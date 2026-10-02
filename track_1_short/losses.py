"""The training loss: chunked softcapped cross-entropy over the vocabulary (or a candidate set).

T4 port of the fused CUDA CE kernel (perf/kernels/cross_entropy.py, deleted): that kernel read an
fp8 lm_head GEMM's e4m3 logits and wrote an e5m2 logit gradient -- both need sm89+ FP8 conversions
and were compiled for sm90, so neither can run on a T4 (sm75). This module keeps the exact math,
in differentiable pure PyTorch with fp16 GEMMs (tensor cores) over chunks of the class axis:

  z      = A * sigmoid((logit + B) / C), (A, B, C) = (23, 5, 7.5)
  lse    = logsumexp_c z
  loss_t = sum_k mtp_weights[k] * (lse - z[target_{t+k}])
           + prefix_weight * (lse - z[prefix_t])      (prefix term only where prefix >= 0)
  dz     = S_w * softmax(z) - (sum of the weights of the predictions targeting each class)
  dlogit = (A / C) * dz * s * (1 - s),  s = sigmoid((logit + B) / C) = z / A

The [T, V] logits are never materialized: a forward pass over class chunks accumulates the
log-sum-exp, and the backward recomputes each chunk's softmax rows on the fly.

Standard autograd semantics (grad_output is applied), unlike the original kernel, which baked
loss.sum()'s gradient in; the caller scales with losses.mean() and the static LOSS_SCALE.

Provenance: loss formula and softcap constants from record #360 (ANVIL2); port for sm75/fp16.
"""
import torch

SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5
# Class rows per logits chunk: [MICRO_BATCH, CLS_CHUNK] fp32 = 134 MB at the training microbatch.
CLS_CHUNK = 2048


def _softcap(logits: torch.Tensor) -> torch.Tensor:
    return SOFTCAP_A * torch.sigmoid((logits + SOFTCAP_B) / SOFTCAP_C)


class SoftcappedCE(torch.autograd.Function):
    """Per-token softcapped CE over `weight`'s columns, with MTP and prefix targets.

    x: [n, D] fp16; weight: [D, M] fp16 (lm_head, or a sampled candidate slab for M = P).
    lm_head_weight: [D, V] fp16, the live lm_head weight. It is unused in the forward; it only
        carries the backward's gradient on the sampled path (record #360's fused kernel did the
        same: its lm_head input "only carries that gradient"). The sampled rows slab is a reused
        buffer that must stay outside autograd (hence the forward can never route grad through it),
        so the backward scatters the [D, P] candidate gradient into a dense [D, V] one and returns
        it in this slot. The full-softmax path passes None here and takes its [D, V] gradient in
        the `weight` slot instead.
    mtp_weights: [K] fp32; prefix_weight: 0-dim fp32 tensor or python float.
    target_cols: [L] int64 class ids (full softmax) or candidate positions (sampled), L >= n;
        row t's k-th target is target_cols[t + k], valid while t + k < L (padding is -1).
    prefix_cols: [n] int64, -1 where there is no prefix target.
    vocab_pos: None for the full softmax; for sampled, [V] int32 class -> candidate column (-1
        outside), which scatters the [D, P] gradient into a dense [D, V] one.
    Returns losses: [n] fp32.
    """
    @staticmethod
    def forward(ctx, x, weight, lm_head_weight, mtp_weights, prefix_weight, target_cols,
                prefix_cols, vocab_pos):
        n, D = x.shape
        M = weight.shape[1]
        K = int(mtp_weights.numel())
        device = x.device
        x_f = x.float()
        mtp_weights = mtp_weights.float()
        prefix_weight = torch.as_tensor(prefix_weight, dtype=torch.float32, device=device)

        # ---- lse over all M classes, chunked (each chunk's logits are freed as we go) ----
        sum_exp = torch.zeros(n, dtype=torch.float32, device=device)
        for lo in range(0, M, CLS_CHUNK):
            hi = min(lo + CLS_CHUNK, M)
            z = _softcap((x @ weight[:, lo:hi]).float())
            sum_exp += (z - SOFTCAP_A).exp().sum(dim=1)
        lse = SOFTCAP_A + sum_exp.log()

        # ---- the K MTP target logits and the prefix logits ----
        pad = max(K - 1, 0)
        if target_cols.numel() < n + pad:
            target_cols = torch.nn.functional.pad(target_cols, (0, n + pad - target_cols.numel()), value=-1)
        cols = torch.stack([target_cols[k:k + n] for k in range(K)], dim=1)      # [n, K]
        valid = cols >= 0
        losses = torch.zeros(n, dtype=torch.float32, device=device)
        for k in range(K):
            # z at row t's k-th target, with -1 (invalid) rows clamped to column 0 and masked out.
            z_k = _softcap(torch.einsum("nd,dn->n", x_f, weight[:, cols[:, k].clamp(0, M - 1)].float()))
            losses += mtp_weights[k] * valid[:, k].float() * (lse - z_k)

        pvalid = prefix_cols >= 0
        z_p = _softcap(torch.einsum("nd,dn->n", x_f, weight[:, prefix_cols.clamp(0, M - 1)].float()))
        losses = losses + prefix_weight * pvalid.float() * (lse - z_p)

        ctx.save_for_backward(x, weight, lse, cols, valid, prefix_cols, pvalid,
                              mtp_weights, prefix_weight, vocab_pos)
        return losses

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, lse, cols, valid, prefix_cols, pvalid, mtp_weights, prefix_weight, vocab_pos = \
            ctx.saved_tensors
        n, D = x.shape
        M = weight.shape[1]
        K = int(mtp_weights.numel())
        device = x.device
        g = grad_out.float()
        # Per-row total weight on the softmax-normalizer term (MTP + a valid prefix term).
        S_w = mtp_weights.sum() + prefix_weight * pvalid.float()                       # [n]
        gs = g * S_w                                                                    # [n]
        A_div_C = SOFTCAP_A / SOFTCAP_C
        prefix_cols_clamped = prefix_cols.clamp(0, M - 1)

        dx = torch.zeros(n, D, dtype=torch.float32, device=device)
        dW = torch.zeros(D, M, dtype=weight.dtype, device=device)
        for lo in range(0, M, CLS_CHUNK):
            hi = min(lo + CLS_CHUNK, M)
            Wc = weight[:, lo:hi]
            z = _softcap((x @ Wc).float())
            s = z / SOFTCAP_A                       # sigmoid((logit + B) / C), fp32
            z.sub_(lse[:, None]).exp_()             # softmax over this chunk's classes (reuses z)
            dlogit = gs[:, None] * z
            del z
            # Subtract the weight of every prediction whose target lands on this chunk's columns.
            in_chunk = (cols >= lo) & (cols < hi) & valid                                # [n, K]
            local = (cols - lo).clamp(0, hi - lo - 1)
            for k in range(K):
                rows = in_chunk[:, k]
                if bool(rows.any()):
                    dlogit[rows, local[rows, k]] -= g[rows] * mtp_weights[k]
            p_in = (prefix_cols >= lo) & (prefix_cols < hi) & pvalid
            if bool(p_in.any()):
                dlogit[p_in, prefix_cols_clamped[p_in] - lo] -= g[p_in] * prefix_weight
            # Softcap derivative folded in place. The per-chunk [n, CLS_CHUNK] fp32 tensors are the
            # peak of the whole backward on a 16 GB T4, so never materialize A_div_C * s * (1 - s).
            dlogit.mul_(s).mul_(A_div_C)
            s.neg_().add_(1)                        # s := 1 - s, the last factor, in place
            dlogit.mul_(s)
            del s, in_chunk, local
            # fp16 tensor-core GEMMs; fp32 accumulation only in the fp32 results.
            # weight is [D, M] (in, out): dx = dlogit @ Wc.T, dW = x.T @ dlogit.
            dx += (dlogit.to(torch.float16) @ Wc.T).float()
            dW[:, lo:hi] = x.T @ dlogit.to(torch.float16)
            del dlogit
        dx = dx.to(x.dtype)
        dW = dW.to(weight.dtype)
        if vocab_pos is not None:
            # Sampled: the rows slab is a grad-free buffer, so the candidate gradient cannot be
            # returned in the `weight` slot (autograd would discard it); scatter it dense and land
            # it on the lm_head_weight carrier instead.
            dense = torch.zeros(D, int(vocab_pos.numel()), dtype=dW.dtype, device=dW.device)
            keep = vocab_pos >= 0
            dense[:, keep] = dW[:, vocab_pos[keep].long()]
            return dx, None, dense, None, None, None, None, None
        return dx, dW, None, None, None, None, None, None


def softcapped_ce(x, weight, lm_head_weight, mtp_weights, prefix_weight, target_cols, prefix_cols,
                  vocab_pos=None):
    """SoftcappedCE over `weight`'s columns; see SoftcappedCE."""
    return SoftcappedCE.apply(x, weight, lm_head_weight, mtp_weights.float(), prefix_weight,
                              target_cols, prefix_cols, vocab_pos)
