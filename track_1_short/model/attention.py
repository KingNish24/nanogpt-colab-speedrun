"""Causal self-attention: YaRN RoPE, sliding-window varlen SDPA, and the per-layer extras (XSA, gates).

Heads have mixed widths (record #360): each layer's module is built with its own query/key width
(qk_dim, 64 or 128) and value/output width (v_dim, 64 or 128). The layer-to-width map lives in
model/gpt.py.

T4 port: the patched FA3 binary (sm90-only) is replaced by segment-wise
F.scaled_dot_product_attention -- one call per packed document/attention segment, with an additive
causal + sliding-window bias built once per (segment length, window) per forward and shared by all
layers through AttnArgs.mask_cache. On sm75 this dispatches to the memory-efficient backend.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from track_1_short.perf.kernels.qkv_rope import qk_norm_rope


class Yarn(nn.Module):
    """RoPE cos/sin tables for one head width, rescaled (YaRN) whenever its layers' window grows.

    Row t of a paired table packs positions 2t and 2t+1 side by side, so every table has max_seq_len
    rows. Each row depends only on its position and the current frequencies, so rebuilding a row range
    gives bitwise the rows a full rebuild would. That makes the partial rebuild legal: once
    GPT.limit_yarn_rebuild sets `rebuild_rows` (right before the clock, to the longest training
    sequence), apply() rebuilds only the rows training reads, and ensure_full() completes the table
    before a validation reads it all (record #360: a window change otherwise rebuilds all 262k rows of
    three tables on the clock).
    """
    # Rotating dims: all of a 64-wide head, the first half of a 128-wide one (half-truncate RoPE by
    # @YouJiacheng); the rest stay fixed. The 32 frequencies are shared by both widths (record #360).
    ROTARY_DIM = 64

    def __init__(self, head_dim, max_seq_len, paired=False, *, attn_scale: float, device):
        super().__init__()
        assert head_dim >= self.ROTARY_DIM
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.base_attn_scale = attn_scale
        width = head_dim if not paired else 2 * head_dim
        self.factor1 = nn.Buffer(torch.empty(max_seq_len, width, dtype=torch.float16, device=device), persistent=False)
        self.factor2 = nn.Buffer(torch.empty(max_seq_len, width, dtype=torch.float16, device=device), persistent=False)
        # Rows apply() rebuilds (all until GPT.limit_yarn_rebuild lowers it), and how many rows are current.
        self.rebuild_rows = max_seq_len
        self.valid_rows = 0
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.ROTARY_DIM // 2, dtype=torch.float32, device=self.device
        )
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        self.angular_freq = torch.cat([
            angular_freq, angular_freq.new_zeros(self.head_dim - self.ROTARY_DIM)
        ])
        self._build_rows(0, self.max_seq_len)
        # inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1839769358388995612
        self.attn_scale = self.base_attn_scale

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        self._build_rows(0, min(self.max_seq_len, self.rebuild_rows))
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

    def ensure_full(self):
        """Rebuild the rows the last apply() skipped (before a forward longer than rebuild_rows)."""
        if self.valid_rows < self.max_seq_len:
            self._build_rows(self.valid_rows, self.max_seq_len)

    def _build_rows(self, lo: int, hi: int):
        """Rows [lo, hi) from the current frequencies; rows [0, hi) are then current (lo is 0 or valid_rows)."""
        assert lo == 0 or lo == self.valid_rows
        t = torch.arange(lo, hi, dtype=torch.float32, device=self.device)
        # copy_ rounds fp32 -> fp16 exactly as .to(torch.float16) does (record #360 stored these bf16;
        # fp16's 10-bit mantissa is strictly sharper, and qk_norm_rope computes on their .float()).
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1[lo:hi].copy_(theta.cos())
            self.factor2[lo:hi].copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1[lo:hi].copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2[lo:hi].copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[lo:hi, 1::2] *= -1
        self.valid_rows = hi

@dataclass(slots=True)
class AttnArgs:
    sa_lambdas: torch.Tensor
    # Attention-segment boundaries [(start, end), ...] summed to the packed batch length, cut at
    # VIRTUAL_SEQ_CAP; computed once per forward in GPT and shared by every layer.
    segments: list[tuple[int, int]]
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor | None
    aux_v: torch.Tensor | None  # added to V; full head width (num_heads * head_dim) on every layer
    xsa_alpha: torch.Tensor | None
    # Per-forward cache of the SDPA biases: (segment length, window) -> [1, 1, L, L] bias (or True
    # when the window covers the whole causal triangle). Built by the first layer that needs it.
    mask_cache: dict = field(default_factory=dict)
    # A 0-D gain folded into the output projection with sa_lambdas[1]: the layer's residual post-lambda,
    # so the residual add takes the attention output as is (perf/residual_fusion.py). None = no gain.
    o_gain: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    """One attention layer at its own head widths. No parameters: the weights come from the GPT's banks.

    forward takes this layer's weights already cut to its widths, all in nn.Linear [out, in] layout:
      qk_w [2 * num_heads * qk_dim, dim]   Q rows then K rows
      v_w  [num_heads * v_dim, dim]
      o_w  [dim, num_heads * v_dim]
    """
    def __init__(
        self, num_heads: int, head_dim: int, qk_dim: int, v_dim: int,
        val_max_seq_len: int, paired: bool = False,
    ):
        super().__init__()
        assert qk_dim <= head_dim and v_dim <= head_dim
        self.val_max_seq_len = val_max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim  # the width aux_v arrives at
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.paired = paired

    def forward(
        self, x: Tensor, attn_args: AttnArgs, qk_w: Tensor, v_w: Tensor, o_w: Tensor,
    ):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        H = self.num_heads
        # unpack attention args
        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        bm_size = attn_args.bm_size
        yarn = attn_args.yarn
        segments = attn_args.segments
        assert yarn.head_dim == self.qk_dim
        # The partial key offset shifts a key's non-rotating dims, so only a head with some can carry it.
        assert not key_offset or self.qk_dim > Yarn.ROTARY_DIM

        # QKV projection, then the QK rms-norm/rotary epilogue in the projection's (differentiable)
        # path -- this is the former validation path, now used for training and validation alike.
        # sa_lambdas are fp32 scalars; cast them so the product stays in x's dtype (a 0-D fp32 tensor
        # times an fp16 tensor would promote to fp32).
        qkv_weight = sa_lambdas[0].type_as(x) * torch.cat((qk_w, v_w)).type_as(x)
        qkv = F.linear(x, qkv_weight)
        qk, v = qkv.split((2 * H * self.qk_dim, H * self.v_dim), dim=-1)
        qk = qk.view(B, T, 2 * H, self.qk_dim)
        v = v.view(B, T, H, self.v_dim)
        q, k = qk_norm_rope(
            qk[0], yarn.factor1[:T], yarn.factor2[:T], H, Yarn.ROTARY_DIM, self.paired, key_offset,
        )
        q, k = q[None], k[None]

        # A narrower V takes the leading v_dim dims of each head of aux_v: the same as adding at full
        # width and dropping the rest.
        if aux_v is not None:
            aux_v = aux_v.view(B, T, H, self.head_dim)[..., :self.v_dim]
        if not self.paired:
            if aux_v is not None:
                v = v + aux_v
        else:
            # Paired heads: adjacent heads' queries attend to each other's keys. Two copies of the
            # input stream are interleaved (q, k already are, by the norm/rotary kernel), which
            # doubles each sequence's length and halves the effective window.
            v = v.reshape(B, T * 2, H // 2, self.v_dim)
            if aux_v is not None:
                v = v + aux_v.reshape(v.shape)
            segments = [(2 * s, 2 * e) for s, e in segments]

        # Sliding-window causal attention per segment (replaces flash_attn_varlen_func).
        y = self._segment_attention(q, k, v, segments, bm_size, yarn.attn_scale, attn_args.mask_cache)
        y = y.view(B, T, H, self.v_dim)
        # Gated XSA (arXiv:2603.09078) with learnable strength: subtract per-head fraction tanh(α)
        # of y aligned with v̂. Non-paired only (v shape doesn't line up for paired layers).
        if attn_args.xsa_alpha is not None and not self.paired:
            dot = (y * v).sum(-1, keepdim=True)
            denom = v.square().sum(-1, keepdim=True).clamp_min(1e-8)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(B, T, H, 1)
            y = y - alpha * (dot / denom) * v
        if attn_gate_w is not None:
            y = y * attn_gate_w.type_as(y).view(B, T, H, 1)
        y = y.contiguous().view(B, T, H * self.v_dim) # re-assemble all head outputs side by side
        # The output scale rounds to fp16 in the one product with O, with or without o_gain.
        o_scale = sa_lambdas[1] if attn_args.o_gain is None else sa_lambdas[1] * attn_args.o_gain
        return F.linear(y, o_scale.type_as(y) * o_w.type_as(y))

    @staticmethod
    def _segment_attention(q, k, v, segments, bm_size, scale, mask_cache):
        """One SDPA call per attention segment, with a causal + left-window bias.

        q, k: [1, T, H, D]; v: [1, T, H, Dv]; segments: [(start, end), ...] over T. The bias of a
        (length, window) pair is cached in mask_cache for the other layers of the same forward.
        """
        outs = []
        for s, e in segments:
            L = e - s
            if L <= 0:
                continue
            qs = q[:, s:e].transpose(1, 2)   # [1, H, L, D]
            ks = k[:, s:e].transpose(1, 2)
            vs = v[:, s:e].transpose(1, 2)
            cached = mask_cache.get((L, bm_size))
            if cached is None:
                if bm_size >= L - 1:
                    # The window covers the whole causal triangle: no bias at all.
                    cached = True
                else:
                    i = torch.arange(L, device=q.device)
                    d = i[:, None] - i[None, :]
                    bias = torch.zeros((L, L), dtype=q.dtype, device=q.device)
                    bias.masked_fill_((d < 0) | (d > bm_size), float("-inf"))
                    cached = bias[None, None]   # [1, 1, L, L]: broadcast over batch and heads
                mask_cache[(L, bm_size)] = cached
            if cached is True:
                out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)
            else:
                out = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=cached, scale=scale)
            outs.append(out.transpose(1, 2))    # [1, L, H, Dv]
        return torch.cat(outs, dim=1)
