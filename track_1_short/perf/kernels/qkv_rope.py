"""QK-norm + RoPE for attention, in differentiable pure PyTorch (fp32 math, cast back).

This replaces the fused Triton forward (which had no backward) and the packed FP8 QKV projection
(unsupported on sm75). The math is identical to the kernels it replaces (record #344/#360):
  - rms-norm over the head's full width with eps 1.1920928955078125e-7 (F.rms_norm's fp32 eps),
  - RoPE on the first `rotary_dim` dims with adjacent-lane swaps against the paired factor tables,
  - the partial key offset: a key's stationary dims (d >= rotary_dim) come from the previous token,
  - paired heads: head h of token t lands at virtual token 2t + h // (num_heads/2), head h %
    (num_heads/2), so adjacent heads attend to each other's keys; odd heads read the second
    half of the paired rotary row.

Q and K keep the layer's own head width (64 or 128).
"""
import torch


def qk_norm_rope(qk, factor1, factor2, num_heads, rotary_dim, paired, key_offset):
    """(q, k) from the packed [tokens, 2 * num_heads, qk_dim] QK projection.

    Each of (q, k) is [tokens, num_heads, qk_dim], or [2 * tokens, num_heads / 2, qk_dim] when
    paired. Differentiable: gradients flow through the norm, the rotary mix and the key offset.
    """
    tokens, heads2, qk_dim = qk.shape
    H = num_heads
    assert heads2 == 2 * H
    assert factor1.shape == factor2.shape == (tokens, qk_dim * (2 if paired else 1))
    assert not (paired and key_offset)
    assert rotary_dim <= qk_dim

    x = qk.float()
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.1920928955078125e-7)
    xn = x * rstd
    # The adjacent-lane swap RoPE needs: lanes 2i and 2i+1 exchange.
    x_flip = xn.reshape(tokens, heads2, qk_dim // 2, 2).flip(-1).reshape(tokens, heads2, qk_dim)

    if paired:
        # The factor tables carry both parities side by side: [tokens, 2 * qk_dim]. Flat head h
        # (Q block 0..H-1, K block H..2H-1) reads its parity's half; H is even, so h % 2 equals
        # (h % H) % 2 for both blocks.
        parity = torch.arange(heads2, device=qk.device) % 2
        f1 = torch.stack((factor1[:, :qk_dim], factor1[:, qk_dim:]), dim=1)[:, parity].float()
        f2 = torch.stack((factor2[:, :qk_dim], factor2[:, qk_dim:]), dim=1)[:, parity].float()
    else:
        f1 = factor1.unsqueeze(1).float()   # [T, 1, D]: shared by every head
        f2 = factor2.unsqueeze(1).float()

    y = f1 * xn + f2 * x_flip

    if key_offset:
        # A key's stationary dims come from the previous token's normed row (its own rstd);
        # token 0 keeps its own row (the mask's token > 0 excludes it).
        prev_x = torch.cat((x[:, :1], x[:, :-1]), dim=0)
        prev_norm = prev_x * torch.rsqrt(prev_x.square().mean(dim=-1, keepdim=True) + 1.1920928955078125e-7)
        shift = (torch.arange(tokens, device=qk.device) > 0)[:, None] & \
                (torch.arange(qk_dim, device=qk.device) >= rotary_dim)[None, :]   # [T, D]
        y = torch.cat((y[:, :H], torch.where(shift[:, None, :], prev_norm[:, H:], y[:, H:])), dim=1)

    q, k = y[:, :H], y[:, H:]
    if paired:
        # head h = a * (H // 2) + b at token t -> virtual row 2t + a, virtual head b.
        q = q.reshape(tokens, 2, H // 2, qk_dim).permute(0, 2, 1, 3).reshape(tokens * 2, H // 2, qk_dim)
        k = k.reshape(tokens, 2, H // 2, qk_dim).permute(0, 2, 1, 3).reshape(tokens * 2, H // 2, qk_dim)
    return q.to(qk.dtype), k.to(qk.dtype)
