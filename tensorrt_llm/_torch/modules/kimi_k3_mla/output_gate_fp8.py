# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MLA output gate fused with the 1x128 FP8 quantization of the o_proj input.

``gate_mul_fp8_packed(attn_out, g)`` returns the E4M3 values and DeepGEMM-packed UE8M0 scales that
``fp8_quantize_1x128_packed_ue8m0(attn_out * g.sigmoid(), False)`` produces, in one kernel instead
of the sigmoid, the multiply and the quantization:

* ``sigmoid`` as PyTorch computes it for BF16: ``bf16(1 / (1 + expf(-float(g))))`` with the
  libdevice ``expf`` and an IEEE division, then ``bf16(float(attn_out) * float(sig))``;
* per row and 128-column block the scale ``2^ceil(log2(amax / 448))`` (amax floored at the BF16
  value of 1e-10), values scaled by its inverse and rounded to E4M3 (nearest even);
* scales as bytes of int32 words ``[ceil(N / 512), pad4(M)]`` (byte ``kb % 4`` of word
  ``(kb // 4, row)``), the bytes past the last block and the rows past ``M`` zeroed.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_BLOCK_M = 32


@triton.jit
def _gate_mul_fp8_kernel(
    a_ptr,
    g_ptr,
    q_ptr,
    s_ptr,
    M,
    stride_a,
    stride_g,
    sf_ld,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    num_kb: tl.constexpr = N // 128
    sf_bytes = s_ptr.to(tl.pointer_type(tl.uint8))
    sf_off = ((kb // 4) * sf_ld + rows) * 4 + kb % 4
    if kb < num_kb:
        cols = kb * 128 + tl.arange(0, 128)
        mask = (rows < M)[:, None]
        a = tl.load(a_ptr + rows[:, None] * stride_a + cols[None, :], mask=mask, other=0.0)
        g = tl.load(g_ptr + rows[:, None] * stride_g + cols[None, :], mask=mask, other=0.0)
        e = libdevice.exp(-g.to(tl.float32))
        sig = tl.math.div_rn(tl.full(e.shape, 1.0, tl.float32), 1.0 + e).to(tl.bfloat16)
        y = (a.to(tl.float32) * sig.to(tl.float32)).to(tl.bfloat16)
        yf = y.to(tl.float32)
        # amax as BF16 magnitude bits (exact: every value is a BF16 value).
        amax_bits = (tl.max(tl.abs(yf), axis=1).to(tl.int32, bitcast=True) >> 16) & 0x7FFF
        exp = (amax_bits >> 7) - 8 + ((amax_bits & 0x7F) > 0x60).to(tl.int32)
        exp = tl.where(amax_bits < 0x2EDC, 85, exp)
        inv = ((254 - exp) << 23).to(tl.float32, bitcast=True)
        q = (yf * inv[:, None]).to(tl.float8e4nv, fp_downcast_rounding="rtne")
        tl.store(q_ptr + rows[:, None] * N + cols[None, :], q, mask=mask)
        tl.store(sf_bytes + sf_off, tl.where(rows < M, exp, 0).to(tl.uint8), mask=rows < sf_ld)
    else:
        tl.store(sf_bytes + sf_off, tl.zeros_like(rows).to(tl.uint8), mask=rows < sf_ld)


def gate_mul_fp8_packed(
    attn_out: torch.Tensor, g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 ``[M, N]`` and packed UE8M0 scales (int32 ``[M, ceil(N / 512)]``, strides
    ``(1, pad4(M))``) of ``attn_out * g.sigmoid()`` (BF16 ``[M, N]`` each, unit column stride)."""
    m, n = attn_out.shape
    if g.shape != attn_out.shape or attn_out.dtype != torch.bfloat16 or g.dtype != torch.bfloat16:
        raise ValueError("attn_out and g must be BF16 tensors of the same [M, N] shape")
    if n % 128 or attn_out.stride(1) != 1 or g.stride(1) != 1:
        raise ValueError("N must be a multiple of 128 and the columns contiguous")
    words = (n // 128 + 3) // 4
    ld = (m + 3) // 4 * 4
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=attn_out.device)
    sf = torch.empty((words, ld), dtype=torch.int32, device=attn_out.device)
    if m:
        grid = (triton.cdiv(ld, _BLOCK_M), words * 4)
        _gate_mul_fp8_kernel[grid](
            attn_out,
            g,
            q,
            sf,
            m,
            attn_out.stride(0),
            g.stride(0),
            ld,
            N=n,
            BLOCK_M=_BLOCK_M,
            num_warps=4,
        )
    return q, sf.t()[:m]


__all__ = ["gate_mul_fp8_packed"]
