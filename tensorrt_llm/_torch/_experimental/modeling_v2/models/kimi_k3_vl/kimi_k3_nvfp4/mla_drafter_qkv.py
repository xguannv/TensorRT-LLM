# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm::k3_drafter_mla_a_post``: what follows the MLA DSpark drafter's fused a-projection, in one launch.

Input ``a`` is ``[q_a | kv_a | k_pe]`` (``[rows, q_lora_rank + kv_lora_rank + rope]``, one GEMV over the concatenated
``q_a_proj`` / ``kv_a_proj_with_mqa`` weights). The kernel returns ``RMSNorm(q_a) * q_a_layernorm.weight`` (the q_b
projection's input) and stores the block's latent rows ``[RMSNorm(kv_a) * kv_a_layernorm.weight | rotated k_pe]`` into
the paged pool at ``slots`` (element offsets), replacing the stock drafter's q_a_layernorm, latent RMSNorm + RoPE and
the pool write of the block. The q_b projection, the query RoPE and the nope absorption stay the stock drafter's (many
CTAs each).

Arithmetic follows the stock drafter's roundings: RMSNorms in fp32 (``x * rsqrt(mean(x^2) + eps)``, times the weight)
rounded to bf16, the adjacent-pair RoPE in fp32 rounded to bf16. Shapes are the K3 drafter's: q_lora_rank 1536,
kv_lora_rank 512, rope 64.
"""

import torch
import triton
import triton.language as tl

Q_LORA = 1536
KV_LORA = 512
ROPE = 64
NOPE = 128
A_WIDTH = Q_LORA + KV_LORA + ROPE
ROW_TILE = 4


@triton.jit
def _drafter_mla_a_post(
    A,
    QA_NORM,
    KV_NORM,
    FREQS,
    POOL,
    SLOTS,
    QA_OUT,
    M,
    A_STRIDE: tl.constexpr,
    QA_EPS: tl.constexpr,
    KV_EPS: tl.constexpr,
):
    # Program (chunk, 0): q_a of ROW_TILE rows; program (chunk, 1): their block latents.
    chunk, part = tl.program_id(0), tl.program_id(1)
    rows = chunk * 4 + tl.arange(0, 4)
    valid = rows < M
    if part == 0:
        cols = tl.arange(0, 2048)
        keep = valid[:, None] & (cols[None, :] < 1536)
        qa = tl.load(A + rows[:, None] * A_STRIDE + cols[None, :], keep, 0).to(tl.float32)
        qa = qa * tl.rsqrt(tl.sum(qa * qa, 1) / 1536 + QA_EPS)[:, None]
        qa = tl.load(QA_NORM + cols, cols < 1536, 0).to(tl.float32)[None, :] * qa
        tl.store(QA_OUT + rows[:, None] * 1536 + cols[None, :], qa.to(tl.bfloat16), keep)
    else:
        latent = tl.arange(0, 512)
        half = tl.arange(0, 32)
        slot = tl.load(SLOTS + rows, valid, 0)
        ckv = tl.load(A + rows[:, None] * A_STRIDE + 1536 + latent[None, :], valid[:, None], 0).to(
            tl.float32
        )
        ckv = ckv * tl.rsqrt(tl.sum(ckv * ckv, 1) / 512 + KV_EPS)[:, None]
        ckv = tl.load(KV_NORM + latent).to(tl.float32)[None, :] * ckv
        tl.store(POOL + slot[:, None] + latent[None, :], ckv.to(tl.bfloat16), valid[:, None])
        cos = tl.load(FREQS + rows[:, None] * 64 + 2 * half[None, :], valid[:, None], 0)
        sin = tl.load(FREQS + rows[:, None] * 64 + 2 * half[None, :] + 1, valid[:, None], 0)
        k_even = tl.load(
            A + rows[:, None] * A_STRIDE + 2048 + 2 * half[None, :], valid[:, None], 0
        ).to(tl.float32)
        k_odd = tl.load(
            A + rows[:, None] * A_STRIDE + 2049 + 2 * half[None, :], valid[:, None], 0
        ).to(tl.float32)
        tl.store(
            POOL + slot[:, None] + 512 + 2 * half[None, :],
            (k_even * cos - k_odd * sin).to(tl.bfloat16),
            valid[:, None],
        )
        tl.store(
            POOL + slot[:, None] + 513 + 2 * half[None, :],
            (k_even * sin + k_odd * cos).to(tl.bfloat16),
            valid[:, None],
        )


def supports(a: torch.Tensor, pool: torch.Tensor) -> bool:
    """Whether the kernel takes these operands: bf16 ``a`` ``[rows, 2112]`` with unit column stride and a bf16 pool
    whose rows are 576 contiguous elements."""
    return (
        a.dtype == torch.bfloat16
        and a.dim() == 2
        and a.shape[1] == A_WIDTH
        and a.stride(1) == 1
        and pool.dtype == torch.bfloat16
        and pool.shape[-1] == KV_LORA + ROPE
        and pool.stride(-1) == 1
    )


@torch.library.custom_op("trtllm::k3_drafter_mla_a_post", mutates_args=("pool",))
def drafter_mla_a_post(
    a: torch.Tensor,
    qa_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    qa_eps: float,
    kv_eps: float,
) -> torch.Tensor:
    """The normalized q_a ``[rows, 1536]`` bf16, and the block latent stored into ``pool`` at ``slots`` (int64 element
    offsets of each row in ``pool``); ``freqs``: fp32 ``[rows, 32, 2]`` (cos, sin) per row."""
    rows = a.shape[0]
    out = a.new_empty((rows, Q_LORA))
    if rows == 0:
        return out
    freqs = freqs.reshape(rows, ROPE).contiguous()
    _drafter_mla_a_post[(triton.cdiv(rows, ROW_TILE), 2)](
        a,
        qa_norm,
        kv_norm,
        freqs,
        pool,
        slots,
        out,
        rows,
        a.stride(0),
        qa_eps,
        kv_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@drafter_mla_a_post.register_fake
def _drafter_mla_a_post_fake(a, qa_norm, kv_norm, freqs, pool, slots, qa_eps, kv_eps):
    return a.new_empty((a.shape[0], Q_LORA))
