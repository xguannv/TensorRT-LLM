# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm::k3_drafter_mla_qkv``: the MLA DSpark drafter's block query and block latent from its fused a-projection.

One launch replaces the stock drafter's chain after its two a-projections (``q_a_layernorm``, ``q_b_proj``, the query
RoPE, the nope absorption into latent space, the latent RMSNorm + RoPE and the block's pool write). Input ``a`` is
``[q_a | kv_a]`` (``[rows, q_lora_rank + kv_lora_rank + rope]``, one GEMV over the concatenated weights). Outputs:

* ``query`` ``[rows, H, kv_lora_rank + rope]``: per head, ``q_nope @ k_b`` then the rotated ``q_pe``, the layout the
  absorbed MLA decode reads;
* the block's latent rows ``[RMSNorm(kv) | rotated k_pe]`` stored into the paged pool at ``slots`` (element offsets).

Arithmetic follows the stock drafter's roundings: RMSNorms in fp32 (``x * rsqrt(mean(x^2) + eps)``, times the weight)
rounded to bf16, bf16 GEMMs with fp32 accumulation rounded to bf16, the adjacent-pair RoPE in fp32 rounded to bf16.
Shapes are the K3 drafter's: q_lora_rank 1536, kv_lora_rank 512, rope 64, nope 128.
"""

from typing import List

import torch
import triton
import triton.language as tl

Q_LORA = 1536
KV_LORA = 512
ROPE = 64
NOPE = 128
A_WIDTH = Q_LORA + KV_LORA + ROPE
ROW_TILE = 16


@triton.jit
def _drafter_mla_qkv(
    A,
    QA_NORM,
    QB,
    KB,
    KV_NORM,
    FREQS,
    POOL,
    SLOTS,
    OUT,
    M,
    A_STRIDE: tl.constexpr,
    H: tl.constexpr,
    KB_S0: tl.constexpr,
    KB_S1: tl.constexpr,
    KB_S2: tl.constexpr,
    FREQ_STRIDE: tl.constexpr,
    QA_EPS: tl.constexpr,
    KV_EPS: tl.constexpr,
):
    head, chunk = tl.program_id(0), tl.program_id(1)
    rows = chunk * 16 + tl.arange(0, 16)
    valid = rows < M
    qa_cols = tl.arange(0, 2048)
    qa = tl.load(
        A + rows[:, None] * A_STRIDE + qa_cols[None, :],
        valid[:, None] & (qa_cols[None, :] < 1536),
        0,
    ).to(tl.float32)
    rms = tl.rsqrt(tl.sum(qa * qa, 1) / 1536 + QA_EPS)
    k = tl.arange(0, 128)
    half = tl.arange(0, 32)
    qb_head = head * 192
    q_nope = tl.zeros((16, 128), tl.float32)
    q_even = tl.zeros((16, 32), tl.float32)
    q_odd = tl.zeros((16, 32), tl.float32)
    for block in range(12):
        cols = block * 128 + k
        a = tl.load(A + rows[:, None] * A_STRIDE + cols[None, :], valid[:, None], 0).to(tl.float32)
        a = a * rms[:, None]
        a = (tl.load(QA_NORM + cols).to(tl.float32)[None, :] * a).to(tl.bfloat16)
        b_nope = tl.load(QB + (qb_head + k)[None, :] * 1536 + cols[:, None])
        q_nope += tl.dot(a, b_nope)
        b_even = tl.load(QB + (qb_head + 128 + 2 * half)[None, :] * 1536 + cols[:, None])
        q_even += tl.dot(a, b_even)
        b_odd = tl.load(QB + (qb_head + 129 + 2 * half)[None, :] * 1536 + cols[:, None])
        q_odd += tl.dot(a, b_odd)
    cos = tl.load(FREQS + rows[:, None] * FREQ_STRIDE + 2 * half[None, :], valid[:, None], 0)
    sin = tl.load(FREQS + rows[:, None] * FREQ_STRIDE + 2 * half[None, :] + 1, valid[:, None], 0)
    latent = tl.arange(0, 512)
    kb = tl.load(KB + head * KB_S0 + k[:, None] * KB_S1 + latent[None, :] * KB_S2)
    absorbed = tl.dot(q_nope.to(tl.bfloat16), kb)
    base = rows[:, None] * (H * 576) + head * 576
    tl.store(OUT + base + latent[None, :], absorbed.to(tl.bfloat16), valid[:, None])
    q_even = q_even.to(tl.bfloat16).to(tl.float32)
    q_odd = q_odd.to(tl.bfloat16).to(tl.float32)
    tl.store(
        OUT + base + 512 + 2 * half[None, :],
        (q_even * cos - q_odd * sin).to(tl.bfloat16),
        valid[:, None],
    )
    tl.store(
        OUT + base + 513 + 2 * half[None, :],
        (q_even * sin + q_odd * cos).to(tl.bfloat16),
        valid[:, None],
    )
    if head == 0:
        slot = tl.load(SLOTS + rows, valid, 0)
        ckv = tl.load(A + rows[:, None] * A_STRIDE + 1536 + latent[None, :], valid[:, None], 0).to(
            tl.float32
        )
        kv_rms = tl.rsqrt(tl.sum(ckv * ckv, 1) / 512 + KV_EPS)
        ckv = ckv * kv_rms[:, None]
        ckv = tl.load(KV_NORM + latent).to(tl.float32)[None, :] * ckv
        tl.store(POOL + slot[:, None] + latent[None, :], ckv.to(tl.bfloat16), valid[:, None])
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


def supports(a: torch.Tensor, q_b: torch.Tensor, k_b: torch.Tensor, pool: torch.Tensor) -> bool:
    """Whether the kernel takes these operands: bf16 ``a`` ``[rows, 2112]`` with unit column stride, a contiguous bf16
    ``q_b`` weight ``[H x 192, 1536]``, bf16 ``k_b`` ``[H, 128, 512]``, and a bf16 pool whose rows are 576 contiguous
    elements."""
    return (
        a.dtype == torch.bfloat16
        and a.dim() == 2
        and a.shape[1] == A_WIDTH
        and a.stride(1) == 1
        and q_b.dtype == torch.bfloat16
        and q_b.is_contiguous()
        and q_b.dim() == 2
        and q_b.shape[1] == Q_LORA
        and q_b.shape[0] % (NOPE + ROPE) == 0
        and k_b.dtype == torch.bfloat16
        and tuple(k_b.shape) == (q_b.shape[0] // (NOPE + ROPE), NOPE, KV_LORA)
        and pool.dtype == torch.bfloat16
        and pool.shape[-1] == KV_LORA + ROPE
        and pool.stride(-1) == 1
    )


@torch.library.custom_op("trtllm::k3_drafter_mla_qkv", mutates_args=("pool",))
def drafter_mla_qkv(
    a: torch.Tensor,
    qa_norm: torch.Tensor,
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    kv_norm: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    qa_eps: float,
    kv_eps: float,
) -> torch.Tensor:
    """``query`` ``[rows, H, 576]`` and the block latent stored into ``pool`` at ``slots`` (int64 element offsets of
    each row in ``pool``); ``freqs``: fp32 ``[rows, 32, 2]`` (cos, sin) per row."""
    heads = q_b.shape[0] // (NOPE + ROPE)
    rows = a.shape[0]
    out = a.new_empty((rows, heads, KV_LORA + ROPE))
    if rows == 0:
        return out
    freqs = freqs.reshape(rows, ROPE).contiguous()
    _drafter_mla_qkv[(heads, triton.cdiv(rows, ROW_TILE))](
        a,
        qa_norm,
        q_b,
        k_b,
        kv_norm,
        freqs,
        pool,
        slots,
        out,
        rows,
        a.stride(0),
        heads,
        *k_b.stride(),
        ROPE,
        qa_eps,
        kv_eps,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return out


@drafter_mla_qkv.register_fake
def _drafter_mla_qkv_fake(a, qa_norm, q_b, k_b, kv_norm, freqs, pool, slots, qa_eps, kv_eps):
    return a.new_empty((a.shape[0], q_b.shape[0] // (NOPE + ROPE), KV_LORA + ROPE))


@triton.jit
def _drafter_mla_ctx_kv(
    LATENT,
    NORM,
    FREQS,
    POS,
    PAGES,
    OFFSETS,
    PTRS,
    LAT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
):
    row, layer = tl.program_id(0), tl.program_id(1)
    latent = tl.arange(0, 512)
    half = tl.arange(0, 32)
    src = LATENT + row * LAT_STRIDE + layer * 576
    x = tl.load(src + latent).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 0) / 512 + EPS)
    x = tl.load(NORM + layer * 512 + latent).to(tl.float32) * x
    pos = tl.load(POS + row).to(tl.int64)
    page = tl.load(PAGES + row).to(tl.int64)
    offset = tl.load(OFFSETS + row).to(tl.int64)
    dst = (
        tl.load(PTRS + layer).to(tl.pointer_type(tl.bfloat16))
        + page * PAGE_STRIDE
        + offset * ROW_STRIDE
    )
    tl.store(dst + latent, x.to(tl.bfloat16))
    k_even = tl.load(src + 512 + 2 * half).to(tl.float32)
    k_odd = tl.load(src + 513 + 2 * half).to(tl.float32)
    cos = tl.load(FREQS + pos * 64 + 2 * half)
    sin = tl.load(FREQS + pos * 64 + 2 * half + 1)
    tl.store(dst + 512 + 2 * half, (k_even * cos - k_odd * sin).to(tl.bfloat16))
    tl.store(dst + 513 + 2 * half, (k_even * sin + k_odd * cos).to(tl.bfloat16))


@torch.library.custom_op("trtllm::k3_drafter_mla_ctx_kv", mutates_args=("pools",))
def drafter_mla_ctx_kv(
    latent: torch.Tensor,
    norm: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    pointers: torch.Tensor,
    pools: List[torch.Tensor],
    eps: float,
) -> None:
    """Store the drafter's context latents into its paged pools: for every row of ``latent`` (``[rows, L x 576]``,
    the raw ``[kv_a | k_pe]`` of each layer) and layer, ``RMSNorm(kv_a) * norm[layer]`` and ``k_pe`` rotated at
    ``positions[row]`` (``freqs``: fp32 ``[max_pos, 32, 2]``), written to page ``pages[row]``, row ``offsets[row]`` of
    ``pools[layer]``. ``pointers``: int64 ``[L]``, each pool's data pointer; ``pools``: the pools (all of one layout,
    ``[pages, 1, 1, page_size, 576]`` bf16), named so the op records the write."""
    rows = latent.shape[0]
    if rows == 0:
        return
    pool = pools[0]
    _drafter_mla_ctx_kv[(rows, len(pools))](
        latent,
        norm,
        freqs,
        positions,
        pages,
        offsets,
        pointers,
        latent.stride(0),
        pool.stride(0),
        pool.stride(3),
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )


@drafter_mla_ctx_kv.register_fake
def _drafter_mla_ctx_kv_fake(latent, norm, freqs, positions, pages, offsets, pointers, pools, eps):
    return None
