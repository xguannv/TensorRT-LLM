# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm::k3_drafter_mla_ctx_kv``: the MLA DSpark drafter's context latents of every layer in one launch.

The stock drafter finishes the latents of its one context GEMM layer by layer (``precompute_context_kv``: an RMSNorm of
the kv_lora_rank latent and the RoPE of k_pe per layer) and scatters them layer by layer into the paged pool
(``_store_context_kv_paged``). This kernel does both for all layers: for every row and layer, ``RMSNorm(kv) * weight``
(fp32, rounded to bf16) and the adjacent-pair RoPE of k_pe at the row's position (fp32, rounded to bf16), stored at the
row's page and offset. Shapes are the K3 drafter's: kv_lora_rank 512, rope 64.
"""

from typing import List

import torch
import triton
import triton.language as tl

KV_LORA = 512
ROPE = 64


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
