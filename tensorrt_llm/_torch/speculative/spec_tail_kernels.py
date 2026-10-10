# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused bookkeeping kernels for the one-model speculative-decoding tail.

Each kernel replaces a chain of small eager PyTorch ops with one launch and
produces bit-identical results:

* :func:`fused_strict_accept` -- strict (token-equality) acceptance of the
  previous draft against the sampled target tokens, including the optional
  synthetic acceptance override of ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``.
* :func:`decode_draft_block_table` -- the DFlash draft-KV block-table decode
  (count of allocated blocks per request, clamped and divided block offsets).

``TLLM_SPEC_TAIL_FUSION=1`` selects them. Launch shapes depend only on host
batch sizes, so they are CUDA-graph capturable.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

SPEC_TAIL_FUSION_ENV_VAR = "TLLM_SPEC_TAIL_FUSION"

_BLOCK_TABLE_BLOCK = 2048


def spec_tail_fusion_enabled() -> bool:
    """Whether ``TLLM_SPEC_TAIL_FUSION=1`` selects the fused tail kernels."""
    return os.environ.get(SPEC_TAIL_FUSION_ENV_VAR, "0") == "1"


@triton.jit
def _strict_accept_kernel(
    target_ptr,
    draft_ptr,
    stride_draft,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    pool_ptr,
    counter_ptr,
    base_total,
    frac,
    max_total,
    DRAFT_LEN: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    width = DRAFT_LEN + 1
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)

    # Context requests: accept the sampled token only.
    ctx_mask = rows < num_ctx
    t_ctx = tl.load(target_ptr + rows, mask=ctx_mask, other=0)
    tl.store(accepted_ptr + rows * width, t_ctx.to(tl.int32), mask=ctx_mask)
    tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ctx_mask)

    # Generation requests: accepted row = the K+1 target tokens; the accepted
    # length is 1 + the number of leading draft tokens equal to them.
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    t_gen = tl.load(
        target_ptr + num_ctx + rows[:, None] * width + cols[None, :], mask=tile_mask, other=0
    ).to(tl.int32)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(
        draft_ptr + rows[:, None] * stride_draft + cols[None, :], mask=draft_mask, other=0
    ).to(tl.int32)
    mismatch = draft_mask & (d != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    n = 1 + first_mismatch
    if FORCE == 1:
        n = tl.full([BLOCK_B], 0, tl.int32) + base_total
    elif FORCE == 2:
        # Same draw as SpecWorkerBase._apply_force_accepted_tokens: bump the
        # device counter, hash (counter, slot) into the fixed uniform pool.
        counter = tl.load(counter_ptr) + 1
        tl.store(counter_ptr, counter)
        idx = (counter * COUNTER_STRIDE + rows.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
        rand = tl.load(pool_ptr + idx, mask=gen_mask, other=1.0)
        extra = (rand < frac).to(tl.int32)
        n = tl.minimum(base_total + extra, max_total)
    tl.store(num_accepted_ptr + num_ctx + rows, n.to(tl.int32), mask=gen_mask)


def fused_strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: Optional[tuple] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance in one launch.

    Args:
        target_tokens: ``[num_contexts + num_gens * (K + 1)]`` sampled target
            tokens (any integer dtype).
        draft_tokens: ``[num_gens, K]`` previous draft tokens (unit inner stride).
        num_contexts, batch_size: request counts.
        force: ``None`` for no override, ``(base_total,)`` for a constant
            accepted length, or ``(base_total, frac, max_total, pool, counter,
            counter_stride, slot_stride)`` for the fractional override.
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` int32 and
        ``num_accepted_tokens [batch_size]`` int32, matching
        ``SpecWorkerBase._sample_and_accept_draft_tokens_base`` (context rows
        only define column 0, as there).
    """
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if batch_size == 0:
        return accepted, num_accepted
    assert draft_tokens.dim() == 2 and (num_gens == 0 or draft_tokens.stride(-1) == 1)
    force_mode, base_total, frac, max_total = 0, 0, 0.0, 0
    pool, counter, counter_stride, slot_stride, pool_mask = target_tokens, target_tokens, 1, 1, 0
    if force is not None and len(force) == 1:
        force_mode, base_total = 1, int(force[0])
    elif force is not None:
        base_total, frac, max_total, pool, counter, counter_stride, slot_stride = force
        force_mode = 2
        pool_mask = pool.numel() - 1
    _strict_accept_kernel[(1,)](
        target_tokens,
        draft_tokens,
        draft_tokens.stride(0) if num_gens > 0 else draft_len,
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        pool,
        counter,
        base_total,
        float(frac),
        max_total,
        DRAFT_LEN=draft_len,
        FORCE=force_mode,
        COUNTER_STRIDE=counter_stride,
        SLOT_STRIDE=slot_stride,
        POOL_MASK=pool_mask,
        BLOCK_B=triton.next_power_of_2(max(num_contexts, num_gens, 1)),
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        num_warps=4,
    )
    return accepted, num_accepted


@triton.jit
def _decode_block_table_kernel(
    src_ptr,
    stride_src,
    stride_src_col,
    n_blocks,
    divisor,
    table_ptr,
    stride_table,
    counts_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    src_row = src_ptr + row.to(tl.int64) * stride_src
    table_row = table_ptr + row.to(tl.int64) * stride_table
    count = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(0, n_blocks, BLOCK):
        cols = start + offs
        mask = cols < n_blocks
        encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
        count += (encoded >= 0).to(tl.int64)
        decoded = tl.maximum(encoded, 0) // divisor
        tl.store(table_row + cols, decoded.to(tl.int32), mask=mask)
    tl.store(counts_ptr + row, tl.sum(count, axis=0))


def decode_draft_block_table(
    src: torch.Tensor, divisor: int, table: torch.Tensor, counts: torch.Tensor
) -> None:
    """``counts[i] = (src[i] >= 0).sum()``; ``table[i] = src[i].clamp(min=0) // divisor``.

    Args:
        src: ``[rows, blocks]`` encoded block offsets.
        divisor: encoded offset -> pool block index divisor.
        table: ``[rows, blocks]`` int32 output (unit inner stride).
        counts: ``[rows]`` int64 output.
    """
    rows, n_blocks = src.shape
    assert table.shape[-1] == n_blocks and table.stride(-1) == 1
    if rows == 0:
        return
    _decode_block_table_kernel[(rows,)](
        src,
        src.stride(0),
        src.stride(1),
        n_blocks,
        int(divisor),
        table,
        table.stride(0),
        counts,
        BLOCK=_BLOCK_TABLE_BLOCK,
        num_warps=8,
    )
