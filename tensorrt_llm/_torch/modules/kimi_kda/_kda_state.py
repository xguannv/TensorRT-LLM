# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi KDA recurrent-state layout and pool-write kernels."""

import torch
import triton
import triton.language as tl

_BLOCK_K = 64
_BLOCK_V = 64
_NUM_WARPS = 8


@triton.jit
def _kda_state_transpose_scatter_kernel(
    k_first_state_ptr,
    state_pool_ptr,
    slot_indices_ptr,
    num_slots,
    num_heads,
    head_k_dim,
    head_v_dim,
    src_stride_n,
    src_stride_h,
    src_stride_k,
    src_stride_v,
    dst_stride_slot,
    dst_stride_h,
    dst_stride_v,
    dst_stride_k,
    slot_stride,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Transpose one state tile while scattering its sequence to a pool slot."""
    nh_idx = tl.program_id(0)
    k_tile_idx = tl.program_id(1)
    v_tile_idx = tl.program_id(2)

    seq_idx = nh_idx // num_heads
    head_idx = nh_idx % num_heads
    slot_idx = tl.load(slot_indices_ptr + seq_idx * slot_stride).to(tl.int64)
    slot_mask = (slot_idx >= 0) & (slot_idx < num_slots)

    k_offsets = k_tile_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offsets = v_tile_idx * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offsets < head_k_dim
    v_mask = v_offsets < head_v_dim

    src_base = seq_idx.to(tl.int64) * src_stride_n + head_idx.to(tl.int64) * src_stride_h
    src_offsets = (
        src_base
        + k_offsets[:, None].to(tl.int64) * src_stride_k
        + v_offsets[None, :].to(tl.int64) * src_stride_v
    )
    state_tile = tl.load(
        k_first_state_ptr + src_offsets,
        mask=k_mask[:, None] & v_mask[None, :],
        other=0.0,
    )

    dst_base = slot_idx * dst_stride_slot + head_idx.to(tl.int64) * dst_stride_h
    dst_offsets = (
        dst_base
        + v_offsets[:, None].to(tl.int64) * dst_stride_v
        + k_offsets[None, :].to(tl.int64) * dst_stride_k
    )
    tl.store(
        state_pool_ptr + dst_offsets,
        tl.trans(state_tile),
        mask=slot_mask & v_mask[:, None] & k_mask[None, :],
    )


def _validate_kda_state_transpose_scatter(
    k_first_state: torch.Tensor,
    state_pool: torch.Tensor,
    slot_indices: torch.Tensor,
) -> None:
    if k_first_state.ndim != 4:
        raise ValueError(
            f"k_first_state must have shape [N, H, K, V], got {tuple(k_first_state.shape)}"
        )
    if state_pool.ndim != 4:
        raise ValueError(
            f"state_pool must have shape [slots, H, V, K], got {tuple(state_pool.shape)}"
        )
    if slot_indices.ndim != 1:
        raise ValueError(f"slot_indices must have shape [N], got {tuple(slot_indices.shape)}")

    num_seqs, num_heads, head_k_dim, head_v_dim = k_first_state.shape
    expected_pool_shape = (num_heads, head_v_dim, head_k_dim)
    if tuple(state_pool.shape[1:]) != expected_pool_shape:
        raise ValueError(
            "state_pool trailing shape must be [H, V, K] matching "
            f"k_first_state; expected {expected_pool_shape}, "
            f"got {tuple(state_pool.shape[1:])}"
        )
    if slot_indices.shape[0] != num_seqs:
        raise ValueError(f"slot_indices has {slot_indices.shape[0]} entries for {num_seqs} states")
    if k_first_state.dtype != torch.float32 or state_pool.dtype != torch.float32:
        raise TypeError(
            "KDA recurrent state transpose-scatter requires fp32 source and destination"
        )
    if slot_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"slot_indices must use torch.int32 or torch.int64, got {slot_indices.dtype}"
        )
    if k_first_state.device != state_pool.device or k_first_state.device != slot_indices.device:
        raise ValueError("source, state_pool, and slot_indices must be on the same device")


@torch.library.custom_op(
    "trtllm::kda_state_transpose_scatter",
    mutates_args=("state_pool",),
    device_types="cuda",
)
def kda_state_transpose_scatter(
    k_first_state: torch.Tensor,
    state_pool: torch.Tensor,
    slot_indices: torch.Tensor,
) -> None:
    """Write K-first states directly to indexed V-first state-pool slots.

    Args:
        k_first_state: Contiguous or strided fp32 states with shape
            ``[N, H, K, V]``.
        state_pool: Mutable fp32 pool with shape ``[slots, H, V, K]``.
        slot_indices: int32 or int64 destination slot for every input state,
            with shape ``[N]``. Runtime callers must provide unique, valid
            slot indices.
    """
    _validate_kda_state_transpose_scatter(k_first_state, state_pool, slot_indices)
    num_seqs, num_heads, head_k_dim, head_v_dim = k_first_state.shape
    if num_seqs == 0:
        return

    grid = (
        num_seqs * num_heads,
        triton.cdiv(head_k_dim, _BLOCK_K),
        triton.cdiv(head_v_dim, _BLOCK_V),
    )
    _kda_state_transpose_scatter_kernel[grid](
        k_first_state,
        state_pool,
        slot_indices,
        state_pool.shape[0],
        num_heads,
        head_k_dim,
        head_v_dim,
        *k_first_state.stride(),
        *state_pool.stride(),
        slot_indices.stride(0),
        BLOCK_K=_BLOCK_K,
        BLOCK_V=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )


@kda_state_transpose_scatter.register_fake
def _(
    k_first_state: torch.Tensor,
    state_pool: torch.Tensor,
    slot_indices: torch.Tensor,
) -> None:
    return None
