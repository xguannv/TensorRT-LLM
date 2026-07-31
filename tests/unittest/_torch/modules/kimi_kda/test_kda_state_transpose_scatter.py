# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused KDA recurrent-state transpose-scatter kernel."""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda._kda_state import (  # noqa: E402
    kda_state_transpose_scatter,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="KDA state transpose-scatter requires CUDA",
)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_kda_state_transpose_scatter_layout_and_slots(index_dtype: torch.dtype) -> None:
    num_seqs, num_heads, head_k_dim, head_v_dim = 3, 3, 17, 13
    num_slots = 7
    slot_indices = torch.tensor([5, 1, 6], dtype=index_dtype, device="cuda")

    values = torch.arange(
        num_seqs * num_heads * head_k_dim * head_v_dim,
        dtype=torch.float32,
        device="cuda",
    )
    k_first_state = values.reshape(
        num_seqs,
        num_heads,
        head_k_dim,
        head_v_dim,
    )

    # Select one plane from a wider allocation so the pool has a non-compact
    # slot stride, matching cache managers that embed layer state in a larger
    # per-slot envelope.
    sentinel = -12345.0
    pool_storage = torch.full(
        (num_slots, 2, num_heads, head_v_dim, head_k_dim),
        sentinel,
        dtype=torch.float32,
        device="cuda",
    )
    state_pool = pool_storage[:, 1]
    assert state_pool.stride(0) != num_heads * head_v_dim * head_k_dim

    expected = state_pool.clone()
    expected.index_copy_(
        0,
        slot_indices.long(),
        k_first_state.transpose(-1, -2).contiguous(),
    )

    kda_state_transpose_scatter(k_first_state, state_pool, slot_indices)

    torch.testing.assert_close(state_pool, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        pool_storage[:, 0],
        torch.full_like(pool_storage[:, 0], sentinel),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_kda_state_transpose_scatter_cuda_graph() -> None:
    num_seqs, num_heads, head_dim = 2, 4, 128
    slot_indices = torch.tensor([3, 0], dtype=torch.int64, device="cuda")
    k_first_state = torch.randn(
        num_seqs,
        num_heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    state_pool = torch.zeros(
        5,
        num_heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )

    # Compile the Triton kernel before capture.
    kda_state_transpose_scatter(k_first_state, state_pool, slot_indices)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kda_state_transpose_scatter(k_first_state, state_pool, slot_indices)

    k_first_state.copy_(torch.randn_like(k_first_state))
    state_pool.fill_(-1)
    graph.replay()

    expected = torch.full_like(state_pool, -1)
    expected.index_copy_(
        0,
        slot_indices,
        k_first_state.transpose(-1, -2).contiguous(),
    )
    torch.testing.assert_close(state_pool, expected, rtol=0, atol=0)
