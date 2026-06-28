# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU unit tests for the NVBug 6280721 sparse-index guardrail.

These tests pin the behaviour of ``mask_indices_outside_pool``: any global
KV-pool index that lands outside ``[0, pool_num_rows)`` must be replaced by the
``-1`` skip sentinel, while in-range indices and pre-existing ``-1`` entries are
left untouched. Pure tensor logic -- no GPU, no model, no server.
"""

import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    mask_indices_outside_pool)

POOL = 100  # valid pool row indices are 0..99


def test_in_range_unchanged():
    idx = torch.tensor([[0, 5, 99]], dtype=torch.int32)
    assert torch.equal(mask_indices_outside_pool(idx, POOL), idx)


def test_out_of_range_to_minus_one():
    idx = torch.tensor([[100, 101, 9999]], dtype=torch.int32)
    assert torch.equal(mask_indices_outside_pool(idx, POOL),
                       torch.full_like(idx, -1))


def test_boundary_is_exclusive():
    # row index == pool_num_rows is already out of bounds (rows are 0..N-1).
    out = mask_indices_outside_pool(torch.tensor([[99, 100]], dtype=torch.int32),
                                    POOL)
    assert out.tolist() == [[99, -1]]


def test_existing_minus_one_preserved():
    # -1 entries produced by the convert kernel must survive untouched.
    out = mask_indices_outside_pool(
        torch.tensor([[-1, 3, -1]], dtype=torch.int32), POOL)
    assert out.tolist() == [[-1, 3, -1]]


def test_mixed_shape_and_dtype_preserved():
    idx = torch.tensor([[-1, 50, 100], [0, 200, 99]], dtype=torch.int32)
    out = mask_indices_outside_pool(idx, POOL)
    assert out.tolist() == [[-1, 50, -1], [0, -1, 99]]
    assert out.dtype == torch.int32
    assert out.shape == idx.shape


def test_does_not_mutate_input():
    idx = torch.tensor([[100, 1]], dtype=torch.int32)
    before = idx.clone()
    _ = mask_indices_outside_pool(idx, POOL)
    assert torch.equal(idx, before)


def test_all_in_range_noop_large():
    idx = torch.arange(POOL, dtype=torch.int32).reshape(10, 10)
    assert torch.equal(mask_indices_outside_pool(idx, POOL), idx)
