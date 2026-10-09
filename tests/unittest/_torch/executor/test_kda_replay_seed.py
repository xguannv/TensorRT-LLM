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
"""Batched seeding of the KDA replay conv caches for disaggregated generation."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import _seed_kda_replay_conv_batched

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _layout(num_layers, num_slots, section_dims, committed, num_spec, conv_dtype):
    conv_dim = sum(section_dims)
    # Per-layer conv states are strided views into one pool with padded slot pages,
    # like the V2 manager's page-backed state buffers.
    page = conv_dim * committed + 64
    pool = torch.randn(num_layers, num_slots, page, device="cuda").to(conv_dtype)
    conv_states = [
        pool[layer].as_strided((num_slots, conv_dim, committed), (page, committed, 1))
        for layer in range(num_layers)
    ]
    buffers = tuple(
        torch.randn(num_layers, num_slots, committed + num_spec, dim, device="cuda")
        .to(torch.bfloat16)
        .transpose(-1, -2)
        for dim in section_dims
    )
    offsets = [0]
    for dim in section_dims:
        offsets.append(offsets[-1] + dim)
    return conv_states, buffers, offsets


def _reference(conv_states, buffers, offsets, state_indices, committed):
    for layer, conv_state in enumerate(conv_states):
        selected = conv_state.index_select(0, state_indices)
        for section, buffer in enumerate(buffers):
            replay_layer = buffer[layer]
            replay_layer.index_fill_(0, state_indices, 0)
            part = selected[:, offsets[section] : offsets[section + 1], :]
            replay_layer[:, :, :committed].index_copy_(
                0, state_indices, part.to(replay_layer.dtype)
            )


@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("slots", [[3], [0, 5, 7], list(range(0, 16, 2))])
def test_batched_seed_matches_per_layer_loop(conv_dtype, slots):
    torch.manual_seed(0)
    num_layers, num_slots, committed, num_spec = 5, 16, 3, 6
    section_dims = [384, 384, 256]
    conv_states, buffers, offsets = _layout(
        num_layers, num_slots, section_dims, committed, num_spec, conv_dtype
    )
    expected = tuple(b.clone() for b in buffers)
    idx = torch.tensor(slots, dtype=torch.long, device="cuda")
    _reference(conv_states, expected, offsets, idx, committed)
    ptrs = torch.tensor([s.data_ptr() for s in conv_states], dtype=torch.int64, device="cuda")
    assert _seed_kda_replay_conv_batched(conv_states, ptrs, idx, buffers, offsets, committed)
    for got, want in zip(buffers, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_mismatched_layouts_fall_back():
    conv_states, buffers, offsets = _layout(2, 4, [128, 128, 128], 3, 6, torch.bfloat16)
    conv_states[1] = conv_states[1].contiguous()
    ptrs = torch.tensor([s.data_ptr() for s in conv_states], dtype=torch.int64, device="cuda")
    idx = torch.tensor([1], dtype=torch.long, device="cuda")
    assert not _seed_kda_replay_conv_batched(conv_states, ptrs, idx, buffers, offsets, 3)
