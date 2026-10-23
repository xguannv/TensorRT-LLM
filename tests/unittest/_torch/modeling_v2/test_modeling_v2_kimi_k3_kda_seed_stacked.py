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
"""``TRTLLM_K3LL_KDA_SEED_STACKED`` (``MambaHybridCacheManagerV2._seed_kda_replay_slots``): switch on = switch off, bit
for bit, on every buffer the seed writes (the three replay conv caches, the draft records, prev_num_accepted_tokens)
and on the conv pool it reads, for K3-like shapes (69 KDA layers, conv sections q / k / v, window 3 + 7 draft columns),
1 / 2 / 4 / 8 new slots, an affine pool (stacked view) and a non-affine one (falls back to the per-layer loop). Runs on
the GPU when there is one, else on the CPU.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags
from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager as mcm

LAYERS, SLOTS, D, W, SPEC = 69, 12, 32, 3, 7


def _manager(seed, affine):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = 3 * D
    m = object.__new__(mcm.MambaHybridCacheManagerV2)
    if (
        affine
    ):  # per-layer views into one pool, base pointers spaced by a constant (as V2 coalesces them)
        pool = torch.randn((SLOTS, LAYERS, c, W), generator=g).to(dev, torch.bfloat16)
        m.all_conv_states = [pool[:, i] for i in range(LAYERS)]
    else:  # per-layer views at irregular offsets of one buffer (separate allocations can come out evenly spaced)
        chunk = SLOTS * c * W
        flat = torch.randn((LAYERS * (chunk + 64),), generator=g).to(dev, torch.bfloat16)
        offs = [i * chunk + (i * i) % 61 for i in range(LAYERS)]
        m.all_conv_states = [
            flat.as_strided((SLOTS, c, W), (c * W, W, 1), storage_offset=o) for o in offs
        ]
    m._stacked_conv_states = mcm.MambaHybridCacheManagerV2._stack_state_views(m.all_conv_states)
    assert (m._stacked_conv_states is not None) == affine
    m.conv_state_shape = (c, W)
    m.conv_section_dims = [D, D, D]
    for name in ("kda_conv_q", "kda_conv_k", "kda_conv_v"):
        setattr(
            m, name, torch.randn((LAYERS, SLOTS, D, W + SPEC), generator=g).to(dev, torch.bfloat16)
        )
    m.kda_qkg_cache = torch.randn((LAYERS, SLOTS, SPEC, 8), generator=g).to(dev)
    m.kda_v_cache = torch.randn((LAYERS, SLOTS, SPEC, 4), generator=g).to(dev)
    m.kda_beta_cache = torch.randn((LAYERS, SLOTS, SPEC), generator=g).to(dev)
    m.prev_num_accepted_tokens = torch.randint(0, 8, (SLOTS,), generator=g).to(dev, torch.int32)
    return m


def _state(m):
    out = [x.clone() for x in m.all_conv_states]
    out += [
        getattr(m, n).clone()
        for n in (
            "kda_conv_q",
            "kda_conv_k",
            "kda_conv_v",
            "kda_qkg_cache",
            "kda_v_cache",
            "kda_beta_cache",
            "prev_num_accepted_tokens",
        )
    ]
    return out


@pytest.mark.parametrize("affine", (True, False))
@pytest.mark.parametrize("slots", ([3], [0, 5], [1, 4, 7, 11], [0, 1, 2, 3, 4, 5, 6, 7]))
def test_seed_stacked_equals_per_layer(monkeypatch, affine, slots):
    got = {}
    for v in (0, 1):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_KDA_SEED_STACKED", v)
        m = _manager(7, affine)
        before = _state(m)
        calls = mcm._KDA_SEED_STACKED_CALLS[0]
        idx = torch.tensor(slots, dtype=torch.long, device=m.prev_num_accepted_tokens.device)
        m._seed_kda_replay_slots(idx)
        got[v] = _state(m)
        stacked_ran = mcm._KDA_SEED_STACKED_CALLS[0] > calls
        assert stacked_ran == (v == 1 and affine)
        # the seed changed the selected slots (the test exercises something)
        assert not torch.equal(before[LAYERS], got[v][LAYERS])
    for a, b in zip(got[0], got[1]):
        assert a.dtype == b.dtype and torch.equal(a, b)
