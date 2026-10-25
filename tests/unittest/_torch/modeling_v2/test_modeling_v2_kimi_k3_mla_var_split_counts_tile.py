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
"""``TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE`` (``kimi_k3_nvfp4/flags.py``): the per-request split-KV counts of
``TRTLLM_K3LL_MLA_VAR_SPLIT`` from one Triton program that scores every (request, k) candidate at once
(``decode_mla_dcp._var_split_counts_tile_kernel``) are the sequential kernel's counts and the python reference's, on
random per-rank lengths (0 included: a request with no page on the rank) at 2 to 8 requests and at the grid of every
batch size (the kernels are launched directly, below the switch's batch floor too), and through ``var_split_counts``
with the switch on and off.
"""

import os
import sys

import pytest
import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")


def _dcp():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
    )

    torch.zeros(
        1, device="cuda"
    )  # a current CUDA context for the occupancy query of the split envelope
    return decode_mla_dcp


def _launch(dcp, tile, lens, grid, budget):
    batch = lens.shape[0]
    out = torch.empty(batch, dtype=torch.int32, device=lens.device)
    args = dict(
        B=batch, BLOCK_B=triton.next_power_of_2(batch), GRID=grid, BUDGET=budget, K_TILE=dcp._K_TILE
    )
    if tile:
        dcp._var_split_counts_tile_kernel[(1,)](
            lens, out, BLOCK_C=triton.next_power_of_2(batch * grid), **args
        )
    else:
        dcp._var_split_counts_kernel[(1,)](lens, out, **args)
    return out.tolist()


@pytest.mark.parametrize("batch", range(2, 9))
def test_tile_equals_sequential(batch):
    import test_modeling_v2_kimi_k3_mla_var_split as var

    dcp = _dcp()
    budget, grid = dcp._var_split_envelope(batch, 96, 8)
    if grid < 1:
        pytest.skip(f"no split grid at {batch} requests")
    g = torch.Generator().manual_seed(100 + batch)
    for _ in range(300):
        lens = torch.randint(0, 60000, (batch,), generator=g)
        lens[torch.rand(batch, generator=g) < 0.1] = 0
        if bool(torch.rand(1, generator=g) < 0.2):
            lens[:] = lens[0]  # equal lengths
        d = lens.to(torch.int32).cuda()
        got = _launch(dcp, True, d, grid, budget)
        assert got == _launch(dcp, False, d, grid, budget), lens.tolist()
        assert got == var._ref_counts(lens.tolist(), budget, grid), lens.tolist()


@pytest.mark.parametrize("tile", [0, 1])
def test_var_split_counts_switch(monkeypatch, tile):
    import test_modeling_v2_kimi_k3_mla_var_split as var

    dcp = _dcp()
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE", tile)
    g = torch.Generator().manual_seed(7)
    for batch in (4, 5, 8):
        budget, grid = dcp._var_split_envelope(batch, 96, 8)
        for _ in range(50):
            lens = torch.randint(0, 60000, (batch,), generator=g)
            counts = dcp.var_split_counts(lens.to(torch.int32).cuda(), 96, 8)
            assert counts is not None
            assert counts.tolist() == var._ref_counts(lens.tolist(), budget, grid), lens.tolist()


def test_default_on():
    dcp = _dcp()
    assert dcp._flags._FLAGS["TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE"][1] == 1
