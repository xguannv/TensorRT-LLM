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
"""``TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY`` (``kimi_k3_nvfp4/flags.py``): with per-request split-KV counts
(``TRTLLM_K3LL_MLA_VAR_SPLIT``) the split kernel's grid is sized for the largest count, and the clusters past their
request's count take no warp role and exit right after the PDL wait. One rank's Helix V-up split read on one GPU (the
helpers of ``test_modeling_v2_kimi_k3_mla_var_split``):

* the value: 1 on, 0 off, the default (-1) on only on ``mla_v_up.MLA_VAR_SPLIT_SKIP_EMPTY_CAPABILITIES`` (SM103; the
  capability is patched, no GPU needed);
* on against off where counts sit below the grid's split count (empty clusters: mixed lengths, and equal lengths at
  half the grid): the V rows and softmax statistics bit for bit, eager and replayed from a CUDA graph (counts and
  split read captured together).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))

CASES = {
    # (global lengths, q_len, rank): one rank of 16, 400K / 150K / 32K-class AgentX contexts
    "bs4_mixed": ([32768, 409600, 32768, 409600], 8, 7),
    "bs4_one_long": ([32768, 32768, 32768, 409600], 8, 2),
    "bs4_spread": ([32768, 153600, 409600, 65536], 8, 5),
    "bs4_equal": ([153600] * 4, 8, 3),
    "bs3_mixed": ([409600, 32768, 153600], 8, 1),
}


def _mla_v_up():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_v_up,
    )

    return mla_v_up


@pytest.mark.parametrize(
    "value,capability,on",
    [
        (1, (10, 3), True),
        (1, (10, 7), True),
        (0, (10, 3), False),
        (0, (10, 7), False),
        (-1, (10, 3), True),
        (-1, (10, 7), False),
        (-1, (10, 0), False),
    ],
)
def test_skip_empty_value(monkeypatch, value, capability, on):
    m = _mla_v_up()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    m._skip_empty_by_default.cache_clear()
    try:
        assert m.var_split_skip_empty(torch.device("cuda", 0)) is on
    finally:
        m._skip_empty_by_default.cache_clear()


def test_default_is_auto():
    m = _mla_v_up()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    assert flags._FLAGS["TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY"][1] == -1
    assert m.MLA_VAR_SPLIT_SKIP_EMPTY_CAPABILITIES == ((10, 3),)


@pytest.mark.skipif(not _GPU, reason="the fused V up-projection runs on SM100/SM103 (tcgen05)")
@pytest.mark.parametrize("name", list(CASES))
def test_on_is_off_bitwise(monkeypatch, name):
    import test_modeling_v2_kimi_k3_mla_var_split as var

    vu, dcp, _, pool, _, _, q, plan = var._setup(CASES[name])
    w, _, _ = vu._w_uv(1, q.device)
    counts = dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q.shape[1])
    grid = dcp.var_split_grid(q.shape[0], vu.HEADS, q.shape[1])
    assert counts is not None and min(counts.tolist()) < grid, "no empty cluster in this case"
    outs = {}
    for skip in (0, 1):
        monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY", skip)
        run = var._Run(vu, dcp, q.clone(), pool, plan, w, True)
        eager = [t.clone() for t in run()]
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        run.v.fill_(float("nan"))
        run.stats.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        outs[skip] = (eager, [run.v.clone(), run.stats.clone()])
    keyed = ~var._keyless(plan, q.shape[1])
    for kind, label in ((0, "eager"), (1, "replayed")):
        (v0, s0), (v1, s1) = outs[0][kind], outs[1][kind]
        assert not torch.isnan(v1[keyed]).any(), (name, label)
        assert torch.equal(v1[keyed].view(torch.int16), v0[keyed].view(torch.int16)), (name, label)
        assert torch.equal(s1[keyed], s0[keyed]), (name, label)
    print(
        f"[{name}] splits {counts.tolist()} on grid {grid}: skip-empty bit for bit, eager and replayed"
    )
