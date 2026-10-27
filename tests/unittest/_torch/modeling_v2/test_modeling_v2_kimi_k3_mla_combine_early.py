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
"""``TRTLLM_K3LL_MLA_COMBINE_EARLY`` (``kimi_k3_nvfp4/flags.py``): the fused V-up reduction's combine warps issue the
LSE and first partial loads before the split count is read. One Helix16 rank's V-up split read on one GPU (helpers of
``test_modeling_v2_kimi_k3_mla_var_split``), at the shipped decode shapes (8 / 12 / 14 / 16 rows) and short ones
(fewer local splits than the early loads, a request with no page on the rank):

* the default (0) and its values;
* switch on (2 / 4 / 8 early splits): the V rows and the softmax statistics bit for bit the switch-off ones, with
  FP32 and with FP16 partials (``TRTLLM_K3LL_MLA_PARTIAL_F16``), a build of its own;
* switch on, replayed from a CUDA graph with new queries: the eager call's bits.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused V up-projection runs on SM100/SM103 (tcgen05)",
)

CASES = {
    # (global lengths, q_len, rank): Helix16 one-rank shares
    "bs1_d7_8rows": ([153600 + 8], 8, 5),
    "bs3_d3_12rows": ([153600 + 4] * 3, 4, 2),
    "bs2_d6_14rows": ([153600 + 7, 120000], 7, 3),
    "bs4_d3_16rows": ([153600 + 4, 90000, 40000, 153600], 4, 7),
    "bs4_short": ([60000, 50, 300, 20000], 4, 7),  # 50: below one K tile; 300: no page on rank 7
    "bs2_short": ([3000, 1200], 8, 1),  # fewer local splits than COMBINE_EARLY_SPLITS
}


def _var():
    import test_modeling_v2_kimi_k3_mla_var_split as var

    return var


def _set(monkeypatch, name, value):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    monkeypatch.setattr(flags, name, value)


def test_defaults():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    assert flags._FLAGS["TRTLLM_K3LL_MLA_COMBINE_EARLY"][1] == 0
    assert flags.accepted_values()["TRTLLM_K3LL_MLA_COMBINE_EARLY"] == (0, 2, 4, 8)


@pytest.mark.parametrize("early", [2, 4, 8])
@pytest.mark.parametrize("partial_f16", [0, 1])
@pytest.mark.parametrize("name", list(CASES))
def test_on_is_off_bitwise(name, partial_f16, early, monkeypatch):
    var = _var()
    vu, dcp, mla_v_up, pool, _, _, q, plan = var._setup(CASES[name], seed=11)
    w, _, _ = vu._w_uv(1, q.device)
    _set(monkeypatch, "TRTLLM_K3LL_MLA_PARTIAL_F16", partial_f16)
    run = var._Run(vu, dcp, q, pool, plan, w, True)
    out = {}
    for value in (0, early, 0):
        _set(monkeypatch, "TRTLLM_K3LL_MLA_COMBINE_EARLY", value)
        run.v.fill_(float("nan"))
        run.stats.fill_(float("nan"))
        v, s = run()
        torch.cuda.synchronize()
        out.setdefault(value, []).append((v.clone(), s.clone()))
    assert any(f"combine_early{early}" in k for k in mla_v_up._compiled), "no switch-on build"
    keyed = ~var._keyless(plan, q.shape[1])
    (v_off, s_off), (v_off2, s_off2) = out[0]
    (v_on, s_on) = out[early][0]
    assert not torch.isnan(v_on[keyed]).any()
    for v, s in ((v_on, s_on), (v_off2, s_off2)):
        assert torch.equal(v[keyed].view(torch.int16), v_off[keyed].view(torch.int16)), name
        assert torch.equal(s[keyed], s_off[keyed]), name
    keyless = var._keyless(plan, q.shape[1])
    if bool(keyless.any()):
        assert torch.equal(s_on[keyless], s_off[keyless])


@pytest.mark.parametrize("early", [4, 8])
@pytest.mark.parametrize("name", ["bs1_d7_8rows", "bs4_d3_16rows", "bs4_short"])
def test_on_graph_replay(name, early, monkeypatch):
    var = _var()
    vu, dcp, _, pool, _, _, q, plan = var._setup(CASES[name], seed=11)
    w, _, _ = vu._w_uv(1, q.device)
    _set(monkeypatch, "TRTLLM_K3LL_MLA_COMBINE_EARLY", early)
    run = var._Run(vu, dcp, q.clone(), pool, plan, w, True)
    run()  # compiles eagerly
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    seq_lens, q_len, _ = CASES[name]
    _, _, q_new = vu._cache(seq_lens, q_len, 23, q.device)
    run.q.copy_(q_new)
    run.v.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replayed, replayed_stats = run.v.clone(), run.stats.clone()
    v_eager, s_eager = var._Run(vu, dcp, run.q, pool, plan, w, True)()
    torch.cuda.synchronize()
    keyed = ~var._keyless(plan, q_len)
    assert torch.equal(replayed[keyed].view(torch.int16), v_eager[keyed].view(torch.int16))
    assert torch.equal(replayed_stats[keyed], s_eager[keyed])
