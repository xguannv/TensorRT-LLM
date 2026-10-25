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
"""``TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS`` (``kimi_k3_nvfp4/flags.py``): the Helix MLA split workspace sized for the
budget's largest per-request split count on the FLAT grid, so ``decode_mla_dcp._var_split_envelope`` no longer caps
the count at the static split's units. One rank's V-up split read on one GPU (helpers of
``test_modeling_v2_kimi_k3_mla_var_split``):

* the default (-1) and its capability tuple;
* the envelope: ``workspace_bytes`` records the units and adds their bytes; with them a 4-request MTP7 batch takes up
  to ``budget - 3`` splits (9 on GB300, was 6), not without the allocation, with the switch at 0, or past the FLAT
  grid's rows;
* equal-length batches keep their counts: V rows and softmax statistics bit for bit the switch-off call;
* one long request among short ones takes more splits: counts as the python reference's on the wider grid, the V
  rows against fp32 attention x fp32 W_UV within the static split's error bound, eager and replayed from a CUDA
  graph the same bits; the census line counted only where the grid widened.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
gpu = pytest.mark.skipif(not _GPU, reason="the fused V up-projection runs on SM100/SM103 (tcgen05)")

CASES = {
    # (global lengths, q_len, rank): Helix16 one-rank shares of AgentX-like bs4 / bs3 steps
    "bs4_one_long": ([409600, 40960, 40960, 40960], 8, 3),
    "bs4_one_long_32k": ([409600, 32768, 32768, 32768], 8, 5),
    "bs3_one_long": ([409600, 40960, 40960], 8, 1),
    "bs4_equal": ([153600] * 4, 8, 2),
}


def _var():
    import test_modeling_v2_kimi_k3_mla_var_split as var

    return var


def _dcp():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
    )

    return decode_mla_dcp


def test_defaults():
    dcp = _dcp()
    assert dcp._flags._FLAGS["TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS"][1] == -1
    assert dcp.MLA_VAR_SPLIT_WIDE_WS_CAPABILITIES == ((10, 3),)


@gpu
def test_envelope(monkeypatch):
    torch.zeros(
        1, device="cuda"
    )  # a CUDA context first: the envelope queries the device's occupancy
    dcp = _dcp()
    heads = _var()._vu().HEADS
    monkeypatch.setattr(dcp, "_WIDE_WS_UNITS", {})
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT", 1)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 0)
    base_bytes = dcp.workspace_bytes(heads, dcp.MAX_Q_LEN, dcp.MAX_REQUESTS)
    budget, static_grid = dcp._var_split_envelope(4, heads, 8)
    assert not dcp._WIDE_WS_UNITS
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 1)
    # no allocation yet: nothing widens
    assert dcp._var_split_envelope(4, heads, 8) == (budget, static_grid)
    wide_bytes = dcp.workspace_bytes(heads, dcp.MAX_Q_LEN, dcp.MAX_REQUESTS)
    units = dcp._WIDE_WS_UNITS[(heads, 8)]
    q_tiles = -(-heads * 8 // 128)
    assert units == max(
        b * q_tiles * min(dcp._MAX_VAR_SPLIT, budget - b + 1)
        for b in range(2, dcp._max_fused_rows() // 8 + 1)
    )
    assert wide_bytes - base_bytes == (units - dcp._static_units(heads)) * 128 * (512 + 1) * 4
    grid4 = dcp._var_split_envelope(4, heads, 8)[1]
    assert grid4 == min(dcp._MAX_VAR_SPLIT, budget - 3) > static_grid
    # past the FLAT grid's rows (5 requests x 8 = 40 rows): unchanged
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 0)
    g5_off = dcp._var_split_envelope(5, heads, 8)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 1)
    assert dcp._var_split_envelope(5, heads, 8) == g5_off
    # FLAT off: unchanged
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT", 0)
    assert dcp._var_split_envelope(4, heads, 8) == (budget, static_grid)
    print(f"bs4 MTP7: units {dcp._static_units(heads)} -> {units}, grid {static_grid} -> {grid4}, "
          f"+{(wide_bytes - base_bytes) / 2**20:.1f} MiB")  # fmt: skip


def _run(var, dcp, case, wide, w=None):
    vu, dcp, _, pool, _, _, q, plan = var._setup(case)
    if w is None:
        w, _, _ = vu._w_uv(1, q.device)
    run = var._Run(vu, dcp, q, pool, plan, w, True)
    v, s = (t.clone() for t in run())
    torch.cuda.synchronize()
    grid = dcp.var_split_grid(q.shape[0], vu.HEADS, q.shape[1])
    counts = dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q.shape[1]).tolist()
    return vu, plan, q, run, w, v, s, grid, counts


@gpu
def test_equal_lengths_bitwise(monkeypatch):
    var, dcp = _var(), _dcp()
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT", 1)
    out = {}
    for wide in (0, 1):
        monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", wide)
        out[wide] = _run(var, dcp, CASES["bs4_equal"], wide)
    (_, plan, q, _, _, v0, s0, g0, c0), (_, _, _, _, _, v1, s1, g1, c1) = out[0], out[1]
    assert g1 > g0 and c0 == c1, (g0, g1, c0, c1)
    keyed = ~var._keyless(plan, q.shape[1])
    assert torch.equal(v1[keyed].view(torch.int16), v0[keyed].view(torch.int16))
    assert torch.equal(s1[keyed], s0[keyed])


@gpu
@pytest.mark.parametrize("name", ["bs4_one_long", "bs4_one_long_32k", "bs3_one_long"])
def test_one_long_counts_and_replay(monkeypatch, name):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import census

    var, dcp = _var(), _dcp()
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT", 1)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 0)
    *_, g_off, c_off = _run(var, dcp, CASES[name], 0)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 1)
    monkeypatch.setattr(census, "ENABLED", True)
    monkeypatch.setattr(census, "_device_tallies", {})
    census._counts.clear()
    try:
        vu, plan, q, run, w, v_eager, s_eager, grid, counts = _run(var, dcp, CASES[name], 1)
        seen = dict(census.counts())
        tally = census._device_tally_text()
    finally:
        census.ENABLED = False
    import re

    above = re.search(r"device:mla_var_split_counts:wide_ws_above_static=(\d+)", tally)
    assert above is not None and int(above.group(1)) >= 1, tally
    assert grid > g_off and max(counts) > max(c_off), (g_off, grid, c_off, counts)
    budget = dcp._var_split_envelope(q.shape[0], vu.HEADS, q.shape[1])[0]
    assert counts == var._ref_counts(plan.cache_seqs.tolist(), budget, grid)
    assert seen.get("dispatch:mla_var_split_wide_ws:eager", 0) >= 1, seen
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    run.v.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    keyed = ~var._keyless(plan, q.shape[1])
    assert torch.equal(run.v[keyed].view(torch.int16), v_eager[keyed].view(torch.int16))
    assert torch.equal(run.stats[keyed], s_eager[keyed])
    print(f"[{name}] splits {c_off} / grid {g_off} -> {counts} / grid {grid}; replay bit for bit")


@gpu
@pytest.mark.parametrize("seq_lens", [[25600, 2560, 2560, 2560], [25600, 2048, 2048, 2048]])
def test_one_long_against_fp32(monkeypatch, seq_lens):
    """WIDE_WS at 4 requests, one long (one rank of 400K + 3 x 40K / 32K under Helix16, whole sequences on one rank):
    the V rows against fp32 attention x fp32 W_UV within the static split's error bound."""
    var, dcp = _var(), _dcp()
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT", 1)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS", 1)
    var.test_on_against_fp32(seq_lens)
