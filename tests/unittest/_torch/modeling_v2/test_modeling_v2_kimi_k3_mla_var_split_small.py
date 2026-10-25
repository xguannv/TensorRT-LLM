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
"""``TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL`` and ``TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY`` (``kimi_k3_nvfp4/flags.py``):
per-request split-KV counts (``TRTLLM_K3LL_MLA_VAR_SPLIT``) from 2 requests on, and the split kernel's clusters past
their request's count exiting right after the PDL wait. One rank's Helix V-up split read on one GPU (the helpers of
``test_modeling_v2_kimi_k3_mla_var_split``):

* the fewest requests that take counts: 2 with SMALL on, 4 off; the default (-1) on only on
  ``decode_mla_dcp.MLA_VAR_SPLIT_SMALL_CAPABILITIES`` (SM103; the capability is patched, no GPU needed);
* SMALL at 2 requests: the counts are the python reference's; equal lengths take the static split's count, and the V
  rows and softmax statistics are bit for bit the switch-off call (the static split: same boundaries);
* at 2 unequal requests, the V rows against fp32 attention x fp32 W_UV within the static split's error bound;
* SKIP_EMPTY on against off, with counts below the grid's split count (mixed lengths at 2 and 4 requests, so empty
  clusters): bit for bit (the empty clusters write nothing the reduction reads), eager and replayed from a CUDA graph.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
gpu = pytest.mark.skipif(not _GPU, reason="the fused V up-projection runs on SM100/SM103 (tcgen05)")

CASES = {
    # (global lengths, q_len, rank): AgentX-like long / short pairs and mixes, scaled to one rank of 16
    "bs2_mixed": ([409600, 32768], 8, 3),
    "bs2_mixed_close": ([409600, 153600], 8, 5),
    "bs2_equal": ([153600, 153600], 8, 3),
    "bs4_mixed": ([32768, 409600, 32768, 409600], 8, 7),
    "bs4_one_long": ([32768, 32768, 32768, 409600], 8, 2),
}


def _var():
    import test_modeling_v2_kimi_k3_mla_var_split as var

    return var


def _dcp():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
    )

    return decode_mla_dcp


@pytest.mark.parametrize(
    "value,capability,min_batch",
    [
        (1, (10, 3), 2),
        (1, (10, 7), 2),
        (0, (10, 3), 4),
        (0, (10, 7), 4),
        (-1, (10, 3), 2),
        (-1, (10, 7), 4),
        (-1, (10, 0), 4),
    ],
)
def test_small_value(monkeypatch, value, capability, min_batch):
    dcp = _dcp()
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    dcp._var_split_small_by_default.cache_clear()
    try:
        assert dcp._var_split_min_batch() == min_batch
    finally:
        dcp._var_split_small_by_default.cache_clear()


def test_defaults():
    dcp = _dcp()
    assert dcp._flags._FLAGS["TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL"][1] == -1
    assert dcp.MLA_VAR_SPLIT_SMALL_CAPABILITIES == ((10, 3),)


@gpu
@pytest.mark.parametrize("name", ["bs2_mixed", "bs2_mixed_close", "bs2_equal"])
def test_small_counts(monkeypatch, name):
    var = _var()
    vu, dcp, _, _, _, _, q, plan = var._setup(CASES[name])
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", 1)
    batch, q_len = q.shape[:2]
    budget, grid = dcp._var_split_envelope(batch, vu.HEADS, q_len)
    counts = dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q_len)
    assert counts is not None and dcp.var_split_grid(batch, vu.HEADS, q_len) == grid > 1
    got = counts.tolist()
    assert got == var._ref_counts(plan.cache_seqs.tolist(), budget, grid)
    if len(set(CASES[name][0])) == 1:
        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner as Runner,
        )

        q_tiles = -(-vu.HEADS * q_len // 128)
        static = Runner.get_default_split_kv(batch, q_tiles, Runner._get_max_active_blocks())
        assert got == [static] * batch
    print(
        f"[{name}] local lengths {plan.cache_seqs.tolist()} -> splits {got} (grid {grid}, budget {budget})"
    )


@gpu
def test_small_equal_is_static_split(monkeypatch):
    var = _var()
    vu, dcp, _, pool, _, _, q, plan = var._setup(CASES["bs2_equal"])
    w, _, _ = vu._w_uv(1, q.device)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", 0)
    v_off, s_off = (t.clone() for t in var._Run(vu, dcp, q, pool, plan, w, True)())
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", 1)
    assert dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q.shape[1]) is not None
    v_on, s_on = (t.clone() for t in var._Run(vu, dcp, q, pool, plan, w, True)())
    torch.cuda.synchronize()
    keyed = ~var._keyless(plan, q.shape[1])
    assert torch.equal(v_on[keyed].view(torch.int16), v_off[keyed].view(torch.int16))
    assert torch.equal(s_on[keyed], s_off[keyed])


@gpu
@pytest.mark.parametrize("name", ["bs2_mixed", "bs2_mixed_close", "bs4_mixed", "bs4_one_long"])
def test_skip_empty_bitwise(monkeypatch, name):
    var = _var()
    vu, dcp, _, pool, _, _, q, plan = var._setup(CASES[name])
    w, _, _ = vu._w_uv(1, q.device)
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", 1)
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
        graph.replay()
        torch.cuda.synchronize()
        outs[skip] = (eager, [run.v.clone(), run.stats.clone()])
    keyed = ~var._keyless(plan, q.shape[1])
    for kind in (0, 1):  # eager, replayed
        (v0, s0), (v1, s1) = outs[0][kind], outs[1][kind]
        assert torch.equal(v1[keyed].view(torch.int16), v0[keyed].view(torch.int16)), (name, kind)
        assert torch.equal(s1[keyed], s0[keyed]), (name, kind)
    print(
        f"[{name}] splits {counts.tolist()} on grid {grid}: skip-empty bit for bit, eager and replayed"
    )


@gpu
@pytest.mark.parametrize(
    "seq_lens",
    [
        [25600, 2048],  # one rank of 400K / 32K under Helix16 (counts [1, 11])
        [43520, 2048],  # 680K / 32K
        [25600, 9600],  # 400K / 150K
    ],
)
def test_small_mixed_against_fp32(monkeypatch, seq_lens):
    """SMALL at 2 requests of unequal lengths (the whole sequence on one rank, the per-rank lengths of the Helix16
    cases): the V rows against fp32 attention x fp32 W_UV within ``FP32_OVER_OFF`` of the static split's error."""
    var = _var()
    dcp = _dcp()
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", 1)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    counts = dcp.var_split_counts(lens, var._vu().HEADS, 8)
    assert counts is not None and len(set(counts.tolist())) > 1, counts
    var.test_on_against_fp32(seq_lens)


@gpu
def test_census_tally(monkeypatch):
    """``TRTLLM_K3LL_CENSUS``: the count kernel tallies its calls and the calls with unequal counts on device, per
    2-3 request (SMALL) and 4+ request batches, also inside a replayed CUDA graph."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import census

    dcp = _dcp()
    monkeypatch.setattr(census, "ENABLED", True)
    monkeypatch.setattr(census, "_device_tallies", {})
    monkeypatch.setattr(dcp._flags, "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL", 1)
    heads = _var()._vu().HEADS
    mixed = torch.tensor([25600, 2048], dtype=torch.int32, device="cuda")
    equal = torch.tensor([9600, 9600], dtype=torch.int32, device="cuda")
    wide = torch.tensor([25600, 2048, 2048, 2048], dtype=torch.int32, device="cuda")
    for lens in (mixed, equal, wide):
        dcp.var_split_counts(lens, heads, 8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dcp.var_split_counts(mixed, heads, 8)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    tally = dcp._counts_tally(mixed.device).tolist()
    assert tally == [4, 3, 1, 1], (
        tally
    )  # small: 2 eager + 2 replays, 3 unequal; wide: 1 call, unequal
    assert "device:mla_var_split_counts:small_unequal=3" in census._device_tally_text()
