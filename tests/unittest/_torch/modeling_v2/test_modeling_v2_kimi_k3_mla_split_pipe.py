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
"""``TRTLLM_K3LL_MLA_SPLIT_PIPE`` (``kimi_k3_nvfp4/flags.py``): the Helix V-up split read
(``trtllm::k3_mla_decode_fp8_v_up``) with the split kernel's V tile loads under an L2 evict-first hint, one rank's
split read on one GPU.

A cache hint only, so every check is bit for bit against the switch at 0 (the trunk kernel):

* ``split_read_attention_v_up``'s V rows and softmax statistics at the K3 shapes (96 heads, MTP7 q_len 8, Helix CP 16,
  rank-local pages) for 2 and 4 requests at 64K / 150K / 300K / 500K tokens (4 requests with the
  ``TRTLLM_K3LL_MLA_VAR_SPLIT`` counts the model uses), and the 1 / 2 / 4-request cases of
  ``test_modeling_v2_kimi_k3_mla_v_up`` (a request this rank holds no page of included), eagerly and from a CUDA graph
  replayed with new queries; the on arm compiles its own ``split_pipe`` variant;
* the census counts ``mla_split_pipe`` once per fused call with the switch on, never with it off, and never on the
  unfused fallback (35 rows), which the switch leaves unchanged;
* the default (-1) is on on SM103 only (``decode_mla_dcp.MLA_SPLIT_PIPE_CAPABILITIES``).
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused V-up split read runs on SM100 / SM103",
)

REPLAYS = (21, 22)


def _vu():
    import test_modeling_v2_kimi_k3_mla_v_up as vu

    return vu


def _mods():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        census,
        flags,
    )

    dcp, mla_v_up, _ = _vu()._mods()
    return dcp, mla_v_up, flags, census


@contextlib.contextmanager
def _switch(monkeypatch, value):
    _, _, flags, census = _mods()
    with monkeypatch.context() as m:
        m.setattr(flags, "TRTLLM_K3LL_MLA_SPLIT_PIPE", int(value))
        m.setattr(census, "ENABLED", True)
        yield


def _census_count(census) -> int:
    return sum(v for k, v in census.counts().items() if k.startswith("mla_split_pipe:"))


def _bits(a, b) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    )


def _setup(seq_lens, q_len, rank, var_split):
    vu = _vu()
    dcp, _, _, _ = _mods()
    device = torch.device("cuda")
    pool, block_rows, q = vu._cache(seq_lens, q_len, 11, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    plan = dcp.make_split_read_plan(block_rows, lens, q_len, vu.PAGE, vu.CP, rank)
    if var_split:
        plan = dcp.with_var_split(plan, vu.HEADS, q_len)
    w, _, _ = vu._w_uv(1, device)
    return vu, dcp, pool, q, plan, w


# Global KV lengths (the step's 8 new tokens included) of the measured shapes, rank 5 of CP 16.
REAL = {
    f"bs{b}_{ctx // 1000}k": ([ctx + 8] * b, 8, 5, b >= 4)
    for b in (2, 4)
    for ctx in (65536, 153600, 307200, 512000)
}
# test_modeling_v2_kimi_k3_mla_v_up's cases (rank 7 holds no page of the 300-token request).
SMALL = {
    "bs1_mtp7": ([153600 + 8], 8, 5, False),
    "bs2_mtp7": ([153600 + 8, 70000 + 8], 8, 3, False),
    "bs4_mtp7": ([153600 + 8, 20000 + 8, 300, 5000], 8, 7, False),
}
CASES = {**REAL, **SMALL}


@pytest.mark.parametrize("case", list(CASES))
def test_on_equals_off(monkeypatch, case):
    seq_lens, q_len, rank, var_split = CASES[case]
    vu, dcp, pool, q, plan, w = _setup(seq_lens, q_len, rank, var_split)
    _, mla_v_up, _, census = _mods()
    if var_split:
        assert plan.block_splits is not None and plan.var_split > 1
    scale = dcp.default_softmax_scale()
    with _switch(monkeypatch, 0):
        before = _census_count(census)
        off_run = vu._VUp(dcp, q, pool, plan, w, scale)
        v_off, s_off = (t.clone() for t in off_run())
        assert _census_count(census) == before
    with _switch(monkeypatch, 1):
        before = _census_count(census)
        on_run = vu._VUp(dcp, q, pool, plan, w, scale)
        v_on, s_on = (t.clone() for t in on_run())
        assert _census_count(census) == before + 1
    torch.cuda.synchronize()
    keys = list(mla_v_up._compiled)
    assert any(k[-1] == "split_pipe" for k in keys) and any(k[-1] != "split_pipe" for k in keys)
    assert not torch.isnan(v_off).any()
    assert _bits(v_on, v_off), f"{case}: V rows differ with the switch on"
    assert _bits(s_on, s_off), f"{case}: softmax statistics differ with the switch on"

    # the on arm captured, replayed with new queries: bit for bit the off arm's eager call
    with _switch(monkeypatch, 1):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            on_run()
    for seed in REPLAYS:
        _, _, q_new = vu._cache(seq_lens, q_len, seed, q.device)
        on_run.q.copy_(q_new)
        on_run.v.fill_(float("nan"))
        on_run.stats.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        with _switch(monkeypatch, 0):
            off_run.q.copy_(q_new)
            v_ref, s_ref = off_run()
            torch.cuda.synchronize()
        assert _bits(on_run.v, v_ref) and _bits(on_run.stats, s_ref), (
            f"{case}: replay {seed} differs"
        )


def test_fallback_unchanged_and_not_counted(monkeypatch):
    """35 rows per head: the op runs the unfused decode + BMM; the switch neither counts nor changes it."""
    seq_lens, q_len, rank = _vu().CASES["bs5_fallback"]
    vu, dcp, pool, q, plan, w = _setup(seq_lens, q_len, rank, False)
    _, _, _, census = _mods()
    scale = dcp.default_softmax_scale()
    with _switch(monkeypatch, 0):
        v_off, s_off = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
    with _switch(monkeypatch, 1):
        before = _census_count(census)
        v_on, s_on = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
        assert _census_count(census) == before
    torch.cuda.synchronize()
    assert _bits(v_on, v_off) and _bits(s_on, s_off)


def test_default_is_sm103_only(monkeypatch):
    dcp, _, flags, _ = _mods()
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_SPLIT_PIPE", -1)
    dcp._split_pipe_by_default.cache_clear()
    try:
        assert dcp.split_pipe_on(device) == (torch.cuda.get_device_capability() == (10, 3))
        monkeypatch.setattr(dcp, "MLA_SPLIT_PIPE_CAPABILITIES", ((10, 7),))
        dcp._split_pipe_by_default.cache_clear()
        assert not dcp.split_pipe_on(device)
        monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_SPLIT_PIPE", 1)
        assert dcp.split_pipe_on(device)
        monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_SPLIT_PIPE", 0)
        monkeypatch.setattr(dcp, "MLA_SPLIT_PIPE_CAPABILITIES", ((10, 3), (10, 0)))
        dcp._split_pipe_by_default.cache_clear()
        assert not dcp.split_pipe_on(device)
    finally:
        monkeypatch.undo()
        dcp._split_pipe_by_default.cache_clear()
