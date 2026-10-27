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
"""``TRTLLM_K3LL_MLA_PARTIAL_F16`` (``kimi_k3_nvfp4/flags.py``): the fused V-up MLA decode's split partials stored as
FP16 (the LSE stays FP32). One Helix16 rank's V-up split read on one GPU (helpers of
``test_modeling_v2_kimi_k3_mla_var_split``), at the shipped decode shapes (8 / 12 / 14 / 16 rows):

* the default (0) and its values; switch off around a switch-on call: the same bits (the switch-off path itself is
  covered against the plain split read + BMM by ``test_modeling_v2_kimi_k3_mla_var_split`` / ``_mla_v_up``);
* switch on: the softmax statistics bit for bit the switch-off ones (the LSE path is untouched), the V rows close to
  the switch-off rows, a build of its own; met first inside a capture, it raises;
* switch on against fp32 attention x fp32 W_UV: within a small factor of the switch-off rows' own error;
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

#: Per (token, head) relative difference of the V rows, switch on vs off (one FP16 rounding per partial).
ON_OFF_TOL = 1e-2
#: Switch-on error against fp32, at most this factor of the switch-off error (max and median).
FP32_OVER_OFF = 1.5
CASES = {
    # (global lengths, q_len, rank): Helix16 one-rank shares of the shipped decode shapes
    "bs1_d7_8rows": ([153600 + 8], 8, 5),
    "bs3_d3_12rows": ([153600 + 4] * 3, 4, 2),
    "bs2_d6_14rows": ([153600 + 7, 120000], 7, 3),
    "bs4_d3_16rows": ([153600 + 4, 90000, 40000, 153600], 4, 7),
}


def _var():
    import test_modeling_v2_kimi_k3_mla_var_split as var

    return var


def _set(monkeypatch, value):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_PARTIAL_F16", value)


def _call(case, monkeypatch, value, seed=11):
    var = _var()
    vu, dcp, mla_v_up, pool, block_rows, lens, q0, plan = var._setup(case, seed=seed)
    w, _, _ = vu._w_uv(1, q0.device)
    _set(monkeypatch, value)
    run = var._Run(vu, dcp, q0.clone(), pool, plan, w, True)
    v, s = (t.clone() for t in run())
    torch.cuda.synchronize()
    return var, vu, dcp, mla_v_up, plan, q0, run, v, s


def test_defaults():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    assert flags._FLAGS["TRTLLM_K3LL_MLA_PARTIAL_F16"][1] == -1
    assert flags.accepted_values()["TRTLLM_K3LL_MLA_PARTIAL_F16"] == (0, 1)


def test_default_resolves_per_arch(monkeypatch):
    """-1 is on only where the device's compute capability is in MLA_PARTIAL_F16_CAPABILITIES (SM103); 0 / 1 are the
    same on every device."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        mla_v_up,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    expected = torch.cuda.get_device_capability(device) in mla_v_up.MLA_PARTIAL_F16_CAPABILITIES
    assert mla_v_up.MLA_PARTIAL_F16_CAPABILITIES == ((10, 3),)
    _set(monkeypatch, -1)
    assert mla_v_up.flags_partial_f16(device) is expected
    for value in (0, 1):
        _set(monkeypatch, value)
        assert mla_v_up.flags_partial_f16(device) is bool(value)


@pytest.mark.parametrize("name", list(CASES))
def test_off_unchanged_around_on(name, monkeypatch):
    """Switch off, then on, then off: the two off calls give the same bits (the on build is a separate entry)."""
    var, vu, dcp, mla_v_up, plan, q, run, v0, s0 = _call(CASES[name], monkeypatch, 0)
    _set(monkeypatch, 1)
    run()
    _set(monkeypatch, 0)
    v1, s1 = (t.clone() for t in run())
    torch.cuda.synchronize()
    keyed = ~var._keyless(plan, q.shape[1])
    assert torch.equal(v1[keyed].view(torch.int16), v0[keyed].view(torch.int16))
    assert torch.equal(s1[keyed], s0[keyed])


@pytest.mark.parametrize("name", list(CASES))
def test_on_close_to_off(name, monkeypatch):
    var, vu, dcp, mla_v_up, plan, q, run, v_off, s_off = _call(CASES[name], monkeypatch, 0)
    _set(monkeypatch, 1)
    v_on, s_on = (t.clone() for t in run())
    torch.cuda.synchronize()
    assert any("partial_f16" in k for k in mla_v_up._compiled), "no switch-on build"
    keyed = ~var._keyless(plan, q.shape[1])
    # the LSE and the Helix statistics do not read the partial O: same bits
    assert torch.equal(s_on[keyed], s_off[keyed])
    assert not torch.isnan(v_on[keyed]).any()
    rel = var._per_head_rel(v_on[keyed], v_off[keyed])
    print(
        f"[{name}] V rows on vs off per (token, head): median {rel.median():.2e} max {rel.max():.2e}"
    )
    assert float(rel.max()) <= ON_OFF_TOL


@pytest.mark.parametrize(
    "seq_lens, q_len", [([9600], 8), ([5000, 9000], 7), ([12000, 3000, 700, 9000], 4)]
)
def test_on_against_fp32(seq_lens, q_len, monkeypatch):
    """The whole sequence on one rank (cp 1): switch on and off against fp32 attention x fp32 W_UV."""
    var = _var()
    vu, dcp, _, pool, block_rows, lens, q, plan = var._setup((seq_lens, q_len, 0), cp=1, seed=7)
    w, codes, scales = vu._w_uv(2, q.device)
    out = {}
    for value in (0, 1):
        _set(monkeypatch, value)
        out[value] = var._Run(vu, dcp, q, pool, plan, w, True)()[0].clone()
    torch.cuda.synchronize()
    latent_ref = vu._fp32_reference(q, pool, block_rows, lens)
    rows = len(seq_lens) * q_len
    w32 = codes.float() * scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    v_ref = torch.einsum(
        "rhk,hvk->rhv", latent_ref.reshape(rows, vu.HEADS, vu.LATENT).double(), w32.double()
    )

    def err(v):
        return ((v.double() - v_ref).norm(dim=-1) / v_ref.norm(dim=-1)).flatten()

    e_on, e_off = err(out[1]), err(out[0])
    print(
        f"[fp32 ref {seq_lens} q{q_len}] on median {e_on.median():.2e} max {e_on.max():.2e}; "
        f"off median {e_off.median():.2e} max {e_off.max():.2e}"
    )
    assert float(e_on.max()) <= FP32_OVER_OFF * float(e_off.max())
    assert float(e_on.median()) <= FP32_OVER_OFF * float(e_off.median())


@pytest.mark.parametrize("name", list(CASES))
def test_on_graph_replay(name, monkeypatch):
    var, vu, dcp, mla_v_up, plan, q, run, _, _ = _call(
        CASES[name], monkeypatch, 1
    )  # compiles eagerly
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
    v_eager, s_eager = var._Run(vu, dcp, run.q, run.pool, plan, run.w, True)()
    torch.cuda.synchronize()
    keyed = ~var._keyless(plan, q_len)
    assert torch.equal(replayed[keyed].view(torch.int16), v_eager[keyed].view(torch.int16))
    assert torch.equal(replayed_stats[keyed], s_eager[keyed])


def test_capture_without_the_build_raises(monkeypatch):
    """A switch-on build met first inside a capture raises (the forward-switch registry's LAZY_RAISE entry)."""
    var, vu, dcp, mla_v_up, plan, q, run, _, _ = _call(CASES["bs1_d7_8rows"], monkeypatch, 0)
    monkeypatch.setattr(
        mla_v_up,
        "_compiled",
        {k: v for k, v in mla_v_up._compiled.items() if "partial_f16" not in k},
    )
    _set(monkeypatch, 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not compiled before capture"):
        run()
