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
"""``TRTLLM_K3LL_MLA_GATE_FUSED_A`` (``kimi_k3_nvfp4/flags.py``) on one GPU: the MLA output gate's FP8 rows stacked
below the fused q_a / kv_a rows (padded to whole 128-row tiles) in one weight buffer, one ``k3_fp8w_lowm_gemm`` for
both (``K3DecodeMLA.build_gate_fused_a`` / ``_gate_fused_rows``), on the stand-in module of
``test_modeling_v2_kimi_k3_mla_q_switches`` at 8 / 16 / 32 / 64 rows:

* the build: both modules' weights become views of the stacked buffer holding their bytes unchanged;
* switch off with the stacked weight built = switch off without it (a fresh stand-in), bit for bit, at the trunk
  defaults (HELIX_Q_ABSORB / GATE_SIDE / FUSED_A at 1) and on the head-group path, eagerly and from a CUDA graph;
* the stacked GEMM's columns against the two GEMMs it replaces: the q_a / kv_a / k_pe columns bit for bit (same
  split-K 6 as the fused_a GEMM); the gate columns within one bf16 ulp of their row's largest value on at most 1 % of
  the elements (the stacked shape runs split-K 6, the gate's own 768 x 7168 GEMM split 8: the FP32 sums round
  differently);
* switch on, the whole decode: output within ``OUT_TOL`` of switch off, query bytes and latent row bit for bit where
  the q_a / kv_a columns are; captured and replayed with new inputs, bit for bit its own eager call.
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="k3_fp8w_lowm_gemm and K3MlaQAbsorb run on SM100 / SM103",
)

ROWS = (8, 16, 32, 64)
REPLAYS = 2
OUT_TOL = 0.01
ULP = 2.0**-7
FUSED = 2112
BASES = {
    "all_head": {"TRTLLM_K3LL_HELIX_Q_ABSORB": 1},
    "head_group": {"TRTLLM_K3LL_HELIX_Q_ABSORB": 0},
}


def _qs():
    import test_modeling_v2_kimi_k3_mla_q_switches as qs

    return qs


@contextlib.contextmanager
def _switches(monkeypatch, **values):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    base = {
        "TRTLLM_K3LL_HELIX_Q_ABSORB": 1,
        "TRTLLM_K3LL_MLA_GATE_SIDE": 1,
        "TRTLLM_K3LL_FP8W_LOWM_FUSED_A": 1,
        "TRTLLM_K3LL_MLA_QKV_PDL": 0,
        "TRTLLM_K3LL_MLA_QNORM_PDL": 0,
        "TRTLLM_K3LL_MLA_PREWAIT_L2": 0,
        "TRTLLM_K3LL_MLA_GATE_FUSED_A": 0,
    }
    with monkeypatch.context() as m:
        for name, value in {**base, **values}.items():
            m.setattr(flags, name, int(value))
        yield


def _meta(rows):
    import types

    reqs = rows // 8
    return types.SimpleNamespace(num_generations=reqs, num_seqs=reqs)


@pytest.fixture(scope="module")
def pair():
    """(stand-in with the stacked weight built, an identical stand-in without it, the pre-build weight bytes)."""
    qs = _qs()
    torch.manual_seed(0)
    built = qs._standin(qs._random_weights(torch.Generator(device="cuda").manual_seed(1234)))
    plain = qs._standin(qs._random_weights(torch.Generator(device="cuda").manual_seed(1234)))
    before = (built.kv_a_proj_with_mqa.weight.clone(), built.g_proj.weight.clone())
    assert built.build_gate_fused_a() is None and built._gate_fused_a is not None
    assert built.build_gate_fused_a() is None  # idempotent
    return built, plain, before


def test_build_views(pair):
    built, _, (fused_w, gate_w) = pair
    weight, scale, gate_col = built._gate_fused_a
    assert gate_col == 2176 and tuple(weight.shape) == (2176 + gate_w.shape[0], fused_w.shape[1])
    assert built.kv_a_proj_with_mqa.weight.data_ptr() == weight.data_ptr()
    assert built.g_proj.weight.data_ptr() == weight[gate_col:].data_ptr()
    assert torch.equal(built.kv_a_proj_with_mqa.weight.view(torch.uint8), fused_w.view(torch.uint8))
    assert torch.equal(built.g_proj.weight.view(torch.uint8), gate_w.view(torch.uint8))
    assert torch.all(weight[FUSED:gate_col].view(torch.uint8) == 0)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("base", list(BASES))
def test_off_equals_unbuilt(pair, monkeypatch, rows, base):
    qs = _qs()
    built, plain, _ = pair
    meta = _meta(rows)
    x = qs._x(rows, 60 + rows)
    with _switches(monkeypatch, **BASES[base]):
        a, a_rope, a_split = qs._run(built, qs._decode(built), x, meta)
        b, b_rope, b_split = qs._run(plain, qs._decode(plain), x, meta)
    assert qs._bits(a, b) and qs._bits(a_rope[1], b_rope[1]) and qs._bits(a_split[1], b_split[1])


@pytest.mark.parametrize("rows", ROWS)
def test_stacked_columns(pair, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        fp8w_lowm,
    )

    qs = _qs()
    built, _, _ = pair
    weight, scale, gate_col = built._gate_fused_a
    x = qs._x(rows, 70 + rows)
    out = fp8w_lowm.gemm(x, weight, scale)
    assert out is not None
    fused, gate = built.kv_a_proj_with_mqa(x), built.g_proj(x)
    torch.cuda.synchronize()
    report = []
    for name, got, ref in (
        ("q_a/kv_a/k_pe", out[:, :FUSED], fused),
        ("gate", out[:, gate_col:], gate),
    ):
        diff = (got.float() - ref.float()).abs()
        # relative to the row's largest value: a rounding difference of the split-K sum is one bf16 ulp of it
        rel = (diff / ref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)).max().item()
        frac = float((diff > 0).float().mean())
        report.append(
            f"{name}: {int((diff > 0).sum())} of {diff.numel()} differ, max rel (row max) {rel:.2e}"
        )
        assert rel <= ULP and frac <= 0.01, report[-1]
    # Same split-K and bytes for the fused q_a / kv_a rows: these columns are the fused_a GEMM's, bit for bit.
    assert torch.equal(out[:, :FUSED].view(torch.int16), fused.view(torch.int16)), report[0]
    print(f"[rows {rows}] " + "; ".join(report))


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("base", list(BASES))
def test_on_close_to_off(pair, monkeypatch, rows, base):
    qs = _qs()
    built, _, _ = pair
    meta = _meta(rows)
    x = qs._x(rows, 80 + rows)
    with _switches(monkeypatch, **BASES[base]):
        off, off_rope, off_split = qs._run(built, qs._decode(built), x, meta)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_GATE_FUSED_A=1, **BASES[base]):
        on, on_rope, on_split = qs._run(built, qs._decode(built), x, meta)
    rel = qs._rel(on, off)
    same_q = qs._bits(on_rope[1], off_rope[1]) and qs._bits(on_split[1], off_split[1])
    print(
        f"[{base} rows {rows}] output rel L2 {rel:.2e}, bitwise {qs._bits(on, off)}; query + latent bitwise {same_q}"
    )
    assert rel <= OUT_TOL


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("base", list(BASES))
@pytest.mark.parametrize("on", [0, 1], ids=["off", "on"])
def test_graph_replay(pair, monkeypatch, rows, base, on):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = _qs()
    built, plain, _ = pair
    meta = _meta(rows)
    x = qs._x(rows, 1)
    values = {"TRTLLM_K3LL_MLA_GATE_FUSED_A": on, **BASES[base]}
    with _switches(monkeypatch, **values):
        qs._decode(built)(x, meta)
        torch.cuda.synchronize()
        built.record.clear()
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.graph(graph):
            out = qs._decode(built)(x, meta)
    for replay in range(REPLAYS):
        x.copy_(qs._x(rows, 400 + replay))
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        with _switches(monkeypatch, **values):
            ref, _, _ = qs._run(built, qs._decode(built), x.clone(), meta)
        assert qs._bits(replayed, ref)
        if not on:
            with _switches(monkeypatch, **BASES[base]):
                ref_plain, _, _ = qs._run(plain, qs._decode(plain), x.clone(), meta)
            assert qs._bits(replayed, ref_plain)
