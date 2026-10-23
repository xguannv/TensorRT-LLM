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
"""``TRTLLM_K3LL_MLA_GATE_EARLY`` (``kimi_k3_nvfp4/flags.py``) on one GPU: with ``TRTLLM_K3LL_MLA_GATE_SIDE`` the side-
stream gate projection is forked before the fused q_a / kv_a instead of after the cache append. Same kernel, same
input, so bit for bit:

* switch off = the pre-change order (the gate forked after the cache append), recorded by a spy;
* switch on: the gate projection runs before the fused q_a / kv_a; output, query bytes and latent row bit for bit the
  switch-off ones at the trunk defaults (HELIX_Q_ABSORB / GATE_SIDE / FUSED_A at 1) and with HELIX_Q_ABSORB at 0
  (the head-group path), 8 / 16 / 32 rows, eagerly and from a CUDA graph (side stream on) replayed with new inputs;
* GATE_SIDE 0: the switch does nothing (the gate stays concatenated behind the fused q_a / kv_a).
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

ROWS = (8, 16, 32)
REPLAYS = 2
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
        "TRTLLM_K3LL_MLA_GATE_EARLY": 0,
    }
    with monkeypatch.context() as m:
        for name, value in {**base, **values}.items():
            m.setattr(flags, name, int(value))
        yield


@contextlib.contextmanager
def _order_spy(mla, monkeypatch):
    """Records, in call order, the fused q_a / kv_a and the gate projection calls into ``mla.record``."""
    kv_a, g = mla.kv_a_proj_with_mqa, mla.g_proj

    class _Spy(torch.nn.Module):
        def __init__(self, inner, tag):
            super().__init__()
            self.inner, self.tag = inner, tag

        def forward(self, x):
            mla.record.append((self.tag,))
            return self.inner(x)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    with monkeypatch.context() as m:
        m.setattr(mla, "kv_a_proj_with_mqa", _Spy(kv_a, "kv_a"), raising=False)
        m.setattr(mla, "g_proj", _Spy(g, "g_proj"), raising=False)
        yield


@pytest.fixture(scope="module")
def mla():
    qs = _qs()
    torch.manual_seed(0)
    return qs._standin(qs._random_weights(torch.Generator(device="cuda").manual_seed(1234)))


def _order(record):
    return [r[0] for r in record if r[0] in ("kv_a", "g_proj")]


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("base", list(BASES))
def test_gate_early_bitwise_eager(mla, monkeypatch, rows, base):
    qs = _qs()
    meta = qs._meta(rows)
    for seed in (30, 31):
        x = qs._x(rows, seed + rows)
        with _switches(monkeypatch, **BASES[base]), _order_spy(mla, monkeypatch):
            off, off_rope, off_split = qs._run(mla, qs._decode(mla), x, meta)
            off_order = _order(mla.record)
        with (
            _switches(monkeypatch, TRTLLM_K3LL_MLA_GATE_EARLY=1, **BASES[base]),
            _order_spy(mla, monkeypatch),
        ):
            on, on_rope, on_split = qs._run(mla, qs._decode(mla), x, meta)
            on_order = _order(mla.record)
        assert off_order == ["kv_a", "g_proj"] and on_order == ["g_proj", "kv_a"], (
            off_order,
            on_order,
        )
        assert on_split[2] is (base == "all_head")
        assert qs._bits(on, off)
        assert qs._bits(on_rope[1], off_rope[1]) and qs._bits(on_split[1], off_split[1])


@pytest.mark.parametrize("rows", ROWS)
def test_gate_early_without_gate_side_is_off(mla, monkeypatch, rows):
    qs = _qs()
    meta = qs._meta(rows)
    x = qs._x(rows, 50 + rows)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_GATE_SIDE=0):
        off, _, off_split = qs._run(mla, qs._decode(mla), x, meta)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_GATE_SIDE=0, TRTLLM_K3LL_MLA_GATE_EARLY=1):
        on, _, on_split = qs._run(mla, qs._decode(mla), x, meta)
    assert qs._bits(on, off) and qs._bits(on_split[1], off_split[1])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("base", list(BASES))
@pytest.mark.parametrize("early", [0, 1], ids=["off", "on"])
def test_gate_early_graph_replay(mla, monkeypatch, rows, base, early):
    """Captured with the side stream on, replayed with new inputs: bit for bit the switch-off eager call."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = _qs()
    meta = qs._meta(rows)
    x = qs._x(rows, 1)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_GATE_EARLY=early, **BASES[base]):
        qs._decode(mla)(x, meta)
        torch.cuda.synchronize()
        mla.record.clear()
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.graph(graph):
            out = qs._decode(mla)(x, meta)
    records = list(mla.record)
    for replay in range(REPLAYS):
        x.copy_(qs._x(rows, 300 + replay))
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        replay_split = next(r for r in records if r[0] == "split")[1].clone()
        with _switches(monkeypatch, **BASES[base]):
            ref, _, ref_split = qs._run(mla, qs._decode(mla), x.clone(), meta)
        assert qs._bits(replayed, ref)
        assert qs._bits(replay_split, ref_split[1])
