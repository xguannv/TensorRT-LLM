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
"""The two weight-before-wait switches of the Kimi K3 NVFP4 MLA decode (``kimi_k3_nvfp4/flags.py``) on one GPU:

* ``TRTLLM_K3LL_MLA_QNORM_PDL``: the Helix all-head query's q_a RMSNorm through ``q_norm_pdl.rmsnorm_release_at_launch``
  (FlashInfer's ``RMSNormKernel`` with the PDL trigger at launch and the weight read before the wait);
* ``TRTLLM_K3LL_MLA_PREWAIT_L2``: ``K3MlaQAbsorb`` with ``prewait`` (every W_UK tile of a CTA staged or L2-prefetched
  before the wait, dependents released after it) and the V-up split read whose split kernel L2-prefetches W_UV before
  its wait and releases ``reduce_v_up`` after it.

Both only move loads ahead of ``griddepcontrol.wait`` and triggers earlier, so every check is bit for bit:

* the kernels: the released norm against ``trtllm::flashinfer_rmsnorm`` and the module (strided q_a views and
  contiguous rows); ``K3MlaQAbsorb`` prewait against the plain kernel and the Triton kernel, every tile shape;
* switch off = the pre-change chain: ``all_head_queries`` with both switches 0 against norm -> q_b -> ``K3MlaQAbsorb``
  as trunk ran them; the V-up op with ``prewait`` 0 compiles and runs the trunk variant (no ``prewait`` key);
* switch on = switch off: ``K3DecodeMLA._quantized_decode`` on the stand-in module of
  ``test_modeling_v2_kimi_k3_mla_q_switches`` at the trunk defaults (HELIX_Q_ABSORB, GATE_SIDE, FUSED_A at 1), each
  arm's output, recorded query bytes and latent row against the off arm, 8 / 16 / 32 rows, eagerly and from a CUDA
  graph (side stream on) replayed with new inputs; ``split_read_attention_v_up`` at 8 / 16 / 32 rows per head, the V
  rows and softmax statistics, eagerly and replayed. Spies confirm the switched kernels ran;
* the QNORM_PDL request-count bound (``decode_mla_q.QNORM_PDL_MAX_REQUESTS``): the released norm runs for 1 to that
  many requests only; one request above it, the switch at 1 runs the module's norm and equals the switch at 0.
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="K3MlaQAbsorb and the fused V-up reduction run on SM100 / SM103",
)

ROWS = (8, 16, 32)
REPLAYS = 2
QA, QA_STRIDE = 1536, 2112
TRUNK_DEFAULTS = {
    "TRTLLM_K3LL_HELIX_Q_ABSORB": 1,
    "TRTLLM_K3LL_MLA_GATE_SIDE": 1,
    "TRTLLM_K3LL_FP8W_LOWM_FUSED_A": 1,
    "TRTLLM_K3LL_MLA_QKV_PDL": 0,
    "TRTLLM_K3LL_MLA_DECODE_PDL": 1,
}
ARMS = {
    "off": {},
    "qnorm": {"TRTLLM_K3LL_MLA_QNORM_PDL": 1},
    "prewait": {"TRTLLM_K3LL_MLA_PREWAIT_L2": 1},
    "both": {"TRTLLM_K3LL_MLA_QNORM_PDL": 1, "TRTLLM_K3LL_MLA_PREWAIT_L2": 1},
}


def _qs():
    import test_modeling_v2_kimi_k3_mla_q_switches as qs

    return qs


def _vu():
    import test_modeling_v2_kimi_k3_mla_v_up as vu

    return vu


def _absorb_test():
    """``tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_mla_q_absorb.py`` (its inputs and Triton reference)."""
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cute_dsl_kernels",
        "kimi_k3",
        "test_k3_mla_q_absorb.py",
    )
    spec = importlib.util.spec_from_file_location("_k3_mla_q_absorb_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flags():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    return flags


@contextlib.contextmanager
def _switches(monkeypatch, **values):
    flags = _flags()
    with monkeypatch.context() as m:
        for name, value in {**TRUNK_DEFAULTS, "TRTLLM_K3LL_MLA_QNORM_PDL": 0,
                            "TRTLLM_K3LL_MLA_PREWAIT_L2": 0, **values}.items():  # fmt: skip
            m.setattr(flags, name, int(value))
        yield


@contextlib.contextmanager
def _spies(monkeypatch):
    """Counts of the released norm and of the K3MlaQAbsorb calls by ``prewait``."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

    seen = {"qnorm": 0, "absorb_prewait": 0, "absorb_plain": 0}
    release, absorb = q_norm_pdl.rmsnorm_release_at_launch, k3_mla_q_absorb.k3_mla_q_absorb

    def release_spy(*args, **kwargs):
        seen["qnorm"] += 1
        return release(*args, **kwargs)

    def absorb_spy(*args, prewait=False, **kwargs):
        seen["absorb_prewait" if prewait else "absorb_plain"] += 1
        return absorb(*args, prewait=prewait, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(q_norm_pdl, "rmsnorm_release_at_launch", release_spy)
        m.setattr(k3_mla_q_absorb, "k3_mla_q_absorb", absorb_spy)
        yield seen


def _bits(a, b) -> bool:
    return _qs()._bits(a, b)


# ---------------------------------------------------------------------------------------------------------------------
# The released q_a norm


@pytest.mark.parametrize("rows", (1, 7, 8, 14, 16, 28, 32))
@pytest.mark.parametrize("strided", (True, False), ids=("q_a_view", "contiguous"))
def test_qnorm_release_kernel_bitwise(rows, strided):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

    qs = _qs()
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    norm = qs._norm((1 + 0.1 * torch.randn(QA, device="cuda", generator=gen)).bfloat16())
    full = (torch.randn(rows, QA_STRIDE, device="cuda", generator=gen) * 3).bfloat16()
    x = full[:, :QA] if strided else full[:, :QA].contiguous()
    assert decode_mla_q._release_norm_applies(x, norm)
    ref_op = torch.ops.trtllm.flashinfer_rmsnorm(x, norm.weight, float(norm.variance_epsilon))
    ref_mod = norm(x)
    out = q_norm_pdl.rmsnorm_release_at_launch(x, norm.weight, float(norm.variance_epsilon))
    torch.cuda.synchronize()
    assert _bits(out, ref_op) and _bits(out, ref_mod)
    released, ran = decode_mla_q.q_a_norm(x, norm, True)
    assert ran and _bits(released, ref_mod)
    plain, ran = decode_mla_q.q_a_norm(x, norm, False)
    assert not ran and _bits(plain, ref_mod)


# ---------------------------------------------------------------------------------------------------------------------
# K3MlaQAbsorb prewait


@pytest.mark.parametrize("num_tokens", (7, 8, 14, 16, 28, 32, 129, 448, 449))
def test_absorb_prewait_bitwise(num_tokens):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb as cq

    t = _absorb_test()
    q, w = t._inputs(num_tokens, num_tokens)
    ref = torch.full((num_tokens, 96, 576), 0x55, dtype=torch.uint8, device="cuda")
    plain = torch.full_like(ref, 0x11)
    pre = torch.full_like(ref, 0xAA)
    t._triton(q, w, ref)
    cq.k3_mla_q_absorb(q, w, plain, 128, 64)
    cq.k3_mla_q_absorb(q, w, pre, 128, 64, prewait=True)
    torch.cuda.synchronize()
    assert torch.equal(pre, plain) and torch.equal(pre, ref)


@pytest.mark.parametrize("tile", [(128, False), (256, False), (128, True), (256, True)])
def test_absorb_prewait_tiles(tile):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb as cq

    t = _absorb_test()
    num_tokens = 300
    q, w = t._inputs(num_tokens, 3)
    ref = torch.full((num_tokens, 96, 576), 0x55, dtype=torch.uint8, device="cuda")
    pre = torch.full_like(ref, 0xAA)
    t._triton(q, w, ref)
    cq.k3_mla_q_absorb(q, w, pre, 128, 64, tile=tile, prewait=True)
    torch.cuda.synchronize()
    assert torch.equal(pre, ref)


# ---------------------------------------------------------------------------------------------------------------------
# Switch off = the pre-change all-head query chain


@pytest.fixture(scope="module")
def mla():
    qs = _qs()
    torch.manual_seed(0)
    return qs._standin(qs._random_weights(torch.Generator(device="cuda").manual_seed(1234)))


@pytest.mark.parametrize("rows", ROWS)
def test_switches_off_equal_pre_change_queries(mla, monkeypatch, rows):
    """``all_head_queries`` with both switches 0 against trunk's chain: the module norm, Q-B, ``K3MlaQAbsorb``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb

    qs = _qs()
    x = qs._x(rows, 40 + rows)
    with _switches(monkeypatch):
        ag = mla.kv_a_proj_with_mqa(x)
        q_a = ag[:, :QA]
        quant_q, q, kernel, q_norm = decode_mla_q.all_head_queries(
            q_a, mla.q_a_layernorm, mla.q_b_proj, mla._q_absorb_w_kc, 128, 64
        )
        ref_norm = mla.q_a_layernorm(q_a)
        ref_q = mla.q_b_proj(ref_norm).view(rows, 96, 192)
        ref_quant = torch.empty_like(quant_q)
        k3_mla_q_absorb.k3_mla_q_absorb(ref_q, mla._q_absorb_w_kc, ref_quant, 128, 64)
    torch.cuda.synchronize()
    assert kernel == "K3MlaQAbsorb"
    assert _bits(q_norm, ref_norm) and _bits(q, ref_q) and _bits(quant_q, ref_quant)


# ---------------------------------------------------------------------------------------------------------------------
# Switch on = switch off, through K3DecodeMLA._quantized_decode


def _decode_arm(mla, monkeypatch, arm, x, meta):
    qs = _qs()
    with _switches(monkeypatch, **ARMS[arm]), _spies(monkeypatch) as seen:
        out, rope, split = qs._run(mla, qs._decode(mla), x, meta)
    return out, rope, split, dict(seen)


def _decode_mla_q():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_q,
    )

    return decode_mla_q


def _check_spies(arm, seen, requests):
    on = ARMS[arm]
    released = bool(on.get("TRTLLM_K3LL_MLA_QNORM_PDL")) and _decode_mla_q().qnorm_pdl_applies(
        requests
    )
    assert (seen["qnorm"] > 0) == released, (arm, requests, seen)
    if on.get("TRTLLM_K3LL_MLA_PREWAIT_L2"):
        assert seen["absorb_prewait"] > 0 and seen["absorb_plain"] == 0, (arm, seen)
    else:
        assert seen["absorb_plain"] > 0 and seen["absorb_prewait"] == 0, (arm, seen)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("arm", ["qnorm", "prewait", "both"])
def test_switch_on_equals_off_eager(mla, monkeypatch, rows, arm):
    meta = _qs()._meta(rows)
    for seed in (10, 11):
        x = _qs()._x(rows, seed + rows)
        off, off_rope, off_split, seen_off = _decode_arm(mla, monkeypatch, "off", x, meta)
        on, on_rope, on_split, seen_on = _decode_arm(mla, monkeypatch, arm, x, meta)
        _check_spies("off", seen_off, meta.num_generations)
        _check_spies(arm, seen_on, meta.num_generations)
        assert on_split[2] is True  # the all-head path (HELIX_Q_ABSORB) is the one switched
        assert _bits(on, off)
        assert _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("arm", list(ARMS))
def test_graph_replay_equals_off_eager(mla, monkeypatch, rows, arm):
    """Each arm captured with the side stream on (every launch after a PDL predecessor), replayed with new inputs:
    bit for bit the off arm's eager call on the same inputs."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    qs = _qs()
    meta = qs._meta(rows)
    x = qs._x(rows, 1)
    with _switches(monkeypatch, **ARMS[arm]), _spies(monkeypatch) as seen:
        qs._decode(mla)(x, meta)  # compiles every kernel outside the capture
        torch.cuda.synchronize()
        mla.record.clear()
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.graph(graph):
            out = qs._decode(mla)(x, meta)
        _check_spies(arm, dict(seen), meta.num_generations)
    records = list(mla.record)
    for replay in range(REPLAYS):
        x.copy_(qs._x(rows, 200 + replay))
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        replay_split = next(r for r in records if r[0] == "split")[1].clone()
        ref, _, ref_split, _ = _decode_arm(mla, monkeypatch, "off", x.clone(), meta)
        assert _bits(replayed, ref)
        assert _bits(replay_split, ref_split[1])


def test_qnorm_pdl_bound():
    d = _decode_mla_q()
    assert d.QNORM_PDL_MAX_REQUESTS >= 1
    assert [n for n in range(0, 10) if d.qnorm_pdl_applies(n)] == list(
        range(1, d.QNORM_PDL_MAX_REQUESTS + 1)
    )


@pytest.mark.parametrize("arm", ["qnorm", "both"])
def test_qnorm_off_above_bound(mla, monkeypatch, arm):
    """One request above ``QNORM_PDL_MAX_REQUESTS`` with the switch at 1: the module's norm runs (the spy sees no
    released norm), and output, query bytes and latent row equal the switch at 0, eagerly."""
    qs = _qs()
    rows = qs.TOKENS_PER_REQUEST * (_decode_mla_q().QNORM_PDL_MAX_REQUESTS + 1)
    meta = qs._meta(rows)
    x = qs._x(rows, 90 + rows)
    off, off_rope, off_split, seen_off = _decode_arm(mla, monkeypatch, "off", x, meta)
    on, on_rope, on_split, seen_on = _decode_arm(mla, monkeypatch, arm, x, meta)
    assert seen_on["qnorm"] == 0
    _check_spies("off", seen_off, meta.num_generations)
    _check_spies(arm, seen_on, meta.num_generations)
    assert _bits(on, off)
    assert _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])


# ---------------------------------------------------------------------------------------------------------------------
# The V-up split read


V_CASES = ("bs1_mtp7", "bs2_mtp7", "bs4_mtp7")


def _v_setup(case):
    vu = _vu()
    dcp, mla_v_up, _ = vu._mods()
    device = torch.device("cuda")
    seq_lens, q_len, rank = vu.CASES[case]
    pool, block_rows, q = vu._cache(seq_lens, q_len, 11, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    plan = dcp.make_split_read_plan(block_rows, lens, q_len, vu.PAGE, vu.CP, rank)
    w, _, _ = vu._w_uv(1, device)
    return vu, dcp, mla_v_up, pool, q, plan, w, (seq_lens, q_len)


@pytest.mark.parametrize("case", V_CASES)
def test_v_up_prewait_bitwise(monkeypatch, case):
    vu, dcp, mla_v_up, pool, q, plan, w, (seq_lens, q_len) = _v_setup(case)
    scale = dcp.default_softmax_scale()
    with _switches(monkeypatch):
        off_run = vu._VUp(dcp, q, pool, plan, w, scale)
        v_off, s_off = (t.clone() for t in off_run())
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_PREWAIT_L2=1):
        on_run = vu._VUp(dcp, q, pool, plan, w, scale)
        v_on, s_on = (t.clone() for t in on_run())
    torch.cuda.synchronize()
    keys = list(mla_v_up._compiled)
    assert any("prewait" in k for k in keys) and any("prewait" not in k for k in keys)
    assert _bits(v_on, v_off) and _bits(s_on, s_off)

    # the prewait arm captured and replayed with new queries: bit for bit the off arm's eager call
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_PREWAIT_L2=1):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            on_run()
    for seed in (21, 22):
        _, _, q_new = vu._cache(seq_lens, q_len, seed, q.device)
        on_run.q.copy_(q_new)
        on_run.v.fill_(float("nan"))
        on_run.stats.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        with _switches(monkeypatch):
            off_run.q.copy_(q_new)
            v_ref, s_ref = off_run()
            torch.cuda.synchronize()
        assert _bits(on_run.v, v_ref) and _bits(on_run.stats, s_ref)


def test_v_up_prewait_needs_pdl(monkeypatch):
    """PREWAIT_L2 without the split kernel's PDL (MLA_DECODE_PDL 0): the trunk variant (no ``prewait`` key)."""
    vu, dcp, mla_v_up, pool, q, plan, w, _ = _v_setup("bs1_mtp7")
    scale = dcp.default_softmax_scale()
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_DECODE_PDL=0):
        v_off, s_off = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
    before = set(mla_v_up._compiled)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_DECODE_PDL=0, TRTLLM_K3LL_MLA_PREWAIT_L2=1):
        v_on, s_on = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
    torch.cuda.synchronize()
    assert not any("prewait" in k for k in set(mla_v_up._compiled) - before)
    assert _bits(v_on, v_off) and _bits(s_on, s_off)
