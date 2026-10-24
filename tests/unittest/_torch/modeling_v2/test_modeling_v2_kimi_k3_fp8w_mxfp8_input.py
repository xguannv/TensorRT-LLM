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
"""``kimi_k3_nvfp4/fp8w_lowm.py``'s pre-quantized MXFP8 input on one GPU: a routed FP8 projection run from the MXFP8
form of its input (``k3_fp8w_lowm_mxfp8``) is bit for bit the low-M kernel on the bf16 input.

* ``forward_mxfp8`` / ``gemm_mxfp8`` against ``k3_fp8w_lowm_gemm`` on the bf16 rows, for the decode projections fed by
  the pre-attention output (KDA QKVG, the fused KDA QKVG + [f_a | b] weight, MLA fused q_a / kv_a with its partial
  last row tile) at 1-64 rows, both ``TRTLLM_K3LL_MX_SF_EARLY_RELEASE`` builds (the switch flipped between calls of
  one module), and the fallback where the MXFP8 kernel does not take the rows (above the low-M range);
* ``takes_mxfp8_input`` for modules and weight triples;
* the offer: a routed module called on the offered tensor (or its leading rows) runs the MXFP8 kernel (``MX_STATS``)
  with the low-M kernel's bits; another tensor, another stream or a withdrawn offer keep the low-M kernel; an offer no
  GEMM took is counted as unconsumed;
* the offer in a CUDA graph (kernels compiled eagerly first), replayed with new inputs, against eager calls; and an
  uncompiled MXFP8 kernel under capture keeps the low-M kernel (same bits).
"""

import functools

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the low-M FP8 GEMMs run on SM100 / SM103",
)

K = 7168
SHAPES = {"qkvg": (3072, False), "qkvg_bfa": (3216, True), "fused_a": (2112, True)}
ROWS = (1, 7, 8, 16, 32, 64)


def _modules():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        flags,
        fp8w_lowm,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    return fp8w_lowm, flags, lowm, mx


@functools.cache
def _weight(n):
    _, _, lowm, _ = _modules()
    g = torch.Generator(device="cuda").manual_seed(n)
    n_pad = -(-n // 128) * 128
    w = torch.randn((n_pad, K), device="cuda", generator=g) * 0.02
    blocks = w.view(n_pad // 128, 128, K // 128, 128)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0))
    )
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n_pad, K)
    return lowm.prepare_k3_fp8w_lowm_weight(w_fp8[:n].contiguous(), scale.float())


class _Projection(nn.Module):
    """A weight-read projection routed as ``fp8w_lowm.install`` routes one (its stock forward must not run)."""

    def __init__(self, n, partial):
        super().__init__()
        fl, _, _, _ = _modules()
        self.weight, self.weight_scale = _weight(n)
        self._locality_domain_weight_shards = None
        self._k3ll_fp8w_lowm = True
        self._k3ll_fp8w_partial_n_tile = partial

        def stock(x, **kwargs):
            raise AssertionError("the stock GEMM ran")

        self.forward = functools.partial(fl._forward, self, stock)


def _x(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, K), device="cuda", generator=g)
    x.view(-1)[torch.randint(0, m * K, (max(1, m * K // 2048),), device="cuda", generator=g)] = (
        300.0
    )
    x[0, :32] = 0
    return x.bfloat16()


def _mx_input(x):
    fl, _, _, mx = _modules()
    xq, xs = fl.new_mxfp8_input(x.shape[0], x.shape[1], x.device)
    rq, rs = mx.mxfp8_quantize_reference(x)
    xq.copy_(rq)
    xs.copy_(rs)
    return xq, xs


def _ref(x, n):
    return torch.ops.trtllm.k3_fp8w_lowm_gemm(x, *_weight(n))


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.fixture
def switches():
    _, flags, _, _ = _modules()
    names = ("TRTLLM_K3LL_MX_SF_EARLY_RELEASE", "TRTLLM_K3LL_FP8W_LOWM_FUSED_A")
    old = {name: getattr(flags, name) for name in names}
    yield flags
    for name, value in old.items():
        setattr(flags, name, value)


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("m", ROWS)
def test_forward_mxfp8_bitwise(name, m, switches):
    """One module, the early-release switch flipped between calls: every call bit for bit the low-M kernel."""
    fl, flags, _, _ = _modules()
    n, partial = SHAPES[name]
    proj = _Projection(n, partial)
    x = _x(m, seed=m + n)
    xq, xs = _mx_input(x)
    want = _ref(x, n)
    assert fl.takes_mxfp8_input(proj, m)
    for early in (0, 1, 0):
        flags.TRTLLM_K3LL_MX_SF_EARLY_RELEASE = early
        before = fl.MX_STATS["mxfp8"]
        got = fl.forward_mxfp8(proj, x, xq, xs)
        got_gemm = fl.gemm_mxfp8(x, *_weight(n), xq, xs, partial_n_tile=partial)
        torch.cuda.synchronize()
        assert fl.MX_STATS["mxfp8"] - before == 2, "the MXFP8 kernel did not run"
        assert _same(got, want) and _same(got_gemm, want), f"early release {early}"


def test_takes_and_fallback(switches):
    """``takes_mxfp8_input`` follows the kernel's range and the partial-tile switch; outside it ``forward_mxfp8`` runs
    the module (the same bits) and ``gemm_mxfp8`` returns None."""
    fl, flags, lowm, _ = _modules()
    n, partial = SHAPES["fused_a"]
    proj = _Projection(n, partial)
    assert fl.takes_mxfp8_input(proj, 8) and fl.takes_mxfp8_input((*_weight(n), True), 8)
    assert not fl.takes_mxfp8_input(
        (*_weight(n), False), 8
    )  # a partial last row tile needs the flag
    assert not fl.takes_mxfp8_input(None, 8) and not fl.takes_mxfp8_input(nn.Linear(4, 4), 8)
    flags.TRTLLM_K3LL_FP8W_LOWM_FUSED_A = 0
    assert not fl.takes_mxfp8_input(proj, 8)
    flags.TRTLLM_K3LL_FP8W_LOWM_FUSED_A = 1
    big = lowm.low_m_max_for(3072, K) + 1
    if big <= lowm.MAX_M:
        proj_q = _Projection(3072, False)
        assert not fl.takes_mxfp8_input(proj_q, big)
        x = _x(big, seed=3)
        xq, xs = _mx_input(x)
        assert fl.gemm_mxfp8(x, *_weight(3072), xq, xs, partial_n_tile=False) is None
        assert _same(fl.forward_mxfp8(proj_q, x, xq, xs), _ref(x, 3072))


def test_offer_matching():
    """The offer is taken by the offered tensor's leading rows on its stream only, and withdrawn after."""
    fl, _, _, _ = _modules()
    n = 3072
    proj = _Projection(n, False)
    x = _x(16, seed=11)
    xq, xs = _mx_input(x)
    other = x.clone()
    stats = fl.MX_STATS
    fl.offer_mxfp8(x, xq, xs)
    try:
        cases = [
            ("the offered tensor", lambda: proj(x), 1),
            ("its leading rows", lambda: proj(x[:5]), 1),
            ("another tensor", lambda: proj(other), 0),
        ]
        for what, run, want_mx in cases:
            before = stats["mxfp8"]
            run()
            torch.cuda.synchronize()
            assert stats["mxfp8"] - before == want_mx, what
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        before = stats["mxfp8"]
        with torch.cuda.stream(side):
            got = proj(x)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        assert stats["mxfp8"] == before, "another stream took the offer"
        assert _same(got, _ref(x, n))
        assert _same(proj(x[:5]), _ref(x[:5].contiguous(), n))
    finally:
        fl.withdraw_mxfp8()
    before = stats["mxfp8"]
    assert _same(proj(x), _ref(x, n)) and stats["mxfp8"] == before, "a withdrawn offer was taken"
    unconsumed = stats["unconsumed"]
    fl.offer_mxfp8(x, xq, xs)
    fl.withdraw_mxfp8()
    assert stats["unconsumed"] == unconsumed + 1


@pytest.mark.parametrize("m", (8, 32))
def test_offer_in_cuda_graph(m, switches):
    """Captured with the kernels compiled eagerly, replayed with new inputs (both early-release builds), against
    eager low-M calls; an MXFP8 kernel never compiled in this process keeps the low-M kernel inside a capture."""
    fl, flags, _, mx = _modules()
    n, partial = SHAPES["qkvg_bfa"]
    w, ws = _weight(n)
    x = _x(m, seed=21)
    xq, xs = _mx_input(x)
    graphs = []
    for early in (0, 1):
        flags.TRTLLM_K3LL_MX_SF_EARLY_RELEASE = early
        fl.offer_mxfp8(x, xq, xs)
        fl.gemm(x, w, ws)  # eager: compiles the build
        fl.withdraw_mxfp8()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        before = fl.MX_STATS["mxfp8"]
        with torch.cuda.graph(graph, stream=stream):
            fl.offer_mxfp8(x, xq, xs)
            out = fl.gemm(x, w, ws)
            fl.withdraw_mxfp8()
        assert fl.MX_STATS["mxfp8"] - before == 1, "the capture does not hold the MXFP8 kernel"
        graphs.append((graph, out))
    for rep in range(3):
        x.copy_(_x(m, seed=100 + rep))
        rq, rs = mx.mxfp8_quantize_reference(x)
        xq.copy_(rq)
        xs.copy_(rs)
        want = _ref(x, n)
        for graph, out in graphs:
            graph.replay()
            torch.cuda.synchronize()
            assert _same(out, want), f"replay {rep}"
    # Not compiled: a weight shape no call of this process has run (N 1024), inside a capture.
    flags.TRTLLM_K3LL_MX_SF_EARLY_RELEASE = 0
    w1, ws1 = _weight(1024)
    x3 = _x(3, seed=5)
    xq3, xs3 = _mx_input(x3)
    assert not mx.is_compiled(
        3, 1024, K, True, False
    )  # (its tactic, cached, chosen here, outside the capture)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    before = fl.MX_STATS["not_compiled"]
    with torch.cuda.graph(graph, stream=stream):
        out = fl.gemm_mxfp8(x3, w1, ws1, xq3, xs3, partial_n_tile=False)
    assert out is None and fl.MX_STATS["not_compiled"] == before + 1
