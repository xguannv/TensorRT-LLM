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
"""``TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU``: the low-M split-K GEMM with the SwiGLU epilogue
(``low_m_bf16_splitk.run_splitk_dense_swiglu``) against the GEMM it replaces followed by ``trtllm::silu_and_mul``.

* Kernel: bit for bit the pinned GEMM (``run_splitk_dense``, same tactic, same weight) then ``silu_and_mul``, at the
  drafter's gate_up shape [1792, 7168] and the gate_up pins' row counts (7 / 8 / 14 / 16 / 28 / 32), with and without
  PDL, and at another shape (I 512, K 1024); the weight is read, never written.
* ``DrafterGemm.project_swiglu``: bit for bit as above; under capture it refuses a build that never ran eagerly; a
  captured graph replayed with rewritten inputs is bit for bit the eager call.
* The switch: 1 on, 0 off, -1 on SM103 only.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    flags,
    mla_drafter_gemm,
)

INTER, HIDDEN = 896, 7168
ROWS = (7, 8, 14, 16, 28, 32)

needs_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the low-M split-K GEMM needs SM100 / SM103",
)


def _same(a, b):
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    )


def _weights(inter, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(2 * inter, k, generator=g, device="cuda") * k**-0.5).bfloat16()
    return w, g


def _pin(rows):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import SplitKTactic

    return SplitKTactic(*next(p for p in mla_drafter_gemm.PINS["gate_up"] if rows <= p[0])[1])


def _reference(x, w, tactic, pdl):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import run_splitk_dense

    h = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
    run_splitk_dense(x, w.t(), None, h, pdl, tactic)
    return torch.ops.trtllm.silu_and_mul(h)


@needs_sm100
@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize("rows", ROWS)
def test_kernel_bitwise(rows, pdl):
    import tensorrt_llm  # noqa: F401  (trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        run_splitk_dense_swiglu,
    )

    w, g = _weights(INTER, HIDDEN, rows)
    w_before = w.clone()
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    tactic = _pin(rows)
    out = torch.empty(rows, INTER, dtype=torch.bfloat16, device="cuda")
    run_splitk_dense_swiglu(x, w, out, tactic, pdl)
    want = _reference(x, w, tactic, pdl)
    assert _same(out, want), (rows, pdl, (out.float() - want.float()).abs().max().item())
    assert _same(w, w_before), "the weight changed"


@needs_sm100
@pytest.mark.parametrize("rows", (5, 16, 32))
def test_kernel_bitwise_other_shape(rows):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        SplitKTactic,
        run_splitk_dense_swiglu,
    )

    w, g = _weights(512, 1024, 100 + rows)
    x = torch.randn(rows, 1024, generator=g, device="cuda").bfloat16()
    tactic = SplitKTactic(64, 16 if rows <= 16 else 32, 2, 6)
    out = torch.empty(rows, 512, dtype=torch.bfloat16, device="cuda")
    run_splitk_dense_swiglu(x, w, out, tactic, False)
    assert _same(out, _reference(x, w, tactic, False))


@needs_sm100
@pytest.mark.parametrize("rows", (14, 32))
def test_project_swiglu_eager_and_graph(rows):
    import tensorrt_llm  # noqa: F401

    w, g = _weights(INTER, HIDDEN, 200 + rows)
    gemm = mla_drafter_gemm.DrafterGemm()
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    pdl = get_env_enable_pdl()
    # Under capture, a build that never ran eagerly is refused.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        refused = gemm.project_swiglu(x, w)
    assert refused is None
    del graph

    got = gemm.project_swiglu(x, w)
    assert got is not None and _same(got, _reference(x, w, _pin(rows), pdl))

    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = gemm.project_swiglu(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    assert out is not None
    for r in range(3):
        x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, gemm.project_swiglu(x, w)), f"replay {r}"


def test_switch(monkeypatch):
    dev = torch.device("cuda", 0)
    mla_drafter_gemm._swiglu_by_default.cache_clear()
    for value, cap, want in (
        (1, (9, 0), True),
        (0, (10, 3), False),
        (-1, (10, 3), True),
        (-1, (10, 0), False),
        (-1, (10, 7), False),
    ):
        monkeypatch.setattr(flags, "TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU", value)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, _c=cap: _c)
        mla_drafter_gemm._swiglu_by_default.cache_clear()
        assert mla_drafter_gemm.swiglu_enabled(dev) is want, (value, cap)
    mla_drafter_gemm._swiglu_by_default.cache_clear()
