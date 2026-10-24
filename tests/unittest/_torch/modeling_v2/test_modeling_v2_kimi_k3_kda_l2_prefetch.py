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
"""``TRTLLM_K3LL_KDA_L2_PREFETCH`` (``kimi_k3_nvfp4/flags.py``): after a KDA verify's projections, a side stream
prefetches the next KDA layer's projection weights into L2 (``k3_l2_prefetch``); the main stream joins it after the
last layer (``join_kda_l2_prefetch``). One GPU, ``K3DecodeKDA``'s methods on stand-in layers:

* the ranges: the next layer's fused ``[q | k | v | g | f_a | b]`` weight and scale (else the QKVG module's), f_b and
  o_proj, as ``(address, bytes)``, built once;
* when it issues: switch on, at most ``KDA_L2_PREFETCH_MAX_ROWS`` rows, a next layer, multi-stream on: one prefetch on
  the side stream behind an event of the calling stream, then a pending join; with any of those off, nothing;
* a CUDA graph capturing two layers' prefetches and the join (a capture must join every stream it forks) replays, and
  the prefetched weights are unchanged (hints only).
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="k3_l2_prefetch runs on SM100 / SM103",
)


def _impl():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        implementation,
    )

    return implementation


def _flags():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    return flags


def _layer(seed: int, fused: bool = True):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def w(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    layer = SimpleNamespace(
        _qkvg_bfa_fp8=(w(3328, 7168, dtype=torch.float8_e4m3fn), w(3328, 14).to(torch.int32))
        if fused
        else None,
        qkvg_proj=SimpleNamespace(
            weight=w(3072, 7168, dtype=torch.float8_e4m3fn), weight_scale=w(3072, 14)
        ),
        f_b_proj=SimpleNamespace(weight=w(768, 128)),
        o_proj=SimpleNamespace(weight=w(7168, 768)),
        _k3_l2_ranges=None,
        _k3_l2_start=None,
        _k3_next_kda=None,
    )
    impl = _impl()
    layer.k3_l2_prefetch_ranges = lambda: impl.K3DecodeKDA.k3_l2_prefetch_ranges(layer)
    return layer


def _issue(layer, x):
    _impl().K3DecodeKDA._k3_issue_l2_prefetch(layer, x)


@pytest.mark.parametrize("fused", [True, False])
def test_ranges(fused):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_l2_prefetch as l2

    layer = _layer(1, fused)
    ranges = layer.k3_l2_prefetch_ranges()
    first = (
        list(layer._qkvg_bfa_fp8)
        if fused
        else [layer.qkvg_proj.weight, layer.qkvg_proj.weight_scale]
    )
    tensors = first + [layer.f_b_proj.weight, layer.o_proj.weight]
    assert ranges == [l2.tensor_range(t) for t in tensors]
    assert layer.k3_l2_prefetch_ranges() is ranges  # built once


def _spy(monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_l2_prefetch as l2

    calls = []
    real = l2.run_k3_l2_prefetch

    def spy(ranges, *args, **kwargs):
        calls.append((list(ranges), torch.cuda.current_stream().cuda_stream))
        return real(ranges, *args, **kwargs)

    monkeypatch.setattr(l2, "run_k3_l2_prefetch", spy)
    return calls


@pytest.mark.parametrize(
    "switch,rows,has_next,multi,issues",
    [
        (1, 8, True, True, True),
        (1, 16, True, True, True),
        (1, 17, True, True, False),
        (1, 32, True, True, False),
        (1, 16, False, True, False),
        (1, 16, True, False, False),
        (0, 16, True, True, False),
    ],
)
def test_when_it_issues(monkeypatch, switch, rows, has_next, multi, issues):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    impl = _impl()
    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_KDA_L2_PREFETCH", switch)
    calls = _spy(monkeypatch)
    layer, nxt = _layer(2), _layer(3)
    layer._k3_next_kda = nxt if has_next else None
    x = torch.randn(rows, 7168, device="cuda").bfloat16()
    with with_multi_stream(multi):
        _issue(layer, x)
    state = impl._kda_l2_state(x.device)
    if issues:
        assert len(calls) == 1 and calls[0][0] == nxt.k3_l2_prefetch_ranges()
        assert calls[0][1] == state[0].cuda_stream != torch.cuda.current_stream().cuda_stream
        assert state[2] is True
        impl.join_kda_l2_prefetch(x.device)
        assert state[2] is False
    else:
        assert calls == [] and state[2] is False
    torch.cuda.synchronize()


def test_graph_capture_joins(monkeypatch):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    impl = _impl()
    monkeypatch.setattr(_flags(), "TRTLLM_K3LL_KDA_L2_PREFETCH", 1)
    a, b, c = _layer(4), _layer(5), _layer(6)
    a._k3_next_kda, b._k3_next_kda = b, c
    x = torch.randn(16, 7168, device="cuda").bfloat16()
    before = [t.clone() for t in (*b._qkvg_bfa_fp8, *c._qkvg_bfa_fp8)]
    with with_multi_stream(True):
        _issue(a, x)  # compiles the prefetch kernel outside the capture
        impl.join_kda_l2_prefetch(x.device)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _issue(a, x)
            _issue(b, x)
            impl.join_kda_l2_prefetch(x.device)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    after = [*b._qkvg_bfa_fp8, *c._qkvg_bfa_fp8]
    assert all(torch.equal(p.view(torch.uint8), q.view(torch.uint8)) for p, q in zip(before, after))
