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
"""``TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA``: the Kimi-K3 MoE block prefetches the next layer's MLA
q_b weight into L2 on its latent-down stream once its routed FC2 is done, only where the next
decoder layer is an MLA layer and a later MoE layer joins that stream, next to (and without
changing) the ``TRTLLM_K3_MOE_L2_PREFETCH`` cases; it captures into a CUDA graph."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)


def _moe_runtime(layer_idx, group, key, stream):
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    rt = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
    nn.Module.__init__(rt)
    rt.layer_idx = layer_idx
    rt.latent_down_stream = stream
    rt.moe_fc2_event = torch.cuda.Event()
    rt._moe_group = group
    rt._moe_group[layer_idx] = rt
    rt._l2_ranges_cache = {}
    rt._layers_key = key
    rt.routed_expert_up_proj = SimpleNamespace(
        weight=torch.zeros(4096, dtype=torch.uint8, device="cuda"), weight_scale=None
    )
    return rt


def _mla_runtime(layer_idx, key):
    """A ``KimiMLARuntime`` holding only a q_b projection (FP8 weight and packed int32 scale)."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    rt = mkl.KimiMLARuntime.__new__(mkl.KimiMLARuntime)
    nn.Module.__init__(rt)
    scale = torch.zeros((3, 1024), dtype=torch.int32, device="cuda").t()
    rt.mixer = SimpleNamespace(
        q_b_proj=SimpleNamespace(
            weight=torch.zeros((1024, 1536), dtype=torch.uint8, device="cuda"),
            weight_scale=scale,
        )
    )
    rt._q_b_l2_ranges = ((), [])
    mkl._K3_MLA_LAYERS.setdefault(key, weakref.WeakValueDictionary())[layer_idx] = rt
    return rt


def test_next_mla_q_b_prefetch(monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_l2_prefetch as pf
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    key = object()  # a model key no other test uses
    key_id = id(key)
    group = weakref.WeakValueDictionary()
    stream = torch.cuda.Stream()
    kda_moe = _moe_runtime(2, group, key_id, stream)  # next layer (3) is MLA
    mla_moe = _moe_runtime(3, group, key_id, stream)  # next layer (4) is not MLA
    last = _moe_runtime(4, group, key_id, stream)
    mla = _mla_runtime(3, key_id)
    w, s = mla.mixer.q_b_proj.weight, mla.mixer.q_b_proj.weight_scale
    assert mla.q_b_l2_prefetch_ranges() == [pf.tensor_range(w), pf.tensor_range(s)]
    assert mla.q_b_l2_prefetch_ranges()[1][1] == s.numel() * 4

    calls = []
    real = pf.run_k3_l2_prefetch

    def spy(ranges, **kwargs):
        calls.append((torch.cuda.current_stream(), list(ranges)))
        real(ranges, **kwargs)

    monkeypatch.setattr(pf, "run_k3_l2_prefetch", spy)
    monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH", frozenset())
    with with_multi_stream(True):
        # Off: nothing.
        monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH_NEXT_MLA", False)
        kda_moe._issue_l2_prefetch(14)
        assert not calls
        monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH_NEXT_MLA", True)
        # Next layer not MLA, too many tokens, or no later MoE layer on the stream: nothing.
        mla_moe._issue_l2_prefetch(14)
        kda_moe._issue_l2_prefetch(65)
        last._issue_l2_prefetch(14)
        assert not calls
        static = torch.zeros(4, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static.add_(1.0)
            kda_moe._issue_l2_prefetch(14)
            # With TRTLLM_K3_MOE_L2_PREFETCH as well, its cases come first, unchanged.
            monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH", frozenset({"lu"}))
            kda_moe._issue_l2_prefetch(14)
            # The next MoE layer's latent-down work follows on the stream and joins main.
            fork, ready = torch.cuda.Event(), torch.cuda.Event()
            fork.record()
            with torch.cuda.stream(stream):
                fork.wait()
                static.add_(1.0)
                ready.record()
            ready.wait()
    q_b = mla.q_b_l2_prefetch_ranges()
    assert [r for _, r in calls] == [q_b, kda_moe.l2_prefetch_ranges("lu"), q_b]
    assert all(st == stream for st, _ in calls)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static, torch.full_like(static, 4.0))
    del mkl._K3_MLA_LAYERS[key_id]
