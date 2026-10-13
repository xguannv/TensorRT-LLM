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
"""``TRTLLM_K3_SHARED_FIRST``: the Kimi-K3 MoE runtime issues the shared expert's GEMMs before
the routed FC1 (which waits for them) and its all-reduce after FC1, on the shared stream.

The modules are stand-ins (elementwise ops on the GPU), so the test checks the scheduling
plumbing: the same result as the default schedule, the all-reduce applied once to the shared
partial on the shared stream, FC1 joining the shared-GEMM event, the FC1 event consumed, the
token-count gate, and CUDA graph capture.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN = 64


class _AllReduce(nn.Module):
    """Stand-in all-reduce: adds 1000 and records the stream it ran on."""

    def __init__(self):
        super().__init__()
        self.streams = []

    def forward(self, x, all_reduce_params=None):
        if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
            return x
        self.streams.append(torch.cuda.current_stream())
        return x + 1000.0


class _SharedExperts(nn.Module):
    """Stand-in for the TP-sharded GatedMLP: partial = 2 x, reduced by down_proj.all_reduce."""

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Module()
        self.down_proj.reduce_output = True
        self.down_proj.use_fused_gemm_allreduce = False
        self.down_proj.all_reduce = _AllReduce()
        self.calls = []

    def forward(self, x, final_all_reduce_params=None):
        self.calls.append(final_all_reduce_params)
        return self.down_proj.all_reduce(x * 2.0, all_reduce_params=final_all_reduce_params)


class _RoutedExperts:
    """Stand-in routed experts: joins ``input_ready_event``, "FC1" (3 x), records the backend's
    ``post_fc1_event`` if set, "FC2" (+1)."""

    def __init__(self, routing):
        self.routing_method = routing
        self.use_dp = False
        self.all_reduce = None
        self.backend = SimpleNamespace(post_fc1_event=None)
        self.ready_events = []

    def __call__(self, value, router_logits, input_ready_event=None, **kwargs):
        self.ready_events.append(input_ready_event)
        if input_ready_event is not None:
            input_ready_event.wait()
        y = value * 3.0
        if self.backend.post_fc1_event is not None:
            self.backend.post_fc1_event.record()
            self.backend.post_fc1_event = None
        return y + 1.0


class _Identity(nn.Module):
    def forward(self, value, **kwargs):
        return value


def _expected(hidden):
    """Routed (3 x + 1) plus reduced shared (2 x + 1000), in the runtime's order."""
    return (hidden * 3.0 + 1.0) + (hidden * 2.0 + 1000.0)


def _runtime(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod

    bias = torch.zeros(896, device="cuda")
    routing = _KimiK3RoutingMethod(
        top_k=16,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(compute_logits=lambda value: value[:, :8].float())
    runtime.routed_experts = _RoutedExperts(routing)
    runtime.routed_expert_down_proj = _Identity()
    runtime.routed_expert_norm = _Identity()
    runtime.routed_expert_up_proj = _Identity()
    runtime.shared_experts = _SharedExperts()
    runtime.shared_down_all_reduce = None
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime._merge_moe_allreduce = False
    runtime.shared_expert_stream = torch.cuda.Stream()
    runtime.latent_down_stream = torch.cuda.Stream()
    for name in (
        "moe_main_event",
        "moe_shared_event",
        "moe_latent_fork_event",
        "moe_latent_ready_event",
        "moe_shared_gemm_event",
        "moe_fc1_event",
    ):
        setattr(runtime, name, torch.cuda.Event())
    runtime.moe_routed_event = None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "0")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_FUSED_ROUTER", "0")
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_fused_latent_input", lambda value: None)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value * 1.0)
    return runtime


@pytest.mark.parametrize("mode", ["1", "router"])
def test_shared_first_matches_default(monkeypatch, mode):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    runtime = _runtime(monkeypatch)
    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn((300, HIDDEN), device="cuda", generator=gen)
    all_reduce = runtime.shared_experts.down_proj.all_reduce
    routed = runtime.routed_experts
    with with_multi_stream(True):
        monkeypatch.setattr(modeling_kimi_linear, "_K3_SHARED_FIRST", "0")
        ref = runtime(hidden)
        assert routed.ready_events[-1] is runtime.moe_latent_ready_event
        monkeypatch.setattr(modeling_kimi_linear, "_K3_SHARED_FIRST", mode)
        n_calls = len(all_reduce.streams)
        out = runtime(hidden)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert torch.equal(ref, _expected(hidden))
    # The shared expert ran without its all-reduce; the reduction ran once, on the shared stream.
    assert not runtime.shared_experts.calls[-1].enable_allreduce
    assert all_reduce.streams[n_calls:] == [runtime.shared_expert_stream]
    assert routed.ready_events[-1] is runtime.moe_shared_gemm_event
    assert routed.backend.post_fc1_event is None


def test_shared_first_token_gate(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    runtime = _runtime(monkeypatch)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_SHARED_FIRST", "1")
    with with_multi_stream(True):
        assert not runtime._shared_first_applies(256)
        assert runtime._shared_first_applies(257)
        assert runtime._shared_first_applies(2048)
        assert not runtime._shared_first_applies(2049)
        runtime._merge_moe_allreduce = True
        assert not runtime._shared_first_applies(896)
        runtime._merge_moe_allreduce = False
    with with_multi_stream(False):
        assert not runtime._shared_first_applies(896)
    hidden = torch.ones((128, HIDDEN), device="cuda")
    with with_multi_stream(True):
        runtime(hidden)
    # Below the gate: the default schedule (FC1 joins the latent chain only).
    assert runtime.routed_experts.ready_events[-1] is runtime.moe_latent_ready_event


def test_shared_first_cuda_graph(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    runtime = _runtime(monkeypatch)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_SHARED_FIRST", "1")
    static = torch.zeros((896, HIDDEN), device="cuda")
    with with_multi_stream(True):
        runtime(static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runtime(static)
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        static.copy_(torch.randn(static.shape, device="cuda", generator=gen))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _expected(static))
