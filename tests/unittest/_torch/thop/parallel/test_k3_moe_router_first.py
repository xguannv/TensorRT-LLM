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
"""``TRTLLM_K3_ROUTER_FIRST``: the Kimi-K3 MoE runtime issues the router GEMM on the main stream
before latent_down on the latent stream, with the latent fork event recorded before both.

The modules are stand-ins (elementwise ops on the GPU), so the test checks the plumbing: the same
result as the default order, the issue order (fork event, router, latent_down) and the streams,
the token-count gate, and CUDA graph capture.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN = 64


class _AllReduce(nn.Module):
    """Stand-in all-reduce: adds 1000."""

    def forward(self, x, all_reduce_params=None):
        if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
            return x
        return x + 1000.0


class _SharedExperts(nn.Module):
    """Stand-in for the TP-sharded GatedMLP: partial = 2 x, reduced by down_proj.all_reduce."""

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Module()
        self.down_proj.reduce_output = True
        self.down_proj.use_fused_gemm_allreduce = False
        self.down_proj.all_reduce = _AllReduce()

    def forward(self, x, final_all_reduce_params=None):
        return self.down_proj.all_reduce(x * 2.0, all_reduce_params=final_all_reduce_params)


class _RoutedExperts:
    """Stand-in routed experts: joins ``input_ready_event``, returns 3 x + 1."""

    def __init__(self, routing):
        self.routing_method = routing
        self.use_dp = False
        self.all_reduce = None
        self.backend = SimpleNamespace(post_fc1_event=None)

    def __call__(self, value, router_logits, input_ready_event=None, **kwargs):
        if input_ready_event is not None:
            input_ready_event.wait()
        return value * 3.0 + 1.0


class _Identity(nn.Module):
    def forward(self, value, **kwargs):
        return value


def _expected(hidden):
    """Routed (3 x + 1) plus reduced shared (2 x + 1000), in the runtime's order."""
    return (hidden * 3.0 + 1.0) + (hidden * 2.0 + 1000.0)


class _ForkEvent:
    """``torch.cuda.Event`` that logs its record calls into ``log``."""

    def __init__(self, log):
        self.event = torch.cuda.Event()
        self.log = log

    def record(self, stream=None):
        self.log.append(("fork", torch.cuda.current_stream()))
        self.event.record(stream)

    def wait(self, stream=None):
        self.event.wait(stream)


def _runtime(monkeypatch, log):
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

    def router(value):
        log.append(("router", torch.cuda.current_stream()))
        return value[:, :8].float()

    def projection_stub(value, projection):
        if projection is runtime.routed_expert_down_proj:
            log.append(("latent_down", torch.cuda.current_stream()))
        return value * 1.0

    runtime.gate = SimpleNamespace(compute_logits=router)
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
        "moe_latent_ready_event",
        "moe_shared_gemm_event",
        "moe_fc1_event",
    ):
        setattr(runtime, name, torch.cuda.Event())
    runtime.moe_latent_fork_event = _ForkEvent(log)
    runtime.moe_routed_event = None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "0")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_FUSED_ROUTER", "0")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_SHARED_FIRST", "0")
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_fused_latent_input", lambda value: None)
    monkeypatch.setattr(runtime, "_lowm_fp4_latent_input", lambda value: None)
    monkeypatch.setattr(runtime, "_routed_projection", projection_stub)
    return runtime


@pytest.mark.parametrize("num_tokens", [14, 64, 65])
def test_router_first_order_and_result(monkeypatch, num_tokens):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    log = []
    runtime = _runtime(monkeypatch, log)
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    hidden = torch.randn((num_tokens, HIDDEN), device="cuda", generator=gen)
    main = torch.cuda.current_stream()
    with with_multi_stream(True):
        monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTER_FIRST", False)
        ref = runtime(hidden)
        default_order = [name for name, _ in log]
        log.clear()
        monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTER_FIRST", True)
        out = runtime(hidden)
    torch.cuda.synchronize()
    assert torch.equal(ref, _expected(hidden))
    assert torch.equal(out, ref)
    assert default_order == ["fork", "latent_down", "router"]
    # The fork event is recorded before the router either way, so latent_down depends on the MoE
    # input only; with the switch the router is issued (captured) first, up to the token gate.
    first = num_tokens <= modeling_kimi_linear._K3_ROUTER_FIRST_MAX_M
    order = ["fork", "router", "latent_down"] if first else default_order
    assert [name for name, _ in log] == order
    streams = dict((name, stream) for name, stream in log)
    assert streams["fork"] == main and streams["router"] == main
    assert streams["latent_down"] == runtime.latent_down_stream


def test_router_first_cuda_graph(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    log = []
    runtime = _runtime(monkeypatch, log)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTER_FIRST", True)
    static = torch.zeros((14, HIDDEN), device="cuda")
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
