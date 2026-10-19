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
"""``TRTLLM_K3_SHARED_TAIL_PRIO``: on the latent all-reduce's own-workspace path, the Kimi-K3 MoE
runtime runs the shared expert's gate_up on the shared-expert stream and its activation, down
projection and all-reduce on a second (priority) stream that the main stream joins.

The modules are stand-ins (elementwise BF16 ops on the GPU), so the test checks the plumbing: the
same result as the unsplit shared expert, the stream of every piece, the token-count gate, and
CUDA graph capture.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN = 64


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, name):
        self.calls.append((name, torch.cuda.current_stream()))


class _DownProj(nn.Module):
    """Stand-in down projection + all-reduce: (x + 1) reduced as x + 1000."""

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.reduce_output = True

    def forward(self, x, all_reduce_params=None, layer_idx=None):
        self.log("down")
        y = x + 1.0
        if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
            return y
        return y + 1000.0


class _GateUp(nn.Module):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def forward(self, x):
        self.log("gate_up")
        return x * 2.0


class _SharedExperts(nn.Module):
    """Stand-in GatedMLP: gate_up (2 x), activation (+3), down (+1) and all-reduce (+1000)."""

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.gate_up_proj = _GateUp(log)
        self.down_proj = _DownProj(log)
        self.layer_idx = 1

    def _apply_activation(self, x):
        self.log("act")
        return x + 3.0

    def forward(self, x, final_all_reduce_params=None):
        return self.down_proj(
            self._apply_activation(self.gate_up_proj(x)), all_reduce_params=final_all_reduce_params
        )


class _RoutedExperts:
    """Stand-in routed experts: joins ``input_ready_event``, returns 3 x + 1."""

    def __init__(self, routing):
        self.routing_method = routing
        self.use_dp = False
        self.all_reduce = object()
        self.backend = SimpleNamespace(post_fc1_event=None)

    def __call__(self, value, router_logits, input_ready_event=None, **kwargs):
        if input_ready_event is not None:
            input_ready_event.wait()
        return value * 3.0 + 1.0


class _Identity(nn.Module):
    def forward(self, value, **kwargs):
        return value


def _expected(hidden):
    """Routed (3 x + 1) plus reduced shared (2 x + 3 + 1 + 1000), as the runtime adds them."""
    return (hidden * 3.0 + 1.0) + (hidden * 2.0 + 3.0 + 1.0 + 1000.0)


def _runtime(monkeypatch, log, tail):
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
    runtime.shared_experts = _SharedExperts(log)
    runtime.shared_down_all_reduce = None
    runtime._reduce_routed_output = True
    runtime._defer_shared_allreduce = False
    runtime._merge_moe_allreduce = False
    runtime._latent_ar_ws = {"max_tokens": 128}
    runtime.shared_expert_stream = torch.cuda.Stream()
    runtime.latent_down_stream = torch.cuda.Stream()
    runtime.shared_tail_stream = tail
    for name in (
        "moe_main_event",
        "moe_shared_event",
        "moe_shared_act_event",
        "moe_latent_fork_event",
        "moe_latent_ready_event",
        "moe_shared_gemm_event",
        "moe_fc1_event",
    ):
        setattr(runtime, name, torch.cuda.Event())
    runtime.moe_routed_event = None
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "0")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_FUSED_ROUTER", "0")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_SHARED_FIRST", "0")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_MOE_L2_PREFETCH", frozenset())
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_fused_latent_input", lambda value: None)
    monkeypatch.setattr(runtime, "_lowm_fp4_latent_input", lambda value: None)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value * 1.0)
    monkeypatch.setattr(runtime, "_latent_up_reads_shared", lambda latent: False)

    def latent_tail(routed_out, shared_out, residual, defer_residual):
        # The own-workspace tail: latent all-reduce + latent_up on main, then the join.
        log("latent_tail")
        runtime.moe_shared_event.wait()
        return routed_out + shared_out

    monkeypatch.setattr(runtime, "_own_workspace_latent_tail", latent_tail)
    # The stand-in shared expert is not a GatedMLP: gate the split on the token count only.
    monkeypatch.setattr(
        runtime,
        "_shared_tail_applies",
        lambda n: tail is not None
        and modeling_kimi_linear._K3_SHARED_TAIL_MIN_M
        <= n
        <= modeling_kimi_linear._K3_SHARED_TAIL_MAX_M,
    )
    return runtime


@pytest.mark.parametrize("num_tokens", [14, 64, 65])
def test_shared_tail_streams_and_result(monkeypatch, num_tokens):
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    log = _Recorder()
    tail = torch.cuda.Stream(priority=-1)
    ref_runtime = _runtime(monkeypatch, log, None)
    runtime = _runtime(monkeypatch, log, tail)
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    hidden = torch.randn((num_tokens, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    with with_multi_stream(True):
        ref = ref_runtime(hidden)
        ref_streams = dict(log.calls)
        log.calls.clear()
        out = runtime(hidden)
    torch.cuda.synchronize()
    assert torch.equal(ref, _expected(hidden))
    assert torch.equal(out, ref)
    streams = dict(log.calls)
    assert ref_streams["act"] == ref_streams["down"] == ref_runtime.shared_expert_stream
    assert streams["gate_up"] == runtime.shared_expert_stream
    split = num_tokens <= modeling_kimi_linear._K3_SHARED_TAIL_MAX_M
    tail_or_shared = tail if split else runtime.shared_expert_stream
    assert streams["act"] == streams["down"] == tail_or_shared
    assert streams["latent_tail"] == torch.cuda.current_stream()


def test_shared_tail_cuda_graph(monkeypatch):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    log = _Recorder()
    runtime = _runtime(monkeypatch, log, torch.cuda.Stream(priority=-1))
    static = torch.zeros((14, HIDDEN), device="cuda", dtype=torch.bfloat16)
    with with_multi_stream(True):
        runtime(static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runtime(static)
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        static.copy_(torch.randn(static.shape, device="cuda", generator=gen).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _expected(static))
