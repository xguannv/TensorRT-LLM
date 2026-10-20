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
"""Kimi K3 MoE block: ``TRTLLM_K3_SHARED_JOIN_AFTER_LU`` (the shared-expert join carried by a marker
kernel issued after latent_up).

``KimiK3MoERuntime.forward`` on the own-workspace latent path (``TRTLLM_K3_LATENT_AR_OWN_WS``) runs on
every rank with stand-ins for the routed experts and a slow TP-sharded shared expert. With the switch
the forward returns the same bits (and the same deferred residual) as without it, eagerly and under
CUDA graph replay; a consumer launched on the main stream right after the forward reads the shared
expert's output only after it is complete; and in the captured graph the next layer's
attention-residual mix (the PDL kernel that consumes the deferred residual) depends on the marker
kernel, not on the shared expert's last kernel.
"""

import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
LATENT = 3584
EPS = 1e-6
MAX_TOKENS = 64


class _RoutedStandIn:
    """Returns this rank's latent partial."""

    def __init__(self, all_reduce):
        self.all_reduce = all_reduce
        self.partial = None
        self.routing_method = None
        self.use_dp = False
        self.backend = SimpleNamespace()

    def __call__(self, x, router_logits, *, all_rank_num_tokens=None, **kwargs):
        return self.partial.clone()


class _SlowSharedStandIn(torch.nn.Module):
    """TP-sharded BF16 shared expert that reduces its partial itself, after a long spin on its
    stream (so a consumer that is not ordered after it reads a stale tensor)."""

    def __init__(self, all_reduce):
        super().__init__()
        self.all_reduce = all_reduce
        self.partial = None
        self.down_proj = torch.nn.Identity()

    def forward(self, x, final_all_reduce_params=None):
        torch.cuda._sleep(2_000_000)
        out = self.partial.clone()
        if final_all_reduce_params is None or final_all_reduce_params.enable_allreduce:
            out = self.all_reduce(out)
        return out


def _rank_inputs(world, num_tokens, seed):
    gen = torch.Generator(device="cuda")
    latents, shareds = [], []
    for r in range(world):
        gen.manual_seed(seed * 1009 + r)
        latents.append(torch.randn((num_tokens, LATENT), generator=gen, device="cuda").bfloat16())
        shareds.append(torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda").bfloat16())
    gen.manual_seed(seed * 1009 + 999)
    residual = (torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda") * 4).bfloat16()
    return latents, shareds, residual


def _graph_deps(graph, name_part):
    """Dependency function names of every kernel node whose name contains ``name_part``."""
    import cuda.bindings.driver as cu

    g = cu.CUgraph(graph.raw_cuda_graph())
    count = cu.cuGraphGetNodes(g, 0)[-1]
    nodes = cu.cuGraphGetNodes(g, count)[1]

    def name(node):
        _, kind = cu.cuGraphNodeGetType(node)
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            return str(kind)
        _, params = cu.cuGraphKernelNodeGetParams(node)
        err, fname = cu.cuFuncGetName(params.func)
        return fname.decode() if err == cu.CUresult.CUDA_SUCCESS else "?"

    out = []
    for node in nodes:
        if name_part in name(node):
            nd = cu.cuGraphNodeGetDependencies(node, 0)[-1]
            deps = cu.cuGraphNodeGetDependencies(node, nd)[1] if nd else []
            out.append([name(d) for d in deps])
    return out


def _join_marker_rank(world):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    try:
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res
        from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
        from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        ar_routed = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
        ar_shared = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
        assert ar_routed.mnnvl_allreduce is not None
        workspace = mkl._k3_latent_ar_workspace(
            ar_routed.mnnvl_allreduce.mapping, torch.bfloat16, LATENT, MAX_TOKENS
        )
        torch.manual_seed(0)
        runtime = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
        torch.nn.Module.__init__(runtime)
        runtime.hidden_size = HIDDEN
        runtime.moe_hidden_size = LATENT
        runtime.gate = SimpleNamespace(
            compute_logits=lambda h: torch.zeros((h.shape[0], 8), device=h.device)
        )
        runtime.routed_experts = _RoutedStandIn(ar_routed)
        runtime.shared_experts = _SlowSharedStandIn(ar_shared)
        runtime.shared_down_all_reduce = None
        runtime.routed_expert_down_proj = torch.nn.Identity()
        runtime.routed_expert_norm = RMSNorm(
            hidden_size=LATENT, eps=EPS, dtype=torch.bfloat16
        ).cuda()
        runtime.routed_expert_norm.weight.data = (
            torch.rand(LATENT, device="cuda") + 0.5
        ).bfloat16()
        runtime.routed_expert_up_proj = torch.nn.Linear(
            LATENT, HIDDEN, bias=False, dtype=torch.bfloat16
        ).cuda()
        runtime.routed_expert_up_proj.weight.data.normal_(0, 0.02)
        runtime._routed_projection = lambda x, projection: projection(x)
        runtime._router_exact = False
        runtime._can_fuse_router_down = lambda h: False
        runtime._reduce_routed_output = True
        runtime._defer_shared_allreduce = False
        runtime._merge_moe_allreduce = False
        runtime.shared_expert_stream = torch.cuda.Stream()
        runtime.latent_down_stream = None
        runtime.moe_main_event = torch.cuda.Event()
        runtime.moe_shared_event = torch.cuda.Event()
        runtime.moe_routed_event = runtime.moe_latent_fork_event = None
        runtime.moe_latent_ready_event = None
        runtime.moe_fc2_event = None
        runtime.shared_tail_stream = None
        runtime.moe_shared_act_event = None
        runtime._latent_ar_ws = workspace
        mkl._K3_LATENT_AR_CUTE = "1"

        # A_pre of the next layer on the deferred residual (the PDL consumer of the join).
        num_snap = 6
        gen = torch.Generator(device="cuda").manual_seed(7)
        weights = (
            (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.05).bfloat16(),
            (1 + torch.randn(HIDDEN, device="cuda", generator=gen) * 0.1).bfloat16(),
            (1 + torch.randn(HIDDEN, device="cuda", generator=gen) * 0.1).bfloat16(),
        )

        def a_pre(deferred, block):
            out = torch.empty_like(deferred.residual)
            upd = torch.empty_like(deferred.residual)
            k3_attn_res.run_k3_attn_res(
                deferred.residual,
                (deferred.routed, deferred.shared),
                block,
                *weights,
                EPS,
                EPS,
                out,
                upd,
            )
            return upd, out

        def forward(join_after_lu, hidden, residual):
            mkl._K3_SHARED_JOIN_AFTER_LU = join_after_lu
            runtime.moe_shared_join_event = torch.cuda.Event() if join_after_lu else None
            with with_multi_stream(True), torch.inference_mode():
                deferred = runtime(hidden, residual=residual, defer_residual=True)
                # Read on the main stream right after the forward (before any synchronization).
                shared_copy = deferred.shared.clone()
            return deferred, shared_copy

        def same(a, b, what):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=what)

        for num_tokens in (1, 7, 14):
            latents, shareds, residual = _rank_inputs(world, num_tokens, 30 + num_tokens)
            runtime.routed_experts.partial = latents[rank]
            runtime.shared_experts.partial = shareds[rank]
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            ref, ref_copy = forward(False, hidden, residual)
            torch.cuda.synchronize()
            out, copy = forward(True, hidden, residual)
            torch.cuda.synchronize()
            for x, y in zip(out, ref):
                same(x, y, f"deferred residual, {num_tokens} tokens")
            same(copy, ref.shared, f"shared read after the join, {num_tokens} tokens")
            same(ref_copy, ref.shared, f"shared read after the event join, {num_tokens} tokens")

        # CUDA graph: forward + A_pre, replayed on new inputs.
        num_tokens = 14
        latents, shareds, residual = _rank_inputs(world, num_tokens, 50)
        static_lat = latents[rank].clone()
        static_sh = shareds[rank].clone()
        static_res = residual.clone()
        block = torch.randn((num_snap, num_tokens, HIDDEN), device="cuda", generator=gen).bfloat16()
        runtime.routed_experts.partial = static_lat
        runtime.shared_experts.partial = static_sh
        hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        for join_after_lu in (False, True):
            for _ in range(2):
                deferred, _ = forward(join_after_lu, hidden, static_res)
                a_pre(deferred, block)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            mkl._K3_SHARED_JOIN_AFTER_LU = join_after_lu
            runtime.moe_shared_join_event = torch.cuda.Event() if join_after_lu else None
            with with_multi_stream(True), torch.inference_mode(), torch.cuda.graph(graph):
                deferred = runtime(hidden, residual=static_res, defer_residual=True)
                graph_out = a_pre(deferred, block)
            deps = _graph_deps(graph, "K3AttnResFastKernel")
            assert len(deps) == 1, deps
            assert any("K3JoinMarkerKernel" in d for d in deps[0]) == join_after_lu, deps
            for step in range(3):
                latents, shareds, residual = _rank_inputs(world, num_tokens, 60 + 7 * step)
                static_lat.copy_(latents[rank])
                static_sh.copy_(shareds[rank])
                static_res.copy_(residual)
                graph.replay()
                torch.cuda.synchronize()
                runtime.routed_experts.partial = latents[rank]
                runtime.shared_experts.partial = shareds[rank]
                ref, _ = forward(False, hidden, residual)
                ref_out = a_pre(ref, block)
                torch.cuda.synchronize()
                runtime.routed_experts.partial = static_lat
                runtime.shared_experts.partial = static_sh
                for x, y in zip(graph_out, ref_out):
                    same(x, y, f"A_pre after the graph's forward, join_after_lu={join_after_lu}")
            del graph
        mkl._K3_SHARED_JOIN_AFTER_LU = False
        mkl._K3_LATENT_AR_CUTE = "0"
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


def _run(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_join_marker_rank, *zip(*[(world,)] * world))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_shared_join_marker_tp2(mpi_pool_executor):
    _run(mpi_pool_executor)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_shared_join_marker_tp4(mpi_pool_executor):
    _run(mpi_pool_executor)
