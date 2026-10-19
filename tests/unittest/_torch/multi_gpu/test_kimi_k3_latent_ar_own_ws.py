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
"""Kimi K3 MoE block: the routed latent all-reduce on its own MNNVL workspace.

``TRTLLM_K3_LATENT_AR_OWN_WS``: ``KimiK3MoERuntime.forward`` runs on every rank with stand-ins for
the routed experts (a per-rank latent partial) and the TP-sharded BF16 shared expert (which reduces
its output on the shared MNNVL workspace), the real latent RMSNorm and a BF16 latent up-projection.
With the latent all-reduce on its own workspace (C++ kernels, or with ``TRTLLM_K3_LATENT_AR_CUTE``
``k3_latent_ar_norm`` for the one-shot sizes; with ``TRTLLM_K3_LATENT_AR_ROW`` its row kernel, with
``TRTLLM_K3_LATENT_AR_SIDE_CLEAR`` the Lamport buffer re-armed on the latent-down stream, with
``TRTLLM_K3_LU_MX_STAGES`` latent_up's MXFP8 GEMM on other stages) the result (and the deferred
residual) must be bit-identical to the shared-workspace path, eagerly and under CUDA graph replay,
and the own workspace must serve exactly the forwards it applies to.
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
# One-shot and two-shot sizes at TP2 / TP4 with the default threshold, up to the workspace limit.
SEQ_LENS = (1, 7, 14, 18, 19, 32, 64)


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


class _SharedStandIn(torch.nn.Module):
    """TP-sharded BF16 shared expert that reduces its partial itself."""

    def __init__(self, all_reduce):
        super().__init__()
        self.all_reduce = all_reduce
        self.partial = None
        self.down_proj = torch.nn.Identity()

    def forward(self, x, final_all_reduce_params=None):
        out = self.partial.clone()
        if final_all_reduce_params is None or final_all_reduce_params.enable_allreduce:
            out = self.all_reduce(out)
        return out


def _rank_inputs(world, num_tokens, seed):
    """Every rank's latent and shared partials, and the residual."""
    gen = torch.Generator(device="cuda")
    latents, shareds = [], []
    for r in range(world):
        gen.manual_seed(seed * 1009 + r)
        latents.append(torch.randn((num_tokens, LATENT), generator=gen, device="cuda").bfloat16())
        shareds.append(torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda").bfloat16())
    gen.manual_seed(seed * 1009 + 999)
    residual = (torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda") * 4).bfloat16()
    return latents, shareds, residual


def _own_ws_rank(world):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    try:
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
        # The same workspace again for the same group; a larger one is refused.
        assert (
            mkl._k3_latent_ar_workspace(
                ar_routed.mnnvl_allreduce.mapping, torch.bfloat16, LATENT, MAX_TOKENS
            )
            is workspace
        )
        with pytest.raises(ValueError):
            mkl._k3_latent_ar_workspace(
                ar_routed.mnnvl_allreduce.mapping, torch.bfloat16, LATENT, MAX_TOKENS + 1
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
        runtime.shared_experts = _SharedStandIn(ar_shared)
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
        flags = workspace["flags"]

        def set_inputs(latents, shareds):
            runtime.routed_experts.partial = latents[rank]
            runtime.shared_experts.partial = shareds[rank]

        def run(own_ws, multi_stream, hidden, residual, defer=False):
            runtime._latent_ar_ws = workspace if own_ws else None
            with with_multi_stream(multi_stream), torch.inference_mode():
                return runtime(hidden, residual=residual, defer_residual=defer)

        def check_rotated(before, launches):
            torch.cuda.synchronize()
            assert int(flags[0].item()) == (before + launches) % 3

        def assert_same(a, b):
            if isinstance(a, mkl._DeferredMoEResidual):
                assert isinstance(b, mkl._DeferredMoEResidual)
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)

        # The C++ one-shot kernel, then (TRTLLM_K3_LATENT_AR_CUTE) k3_latent_ar_norm on the own workspace.
        for cute_mode, num_tokens in [(c, n) for c in ("0", "1") for n in SEQ_LENS]:
            mkl._K3_LATENT_AR_CUTE = cute_mode
            latents, shareds, residual = _rank_inputs(world, num_tokens, num_tokens)
            set_inputs(latents, shareds)
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            for defer in (False, True):
                ref = run(False, True, hidden, residual, defer)
                torch.cuda.synchronize()
                before = int(flags[0].item())
                out = run(True, True, hidden, residual, defer)
                check_rotated(before, 1)
                assert_same(out, ref)
            # Single stream: the join cannot move, the shared workspace serves the forward.
            before = int(flags[0].item())
            out = run(True, False, hidden, residual)
            check_rotated(before, 0)
            assert_same(out, run(False, False, hidden, residual))
            # A latent_up that adds the shared output itself keeps the shared workspace.
            runtime._latent_up_reads_shared = lambda latent: True
            before = int(flags[0].item())
            out = run(True, True, hidden, residual)
            check_rotated(before, 0)
            del runtime._latent_up_reads_shared
            assert_same(out, run(False, True, hidden, residual))

            # CUDA graph of the own-workspace path, replayed on new partials between eager
            # shared-workspace forwards.
            static_lat = latents[rank].clone()
            static_sh = shareds[rank].clone()
            static_res = residual.clone()
            runtime.routed_experts.partial = static_lat
            runtime.shared_experts.partial = static_sh
            for _ in range(2):
                run(True, True, hidden, static_res)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            runtime._latent_ar_ws = workspace
            with with_multi_stream(True), torch.inference_mode(), torch.cuda.graph(graph):
                graph_out = runtime(hidden, residual=static_res)
            for step in range(3):
                latents, shareds, residual = _rank_inputs(world, num_tokens, 100 + 7 * step)
                static_lat.copy_(latents[rank])
                static_sh.copy_(shareds[rank])
                static_res.copy_(residual)
                before = int(flags[0].item())
                graph.replay()
                check_rotated(before, 1)
                set_inputs(latents, shareds)
                assert_same(graph_out, run(False, True, hidden, residual))
            del graph

        # FP8 latent_up (low-M kernel): with TRTLLM_K3_LU_MXFP8 the CuTe all-reduce also hands it
        # the MXFP8 form of the latent (where this build has forward_mxfp8).
        mkl._K3_FP8W_LOWM_GEMM = True
        dense = torch.nn.Linear(LATENT, HIDDEN, bias=False, dtype=torch.bfloat16)
        dense.weight.data = (torch.randn_like(dense.weight) * 0.02).cuda()
        up = mkl._Fp8BlockScaleWeightReadLinear.from_linear(dense)
        runtime.routed_expert_up_proj = up
        mx_calls = []
        if hasattr(up, "forward_mxfp8"):
            forward_mxfp8 = up.forward_mxfp8

            def counting_forward_mxfp8(x, xq, xs):
                mx_calls.append(x.shape[0])
                return forward_mxfp8(x, xq, xs)

            up.forward_mxfp8 = counting_forward_mxfp8
        mkl._K3_LATENT_AR_CUTE = "1"
        mkl._K3_LU_MXFP8 = True
        for num_tokens in (1, 7, 14):
            latents, shareds, residual = _rank_inputs(world, num_tokens, 40 + num_tokens)
            set_inputs(latents, shareds)
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            for defer in (False, True):
                assert_same(
                    run(True, True, hidden, residual, defer),
                    run(False, True, hidden, residual, defer),
                )
        if hasattr(up, "forward_mxfp8"):
            assert mx_calls == [1, 1, 7, 7, 14, 14]

        # TRTLLM_K3_LATENT_AR_ROW (row kernel), TRTLLM_K3_LU_MX_STAGES (latent_up's MXFP8 GEMM stages)
        # and TRTLLM_K3_LATENT_AR_SIDE_CLEAR (the latent-down stream re-arms the Lamport buffer after
        # latent_up and marks the state clean; the test joins that stream after every forward, as the
        # next MoE layer's latent_down join does). Eagerly and under CUDA graph replay.
        latent_stream = torch.cuda.Stream()
        next_layer = SimpleNamespace(latent_down_stream=latent_stream)
        runtime._next_moe_layer = lambda: next_layer
        runtime.latent_ar_clear_stream = latent_stream
        mkl._K3_LATENT_AR_ROW = True
        for stages, side_clear in (("", False), ("12:8", True), ("12:6", True)):
            mkl._K3_LU_MX_STAGES = stages
            runtime.moe_lu_event = torch.cuda.Event() if side_clear else None
            for num_tokens in (1, 7, 14, 18):
                latents, shareds, residual = _rank_inputs(world, num_tokens, 60 + num_tokens)
                set_inputs(latents, shareds)
                hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
                for defer in (False, True):
                    ref = run(False, True, hidden, residual, defer)
                    out = run(True, True, hidden, residual, defer)
                    torch.cuda.current_stream().wait_stream(latent_stream)
                    assert_same(out, ref)
                    torch.cuda.synchronize()
                    # The side clear left the state clean (no bytes to clear for the next call).
                    assert (int(flags[3].item()) == 0) == side_clear
            # CUDA graph: two forwards per replay, the latent-down stream joined after each.
            num_tokens = 14
            latents, shareds, residual = _rank_inputs(world, num_tokens, 80)
            static_lat = latents[rank].clone()
            static_sh = shareds[rank].clone()
            static_res = residual.clone()
            runtime.routed_experts.partial = static_lat
            runtime.shared_experts.partial = static_sh
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            for _ in range(2):
                run(True, True, hidden, static_res)
                torch.cuda.current_stream().wait_stream(latent_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            runtime._latent_ar_ws = workspace
            with with_multi_stream(True), torch.inference_mode(), torch.cuda.graph(graph):
                first = runtime(hidden, residual=static_res)
                if side_clear:
                    torch.cuda.current_stream().wait_stream(latent_stream)
                second = runtime(hidden, residual=static_res)
                if side_clear:
                    torch.cuda.current_stream().wait_stream(latent_stream)
            for step in range(3):
                latents, shareds, residual = _rank_inputs(world, num_tokens, 90 + 7 * step)
                static_lat.copy_(latents[rank])
                static_sh.copy_(shareds[rank])
                static_res.copy_(residual)
                before = int(flags[0].item())
                graph.replay()
                check_rotated(before, 2)
                set_inputs(latents, shareds)
                ref = run(False, True, hidden, residual)
                assert_same(first, ref)
                assert_same(second, ref)
            del graph
        runtime.moe_lu_event = None
        runtime.latent_ar_clear_stream = None
        del runtime._next_moe_layer
        mkl._K3_LATENT_AR_ROW = False
        mkl._K3_LU_MX_STAGES = ""
        mkl._K3_LU_MXFP8 = False
        mkl._K3_LATENT_AR_CUTE = "0"
        # Above the workspace's token limit the shared workspace serves the forward.
        num_tokens = MAX_TOKENS + 1
        latents, shareds, residual = _rank_inputs(world, num_tokens, 5)
        set_inputs(latents, shareds)
        hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        before = int(flags[0].item())
        out = run(True, True, hidden, residual)
        check_rotated(before, 0)
        assert_same(out, run(False, True, hidden, residual))
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


def _run(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_own_ws_rank, *zip(*[(world,)] * world))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_latent_ar_own_workspace_tp2(mpi_pool_executor):
    _run(mpi_pool_executor)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_latent_ar_own_workspace_tp4(mpi_pool_executor):
    _run(mpi_pool_executor)
