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
"""Kimi K3 MoE block back-end: one merged all-reduce vs the latent + shared all-reduces.

``KimiK3MoERuntime.forward`` runs on every rank with stand-ins for the routed experts (a
per-rank latent partial) and the TP-sharded shared expert (a per-rank partial, either reduced by
the module itself like the BF16 ``GatedMLP`` or by ``shared_down_all_reduce`` after an FP8
weight-read down projection), real MNNVL all-reduces, the real latent RMSNorm and a BF16
latent up-projection. The merged path must match the two-all-reduce path and an FP32 reference,
eagerly (with and without the shared-expert stream) and under CUDA graph replay.
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
SHARED_SHARD = 384
EPS = 1e-6
# 1 and 7 run one-shot at TP2 with the default threshold, 224 two-shot.
SEQ_LENS = (1, 7, 32, 224)


class _RoutedStandIn:
    """Returns this rank's latent partial; writes it into ``moe_output`` when asked to."""

    def __init__(self, all_reduce, honor_output):
        self.all_reduce = all_reduce
        self.honor_output = honor_output
        self.partial = None
        self.routing_method = None
        self.use_dp = False
        self.backend = SimpleNamespace()
        self.wrote_into_output = False

    def __call__(self, x, router_logits, *, all_rank_num_tokens=None, moe_output=None, **kwargs):
        if moe_output is not None and self.honor_output:
            moe_output.copy_(self.partial)
            self.wrote_into_output = True
            return moe_output
        return self.partial.clone()


class _SharedStandIn(torch.nn.Module):
    """TP-sharded shared expert: BF16 (reduces itself) or FP8 down projection (does not)."""

    def __init__(self, all_reduce, fp8_down):
        super().__init__()
        self.all_reduce = all_reduce
        self.partial = None
        self.activation = None
        self.down_proj = fp8_down if fp8_down is not None else torch.nn.Identity()

    def forward(self, x, final_all_reduce_params=None):
        if not isinstance(self.down_proj, torch.nn.Identity):
            return self.down_proj(self.activation)
        out = self.partial.clone()
        if final_all_reduce_params is None or final_all_reduce_params.enable_allreduce:
            out = self.all_reduce(out)
        return out


def _rank_inputs(world, rank, num_tokens, seed):
    """Per-rank partials (and every rank's, for the reference) and the shared residual."""
    gen = torch.Generator(device="cuda")
    latents, shareds, acts = [], [], []
    for r in range(world):
        gen.manual_seed(seed * 1009 + r)
        latents.append(torch.randn((num_tokens, LATENT), generator=gen, device="cuda").bfloat16())
        shareds.append(torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda").bfloat16())
        acts.append(
            torch.randn((num_tokens, SHARED_SHARD), generator=gen, device="cuda").bfloat16()
        )
    gen.manual_seed(seed * 1009 + 999)
    residual = (torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda") * 4).bfloat16()
    return latents, shareds, acts, residual


def _merged_moe_rank(world, honor_output, shared_fp8):
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

        torch.manual_seed(0)
        fp8_down = None
        direct_writes = []
        if shared_fp8:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

            gemm_out = lowm.k3_fp8w_lowm_gemm_out

            def counting_gemm_out(x, w, w_scale, out):
                direct_writes.append(out.data_ptr())
                return gemm_out(x, w, w_scale, out)

            lowm.k3_fp8w_lowm_gemm_out = counting_gemm_out
            mkl._K3_FP8W_LOWM_GEMM = True
            dense = torch.nn.Linear(SHARED_SHARD, HIDDEN, bias=False, dtype=torch.bfloat16)
            dense.weight.data = (torch.randn_like(dense.weight) * 0.05).cuda()
            fp8_down = mkl._Fp8BlockScaleWeightReadLinear.from_linear(dense)

        runtime = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
        torch.nn.Module.__init__(runtime)
        runtime.hidden_size = HIDDEN
        runtime.moe_hidden_size = LATENT
        runtime.gate = SimpleNamespace(
            compute_logits=lambda h: torch.zeros((h.shape[0], 8), device=h.device)
        )
        runtime.routed_experts = _RoutedStandIn(ar_routed, honor_output)
        runtime.shared_experts = _SharedStandIn(ar_shared, fp8_down)
        runtime.shared_down_all_reduce = ar_shared if shared_fp8 else None
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
        runtime.shared_expert_stream = torch.cuda.Stream()
        runtime.latent_down_stream = None
        runtime.moe_main_event = torch.cuda.Event()
        runtime.moe_shared_event = torch.cuda.Event()
        runtime.moe_routed_event = runtime.moe_latent_fork_event = None
        runtime.moe_latent_ready_event = None
        mkl._K3_MERGED_MOE_AR_MAX_TOKENS = 0
        norm_w = runtime.routed_expert_norm.weight.float()

        def set_inputs(latents, shareds, acts):
            runtime.routed_experts.partial = latents[rank]
            runtime.shared_experts.partial = shareds[rank]
            runtime.shared_experts.activation = acts[rank]

        def run(merge, multi_stream, hidden, residual):
            runtime._merge_moe_allreduce = merge
            with with_multi_stream(multi_stream), torch.inference_mode():
                return runtime(hidden, residual=residual)

        def reference(latents, shareds, acts, residual):
            lat = sum(x.float() for x in latents).bfloat16().float()
            lat = (lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + EPS) * norm_w).bfloat16()
            routed = runtime.routed_expert_up_proj(lat).float()
            if shared_fp8:
                shareds = [fp8_down(a) for a in acts]
            shared = sum(x.float() for x in shareds).bfloat16().float()
            return residual.float() + (routed + shared).bfloat16().float()

        for num_tokens in SEQ_LENS:
            latents, shareds, acts, residual = _rank_inputs(world, rank, num_tokens, num_tokens)
            set_inputs(latents, shareds, acts)
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            ref = reference(latents, shareds, acts, residual)
            old = run(False, False, hidden, residual)
            torch.testing.assert_close(old.float(), ref, rtol=0.02, atol=0.05)
            for multi_stream in (False, True):
                runtime.routed_experts.wrote_into_output = False
                direct_writes.clear()
                new = run(True, multi_stream, hidden, residual)
                assert runtime.routed_experts.wrote_into_output == honor_output
                if shared_fp8 and lowm.can_run(acts[rank], fp8_down.weight, fp8_down.weight_scale):
                    # The low-M down projection wrote straight into the merged buffer.
                    assert len(direct_writes) == 1
                torch.testing.assert_close(new.float(), ref, rtol=0.02, atol=0.05)
                # Same sums and the same RMSNorm math: at most one BF16 rounding step apart.
                torch.testing.assert_close(new, old, rtol=2**-7, atol=2**-6)

            # CUDA graph: capture the merged path once, replay it on new partials.
            static_lat = latents[rank].clone()
            static_sh = shareds[rank].clone()
            static_act = acts[rank].clone()
            static_res = residual.clone()
            runtime.routed_experts.partial = static_lat
            runtime.shared_experts.partial = static_sh
            runtime.shared_experts.activation = static_act
            for _ in range(2):
                run(True, True, hidden, static_res)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            runtime._merge_moe_allreduce = True
            with with_multi_stream(True), torch.inference_mode(), torch.cuda.graph(graph):
                graph_out = runtime(hidden, residual=static_res)
            for step in range(3):
                latents, shareds, acts, residual = _rank_inputs(
                    world, rank, num_tokens, 100 + 7 * step + num_tokens
                )
                static_lat.copy_(latents[rank])
                static_sh.copy_(shareds[rank])
                static_act.copy_(acts[rank])
                static_res.copy_(residual)
                graph.replay()
                torch.cuda.synchronize()
                set_inputs(latents, shareds, acts)
                old = run(False, False, hidden, residual)
                torch.testing.assert_close(graph_out, old, rtol=2**-7, atol=2**-6)
                torch.testing.assert_close(
                    graph_out.float(),
                    reference(latents, shareds, acts, residual),
                    rtol=0.02,
                    atol=0.05,
                )
            del graph

        # Above TRTLLM_K3_MERGED_MOE_AR_MAX_TOKENS the runtime keeps the two all-reduces.
        num_tokens = SEQ_LENS[-1]
        mkl._K3_MERGED_MOE_AR_MAX_TOKENS = num_tokens - 1
        latents, shareds, acts, residual = _rank_inputs(world, rank, num_tokens, 5)
        set_inputs(latents, shareds, acts)
        hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        runtime.routed_experts.wrote_into_output = False
        limited = run(True, True, hidden, residual)
        assert not runtime.routed_experts.wrote_into_output
        torch.testing.assert_close(limited, run(False, True, hidden, residual), rtol=0, atol=0)
        mkl._K3_MERGED_MOE_AR_MAX_TOKENS = 0
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


def _run(mpi_pool_executor, honor_output, shared_fp8):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _merged_moe_rank, *zip(*[(world, honor_output, shared_fp8)] * world)
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("honor_output", [True, False], ids=["direct", "copied"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_merged_moe_allreduce_bf16_shared(mpi_pool_executor, honor_output):
    _run(mpi_pool_executor, honor_output, shared_fp8=False)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the FP8 low-M GEMM runs on SM100/SM103",
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_merged_moe_allreduce_fp8_shared_tp4(mpi_pool_executor):
    _run(mpi_pool_executor, honor_output=True, shared_fp8=True)
