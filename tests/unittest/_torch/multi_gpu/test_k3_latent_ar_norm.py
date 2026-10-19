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
"""``k3_latent_ar_norm``: CuTe one-shot MNNVL all-reduce + RMSNorm vs the C++ kernel.

On every rank: the CuTe kernel (early and late PDL trigger, eagerly and under CUDA graph replay)
must match ``mnnvl_fusion_allreduce`` (``RMS_NORM``, and ``NONE`` for the plain sum) bit for bit,
while it shares its Lamport workspace with the C++ one-shot and two-shot kernels in any order.
The 16-rank chunked summation order is checked on the available GPUs by filling the other ranks'
slots of the Lamport buffer through the multicast address (sum only, against an FP32 reference).
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 3584
EPS = 1e-6
MAX_TOKENS = 64
ROWS = (1, 2, 7, 14, 16, 18)


def _inputs(world, num_tokens, seed):
    """Every rank's partial; seed 2 adds large values, exact zeros and BF16 -0."""
    gen = torch.Generator(device="cuda")
    xs = []
    for r in range(world):
        gen.manual_seed(seed * 1009 + r)
        x = torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda")
        if seed % 3 == 2:
            x[:, ::7] *= 300.0
            x[:, 1::11] = 0.0
            x[:, 2::13] = -0.0
        xs.append(x.bfloat16())
    return xs


def _ar_norm_rank(world):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    try:
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_latent_ar_norm as kan
        from tensorrt_llm._torch.distributed import AllReduce
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
        from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
        from tensorrt_llm.functional import AllReduceStrategy

        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
        assert ar.mnnvl_allreduce is not None
        shared_ws = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        own = mkl._k3_latent_ar_workspace(mapping, torch.bfloat16, HIDDEN, MAX_TOKENS)
        comm = shared_ws["mpi_comm"]
        torch.manual_seed(0)
        gamma = (torch.rand(HIDDEN, device="cuda") + 0.5).bfloat16()

        def cpp(x, op, workspace):
            if workspace is own:
                buf, flags = own["buffer"], own["flags"]
            else:
                buf = workspace["uc_buffer"].view(torch.bfloat16).view(3, -1)
                flags = workspace["buffer_flags"]
            norm = op != AllReduceFusionOp.NONE
            (out,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                x,
                gamma if norm else None,
                None,
                EPS if norm else None,
                buf,
                flags,
                norm,
                None,
                int(op),
            )
            return out

        def cute_ar(x, early=True, norm=True, world_size=world):
            out = torch.empty_like(x)
            kan.run_k3_latent_ar_norm(
                x,
                gamma,
                EPS,
                out,
                uc_addr=own["buffer"].data_ptr(),
                mc_addr=own["mc"].data_ptr(),
                flags=own["flags"],
                rank=rank,
                world_size=world_size,
                early_trigger=early,
                norm=norm,
            )
            return out

        def same(a, b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

        # Bit-identical to the C++ kernel; the C++ one-shot and two-shot kernels use the same
        # workspace before and after it.
        for num_tokens in ROWS:
            for seed in range(3):
                x = _inputs(world, num_tokens, seed)[rank]
                assert kan.can_run(x, gamma, world)
                ref = cpp(x, AllReduceFusionOp.RMS_NORM, shared_ws)
                ref_sum = cpp(x, AllReduceFusionOp.NONE, shared_ws)
                same(cute_ar(x), ref)
                same(cute_ar(x, early=False), ref)
                same(cpp(x, AllReduceFusionOp.RMS_NORM, own), ref)
                same(cute_ar(x, norm=False), ref_sum)
                # Two-shot sizes on the same workspace in between (and the one-shot again).
                big = _inputs(world, MAX_TOKENS, seed + 10)[rank]
                same(
                    cpp(big, AllReduceFusionOp.RMS_NORM, own),
                    cpp(big, AllReduceFusionOp.RMS_NORM, shared_ws),
                )
                same(cute_ar(x), ref)
        torch.cuda.synchronize()

        # CUDA graph: two launches per replay, new inputs each replay.
        for num_tokens in (7, 14):
            static_x = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            static_x.copy_(_inputs(world, num_tokens, 0)[rank])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                cute_ar(static_x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    first = cute_ar(static_x)
                    second = cute_ar(first, early=False)
            torch.cuda.synchronize()
            for step in range(3):
                static_x.copy_(_inputs(world, num_tokens, 20 + step)[rank])
                comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                ref = cpp(static_x, AllReduceFusionOp.RMS_NORM, shared_ws)
                same(first, ref)
                same(second, cpp(ref, AllReduceFusionOp.RMS_NORM, shared_ws))
            del graph

        # 16 ranks on `world` GPUs (sum only): rank r also stands for ranks r + world, ...;
        # their slots of the current buffer are filled through the multicast address first.
        if 16 % world == 0 and world < 16:
            buf_bytes = int(own["flags"][2].item())
            for num_tokens in (1, 7, 16):
                if num_tokens * 16 * HIDDEN * 2 > buf_bytes:
                    # The workspace is sized for the real world size.
                    continue
                xs = _inputs(16, num_tokens, 3)
                torch.cuda.synchronize()
                cur = int(own["flags"][0].item())
                mc = own["mc"].view(torch.bfloat16)
                start = cur * buf_bytes // 2
                stage = mc[start : start + num_tokens * 16 * HIDDEN].view(num_tokens, 16, HIDDEN)
                for v in range(rank + world, 16, world):
                    sanitized = torch.where(xs[v] == 0, torch.zeros_like(xs[v]), xs[v])
                    stage[:, v, :].copy_(sanitized)
                torch.cuda.synchronize()
                comm.Barrier()
                out = cute_ar(xs[rank], norm=False, world_size=16)
                torch.cuda.synchronize()
                chunks = []
                for base in (0, 8):
                    c = torch.zeros((num_tokens, HIDDEN), dtype=torch.float32, device="cuda")
                    for v in range(base, base + 8):
                        c = c + xs[v].float()
                    chunks.append(c)
                expect = ((torch.zeros_like(chunks[0]) + chunks[0]) + chunks[1]).bfloat16()
                same(out, expect)
                comm.Barrier()
            # The C++ kernel resumes on the workspace after the 16-rank launches.
            x = _inputs(world, 14, 1)[rank]
            same(cpp(x, AllReduceFusionOp.RMS_NORM, own), cute_ar(x))
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


def _run(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_ar_norm_rank, *zip(*[(world,)] * world))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)), reason="SM100 / SM103 kernel"
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_latent_ar_norm_tp2(mpi_pool_executor):
    _run(mpi_pool_executor)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)), reason="SM100 / SM103 kernel"
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_latent_ar_norm_tp4(mpi_pool_executor):
    _run(mpi_pool_executor)
