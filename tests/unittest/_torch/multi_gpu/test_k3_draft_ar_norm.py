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
"""One-kernel MNNVL all-reduce + residual + RMSNorm (k3_draft_ar_norm, TLLM_DSPARK_AR_NORM): bitwise
equal to the C++ RESIDUAL_RMS_NORM fusion (one- and two-shot kernels, every norm launch shape), with
C++ calls interleaved on the workspace and inside a CUDA graph."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)


def _bf16(m, h, seed, scale, special=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, h, device="cuda", generator=gen) * scale).to(torch.bfloat16)
    if special:
        bits = x.view(torch.int16)
        r = torch.rand(m, h, device="cuda", generator=gen)
        bits[r < 0.05] = -32768  # -0
        bits[(r >= 0.05) & (r < 0.08)] = 0
        den = (r >= 0.08) & (r < 0.11)
        bits[den] = torch.randint(1, 127, (int(den.sum()),), device="cuda", generator=gen).to(
            torch.int16
        )
    return x


class _Norm:
    def __init__(self, h, seed, eps):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.weight = (1 + 0.2 * torch.randn(h, device="cuda", generator=gen)).to(torch.bfloat16)
        self.variance_epsilon = eps


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    os.environ["TLLM_DSPARK_AR_NORM"] = "1"
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_draft_ar_norm as kernel
    from tensorrt_llm._torch.distributed import (
        AllReduce,
        AllReduceFusionOp,
        AllReduceParams,
        AllReduceStrategy,
    )
    from tensorrt_llm._torch.speculative.dspark_ar_norm import draft_ar_norm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        sms = torch.cuda.get_device_properties(rank).multi_processor_count

        def cpp(x, res, norm):
            return allreduce(
                x,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=res,
                    norm_weight=norm.weight,
                    eps=norm.variance_epsilon,
                ),
            )

        def differ(a, b):
            return int((a.view(torch.int16) != b.view(torch.int16)).sum())

        bad = 0
        shapes = set()
        # Rows covering both C++ transports and the norm launch shapes 112 x 8, 224 x 4, 448 x 2
        # (7168) and 112 x 4 (3584), with C++ calls on the workspace before and after each.
        for h in (7168, 3584):
            for m in (1, 7, 14, 19, 28, 32, 40):
                cfg = kernel.grid_config(m, h, sms)
                if cfg is None:
                    continue
                shapes.add(cfg)
                for kind in range(3):
                    x = _bf16(m, h, 100 * m + 10 * kind + rank, 0.5, special=kind == 1)
                    res = _bf16(m, h, 7 + m + kind, 2.0 if kind == 2 else 1.0)
                    norm = _Norm(h, 3 + kind, 1e-5 if kind else 1e-6)
                    ref_out, ref_res = cpp(x, res, norm)
                    got = draft_ar_norm(allreduce, x, res, norm)
                    assert got is not None, (m, h)
                    bad += differ(ref_out, got[0]) + differ(ref_res, got[1])
                    again_out, again_res = cpp(x, res, norm)
                    bad += differ(ref_out, again_out) + differ(ref_res, again_res)
        assert {(112, 8), (224, 4)} <= shapes, shapes

        # One CUDA graph alternating the CuTe and the C++ path on the workspace, replayed with
        # new inputs.
        m, h = 14, 7168
        norm = _Norm(h, 11, 1e-5)
        xs = [_bf16(m, h, 500 + i, 0.5) for i in range(4)]
        rs = [_bf16(m, h, 600 + i, 1.0) for i in range(4)]

        def calls():
            outs = []
            for i in range(4):
                if i % 2 == 0:
                    outs.append(draft_ar_norm(allreduce, xs[i], rs[i], norm))
                else:
                    outs.append(cpp(xs[i], rs[i], norm))
            return outs

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(4):
                xs[i].copy_(_bf16(m, h, 900 + 10 * rep + i + 100 * rank, 0.5))
                rs[i].copy_(_bf16(m, h, 950 + 10 * rep + i, 1.0))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for i in range(4):
                ref_out, ref_res = cpp(xs[i], rs[i], norm)
                bad += differ(ref_out, results[i][0]) + differ(ref_res, results[i][1])
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the C++ path"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_draft_ar_norm(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
