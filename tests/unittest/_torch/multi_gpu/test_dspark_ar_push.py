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
"""DSpark drafter projections pushed into their all-reduce (TLLM_DSPARK_AR_PUSH): the low-M GEMM
push epilogue + the finish / residual / RMSNorm kernel (with its cluster sums on a cluster barrier
or, TLLM_DSPARK_AR_FINISH_ASYNC, on an mbarrier) are bitwise equal to F.linear followed by the C++
RESIDUAL_RMS_NORM fusion, with C++ calls interleaved on the workspace and inside a CUDA graph."""

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

HIDDEN = 7168


class _Norm:
    def __init__(self, seed, eps):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.weight = (1 + 0.2 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(
            torch.bfloat16
        )
        self.variance_epsilon = eps


def _rnd(shape, seed, scale):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)


def _run_rank(world_size, finish_async):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    os.environ["TLLM_DSPARK_AR_PUSH"] = "1"
    os.environ["TLLM_DSPARK_AR_FINISH_ASYNC"] = "1" if finish_async else "0"
    import torch.nn.functional as F

    from tensorrt_llm._torch.distributed import (
        AllReduce,
        AllReduceFusionOp,
        AllReduceParams,
        AllReduceStrategy,
    )
    from tensorrt_llm._torch.speculative.dspark_ar_push import (
        PushedPartial,
        draft_ar_finish_norm,
        draft_gemm_push,
    )
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

        def cpp(partial, res, norm):
            return allreduce(
                partial,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=res,
                    norm_weight=norm.weight,
                    eps=norm.variance_epsilon,
                ),
            )

        def pushed(x, w, res, norm):
            p = draft_gemm_push(allreduce, x, w)
            assert isinstance(p, PushedPartial)
            return draft_ar_finish_norm(p, res, norm)

        def differ(a, b):
            return int((a.view(torch.int16) != b.view(torch.int16)).sum())

        bad = 0
        for k in (512, 896):
            w = _rnd((HIDDEN, k), 100 + k + rank, 0.03)
            for m in (1, 7, 14, 19, 28, 32):
                for kind in range(2):
                    x = _rnd((m, k), 10 * m + kind + 1000 * rank, 1.0 if kind == 0 else 4.0)
                    res = _rnd((m, HIDDEN), 7 + m + kind, 1.0)
                    norm = _Norm(3 + kind, 1e-5 if kind else 1e-6)
                    ref_out, ref_res = cpp(F.linear(x, w), res, norm)
                    got_out, got_res = pushed(x, w, res, norm)
                    bad += differ(ref_out, got_out) + differ(ref_res, got_res)
                    again_out, again_res = cpp(F.linear(x, w), res, norm)
                    bad += differ(ref_out, again_out) + differ(ref_res, again_res)

        # One CUDA graph alternating pushed and C++ calls on the workspace, replayed with new
        # inputs.
        m, k = 14, 512
        w = _rnd((HIDDEN, k), 77 + rank, 0.03)
        norm = _Norm(11, 1e-5)
        xs = [_rnd((m, k), 500 + i + 10 * rank, 1.0) for i in range(4)]
        rs = [_rnd((m, HIDDEN), 600 + i, 1.0) for i in range(4)]

        def calls():
            outs = []
            for i in range(4):
                if i % 2 == 0:
                    outs.append(pushed(xs[i], w, rs[i], norm))
                else:
                    outs.append(cpp(F.linear(xs[i], w), rs[i], norm))
            return outs

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(4):
                xs[i].copy_(_rnd((m, k), 900 + 10 * rep + i + 100 * rank, 1.0))
                rs[i].copy_(_rnd((m, HIDDEN), 950 + 10 * rep + i, 1.0))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for i in range(4):
                ref_out, ref_res = cpp(F.linear(xs[i], w), rs[i], norm)
                bad += differ(ref_out, results[i][0]) + differ(ref_res, results[i][1])
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the C++ path"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("finish_async", [False, True])
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_dspark_ar_push(mpi_pool_executor, finish_async):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size, finish_async)] * world_size))
    for r in results:
        assert r is True
