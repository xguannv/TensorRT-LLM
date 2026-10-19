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
"""Pushed MNNVL all-reduce (k3_lamport_ar + the low-M GEMM push epilogue): bitwise equal to the
C++ MNNVL all-reduce, with fused and C++ calls interleaved on one workspace."""

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


def _special(m, h, gen):
    """BF16 ``[m, h]`` with -0, +0, denormals, large values, inf and NaN lanes."""
    x = torch.randn(m, h, device="cuda", generator=gen) * torch.exp2(
        torch.randint(-20, 20, (m, 1), device="cuda", generator=gen).float()
    )
    x = x.to(torch.bfloat16)
    bits = x.view(torch.int16)
    r = torch.rand(m, h, device="cuda", generator=gen)
    bits[r < 0.05] = -32768
    bits[(r >= 0.05) & (r < 0.08)] = 0
    den = (r >= 0.08) & (r < 0.11)
    bits[den] = torch.randint(1, 127, (int(den.sum()),), device="cuda", generator=gen).to(
        torch.int16
    )
    bits[(r >= 0.11) & (r < 0.115)] = 0x7F80
    bits[(r >= 0.115) & (r < 0.117)] = 0x7FC1
    return x


def _input(m, h, seed, rank, kind):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    if kind == 0:
        return (torch.randn(m, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    if kind == 1:
        return _special(m, h, gen)
    # Every rank sends denormals only: the sums stay denormal (no flush to zero anywhere).
    bits = torch.randint(-127, 127, (m, h), device="cuda", generator=gen).to(torch.int16)
    return bits.view(torch.bfloat16)


def _weights(n, k, seed):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.exp2(
        torch.randint(-8, -2, (n // 128, k // 128), device="cuda", generator=gen).float()
    )
    return lowm.prepare_k3_fp8w_lowm_weight(w, scale)


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_lamport_ar as lamport
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy

    # Imported by name: an attribute named ``ops`` next to the global ``torch`` would make
    # cloudpickle ship the ``torch.ops`` object with this function.
    from tensorrt_llm._torch.distributed.ops import (
        _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
        get_or_scale_allreduce_mnnvl_workspace,
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
        workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        ws = lamport.LamportWorkspace.from_mnnvl(workspace, world_size, rank)
        threshold = _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        bad = 0

        def differ(ref, out):
            return int((ref.view(torch.int16) != out.view(torch.int16)).sum())

        # Push kernel + finish kernel against the C++ all-reduce, both transports reproducing
        # the C++ kernel the size selects, with C++ calls in between.
        for m, h in ((1, 7168), (14, 7168), (64, 7168), (300, 7168), (14, 3584), (5, 64)):
            cpp_one_shot = lamport.one_shot_for(m, h, world_size, threshold)
            for kind in (0, 1, 2):
                x = _input(m, h, 3 * m + kind, rank, kind)
                ref = allreduce(x)
                for one_shot in (False, True):
                    if lamport.required_bytes(m, h, world_size, one_shot) > ws.bytes_per_buffer:
                        continue
                    out = torch.empty_like(x)
                    lamport.run_lamport_push(ws, x, one_shot)
                    lamport.run_lamport_finish(ws, out, one_shot, cpp_one_shot=cpp_one_shot)
                    bad += differ(ref, out)
                    bad += differ(ref, allreduce(x))

        # Two-shot finish in place + gather (what a consumer polling the result rows reads).
        for m in (1, 100, 300):
            cpp_one_shot = lamport.one_shot_for(m, 7168, world_size, threshold)
            x = _input(m, 7168, 41 * m, rank, 1)
            ref = allreduce(x)
            lamport.run_lamport_push(ws, x, False)
            lamport.run_lamport_finish_in_place(ws, m, 7168, cpp_one_shot=cpp_one_shot)
            out = lamport.run_lamport_gather(ws, torch.empty_like(x))
            bad += differ(ref, out)
            bad += differ(ref, allreduce(x))

        # GEMM push epilogue + finish against the GEMM followed by the C++ all-reduce.
        for n, k in ((7168, 768), (7168, 384)):
            w, w_scale = _weights(n, k, 11 + rank)
            for m in (1, 7, 14, 32, 64):
                cpp_one_shot = lamport.one_shot_for(m, n, world_size, threshold)
                x = _input(m, k, 5 * m, rank, 0)
                y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                lowm.run_k3_fp8w_lowm_gemm(x, w, w_scale, y)
                ref = allreduce(y)
                for one_shot in (False, True):
                    lowm.run_k3_fp8w_lowm_gemm_ar_push(x, w, w_scale, ws, one_shot)
                    out = lamport.run_lamport_finish(
                        ws, torch.empty_like(y), one_shot, cpp_one_shot=cpp_one_shot
                    )
                    bad += differ(ref, out)

        # One CUDA graph with pushed and C++ calls on the workspace, replayed with new inputs.
        n, k, m = 7168, 768, 14
        cpp_one_shot = lamport.one_shot_for(m, n, world_size, threshold)
        w, w_scale = _weights(n, k, 99 + rank)
        xs = [_input(m, k, 700 + i, rank, 0) for i in range(4)]
        ys = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        outs = [torch.empty_like(y) for y in ys]

        def calls():
            res = []
            for i in range(4):
                if i % 2 == 0:
                    lowm.run_k3_fp8w_lowm_gemm_ar_push(xs[i], w, w_scale, ws, cpp_one_shot)
                    res.append(
                        lamport.run_lamport_finish(
                            ws, outs[i], cpp_one_shot, cpp_one_shot=cpp_one_shot
                        )
                    )
                else:
                    lowm.run_k3_fp8w_lowm_gemm(xs[i], w, w_scale, ys[i])
                    res.append(allreduce(ys[i]))
            return res

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(4):
                xs[i].copy_(_input(m, k, 900 + 10 * rep + i, rank, 0))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for i in range(4):
                y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                lowm.run_k3_fp8w_lowm_gemm(xs[i], w, w_scale, y)
                bad += differ(allreduce(y), results[i])
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the C++ path"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_amid(world_size):
    """o_proj push -> in-place finish -> A_mid reading the Lamport rows (TRTLLM_K3_AR_PUSH_AMID)
    against o_proj -> C++ all-reduce -> A_mid, through the model helpers."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import importlib

    from torch import nn

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    # Not ``from ... import ops``: see _run_rank.
    dist_ops = importlib.import_module("tensorrt_llm._torch.distributed.ops")
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    switches = {
        (dist_ops, "_K3_AR_PUSH"): True,
        (kimi, "_K3_AR_PUSH_AMID"): True,
        (kimi, "_K3_FP8W_LOWM_GEMM"): True,
        (kimi, "_K3_ATTN_RES_PDL"): True,
    }
    saved = {key: getattr(*key) for key in switches}
    try:
        for (module, name), value in switches.items():
            setattr(module, name, value)
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        hidden = 7168
        w, w_scale = _weights(hidden, 768, 21 + rank)
        linear = kimi._Fp8BlockScaleWeightReadLinear(w, w_scale, hidden)
        gen = torch.Generator(device="cuda").manual_seed(3)
        proj = nn.Linear(hidden, 1, bias=False, dtype=torch.bfloat16, device="cuda")
        proj.weight.data.copy_(torch.randn(1, hidden, device="cuda", generator=gen) * 0.05)
        norm = kimi.KimiK3RMSNorm(hidden, eps=1e-6, dtype=torch.bfloat16, device="cuda")
        norm.weight.data.copy_(1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen))
        out_norm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16).cuda()
        out_norm.weight.data.copy_(1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen))
        bad = 0
        in_place = 0
        for m in (14, 32, 64):
            x = _input(m, 768, 9 * m, rank, 0)
            prefix = (torch.randn(m, hidden, device="cuda", generator=gen)).to(torch.bfloat16)
            block = (torch.randn(4, m, hidden, device="cuda", generator=gen)).to(torch.bfloat16)
            ref = kimi._apply_attn_res_add_and_rmsnorm(
                prefix, allreduce(linear(x)), block, proj, norm, out_norm
            )
            target = allreduce.lamport_push_target(m, hidden, in_place=True)
            linear.next_ar_push = target
            linear(x)
            assert target.pushed
            result = allreduce.finish_lamport_push(target, in_place=True)
            in_place += int(isinstance(result, dist_ops.LamportResult))
            got = kimi._apply_attn_res_add_and_rmsnorm(prefix, result, block, proj, norm, out_norm)
            for a, b in zip(ref, got):
                bad += int((a.view(torch.int16) != b.view(torch.int16)).sum())
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
        # Every size took the in-place path (the two-shot transport, also where the C++ kernel
        # is one-shot).
        assert in_place == 3
    except Exception:
        traceback.print_exc()
        raise
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_ar_push_amid(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank_amid, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_lamport_ar(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
