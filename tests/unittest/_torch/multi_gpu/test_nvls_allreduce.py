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
"""Multicast (NVLink SHARP) all-reduce: values, cross-rank bit identity, CUDA graphs."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_hopper

import tensorrt_llm
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.distributed.nvls_allreduce import NVLSAllReduce
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)


def _inputs(world_size, rows, hidden, seed):
    gen = torch.Generator(device="cuda")
    xs = []
    for r in range(world_size):
        gen.manual_seed(seed * 1000 + r)
        x = torch.randn(rows, hidden, device="cuda", generator=gen) * (1 + seed % 7)
        xs.append(x.to(torch.bfloat16))
    return xs


def _reference(xs):
    # FP64: an FP32 running sum loses the exact result under cancellation.
    acc = torch.zeros_like(xs[0], dtype=torch.float64)
    for x in xs:
        acc += x.double()
    return acc


def _bits_digest(t):
    v = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    idx = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 1000003 + 1
    return int((v * idx).sum().item())


def _check(out, exact, comm):
    # The switch rounds the FP32 sum to BF16 within one ulp of the exact sum.
    ulp = torch.ldexp(torch.ones_like(exact), torch.frexp(exact.abs().clamp_min(1e-30))[1] - 8)
    assert ((out.double() - exact).abs() <= ulp).all(), "all-reduce off by more than one BF16 ulp"
    digests = comm.allgather(_bits_digest(out))
    assert len(set(digests)) == 1, "ranks disagree on the all-reduce bits"


def _run_rank(world_size, shapes, use_allreduce_module, version):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TRTLLM_NVLS_AR"] = "1"
    os.environ["TRTLLM_NVLS_AR_V2"] = "1" if version == 2 else "0"
    os.environ["TRTLLM_NVLS_AR_MIN_WORLD_SIZE"] = str(world_size)
    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
        if use_allreduce_module:
            ar = AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
            assert ar.nvls_allreduce is not None
            assert ar.nvls_allreduce.version == version
        else:
            # Every size through the requested kernel (the AllReduce path keeps the
            # size threshold, so the pipelined case there mixes both kernels on one
            # workspace).
            ar = NVLSAllReduce(
                mapping,
                slot=f"unittest_v{version}",
                version=version,
                v2_config={"min_bytes": 0},
            )

        # Eager: mixed shapes back to back on one workspace.
        for i, (rows, hidden) in enumerate(shapes):
            xs = _inputs(world_size, rows, hidden, seed=i + 1)
            _check(ar(xs[rank]), _reference(xs), comm)

        # CUDA graph: the same shapes, replayed with new inputs.
        inputs = [
            torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
            for rows, hidden in shapes
        ]
        for x in inputs:
            ar(x)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [ar(x) for x in inputs]
        for rep in range(3):
            refs = []
            for i, x in enumerate(inputs):
                xs = _inputs(world_size, x.shape[0], x.shape[1], seed=100 + 10 * rep + i)
                x.copy_(xs[rank])
                refs.append(_reference(xs))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for out, ref in zip(outs, refs):
                _check(out, ref, comm)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_decode_pattern(world_size, rows, num_layers):
    """K3 low-batch decode pattern on one mapping: per layer an o_proj all-reduce (NVLS) feeding a
    GEMM, then a shared-expert all-reduce (NVLS) on a side stream next to a latent all-reduce +
    RMSNorm (MNNVL); eager on one stream, eager on two streams, and a two-stream CUDA graph."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TRTLLM_NVLS_AR"] = "1"
    os.environ["TRTLLM_NVLS_AR_V2"] = "0"
    os.environ["TRTLLM_NVLS_AR_MIN_WORLD_SIZE"] = str(world_size)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)

        def new_ar():
            return AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)

        o_ars = [new_ar() for _ in range(num_layers)]
        sh_ars = [new_ar() for _ in range(num_layers)]
        lat_ars = [new_ar() for _ in range(num_layers)]
        assert o_ars[0].nvls_allreduce is not None
        hidden, latent = 7168, 3584
        gamma = torch.full((latent,), 0.75, device="cuda", dtype=torch.bfloat16)
        weight = torch.full((hidden, 256), 0.01, device="cuda", dtype=torch.bfloat16)
        aux = torch.cuda.Stream()
        forks = [torch.cuda.Event() for _ in range(num_layers)]
        joins = [torch.cuda.Event() for _ in range(num_layers)]
        x_o = [torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16) for _ in forks]
        x_sh = [torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16) for _ in forks]
        x_lat = [torch.zeros(rows, latent, device="cuda", dtype=torch.bfloat16) for _ in forks]
        rms = AllReduceParams(fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=gamma, eps=1e-6)

        def layers(two_streams):
            outs = []
            for i in range(num_layers):
                o = o_ars[i](x_o[i])
                torch.matmul(o, weight)
                if two_streams:
                    forks[i].record()
                    with torch.cuda.stream(aux):
                        forks[i].wait()
                        sh = sh_ars[i](x_sh[i])
                        joins[i].record()
                else:
                    sh = sh_ars[i](x_sh[i])
                lat_ars[i](x_lat[i], all_reduce_params=rms)
                if two_streams:
                    joins[i].wait()
                outs.append((o, sh))
            return outs

        seed = [0]

        def fill():
            refs = []
            for i in range(num_layers):
                seed[0] += 3
                a = _inputs(world_size, rows, hidden, seed[0])
                b = _inputs(world_size, rows, hidden, seed[0] + 1)
                c = _inputs(world_size, rows, latent, seed[0] + 2)
                x_o[i].copy_(a[rank])
                x_sh[i].copy_(b[rank])
                x_lat[i].copy_(c[rank])
                refs.append((_reference(a), _reference(b)))
            return refs

        def check(outs, refs):
            for (o, sh), (ref_o, ref_sh) in zip(outs, refs):
                _check(o, ref_o, comm)
                _check(sh, ref_sh, comm)

        for two_streams in (False, True):
            for _ in range(2):
                refs = fill()
                with with_multi_stream(two_streams):
                    outs = layers(two_streams)
                torch.cuda.synchronize()
                check(outs, refs)

        torch.matmul(x_o[0], weight)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.graph(graph):
            outs = layers(True)
        for _ in range(3):
            refs = fill()
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            check(outs, refs)
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("rows", [128, 200])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_nvls_allreduce_decode_pattern(rows, mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _run_rank_decode_pattern, *zip(*[(world_size, rows, 4)] * world_size)
    )
    for r in results:
        assert r is True


# (rows, hidden): K3 decode shapes (o_proj / shared expert all-reduce at 7168,
# latent 3584), a ragged row count and a tiny one.
SHAPES = ((224, 7168), (32, 7168), (896, 3584), (37, 3584), (3, 7168), (896, 7168))


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("version", [1, 2], ids=["oneshot", "pipelined"])
@pytest.mark.parametrize("use_allreduce_module", [False, True], ids=["direct", "allreduce"])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_nvls_allreduce(use_allreduce_module, version, mpi_pool_executor):
    if use_allreduce_module:
        # AllReduce routes only messages inside the size window.
        shapes = tuple(s for s in SHAPES if s[0] * s[1] * 2 >= 1 << 20)
    else:
        shapes = SHAPES
    world_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _run_rank, *zip(*[(world_size, shapes, use_allreduce_module, version)] * world_size)
    )
    for r in results:
        assert r is True
