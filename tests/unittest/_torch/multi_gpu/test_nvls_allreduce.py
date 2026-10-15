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


def _run_rank(
    world_size,
    shapes,
    use_allreduce_module,
    version,
    protocol="gpu",
    v2_extra=None,
    staging="copy",
    zero_copy=False,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TRTLLM_NVLS_AR"] = "1"
    os.environ["TRTLLM_NVLS_AR_V2"] = "1" if version == 2 else "0"
    os.environ["TRTLLM_NVLS_AR_MIN_WORLD_SIZE"] = str(world_size)
    os.environ["TRTLLM_NVLS_AR_PROTOCOL"] = protocol
    os.environ["TRTLLM_NVLS_AR_PTX_STAGING"] = staging
    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
        if use_allreduce_module:
            ar = AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
            assert ar.nvls_allreduce is not None
            assert ar.nvls_allreduce.version == version
            assert ar.nvls_allreduce.protocol == protocol
        else:
            # Every size through the requested kernel (the AllReduce path keeps the
            # size threshold, so the pipelined case there mixes both kernels on one
            # workspace).
            ar = NVLSAllReduce(
                mapping,
                slot=f"unittest_v{version}_{protocol}_{staging}_{sorted((v2_extra or {}).items())}",
                version=version,
                v2_config={"min_bytes": 0, **(v2_extra or {})},
            )
            assert ar.protocol == protocol and ar.ptx_staging == staging
        # Every protocol produces the bits of the default one (only the synchronization differs).
        same_as = None
        if protocol != "gpu":
            same_as = NVLSAllReduce(
                mapping,
                slot=f"unittest_v{version}_gpu_ref",
                version=version,
                v2_config={"min_bytes": 0},
                protocol="gpu",
            )

        # Eager: mixed shapes back to back on one workspace.
        for i, (rows, hidden) in enumerate(shapes):
            xs = _inputs(world_size, rows, hidden, seed=i + 1)
            if zero_copy:
                x = ar.input_buffer((rows, hidden))
                x.copy_(xs[rank])
            else:
                x = xs[rank]
            out = ar(x)
            _check(out, _reference(xs), comm)
            if same_as is not None and same_as.can_use(xs[rank]):
                assert torch.equal(out.view(torch.int16), same_as(xs[rank]).view(torch.int16))

        # CUDA graph: the same shapes, replayed with new inputs.
        inputs = [
            torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
            for rows, hidden in shapes
        ]
        for x in inputs:
            ar(x)
        torch.cuda.synchronize()
        comm.Barrier()

        def call(x):
            if zero_copy:
                # A producer writing into the input buffer (here: a copy in the graph).
                buf = ar.input_buffer(tuple(x.shape))
                buf.copy_(x)
                return ar(buf)
            return ar(x)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [call(x) for x in inputs]
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


def _run_rank_decode_pattern(world_size, rows, num_layers, protocol="gpu"):
    """K3 low-batch decode pattern on one mapping: per layer an o_proj all-reduce (NVLS) feeding a
    GEMM, then a shared-expert all-reduce (NVLS) on a side stream next to a latent all-reduce +
    RMSNorm (MNNVL); eager on one stream, eager on two streams, and a two-stream CUDA graph."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TRTLLM_NVLS_AR"] = "1"
    os.environ["TRTLLM_NVLS_AR_V2"] = "0"
    os.environ["TRTLLM_NVLS_AR_MIN_WORLD_SIZE"] = str(world_size)
    os.environ["TRTLLM_NVLS_AR_PROTOCOL"] = protocol
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


class _ZeroCopySpy:
    """Records, for every NVLS all-reduce call, whether its input is the workspace's input
    buffer (a zero-copy call)."""

    def __enter__(self):
        self.calls = []
        self._call = NVLSAllReduce.__call__
        spy = self

        def call(nvls, inp, out=None):
            spy.calls.append(inp.data_ptr() == nvls._ws["in_buf"].data_ptr())
            return spy._call(nvls, inp, out)

        NVLSAllReduce.__call__ = call
        return self

    def __exit__(self, *exc):
        NVLSAllReduce.__call__ = self._call


def _k3_o_proj(n, k, seed):
    """The K3 FP8 weight-read o_proj (``_Fp8BlockScaleWeightReadLinear``) with its mid-M GEMM."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_fp8_mid_gemm import expand_weight_scale
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w_fp8, w_scale = _Fp8BlockScaleWeightReadLinear.quantize_weight(w)
    lin = _Fp8BlockScaleWeightReadLinear(w_fp8, w_scale, n)
    lin.mid_weight_sf = expand_weight_scale(w_scale, n, k)
    return lin


def _row_linear(mapping, k_per_rank, n, seed):
    """A row-parallel BF16 ``Linear`` that all-reduces its output (the K3 shared-expert down
    projection), opted in to zero-copy NVLS input."""
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

    lin = Linear(
        k_per_rank * mapping.tp_size,
        n,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=True,
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lin.weight.data.copy_(
        torch.randn(lin.weight.shape, device="cuda", generator=gen).to(torch.bfloat16) * 0.05
    )
    lin.nvls_zero_copy = True
    return lin


def _assert_same_bits(outs, refs, comm):
    for out, ref in zip(outs, refs):
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
            "zero-copy output differs from the copied-input output"
        )
    digests = comm.allgather([_bits_digest(out) for out in outs])
    assert all(d == digests[0] for d in digests), "ranks disagree on the all-reduce bits"


def _run_rank_zero_copy_producers(world_size, rows, protocol):
    """GEMM producers writing straight into ``AllReduce.nvls_input_buffer``
    (TRTLLM_NVLS_AR_ZERO_COPY=1): the K3 o_proj (FP8 mid-M GEMM on BF16 or prequantized input)
    before a pipelined-kernel all-reduce, and a row-parallel ``Linear`` (cuBLAS, the shared-expert
    down projection) before a one-shot all-reduce, on one workspace. Per layer the o_proj
    all-reduce runs on the main stream and the shared expert on a side stream after it, joined
    before the next layer, as in the K3 decode step. The outputs are bitwise those of the
    copied-input path, eager on one and two streams and in CUDA graphs replayed with new inputs."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TRTLLM_NVLS_AR"] = "1"
    os.environ["TRTLLM_NVLS_AR_V2"] = "1"
    os.environ["TRTLLM_NVLS_AR_MIN_WORLD_SIZE"] = str(world_size)
    os.environ["TRTLLM_NVLS_AR_PROTOCOL"] = protocol
    os.environ["TRTLLM_NVLS_AR_PTX_STAGING"] = "copy"
    os.environ["TRTLLM_NVLS_AR_ZERO_COPY"] = "1"
    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
        hidden, k_o, k_sh, num_layers = 7168, 768, 384, 3
        o_ars = [
            AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
            for _ in range(num_layers)
        ]
        o_projs = [_k3_o_proj(hidden, k_o, seed=100 * rank + i) for i in range(num_layers)]
        sh_downs = [
            _row_linear(mapping, k_sh, hidden, seed=100 * rank + 50 + i) for i in range(num_layers)
        ]
        for lin in sh_downs:
            # The K3 shared expert keeps the one-shot kernel.
            lin.all_reduce.nvls_allreduce.version = 1
        nvls_ars = [ar.nvls_allreduce for ar in o_ars] + [
            lin.all_reduce.nvls_allreduce for lin in sh_downs
        ]
        assert all(nvls.zero_copy and nvls.slot == "main" for nvls in nvls_ars)
        x_o = [torch.zeros(rows, k_o, device="cuda", dtype=torch.bfloat16) for _ in o_ars]
        x_sh = [torch.zeros(rows, k_sh, device="cuda", dtype=torch.bfloat16) for _ in o_ars]
        consumer = torch.full((hidden, 64), 0.01, device="cuda", dtype=torch.bfloat16)
        aux = torch.cuda.Stream()
        forks = [torch.cuda.Event() for _ in o_ars]
        joins = [torch.cuda.Event() for _ in o_ars]

        def set_zero_copy(on):
            for nvls in nvls_ars:
                nvls.zero_copy = on

        def layers(packed, two_streams):
            outs = []
            for i in range(num_layers):
                o_projs[i].next_output = o_ars[i].nvls_input_buffer((rows, hidden))
                if packed:
                    xq, xs = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x_o[i], False)
                    partial = o_projs[i].forward_fp8_packed(xq, xs)
                else:
                    partial = o_projs[i](x_o[i])
                o = o_ars[i](partial)
                consumed = torch.matmul(o, consumer)
                if two_streams:
                    forks[i].record()
                    with torch.cuda.stream(aux):
                        forks[i].wait()
                        sh = sh_downs[i](x_sh[i])
                        joins[i].record()
                    joins[i].wait()
                else:
                    sh = sh_downs[i](x_sh[i])
                outs += [o, consumed, sh]
            return outs

        seed = [rank]

        def fill():
            gen = torch.Generator(device="cuda")
            for i in range(num_layers):
                seed[0] += 7
                gen.manual_seed(seed[0])
                x_o[i].copy_(torch.randn(x_o[i].shape, device="cuda", generator=gen) * 2)
                x_sh[i].copy_(torch.randn(x_sh[i].shape, device="cuda", generator=gen))

        for packed in (False, True):
            for two_streams in (False, True):
                fill()
                with with_multi_stream(two_streams):
                    set_zero_copy(False)
                    with _ZeroCopySpy() as spy:
                        refs = layers(packed, two_streams)
                    assert spy.calls == [False] * (2 * num_layers), spy.calls
                    set_zero_copy(True)
                    with _ZeroCopySpy() as spy:
                        outs = layers(packed, two_streams)
                    assert spy.calls == [True] * (2 * num_layers), spy.calls
                torch.cuda.synchronize()
                _assert_same_bits(outs, refs, comm)

        for packed in (False, True):
            graphs, outs = [], []
            for on in (False, True):
                set_zero_copy(on)
                with with_multi_stream(True):
                    layers(packed, True)
                torch.cuda.synchronize()
                comm.Barrier()
                graph = torch.cuda.CUDAGraph()
                with with_multi_stream(True), torch.cuda.graph(graph):
                    outs.append(layers(packed, True))
                graphs.append(graph)
            for _ in range(3):
                fill()
                torch.cuda.synchronize()
                comm.Barrier()
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                _assert_same_bits(outs[1], outs[0], comm)
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the K3 mid-M FP8 GEMM needs SM100 / SM103",
)
@pytest.mark.parametrize("protocol", ["gpu", "ptx"])
@pytest.mark.parametrize("rows", [256, 896], ids=["b32", "b128"])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_nvls_allreduce_zero_copy_producers(rows, protocol, mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _run_rank_zero_copy_producers, *zip(*[(world_size, rows, protocol)] * world_size)
    )
    for r in results:
        assert r is True


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("protocol", ["gpu", "ptx"])
@pytest.mark.parametrize("rows", [128, 200])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_nvls_allreduce_decode_pattern(rows, protocol, mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _run_rank_decode_pattern, *zip(*[(world_size, rows, 4, protocol)] * world_size)
    )
    for r in results:
        assert r is True


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize(
    "version,protocol,staging,v2_extra,zero_copy",
    [
        (1, "ptx", "copy", None, False),
        (1, "ptx", "copy", None, True),
        (1, "ptx", "inline", None, False),
        (1, "sys", "copy", None, False),
        (2, "ptx", "copy", None, False),
        (2, "ptx", "copy", None, True),
        (2, "ptx", "once", {"stagers": 7}, False),
        (2, "ptx", "inline", None, False),
        (2, "gpu", "copy", None, True),
    ],
    ids=[
        "oneshot-ptx-copy",
        "oneshot-ptx-zerocopy",
        "oneshot-ptx-inline",
        "oneshot-sys",
        "pipelined-ptx-copy",
        "pipelined-ptx-zerocopy",
        "pipelined-ptx-once-7stagers",
        "pipelined-ptx-inline",
        "pipelined-gpu-zerocopy",
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_nvls_allreduce_protocols(
    version, protocol, staging, v2_extra, zero_copy, mpi_pool_executor
):
    """The arrival protocols (TRTLLM_NVLS_AR_PROTOCOL), their staging modes and the
    zero-copy input buffer: exact within one ulp, identical across ranks, bitwise equal
    to the default protocol, CUDA graph replays with new inputs."""
    world_size = mpi_pool_executor.num_workers
    args = (world_size, SHAPES, False, version, protocol, v2_extra, staging, zero_copy)
    results = mpi_pool_executor.map(_run_rank, *zip(*[args] * world_size))
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
