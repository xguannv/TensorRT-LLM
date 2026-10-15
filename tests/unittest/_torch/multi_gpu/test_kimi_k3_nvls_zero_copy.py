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
"""Kimi-K3 zero-copy NVLS all-reduce input (``TRTLLM_NVLS_AR_ZERO_COPY=1``) in the K3 modules.

The KDA mixer and the MLA runtime hand the NVLS all-reduce's input buffer to their o_proj (the
FP8 weight-read projection; the MLA one behind the real FP8 output-gate path), and the shared
expert (a real ``GatedMLP``) lets its row-parallel down projection write there. The decode steps
are stand-ins (KDA verify: a fixed activation quantized for o_proj as the FP8-output verify
kernel does; MLA attention: a fixed output), so the test checks the plumbing: every all-reduce
takes the zero-copy input, and the results are bitwise those of the copied-input path, eager and
in a CUDA graph with the shared expert on a side stream.
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
from torch import nn
from utils.util import skip_pre_hopper

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
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

HIDDEN, O_IN, SHARED_INTER = 7168, 768, 384


def _randn(shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)


def _o_proj(seed):
    """K3 attention o_proj after weight loading: FP8 weight read with the mid-M GEMM."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_fp8_mid_gemm import expand_weight_scale
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    w_fp8, w_scale = _Fp8BlockScaleWeightReadLinear.quantize_weight(
        _randn((HIDDEN, O_IN), seed, 0.02)
    )
    lin = _Fp8BlockScaleWeightReadLinear(w_fp8, w_scale, HIDDEN)
    lin.mid_weight_sf = expand_weight_scale(w_scale, HIDDEN, O_IN)
    return lin


def _kda(mapping, seed, rows):
    """``KimiKDALinearAttention`` with its real forward / o_proj / all-reduce; the MTP verify
    step is a stand-in that hands o_proj a fixed activation as the FP8-output verify kernel does
    (E4M3 + packed 1x128 scales, ``forward_fp8_packed``)."""
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    kda = KimiKDALinearAttention.__new__(KimiKDALinearAttention)
    nn.Module.__init__(kda)
    kda.layer_idx = 0
    kda.hidden_size = HIDDEN
    kda._use_indexed_ssm_pool = False
    kda._o_allreduce = AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
    kda.o_proj = _o_proj(seed)
    kda.act = _randn((rows, O_IN), seed + 1)

    def forward_verify(x, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices):
        assert x.shape[0] == rows
        xq, xs = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(kda.act, False)
        return kda.o_proj.forward_fp8_packed(xq, xs)

    kda.forward_verify = forward_verify
    return kda


def _kda_metadata(batch):
    return SimpleNamespace(
        num_contexts=0,
        num_ctx_tokens=0,
        seq_lens=torch.ones(batch, dtype=torch.int32),
        mamba_metadata=SimpleNamespace(
            state_indices=torch.zeros(batch, dtype=torch.int32, device="cuda"),
            query_start_loc_long=torch.zeros(1, dtype=torch.int64, device="cuda"),
        ),
        kv_cache_manager=SimpleNamespace(
            mamba_layer_cache=lambda idx: SimpleNamespace(conv=None, temporal=None)
        ),
    )


class _MlaMixer(nn.Module):
    """``KimiK3MLAAttention`` after attention: its real output gate + o_proj on a fixed attention
    output."""

    def __init__(self, seed, rows):
        super().__init__()
        self.use_output_gate = True
        self.g_proj = nn.Linear(HIDDEN, O_IN, bias=False, dtype=torch.bfloat16, device="cuda")
        self.g_proj.weight.data.copy_(_randn((O_IN, HIDDEN), seed + 2, 0.02))
        self.o_proj = _o_proj(seed)
        self.attn_out = _randn((rows, O_IN), seed + 3)

    def forward(self, hidden_states, attn_metadata):
        from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention

        return KimiK3MLAAttention._apply_output_gate_and_o_proj(self, hidden_states, self.attn_out)


def _mla(mapping, seed, rows):
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiMLARuntime

    mla = KimiMLARuntime.__new__(KimiMLARuntime)
    nn.Module.__init__(mla)
    mla._o_allreduce = AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
    mla.mixer = _MlaMixer(seed, rows)
    return mla


def _shared_expert(mapping, seed):
    """The K3 shared expert: a TP ``GatedMLP`` whose down projection reduces its output, opted in
    to zero-copy NVLS input as ``KimiK3MoERuntime`` does."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    mlp = GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=SHARED_INTER * mapping.tp_size,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(mapping=mapping),
        reduce_output=True,
        is_shared_expert=True,
    ).cuda()
    mlp.gate_up_proj.weight.data.copy_(_randn(mlp.gate_up_proj.weight.shape, seed + 4, 0.02))
    mlp.down_proj.weight.data.copy_(_randn(mlp.down_proj.weight.shape, seed + 5, 0.05))
    mlp.down_proj.nvls_zero_copy = True
    mlp.down_proj.all_reduce.nvls_allreduce.version = 1
    return mlp


def _bits_digest(t):
    v = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    idx = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 1000003 + 1
    return int((v * idx).sum().item())


def _run_rank(world_size, batch, steps, protocol, gate_fp8):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TRTLLM_NVLS_AR"] = "1"
    os.environ["TRTLLM_NVLS_AR_V2"] = "1"
    os.environ["TRTLLM_NVLS_AR_MIN_WORLD_SIZE"] = str(world_size)
    os.environ["TRTLLM_NVLS_AR_PROTOCOL"] = protocol
    os.environ["TRTLLM_NVLS_AR_PTX_STAGING"] = "copy"
    os.environ["TRTLLM_NVLS_AR_ZERO_COPY"] = "1"
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention

    saved_gate_fp8 = kimi_k3_mla_attention._K3_MLA_GATE_FP8
    kimi_k3_mla_attention._K3_MLA_GATE_FP8 = gate_fp8
    calls = []
    nvls_call = NVLSAllReduce.__call__

    def spy(nvls, inp, out=None):
        calls.append(inp.data_ptr() == nvls._ws["in_buf"].data_ptr())
        return nvls_call(nvls, inp, out)

    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
        rows = batch * steps
        kda = _kda(mapping, 1000 * rank, rows)
        mla = _mla(mapping, 1000 * rank + 100, rows)
        shared = _shared_expert(mapping, 1000 * rank + 200)
        nvls_ars = [
            kda._o_allreduce.nvls_allreduce,
            mla._o_allreduce.nvls_allreduce,
            shared.down_proj.all_reduce.nvls_allreduce,
        ]
        assert all(nvls is not None and nvls.zero_copy for nvls in nvls_ars)
        metadata = _kda_metadata(batch)
        x = _randn((rows, HIDDEN), 1000 * rank + 300)
        aux = torch.cuda.Stream()
        fork, join = torch.cuda.Event(), torch.cuda.Event()

        def step(two_streams):
            """KDA layer, shared expert (side stream), MLA layer, as in the K3 decode step."""
            a = kda(x, metadata)
            if two_streams:
                fork.record()
                with torch.cuda.stream(aux):
                    fork.wait()
                    s = shared(a)
                    join.record()
                join.wait()
            else:
                s = shared(a)
            m = mla(s, None)
            return [a, s, m]

        def run(zero_copy, two_streams):
            for nvls in nvls_ars:
                nvls.zero_copy = zero_copy
            calls.clear()
            NVLSAllReduce.__call__ = spy
            try:
                with with_multi_stream(two_streams):
                    outs = step(two_streams)
            finally:
                NVLSAllReduce.__call__ = nvls_call
            assert calls == [zero_copy] * 3, calls
            assert kda.o_proj.next_output is None and mla.mixer.o_proj.next_output is None
            return outs

        def same_bits(outs, refs):
            for out, ref in zip(outs, refs):
                assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
            digests = comm.allgather([_bits_digest(out) for out in outs])
            assert all(d == digests[0] for d in digests), "ranks disagree"

        for two_streams in (False, True):
            refs = run(False, two_streams)
            outs = run(True, two_streams)
            torch.cuda.synchronize()
            same_bits(outs, refs)

        graphs, graph_outs = [], []
        for zero_copy in (False, True):
            run(zero_copy, True)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with with_multi_stream(True), torch.cuda.graph(graph):
                graph_outs.append(step(True))
            graphs.append(graph)
        for rep in range(3):
            x.copy_(_randn((rows, HIDDEN), 1000 * rank + 400 + rep))
            kda.act.copy_(_randn((rows, O_IN), 1000 * rank + 500 + rep))
            torch.cuda.synchronize()
            comm.Barrier()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            same_bits(graph_outs[1], graph_outs[0])
    except Exception:
        traceback.print_exc()
        raise
    finally:
        kimi_k3_mla_attention._K3_MLA_GATE_FP8 = saved_gate_fp8
    return True


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the K3 mid-M FP8 GEMM needs SM100 / SM103",
)
@pytest.mark.parametrize("gate_fp8", [True, False], ids=["gate_fp8", "gate_bf16"])
@pytest.mark.parametrize("protocol", ["ptx", "gpu"])
@pytest.mark.parametrize("batch,steps", [(32, 8), (128, 7)], ids=["b32", "b128"])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_kimi_k3_nvls_zero_copy(batch, steps, protocol, gate_fp8, mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    args = (world_size, batch, steps, protocol, gate_fp8)
    results = mpi_pool_executor.map(_run_rank, *zip(*[args] * world_size))
    for r in results:
        assert r is True
