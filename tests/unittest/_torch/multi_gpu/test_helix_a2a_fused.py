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
"""Fused Helix all-to-all + combine: bitwise equal to the FIFO all-to-all + post-process."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

from tensorrt_llm._mnnvl_utils import init_helix_cp_comm
from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)


def _inputs(num_tokens, cp_size, num_heads, head_dim, seed, specials):
    """One rank's partials, softmax statistics and zero-KV mask."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    partial_o = torch.randn(
        num_tokens, cp_size * num_heads * head_dim, device="cuda", generator=gen
    ).to(torch.bfloat16)
    stats = torch.empty(num_tokens, cp_size * num_heads, 2, device="cuda")
    stats[..., 0] = torch.randn(num_tokens, cp_size * num_heads, device="cuda", generator=gen) * 3
    stats[..., 1] = (
        torch.rand(num_tokens, cp_size * num_heads, device="cuda", generator=gen) * 50 + 1
    )
    mask = torch.rand(num_tokens, device="cuda", generator=gen) < 0.25
    if specials and num_tokens > 4:
        # Words equal to the receive-buffer sentinel (0xffffffff), NaN, a far smaller max,
        # and a token without any KV on this rank.
        partial_o.view(torch.int16)[0, :64] = -1
        stats.view(torch.int32)[1, 0, 0] = -1
        stats[2, :, 0] = -1.0e30
        stats[2, 0, 0] = 80.0
        stats[3, :, 0] = float("-inf")
        stats[3, :, 1] = 0.0
    return partial_o, stats, mask


def _reference(helix, partial_o, stats, mask, cp_size, num_heads, head_dim):
    num_tokens = partial_o.shape[0]
    o, s = helix.alltoall_native(
        partial_o.view(num_tokens, cp_size, num_heads * head_dim),
        stats.view(num_tokens, cp_size, num_heads * 2),
        mask,
    )
    # getattr keeps ``torch.ops`` (a module object) out of what cloudpickle ships by value.
    return getattr(torch, "ops").trtllm.helix_post_process_native(
        o.view(num_tokens, cp_size, num_heads, head_dim),
        s.view(num_tokens, cp_size, num_heads, 2),
        1.0,
        1,
    )


def _run_rank(world_size, num_heads, head_dim, token_counts):
    from tensorrt_llm._torch.distributed.helix_a2a_fused import HelixAllToAllFused
    from tensorrt_llm._torch.distributed.ops import HelixAllToAllNative
    from tensorrt_llm._torch.modules import attention

    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            cp_size=world_size,
            cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2},
        )
        init_helix_cp_comm(mapping)
        helix = HelixAllToAllNative.get(mapping)
        fused = HelixAllToAllFused.get(mapping)
        args = (world_size, num_heads, head_dim)
        bad = 0

        # Eager, with and without a zero-KV mask, including sentinel / NaN / -inf inputs.
        for i, num_tokens in enumerate(token_counts):
            partial_o, stats, mask = _inputs(num_tokens, *args, seed=17 * i + rank, specials=True)
            for m in (mask, None):
                ref = _reference(helix, partial_o, stats, m, *args)
                out = fused(partial_o, stats, num_heads, head_dim, m)
                bad += int((ref.view(torch.int16) != out.view(torch.int16)).sum())

        # Through _helix_post_process with the switch on.
        attention.HELIX_A2A_FUSED = True
        partial_o, stats, mask = _inputs(token_counts[-1], *args, seed=5 + rank, specials=False)
        out = attention._helix_post_process(
            partial_o, stats, mapping, num_heads, head_dim, zero_kv_mask=mask
        )
        ref = _reference(helix, partial_o, stats, mask, *args)
        bad += int((ref.view(torch.int16) != out.view(torch.int16)).sum())

        # One CUDA graph with every token count (the grid changes between launches), replayed
        # with new inputs.
        inputs = [_inputs(n, *args, seed=0, specials=False) for n in token_counts]
        for partial_o, stats, mask in inputs:
            fused(partial_o, stats, num_heads, head_dim, mask)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [fused(p, s, num_heads, head_dim, m) for p, s, m in inputs]
        for rep in range(3):
            for i, (partial_o, stats, mask) in enumerate(inputs):
                new = _inputs(
                    partial_o.shape[0], *args, seed=1000 * rep + 7 * i + rank, specials=rep == 0
                )
                for dst, src in zip((partial_o, stats, mask), new):
                    dst.copy_(src)
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for (partial_o, stats, mask), out in zip(inputs, outs):
                ref = _reference(helix, partial_o, stats, mask, *args)
                bad += int((ref.view(torch.int16) != out.view(torch.int16)).sum())
        del graph
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the FIFO path"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize(
    "num_heads,head_dim", [(6, 128), (24, 128), (4, 256)], ids=["k3_cp16", "k3_cp4", "d256"]
)
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_helix_a2a_fused(num_heads, head_dim, mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    token_counts = (1, 8, 37, 224, 896)
    results = mpi_pool_executor.map(
        _run_rank, *zip(*[(world_size, num_heads, head_dim, token_counts)] * world_size)
    )
    for r in results:
        assert r is True
