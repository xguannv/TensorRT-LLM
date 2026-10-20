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
"""CuTe DSpark Markov chain (TLLM_DSPARK_MARKOV_V2=3) against the Triton one-launch chain (=2):
the same draft tokens over random and adversarial inputs, eager and in a replayed CUDA graph, with
the bias MMA's fragments by ldmatrix (default) and by 4-byte loads (TLLM_DSPARK_MARKOV_LDSM=0)."""

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

SHARD = 10240
MARKOV_RANK = 256


def _case(b, k, kind, tp, w1, w2, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    vocab = SHARD * tp
    local = (torch.randn(b, k, SHARD, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    anchors = torch.randint(0, vocab, (b,), device="cuda", generator=gen)
    if kind == 1:  # ties everywhere: integer logits and a zero bias
        local = torch.randint(-3, 3, (b, k, SHARD), device="cuda", generator=gen).to(torch.bfloat16)
        w2 = torch.zeros_like(w2)
    elif kind == 2:  # NaN lanes
        local[torch.rand(b, k, SHARD, device="cuda", generator=gen) < 1e-4] = float("nan")
    elif kind == 3:  # out-of-vocab anchors, int32 anchors
        anchors = torch.tensor([-1, vocab + 5] * b, device="cuda")[:b].to(torch.int32)
    elif kind == 4:  # large logits and bias
        local = (local.float() * 64).to(torch.bfloat16)
        w1 = (w1.float() * 8).to(torch.bfloat16)
    return local, anchors, w1, w2


def _run_rank(world_size, ldsm):
    os.environ["TLLM_DSPARK_MARKOV_LDSM"] = "1" if ldsm else "0"
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    os.environ["TLLM_SPEC_VOCAB_PARALLEL"] = "1"
    os.environ["TLLM_SPEC_VOCAB_PARALLEL_COMM"] = "mnnvl"
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.speculative.vocab_parallel import VocabParallelHead
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        vocab = SHARD * world_size
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        lm_head = LMHead(
            vocab,
            256,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        ).cuda()
        head = VocabParallelHead(lm_head, vocab)
        gen = torch.Generator(device="cuda").manual_seed(3)
        w1 = torch.randn(vocab, MARKOV_RANK, device="cuda", generator=gen).to(torch.bfloat16)
        w2 = (torch.randn(vocab, MARKOV_RANK, device="cuda", generator=gen) * 0.1).to(
            torch.bfloat16
        )

        def chain(local, anchors, a, b, mode):
            os.environ["TLLM_DSPARK_MARKOV_V2"] = mode
            return head.markov_tokens_chain(local, anchors, a, b)

        bad = 0
        for b in (1, 2, 3, 4, 8):
            for k in (6, 7):
                for kind in range(5):
                    local, anchors, a, w = _case(
                        b, k, kind, world_size, w1, w2, 97 * b + 7 * k + kind
                    )
                    ref = chain(local, anchors, a, w, "2").clone()
                    got = chain(local, anchors, a, w, "3").clone()
                    bad += int((ref != got).sum())

        # A captured CuTe chain replayed with new inputs: the exchange sets rotate across replays.
        b, k = 2, 6
        inputs = [_case(b, k, 0, world_size, w1, w2, 500 + i) for i in range(3)]
        os.environ["TLLM_DSPARK_MARKOV_V2"] = "3"
        for local, anchors, a, w in inputs:
            head.markov_tokens_chain(local, anchors, a, w)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [head.markov_tokens_chain(*x) for x in inputs]
        for rep in range(4):
            for i, (local, anchors, _, _) in enumerate(inputs):
                new_local, new_anchors, _, _ = _case(
                    b, k, 0, world_size, w1, w2, 900 + 10 * rep + i
                )
                local.copy_(new_local)
                anchors.copy_(new_anchors)
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for x, out in zip(inputs, outs):
                bad += int((chain(*x, "2") != out).sum())
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} tokens differ from the Triton chain"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("ldsm", [True, False])
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_dspark_markov_chain_cute(mpi_pool_executor, ldsm):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size, ldsm)] * world_size))
    for r in results:
        assert r is True
