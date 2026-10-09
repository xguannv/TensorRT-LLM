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
"""The fused DSpark Markov chain must be bit-identical to the reference chain."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_chain_logits
from tensorrt_llm._torch.speculative import dspark as dspark_module
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.dspark_markov_fused import (
    FUSED_MARKOV_ENV_VAR,
    fused_markov_chain_logits,
    fused_markov_supported,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

VOCAB = 20000 + 37  # not a multiple of the 2048-wide argmax tiles


def _plain_argmax(step_logits):
    # What DSparkWorker's argmax_fn reduces to for full-vocab logits.
    return torch.argmax(step_logits, dim=-1).type(torch.int32).long()


def _reference(base, anchor, w1, w2):
    return dspark_markov_chain_logits(base, anchor, w1, w2, argmax_fn=_plain_argmax)


def _weights(rank, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randn(VOCAB, rank, device="cuda", generator=gen).to(torch.bfloat16)
    w2 = (torch.randn(VOCAB, rank, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    return w1, w2


def _assert_bit_identical(actual, expected):
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize("rank", [256, 96])
@pytest.mark.parametrize("num_reqs,block_size", [(1, 1), (1, 7), (3, 6), (8, 8), (33, 7)])
def test_matches_reference(num_reqs, block_size, rank):
    w1, w2 = _weights(rank)
    gen = torch.Generator(device="cuda").manual_seed(num_reqs * 100 + block_size)
    base = torch.randn(num_reqs, block_size, VOCAB, device="cuda", generator=gen) * 3.0
    anchor = torch.randint(0, VOCAB, (num_reqs,), device="cuda", generator=gen)
    assert fused_markov_supported(base, w1, w2)
    _assert_bit_identical(
        fused_markov_chain_logits(base, anchor, w1, w2), _reference(base, anchor, w1, w2)
    )


@pytest.mark.parametrize("anchor_dtype", [torch.int64, torch.int32])
def test_out_of_vocab_anchor_gives_zero_bias(anchor_dtype):
    w1, w2 = _weights(256)
    base = torch.randn(5, 7, VOCAB, device="cuda") * 3.0
    anchor = torch.tensor([-1, VOCAB, VOCAB + 5, 0, VOCAB - 1], device="cuda", dtype=anchor_dtype)
    out = fused_markov_chain_logits(base, anchor, w1, w2)
    _assert_bit_identical(out, _reference(base, anchor, w1, w2))
    _assert_bit_identical(out[:3, 0], base[:3, 0])


def test_argmax_ties_nan_and_inf_follow_torch():
    w1, w2 = _weights(256)
    base = torch.zeros(6, 7, VOCAB, device="cuda")
    base[1] = 2.5  # constant row: the bf16 bias alone decides, with many exact ties
    base[2, :, 100] = float("inf")
    base[2, :, 30] = float("inf")
    base[3] = float("-inf")
    base[4, 2, 12345] = float("nan")
    base[4, 2, 7] = float("nan")
    base[5] = (torch.randn(7, VOCAB, device="cuda") * 2).round()
    anchor = torch.tensor([-1, 5, 6, 7, 8, 9], device="cuda")
    _assert_bit_identical(
        fused_markov_chain_logits(base, anchor, w1, w2), _reference(base, anchor, w1, w2)
    )


def test_strided_block_logits():
    w1, w2 = _weights(256)
    full = torch.randn(8, 9, VOCAB, device="cuda") * 3.0
    base = full[::2, 1:8]
    anchor = torch.randint(0, VOCAB, (4,), device="cuda")
    _assert_bit_identical(
        fused_markov_chain_logits(base, anchor, w1, w2), _reference(base, anchor, w1, w2)
    )


def test_cuda_graph_replay():
    w1, w2 = _weights(256)
    base = torch.randn(4, 7, VOCAB, device="cuda") * 3.0
    anchor = torch.randint(0, VOCAB, (4,), device="cuda")
    fused_markov_chain_logits(base, anchor, w1, w2)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_markov_chain_logits(base, anchor, w1, w2)
    for _ in range(2):
        base.copy_(torch.randn_like(base) * 3.0)
        anchor.copy_(torch.randint(0, VOCAB, (4,), device="cuda"))
        graph.replay()
        _assert_bit_identical(out, _reference(base, anchor, w1, w2))


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_env_gate(monkeypatch, enabled):
    w1, w2 = _weights(256)
    base = torch.randn(3, 7, VOCAB, device="cuda") * 3.0
    anchor = torch.randint(0, VOCAB, (3,), device="cuda")
    fused_calls = []

    def _spy(*args):
        fused_calls.append(1)
        return fused_markov_chain_logits(*args)

    monkeypatch.setattr(dspark_module, "fused_markov_chain_logits", _spy)
    if enabled:
        monkeypatch.setenv(FUSED_MARKOV_ENV_VAR, "1")
    else:
        monkeypatch.delenv(FUSED_MARKOV_ENV_VAR, raising=False)
    worker = SimpleNamespace(
        mapping=None,
        _draft_logits_are_sharded=lambda logits, spec_metadata: False,
        greedy_sample_draft_with_tp_gather=lambda logits, spec_metadata: torch.argmax(
            logits, dim=-1
        ).type(torch.int32),
    )
    draft_model = SimpleNamespace(
        markov_w1=w1,
        markov_w2=w2,
        apply_markov_chain_logits=lambda logits, first, argmax_fn, vocab_slice: (
            dspark_markov_chain_logits(logits, first, w1, w2, argmax_fn=argmax_fn)
        ),
    )
    out = DSparkWorker._apply_dspark_markov_bias(worker, draft_model, base, anchor, None)
    assert len(fused_calls) == int(enabled)
    _assert_bit_identical(out, _reference(base, anchor, w1, w2))
