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
"""The fused spec-dec tail kernels must be bit-identical to the eager chains."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import interface as interface_module
from tensorrt_llm._torch.speculative.interface import FORCE_ACCEPT_KEY_ENV_VAR, SpecWorkerBase
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    SPEC_TAIL_FUSION_ENV_VAR,
    decode_draft_block_table,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class _StubSpecWorker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def _metadata(is_cuda_graph=False):
    return SimpleNamespace(is_all_greedy_sample=True, is_cuda_graph=is_cuda_graph)


def _accept(worker, logits, draft, num_contexts, batch_size, fused, monkeypatch):
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1" if fused else "0")
    return worker._sample_and_accept_draft_tokens_base(
        logits, draft, num_contexts, batch_size, _metadata()
    )


def _case(num_contexts, num_gens, draft_len, vocab=50, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = num_contexts + num_gens * (draft_len + 1)
    # Small vocab with integer logits: frequent argmax ties.
    logits = torch.randint(0, 6, (rows, vocab), device="cuda", generator=gen).float()
    target = torch.argmax(logits, dim=-1)
    gen_target = target[num_contexts:].view(num_gens, draft_len + 1)[:, :draft_len]
    # Drafts that match the target for a random prefix length.
    draft = torch.randint(0, vocab, (num_gens, draft_len), device="cuda", generator=gen)
    prefix = torch.randint(0, draft_len + 1, (num_gens,), device="cuda", generator=gen)
    keep = torch.arange(draft_len, device="cuda")[None, :] < prefix[:, None]
    draft = torch.where(keep, gen_target, draft).to(torch.int32)
    return logits, draft


def _assert_same(fused, ref, num_contexts):
    acc_f, n_f = fused
    acc_r, n_r = ref
    assert torch.equal(n_f, n_r)
    # Context rows only define column 0.
    assert torch.equal(acc_f[:num_contexts, 0], acc_r[:num_contexts, 0])
    assert torch.equal(acc_f[num_contexts:], acc_r[num_contexts:])


@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 4), (0, 32), (0, 128), (3, 5)])
@pytest.mark.parametrize("draft_len", [1, 6, 7])
def test_strict_accept_matches_eager(num_contexts, num_gens, draft_len, monkeypatch):
    worker = _StubSpecWorker()
    logits, draft = _case(num_contexts, num_gens, draft_len, seed=num_gens * 10 + draft_len)
    bs = num_contexts + num_gens
    ref = _accept(worker, logits, draft, num_contexts, bs, False, monkeypatch)
    fused = _accept(worker, logits, draft, num_contexts, bs, True, monkeypatch)
    _assert_same(fused, ref, num_contexts)


@pytest.mark.parametrize("force", [1.0, 3.0, 9.0, 2.75, 2.6, 0.5, 6.4])
@pytest.mark.parametrize("num_contexts,num_gens,draft_len", [(0, 8, 6), (2, 33, 7)])
def test_strict_accept_forced_matches_eager(num_contexts, num_gens, draft_len, force, monkeypatch):
    logits, draft = _case(num_contexts, num_gens, draft_len)
    bs = num_contexts + num_gens
    ref_worker, fused_worker = _StubSpecWorker(), _StubSpecWorker()
    ref_worker.force_num_accepted_tokens = force
    fused_worker.force_num_accepted_tokens = force
    # Several iterations: the fractional path advances a device counter.
    for _ in range(5):
        ref = _accept(ref_worker, logits, draft, num_contexts, bs, False, monkeypatch)
        fused = _accept(fused_worker, logits, draft, num_contexts, bs, True, monkeypatch)
        _assert_same(fused, ref, num_contexts)
    if ref_worker._force_accept_rng_counter is not None:
        assert torch.equal(
            ref_worker._force_accept_rng_counter, fused_worker._force_accept_rng_counter
        )


def test_strict_accept_forced_skipped_in_warmup(monkeypatch):
    worker = _StubSpecWorker()
    worker.force_num_accepted_tokens = 2.75
    logits, draft = _case(0, 4, 6)
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1")
    _, n = worker._sample_and_accept_draft_tokens_base(
        logits, draft, 0, 4, _metadata(is_cuda_graph=True)
    )
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "0")
    _, n_ref = worker._sample_and_accept_draft_tokens_base(
        logits, draft, 0, 4, _metadata(is_cuda_graph=True)
    )
    assert torch.equal(n, n_ref)
    assert worker._force_accept_rng_counter is None


def test_strict_accept_cuda_graph_replay_matches_eager(monkeypatch):
    num_gens, draft_len = 16, 6
    logits, draft = _case(0, num_gens, draft_len)
    ref_worker, fused_worker = _StubSpecWorker(), _StubSpecWorker()
    ref_worker.force_num_accepted_tokens = 2.75
    fused_worker.force_num_accepted_tokens = 2.75
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1")
    # Allocate the RNG pool eagerly (as the warmup-then-capture flow does).
    fused_worker._ensure_force_accept_rng_state(torch.device("cuda"))
    ref_worker._ensure_force_accept_rng_state(torch.device("cuda"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fused_worker._sample_and_accept_draft_tokens_base(logits, draft, 0, num_gens, _metadata())
    torch.cuda.synchronize()
    fused_worker._force_accept_rng_counter.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        acc, n = fused_worker._sample_and_accept_draft_tokens_base(
            logits, draft, 0, num_gens, _metadata()
        )
    for _ in range(4):
        graph.replay()
        ref = _accept(ref_worker, logits, draft, 0, num_gens, False, monkeypatch)
        monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1")
        torch.cuda.synchronize()
        _assert_same((acc, n), ref, 0)


def _state_key_inputs(num_contexts, num_gens, draft_len, seed, ties=0):
    """Step inputs (input_ids, position_ids [1, N], num_ctx_tokens) for the state key: five
    tokens per context request, then one verify group of draft_len + 1 tokens per generation
    request starting at a random position with a random newest committed token; the first
    ``ties`` requests after request 0 copy its position and newest token."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_ctx_tokens = 5 * num_contexts
    total = num_ctx_tokens + num_gens * (draft_len + 1)
    ids = torch.randint(0, 163840, (total,), device="cuda", generator=gen).to(torch.int32)
    starts = torch.randint(0, 1 << 20, (num_gens,), device="cuda", generator=gen)
    for g in range(1, min(ties + 1, num_gens)):
        starts[g] = starts[0]
        ids[num_ctx_tokens + g * (draft_len + 1)] = ids[num_ctx_tokens]
    offsets = torch.arange(draft_len + 1, device="cuda")
    pos = torch.cat(
        [
            torch.arange(num_ctx_tokens, device="cuda"),
            (starts[:, None] + offsets[None, :]).flatten(),
        ]
    ).to(torch.int32)
    return ids, pos.unsqueeze(0), num_ctx_tokens


@pytest.mark.parametrize(
    "num_contexts,num_gens,draft_len", [(0, 2, 6), (0, 8, 6), (2, 33, 7), (0, 128, 6)]
)
@pytest.mark.parametrize("ties", [0, 2])
def test_strict_accept_forced_state_key_matches_eager(
    num_contexts, num_gens, draft_len, ties, monkeypatch
):
    monkeypatch.setenv(FORCE_ACCEPT_KEY_ENV_VAR, "state")
    logits, draft = _case(num_contexts, num_gens, draft_len)
    bs = num_contexts + num_gens
    ref_worker, fused_worker = _StubSpecWorker(), _StubSpecWorker()
    ref_worker.force_num_accepted_tokens = 2.75
    fused_worker.force_num_accepted_tokens = 2.75
    seen = set()
    for step in range(6):
        inputs = _state_key_inputs(num_contexts, num_gens, draft_len, seed=step, ties=ties)
        ref_worker._force_accept_inputs = inputs
        fused_worker._force_accept_inputs = inputs
        ref = _accept(ref_worker, logits, draft, num_contexts, bs, False, monkeypatch)
        fused = _accept(fused_worker, logits, draft, num_contexts, bs, True, monkeypatch)
        _assert_same(fused, ref, num_contexts)
        seen.update(ref[1][num_contexts:].tolist())
    assert seen == {3, 4}
    # The state key leaves the counter alone.
    assert int(fused_worker._force_accept_rng_counter.item()) == 0


def test_strict_accept_state_key_cuda_graph_replay_matches_eager(monkeypatch):
    monkeypatch.setenv(FORCE_ACCEPT_KEY_ENV_VAR, "state")
    num_gens, draft_len = 2, 6
    logits, draft = _case(0, num_gens, draft_len)
    ref_worker, fused_worker = _StubSpecWorker(), _StubSpecWorker()
    ref_worker.force_num_accepted_tokens = 2.75
    fused_worker.force_num_accepted_tokens = 2.75
    ids, pos, _ = _state_key_inputs(0, num_gens, draft_len, seed=0)
    fused_worker._force_accept_inputs = (ids, pos, 0)
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1")
    fused_worker._ensure_force_accept_rng_state(torch.device("cuda"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fused_worker._sample_and_accept_draft_tokens_base(logits, draft, 0, num_gens, _metadata())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        acc, n = fused_worker._sample_and_accept_draft_tokens_base(
            logits, draft, 0, num_gens, _metadata()
        )
    for step in range(1, 7):
        new_ids, new_pos, _ = _state_key_inputs(0, num_gens, draft_len, seed=step)
        ids.copy_(new_ids)
        pos.copy_(new_pos)
        graph.replay()
        ref_worker._force_accept_inputs = (new_ids, new_pos, 0)
        ref = _accept(ref_worker, logits, draft, 0, num_gens, False, monkeypatch)
        torch.cuda.synchronize()
        _assert_same((acc, n), ref, 0)


def _reference_block_table(src, divisor):
    encoded = src.to(torch.int64)
    counts = (encoded >= 0).sum(dim=1)
    decoded = encoded.clamp(min=0).div(divisor, rounding_mode="floor")
    return decoded.to(torch.int32), counts


@pytest.mark.parametrize("rows,blocks", [(1, 7), (4, 15589), (32, 2048), (128, 5000)])
@pytest.mark.parametrize("divisor", [1, 2, 10])
def test_decode_draft_block_table_matches_eager(rows, blocks, divisor):
    gen = torch.Generator(device="cuda").manual_seed(rows + blocks)
    # [pools, max_batch, K/V, blocks] like draft_kv_cache_block_offsets.
    full = torch.randint(0, 1 << 20, (2, rows + 3, 2, blocks), device="cuda", generator=gen)
    full = full.to(torch.int32)
    allocated = torch.randint(0, blocks + 1, (rows + 3,), device="cuda", generator=gen)
    tail = torch.arange(blocks, device="cuda")[None, :] >= allocated[:, None]
    full[1, :, 0] = torch.where(tail, torch.full_like(full[1, :, 0], -1), full[1, :, 0])
    src = full[1, :rows, 0]
    table = torch.full((rows + 2, blocks), 7, dtype=torch.int32, device="cuda")
    counts = torch.full((rows + 2,), 9, dtype=torch.long, device="cuda")
    decode_draft_block_table(src, divisor, table[:rows], counts[:rows])
    ref_table, ref_counts = _reference_block_table(src, divisor)
    assert torch.equal(table[:rows], ref_table)
    assert torch.equal(counts[:rows], ref_counts)
    # Rows past the batch are untouched.
    assert bool((table[rows:] == 7).all()) and bool((counts[rows:] == 9).all())


def test_interface_imports_fused_kernels():
    assert interface_module.fused_strict_accept is not None


@pytest.mark.parametrize("rows,width", [(1, 576), (192, 576), (768, 576), (5, 100)])
def test_scatter_rows_to_pool_matches_eager(rows, width):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool

    gen = torch.Generator(device="cuda").manual_seed(rows)
    full = torch.randn(64, 1, 1, 32, width, device="cuda", generator=gen).to(torch.bfloat16)
    pool = full[:, 0, 0]
    flat = torch.randperm(64 * 32, device="cuda", generator=gen)[:rows]
    pages, offsets = flat // 32, flat % 32
    src = torch.randn(rows, width, device="cuda", generator=gen).to(torch.bfloat16)
    expected = full.clone()
    expected[:, 0, 0][pages, offsets] = src
    scatter_rows_to_pool(src, pool, pages, offsets)
    assert torch.equal(full, expected)


@pytest.mark.parametrize("batch,draft_len", [(1, 6), (32, 6), (128, 7), (300, 6)])
def test_fused_next_new_tokens_matches_eager(batch, draft_len):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import fused_next_new_tokens

    gen = torch.Generator(device="cuda").manual_seed(batch)
    accepted = torch.randint(0, 1000, (batch, draft_len + 1), device="cuda", generator=gen).int()
    nacc = torch.randint(1, draft_len + 2, (batch,), device="cuda", generator=gen).int()
    draft = torch.randint(0, 1000, (batch, draft_len), device="cuda", generator=gen).int()
    worker = _StubSpecWorker()
    expected = worker._prepare_next_new_tokens(
        accepted, draft, torch.arange(batch, device="cuda"), batch, nacc
    )
    out = fused_next_new_tokens(accepted, draft, nacc)
    assert out.dtype == expected.dtype and torch.equal(out, expected)
