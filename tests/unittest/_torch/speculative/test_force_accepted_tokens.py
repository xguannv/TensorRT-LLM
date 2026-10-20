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
"""Unit tests for the synthetic forced acceptance rate in 1-model spec dec.

These tests exercise ``SpecWorkerBase._apply_force_accepted_tokens``, which
honors the ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` environment
variable. The implementation must be:

  - Faithful to fractional rates: integer part always accepts, fractional
    part acts as the per-iteration probability of accepting one extra draft
    token.
  - Tensor-parallel deterministic: identical seed/counter state across
    ranks so all ranks agree on per-request accepted counts (otherwise
    downstream collectives expecting matching shapes deadlock).
  - CUDA-graph compatible: pure-tensor ops with statically-known shapes,
    captureable and replayable.
"""

import os
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.interface import (
    FORCE_ACCEPT_KEY_ENV_VAR,
    FORCE_ACCEPT_KEY_STATE,
    FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR,
    SpecWorkerBase,
    get_force_accept_key,
    get_force_num_accepted_tokens,
    get_force_num_accepted_tokens_float,
)


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the abstract API.

    Used purely to drive ``_apply_force_accepted_tokens`` in isolation.
    """

    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _make_worker(value: Optional[float] = None) -> _StubSpecWorker:
    worker = _StubSpecWorker()
    if value is not None:
        worker.force_num_accepted_tokens = value
    return worker


def _make_input(batch_size: int, device: str = "cuda") -> torch.Tensor:
    # Initial counts: 1 (target token, no draft accepts yet). The forced
    # override only ever rewrites entries from ``num_contexts:`` onward.
    return torch.ones(batch_size, dtype=torch.int32, device=device)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this test.")


# ---------------- env-var parsing helpers -----------------------------------


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("0", 0.0),
        ("3", 3.0),
        ("2.6", 2.6),
        ("0.5", 0.5),
        ("not-a-number", 0.0),
    ],
)
def test_get_force_num_accepted_tokens_float(env_value, expected):
    with patch.dict(os.environ, {FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR: env_value}):
        assert get_force_num_accepted_tokens_float() == pytest.approx(expected)


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("0", 0),
        ("3", 3),
        ("2.6", 0),
    ],
)
def test_get_force_num_accepted_tokens_int_unchanged(env_value, expected):
    """The int helper must keep its original behavior (used by 2-model)."""
    with patch.dict(os.environ, {FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR: env_value}):
        assert get_force_num_accepted_tokens() == expected


# ---------------- _apply_force_accepted_tokens semantics --------------------


def test_zero_value_is_noop():
    _require_cuda()
    worker = _make_worker(0.0)
    base = _make_input(batch_size=4)
    before = base.clone()
    out = worker._apply_force_accepted_tokens(base, num_contexts=0, runtime_draft_len=4)
    assert torch.equal(out, before)
    # No RNG state should have been touched in the early-exit path.
    assert worker._force_accept_rng_pool is None
    assert worker._force_accept_rng_counter is None


@pytest.mark.parametrize("value", [1.0, 2.0, 4.0])
def test_integer_value_matches_legacy_behavior(value):
    _require_cuda()
    worker = _make_worker(value)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=8), num_contexts=0, runtime_draft_len=4
    )
    expected = min(int(value) + 1, 4 + 1)
    assert torch.all(out == expected)
    # Pure-integer path must not allocate the RNG pool.
    assert worker._force_accept_rng_pool is None


def test_integer_value_caps_at_runtime_draft_len_plus_one():
    _require_cuda()
    worker = _make_worker(10.0)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=4), num_contexts=0, runtime_draft_len=2
    )
    # 10 draft tokens requested but only 2 available → all 3 (= 2 + target).
    assert torch.all(out == 3)


def test_fractional_only_emits_two_values():
    _require_cuda()
    worker = _make_worker(2.6)
    seen = set()
    for _ in range(50):
        out = worker._apply_force_accepted_tokens(
            _make_input(batch_size=64), num_contexts=0, runtime_draft_len=4
        )
        seen.update(out.unique().tolist())
    # Either 2 draft + target = 3, or 3 draft + target = 4.
    assert seen == {3, 4}


def test_fractional_distribution_matches_target_probability():
    """Average over many draws should converge to the target fraction."""
    _require_cuda()
    worker = _make_worker(2.6)
    target_frac = 0.6
    n_iters = 200
    batch = 64
    extra_count = 0
    total = 0
    for _ in range(n_iters):
        out = worker._apply_force_accepted_tokens(
            _make_input(batch_size=batch), num_contexts=0, runtime_draft_len=4
        )
        extra_count += int((out == 4).sum().item())
        total += batch
    measured = extra_count / total
    # Pool-based RNG is deterministic; the empirical mean over ~13k draws
    # is tight against 0.6.
    assert abs(measured - target_frac) < 0.03, (
        f"measured fraction {measured:.4f} differs from target {target_frac}"
    )


def test_fractional_capped_when_no_room():
    """If ``int_part + 1`` already saturates the cap, no extra is granted."""
    _require_cuda()
    worker = _make_worker(9.5)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=8), num_contexts=0, runtime_draft_len=2
    )
    # max_total = runtime_draft_len + 1 = 3, base_total = min(10, 3) = 3.
    # frac_part > 0 but ``base_total < max_total`` is False → all 3, no RNG.
    assert torch.all(out == 3)
    assert worker._force_accept_rng_pool is None


def test_num_contexts_offset_is_respected():
    """Context (prefill) rows must be left untouched by the override."""
    _require_cuda()
    worker = _make_worker(2.6)
    batch_size = 8
    num_contexts = 3
    base = _make_input(batch_size=batch_size)
    sentinel = torch.iinfo(base.dtype).max
    base[:num_contexts] = sentinel
    out = worker._apply_force_accepted_tokens(base, num_contexts=num_contexts, runtime_draft_len=4)
    assert torch.all(out[:num_contexts] == sentinel)
    assert torch.all((out[num_contexts:] == 3) | (out[num_contexts:] == 4))


# ---------------- TP determinism --------------------------------------------


def test_tp_determinism_across_independent_workers():
    """
    Two independently-instantiated workers (simulating two TP ranks) must
    produce bit-identical accepted-token counts when fed the same call
    sequence. This is the property whose absence used to hang TP>1.
    """
    _require_cuda()
    rank0 = _make_worker(2.6)
    rank1 = _make_worker(2.6)
    # Use a non-power-of-two batch to make accidental shape coincidences
    # less likely to mask divergence.
    for _ in range(32):
        out0 = rank0._apply_force_accepted_tokens(
            _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
        )
        out1 = rank1._apply_force_accepted_tokens(
            _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
        )
        assert torch.equal(out0, out1)


def test_tp_determinism_survives_default_generator_drift():
    """
    Even if the default CUDA generator state diverges between ranks (e.g.
    because some other component consumed random numbers), the synthetic
    AR draws must still match. This is the regression guard for the
    original bug, where ``torch.rand`` against the default generator gave
    each rank a different sequence.
    """
    _require_cuda()
    rank0 = _make_worker(2.6)
    rank1 = _make_worker(2.6)

    def _advance_default_generator(seed: int):
        # Seed the default CUDA generator on the same device differently
        # for each "rank" and consume some randoms, simulating drift.
        torch.cuda.manual_seed(seed)
        _ = torch.rand(128, device="cuda")

    _advance_default_generator(seed=11)
    out0 = rank0._apply_force_accepted_tokens(
        _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
    )
    _advance_default_generator(seed=999)
    out1 = rank1._apply_force_accepted_tokens(
        _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
    )
    assert torch.equal(out0, out1)


# ---------------- CUDA graph capture/replay ---------------------------------


def test_cuda_graph_capture_and_replay_match_eager():
    """
    Capturing the override into a CUDA graph and replaying it must produce
    the exact same per-replay output sequence as running it eagerly.
    """
    _require_cuda()

    n_iters = 8
    batch_size = 16
    runtime_draft_len = 4
    value = 2.6

    # 1) Eager reference.
    eager_worker = _make_worker(value)
    eager_outputs = []
    for _ in range(n_iters):
        out = eager_worker._apply_force_accepted_tokens(
            _make_input(batch_size=batch_size), num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        eager_outputs.append(out.clone())

    # 2) Captured-graph run, aligned to the eager reference.
    graph_worker = _make_worker(value)
    static_input = _make_input(batch_size=batch_size)
    # Eager warmup: forces lazy RNG state allocation OUTSIDE capture.
    graph_worker._apply_force_accepted_tokens(
        static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
    )
    # Realign the device-side counter so the captured graph's first replay
    # advances 0 → 1, matching the eager loop's first iteration.
    graph_worker._force_accept_rng_counter.zero_()

    static_input.fill_(1)
    static_output = torch.empty_like(static_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_worker._apply_force_accepted_tokens(
            static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        # ``_apply_force_accepted_tokens`` mutates ``static_input`` in place;
        # mirror it into ``static_output`` so we can snapshot per replay.
        static_output.copy_(static_input)

    graph_outputs = []
    for _ in range(n_iters):
        # Reset input each replay so the override semantics match eager
        # (which also starts from all-1s every iteration).
        static_input.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        graph_outputs.append(static_output.clone())

    for i, (eager, graphed) in enumerate(zip(eager_outputs, graph_outputs)):
        assert torch.equal(eager, graphed), (
            f"Iteration {i}: eager={eager.tolist()} vs graphed={graphed.tolist()}"
        )


def test_cuda_graph_replay_advances_rng_state():
    """
    Across replays, the captured counter must advance, producing different
    (but deterministic) draws. A constant output across replays would mean
    the counter increment was not captured.
    """
    _require_cuda()

    worker = _make_worker(0.5)
    batch_size = 64
    runtime_draft_len = 4

    static_input = _make_input(batch_size=batch_size)
    worker._apply_force_accepted_tokens(
        static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
    )
    worker._force_accept_rng_counter.zero_()

    static_input.fill_(1)
    static_output = torch.empty_like(static_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._apply_force_accepted_tokens(
            static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        static_output.copy_(static_input)

    snapshots = []
    for _ in range(8):
        static_input.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        snapshots.append(static_output.clone())

    # At least two replays must differ; otherwise the captured counter is
    # not advancing (the original bug-class).
    assert any(not torch.equal(snapshots[0], s) for s in snapshots[1:]), (
        "Captured graph produced identical outputs on every replay — "
        "RNG counter is not advancing inside the captured graph."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class _FailingSpecWorker(_StubSpecWorker):
    """Stub whose forward saves spec-dec metadata state and then fails."""

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        attn_metadata = kwargs["attn_metadata"]
        attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        raise RuntimeError("simulated draft failure")


def test_forward_restores_spec_dec_state_on_failure() -> None:
    """A failure between prepare_for_spec_dec and restore must not leak saved
    attn-metadata state; SpecWorkerBase.forward restores it in its cleanup
    (https://nvbugs/6442074)."""
    _require_cuda()
    attn_metadata = AttentionMetadata(max_num_requests=2, max_num_tokens=16)
    attn_metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    worker = _FailingSpecWorker()
    for _ in range(2):
        # The second iteration would trip the pairing assert inside
        # prepare_for_spec_dec if the first failure had leaked saved state.
        with pytest.raises(RuntimeError, match="simulated draft failure"):
            worker(attn_metadata=attn_metadata, spec_metadata=None)
        assert not attn_metadata.has_spec_dec_saved_state


# ---------------- state-keyed draws (TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY) ------

_WIDTH = 7  # MTP6: the newest committed token + 6 drafts per verify group


def _make_state_worker(value: float = 2.75) -> _StubSpecWorker:
    with patch.dict(os.environ, {FORCE_ACCEPT_KEY_ENV_VAR: FORCE_ACCEPT_KEY_STATE}):
        worker = _StubSpecWorker()
    worker.force_num_accepted_tokens = value
    return worker


def _step_inputs(positions, tokens, num_ctx_tokens=0, width=_WIDTH, draft_seed=0):
    """``input_ids`` and ``position_ids`` ``[1, N]`` of a target step: ``num_ctx_tokens`` context
    tokens, then generation request ``g``'s verify group, its newest committed token
    ``tokens[g]`` at ``positions[g]`` followed by ``width - 1`` drafts."""
    n = len(positions)
    total = num_ctx_tokens + n * width
    pos = torch.empty(total, dtype=torch.int32)
    ids = torch.empty(total, dtype=torch.int32)
    pos[:num_ctx_tokens] = torch.arange(num_ctx_tokens, dtype=torch.int32)
    ids[:num_ctx_tokens] = 7
    drafts = torch.randint(
        0, 163840, (n, width - 1), generator=torch.Generator().manual_seed(draft_seed)
    )
    for g in range(n):
        start = num_ctx_tokens + g * width
        pos[start : start + width] = torch.arange(positions[g], positions[g] + width)
        ids[start] = tokens[g]
        ids[start + 1 : start + width] = drafts[g]
    return ids.cuda(), pos.cuda().unsqueeze(0)


def _apply_state(worker, positions, tokens, num_contexts=0, num_ctx_tokens=0, draft_seed=0):
    """``_apply_force_accepted_tokens`` with the step inputs forward() would hold."""
    ids, pos = _step_inputs(positions, tokens, num_ctx_tokens, draft_seed=draft_seed)
    worker._force_accept_inputs = (ids, pos, num_ctx_tokens)
    try:
        return worker._apply_force_accepted_tokens(
            _make_input(num_contexts + len(positions)),
            num_contexts=num_contexts,
            runtime_draft_len=_WIDTH - 1,
        )
    finally:
        worker._force_accept_inputs = None


def _expected_state(positions, tokens, value=2.75):
    """Reference: entry (position * 6007 + token * 1009 + ties * 4093) mod 2^16 of the fixed-seed
    pool, ties = the number of earlier rows with the same (position, token)."""
    from tensorrt_llm._torch.speculative import interface as itf

    gen = torch.Generator(device="cpu").manual_seed(itf._FORCE_ACCEPT_RNG_SEED)
    pool = torch.rand(itf._FORCE_ACCEPT_RNG_POOL_SIZE, dtype=torch.float32, generator=gen)
    keys = list(zip(positions, tokens))
    ties = [keys[:g].count(keys[g]) for g in range(len(keys))]
    idx = [
        (
            p * itf._FORCE_ACCEPT_RNG_COUNTER_STRIDE
            + t * itf._FORCE_ACCEPT_RNG_SLOT_STRIDE
            + k * 4093
        )
        & (itf._FORCE_ACCEPT_RNG_POOL_SIZE - 1)
        for (p, t), k in zip(keys, ties)
    ]
    base_total = min(int(value) + 1, _WIDTH)
    extra = (pool[idx] < value - int(value)).to(torch.int32)
    return (base_total + extra).clamp(max=_WIDTH)


@pytest.mark.parametrize(
    "env_value, expected",
    [
        (None, "counter"),
        ("counter", "counter"),
        ("state", "state"),
        (" State ", "state"),
        ("x", "counter"),
    ],
)
def test_get_force_accept_key(env_value, expected, monkeypatch):
    if env_value is None:
        monkeypatch.delenv(FORCE_ACCEPT_KEY_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FORCE_ACCEPT_KEY_ENV_VAR, env_value)
    assert get_force_accept_key() == expected


def test_state_key_draws_from_newest_committed_position_and_token():
    _require_cuda()
    worker = _make_state_worker()
    positions = [163840, 163840, 5, 1000003, 77]
    tokens = [11, 12, 163000, 0, 11]
    out = _apply_state(worker, positions, tokens)
    assert torch.equal(out.cpu(), _expected_state(positions, tokens))
    # The drafts of the verify groups do not enter the key.
    assert torch.equal(_apply_state(worker, positions, tokens, draft_seed=9), out)
    # The counter is neither used nor advanced.
    assert torch.equal(worker._force_accept_rng_counter.cpu(), torch.zeros(1, dtype=torch.int64))


def test_state_key_invariant_to_batch_row_and_counter():
    _require_cuda()
    worker = _make_state_worker()
    positions = [163840 + 3 * i for i in range(9)]
    tokens = [(i * 7919) % 163840 for i in range(9)]
    ref = _apply_state(worker, positions, tokens)
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(3)).tolist()
    permuted = _apply_state(worker, [positions[i] for i in perm], [tokens[i] for i in perm])
    assert torch.equal(permuted, ref[perm])
    # Earlier replays (a different counter) do not change the draws.
    worker._force_accept_rng_counter.fill_(12345)
    assert torch.equal(_apply_state(worker, positions, tokens), ref)


def test_state_key_tells_apart_requests_in_the_same_state():
    """Rows with the same (position, token) draw the entries of their tie counts 0, 1, ...; the
    other rows do not change, and permuting the batch permutes the draws of distinct requests and
    keeps the multiset of the tied ones."""
    _require_cuda()
    worker = _make_state_worker(0.5)
    positions = [163840, 900, 163840, 163840, 4242, 900]
    tokens = [17, 3, 17, 17, 8, 3]
    out = _apply_state(worker, positions, tokens)
    assert torch.equal(out.cpu(), _expected_state(positions, tokens, 0.5))
    alone = _apply_state(worker, [163840, 4242], [17, 8])
    assert torch.equal(out[[0, 4]], alone)
    perm = [4, 3, 5, 0, 2, 1]
    permuted = _apply_state(worker, [positions[i] for i in perm], [tokens[i] for i in perm])
    assert permuted[0] == out[4]
    assert sorted(permuted[[1, 3, 4]].tolist()) == sorted(out[[0, 2, 3]].tolist())
    assert sorted(permuted[[2, 5]].tolist()) == sorted(out[[1, 5]].tolist())


def test_state_key_identical_on_independent_ranks():
    _require_cuda()
    rank0, rank1 = _make_state_worker(), _make_state_worker()
    torch.cuda.manual_seed(11)
    positions = [10_000 + 37 * i for i in range(33)]
    tokens = [(i * 104729) % 163840 for i in range(33)]
    out0 = _apply_state(rank0, positions, tokens)
    torch.cuda.manual_seed(999)
    _ = torch.rand(128, device="cuda")
    out1 = _apply_state(rank1, positions, tokens)
    assert torch.equal(out0, out1)


def test_state_key_skips_context_rows():
    _require_cuda()
    worker = _make_state_worker()
    num_contexts, num_ctx_tokens = 2, 19
    positions, tokens = [4096, 333, 90001], [1, 2, 3]
    ids, pos = _step_inputs(positions, tokens, num_ctx_tokens)
    worker._force_accept_inputs = (ids, pos, num_ctx_tokens)
    base = _make_input(num_contexts + 3)
    base[:num_contexts] = 1
    out = worker._apply_force_accepted_tokens(base, num_contexts, runtime_draft_len=_WIDTH - 1)
    assert torch.all(out[:num_contexts] == 1)
    assert torch.equal(out[num_contexts:].cpu(), _expected_state(positions, tokens))


def test_state_key_rejects_uneven_layout():
    _require_cuda()
    worker = _make_state_worker()
    ids, pos = _step_inputs([100, 200], [5, 6])
    worker._force_accept_inputs = (ids[:-1], pos[:, :-1], 0)
    with pytest.raises(RuntimeError, match="equal-width"):
        worker._apply_force_accepted_tokens(_make_input(2), 0, runtime_draft_len=_WIDTH - 1)
    worker._force_accept_inputs = None
    with pytest.raises(RuntimeError, match="forward"):
        worker._apply_force_accepted_tokens(_make_input(2), 0, runtime_draft_len=_WIDTH - 1)


def _toy_next_token(position: int, token: int) -> int:
    """A deterministic stand-in for the model: the newest committed token after a step."""
    return (position * 2654435761 + token * 40503) % 163840


def _simulate_run(
    worker, admit_step, order, num_steps, prompt_len=163840, warmup_calls=0, first_token=None
):
    """Per-request accepted counts of a run of a toy batch-invariant model.

    Request ``r`` joins the batch at step ``admit_step[r]`` with ``first_token(r)`` (default
    ``1000 + r``) as its newest token and generates ``num_steps`` steps; ``order(step, active)``
    gives the batch rows; ``warmup_calls`` replays happen first."""
    for _ in range(warmup_calls):
        if worker.force_accept_key == "state":
            _apply_state(worker, [1, 2], [3, 4])
        else:
            worker._apply_force_accepted_tokens(_make_input(2), 0, runtime_draft_len=_WIDTH - 1)
    first_token = first_token or (lambda r: 1000 + r)
    state = {r: (prompt_len, first_token(r)) for r in admit_step}
    history = {r: [] for r in admit_step}
    step = 0
    while any(len(h) < num_steps for h in history.values()):
        active = [r for r in admit_step if admit_step[r] <= step and len(history[r]) < num_steps]
        if active:
            rows = order(step, active)
            positions = [state[r][0] for r in rows]
            tokens = [state[r][1] for r in rows]
            if worker.force_accept_key == "state":
                out = _apply_state(worker, positions, tokens).tolist()
            else:
                out = worker._apply_force_accepted_tokens(
                    _make_input(len(rows)), 0, runtime_draft_len=_WIDTH - 1
                ).tolist()
            for r, n in zip(rows, out):
                history[r].append(n)
                pos = state[r][0] + n
                state[r] = (pos, _toy_next_token(pos, state[r][1]))
        step += 1
    return history


@pytest.mark.parametrize("key", ["counter", "state"])
def test_runs_replay_the_same_streams_only_with_state_key(key):
    """Two runs of the same requests whose admission step, batch rows and replay count differ:
    with the state key every request accepts the same sequence (so a bit-exact configuration
    generates the same tokens); with the counter key the sequences differ."""
    _require_cuda()

    def make():
        if key == "state":
            return _make_state_worker()
        return _make_worker(2.75)

    run_a = _simulate_run(make(), {0: 0, 1: 0, 2: 0, 3: 0}, lambda s, a: a, num_steps=60)
    run_b = _simulate_run(
        make(),
        {0: 3, 1: 0, 2: 1, 3: 0},
        lambda s, a: list(reversed(a)) if s % 2 else a,
        num_steps=60,
        warmup_calls=5,
    )
    if key == "state":
        assert run_a == run_b
    else:
        assert run_a != run_b


def test_state_key_runs_with_identical_requests():
    """Two requests in the same state (a duplicated prompt: same position, same newest token) do
    not run in lockstep: the tie count gives them different draws. Which one gets which draw
    depends on the rows, so runs agree on the multiset of streams."""
    _require_cuda()

    def first_token(r):  # requests 0 and 2 are copies of one prompt
        return 1000 + (r % 2)

    def run(order, admit):
        worker = _make_state_worker()
        hist = _simulate_run(worker, admit, order, num_steps=60, first_token=first_token)
        return hist

    run_a = run(lambda s, a: a, {0: 0, 1: 0, 2: 0})
    run_b = run(lambda s, a: list(reversed(a)), {0: 0, 1: 0, 2: 0})
    assert run_a[0] != run_a[2]  # the copies diverge
    assert run_a[1] == run_b[1]  # the distinct request is unaffected
    assert sorted([run_a[0], run_a[2]]) == sorted([run_b[0], run_b[2]])


@pytest.mark.parametrize("value", [2.75, 2.6, 0.5])
def test_state_key_mean_matches_target(value):
    """Mean accepted drafts over long toy streams match the forced value."""
    _require_cuda()
    worker = _make_state_worker(value)
    num_requests, num_steps = 64, 300
    positions = [163840 + 17 * r for r in range(num_requests)]
    tokens = [(r * 7919) % 163840 for r in range(num_requests)]
    total = 0
    for _ in range(num_steps):
        out = _apply_state(worker, positions, tokens).tolist()
        total += sum(out) - num_requests
        for r, n in enumerate(out):
            positions[r] += n
            tokens[r] = _toy_next_token(positions[r], tokens[r])
    mean = total / (num_requests * num_steps)
    # Bernoulli(frac) over 19200 draws: std of the mean <= 0.0036.
    assert abs(mean - value) < 0.02, f"mean accepted drafts {mean:.4f} vs {value}"


def test_state_key_cuda_graph_replay_matches_eager():
    _require_cuda()
    eager, graphed = _make_state_worker(), _make_state_worker()
    num_gens = 4
    ids, pos = _step_inputs([0] * num_gens, [0] * num_gens)
    static_in = _make_input(num_gens)
    graphed._ensure_force_accept_rng_state(torch.device("cuda"))
    graphed._force_accept_inputs = (ids, pos, 0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graphed._apply_force_accepted_tokens(static_in, 0, runtime_draft_len=_WIDTH - 1)
    torch.cuda.synchronize()
    out = torch.empty_like(static_in)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed._apply_force_accepted_tokens(static_in, 0, runtime_draft_len=_WIDTH - 1)
        out.copy_(static_in)
    graphed._force_accept_inputs = None
    for step in range(6):
        positions = [163840 + 4 * step + g for g in range(num_gens)]
        tokens = [(step * 31 + g * 7) % 163840 for g in range(num_gens)]
        new_ids, new_pos = _step_inputs(positions, tokens, draft_seed=step)
        ids.copy_(new_ids)
        pos.copy_(new_pos)
        static_in.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _apply_state(eager, positions, tokens))
        assert torch.equal(out.cpu(), _expected_state(positions, tokens))


class _ForwardStateWorker(_StubSpecWorker):
    """Forced acceptance from inside forward(), as every one-model worker does."""

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        return self._apply_force_accepted_tokens(
            _make_input(attn_metadata.num_seqs),
            attn_metadata.num_contexts,
            runtime_draft_len=_WIDTH - 1,
        )


@pytest.mark.parametrize("key", ["counter", "state"])
def test_forward_holds_the_step_inputs_only_with_state_key(key):
    from types import SimpleNamespace

    _require_cuda()
    with patch.dict(os.environ, {FORCE_ACCEPT_KEY_ENV_VAR: key}):
        worker = _ForwardStateWorker()
    worker.force_num_accepted_tokens = 2.75
    positions, tokens = [163840, 163841, 9], [5, 5, 160000]
    ids, pos = _step_inputs(positions, tokens, num_ctx_tokens=0)
    md = SimpleNamespace(
        num_seqs=3, num_contexts=0, num_ctx_tokens=0, has_spec_dec_saved_state=False
    )
    seen = []
    original = worker._apply_force_accepted_tokens

    def spy(*args, **kwargs):
        seen.append(worker._force_accept_inputs)
        return original(*args, **kwargs)

    worker._apply_force_accepted_tokens = spy
    out = worker(
        input_ids=ids,
        position_ids=pos,
        hidden_states=None,
        logits=None,
        attn_metadata=md,
        spec_metadata=None,
        draft_model=None,
    )
    assert worker._force_accept_inputs is None
    if key == "state":
        assert seen[0][0] is ids and seen[0][1] is pos and seen[0][2] == 0
        assert torch.equal(out.cpu(), _expected_state(positions, tokens))
    else:
        assert seen == [None]
        assert int(worker._force_accept_rng_counter.item()) == 1


def test_stream_log_records_finished_requests(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm._torch.speculative import spec_sampler_base

    log = spec_sampler_base._StreamLog(str(tmp_path / "streams"))
    tokens = list(range(100, 110))
    req = SimpleNamespace(
        py_request_id=42,
        py_prompt_len=3,
        total_input_len_cp=163840,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        get_tokens=lambda beam: [1, 2, 3] + tokens,
    )
    log.on_step(req, 4)
    log.on_step(req, 3)
    assert not list((tmp_path / "streams").glob("*.jsonl"))
    req.state = LlmRequestState.GENERATION_COMPLETE
    log.on_step(req, 3)
    (path,) = list((tmp_path / "streams").glob("rank*.jsonl"))
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["request_id"] == 42
    assert record["prompt_len"] == 163840
    assert record["steps"] == 3 and record["tokens"] == 10
    assert record["first_tokens"] == tokens[:8]
    assert record["token_digest"] == spec_sampler_base._digest(tokens)
    assert record["accept_digest"] == spec_sampler_base._digest([4, 3, 3])
    assert 42 not in log._accepted
