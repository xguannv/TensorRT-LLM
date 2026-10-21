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
"""TLLM_SPEC_DECODE_FORCE_REPLAY: replaying requests commit their recorded streams and take the
recorded continuation as drafts, whatever the target selects and the drafter proposes."""

import json

import pytest
import torch

from tensorrt_llm._torch.speculative.force_replay import (
    ForceReplay,
    apply_replay_drafts,
    load_reference_streams,
    start_reference,
)
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    ForceAcceptKey,
    fused_next_new_tokens,
    fused_next_new_tokens_rewind,
    fused_select_accept,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

K = 6  # drafts per step (MTP6)
W = K + 1  # verify group width
VOCAB = 163840
POOL_SIZE = 1 << 16


def _streams(seed=0, lengths=(300, 260, 280), bases=(163840, 163842, 163840)):
    """Recorded streams: two share a prompt length (different first tokens), one is 2 longer."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for base, n in zip(bases, lengths):
        out.append((base, torch.randint(0, VOCAB, (n,), generator=gen).tolist()))
    return out


def _pool(seed=1):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(POOL_SIZE, generator=gen).cuda()


def _pairs(targets):
    """One (value, index) slot per row whose argmax is ``targets`` (the target's selection)."""
    buf = torch.empty((targets.numel(), 2), dtype=torch.float32, device="cuda")
    buf[:, 0] = 1.0
    buf[:, 1] = targets.float()
    return buf, 2, 2, 1


def _step(replay, pool, positions, groups, targets, drafts_in, drafter_out, frac=0.75, base=3):
    """One generation step through the fused select-accept and next-new-tokens kernels.

    ``positions[g]`` / ``groups[g]``: global position of request g's newest committed token and
    its verify group (the step's input ids); ``targets`` the target's selected tokens, ``drafts_in``
    the step's drafts, ``drafter_out`` the drafter's next drafts. Returns (accepted, n, next inputs,
    next drafts)."""
    num_gens = len(positions)
    pos = torch.tensor(
        [[p + c for c in range(W)] for p in positions], dtype=torch.int32, device="cuda"
    ).flatten()
    ids = torch.tensor(groups, dtype=torch.int32, device="cuda").flatten()
    key = ForceAcceptKey(pos, ids, W)
    force = (base, frac, W, pool, key, 6007, 1009)
    acc, n, _ = fused_select_accept(
        _pairs(torch.tensor(targets, device="cuda").flatten()),
        torch.tensor(drafts_in, dtype=torch.int32, device="cuda"),
        0,
        num_gens,
        force,
        replay=replay,
    )
    nxt_drafts = torch.tensor(drafter_out, dtype=torch.int32, device="cuda")
    nxt = fused_next_new_tokens(acc, nxt_drafts, n, replay=(replay, 0, num_gens))
    return acc, n, nxt, nxt_drafts


def test_load_reference_streams_dedupes_and_starts(tmp_path):
    path = tmp_path / "ref.jsonl"
    recs = [
        {"prompt_len": 10, "token_ids": [5, 6, 7]},
        {"prompt_len": 10, "token_ids": [5, 9, 9]},  # same (prompt_len, first token): dropped
        {"prompt_len": 10, "token_ids": [8, 1]},
        {"prompt_len": 12, "token_ids": []},  # no tokens: dropped
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    streams = load_reference_streams(str(path))
    assert streams == [(10, [5, 6, 7]), (10, [8, 1])]
    assert start_reference(streams, 10, 8) == 1
    assert start_reference(streams, 10, 4) == -1
    assert start_reference(streams, 11, 5) == -1


def test_replayed_rows_commit_the_recording_and_take_its_drafts():
    streams = _streams()
    replay = ForceReplay.build(streams, torch.device("cuda"), 16, K)
    pool = _pool()
    base0, s0 = streams[0]
    base1, s1 = streams[1]
    gen = torch.Generator().manual_seed(3)
    rnd = lambda *shape: torch.randint(0, VOCAB, shape, generator=gen).tolist()  # noqa: E731
    j1 = 37
    positions = [
        base0,  # first step of stream 0: context token + dummy zero drafts
        base1 + j1,  # mid-stream of stream 1: the group is the recording
        base0 + 5,  # a group that follows no recording
        base1,  # first step with a context token no recording starts with
    ]
    groups = [
        [s0[0]] + [0] * K,
        s1[j1 : j1 + W],
        rnd(W),
        [(s0[0] + 1) % VOCAB] + [0] * K,
    ]
    targets = rnd(4, W)
    drafter = rnd(4, K)
    acc, n, nxt, nxt_drafts = _step(replay, pool, positions, groups, targets, rnd(4, K), drafter)
    acc, n, nxt, nxt_drafts = acc.cpu(), n.cpu(), nxt.cpu(), nxt_drafts.cpu()
    assert set(n.tolist()) <= {3, 4}
    expect = [(s0, 0), (s1, j1), None, None]
    for g, e in enumerate(expect):
        ng = int(n[g])
        if e is None:
            assert acc[g].tolist() == targets[g]
            assert nxt_drafts[g].tolist() == drafter[g]
            assert nxt[g].tolist() == [targets[g][ng - 1]] + drafter[g]
        else:
            s, j = e
            assert acc[g].tolist() == s[j + 1 : j + 1 + W]
            assert nxt[g].tolist() == s[j + ng : j + ng + W]
            assert nxt_drafts[g].tolist() == s[j + ng + 1 : j + ng + W]
    assert replay.flags[:4].tolist() == [1, 1, 0, 0]


def test_replay_ends_with_the_recording():
    """At the end of a stream the committed tokens fall back to the target's and the drafts
    repeat the last recorded token."""
    streams = _streams()
    replay = ForceReplay.build(streams, torch.device("cuda"), 16, K)
    base, s = streams[1]
    j = len(s) - 3  # two recorded tokens follow the newest committed one
    gen = torch.Generator().manual_seed(5)
    targets = torch.randint(0, VOCAB, (1, W), generator=gen).tolist()
    group = (s + [s[-1]] * W)[j : j + W]
    acc, n, nxt, _ = _step(replay, _pool(), [base + j], [group], targets, [[0] * K], [[1] * K])
    acc = acc[0].tolist()
    left = len(s) - j - 1
    assert acc[:left] == s[j + 1 :]
    assert acc[left:] == targets[0][left:]
    ng = int(n[0])
    tail = (s + [s[-1]] * 3 * W)[j + ng : j + ng + W]
    assert nxt[0].tolist() == [acc[ng - 1]] + tail[1:]


def _run(streams, starts, noise_seed, steps=60, frac=0.75, rewind=False, base=3):
    """A toy decode loop: every step the target selects random tokens and the drafter proposes
    random drafts (numerics noise). Returns each request's committed tokens and counts."""
    replay = ForceReplay.build(streams, torch.device("cuda"), 16, K)
    pool = _pool()
    gen = torch.Generator().manual_seed(noise_seed)
    # Request r starts from stream starts[r]: its context token at the stream's prompt length.
    pos = [streams[r][0] for r in starts]
    groups = [[streams[r][1][0]] + [0] * K for r in starts]
    drafts = [[0] * K for _ in starts]
    committed = [[streams[r][1][0]] for r in starts]
    counts = [[] for _ in starts]
    kv = torch.zeros(len(starts), dtype=torch.int32, device="cuda")
    for _ in range(steps):
        targets = torch.randint(0, VOCAB, (len(starts), W), generator=gen).tolist()
        drafter = torch.randint(0, VOCAB, (len(starts), K), generator=gen).tolist()
        if rewind:
            num_gens = len(starts)
            p = torch.tensor(
                [[q + c for c in range(W)] for q in pos], dtype=torch.int32, device="cuda"
            ).flatten()
            ids = torch.tensor(groups, dtype=torch.int32, device="cuda").flatten()
            force = (base, frac, W, pool, ForceAcceptKey(p, ids, W), 6007, 1009)
            acc, n, rw = fused_select_accept(
                _pairs(torch.tensor(targets, device="cuda").flatten()),
                torch.tensor(drafts, dtype=torch.int32, device="cuda"),
                0,
                num_gens,
                force,
                kv_lens=kv,
                replay=replay,
            )
            nd = torch.tensor(drafter, dtype=torch.int32, device="cuda")
            nxt = fused_next_new_tokens_rewind(acc, nd, n, kv, rw, 0, replay=(replay, 0, num_gens))
        else:
            acc, n, nxt, nd = _step(replay, pool, pos, groups, targets, drafts, drafter, frac, base)
        acc, n, nxt, nd = acc.tolist(), n.tolist(), nxt.tolist(), nd.tolist()
        for g in range(len(starts)):
            committed[g] += acc[g][: n[g]]
            counts[g].append(n[g])
            pos[g] += n[g]
            groups[g] = nxt[g]
            drafts[g] = nd[g]
    return committed, counts


@pytest.mark.parametrize("rewind", [False, True])
def test_replay_is_independent_of_the_numerics(rewind):
    """Two runs whose target and drafter outputs differ commit the same streams: the recorded
    ones, with the same accepted counts (keyed by the replayed tokens)."""
    streams = _streams(lengths=(400, 400, 400))
    a_tokens, a_counts = _run(streams, [0, 1], noise_seed=1, rewind=rewind)
    b_tokens, b_counts = _run(streams, [0, 1], noise_seed=2, rewind=rewind)
    assert a_tokens == b_tokens and a_counts == b_counts
    for g, r in enumerate([0, 1]):
        assert a_tokens[g] == streams[r][1][: len(a_tokens[g])]


def test_replay_mean_acceptance():
    streams = _streams(lengths=(1300, 1300, 1300), seed=7)
    _, counts = _run(streams, [0, 1, 2], noise_seed=3, steps=280)
    flat = [c for cs in counts for c in cs]
    mean_drafts = sum(flat) / len(flat) - 1
    # 840 Bernoulli(0.75) draws: std of the mean 0.015.
    assert abs(mean_drafts - 2.75) < 0.06, mean_drafts


@pytest.mark.parametrize("rewind", [False, True])
def test_replay_whole_forced_count(rewind):
    """A whole forced count (e.g. 3.00 drafts at MTP8: base 4, frac 0) replays like a fractional
    one: every step commits exactly 4 recorded tokens, whatever the numerics."""
    streams = _streams(lengths=(400, 400, 400))
    a_tokens, a_counts = _run(streams, [0, 1], noise_seed=1, frac=0.0, rewind=rewind, base=4)
    b_tokens, b_counts = _run(streams, [0, 1], noise_seed=2, frac=0.0, rewind=rewind, base=4)
    assert a_tokens == b_tokens and a_counts == b_counts
    assert all(c == 4 for cs in a_counts for c in cs)
    for g, r in enumerate([0, 1]):
        assert a_tokens[g] == streams[r][1][: len(a_tokens[g])]


@pytest.mark.parametrize(
    "forced,replaying,expect",
    [
        (3.0, True, (4, 0.0)),  # whole count: state-keyed form (the replay rides on it)
        (3.0, False, (4,)),  # whole count without replay: the constant form
        (2.75, True, (3, 0.75)),
        (2.75, False, (3, 0.75)),
    ],
)
def test_force_kernel_args_whole_count_keeps_the_replay(forced, replaying, expect):
    """``SpecWorkerBase._force_accept_kernel_args`` (the accept kernels' ``force`` argument)
    takes the state-keyed form whenever the run replays, also for a whole forced count."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.speculative.interface import FORCE_ACCEPT_KEY_STATE, SpecWorkerBase

    key = ForceAcceptKey(
        torch.zeros(W, dtype=torch.int32, device="cuda"),
        torch.zeros(W, dtype=torch.int32, device="cuda"),
        W,
    )
    worker = SimpleNamespace(
        force_num_accepted_tokens=forced,
        force_accept_key=FORCE_ACCEPT_KEY_STATE,
        _force_replay_streams=_streams() if replaying else None,
        _force_accept_rng_pool=_pool(),
        _force_accept_rng_counter=None,
        _ensure_force_accept_rng_state=lambda device: None,
        _force_accept_state_key=lambda num_gens: key,
    )
    force = SpecWorkerBase._force_accept_kernel_args(worker, K, None, torch.device("cuda"), 1)
    assert force[: len(expect)] == expect
    if len(expect) > 1:
        assert len(force) == 7 and force[2] == K + 1 and force[4] is key


def test_copies_of_one_recording_diverge():
    """Two requests starting from the same recording (a duplicated prompt) are told apart by
    the tie count: both replay the recording, at different positions."""
    streams = _streams(lengths=(400, 400, 400))
    tokens, counts = _run(streams, [0, 0], noise_seed=4, steps=40)
    assert counts[0] != counts[1]
    for g in range(2):
        assert tokens[g] == streams[0][1][: len(tokens[g])]


def test_apply_replay_drafts_matches_the_kernel():
    streams = _streams()
    replay = ForceReplay.build(streams, torch.device("cuda"), 16, K)
    replay.flags[:3].copy_(torch.tensor([1, 0, 1], dtype=torch.int32))
    replay.drafts[:3].copy_(torch.arange(3 * K, dtype=torch.int32).view(3, K))
    drafter = torch.randint(0, VOCAB, (5, K), dtype=torch.int32, device="cuda")
    accepted = torch.randint(0, VOCAB, (5, W), dtype=torch.int32, device="cuda")
    n = torch.tensor([1, 2, 3, 4, 1], dtype=torch.int32, device="cuda")
    ref = drafter.clone()
    apply_replay_drafts(ref, replay, 2, 3)
    got = drafter.clone()
    nxt = fused_next_new_tokens(accepted, got, n, replay=(replay, 2, 3))
    assert torch.equal(got, ref)
    assert torch.equal(nxt[:, 1:], ref)
    assert torch.equal(ref[:2], drafter[:2])
    assert torch.equal(ref[3], drafter[3])
    assert torch.equal(ref[2], replay.drafts[0]) and torch.equal(ref[4], replay.drafts[2])


def test_replay_cuda_graph_matches_eager():
    streams = _streams(lengths=(400, 400, 400))
    replay = ForceReplay.build(streams, torch.device("cuda"), 16, K)
    pool = _pool()
    num_gens = 2
    pos = torch.zeros(num_gens * W, dtype=torch.int32, device="cuda")
    ids = torch.zeros(num_gens * W, dtype=torch.int32, device="cuda")
    pairs = torch.zeros((num_gens * W, 2), dtype=torch.float32, device="cuda")
    drafts = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
    drafter = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
    force = (3, 0.75, W, pool, ForceAcceptKey(pos, ids, W), 6007, 1009)

    def launch():
        acc, n, _ = fused_select_accept((pairs, 2, 2, 1), drafts, 0, num_gens, force, replay=replay)
        nxt = fused_next_new_tokens(acc, drafter, n, replay=(replay, 0, num_gens))
        return acc, n, nxt

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_acc, g_n, g_nxt = launch()
    gen = torch.Generator().manual_seed(9)
    state = [
        (streams[0][0], [streams[0][1][0]] + [0] * K),
        (streams[1][0], [streams[1][1][0]] + [0] * K),
    ]
    for _ in range(8):
        targets = torch.randint(0, VOCAB, (num_gens, W), generator=gen)
        dr = torch.randint(0, VOCAB, (num_gens, K), generator=gen)
        pos.copy_(torch.tensor([[p + c for c in range(W)] for p, _ in state]).flatten())
        ids.copy_(torch.tensor([grp for _, grp in state]).flatten())
        pairs[:, 0] = 1.0
        pairs[:, 1] = targets.flatten().float().cuda()
        drafter.copy_(dr)
        graph.replay()
        e_acc, e_n, e_nxt, _ = _step(
            replay,
            pool,
            [p for p, _ in state],
            [grp for _, grp in state],
            targets.tolist(),
            drafts.tolist(),
            dr.tolist(),
        )
        torch.cuda.synchronize()
        assert torch.equal(g_acc, e_acc) and torch.equal(g_n, e_n) and torch.equal(g_nxt, e_nxt)
        state = [(p + int(g_n[g]), g_nxt[g].tolist()) for g, (p, _) in enumerate(state)]
