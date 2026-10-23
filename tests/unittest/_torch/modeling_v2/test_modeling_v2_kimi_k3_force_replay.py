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
"""``TRTLLM_K3LL_FORCE_REPLAY`` (``kimi_k3_nvfp4/force_replay.py``, measurement only) on one GPU:

* switch off: ``replay_begin`` returns None and launches nothing (the MLA DSpark worker's step is then exactly the
  pre-change step), also with the switch on but forced acceptance off;
* the lookup and the commit against a Python reference: rows that follow an entry (any offset, including rows past
  the end of their stream and rows of no stream), several placements of one stream, the RNG counter keyed by row 0's
  offset, at 1 / 2 / 4 / 8 generation rows (8 / 16 / 32 / 64 tokens) behind context tokens, eagerly and from a CUDA
  graph replayed with new inputs;
* a decode loop: rows started on their streams commit exactly the streams' tokens whatever the accepted counts.
"""

import json
import random

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")

ROWS = (1, 2, 4, 8)
K = 7


def _mod():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        force_replay,
    )

    return force_replay


def _flags():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    return flags


def _entries(rng, n_streams=5, bases=(153600, 65536), length=(60, 90)):
    streams = [
        [rng.randrange(1000, 150000) for _ in range(rng.randrange(*length))]
        for _ in range(n_streams)
    ]
    return [(b, s) for b in bases for s in streams]


def _ref_lookup(entries, ids, pos, tok0, rows, k1):
    out = []
    for g in range(rows):
        group = ids[tok0 + g * k1 : tok0 + (g + 1) * k1]
        p = pos[tok0 + g * k1]
        hit = (-1, 0)
        for e, (base, tok) in enumerate(entries):
            j = p - base
            if 0 <= j < len(tok):
                padded = tok + [tok[-1]] * (2 * (K + 2))
                if padded[j : j + k1] == group:
                    hit = (e, j)
                    break
        out.append(hit)
    return out


def _padded(entries, e, i):
    tok = entries[e][1]
    return tok[i] if i < len(tok) else tok[-1]


def _case(rng, entries, rows, nctx):
    """Input ids / positions of ``nctx`` context tokens then ``rows`` groups of K + 1: matching rows at random offsets
    (some near / past the end), rows with a wrong token, rows at a position of no entry."""
    k1 = K + 1
    ids, pos = [], []
    for _ in range(nctx):
        ids.append(rng.randrange(1000, 150000))
        pos.append(rng.randrange(0, 1000))
    for g in range(rows):
        kind = rng.choice(["match", "match", "match", "end", "wrong", "nowhere"])
        e = rng.randrange(len(entries))
        base, tok = entries[e]
        if kind == "end":
            j = rng.randrange(max(len(tok) - 3, 0), len(tok) + 3)
        else:
            j = rng.randrange(0, len(tok) - 1)
        group = [_padded(entries, e, j + c) for c in range(k1)]
        p = base + j
        if kind == "wrong":
            group[rng.randrange(k1)] = 7
        if kind == "nowhere":
            p = 10_000_000 + g
        ids.extend(group)
        pos.extend(range(p, p + k1))
    return ids, pos


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("nctx", (0, 5))
@pytest.mark.parametrize("graph", (False, True))
def test_lookup_commit_reference(rows, nctx, graph):
    fr = _mod()
    rng = random.Random(1000 * rows + 10 * nctx + int(graph))
    entries = _entries(rng)
    dev = torch.device("cuda")
    tables = fr.ReplayTables.build(entries, dev, K)
    k1 = K + 1
    ntok = nctx + rows * k1
    ncreq = 1 if nctx else 0
    batch = ncreq + rows
    ids_buf = torch.zeros(ntok, dtype=torch.int32, device=dev)
    pos_buf = torch.zeros(1, ntok, dtype=torch.int32, device=dev)
    counter = torch.full((1,), -5, dtype=torch.int64, device=dev)
    new = torch.zeros(batch, k1, dtype=torch.int32, device=dev)
    lens = torch.zeros(batch, dtype=torch.int32, device=dev)
    nd_base = torch.zeros(batch, K + 3, dtype=torch.int32, device=dev)
    nd = nd_base[:, :K]  # a strided view, as a slice of a wider buffer may be
    nn = torch.zeros(batch, k1, dtype=torch.int64, device=dev)

    def step():
        fr.replay_lookup(tables, ids_buf, pos_buf.reshape(-1), nctx, rows, k1, counter)
        fr.replay_commit(tables, new, lens, nd, nn, ncreq, rows, K)

    g = None
    if graph:
        step()  # compile outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
    for trial in range(3):
        ids, pos = _case(rng, entries, rows, nctx)
        n = [rng.randrange(1, k1 + 1) for _ in range(batch)]
        ids_buf.copy_(torch.tensor(ids, dtype=torch.int32))
        pos_buf.copy_(torch.tensor([pos], dtype=torch.int32))
        lens.copy_(torch.tensor(n, dtype=torch.int32))
        new.fill_(-1)
        nd_base.fill_(-1)
        nn.fill_(-1)
        counter.fill_(-5)
        if graph:
            g.replay()
        else:
            step()
        torch.cuda.synchronize()
        hits = _ref_lookup(entries, ids, pos, nctx, rows, k1)
        state = tables.state[:rows].tolist()
        assert [tuple(s) for s in state] == hits, (trial, state, hits)
        new_h, nd_h, nn_h = new.tolist(), nd_base.tolist(), nn.tolist()
        for r in range(batch):
            g_ = r - ncreq
            if g_ < 0 or hits[g_][0] < 0:
                assert new_h[r] == [-1] * k1 and nn_h[r] == [-1] * k1 and nd_h[r] == [-1] * (K + 3)
                continue
            e, j = hits[g_]
            assert new_h[r] == [_padded(entries, e, j + 1 + c) for c in range(k1)]
            assert nn_h[r] == [_padded(entries, e, j + n[r] + c) for c in range(k1)]
            assert nd_h[r][:K] == [_padded(entries, e, j + n[r] + 1 + c) for c in range(K)]
            assert nd_h[r][K:] == [-1, -1]
        want_counter = hits[0][1] if hits[0][0] >= 0 else -5
        assert counter.item() == want_counter


def test_decode_loop_follows_streams():
    fr = _mod()
    rng = random.Random(7)
    entries = _entries(rng, n_streams=4, bases=(153600,), length=(300, 301))
    dev = torch.device("cuda")
    tables = fr.ReplayTables.build(entries, dev, K)
    rows, k1 = 4, K + 1
    committed = [[entries[g][1][0]] for g in range(rows)]  # the context phase's token
    group = [entries[g][1][:k1] for g in range(rows)]
    pos = [153600] * rows
    while min(len(c) for c in committed) < 250:
        ids = torch.tensor(sum(group, []), dtype=torch.int32, device=dev)
        posv = torch.tensor(
            sum([list(range(p, p + k1)) for p in pos], []), dtype=torch.int32, device=dev
        )
        n = [rng.randrange(1, k1 + 1) for _ in range(rows)]
        lens = torch.tensor(n, dtype=torch.int32, device=dev)
        new = torch.randint(
            0, 100, (rows, k1), dtype=torch.int32, device=dev
        )  # the target's (wrong) tokens
        nd = torch.randint(
            0, 100, (rows, K), dtype=torch.int32, device=dev
        )  # the drafter's (wrong) drafts
        nn = torch.cat([new[:, :1], nd], dim=1)
        fr.replay_lookup(tables, ids, posv, 0, rows, k1, None)
        fr.replay_commit(tables, new, lens, nd, nn, 0, rows, K)
        for g in range(rows):
            committed[g].extend(new[g, : n[g]].tolist())
            group[g] = nn[g].tolist()
            assert group[g][1:] == nd[g].tolist()
            pos[g] += n[g]
    for g in range(rows):
        assert committed[g] == entries[g][1][: len(committed[g])]


def test_switch_off_launches_nothing(monkeypatch, tmp_path):
    fr = _mod()
    flags = _flags()

    class Worker:
        force_num_accepted_tokens = 2.84
        max_draft_len = K

        def _ensure_force_accept_rng_state(self, device):
            raise AssertionError("switch off must not touch the RNG state")

    class Meta:
        num_contexts, num_seqs, num_ctx_tokens = 0, 2, 0

    class Spec:
        runtime_draft_len = K

    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"prompt_len": 100, "token_ids": list(range(1000, 1100))}) + "\n")
    monkeypatch.setenv(fr.FILE_ENV, str(path))
    ids = torch.zeros(2 * (K + 1), dtype=torch.int32, device="cuda")
    pos = torch.zeros(1, 2 * (K + 1), dtype=torch.int32, device="cuda")
    calls = []
    monkeypatch.setattr(fr, "replay_lookup", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(flags, "TRTLLM_K3LL_FORCE_REPLAY", 0)
    w = Worker()
    assert fr.replay_begin(w, ids, pos, Meta, Spec) is None
    assert not calls and getattr(w, "_k3_replay_tables", None) is None
    # on, but without forced acceptance: nothing either
    monkeypatch.setattr(flags, "TRTLLM_K3LL_FORCE_REPLAY", 1)
    w.force_num_accepted_tokens = 0.0
    assert fr.replay_begin(w, ids, pos, Meta, Spec) is None
    assert not calls and getattr(w, "_k3_replay_tables", None) is None


def test_load_entries(tmp_path):
    fr = _mod()
    path = tmp_path / "r.jsonl"
    recs = [
        {"prompt_len": 10, "token_ids": [1, 2, 3]},
        {"prompt_len": 10, "token_ids": [1, 2, 3]},
        {"prompt_len": 20, "token_ids": [1, 2, 3]},
        {"prompt_len": 10, "token_ids": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert fr.load_entries(str(path)) == [(10, [1, 2, 3]), (20, [1, 2, 3])]
