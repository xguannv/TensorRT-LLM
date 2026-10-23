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
"""``TRTLLM_K3LL_MLA_VAR_SPLIT`` (``kimi_k3_nvfp4/flags.py``): the Helix V-up split read with each request's own
split-KV count (``decode_mla_dcp.var_split_counts``, on device), one rank's split read on one GPU (the helpers of
``test_modeling_v2_kimi_k3_mla_v_up``), at 8 / 16 / 32 / 64 rows per head (1 / 2 / 4 / 8 requests x 8 tokens), on
mixed lengths (AgentX's decode-time quantiles, scaled down), a request shorter than one K tile, a request with no page
on the rank, and equal lengths:

* the counts: at least 1, at most the grid's split count, within the one-wave budget, a python reference's choice;
  equal lengths split equally;
* switch off (no counts): the split read + BF16 BMM bit for bit, as before (the decode runner without the optional
  ``block_split_kvs`` input);
* switch on: a variable-split variant compiled; the softmax statistics within 1e-3 (LSE) and the V rows within
  ``ON_OFF_TOL`` (per (token, head) relative L2: other split boundaries round P to FP8 at other running maxima, the
  Helix V-up op matrix's on/off bound) of the switch-off rows; a request with no key on the rank gets the combine's
  identity; against fp32 attention x fp32 W_UV (the whole sequence on one rank) within ``FP32_OVER_OFF`` of the
  switch-off error;
* the counts and the split read captured in one CUDA graph, replayed with new queries and shorter lengths: bit for bit
  the eager switch-on call;
* above 32 rows (the op's own split read and reduction): a request split one way written from the split kernel's
  epilogue (``var_split_direct``) is bit for bit the one-split partial + reduction of the same counts (output latent,
  V rows, statistics of every row with keys on the rank).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused V up-projection runs on SM100/SM103 (tcgen05)",
)

ON_OFF_TOL = 8e-2
FP32_OVER_OFF = 1.25
CASES = {
    # (global lengths, q_len, rank)
    "bs1_8rows": ([153600 + 8], 8, 5),
    "bs2_16rows": ([70000 + 8, 9000], 8, 3),
    "bs4_32rows_agentx": ([74656, 22480, 11696, 4936], 8, 7),  # p95 / p60 / p25 / p5, / 8
    "bs8_64rows_agentx": ([66200, 32936, 25960, 21088, 16720, 13176, 9800, 5400], 8, 2),
    "bs4_short": ([60000, 50, 300, 20000], 8, 7),  # 50: below one K tile; 300: no page on rank 7
    "bs4_equal": ([20000] * 4, 8, 4),
    "bs8_equal": ([12000] * 8, 8, 1),
}


def _vu():
    import test_modeling_v2_kimi_k3_mla_v_up as vu

    return vu


def _setup(case, cp=None, seed=11):
    vu = _vu()
    dcp, mla_v_up, _ = vu._mods()
    device = torch.device("cuda")
    seq_lens, q_len, rank = case
    pool, block_rows, q = vu._cache(seq_lens, q_len, seed, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    plan = dcp.make_split_read_plan(
        block_rows, lens, q_len, vu.PAGE, vu.CP if cp is None else cp, rank if cp is None else 0
    )
    return vu, dcp, mla_v_up, pool, block_rows, lens, q, plan


class _Run:
    """``split_read_attention_v_up`` on persistent buffers; ``var`` computes the counts inside the call."""

    def __init__(self, vu, dcp, q, pool, plan, w, var):
        batch, q_len = q.shape[:2]
        self.vu, self.dcp, self.q, self.pool, self.plan, self.w, self.var = (
            vu,
            dcp,
            q,
            pool,
            plan,
            w,
            var,
        )
        self.latent = torch.empty(
            batch, q_len, vu.HEADS, vu.LATENT, dtype=torch.bfloat16, device=q.device
        )
        self.stats = torch.full((batch * q_len, vu.HEADS, 2), float("nan"), device=q.device)
        self.v = torch.full(
            (batch * q_len, vu.HEADS, vu.V), float("nan"), dtype=torch.bfloat16, device=q.device
        )
        self.ws = vu._workspace(dcp, q.device)
        self.scale = dcp.default_softmax_scale()

    def __call__(self):
        plan = self.plan
        if self.var:
            plan = self.dcp.with_var_split(plan, self.vu.HEADS, self.q.shape[1])
        self.dcp.split_read_attention_v_up(
            self.q, self.pool, plan, self.latent, self.stats, self.ws, self.scale, 1.0,
            self.dcp.MAX_REQUESTS, self.w, self.v,
        )  # fmt: skip
        return self.v, self.stats


def _ref_counts(local_lens, budget, grid):
    """The choice ``var_split_counts`` makes, in python."""
    tiles = [max(1, -(-n // 128)) for n in local_lens]
    best = None
    for t in tiles:
        for k in range(1, grid + 1):
            u = -(-t // k)
            s = [min(grid, max(1, -(-x // u))) for x in tiles]
            total = sum(s)
            largest = max(-(-x // si) for x, si in zip(tiles, s))
            if total <= budget:
                cost = largest * (budget + 1) + total
                if best is None or cost < best[0]:
                    best = (cost, s)
    return best[1]


def _keyless(plan, q_len):
    return (plan.cache_seqs == 0).repeat_interleave(q_len)


def _per_head_rel(a, b):
    a, b = a.double(), b.double()
    rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).flatten()
    return rel[torch.isfinite(rel)]


@pytest.mark.parametrize("name", list(CASES))
def test_counts(name):
    vu, dcp, _, _, _, _, q, plan = _setup(CASES[name])
    batch, q_len = q.shape[:2]
    budget, grid = dcp._var_split_envelope(batch, vu.HEADS, q_len)
    counts = dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q_len)
    assert counts is not None and dcp.var_split_grid(batch, vu.HEADS, q_len) == grid > 1
    got = counts.tolist()
    assert all(1 <= c <= grid for c in got) and sum(got) <= budget
    assert got == _ref_counts(plan.cache_seqs.tolist(), budget, grid)
    if len(set(CASES[name][0])) == 1:
        assert len(set(got)) == 1
    print(
        f"[{name}] local lengths {plan.cache_seqs.tolist()} -> splits {got} (grid {grid}, budget {budget})"
    )


@pytest.mark.parametrize("batch", range(1, 9))
def test_counts_random(batch):
    """The one-kernel selection against the python reference on random per-rank lengths (0 included: a request
    with no page on the rank), 200 batches per size."""
    vu = _vu()
    dcp = vu._mods()[0]
    budget, grid = dcp._var_split_envelope(batch, vu.HEADS, 8)
    g = torch.Generator().manual_seed(batch)
    for _ in range(200):
        lens = torch.randint(0, 40000, (batch,), generator=g)
        lens[torch.rand(batch, generator=g) < 0.1] = 0
        counts = dcp.var_split_counts(lens.to(torch.int32).cuda(), vu.HEADS, 8)
        assert counts is not None and counts.dtype == torch.int32
        assert counts.tolist() == _ref_counts(lens.tolist(), budget, grid), lens.tolist()


@pytest.mark.parametrize("name", list(CASES))
def test_off_is_split_read_then_bmm(name):
    vu, dcp, _, pool, _, _, q, plan = _setup(CASES[name])
    w, _, _ = vu._w_uv(1, q.device)
    v, stats = _Run(vu, dcp, q, pool, plan, w, False)()
    out, stats_ref = vu._plain(dcp, q, pool, plan, dcp.default_softmax_scale())
    v_ref = vu._bmm(out, w)
    torch.cuda.synchronize()
    keyed = ~_keyless(plan, q.shape[1])
    if q.shape[0] * q.shape[1] > 32:  # the unfused path: the op's own split read + BMM, bit for bit
        assert torch.equal(v[keyed].view(torch.int16), v_ref[keyed].view(torch.int16))
        assert torch.equal(stats[keyed], stats_ref[keyed])
    else:
        vu._compare(v, stats, v_ref, stats_ref, name)


@pytest.mark.parametrize("name", list(CASES))
def test_on_close_to_off(name):
    vu, dcp, mla_v_up, pool, _, _, q, plan = _setup(CASES[name])
    w, _, _ = vu._w_uv(1, q.device)
    v_off, s_off = (t.clone() for t in _Run(vu, dcp, q, pool, plan, w, False)())
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    before = set(mla_v_up._compiled) | set(CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache)
    v_on, s_on = (t.clone() for t in _Run(vu, dcp, q, pool, plan, w, True)())
    torch.cuda.synchronize()
    new = (set(mla_v_up._compiled) | set(CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache)) - before
    counts = dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q.shape[1])
    if int(counts.max()) > 1 and not any("var_split" in k for k in new):
        # a variant may already be compiled by an earlier test; then it must exist in a cache
        assert any(
            "var_split" in k
            for k in set(mla_v_up._compiled) | set(CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache)
        ), "no variable-split variant compiled"
    keyless = _keyless(plan, q.shape[1])
    if bool(keyless.any()):
        assert torch.all(s_on[keyless, :, 0] == float("-inf")) and torch.all(
            s_on[keyless, :, 1] == 0
        )
    keyed = ~keyless
    assert not torch.isnan(v_on[keyed]).any()
    lse_on, lse_off = s_on[keyed][..., 0], s_off[keyed][..., 0]
    finite = torch.isfinite(lse_off)
    assert torch.equal(finite, torch.isfinite(lse_on))
    assert float((lse_on[finite] - lse_off[finite]).abs().max()) < 1e-3
    assert torch.equal(s_on[keyed][..., 1][finite], s_off[keyed][..., 1][finite])
    rel = _per_head_rel(v_on[keyed], v_off[keyed])
    same = torch.equal(v_on[keyed].view(torch.int16), v_off[keyed].view(torch.int16))
    print(
        f"[{name}] splits {counts.tolist()}: V rows vs switch off per (token, head) median {rel.median():.2e} "
        f"max {rel.max():.2e}; bitwise {same}"
    )
    assert float(rel.max()) <= ON_OFF_TOL


@pytest.mark.parametrize(
    "seq_lens",
    [
        [9600],
        [5000, 9000],
        [12000, 3000, 700, 9000],
        [6000, 1000, 2500, 4000, 300, 9000, 800, 3000],
    ],
)
def test_on_against_fp32(seq_lens):
    """The whole sequence on one rank (cp 1): the switch-on V rows against fp32 attention x fp32 W_UV, beside the
    switch-off rows' own error."""
    vu, dcp, _, pool, block_rows, lens, q, plan = _setup((seq_lens, 8, 0), cp=1, seed=7)
    w, codes, scales = vu._w_uv(2, q.device)
    v_on, _ = (t.clone() for t in _Run(vu, dcp, q, pool, plan, w, True)())
    v_off, _ = (t.clone() for t in _Run(vu, dcp, q, pool, plan, w, False)())
    latent_ref = vu._fp32_reference(q, pool, block_rows, lens)
    rows = len(seq_lens) * 8
    w32 = codes.float() * scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    v_ref = torch.einsum(
        "rhk,hvk->rhv", latent_ref.reshape(rows, vu.HEADS, vu.LATENT).double(), w32.double()
    )

    def err(v):
        return ((v.double() - v_ref).norm(dim=-1) / v_ref.norm(dim=-1)).flatten()

    e_on, e_off = err(v_on), err(v_off)
    print(
        f"[fp32 ref {seq_lens}] splits {dcp.var_split_counts(plan.cache_seqs, vu.HEADS, 8).tolist()}: on median "
        f"{e_on.median():.2e} max {e_on.max():.2e}; off median {e_off.median():.2e} max {e_off.max():.2e}"
    )
    assert float(e_on.max()) <= FP32_OVER_OFF * float(e_off.max())
    assert float(e_on.median()) <= FP32_OVER_OFF * float(e_off.median())


@pytest.mark.parametrize("name", ["bs4_32rows_agentx", "bs8_64rows_agentx", "bs4_short"])
def test_graph_replay(name):
    """Counts and split read captured together; replayed with new queries, then with shorter lengths on the rank."""
    vu, dcp, _, pool, _, _, q, plan = _setup(CASES[name])
    w, _, _ = vu._w_uv(1, q.device)
    run = _Run(vu, dcp, q.clone(), pool, plan, w, True)
    run()  # compiles eagerly
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    seq_lens, q_len, _ = CASES[name]
    full = plan.cache_seqs.clone()
    for seed, shrink in ((21, False), (22, True)):
        _, _, q_new = vu._cache(seq_lens, q_len, seed, q.device)
        run.q.copy_(q_new)
        if shrink:  # fewer of this rank's tokens per request (the pages stay): other counts
            plan.cache_seqs.copy_(full // 3)
            plan.kv_bounds.copy_(plan.kv_bounds.clamp(max=full.repeat_interleave(q_len) // 3))
        run.v.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_stats = run.v.clone(), run.stats.clone()
        v_eager, s_eager = _Run(vu, dcp, run.q, pool, plan, w, True)()
        torch.cuda.synchronize()
        keyed = ~_keyless(plan, q_len)
        assert torch.equal(replayed[keyed].view(torch.int16), v_eager[keyed].view(torch.int16)), (
            name,
            seed,
        )
        assert torch.equal(replayed_stats[keyed], s_eager[keyed]), (name, seed)
    plan.cache_seqs.copy_(full)


@pytest.mark.parametrize("name", ["bs8_64rows_agentx", "bs8_equal"])
def test_direct_equals_one_split_reduction(name, monkeypatch):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    vu, dcp, _, pool, _, _, q, plan = _setup(CASES[name])
    w, _, _ = vu._w_uv(1, q.device)
    counts = dcp.var_split_counts(plan.cache_seqs, vu.HEADS, q.shape[1])
    assert 1 in counts.tolist() and q.shape[0] * q.shape[1] > 32
    outs = {}
    for direct in (False, True):
        monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_VAR_SPLIT_DIRECT", direct)
        run = _Run(vu, dcp, q, pool, plan, w, True)
        v, stats = run()
        torch.cuda.synchronize()
        outs[direct] = (run.latent.clone(), v.clone(), stats.clone())
    keyed = ~plan.empty.to(q.device)
    rows = q.shape[0] * q.shape[1]
    lat0, v0, s0 = outs[False]
    lat1, v1, s1 = outs[True]
    lat0, lat1 = lat0.view(rows, vu.HEADS, vu.LATENT), lat1.view(rows, vu.HEADS, vu.LATENT)
    print(
        f"[{name}] splits {counts.tolist()}: {int(keyed.sum())} of {rows} rows with keys compared"
    )
    assert torch.equal(lat1[keyed].view(torch.int16), lat0[keyed].view(torch.int16))
    assert torch.equal(v1[keyed].view(torch.int16), v0[keyed].view(torch.int16))
    assert torch.equal(s1[keyed], s0[keyed])
