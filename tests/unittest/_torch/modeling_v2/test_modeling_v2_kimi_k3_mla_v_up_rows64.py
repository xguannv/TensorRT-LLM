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
"""``TRTLLM_K3LL_MLA_V_UP_ROWS64`` (``kimi_k3_nvfp4/flags.py``): the Helix V-up split read's fused split-KV reduction +
V up-projection (``trtllm::k3_mla_decode_fp8_v_up``) also for 33-64 rows per head (bs8 at draft 7: 8 requests x 8
tokens), with at least 2 KV splits; without it those rows take the unfused split kernel and a BF16 BMM. One rank's
split read on one GPU (the helpers of ``test_modeling_v2_kimi_k3_mla_v_up``):

* up to 32 rows (8 / 16 / 32): switch on = switch off bit for bit (V rows and softmax statistics), eagerly and replayed
  from a CUDA graph with new queries;
* 64 rows, switch off: the fallback, bit for bit the split read + BMM it is (no fused variant compiled for 64 rows;
  the statistics of the rows that have keys on the rank);
* 64 rows, switch on: the fused path ran (a 64-row, >= 2-split variant compiled); a request with no key on the rank
  gets the combine's identity (statistics (-inf, 0), V rows 0); the softmax statistics and V rows of the others
  against the switch-off fallback within the V-up tolerance (per (token, head) relative L2 ``ON_OFF_TOL``, the Helix
  V-up op matrix's on/off bound), and against fp32 attention x fp32 W_UV (the whole sequence on one rank) within
  ``FP32_TOL``, beside the fallback's own error; replayed from a CUDA graph with new queries, bit for bit its eager
  call.
"""

import contextlib
import math
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
FP32_TOL = 3e-2
LE32 = {
    "bs1_mtp7": ([153600 + 8], 8, 5),
    "bs2_mtp7": ([153600 + 8, 70000 + 8], 8, 3),
    "bs4_mtp7": ([153600 + 8, 20000 + 8, 300, 5000], 8, 7),
}
BS8 = (
    [153600 + 8, 120000, 90000, 60000, 30000, 9000, 300, 153600 + 8],
    8,
    7,
)  # rank 7 holds no page of the 300


def _vu():
    import test_modeling_v2_kimi_k3_mla_v_up as vu

    return vu


@contextlib.contextmanager
def _switches(monkeypatch, rows64):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags

    with monkeypatch.context() as m:
        m.setattr(flags, "TRTLLM_K3LL_MLA_DECODE_PDL", 1)
        m.setattr(flags, "TRTLLM_K3LL_MLA_PREWAIT_L2", 0)
        m.setattr(flags, "TRTLLM_K3LL_MLA_V_UP_ROWS64", int(rows64))
        yield


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
    w, codes, scales = vu._w_uv(1, device)
    return vu, dcp, mla_v_up, pool, block_rows, lens, q, plan, w, codes, scales


def _keyless_rows(plan, q_len):
    """Rows (``batch * q_len``) of the requests that hold no key on this rank (at least one, not all)."""
    rows = (plan.cache_seqs == 0).repeat_interleave(q_len)
    assert 0 < int(rows.sum()) < rows.numel()
    return rows


def _per_head_rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).flatten()


@pytest.mark.parametrize("case", list(LE32))
def test_le32_rows_bitwise(monkeypatch, case):
    vu, dcp, _, pool, _, _, q, plan, w, _, _ = _setup(LE32[case])
    scale = dcp.default_softmax_scale()
    with _switches(monkeypatch, False):
        off = vu._VUp(dcp, q, pool, plan, w, scale)
        v_off, s_off = (t.clone() for t in off())
    with _switches(monkeypatch, True):
        on = vu._VUp(dcp, q, pool, plan, w, scale)
        v_on, s_on = (t.clone() for t in on())
        torch.cuda.synchronize()
        assert torch.equal(v_on.view(torch.int16), v_off.view(torch.int16)) and torch.equal(
            s_on, s_off
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            on()
    seq_lens, q_len, _ = LE32[case]
    for seed in (21, 22):
        _, _, q_new = vu._cache(seq_lens, q_len, seed, q.device)
        on.q.copy_(q_new)
        on.v.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        with _switches(monkeypatch, False):
            off.q.copy_(q_new)
            v_ref, s_ref = off()
            torch.cuda.synchronize()
        assert torch.equal(on.v.view(torch.int16), v_ref.view(torch.int16)) and torch.equal(
            on.stats, s_ref
        )


def test_rows64_off_is_the_fallback(monkeypatch):
    vu, dcp, mla_v_up, pool, _, _, q, plan, w, _, _ = _setup(BS8)
    scale = dcp.default_softmax_scale()
    before = set(mla_v_up._compiled)
    with _switches(monkeypatch, False):
        v, stats = vu._VUp(dcp, q, pool, plan, w, scale)()
    out, stats_ref = vu._plain(dcp, q, pool, plan, scale)
    v_ref = vu._bmm(out, w)
    torch.cuda.synchronize()
    assert not any(k[-2] == 64 for k in set(mla_v_up._compiled) - before)
    assert torch.equal(v.view(torch.int16), v_ref.view(torch.int16))
    # One split: the decode leaves the statistics of a request with no key on this rank unwritten (the exchange
    # masks it), so compare the requests that have keys here.
    keyed = ~_keyless_rows(plan, BS8[1]).to(stats.device)
    assert torch.equal(stats[keyed], stats_ref[keyed])


def test_rows64_on_close_to_off(monkeypatch):
    vu, dcp, mla_v_up, pool, _, _, q, plan, w, _, _ = _setup(BS8)
    scale = dcp.default_softmax_scale()
    with _switches(monkeypatch, False):
        v_off, s_off = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
    before = set(mla_v_up._compiled)
    with _switches(monkeypatch, True):
        on = vu._VUp(dcp, q, pool, plan, w, scale)
        v_on, s_on = (t.clone() for t in on())
    torch.cuda.synchronize()
    new = set(mla_v_up._compiled) - before
    # key = runner id + (mma_qk, mma_pv, split_kv, is_persistent, kv_bounds?, v_rows, use_pdl)
    assert any(k[-2] == 64 and k[-5] >= 2 for k in new), f"no fused 64-row variant compiled: {new}"
    assert not torch.isnan(v_on).any()
    # rows with no key on this rank: the fused reduction writes the combine's identity; the one-split fallback
    # leaves them unwritten (the exchange masks them), so the comparison below takes the keyed rows
    empty = _keyless_rows(plan, BS8[1]).to(s_on.device)
    assert torch.all(s_on[empty, :, 0] == float("-inf")) and torch.all(s_on[empty, :, 1] == 0)
    assert torch.all(v_on[empty] == 0)
    s_on, s_off, v_on_k, v_off_k = s_on[~empty], s_off[~empty], v_on[~empty], v_off[~empty]
    assert torch.equal(torch.isinf(s_on), torch.isinf(s_off))
    lse_on, lse_off = s_on[..., 0], s_off[..., 0]
    finite = torch.isfinite(lse_off)
    assert (lse_on[finite] - lse_off[finite]).abs().max().item() < 1e-3
    rel = _per_head_rel(v_on_k, v_off_k)
    rel = rel[torch.isfinite(rel)]
    print(
        f"[bs8 rows 64] fused vs fallback per (token, head): median {rel.median():.2e} max {rel.max():.2e}"
    )
    assert float(rel.max()) <= ON_OFF_TOL

    with _switches(monkeypatch, True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            on()
        seq_lens, q_len, _ = BS8
        for seed in (31, 32):
            _, _, q_new = vu._cache(seq_lens, q_len, seed, q.device)
            on.q.copy_(q_new)
            on.v.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            replayed = on.v.clone()
            v_eager, _ = vu._VUp(dcp, on.q, pool, plan, w, scale)()
            torch.cuda.synchronize()
            assert torch.equal(replayed.view(torch.int16), v_eager.view(torch.int16))


def test_rows64_against_fp32(monkeypatch):
    """The whole sequence on one rank (cp 1), 8 requests x 8 tokens: fused (switch on) and fallback (off) against
    fp32 attention x fp32 W_UV."""
    case = ([3000, 5000, 9000, 4000, 7000, 2000, 6000, 8000], 8, 0)
    vu, dcp, _, pool, block_rows, lens, q, plan, w, codes, scales = _setup(case, cp=1, seed=7)
    scale = dcp.default_softmax_scale()
    with _switches(monkeypatch, True):
        v_on, _ = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
    with _switches(monkeypatch, False):
        v_off, _ = (t.clone() for t in vu._VUp(dcp, q, pool, plan, w, scale)())
    latent_ref = vu._fp32_reference(q, pool, block_rows, lens)
    rows = len(case[0]) * case[1]
    w32 = codes.float() * scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    v_ref = torch.einsum(
        "rhk,hvk->rhv", latent_ref.reshape(rows, vu.HEADS, vu.LATENT).double(), w32.double()
    )
    err_on, err_off = _per_head_rel(v_on, v_ref), _per_head_rel(v_off, v_ref)
    print(
        f"[fp32 ref, 64 rows] fused median {err_on.median():.2e} max {err_on.max():.2e}; "
        f"fallback median {err_off.median():.2e} max {err_off.max():.2e}"
    )
    assert float(err_on.max()) < FP32_TOL
    assert math.isfinite(float(err_off.max()))
