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
"""``TRTLLM_K3LL_MLA_SPLIT_LOSSY`` (``kimi_k3_nvfp4/flags.py``): the Helix V-up split read
(``trtllm::k3_mla_decode_fp8_v_up``) with a rescale threshold (P's exponent bias lowered to fit e4m3) and / or BF16
split-KV partials. One rank's split read on one GPU at the K3 shapes (96 heads, MTP7 q_len 8, CP 16 rank-local pages).

* switch 0 compiles and runs the current kernel (no ``split_lossy`` compile key), bit for bit;
* every option (1 both, 2 threshold, 3 BF16 partials) against an fp32 reference of the same FP8 inputs: the V
  rows' relative L2 error within ``REL_SLACK`` times the current kernel's own error plus ``REL_ABS``, and the Helix
  LSE (the options change only P's FP8 rounding; the row sum stays FP32) within ``LSE_TOL``; eagerly and replayed
  from a CUDA graph (replay == eager, bit for bit);
* the softmax tail (``test_modeling_v2_kimi_k3_mla_v_up._tail_case``) is still kept with the lowered bias: the same
  0.015 bound as the current kernel;
* the census counts ``mla_split_lossy`` once per fused call with the switch on and at least
  ``decode_mla_dcp.SPLIT_LOSSY_MIN_BATCH`` requests, never below it or with the switch off;
* the exponent budget: a bias plus threshold above ``mla_decode_fp8.P_EXP_BUDGET`` is refused.
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused V-up split read runs on SM100 / SM103",
)

HEADS, Q_LEN, PAGE, CP, LATENT, ROPE, V = 96, 8, 64, 16, 512, 64, 128
REL_SLACK, REL_ABS, LSE_TOL = 1.5, 2e-3, 1e-4
MODES = {"both": 1, "threshold": 2, "bf16_partials": 3}


def _vu():
    import test_modeling_v2_kimi_k3_mla_v_up as vu

    return vu


def _mods():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        census,
        flags,
    )

    dcp, mla_v_up, _ = _vu()._mods()
    return dcp, mla_v_up, flags, census


@contextlib.contextmanager
def _switch(monkeypatch, value):
    _, _, flags, census = _mods()
    with monkeypatch.context() as m:
        m.setattr(flags, "TRTLLM_K3LL_MLA_SPLIT_LOSSY", int(value))
        m.setattr(flags, "TRTLLM_K3LL_MLA_SPLIT_PIPE", 0)
        m.setattr(census, "ENABLED", True)
        yield


def _lossy_count(census) -> int:
    return sum(v for k, v in census.counts().items() if k.startswith("mla_split_lossy:"))


def _inputs(batch, ctx, seed=7):
    """One rank's pages: every request holds ``ctx // CP`` local tokens, every query token sees all of them."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    local = ctx // CP
    nb = -(-local // PAGE)
    num_pages = batch * nb + 3
    block_rows = (
        torch.randperm(num_pages, generator=g)[: batch * nb].view(batch, nb).to(torch.int32)
    )
    pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.5).to(torch.float8_e4m3fn)
    q = (torch.randn(batch, Q_LEN, HEADS, LATENT + ROPE, generator=g) * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(HEADS, V, LATENT, generator=g) * 0.05).to(torch.bfloat16)
    dev = torch.device("cuda")
    return dict(
        pool=pool.to(dev),
        block_rows=block_rows.to(dev),
        q=q.to(dev),
        w=w.to(dev).contiguous(),
        local=local,
    )


def _plan(t):
    dcp, _, flags, _ = _mods()
    batch = t["q"].shape[0]
    seqs = torch.full((batch,), t["local"], dtype=torch.int32, device="cuda")
    bounds = torch.full((batch * Q_LEN,), t["local"], dtype=torch.int32, device="cuda")
    plan = dcp.SplitReadPlan(t["block_rows"].contiguous().t(), seqs, bounds, bounds == 0)
    if flags.TRTLLM_K3LL_MLA_VAR_SPLIT:
        plan = dcp.with_var_split(plan, HEADS, Q_LEN)
    return plan


def _reference(t):
    q = t["q"].float()
    scale = _mods()[0].default_softmax_scale()
    v_rows, lses = [], []
    for b in range(q.shape[0]):
        kv = (
            t["pool"].view(torch.uint8)[t["block_rows"][b].long()].view(torch.float8_e4m3fn).float()
        )
        kv = kv.reshape(-1, LATENT + ROPE)[: t["local"]]
        s = torch.einsum("qhd,nd->qhn", q[b], kv) * scale
        lses.append(torch.logsumexp(s, dim=-1))
        lat = torch.einsum("qhn,nd->qhd", torch.softmax(s, dim=-1), kv[:, :LATENT])
        v_rows.append(torch.einsum("qhd,hvd->qhv", lat, t["w"].float()))
    return torch.cat(v_rows), torch.cat(lses)


class _Run:
    def __init__(self, t):
        dcp, _, _, _ = _mods()
        batch = t["q"].shape[0]
        self.t, self.plan = t, _plan(t)
        self.latent = torch.empty(batch, Q_LEN, HEADS, LATENT, dtype=torch.bfloat16, device="cuda")
        self.stats = torch.full((batch * Q_LEN, HEADS, 2), float("nan"), device="cuda")
        self.v = torch.full(
            (batch * Q_LEN, HEADS, V), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        self.ws = torch.empty(
            dcp.workspace_bytes(HEADS, dcp.MAX_Q_LEN, dcp.MAX_REQUESTS),
            dtype=torch.uint8,
            device="cuda",
        )

    def __call__(self):
        dcp, _, _, _ = _mods()
        dcp.split_read_attention_v_up(
            self.t["q"], self.t["pool"], self.plan, self.latent, self.stats, self.ws,
            dcp.default_softmax_scale(), 1.0, dcp.MAX_REQUESTS, self.t["w"], self.v,
        )  # fmt: skip
        return self.v, self.stats


def _errors(v, stats, v_ref, lse_ref):
    rel = ((v.float() - v_ref).norm() / v_ref.norm()).item()
    lse = (stats[..., 0] - lse_ref).abs().max().item()
    return rel, lse


CASES = {"bs4_150k": (4, 153600), "bs4_500k": (4, 512000), "bs3_64k": (3, 65536)}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mode", list(MODES))
def test_option_against_fp32(monkeypatch, case, mode):
    batch, ctx = CASES[case]
    t = _inputs(batch, ctx)
    v_ref, lse_ref = _reference(t)
    _, mla_v_up, _, census = _mods()
    with _switch(monkeypatch, 0):
        before = set(mla_v_up._compiled)
        v_off, s_off = (x.clone() for x in _Run(t)())
        assert not any("split_lossy" in k for k in set(mla_v_up._compiled) - before)
    with _switch(monkeypatch, MODES[mode]):
        count = _lossy_count(census)
        run = _Run(t)
        v_on, s_on = (x.clone() for x in run())
        assert _lossy_count(census) == count + 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    torch.cuda.synchronize()
    rel_off, lse_off = _errors(v_off, s_off, v_ref, lse_ref)
    rel_on, lse_on = _errors(v_on, s_on, v_ref, lse_ref)
    print(
        f"[{case} {mode}] V rel L2 vs fp32: off {rel_off:.3e} on {rel_on:.3e}; LSE max abs off {lse_off:.2e} "
        f"on {lse_on:.2e}"
    )
    assert not torch.isnan(v_on).any()
    assert rel_on <= REL_SLACK * rel_off + REL_ABS, (
        f"{case} {mode}: V rel L2 {rel_on:.3e} (off {rel_off:.3e})"
    )
    assert lse_on <= LSE_TOL, f"{case} {mode}: LSE off fp32 by {lse_on:.3e}"
    run.v.fill_(float("nan"))
    run.stats.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(run.v, v_on) and torch.equal(run.stats, s_on), (
        f"{case} {mode}: replay differs"
    )


def test_below_min_batch_and_off_unchanged(monkeypatch):
    """Fewer than SPLIT_LOSSY_MIN_BATCH requests: the switch at 1 runs the current kernel (bitwise, not counted)."""
    dcp, _, _, census = _mods()
    t = _inputs(dcp.SPLIT_LOSSY_MIN_BATCH - 1, 153600)
    with _switch(monkeypatch, 0):
        v_off, s_off = (x.clone() for x in _Run(t)())
    with _switch(monkeypatch, 1):
        count = _lossy_count(census)
        v_on, s_on = (x.clone() for x in _Run(t)())
        assert _lossy_count(census) == count
    torch.cuda.synchronize()
    assert torch.equal(v_on, v_off) and torch.equal(s_on, s_off)


def test_softmax_tail_kept(monkeypatch):
    """The lowered bias still keeps oak's softmax tail (the current kernel's bound)."""
    vu = _vu()
    dcp, _, _, _ = _mods()
    monkeypatch.setattr(dcp, "SPLIT_LOSSY_MIN_BATCH", 1)
    for mode in MODES.values():
        with _switch(monkeypatch, mode):
            err, _ = vu._tail_error(torch.device("cuda"))
        print(f"[tail] mode {mode}: V rows vs fp32 {err:.4f}")
        assert err < 0.015, f"mode {mode}: the softmax tail is lost ({err:.4f})"


def test_exponent_budget():
    import cutlass

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla import mla_decode_fp8

    def make(threshold, bias):
        return mla_decode_fp8.BlackwellMultiHeadLatentAttentionForwardFP8(
            cutlass.Float32, cutlass.Float32, (128, 128), (128, 256), 74, PAGE, threshold, False, True,
            False, num_heads=HEADS, seq_len_q=Q_LEN, p_exp2_bias=bias,
        )  # fmt: skip

    _, mla_v_up, _, _ = _mods()
    t = mla_v_up.SPLIT_LOSSY_THRESHOLD
    assert make(t, mla_decode_fp8.P_EXP_BUDGET - t).p_exp2_bias + t == mla_decode_fp8.P_EXP_BUDGET
    with pytest.raises(ValueError):
        make(t, mla_decode_fp8.P_EXP2_BIAS)
    assert make(0.0, None).p_exp2_bias == mla_decode_fp8.P_EXP2_BIAS
