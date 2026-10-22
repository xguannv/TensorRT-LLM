# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix MLA decode with the V up-projection in the split-KV reduction (``kimi_k3_nvfp4/mla_v_up.py``,
``TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A``), one rank's split read on one GPU.

* ``trtllm::k3_mla_decode_fp8_v_up`` against the split read it replaces (``split_read_attention``, the same op and
  tactic) followed by a BF16 BMM of its latent rows with the same ``W_UV``: softmax statistics bit for bit, the V rows
  within one bf16 ulp of their magnitude (the fused reduction's MMA and the BMM sum the 512 products in their own
  orders); 8 / 16 / 32 rows per head (bs 1 / 2 / 4 at MTP7) take the fused reduction, 35 rows the fallback; requests
  this rank holds no page of; eagerly and replayed from a CUDA graph with new inputs.
* against fp32 attention and an fp32 ``W_UV``, beside the current path's FP8 V-B (``k3_fp8_mla_output``).
* the softmax tail below e4m3's normal range (the P @ V bias of ``mla_decode_fp8.py``): kept on this path; with the
  bias removed (the kernel before the fix, recompiled) the same case fails, so the check can see the bug.
"""

import math
import os
import sys

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103),
    reason="the fused V up-projection runs on SM100/SM103 (tcgen05)",
)

HEADS, LATENT, ROPE, PAGE, V = 96, 512, 64, 64, 128
CP = 16
ULP = 2.0**-7  # one bf16 ulp at a value's magnitude, relative (with margin)


def _mods():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        attention,
        decode_mla_dcp,
        mla_v_up,
    )

    return decode_mla_dcp, mla_v_up, attention


def _cache(seq_lens, q_len, seed, device):
    """A shuffled FP8 page pool holding each request's pages, the block rows, FP8 queries of every head."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = [max(1, math.ceil(n / PAGE)) for n in seq_lens]
    num_pages = sum(blocks) + 5
    perm = torch.randperm(num_pages, generator=g)
    block_rows = torch.zeros(len(seq_lens), max(blocks), dtype=torch.int32)
    k = 0
    for b, nb in enumerate(blocks):
        block_rows[b, :nb] = perm[k : k + nb].to(torch.int32)
        k += nb
    pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.5).to(device)
    q = (torch.randn(len(seq_lens), q_len, HEADS, LATENT + ROPE, generator=g) * 0.5).to(device)
    return pool.to(torch.float8_e4m3fn), block_rows.to(device), q.to(torch.float8_e4m3fn)


def _w_uv(seed, device):
    """BF16 W_UV as the target builds it: FP8 V-B codes times 128 x 128 block scales (and those codes / scales)."""
    _, mla_v_up, _ = _mods()
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = (
        (torch.randn(HEADS, V, LATENT, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    scales = torch.rand(HEADS, 1, LATENT // 128, generator=g) * 2e-3 + 5e-4
    codes, scales = codes.to(device), scales.to(device)
    return mla_v_up.dequantize_v_b(codes, scales), codes, scales


def _workspace(dcp, device):
    return torch.empty(
        dcp.workspace_bytes(HEADS, dcp.MAX_Q_LEN, dcp.MAX_REQUESTS),
        dtype=torch.uint8,
        device=device,
    )


def _plain(dcp, q, pool, plan, scale):
    batch, q_len = q.shape[:2]
    out = torch.empty(batch, q_len, HEADS, LATENT, dtype=torch.bfloat16, device=q.device)
    stats = torch.empty(batch * q_len, HEADS, 2, dtype=torch.float32, device=q.device)
    dcp.split_read_attention(
        q, pool, plan, out, stats, _workspace(dcp, q.device), scale, 1.0, dcp.MAX_REQUESTS
    )
    return out, stats


def _bmm(out, w):
    rows = out.shape[0] * out.shape[1]
    v = torch.empty(rows, HEADS, V, dtype=torch.bfloat16, device=out.device)
    torch.ops.trtllm.bmm_out(
        out.view(rows, HEADS, LATENT).transpose(0, 1), w.transpose(1, 2), v.transpose(0, 1)
    )
    return v


class _VUp:
    """One rank's ``split_read_attention_v_up`` on persistent buffers (capturable)."""

    def __init__(self, dcp, q, pool, plan, w, scale):
        batch, q_len = q.shape[:2]
        self.dcp, self.q, self.pool, self.plan, self.w, self.scale = dcp, q, pool, plan, w, scale
        self.latent = torch.empty(
            batch, q_len, HEADS, LATENT, dtype=torch.bfloat16, device=q.device
        )
        self.stats = torch.full((batch * q_len, HEADS, 2), float("nan"), device=q.device)
        self.v = torch.full(
            (batch * q_len, HEADS, V), float("nan"), dtype=torch.bfloat16, device=q.device
        )
        self.ws = _workspace(dcp, q.device)

    def __call__(self):
        self.dcp.split_read_attention_v_up(
            self.q, self.pool, self.plan, self.latent, self.stats, self.ws, self.scale, 1.0,
            self.dcp.MAX_REQUESTS, self.w, self.v,
        )  # fmt: skip
        return self.v, self.stats


def _compare(v, stats, v_ref, stats_ref, what):
    assert torch.equal(torch.isnan(stats), torch.isnan(stats_ref)), what
    assert torch.equal(torch.nan_to_num(stats), torch.nan_to_num(stats_ref)), (
        f"{what}: softmax statistics differ from the split read's"
    )
    assert not torch.isnan(v).any(), f"{what}: NaN in the V rows"
    scale = v_ref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    rel = ((v.float() - v_ref.float()).abs() / scale).max().item()
    assert rel <= ULP, f"{what}: V rows off the split read + BMM by {rel:.3e} of their magnitude"
    return torch.equal(v, v_ref)


CASES = {
    # (global lengths, q_len, rank): rows per head = batch * q_len
    "bs1_mtp7": ([153600 + 8], 8, 5),  # 8 rows
    "bs2_mtp7": ([153600 + 8, 70000 + 8], 8, 3),  # 16 rows
    "bs4_mtp7": (
        [153600 + 8, 20000 + 8, 300, 5000],
        8,
        7,
    ),  # 32 rows; rank 7 holds no page of the 300
    "bs1_q1": ([9000], 1, 0),  # 1 row
    "bs5_fallback": ([3000, 4000, 5000, 6000, 7000], 7, 2),  # 35 rows: decode + BMM
}


@pytest.mark.parametrize("case", list(CASES))
def test_v_up_equals_split_read_then_bmm(case):
    dcp, mla_v_up, _ = _mods()
    device = torch.device("cuda")
    seq_lens, q_len, rank = CASES[case]
    pool, block_rows, q = _cache(seq_lens, q_len, 11, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    plan = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, CP, rank)
    scale = dcp.default_softmax_scale()
    w, _, _ = _w_uv(1, device)

    out, stats_ref = _plain(dcp, q, pool, plan, scale)
    v_ref = _bmm(out, w)
    run = _VUp(dcp, q, pool, plan, w, scale)
    v, stats = run()
    torch.cuda.synchronize()
    bitwise = _compare(v, stats, v_ref, stats_ref, f"{case} eager")
    if 0 in plan.cache_seqs.tolist():
        b = plan.cache_seqs.tolist().index(0)
        rows = slice(b * q_len, (b + 1) * q_len)
        assert torch.all(stats[rows, :, 0] == float("-inf")) and torch.all(stats[rows, :, 1] == 0)
        assert torch.all(v[rows] == 0)
    print(f"[{case}] rows={len(seq_lens) * q_len} V bitwise equal to split read + BMM: {bitwise}")

    # captured, replayed with new queries
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (21, 22):
        _, _, q_new = _cache(seq_lens, q_len, seed, device)
        run.q.copy_(q_new)
        run.v.fill_(float("nan"))
        run.stats.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        out, stats_ref = _plain(dcp, run.q, pool, plan, scale)
        _compare(run.v, run.stats, _bmm(out, w), stats_ref, f"{case} replay {seed}")


def _fp32_reference(q, pool, block_rows, seq_lens):
    from test_modeling_v2_kimi_k3_mla_dcp import _torch_reference

    dcp, _, _ = _mods()
    return _torch_reference(q, pool, block_rows, seq_lens, dcp.default_softmax_scale())


@pytest.mark.parametrize("seq_lens,q_len", [([9600], 8), ([5000, 9000], 8), ([3000] * 4, 8)])
def test_v_up_against_fp32(seq_lens, q_len):
    """The whole sequence on one rank (Helix form, cp 1): the V rows against fp32 attention x fp32 W_UV; and the
    current path's FP8 V-B (``k3_fp8_mla_output`` with a gate of 1) on the plain split read's latent, for the record."""
    dcp, _, attention = _mods()
    device = torch.device("cuda")
    pool, block_rows, q = _cache(seq_lens, q_len, 7, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    plan = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, 1, 0)
    scale = dcp.default_softmax_scale()
    w, codes, scales = _w_uv(2, device)
    v, _ = _VUp(dcp, q, pool, plan, w, scale)()
    latent_ref = _fp32_reference(q, pool, block_rows, lens)  # [B, T, H, 512]
    rows = len(seq_lens) * q_len
    w32 = codes.float() * scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    v_ref = torch.einsum(
        "rhk,hvk->rhv", latent_ref.reshape(rows, HEADS, LATENT).double(), w32.double()
    )
    err_up = ((v.double() - v_ref).norm(dim=-1) / v_ref.norm(dim=-1)).flatten()

    out, _ = _plain(dcp, q, pool, plan, scale)
    gate = torch.full(
        (rows, HEADS * V), 30.0, dtype=torch.bfloat16, device=device
    )  # sigmoid = 1.0 in fp32
    vb = attention.mla_output(out.view(rows, HEADS * LATENT), codes, scales, gate, 0)
    err_vb = (
        (vb.view(rows, HEADS, V).double() - v_ref).norm(dim=-1) / v_ref.norm(dim=-1)
    ).flatten()
    print(
        f"[fp32 ref {seq_lens}] per (token, head) relative error: BF16 V-up median {err_up.median():.2e} max "
        f"{err_up.max():.2e}; FP8 V-B median {err_vb.median():.2e} max {err_vb.max():.2e}"
    )
    assert float(err_up.max()) < 3e-2, f"V-up off fp32 by up to {float(err_up.max()):.3e}"


def _tail_case(device):
    """oak's tail case (``test_softmax_tail_below_fp8_normal_range_is_kept``): a sink every 256 positions of a 9600-
    token request, the other keys at 2^-10.2 of a sink's probability, value rows that differ between the two."""
    n, q_len, every = 9600, 8, 256
    g = torch.Generator(device="cpu").manual_seed(5)
    blocks = math.ceil(n / PAGE)
    num_pages = blocks + 3
    block_rows = torch.randperm(num_pages, generator=g)[:blocks].view(1, -1).to(torch.int32)
    pool = torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.05
    pool[..., 0] = 1.875
    pool[..., 1] = 1.0
    for pos in range(0, n, every):
        page, row = int(block_rows[0, pos // PAGE]), pos % PAGE
        pool[page, row, 0] = 8.0
        pool[page, row, 1] = -1.0
    pool = pool.to(device).to(torch.float8_e4m3fn)
    block_rows = block_rows.to(device)
    q = torch.zeros(1, q_len, HEADS, LATENT + ROPE)
    q[..., 0] = 16.0
    return pool, block_rows, q.to(device).to(torch.float8_e4m3fn), n, q_len


def _tail_error(device):
    dcp, _, _ = _mods()
    pool, block_rows, q, n, q_len = _tail_case(device)
    lens = torch.tensor([n], dtype=torch.int32, device=device)
    plan = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, 1, 0)
    w, codes, scales = _w_uv(3, device)
    v, stats = _VUp(dcp, q, pool, plan, w, dcp.default_softmax_scale())()
    latent_ref = _fp32_reference(q, pool, block_rows, lens)
    w32 = codes.float() * scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    v_ref = torch.einsum("rhk,hvk->rhv", latent_ref.reshape(q_len, HEADS, LATENT), w32)
    err = (v.float() - v_ref).norm(dim=-1) / v_ref.norm(dim=-1)
    return float(err.max()), stats


def test_softmax_tail_below_fp8_normal_range_is_kept_with_v_up(monkeypatch):
    """The V-up path's split kernel keeps the softmax tail (P @ V on probabilities biased by 2^P_EXP2_BIAS): ~0.2 %
    per (token, head). Negative control: the same kernel recompiled with the bias at 0 (the pre-fix arithmetic) loses
    the tail's share and fails the same threshold."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla import mla_decode_fp8

    _, mla_v_up, _ = _mods()
    device = torch.device("cuda")
    assert mla_decode_fp8.P_EXP2_BIAS == 8.0
    err, stats = _tail_error(device)
    assert err < 0.015, f"per-(token, head) relative error of the V rows up to {err:.3f}"
    saved = dict(mla_v_up._compiled)
    try:
        mla_v_up._compiled.clear()
        monkeypatch.setattr(mla_decode_fp8, "P_EXP2_BIAS", 0.0)
        err_unbiased, _ = _tail_error(device)
    finally:
        monkeypatch.undo()
        mla_v_up._compiled.clear()
        mla_v_up._compiled.update(saved)
    print(
        f"[tail] V rows vs fp32: {err:.4f} (biased P), {err_unbiased:.4f} (bias 0, the pre-fix kernel)"
    )
    assert err_unbiased > 0.03, (
        f"the check does not see the tail bug: {err_unbiased:.3f} with the bias removed"
    )
