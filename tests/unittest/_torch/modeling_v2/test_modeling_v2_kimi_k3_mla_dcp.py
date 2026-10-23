# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-read DCP for K3 MLA decode: per-rank partials over round-robin pages, combined, equal full attention.

All ranks are emulated on one GPU: the "all-to-all" is a stack of the per-rank outputs.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="CuTe DSL MLA decode runs on SM100/SM103/SM107 only",
)

HEADS, LATENT, ROPE, PAGE = 96, 512, 64, 64


def _dcp():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_dcp,
    )

    return decode_mla_dcp


def _make_cache(seq_lens, q_len, device, seed=3):
    """A shuffled page pool holding each request's pages, and FP8 queries for all heads."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = [math.ceil(n / PAGE) for n in seq_lens]
    max_blocks = max(blocks)
    num_pages = sum(blocks) + 5
    perm = torch.randperm(num_pages, generator=g)
    block_rows = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    k = 0
    for b, nb in enumerate(blocks):
        block_rows[b, :nb] = perm[k : k + nb].to(torch.int32)
        k += nb
    pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.5).to(device)
    pool = pool.to(torch.float8_e4m3fn)
    q = (torch.randn(len(seq_lens), q_len, HEADS, LATENT + ROPE, generator=g) * 0.5).to(device)
    q = q.to(torch.float8_e4m3fn)
    return pool, block_rows.to(device), q


def _run(dcp, q, pool, plan, max_batch, scale):
    batch, q_len = q.shape[:2]
    out = torch.empty(batch, q_len, HEADS, LATENT, dtype=torch.bfloat16, device=q.device)
    stats = torch.empty(batch * q_len, HEADS, 2, dtype=torch.float32, device=q.device)
    ws = torch.empty(
        dcp.workspace_bytes(HEADS, q_len, max_batch), dtype=torch.uint8, device=q.device
    )
    dcp.split_read_attention(q, pool, plan, out, stats, ws, scale, 1.0, max_batch)
    return out, stats


def _torch_reference(q, pool, block_rows, seq_lens, scale):
    """FP32 causal MLA over the global sequence: token t of a request sees positions < L - q_len + 1 + t."""
    batch, q_len = q.shape[:2]
    outs = []
    for b in range(batch):
        n = int(seq_lens[b])
        pages = block_rows[b, : math.ceil(n / PAGE)].long()
        kv = pool[pages].float().reshape(-1, LATENT + ROPE)[:n]
        qb = q[b].float()  # [T, H, 576]
        scores = torch.einsum("qhd,nd->qhn", qb, kv) * scale
        pos = torch.arange(n, device=q.device)
        limit = n - q_len + 1 + torch.arange(q_len, device=q.device)
        scores = scores.masked_fill(pos.view(1, 1, -1) >= limit.view(-1, 1, 1), float("-inf"))
        outs.append(torch.softmax(scores, -1) @ kv[:, :LATENT])
    return torch.stack(outs)  # [B, T, H, 512]


@pytest.mark.parametrize(
    "seq_lens,q_len",
    [
        ([3 * 16 * PAGE + 37], 7),  # three full ledger blocks plus a partial page
        ([16 * PAGE * 2], 8),  # ends exactly on a ledger block
        ([PAGE * 5 + 3, 16 * PAGE + 70, 9000], 7),  # short request: most ranks have nothing
        ([20000, 300], 1),
        ([PAGE * 16 + 2], 7),  # the q_len tokens straddle a page boundary across two ranks
    ],
)
@pytest.mark.parametrize("cp", [16])
def test_split_read_equals_full_attention(seq_lens, q_len, cp):
    dcp = _dcp()
    device = torch.device("cuda")
    pool, block_rows, q = _make_cache(seq_lens, q_len, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    batch = len(seq_lens)
    scale = dcp.default_softmax_scale()

    full_plan = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, 1, 0)
    ref_out, _ = _run(dcp, q, pool, full_plan, batch, scale)

    outs, stats = [], []
    for r in range(cp):
        plan = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, cp, r)
        assert int(plan.cache_seqs.sum()) >= 0
        o, s = _run(dcp, q, pool, plan, batch, scale)
        dcp.sanitize_empty(o, s, plan.empty)
        outs.append(o.view(batch * q_len, HEADS * LATENT))
        stats.append(s)
    # every token's keys are spread over the ranks exactly once
    total = sum(
        dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, cp, r).kv_bounds for r in range(cp)
    )
    expect = (lens.view(-1, 1) - q_len + 1 + torch.arange(q_len, device=device)).reshape(-1)
    torch.testing.assert_close(total, expect.to(total.dtype))

    combined = dcp.combine(torch.stack(outs), torch.stack(stats)).view(batch, q_len, HEADS, LATENT)
    torch.testing.assert_close(combined.float(), ref_out.float(), rtol=2e-2, atol=2e-2)

    if max(seq_lens) <= 20000:
        torch_ref = _torch_reference(q, pool, block_rows, lens, scale)
        torch.testing.assert_close(combined.float(), torch_ref, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(ref_out.float(), torch_ref, rtol=5e-2, atol=5e-2)


def _helix_shard(pool, block_rows, seq_lens, cp, rank, device, seed):
    """Rank ``rank``'s Helix storage: a fresh pool holding only its round-robin pages (global page ``j * cp + rank``
    of a request is its ``j``-th local page) at new page ids, every other page garbage, and its local block table."""
    g = torch.Generator(device="cpu").manual_seed(seed * 100 + rank)
    local_counts = [len(range(rank, math.ceil(n / PAGE), cp)) for n in seq_lens]
    num_local = sum(local_counts)
    num_pages = num_local + 7
    perm = torch.randperm(num_pages, generator=g)
    local_pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 50).to(device)
    local_pool = local_pool.to(torch.float8_e4m3fn)
    width = max(1, max(local_counts))
    local_rows = torch.zeros(len(seq_lens), width, dtype=torch.int32)
    k = 0
    for b, count in enumerate(local_counts):
        for j in range(count):
            page = int(perm[k])
            local_rows[b, j] = page
            local_pool[page] = pool[int(block_rows[b, j * cp + rank])]
            k += 1
    return local_pool, local_rows.to(device)


@pytest.mark.parametrize(
    "seq_lens,q_len",
    [
        ([3 * 16 * PAGE + 37], 7),
        ([16 * PAGE * 2], 8),
        (
            [PAGE * 5 + 3, 16 * PAGE + 70, 9000],
            7,
        ),  # short request: most ranks hold none of its pages
        ([20000, 300], 1),
        ([PAGE * 16 + 2], 8),  # the q_len tokens straddle a page boundary across two ranks
        ([153600, 153600 + 9], 8),
    ],
)
def test_helix_sharded_storage_equals_full_attention(seq_lens, q_len):
    """Helix: each rank stores only its pages (KV manager layout), reads its local table / local lengths, bounds
    from the tokens' global positions; the combined partials equal full attention over the whole sequence."""
    dcp = _dcp()
    from tensorrt_llm._torch.utils import helix_local_len_tensor

    cp = 16
    device = torch.device("cuda")
    pool, block_rows, q = _make_cache(seq_lens, q_len, device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    batch = len(seq_lens)
    scale = dcp.default_softmax_scale()
    ref_out, _ = _run(
        dcp, q, pool, dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, 1, 0), batch, scale
    )

    positions = (lens.view(-1, 1) - q_len + torch.arange(q_len, device=device)).reshape(-1)
    outs, stats, bounds = [], [], []
    for r in range(cp):
        local_pool, local_rows = _helix_shard(pool, block_rows, seq_lens, cp, r, device, seed=5)
        local_lens = helix_local_len_tensor(lens, PAGE, cp, r)
        plan = dcp.make_helix_plan(local_rows, local_lens, positions, PAGE, cp, r)
        # the same bounds the full-table split read derives
        twin = dcp.make_split_read_plan(block_rows, lens, q_len, PAGE, cp, r)
        torch.testing.assert_close(plan.kv_bounds, twin.kv_bounds)
        torch.testing.assert_close(plan.cache_seqs, twin.cache_seqs)
        o, s = _run(dcp, q, local_pool, plan, batch, scale)
        dcp.sanitize_empty(o, s, plan.empty)
        outs.append(o.view(batch * q_len, HEADS * LATENT))
        stats.append(s)
        bounds.append(plan.kv_bounds)
    torch.testing.assert_close(sum(bounds), (positions + 1).to(torch.int32))

    combined = dcp.combine(torch.stack(outs), torch.stack(stats)).view(batch, q_len, HEADS, LATENT)
    torch.testing.assert_close(combined.float(), ref_out.float(), rtol=2e-2, atol=2e-2)
    if max(seq_lens) <= 20000:
        torch_ref = _torch_reference(q, pool, block_rows, lens, scale)
        torch.testing.assert_close(combined.float(), torch_ref, rtol=5e-2, atol=5e-2)


def test_helix_step_plan_reads_the_helix_metadata():
    """Under a Helix mapping the split read's group is the CP group, the query all-gather runs on the TP mapping
    over the same ranks, and the step plan reads the local table as is with bounds from helix_position_offsets."""
    import types

    dcp = _dcp()
    device = torch.device("cuda")
    cp, rank, q_len = 16, 3, 8
    helix = types.SimpleNamespace(
        has_cp_helix=lambda: True, cp_size=cp, cp_rank=rank, cp_group=list(range(cp)), tp_rank=0
    )
    gather = types.SimpleNamespace(tp_size=cp, tp_rank=rank)
    split = dcp.MlaSplitRead(helix, 6, mode=1, gather_mapping=gather)
    assert (split.helix, split.cp, split.rank, split.heads) == (True, cp, rank, 96)
    assert split.gather_mapping is gather
    with pytest.raises(ValueError, match="TP rank is the CP rank"):
        dcp.MlaSplitRead(
            helix, 6, mode=1, gather_mapping=types.SimpleNamespace(tp_size=cp, tp_rank=0)
        )
    split.register(7)
    lens = torch.tensor([153600 + 8, 70000 + 8], dtype=torch.int32, device=device)
    positions = (
        (lens.view(-1, 1) - q_len + torch.arange(q_len, device=device)).reshape(-1).to(torch.int32)
    )
    meta = types.SimpleNamespace(
        helix_position_offsets=torch.cat([positions, positions.new_full((5,), -7)])
    )
    local_rows = torch.arange(2 * 200, dtype=torch.int32, device=device).view(2, 200)
    from tensorrt_llm._torch.utils import helix_local_len_tensor

    local_lens = helix_local_len_tensor(lens, PAGE, cp, rank)
    plan = split._step_plan(7, local_rows, local_lens, q_len, meta)
    torch.testing.assert_close(plan.page_table, local_rows.t())
    torch.testing.assert_close(plan.cache_seqs, local_lens)
    torch.testing.assert_close(
        plan.kv_bounds, helix_local_len_tensor(positions + 1, PAGE, cp, rank)
    )


@pytest.mark.parametrize("switch", [0, 1])
def test_var_split_counts_only_for_the_v_up_split_read(monkeypatch, switch):
    """TRTLLM_K3LL_MLA_VAR_SPLIT: only the Helix V-up split read takes per-request split counts. Steps whose split
    reads are the plain one (no W_UV: SM107, or TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A off) compute none and read the
    static plan; a step of V-up split reads computes them once, at its first MLA layer, and every layer reads those
    counts; the next step computes them again. With the switch off no step computes any."""
    import types

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags
    from tensorrt_llm._torch.utils import helix_local_len_tensor

    dcp = _dcp()
    device = torch.device("cuda")
    cp, rank, q_len, layers = 16, 3, 8, (7, 9)
    helix = types.SimpleNamespace(
        has_cp_helix=lambda: True, cp_size=cp, cp_rank=rank, cp_group=list(range(cp)), tp_rank=0
    )
    split = dcp.MlaSplitRead(
        helix, 6, mode=1, gather_mapping=types.SimpleNamespace(tp_size=cp, tp_rank=rank)
    )
    for layer in layers:
        split.register(layer)
    lens = torch.tensor([153600 + q_len, 70000 + q_len], dtype=torch.int32, device=device)
    positions = (
        (lens.view(-1, 1) - q_len + torch.arange(q_len, device=device)).reshape(-1).to(torch.int32)
    )
    meta = types.SimpleNamespace(num_generations=2, helix_position_offsets=positions)
    local_rows = torch.arange(2 * 200, dtype=torch.int32, device=device).view(2, 200)
    local_lens = helix_local_len_tensor(lens, PAGE, cp, rank)

    monkeypatch.setattr(flags, "TRTLLM_K3LL_MLA_VAR_SPLIT", switch)
    monkeypatch.setattr(flags, "TRTLLM_K3LL_HELIX_A2A_FUSED", 1)
    counted = []
    real_counts = dcp.var_split_counts
    monkeypatch.setattr(
        dcp, "var_split_counts", lambda *a, **k: counted.append(1) or real_counts(*a, **k)
    )
    plans = []
    monkeypatch.setattr(dcp, "split_read_attention", lambda q, pages, plan, *a: plans.append(plan))
    monkeypatch.setattr(dcp, "exchange_combine_fused", lambda out, *a: out)
    split._a2a_fused = object()
    split._forward_v_up = lambda q_all, pages, plan, *a: plans.append(plan)
    split._layer_inputs = lambda mqa, meta, n: (None, local_rows, 0, local_lens)
    split._layer_scales = lambda *a: (1.0, 1.0)
    quant_q = torch.zeros(2 * q_len, HEADS, LATENT + ROPE, dtype=torch.uint8, device=device)
    w_uv = torch.zeros(1, device=device)
    gate = torch.zeros(1, device=device)

    def step(v_up: bool):
        plans.clear()
        for layer in layers:
            split.forward(
                types.SimpleNamespace(layer_idx=layer), quant_q, meta, q_len, None, None,
                all_heads=True, v_up_weight=w_uv if v_up else None, v_up_gate=gate if v_up else None,
            )  # fmt: skip
        assert len(plans) == len(layers)
        return list(plans)

    for _ in range(2):
        assert all(p.block_splits is None for p in step(v_up=False))
    assert counted == []
    first = step(v_up=True)
    if not switch:
        assert counted == [] and all(p.block_splits is None for p in first)
        return
    assert len(counted) == 1
    assert first[0].block_splits is not None and first[0].var_split > 1
    assert all(p.block_splits is first[0].block_splits for p in first)
    step(v_up=True)
    assert len(counted) == 2
    assert all(p.block_splits is None for p in step(v_up=False)) and len(counted) == 2


@pytest.mark.parametrize(
    "seq_lens,q_len,heads", [([5000, 300], 7, 6), ([PAGE * 16 + 2], 8, 6), ([9000], 1, 96)]
)
def test_direct_local_heads_without_stats(seq_lens, q_len, heads):
    """TRTLLM_K3LL_MLA_DCP=1: this rank's heads over the full KV, no softmax stats, equals the fp32 reference."""
    global HEADS
    dcp = _dcp()
    device = torch.device("cuda")
    saved, HEADS = HEADS, heads
    try:
        pool, block_rows, q = _make_cache(seq_lens, q_len, device)
        lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch = len(seq_lens)
        scale = dcp.default_softmax_scale()
        # mode 1's operands: the stock [blocks, R] view of the block rows, global lengths, the kernel's own causal mask
        plan = dcp.SplitReadPlan(block_rows.t(), lens, None, None)
        out = torch.empty(batch, q_len, heads, LATENT, dtype=torch.bfloat16, device=device)
        ws = torch.empty(dcp.workspace_bytes(heads, q_len, batch), dtype=torch.uint8, device=device)
        dcp.split_read_attention(q, pool, plan, out, None, ws, scale, 1.0, batch)
        torch_ref = _torch_reference(q, pool, block_rows, lens, scale)
        torch.testing.assert_close(out.float(), torch_ref, rtol=5e-2, atol=5e-2)
    finally:
        HEADS = saved


@pytest.mark.parametrize("heads,helix_form", [(6, False), (96, True)])
def test_softmax_tail_below_fp8_normal_range_is_kept(heads, helix_form):
    """A sink key every 256 positions of a 9600-token request and keys whose probabilities sit below e4m3's flush
    point relative to a sink (2^-10.2), with value rows that differ between the two (latent dim 1: -1 on the sinks,
    +1 on the tail). The kernel's P @ V runs on FP8 probabilities relative to the running row max; the tail must
    count there and not only in the fp32 row sum, otherwise the output loses the tail's share. Each KV split
    normalizes by its own max, so only splits that see a sink lose their tail; with this spacing the error was ~4.4 %
    per (token, head) (the fix: ~0.2 %). Checked in both call forms the staircase uses (mode 1: own heads, no stats,
    built-in causal mask; Helix: stats + per-token bounds), and the Helix form's exported log-sum-exp."""
    global HEADS
    dcp = _dcp()
    device = torch.device("cuda")
    n, q_len, every = 9600, 8, 256
    saved, HEADS = HEADS, heads
    try:
        g = torch.Generator(device="cpu").manual_seed(5)
        blocks = math.ceil(n / PAGE)
        num_pages = blocks + 3
        block_rows = torch.randperm(num_pages, generator=g)[:blocks].view(1, -1).to(torch.int32)
        pool = torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.05
        # logits come from latent dim 0 only (q is zero elsewhere): 16 * 1.875 on the tail, 16 * 8 on the sinks, i.e.
        # a tail probability of 2^(-98 * scale * log2 e) = 2^-10.2 of a sink's
        pool[..., 0] = 1.875
        pool[..., 1] = 1.0
        for pos in range(0, n, every):
            page, row = int(block_rows[0, pos // PAGE]), pos % PAGE
            pool[page, row, 0] = 8.0
            pool[page, row, 1] = -1.0
        pool = pool.to(device).to(torch.float8_e4m3fn)
        block_rows = block_rows.to(device)
        q = torch.zeros(1, q_len, heads, LATENT + ROPE)
        q[..., 0] = 16.0
        q = q.to(device).to(torch.float8_e4m3fn)
        lens = torch.tensor([n], dtype=torch.int32, device=device)
        scale = dcp.default_softmax_scale()

        out = torch.empty(1, q_len, heads, LATENT, dtype=torch.bfloat16, device=device)
        ws = torch.empty(
            dcp.workspace_bytes(heads, dcp.MAX_Q_LEN, dcp.MAX_REQUESTS),
            dtype=torch.uint8,
            device=device,
        )
        stats = None
        if helix_form:
            bounds = (lens.view(-1, 1) - q_len + 1 + torch.arange(q_len, device=device)).reshape(-1)
            bounds = bounds.to(torch.int32)
            plan = dcp.SplitReadPlan(block_rows.contiguous().t(), lens, bounds, bounds == 0)
            stats = torch.empty(q_len, heads, 2, dtype=torch.float32, device=device)
        else:
            plan = dcp.SplitReadPlan(block_rows.t(), lens, None, None)
        dcp.split_read_attention(q, pool, plan, out, stats, ws, scale, 1.0, dcp.MAX_REQUESTS)

        ref = _torch_reference(q, pool, block_rows, lens, scale)
        err = (out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)
        assert float(err.max()) < 0.015, (
            f"per-(token, head) relative error up to {float(err.max()):.3f}"
        )
        if helix_form:
            kv = pool[block_rows[0].long()].float().reshape(-1, LATENT + ROPE)[:n]
            scores = torch.einsum("qhd,nd->qhn", q[0].float(), kv) * scale
            limit = n - q_len + 1 + torch.arange(q_len, device=device)
            pos = torch.arange(n, device=device)
            scores = scores.masked_fill(pos.view(1, 1, -1) >= limit.view(-1, 1, 1), float("-inf"))
            # stats = (log-sum-exp in natural log, 1)
            torch.testing.assert_close(
                stats[..., 0], torch.logsumexp(scores, -1), rtol=0, atol=2e-2
            )
            torch.testing.assert_close(stats[..., 1], torch.ones_like(stats[..., 1]))
    finally:
        HEADS = saved
