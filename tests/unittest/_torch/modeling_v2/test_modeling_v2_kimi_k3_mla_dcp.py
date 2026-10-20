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
