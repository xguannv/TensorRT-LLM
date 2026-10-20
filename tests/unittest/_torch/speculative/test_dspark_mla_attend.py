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
"""Fused DSpark MLA block attention (dspark_mla_attend, TLLM_DSPARK_MLA_FUSED) against an FP32
reference with the same BF16 roundings (absorbed query, normalized output), eager and in a
replayed CUDA graph with changed KV lengths."""

import math

import pytest
import torch
from utils.util import skip_pre_blackwell

NOPE, ROPE, LAT, VD = 128, 64, 512, 128


def _case(batch, block, heads, ctx, page, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, s=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * s).to(torch.bfloat16)

    kv_lens = torch.tensor(
        [max(0, ctx - 37 * b) + block for b in range(batch)], dtype=torch.int32, device="cuda"
    )
    pages_per_req = (ctx + block + page - 1) // page + 1
    num_pages = batch * pages_per_req + 3
    pool = rnd(num_pages, page, LAT + ROPE, s=0.5)
    perm = torch.randperm(num_pages - 3, device="cuda", generator=gen)[: batch * pages_per_req]
    page_tables = (perm + 2).view(batch, pages_per_req).to(torch.int32)
    q = rnd(batch, block, heads, NOPE + ROPE)
    query = torch.zeros(batch, block, heads, LAT + ROPE, dtype=torch.bfloat16, device="cuda")
    query[..., LAT:] = rnd(batch, block, heads, ROPE)
    k_b = rnd(heads, NOPE, LAT, s=0.08)
    v_b = rnd(heads, VD, LAT, s=0.05)
    return q, query, pool, page_tables, kv_lens, k_b, v_b


def _reference(q, query, pool, page_tables, kv_lens, k_b, v_b, scale):
    page = pool.shape[1]
    q_lat = torch.einsum("bthk,hkc->bthc", q[..., :NOPE].float(), k_b.float()).to(torch.bfloat16)
    full = torch.cat([q_lat, query[..., LAT:]], dim=-1).float()
    outs = []
    for b in range(q.shape[0]):
        pos = torch.arange(int(kv_lens[b]), device="cuda")
        rows = pool[page_tables[b].long()[pos // page], pos % page].float()
        p = torch.softmax(torch.einsum("thd,nd->thn", full[b], rows) * scale, dim=-1)
        o = torch.einsum("thn,nc->thc", p, rows[:, :LAT]).to(torch.bfloat16).float()
        outs.append(torch.einsum("thc,hvc->thv", o, v_b.float()))
    return torch.stack(outs)


def _check(out, ref):
    err = (out.float() - ref).abs().max().item()
    assert math.isfinite(err) and err <= 1e-2 * ref.abs().max().item() + 1e-3, err


@skip_pre_blackwell
@pytest.mark.parametrize("cluster", [8, 16])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("block", [7, 8])
@pytest.mark.parametrize("batch,heads", [(1, 4), (2, 4), (3, 2), (4, 4)])
def test_dspark_mla_attend_matches_reference(batch, heads, block, page, cluster):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dspark_mla_attend import (
        can_run,
        run_dspark_mla_attend,
    )

    scale = 1.0 / math.sqrt(NOPE + ROPE)
    for ctx in (0, 5, 57, 1000, 4500):
        q, query, pool, pt, kv_lens, k_b, v_b = _case(batch, block, heads, ctx, page, ctx + 7)
        k_b_t = k_b.transpose(1, 2).contiguous()
        assert can_run(q, query, pool, pt, kv_lens, k_b_t, v_b)
        out = torch.full(
            (batch, block, heads, VD), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        run_dspark_mla_attend(q, query, pool, pt, kv_lens, k_b_t, v_b, scale, out, cluster=cluster)
        _check(out, _reference(q, query, pool, pt, kv_lens, k_b, v_b, scale))


@skip_pre_blackwell
def test_dspark_mla_attend_graph_replay():
    """A captured launch reads the KV lengths and page tables at replay time."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dspark_mla_attend import (
        run_dspark_mla_attend,
    )

    batch, block, heads, page = 2, 7, 4, 64
    scale = 1.0 / math.sqrt(NOPE + ROPE)
    q, query, pool, pt, kv_lens, k_b, v_b = _case(batch, block, heads, 900, page, 1)
    k_b_t = k_b.transpose(1, 2).contiguous()
    out = torch.empty((batch, block, heads, VD), dtype=torch.bfloat16, device="cuda")
    run_dspark_mla_attend(q, query, pool, pt, kv_lens, k_b_t, v_b, scale, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_dspark_mla_attend(q, query, pool, pt, kv_lens, k_b_t, v_b, scale, out)
    for rep, lens in enumerate(([12, 900], [907, 300], [64, 65])):
        kv_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        gen = torch.Generator(device="cuda").manual_seed(100 + rep)
        q.copy_(torch.randn(q.shape, device="cuda", generator=gen).to(torch.bfloat16))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(out, _reference(q, query, pool, pt, kv_lens, k_b, v_b, scale))
