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
"""The staircase's MLA FP8 decode on the ll4 kernel (balanced and uniform split) equals an fp32 reference."""

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

LATENT, ROPE, PAGE = 512, 64, 64


def _ll4():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        decode_mla_ll4,
    )

    return decode_mla_ll4


def _make_cache(seq_lens, q_len, heads, device, seed=5):
    """A shuffled page pool holding each request's pages, and FP8 queries."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = [math.ceil(n / PAGE) for n in seq_lens]
    num_pages = sum(blocks) + 3
    perm = torch.randperm(num_pages, generator=g)
    block_rows = torch.zeros(len(seq_lens), max(blocks), dtype=torch.int32)
    k = 0
    for b, nb in enumerate(blocks):
        block_rows[b, :nb] = perm[k : k + nb].to(torch.int32)
        k += nb
    pool = (torch.randn(num_pages, PAGE, LATENT + ROPE, generator=g) * 0.5).to(device)
    q = (torch.randn(len(seq_lens), q_len, heads, LATENT + ROPE, generator=g) * 0.5).to(device)
    return pool.to(torch.float8_e4m3fn), block_rows.to(device), q.to(torch.float8_e4m3fn)


def _reference(q, pool, block_rows, seq_lens, scale):
    """FP32 causal MLA: token t of a request of length L sees positions < L - q_len + 1 + t."""
    q_len = q.shape[1]
    outs = []
    for b, n in enumerate(seq_lens):
        kv = (
            pool[block_rows[b, : math.ceil(n / PAGE)].long()].float().reshape(-1, LATENT + ROPE)[:n]
        )
        scores = torch.einsum("qhd,nd->qhn", q[b].float(), kv) * scale
        pos = torch.arange(n, device=q.device)
        limit = n - q_len + 1 + torch.arange(q_len, device=q.device)
        scores = scores.masked_fill(pos.view(1, 1, -1) >= limit.view(-1, 1, 1), float("-inf"))
        outs.append(torch.softmax(scores, -1) @ kv[:, :LATENT])
    return torch.stack(outs)


def _run(ll4, runner, q, pool, block_rows, seq_lens, scale):
    device = q.device
    batch, q_len, heads, _ = q.shape
    out = torch.full(
        (batch, q_len, heads, LATENT), float("nan"), dtype=torch.bfloat16, device=device
    )
    ws = torch.empty(runner.workspace_bytes(), dtype=torch.uint8, device=device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    runner(
        q,
        pool,
        block_rows.t(),
        lens,
        out,
        ws,
        torch.tensor([scale], dtype=torch.float32, device=device),
        torch.tensor([1.0], dtype=torch.float32, device=device),
    )
    return out


@pytest.mark.parametrize(
    "seq_lens,q_len",
    [
        ([4096], 8),
        ([PAGE * 5 + 3, 9000], 8),  # a short request beside a longer one
        ([16 * PAGE + 2, 300, 5000, 12000], 8),  # 4 requests of mixed length
        ([20000], 1),
        ([40], 8),  # shorter than one K tile
    ],
)
def test_balanced_split_equals_reference(seq_lens, q_len):
    ll4 = _ll4()
    heads = 6
    pool, block_rows, q = _make_cache(seq_lens, q_len, heads, torch.device("cuda"))
    scale = 1.0 / math.sqrt(192)
    runner = ll4.MlaLl4Decode(heads, q_len, PAGE, max_batch=8, balanced=True)
    out = _run(ll4, runner, q, pool, block_rows, seq_lens, scale)
    assert torch.isfinite(out.float()).all()
    ref = _reference(q, pool, block_rows, seq_lens, scale)
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("split_kv", [1, 16, 32])
def test_uniform_split_equals_reference(split_kv):
    ll4 = _ll4()
    heads, q_len, seq_lens = 6, 8, [3000, 9000]
    pool, block_rows, q = _make_cache(seq_lens, q_len, heads, torch.device("cuda"))
    scale = 1.0 / math.sqrt(192)
    runner = ll4.MlaLl4Decode(heads, q_len, PAGE, max_batch=8, balanced=False, split_kv=split_kv)
    out = _run(ll4, runner, q, pool, block_rows, seq_lens, scale)
    assert torch.isfinite(out.float()).all()
    ref = _reference(q, pool, block_rows, seq_lens, scale)
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)
