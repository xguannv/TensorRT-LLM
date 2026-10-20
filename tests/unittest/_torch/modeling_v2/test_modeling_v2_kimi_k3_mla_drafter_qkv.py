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
"""``trtllm::k3_drafter_mla_qkv`` against the stock MLA DSpark drafter's chain at the K3 drafter's shapes (4 heads per
rank): q_a RMSNorm, q_b projection, query RoPE, nope absorption, latent RMSNorm + RoPE and the block's pool write."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    mla_drafter_qkv as qkv,
)
from tensorrt_llm._torch.models.modeling_dspark import _rmsnorm, _rmsnorm_rope_batched

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bf16(*shape, scale=0.05, gen=None, device="cuda"):
    return (torch.randn(*shape, generator=gen, device=device) * scale).to(torch.bfloat16)


@needs_cuda
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_drafter_mla_qkv_matches_the_stock_chain(batch):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    heads, block, page = 4, 8, 64
    rows = batch * block
    a = _bf16(rows, qkv.A_WIDTH, scale=1.0, gen=g)
    qa_norm = _bf16(qkv.Q_LORA, gen=g).abs() + 1.0
    q_b = _bf16(heads * (qkv.NOPE + qkv.ROPE), qkv.Q_LORA, gen=g)
    k_b = _bf16(heads, qkv.NOPE, qkv.KV_LORA, gen=g, scale=0.1)
    kv_norm = _bf16(qkv.KV_LORA, gen=g).abs() + 1.0
    angles = torch.rand(batch, block, qkv.ROPE // 2, generator=g, device=dev) * 6.28
    freqs = torch.polar(torch.ones_like(angles), angles)  # complex64 [B, block, 32]
    pages = 4 * batch
    pool = _bf16(pages, 1, 1, page, qkv.KV_LORA + qkv.ROPE, gen=g)
    before = pool.clone()
    # Block j of request b at a position crossing a page boundary: page 4b + 1, offsets 60..67 -> pages 4b+1, 4b+2.
    pos = 60 + torch.arange(block, device=dev)
    page_of = (4 * torch.arange(batch, device=dev)[:, None] + 1 + pos[None, :] // page).reshape(-1)
    off_of = (pos[None, :] % page).expand(batch, block).reshape(-1)
    slots = page_of * pool.stride(0) + off_of * pool.stride(3)

    query = torch.ops.trtllm.k3_drafter_mla_qkv(
        a,
        qa_norm,
        q_b,
        k_b,
        kv_norm,
        torch.view_as_real(freqs).reshape(rows, -1),
        pool,
        slots,
        1e-5,
        1e-5,
    )

    # The stock drafter's chain.
    q_a = _rmsnorm(a[:, : qkv.Q_LORA], qa_norm, 1e-5)
    q = F.linear(q_a, q_b).view(batch, block, heads, qkv.NOPE + qkv.ROPE)
    q = _rmsnorm_rope_batched(
        q, None, 0.0, qkv.ROPE, freqs, num_heads=heads, apply_weight=False, apply_rmsnorm=False
    )
    absorbed = torch.bmm(
        q[..., : qkv.NOPE].reshape(rows, heads, qkv.NOPE).transpose(0, 1), k_b
    ).transpose(0, 1)
    expected = torch.cat([absorbed, q[..., qkv.NOPE :].reshape(rows, heads, qkv.ROPE)], -1)
    blk_kv = _rmsnorm_rope_batched(
        a[:, qkv.Q_LORA :].reshape(batch, block, -1),
        kv_norm,
        1e-5,
        qkv.ROPE,
        freqs,
        norm_dim=qkv.KV_LORA,
    ).reshape(rows, -1)

    # bf16 GEMMs with another reduction order: within a few bf16 ULPs of the value.
    def close(x, y, what):
        err = (x.float() - y.float()).abs()
        tol = 2**-6 * y.float().abs() + 1e-3
        bad = (err > tol).float().mean().item()
        assert bad < 1e-3, f"{what}: {bad:.2%} of elements off, max err {err.max().item()}"

    close(query[..., : qkv.KV_LORA], expected[..., : qkv.KV_LORA], "absorbed query")
    close(query[..., qkv.KV_LORA :], expected[..., qkv.KV_LORA :], "query rope")
    flat = pool.view(-1)
    written = torch.stack([flat[s : s + qkv.KV_LORA + qkv.ROPE] for s in slots.tolist()])
    close(written, blk_kv, "block latent")
    # Nothing else in the pool moved.
    mask = torch.ones(flat.numel(), dtype=torch.bool, device=dev)
    for s in slots.tolist():
        mask[s : s + qkv.KV_LORA + qkv.ROPE] = False
    assert torch.equal(flat[mask], before.view(-1)[mask])


def test_supports_rejects_other_shapes():
    a = torch.empty(8, qkv.A_WIDTH, dtype=torch.bfloat16)
    q_b = torch.empty(4 * 192, qkv.Q_LORA, dtype=torch.bfloat16)
    k_b = torch.empty(4, qkv.NOPE, qkv.KV_LORA, dtype=torch.bfloat16)
    pool = torch.empty(2, 1, 1, 64, 576, dtype=torch.bfloat16)
    assert qkv.supports(a, q_b, k_b, pool)
    assert not qkv.supports(a[:, :-1], q_b, k_b, pool)
    assert not qkv.supports(a, q_b.float(), k_b, pool)
    assert not qkv.supports(a, q_b, k_b[:3], pool)
