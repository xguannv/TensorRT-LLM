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
"""``trtllm::k3_drafter_mla_a_post`` against the stock MLA DSpark drafter's chain after its two a-projections at the K3
drafter's shapes: the q_a RMSNorm (q_b's input) and the block latent's RMSNorm + RoPE written into the pool."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    mla_drafter_qkv as qkv,
)
from tensorrt_llm._torch.models.modeling_dspark import _rmsnorm, _rmsnorm_rope_batched

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bf16(*shape, scale=0.05, gen=None, device="cuda"):
    return (torch.randn(*shape, generator=gen, device=device) * scale).to(torch.bfloat16)


@needs_cuda
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_drafter_mla_a_post_matches_the_stock_chain(batch):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    block, page = 8, 64
    rows = batch * block
    # A strided [q_a | kv_a | k_pe] view, as the GEMV's output may be.
    a = _bf16(rows, qkv.A_WIDTH + 32, scale=1.0, gen=g)[:, : qkv.A_WIDTH]
    qa_norm = _bf16(qkv.Q_LORA, gen=g).abs() + 1.0
    kv_norm = _bf16(qkv.KV_LORA, gen=g).abs() + 1.0
    angles = torch.rand(batch, block, qkv.ROPE // 2, generator=g, device=dev) * 6.28
    freqs = torch.polar(torch.ones_like(angles), angles)  # complex64 [B, block, 32]
    pool = _bf16(4 * batch, 1, 1, page, qkv.KV_LORA + qkv.ROPE, gen=g)
    before = pool.clone()
    # Block positions 60..67 of request b cross from page 4b + 1 into page 4b + 2.
    pos = 60 + torch.arange(block, device=dev)
    page_of = (4 * torch.arange(batch, device=dev)[:, None] + 1 + pos[None, :] // page).reshape(-1)
    off_of = (pos[None, :] % page).expand(batch, block).reshape(-1)
    slots = page_of * pool.stride(0) + off_of * pool.stride(3)

    q_a = torch.ops.trtllm.k3_drafter_mla_a_post(
        a, qa_norm, kv_norm, torch.view_as_real(freqs).reshape(rows, -1), pool, slots, 1e-5, 1e-5
    )

    expected_q_a = _rmsnorm(a[:, : qkv.Q_LORA].contiguous(), qa_norm, 1e-5)
    blk_kv = _rmsnorm_rope_batched(
        a[:, qkv.Q_LORA :].contiguous().reshape(batch, block, -1),
        kv_norm,
        1e-5,
        qkv.ROPE,
        freqs,
        norm_dim=qkv.KV_LORA,
    ).reshape(rows, -1)

    def close(x, y, what):
        err = (x.float() - y.float()).abs()
        tol = 2**-7 * y.float().abs() + 1e-3
        assert (err > tol).sum().item() == 0, f"{what}: max err {err.max().item()}"

    close(q_a, expected_q_a, "q_a")
    flat = pool.view(-1)
    width = qkv.KV_LORA + qkv.ROPE
    written = torch.stack([flat[s : s + width] for s in slots.tolist()])
    close(written, blk_kv, "block latent")
    mask = torch.ones(flat.numel(), dtype=torch.bool, device=dev)
    for s in slots.tolist():
        mask[s : s + width] = False
    assert torch.equal(flat[mask], before.view(-1)[mask]), (
        "the kernel wrote outside the block's slots"
    )


def test_supports_rejects_other_shapes():
    a = torch.empty(8, qkv.A_WIDTH, dtype=torch.bfloat16)
    pool = torch.empty(2, 1, 1, 64, 576, dtype=torch.bfloat16)
    assert qkv.supports(a, pool)
    assert not qkv.supports(a[:, :-1], pool)
    assert not qkv.supports(a.float(), pool)
    assert not qkv.supports(a, pool[..., :512])
