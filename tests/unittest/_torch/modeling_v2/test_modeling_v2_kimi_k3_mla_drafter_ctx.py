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
"""``trtllm::k3_drafter_mla_ctx_kv`` against the stock MLA DSpark drafter's context path at the K3 drafter's shapes:
per layer the latent RMSNorm + RoPE of ``precompute_context_kv`` and the paged scatter of
``_store_context_kv_paged``."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    mla_drafter_ctx as ctx,
)
from tensorrt_llm._torch.models.modeling_dspark import _rmsnorm_rope_batched

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bf16(*shape, scale=0.05, gen=None, device="cuda"):
    return (torch.randn(*shape, generator=gen, device=device) * scale).to(torch.bfloat16)


@needs_cuda
def test_drafter_mla_ctx_kv_matches_the_stock_store():
    """``trtllm::k3_drafter_mla_ctx_kv`` against the stock context path: per layer, the latent RMSNorm + RoPE of
    ``precompute_context_kv`` and the paged scatter of ``_store_context_kv_paged``; masked (zero) rows stay zero."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    layers, rows, page, max_pos = 5, 24, 64, 4096
    latent = _bf16(rows, layers * (ctx.KV_LORA + ctx.ROPE), scale=1.0, gen=g)
    latent.view(rows, layers, -1)[-3:] = (
        0  # the stock mask multiply zeroes the rows past the accepted ones
    )
    norm = _bf16(layers, ctx.KV_LORA, gen=g).abs() + 1.0
    angles = torch.rand(max_pos, ctx.ROPE // 2, generator=g, device=dev) * 6.28
    table = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randperm(max_pos, generator=g, device=dev)[:rows]
    # Distinct (page, offset) targets; the RoPE position of a row is independent of where it is stored.
    pages = torch.arange(rows, device=dev) // 2
    offsets = (torch.arange(rows, device=dev) % 2) * 32 + positions % 32
    pools = [_bf16(16, 1, 1, page, ctx.KV_LORA + ctx.ROPE, gen=g) for _ in range(layers)]
    expected = [p.clone() for p in pools]
    pointers = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device=dev)

    torch.ops.trtllm.k3_drafter_mla_ctx_kv(
        latent,
        norm,
        torch.view_as_real(table).contiguous(),
        positions,
        pages,
        offsets,
        pointers,
        pools,
        1e-5,
    )

    freqs = table[positions]
    for layer in range(layers):
        out = _rmsnorm_rope_batched(
            latent.view(rows, layers, -1)[:, layer].unsqueeze(0).contiguous(),
            norm[layer],
            1e-5,
            ctx.ROPE,
            freqs.unsqueeze(0),
            norm_dim=ctx.KV_LORA,
        ).squeeze(0)
        expected[layer][pages, 0, 0, offsets] = out
    for layer in range(layers):
        err = (pools[layer].float() - expected[layer].float()).abs()
        tol = 2**-7 * expected[layer].float().abs() + 1e-3
        assert (err > tol).sum().item() == 0, f"layer {layer}: max err {err.max().item()}"
