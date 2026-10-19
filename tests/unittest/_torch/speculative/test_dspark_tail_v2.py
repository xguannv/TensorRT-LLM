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
"""``TLLM_DSPARK_TAIL_V2`` kernels write bitwise what the launches they replace write."""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs an SM100-family GPU"
)

KV_LORA = 512
ROPE = 64
NOPE = 128
HEAD_DIM = KV_LORA + ROPE
EPS = 1e-5


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _freqs_table(num_positions: int, gen) -> torch.Tensor:
    """A ``[P, ROPE // 2, 2]`` FP32 (cos, sin) table with YaRN-like magnitudes."""
    angle = torch.rand(num_positions, ROPE // 2, device="cuda", generator=gen) * 6.3
    scale = 1.0 + 0.3 * torch.rand(num_positions, 1, device="cuda", generator=gen)
    return torch.stack([angle.cos() * scale, angle.sin() * scale], dim=-1).contiguous()


def _rows(n, width, gen, row_stride=None):
    """BF16 rows with a spread of magnitudes; optionally a row-strided view."""
    row_stride = row_stride or width
    base = torch.randn(n, row_stride, device="cuda", generator=gen)
    base = base * torch.logspace(-2, 2, n, device="cuda").unsqueeze(1)
    return base.to(torch.bfloat16)[:, :width]


@pytest.mark.parametrize("num_tokens", [1, 7, 14, 28, 33])
def test_ctx_norm_rope_matches_per_layer_kernel(num_tokens):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_fused_dspark_rmsnorm_rope,
        dspark_mla_ctx_norm_rope,
    )

    layers = 5
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    latent = _rows(num_tokens, layers * HEAD_DIM, gen).view(num_tokens, layers, HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(layers, KV_LORA, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    table = _freqs_table(4096, gen)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda", generator=gen)

    out = dspark_mla_ctx_norm_rope(latent, weight, positions, table, EPS)

    compiled = _compile_fused_dspark_rmsnorm_rope(
        HEAD_DIM, ROPE, 1, EPS, True, True, False, KV_LORA
    )
    gathered = table[positions].contiguous()
    ref = torch.empty_like(out)
    for layer in range(layers):
        compiled(latent[:, layer], weight[layer], gathered, ref[layer])
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("num_tokens", [7, 14, 28, 32])
@pytest.mark.parametrize("page_size", [32, 64])
def test_block_rope_matches_separate_launches(num_tokens, page_size):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rope_into,
        dspark_mla_block_rope,
    )
    from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool

    heads = 4
    gen = torch.Generator(device="cuda").manual_seed(100 * num_tokens + page_size)
    q = _rows(num_tokens * heads, NOPE + ROPE, gen).view(num_tokens, heads, NOPE + ROPE)
    # The block latents are the kv half of the fused [q_a | kv_a] GEMM output.
    kv = _rows(num_tokens, HEAD_DIM, gen, row_stride=1536 + HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(KV_LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(8192, gen)
    positions = torch.randint(0, 8192, (num_tokens,), device="cuda", generator=gen)
    # An interleaved multi-layer page space, like the draft KV manager's view.
    num_pages = 40
    space = torch.randn(num_pages, 3, page_size, HEAD_DIM, device="cuda", generator=gen)
    pool = space.to(torch.bfloat16)[:, 1]
    pages = torch.randperm(num_pages, device="cuda", generator=gen)[:num_tokens].long()
    offsets = torch.randint(0, page_size, (num_tokens,), device="cuda", generator=gen)

    query = torch.full((num_tokens, heads, HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda")
    pool_new = pool.clone()
    dspark_mla_block_rope(q, query, kv, weight, positions, table, pool_new, pages, offsets, EPS)

    gathered = table[positions].contiguous()
    query_ref = torch.full_like(query, 7.0)
    cute_dsl_dspark_rope_into(q, gathered, query_ref, heads, ROPE, KV_LORA)
    blk_kv = cute_dsl_dspark_rmsnorm_rope(
        kv, weight, gathered, 1, ROPE, EPS, True, True, False, KV_LORA
    )
    pool_ref = pool.clone()
    scatter_rows_to_pool(blk_kv, pool_ref, pages, offsets)
    torch.cuda.synchronize()
    assert torch.equal(_bits(query), _bits(query_ref))
    assert torch.equal(_bits(pool_new), _bits(pool_ref))


@pytest.mark.parametrize(
    "rows,n_blocks,holes",
    [(1, 31, False), (2, 2048, False), (3, 4097, True), (8, 30000, False), (64, 6000, True)],
)
def test_block_table_decode_parallel_matches(rows, n_blocks, holes):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        BLOCK_TABLE_CHUNK,
        decode_draft_block_table,
        decode_draft_block_table_parallel,
    )

    gen = torch.Generator(device="cuda").manual_seed(rows * 7 + n_blocks)
    divisor = 10
    src_full = torch.randint(0, 50000, (2, rows + 1, 2, n_blocks), device="cuda", generator=gen)
    src_full = src_full.to(torch.int32)
    lengths = torch.randint(0, n_blocks + 1, (rows + 1,), device="cuda", generator=gen)
    cols = torch.arange(n_blocks, device="cuda")
    src_full[1] = torch.where(cols < lengths.view(-1, 1, 1), src_full[1], -1)
    if holes:
        src_full[1, :, :, ::7] = -1
    src = src_full[1, :rows, 0]

    table_ref = torch.full((rows + 1, n_blocks), -5, dtype=torch.int32, device="cuda")
    counts_ref = torch.full((rows + 1,), -5, dtype=torch.long, device="cuda")
    decode_draft_block_table(src, divisor, table_ref[:rows], counts_ref[:rows])

    table = torch.full_like(table_ref, -5)
    counts = torch.full_like(counts_ref, -5)
    partial = torch.full(
        (rows * -(-n_blocks // BLOCK_TABLE_CHUNK),), -1, dtype=torch.int32, device="cuda"
    )
    decode_draft_block_table_parallel(src, divisor, table[:rows], counts[:rows], partial)
    torch.cuda.synchronize()
    assert torch.equal(table, table_ref)
    assert torch.equal(counts, counts_ref)
