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
"""The fused decode-step head writes what the eager head writes."""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.spec_head_kernels import embedding_gather, fused_spec_head_prep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _fake_metadata(num_contexts, kv_lens, helix_pos, phys, cp_size, cp_rank, max_tokens):
    md = SimpleNamespace(
        num_contexts=num_contexts,
        kv_lens_cuda=kv_lens,
        helix_position_offsets=helix_pos,
        helix_local_slots=torch.full((max_tokens,), -7, dtype=torch.int32, device="cuda"),
        helix_kv_bounds=torch.full((max_tokens,), -7, dtype=torch.int32, device="cuda"),
        kv_cache_manager=SimpleNamespace(tokens_per_block=phys),
        mapping=SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank),
    )
    md.helix_local_len_vec = types.MethodType(TrtllmAttentionMetadata.helix_local_len_vec, md)
    return md


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize(
    "num_seqs,tokens_per_seq,phys,cp_size",
    [(1, 7, 32, 16), (2, 7, 32, 16), (4, 8, 64, 16), (3, 7, 32, 4), (16, 7, 32, 16)],
)
def test_fused_head_prep_matches_eager(overlap, num_seqs, tokens_per_seq, phys, cp_size):
    gen = torch.Generator(device="cuda").manual_seed(num_seqs * 131 + phys + int(overlap))
    max_tokens = 256
    n = num_seqs * tokens_per_seq
    num_contexts = 0
    for cp_rank in sorted({0, 1, cp_size // 2, cp_size - 1}):
        # Group starts near ledger-page boundaries so groups straddle owners.
        base = torch.randint(0, 1 << 18, (num_seqs,), device="cuda", generator=gen)
        base = base // phys * phys + torch.randint(
            -3, 3, (num_seqs,), device="cuda", generator=gen
        ).clamp(min=0)
        positions = (base.view(-1, 1) + torch.arange(tokens_per_seq, device="cuda")).view(-1)
        positions = positions.to(torch.int32)
        pos_offsets = torch.randint(-2, 3, (max_tokens,), device="cuda", generator=gen)
        pos_offsets = pos_offsets.to(torch.int32)
        kv_offsets = torch.randint(-2, 3, (64,), device="cuda", generator=gen).to(torch.int32)
        kv_lens = torch.randint(0, 1 << 16, (64,), device="cuda", generator=gen).to(torch.int32)
        position_ids = torch.randint(0, 1 << 18, (1, n), device="cuda", generator=gen)
        position_ids = position_ids.to(torch.int32)
        helix_pos = torch.zeros(max_tokens, dtype=torch.int32, device="cuda")
        helix_pos[:n] = positions

        # Eager: model_engine._preprocess_inputs + recompute_helix_spec_buffers.
        ref_pos_ids = position_ids.clone()
        ref = _fake_metadata(
            num_contexts, kv_lens.clone(), helix_pos.clone(), phys, cp_size, cp_rank, max_tokens
        )
        if overlap:
            ref_pos_ids[0, 0:] += pos_offsets[:n]
            ref.kv_lens_cuda[num_contexts:num_seqs] += kv_offsets[:num_seqs]
            ref.helix_position_offsets[:n] += pos_offsets[:n]
        TrtllmAttentionMetadata.recompute_helix_spec_buffers(ref, 0, n, tokens_per_seq)

        out_pos_ids = position_ids.clone()
        out = _fake_metadata(
            num_contexts, kv_lens.clone(), helix_pos.clone(), phys, cp_size, cp_rank, max_tokens
        )
        fused_spec_head_prep(
            out_pos_ids[0, 0:],
            pos_offsets,
            n,
            out.kv_lens_cuda,
            kv_offsets,
            num_contexts,
            num_seqs,
            helix_positions=out.helix_position_offsets,
            helix_slots=out.helix_local_slots,
            helix_bounds=out.helix_kv_bounds,
            num_helix=n,
            tokens_per_block=phys,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_bound_start=num_contexts,
            tokens_per_seq=tokens_per_seq,
            has_overlap=overlap,
        )
        torch.cuda.synchronize()
        assert torch.equal(out_pos_ids, ref_pos_ids)
        assert torch.equal(out.kv_lens_cuda, ref.kv_lens_cuda)
        assert torch.equal(out.helix_position_offsets, ref.helix_position_offsets)
        assert torch.equal(out.helix_local_slots, ref.helix_local_slots)
        assert torch.equal(out.helix_kv_bounds, ref.helix_kv_bounds)


def test_fused_head_prep_overlap_only_chunked_range():
    """Without Helix buffers only the overlap correction runs, on any KV range."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    position_ids = torch.randint(0, 1000, (1, 40), device="cuda", generator=gen).to(torch.int32)
    pos_offsets = torch.randint(-3, 4, (64,), device="cuda", generator=gen).to(torch.int32)
    kv_lens = torch.randint(0, 1000, (16,), device="cuda", generator=gen).to(torch.int32)
    kv_offsets = torch.randint(-3, 4, (16,), device="cuda", generator=gen).to(torch.int32)
    ref_pos, ref_kv = position_ids.clone(), kv_lens.clone()
    ref_pos[0, 12:] += pos_offsets[:28]
    ref_kv[3:5] += kv_offsets[:2]
    fused_spec_head_prep(position_ids[0, 12:], pos_offsets, 28, kv_lens, kv_offsets, 3, 2)
    torch.cuda.synchronize()
    assert torch.equal(position_ids, ref_pos)
    assert torch.equal(kv_lens, ref_kv)


@pytest.mark.parametrize("num_ids", [1, 7, 14, 32])
def test_embedding_gather_matches_embedding(num_ids):
    gen = torch.Generator(device="cuda").manual_seed(num_ids)
    weight = torch.randn(5000, 7168, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, 5000, (num_ids,), device="cuda", generator=gen).to(torch.int32)
    ids[0] = 4999
    out = embedding_gather(ids, weight)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), F.embedding(ids, weight).view(torch.int16))
