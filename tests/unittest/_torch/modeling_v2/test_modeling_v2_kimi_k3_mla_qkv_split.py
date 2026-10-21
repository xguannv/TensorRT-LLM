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
"""TRTLLM_K3LL_MLA_QKV_SPLIT: trtllm::k3_fp8_mla_qkv (paged latent write) and trtllm::k3_fp8_mla_project_qkv by the
three split kernels give the bits of the one-program-a-head _mla_qkv, at the TP16 shapes (6 local heads), M 1 to 16,
both Q-B scale recipes."""

import pytest
import torch

H, QA, NOPE, ROPE, KVL = 6, 1536, 128, 64, 512
AG_COLS = QA + KVL + ROPE + 64 * H  # q_a | kv_a | k_pe | the gate columns after


def _cuda_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(not _cuda_ok(), reason="needs an FP8 GPU")


def _attention():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        attention,
    )

    return attention


def _weights(gen):
    dev = "cuda"
    qb = (torch.randn(H * (NOPE + ROPE), QA, device=dev, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    qb_scale = (
        torch.rand(-(-H * (NOPE + ROPE) // 128), QA // 128, device=dev, generator=gen) * 1e-2 + 1e-3
    )
    kb = (torch.randn(H, KVL, NOPE, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    kb_scale = torch.rand(H, KVL // 128, 1, device=dev, generator=gen) * 1e-2 + 1e-3
    qa_norm = (1 + 0.1 * torch.randn(QA, device=dev, generator=gen)).bfloat16()
    kv_norm = (1 + 0.1 * torch.randn(KVL, device=dev, generator=gen)).bfloat16()
    return qa_norm, qb, qb_scale, kb, kb_scale, kv_norm


def _bits_equal(a, b) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("m", [1, 3, 8, 16])
def test_project_qkv_split_bits(monkeypatch, m, ue8m0):
    attention = _attention()
    gen = torch.Generator(device="cuda").manual_seed(100 + m)
    qa_norm, qb, qb_scale, kb, kb_scale, kv_norm = _weights(gen)
    ag = torch.randn(m, AG_COLS, device="cuda", generator=gen).bfloat16()
    args = (ag, qa_norm, qb, qb_scale, kb, kb_scale, kv_norm, 1e-6, 1e-6, ue8m0)
    monkeypatch.setattr(attention._flags, "TRTLLM_K3LL_MLA_QKV_SPLIT", 0)
    ref_query, ref_latent = attention.mla_project_qkv(*args)
    monkeypatch.setattr(attention._flags, "TRTLLM_K3LL_MLA_QKV_SPLIT", 1)
    query, latent = attention.mla_project_qkv(*args)
    torch.cuda.synchronize()
    assert _bits_equal(query, ref_query)
    assert _bits_equal(latent, ref_latent)


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("seqs,tokens", [(1, 1), (1, 8), (2, 8), (3, 5)])
def test_paged_qkv_split_bits(monkeypatch, seqs, tokens, ue8m0):
    attention = _attention()
    gen = torch.Generator(device="cuda").manual_seed(7 * seqs + tokens)
    qa_norm, qb, qb_scale, kb, kb_scale, kv_norm = _weights(gen)
    m = seqs * tokens
    ag = torch.randn(m, AG_COLS, device="cuda", generator=gen).bfloat16()
    pages_per_seq, page_offset, row_stride = 4, 1, 576
    pages = torch.randperm(seqs * pages_per_seq, device="cuda", generator=gen).int()
    pages = pages.view(seqs, pages_per_seq).contiguous()
    lengths = torch.randint(tokens, pages_per_seq * 64, (seqs,), device="cuda", generator=gen).int()
    pool_rows = (seqs * pages_per_seq + page_offset) * 64

    def run(split):
        monkeypatch.setattr(attention._flags, "TRTLLM_K3LL_MLA_QKV_SPLIT", split)
        pool = torch.zeros(pool_rows, row_stride, dtype=torch.bfloat16, device="cuda")
        out = attention.mla_qkv(
            ag, qa_norm, qb, qb_scale, kb, kb_scale, kv_norm, pool, pages, lengths, row_stride,
            page_offset, tokens, 1e-6, 1e-6, ue8m0,
        )  # fmt: skip
        return out, pool

    ref_out, ref_pool = run(0)
    out, pool = run(1)
    torch.cuda.synchronize()
    assert _bits_equal(out, ref_out)
    assert _bits_equal(pool, ref_pool)
    assert bool((ref_pool != 0).any())
