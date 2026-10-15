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
"""TRTLLM_K3_MLA_CTX_FMHA=cute: the CuTe DSL FP8 MLA context FMHA against an FP32 reference and trtllm-gen."""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_mla_ctx_fmha import (
    mla_ctx_cute_fmha,
    mla_ctx_cute_fmha_available,
)

H, DQK, DV = 96, 192, 128

pytestmark = pytest.mark.skipif(
    not mla_ctx_cute_fmha_available(), reason="needs FlashInfer CuTe DSL FMHA on SM100/SM103"
)


def _reference(q, k, v, cu_q, cu_k, causal, scale):
    outs, lses = [], []
    for i in range(cu_q.numel() - 1):
        qs, qe, ks, ke = (int(x) for x in (cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]))
        s = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), k[ks:ke].float()) * scale
        if causal:
            tq, tk = qe - qs, ke - ks
            mask = torch.arange(tk, device=s.device)[None, :] > (
                torch.arange(tq, device=s.device)[:, None] + tk - tq
            )
            s = s.masked_fill(mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v[ks:ke].float()))
        lses.append(torch.logsumexp(s, -1).t() / math.log(2.0))
    return torch.cat(outs), torch.cat(lses)


@pytest.mark.parametrize(
    "q_lens,kv_lens,causal,amp",
    [
        ([300, 7, 1000], [5000, 64, 2000], False, 1.0),
        ([512], [8192], False, 3.0),
        ([129, 640], [129, 640], True, 2.0),
        ([128], [64], False, 1.0),
        ([184], [8], False, 1.0),
        ([128, 184], [64, 8], False, 1.0),
        ([128, 184], [128, 184], True, 1.0),
    ],
)
def test_cute_fmha_matches_reference_like_trtllm_gen(q_lens, kv_lens, causal, amp):
    from flashinfer.prefill import trtllm_ragged_attention_deepseek

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    tq, tk = sum(q_lens), sum(kv_lens)
    q = (torch.randn(tq, H, DQK, device=dev, generator=gen) * amp).to(torch.float8_e4m3fn)
    k = (torch.randn(tk, H, DQK, device=dev, generator=gen) * amp).to(torch.float8_e4m3fn)
    v = torch.randn(tk, H, DV, device=dev, generator=gen).to(torch.float8_e4m3fn)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), dtype=torch.int32, device=dev
    )
    scale = 1.0 / math.sqrt(DQK)
    o_ref, lse_ref = _reference(q, k, v, cu_q, cu_k, causal, scale)

    out = torch.empty(tq, H, DV, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(tq, H, device=dev, dtype=torch.float32)
    mla_ctx_cute_fmha(q, k, v, out, lse, cu_q, cu_k, max(q_lens), max(kv_lens), scale, causal)

    ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    out_gen = torch.empty_like(out)
    lse_gen = torch.empty_like(lse)
    trtllm_ragged_attention_deepseek(
        q,
        k,
        v,
        ws,
        cu_k[1:] - cu_k[:-1],
        max(q_lens),
        max(kv_lens),
        scale,
        1.0,
        1.0,
        len(q_lens),
        -1,
        cu_q,
        cu_k,
        False,
        causal,
        True,
        out=out_gen,
        lse=lse_gen,
        skip_all_rows_active_check=True,
    )
    torch.cuda.synchronize()
    err = ((out.float() - o_ref).norm() / o_ref.norm()).item()
    err_gen = ((out_gen.float() - o_ref).norm() / o_ref.norm()).item()
    assert err <= 1.2 * err_gen + 1e-4, (err, err_gen)
    assert (lse - lse_ref).abs().max().item() < 1e-3
