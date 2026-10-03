# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL FP8 MLA decode: flat (q_tok, head) fold vs the unfolded kernel.

K3 has 96 MLA heads and verifies 1 + 3 draft tokens per request. 96 heads fill
only 3/4 of the 128-row M tile and no integer token fold fits, so the kernel
used to run S_q = 4 tiles of 96 valid rows per request. The flat fold packs the
384 (q_tok, head) rows into 3 full M tiles; each row's math is unchanged, so the
output must be BITWISE identical to the unfolded kernel (TRTLLM_MLA_FLAT_FOLD=0)
at any split_kv, and agree with an fp32 reference of the MTP-causal attention.

Requires 1 SM100/SM103 GPU and nvidia-cutlass-dsl.
"""

import math

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
    )
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_sm100_or_sm103():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_sm100_or_sm103(), reason="needs sm100/sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

D, R, PAGE = 512, 64, 64


def _make_inputs(lens, S, H, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B = len(lens)
    pages_per = [math.ceil(L / PAGE) for L in lens]
    max_blocks = max(pages_per) + 1
    num_pages = sum(pages_per) + 8
    perm = torch.randperm(num_pages, device=dev).to(torch.int32)
    pt = torch.zeros(B, max_blocks, dtype=torch.int32, device=dev)
    off = 0
    for b, n in enumerate(pages_per):
        pt[b, :n] = perm[off : off + n]
        off += n
    kv = (torch.randn(num_pages, 1, PAGE, 1, D + R, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv_pages = kv[:, 0, :, 0, :]
    q = (torch.randn(B, S, H, D + R, device=dev) * 0.5).to(torch.float8_e4m3fn)
    o = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=dev)
    return dict(
        q=q,
        kv_pages=kv_pages,
        pt=pt,
        o=o,
        inputs=[
            q[..., :D].permute(2, 3, 1, 0),
            q[..., D:].permute(2, 3, 1, 0),
            kv_pages[..., :D].permute(1, 2, 0),
            kv_pages[..., D:].permute(1, 2, 0),
            pt.transpose(0, 1),
            torch.tensor(lens, dtype=torch.int32, device=dev),
            o.permute(2, 3, 1, 0),
            torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
            None,
            None,
            torch.tensor([1.0 / math.sqrt(192)], dtype=torch.float32, device=dev),
            torch.tensor([1.0], dtype=torch.float32, device=dev),
        ],
    )


def _run(x, H, S, split_kv, flat_fold, monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1" if flat_fold else "0")
    # the compiled-kernel cache is keyed by shape/tactic only
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", {})
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN, num_heads=H, seq_len_q=S, page_size=PAGE, max_batch_size=64
    )
    runner.forward(x["inputs"], ((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()
    return x["o"].clone()


def _reference(x, lens, S, H):
    q = x["q"].float()
    kv = x["kv_pages"].float()
    out = torch.empty(len(lens), S, H, D, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(192)
    for b, L in enumerate(lens):
        pages = x["pt"][b, : math.ceil(L / PAGE)].long()
        c = kv[pages].reshape(-1, D + R)[:L]
        for s in range(S):
            bound = L - (S - 1) + s
            logits = (q[b, s] @ c[:bound].T) * scale  # [H, bound]
            p = torch.softmax(logits, dim=-1)
            out[b, s] = p @ c[:bound, :D]
    return out


def test_k3_shape_takes_the_flat_fold(monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1")
    k = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=96,
        seq_len_q=4,
        fold_sq=True,
    )
    assert k.fold_rows == 128 and k.flat_fold
    # integer folds are unchanged
    k32 = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=32,
        seq_len_q=4,
        fold_sq=True,
    )
    assert k32.fold_rows == 128 and not k32.flat_fold


@pytest.mark.parametrize("split_kv", [1, 2, 4])
@pytest.mark.parametrize("H,S", [(96, 4), (48, 8)])
def test_flat_fold_is_bitwise_identical(H, S, split_kv, monkeypatch):
    lens = [3000, 70, 1500, 8191, 64, 513, 2048, 4500]
    x = _make_inputs(lens, S, H, seed=7)
    ref = _run(x, H, S, split_kv, flat_fold=False, monkeypatch=monkeypatch)
    new = _run(x, H, S, split_kv, flat_fold=True, monkeypatch=monkeypatch)
    assert torch.equal(new, ref)
    gold = _reference(x, lens, S, H)
    err = (new.float() - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err
