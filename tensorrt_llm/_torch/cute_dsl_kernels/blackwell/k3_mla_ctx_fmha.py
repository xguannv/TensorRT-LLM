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
"""Kimi-K3 MLA context FMHA on FP8 Q / K / V: FlashInfer's CuTe DSL FMHA with exact softmax correction.

The MLA FP8 chunked-prefill path (``TRTLLM_K3LL_MLA_CTX_FP8``, ``attention/mla.py``) attends FP8 Q
(``d_qk = 192``) against FP8 K / V (``d_v = 128``) of every cached-KV partition and of the new
tokens, writing BF16 O and the base-2 log-sum-exp of the scaled logits for the partition merge.  FlashInfer's prebuilt
CuTe DSL FMHA artifacts are compiled with lazy softmax correction: the running row maximum is
only refreshed when it grows by more than 8 (log2), so P is converted to E4M3 relative to a
stale maximum without the ``2^8`` pre-scale, and its small entries fall into E4M3's subnormal
range.  On peaked attention rows that costs 2-3x the output error of the trtllm-gen kernel.

This module JIT-compiles FlashInfer's ``BlackwellFusedMultiHeadAttentionForward`` with exact
correction instead (the maximum is refreshed every tile and P is pre-scaled by ``2^8`` before
the E4M3 conversion), which matches the trtllm-gen kernel's error against an FP32 reference
and is 11-16% faster than it on the cached-KV partitions on GB300 (equal on the causal part).

Selected by ``TRTLLM_K3LL_MLA_CTX_CUTE_FMHA`` (``attention/mla.py``).
"""

from __future__ import annotations

import functools
import math
import threading
from typing import Callable, Dict, Tuple

import torch


@functools.lru_cache(maxsize=None)
def mla_ctx_cute_fmha_available() -> bool:
    """FlashInfer's CuTe DSL FMHA source is importable and the GPU is SM100 / SM103."""
    try:
        from flashinfer.cute_dsl.attention.fmha.fmha import (  # noqa: F401
            BlackwellFusedMultiHeadAttentionForward,
        )
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


_compiled: Dict[Tuple[int, int, int, bool], Callable] = {}
_lock = threading.Lock()


def _compile(num_heads: int, head_dim_qk: int, head_dim_v: int, causal: bool) -> Callable:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cutlass.cute.typing import Int32
    from flashinfer.cute_dsl.attention.fmha.fmha import BlackwellFusedMultiHeadAttentionForward
    from flashinfer.cute_dsl.attention.fmha.helpers import fmha_helpers

    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128),
        head_dim=(head_dim_qk, head_dim_v),
        is_persistent=False,
        # Bottom-right aligned causal mask (query i attends keys up to i + kv_len - q_len).
        mask_type=(
            fmha_helpers.MaskEnum.WINDOW_MASK_INFERENCE
            if causal
            else fmha_helpers.MaskEnum.RESIDUAL_MASK
        ),
        enable_ex2_emulation=False,
        enable_skip_correction=False,
        use_tma_store=False,
    )
    sym_b, sym_q, sym_k, sym_hk, sym_hr, sym_seq = (cute.sym_int() for _ in range(6))
    fp8 = cutlass.Float8E4M3FN
    order5 = (4, 3, 2, 1, 0)
    q = make_fake_compact_tensor(
        fp8, (sym_b, sym_q, sym_hk, sym_hr, head_dim_qk), stride_order=order5, assumed_align=16
    )
    k = make_fake_compact_tensor(
        fp8, (sym_b, sym_k, sym_hk, 1, head_dim_qk), stride_order=order5, assumed_align=16
    )
    v = make_fake_compact_tensor(
        fp8, (sym_b, sym_k, sym_hk, 1, head_dim_v), stride_order=order5, assumed_align=16
    )
    o = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_q, sym_hk, sym_hr, head_dim_v),
        stride_order=order5,
        assumed_align=32,
    )
    lse = make_fake_compact_tensor(
        cutlass.Float32, (sym_b, sym_q, sym_hk, sym_hr), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    cu_q = make_fake_compact_tensor(Int32, (sym_seq,), assumed_align=16)
    cu_k = make_fake_compact_tensor(Int32, (sym_seq,), assumed_align=16)
    problem = (1, 1, 1, 1, num_heads, num_heads, head_dim_qk, head_dim_v)
    return cute.compile(
        fmha,
        q,
        k,
        v,
        o,
        problem,
        cu_q,
        cu_k,
        lse,
        None,
        1.0,
        1.0,
        1.0,
        None,
        None,
        Int32(0) if causal else None,
        None,
        None,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        False,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _kernel(num_heads: int, head_dim_qk: int, head_dim_v: int, causal: bool) -> Callable:
    key = (num_heads, head_dim_qk, head_dim_v, causal)
    fn = _compiled.get(key)
    if fn is None:
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compile(num_heads, head_dim_qk, head_dim_v, causal)
                _compiled[key] = fn
    return fn


def mla_ctx_cute_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    max_q: int,
    max_kv: int,
    softmax_scale: float,
    causal: bool,
) -> None:
    """Ragged FP8 attention of ``q`` [Tq, H, d_qk] over ``k`` [Tk, H, d_qk] / ``v`` [Tk, H, d_v].

    Writes ``out`` [Tq, H, d_v] (BF16, 32-byte aligned) and ``lse`` [Tq, H] (FP32, base-2
    log-sum-exp of ``softmax_scale * q.k``).  ``cu_q`` / ``cu_kv`` are int32 cumulative lengths;
    every request needs a positive query and KV length.  ``causal`` uses the bottom-right
    aligned mask.  Runs on the current stream.
    """
    from cutlass.cute.typing import Float32, Int32

    num_tokens, num_heads, head_dim_qk = q.shape
    num_kv, _, head_dim_v = v.shape
    fn = _kernel(num_heads, head_dim_qk, head_dim_v, causal)
    fn(
        q.view(1, num_tokens, num_heads, 1, head_dim_qk),
        k.view(1, num_kv, num_heads, 1, head_dim_qk),
        v.view(1, num_kv, num_heads, 1, head_dim_v),
        out.view(1, num_tokens, num_heads, 1, head_dim_v),
        (
            cu_q.numel() - 1,
            max_q,
            num_tokens,
            max_kv,
            num_heads,
            num_heads,
            head_dim_qk,
            head_dim_v,
        ),
        cu_q,
        cu_kv,
        lse.view(1, num_tokens, num_heads, 1),
        None,
        Float32(softmax_scale * math.log2(math.e)),
        Float32(softmax_scale),
        Float32(1.0),
        None,
        None,
        Int32(0) if causal else None,
        None,
        None,
        False,
    )
