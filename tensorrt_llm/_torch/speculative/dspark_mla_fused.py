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
"""The DSpark MLA drafter's block attention in one launch.

``TLLM_DSPARK_MLA_FUSED=1`` replaces, per drafter layer (``TLLM_DSPARK_TAIL_V2`` block RoPE
path), the absorption BMM, the cute-dsl MLA decode (split-KV kernel and reduction) and the V
up-projection BMM by :mod:`~tensorrt_llm._torch.cute_dsl_kernels.blackwell.dspark_mla_attend`.
It sums in a different order (draft tokens only): the target's verification keeps the output
distribution, the acceptance can move. ``TLLM_DSPARK_MLA_FUSED_ASYNC`` and
``TLLM_DSPARK_MLA_FUSED_STAGED`` (see the kernel module) keep its values bit for bit.
"""

import os
from typing import Optional

import torch

MLA_FUSED_ENV_VAR = "TLLM_DSPARK_MLA_FUSED"


def mla_fused_enabled() -> bool:
    """Whether ``TLLM_DSPARK_MLA_FUSED=1`` runs the drafter's block attention in one kernel."""
    return os.environ.get(MLA_FUSED_ENV_VAR, "0") == "1"


def _weights(attn) -> tuple:
    """``k_b`` transposed to ``[H, 512, 128]`` and ``v_b`` ``[H, 128, 512]``, BF16 contiguous,
    cached on the attention module (rebuilt when its projection buffers change)."""
    k_b = attn._k_b_proj
    v_b = attn._v_b_proj
    key = (k_b.data_ptr(), v_b.data_ptr())
    cached = attn.__dict__.get("_dspark_mla_fused_weights")
    if cached is None or cached[0] != key:
        k_b_t = k_b.detach().to(torch.bfloat16).transpose(1, 2).contiguous()
        v_b_c = v_b.detach().to(torch.bfloat16).contiguous()
        cached = (key, k_b_t, v_b_c)
        attn.__dict__["_dspark_mla_fused_weights"] = cached
    return cached[1], cached[2]


def _staged_slices(attn, v_b: torch.Tensor, cluster: int) -> Optional[torch.Tensor]:
    """``v_b`` re-laid per cluster CTA (``TLLM_DSPARK_MLA_FUSED_STAGED``), cached on the module;
    None under CUDA graph capture before the warm-up forwards made it."""
    from ..cute_dsl_kernels.blackwell.dspark_mla_attend import v_b_slices

    key = (v_b.data_ptr(), cluster)
    cached = attn.__dict__.get("_dspark_mla_fused_v_b_slices")
    if cached is None or cached[0] != key:
        if torch.cuda.is_current_stream_capturing():
            return None
        cached = (key, v_b_slices(v_b, cluster))
        attn.__dict__["_dspark_mla_fused_v_b_slices"] = cached
    return cached[1]


def draft_mla_attend(
    attn,
    q: torch.Tensor,
    query: torch.Tensor,
    pool: torch.Tensor,
    page_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    softmax_scale: float,
    next_attn=None,
) -> Optional[torch.Tensor]:
    """The layer's ``[B * T, H, 128]`` attention output (what the absorption BMM, the block decode
    and the V up-projection produce), or None when the switch is off or the kernel does not serve
    these operands (the caller runs the separate kernels).

    ``q [B, T, H, 192]`` is the q_b output, ``query [B, T, H, 576]`` holds the rotated ``q_pe`` in
    its last 64 columns (the block RoPE kernel's output), ``pool [pages, page_size, 576]`` the
    layer's latent cache with the block rows stored, ``page_tables [>= B, pages]`` and
    ``kv_lens [>= B]`` int32. ``next_attn``: the next drafter layer's attention module, whose
    weight slices the staged kernel (``TLLM_DSPARK_MLA_FUSED_STAGED``) prefetches into L2.
    """
    if not mla_fused_enabled():
        return None
    from ..cute_dsl_kernels.blackwell.dspark_mla_attend import (
        can_run,
        default_cluster,
        run_dspark_mla_attend,
        staged_enabled,
    )
    from ..flashinfer_utils import get_env_enable_pdl

    k_b_t, v_b = _weights(attn)
    if not can_run(q, query, pool, page_tables, kv_lens, k_b_t, v_b):
        return None
    batch, block, heads, _ = q.shape
    cluster = default_cluster(batch)
    v_b_sliced = None
    prefetch = None
    if staged_enabled():
        v_b_sliced = _staged_slices(attn, v_b, cluster)
        if v_b_sliced is None:
            return None
        if next_attn is not None:
            next_k_b_t, next_v_b = _weights(next_attn)
            next_sliced = _staged_slices(next_attn, next_v_b, cluster)
            if next_sliced is not None and next_k_b_t.shape == k_b_t.shape:
                prefetch = (next_k_b_t, next_sliced)
    out = torch.empty((batch, block, heads, v_b.shape[1]), dtype=q.dtype, device=q.device)
    run_dspark_mla_attend(
        q,
        query,
        pool,
        page_tables,
        kv_lens,
        k_b_t,
        v_b,
        softmax_scale,
        out,
        use_pdl=get_env_enable_pdl(),
        cluster=cluster,
        v_b_sliced=v_b_sliced,
        prefetch=prefetch,
    )
    return out.view(batch * block, heads, v_b.shape[1])
