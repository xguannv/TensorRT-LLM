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
"""The DSpark drafter's all-reduce + residual + RMSNorm in one CuTe kernel.

``TLLM_DSPARK_AR_NORM=1`` routes the drafter's fused ``RESIDUAL_RMS_NORM`` all-reduces (after
o_proj and down_proj, ``TLLM_DSPARK_DRAFT_FUSION``) through
:mod:`~tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_draft_ar_norm`: same values as the C++
MNNVL kernels the size selects, one launch instead of two, launched with the max-shared carveout
so that the drafter GEMM after it can become resident (and fetch its weight) while it waits for
the other ranks.
"""

import os
from typing import Optional, Tuple

import torch

AR_NORM_ENV_VAR = "TLLM_DSPARK_AR_NORM"


def draft_ar_norm_enabled() -> bool:
    """Whether ``TLLM_DSPARK_AR_NORM=1`` selects the one-kernel all-reduce + RMSNorm."""
    return os.environ.get(AR_NORM_ENV_VAR, "0") == "1"


def draft_ar_norm(
    all_reduce, partial: torch.Tensor, residual: torch.Tensor, norm
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(norm(all_reduce(partial) + residual), all_reduce(partial) + residual)`` with the CuTe
    kernel, or None (the caller keeps the C++ path) when the switch is off or the call does not
    fit it: MNNVL all-reduce of BF16 rows whose norm launch the kernel reproduces.

    The workspace view (a device table of the peers' mappings) is built on the first call, which
    must not be under CUDA graph capture (the warm-up forwards make it); under capture without it
    this returns None.
    """
    if not draft_ar_norm_enabled():
        return None
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if mnnvl is None or partial.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return None
    from ..cute_dsl_kernels.blackwell import k3_draft_ar_norm as kernel
    from ..distributed.ops import (
        _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from ..flashinfer_utils import get_env_enable_pdl

    hidden = partial.shape[-1]
    x = partial.reshape(-1, hidden)
    world = mnnvl.mapping.tp_size
    if not x.is_contiguous() or not kernel.can_run(x, world):
        return None
    m = x.shape[0]
    cpp_one_shot = m * hidden * world * 2 <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden, world, torch.bfloat16),
        kernel.required_bytes(m, hidden, world),
    )
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mnnvl.mapping, torch.bfloat16, buffer_size_bytes=need
    )
    if (
        workspace["uc_buffer"].data_ptr() not in MNNVLAllReduce._lamport_views
        and torch.cuda.is_current_stream_capturing()
    ):
        return None
    view = mnnvl._lamport_view(workspace)
    out, res_out = kernel.run_k3_draft_ar_norm(
        x,
        residual.reshape(-1, hidden),
        norm.weight.detach(),
        norm.variance_epsilon,
        view,
        cpp_one_shot,
        use_pdl=get_env_enable_pdl(),
    )
    return out.view(partial.shape), res_out.view(residual.shape)
