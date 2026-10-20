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
"""The DSpark drafter's o_proj / down_proj all-reduces pushed from the GEMM epilogue.

``TLLM_DSPARK_AR_PUSH=1``: the drafter's row-parallel projections that feed a fused
``RESIDUAL_RMS_NORM`` all-reduce (o_proj, down_proj; ``TLLM_DSPARK_DRAFT_FUSION``) run on the
low-M BF16 GEMM without split-K (``low_m_bf16_splitk.run_splitk_dense_ar_push``), whose epilogue
stores each token row segment straight into the row owner's Lamport slot of the two-shot MNNVL
all-reduce; ``k3_draft_ar_norm.run_k3_draft_ar_finish_norm`` then completes the all-reduce, the
residual add and the RMSNorm in one kernel that polls from the moment it is resident.

Values: that GEMM sums K in the same order as the cuBLAS GEMM it replaces (the unit test checks
the drafter shapes bit for bit), and the finish kernel reproduces the C++ fusion, so the drafter's
values do not change.
"""

import dataclasses
import os
from typing import Optional, Tuple

import torch

AR_PUSH_ENV_VAR = "TLLM_DSPARK_AR_PUSH"
#: Largest token count the push GEMM serves (the low-M GEMM's limit).
MAX_TOKENS = 32


def ar_push_enabled() -> bool:
    """Whether ``TLLM_DSPARK_AR_PUSH=1`` pushes the drafter's all-reduce inputs."""
    return os.environ.get(AR_PUSH_ENV_VAR, "0") == "1"


#: ``TLLM_DSPARK_AR_FINISH_ASYNC=1``: the finish kernel's cluster sums go by ``st.async`` on an
#: mbarrier instead of a cluster barrier (``K3DraftArFinishNormKernel(async_sum=True)``).
FINISH_ASYNC_ENV_VAR = "TLLM_DSPARK_AR_FINISH_ASYNC"


def finish_async_enabled() -> bool:
    """Whether ``TLLM_DSPARK_AR_FINISH_ASYNC=1`` exchanges the finish kernel's sums by mbarrier."""
    return os.environ.get(FINISH_ASYNC_ENV_VAR, "0") == "1"


#: ``TLLM_DSPARK_AR_FINISH_POLL``: how the finish kernel's row owner polls its slots, ``wait``
#: (default), ``fast`` or ``serial`` (``k3_draft_ar_norm.K3DraftArFinishNormKernel``).
FINISH_POLL_ENV_VAR = "TLLM_DSPARK_AR_FINISH_POLL"


def finish_poll() -> str:
    """The finish kernel's poll mode (``TLLM_DSPARK_AR_FINISH_POLL``, default ``wait``)."""
    return os.environ.get(FINISH_POLL_ENV_VAR, "wait")


#: ``TLLM_DSPARK_AR_PUSH_STATE_EARLY`` (default 1): the push GEMM's epilogue reads the workspace
#: state while the main loop runs (``run_splitk_dense_ar_push(state_early=True)``); 0 reads it
#: after the accumulator, on the push's critical path.
STATE_EARLY_ENV_VAR = "TLLM_DSPARK_AR_PUSH_STATE_EARLY"


def state_early_enabled() -> bool:
    """Whether the push GEMM reads the workspace state early (default; ``=0`` disables)."""
    return os.environ.get(STATE_EARLY_ENV_VAR, "1") != "0"


@dataclasses.dataclass
class PushedPartial:
    """A projection output that every rank pushed into the next two-shot all-reduce on
    ``workspace`` (a ``k3_lamport_ar.LamportWorkspace``) instead of storing it."""

    workspace: object
    cpp_one_shot: bool
    shape: Tuple[int, ...]
    dtype: torch.dtype


def _tactic(m: int, k: int):
    from ..cute_dsl_kernels.blackwell.low_m_bf16_splitk import SplitKTactic

    mma_n = 8 if m <= 8 else (16 if m <= 16 else 32)
    # Every K tile in flight when it fits: the weight streams from HBM.
    return SplitKTactic(64, mma_n, 1, max(2, min(k // 128, 8)))


_SHAPE_OUT = {}


def _shape_out(m: int, n: int, device) -> torch.Tensor:
    """A BF16 ``[m, n]`` tensor that only gives the push GEMM its output layout."""
    key = (m, n, str(device))
    t = _SHAPE_OUT.get(key)
    if t is None:
        t = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        _SHAPE_OUT[key] = t
    return t


def draft_gemm_push(
    all_reduce, x: torch.Tensor, weight: torch.Tensor, role: str = ""
) -> Optional[PushedPartial]:
    """``x @ weight.T`` pushed into the two-shot MNNVL all-reduce that ``all_reduce`` runs next
    (a ``RESIDUAL_RMS_NORM`` fusion that ``draft_ar_finish_norm`` completes), or None (the caller
    keeps its GEMM and all-reduce) when the switch is off or the call does not fit.

    ``role`` (``"o_proj"`` / ``"down"``) selects the GEMM's exact shared-memory launch under
    ``TLLM_DSPARK_DRAFT_EXACT_SMEM`` (``dspark_draft_gemm``).

    The workspace view is built on the first call, which must not be under CUDA graph capture
    (the warm-up forwards make it); under capture without it this returns None.
    """
    if not ar_push_enabled():
        return None
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if mnnvl is None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    from ..cute_dsl_kernels.blackwell import k3_draft_ar_norm, k3_lamport_ar
    from ..cute_dsl_kernels.blackwell.low_m_bf16_splitk import run_splitk_dense_ar_push
    from ..distributed.ops import (
        _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from ..flashinfer_utils import get_env_enable_pdl
    from .dspark_draft_gemm import _exact_smem_for

    n, k = weight.shape
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    world = mnnvl.mapping.tp_size
    if (
        world < 2
        or not 0 < m <= MAX_TOKENS
        or not x2.is_contiguous()
        or not weight.is_contiguous()
        or k % 128
        or n % 64
        or not k3_draft_ar_norm.can_run(_shape_out(m, n, x.device), world)
    ):
        return None
    cpp_one_shot = m * n * world * 2 <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, n, world, torch.bfloat16),
        k3_lamport_ar.required_bytes(m, n, world, False),
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
    run_splitk_dense_ar_push(
        x2,
        weight.detach().t(),
        _shape_out(m, n, x.device),
        _tactic(m, k),
        view,
        get_env_enable_pdl(),
        exact_smem=bool(role) and _exact_smem_for(role),
        state_early=state_early_enabled(),
    )
    return PushedPartial(view, cpp_one_shot, (*x.shape[:-1], n), x.dtype)


def draft_ar_finish_norm(
    pushed: PushedPartial, residual: torch.Tensor, norm
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(norm(all_reduce(x) + residual), all_reduce(x) + residual)`` for the pushed ``x``."""
    from ..cute_dsl_kernels.blackwell.k3_draft_ar_norm import run_k3_draft_ar_finish_norm
    from ..flashinfer_utils import get_env_enable_pdl

    hidden = pushed.shape[-1]
    out, res_out = run_k3_draft_ar_finish_norm(
        residual.reshape(-1, hidden),
        norm.weight.detach(),
        norm.variance_epsilon,
        pushed.workspace,
        pushed.cpp_one_shot,
        use_pdl=get_env_enable_pdl(),
        async_sum=finish_async_enabled(),
        poll=finish_poll(),
    )
    return out.view(pushed.shape), res_out.view(residual.shape)
