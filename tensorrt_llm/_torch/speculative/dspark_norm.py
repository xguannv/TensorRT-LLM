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
"""The DSpark drafter's residual-free RMSNorms with an early PDL trigger.

``TLLM_DSPARK_NORM_EARLY_TRIGGER=1`` runs ``q_a_layernorm`` (and the first layer's
``input_layernorm``) through
:mod:`~tensorrt_llm._torch.cute_dsl_kernels.blackwell.rmsnorm_early_trigger`: FlashInfer's CuTe
RMSNorm, the kernel the ``RMSNorm`` module runs here, with its PDL trigger moved to right after
its wait, so the GEMM after it (q_b, [q_a | kv_a]) becomes resident and fetches its weight while
the norm runs. Same values.
"""

import os
from typing import Optional

import torch

NORM_EARLY_TRIGGER_ENV_VAR = "TLLM_DSPARK_NORM_EARLY_TRIGGER"


def draft_rmsnorm(norm, x: torch.Tensor) -> Optional[torch.Tensor]:
    """``norm(x)`` with the early-trigger kernel, or None (the caller runs ``norm``) when the
    switch is off or the module would not run FlashInfer's RMSNorm on ``x``."""
    if os.environ.get(NORM_EARLY_TRIGGER_ENV_VAR, "0") != "1":
        return None
    from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

    if not IS_FLASHINFER_AVAILABLE or not get_env_enable_pdl():
        return None
    if (
        getattr(norm, "use_cuda_tile", True)
        or getattr(norm, "use_gemma", True)
        or getattr(norm, "return_hp_output", True)
        or (getattr(norm, "is_nvfp4", False) and getattr(norm, "nvfp4_scale", None) is not None)
    ):
        return None
    from ..utils import is_nvfp4_marlin_enabled

    if is_nvfp4_marlin_enabled():
        return None
    from ..cute_dsl_kernels.blackwell.rmsnorm_early_trigger import can_run, rmsnorm_early_trigger

    weight = norm.weight.detach()
    if not can_run(x, weight):
        return None
    return rmsnorm_early_trigger(x, weight, norm.variance_epsilon)
