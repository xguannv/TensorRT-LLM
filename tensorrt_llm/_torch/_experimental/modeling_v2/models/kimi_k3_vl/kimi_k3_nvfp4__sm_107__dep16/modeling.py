# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), dep16: the target class.

Serves generation with attention DP over 16 ranks (tp 16), the routed experts over EP16 on
MEGAMOE_CUTEDSL: the c383 recipe.

The model is the family's SM107 code (``sm107_parts/``), run with this target's constants
(``constants.py``).

Engine settings the recipe carries, not this class: the model loader applies model defaults of the
checkpoint's built-in class before a Modeling V2 target is routed, so a target cannot set
``llm_args``. The c383 GEN recipe turns ``attention_dp_config.enable_kv_balanced_admission`` on,
which runs this target's planner.
"""

from dataclasses import replace
from typing import ClassVar

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.modeling import KimiK3Sm107ForConditionalGeneration
from .constants import CONSTANTS
from .kv_balance import K3Dep16KvBalancedAdmission

_KV_BALANCED_ADMISSION = K3Dep16KvBalancedAdmission()


class ModelingV2KimiK3Nvfp4Sm107Dep16(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, dep16."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The family's hooks and the KV-balanced admission planner the c383 GEN servers run
        (window 32, the KV load closest to the mean)."""
        return replace(super().runtime_hooks, kv_balanced_admission_planner=_KV_BALANCED_ADMISSION)


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Dep16"]
