# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), dep16: the target class.

Serves generation with attention DP over 16 ranks (tp 16), the routed experts over EP16 on
MEGAMOE_CUTEDSL: the c383 recipe.

The model is the family's SM107 code (``sm107_parts/``), run with this target's constants
(``constants.py``).
"""

from typing import ClassVar

from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.modeling import KimiK3Sm107ForConditionalGeneration
from .constants import CONSTANTS


class ModelingV2KimiK3Nvfp4Sm107Dep16(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, dep16."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        """Block reuse off with 128-token pages, and KV-balanced admission over the attention-DP ranks,
        as the c383 recipe runs."""
        return {
            "kv_cache_config": {"enable_block_reuse": False, "tokens_per_block": 128},
            "attention_dp_config": {"enable_kv_balanced_admission": True},
        }


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Dep16"]
