# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), helix8_moeep8: the target class.

Serves generation over Helix CP8 (tp 1, no attention DP), the routed experts split MoE EP8 on
CUTEDSL_FC12: the c24 and c75 recipes.

The model is the family's SM107 code (``sm107_parts/``), run with this target's constants
(``constants.py``).
"""

from typing import ClassVar

from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.modeling import KimiK3Sm107ForConditionalGeneration
from .constants import CONSTANTS


class ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, helix8_moeep8."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        """Block reuse off with 128-token pages, as every generation recipe runs."""
        return {"kv_cache_config": {"enable_block_reuse": False, "tokens_per_block": 128}}


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8"]
