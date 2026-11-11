# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), tp1: the target class.

Serves the context servers of every deployment: one rank per process (world 1, attention DP over
that one rank), DWDP, the CUTEDSL routed experts. Context only: no CUDA graphs, the overlap
scheduler off, chunked prefill.

The model is the family's SM107 code (``sm107_parts/``), run with this target's constants
(``constants.py``).
"""

from typing import ClassVar

from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.modeling import KimiK3Sm107ForConditionalGeneration
from .constants import CONSTANTS


class ModelingV2KimiK3Nvfp4Sm107Tp1(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, tp1."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        """Block reuse on with 128-token pages, the overlap scheduler off, chunked prefill on and no
        CUDA graphs: what every recipe's context servers run. The DWDP group, pool ratios, memory
        fractions and demotion thresholds differ between recipes and stay in the recipe."""
        return {
            "kv_cache_config": {"enable_block_reuse": True, "tokens_per_block": 128},
            "disable_overlap_scheduler": True,
            "enable_chunked_prefill": True,
            "cuda_graph_config": None,
        }


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Tp1"]
