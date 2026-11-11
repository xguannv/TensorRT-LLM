# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107): the target class.

The model is the family's SM107 code (``sm107_parts/``); the deployment's optimizations come
from its ``sm107_parts/features.py`` profile, selected by ``routing.py`` before this module is
imported.
"""

from ..sm107_parts.modeling import KimiK3Sm107ForConditionalGeneration


class ModelingV2KimiK3Nvfp4Sm107Default(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, sharded as the engine's mapping says."""


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Default"]
