# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), helix8_moetp2ep4: the target class.

Serves generation over Helix CP8 (tp 1, no attention DP), the routed experts split MoE TP2 x EP4
on CUTEDSL_FC12: the c240 recipe.

The model is the helix8 targets' SM107 glue (``sm107_helix8_parts/``), built for these constants
(``constants.py``).

Engine settings (block reuse, page size) are the recipe's: the model loader applies model defaults of
the checkpoint's built-in class before a Modeling V2 target is routed, so a target cannot set
``llm_args``.
"""

from dataclasses import replace
from typing import ClassVar

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

from ..sm107_helix8_parts.modeling import KimiK3Sm107ForConditionalGeneration
from ..sm107_parts.constants import K3Sm107Constants
from .constants import CONSTANTS


class ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, helix8_moetp2ep4."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The family's hooks; the recurrent-state indices and the dummy-request mask are uploaded
        with one kernel when they change, as the reference's helix8 GEN servers run."""
        hooks = super().runtime_hooks
        return replace(
            hooks, hybrid_cache=replace(hooks.hybrid_cache, state_index_upload_kernel=True)
        )


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4"]
