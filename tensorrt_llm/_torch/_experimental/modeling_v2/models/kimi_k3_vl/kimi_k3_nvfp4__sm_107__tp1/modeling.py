# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), tp1: the target class.

Serves the context servers of every deployment: one rank per process (world 1, attention DP over
that one rank), DWDP, the CUTEDSL routed experts. Context only: no CUDA graphs, the overlap
scheduler off, chunked prefill.

The model is the family's SM107 code (``sm107_parts/``), run with this target's constants
(``constants.py``). Its DSpark drafter only writes the context KV (``dspark_context_kv_only``) and
keeps it in FP8 (the drafter's ``draft_kv_dtype``).

Engine settings (block reuse with 128-token pages, the overlap scheduler, chunked prefill, CUDA
graphs, the DWDP group, pool ratios, memory fractions, demotion thresholds) are the recipe's: the
model loader applies model defaults of the checkpoint's built-in class before a Modeling V2 target
is routed, so a target cannot set ``llm_args``.
"""

from dataclasses import replace
from typing import ClassVar

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

from ..common.kda_metadata import KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT
from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.modeling import KimiK3Sm107ForConditionalGeneration
from .constants import CONSTANTS

# Context requests shorter than this many tokens keep their recurrent-state pages at the lowest
# eviction priority (every reference recipe's context servers).
_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN = 16384


class ModelingV2KimiK3Nvfp4Sm107Tp1(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, tp1."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The family's hooks and the context servers' recurrent-state cache policies: KDA
        mid-chunk snapshots at the prefill chunk, the lowest eviction priority for short prompts'
        recurrent pages, and exact long-prefill demotion (in effect where the recipe sets a
        demotion threshold)."""
        hooks = super().runtime_hooks
        return replace(
            hooks,
            hybrid_cache=replace(
                hooks.hybrid_cache,
                kda_midchunk_snapshot_alignment=KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT,
                low_priority_snapshot_prompt_len=_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN,
            ),
            exact_context_demotion=True,
        )


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Tp1"]
