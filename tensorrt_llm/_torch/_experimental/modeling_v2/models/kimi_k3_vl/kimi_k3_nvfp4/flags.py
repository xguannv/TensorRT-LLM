# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switches of the Kimi K3 NVFP4 low-latency decode path.

One ``_flag(...)`` line per switch: environment variable name, default, one sentence. A switch is read once, at
import, from the environment of every rank; ``log_flags`` prints every switch's effective value when the model is
built, so a log always states which variant ran. With every switch at its default the target behaves as before the
switch was added.
"""

import os
from typing import Dict, Tuple

from tensorrt_llm.logger import logger

_FLAGS: Dict[str, Tuple[int, int, str]] = {}


def _flag(name: str, default: int, doc: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as err:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from err
    _FLAGS[name] = (value, default, doc)
    return value


TRTLLM_K3LL_CENSUS = _flag(
    "TRTLLM_K3LL_CENSUS", 0, "count and log generic fallbacks in decode steps"
)
TRTLLM_K3LL_NVFP4_DECODE_MOE = _flag(
    "TRTLLM_K3LL_NVFP4_DECODE_MOE", 0, "decode MoE on k3_moe_nvfp4 with the sharded head and tail"
)
TRTLLM_K3LL_MOE_LOWM = _flag(
    "TRTLLM_K3LL_MOE_LOWM",
    0,
    "MoE decode path: NVFP4-latent front + k3_moe_lowm_dyn on CuteDSL NVFP4 buffers (decode_moe_lowm.py)",
)


def flags() -> Dict[str, Tuple[int, int, str]]:
    """Every switch: name -> (value, default, doc)."""
    return dict(_FLAGS)


def log_flags() -> None:
    """Print every switch's effective value (``*`` marks a non-default value)."""
    parts = [
        f"{name}={value}{'*' if value != default else ''}"
        for name, (value, default, _) in _FLAGS.items()
    ]
    logger.info("Kimi K3 low-latency switches: " + " ".join(parts))
