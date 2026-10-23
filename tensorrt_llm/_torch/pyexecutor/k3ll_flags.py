# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 low-latency switches read from executor code (registry: the K3 NVFP4 target's ``flags.py``).

``k3ll_flag`` looks the value up on the flags module object at every call, so a switch set at runtime on every rank
(one-process A/B) takes effect, and returns 0 when the K3 NVFP4 target is not loaded in this process.
"""

import sys

K3LL_FLAGS_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.flags"
)


def k3ll_flag(name: str) -> int:
    flags = sys.modules.get(K3LL_FLAGS_MODULE)
    return int(getattr(flags, name, 0)) if flags is not None else 0
