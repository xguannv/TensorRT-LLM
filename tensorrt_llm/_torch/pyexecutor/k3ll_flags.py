# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 low-latency switches read from executor code (registry: the K3 NVFP4 target's ``flags.py``).

``k3ll_flag`` looks the value up on the flags module object at every call, so a switch set at runtime on every rank
(one-process A/B) takes effect, and returns 0 when the K3 NVFP4 target is not loaded in this process.

``k3ll_census_dispatch`` counts one launch on a switched executor path in the K3 path census (``census.py`` next to
``flags.py``, on with ``TRTLLM_K3LL_CENSUS=1``); it does nothing when the census is off or the target is not loaded.
"""

import sys

K3LL_FLAGS_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.flags"
)
K3LL_CENSUS_MODULE = K3LL_FLAGS_MODULE.rsplit(".", 1)[0] + ".census"


def k3ll_flag(name: str) -> int:
    flags = sys.modules.get(K3LL_FLAGS_MODULE)
    return int(getattr(flags, name, 0)) if flags is not None else 0


def k3ll_flag_any_target(name: str) -> int:
    """``k3ll_flag`` for a switch read in a process that may not load the K3 NVFP4 target (e.g. a disaggregated
    ctx worker on the built-in model): imports the flags module (environment reads only) if it is not loaded."""
    import importlib

    flags = sys.modules.get(K3LL_FLAGS_MODULE) or importlib.import_module(K3LL_FLAGS_MODULE)
    return int(getattr(flags, name, 0))


def k3ll_census_dispatch(path: str) -> None:
    """Count the census key ``dispatch:<path>:<mode>`` (``mode``: ``capture`` under CUDA graph capture, else
    ``eager``; a graph replay runs no Python and is not counted)."""
    census = sys.modules.get(K3LL_CENSUS_MODULE)
    if census is not None and getattr(census, "ENABLED", False):
        census.count(f"dispatch:{path}:{census.mode()}")
