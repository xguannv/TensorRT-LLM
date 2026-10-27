# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 low-latency switches read from executor code (registry: the K3 NVFP4 target's ``flags.py``).

``k3ll_flag`` looks the value up on the flags module object at every call, so a switch set at runtime on every rank
(one-process A/B) takes effect, and returns 0 when the K3 NVFP4 target is not loaded in this process.

``k3ll_census_dispatch`` counts one launch on a switched executor path in the K3 path census (``census.py`` next to
``flags.py``, on with ``TRTLLM_K3LL_CENSUS=1``); it does nothing when the census is off or the target is not loaded.

``drafter_prompt_window`` is the one reading of ``TRTLLM_K3LL_DRAFTER_PROMPT_CTX``: the window in tokens, 0 off.
"""

import sys
from typing import Optional, Tuple

K3LL_FLAGS_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.flags"
)
K3LL_CENSUS_MODULE = K3LL_FLAGS_MODULE.rsplit(".", 1)[0] + ".census"

#: The drafter prompt-context switch (registered in ``flags.py``).
DRAFTER_PROMPT_CTX = "TRTLLM_K3LL_DRAFTER_PROMPT_CTX"
#: The window in tokens that ``TRTLLM_K3LL_DRAFTER_PROMPT_CTX`` = 1, and -1 where it is on, select.
DRAFTER_PROMPT_CTX_WINDOW = 16384
#: Where -1 (the default) turns the switch on: SM103 (GB300). SM100 / SM107 stay off.
DRAFTER_PROMPT_CTX_CAPABILITIES = ((10, 3),)


def k3ll_flag(name: str) -> int:
    flags = sys.modules.get(K3LL_FLAGS_MODULE)
    return int(getattr(flags, name, 0)) if flags is not None else 0


def k3ll_flag_any_target(name: str) -> int:
    """``k3ll_flag`` for a switch read in a process that may not load the K3 NVFP4 target (e.g. a disaggregated
    ctx worker on the built-in model): imports the flags module (environment reads only) if it is not loaded."""
    import importlib

    flags = sys.modules.get(K3LL_FLAGS_MODULE) or importlib.import_module(K3LL_FLAGS_MODULE)
    return int(getattr(flags, name, 0))


def resolve_drafter_prompt_window(value: int, capability: Optional[Tuple[int, int]]) -> int:
    """The drafter prompt window in tokens that ``TRTLLM_K3LL_DRAFTER_PROMPT_CTX`` = ``value`` selects on a device of
    compute ``capability`` (``(major, minor)``, None when unknown); 0 is off.

    -1 (the default): ``DRAFTER_PROMPT_CTX_WINDOW`` on ``DRAFTER_PROMPT_CTX_CAPABILITIES``, else 0; 0: 0 (off);
    1: ``DRAFTER_PROMPT_CTX_WINDOW`` on any device; W >= 2: W.
    """
    if value < -1:
        raise ValueError(f"{DRAFTER_PROMPT_CTX}={value} is not accepted: expected an integer >= -1")
    if value == -1:
        on = capability is not None and tuple(capability) in DRAFTER_PROMPT_CTX_CAPABILITIES
        return DRAFTER_PROMPT_CTX_WINDOW if on else 0
    if value == 1:
        return DRAFTER_PROMPT_CTX_WINDOW
    return value


def _device_capability() -> Optional[Tuple[int, int]]:
    """This process's GPU compute capability, without creating a CUDA context when none exists yet (a serve front-end
    process reads it before it spawns the model ranks): NVML first, else torch. None without a GPU."""
    import torch

    if torch.cuda.is_initialized():
        return tuple(torch.cuda.get_device_capability())
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return tuple(int(x) for x in pynvml.nvmlDeviceGetCudaComputeCapability(handle))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - NVML absent or failing: fall back to torch below
        pass
    if torch.cuda.is_available():
        return tuple(torch.cuda.get_device_capability())
    return None


def drafter_prompt_window(capability: Optional[Tuple[int, int]] = None) -> int:
    """The effective ``TRTLLM_K3LL_DRAFTER_PROMPT_CTX`` window in tokens (0: off), resolved for ``capability``
    (default: this process's GPU, read only when the switch is -1). Every reader of the switch goes through here."""
    value = k3ll_flag_any_target(DRAFTER_PROMPT_CTX)
    if value == -1 and capability is None:
        capability = _device_capability()
    return resolve_drafter_prompt_window(value, capability)


def k3ll_census_dispatch(path: str) -> None:
    """Count the census key ``dispatch:<path>:<mode>`` (``mode``: ``capture`` under CUDA graph capture, else
    ``eager``; a graph replay runs no Python and is not counted)."""
    census = sys.modules.get(K3LL_CENSUS_MODULE)
    if census is not None and getattr(census, "ENABLED", False):
        census.count(f"dispatch:{path}:{census.mode()}")
