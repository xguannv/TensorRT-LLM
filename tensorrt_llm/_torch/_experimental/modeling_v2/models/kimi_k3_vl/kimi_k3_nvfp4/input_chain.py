# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executor's decode-step input chain as a Kimi K3 default (``TRTLLM_K3LL_INPUT_CHAIN``).

``pyexecutor/input_commit.py`` holds four switches of the executor's input preparation, each off unless set in the
environment. Where this switch is on, the K3 NVFP4 target makes them default to on (`INPUT_CHAIN_DEFAULTS`) through
``input_commit.set_model_defaults``, called from the target's constructor (`install`):

* ``TRTLLM_BATCHED_INPUT_COMMIT=1``: the H2D input copies and the overlap gathers of a CUDA-graph decode step in one
  ``_input_commit_kernel`` before the replay (census ``dispatch:input_commit``);
* ``TRTLLM_KV_BLOCK_OFFSETS_TRITON=2``: the target and draft block-offset tables in one
  ``_copy_block_offsets_pair_kernel`` (``dispatch:block_offsets_pair``; a single table on the Triton copy:
  ``dispatch:block_offsets_triton``);
* ``TRTLLM_STATE_INDEX_COPY_KERNEL=1``: the KDA state indices and the dummy mask in one ``_small_copy_pair_kernel``
  where either changes (``dispatch:state_index_kernel``);
* ``TRTLLM_SPEC_SAMPLER_FUSED_SCATTER=1``: the spec sampler's store scatters in one ``_spec_store_scatter_kernel``
  (``dispatch:spec_store_scatter``).

A switch set in the environment keeps its value, either way. The defaults apply to the process once the target is
built: the model engine builds the model before its runner (which creates the InputCommit and reads
``batched_input_commit_enabled``), and the executor creator builds the sampler (``spec_sampler_fused_scatter_enabled``)
and the KV cache managers (read per call) after the model engine. Another model in the same process without the K3
target keeps the environment-only behavior; `install` with the switch off clears the defaults.
"""

import functools
from typing import Dict, Tuple

import torch

from tensorrt_llm._torch.pyexecutor import input_commit as _input_commit
from tensorrt_llm.logger import logger

from . import flags as _flags

INPUT_CHAIN_CAPABILITIES: Tuple[Tuple[int, int], ...] = ((10, 3),)

INPUT_CHAIN_DEFAULTS: Dict[str, str] = {
    "TRTLLM_BATCHED_INPUT_COMMIT": "1",
    "TRTLLM_KV_BLOCK_OFFSETS_TRITON": "2",
    "TRTLLM_STATE_INDEX_COPY_KERNEL": "1",
    "TRTLLM_SPEC_SAMPLER_FUSED_SCATTER": "1",
}

_logged = set()


@functools.lru_cache(maxsize=None)
def _on_by_default() -> bool:
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability() in INPUT_CHAIN_CAPABILITIES
    )


def enabled() -> bool:
    """``TRTLLM_K3LL_INPUT_CHAIN`` (read at model construction): 1 on, 0 off, -1 (default) on where the device is in
    ``INPUT_CHAIN_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_INPUT_CHAIN)
    if value >= 0:
        return value > 0
    return _on_by_default()


def _reason() -> str:
    value = int(_flags.TRTLLM_K3LL_INPUT_CHAIN)
    if value >= 0:
        return f"TRTLLM_K3LL_INPUT_CHAIN={value}"
    if not torch.cuda.is_available():
        return "TRTLLM_K3LL_INPUT_CHAIN=-1 without a CUDA device"
    major, minor = torch.cuda.get_device_capability()
    return f"TRTLLM_K3LL_INPUT_CHAIN=-1 on SM{major}{minor}"


def install() -> Dict[str, Tuple[str, str]]:
    """Install the target's input-chain defaults (none where the switch is off) and log, once per outcome, which
    input-chain switches are on and why. Returns ``{switch: (value, source)}``."""
    on = enabled()
    switches = _input_commit.set_model_defaults(INPUT_CHAIN_DEFAULTS if on else {})
    line = " ".join(f"{name}={value}({source})" for name, (value, source) in switches.items())
    key = (on, line)
    if key not in _logged:
        _logged.add(key)
        logger.info(f"Kimi K3 input chain {'on' if on else 'off'} ({_reason()}): {line}")
    return switches
