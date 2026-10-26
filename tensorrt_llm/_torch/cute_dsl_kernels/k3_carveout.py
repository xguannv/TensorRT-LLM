# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The max-shared carveout for a CuteDSL build (``TRTLLM_K3LL_MAXCARVE``).

A CTA that needs the full shared-memory carveout (the 200+ KB low-M GEMMs, the MoE front) can become resident on an SM
only when that SM's configuration has it; an SM running CTAs at a smaller carveout must drain first. The runtime gives a
kernel with little shared memory a small carveout, so a big successor launched early by PDL joins only as the small
kernel's SMs empty. A small kernel launched at the max-shared carveout leaves its SMs in the configuration the
successor needs.

``launch(preferred_smem_carveout=...)`` does not do this for builds executed by CuteDSL's CUDA-dialect executor (every
``cute.compile`` here): the value is recorded as a CUfunction attribute that executor never applies. :func:`apply` sets
``cudaFuncAttributePreferredSharedMemoryCarveout`` on the build's kernels itself, through the build's CUDA library
(loaded by ``compiled.library``, without a launch), once per build, before any launch: CUDA graphs captured afterwards
launch the kernels with it.

Callers: every vectorized ``K3AttnResFastKernel`` build (``k3_attn_res._compiled`` without ``exact``: the o_proj push
finish, the MoE tail LU / COL consumers, the torch ops; the per-element exact kernel keeps the runtime's carveout) and
every ``k3_ctm_gemv`` wide build (``k3_ctm_gemv_wide``, ``_wide_ar_push``, ``_wide_col_publish``).

The variable is read once, at import, never at forward time: 1 on, 0 off, -1 (default) on where the device is in
:data:`MAXCARVE_CAPABILITIES`; any other value raises at import.
"""

import functools
import os

import torch
from cuda.bindings import runtime as _rt

#: The values ``TRTLLM_K3LL_MAXCARVE`` takes.
MAXCARVE_VALUES = (-1, 0, 1)
#: ``TRTLLM_K3LL_MAXCARVE`` (``kimi_k3_nvfp4/flags.py`` logs it), read here once.
MAXCARVE = int(os.environ.get("TRTLLM_K3LL_MAXCARVE") or -1)
if MAXCARVE not in MAXCARVE_VALUES:
    raise ValueError(f"TRTLLM_K3LL_MAXCARVE must be one of {MAXCARVE_VALUES}, got {MAXCARVE}")
#: Where -1 turns it on: SM103 (measured on GB300).
MAXCARVE_CAPABILITIES = ((10, 3),)
#: The preferred carveout set, in percent of the maximum shared memory per SM.
CARVEOUT_PERCENT = 100


@functools.cache
def enabled(device_index: int) -> bool:
    """Whether builds compiled for ``device_index`` get the max-shared carveout."""
    if MAXCARVE != -1:
        return MAXCARVE == 1
    return torch.cuda.get_device_capability(device_index) in MAXCARVE_CAPABILITIES


def _check(err) -> None:
    if err != _rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"k3_carveout: {err}")


def apply(compiled) -> bool:
    """Set the max-shared carveout on every kernel of ``compiled`` (a ``cute.compile`` result) for the current
    device, when :func:`enabled`; loads the build's CUDA library if needed, launches nothing. Returns whether it set
    it."""
    device = torch.cuda.current_device()
    if not enabled(device):
        return False
    library = compiled.library
    err, count = _rt.cudaLibraryGetKernelCount(library)
    _check(err)
    err, kernels = _rt.cudaLibraryEnumerateKernels(count, library)
    _check(err)
    for kernel in kernels:
        (err,) = _rt.cudaKernelSetAttributeForDevice(
            kernel,
            _rt.cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout,
            CARVEOUT_PERCENT,
            device,
        )
        _check(err)
    return True
