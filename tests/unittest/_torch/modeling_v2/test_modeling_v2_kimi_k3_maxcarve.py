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
"""``TRTLLM_K3LL_MAXCARVE`` (``kimi_k3_nvfp4/flags.py``, ``cute_dsl_kernels/k3_carveout.py``): K3AttnResFastKernel and
the k3_ctm_gemv_wide builds get the max-shared carveout, set on each build's kernels when it is compiled.

* The variable is read once, at import: each value is checked in a fresh interpreter, with the device capability
  patched (no GPU): 1 on, 0 off, -1 (default) on SM103 only; a later change of the environment changes nothing; the
  flags module logs the same value.
* ``k3_carveout.apply`` on a real AttnResFast build sets the kernel attribute to 100 (read back from the driver), and
  leaves it alone when off.
"""

import os
import subprocess
import sys

import pytest
import torch

_PROBE = """
import os
import torch
torch.cuda.get_device_capability = lambda *a, **k: ({major}, {minor})
from tensorrt_llm._torch.cute_dsl_kernels import k3_carveout as c
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags as f
first = c.enabled(0)
os.environ["TRTLLM_K3LL_MAXCARVE"] = "0" if first else "1"
c.enabled.cache_clear()
print("MAXCARVE", c.MAXCARVE, f.TRTLLM_K3LL_MAXCARVE, first, c.enabled(0))
"""


@pytest.mark.parametrize(
    "value,capability,on",
    [
        (None, (10, 3), True),
        (None, (10, 7), False),
        (None, (10, 0), False),
        ("-1", (10, 3), True),
        ("0", (10, 3), False),
        ("1", (10, 7), True),
    ],
)
def test_maxcarve_is_read_once_at_import(value, capability, on):
    env = dict(os.environ)
    env.pop("TRTLLM_K3LL_MAXCARVE", None)
    if value is not None:
        env["TRTLLM_K3LL_MAXCARVE"] = value
    code = _PROBE.format(major=capability[0], minor=capability[1])
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout
    line = [ln for ln in out.splitlines() if ln.startswith("MAXCARVE ")][-1].split()
    expect_value = str(int(value)) if value is not None else "-1"
    # the module's value, the flag's logged value, the resolution, and the resolution after the env changed
    assert line[1:] == [expect_value, expect_value, str(on), str(on)]


def _carveout_of(kernel, device: int) -> int:
    from cuda.bindings import driver

    err, value = driver.cuKernelGetAttribute(
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        driver.CUkernel(int(kernel)),
        driver.CUdevice(device),
    )
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return value


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="K3AttnResFastKernel needs SM100 / SM103",
)
@pytest.mark.parametrize("on", [False, True])
def test_apply_sets_the_kernel_attribute(on, monkeypatch):
    from cuda.bindings import runtime as rt

    from tensorrt_llm._torch.cute_dsl_kernels import k3_carveout
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res as ar

    monkeypatch.setattr(k3_carveout, "MAXCARVE", 1 if on else 0)
    k3_carveout.enabled.cache_clear()
    try:
        compiled = ar._compiled(
            8, ar.ADD_NONE, ar._USE_PDL, ar.fast_cluster_size(32, 8), False, False, False,
            ar._EARLY_TRIGGER, False, False,
        )  # fmt: skip
        device = torch.cuda.current_device()
        lib = compiled.library
        err, count = rt.cudaLibraryGetKernelCount(lib)
        assert err == rt.cudaError_t.cudaSuccess and count >= 1
        err, kernels = rt.cudaLibraryEnumerateKernels(count, lib)
        assert err == rt.cudaError_t.cudaSuccess
        before = [_carveout_of(k, device) for k in kernels]
        assert k3_carveout.apply(compiled) is on
        after = [_carveout_of(k, device) for k in kernels]
        if on:
            assert after == [k3_carveout.CARVEOUT_PERCENT] * count
        else:
            assert after == before
    finally:
        k3_carveout.enabled.cache_clear()
