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
"""``TRTLLM_K3LL_MAXCARVE`` (``kimi_k3_nvfp4/flags.py``, ``cute_dsl_kernels/k3_carveout.py``): every vectorized
K3AttnResFastKernel build and every k3_ctm_gemv wide build get the max-shared carveout, set on each build's kernels
when it is compiled.

* The variable is read once, at import: each value is checked in a fresh interpreter, with the device capability
  patched (no GPU): 1 on, 0 off, -1 (default) on SM103 only; a later change of the environment changes nothing; the
  flags module logs the same value. Other values raise at import.
* In a fresh interpreter per value (0 / 1; nothing compiled before), AttnResFast (``k3_attn_res.run_k3_attn_res``,
  vectorized, 0 / 1 / 2 addends) and ``trtllm::k3_ctm_gemv_wide`` (bf16, sigmoid columns, fp32) run a few K3 shapes;
  their builds' kernel attribute, read back from the driver, is 100 on every kernel with 1 and on none with 0: the
  compile path itself sets it. The outputs of 0 and 1 are bitwise equal.
"""

import json
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


@pytest.mark.parametrize("value", ["2", "3", "-2"])
def test_maxcarve_rejects_other_values(value):
    env = dict(os.environ, TRTLLM_K3LL_MAXCARVE=value)
    run = subprocess.run(
        [sys.executable, "-c", "from tensorrt_llm._torch.cute_dsl_kernels import k3_carveout"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "TRTLLM_K3LL_MAXCARVE must be one of (-1, 0, 1)" in run.stderr


_RUN = """
import json
import sys

import torch
from cuda.bindings import driver
from cuda.bindings import runtime as rt

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res as ar
from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op

dev = torch.cuda.current_device()
gen = torch.Generator(device="cuda").manual_seed(7)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, generator=gen, device="cuda") * scale).bfloat16()


def carveouts(compiled):
    lib = compiled.library
    err, count = rt.cudaLibraryGetKernelCount(lib)
    assert err == rt.cudaError_t.cudaSuccess and count >= 1, err
    err, kernels = rt.cudaLibraryEnumerateKernels(count, lib)
    assert err == rt.cudaError_t.cudaSuccess, err
    values = []
    for k in kernels:
        err, v = driver.cuKernelGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            driver.CUkernel(int(k)),
            driver.CUdevice(dev),
        )
        assert err == driver.CUresult.CUDA_SUCCESS, err
        values.append(v)
    return values


out, seen = {}, {"attn_res": [], "ctm_wide": []}
H = ar.HIDDEN_SIZE
for m, n, nadd in ((1, 4, 0), (8, 9, 1), (32, 12, 2)):
    layer, block = rnd(m, H), rnd(n - 1, m, H)
    addends = tuple(rnd(m, H) for _ in range(nadd))
    weights = [rnd(H) for _ in range(3)]
    output = torch.empty_like(layer)
    updated = torch.empty_like(layer) if nadd else None
    assert ar.can_use_k3_attn_res(layer, addends, block, *weights)
    ar.run_k3_attn_res(layer, addends, block, *weights, 1e-6, 1e-6, output, updated, exact=False)
    out[f"attn_res m{m} n{n} add{nadd}"] = output
    if updated is not None:
        out[f"attn_res m{m} n{n} add{nadd} updated"] = updated
    # the build run_k3_attn_res compiled above (functools.cache: the same object)
    compiled = ar._compiled(
        n, nadd, ar._USE_PDL, ar.fast_cluster_size(m, n), False, False, False, ar._EARLY_TRIGGER, False, False
    )
    seen["attn_res"] += carveouts(compiled)
for m, (n, k, sig, fp32) in ((8, (7168, 768, -1, False)), (32, (2880, 7168, 2112, False)), (16, (280, 7168, -1, True))):
    x, w = rnd(m, k), rnd(n, k, scale=0.02)
    out[f"ctm_wide m{m} n{n} k{k}"] = torch.ops.trtllm.k3_ctm_gemv_wide(x, w, sig, fp32)
for key, fn in op._compiled.items():
    if key[0] == "k3_ctm_gemv_wide":
        seen["ctm_wide"] += carveouts(fn)
torch.cuda.synchronize()
torch.save({k: v.cpu() for k, v in out.items()}, sys.argv[1])
print("CARVEOUTS", json.dumps(seen))
"""


def _gpu_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``value -> (carveouts by family, outputs)``, each value in a fresh interpreter."""
    res = {}
    for value in ("0", "1"):
        path = tmp_path_factory.mktemp("maxcarve") / f"out_{value}.pt"
        env = dict(os.environ, TRTLLM_K3LL_MAXCARVE=value)
        stdout = subprocess.run(
            [sys.executable, "-c", _RUN, str(path)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        line = [ln for ln in stdout.splitlines() if ln.startswith("CARVEOUTS ")][-1]
        res[value] = (json.loads(line[len("CARVEOUTS ") :]), torch.load(path))
    return res


@pytest.mark.skipif(not _gpu_ok(), reason="K3AttnResFastKernel needs SM100 / SM103")
@pytest.mark.parametrize("value", ["0", "1"])
def test_compile_sets_the_kernel_attribute(runs, value):
    from tensorrt_llm._torch.cute_dsl_kernels import k3_carveout

    seen, _ = runs[value]
    for family in ("attn_res", "ctm_wide"):
        assert len(seen[family]) >= 3, (family, seen[family])
        if value == "1":
            assert seen[family] == [k3_carveout.CARVEOUT_PERCENT] * len(seen[family]), (
                family,
                seen[family],
            )
        else:
            assert k3_carveout.CARVEOUT_PERCENT not in seen[family], (family, seen[family])


@pytest.mark.skipif(not _gpu_ok(), reason="K3AttnResFastKernel needs SM100 / SM103")
def test_on_and_off_outputs_are_bitwise_equal(runs):
    off, on = runs["0"][1], runs["1"][1]
    # 3 AttnResFast outputs + 2 updated prefixes (1 and 2 addends) + 3 ctm wide outputs
    assert sorted(off) == sorted(on) and len(off) == 8
    for name in off:
        a, b = off[name], on[name]
        assert a.dtype == b.dtype and a.shape == b.shape, name
        view = torch.int32 if a.dtype == torch.float32 else torch.int16
        assert torch.equal(a.view(view), b.view(view)), name
