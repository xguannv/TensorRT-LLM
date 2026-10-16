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
"""``TRTLLM_FP8_QUANT_1X128_TACTIC`` on ``trtllm::fp8_swap_ab_gemm`` (SM100/SM103).

The pin replaces the autotuned choice between the CUDA and the Triton 1x128 activation
quantization. It must skip that tuning, run the pinned kernel, and give the same GEMM output
as the other backend.
"""

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops import torch_custom_ops as ops
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(),
    reason="fp8_swap_ab_gemm quantizes for DeepGEMM on SM100/SM103",
)

K = 7168
N = 2048
# Kimi K3 GEN token counts that reach the DeepGEMM swap-AB path (b32 MTP7, b64 MTP5, b128 MTP6)
# plus an unaligned one.
M_VALUES = (256, 384, 896, 133)


def _packed_scale_values(scale: torch.Tensor, m: int) -> torch.Tensor:
    """The ``[m, ceil(K / 512)]`` int32 words of a packed MN-major UE8M0 scale tensor."""
    return scale[:m].contiguous()


@pytest.mark.parametrize("m", M_VALUES)
def test_cuda_and_triton_quantization_match(m):
    torch.manual_seed(m)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 3
    x[0, :128] = 0  # an all-zero 1x128 block
    q_cuda, s_cuda = ops._fp8_quantize_1x128_ue8m0(x, ops.Fp8QuantKernelRunner.TACTIC_CUDA)
    q_triton, s_triton = ops._fp8_quantize_1x128_ue8m0(x, ops.Fp8QuantKernelRunner.TACTIC_TRITON)
    assert s_cuda.shape == s_triton.shape and s_cuda.stride() == s_triton.stride()
    assert torch.equal(_packed_scale_values(s_cuda, m), _packed_scale_values(s_triton, m))
    assert torch.equal(q_cuda.view(torch.uint8), q_triton.view(torch.uint8))


@pytest.mark.parametrize("pin", ["cuda", "triton"])
@pytest.mark.parametrize("m", M_VALUES)
def test_pin_skips_tuning_and_runs_pinned_kernel(monkeypatch, pin, m):
    torch.manual_seed(m)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w_scale = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5

    seen_tactics = []
    quantize = ops._fp8_quantize_1x128_ue8m0

    def recording_quantize(inp, tactic):
        seen_tactics.append(tactic)
        return quantize(inp, tactic)

    tuned_ops = []
    choose_one = AutoTuner.choose_one

    def recording_choose_one(self, custom_op, *args, **kwargs):
        tuned_ops.append(custom_op)
        return choose_one(self, custom_op, *args, **kwargs)

    monkeypatch.setattr(ops, "_fp8_quantize_1x128_ue8m0", recording_quantize)
    monkeypatch.setattr(AutoTuner, "choose_one", recording_choose_one)
    monkeypatch.setattr(ops, "_FP8_QUANT_1X128_TACTIC", pin)
    with autotune():
        torch.ops.trtllm.fp8_swap_ab_gemm(x, w, w_scale, disable_ue8m0_cast=True)
    seen_tactics.clear()
    out_pinned = torch.ops.trtllm.fp8_swap_ab_gemm(x, w, w_scale, disable_ue8m0_cast=True)

    expected = (
        ops.Fp8QuantKernelRunner.TACTIC_TRITON
        if pin == "triton"
        else ops.Fp8QuantKernelRunner.TACTIC_CUDA
    )
    assert "trtllm::fp8_quant_1x128_tactic" not in tuned_ops
    assert seen_tactics == [expected]

    other = (
        ops.Fp8QuantKernelRunner.TACTIC_CUDA
        if pin == "triton"
        else ops.Fp8QuantKernelRunner.TACTIC_TRITON
    )
    runner = ops.fp8SwapABGemmRunner(torch.bfloat16, True, quant_tactic=other)
    out_other = runner(inputs=[x, w, w_scale], tactic=0)
    assert torch.equal(out_pinned, out_other)
