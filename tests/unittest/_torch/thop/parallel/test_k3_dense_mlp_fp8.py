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
"""``KIMI_K3_FP8_WEIGHT_READ_DENSE_MLP``: the dense layer's ``GatedMLP`` on FP8 weight reads.

The per-rank intermediate size of Kimi K3's dense MLP (2112 at TP4) is not a multiple of the
128-wide FP8 scale blocks; the conversion pads each half of ``gate_up`` with zero rows and ``down``
with zero columns. The padded intermediate columns must be exactly zero, the converted MLP must
stay within FP8 weight-read accuracy of the BF16 one (lossy, like the other FP8 weight reads), and
the converted projections must run the low-M FP8 GEMM at decode batch sizes.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.situ import SituAndMul

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)

HIDDEN = 7168
INTER = 2112
PADDED = 2176


def _mlp(seed, inter=INTER):
    torch.manual_seed(seed)
    mlp = GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=inter,
        bias=False,
        activation=SituAndMul(beta=4.0, linear_beta=25.0, use_fused_activation=True),
        dtype=torch.bfloat16,
        reduce_output=False,
    ).cuda()
    with torch.no_grad():
        mlp.gate_up_proj.weight.normal_(0, 0.02)
        mlp.down_proj.weight.normal_(0, 0.02)
    return mlp


@pytest.mark.parametrize("m", [1, 7, 14, 32])
def test_dense_mlp_fp8_weight_read(monkeypatch, m):
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    monkeypatch.setattr(mkl, "_K3_FP8W_LOWM_GEMM", True)
    mlp = _mlp(m)
    ref_gate_up = mlp.gate_up_proj.weight.detach().clone()
    ref_down = mlp.down_proj.weight.detach().clone()
    x = torch.randn((m, HIDDEN), device="cuda").to(torch.bfloat16)
    ref = mlp(x).float()

    model = SimpleNamespace(
        layers=[SimpleNamespace(is_moe=True), SimpleNamespace(is_moe=False, mlp=mlp)]
    )
    assert mkl._convert_dense_mlps_to_fp8_weight_read(model) == 2
    assert isinstance(mlp.gate_up_proj, mkl._Fp8BlockScaleWeightReadLinear)
    assert mlp.gate_up_proj.out_features == 2 * PADDED
    assert isinstance(mlp.down_proj, mkl._Fp8RowParallelLinear)
    assert mlp.down_proj.in_features == PADDED and mlp.down_proj.all_reduce is None

    # The padded intermediate columns are exactly zero.
    act = mlp.activation(mlp.gate_up_proj(x))
    assert act.shape == (m, PADDED)
    assert torch.count_nonzero(act[:, INTER:]) == 0

    # Lossy like the other FP8 weight reads (FP8 weights, MXFP8 activations); GSM8K decides.
    out = mlp(x).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.15, f"relative error {rel:.4f}"

    # The padding is exact: the same MLP built at the padded size with zero rows / columns, then
    # converted without padding, gives the same FP8 weights and bitwise the same output.
    padded = _mlp(m, PADDED)
    with torch.no_grad():
        gu = padded.gate_up_proj.weight
        gu.zero_()
        gu[:INTER] = ref_gate_up[:INTER]
        gu[PADDED : PADDED + INTER] = ref_gate_up[INTER:]
        padded.down_proj.weight.zero_()
        padded.down_proj.weight[:, :INTER] = ref_down
    assert (
        mkl._convert_dense_mlps_to_fp8_weight_read(
            SimpleNamespace(layers=[SimpleNamespace(is_moe=False, mlp=padded)])
        )
        == 2
    )
    assert torch.equal(
        padded.gate_up_proj.weight.view(torch.uint8), mlp.gate_up_proj.weight.view(torch.uint8)
    )
    assert torch.equal(padded.gate_up_proj.weight_scale, mlp.gate_up_proj.weight_scale)
    assert torch.equal(
        padded.down_proj.fp8.weight.view(torch.uint8), mlp.down_proj.fp8.weight.view(torch.uint8)
    )
    torch.testing.assert_close(padded(x).float(), out, rtol=0, atol=0)

    # The low-M FP8 GEMM serves both projections at decode batch sizes.
    calls = []
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

    real = lowm.run_k3_fp8w_lowm_gemm

    def spy(*args, **kwargs):
        calls.append(args[2].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(lowm, "run_k3_fp8w_lowm_gemm", spy)
    torch.testing.assert_close(mlp(x).float(), out, rtol=0, atol=0)
    assert len(calls) == 2


def test_dense_mlp_fp8_skips_unsupported():
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    mlp = _mlp(0)
    mlp.gate_up_proj.weight.data = mlp.gate_up_proj.weight.data.float()
    model = SimpleNamespace(layers=[SimpleNamespace(is_moe=False, mlp=mlp)])
    assert mkl._convert_dense_mlps_to_fp8_weight_read(model) == 0
    assert not isinstance(mlp.gate_up_proj, mkl._Fp8BlockScaleWeightReadLinear)
