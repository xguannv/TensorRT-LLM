# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

import pytest
import torch
from utils.util import check_accuracy, skip_pre_blackwell

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE, IS_CUTLASS_DSL_INTERNAL_AVAILABLE)
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.ugpu.policy import UgpuPolicy
from tensorrt_llm._torch.ugpu_utils import is_ugpu_enabled
from tensorrt_llm._torch.utils import (Fp4QuantizedTensor,
                                       is_nvfp4_marlin_supported_sm,
                                       model_extra_attrs)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

scaling_vector_size = 16


@skip_pre_blackwell
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16]
)  # TODO: Do we need float32 test case? fp4_quantize only supports fp16, bf16, fp8_e4m3
@pytest.mark.parametrize("mnk", [(1, 192, 128), (4, 192, 128), (8, 7168, 16384),
                                 (128, 7168, 16384)])
def test_fp4_linear(dtype, mnk):
    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(
        in_features=HIDDEN_SIZE,
        out_features=OUTPUT_SIZE,
        bias=False,
        dtype=dtype,
        quant_config=qc,
        nvfp4_allowed_backends=['cutlass'])  # Force CUTLASS to match reference

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

    l_fp4.load_weights([{
        'input_scale':
        1.0 / x_sf_global.cpu(),  # Simulates amax/(448*6) in modelopt ckpt
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(
            torch.float8_e4m3fn),  # Simulates float8_e4m3fn in modelopt ckpt
        'weight_scale_2':
        1.0 / w_sf_global.cpu()  # Simulates amax/(448*6) in modelopt ckpt
    }])
    l_fp4 = l_fp4.cuda()

    torch.testing.assert_close(l_fp4.weight, w_fp4)
    torch.testing.assert_close(l_fp4.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4.alpha[0], alpha_ref)

    with torch.inference_mode(), autotune():
        output = l_fp4.forward(x)

    output = l_fp4.forward(x)

    # ref linear
    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)
        output_ref = torch.ops.trtllm.fp4_gemm(
            x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, dtype)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="cutlass-dsl 4.1.0 requires Python 3.12+")
@pytest.mark.skipif(
    get_sm_version() not in [100, 103, 107],
    reason="This test is only supported in sm100, sm103, and sm107 architecture",
)
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE,
                    reason="cutlass-dsl is not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(128, 7168, 16384), (128, 24576, 1536),
                                 (128, 2112, 7168), (128, 4096, 7168),
                                 (128, 7168, 2048), [127, 1024, 3200]])
def test_fp4_linear_cute_dsl(dtype, mnk):

    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(in_features=HIDDEN_SIZE,
                   out_features=OUTPUT_SIZE,
                   bias=False,
                   dtype=dtype,
                   quant_config=qc,
                   nvfp4_allowed_backends=['cutedsl'])

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

    l_fp4.load_weights([{
        'input_scale':
        1.0 / x_sf_global.cpu(),  # Simulates amax/(448*6) in modelopt ckpt
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(
            torch.float8_e4m3fn),  # Simulates float8_e4m3fn in modelopt ckpt
        'weight_scale_2':
        1.0 / w_sf_global.cpu()  # Simulates amax/(448*6) in modelopt ckpt
    }])
    l_fp4 = l_fp4.cuda()

    torch.testing.assert_close(l_fp4.weight, w_fp4)
    torch.testing.assert_close(l_fp4.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4.alpha[0], alpha_ref)

    with torch.inference_mode(), autotune():
        output = l_fp4.forward(x)

    output = l_fp4.forward(x)

    # ref linear
    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)
        output_ref = torch.ops.trtllm.fp4_gemm(
            x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, dtype)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)


def _skip_if_no_ugpu():
    is_ugpu_enabled.cache_clear()
    if not is_ugpu_enabled():
        pytest.skip("uGPU localization is not enabled/supported on this system")


def _create_fp4_weights(output_size, hidden_size, dtype):
    weight = torch.randn((output_size, hidden_size), dtype=dtype).cuda()
    weight_sf_global = (448 * 6) / weight.abs().max().float()
    weight_fp4, weight_sf_block = torch.ops.trtllm.fp4_quantize(
        weight, weight_sf_global, scaling_vector_size, False)
    weight_sf_block_unswizzled = (
        torch.ops.trtllm.block_scale_interleave_reverse(
            weight_sf_block.cpu().view(pad_up(output_size, 128), -1)))
    return weight_fp4, weight_sf_block, weight_sf_block_unswizzled, weight_sf_global


def _create_fp4_input(seq_len, hidden_size, dtype):
    input_raw = torch.randn(seq_len, hidden_size, dtype=dtype).cuda()
    input_sf_global = (448 * 6) / input_raw.abs().max().float()
    input_fp4, input_sf_block = torch.ops.trtllm.fp4_quantize(
        input_raw, input_sf_global, scaling_vector_size, False)
    return Fp4QuantizedTensor(input_fp4, input_sf_block), input_sf_global


def _make_ugpu_weight_dict(weight_fp4,
                           weight_sf_block_unswizzled,
                           input_sf_global,
                           weight_sf_global,
                           bias=None):
    weight_dict = {
        "input_scale": 1.0 / input_sf_global.cpu(),
        "weight": weight_fp4.cpu(),
        "weight_scale": weight_sf_block_unswizzled.view(torch.float8_e4m3fn),
        "weight_scale_2": 1.0 / weight_sf_global.cpu(),
    }
    if bias is not None:
        weight_dict["bias"] = bias
    return [weight_dict]


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("mnk", [(128, 7168, 2112), (128, 1536, 12288)])
def test_fp4_linear_ugpu_correctness(mnk):
    _skip_if_no_ugpu()
    seq_len, output_size, hidden_size = mnk
    dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    base_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=False,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=False))
    ugpu_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=False,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=True))

    weight_fp4, _, weight_sf_unswizzled, weight_sf_global = _create_fp4_weights(
        output_size, hidden_size, dtype)
    input_tensor, input_sf_global = _create_fp4_input(seq_len, hidden_size,
                                                      dtype)
    weight_dict = _make_ugpu_weight_dict(weight_fp4, weight_sf_unswizzled,
                                         input_sf_global, weight_sf_global)

    base_linear.load_weights(weight_dict)
    base_linear = base_linear.cuda()
    base_linear.post_load_weights()
    ugpu_linear.load_weights(weight_dict)
    ugpu_linear = ugpu_linear.cuda()
    ugpu_linear.post_load_weights()

    with torch.inference_mode(), autotune(tune_mode=True):
        output_base = base_linear.forward(input_tensor)
    with torch.inference_mode(), autotune(tune_mode=True):
        output_ugpu = ugpu_linear.forward(input_tensor)

    torch.cuda.synchronize()
    assert ugpu_linear.partition_plan.enabled
    assert ugpu_linear.partition_plan.num_partitions == 2
    torch.testing.assert_close(output_base, output_ugpu, rtol=1e-2, atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.skipif(not IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
                    reason="Rubin CuTe DSL internal package is not available")
def test_fp4_linear_cute_dsl_mixed_cluster_ugpu_strided_output():
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    m = 256
    n = 4096
    k = 7168
    packed_k = k // 2
    sf_vec_size = 16
    sf_m = pad_up(m, 128)
    sf_n = pad_up(n, 128)
    sf_k = pad_up(k // sf_vec_size, 4)

    act_fp4 = torch.randint(0,
                            16, (m, packed_k),
                            dtype=torch.uint8,
                            device="cuda")
    weight_fp4 = torch.randint(0,
                               16, (n, packed_k),
                               dtype=torch.uint8,
                               device="cuda")
    act_sf = torch.ones(sf_m * sf_k, dtype=torch.float8_e4m3fn,
                        device="cuda").view(torch.uint8)
    weight_sf = torch.ones(sf_n * sf_k,
                           dtype=torch.float8_e4m3fn,
                           device="cuda").view(torch.uint8)
    alpha_one = torch.ones(1, dtype=torch.float32, device="cuda")
    alpha_scaled = torch.tensor([0.25], dtype=torch.float32, device="cuda")

    runner = cute_dsl_custom_ops.CuteDSLNVFP4InplaceRubinLinear(
        torch.bfloat16, to_userbuffers=False, use_tvm_ffi=True)
    output_one = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
    output_scaled = torch.empty_like(output_one)
    tactics = runner.get_valid_tactics(
        [act_fp4, weight_fp4, act_sf, weight_sf, alpha_one, output_one], None)
    mixed_tactics = [
        tactic for tactic in tactics
        if tactic[0] == "mixed_clusters" and tactic[-1] is False
    ]
    assert mixed_tactics, "expected at least one no-prefetch mixed-cluster tactic"

    preferred_tactic = ("mixed_clusters", (128, 64, 256), (128, 64, 128),
                        (4, 2), (2, 1), True, False)
    tactic = preferred_tactic if preferred_tactic in mixed_tactics else mixed_tactics[
        0]

    for ugpu_id in range(2):
        output_one.fill_(float("nan"))
        runner([act_fp4, weight_fp4, act_sf, weight_sf, alpha_one, output_one],
               tactic=tactic,
               partition_id=ugpu_id)
        torch.cuda.synchronize()

        output_scaled.fill_(float("nan"))
        runner([
            act_fp4, weight_fp4, act_sf, weight_sf, alpha_scaled, output_scaled
        ],
               tactic=tactic,
               partition_id=ugpu_id)
        torch.cuda.synchronize()

        output_one_slice = output_one[:, ugpu_id * n:(ugpu_id + 1) * n]
        output_scaled_slice = output_scaled[:, ugpu_id * n:(ugpu_id + 1) * n]
        other_one_slice = output_one[:, (1 - ugpu_id) * n:(2 - ugpu_id) * n]
        other_scaled_slice = output_scaled[:,
                                           (1 - ugpu_id) * n:(2 - ugpu_id) * n]
        assert torch.isfinite(output_one_slice.float()).all()
        assert torch.isfinite(output_scaled_slice.float()).all()
        assert torch.isnan(other_one_slice).all()
        assert torch.isnan(other_scaled_slice).all()
        assert output_one_slice.float().abs().max() > 0
        torch.testing.assert_close(output_scaled_slice.float(),
                                   output_one_slice.float() *
                                   alpha_scaled.item(),
                                   rtol=1e-2,
                                   atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.skipif(
    not IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
    reason="Rubin CuTe DSL internal package is not available",
)
@pytest.mark.parametrize(
    "m,mma_tiler,mma_inst_shape",
    [
        (4096, (128, 192, 256), (128, 192, 128)),
        (512, (256, 192, 256), (256, 192, 128)),
    ],
)
def test_cute_dsl_nvfp4_rubin_mixed_clusters_n192_fallback(
        m, mma_tiler, mma_inst_shape):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        CuteDSLNVFP4RubinLinear

    n, k = 7168, 384
    sf_vec_size = 16
    packed_k = k // 2
    sf_m = pad_up(m, 128)
    sf_n = pad_up(n, 128)
    sf_k = pad_up(k // sf_vec_size, 4)

    torch.manual_seed(41)
    act_fp4 = torch.randint(0,
                            256, (m, packed_k),
                            dtype=torch.uint8,
                            device="cuda")
    weight_fp4 = torch.randint(0,
                               256, (n, packed_k),
                               dtype=torch.uint8,
                               device="cuda")
    act_sf = torch.ones(sf_m * sf_k, dtype=torch.float8_e4m3fn,
                        device="cuda").view(torch.uint8)
    weight_sf = torch.ones(sf_n * sf_k,
                           dtype=torch.float8_e4m3fn,
                           device="cuda").view(torch.uint8)
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    inputs = [act_fp4, weight_fp4, act_sf, weight_sf, alpha]

    runner = CuteDSLNVFP4RubinLinear(output_dtype=torch.bfloat16,
                                     use_tvm_ffi=True)
    base_tactic = (
        "base",
        mma_tiler,
        mma_inst_shape,
        (2, 1),
        False,
        True,
    )
    mixed_tactic = (
        "mixed_clusters",
        mma_tiler,
        mma_inst_shape,
        (4, 2),
        (2, 1),
        False,
        True,
    )
    tactics = runner.get_valid_tactics(inputs, None)
    assert base_tactic in tactics
    assert mixed_tactic in tactics

    base_output = runner(inputs, tactic=base_tactic)
    mixed_output = runner(inputs, tactic=mixed_tactic)
    torch.cuda.synchronize()

    assert torch.isfinite(mixed_output).all()
    assert mixed_output.float().abs().max() > 0
    torch.testing.assert_close(
        mixed_output,
        base_output,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("nk", [(7168, 2112), (1536, 12288)])
def test_split_weights_for_ugpu(nk):
    _skip_if_no_ugpu()
    output_size, hidden_size = nk
    dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    linear = Linear(in_features=hidden_size,
                    out_features=output_size,
                    bias=True,
                    dtype=dtype,
                    quant_config=quant_config,
                    nvfp4_allowed_backends=["cutedsl"],
                    ugpu_policy=UgpuPolicy(enabled=True))
    assert linear.partition_plan.enabled

    weight_fp4, _, weight_sf_unswizzled, weight_sf_global = _create_fp4_weights(
        output_size, hidden_size, dtype)
    input_sf_global = torch.tensor(1.0 / (448 * 6))
    bias_data = torch.randn(output_size, dtype=dtype)
    weight_dict = _make_ugpu_weight_dict(weight_fp4, weight_sf_unswizzled,
                                         input_sf_global, weight_sf_global,
                                         bias_data)

    linear.load_weights(weight_dict)
    linear = linear.cuda()

    full_weight = linear.weight.data.clone()
    full_bias = linear.bias.data.clone()

    linear.post_load_weights()
    shards = linear._ugpu_weight_shards

    num_partitions = linear.partition_plan.num_partitions
    assert len(shards) == num_partitions

    layout = linear.partition_plan.layout
    assert layout is not None
    assert layout.padded_axis_extent == full_weight.size(0)
    partition_n = layout.per_partition_axis_extent(padded=True)
    for shard in shards:
        assert shard["weight"].shape == (partition_n, full_weight.shape[1])

    reconstructed = torch.cat([shard["weight"] for shard in shards], dim=0)
    assert torch.equal(reconstructed.cpu(), full_weight.cpu())

    reconstructed_bias = torch.cat([shard["bias"] for shard in shards], dim=0)
    assert torch.equal(reconstructed_bias.cpu(), full_bias.cpu())

    assert linear.weight.numel() == 0
    assert linear.weight_scale.numel() == 0
    assert linear.bias is not None
    assert torch.equal(linear.bias.cpu(), full_bias.cpu())

    expected_keys = {"weight", "weight_scale", "bias"}
    for shard in shards:
        assert set(shard.keys()) == expected_keys


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("mnk", [(128, 7168, 2112)])
def test_fp4_linear_ugpu_bias_correctness(mnk):
    _skip_if_no_ugpu()
    seq_len, output_size, hidden_size = mnk
    dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    base_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=True,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=False))
    ugpu_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=True,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=True))

    weight_fp4, _, weight_sf_unswizzled, weight_sf_global = _create_fp4_weights(
        output_size, hidden_size, dtype)
    input_tensor, input_sf_global = _create_fp4_input(seq_len, hidden_size,
                                                      dtype)
    bias_data = torch.randn(output_size, dtype=dtype)
    weight_dict = _make_ugpu_weight_dict(weight_fp4, weight_sf_unswizzled,
                                         input_sf_global, weight_sf_global,
                                         bias_data)

    base_linear.load_weights(weight_dict)
    base_linear = base_linear.cuda()
    base_linear.post_load_weights()
    ugpu_linear.load_weights(weight_dict)
    ugpu_linear = ugpu_linear.cuda()
    ugpu_linear.post_load_weights()

    assert ugpu_linear._ugpu_weight_shards is not None
    assert "bias" in ugpu_linear._ugpu_weight_shards[0]

    with torch.inference_mode():
        output_base_forward = base_linear(input_tensor)
        output_ugpu_forward = ugpu_linear(input_tensor)
        output_base = base_linear.apply_linear(input_tensor, base_linear.bias)
        output_ugpu = ugpu_linear.apply_linear(input_tensor, True)
        output_base_no_bias = base_linear.apply_linear(input_tensor, None)
        output_ugpu_no_bias = ugpu_linear.apply_linear(input_tensor, None)

    torch.cuda.synchronize()
    torch.testing.assert_close(output_base_forward,
                               output_ugpu_forward,
                               rtol=1e-2,
                               atol=0.15)
    torch.testing.assert_close(output_base, output_ugpu, rtol=1e-2, atol=0.15)
    torch.testing.assert_close(output_base_no_bias,
                               output_ugpu_no_bias,
                               rtol=1e-2,
                               atol=0.15)
    assert (output_ugpu - output_ugpu_no_bias).abs().max() > 0


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
def test_cute_dsl_nvfp4_inplace_rubin_tvm_ffi_ugpu_correctness():
    _skip_if_no_ugpu()
    seq_len, output_size, hidden_size = (128, 1024, 3200)
    dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    base_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=False,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=False))
    ugpu_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=False,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=True))

    weight_fp4, _, weight_sf_unswizzled, weight_sf_global = _create_fp4_weights(
        output_size, hidden_size, dtype)
    input_tensor, input_sf_global = _create_fp4_input(seq_len, hidden_size,
                                                      dtype)
    weight_dict = _make_ugpu_weight_dict(weight_fp4, weight_sf_unswizzled,
                                         input_sf_global, weight_sf_global)

    base_linear.load_weights(weight_dict)
    base_linear = base_linear.cuda()
    base_linear.post_load_weights()
    ugpu_linear.load_weights(weight_dict)
    ugpu_linear = ugpu_linear.cuda()
    ugpu_linear.post_load_weights()

    assert ugpu_linear.partition_plan.enabled
    act_fp4, act_sf, alpha = ugpu_linear.quant_method._input_prepare(
        ugpu_linear, input_tensor)
    output_ffi = torch.empty(
        act_fp4.shape[0],
        ugpu_linear.partition_plan.layout.padded_axis_extent,
        dtype=dtype,
        device="cuda")

    with torch.inference_mode():
        output_base = base_linear.forward(input_tensor)
        for partition_id, shard in enumerate(ugpu_linear._ugpu_weight_shards):
            torch.ops.trtllm.cute_dsl_nvfp4_gemm_inplace_rubin(
                act_fp4,
                shard["weight"],
                act_sf,
                shard["weight_scale"],
                alpha,
                dtype,
                False,
                True,
                output_ffi,
                partition_id,
            )

    torch.cuda.synchronize()
    torch.testing.assert_close(output_base,
                               output_ffi[:, :output_size],
                               rtol=1e-2,
                               atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("use_tvm_ffi", [True, False])
@pytest.mark.parametrize("swap_ab", [False, True])
def test_cute_dsl_nvfp4_inplace_rubin_mixed_clusters_ugpu_correctness(
        use_tvm_ffi, swap_ab):
    _skip_if_no_ugpu()
    from tensorrt_llm._torch.autotuner import AutoTuner

    seq_len, output_size, hidden_size = (512, 1024, 2048)
    dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    base_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=False,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=False))
    ugpu_linear = Linear(in_features=hidden_size,
                         out_features=output_size,
                         bias=False,
                         dtype=dtype,
                         quant_config=quant_config,
                         nvfp4_allowed_backends=["cutedsl"],
                         ugpu_policy=UgpuPolicy(enabled=True))

    weight_fp4, _, weight_sf_unswizzled, weight_sf_global = _create_fp4_weights(
        output_size, hidden_size, dtype)
    input_tensor, input_sf_global = _create_fp4_input(seq_len, hidden_size,
                                                      dtype)
    weight_dict = _make_ugpu_weight_dict(weight_fp4, weight_sf_unswizzled,
                                         input_sf_global, weight_sf_global)

    base_linear.load_weights(weight_dict)
    base_linear = base_linear.cuda()
    base_linear.post_load_weights()
    ugpu_linear.load_weights(weight_dict)
    ugpu_linear = ugpu_linear.cuda()
    ugpu_linear.post_load_weights()

    assert ugpu_linear.partition_plan.enabled
    act_fp4, act_sf, alpha = ugpu_linear.quant_method._input_prepare(
        ugpu_linear, input_tensor)
    output_mixed = torch.empty(
        act_fp4.shape[0],
        ugpu_linear.partition_plan.layout.padded_axis_extent,
        dtype=dtype,
        device="cuda")

    first_shard = ugpu_linear._ugpu_weight_shards[0]
    with AutoTuner.get().capture() as capture, torch.inference_mode():
        torch.ops.trtllm.cute_dsl_nvfp4_gemm_inplace_rubin(
            act_fp4,
            first_shard["weight"],
            act_sf,
            first_shard["weight_scale"],
            alpha,
            dtype,
            False,
            use_tvm_ffi,
            output_mixed,
            0,
        )

    mixed_tactic = None
    for tactic in capture:
        _, tactic_value = tactic[0]
        if (isinstance(tactic_value, tuple)
                and tactic_value[0] == "mixed_clusters"
                and tactic_value[5] == swap_ab):
            mixed_tactic = tactic
            break
    if mixed_tactic is None:
        pytest.skip(
            f"No mixed_clusters tactic is available for swap_ab={swap_ab}")

    output_mixed.zero_()
    with torch.inference_mode():
        output_base = base_linear.forward(input_tensor)
        for partition_id, shard in enumerate(ugpu_linear._ugpu_weight_shards):
            with AutoTuner.get().replay(mixed_tactic):
                torch.ops.trtllm.cute_dsl_nvfp4_gemm_inplace_rubin(
                    act_fp4,
                    shard["weight"],
                    act_sf,
                    shard["weight_scale"],
                    alpha,
                    dtype,
                    False,
                    use_tvm_ffi,
                    output_mixed,
                    partition_id,
                )

    torch.cuda.synchronize()
    torch.testing.assert_close(output_base,
                               output_mixed[:, :output_size],
                               rtol=1e-2,
                               atol=0.15)


def test_nvfp4_gemm_fake_rejects_legacy_output_args():
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available")

    from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401

    with FakeTensorMode():
        act_fp4 = torch.empty((2, 4), dtype=torch.uint8, device="cuda")
        weight = torch.empty((8, 4), dtype=torch.uint8, device="cuda")
        act_sf = torch.empty((2, 1), dtype=torch.uint8, device="cuda")
        weight_scale = torch.empty((8, 1), dtype=torch.uint8, device="cuda")
        alpha = torch.empty((1, ), dtype=torch.float32, device="cuda")
        output_tensor = torch.empty((2, 16),
                                    dtype=torch.bfloat16,
                                    device="cuda")

        with pytest.raises(ValueError, match="nvfp4_gemm_inplace"):
            torch.ops.trtllm.nvfp4_gemm(
                act_fp4,
                weight,
                act_sf,
                weight_scale,
                alpha,
                torch.bfloat16,
                False,
                "cutlass",
                None,
                output_tensor,
                0,
            )

        with pytest.raises(ValueError, match="partition_id"):
            torch.ops.trtllm.nvfp4_gemm(
                act_fp4,
                weight,
                act_sf,
                weight_scale,
                alpha,
                torch.bfloat16,
                False,
                "cutlass",
                None,
                None,
                0,
            )


def test_cute_dsl_rubin_fake_rejects_legacy_output_args():
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available")

    import tensorrt_llm._torch.custom_ops  # noqa: F401

    try:
        op = torch.ops.trtllm.cute_dsl_nvfp4_gemm_rubin
    except AttributeError:
        pytest.skip("internal CuteDSL Rubin GEMM op is not registered")

    with FakeTensorMode():
        act_fp4 = torch.empty((2, 4), dtype=torch.uint8, device="cuda")
        weight = torch.empty((8, 4), dtype=torch.uint8, device="cuda")
        act_sf = torch.empty((2, 1), dtype=torch.uint8, device="cuda")
        weight_scale = torch.empty((8, 1), dtype=torch.uint8, device="cuda")
        alpha = torch.empty((1, ), dtype=torch.float32, device="cuda")
        output_tensor = torch.empty((2, 16),
                                    dtype=torch.bfloat16,
                                    device="cuda")

        with pytest.raises(ValueError,
                           match="cute_dsl_nvfp4_gemm_inplace_rubin"):
            op(
                act_fp4,
                weight,
                act_sf,
                weight_scale,
                alpha,
                torch.bfloat16,
                False,
                True,
                output_tensor,
                0,
            )

        with pytest.raises(ValueError, match="partition_id"):
            op(
                act_fp4,
                weight,
                act_sf,
                weight_scale,
                alpha,
                torch.bfloat16,
                False,
                True,
                None,
                0,
            )


def fp4_linear_perf_test(dtype, SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE):
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(in_features=HIDDEN_SIZE,
                   out_features=OUTPUT_SIZE,
                   bias=False,
                   dtype=dtype,
                   quant_config=qc,
                   nvfp4_allowed_backends=['cutedsl'])

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

    l_fp4.load_weights([{
        'input_scale':
        1.0 / x_sf_global.cpu(),  # Simulates amax/(448*6) in modelopt ckpt
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(
            torch.float8_e4m3fn),  # Simulates float8_e4m3fn in modelopt ckpt
        'weight_scale_2':
        1.0 / w_sf_global.cpu()  # Simulates amax/(448*6) in modelopt ckpt
    }])
    l_fp4 = l_fp4.cuda()

    torch.testing.assert_close(l_fp4.weight, w_fp4)
    torch.testing.assert_close(l_fp4.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4.alpha[0], alpha_ref)

    with torch.inference_mode(), autotune():
        output = l_fp4.forward(x)

    l_fp4_ref = Linear(in_features=HIDDEN_SIZE,
                       out_features=OUTPUT_SIZE,
                       bias=False,
                       dtype=dtype,
                       quant_config=qc,
                       nvfp4_allowed_backends=['cutlass'
                                               ])  # Use CUTLASS as reference

    assert l_fp4_ref.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4_ref.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

    l_fp4_ref.load_weights([{
        'input_scale':
        1.0 / x_sf_global.cpu(),  # Simulates amax/(448*6) in modelopt ckpt
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(
            torch.float8_e4m3fn),  # Simulates float8_e4m3fn in modelopt ckpt
        'weight_scale_2':
        1.0 / w_sf_global.cpu()  # Simulates amax/(448*6) in modelopt ckpt
    }])
    l_fp4_ref = l_fp4_ref.cuda()

    torch.testing.assert_close(l_fp4_ref.weight, w_fp4)
    torch.testing.assert_close(l_fp4_ref.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4_ref.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4_ref.alpha[0], alpha_ref)

    with torch.inference_mode(), autotune():
        output_ref = l_fp4_ref.forward(x)

    for _ in range(5):
        output = l_fp4.forward(x)

    for i in range(10):
        output = l_fp4.forward(x)

    for _ in range(5):
        output_ref = l_fp4_ref.forward(x)

    for i in range(10):
        output_ref = l_fp4_ref.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)


# cold L2 cache for benchmarking (using circular buffer)
def nvfp4_gemm_perf_test(
    dtype,
    SEQ_LEN,
    OUTPUT_SIZE,
    HIDDEN_SIZE,
    test_ref=True,
    use_cold_l2_cache=True,
    warmup_iterations=2,
    iterations=1000,
):
    import cutlass.cute as cute
    import nvtx

    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()
    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)
    x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(x, x_sf_global,
                                                      scaling_vector_size,
                                                      False)

    if use_cold_l2_cache:
        one_workspace_bytes = (x_fp4.numel() * x_fp4.element_size() +
                               w_fp4.numel() * w_fp4.element_size() +
                               x_sf_block.numel() * x_sf_block.element_size() +
                               w_sf_block.numel() * w_sf_block.element_size())
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations)
        x_fp4_list = [x_fp4]
        w_fp4_list = [w_fp4]
        x_sf_block_list = [x_sf_block]
        w_sf_block_list = [w_sf_block]
        for _ in range(workspace_count - 1):
            x_fp4_list.append(x_fp4.clone())
            w_fp4_list.append(w_fp4.clone())
            x_sf_block_list.append(x_sf_block.clone())
            w_sf_block_list.append(w_sf_block.clone())
    else:
        workspace_count = 1
        x_fp4_list = [x_fp4]
        w_fp4_list = [w_fp4]
        x_sf_block_list = [x_sf_block]
        w_sf_block_list = [w_sf_block]

    alpha_tensor = torch.tensor([1.0]).cuda()
    with torch.inference_mode(), autotune():
        with nvtx.annotate(
                f"cute_dsl tune, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="orange",
        ):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_tensor, dtype)
    from tensorrt_llm._torch.autotuner import AutoTuner
    AutoTuner.get().print_statistics()

    if test_ref:
        with nvtx.annotate(
                f"ref tune, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="orange"):
            with torch.inference_mode(), autotune():
                output_ref = torch.ops.trtllm.nvfp4_gemm_cutlass(
                    x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_tensor, dtype)
        torch.testing.assert_close(output, output_ref)
        print(f"PASSED")

    buffer_idx = 0
    with nvtx.annotate(
            f"cute_dsl warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
            color="green"):
        for _ in range(warmup_iterations):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4_list[buffer_idx % workspace_count],
                w_fp4_list[buffer_idx % workspace_count],
                x_sf_block_list[buffer_idx % workspace_count],
                w_sf_block_list[buffer_idx % workspace_count],
                alpha_tensor,
                dtype,
            )
            buffer_idx = buffer_idx + 1

    with nvtx.annotate(
            f"cute_dsl run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
            color="green"):
        for i in range(iterations):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4_list[buffer_idx % workspace_count],
                w_fp4_list[buffer_idx % workspace_count],
                x_sf_block_list[buffer_idx % workspace_count],
                w_sf_block_list[buffer_idx % workspace_count],
                alpha_tensor,
                dtype,
            )
            buffer_idx = buffer_idx + 1

    if test_ref:
        torch.testing.assert_close(output, output_ref)
        print(f"PASSED")

        buffer_idx = 0
        with nvtx.annotate(
                f"ref warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="red"):
            for _ in range(warmup_iterations):
                output_ref = torch.ops.trtllm.nvfp4_gemm_cutlass(
                    x_fp4_list[buffer_idx % workspace_count],
                    w_fp4_list[buffer_idx % workspace_count],
                    x_sf_block_list[buffer_idx % workspace_count],
                    w_sf_block_list[buffer_idx % workspace_count],
                    alpha_tensor,
                    dtype,
                )
                buffer_idx = buffer_idx + 1
        with nvtx.annotate(
                f"ref run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="red"):
            for i in range(iterations):
                output_ref = torch.ops.trtllm.nvfp4_gemm_cutlass(
                    x_fp4_list[buffer_idx % workspace_count],
                    w_fp4_list[buffer_idx % workspace_count],
                    x_sf_block_list[buffer_idx % workspace_count],
                    w_sf_block_list[buffer_idx % workspace_count],
                    alpha_tensor,
                    dtype,
                )
                buffer_idx = buffer_idx + 1


@skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "mnk",
    [
        # Small batch sizes (M <= 16) - test small M handling
        (1, 4096, 4096, "Batch=1, Square 4K"),
        (4, 4096, 4096, "Batch=4, Square 4K"),
        (16, 4096, 4096, "Batch=16, Square 4K"),

        # Odd M values
        (3, 4096, 4096, "Odd M: M=3"),
        (7, 4096, 4096, "Odd M: M=7"),
        (9, 4096, 4096, "Odd M: M=9"),

        # Medium batch sizes - common inference scenarios
        (128, 4096, 4096, "Batch=128, Square 4K"),
        (128, 7168, 16384, "Batch=128, Large K/N"),
        (128, 4096, 7168, "Batch=128, Asymmetric"),

        # Large batch sizes - training scenarios
        (512, 4096, 4096, "Batch=512, Square 4K"),
        (1024, 4096, 4096, "Batch=1024, Square 4K"),

        # Very large batch - maximum performance
        (2048, 4096, 4096, "Batch=2048, Square 4K"),
        (4096, 4096, 4096, "Batch=4096, Square 4K"),

        # Large K and N - test memory bandwidth
        (128, 8192, 8192, "Batch=128, Square 8K"),
        (256, 16384, 16384, "Batch=256, Square 16K"),

        # Size asymmetry tests
        (1024, 128, 4096, "Wide M: M >> N"),
        (128, 16384, 128, "Wide N: N >> K"),
    ])
def test_nvfp4_gemm_unified_all_tactics(dtype, mnk):
    """Test nvfp4_gemm with auto backend selection, ensuring all tactics are tested."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE

    # Unpack mnk with optional description
    if len(mnk) == 4:
        SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE, desc = mnk
    else:
        SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
        desc = f"M={SEQ_LEN}, K={HIDDEN_SIZE}, N={OUTPUT_SIZE}"
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    # Prepare input
    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)
        alpha_ref = 1.0 / (w_sf_global * x_sf_global)
        alpha_tensor = torch.tensor([alpha_ref], dtype=torch.float32).cuda()

    # Reference: Use CUTLASS backend explicitly for reference output
    with torch.inference_mode():
        output_ref = torch.ops.trtllm.nvfp4_gemm(act_fp4=x_fp4,
                                                 weight=w_fp4,
                                                 act_sf=x_sf_block,
                                                 weight_scale=w_sf_block,
                                                 alpha=alpha_tensor,
                                                 output_dtype=dtype,
                                                 output_buffer_kind=0,
                                                 allowed_backends='cutlass')

    # Test auto backend selection with autotuning
    with torch.inference_mode(), autotune():
        output_auto = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            output_buffer_kind=0,
            allowed_backends='cutlass,cublaslt,cuda_core,cutedsl')

    AutoTuner.get().print_profiling_cache()

    # Verify auto mode result matches reference
    torch.cuda.synchronize()
    torch.testing.assert_close(output_auto, output_ref, rtol=1e-2, atol=0.15)

    # Test all combinations of outer layer (backend selection) and inner layer (backend tactics)
    # Outer layer: nvfp4_gemm selects backend
    # Inner layer: each backend has its own tactics
    from collections import defaultdict

    print(f"\n{'='*80}")
    print(f"Testing nvfp4_gemm (2-layer tactics): {desc}")
    print(f"Shape: M={SEQ_LEN}, K={HIDDEN_SIZE}, N={OUTPUT_SIZE}")
    print(f"{'='*80}")

    print(f"\n[Outer Layer] Capturing backend selection tactics...")
    with AutoTuner.get().capture() as outer_capture, torch.inference_mode():
        output = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            output_buffer_kind=0,
            allowed_backends='cutlass,cublaslt,cuda_core,cutedsl')

    outer_tactics_list = list(outer_capture)
    print(f"  Found {len(outer_tactics_list)} outer layer tactics (backends)")

    # Parse outer tactics to get backend names
    backend_map = {}
    for outer_tactic in outer_tactics_list:
        outer_runner, backend_name = outer_tactic[0]
        backend_map[backend_name] = outer_tactic
        print(f"    - Backend: {backend_name}")

    print(f"\n[Inner Layer] Testing tactics for each backend...")

    # All backends have independent APIs, but cuda_core needs special handling, because it requires unswizzled scale factors
    backend_apis = {}
    if IS_CUTLASS_DSL_AVAILABLE:
        if 'cutlass' in backend_map:
            backend_apis['cutlass'] = torch.ops.trtllm.nvfp4_gemm_cutlass
    if IS_CUBLASLT_AVAILABLE:
        if 'cublaslt' in backend_map:
            backend_apis['cublaslt'] = torch.ops.trtllm.nvfp4_gemm_cublaslt
    if IS_CUTLASS_DSL_AVAILABLE:
        if 'cutedsl' in backend_map:
            backend_apis[
                'cutedsl'] = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell

    # cuda_core needs special handling (different parameters, single tactic)
    test_cuda_core = 'cuda_core' in backend_map

    # Step 3: For each backend, capture and immediately test all tactics
    # Must test immediately after capture to avoid _last_capture being overwritten
    tactics_by_backend = defaultdict(list)
    total_tactics_tested = 0

    for backend_name, backend_api in backend_apis.items():
        print(f"\n  Backend: {backend_name}")

        # Capture inner tactics for this backend
        with AutoTuner.get().capture() as inner_capture, torch.inference_mode():
            output = backend_api(
                x_fp4,  # input/act_fp4
                w_fp4,  # weight
                x_sf_block,  # input_scale/act_sf
                w_sf_block,  # weight_scale
                alpha_tensor,  # alpha
                dtype  # output_dtype
            )

        inner_tactics_list = list(inner_capture)
        print(f"    Found {len(inner_tactics_list)} inner tactics")

        # Verify tactics uniqueness (ensure we're testing different tactics, not repeating the same one)
        tactic_values = [t[0][1] for t in inner_tactics_list]
        unique_tactics = len(set(tactic_values))
        assert len(tactic_values) == unique_tactics, \
            f"Duplicate tactics detected! Total: {len(tactic_values)}, Unique: {unique_tactics}"

        # Test each tactic immediately (while _last_capture is still valid)
        for tactic_idx, inner_tactic in enumerate(inner_tactics_list):
            inner_runner, inner_tactic_value = inner_tactic[0]
            runner_name = inner_runner.__class__.__name__

            # Replay this tactic
            with AutoTuner.get().replay(inner_tactic), torch.inference_mode():
                # Call backend API directly (using positional args)
                output = backend_api(
                    x_fp4,  # input/act_fp4
                    w_fp4,  # weight
                    x_sf_block,  # input_scale/act_sf
                    w_sf_block,  # weight_scale
                    alpha_tensor,  # alpha
                    dtype  # output_dtype
                )

                # Verify correctness
                torch.testing.assert_close(output,
                                           output_ref,
                                           rtol=1e-2,
                                           atol=0.15)

            total_tactics_tested += 1
            tactics_by_backend[runner_name].append(total_tactics_tested)
            print(f"    ✓ Tactic {tactic_idx+1}/{len(inner_tactics_list)}: "
                  f"{runner_name} tactic={inner_tactic_value} - PASSED")

    # Step 4: Test cuda_core if it's available (single tactic, no capture needed)
    if test_cuda_core:
        print(f"\n  Backend: cuda_core")
        print(f"    Found 1 tactic (single implementation, no autotuning)")

        with torch.inference_mode():
            output_cuda_core = torch.ops.trtllm.nvfp4_gemm(
                act_fp4=x_fp4,
                weight=w_fp4,
                act_sf=x_sf_block,
                weight_scale=w_sf_block,
                alpha=alpha_tensor,
                output_dtype=dtype,
                output_buffer_kind=0,
                allowed_backends='cuda_core')

            torch.testing.assert_close(output_cuda_core,
                                       output_ref,
                                       rtol=1e-2,
                                       atol=0.15)

        total_tactics_tested += 1
        tactics_by_backend['CudaCoreNVFP4Runner'].append(total_tactics_tested)
        print(f"    ✓ Tactic 1/1: CudaCoreNVFP4Runner tactic=0 - PASSED")

    print(f"\n{'='*80}")
    print(f"All {total_tactics_tested} tactics verified successfully!")
    print(f"\nBreakdown by backend:")
    for runner_name, indices in tactics_by_backend.items():
        print(f"  - {runner_name}: {len(indices)} tactics")
    if test_cuda_core:
        print(f"\n  Note: cuda_core has no autotuning (single tactic)")
    print(f"  Note: Tested all inner layer tactics for each backend")
    print(
        f"  Outer layer (backend selection) was tested separately with all backends allowed"
    )
    print(f"{'='*80}\n")


@pytest.mark.skipif(
    get_sm_version() not in [100, 103],
    reason="This test is only supported in Blackwell architecture",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(128, 7168, 16384), (128, 24576, 1536),
                                 (128, 2112, 7168), (128, 4096, 7168),
                                 (128, 7168, 2048), [127, 1024, 3200]])
def test_fp4_linear_cublaslt(dtype, mnk):
    """Test cuBLASLt FP4 GEMM implementation and compare with nvfp4_gemm_cutlass"""
    from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE
    if not IS_CUBLASLT_AVAILABLE:
        pytest.skip("cuBLASLt FP4 GEMM not available in this build")

    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)

        alpha_ref = 1.0 / (w_sf_global * x_sf_global)
        alpha_tensor = torch.tensor(alpha_ref, dtype=torch.float32).cuda()

        # Use cuBLASLt FP4 GEMM with autotuning support
        with autotune():
            output_cublaslt = torch.ops.trtllm.nvfp4_gemm_cublaslt(
                act_fp4=x_fp4,
                weight=w_fp4,
                act_sf=x_sf_block,
                weight_scale=w_sf_block,
                alpha=alpha_tensor,
                output_dtype=dtype)

    # Reference implementation: use torch.ops.trtllm.nvfp4_gemm_cutlass (CUTLASS)
    with torch.inference_mode():
        output_cutlass = torch.ops.trtllm.nvfp4_gemm_cutlass(
            x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref, dtype)

    # Compare results
    torch.cuda.synchronize()
    torch.testing.assert_close(output_cublaslt, output_cutlass)


@pytest.mark.skipif(
    get_sm_version() < 100,
    reason="CUDA Core backend requires SM >= 100 (Blackwell or newer)",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(1, 4096, 7168), (4, 7168, 16384),
                                 (8, 2112, 7168)])
def test_fp4_linear_cuda_core(dtype, mnk):
    """Test CUDA Core NVFP4 GEMM implementation on SM >= 100 (M <= 8)"""

    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)

        alpha_ref = 1.0 / (w_sf_global * x_sf_global)
        alpha_tensor = torch.tensor(alpha_ref, dtype=torch.float32).cuda()

        # Reference: Use CUTLASS backend
        output_ref = torch.ops.trtllm.nvfp4_gemm(act_fp4=x_fp4,
                                                 weight=w_fp4,
                                                 act_sf=x_sf_block,
                                                 weight_scale=w_sf_block,
                                                 alpha=alpha_tensor,
                                                 output_dtype=dtype,
                                                 output_buffer_kind=0,
                                                 allowed_backends='cutlass')

        # Test CUDA Core backend
        output_cuda_core = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            output_buffer_kind=0,
            allowed_backends='cuda_core')

    # Compare results
    torch.cuda.synchronize()
    torch.testing.assert_close(output_cuda_core,
                               output_ref,
                               rtol=1e-2,
                               atol=0.15)
    print(
        f"✓ CUDA Core test passed for M={SEQ_LEN}, N={OUTPUT_SIZE}, K={HIDDEN_SIZE}"
    )


@pytest.mark.skipif(
    not (89 <= get_sm_version() < 100 or get_sm_version() in (120, 121)),
    reason="Dense Marlin NVFP4 runs on SM89-99 and SM120/121",
)
@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "mnk",
    [
        (1, 1024, 1024),
        (8, 1024, 2048),
        (128, 2048, 1024),
        (1, 18560, 4096),
        (128, 18560, 4096),
        (1, 4096, 8192),
        (128, 4096, 8192),
        # Non-64-aligned N and/or K (e.g. from TP sharding)
        (8, 2576, 672),
        (128, 2576, 672),
        (8, 2576, 4096),
        (8, 4096, 2576),
        (8, 2576, 544),
        (1, 96, 80),
        (3, 176, 144),
        (128, 928, 1360),
    ])
def test_fp4_linear_marlin(quant_algo, dtype, mnk):
    if quant_algo == QuantAlgo.NVFP4 and get_sm_version() in (120, 121):
        pytest.skip(
            "Marlin backend shouldn't be used for NVFP4 quant on SM120/121")
    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    w_float = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=torch.float32)
    w_fp4, w_sf_swizzled, w_dequant = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
        w_float,
        scaling_vector_size,
        1,  # ufp8_type=1 (e4m3)
        True,  # is_sf_swizzled_layout=True (modelopt checkpoint native layout)
    )
    assert torch.iinfo(w_sf_swizzled.dtype).bits == 8  # torch.uint8
    w_sf_2d = torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_swizzled.view(pad_up(OUTPUT_SIZE, 128),
                           -1)).view(torch.float8_e4m3fn)

    with model_extra_attrs({'nvfp4_gemm_allowed_backends': ['marlin']}):
        l_marlin = Linear(
            in_features=HIDDEN_SIZE,
            out_features=OUTPUT_SIZE,
            bias=False,
            dtype=dtype,
            quant_config=QuantConfig(quant_algo=quant_algo),
            nvfp4_allowed_backends=['marlin'],  # key
        )

        # ``float_to_e2m1_and_ufp8sf_scale`` returns ``w_dequant`` that already
        # encodes the per-block FP8 scale. The Marlin BF16-activation path
        # multiplies the kernel output by ``weight_global_scale`` (derived from
        # ``weight_scale_2``); we want that scalar to be 1 so the kernel result
        # matches the reference ``torch.mm(x, w_dequant.T)``. Mirrors the
        # passing GEMM test (test_fp4_gemm.py:453-454, is_bf16_act=True branch).
        l_marlin.load_weights([{
            'weight':
            w_fp4,
            'weight_scale':
            w_sf_2d,
            'weight_scale_2':
            torch.tensor(1.0, dtype=torch.float32),
        }])
        l_marlin = l_marlin.cuda()

        l_marlin.post_load_weights()

        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
        x_sf_global = (448 * 6) / x.abs().max().float()
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)

        with torch.inference_mode():
            output = l_marlin(x)

    w_dequant_bf16 = w_dequant.to(dtype).cuda()
    with torch.inference_mode():
        ref_output = torch.mm(x, w_dequant_bf16.T)

    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, atol=0.5, rtol=2e-2)


if __name__ == "__main__":
    # m, n, k
    nvfp4_gemm_perf_test(torch.bfloat16, 128, 7168, 16384)

    # # group-1 test cases
    # for tokens in [128, 8192]:
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 7168, 16384)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 24576, 1536)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 2112, 7168)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 4096, 7168)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 7168, 2048)

    # # group-2 test cases
    # for m in [128, 256, 512]:
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 131584, 7168)
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 7168, 65792)
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 227368, 2560, test_ref=False)
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 2560, 113664)


def _make_nvfp4_inputs_for_bias_test(m, n, k, dtype=torch.bfloat16):
    """Quantize random bf16 act+weight; return (act_fp4, weight_fp4, act_sf, weight_sf, alpha)."""
    torch.manual_seed(0)
    act = torch.randn(m, k, dtype=dtype, device="cuda")
    weight = torch.randn(n, k, dtype=dtype, device="cuda")
    act_gs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    w_gs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(act, act_gs, 16, False,
                                                    True)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(weight, w_gs, 16, False, True)
    alpha = (1.0 / act_gs * 1.0 / w_gs).to(torch.float32).reshape(())
    return act_fp4, w_fp4, act_sf, w_sf, alpha


@skip_pre_blackwell
@pytest.mark.parametrize("backend", ["cutlass", "cublaslt", "cuda_core"])
@pytest.mark.parametrize("mnk", [(8, 4096, 4096), (252, 2048, 2048),
                                 (1024, 4096, 4096), (4096, 6144, 4096)])
def test_fp4_gemm_bias_per_backend(backend, mnk):
    """Per-backend numerical parity: nvfp4_gemm(bias=B) ≈ nvfp4_gemm(bias=None) + B."""
    m, n, k = mnk
    if backend == "cuda_core" and m > 8:
        pytest.skip("cuda_core backend only supports M <= 8")

    act_fp4, w_fp4, act_sf, w_sf, alpha = _make_nvfp4_inputs_for_bias_test(
        m, n, k)
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.5

    out_no_bias = torch.ops.trtllm.nvfp4_gemm(act_fp4,
                                              w_fp4,
                                              act_sf,
                                              w_sf,
                                              alpha,
                                              torch.bfloat16,
                                              allowed_backends=backend)
    ref = out_no_bias + bias

    out_fused = torch.ops.trtllm.nvfp4_gemm(act_fp4,
                                            w_fp4,
                                            act_sf,
                                            w_sf,
                                            alpha,
                                            torch.bfloat16,
                                            allowed_backends=backend,
                                            bias=bias)

    # bf16 1-ULP at cast boundary (~0.0039) — fused vs gemm+add differs by ULP-level rounding.
    check_accuracy(out_fused, ref, atol=5e-3, rtol=1e-2, percent=0.998)
