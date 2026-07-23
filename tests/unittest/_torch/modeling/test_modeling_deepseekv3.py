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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
    _dequantize_fp8_block_scaled_linear,
)
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _mixed_model_config(
    quant_config_dict: dict[str, QuantConfig],
) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            first_k_dense_replace=3,
            moe_layer_freq=1,
            n_routed_experts=256,
        ),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict=quant_config_dict,
    )


@pytest.mark.parametrize(
    ("layer_idx", "layer_quant_name", "expected_quant_algo"),
    [
        (0, "model.layers.0.mlp.down_proj", QuantAlgo.FP8_BLOCK_SCALES),
        (3, "model.layers.3.mlp.experts.0.down_proj", QuantAlgo.NVFP4),
    ],
)
def test_get_mixed_decoder_layer_quant_config(
    layer_idx: int,
    layer_quant_name: str,
    expected_quant_algo: QuantAlgo,
) -> None:
    layer_quant_config = QuantConfig(quant_algo=expected_quant_algo)
    model_config = _mixed_model_config({layer_quant_name: layer_quant_config})
    decoder_layer = object.__new__(DeepseekV3DecoderLayer)

    actual = decoder_layer._get_decoder_layer_quant_config(model_config, layer_idx)

    assert actual is layer_quant_config


def test_get_mixed_decoder_layer_quant_config_requires_layer_entry() -> None:
    model_config = _mixed_model_config({})
    decoder_layer = object.__new__(DeepseekV3DecoderLayer)

    with pytest.raises(
        ValueError, match="Cannot resolve the MoE quantization config for model.layers.3"
    ):
        decoder_layer._get_decoder_layer_quant_config(model_config, 3)


@patch(
    "tensorrt_llm._torch.models.modeling_deepseekv3.weight_dequant",
    return_value=torch.full((2, 3), 4.0, dtype=torch.bfloat16),
)
def test_dequantize_fp8_block_scaled_linear_uses_dense_method(
    mock_weight_dequant,
) -> None:
    linear = Linear(
        3,
        2,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
            kv_cache_quant_algo=QuantAlgo.FP8,
            group_size=128,
        ),
    )

    assert _dequantize_fp8_block_scaled_linear(linear)

    mock_weight_dequant.assert_called_once()
    assert mock_weight_dequant.call_args.kwargs["output_dtype"] == torch.bfloat16
    assert linear.weight.dtype == torch.bfloat16
    assert torch.equal(linear.weight, torch.full((2, 3), 4.0, dtype=torch.bfloat16))
    assert isinstance(linear.quant_method, UnquantizedLinearMethod)
    assert linear.quant_config.quant_algo is None
    assert linear.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
