# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4, SM107, attention TP8 and MoE TP2 x EP4: a context-only (prefill) target for a
disaggregated ctx worker. Every step runs the generic path; no decode state is built."""

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..kimi_k3_nvfp4.implementation import K3TargetSpec, ModelingV2KimiK3
from ..kimi_k3_nvfp4.implementation import _check_construction as _check_target_construction

_TARGET = K3TargetSpec(
    sm=(10, 7), quant_algo=QuantAlgo.NVFP4, moe_tp=2, moe_ep=4, tp=8, ctx_only=True
)


def _check_construction(model_config: ModelConfig) -> None:
    _check_target_construction(model_config, _TARGET)


class ModelingV2KimiK3Nvfp4Sm107Tp8Moetp2ep4(ModelingV2KimiK3):
    """Concrete deployment of the shared Kimi K3 text model (context only)."""

    target_spec = _TARGET
