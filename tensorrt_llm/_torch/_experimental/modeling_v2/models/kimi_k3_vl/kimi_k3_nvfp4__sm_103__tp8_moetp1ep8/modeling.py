# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4, SM103, attention TP8 and MoE TP1 x EP8: a context-only (prefill) target for a
disaggregated ctx worker. Every step runs the generic path; no decode state is built."""

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..kimi_k3_nvfp4.implementation import K3TargetSpec, ModelingV2KimiK3
from ..kimi_k3_nvfp4.implementation import _check_construction as _check_target_construction

_TARGET = K3TargetSpec(
    sm=(10, 3), quant_algo=QuantAlgo.NVFP4, moe_tp=1, moe_ep=8, tp=8, ctx_only=True
)


def _check_construction(model_config: ModelConfig) -> None:
    _check_target_construction(model_config, _TARGET)


class ModelingV2KimiK3Nvfp4Sm103Tp8Moetp1ep8(ModelingV2KimiK3):
    """Concrete deployment of the shared Kimi K3 text model (context only)."""

    target_spec = _TARGET
