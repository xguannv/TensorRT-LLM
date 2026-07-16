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
"""
uGPU Execution Planning: centralized enable/disable decisions.

All scattered checks (IS_CUTLASS_DSL_AVAILABLE, is_ugpu_enabled, has_nvfp4,
weight_mode==VANILLA, out_features%2==0, etc.) are consolidated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from tensorrt_llm._torch.ugpu.layout import PartitionedTensorLayout, make_nvfp4_linear_output_layout


@dataclass(frozen=True)
class UgpuPolicy:
    """Top-level uGPU configuration, typically stored in ModelConfig."""

    enabled: bool = False
    num_partitions: int = 2
    allowed_ops: frozenset = field(
        default_factory=lambda: frozenset({"nvfp4_linear", "nvfp4_moe", "bf16_moe"})
    )
    allowed_backends: tuple = ("cutlass", "cublaslt", "cuda_core")

    def __post_init__(self):
        # Runtime (streams, mempools, ugpu_device) only supports 2 partitions.
        if self.num_partitions != 2:
            raise ValueError(
                f"uGPU only supports num_partitions=2, got {self.num_partitions}. "
                f"Runtime resources (streams, mempools, TPC masks) are "
                f"hardcoded for exactly 2 partitions."
            )


@dataclass(frozen=True)
class PartitionPlan:
    """Base class for partition decisions."""

    enabled: bool
    num_partitions: int = 2
    backend: str = "cutedsl"
    merge_kind: Literal["concat", "scatter_add", "none"] = "concat"
    reason_if_disabled: Optional[str] = None


@dataclass(frozen=True)
class LinearPartitionPlan(PartitionPlan):
    """Partition plan specific to Linear layers."""

    partition_axis: int = 0  # partition along output dimension
    layout: Optional[PartitionedTensorLayout] = None


# Singleton disabled plan
DISABLED_PLAN = LinearPartitionPlan(
    enabled=False,
    reason_if_disabled="uGPU not enabled or not applicable",
)


class UgpuExecutionPlanner:
    """Produces PartitionPlans based on hardware capability and module config.

    Centralizes all enable/disable logic that was previously scattered in:
    - Linear.maybe_create_ugpu_sub_modules() (linear.py:2704-2708)
    - attention.py: is_ugpu_supported() backend patching
    - fused_moe_cute_dsl.py: CuteDslGroupGemmMLP.__init__
    """

    def __init__(self, policy: UgpuPolicy):
        self.policy = policy

    def plan_linear(
        self,
        in_features: int,
        out_features: int,
        quant_config,
        weight_mode,
    ) -> LinearPartitionPlan:
        """Decide whether to partition a Linear layer for uGPU execution.

        Returns LinearPartitionPlan with enabled=True only if ALL conditions are met.
        The planner owns the backend decision (always cutedsl for uGPU) — callers
        do NOT need to pre-add 'cutedsl' to their allowed_backends list.
        """
        from tensorrt_llm._torch.cute_dsl_utils import (
            IS_CUTLASS_DSL_AVAILABLE,
            IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
        )
        from tensorrt_llm._torch.modules.linear import WeightMode
        from tensorrt_llm._torch.ugpu_utils import is_ugpu_enabled

        # Policy check
        if not self.policy.enabled:
            return LinearPartitionPlan(enabled=False, reason_if_disabled="UgpuPolicy is disabled")

        # Hardware support
        if not is_ugpu_enabled():
            return LinearPartitionPlan(
                enabled=False, reason_if_disabled="uGPU not supported on this hardware"
            )

        # Op type check
        if "nvfp4_linear" not in self.policy.allowed_ops:
            return LinearPartitionPlan(
                enabled=False, reason_if_disabled="nvfp4_linear not in allowed_ops"
            )

        # Quantization check
        is_nvfp4 = (
            quant_config is not None
            and hasattr(quant_config, "layer_quant_mode")
            and quant_config.layer_quant_mode.has_nvfp4()
        )
        if not is_nvfp4:
            return LinearPartitionPlan(enabled=False, reason_if_disabled="Not NVFP4 quantization")

        # uGPU uses Rubin-only CuteDSL kernels from the internal package.
        if not IS_CUTLASS_DSL_AVAILABLE:
            return LinearPartitionPlan(enabled=False, reason_if_disabled="CuteDSL not available")
        if not IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
            return LinearPartitionPlan(
                enabled=False, reason_if_disabled="CuteDSL internal Rubin kernels not available"
            )

        # Weight mode check (only vanilla supported)
        if weight_mode != WeightMode.VANILLA:
            return LinearPartitionPlan(
                enabled=False,
                reason_if_disabled=f"Weight mode {weight_mode} not supported, "
                f"only VANILLA is supported for uGPU",
            )

        layout = make_nvfp4_linear_output_layout(
            out_features,
            in_features,
            self.policy.num_partitions,
        )
        layout_reason = layout.disabled_reason_for_padding_free_split()
        if layout_reason is not None:
            return LinearPartitionPlan(enabled=False, reason_if_disabled=layout_reason)

        return LinearPartitionPlan(
            enabled=True,
            num_partitions=self.policy.num_partitions,
            backend="cutedsl",
            merge_kind="concat",
            partition_axis=0,
            layout=layout,
        )

    def plan_moe(
        self,
        quant_config,
        *,
        moe_backend: str = "CUTEDSL",
        use_fused_finalize: bool = True,
        dtype_activation: torch.dtype = torch.bfloat16,
    ) -> PartitionPlan:
        """Decide whether to partition a MoE GroupGemm for uGPU execution.

        MoE uGPU replicates weights on each partition (not partitioned like Linear).
        Each partition runs the full GroupGemm with inplace output into shared buffers.
        """
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_INTERNAL_AVAILABLE
        from tensorrt_llm._torch.ugpu_utils import is_ugpu_enabled

        if not self.policy.enabled:
            return PartitionPlan(enabled=False, reason_if_disabled="UgpuPolicy is disabled")

        if not is_ugpu_enabled():
            return PartitionPlan(
                enabled=False, reason_if_disabled="uGPU not supported on this hardware"
            )

        if moe_backend.upper() != "CUTEDSL":
            return PartitionPlan(
                enabled=False,
                reason_if_disabled=f"uGPU MoE requires CuteDSL backend, got {moe_backend}",
            )

        is_nvfp4 = (
            quant_config is not None
            and hasattr(quant_config, "quant_mode")
            and quant_config.quant_mode.has_nvfp4()
        )
        has_any_quant = (
            quant_config is not None
            and hasattr(quant_config, "quant_mode")
            and quant_config.quant_mode.has_any_quant()
        )
        is_bf16 = not has_any_quant and dtype_activation == torch.bfloat16

        if is_nvfp4:
            op_name = "nvfp4_moe"
        elif is_bf16:
            op_name = "bf16_moe"
        elif not has_any_quant:
            return PartitionPlan(
                enabled=False,
                reason_if_disabled=f"BF16 uGPU MoE requires bfloat16 activation, got {dtype_activation}",
            )
        else:
            return PartitionPlan(
                enabled=False, reason_if_disabled="uGPU MoE only supports NVFP4 or BF16"
            )

        if op_name not in self.policy.allowed_ops:
            return PartitionPlan(enabled=False, reason_if_disabled=f"{op_name} not in allowed_ops")

        if not use_fused_finalize:
            return PartitionPlan(
                enabled=False, reason_if_disabled="uGPU MoE requires fused finalize"
            )

        if not IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
            return PartitionPlan(
                enabled=False, reason_if_disabled="CuteDSL internal Rubin kernels not available"
            )

        return PartitionPlan(
            enabled=True,
            num_partitions=self.policy.num_partitions,
            backend="cutedsl",
            merge_kind="none",
        )
