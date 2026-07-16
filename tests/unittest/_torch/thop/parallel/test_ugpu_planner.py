# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Unit tests for uGPU Execution Planner.

Tests cover:
- Policy dataclass behavior (frozen, defaults)
- PartitionPlan / LinearPartitionPlan creation
- PartitionedTensorLayout metadata
- UgpuExecutionPlanner enable/disable decisions
- Singleton DISABLED_PLAN
"""

from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.ugpu.layout import make_nvfp4_linear_output_layout
from tensorrt_llm._torch.ugpu.policy import (
    DISABLED_PLAN,
    LinearPartitionPlan,
    PartitionPlan,
    UgpuExecutionPlanner,
    UgpuPolicy,
)
from tensorrt_llm._torch.utils import model_extra_attrs


@pytest.fixture(autouse=True)
def _enable_cutedsl_internal(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_INTERNAL_AVAILABLE", True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeQuantMode:
    """Minimal stub for quant_config.layer_quant_mode."""

    def __init__(self, nvfp4: bool = True):
        self._nvfp4 = nvfp4

    def has_nvfp4(self):
        return self._nvfp4


class _FakeQuantConfig:
    """Minimal stub for QuantConfig."""

    def __init__(self, nvfp4: bool = True):
        self.layer_quant_mode = _FakeQuantMode(nvfp4)


# Import WeightMode lazily to avoid heavy module load
def _vanilla_weight_mode():
    from tensorrt_llm._torch.modules.linear import WeightMode

    return WeightMode.VANILLA


def _fused_qkv_weight_mode():
    from tensorrt_llm._torch.modules.linear import WeightMode

    return WeightMode.FUSED_QKV_LINEAR


# ---------------------------------------------------------------------------
# Policy dataclass tests
# ---------------------------------------------------------------------------


class TestUgpuPolicy:
    """Tests for UgpuPolicy frozen dataclass."""

    def test_default_values(self):
        policy = UgpuPolicy()
        assert not policy.enabled
        assert policy.num_partitions == 2
        assert "nvfp4_linear" in policy.allowed_ops
        assert "nvfp4_moe" in policy.allowed_ops
        assert "bf16_moe" in policy.allowed_ops

    def test_enabled(self):
        policy = UgpuPolicy(enabled=True)
        assert policy.enabled

    def test_frozen(self):
        policy = UgpuPolicy()
        with pytest.raises(AttributeError):
            policy.enabled = True

    def test_custom_partitions_rejected(self):
        # Runtime only supports num_partitions=2
        with pytest.raises(ValueError, match="only supports num_partitions=2"):
            UgpuPolicy(num_partitions=4)

    def test_default_partitions(self):
        policy = UgpuPolicy()
        assert policy.num_partitions == 2

    def test_custom_allowed_ops(self):
        policy = UgpuPolicy(allowed_ops=frozenset({"nvfp4_linear"}))
        assert "nvfp4_moe" not in policy.allowed_ops


# ---------------------------------------------------------------------------
# PartitionPlan tests
# ---------------------------------------------------------------------------


class TestPartitionPlan:
    """Tests for PartitionPlan and LinearPartitionPlan."""

    def test_disabled_plan(self):
        plan = PartitionPlan(enabled=False)
        assert not plan.enabled
        assert plan.num_partitions == 2
        assert plan.backend == "cutedsl"

    def test_enabled_plan(self):
        plan = PartitionPlan(enabled=True, num_partitions=2)
        assert plan.enabled
        assert plan.merge_kind == "concat"

    def test_linear_partition_plan(self):
        plan = LinearPartitionPlan(enabled=True)
        assert plan.partition_axis == 0
        assert isinstance(plan, PartitionPlan)

    def test_singleton_disabled_plan(self):
        assert not DISABLED_PLAN.enabled
        assert DISABLED_PLAN.reason_if_disabled is not None

    def test_frozen(self):
        plan = LinearPartitionPlan(enabled=True)
        with pytest.raises(AttributeError):
            plan.enabled = False


# ---------------------------------------------------------------------------
# UgpuExecutionPlanner tests
# ---------------------------------------------------------------------------


class TestPartitionedTensorLayout:
    """Tests for logical/padded partition metadata."""

    def test_nvfp4_linear_layout_slices(self):
        layout = make_nvfp4_linear_output_layout(
            out_features=7168,
            in_features=2048,
            num_partitions=2,
        )
        assert layout.logical_shape == (7168, 2048)
        assert layout.padded_shape == (7168, 2048)
        assert layout.partition_axis_slice(0, padded=False) == slice(0, 3584)
        assert layout.partition_axis_slice(1, padded=True) == slice(3584, 7168)
        assert layout.disabled_reason_for_padding_free_split() is None

    def test_nvfp4_linear_layout_reports_padding_gap(self):
        layout = make_nvfp4_linear_output_layout(
            out_features=7170,
            in_features=2048,
            num_partitions=2,
        )
        assert layout.logical_shape == (7170, 2048)
        assert layout.padded_shape == (7200, 2048)
        assert "NVFP4 row alignment" in layout.disabled_reason_for_padding_free_split()


class TestUgpuExecutionPlanner:
    """Tests for plan_linear() enable/disable decisions."""

    def _plan(
        self, policy=None, in_features=2048, out_features=7168, quant_config=None, weight_mode=None
    ):
        """Helper: run plan_linear with sensible defaults."""
        if policy is None:
            policy = UgpuPolicy(enabled=True)
        if quant_config is None:
            quant_config = _FakeQuantConfig(nvfp4=True)
        if weight_mode is None:
            weight_mode = _vanilla_weight_mode()
        planner = UgpuExecutionPlanner(policy)
        return planner.plan_linear(in_features, out_features, quant_config, weight_mode)

    # --- Policy checks ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_disabled_policy_disables(self, mock_ugpu):
        plan = self._plan(policy=UgpuPolicy(enabled=False))
        assert not plan.enabled
        assert "disabled" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_enabled_policy_enables_when_all_conditions_met(self, mock_ugpu):
        plan = self._plan(policy=UgpuPolicy(enabled=True))
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.num_partitions == 2

    # --- Hardware check ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=False)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_ugpu_not_supported_disables(self, mock_ugpu):
        plan = self._plan()
        assert not plan.enabled
        assert "hardware" in plan.reason_if_disabled.lower()

    # --- Op type check ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_op_not_allowed_disables(self, mock_ugpu):
        policy = UgpuPolicy(enabled=True, allowed_ops=frozenset({"nvfp4_moe"}))
        plan = self._plan(policy=policy)
        assert not plan.enabled
        assert "allowed_ops" in plan.reason_if_disabled

    # --- Quantization check ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_non_nvfp4_quant_disables(self, mock_ugpu):
        plan = self._plan(quant_config=_FakeQuantConfig(nvfp4=False))
        assert not plan.enabled
        assert "NVFP4" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_no_quant_config_disables(self, mock_ugpu):
        """quant_config=None should disable uGPU (no quantization)."""
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_linear(2048, 7168, None, _vanilla_weight_mode())
        assert not plan.enabled

    # --- CuteDSL availability ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", False)
    def test_cutedsl_unavailable_disables(self, mock_ugpu):
        plan = self._plan()
        assert not plan.enabled
        assert "CuteDSL" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_cutedsl_internal_unavailable_disables(self, mock_ugpu, monkeypatch):
        monkeypatch.setattr(
            "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_INTERNAL_AVAILABLE", False
        )
        plan = self._plan()
        assert not plan.enabled
        assert "internal" in plan.reason_if_disabled

    # --- Weight mode check ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_non_vanilla_weight_mode_disables(self, mock_ugpu):
        plan = self._plan(weight_mode=_fused_qkv_weight_mode())
        assert not plan.enabled
        assert "VANILLA" in plan.reason_if_disabled

    # --- Divisibility check ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_odd_out_features_disables(self, mock_ugpu):
        plan = self._plan(out_features=7169)
        assert not plan.enabled
        assert "divisible" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_unaligned_partition_out_features_disables(self, mock_ugpu):
        plan = self._plan(out_features=7170)
        assert not plan.enabled
        assert "row alignment" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_custom_partition_count_rejected(self, mock_ugpu):
        # Runtime only supports num_partitions=2
        with pytest.raises(ValueError, match="only supports num_partitions=2"):
            UgpuPolicy(num_partitions=4)

    # --- Enabled plan properties ---

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_enabled_plan_properties(self, mock_ugpu):
        plan = self._plan()
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.merge_kind == "concat"
        assert plan.partition_axis == 0
        assert plan.layout == make_nvfp4_linear_output_layout(7168, 2048, 2)
        assert plan.reason_if_disabled is None

    def test_linear_reads_policy_from_model_extra_attrs(self):
        from tensorrt_llm._torch.modules.linear import Linear

        with model_extra_attrs({"ugpu_policy": UgpuPolicy(enabled=False)}):
            linear = Linear(
                in_features=2048,
                out_features=7168,
                bias=False,
                quant_config=_FakeQuantConfig(nvfp4=True),
                skip_create_weights_in_init=True,
            )

        assert not linear.partition_plan.enabled
        assert "disabled" in linear.partition_plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_linear_copies_backend_list_before_ugpu_append(self, mock_ugpu):
        from tensorrt_llm._torch.modules.linear import Linear

        shared_backends = ["cutlass", "cublaslt", "cuda_core"]
        with model_extra_attrs(
            {
                "ugpu_policy": UgpuPolicy(enabled=True),
                "nvfp4_gemm_allowed_backends": shared_backends,
            }
        ):
            linear = Linear(
                in_features=2048,
                out_features=7168,
                bias=False,
                quant_config=_FakeQuantConfig(nvfp4=True),
                skip_create_weights_in_init=True,
            )

        assert shared_backends == ["cutlass", "cublaslt", "cuda_core"]
        assert linear.nvfp4_allowed_backends is not shared_backends
        assert linear.nvfp4_allowed_backends == [
            "cutlass",
            "cublaslt",
            "cuda_core",
            "cutedsl",
        ]

    def test_model_config_exports_ugpu_policy(self):
        from tensorrt_llm._torch.model_config import ModelConfig

        policy = UgpuPolicy(enabled=False)
        config = ModelConfig(ugpu_policy=policy)

        assert config.extra_attrs["ugpu_policy"] is policy


# ---------------------------------------------------------------------------
# MoE planner tests
# ---------------------------------------------------------------------------


class _FakeMoeQuantMode:
    """Stub for MoE quant_config.quant_mode."""

    def __init__(self, nvfp4: bool = True, any_quant: Optional[bool] = None):
        self._nvfp4 = nvfp4
        self._any_quant = nvfp4 if any_quant is None else any_quant

    def has_nvfp4(self):
        return self._nvfp4

    def has_any_quant(self):
        return self._any_quant


class _FakeMoeQuantConfig:
    """Stub for MoE QuantConfig."""

    def __init__(self, nvfp4: bool = True, any_quant: Optional[bool] = None):
        self.quant_mode = _FakeMoeQuantMode(nvfp4, any_quant)


class TestUgpuMoePlanner:
    """Tests for plan_moe() enable/disable decisions."""

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_enabled_when_nvfp4(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.merge_kind == "none"

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_enabled_when_bf16(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(None, dtype_activation=torch.bfloat16)
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.merge_kind == "none"

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_without_fused_finalize(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True), use_fused_finalize=False)
        assert not plan.enabled
        assert "fused finalize" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_when_backend_not_cutedsl(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True), moe_backend="CUTLASS")
        assert not plan.enabled
        assert "CuteDSL backend" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_when_quantized_but_not_nvfp4(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=False, any_quant=True))
        assert not plan.enabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_when_bf16_not_in_allowed_ops(self, mock_ugpu):
        policy = UgpuPolicy(enabled=True, allowed_ops=frozenset({"nvfp4_linear", "nvfp4_moe"}))
        planner = UgpuExecutionPlanner(policy)
        plan = planner.plan_moe(None, dtype_activation=torch.bfloat16)
        assert not plan.enabled
        assert "bf16_moe" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_when_unquantized_not_bf16(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(None, dtype_activation=torch.float16)
        assert not plan.enabled
        assert "bfloat16" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_when_policy_disabled(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=False))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled
        assert "disabled" in plan.reason_if_disabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=False)
    def test_moe_disabled_when_no_hardware(self, mock_ugpu):
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_when_not_in_allowed_ops(self, mock_ugpu):
        policy = UgpuPolicy(enabled=True, allowed_ops=frozenset({"nvfp4_linear"}))
        planner = UgpuExecutionPlanner(policy)
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled

    @patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_enabled", return_value=True)
    def test_moe_disabled_without_cutedsl_internal(self, mock_ugpu, monkeypatch):
        monkeypatch.setattr(
            "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_INTERNAL_AVAILABLE", False
        )
        planner = UgpuExecutionPlanner(UgpuPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled
        assert "internal" in plan.reason_if_disabled
