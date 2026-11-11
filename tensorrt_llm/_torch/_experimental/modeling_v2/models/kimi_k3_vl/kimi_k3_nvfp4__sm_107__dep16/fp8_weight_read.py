# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), dep16: projections that read FP8 block-scaled weights.

``_Fp8BlockScaleWeightReadLinear`` keeps a bias-free projection as E4M3 codes with 128x128
block scales: the checkpoint FP8 attention projections, and the MoE layers' BF16 MLPs, which
the target quantizes after loading. The K3 low-M GEMM runs the low-M rows; the rest runs
DeepGEMM on SM100/SM103 and the native MXFP8 GEMM on SM107. ``_swap_linear_to_fp8_weight_read``
and ``_convert_moe_mlps_to_fp8_weight_read`` put it in place of a loaded linear.
"""

from __future__ import annotations

import gc
from typing import Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from ..ops.gemm.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm as k3_fp8w_lowm_gemm_op
from ..ops.gemm.k3_mxfp8_gemm import k3_mxfp8_gemm as k3_mxfp8_gemm_op
from .layers import Bf16Linear, Fp8BlockScaleLinear, fp8_block_scale_gemm, prepare_fp8_block_weight

# ---------------------------------------------------------------------------
# FP8 block-scale weight read for attention and replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Recomputed rather than cached: the SM version cannot change within a run,
    so weight preparation and forward always agree.
    """
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _is_fp8_block_projection(module: nn.Module) -> bool:
    """A checkpoint FP8 128 x 128 block-scale projection: the loader reads its codes and scales
    together and post-load swaps it for ``_Fp8BlockScaleWeightReadLinear``."""
    return isinstance(module, Fp8BlockScaleLinear)


def _k3_lowm_gemm_can_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    allow_partial_n_tile: bool = False,
) -> bool:
    """Whether the K3 low-M GEMM (``ops.gemm.k3_fp8w_lowm_gemm``) serves this projection."""
    from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm.can_run(
        x, weight, weight_scale, allow_partial_n_tile=allow_partial_n_tile
    )


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. On SM100/SM103 the UE8M0
    scales are packed for DeepGEMM, which quantizes BF16 activations inside the
    GEMM. On Rubin (SM107) the scales are packed into the CuTe DSL K32 R128c4
    layout and the activation is quantized by the specialized CuTe quantizer
    before the native MXFP8 GEMM, which runs with K3's fine-grained low-M tuning
    buckets. That dispatch is ``layers.fp8_block_scale_gemm``.

    The Rubin agentic kernels sit in front of that dispatch: the K3 low-M GEMM
    (``kernels/k3_fp8w_lowm_gemm.py``) owns the low-M rows (per-shape, per-architecture limit,
    see ``k3_fp8w_lowm_gemm.max_m_for``), so the generic low-M GEMM
    (``trtllm::fp8_block_scale_low_m_gemm``) is not prepared. On SM107 the KDA MTP verify's
    o_proj may take the core's MXFP8 output (:meth:`forward_prequantized`).
    """

    def __init__(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # SM107: the weight scale in the DeepGEMM-packed UE8M0 layout of the K3 low-M GEMM
        # (``weight_scale`` holds the CuTe MXFP8 layout there).
        self.lowm_weight_scale: Optional[torch.Tensor] = None

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether a producer may hand this GEMM an FP8 activation in CuTe layout."""
        return _rubin_mxfp8_is_available()

    def takes_prequantized_input(self, num_rows: int) -> bool:
        """Whether a producer that can emit this GEMM's scale layout should hand ``num_rows``
        rows to :meth:`forward_prequantized` rather than BF16 rows to :meth:`forward`.

        False where :meth:`forward` serves BF16 rows with the low-M GEMM (activation quantized
        inside the kernel) faster than the CuTe MXFP8 GEMM runs on the prequantized operand.
        """
        if not self.supports_prequantized_input:
            return False
        if self.lowm_weight_scale is None:
            return True
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return num_rows > k3_fp8w_lowm_gemm.bf16_over_prequantized_max_m(
            self.out_features, self.in_features
        )

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> (FP8 weight, backend-ready scale).

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy imports: only pulled in on the FP8 path.
        from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        # Both GEMMs consume UE8M0 scales rather than the FP32 grid.
        return prepare_fp8_block_weight(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_scale_layout(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
        from tensorrt_llm.quantization.utils.fp8_utils import (
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            return transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        return transform_sf_into_required_layout(
            weight_scale,
            mn=n,
            k=k,
            recipe=(1, 128, 128),
            is_sfa=False,
        )

    @classmethod
    def from_linear(cls, linear: nn.Module) -> "_Fp8BlockScaleWeightReadLinear":
        """The FP8 weight read of a loaded projection, sharing its codes and scales:

        * ``core.Fp8BlockScaleLinear`` (checkpoint E4M3 codes and FP32 128 x 128 scales, or their
          ``prepare``d form);
        * a bias-free BF16 projection (``nn.Linear`` or ``core.Bf16Linear``): quantized here.
        """
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, Fp8BlockScaleLinear):
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = getattr(linear, "prepared", False)
            result.fp8w_lowm_partial_n_tile = getattr(linear, "fp8w_lowm_partial_n_tile", False)
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                if self.weight_scale.dtype != torch.float32 or self.weight_scale.ndim != 2:
                    raise ValueError("KDA fused FP8 views require a raw FP32 scale grid")
                # Codes and raw scale grids alias the fused projection. Only its
                # owner may resmooth them in place; each view packs the updated
                # scales without modifying the shared codes again.
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
                self.weight_scale = self._prepare_scale_layout(
                    self.weight_scale, self.out_features, self.in_features
                )
            else:
                self.weight, self.weight_scale = prepare_fp8_block_weight(
                    self.weight, self.weight_scale
                )
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.prepare_lowm_gemm()

    def prepare_lowm_gemm(self) -> bool:
        """Build ``lowm_weight_scale`` on SM107 for the K3 low-M GEMM.

        Derived from the transformed MXFP8 scale, so it follows weight reloads; rebuilt on
        every call (post-load), before CUDA graph capture. Costs N * K / 128 bytes (0.8% of
        the FP8 weight). SM100/SM103 run these GEMMs on ``weight_scale`` itself.
        """
        self.lowm_weight_scale = None
        if not _rubin_mxfp8_is_available():
            return False
        if self._fused_projection is not None:
            # A row view of the KDA fused q/k/v/g weight; the fused module serves forward.
            return False
        self.transform_weights()
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        n, k = self.weight.shape
        if n % k3_fp8w_lowm_gemm.N_ALIGN or k % 128:
            return False
        self.lowm_weight_scale = k3_fp8w_lowm_gemm.pack_block_scale_from_mxfp8_layout(
            self.weight_scale, n, k
        )
        return True

    def _lowm_gemm_applies(self, x_flat: torch.Tensor, weight_scale: torch.Tensor) -> bool:
        """Whether the ``k3_fp8w_lowm_gemm`` entry serves ``x_flat`` with ``weight_scale``."""
        return x_flat.dtype == torch.bfloat16 and _k3_lowm_gemm_can_run(
            x_flat,
            self.weight,
            weight_scale,
            allow_partial_n_tile=self.fp8w_lowm_partial_n_tile,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, x.shape[-1])

        # The K3 low-M GEMM up to its per-shape M gate, on SM107 with ``lowm_weight_scale``, on
        # SM100 / SM103 with ``weight_scale``; else the MXFP8 (SM107) / swap-AB (SM100 / SM103)
        # GEMM of ``layers.fp8_block_scale_gemm``.
        lowm_scale = self.lowm_weight_scale if _rubin_mxfp8_is_available() else self.weight_scale
        if lowm_scale is not None and self._lowm_gemm_applies(x_flat, lowm_scale):
            out = k3_fp8w_lowm_gemm_op(x_flat, self.weight, lowm_scale)
        else:
            out = fp8_block_scale_gemm(x_flat, self.weight, self.weight_scale)
        return out.reshape(out_shape)

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an FP8 activation a fused producer already quantized.

        Args:
            activation: ``[..., in_features]`` ``float8_e4m3fn`` activation.
            activation_scale: ``uint8`` UE8M0 scales in the CuTe K32 R128c4
                layout that ``fp8_quantize_1x128_cutedsl_ue8m0`` produces.
        """
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        out = k3_mxfp8_gemm_op(
            activation.reshape(-1, activation.shape[-1]),
            self.weight,
            activation_scale,
            self.weight_scale,
        )
        return out.reshape(activation.shape[:-1] + (self.out_features,))


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    converted = _Fp8BlockScaleWeightReadLinear.from_linear(child)
    setattr(parent, attr, converted)
    child.weight.data = child.weight.data.new_empty(0)
    return 1


def _has_weights(module: nn.Module) -> bool:
    """False once ``modeling_utils.remove_weights()`` has stripped a module.

    Post-load finalization walks every decoder layer, so it must skip layers
    whose parameters were dropped — the layer-wise benchmarks keep only the
    profiled slice resident.
    """
    return not getattr(module, "_weights_removed", False)


def _convert_moe_mlps_to_fp8_weight_read(
    model: nn.Module, include_fused_gate_up: bool = True
) -> int:
    """Swap the replicated MoE-layer MLP projections to an FP8 weight read.

    Targets the shared-expert MLP (gate/up/down) and the latent up/down
    projection on every MoE layer — the bias-free BF16 projections that
    attention data-parallelism re-reads in full each decode step. Attention
    follows checkpoint quantization separately. Routed experts and the dense
    layer-0 MLP are not converted. Returns the number of projections converted.
    """
    count = 0

    for layer in model.layers:
        if not _has_weights(layer):
            continue
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            # GatedMlp fuses gate and up into gate_up_proj; keep the split
            # names too so either MLP layout converts. The fused gate_up read
            # only pays off when attention DP re-reads it per rank per step;
            # under TP the bf16 GEMM overlaps on the aux stream and the FP8
            # quantize+GEMM would serialize onto the critical path.
            shared_attrs = (
                ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
                if include_fused_gate_up
                else ("gate_proj", "up_proj", "down_proj")
            )
            for attr in shared_attrs:
                child = getattr(shared, attr, None)
                if isinstance(child, Bf16Linear) and child.tp_size != 1:
                    continue
                count += _swap_linear_to_fp8_weight_read(
                    shared, attr, linear_types=(nn.Linear, Bf16Linear)
                )
        for attr in ("routed_expert_down_proj", "routed_expert_up_proj"):
            count += _swap_linear_to_fp8_weight_read(moe, attr)

    # Return the freed BF16 blocks to the driver so the raw (non-caching-
    # allocator) allocations made during executor creation succeed on the
    # tight DEP16 memory headroom.
    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count
