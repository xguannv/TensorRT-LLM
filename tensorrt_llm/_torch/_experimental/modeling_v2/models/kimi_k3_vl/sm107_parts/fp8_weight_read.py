# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107): projections that read FP8 block-scaled weights.

``_Fp8BlockScaleWeightReadLinear`` keeps a bias-free projection as E4M3 codes with 128x128
block scales: the checkpoint FP8 attention projections, and the MoE layers' BF16 MLPs when
they are quantized after loading. Low-M rows run the low-M GEMMs; the rest runs DeepGEMM on
SM100/SM103 and the native MXFP8 GEMM on SM107. ``_swap_linear_to_fp8_weight_read`` and
``_convert_moe_mlps_to_fp8_weight_read`` put it in place of a loaded linear.
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
from ..ops.gemm.k3_fp8w_lowm_gemm_push import k3_fp8w_lowm_gemm_push as k3_fp8w_lowm_gemm_push_op
from ..ops.gemm.k3_fp8w_lowm_mxfp8_gemm import k3_fp8w_lowm_mxfp8_gemm as k3_fp8w_lowm_mxfp8_gemm_op
from ..ops.gemm.k3_fp8w_lowm_mxfp8_push import k3_fp8w_lowm_mxfp8_push as k3_fp8w_lowm_mxfp8_push_op
from ..ops.gemm.k3_fp8w_mid_gemm import k3_fp8w_mid_gemm as k3_fp8w_mid_gemm_op
from ..ops.gemm.k3_fp8w_mid_gemm_push import k3_fp8w_mid_gemm_push as k3_fp8w_mid_gemm_push_op
from ..ops.gemm.k3_mxfp8_gemm import k3_mxfp8_gemm as k3_mxfp8_gemm_op
from .features import active as _features
from .layers import (
    Bf16Linear,
    Fp8BlockScaleLinear,
    fp8_block_scale_gemm,
    low_m_gemm_rows,
    low_m_gemm_scale,
    prepare_fp8_block_weight,
)

_F = _features()

# FP8 weight-read projections with few rows (per-shape, per-architecture limit, see
# k3_fp8w_lowm_gemm.max_m_for) run the CuTe low-M GEMM: on SM100/SM103 instead of DeepGEMM
# swap-AB, on SM107 instead of the CuTe MXFP8 GEMM (with a DeepGEMM-packed copy of the weight
# scale, built at post-load, see _Fp8BlockScaleWeightReadLinear.prepare_lowm_gemm).
_K3_FP8W_LOWM_GEMM = _F.fp8w_lowm_gemm

# Fused latent projection (cute_dsl_kernels/blackwell/k3_latent_fp8_gemm.py), for token counts
# in [_K3_LD_FUSED_MIN_M, _K3_LD_FUSED_MAX_M]:
#   TRTLLM_K3_LD_FUSED: latent_down + the routed experts' NVFP4 input quantization in one chain
#     (the k3_latent_down_fp4 entry), bit-exact with fp8_swap_ab_gemm followed by fp4_quantize.
# It fuses only where the unfused projection would run fp8_swap_ab_gemm, not where it runs the
# low-M kernel (TRTLLM_K3_FP8W_LOWM_GEMM). On SM107 the op runs the low-M kernel's split-K GEMM
# with the fused epilogue up to k3_latent_fp8_gemm.lowm_route_max_m tokens (bit-exact with that
# GEMM at the same tactic followed by fp4_quantize) and the persistent kernel above; both read a
# DeepGEMM-packed copy of the weight scale built at post-load
# (_Fp8BlockScaleWeightReadLinear.prepare_lowm_gemm).
_K3_LD_FUSED = _F.ld_fused


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
    buckets. On both, the low-M GEMM (``trtllm::fp8_block_scale_low_m_gemm``) serves
    the rows ``layers.low_m_gemm_rows`` gives for the weight shape; it reads a
    DeepGEMM-packed scale, which is ``weight_scale`` on SM100/SM103 and a copy
    repacked from the MXFP8 scale on SM107. That dispatch is
    ``layers.fp8_block_scale_gemm``.

    The Rubin agentic kernels sit in front of that dispatch: the K3 low-M GEMM
    (``TRTLLM_K3_FP8W_LOWM_GEMM``, ``kernels/k3_fp8w_lowm_gemm.py``) with its gated,
    sigmoid, MXFP8-input and all-reduce-push variants. With ``TRTLLM_K3_FP8W_LOWM_GEMM``
    set the K3 low-M GEMM owns the low-M rows exactly as on that branch and the generic low-M GEMM is not
    prepared; with it unset the generic one serves them as on main.
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
        # Scales of the output-row slices :meth:`forward_slice` serves (an
        # ``Fp8BlockScaleWeightSliceCache``), prepared by :meth:`prepare_output_slice`.
        self._output_slice_cache: Optional[nn.Module] = None
        # Rows the low-M GEMM serves (0: none) and the DeepGEMM-packed scale it reads; set at
        # post-load.
        self._low_m_max = 0
        self._low_m_weight_scale: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only: a contiguous BF16 ``[M, out_features]``
        # tensor that the low-M GEMM writes its result into. Other paths return their own
        # tensor, so the caller compares the result with the buffer it handed in.
        self.next_output: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only (every forward consumes and clears it): the
        # ``LamportPushTarget`` (``layers.lamport_push_target``) of the all-reduce that
        # follows this projection. Where the low-M GEMM without split-K (or the mid-M GEMM with
        # its push epilogue) serves ``forward`` / ``forward_gated``, it stores its output into
        # that all-reduce's Lamport buffers instead, marks the target pushed and returns the
        # all-reduce's output tensor, which ``layers.finish_lamport_push`` fills.
        self.next_ar_push = None
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # Whether the fused latent projection (TRTLLM_K3_LD_FUSED) may replace this one (set when
        # it is swapped in).
        self.k3_latent_fused = False
        # SM107 with TRTLLM_K3_FP8W_LOWM_GEMM or ``k3_latent_fused``: the weight scale in the
        # DeepGEMM-packed UE8M0 layout of the low-M GEMM and the fused latent projections
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

    @property
    def packed_weight_scale(self) -> Optional[torch.Tensor]:
        """The weight scale in DeepGEMM's packed UE8M0 layout, as the low-M GEMM and the fused
        latent projections read it: ``weight_scale`` on SM100/SM103, ``lowm_weight_scale`` on
        SM107 (``None`` unless :meth:`prepare_lowm_gemm` built it)."""
        if _rubin_mxfp8_is_available():
            return self.lowm_weight_scale
        return self.weight_scale

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
            self._output_slice_cache = None
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
        self._prepare_low_m_gemm()

    def _prepare_low_m_gemm(self) -> None:
        """Select the low-M GEMM for this weight where it has a DeepGEMM-packed scale.

        On SM107 that scale is a copy (N * K / 128 bytes) derived from the MXFP8
        ``weight_scale``; row views of a fused KDA projection have none (the fused module
        serves decode). Not prepared under ``TRTLLM_K3_FP8W_LOWM_GEMM``: the K3 low-M GEMM
        owns those rows then (see the class docstring).
        """
        self._low_m_max, self._low_m_weight_scale = 0, None
        if _K3_FP8W_LOWM_GEMM:
            return
        n, k = self.out_features, self.in_features
        low_m_max = low_m_gemm_rows(n, k)
        if not _rubin_mxfp8_is_available():
            self._low_m_max, self._low_m_weight_scale = low_m_max, self.weight_scale
        elif low_m_max and self._fused_projection is None:
            self._low_m_max = low_m_max
            self._low_m_weight_scale = low_m_gemm_scale(self.weight_scale, n, k)

    def prepare_lowm_gemm(self) -> bool:
        """Build ``lowm_weight_scale`` on SM107 if TRTLLM_K3_FP8W_LOWM_GEMM is set or a fused
        latent projection may replace this one (``k3_latent_fused``).

        Derived from the transformed MXFP8 scale, so it follows weight reloads; rebuilt on
        every call (post-load), before CUDA graph capture. Costs N * K / 128 bytes (0.8% of
        the FP8 weight). SM100/SM103 run these GEMMs on ``weight_scale`` itself.
        """
        self.lowm_weight_scale = None
        if not ((_K3_FP8W_LOWM_GEMM or self.k3_latent_fused) and _rubin_mxfp8_is_available()):
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

    def _run_lowm_gemm(
        self,
        x_flat: torch.Tensor,
        weight_scale: torch.Tensor,
        out_buffer: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The low-M GEMM, into ``out_buffer`` when the caller handed in a usable one."""
        if (
            out_buffer is not None
            and out_buffer.dtype == torch.bfloat16
            and out_buffer.shape == (x_flat.shape[0], self.out_features)
            and out_buffer.is_contiguous()
        ):
            return k3_fp8w_lowm_gemm_op(x_flat, self.weight, weight_scale, out_buffer)
        return k3_fp8w_lowm_gemm_op(x_flat, self.weight, weight_scale)

    def _take_next_output(self, num_rows: int) -> Optional[torch.Tensor]:
        """Clear :attr:`next_output`; return it if it is a contiguous, 16-byte aligned BF16
        ``[num_rows, out_features]`` tensor."""
        out, self.next_output = self.next_output, None
        if (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (num_rows, self.out_features)
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            return None
        return out

    def _take_next_ar_push(self, x_flat: torch.Tensor, gated: bool = False):
        """Clear :attr:`next_ar_push`; return it if the low-M GEMM can push this forward's
        ``[M, out_features]`` output into it (low-M kernel without split-K; above the low-M
        range, on ungated rows, the mid-M GEMM's push epilogue)."""
        push, self.next_ar_push = self.next_ar_push, None
        if (
            push is None
            or x_flat.dtype != torch.bfloat16
            or tuple(push.output.shape) != (x_flat.shape[0], self.out_features)
        ):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.ar_push_tactic(
            x_flat.shape[0], self.out_features, self.in_features, gated=gated
        )
        if tactic is None and not gated:
            tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(
                x_flat.shape[0], self.out_features, self.in_features
            )
        return push if tactic is not None else None

    def mid_push_pdl_ready(self, num_rows: int) -> bool:
        """``TRTLLM_K3_KDA_MX_PDL``: whether :meth:`forward_prequantized` on the producer's
        ``[num_rows, in_features]`` R128c4 operand will push with the mid-M GEMM (the conditions
        of :meth:`_mid_prequantized_push_tactic` that do not depend on the tensors), which reads
        nothing the producer writes before its ``griddepcontrol.wait``: the producer may then
        release it early."""
        push = self.next_ar_push
        if push is None or self.lowm_weight_scale is None:
            return False
        if num_rows > 128:
            return False
        if tuple(push.output.shape) != (num_rows, self.out_features):
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(num_rows, self.out_features, self.in_features)
        return tactic is not None and tactic.mma_n <= 128

    def _mid_prequantized_push_tactic(self, activation: torch.Tensor, scale: torch.Tensor, push):
        """The mid-M tactic :meth:`forward_prequantized` pushes the KDA verify's MXFP8 operand with,
        or None. Up to 128 rows (one scale-factor block), where the R128c4 scales of the operand are
        the mid-M GEMM's ``[K / 128][1][512 B]`` layout; the mid-M GEMM instead of the CuTe MXFP8
        GEMM, so the projection's FP32 accumulation order can differ there."""
        if push is None or self.lowm_weight_scale is None:
            return None
        if activation.dim() != 2 or not activation.is_contiguous():
            return None
        m, k = activation.shape
        if k != self.in_features or tuple(push.output.shape) != (m, self.out_features):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(m, self.out_features, k)
        if tactic is None or tactic.mma_n > 128:
            return None
        if scale.numel() != (k // 128) * 512 or not scale.is_contiguous():
            return None
        return tactic

    def _run_ar_push(self, x_flat: torch.Tensor, push, gate=None) -> torch.Tensor:
        """The low-M GEMM with the all-reduce push epilogue; returns the all-reduce output."""
        k3_fp8w_lowm_gemm_push_op(
            x_flat, self.weight, self.packed_weight_scale, push.workspace, push.one_shot, gate=gate
        )
        push.pushed = True
        return push.output

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
        out_buffer, self.next_output = self.next_output, None
        ar_push = self._take_next_ar_push(x_flat)

        # The K3 low-M GEMM (``TRTLLM_K3_FP8W_LOWM_GEMM``) up to its per-shape M gate, on SM107
        # with ``lowm_weight_scale``, on SM100 / SM103 with ``weight_scale``; else the low-M /
        # MXFP8 (SM107) / swap-AB (SM100 / SM103) GEMM of ``layers.fp8_block_scale_gemm``.
        lowm_scale = self.lowm_weight_scale if _rubin_mxfp8_is_available() else self.weight_scale
        if (
            _K3_FP8W_LOWM_GEMM
            and lowm_scale is not None
            and self._lowm_gemm_applies(x_flat, lowm_scale)
        ):
            if ar_push is not None:
                out = self._run_ar_push(x_flat, ar_push)
            else:
                out = self._run_lowm_gemm(x_flat, lowm_scale, out_buffer)
        else:
            out = fp8_block_scale_gemm(
                x_flat,
                self.weight,
                self.weight_scale,
                low_m_max=self._low_m_max,
                low_m_scale=self._low_m_weight_scale,
            )
        return out.reshape(out_shape)

    def prepare_output_slice(self, start: int, end: int) -> None:
        """Prepare the scales of output features ``[start, end)`` for :meth:`forward_slice`.

        The shared MLA calls this at post-load for the context heads of a Helix rank. The slice
        keeps the 128-row scale grid: it covers the whole scale tiles around ``[start, end)``.
        """
        from tensorrt_llm.quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

        if not 0 <= start < end <= self.out_features:
            raise ValueError(f"Invalid output feature range [{start}, {end})")
        self.transform_weights()
        if self._output_slice_cache is None:
            self._output_slice_cache = Fp8BlockScaleWeightSliceCache()
        self._output_slice_cache.prepare(self.weight, self.weight_scale, start, end)

    def forward_slice(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Output features ``[start, end)`` of :meth:`forward`, contiguous.

        One GEMM over the weight rows of the slice's scale tiles (the general path of
        :meth:`forward`: the CuTe MXFP8 GEMM on SM107, ``fp8_swap_ab_gemm`` elsewhere), cropped
        to the slice.
        """
        self.prepare_output_slice(start, end)
        weight, scale, crop = self._output_slice_cache.get(self.weight, start, end)
        x_flat = x.reshape(-1, x.shape[-1])
        if _rubin_mxfp8_is_available():
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = k3_mxfp8_gemm_op(act_fp8, weight, act_scale, scale)
        else:
            out = torch.ops.trtllm.fp8_swap_ab_gemm(
                x_flat, weight, scale, output_dtype=x.dtype, disable_ue8m0_cast=True
            )
        out = out.reshape(*x.shape[:-1], weight.shape[0])
        return out[..., crop : crop + end - start].contiguous()

    def _lowm_forward_scale(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """The weight scale :meth:`forward` runs the K3 low-M GEMM with on the BF16 rows
        ``x_flat`` (``lowm_weight_scale`` on SM107, ``weight_scale`` on SM100/SM103), or None
        where :meth:`forward` runs another GEMM."""
        if not _K3_FP8W_LOWM_GEMM or x_flat.dtype != torch.bfloat16:
            return None
        if self.lowm_weight_scale is not None:
            scale = self.lowm_weight_scale
        elif _rubin_mxfp8_is_available():
            return None
        else:
            scale = self.weight_scale
        return scale if self._lowm_gemm_applies(x_flat, scale) else None

    def _runs_low_m_kernel(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows would run the low-M kernel (the
        input-independent part of its dispatch: the low-M GEMM on, not the Rubin CuTe MXFP8
        path, ``num_rows`` within the low-M kernel's range). On SM107 forward runs it on
        ``lowm_weight_scale`` up to the per-shape gate (``max_m_for``)."""
        if not _K3_FP8W_LOWM_GEMM:
            return False
        if _rubin_mxfp8_is_available() and self.lowm_weight_scale is None:
            return False
        if self.weight.shape[0] % 128 and not self.fp8w_lowm_partial_n_tile:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        limit = k3_fp8w_lowm_gemm.low_m_max_for(*self.weight.shape)
        if self.lowm_weight_scale is not None:
            limit = min(limit, k3_fp8w_lowm_gemm.max_m_for(*self.weight.shape))
        return 1 <= num_rows <= limit

    def takes_mxfp8_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_mxfp8` the MXFP8 form of ``num_rows``
        rows: where :meth:`forward` would run the low-M kernel, whose in-kernel quantization the
        MXFP8 GEMM replaces (the attention's first projection also asks
        :meth:`takes_mid_mxfp8_input`)."""
        return self._runs_low_m_kernel(num_rows)

    def takes_mid_mxfp8_input(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows runs the mid-M path of
        ``k3_fp8w_lowm_gemm`` (above the low-M cut-over, up to the per-shape gate) with one
        128-row scale block, so that a producer may hand :meth:`forward_mxfp8` the rows' MXFP8 form
        with the scales in that GEMM's layout (``k3_attn_res`` ``mx_atom``): the GEMM then runs
        without its quantization kernel, bitwise the BF16 path."""
        return self.mid_operand_tactic(num_rows) is not None

    def mid_operand_tactic(self, num_rows: int):
        """c240: the mid-M tactic :meth:`forward` runs on ``num_rows`` BF16 rows (SM107 low-M GEMM
        switch on, above the low-M cut-over, up to the per-shape gate) when it has one 128-row
        scale block, else None. A producer may then store the rows' MXFP8 form in that GEMM's
        layout (``mid_prequant_plan``) for :meth:`forward_mxfp8` / :meth:`forward_mid_mxfp8`."""
        if not _K3_FP8W_LOWM_GEMM:
            return None
        if self.lowm_weight_scale is None:
            return None
        n, k = self.weight.shape
        if n % 128 and not self.fp8w_lowm_partial_n_tile:
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        if (
            not k3_fp8w_lowm_gemm.low_m_max_for(n, k)
            < num_rows
            <= min(128, k3_fp8w_lowm_gemm.max_m_for(n, k))
        ):
            return None
        plan = k3_fp8w_lowm_gemm.mid_prequant_plan(num_rows, n, k)
        return plan[0] if plan is not None and plan[1] == 1 else None

    def forward_mid_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """``TRTLLM_K3_MID_OPROJ_PC`` (c240): :meth:`forward` on rows whose MXFP8 form a producer
        stored with the scales in the mid-M GEMM's layout (``xs`` 1-D, see
        :meth:`mid_operand_tactic`; ``x`` only gives the output shape): the mid-M GEMM without
        its quantization kernel, with the all-reduce push epilogue where :attr:`next_ar_push`
        takes it. Bitwise :meth:`forward` on the rows the producer quantized."""

        m = xq.shape[0]
        tactic = self.mid_operand_tactic(m)
        if tactic is None or xs.dim() != 1:
            raise RuntimeError("forward_mid_mxfp8 needs the mid-M operand layout")
        x_flat = x.reshape(-1, x.shape[-1])
        push = self._take_next_ar_push(x_flat)
        self.next_output = None
        if push is not None:
            k3_fp8w_mid_gemm_push_op(
                xq.reshape(m, -1),
                xs,
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
        out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_mid_gemm_op(xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def _staged_above_cutover(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` serves ``num_rows`` rows above the
        low-M kernel's cut-over with the staged MXFP8 low-M GEMM: :meth:`forward` runs the mid-M
        path of ``k3_fp8w_lowm_gemm`` there, whose bits the staged kernel reproduces
        (``k3_fp8w_lowm_mxfp8.staged_matches_mid_m``)."""
        if not _K3_FP8W_LOWM_GEMM or self.lowm_weight_scale is None:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm, k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if num_rows > k3_fp8w_lowm_gemm.max_m_for(n, k):
            return False
        return k3_fp8w_lowm_mxfp8.staged_matches_mid_m(num_rows, n, k)

    def takes_staged_mxfp8_input(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` runs the staged MXFP8 low-M GEMM on
        the MXFP8 form of ``num_rows`` rows (bitwise :meth:`forward`): where :meth:`forward` runs
        the low-M kernel, and above its cut-over where it runs the mid-M path with the same
        split-K (up to 64 rows)."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if self._runs_low_m_kernel(num_rows):
            # Also where the shared-memory ring alone holds the CTA's whole weight slice.
            return k3_fp8w_lowm_mxfp8.weights_on_chip(num_rows, n, k)
        return self._staged_above_cutover(num_rows)

    def forward_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, stage_weights: bool = False
    ) -> torch.Tensor:
        """:meth:`forward` on the BF16 rows ``x``, bit for bit, given their MXFP8 form ``(xq,
        xs)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, e.g. from
        the ``k3_attn_res_norm`` MXFP8 variants): the low-M GEMM on the prequantized operand where
        :meth:`forward` would run the low-M kernel, else :meth:`forward` on ``x``.
        ``stage_weights``: the low-M GEMM keeps its whole weight slice on chip before its
        ``griddepcontrol.wait`` (``k3_fp8w_lowm_mxfp8.staged_mx_tactic``, the same bits), also
        above the low-M cut-over where :meth:`takes_staged_mxfp8_input` says so."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        x_flat = x.reshape(-1, x.shape[-1])
        if xs.dim() == 1 and self.takes_mid_mxfp8_input(x_flat.shape[0]):
            # The producer stored the scales in the mid-M GEMM's layout.
            from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

            m = x_flat.shape[0]
            tactic, _ = k3_fp8w_lowm_gemm.mid_prequant_plan(m, self.out_features, self.in_features)
            self.next_ar_push = None
            out = self._take_next_output(m)
            if out is None:
                out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=x.device)
            k3_fp8w_mid_gemm_op(
                xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic
            )
            return out.reshape(x.shape[:-1] + (self.out_features,))
        allow_partial = self.fp8w_lowm_partial_n_tile
        # The scale forward's low-M branch uses (lowm_weight_scale on SM107).
        w_scale = self.packed_weight_scale
        above_cutover = (
            stage_weights
            and not self._runs_low_m_kernel(x_flat.shape[0])
            and self._staged_above_cutover(x_flat.shape[0])
        )
        if not (
            x.dtype == torch.bfloat16
            and (above_cutover or self._runs_low_m_kernel(x_flat.shape[0]))
            and xq.shape[0] == x_flat.shape[0]
            and _k3_lowm_gemm_can_run(
                x_flat, self.weight, w_scale, allow_partial_n_tile=allow_partial
            )
            and k3_fp8w_lowm_mxfp8.can_run_mxfp8(
                xq,
                xs,
                self.weight,
                w_scale,
                allow_partial_n_tile=allow_partial,
                above_cutover=above_cutover,
            )
        ):
            return self.forward(x)
        out = self._take_next_output(x_flat.shape[0])
        # The MXFP8 GEMM has no all-reduce push epilogue: the caller runs the all-reduce.
        self.next_ar_push = None
        if out is None:
            out = torch.empty(
                (x_flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        stage_weights = stage_weights and k3_fp8w_lowm_mxfp8.weights_on_chip(
            x_flat.shape[0], self.out_features, self.in_features
        )
        k3_fp8w_lowm_mxfp8_gemm_op(xq, xs, self.weight, w_scale, out, stage_weights=stage_weights)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def gated_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated` serves ``x`` and ``gate`` with one gated low-M GEMM.

        That is where :meth:`forward` would run the low-M GEMM on ``x * gate`` (same kernel
        family and split), so the gated launch is bitwise equal to the multiply plus
        :meth:`forward`.
        """
        if x.dim() != 2 or not x.is_contiguous():
            return False
        scale = self._lowm_forward_scale(x)
        if scale is None:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.can_run_gated(
            x,
            gate,
            self.weight,
            scale,
            allow_partial_n_tile=self.fp8w_lowm_partial_n_tile,
        )

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one low-M GEMM launch with a sigmoid
        epilogue where :meth:`forward` would run the low-M GEMM."""
        x_flat = x.reshape(-1, x.shape[-1])
        scale = self._lowm_forward_scale(x_flat)
        if scale is not None:
            from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

            if x_flat.shape[0] <= k3_fp8w_lowm_gemm.low_m_max_for(
                self.out_features, self.in_features
            ):
                self.next_output = None
                self.next_ar_push = None
                out = k3_fp8w_lowm_gemm_op(x_flat, self.weight, scale, sigmoid=True)
                return out.reshape(x.shape[:-1] + (self.out_features,))
        return self.forward(x).sigmoid()

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * gate)`` in one gated low-M GEMM launch (where :meth:`gated_applies`).

        ``x`` and ``gate`` are BF16 ``[M, in_features]``; the gate may be a row-strided view.
        """
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            return self._run_ar_push(x, ar_push, gate=gate)
        return k3_fp8w_lowm_gemm_op(
            x,
            self.weight,
            self.packed_weight_scale,
            out=self._take_next_output(x.shape[0]),
            gate=gate,
        )

    def gated_mxfp8_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated_mxfp8` runs the MXFP8 low-M GEMM on the producer's MXFP8
        form of ``bf16(x * gate)``: where :meth:`forward_gated` runs
        the gated low-M GEMM, whose tiling the MXFP8 kernel shares (same bits)."""
        if not self.gated_applies(x, gate):
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        m, k = x.shape
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(m, self.out_features, k)
        except ValueError:
            return False
        return k % 32 == 0

    def forward_gated_mxfp8(
        self, x: torch.Tensor, gate: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """:meth:`forward_gated` ``(x, gate)``, bit for bit, given the MXFP8 form ``(xq, xs)`` of
        ``bf16(x * gate)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, the gated
        low-M GEMM's own quantization): the MXFP8 low-M GEMM, with the all-reduce push epilogue
        where :attr:`next_ar_push` is set, so the activation path after its
        ``griddepcontrol.wait`` is two TMA loads instead of the BF16 loads, the gate multiply and
        the quantization."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        w_scale = self.packed_weight_scale
        if w_scale is None or not k3_fp8w_lowm_mxfp8.can_run_mxfp8(xq, xs, self.weight, w_scale):
            return self.forward_gated(x, gate)
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            k3_fp8w_lowm_mxfp8_push_op(
                xq, xs, self.weight, w_scale, ar_push.workspace, ar_push.one_shot
            )
            ar_push.pushed = True
            return ar_push.output
        out = self._take_next_output(x.shape[0])
        if out is None:
            out = torch.empty(
                (x.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        k3_fp8w_lowm_mxfp8_gemm_op(xq, xs, self.weight, w_scale, out)
        return out

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
        push, self.next_ar_push = self.next_ar_push, None
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        rows = activation.reshape(-1, activation.shape[-1])
        tactic = self._mid_prequantized_push_tactic(rows, activation_scale, push)
        if tactic is not None:
            # The mid-M GEMM pushes into the o_proj all-reduce.
            k3_fp8w_mid_gemm_push_op(
                rows,
                activation_scale.reshape(-1),
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
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
    converted.k3_latent_fused = attr == "routed_expert_down_proj" and _K3_LD_FUSED
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
