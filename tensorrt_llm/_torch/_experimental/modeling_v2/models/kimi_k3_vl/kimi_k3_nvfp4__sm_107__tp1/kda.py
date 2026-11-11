# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) for the Kimi K3 text model of the tp1 target (one rank).

``K3KDALinearAttention`` holds the layer's weights in K3's own containers (``layers.py``) and
runs one explicit kernel chain per phase (prefill, decode, multi-token verify) on the package's
KDA kernels (``kernels/kda``, ``ops/ssm/_kda_prefill.py``). The
engine-owned cache manager stays main's ``modules/kimi_kda/cache_manager.py``.

KDA is the linear-attention block used at ``linear_attn_config.kda_layers``
positions in the Kimi K3 text-core. It carries a short-convolution state and a
delta-rule recurrent state per layer, so it follows the hybrid-cache /
mamba ownership pattern rather than the paged-KV FMHA attention-backend
interface.

Kimi K3 KDA layer: K3-owned weights and one explicit kernel chain per phase.

Weights (checkpoint names under ``model.layers.<i>.self_attn.``, this rank's head shard):

* ``q_proj`` / ``k_proj`` / ``v_proj`` / ``g_proj`` / ``o_proj``: FP8 128 x 128 block-scale
  projections (``layers.Fp8BlockScaleLinear``). :meth:`finalize_decode_weights` fuses
  ``[q | k | v | g]`` into ``qkvg_proj`` and repoints the four to row views of it; the model loader
  then swaps every FP8 projection to the K3 FP8 weight-read GEMM (``text_model``).
* ``f_a_proj`` / ``b_proj`` / ``f_b_proj``: BF16 (``layers.Bf16Linear``); ``[f_a | b]`` is
  fused into ``_bfa_proj_weight`` (rows padded to 8).
* ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d``: depthwise short-convolution weights
  (``layers.ShortConvWeight``); ``o_norm``: the gated output RMSNorm weight
  (``layers.RMSNormWeight``); ``A_log`` / ``dt_bias``.

Kernel chains (``H`` heads of this rank, ``D = 128``; ``x`` the BF16 rows):

* **Prefill** (:meth:`forward_prefill`): QKVG GEMM -> ``F.linear`` ``[f_a | b]`` and ``F.linear``
  ``f_b`` -> short convolution with its conv-pool write-back, SiLU and Q/K L2 norm, a single
  unaligned sequence's Q/K/V/beta zero-padded to the prefill chunk
  (``fused_kda_conv_prefill``) -> ``kda_prefill`` (fused K1-K3, A_kk inverse, persistent
  K4) on fp32 state rows (a BF16 pool is staged through an fp32 copy and written back) -> gated
  RMSNorm (``rms_norm_gated_token_major``) -> o_proj.
* **Decode** (:meth:`forward_decode`, one token per request): QKVG GEMM, next to ``[f_a | b]`` /
  ``f_b`` on the projection aux stream -> the fused CUDA decode kernel (convolution, gate,
  recurrence, gated RMSNorm, conv / state pools updated by slot) -> o_proj.
* **Verify** (:meth:`forward_verify_fused`, ``1 + num_spec`` tokens per request): QKVG GEMM next
  to the gate kernel on the aux stream (``k3_kda_fab_fa`` -> the forget gate computed in the
  verify kernel; ``k3_kda_fab_proj``; or ``F.linear`` ``[f_a | b]`` and ``f_b``) ->
  ``kda_mtp_decode`` (forget gate from ``f_a`` in the kernel, or BF16 core, or MXFP8 core for
  o_proj) on the replay caches -> o_proj.

tp1 serves context requests; the decode and verify chains run on the generation-shaped rows a
context server can still see (warmup, dummy rows) and in the test harness.

The kernels are called through the family's ``ops/ssm`` entries and
``rms_norm_gated_token_major``.

Cache ownership: three short-convolution states in the combined ``[slots, 3D, W - 1]`` BF16 pool
and one delta-rule recurrent state in the ``[slots, H, V, K]`` pool (fp32 or BF16), plus the replay
caches of the fused verify kernel, all allocated by the engine's ``KimiK3HybridCacheManager``.

The step metadata (``K3MambaMetadata``, its ``kda_snapshot`` plan of mid-chunk and in-forward
snapshots) is the family's ``common/kda_metadata.py``.
"""

from __future__ import annotations

import functools
import math
import weakref
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common.kda_metadata import KDASnapshotPlan
from ..kernels.k3r_kda._kda_kernels import (
    K3KDAKernelDispatch,
    kda_mtp_verify_ctas,
    reset_recurrent_state_rows,
)
from ..ops.norm.rms_norm_gated_token_major import rms_norm_gated_token_major
from ..sm107_common.layers import (
    Bf16Linear,
    Fp8BlockScaleLinear,
    RMSNormWeight,
    ShortConvWeight,
    fork_join,
)

# Prefill chunk of the kda_prefill entry.
_PREFILL_CHUNK = 64
# Single-token decode: [f_a | b] -> f_b runs on the projection aux stream next to the QKVG GEMM up
# to this many rows.
_KDA_BFA_MULTISTREAM_MAX_ROWS = 128
# Decode verify without the forget-gate round trip. Up to _KDA_FAB_FUSED_MAX_ROWS rows,
# k3_kda_fab runs next to the QKVG GEMM on the aux stream: in f_a mode (BF16 [f_a | b] only,
# the verify kernel computes the forget gate from f_a itself) when that kernel runs in one wave
# or the QKVG GEMM has left the low-M kernel (more rows than _KDA_FUSED_FB_LOWM_ROWS), else as
# the full fab kernel (beta and the forget gate from one fused [f_a | b] -> f_b kernel). The f_b
# MMA costs every wave of the verify kernel, so multi-wave low-M batches keep the full fab kernel.
# Larger row counts run the [f_a | b] -> f_b chain on the aux stream as well. The fab kernels take
# at most 16 heads (the Helix head shards); other shapes keep the [f_a | b] -> f_b chain.
_KDA_FAB_FUSED_MAX_ROWS = 256
_KDA_FUSED_FB_LOWM_ROWS = 224
_FP8_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
_QKVG_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj")


def _use_fused_fab(
    x: torch.Tensor, bfa_weight: torch.Tensor, fb_weight: torch.Tensor, num_heads: int
) -> bool:
    """Whether the ``k3_kda_fab_proj`` entry accepts these operands."""
    if not x.is_contiguous():
        return False
    from ..kernels.k3r_kda import k3_kda_fab

    return k3_kda_fab.can_use_k3_kda_fab(
        x.reshape(-1, x.shape[-1]), bfa_weight, fb_weight, num_heads
    )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _verify_kernel_single_wave(num_heads: int, num_requests: int, device: torch.device) -> bool:
    """Whether the v2 verify kernel's grid (``kda_mtp_verify_ctas``) fits in one wave of one CTA
    per SM."""
    ctas = kda_mtp_verify_ctas(num_heads, num_requests, device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return ctas <= _num_sms(index)


def _use_fab_fa(
    x: torch.Tensor, bfa_weight: torch.Tensor, fb_weight: torch.Tensor, num_heads: int
) -> bool:
    """Whether the verify gate comes from the ``k3_kda_fab_fa`` entry + the in-kernel f_b.

    ``x`` is the ``[requests, steps, hidden]`` verify input."""
    if not x.is_contiguous():
        return False
    num_rows = x.shape[0] * x.shape[1]
    if num_rows <= _KDA_FUSED_FB_LOWM_ROWS and not _verify_kernel_single_wave(
        num_heads, x.shape[0], x.device
    ):
        return False
    if (
        fb_weight.dtype != torch.bfloat16
        or not fb_weight.is_contiguous()
        or tuple(fb_weight.shape) != (num_heads * 128, 128)
        or fb_weight.data_ptr() % 16
    ):
        return False
    from ..kernels.k3r_kda import k3_kda_fab

    return k3_kda_fab.can_use_k3_kda_fab_fa(
        x.reshape(-1, x.shape[-1]), bfa_weight, num_heads, fb_weight
    )


def _o_proj_takes_mxfp8(o_proj: nn.Module, num_rows: int) -> bool:
    """Whether ``o_proj`` takes ``num_rows`` rows as the Rubin MXFP8 operand of the verify
    kernel's epilogue (``forward_prequantized``) rather than BF16 (the K3 FP8 weight-read
    projection decides per row count, see ``takes_prequantized_input``)."""
    takes = getattr(o_proj, "takes_prequantized_input", None)
    if takes is not None:
        return bool(takes(num_rows))
    return bool(getattr(o_proj, "supports_prequantized_input", False))


class _KdaPrequantizedCore(NamedTuple):
    """MXFP8 KDA core emitted by the fused verify epilogue.

    ``activation`` is the ``[tokens, H * V]`` E4M3 post-o_norm core and
    ``scale`` its UE8M0 1x128 scales in the flat uint8 R128c4 layout that
    ``o_proj.forward_prequantized`` consumes.
    """

    activation: torch.Tensor
    scale: torch.Tensor


@dataclass
class _KDAPrefillState:
    """The fp32 state rows the ``kda_prefill`` entry updates in place: the pool itself
    (``kernel_pool`` / ``kernel_indices``), or ``staged_state``, a dense fp32 copy of the
    addressed rows (plus the snapshot rows of a plan) that is written back afterwards."""

    kernel_pool: torch.Tensor
    kernel_indices: torch.Tensor
    staged_state: torch.Tensor | None
    # With a mid-chunk snapshot plan: the snapshot rows of ``kernel_pool`` (the pool's own slots,
    # or the rows after the sequences' in the staged copy).
    snapshot_rows: torch.Tensor | None = None


def _stage_state_rows(ssm_pool: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
    """Dense fp32 copy of the addressed recurrent-state rows (``_writeback_state_rows`` rounds it
    back into the pool)."""
    return ssm_pool.index_select(0, slot_indices.long()).float()


def _writeback_state_rows(
    ssm_pool: torch.Tensor, slot_indices: torch.Tensor, rows: torch.Tensor
) -> None:
    """Scatter dense state rows back into the pool, rounded to the pool dtype."""
    ssm_pool.index_copy_(0, slot_indices.long(), rows.to(ssm_pool.dtype))


def _stage_state_rows_with_snapshot_rows(
    ssm_pool: torch.Tensor, slot_indices: torch.Tensor, num_snapshots: int
) -> torch.Tensor:
    """``_stage_state_rows`` plus ``num_snapshots`` uninitialized rows after them.

    The prefill kernel writes mid-chunk snapshots into the extra rows.
    """
    num_rows = slot_indices.shape[0]
    staged = torch.empty(
        (num_rows + num_snapshots, *ssm_pool.shape[1:]),
        dtype=torch.float32,
        device=ssm_pool.device,
    )
    staged[:num_rows].copy_(ssm_pool.index_select(0, slot_indices.long()))
    return staged


def _snapshot_conv_windows(
    conv_input: torch.Tensor,
    conv_pool: torch.Tensor,
    snapshot: "KDASnapshotPlan",
    token_major: bool,
) -> torch.Tensor:
    """The raw convolution window ending before each snapshot point, ``[n, 3D, W - 1]`` in
    ``conv_pool``'s dtype.

    ``conv_input`` is the pre-convolution projection, ``[3D, T]`` or with
    ``token_major`` ``[T, 3D]``; ``conv_pool`` is ``[slots, 3D, W - 1]``. A
    snapshot point is at least one KDA chunk into its sequence's step, so the
    window never reaches the sequence's previous convolution state.
    """
    window = conv_pool.shape[-1]
    offsets = torch.arange(-window, 0, device=conv_input.device)
    token_indices = (snapshot.token_indices[:, None] + offsets).reshape(-1)
    if token_major:
        rows = conv_input.index_select(0, token_indices)
        windows = rows.view(snapshot.num_snapshots, window, -1).transpose(1, 2)
    else:
        columns = conv_input.index_select(1, token_indices)
        windows = columns.view(-1, snapshot.num_snapshots, window).transpose(0, 1)
    return windows.to(conv_pool.dtype)


def _swap_inforward_conv_windows(
    conv_pool: torch.Tensor,
    stage_conv: torch.Tensor,
    snapshot: "KDASnapshotPlan",
    windows: torch.Tensor,
) -> None:
    """Move each snapshotted sequence's chunk-end convolution window from its live slot
    (``snapshot.slots``) to its staging row and put the window ending at the snapshot point
    (``windows``, :func:`_snapshot_conv_windows`) in the live slot."""
    stage_conv.index_copy_(0, snapshot.stage_rows, conv_pool.index_select(0, snapshot.slots))
    conv_pool.index_copy_(0, snapshot.slots, windows)


def _merge_row_weights(modules: Tuple[nn.Module, ...], pad_rows_to: int = 1) -> torch.Tensor:
    """Concatenate the ``weight`` rows of ``modules`` (zero rows up to a multiple of
    ``pad_rows_to``) and repoint each module's ``weight`` to its row view of the result."""
    weights = [module.weight.data for module in modules]
    padding = (-sum(weight.shape[0] for weight in weights)) % pad_rows_to
    if padding:
        weights.append(weights[0].new_zeros((padding, weights[0].shape[1])))
    fused = torch.cat(weights, dim=0).contiguous()
    offset = 0
    for module in modules:
        rows = module.weight.shape[0]
        module.weight.data = fused[offset : offset + rows]
        offset += rows
    return fused


def _extract_kda_extra_attrs(layer_idx: str):
    """Resolve the live attention metadata and KDA module for ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata")
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata), "Metadata must be AttentionMetadata"

    kda_layers = extra_attrs.get("kda_layers")
    assert kda_layers is not None, "KDA layer registry is not set"
    kda_layer_ref = kda_layers.get(layer_idx)
    assert kda_layer_ref is not None, f"Cannot find KDA layer for layer {layer_idx}"
    kda_layer = kda_layer_ref()
    assert isinstance(kda_layer, K3KDALinearAttention), "KDA layer must be K3KDALinearAttention"
    return metadata, kda_layer


def kda_core_inplace(hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor) -> None:
    """Run the metadata-dependent KDA core and write it into ``output``."""
    metadata, kda_layer = _extract_kda_extra_attrs(layer_idx)
    kda_layer._forward_impl(hidden_states, metadata, output=output)


maybe_bcg_kda_core_inplace = eager_on_graph(kda_core_inplace)


def _check_kda_shape(lin: dict) -> None:
    """The KDA kernels take Kimi K3's shape only."""
    gate_lower_bound = lin.get("gate_lower_bound", None)
    problems = []
    if lin["head_dim"] != 128:
        problems.append(f"head_dim {lin['head_dim']} (the kernels take 128)")
    if lin["short_conv_kernel_size"] != 4:
        problems.append(f"short_conv_kernel_size {lin['short_conv_kernel_size']} (4)")
    if not lin.get("use_full_rank_gate", True):
        problems.append("a low-rank output gate (the full-rank g_proj)")
    if gate_lower_bound is None or not (math.isfinite(gate_lower_bound) and gate_lower_bound < 0):
        problems.append(f"gate_lower_bound {gate_lower_bound} (a finite negative bound)")
    if problems:
        raise ValueError("Kimi K3 KDA: unsupported linear_attn_config: " + "; ".join(problems))


class K3KDALinearAttention(nn.Module):
    """One Kimi K3 KDA layer (head-TP shard) on K3-owned weights."""

    def __init__(
        self,
        cfg,
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__()
        lin = cfg.linear_attn_config
        _check_kda_shape(lin)
        self.hidden_size = cfg.hidden_size
        self.layer_idx = layer_idx
        self.head_dim = lin["head_dim"]
        self.head_k_dim = self.head_dim
        self.conv_size = lin["short_conv_kernel_size"]
        self.gate_lower_bound = lin["gate_lower_bound"]
        self.rms_norm_eps = cfg.rms_norm_eps

        # One rank: every head, no o_proj all-reduce.
        self._kda_tp_size = 1
        self._kda_tp_rank = 0
        self.num_heads = lin["num_heads"]
        projection_size = self.num_heads * self.head_dim
        self.proj_size = projection_size

        projection_quant_configs = (model_config.quant_config_dict or {}) if model_config else {}
        default_quant_config = (
            model_config.quant_config if model_config else None
        ) or QuantConfig()
        declared = {
            name: projection_quant_configs.get(name, default_quant_config).quant_algo
            for name in _FP8_PROJECTIONS
        }
        if any(algo != QuantAlgo.FP8_BLOCK_SCALES for algo in declared.values()):
            raise ValueError(
                "Kimi K3 KDA takes FP8 128 x 128 block-scale q/k/v/g/o projections (the "
                f"checkpoint's FP8_PB_WO); declared {declared}"
            )
        if self.hidden_size % 128 or projection_size % 128:
            raise ValueError("KDA FP8 projection shards must be aligned to 128x128 blocks")

        bf16 = torch.bfloat16
        # This rank's shard (the loader slices the checkpoint by head), so the containers hold
        # rank-local dimensions.
        self.q_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.k_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.v_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.q_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        self.k_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        self.v_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        # Initialized (checkpoint loading overwrites them) with single random-init ops, which
        # MetaInitMode allows: uninitialized NaN bit patterns would poison randomly built layers.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=bf16).uniform_(1, 16))
        )
        self.f_a_proj = Bf16Linear(self.hidden_size, self.head_dim, dtype=bf16)
        self.f_b_proj = Bf16Linear(self.head_dim, projection_size, dtype=bf16)
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=bf16).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.b_proj = Bf16Linear(self.hidden_size, self.num_heads, dtype=bf16)
        self.g_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.o_norm = RMSNormWeight(self.head_dim, self.rms_norm_eps, dtype=bf16)
        self.o_proj = Fp8BlockScaleLinear(projection_size, self.hidden_size)
        # Built from q/k/v/g by finalize_decode_weights: the fused [q | k | v | g] projection and
        # its output sections.
        self.qkvg_proj: Optional[nn.Module] = None
        self.qkvg_split_sizes: Optional[List[int]] = None

        self._dispatch = K3KDAKernelDispatch()

        # Built after checkpoint load (finalize_decode_weights): the fused BF16 [f_a | b] weight
        # (rows padded to 8) and the kernel-layout constants.
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        # Fork/join state for overlapping the small [f_a | b] -> f_b chain
        # with the wide QKVG projection during CUDA-graph execution.
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()
        # Output buffer of the in-place decode kernel outside breakable CUDA graphs.
        self._o_dense: Optional[torch.Tensor] = None
        # The [W, 3D] BF16 conv weight of fused_kda_conv_prefill (from the packed [3D, W] one) and
        # the fp32 [D, W] weights of the verify kernels.
        self._packed_conv_weight: Optional[torch.Tensor] = None
        self._fused_conv_weight_t: Optional[torch.Tensor] = None
        self._mtp_conv_weights: Optional[Tuple[torch.Tensor, ...]] = None

        self.register_to_config = False
        self.layer_idx_str = str(layer_idx)
        if model_config is not None:
            kda_layers = model_config.extra_attrs.setdefault("kda_layers", {})
            suffix = 0
            while self.layer_idx_str in kda_layers:
                self.layer_idx_str = f"{layer_idx}_{suffix}"
                suffix += 1
            kda_layers[self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

    # ------------------------------------------------------------------
    # Post-load weight preparation.
    # ------------------------------------------------------------------

    def finalize_decode_weights(self) -> None:
        """Build the fused projection weights and kernel constants after weight load.

        1. ``qkvg_proj``: one FP8 ``[q | k | v | g]`` projection from the checkpoint codes and
           scales (the swap to the FP8 weight-read GEMM transforms its scales afterwards); the
           four sources become row views of it. ``_bfa_proj_weight``: BF16 ``[f_a | b]`` with
           rows padded to 8; ``f_a_proj`` / ``b_proj`` become row views of it.
        2. Kernel-layout constants: transposed BF16 conv weights ``[W, D]``, fp32 ``A_log`` /
           ``dt_bias`` / ``o_norm.weight`` and the packed / fp32 conv weights of the prefill and
           verify kernels.
        """
        if self.q_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            sources = tuple(getattr(self, name) for name in _QKVG_PROJECTIONS)
            if all(isinstance(module, Fp8BlockScaleLinear) for module in sources):
                self.qkvg_proj = self._fuse_checkpoint_projections(sources)
                self.qkvg_split_sizes = [module.out_features for module in sources]
                # Without it the four sources keep their own storage next to the fused copy.
                self._alias_projections_into_fused(sources, self.qkvg_proj)
            self._bfa_proj_weight = _merge_row_weights((self.f_a_proj, self.b_proj), pad_rows_to=8)
            self._build_decode_kernel_constants()

    @staticmethod
    def _fuse_checkpoint_projections(
        modules: Tuple[Fp8BlockScaleLinear, ...],
    ) -> Fp8BlockScaleLinear:
        """One FP8 projection over the rows of ``modules`` (block-aligned): their checkpoint codes
        and FP32 scale grids, copied before any scale preparation."""
        if any(module.out_features % 128 for module in modules[:-1]):
            raise ValueError("KDA FP8 fusion boundaries must be aligned to 128 rows")
        first = modules[0]
        fused = Fp8BlockScaleLinear(
            first.in_features,
            sum(module.out_features for module in modules),
            device=first.weight.device,
        )
        offset = 0
        scale_offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            fused.weight.data[offset : offset + rows].copy_(module.weight.data)
            offset += rows
            scale_rows = module.weight_scale.shape[0]
            fused.weight_scale.data[scale_offset : scale_offset + scale_rows].copy_(
                module.weight_scale.data
            )
            scale_offset += scale_rows
        fused.input_scale.data = first.input_scale.data
        fused.inv_input_scale.data = first.inv_input_scale.data
        return fused

    @staticmethod
    def _alias_projections_into_fused(modules: Tuple[nn.Module, ...], fused: nn.Module) -> None:
        """Repoint the FP8 source projections at row views of the fused codes and scale grid."""
        offset = 0
        scale_offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = fused.weight.data[offset : offset + rows]
            offset += rows
            scale_rows = module.weight_scale.shape[0]
            module.weight_scale.data = fused.weight_scale.data[
                scale_offset : scale_offset + scale_rows
            ]
            scale_offset += scale_rows

    def _build_decode_kernel_constants(self) -> None:
        """Kernel-layout constants of the decode / verify / prefill kernels."""
        self._w_q_t = (
            self.q_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_k_t = (
            self.k_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_v_t = (
            self.v_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._A_log_f32 = self.A_log.detach().float().contiguous()
        self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
        self._onorm_w_f32 = self.o_norm.weight.detach().float().contiguous()
        # Built eagerly too, so the first verify call never allocates (a capture-unsafe lazy
        # allocation).
        self._build_mtp_conv_weights()

    def _build_mtp_conv_weights(self) -> None:
        """Prebuild the packed-prefill and fused-verify convolution weights.

        Building them lazily at first use would allocate at runtime; under
        CUDA graph capture that would bake capture-pool pointers into the
        cached tensors.
        """
        conv_weights = tuple(
            conv.weight.detach().squeeze(1)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        self._packed_conv_weight = torch.cat(conv_weights, dim=0).to(torch.bfloat16).contiguous()
        # Channel-contiguous taps load in the input tile's layout in the fused kernel.
        self._fused_conv_weight_t = self._packed_conv_weight.t().contiguous()
        self._mtp_conv_weights = tuple(weight.float().contiguous() for weight in conv_weights)

    def _require_finalized(self) -> None:
        if self.qkvg_proj is None or self._bfa_proj_weight is None or self._onorm_w_f32 is None:
            raise RuntimeError(
                "Kimi K3 KDA runs on the fused projections and kernel constants that "
                "finalize_decode_weights() builds after weight load"
            )

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Select the eager path or breakable-CUDA-graph path.

        Returns the o_proj output.
        """
        if self.register_to_config and is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_heads, self.head_dim),
                dtype=torch.bfloat16,
            )
            maybe_bcg_kda_core_inplace(hidden_states, self.layer_idx_str, core)
            return self._project_output(core)

        return self._project_output(self._forward_impl(hidden_states, attn_metadata))

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Split the batch into its prefill, decode and verify rows and run their chains.

        ``output`` is the BCG post-o_norm, pre-o_proj core buffer. When it
        is supplied, subpaths fill and return its corresponding slices;
        otherwise this method returns an allocated core tensor. A
        verify-only batch without ``output`` may instead return the fused
        verify epilogue's MXFP8 core (see ``forward_verify_fused``).
        """
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        state_indices = mamba_metadata.state_indices[:batch_size]
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W - 1] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32 or bf16
        generation_state_indices = getattr(mamba_metadata, "generation_state_indices", None)
        if generation_state_indices is None:
            generation_state_indices = state_indices[num_prefills:]

        cores: List[torch.Tensor] = []
        if num_prefills > 0:
            snapshot = getattr(mamba_metadata, "kda_snapshot", None)
            inforward_stage = None
            if snapshot is not None and snapshot.stage_rows is not None:
                inforward_stage = attn_metadata.kv_cache_manager.inforward_snapshot_stage_views(
                    self.layer_idx
                )
            prefill_core = self.forward_prefill(
                hidden_states[:num_ctx_tokens],
                cu_seqlens,
                mamba_metadata,
                num_prefills,
                conv_pool,
                ssm_pool,
                state_indices[:num_prefills],
                layer_cache,
                output=output[:num_ctx_tokens] if output is not None else None,
                inforward_stage=inforward_stage,
            )
            if output is None:
                cores.append(prefill_core)
        if num_decodes > 0:
            decode_rows = num_tokens - num_ctx_tokens
            if decode_rows == num_decodes:
                decode_core = self.forward_decode(
                    hidden_states[num_ctx_tokens:num_tokens],
                    conv_pool,
                    ssm_pool,
                    generation_state_indices,
                    layer_cache,
                    output=output[num_ctx_tokens:num_tokens] if output is not None else None,
                )
                if output is None:
                    cores.append(decode_core)
            else:
                # Speculative verification: each generation request carries
                # 1 + draft_len tokens (drafts are padded to the static max,
                # so T is uniform).
                assert decode_rows % num_decodes == 0, (
                    f"ragged generation batch: {decode_rows} tokens for {num_decodes} requests"
                )
                verify_core = self.forward_verify(
                    hidden_states[num_ctx_tokens:num_tokens],
                    decode_rows // num_decodes,
                    layer_cache,
                    ssm_pool,
                    generation_state_indices,
                    output=(output[num_ctx_tokens:num_tokens] if output is not None else None),
                    # The MXFP8 core cannot be concatenated with a prefill
                    # core or stored into the BCG buffer.
                    allow_prequantized_output=output is None and num_prefills == 0,
                    context_only=getattr(attn_metadata.kv_cache_manager, "kda_context_only", False)
                    is True,
                )
                if output is None:
                    cores.append(verify_core)
        if output is not None:
            return output
        return cores[0] if len(cores) == 1 else torch.cat(cores, dim=0)

    def _project_output(self, core: Union[torch.Tensor, _KdaPrequantizedCore]) -> torch.Tensor:
        """o_proj on the post-o_norm core (the BF16 rows, or the verify epilogue's MXFP8 form)."""
        if isinstance(core, _KdaPrequantizedCore):
            return self.o_proj.forward_prequantized(core.activation, core.scale)
        return self.o_proj(core.reshape(-1, self.proj_size))

    def _sync_kda_replay_conv_window(self, layer_cache, slot_indices, conv_pool) -> None:
        """Seed the fused verify kernel's replay conv caches from the live committed conv pool.

        Their committed window (columns ``[0, W-1)``) must hold the live pool's ``W - 1`` raw
        inputs whenever prefill or plain decode advances it. No-op without replay caches (no
        speculative decoding, or a context-only server).
        """
        if not bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            return
        layer_cache.commit_conv_window(slot_indices, conv_pool)

    def _project_gate_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(beta, forget_gate)``: ``F.linear`` on the fused ``[f_a | b]`` weight, then
        ``F.linear`` of its ``f_a`` columns on ``f_b``. ``beta`` and the forget gate are the raw
        logits (the kernels apply the sigmoid / gate)."""
        hd, H = self.head_dim, self.num_heads
        bfa = F.linear(x, self._bfa_proj_weight)
        return bfa[..., hd : hd + H], F.linear(bfa[..., :hd], self.f_b_proj.weight)

    def _project_packed_conv_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(conv_input, onorm_g)`` of the prefill rows ``x`` ``[1, T, hidden]``: the QKVG GEMM,
        its ``[q | k | v]`` columns as a ``[T, 3D]`` view (for ``fused_kda_conv_prefill``), and
        its output-gate columns."""
        d = self.proj_size
        qkvg = self.qkvg_proj(x)
        onorm_g = qkvg[..., 3 * d : 4 * d]
        qkvg_2d = qkvg.squeeze(0)
        return qkvg_2d[:, : 3 * d], onorm_g

    def forward_prefill(
        self,
        x2d,
        cu_seqlens,
        mamba_metadata,
        num_prefills,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
        inforward_stage: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill ``num_prefills`` packed sequences and update their conv / recurrent rows.

        The convolution is ``fused_kda_conv_prefill``; Q/K leave it L2-normalized for
        ``kda_prefill``. A single unaligned sequence is chunk-padded by that kernel (with beta)
        where beta is unit-stride, else by the op. A BF16 state pool
        (and every in-forward snapshot plan) is staged through an fp32 copy; a ``kda_snapshot``
        plan also stores mid-chunk states and convolution windows.
        """
        if x2d.shape[0] == 0:
            reset_recurrent_state_rows(
                ssm_pool, slot_indices, mamba_metadata.has_initial_states[:num_prefills], conv_pool
            )
            self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
            return self._store_core(x2d.new_empty((0, self.num_heads, self.head_dim)), output)
        self._require_finalized()
        x = x2d.unsqueeze(0)  # [1, T, hidden]
        # Metadata owns stable int32 indices; never convert per layer.
        slot_indices = mamba_metadata.state_indices[:num_prefills]

        # Mid-chunk snapshots: the recurrent state and convolution window of a sequence after its
        # first tokens of this step. The prefill op writes the state (snapshot_rows /
        # snapshot_chunks). With ``inforward_stage`` (the in-forward snapshots of the cache
        # manager, ``snapshot.stage_rows`` set) the live slot keeps the state at the snapshot
        # point and the staging row receives the chunk-end state; otherwise the snapshot goes to
        # a pool slot of its own.
        snapshot = getattr(mamba_metadata, "kda_snapshot", None)
        if (snapshot is not None and snapshot.stage_rows is not None) != (
            inforward_stage is not None
        ):
            raise RuntimeError(
                "Kimi K3 KDA in-forward snapshots need both the plan's staging rows and the "
                "cache manager's staging views"
            )

        packed_conv, onorm_g = self._project_packed_conv_input(x)
        snapshot_windows = None
        if snapshot is not None:
            # The window is read before the convolution below, which may overwrite its input in
            # place. A live slot's window still feeds that convolution; it is swapped after it.
            snapshot_windows = _snapshot_conv_windows(
                packed_conv, conv_pool, snapshot, token_major=True
            )
            if inforward_stage is None:
                conv_pool.index_copy_(0, snapshot.slots, snapshot_windows)

        # Initial states: present for continuation chunks (chunked prefill)
        # and for prefix-cache hits (block reuse), where the previous
        # conv/recurrent state was onboarded into this request's slot.
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        prepared_state = self._prepare_prefill_state(
            cu_seqlens,
            mamba_metadata,
            num_prefills,
            ssm_pool,
            slot_indices,
            snapshot=snapshot,
        )
        # The prefill op needs a single unaligned sequence zero-padded to a chunk multiple. The
        # convolution kernel produces Q/K/V/beta at that length directly instead of padding
        # copies; g needs no pad because the op rebuilds g's sentinel tail from the real prefix.
        num_tokens = x2d.shape[0]
        padded_tokens = None
        single_sequence_length = getattr(mamba_metadata, "kda_single_sequence_length", None)
        if (
            num_prefills == 1
            and single_sequence_length is not None
            and single_sequence_length == num_tokens
            and num_tokens % _PREFILL_CHUNK != 0
        ):
            padded_tokens = num_tokens + _PREFILL_CHUNK - num_tokens % _PREFILL_CHUNK

        beta, g = self._project_gate_inputs(x)
        g = g.unflatten(-1, (self.num_heads, self.head_dim))

        # Imported here: the entry imports this package's kernels, and the package imports this
        # module.
        from ..ops.ssm.fused_kda_conv_prefill import fused_kda_conv_prefill

        prepad_beta = padded_tokens is not None and beta.stride(-1) == 1
        if not prepad_beta:
            beta = beta.float()
        outputs = fused_kda_conv_prefill(
            packed_conv,
            self._fused_conv_weight_t.t(),
            conv_pool,
            query_start_loc=mamba_metadata.query_start_loc[: num_prefills + 1],
            cache_indices=mamba_metadata.state_indices[:num_prefills],
            has_initial_state=has_init,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            padded_tokens=padded_tokens if prepad_beta else None,
            beta=beta if prepad_beta else None,
            normalize_qk=True,
        )
        if prepad_beta:
            q, k, v, beta = outputs
        else:
            q, k, v = outputs

        o = self._prefill_recurrence(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            mamba_metadata,
            num_prefills,
            ssm_pool,
            slot_indices,
            prepared_state,
            snapshot=snapshot,
            inforward_stage_ssm=None if inforward_stage is None else inforward_stage[0],
        )
        # Fused-verify replay caches: seed the committed conv window so the
        # first verify round convolves the correct history (pending drafts
        # are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
        if inforward_stage is not None:
            _swap_inforward_conv_windows(conv_pool, inforward_stage[1], snapshot, snapshot_windows)

        core = self._output_gate(o, onorm_g)
        return self._store_core(core, output)

    def _prepare_prefill_state(
        self,
        cu_seqlens: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        ssm_pool: torch.Tensor,
        slot_indices: torch.Tensor,
        snapshot: Optional["KDASnapshotPlan"] = None,
    ) -> _KDAPrefillState:
        """The state rows the prefill op updates in place, zeroed where a sequence starts fresh.

        The op takes fp32 V-first pool rows. A BF16 pool is staged through a dense fp32 copy of
        the addressed rows (plus the snapshot rows of a ``snapshot`` plan), as is an fp32 pool
        under an in-forward snapshot plan (whose snapshot rows are the live slots, which the
        kernel's chunk-end write would overwrite); :meth:`_prefill_recurrence` writes the copy
        back.
        """
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        staged_state = None
        kernel_pool, kernel_indices = ssm_pool, slot_indices
        snapshot_rows = None if snapshot is None else snapshot.pool_rows
        inforward = snapshot is not None and snapshot.stage_rows is not None
        if ssm_pool.dtype != torch.float32 or inforward:
            if snapshot is None:
                staged_state = _stage_state_rows(ssm_pool, slot_indices)
            else:
                staged_state = _stage_state_rows_with_snapshot_rows(
                    ssm_pool, slot_indices, snapshot.num_snapshots
                )
                snapshot_rows = snapshot.staged_rows
            kernel_pool = staged_state
            kernel_indices = mamba_metadata._arange_buffer[:num_prefills]
        if not self._dispatch.can_use_indexed_prefill(
            state_pool=kernel_pool,
            state_indices=kernel_indices,
            has_initial_states=has_init,
            cu_seqlens=cu_seqlens,
            num_sequences=num_prefills,
        ):
            raise RuntimeError(
                "Kimi K3 KDA prefill: the kda_prefill entry does not take this state pool / "
                "metadata layout (fp32 [slots, H, V, K] rows, 16-byte aligned int32 / int64 "
                "indices, bool has_initial_states)"
            )
        # The convolution clears fresh convolution rows itself.
        reset_recurrent_state_rows(kernel_pool, kernel_indices, has_init)
        return _KDAPrefillState(
            kernel_pool, kernel_indices, staged_state, snapshot_rows=snapshot_rows
        )

    def _prefill_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        ssm_pool: torch.Tensor,
        slot_indices: torch.Tensor,
        state: _KDAPrefillState,
        *,
        snapshot: Optional["KDASnapshotPlan"] = None,
        inforward_stage_ssm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``kda_prefill`` on the token-major L2-normalized Q/K, V, the raw gate and
        beta logits, updating ``state``'s fp32 V-first ``[slots, H, D, D]`` rows in place, then
        the write-back of a staged copy (and the snapshot moves)."""
        o = self._dispatch.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self._A_log_f32,
            dt_bias=self._dt_bias_f32,
            scale=self.head_k_dim**-0.5,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=getattr(mamba_metadata, "kda_chunk_indices", None),
            chunk_size=_PREFILL_CHUNK,
            state_pool=state.kernel_pool,
            state_indices=state.kernel_indices,
            varlen_is_aligned=getattr(mamba_metadata, "kda_varlen_is_aligned", None),
            single_sequence_length=getattr(mamba_metadata, "kda_single_sequence_length", None),
            snapshot_rows=state.snapshot_rows if snapshot is not None else None,
            snapshot_chunks=None if snapshot is None else snapshot.chunks,
        )

        # The convolution persisted the live convolution pool in place.
        staged_state = state.staged_state
        if staged_state is not None:
            if snapshot is None:
                _writeback_state_rows(ssm_pool, slot_indices, staged_state)
            else:
                _writeback_state_rows(ssm_pool, slot_indices, staged_state[:num_prefills])
                if snapshot.stage_rows is not None:
                    # In-forward: the chunk-end states move to the staging rows; the live
                    # slots (``snapshot.slots``) take the states at the snapshot points.
                    inforward_stage_ssm.index_copy_(
                        0, snapshot.stage_rows, ssm_pool.index_select(0, snapshot.slots)
                    )
                _writeback_state_rows(ssm_pool, snapshot.slots, staged_state[num_prefills:])
        elif snapshot is not None and snapshot.stage_rows is not None:
            raise RuntimeError("Kimi K3 KDA in-forward snapshots need a staged state copy")
        return o

    def forward_decode(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One token per request: the QKVG GEMM (next to ``[f_a | b]`` -> ``f_b`` on the
        projection aux stream up to 128 rows in CUDA graphs), then the fused decode kernel, which
        convolves, gates, runs the recurrence and the gated RMSNorm, and updates the conv and
        fp32 state pools by slot. A BF16 state pool is staged through dense copies of the
        addressed rows (fp32 state, conv window as is) and written back (state rounded).
        """
        self._require_finalized()
        d = self.proj_size
        hd = self.head_dim
        H = self.num_heads
        B = x2d.shape[0]

        staged_state = staged_conv = None
        slots_long = None
        kernel_state, kernel_state_indices, conv_src = ssm_pool, slot_indices, conv_pool
        if ssm_pool.dtype != torch.float32:
            slots_long = slot_indices.long()
            staged_state = _stage_state_rows(ssm_pool, slots_long)
            staged_conv = conv_pool.index_select(0, slots_long)
            kernel_state, kernel_state_indices, conv_src = staged_state, None, staged_conv

        # The decode kernel writes its core in place: the BCG core buffer, or outside breakable
        # CUDA graphs a persistent buffer whose pointer stays stable across graph captures.
        if output is not None:
            kda_out = output.view(B, 1, H, hd)
        else:
            if self._o_dense is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Kimi K3 KDA decode: its output buffer is allocated by the eager forward "
                        "that precedes CUDA graph capture"
                    )
                self._o_dense = torch.empty(
                    max(conv_pool.shape[0], B),
                    1,
                    H,
                    hd,
                    dtype=torch.bfloat16,
                    device=x2d.device,
                )
            else:
                assert self._o_dense.shape[0] >= B, (
                    f"KDA decode output buffer holds {self._o_dense.shape[0]} rows "
                    f"but the decode batch is {B}; reallocating would corrupt "
                    f"previously captured CUDA graphs"
                )
            kda_out = self._o_dense[:B]

        projection_aux_stream = (
            self._projection_aux_stream if B <= _KDA_BFA_MULTISTREAM_MAX_ROWS else None
        )
        qkvg, (beta, g) = fork_join(
            lambda: self.qkvg_proj(x2d),
            lambda: self._project_gate_inputs(x2d),
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
        )
        x_qkvg = qkvg[:, : 3 * d]
        onorm_g = qkvg[:, 3 * d : 4 * d]

        # Section views retain the live pool's slot stride, including V2
        # manager padding. The kernel uses ssm_state_indices for both pools.
        o = self._dispatch.decode_kda(
            x_q=x_qkvg[:, :d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_k=x_qkvg[:, d : 2 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_v=x_qkvg[:, 2 * d : 3 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            w_q_t=self._w_q_t,
            w_k_t=self._w_k_t,
            w_v_t=self._w_v_t,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=conv_src[:, :d],
            cs_k=conv_src[:, d : 2 * d],
            cs_v=conv_src[:, 2 * d :],
            A_log=self._A_log_f32,
            g=g.unflatten(-1, (H, hd)).unsqueeze(0),
            dt_bias=self._dt_bias_f32,
            beta=beta.unsqueeze(0),
            state=kernel_state,
            onorm_g=onorm_g.unflatten(-1, (H, hd)).unsqueeze(0),
            onorm_weight=self._onorm_w_f32,
            out=kda_out,
            ssm_state_indices=kernel_state_indices,
            scale=hd**-0.5,
            onorm_eps=self.o_norm.eps,
            lower_bound=self.gate_lower_bound,
            use_beta_sigmoid_in_kernel=True,
            verbose=False,
            update_conv_cache=True,
        )
        if staged_state is not None:
            conv_pool.index_copy_(0, slots_long, staged_conv)
            _writeback_state_rows(ssm_pool, slots_long, staged_state)
        # Fused-verify replay caches (spec decoding only): keep the
        # committed conv window in sync with the plain-decode advance.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

        # The kernel applied the output gate / RMSNorm and returns the supplied buffer.
        return o.view(B, H, hd)

    def forward_verify(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
        allow_prequantized_output: bool = False,
        context_only: bool = False,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Speculative verification: advance each request ``num_steps`` tokens (1 golden +
        ``num_steps - 1`` padded drafts) with the fused verify kernel on the cache manager's
        replay caches (:meth:`forward_verify_fused`).

        On a context-only server the rows are dummies and take
        ``forward_verify_context_only`` instead (the manager marks such a server with
        ``kda_context_only``, passed in as ``context_only``).
        """
        if bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            return self.forward_verify_fused(
                x2d,
                num_steps,
                layer_cache,
                ssm_pool,
                slot_indices,
                output=output,
                allow_prequantized_output=allow_prequantized_output,
            )
        if context_only or getattr(layer_cache, "context_only", False):
            return self.forward_verify_context_only(x2d, output)
        raise RuntimeError(
            "Kimi K3 KDA verification runs the fused verify kernel on the cache manager's KDA "
            "replay caches, which this layer cache does not have"
        )

    def forward_verify_context_only(
        self, x2d: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Verification rows on a context-only server.

        Such a server (disaggregated context server) runs no real generation
        request; its generation rows are dummies (warmup, attention-DP
        padding) whose outputs are discarded, and its cache manager keeps no
        replay caches for them. The recurrence is skipped and the live pools
        are left untouched; the output projection (``_project_output``) still
        runs on a zero attention core, so the layer's projection and collectives
        match those of a real verification step.
        """
        core = x2d.new_zeros(x2d.shape[0], self.num_heads, self.head_dim)
        return self._store_core(core, output)

    def _project_verify_inputs(self, x: torch.Tensor, num_rows: int, gate_from_fa: bool) -> tuple:
        """The verify projections: the QKVG GEMM on the calling stream, the gate projection on
        the projection aux stream.

        Returns ``(q, k, v, forget_gate, beta, onorm_g, f_a)``, the q / k / v / output-gate
        sections as strided views of the QKVG output (the verify kernel reads them in place) and
        beta as one of the ``[f_a | b]`` output. ``f_a`` is the BF16 ``f_a`` view where
        ``gate_from_fa`` and the f_a-mode fab kernel served these rows (``forget_gate`` is then
        None and the verify kernel computes it), else None.

        The gate is
        ``k3_kda_fab_fa`` (f_a mode), ``k3_kda_fab_proj`` (beta and the forget gate) where they
        accept the shapes, else ``F.linear`` ``[f_a | b]`` -> ``f_b``.
        """
        fused_qkvg = self.qkvg_proj

        def _project_qkvg() -> torch.Tensor:
            return fused_qkvg(x)

        bfa_weight = self._bfa_proj_weight
        fb_weight = self.f_b_proj.weight
        fab_fa = gate_from_fa and _use_fab_fa(x, bfa_weight, fb_weight, self.num_heads)
        fused_fab = not fab_fa and _use_fused_fab(x, bfa_weight, fb_weight, self.num_heads)

        def _project_gates() -> tuple:
            if fab_fa:
                from ..ops.ssm.k3_kda_fab_fa import k3_kda_fab_fa

                # (beta, f_a): the verify kernel derives the forget gate from f_a.
                # W_fb is warmed in L2 for the verify kernel's per-head load.
                bfa = k3_kda_fab_fa(
                    x.reshape(-1, x.shape[-1]), bfa_weight, self.num_heads, fb_weight
                ).view(*x.shape[:-1], bfa_weight.shape[0])
                return bfa[..., self.head_dim : self.head_dim + self.num_heads], bfa[
                    ..., : self.head_dim
                ]
            if fused_fab:
                from ..ops.ssm.k3_kda_fab_proj import k3_kda_fab_proj

                beta, forget_gate = k3_kda_fab_proj(
                    x.reshape(-1, x.shape[-1]), bfa_weight, fb_weight, self.num_heads
                )
                return (
                    beta.view(*x.shape[:-1], self.num_heads),
                    forget_gate.view(*x.shape[:-1], fb_weight.shape[0]),
                )
            return self._project_gate_inputs(x)

        if fab_fa or fused_fab:
            max_rows = _KDA_FAB_FUSED_MAX_ROWS
        else:
            # Large batches: the unfused chain still overlaps the (all-SM) QKVG GEMM better
            # than it runs after it.
            max_rows = num_rows
        projection_aux_stream = self._projection_aux_stream if 0 < num_rows <= max_rows else None
        qkvg, (beta, gate) = fork_join(
            _project_qkvg,
            _project_gates,
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
        )

        d = self.proj_size
        q_proj, k_proj, v_proj = qkvg[..., : 3 * d].split(d, dim=-1)
        onorm_g = qkvg[..., 3 * d : 4 * d]
        if fab_fa:
            return q_proj, k_proj, v_proj, None, beta, onorm_g, gate
        return q_proj, k_proj, v_proj, gate, beta, onorm_g, None

    def forward_verify_fused(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
        allow_prequantized_output: bool = False,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Fused multi-token verify (``mtp_ops.py``: the v1 / v2 / two-CTA kernels).

        The runtime packs ``num_steps`` new rows contiguously per request, so the kernel derives
        the row offsets from the request index and reads each request's pending accepted count
        from the slot-indexed replay pool (``packed_token_layout``). One launch replays the
        previous round's accepted drafts from the per-slot replay caches, then processes the new
        tokens, committing the recurrent state and conv windows in place after the golden token
        and caching the new drafts, and applies the gated RMSNorm in its epilogue:

        * with ``fa`` / ``w_fb``: the forget gate ``bf16(f_a @ W_fb^T)`` computed in the
          kernel (BF16 core);
        * ``quantize_output``: where ``allow_prequantized_output`` and ``o_proj`` takes a
          prequantized activation for this row count (SM107), the core quantized to MXFP8 for
          ``o_proj.forward_prequantized``;
        * otherwise the BF16 core, a programmatic dependent of the QKVG GEMM.
        """
        self._require_finalized()
        num_generations = x2d.shape[0] // num_steps
        num_spec = num_steps - 1
        H = self.num_heads
        K = self.head_k_dim
        x = x2d.view(num_generations, num_steps, -1)  # [B, T, hidden]
        T_total = num_generations * num_steps

        # The MXFP8 core cannot be concatenated with a prefill core or stored into the BCG buffer
        # (``allow_prequantized_output``).
        mxfp8_output = (
            allow_prequantized_output
            and output is None
            and _o_proj_takes_mxfp8(self.o_proj, T_total)
            and callable(getattr(self.o_proj, "forward_prequantized", None))
        )
        q_proj, k_proj, v_proj, forget_gate, beta_proj, onorm_g, f_a = self._project_verify_inputs(
            x, T_total, gate_from_fa=not mxfp8_output
        )
        x_q = q_proj.view(1, T_total, H, K)
        x_k = k_proj.view(1, T_total, H, K)
        x_v = v_proj.view(1, T_total, H, self.head_dim)
        # Raw gate / beta: the kernel applies dt_bias, A_log, the
        # lower-bound sigmoid gate, and the beta sigmoid itself.
        beta = beta_proj.view(1, T_total, H)
        w_q, w_k, w_v = self._mtp_conv_weights
        onorm_g_view = onorm_g.view(1, T_total, H, self.head_dim)

        if f_a is not None:
            out = self._dispatch.mtp_verify_fused_fb(
                x_q=x_q,
                x_k=x_k,
                x_v=x_v,
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                cs_q=layer_cache.kda_conv_q,
                cs_k=layer_cache.kda_conv_k,
                cs_v=layer_cache.kda_conv_v,
                fa=f_a.view(1, T_total, K),
                w_fb=self.f_b_proj.weight.detach(),
                beta=beta,
                A_log=self.A_log.detach(),
                dt_bias=self.dt_bias.detach(),
                recurrent_state=ssm_pool,
                k_cache=layer_cache.kda_k_cache,
                g_cache=layer_cache.kda_g_cache,
                v_cache=layer_cache.kda_v_cache,
                beta_cache=layer_cache.kda_beta_cache,
                ssm_state_indices=slot_indices,
                num_spec=num_spec,
                num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
                lower_bound=self.gate_lower_bound,
                scale=self.head_k_dim**-0.5,
                onorm_g=onorm_g_view,
                onorm_weight=self._onorm_w_f32,
                onorm_eps=self.o_norm.eps,
                use_pdl=True,
            )
            return self._store_core(out, output)

        out = self._dispatch.mtp_verify(
            mxfp8_output=mxfp8_output,
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            g=forget_gate.view(1, T_total, H, K),
            beta=beta,
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking tensors.
            A_log=self.A_log.detach(),
            dt_bias=self.dt_bias.detach(),
            recurrent_state=ssm_pool,
            k_cache=layer_cache.kda_k_cache,
            g_cache=layer_cache.kda_g_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices,
            cu_seqlens=None,
            num_spec=num_spec,
            num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            packed_token_layout=True,
            lower_bound=self.gate_lower_bound,
            scale=self.head_k_dim**-0.5,
            onorm_g=onorm_g_view,
            onorm_weight=self._onorm_w_f32,
            onorm_eps=self.o_norm.eps,
            fuse_output_norm=True,
            # A BF16-output verify launches as a programmatic dependent of the QKVG projection:
            # its state, replay and weight loads run before griddepcontrol.wait.
            use_pdl=not mxfp8_output,
        )
        if mxfp8_output:
            if not isinstance(out, tuple):
                raise RuntimeError("MXFP8-output MTP KDA must return activation and scale")
            return _KdaPrequantizedCore(*out)
        return self._store_core(out, output)

    def _output_gate(self, o: torch.Tensor, onorm_g: torch.Tensor) -> torch.Tensor:
        """The gated output RMSNorm ``rmsnorm(o) * sigmoid(g)`` (``rms_norm_gated_token_major``)
        of the prefill core."""
        return rms_norm_gated_token_major(
            o.reshape(-1, self.head_dim),
            onorm_g.reshape(-1, self.num_heads, self.head_dim),
            self.o_norm.weight,
            self.o_norm.eps,
            gate_activation="sigmoid",
        )

    def _store_core(self, core: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the normalized core ``[tokens, H, D]``, copied into the BCG buffer ``output``
        where one is given."""
        core = core.reshape(-1, self.num_heads, self.head_dim)
        if output is None:
            return core
        output.copy_(core)
        return output
