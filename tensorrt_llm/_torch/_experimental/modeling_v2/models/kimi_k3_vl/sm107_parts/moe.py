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
"""Kimi K3's routed experts of one MoE layer on this rank, as a chain of kernel launches.

:class:`K3RoutedExperts` runs one forward on its NVFP4 weights (:class:`K3Nvfp4Experts`, the
``backend`` attribute DWDP binds) with one of the CuTe DSL kernel sets:

``CUTEDSL`` (context)::

    routing (route tail | noaux_tc_route_sort | noaux_tc_op)
    fp4_quantize                                     [on the aux stream up to 1024 rows]
    moe_sort                                         [unless the routing wrote the tables]
    moe_output_memset_inplace                        [aux stream]
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_{rubin,blackwell}   FC1 + SiTU + FP4 requant
    cute_dsl_nvfp4_grouped_gemm_finalize_inplace_{rubin,blackwell}    FC2 + weighted scatter-add

``CUTEDSL_FC12`` (generation, SM107)::

    routing, fp4_quantize, moe_sort (as above)
    cute_dsl_nvfp4_fc12_fused_rubin                  memset + FC1 + SiTU + FC2 + scatter-add

The routing is the caller's (a ``route`` callable run per chunk, returning a
:class:`SortedRouting`): the experts only run the kernels. The output is this rank's partial sum
(its experts, its slice of the intermediate dim); the caller reduces it.

Kimi K3's NVFP4 routed-expert weights on one rank, in the layout the CuTe DSL MoE kernels read.

:class:`K3Nvfp4Experts` holds one MoE layer's experts ``[slot_start, slot_end)`` (all of them under
DWDP once the setup widened the range) as the kernels take them:

* ``w3_w1_weight`` ``[E, N1, H / 16]`` int64 (16 E2M1 values each) and ``w3_w1_weight_scale``
  ``[E, N1, H / 64]`` int32 (4 E4M3 block scales each, 128x4-swizzled), ``N1 = align128(2 I)``;
  FC1 rows in the gate/up order of the kernels of this GPU (:func:`fc1_rows_for_kernels`).
* ``w2_weight`` ``[E, H, N1 / 32]`` int64 and ``w2_weight_scale`` ``[E, H, N1 / 128]`` int32
  (swizzled).
* ``fc31_input_scale`` / ``fc2_input_scale``: the reciprocal of the largest checkpoint
  ``input_scale`` of the rank's experts (FC1 input, FC2 input); ``fc31_alpha`` / ``fc2_alpha``
  ``[E]``: the per-expert dequantization scales of the two GEMMs.

The checkpoint streams in expert by expert (:meth:`load_expert`); :meth:`finalize` runs once the
last of the rank's experts is in. The parameter names are the ones DWDP rebinds and exchanges.

What the Kimi K3 MoE routing hands to the routed experts of the same forward.

``SortTables``: the six ``moe_sort`` outputs of this rank's experts at one tile size.
``SortedRouting``: the routing a kernel computed (ids, weights) together with this rank's
``SortTables`` where the routing kernel wrote them.
"""

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..common.multi_stream import do_multi_stream
from ..common.parallel import TensorParallelMode
from .layers import load_weight_shard


@dataclass(frozen=True)
class SortTables:
    """The outputs of ``trtllm::moe_sort`` (or of a routing kernel that writes the same tables) for
    this rank's experts at ``tile_size`` rows per tile: ``(tile_idx_to_group_idx,
    tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
    total_num_padded_tokens, num_non_exiting_tiles)``."""

    tile_size: int
    outputs: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class SortedRouting:
    """The routing of one forward: the top-k ids and weights, with this rank's ``moe_sort`` tables
    where the routing kernel wrote them (``None``: the experts run ``moe_sort``)."""

    token_selected_experts: torch.Tensor  # int32 [num_tokens, top_k]
    token_final_scales: torch.Tensor  # fp32 [num_tokens, top_k]
    sort_tables: Optional[SortTables]


#: Values per NVFP4 block scale.
SF_VEC_SIZE = 16
#: FC1 rows (gate + up) are padded to a multiple of this.
_ROW_ALIGNMENT = 128


def ep_partition(num_experts: int, ep_size: int, ep_rank: int) -> Tuple[int, int]:
    """``[start, end)`` of ``ep_rank``'s experts: ranks below ``num_experts % ep_size`` hold one
    expert more."""
    if num_experts < ep_size:
        raise ValueError(f"{num_experts} experts cannot be split over {ep_size} ranks")
    base, remainder = divmod(num_experts, ep_size)
    start = ep_rank * base + min(ep_rank, remainder)
    return start, start + base + (1 if ep_rank < remainder else 0)


def fc1_rows_for_kernels(x: torch.Tensor) -> torch.Tensor:
    """FC1 rows (dim 0: the up half, then the gate half) in the order the CuTe DSL FC1 kernels of
    this GPU read: Rubin (SM107) groups of 16 gate rows then 16 up rows, Blackwell groups of 64
    up rows then 64 gate rows. Weights and their block scales take the same order."""
    rows = x.shape[0]
    group, gate_first = (16, True) if get_sm_version() == 107 else (64, False)
    assert rows % (2 * group) == 0, (rows, group)
    halves = x.view(2, rows // (2 * group), group, *x.shape[1:])
    if gate_first:
        halves = halves.flip(0)
    return halves.transpose(0, 1).contiguous().view(x.shape)


def _pad_to(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """``src`` zero-padded at the end of both dims to ``dst``'s 2D shape."""
    rows, cols = src.shape
    if (rows, cols) != tuple(dst.shape):
        src = torch.nn.functional.pad(
            src, (0, dst.shape[1] - cols, 0, dst.shape[0] - rows), "constant", 0
        ).contiguous()
    return src


def _swizzle_block_scales(dst: torch.Tensor) -> None:
    """Swizzle one expert's linear E4M3 block scales (int32-packed, in place) to the 128x4 layout."""
    swizzled = (
        torch.ops.trtllm.block_scale_interleave(dst.view(torch.uint8))
        .view(dst.dtype)
        .reshape(dst.shape)
    )
    torch.cuda.synchronize()
    dst.copy_(swizzled)


class K3Nvfp4Experts(nn.Module):
    """One MoE layer's NVFP4 routed experts on this rank (see the module docstring).

    ``tp_size`` / ``tp_rank``: MoE tensor parallelism, slicing every expert's intermediate dim (FC1
    rows, FC2 columns); ``ep_size`` / ``ep_rank``: which experts the rank holds.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
    ) -> None:
        super().__init__()
        if hidden_size % 4 != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by 4")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.ep_size, self.ep_rank = ep_size, ep_rank
        self.intermediate_size_per_partition = intermediate_size // tp_size
        self.scaling_vector_size = SF_VEC_SIZE
        # DWDP's fixup re-materializes activation constants only on modules that declare one.
        self.activation = None
        self.set_expert_range(*ep_partition(num_experts, ep_size, ep_rank))
        self._lock = threading.Lock()
        self._staged: Optional[Dict] = None
        self._streamed_expert_slots: set = set()

    def set_expert_range(self, start: int, end: int) -> None:
        """Hold experts ``[start, end)`` (before the weights are created)."""
        self.num_slots = self.num_experts
        self.expert_size_per_partition = end - start
        self.slot_start, self.slot_end = start, end
        self.initial_local_expert_ids = list(range(start, end))
        self.initial_global_assignments = list(range(self.num_experts))

    def create_weights(self) -> None:
        num_local = self.expert_size_per_partition
        n1 = -(-2 * self.intermediate_size_per_partition // _ROW_ALIGNMENT) * _ROW_ALIGNMENT
        h = self.hidden_size
        packed = 16  # E2M1 values per int64
        scales = 4  # E4M3 block scales per int32

        def param(shape, dtype, fill=None):
            data = (
                torch.empty(shape, dtype=dtype)
                if fill is None
                else torch.full(shape, fill, dtype=dtype)
            )
            return nn.Parameter(data, requires_grad=False)

        self.w3_w1_weight = param((num_local, n1, h // packed), torch.int64)
        self.w2_weight = param((num_local, h, n1 // 2 // packed), torch.int64)
        self.w3_w1_weight_scale = param(
            (num_local, n1, h // SF_VEC_SIZE // scales), torch.int32, fill=1
        )
        self.w2_weight_scale = param(
            (num_local, h, n1 // 2 // SF_VEC_SIZE // scales), torch.int32, fill=1
        )
        self.fc31_input_scale = param((), torch.float32, fill=1.0)
        self.fc2_input_scale = param((), torch.float32, fill=1.0)
        self.fc31_alpha = param((num_local,), torch.float32, fill=1.0)
        self.fc2_alpha = param((num_local,), torch.float32, fill=1.0)

    # ------------------------------------------------------------------
    # Loading: expert by expert, then one finalize per layer.
    # ------------------------------------------------------------------
    def prepare_streaming_load(self) -> None:
        """Create the per-layer staging the loader threads fill (before any of them runs)."""
        self._staged = {"input_scales": {}, "weight_scale_2": {}}
        self._streamed_expert_slots = set()

    def load_expert(
        self,
        *,
        global_expert_id: int,
        local_slot_id: int,
        w1_weight: torch.Tensor,
        w1_weight_scale: torch.Tensor,
        w1_weight_scale_2: torch.Tensor,
        w1_input_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_weight_scale_2: torch.Tensor,
        w2_input_scale: torch.Tensor,
        w3_weight: torch.Tensor,
        w3_weight_scale: torch.Tensor,
        w3_weight_scale_2: torch.Tensor,
        w3_input_scale: torch.Tensor,
    ) -> None:
        """Load one checkpoint expert (packed E2M1 ``uint8`` weights, E4M3 block scales, FP32
        global scales) into ``local_slot_id``: this rank's slice of the intermediate dim, FC1 as
        ``[w3 | w1]`` padded to ``N1`` rows, block scales swizzled."""
        if not 0 <= local_slot_id < self.expert_size_per_partition:
            raise IndexError(
                f"local_slot_id={local_slot_id} is outside [0, {self.expert_size_per_partition})"
            )
        if global_expert_id != self.initial_local_expert_ids[local_slot_id]:
            raise ValueError(
                f"local slot {local_slot_id} holds expert "
                f"{self.initial_local_expert_ids[local_slot_id]}, got {global_expert_id}"
            )
        if self._staged is None:
            raise RuntimeError("prepare_streaming_load() must run before load_expert()")
        if local_slot_id in self._streamed_expert_slots:
            raise ValueError(f"NVFP4 local slot {local_slot_id} was loaded twice")
        for name, value in (("w1", w1_weight), ("w2", w2_weight), ("w3", w3_weight)):
            if value.dtype != torch.uint8:
                raise TypeError(f"{name}_weight must hold packed NVFP4 uint8, got {value.dtype}")

        tp, rank = self.tp_size, self.tp_rank
        column, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW

        dst = self.w3_w1_weight.data[local_slot_id]
        w1 = load_weight_shard(w1_weight, tp, rank, column, device=dst.device)
        w3 = load_weight_shard(w3_weight, tp, rank, column, device=dst.device)
        fc1 = torch.cat([w3.contiguous().view(dst.dtype), w1.contiguous().view(dst.dtype)], dim=0)
        dst.copy_(_pad_to(fc1, dst), non_blocking=True)

        dst = self.w2_weight.data[local_slot_id]
        w2 = load_weight_shard(w2_weight, tp, rank, row, device=dst.device)
        dst.copy_(_pad_to(w2.contiguous().view(dst.dtype), dst), non_blocking=True)

        dst = self.w3_w1_weight_scale.data[local_slot_id]
        w1_s = load_weight_shard(w1_weight_scale, tp, rank, column, device=dst.device)
        w3_s = load_weight_shard(w3_weight_scale, tp, rank, column, device=dst.device)
        fc1_s = torch.cat(
            [w3_s.contiguous().view(dst.dtype), w1_s.contiguous().view(dst.dtype)], dim=0
        )
        dst.copy_(_pad_to(fc1_s, dst))
        _swizzle_block_scales(dst)

        dst = self.w2_weight_scale.data[local_slot_id]
        w2_s = load_weight_shard(w2_weight_scale, tp, rank, row, device=dst.device)
        cols = dst.shape[1] * 4
        if cols > w2_s.shape[1]:
            w2_s = torch.nn.functional.pad(
                w2_s, (0, cols - w2_s.shape[1]), "constant", 0
            ).contiguous()
        dst.copy_(_pad_to(w2_s.view(dst.dtype), dst))

        self._staged["weight_scale_2"][local_slot_id] = {
            "w1": w1_weight_scale_2,
            "w3": w3_weight_scale_2,
            "w2": w2_weight_scale_2,
        }
        self._staged["input_scales"][global_expert_id] = {
            "w1": w1_input_scale[...].reshape([]),
            "w3": w3_input_scale[...].reshape([]),
            "w2": w2_input_scale[...].reshape([]),
        }
        self._streamed_expert_slots.add(local_slot_id)

    def loaded_slots(self) -> set:
        return self._streamed_expert_slots

    def finalize(self) -> None:
        """After the last expert of the layer: swizzle the FC2 block scales, compute the global
        input scales and the per-expert alphas, and put FC1 rows and block scales in the kernels'
        gate/up order."""
        staged, self._staged = self._staged, None
        if staged is None:
            return
        for slot in range(self.w2_weight_scale.shape[0]):
            _swizzle_block_scales(self.w2_weight_scale.data[slot])

        fc31_values, fc2_values = [], []
        for expert_id, entry in staged["input_scales"].items():
            assert torch.allclose(entry["w1"], entry["w3"]), (
                f"expert {expert_id}: w1 input_scale != w3 input_scale"
            )
            fc31_values.append(entry["w1"])
            fc2_values.append(entry["w2"])
        if fc31_values:
            self.fc31_input_scale.data.copy_(torch.stack(fc31_values).max().reciprocal())
        if fc2_values:
            self.fc2_input_scale.data.copy_(torch.stack(fc2_values).max().reciprocal())

        fc31_input_scale = self.fc31_input_scale.data
        fc2_input_scale = self.fc2_input_scale.data
        for slot, scales in staged["weight_scale_2"].items():
            w1_ws2, w3_ws2 = scales["w1"], scales["w3"]
            if not torch.allclose(w1_ws2, w3_ws2):
                logger.warning(
                    f"w1_weight_scale_2 != w3_weight_scale_2 ({w1_ws2} != {w3_ws2}), "
                    "selecting the larger value. Accuracy may be affected."
                )
                w1_ws2 = torch.max(w1_ws2, w3_ws2)
            gate_ws2 = 1.0 / w1_ws2[...].reshape([])
            self.fc31_alpha.data[slot].copy_(1.0 / (fc31_input_scale * gate_ws2))
            w2_ws2 = 1.0 / scales["w2"][...].reshape([])
            self.fc2_alpha.data[slot].copy_(1.0 / (fc2_input_scale * w2_ws2))

        n = self.intermediate_size_per_partition * 2
        k = self.hidden_size
        sf_cols = k // SF_VEC_SIZE
        for slot in range(self.w3_w1_weight.shape[0]):
            weight = self.w3_w1_weight.data[slot]
            weight.copy_(fc1_rows_for_kernels(weight.view(torch.uint8)).view(weight.dtype))
            scale = self.w3_w1_weight_scale.data[slot]
            linear = torch.ops.trtllm.block_scale_interleave_reverse(
                scale.view(torch.uint8).view(-1, n, sf_cols)
            ).view(n, sf_cols)
            swizzled = torch.ops.trtllm.block_scale_interleave(
                fc1_rows_for_kernels(linear).view(-1, n, sf_cols)
            ).view(n, sf_cols)
            scale.copy_(swizzled.view(scale.dtype))


#: The kernel sets :class:`K3RoutedExperts` runs, by ``moe_config.backend``.
KERNEL_SETS = ("CUTEDSL", "CUTEDSL_FC12")

# The CUTEDSL input quantization runs on the auxiliary stream up to this many tokens
# (``async_input_quant``).
_ASYNC_INPUT_QUANT_MAX_TOKENS = 1024

#: A route: ``route(router_logits, local_expert_offset, local_num_experts)``.
Route = Callable[[torch.Tensor, int, int], SortedRouting]


def _wait(event: Optional[torch.cuda.Event]) -> None:
    if event is not None:
        event.wait()


def _tile_runners():
    """The AutoTuner adapters of the two kernel sets: the routing tile is the tuned tactic."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.fused_fc12 import CuteDslFc12FusedMoENvfp4Runner
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.runners import CuteDslFusedMoENvfp4Runner

    return {
        "CUTEDSL": (CuteDslFusedMoENvfp4Runner, "CuteDslFusedMoE::run_moe_nvfp4"),
        "CUTEDSL_FC12": (CuteDslFc12FusedMoENvfp4Runner, "CuteDslFc12FusedMoE::run_moe_nvfp4"),
    }


class K3RoutedExperts(nn.Module):
    """One MoE layer's routed experts on this rank (see the module docstring).

    ``mapping``: the routed experts' mapping (``moe_tp_size`` x ``moe_ep_size`` over the TP
    group). ``aux_stream``: the stream the CUTEDSL output memset and input quantization fork to.
    ``max_chunk_tokens``: rows per launch chain (``moe_max_num_tokens``). ``fc12_die_major``: the
    target's die-major FC12 expert layout and schedule (``K3Sm107Constants.fc12_die_major``; not
    run yet, see :meth:`_run`).
    """

    def __init__(
        self,
        *,
        kernels: str,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        layer_idx: int,
        situ_beta: float,
        situ_linear_beta: float,
        aux_stream: Optional[torch.cuda.Stream],
        max_chunk_tokens: int,
        fc12_die_major: bool,
        async_input_quant: bool,
    ) -> None:
        super().__init__()
        if kernels not in KERNEL_SETS:
            raise ValueError(f"Kimi K3 routed experts run {KERNEL_SETS}, not {kernels!r}")
        if kernels == "CUTEDSL_FC12" and (
            get_sm_version() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE
        ):
            raise ValueError("CUTEDSL_FC12 runs on SM107 with the Rubin CuTe DSL only")
        if mapping.enable_attention_dp and mapping.tp_size > 1:
            raise NotImplementedError(
                "Kimi K3 CuTe DSL routed experts take the tokens of their own rank only "
                "(no attention-DP dispatch)"
            )
        self.kernels = kernels
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.situ_beta, self.situ_linear_beta = float(situ_beta), float(situ_linear_beta)
        self.moe_ep_size = mapping.moe_ep_size
        self.max_chunk_tokens = max(1, int(max_chunk_tokens))
        self.fc12_die_major = bool(fc12_die_major)
        self.async_input_quant = bool(async_input_quant)
        self.backend = K3Nvfp4Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=mapping.moe_tp_size,
            tp_rank=mapping.moe_tp_rank,
            ep_size=mapping.moe_ep_size,
            ep_rank=mapping.moe_ep_rank,
        )
        self.dwdp_manager = None
        self.enable_dwdp = False
        self._decide_dwdp()
        self.backend.create_weights()
        # SMs the FC12 grids leave to kernels running beside them (``reserved_sms``).
        self.sm_reserve = 0
        self._aux_stream = aux_stream
        self._main_event = torch.cuda.Event()
        self._memset_event = torch.cuda.Event()
        self._quant_event = torch.cuda.Event()

    def _decide_dwdp(self) -> None:
        """Under a DWDP manager, store only this rank's range of its expert partition and register
        the layer; DWDP widens the range to all experts once it is set up."""
        from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager

        manager = get_global_dwdp_manager()
        if manager is None:
            return
        start, end = manager.expert_partition(self.backend.num_experts).range_of(manager.dwdp_rank)
        self.backend.set_expert_range(start, end)
        self.dwdp_manager = manager
        self.enable_dwdp = True
        manager.add_layer(layer_idx=self.layer_idx)

    @property
    def consumes_sort_tables(self) -> bool:
        """Whether the kernels can run on tables a routing kernel wrote (no slot remap)."""
        return self.backend.num_slots == self.backend.num_experts

    # ------------------------------------------------------------------
    # Input quantization.
    # ------------------------------------------------------------------
    def quantize_input(self, x: Union[torch.Tensor, Fp4QuantizedTensor]):
        """``(x, x_sf)``: packed E2M1 ``[M, H / 2]`` and linear E4M3 block scales ``[M, H / 16]``
        of ``x`` (BF16, quantized against ``fc31_input_scale``; or an unswizzled
        ``Fp4QuantizedTensor`` produced upstream with the same scale)."""
        rows = x.shape[0]
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, "the routed experts take linear (unswizzled) block scales"
            x, x_sf = x.fp4_tensor, x.scaling_factor
        else:
            x, x_sf = fp4_quantize(x, self.backend.fc31_input_scale, SF_VEC_SIZE, False, False)
        return x, x_sf.view(rows, -(-self.hidden_size // SF_VEC_SIZE))

    def _aux(self) -> torch.cuda.Stream:
        if self._aux_stream is None:
            self._aux_stream = torch.cuda.Stream()
        return self._aux_stream

    def _quantize_on_aux(self, x: torch.Tensor, input_ready_event: Optional[torch.cuda.Event]):
        """Quantize ``x`` on the aux stream (it overlaps the routing); the returned event marks
        the result ready."""
        main_stream = torch.cuda.current_stream()
        aux = self._aux()
        self._main_event.record()
        x.record_stream(aux)
        with torch.cuda.stream(aux):
            self._main_event.wait()
            _wait(input_ready_event)
            x_q, x_sf = self.quantize_input(x)
            self._quant_event.record()
        x_q.record_stream(main_stream)
        x_sf.record_stream(main_stream)
        return x_q, x_sf, self._quant_event

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        route: Route,
        input_ready_event: Optional[torch.cuda.Event] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """This rank's routed output ``[M, hidden]`` (BF16) of ``x`` (``[M, hidden]`` BF16 or
        NVFP4), routed by ``route`` on ``router_logits``.

        ``input_ready_event``: ``x`` is written on another stream, joined before its first read.
        """
        if self.enable_dwdp:
            self.dwdp_manager.wait_and_bind(self.backend, self.layer_idx)
        out = self._forward_chunked(x, router_logits, route, input_ready_event, all_rank_num_tokens)
        if self.enable_dwdp:
            self.dwdp_manager.finish_layer(self.layer_idx)
        return out

    def _prologue(self, x, router_logits, route, input_ready_event):
        """Route and quantize: ``(x, x_sf, routing)``; a BF16 ``x`` is joined before the routing,
        an NVFP4 one after it."""
        if isinstance(x, torch.Tensor):
            _wait(input_ready_event)
            input_ready_event = None
        routing = route(
            router_logits, self.backend.slot_start, self.backend.expert_size_per_partition
        )
        _wait(input_ready_event)
        x_q, x_sf = self.quantize_input(x)
        return x_q, x_sf, routing

    def _forward_chunked(self, x, router_logits, route, input_ready_event, all_rank_num_tokens):
        num_tokens = router_logits.shape[0]
        rows = max(all_rank_num_tokens) if self.enable_dwdp and all_rank_num_tokens else num_tokens
        num_chunks = max(1, -(-rows // self.max_chunk_tokens))
        if num_chunks == 1:
            return self._forward_chunk(x, router_logits, route, input_ready_event)
        _wait(input_ready_event)
        base, extra = divmod(num_tokens, num_chunks)
        sizes = [base + 1] * extra + [base] * (num_chunks - extra)
        if isinstance(x, Fp4QuantizedTensor):
            x_chunks = [
                Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)
                for codes, scales in zip(
                    x.fp4_tensor.split(sizes), x.scaling_factor.view(num_tokens, -1).split(sizes)
                )
            ]
        else:
            x_chunks = x.split(sizes)
        outs = [
            self._forward_chunk(xc, lc, route, None)
            for xc, lc in zip(x_chunks, router_logits.split(sizes))
        ]
        return torch.cat(outs)

    def _forward_chunk(self, x, router_logits, route, input_ready_event):
        x_ready = None
        if (
            self.kernels == "CUTEDSL"
            and self.async_input_quant
            and isinstance(x, torch.Tensor)
            and x.shape[0] <= _ASYNC_INPUT_QUANT_MAX_TOKENS
            and (do_multi_stream() or torch.cuda.is_current_stream_capturing())
        ):
            x_q, x_sf, x_ready = self._quantize_on_aux(x, input_ready_event)
            routing = route(
                router_logits, self.backend.slot_start, self.backend.expert_size_per_partition
            )
        else:
            x_q, x_sf, routing = self._prologue(x, router_logits, route, input_ready_event)
        return self._run_tuned(
            x_q,
            x_sf,
            routing.token_selected_experts,
            routing.token_final_scales,
            routing.sort_tables,
            x_ready,
        )

    def _run_tuned(self, x, x_sf, ids, scales, tables: Optional[SortTables], x_ready):
        """The AutoTuner's routing tile for this shape, then the kernel chain at it (on
        ``tables`` where they are at that tile)."""
        b = self.backend
        moe_output = torch.empty(
            (scales.size(0), self.hidden_size), dtype=torch.bfloat16, device=x.device
        )
        if ids.size(0) == 0:
            _wait(x_ready)
            return moe_output
        runner_cls, key = _tile_runners()[self.kernels]
        extra = {}
        if self.kernels == "CUTEDSL_FC12":
            extra["workload_identity"] = (
                int(ActivationType.SiTu),
                self.situ_beta,
                self.situ_linear_beta,
            )
        runner = runner_cls(
            forward_impl=self._run,
            num_experts=b.num_slots,
            top_k=ids.size(-1),
            num_local_experts=b.expert_size_per_partition,
            local_expert_offset=b.slot_start,
            enable_finalize_fusion=True,
            enable_alltoall=False,
            **extra,
        )
        self._pending_x_ready = x_ready
        inputs = [x, ids, scales, x_sf, moe_output, b]
        _, tactic = AutoTuner.get().choose_one(key, [runner], runner.get_tuning_config(), inputs)
        if tables is not None and runner.tile_size_of(tactic) == tables.tile_size:
            return self._run(*inputs, tile_size=tables.tile_size, sort_tables=tables)
        return runner(inputs, tactic=tactic)

    def _join_x_ready(self) -> None:
        event, self._pending_x_ready = getattr(self, "_pending_x_ready", None), None
        _wait(event)

    def _run(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        scales: torch.Tensor,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weights: K3Nvfp4Experts,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        sort_tables: Optional[SortTables] = None,
        **unused,
    ) -> torch.Tensor:
        """The kernel chain at ``tile_size`` (on ``sort_tables`` if given, else ``moe_sort``)."""
        del enable_alltoall, unused
        top_k = ids.size(1)
        offset, num_local = weights.slot_start, weights.expert_size_per_partition
        if sort_tables is not None:
            assert sort_tables.tile_size == tile_size
            outs = sort_tables.outputs
        else:
            outs = torch.ops.trtllm.moe_sort(
                token_selected_experts=ids,
                token_final_scales=scales,
                num_experts=weights.num_slots,
                top_k=top_k,
                local_expert_offset=offset,
                local_num_experts=num_local,
                tile_tokens_dim=tile_size,
            )
        tile_to_expert, tile_mn_limit, expanded_to_permuted, permuted_to_expanded, _, num_tiles = (
            outs
        )
        fc1_weight = weights.w3_w1_weight.view(torch.float4_e2m1fn_x2)
        fc1_scale = weights.w3_w1_weight_scale.view(torch.uint8)
        fc2_weight = weights.w2_weight.view(torch.float4_e2m1fn_x2)
        fc2_scale = weights.w2_weight_scale.view(torch.uint8)
        if self.kernels == "CUTEDSL_FC12":
            self._join_x_ready()
            # TODO(fam_moe): die-major FC12 layout/schedule (084745cde4 ran it via
            # TRTLLM_K3_FC12_DIE_LAYOUT/SCHED in main's fused_fc12.py) -- pass
            # die_major=self.fc12_die_major once the family FC12 module lands.
            torch.ops.trtllm.cute_dsl_nvfp4_fc12_fused_rubin(
                input=x.view(torch.float4_e2m1fn_x2),
                fc1_weight=fc1_weight,
                input_scale=x_sf.view(torch.uint8),
                fc1_weight_scale=fc1_scale,
                fc1_alpha=weights.fc31_alpha,
                tile_idx_to_group_idx=tile_to_expert,
                tile_idx_to_mn_limit=tile_mn_limit,
                permuted_idx_to_expanded_idx=permuted_to_expanded,
                num_non_exiting_tiles=num_tiles,
                global_sf=weights.fc2_input_scale,
                fc2_weight=fc2_weight,
                fc2_weight_scale=fc2_scale,
                fc2_alpha=weights.fc2_alpha,
                output=moe_output,
                token_final_scales=scales,
                expanded_idx_to_permuted_idx=expanded_to_permuted,
                num_experts=weights.num_slots,
                top_k=top_k,
                num_local_experts=num_local,
                local_expert_offset=offset,
                tile_size=tile_size,
                swiglu_limit=float("inf"),
                ep_size=self.moe_ep_size,
                enable_alltoall=False,
                scaling_vector_size=SF_VEC_SIZE,
                activation_type=int(ActivationType.SiTu),
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                reserved_sms=self.sm_reserve,
            )
            return moe_output

        rubin = get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE
        memset_stream = self._aux()
        self._main_event.record()
        moe_output.record_stream(memset_stream)
        with torch.cuda.stream(memset_stream):
            self._main_event.wait()
            torch.ops.trtllm.moe_output_memset_inplace(
                input=moe_output,
                tile_idx_to_mn_limit=tile_mn_limit,
                expanded_idx_to_permuted_idx=expanded_to_permuted,
                permuted_idx_to_expanded_idx=permuted_to_expanded,
                num_non_exiting_tiles=num_tiles,
                tile_tokens_dim=tile_size,
                top_k=top_k,
                ep_size=self.moe_ep_size,
                enable_alltoall=False,
            )
            self._memset_event.record()
        fc1 = (
            torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin
            if rubin
            else torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell
        )
        fc1_kwargs = {"output_tensor": None, "output_sf_tensor": None} if rubin else {}
        self._join_x_ready()
        h, h_sf = fc1(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=fc1_weight,
            input_scale=x_sf.view(torch.uint8),
            weight_scale=fc1_scale,
            alpha=weights.fc31_alpha,
            tile_idx_to_group_idx=tile_to_expert,
            tile_idx_to_mn_limit=tile_mn_limit,
            permuted_idx_to_expanded_idx=permuted_to_expanded,
            num_non_exiting_tiles=num_tiles,
            global_sf=weights.fc2_input_scale,
            num_experts=weights.num_slots,
            top_k=top_k,
            num_local_experts=num_local,
            local_expert_offset=offset,
            tile_size=tile_size,
            **fc1_kwargs,
            activation_type=ActivationType.SiTu,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )
        self._memset_event.wait()
        fc2 = (
            torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin
            if rubin
            else torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell
        )
        fc2_kwargs = {} if rubin else {"expert_counts": None, "expert_capacity": 0}
        fc2(
            input=h.view(torch.float4_e2m1fn_x2),
            weight=fc2_weight,
            input_scale=h_sf.view(torch.uint8),
            weight_scale=fc2_scale,
            alpha=weights.fc2_alpha,
            output=moe_output,
            tile_idx_to_group_idx=tile_to_expert,
            tile_idx_to_mn_limit=tile_mn_limit,
            permuted_idx_to_expanded_idx=permuted_to_expanded,
            num_non_exiting_tiles=num_tiles,
            token_final_scales=scales,
            num_experts=weights.num_slots,
            top_k=top_k,
            num_local_experts=num_local,
            local_expert_offset=offset,
            tile_size=tile_size,
            output_dtype=torch.bfloat16,
            **fc2_kwargs,
        )
        return moe_output


__all__ = [
    "K3RoutedExperts",
    "KERNEL_SETS",
]
