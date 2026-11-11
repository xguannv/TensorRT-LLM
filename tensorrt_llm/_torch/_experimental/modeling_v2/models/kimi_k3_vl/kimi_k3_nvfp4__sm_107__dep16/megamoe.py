# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts of one MoE layer for the dep16 target (EP16, attention DP), as one
MegaMoE CuTe DSL launch per forward.

The MegaMoE kernel (``run_cute_dsl_megamoe_nvfp4_blackwell``, which dispatches to the Rubin or
Blackwell kernel of the GPU) does the whole routed MoE across the EP group in one launch: it
sends every token to the ranks of its top-k experts over NVLink, runs FC1 + SiTU + FP4 requant +
FC2 there, sends the rows back and reduces the top-k on the token's own rank (the deterministic
standalone TopkReduce: no in-kernel FC2 reduce). The output on each rank is the complete routed
output of that rank's tokens.

:class:`K3MegaMoE` runs one forward, in this order::

    join input_ready_event                         [before the routing]
    strip the attention-DP padding                 x[:all_rank_num_tokens[ep_rank]]
    routing (the caller's ``routing``, else its ``route`` on the router logits)
    fp4_quantize                                   NVFP4 x + linear E4M3 block scales
    bucket = smallest ladder rung >= max(all_rank_num_tokens)
    [prime the other buckets]                      once per process, AutoTuner tuning mode only
    stage x, x_sf, top-k into the bucket's buffers  unless the bucket takes them directly
    MegaMoE kernel

The weights (:class:`K3MegaMoeNvfp4Experts`, the ``backend`` attribute) are kept only in the
kernel's layout; the checkpoint streams in expert by expert (``prepare_streaming_load`` /
``load_expert`` / ``finalize``, the API of the family's ``K3Nvfp4Experts``).

Launch buffers. Each ladder bucket ``max_T`` (one compiled kernel per bucket) has a symmetric
memory provider holding ``activation [max_T, H / 2]``, ``activation_sf``, ``topk_weights``, the
combine output ``[max_T, 1, H]`` and the kernel's shared workspace, plus a local ``topk_idx``
staging buffer of this module (``-1`` outside the live rows). Providers are shared by every layer
of the same layout (the custom op's cache), so the returned tensor is a view of the shared combine
region: it holds this layer's output until the next MegaMoE launch in the same bucket, i.e. the
next MoE layer. A bucket whose kernel is the Rubin generation-phase kernel reads ``x``, ``x_sf``
and the top-k in place when the rank's token count equals the bucket.

Every EP rank must launch every forward (also with zero tokens) and pick the same bucket: the
kernel synchronizes the ranks with an in-kernel NVLink barrier. The first launch of each
(local workspace, shared workspace) pairing runs a host-side barrier-reset fence (stream sync and
two WORLD barriers), which CUDA graph capture cannot contain; the priming of the ladder in the
AutoTuner's tuning pass (the warmup) runs that launch for every bucket ahead of capture.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe.k3r_experts import (
    Route,
    SortedRouting,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    megamoe_activation_sf_bytes_per_row,
)
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: Values per NVFP4 block scale.
SF_VEC_SIZE = 16
#: Wire format of the combine (the returned rows travel as BF16).
COMBINE_FORMAT = "bf16"
#: Largest top-k the kernel is run at.
MAX_TOP_K = 16
#: FC1 rows are stored as alternating groups of this many gate rows and up rows.
FC1_GATE_UP_INTERLEAVE = 16
#: Rungs of the per-rank token bucket ladder, by architecture (each rung is one compiled kernel).
BUCKET_RUNGS_RUBIN = (32, 64, 128, 192, 256, 384, 512, 1024, 4096)
BUCKET_RUNGS_BLACKWELL = (256, 1024, 4096)

# Ladders whose buckets have been primed in this process, by layout.
_PRIMED_LADDERS: set = set()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def fc1_sf_elements(hidden_size: int, intermediate_size: int) -> int:
    """E4M3 block scales of one expert's FC1 in the kernel's blocked layout (rows padded to 128,
    scale columns to 4)."""
    return _align_up(2 * intermediate_size, 128) * _align_up(hidden_size // SF_VEC_SIZE, 4)


def fc2_sf_elements(hidden_size: int, intermediate_size: int) -> int:
    """E4M3 block scales of one expert's FC2 in the kernel's blocked layout."""
    return _align_up(hidden_size, 128) * _align_up(intermediate_size // SF_VEC_SIZE, 4)


def resolve_buckets(max_num_tokens: int) -> List[int]:
    """The per-rank token bucket ladder: the architecture's rungs below ``max_num_tokens``, then
    ``max_num_tokens`` itself. A pure function of rank-identical inputs, so every EP rank has the
    same ladder."""
    full = int(max_num_tokens)
    if full <= 0:
        full = 4096
    rungs = BUCKET_RUNGS_RUBIN if get_sm_version() == 107 else BUCKET_RUNGS_BLACKWELL
    buckets = {b for b in rungs if b < full}
    buckets.add(full)
    return sorted(buckets)


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """One expert's FC1 rows in the kernel's order: ``[gate[0:16], up[0:16], gate[16:32], ...]``
    (``gate`` = w1, ``up`` = w3, ``[I, C]`` each; weights and their block scales alike)."""
    rows, cols = gate.shape
    pairs = rows // FC1_GATE_UP_INTERLEAVE
    return torch.stack(
        [
            gate.reshape(pairs, FC1_GATE_UP_INTERLEAVE, cols),
            up.reshape(pairs, FC1_GATE_UP_INTERLEAVE, cols),
        ],
        dim=1,
    ).reshape(2 * rows, cols)


def blocked_block_scales(sf: torch.Tensor, flat_size: int) -> torch.Tensor:
    """One expert's linear E4M3 block scales (``uint8 [rows, cols]``) in the kernel's blocked
    layout, flattened: zero-padded to 128 rows and 4 columns, then each 128x4 block in 32x4x4 atom
    order, zero-padded at the end to ``flat_size``."""
    rows, cols = sf.shape
    padded = F.pad(sf, (0, -cols % 4, 0, -rows % 128))
    blocked = (
        padded.view((rows + 127) // 128, 4, 32, (cols + 3) // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(-1)
    )
    if blocked.numel() == flat_size:
        return blocked
    out = torch.zeros(flat_size, dtype=blocked.dtype, device=blocked.device)
    out[: blocked.numel()] = blocked
    return out


def _as_nvfp4(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float4_e2m1fn_x2 else t.view(torch.float4_e2m1fn_x2)


def _as_fp8_sf(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)


def _leading_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    return t if t.shape[0] == rows else t[:rows]


def _resolve_ep_process_group(mapping: Mapping, ep_size: int):
    """The EP process group: the mapping's (Ray / DeviceMesh), else WORLD where it is the EP
    group."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Kimi K3 MegaMoE needs torch.distributed initialized before the layer is built"
        )
    try:
        return mapping.moe_ep_group_pg
    except (NotImplementedError, AttributeError):
        pass
    if dist.get_world_size() == ep_size:
        return dist.group.WORLD
    raise RuntimeError(
        f"Kimi K3 MegaMoE: the mapping has no EP process group and EP ({ep_size}) is not the "
        f"whole world ({dist.get_world_size()})"
    )


class K3MegaMoeNvfp4Experts(nn.Module):
    """One MoE layer's NVFP4 routed experts on this rank, in the MegaMoE kernel's layout.

    Holds experts ``[slot_start, slot_end)`` (``num_experts / ep_size`` contiguous experts;
    ``E`` below), ``H`` = hidden size, ``I`` = intermediate size:

    * ``mega_fc1_weight`` ``uint8 [E, 2 I, H / 2]``: packed E2M1 FC1 rows, gate (w1) and up (w3)
      interleaved in groups of 16 (:func:`interleave_gate_up`); ``mega_fc1_weight_sf``
      ``uint8 [E, align128(2 I) * align4(H / 16)]``: their E4M3 block scales, same row order,
      blocked (:func:`blocked_block_scales`).
    * ``mega_fc2_weight`` ``uint8 [E, H, I / 2]`` (the checkpoint bytes) and ``mega_fc2_weight_sf``
      ``uint8 [E, align128(H) * align4(I / 16)]`` (blocked).
    * ``fc31_input_scale``: the reciprocal of the largest checkpoint ``w1.input_scale`` of the
      rank's experts, the global scale ``x`` is quantized with; ``fc2_input_scale`` likewise for
      ``w2`` (not read by the kernel).
    * ``fc31_alpha`` ``[E]``: ``w1.weight_scale_2 / fc31_input_scale``; ``fc2_alpha`` ``[E]``:
      ``w2.weight_scale_2 * w2.input_scale``; ``fc1_norm_const`` ``[E]``: ``1 / w2.input_scale``,
      the global scale of the kernel's FP4 requantization of each expert's FC1 output.

    The checkpoint streams in expert by expert (:meth:`load_expert`, from any thread once
    :meth:`prepare_streaming_load` ran); :meth:`finalize` runs once the rank's last expert is in.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        ep_size: int,
        ep_rank: int,
    ) -> None:
        super().__init__()
        if num_experts % ep_size != 0:
            raise ValueError(f"{num_experts} experts do not split evenly over {ep_size} EP ranks")
        if hidden_size <= 0 or hidden_size % 64 != 0:
            raise ValueError(f"hidden_size {hidden_size} must be a positive multiple of 64")
        if intermediate_size <= 0 or intermediate_size % 64 != 0:
            raise ValueError(
                f"intermediate_size {intermediate_size} must be a positive multiple of 64"
            )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size
        self.expand_intermediate_size_per_partition = 2 * intermediate_size
        self.ep_size, self.ep_rank = ep_size, ep_rank
        self.scaling_vector_size = SF_VEC_SIZE
        per_rank = num_experts // ep_size
        self.num_slots = num_experts
        self.expert_size_per_partition = per_rank
        self.slot_start = ep_rank * per_rank
        self.slot_end = self.slot_start + per_rank
        self.initial_local_expert_ids = list(range(self.slot_start, self.slot_end))
        self.initial_global_assignments = list(range(num_experts))
        self._staged: Optional[Dict] = None
        self._streamed_expert_slots: set = set()
        self._weights_created = False

    def create_weights(self) -> None:
        """Create the parameters (once; on the current default device)."""
        if self._weights_created:
            return
        e = self.expert_size_per_partition
        h, i = self.hidden_size, self.intermediate_size_per_partition

        def param(tensor: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(tensor, requires_grad=False)

        self.mega_fc1_weight = param(torch.empty((e, 2 * i, h // 2), dtype=torch.uint8))
        self.mega_fc1_weight_sf = param(torch.empty((e, fc1_sf_elements(h, i)), dtype=torch.uint8))
        self.mega_fc2_weight = param(torch.empty((e, h, i // 2), dtype=torch.uint8))
        self.mega_fc2_weight_sf = param(torch.empty((e, fc2_sf_elements(h, i)), dtype=torch.uint8))
        self.fc31_input_scale = param(torch.tensor(1.0, dtype=torch.float32))
        self.fc2_input_scale = param(torch.tensor(1.0, dtype=torch.float32))
        self.fc31_alpha = param(torch.ones(e, dtype=torch.float32))
        self.fc2_alpha = param(torch.ones(e, dtype=torch.float32))
        self.fc1_norm_const = param(torch.ones(e, dtype=torch.float32))
        self._weights_created = True

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
        global scales) into ``local_slot_id``, directly in the kernel's layout; the global scales
        are kept for :meth:`finalize`."""
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
        h, i = self.hidden_size, self.intermediate_size_per_partition
        device = self.mega_fc1_weight.device

        def take(t: torch.Tensor, name: str, shape: Tuple[int, int], weight: bool) -> torch.Tensor:
            if weight and t.dtype != torch.uint8:
                raise TypeError(f"{name} must hold packed NVFP4 uint8, got {t.dtype}")
            if not weight and t.element_size() != 1:
                raise TypeError(f"{name} must hold one-byte E4M3 block scales, got {t.dtype}")
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {shape}")
            return t.to(device).contiguous().view(torch.uint8)

        w1 = take(w1_weight, "w1_weight", (i, h // 2), True)
        w3 = take(w3_weight, "w3_weight", (i, h // 2), True)
        w2 = take(w2_weight, "w2_weight", (h, i // 2), True)
        w1_sf = take(w1_weight_scale, "w1_weight_scale", (i, h // SF_VEC_SIZE), False)
        w3_sf = take(w3_weight_scale, "w3_weight_scale", (i, h // SF_VEC_SIZE), False)
        w2_sf = take(w2_weight_scale, "w2_weight_scale", (h, i // SF_VEC_SIZE), False)

        self.mega_fc1_weight.data[local_slot_id].copy_(interleave_gate_up(w1, w3))
        self.mega_fc2_weight.data[local_slot_id].copy_(w2)
        self.mega_fc1_weight_sf.data[local_slot_id].copy_(
            blocked_block_scales(interleave_gate_up(w1_sf, w3_sf), self.mega_fc1_weight_sf.shape[1])
        )
        self.mega_fc2_weight_sf.data[local_slot_id].copy_(
            blocked_block_scales(w2_sf, self.mega_fc2_weight_sf.shape[1])
        )

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
        """After the last expert of the layer: the global input scales, the per-expert alphas and
        ``fc1_norm_const``."""
        staged, self._staged = self._staged, None
        if staged is None:
            return
        missing = sorted(set(range(self.expert_size_per_partition)) - self._streamed_expert_slots)
        if missing:
            raise RuntimeError(
                f"Kimi K3 MegaMoE finalize with local slots {missing[:10]} never loaded"
            )
        input_scales = staged["input_scales"]

        norm_const = self.fc1_norm_const.data
        norm_const.copy_(
            torch.stack(
                [
                    input_scales[e]["w2"]
                    .to(device=norm_const.device, dtype=torch.float32)
                    .reciprocal()
                    for e in self.initial_local_expert_ids
                ]
            )
        )

        fc31_values, fc2_values = [], []
        for expert_id, entry in input_scales.items():
            assert torch.allclose(entry["w1"], entry["w3"]), (
                f"Expert {expert_id}: w1_input_scale != w3_input_scale"
            )
            fc31_values.append(entry["w1"])
            fc2_values.append(entry["w2"])
        self.fc31_input_scale.data.copy_(torch.stack(fc31_values).max().reciprocal())
        self.fc2_input_scale.data.copy_(torch.stack(fc2_values).max().reciprocal())

        fc31_input_scale = self.fc31_input_scale.data
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
            expert_id = self.initial_local_expert_ids[slot]
            fc2_input_scale = (
                input_scales[expert_id]["w2"][...].reshape([]).to(dtype=torch.float32).reciprocal()
            )
            w2_ws2 = 1.0 / scales["w2"][...].reshape([])
            self.fc2_alpha.data[slot].copy_(1.0 / (fc2_input_scale * w2_ws2))


class _LaunchBuffers(NamedTuple):
    """One bucket's kernel buffers: the symmetric regions (peer-mapped through ``peer_offsets``)
    and this module's local ``topk_idx`` staging."""

    activation: torch.Tensor  # uint8 [max_T, H / 2]
    activation_sf: torch.Tensor  # uint8 [max_T, sf bytes per row]
    topk_weights: torch.Tensor  # fp32 [max_T, top_k]
    combine_output: torch.Tensor  # bf16 [max_T, 1, H]
    shared_workspace: torch.Tensor  # uint8
    peer_offsets: Tuple[int, ...]
    topk_idx: torch.Tensor  # [max_T, top_k], int32 on Rubin, int64 on Blackwell


class K3MegaMoE(nn.Module):
    """One MoE layer's routed experts on this rank as the MegaMoE kernel (see the module
    docstring).

    ``mapping``: the routed experts' mapping (attention DP over the TP group, ``moe_ep_size`` =
    ``tp_size`` > 1, ``moe_tp_size`` 1). ``max_num_tokens``: tokens per rank per forward, the top
    of the bucket ladder (``min(moe_max_num_tokens, max_num_tokens)`` of the engine).
    ``skip_create_weights_in_init``: leave :meth:`create_weights` (the symmetric-memory
    rendezvous of every bucket, collective over the EP group, then the parameters) to the caller.

    ``torch.distributed`` is initialized here under MPI where it is not yet (a NCCL WORLD group
    over the MPI world, which must be the EP group).
    """

    #: The output is the complete routed output of this rank's tokens (no reduction left).
    returns_global_routed_output = True

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        layer_idx: int,
        situ_beta: float,
        situ_linear_beta: float,
        max_num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
        skip_create_weights_in_init: bool = False,
    ) -> None:
        super().__init__()
        sm = get_sm_version()
        if sm not in (100, 103, 107):
            raise RuntimeError(f"the MegaMoE kernel runs on SM100, SM103 or SM107, not SM{sm}")
        if mapping.moe_tp_size != 1 or mapping.moe_cluster_size != 1:
            raise NotImplementedError("Kimi K3 MegaMoE splits the experts by EP only")
        ep_size = mapping.moe_ep_size
        if ep_size <= 1:
            raise NotImplementedError("Kimi K3 MegaMoE runs over an EP group of several ranks")
        if not mapping.enable_attention_dp or mapping.tp_size != ep_size:
            raise NotImplementedError(
                "Kimi K3 MegaMoE takes the tokens of attention-DP ranks, one rank per EP rank"
            )
        if not 0 < top_k <= MAX_TOP_K:
            raise ValueError(f"top_k {top_k} is outside [1, {MAX_TOP_K}]")
        if situ_beta <= 0 or situ_linear_beta <= 0:
            raise ValueError("the SiTU soft-caps must be positive")
        if max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
        if dtype != torch.bfloat16:
            raise ValueError(f"Kimi K3 MegaMoE takes BF16 activations, not {dtype}")
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size, self.ep_rank = ep_size, mapping.moe_ep_rank
        self.situ_beta, self.situ_linear_beta = float(situ_beta), float(situ_linear_beta)
        self.dtype = dtype
        self.max_num_tokens = int(max_num_tokens)
        self.all_reduce = None
        self.backend = K3MegaMoeNvfp4Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            ep_rank=self.ep_rank,
        )
        self._buckets = resolve_buckets(self.max_num_tokens)
        self._topk_idx_dtype = torch.int32 if sm == 107 else torch.int64
        self._sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden_size)
        # bucket -> MegaMoeSymmMemProvider (filled by create_weights).
        self._symm_providers: Dict[int, object] = {}
        # Buckets whose kernel reads x, x_sf and the top-k in place at an exact token count.
        self._direct_input_buckets: set = set()
        # (bucket, inference mode) -> buffers; views made under inference mode cannot be updated
        # in place outside it.
        self._launch_buffers: Dict[Tuple[int, bool], _LaunchBuffers] = {}
        # bucket -> topk_idx staging, and the live rows its last staging wrote.
        self._topk_idx_staging: Dict[int, torch.Tensor] = {}
        self._last_staged_rows: Dict[int, int] = {}
        # Cute views of this module's weights and topk_idx staging, reused across launches.
        self._cute_weight_cache: Dict = {}
        self._weights_created = False

        from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

        ensure_cutedsl_torch_distributed(ep_size)
        self._ep_pg = _resolve_ep_process_group(mapping, ep_size)
        # The kernel's barrier-reset fence runs on WORLD.
        if dist.get_world_size() != ep_size:
            raise ValueError(
                f"Kimi K3 MegaMoE needs torch.distributed WORLD == EP; got "
                f"WORLD={dist.get_world_size()}, EP={ep_size}"
            )
        if not skip_create_weights_in_init:
            self.create_weights()

    @property
    def buckets(self) -> List[int]:
        """The per-rank token bucket ladder."""
        return list(self._buckets)

    # ------------------------------------------------------------------
    # Construction-time collectives and parameters.
    # ------------------------------------------------------------------
    def create_weights(self) -> None:
        """Rendezvous the symmetric memory of every bucket (collective over the EP group: every
        rank calls this in the same order), then create the parameters. Once per module."""
        if self._weights_created:
            return
        self._alloc_symm_providers()
        self.backend.create_weights()
        self._weights_created = True

    def _alloc_symm_providers(self) -> None:
        from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

        if not op.IS_MEGAMOE_OP_AVAILABLE:
            raise RuntimeError(
                f"the MegaMoE CuTe DSL op is unavailable: {op.MEGAMOE_OP_UNAVAILABLE_REASON}"
            )
        b = self.backend
        for max_t in self._buckets:
            shared_workspace_bytes, direct_inputs = op.query_megamoe_kernel_properties(
                world_size=self.ep_size,
                num_topk=self.top_k,
                num_experts_per_rank=b.expert_size_per_partition,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=self.intermediate_size,
                expand_intermediate_size_per_partition=2 * self.intermediate_size,
                max_tokens_per_rank=max_t,
                apply_topk_in_fc1=False,
                gate_up_clamp=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                in_kernel_fc2_reduce=False,
                combine_format=COMBINE_FORMAT,
            )
            if direct_inputs:
                self._direct_input_buckets.add(max_t)
            self._symm_providers[max_t] = op.get_megamoe_symm_provider(
                process_group=self._ep_pg,
                world_size=self.ep_size,
                rank=self.ep_rank,
                hidden_size=self.hidden_size,
                max_tokens_per_rank=max_t,
                num_topk=self.top_k,
                output_dtype=self.dtype,
                combine_format=COMBINE_FORMAT,
                shared_workspace_bytes=shared_workspace_bytes,
            )
        if len(self._buckets) > 1 and self.ep_rank == 0:
            logger.info(
                "[K3MegaMoE] max_tokens_per_rank buckets=%s (one symmetric provider + kernel per "
                "bucket; per-launch bucket = smallest >= max(all_rank_num_tokens))",
                self._buckets,
            )

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        *,
        route: Optional[Route] = None,
        routing: Optional[SortedRouting] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        input_ready_event: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        """The routed output ``[n, hidden]`` (``x.dtype``) of this rank's ``n`` tokens of ``x``
        (BF16 ``[M, hidden]``, ``M >= n``; rows past ``n`` are attention-DP padding).

        ``all_rank_num_tokens``: every EP rank's token count; ``n`` is this rank's entry and the
        largest entry picks the bucket. Without it (or with one entry) ``n`` is ``M`` (that entry)
        and the launch takes the top bucket.

        The top-k (``int32 [n, top_k]`` global expert ids, ``fp32`` weights): ``routing``'s
        leading ``n`` rows, else ``route(router_logits[:n], slot_start, local experts)``.

        ``input_ready_event``: ``x`` is written on another stream; joined first, before the
        routing.

        The result is a view of the bucket's shared combine region (see the module docstring).
        """
        if input_ready_event is not None:
            input_ready_event.wait()
        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError("Kimi K3 MegaMoE quantizes a BF16 input itself")
        if not self._weights_created:
            raise RuntimeError("create_weights() must run before the forward")
        output_dtype = x.dtype

        if all_rank_num_tokens is not None and len(all_rank_num_tokens) > 1:
            counts = [int(v) for v in all_rank_num_tokens]
            if len(counts) != self.ep_size:
                raise ValueError(
                    f"all_rank_num_tokens has {len(counts)} entries for EP size {self.ep_size}"
                )
            num_tokens = counts[self.ep_rank]
            bucket_hint: Optional[int] = max(counts)
        else:
            num_tokens = int(all_rank_num_tokens[0]) if all_rank_num_tokens else int(x.shape[0])
            bucket_hint = None
        if num_tokens > x.shape[0]:
            raise ValueError(f"{num_tokens} tokens on this rank but x has {x.shape[0]} rows")
        x = _leading_rows(x, num_tokens)

        if num_tokens > 0:
            if routing is None:
                if route is None or router_logits is None:
                    raise ValueError("the forward needs a routing, or a route and router logits")
                routing = route(
                    _leading_rows(router_logits, num_tokens),
                    self.backend.slot_start,
                    self.backend.expert_size_per_partition,
                )
            ids = _leading_rows(routing.token_selected_experts, num_tokens)
            scales = _leading_rows(routing.token_final_scales, num_tokens)
            if ids.dtype != torch.int32:
                ids = ids.to(torch.int32)
            if scales.dtype != torch.float32 or not scales.is_contiguous():
                scales = scales.to(torch.float32).contiguous()
        else:
            ids = torch.empty((0, self.top_k), dtype=torch.int32, device=x.device)
            scales = torch.empty((0, self.top_k), dtype=torch.float32, device=x.device)

        x_fp4, x_sf = self.quantize_input(x)
        return self._run(x_fp4, x_sf, ids, scales, num_tokens, bucket_hint, output_dtype)

    def quantize_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(x, x_sf)``: packed E2M1 ``uint8 [n, H / 2]`` and linear E4M3 block scales
        ``[n, H / 16]`` of ``x`` against ``fc31_input_scale``."""
        hidden = x.shape[1]
        x_bf16 = x.to(torch.bfloat16).contiguous()
        rows = x_bf16.shape[0]
        if rows == 0:
            return (
                torch.empty((0, hidden // 2), dtype=torch.uint8, device=x.device),
                torch.empty((0, self._sf_bytes_per_row), dtype=torch.uint8, device=x.device),
            )
        x_fp4, x_sf = fp4_quantize(x_bf16, self.backend.fc31_input_scale, SF_VEC_SIZE, False, False)
        return x_fp4, x_sf.view(rows, self._sf_bytes_per_row)

    def _select_bucket(self, hint: Optional[int]) -> int:
        """The smallest bucket that holds ``hint`` tokens (rank-identical), the top one without."""
        if hint is None or hint <= 0:
            return self._buckets[-1]
        for bucket in self._buckets:
            if hint <= bucket:
                return bucket
        raise RuntimeError(
            f"Kimi K3 MegaMoE: {hint} tokens per rank exceed the top bucket {self._buckets[-1]}"
        )

    def _run(
        self,
        x_fp4: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        bucket_hint: Optional[int],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pick the bucket, prime the ladder in tuning mode, stage (or hand over) the inputs and
        launch."""
        max_t = self._select_bucket(bucket_hint)
        if num_tokens > max_t:
            raise RuntimeError(
                f"Kimi K3 MegaMoE got {num_tokens} tokens but the bucket holds {max_t} "
                f"(buckets={self._buckets})"
            )
        device = x_fp4.device
        if AutoTuner.get().is_tuning_mode:
            self._prime_ladder(max_t, device, output_dtype)

        bufs = self._acquire_buffers(max_t, device)
        direct_inputs = (
            num_tokens == max_t
            and max_t in self._direct_input_buckets
            and x_fp4.shape == bufs.activation.shape
            and x_sf.shape == bufs.activation_sf.shape
            and x_fp4.is_contiguous()
            and x_sf.is_contiguous()
        )
        direct_routing = (
            direct_inputs
            and topk_idx.shape == (max_t, self.top_k)
            and topk_weights.shape == (max_t, self.top_k)
            and topk_idx.dtype == self._topk_idx_dtype
            and topk_weights.dtype == torch.float32
            and topk_idx.device == device
            and topk_weights.device == device
            and topk_idx.is_contiguous()
            and topk_weights.is_contiguous()
            and topk_idx.data_ptr() % 16 == 0
            and topk_weights.data_ptr() % 16 == 0
        )
        if not direct_routing:
            self._stage_inputs(
                bufs, x_fp4, x_sf, topk_idx, topk_weights, num_tokens, not direct_inputs
            )
        return self._launch(
            bufs,
            activation=x_fp4 if direct_inputs else bufs.activation,
            activation_sf=x_sf if direct_inputs else bufs.activation_sf,
            topk_idx=topk_idx if direct_routing else bufs.topk_idx,
            topk_weights=topk_weights if direct_routing else bufs.topk_weights[:, : self.top_k],
            num_tokens=num_tokens,
            max_t=max_t,
            output_dtype=output_dtype,
            staged_inputs=not direct_inputs,
        )

    def _acquire_buffers(self, max_t: int, device: torch.device) -> _LaunchBuffers:
        key = (max_t, torch.is_inference_mode_enabled())
        bufs = self._launch_buffers.get(key)
        if bufs is None:
            topk_idx = self._topk_idx_staging.get(max_t)
            if topk_idx is None:
                # -1: the kernel's dispatch skips the rows past the live tokens.
                topk_idx = torch.full(
                    (max_t, self.top_k), -1, dtype=self._topk_idx_dtype, device=device
                )
                self._topk_idx_staging[max_t] = topk_idx
            regions = self._symm_providers[max_t].get_regions()
            bufs = _LaunchBuffers(
                activation=regions.activation,
                activation_sf=regions.activation_sf,
                topk_weights=regions.topk_weights,
                combine_output=regions.combine_output,
                shared_workspace=regions.shared_workspace,
                peer_offsets=regions.peer_offsets,
                topk_idx=topk_idx,
            )
            self._launch_buffers[key] = bufs
        return bufs

    def _stage_inputs(
        self,
        bufs: _LaunchBuffers,
        x_fp4: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens: int,
        stage_activation: bool,
    ) -> None:
        """Copy the live rows into the bucket's buffers; ``topk_idx`` is ``-1`` and
        ``topk_weights`` 0 on every other row (the combine reads a weight for every
        ``(token, k)``)."""
        max_t = bufs.topk_idx.shape[0]
        last = self._last_staged_rows.get(max_t)
        if last is not None and last > num_tokens:
            bufs.topk_idx[num_tokens:last].fill_(-1)
        if num_tokens > 0:
            if stage_activation:
                bufs.activation[:num_tokens].copy_(x_fp4.view(torch.uint8), non_blocking=True)
                bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
            bufs.topk_weights[:num_tokens, : self.top_k].copy_(topk_weights, non_blocking=True)
        if num_tokens < max_t:
            bufs.topk_weights[num_tokens:max_t, : self.top_k].zero_()
        if num_tokens > 0:
            bufs.topk_idx[:num_tokens].copy_(topk_idx, non_blocking=True)
        self._last_staged_rows[max_t] = num_tokens

    def _launch(
        self,
        bufs: _LaunchBuffers,
        *,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        max_t: int,
        output_dtype: torch.dtype,
        staged_inputs: bool,
    ) -> torch.Tensor:
        """The kernel on ``max_t``'s buffers; ``staged_inputs``: the inputs are those buffers,
        whose cute views the bucket's provider keeps."""
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            run_cute_dsl_megamoe_nvfp4_blackwell,
            set_active_megamoe_cute_buffer_cache,
            set_active_megamoe_cute_weight_cache,
        )

        b = self.backend
        set_active_megamoe_cute_weight_cache(self._cute_weight_cache)
        provider = self._symm_providers.get(max_t) if staged_inputs else None
        set_active_megamoe_cute_buffer_cache(provider.cute_views if provider is not None else None)
        try:
            run_cute_dsl_megamoe_nvfp4_blackwell(
                activation=_as_nvfp4(activation),
                activation_sf=_as_fp8_sf(activation_sf),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                fc1_weight=_as_nvfp4(b.mega_fc1_weight),
                fc1_weight_sf=_as_fp8_sf(b.mega_fc1_weight_sf),
                fc2_weight=_as_nvfp4(b.mega_fc2_weight),
                fc2_weight_sf=_as_fp8_sf(b.mega_fc2_weight_sf),
                fc1_alpha=b.fc31_alpha,
                fc2_alpha=b.fc2_alpha,
                fc1_norm_const=b.fc1_norm_const,
                combine_output=bufs.combine_output,
                shared_workspace=bufs.shared_workspace,
                world_size=self.ep_size,
                local_rank=self.ep_rank,
                num_topk=self.top_k,
                num_experts_per_rank=b.expert_size_per_partition,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=self.intermediate_size,
                expand_intermediate_size_per_partition=2 * self.intermediate_size,
                max_tokens_per_rank=int(max_t),
                peer_offsets=bufs.peer_offsets,
                apply_topk_in_fc1=False,
                gate_up_clamp=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                act_alpha=self.situ_beta,
                act_beta=self.situ_linear_beta,
                in_kernel_fc2_reduce=False,
                combine_format=COMBINE_FORMAT,
                tactic_autotune=False,
                num_tokens=num_tokens,
                helper_expert_count=0,
                hot_expert_weight_ready_flags=None,
                hot_expert_weight_ready_generation=0,
            )
        finally:
            set_active_megamoe_cute_weight_cache(None)
            set_active_megamoe_cute_buffer_cache(None)
        if num_tokens == 0:
            return torch.empty(
                (0, self.hidden_size), dtype=output_dtype, device=bufs.combine_output.device
            )
        return bufs.combine_output[:num_tokens].squeeze(1).to(output_dtype)

    def _prime_ladder(self, live_t: int, device: torch.device, output_dtype: torch.dtype) -> None:
        """Launch every other bucket once with zero tokens (once per process and layout), so no
        bucket compiles or fences inside a serving iteration or a CUDA graph capture."""
        key = (
            tuple(self._buckets),
            self.ep_size,
            self.hidden_size,
            self.intermediate_size,
            self.backend.expert_size_per_partition,
            self.top_k,
            COMBINE_FORMAT,
            None,
            self.situ_beta,
            self.situ_linear_beta,
            False,
            str(device),
            output_dtype,
        )
        if key in _PRIMED_LADDERS:
            return
        primed = []
        for max_t in self._buckets:
            if max_t == live_t:
                continue
            bufs = self._acquire_buffers(max_t, device)
            self._stage_inputs(bufs, None, None, None, None, 0, False)
            self._launch(
                bufs,
                activation=bufs.activation,
                activation_sf=bufs.activation_sf,
                topk_idx=bufs.topk_idx,
                topk_weights=bufs.topk_weights[:, : self.top_k],
                num_tokens=0,
                max_t=max_t,
                output_dtype=output_dtype,
                staged_inputs=False,
            )
            primed.append(max_t)
        _PRIMED_LADDERS.add(key)
        if self.ep_rank == 0:
            logger.debug("[K3MegaMoE] bucket priming complete: count=%s", len(primed))


__all__ = [
    "BUCKET_RUNGS_BLACKWELL",
    "BUCKET_RUNGS_RUBIN",
    "K3MegaMoE",
    "K3MegaMoeNvfp4Experts",
    "blocked_block_scales",
    "fc1_sf_elements",
    "fc2_sf_elements",
    "interleave_gate_up",
    "resolve_buckets",
]
