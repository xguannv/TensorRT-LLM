# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107): the MoE block.

``K3MoERuntime`` runs one layer's latent MoE: the router (``K3MoEGate``), the latent
down projection, the routed experts (the CuTe DSL kernel chain of ``moe.py`` or the
MegaMoE module), the latent all-reduce, ``routed_expert_norm`` / ``routed_expert_up_proj``,
and the shared experts (``_K3SharedExpert``) on an auxiliary stream. Its decode tails leave the
residual update to the next layer's attention-residual mix (``attn_res.py``).

Also here: the routed-expert checkpoint specs (key families and the loader of each
quantization) and the workspace the routed latent all-reduce keeps across layers. The MoE
switches come from the target's ``K3Sm107Constants``, handed to ``K3MoERuntime``.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from ..ops.comm.k3_latent_ar_rmsnorm import k3_latent_ar_rmsnorm
from ..ops.gemm.k3_fp8w_lowm_gemm_push import k3_fp8w_lowm_gemm_push as k3_fp8w_lowm_gemm_push_op
from ..ops.gemm.k3_fp8w_mid_gemm import k3_fp8w_mid_gemm as k3_fp8w_mid_gemm_op
from ..ops.gemm.k3_latent_down_fp4 import k3_latent_down_fp4 as k3_latent_down_fp4_op
from ..ops.moe.k3_moe_residual_add import k3_moe_residual_add as k3_moe_residual_add_op
from ..ops.moe.k3_route_tail import k3_route_tail as k3_route_tail_op
from ..ops.moe.k3_router_logits import k3_router_logits as k3_router_logits_op
from .attn_res import _DeferredMoEResidual, _DeferredRpTail
from .constants import K3Sm107Constants
from .fp8_weight_read import _Fp8BlockScaleWeightReadLinear, _k3_lowm_gemm_can_run
from .layers import (
    Bf16Linear,
    GatedMlp,
    RMSNormWeight,
    TpAllReduce,
    finish_lamport_push,
    fork_join,
    gated_activation,
    lamport_push_target,
    subgroup_mapping,
)
from .layers import mnnvl_workspace_mpi_comm as _k3_mnnvl_mpi_comm
from .moe import KERNEL_SETS as _K3_ROUTED_KERNEL_SETS
from .moe import K3Nvfp4Experts, K3RoutedExperts, SortedRouting, SortTables
from .text_config import _is_kda_layer, _resolve_kimi_situ_betas

# Decode-batch routing tail of the CuteDSL MoE (see k3_route_tail): replaces noaux_tc_op ->
# moe_sort with one launch on a cluster of at most eight CTAs (same logits, same routing).
# Largest token count the routing tail serves: up to 256 tokens one cluster, above that (at most
# 2048 tokens, see k3_route_tail.wide_sort_fits) a grid of CTAs whose last CTA does the sort.
_K3_ROUTE_TAIL_MAX_M = 2048

_K3_LD_FUSED_MIN_M = 1
_K3_LD_FUSED_MAX_M = 4096

# For up to _K3_ROUTER_FIRST_MAX_M tokens, the router GEMM is issued (captured) on the main stream
# before latent_down is issued on the latent stream. Both depend only on the MoE input (the latent
# fork event is recorded before the router), so the graph has the same edges as the reverse
# order; only the node creation order differs, which decides which of the two the
# graph launches first once the MoE input's producer releases its dependents (PDL). At decode
# batch sizes latent_down's low-M GEMM is 112 CTAs x ~225 KB of shared memory, and with up to 16
# tokens the router GEMM (dsv3 router MMA kernel) is 112 CTAs whose registers fill most of an SM,
# so no SM holds one of each: launched second, the router gets only the SMs latent_down leaves
# and runs in several waves, and the routing (which gates the routed FC1 together with
# latent_down) ends late. Launched first, the router takes its SMs and latent_down starts on the
# rest. nsys node tracing turns latent_down's cross-stream programmatic edge into a full edge (it
# then waits for the MoE input's producer to complete), so such traces show the router first in
# either order.
_K3_ROUTER_FIRST_MAX_M = 64

# ``K3Sm107Constants.shared_sm`` (c240): SM-count target of the shared expert's cuBLAS GEMMs (0:
# off). FC12's persistent grid needs whole SMs, so shared-expert GEMMs still resident when routing
# finishes delay its launch; with N > 0 they run on N SMs (torch SM carveout, captured into CUDA
# graphs) and the routed experts' backend keeps N SMs free of its persistent grids
# (``reserved_sms``, ``_k3_routed_backend_opt_ins``).


@contextmanager
def _k3_shared_sm_scope(shared_sm: int):
    """Run the enclosed cuBLAS GEMMs on ``shared_sm`` SMs (no-op when 0)."""
    if shared_sm <= 0:
        yield
        return
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    prev = torch._C._get_sm_carveout_experimental()
    torch._C._set_sm_carveout_experimental(max(0, num_sms - shared_sm))
    try:
        yield
    finally:
        torch._C._set_sm_carveout_experimental(prev)


# ``K3Sm107Constants.router_exact_kda`` (bit-identical to the path it replaces): in the MoE blocks
# of KDA layers, the router logits of M <= 16 tokens come from k3_router_exact (bit-identical to
# dsv3_router_gemm_op; its weight is copied into shared memory before griddepcontrol.wait). The
# faster router also starts the route tail earlier, inside the burst of latent_down and the shared
# gate_up; after an MLA attention block the route tail then stretches by ~3 us (b2 gen-only,
# GB300) and the routing chain gets longer than with dsv3_router_gemm_op, while after a KDA
# attention block the front gets shorter.
# ``K3Sm107Constants.latent_ar_own_ws``: for up to _K3_LATENT_AR_OWN_WS_MAX_M tokens, the routed
# latent all-reduce + RMSNorm (the same MNNVL Lamport kernels) runs on a workspace of its own instead of the
# one it shares with the shared expert's all-reduce. It then no longer has to follow the shared-expert
# chain: the main stream issues it and latent_up right after the routed FC2 and joins the shared
# stream after latent_up. Applies where latent_up does not add the shared output itself (no fused
# latent_up + residual variant); the output is bit-identical.
# Up to _K3_LATENT_AR_ONESHOT_MAX_BYTES (rows x latent x ranks x 2 B; the workspace is sized for it,
# the other MNNVL all-reduces keep the global one-shot limit: 6291456 runs the 96-row latent
# all-reduce of 8 ranks, 5.5 MB, one-shot) that all-reduce runs k3_latent_ar_norm, bit-identical to
# the C++ one-shot kernel but triggering its programmatic dependents (latent_up) right after its
# griddepcontrol.wait instead of after the cross-rank reduction: its row kernel where that serves
# the size (one 512-thread CTA per row instead of a cluster of four; the rank's own slot is summed
# from registers instead of being polled back from the multicast loopback; slots that arrived are
# not reloaded), else the cluster kernel; elsewhere the C++ kernel (trtllm::mnnvl_fusion_allreduce)
# runs on the same workspace.
# Where latent_up runs the mid-M path of the k3_fp8w_lowm_gemm entry (above the low-M cut-over, e.g.
# the 96-row helix8 b16 mtp5 verify step), the row kernel also stores the MXFP8 form of the
# normalized latent in that GEMM's scale layout and latent_up runs the mid-M GEMM on it
# (k3_fp8w_lowm_gemm.run_mid_prequantized) instead of quantizing the BF16 rows in a kernel of its
# own: bitwise the same projection, one kernel less after the all-reduce.
_K3_LATENT_AR_OWN_WS_MAX_M = 128
_K3_LATENT_AR_ONESHOT_MAX_BYTES = 6291456
# The routed latent all-reduce's own MNNVL workspaces, by all-reduce mapping.
_K3_LATENT_AR_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


def _k3_latent_ar_workspace(
    mapping: Mapping, dtype: torch.dtype, hidden_size: int, max_tokens: int
) -> Dict[str, Any]:
    """MNNVL Lamport workspace of the routed latent all-reduce alone (three buffers and the state
    flags, laid out as ``get_or_scale_allreduce_mnnvl_workspace`` lays out the shared one), sized
    for up to ``max_tokens`` rows of ``hidden_size``. Collective over the all-reduce group: every
    rank builds it at the same point of model construction."""
    workspace = _K3_LATENT_AR_WORKSPACES.get(mapping)
    if workspace is not None:
        if workspace["max_tokens"] < max_tokens:
            raise ValueError(
                f"Kimi K3 latent all-reduce workspace serves {workspace['max_tokens']} tokens, "
                f"{max_tokens} requested"
            )
        return workspace
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    # The shared workspace's communicator (split by the all-reduce group) is the same group.
    comm = _k3_mnnvl_mpi_comm(get_or_scale_allreduce_mnnvl_workspace(mapping, dtype))
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden_size, mapping.tp_size, dtype)
        for m in range(1, max_tokens + 1)
    )
    # One-shot rows under the latent all-reduce's own limit: every rank's rows in one buffer.
    row_bytes = hidden_size * mapping.tp_size * torch.tensor([], dtype=dtype).element_size()
    one_shot_rows = min(max_tokens, _K3_LATENT_AR_ONESHOT_MAX_BYTES // row_bytes)
    need = max(need, one_shot_rows * row_bytes)
    buffer_bytes = -(-need // (1 << 20)) << 20
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    handle = McastGPUBuffer(
        3 * buffer_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        use_fabric_handle,
        comm.py2f(),
    )
    # Lamport-clean (-0.0) before any rank writes into it.
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    # [current index, dirty index, bytes per buffer, dirty stages, bytes to clear x4, arrivals]
    flags = torch.tensor(
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    workspace = {
        "handle": handle,
        "buffer": buffer.view(dtype).view(3, -1),
        # The multicast view of the same buffers (stores reach every rank's copy).
        "mc": handle.get_mc_buffer((3 * buffer_bytes // 4,), torch.float32, 0),
        "flags": flags,
        "max_tokens": max_tokens,
        # Bytes of one of the three Lamport buffers.
        "buffer_bytes": buffer_bytes,
        "rank": mapping.tp_rank,
        "world_size": mapping.tp_size,
    }
    _K3_LATENT_AR_WORKSPACES[mapping] = workspace
    return workspace


# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)


class K3MoEGate(nn.Module):
    """Kimi K3 gate: the router weight, the score-correction bias and the routing kernels.

    Routing is ungrouped sigmoid top-k with the bias added for the selection and the selected
    scores renormalized and scaled (``noaux_tc``); :meth:`route`, :meth:`route_with_sort` and
    :meth:`route_tail` are three kernels of the same routing.
    """

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        if self.moe_router_activation_func != "sigmoid" or not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 routing takes sigmoid scores with top-k weight renormalization"
            )
        if self.num_expert_group != 1 or self.topk_group != 1:
            raise ValueError("Kimi K3 routing is ungrouped (num_expert_group = topk_group = 1)")
        if self.num_experts > 1024 or self.top_k > 32:
            raise ValueError(
                f"the fused routing kernel serves up to 1024 experts and top-32, got "
                f"{self.num_experts} and top-{self.top_k}"
            )

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        weight_dtype = logits_gemm_dtype or torch.float32
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        if self.weight.dtype == torch.bfloat16 and hidden_2d.dtype == torch.bfloat16:
            return torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_2d.contiguous(),
                self.weight.t(),
                bias=None,
                out_dtype=torch.float32,
            )
        return torch.nn.functional.linear(
            hidden_2d.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(ids int32, weights fp32)``, each ``[num_tokens, top_k]``: ``noaux_tc_op``."""
        values, indices = noaux_tc_op(
            logits,
            self.e_score_correction_bias,
            self.num_expert_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )
        return indices.to(torch.int32), values.to(torch.float32)

    def route_with_sort(
        self, logits: torch.Tensor, local_expert_offset: int, local_num_experts: int
    ) -> Optional[SortedRouting]:
        """The routing with the ``moe_sort`` tables of experts ``[local_expert_offset,
        local_expert_offset + local_num_experts)`` in one launch (``noaux_tc_route_sort``, bitwise
        ``noaux_tc_op`` + ``moe_sort``), or ``None`` where that kernel does not serve."""
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not IS_CUTLASS_DSL_AVAILABLE:
            return None
        from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_routing

        if self.top_k != kimi_k3_routing.TOP_K or not kimi_k3_routing.serves(
            logits.shape[0], logits.shape[1], local_expert_offset, local_num_experts, logits.device
        ):
            return None
        ids, weights, *tables = torch.ops.trtllm.noaux_tc_route_sort(
            logits,
            self.e_score_correction_bias,
            float(self.routed_scaling_factor),
            local_expert_offset,
            local_num_experts,
        )
        return SortedRouting(ids, weights, SortTables(kimi_k3_routing.TILE_SIZE, tuple(tables)))

    def route_tail(
        self, logits: torch.Tensor, local_expert_offset: int, local_num_experts: int
    ) -> SortedRouting:
        """The CuTe routing tail (``k3_route_tail``, top-k bitwise ``noaux_tc_op``) with the
        ``moe_sort`` tables of experts ``[local_expert_offset, local_expert_offset +
        local_num_experts)``."""
        from ..kernels.k3r_moe import k3_route_tail

        # Above the cluster kernel's tokens, the multi-CTA sort's persistent workspace.
        workspace = (
            k3_route_tail.wide_workspace(logits.device)
            if logits.shape[0] > k3_route_tail.MAX_M
            else None
        )
        outs = k3_route_tail_op(
            logits,
            self.e_score_correction_bias,
            float(self.routed_scaling_factor),
            local_expert_offset,
            local_num_experts,
            k3_route_tail.TILE,
            workspace=workspace,
        )
        return SortedRouting(outs[0], outs[1], SortTables(k3_route_tail.TILE, tuple(outs[2:8])))


def _router_exact_for_layer(router_exact_kda: bool, cfg, layer_idx: int) -> bool:
    """Whether ``k3_router_exact`` serves the router of this layer's MoE block
    (``router_exact_kda``: in the KDA layers)."""
    return router_exact_kda and _is_kda_layer(cfg, layer_idx)


# Row-parallel MoE tail for generation-only batches of up to _K3_RP_TAIL_MAX_M tokens (the low-M
# GEMM's range), where the routed latent is all-reduced by the model (MoE TP/EP over the all-reduce
# group, TP-sharded shared expert) and the layer's residual update is fused
# (``fuse_moe_residual``). After the latent all-reduce + RMSNorm, every rank runs one low-M
# FP8 GEMM over its whole 128-column blocks of the normalized latent with the matching latent_up
# columns (FP8 codes and block scales read in place) and pushes its partial [M, hidden] plus its
# shared-expert partial into the next all-reduce on the latent all-reduce's MNNVL workspace: the
# shared expert keeps its BF16 down GEMM (row-parallel shard, no all-reduce) on its stream, and the
# tail GEMM's push epilogue adds that BF16 partial to its BF16 output. The next layer's
# pre-attention mix finishes that all-reduce (one-shot transport up to
# lamport._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot above) and adds it to the residual
# (``_DeferredRpTail``). This replaces the shared expert's all-reduce and the replicated latent_up.
# Not bitwise: the latent_up sum is split over the ranks.
_K3_RP_TAIL_MAX_M = 32


# ---------------------------------------------------------------------------
# Latent MoE block: router, routed experts (kernel chains of moe.py), shared expert.
# ---------------------------------------------------------------------------

# Routed-expert key spellings that ModelOpt emits for Kimi K3. The NVFP4
# checkpoint (``nvidia/Kimi-K3-NVFP4``) lists every prefix x module-name
# combination in ``quantized_layers``, so a lookup over this product finds it
# without needing the MiniMax-M3-style prefix normalization in ``ModelConfig``.
_K3_ROUTED_EXPERT_KEY_PREFIXES = ("language_model.model.", "model.", "")
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# The subset of the above that can be a real module path. ``exclude_modules``
# matches with wildcards and walks ancestor prefixes, so an empty prefix would
# widen what matches instead of just missing, as it does in the dict lookup.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")

# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


# ---------------------------------------------------------------------------
# Routed-expert checkpoint layout.
#
# K3 streams routed experts one at a time (see ``load_weights``): the tensor names, the per-expert
# loader and the per-layer finalize it needs are decided by the checkpoint's routed-expert
# quantization. The expected-key plan, the loader and the file grouping read them off one spec.
# ---------------------------------------------------------------------------


def _load_nvfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    for w in ("w1", "w3"):
        if get_optional_tensor(f"{base}.{expert_idx}.{w}.pre_quant_scale") is not None:
            raise NotImplementedError(
                "Kimi K3 routed experts take plain NVFP4 checkpoints; this one carries a "
                "pre-quant activation scale (NVFP4_AWQ / NVFP4_ARC)"
            )
    tensors = {
        f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
        for w in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    if isinstance(backend, K3Nvfp4Experts):
        backend.load_expert(global_expert_id=expert_idx, local_slot_id=local_slot_id, **tensors)
    else:
        backend.quant_method.load_streaming_nvfp4_expert(
            backend, global_expert_id=expert_idx, local_slot_id=local_slot_id, **tensors
        )


def _k3_routed_backend_opt_ins(routed_experts, *, shared_sm: int, fc12_die_major: bool) -> None:
    """The routed experts' opt-ins of a MoE layer, made once at construction.

    SMs the FC12 kernels' persistent grids leave to the kernels the shared stream runs beside them
    (the shared expert's GEMMs confined to ``shared_sm`` SMs). The CUTEDSL kernels take no
    reserve. ``fc12_die_major``: the target asks for die-major expert weights, which the routed
    experts do not have yet (logged once; they keep the loaded layout).
    """
    if fc12_die_major:
        logger.info_once(
            "TRTLLM_K3_FC12_DIE_LAYOUT: die-major expert weights are not available; the routed "
            "experts keep the loaded layout",
            key="k3_fc12_die_major_off",
        )
    if getattr(routed_experts, "kernels", None) != "CUTEDSL_FC12":
        return
    if shared_sm > 0:
        routed_experts.sm_reserve = shared_sm


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # The loader stages per-expert state that the layer's finalize consumes.
    needs_layer_finalize: bool
    # Whole per-expert key suffixes the loader reads WHEN PRESENT (to reject them).
    optional_suffixes: Tuple[str, ...] = ()


_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.NVFP4: _K3ExpertCkptSpec(
        kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
        loader=_load_nvfp4_expert,
        loaded_slots_attr="_streamed_expert_slots",
        needs_layer_finalize=True,
        optional_suffixes=("w1.pre_quant_scale", "w3.pre_quant_scale"),
    ),
}


class _RpTailOperands(NamedTuple):
    """One rank's operands of the row-parallel MoE tail GEMM: K is this
    rank's ``k1`` latent columns from column ``k0`` (``k1 / 128`` whole blocks of latent_up, whose
    FP8 codes the GEMM reads in place); the GEMM's push epilogue adds the shared expert's BF16
    partial."""

    k0: int
    k1: int
    # DeepGEMM-packed UE8M0 scale of the [hidden, k1] tail weight.
    scale: torch.Tensor


def _rp_tail_latent_slice(latent_size: int, world: int, rank: int) -> Tuple[int, int]:
    """``(first column, columns)`` of ``rank``'s 128-column blocks of the latent: whole blocks,
    as even as possible (28 blocks over 8 ranks: 4 on ranks 0-3, 3 on ranks 4-7)."""
    blocks = latent_size // 128
    base, extra = divmod(blocks, world)
    first = rank * base + min(rank, extra)
    return first * 128, (base + (rank < extra)) * 128


def _k3_expert_ckpt_spec(quant_config: QuantConfig) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_config.quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_config.quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    group_size = getattr(quant_config, "group_size", None)
    if group_size not in (None, 16):
        raise NotImplementedError(
            f"Kimi K3 NVFP4 routed experts take 16-element scale blocks, got group_size={group_size}"
        )
    return spec


class _K3SharedExpert(GatedMlp):
    """The MoE layer's shared expert: :class:`GatedMlp` called through its projection modules,
    so a projection ``_convert_moe_mlps_to_fp8_weight_read`` swapped for an FP8 weight read runs
    its own kernel (a ``Bf16Linear`` without bias or all-reduce is ``F.linear`` either way)."""

    def forward(
        self, x: torch.Tensor, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        h = self.gate_up_proj(x)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a, all_reduce_params=final_all_reduce_params)


def _megamoe_routed_experts(
    gate: "K3MoEGate",
    routed_moe_model_config: ModelConfig,
    routed_quant_config: QuantConfig,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    layer_idx: int,
    aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    situ_beta: float,
    situ_linear_beta: float,
) -> nn.Module:
    """The routed experts of a ``MEGAMOE_CUTEDSL`` layer: the MegaMoE module, which runs the
    token exchange, both GEMMs and the combine in one kernel over symmetric memory."""
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

    routing_method = DeepSeekV3MoeRoutingMethod(
        top_k=gate.top_k,
        n_group=gate.num_expert_group,
        topk_group=gate.topk_group,
        routed_scaling_factor=gate.routed_scaling_factor,
        callable_e_score_correction_bias=lambda: gate.e_score_correction_bias,
        is_fused=True,
    )
    moe = create_moe(
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        # Kimi owns the latent reduction so it can order that collective
        # after the shared expert's auxiliary-stream reduction.
        reduce_results=False,
        # Lets the fused-communication kernel return its globally combined routed output.
        accepts_global_routed_output=True,
        model_config=routed_moe_model_config,
        override_quant_config=routed_quant_config,
        layer_idx=layer_idx,
        aux_stream_dict=aux_stream_dict,
        communication_method=None,
        activation=SiTuActivation(gate_softcap=situ_beta, linear_softcap=situ_linear_beta),
        allow_backend_degradation=False,
    )
    if not isinstance(moe, ConfigurableMoE) or moe.layer_load_balancer is not None:
        raise RuntimeError("Kimi K3 MegaMoE layers need ConfigurableMoE without EPLB")
    return moe


class K3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block: the gate routes, the routed experts run the CuTe DSL kernel chain
    of ``moe`` (MEGAMOE_CUTEDSL: the MegaMoE module), the shared expert runs beside
    them."""

    # ``latent_ar_own_ws``: the routed latent all-reduce's own MNNVL workspace (``None``: it
    # shares the one of the shared expert's all-reduce).
    _latent_ar_ws: Optional[Dict[str, Any]] = None
    # ``router_exact_kda``: whether this layer's router logits come from k3_router_exact (set
    # per layer in ``__init__``; off unless the constant selects the layer).
    _router_exact = False
    # The row-parallel tail: this rank's tail GEMM operands, built after loading (``None``: not
    # applicable to this layer).
    _rp_tail: Optional[_RpTailOperands] = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        *,
        constants: K3Sm107Constants,
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.

        ``constants``: the target's switches (``shared_sm``, ``router_exact_kda``,
        ``latent_ar_own_ws``, ``fc12_die_major``, ``fp8w_lowm_gemm``, ``ld_fused``).
        """
        super().__init__()
        self._constants = constants
        self.layer_idx = layer_idx
        self._router_exact = _router_exact_for_layer(constants.router_exact_kda, cfg, layer_idx)
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        dtype = torch.bfloat16

        # Routing scores stay fp32; with attention-DP off the gate GEMM runs
        # bf16xbf16 with fp32 accumulate/output (checkpoint stores the gate
        # weight in bf16; saves a per-layer input cast + fp32 splitK pair on
        # the bs1 decode path). Under attention-DP the legacy upcast-to-fp32
        # GEMM is kept: the bf16-input min-latency GEMM's different reduction
        # order flips borderline top-16 picks (GSM8K 96.7 -> 96.1/96.4,
        # 3-run bisect on 62b20dd868), and the bs1-latency win is irrelevant
        # at DEP batch sizes. helix_a2a ranks share requests and keep bf16.
        # KIMI_K3_ROUTER_BF16=1/0 forces either path.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = (
            _router_bf16_env == "1"
            if _router_bf16_env is not None
            else model_config.mapping.tp_group_shares_requests
        )
        self.gate = K3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config)
        self._moe_backend = routed_moe_model_config.moe_backend
        self._attention_dp = model_config.mapping.enable_attention_dp
        routed_mapping = routed_moe_model_config.mapping
        if self._moe_backend in _K3_ROUTED_KERNEL_SETS:
            if routed_moe_model_config.moe_disable_finalize_fusion:
                raise NotImplementedError(
                    "Kimi K3 routed experts run the fused-finalize kernels "
                    "(moe_disable_finalize_fusion / FORCE_DETERMINISTIC is not supported)"
                )
            from tensorrt_llm._torch.locality_domain.policy import LocalityDomainExecutionPlanner

            locality_plan = LocalityDomainExecutionPlanner(
                routed_moe_model_config.locality_domain_policy
            ).plan_moe(
                routed_quant_config,
                moe_backend=self._moe_backend,
                use_fused_finalize=True,
                dtype_activation=dtype,
                activation=ActivationType.SiTu.name,
            )
            if locality_plan.enabled:
                raise NotImplementedError(
                    "Kimi K3 routed experts do not run the locality-domain MoE kernels; "
                    "disable enable_locality_domains"
                )
            self.routed_experts = K3RoutedExperts(
                kernels=self._moe_backend,
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.moe_hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                mapping=routed_mapping,
                layer_idx=layer_idx,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                aux_stream=aux_stream_dict.get(AuxStreamType.MoeOutputMemset),
                max_chunk_tokens=routed_moe_model_config.moe_max_num_tokens,
                fc12_die_major=constants.fc12_die_major,
            )
            # The routed latent partial sums, reduced over the TP group (Kimi orders this
            # collective after the shared expert's).
            self.routed_experts.all_reduce = (
                TpAllReduce(routed_mapping, dtype, routed_moe_model_config.allreduce_strategy)
                if not routed_mapping.enable_attention_dp and routed_mapping.tp_size > 1
                else None
            )
            returns_global_routed_output = False
            self._routed_fused_comm = False
        else:
            self.routed_experts = _megamoe_routed_experts(
                self.gate,
                routed_moe_model_config,
                routed_quant_config,
                self.num_experts,
                self.moe_hidden_size,
                cfg.moe_intermediate_size,
                dtype,
                layer_idx,
                aux_stream_dict,
                situ_beta,
                situ_linear_beta,
            )
            returns_global_routed_output = self.routed_experts.returns_global_routed_output
            self._routed_fused_comm = True
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        self._shared_intermediate = shared_intermediate
        attention_dp = model_config.mapping.enable_attention_dp
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
        # Fused-comm MegaMoE runs its own cross-rank exchange inside the
        # kernel. Order the shared-expert all-reduce after the routed MoE
        # instead of running the two collectives concurrently (startup
        # deadlocks). CuteDSL is the validated scope, not a correctness
        # dependency: DeepGEMM rejects TP without attention DP.
        self._defer_shared_allreduce = use_shared_tp and self._routed_fused_comm
        self._reduce_routed_output = use_shared_tp and not returns_global_routed_output
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the routed all-reduce."
            )
        if model_config.use_cute_dsl_bf16_gemm:
            raise NotImplementedError(
                "Kimi K3's shared expert runs the cuBLAS BF16 GEMMs "
                "(use_cute_dsl_bf16_gemm is not supported)"
            )
        self.shared_experts = _K3SharedExpert(
            cfg.hidden_size,
            shared_intermediate,
            activation="situ",
            beta=situ_beta,
            linear_beta=situ_linear_beta,
            dtype=dtype,
            mapping=(
                subgroup_mapping(model_config.mapping, 1) if attention_dp else model_config.mapping
            ),
            reduce_output=use_shared_tp,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        shared_sm = constants.shared_sm
        if shared_sm > 0:
            shared_forward = self.shared_experts.forward

            def _shared_forward_confined(*args, **kwargs):
                with _k3_shared_sm_scope(shared_sm):
                    return shared_forward(*args, **kwargs)

            self.shared_experts.forward = _shared_forward_confined
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        self.moe_routed_event = torch.cuda.Event() if self._defer_shared_allreduce else None
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if self._moe_backend in ("CUTEDSL", "CUTEDSL_FC12")
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # FlashInfer RMSNorm on the reduced latent (or folded into the routed all-reduce).
        self.routed_expert_norm = RMSNormWeight(self.moe_hidden_size, cfg.rms_norm_eps, dtype=dtype)
        if constants.latent_ar_own_ws and self._reduce_routed_output:
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                self._latent_ar_ws = _k3_latent_ar_workspace(
                    mnnvl.mapping, mnnvl.dtype, self.moe_hidden_size, _K3_LATENT_AR_OWN_WS_MAX_M
                )
        _k3_routed_backend_opt_ins(
            self.routed_experts, shared_sm=shared_sm, fc12_die_major=constants.fc12_die_major
        )

    def _lu_mid_prequant_plan(self, latent: torch.Tensor):
        """``(tactic, sf_blocks)`` of the mid-M path that
        latent_up's :meth:`_Fp8BlockScaleWeightReadLinear.forward` runs on the BF16 rows ``latent``
        (SM107: the K3 low-M GEMM on ``lowm_weight_scale`` above the low-M cut-over),
        or ``None`` where it runs another GEMM."""
        up = self.routed_expert_up_proj
        if (
            not self._constants.fp8w_lowm_gemm
            or not isinstance(up, _Fp8BlockScaleWeightReadLinear)
            or up.lowm_weight_scale is None
            or up.next_ar_push is not None
        ):
            return None
        if not up._lowm_gemm_applies(latent, up.lowm_weight_scale):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.mid_prequant_plan(latent.shape[0], up.out_features, up.in_features)

    def _latent_up_mid_prequantized(
        self, routed_latent: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, tactic
    ) -> torch.Tensor:
        """latent_up on the MXFP8 form ``(xq, xs)`` of ``routed_latent`` in the mid-M GEMM's
        layout: what :meth:`_routed_projection` computes, bit for
        bit, without its quantization kernel."""
        up = self.routed_expert_up_proj
        m = xq.shape[0]
        out = up._take_next_output(m)
        if out is None:
            out = torch.empty((m, up.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_mid_gemm_op(xq, xs, up.weight, up.lowm_weight_scale, out, tactic)
        return out.reshape(routed_latent.shape[:-1] + (up.out_features,))

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits; ``router_exact_kda`` serves ``M <= 16`` BF16 tokens with
        ``k3_router_exact`` (bit-identical to the ``dsv3_router_gemm_op`` it replaces) in the
        layers it selects."""
        if self._router_exact:
            from ..kernels.k3r_moe import k3_router_exact

            x = hidden_states.reshape(-1, self.gate.gating_dim)
            if k3_router_exact.can_run(x, self.gate.weight):
                return k3_router_logits_op(x, self.gate.weight)
        return self.gate.compute_logits(hidden_states)

    def _shared_needs_all_reduce(self) -> bool:
        """Whether the shared expert's output is a row-parallel partial to all-reduce."""
        return bool(getattr(self.shared_experts.down_proj, "reduce_output", False))

    def _shared_all_reduce(self) -> TpAllReduce:
        """The shared expert's row-parallel reduction."""
        return self.shared_experts.down_proj.all_reduce

    def post_load_weights(self) -> None:
        self.prepare_rp_tail()

    def cache_derived_state(self) -> None:
        self.prepare_rp_tail()

    def prepare_expert_load(self) -> None:
        """Before this layer's first routed expert streams in."""
        backend = self.routed_experts.backend
        if isinstance(backend, K3Nvfp4Experts):
            backend.prepare_streaming_load()
        else:
            backend.quant_method.prepare_streaming_expert_load(backend)

    def finalize_expert_load(self) -> None:
        """After this layer's last routed expert streamed in."""
        backend = self.routed_experts.backend
        if isinstance(backend, K3Nvfp4Experts):
            backend.finalize()
        else:
            backend.process_weights_after_loading()

    def _rp_tail_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the row-parallel MoE tail, or ``None``."""
        if not self._reduce_routed_output or self._defer_shared_allreduce:
            return "the routed latent is not all-reduced by the model"
        moe_all_reduce = self.routed_experts.all_reduce
        if getattr(moe_all_reduce, "mnnvl_allreduce", None) is None:
            return "the routed all-reduce is not the MNNVL all-reduce"
        up = self.routed_expert_up_proj
        if not isinstance(up, _Fp8BlockScaleWeightReadLinear) or up._fused_projection is not None:
            return (
                "latent_up is not an FP8 weight-read projection (KIMI_K3_FP8_WEIGHT_READ_MOE_MLP)"
            )
        mlp = self.shared_experts
        if not isinstance(mlp, _K3SharedExpert) or not isinstance(mlp.down_proj, Bf16Linear):
            return "the shared expert does not run the BF16 gate_up -> activation -> down path"
        if not self._shared_needs_all_reduce():
            return "the shared expert is not row-parallel"
        down = mlp.down_proj
        if getattr(down, "bias", None) is not None:
            return "the shared down projection has a bias"
        world = moe_all_reduce.mapping.tp_size
        shard = down.weight.shape[1]
        if shard * world != self._shared_intermediate:
            return f"the shared down shard ({shard}) is not a 1/{world} shard"
        if (
            tuple(up.weight.shape) != (self.hidden_size, self.moe_hidden_size)
            or self.moe_hidden_size % 128
            or self.hidden_size % 128
            or self.moe_hidden_size // 128 < world
        ):
            return "the latent is not split into whole 128-column blocks over the ranks"
        if tuple(down.weight.shape) != (self.hidden_size, shard):
            return "unexpected shared down weight shape"
        return None

    def prepare_rp_tail(self) -> bool:
        """Build this rank's row-parallel tail operands (``_RpTailOperands``): the packed UE8M0
        scale of this rank's latent_up blocks (whose FP8 codes stay where they are). Rebuilt on
        every call (post-load), before CUDA graph capture."""
        self._rp_tail = None
        reason = self._rp_tail_unsupported()
        if reason is not None:
            logger.info_once(
                f"Kimi K3 MoE layers keep the replicated tail (no row-parallel tail): {reason}",
                key=f"k3_rp_tail_off_{reason}",
            )
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        up = self.routed_expert_up_proj
        up.transform_weights()
        down = self.shared_experts.down_proj
        mapping = self.routed_experts.all_reduce.mapping
        k0, k1 = _rp_tail_latent_slice(self.moe_hidden_size, mapping.tp_size, mapping.tp_rank)
        hidden, shard = down.weight.shape
        with torch.no_grad():
            exponents = k3_fp8w_lowm_gemm.block_exponents_of(
                up.weight_scale, hidden, self.moe_hidden_size
            )[:, k0 // 128 : (k0 + k1) // 128]
            scale = k3_fp8w_lowm_gemm.pack_ue8m0_exponents(exponents.contiguous(), hidden, k1)
        self._rp_tail = _RpTailOperands(k0, k1, scale)
        logger.info_once(
            f"Kimi K3 row-parallel MoE tail, K = latent [{k0}, {k0 + k1}), BF16 shared "
            f"partial added (rank {mapping.tp_rank} of {mapping.tp_size})",
            key=f"k3_rp_tail_on_{k1}_{shard}",
        )
        return True

    def _rp_tail_target(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor], rp_tail_ok: bool
    ):
        """The all-reduce push target of this forward's row-parallel MoE tail, or ``None`` where
        the forward keeps the replicated tail: a generation-only batch of at most
        ``_K3_RP_TAIL_MAX_M`` BF16 tokens with the fused residual update, served by the low-M
        GEMM's push epilogue. The consumer finishes the call: one-shot transport up to
        lamport._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot above."""
        rp = self._rp_tail
        num_tokens = hidden_states.shape[0]
        if (
            rp is None
            or not rp_tail_ok
            or residual is None
            or hidden_states.dtype != torch.bfloat16
            or not 1 <= num_tokens <= _K3_RP_TAIL_MAX_M
        ):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        if k3_fp8w_lowm_gemm.split_source_tactic(num_tokens, self.hidden_size, rp.k1) is None:
            return None
        return lamport_push_target(
            self.routed_experts.all_reduce,
            num_tokens,
            self.hidden_size,
            deferred=True,
        )

    def _rp_shared_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The shared expert's part of the row-parallel tail on this rank: its row-parallel BF16
        down-projection partial without the all-reduce."""
        return self.shared_experts(
            hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
        )

    def _rp_tail_forward(
        self,
        routed_output: Callable[[], torch.Tensor],
        identity: torch.Tensor,
        residual: torch.Tensor,
        target,
        defer_residual: bool,
    ):
        """The row-parallel tail: the routed experts (``routed_output``) and the shared expert's
        part (``_rp_shared_output``), then the latent all-reduce + RMSNorm, then this rank's tail
        GEMM pushing its partial into ``target``. Returns a ``_DeferredRpTail``
        (``defer_residual``) or the updated residual."""
        if self.shared_expert_stream is not None and do_multi_stream():
            self.moe_main_event.record()
            routed_out = routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_part = self._rp_shared_output(identity)
                self.moe_shared_event.record()
            self.moe_shared_event.wait()
        else:
            routed_out = routed_output()
            shared_part = self._rp_shared_output(identity)
        moe_all_reduce = self.routed_experts.all_reduce
        routed_latent = moe_all_reduce(
            routed_out,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM,
                norm_weight=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
            ),
        )
        latent = routed_latent.reshape(-1, self.moe_hidden_size)
        num_tokens = latent.shape[0]
        shared_part = shared_part.reshape(num_tokens, -1)
        if not shared_part.is_contiguous():
            shared_part = shared_part.contiguous()
        rp = self._rp_tail
        lo, hi = rp.k0, rp.k0 + rp.k1
        k3_fp8w_lowm_gemm_push_op(
            latent[:, lo:hi],
            self.routed_expert_up_proj.weight[:, lo:hi],
            rp.scale,
            target.workspace,
            target.one_shot,
            addend=shared_part,
            strided=True,
        )
        target.pushed = True
        result = finish_lamport_push(moe_all_reduce, target, deferred=True)
        deferred = _DeferredRpTail(residual.reshape(num_tokens, -1), result)
        if defer_residual:
            return deferred
        return deferred.materialize().view(residual.shape)

    def _route_tail_target(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        """``(local_expert_offset, local_num_experts)`` if the CuTe routing tail serves this
        forward, else ``None``."""
        if not isinstance(self.routed_experts, K3RoutedExperts) or self._attention_dp:
            return None
        backend = self.routed_experts.backend
        slot_start = backend.slot_start
        num_local = backend.expert_size_per_partition
        if backend.num_slots != self.gate.num_experts:
            return None
        from ..kernels.k3r_moe import k3_route_tail

        device = self.gate.e_score_correction_bias.device
        if not k3_route_tail.supports_device(device):
            return None
        if not 1 <= num_tokens <= min(_K3_ROUTE_TAIL_MAX_M, k3_route_tail.WIDE_MAX_M):
            return None
        if num_tokens > k3_route_tail.MAX_M:
            if not k3_route_tail.wide_sort_fits(num_tokens, num_local):
                return None
            # The multi-CTA kernel's workspace must exist before graph capture.
            if not torch.cuda.is_current_stream_capturing():
                k3_route_tail.wide_workspace(device)
            elif not k3_route_tail.wide_workspace_ready(device):
                return None
        return slot_start, num_local

    def _router(self, tail_target: Optional[Tuple[int, int]]):
        """The routing of this forward's router logits, for the experts
        ``[local_expert_offset, local_expert_offset + local_num_experts)`` the routed experts ask
        for: the CuTe routing tail where ``tail_target`` is set; else ``noaux_tc_route_sort``
        where it serves; else ``noaux_tc_op`` (the experts then run ``moe_sort``)."""
        gate = self.gate
        consumes_tables = self.routed_experts.consumes_sort_tables

        def route(logits: torch.Tensor, local_expert_offset: int, local_num_experts: int):
            target = (local_expert_offset, local_num_experts)
            if tail_target is not None:
                routed = gate.route_tail(logits, *tail_target)
                if consumes_tables and target == tail_target:
                    return routed
                return SortedRouting(routed.token_selected_experts, routed.token_final_scales, None)
            if consumes_tables:
                routed = gate.route_with_sort(logits, *target)
                if routed is not None:
                    return routed
            return SortedRouting(*gate.route(logits), None)

        return route

    def _fused_projection_applies(self, x: torch.Tensor, projection: nn.Module) -> bool:
        """Whether the fused latent projection replaces ``projection(x)`` (``ld_fused``)."""
        if not self._constants.ld_fused:
            return False
        if not isinstance(projection, _Fp8BlockScaleWeightReadLinear):
            return False
        packed_scale = projection.packed_weight_scale
        if packed_scale is None or x.dtype != torch.bfloat16:
            return False
        if not _K3_LD_FUSED_MIN_M <= x.shape[0] <= _K3_LD_FUSED_MAX_M:
            return False
        from ..kernels.k3r_gemm import k3_latent_fp8_gemm

        # Where the fused op itself runs the low-M GEMM (SM107), it only adds the epilogue.
        if x.shape[0] <= k3_latent_fp8_gemm.lowm_route_max_m(*projection.weight.shape):
            return True
        if self._constants.fp8w_lowm_gemm and _k3_lowm_gemm_can_run(
            x, projection.weight, packed_scale
        ):
            return False
        return True

    def _fused_latent_input(self, hidden_states: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """NVFP4 routed-expert input straight from the hidden states, or ``None``.

        Runs the latent_down projection with the experts' input quantization fused in
        (``ops.gemm.k3_latent_down_fp4``) when enabled, for the FC12 routed experts (which quantize
        their input with ``fp4_quantize`` against ``fc31_input_scale``) of one launch chain.
        """
        down = self.routed_expert_down_proj
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not self._fused_projection_applies(x, down):
            return None
        moe = self.routed_experts
        if getattr(moe, "kernels", None) != "CUTEDSL_FC12" or x.shape[0] > moe.max_chunk_tokens:
            return None
        from ..kernels.k3r_gemm import k3_latent_fp8_gemm

        global_sf = moe.backend.fc31_input_scale
        w_scale = down.packed_weight_scale
        if not k3_latent_fp8_gemm.can_run(x, down.weight, w_scale, global_sf):
            return None
        codes, scales = k3_latent_down_fp4_op(x, down.weight, w_scale, global_sf)
        return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)

    def _own_workspace_latent_tail(
        self,
        routed_out: torch.Tensor,
        shared_out: torch.Tensor,
        residual: Optional[torch.Tensor],
        defer_residual: bool,
    ):
        """``latent_ar_own_ws``: the routed latent all-reduce + RMSNorm on its own MNNVL
        workspace (``k3_latent_ar_norm`` up to ``_K3_LATENT_AR_ONESHOT_MAX_BYTES``, the C++ fused
        all-reduce otherwise) and latent_up, issued on the main stream before it joins the shared-expert
        stream, whose output only the residual update reads. Returns what ``forward`` returns."""
        workspace = self._latent_ar_ws
        gamma = self.routed_expert_norm.weight
        eps = float(self.routed_expert_norm.variance_epsilon)
        latent = routed_out.reshape(-1, routed_out.shape[-1]).contiguous()
        routed_latent = None
        latent_mxfp8 = None
        mid_plan = None
        from ..kernels.k3r_comm import k3_latent_ar_norm

        world_size = workspace["world_size"]
        one_shot = (
            latent.numel() * world_size * latent.element_size() <= _K3_LATENT_AR_ONESHOT_MAX_BYTES
        )
        row = k3_latent_ar_norm.can_run_row(latent, gamma, world_size, workspace["buffer_bytes"])
        if one_shot and (row or k3_latent_ar_norm.can_run(latent, gamma, world_size)):
            routed_latent = torch.empty_like(latent)
            m, k = latent.shape
            if row:
                # The MXFP8 operand of latent_up's mid-M GEMM, in its scale layout.
                mid_plan = self._lu_mid_prequant_plan(latent)
                if mid_plan is not None:
                    latent_mxfp8 = (
                        torch.empty((m, k), dtype=torch.float8_e4m3fn, device=latent.device),
                        torch.empty(
                            (k // 128) * mid_plan[1] * 512,
                            dtype=torch.uint8,
                            device=latent.device,
                        ),
                    )
            k3_latent_ar_rmsnorm(
                latent,
                gamma,
                eps,
                routed_latent,
                uc_addr=workspace["buffer"].data_ptr(),
                mc_addr=workspace["mc"].data_ptr(),
                flags=workspace["flags"],
                rank=workspace["rank"],
                world_size=world_size,
                mxfp8_out=latent_mxfp8,
                sf_blocks=mid_plan[1] if mid_plan is not None else 0,
                row=row,
            )
        if routed_latent is None:
            (routed_latent,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                latent,
                gamma,
                None,
                eps,
                workspace["buffer"],
                workspace["flags"],
                True,
                None,
                int(AllReduceFusionOp.RMS_NORM),
            )
        routed_latent = routed_latent.view(routed_out.shape)
        if mid_plan is not None:
            routed_out = self._latent_up_mid_prequantized(routed_latent, *latent_mxfp8, mid_plan[0])
        else:
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        self.moe_shared_event.wait()
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            return k3_moe_residual_add_op(routed_out, shared_out, residual)
        return routed_out + shared_out

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    def _can_fuse_router_down(self, hidden_states: torch.Tensor) -> bool:
        """Whether one fused GEMM can produce the router logits and the latent
        down-projection (trtllm::dsv3_router_latent_gemm_op).

        Scoped to the validated configuration (TP MEGAMOE_CUTEDSL with fused
        communication on SM107); the fusion itself does not depend on it.
        """
        if (
            _K3_DISABLE_MIN_LATENCY_LATENT_PROJ
            or not hidden_states.is_cuda
            or self._attention_dp
            or self._moe_backend != "MEGAMOE_CUTEDSL"
            or not self._routed_fused_comm
            or self.routed_experts.parallel_size <= 1
            or get_sm_version() != 107
        ):
            return False
        down = self.routed_expert_down_proj
        if not isinstance(down, nn.Linear) or down.bias is not None:
            return False
        if (
            hidden_states.ndim != 2
            or not 1 <= hidden_states.shape[0] <= 16
            or hidden_states.shape[1] != 7168
            or self.gate.weight.shape != (896, 7168)
            or down.weight.shape != (3584, 7168)
        ):
            return False
        # Inspect the loaded weights: FP8 conversion can replace the down module.
        return all(
            tensor.dtype == torch.bfloat16
            and tensor.is_cuda
            and tensor.device == hidden_states.device
            and tensor.stride() == (7168, 1)
            and tensor.data_ptr() % 16 == 0
            for tensor in (hidden_states, self.gate.weight, down.weight)
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``, taken from the checkpoint.

        ``nvidia/Kimi-K3-NVFP4`` declares the routed experts per layer as
        ``NVFP4`` with ``group_size=16``; the original ``moonshotai/Kimi-K3``
        declares nothing per layer and keeps the historical
        ``W4A8_MXFP4_MXFP8`` default. Reading the checkpoint instead of
        hardcoding is what lets one code path serve both.

        An exclusion outranks the per-layer entry and the default below:
        ``create_weights`` treats an override as authoritative over anything
        ``__post_init__`` wrote, so this return value stands in for both
        quantization passes and exclusion is the one that runs second. It is
        matched as a pattern, so it is asked only about real module names.
        """
        quant_config = model_config.quant_config
        if quant_config is not None and any(
            quant_config.is_module_excluded_from_quantization(
                f"{prefix}layers.{layer_idx}.{suffix}"
            )
            for prefix in _K3_ROUTED_EXPERT_MODULE_PREFIXES
            for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES
        ):
            logger.debug(
                "Kimi K3 layer %d routed experts: excluded from quantization, "
                "keeping them unquantized",
                layer_idx,
            )
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

        per_layer = getattr(model_config, "quant_config_dict", None)
        if per_layer:
            for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
                for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES:
                    cfg = per_layer.get(f"{prefix}layers.{layer_idx}.{suffix}")
                    if cfg is not None and cfg.quant_algo is not None:
                        # Logged once per layer: the routed-expert format decides
                        # which MoE backends can serve this checkpoint at all.
                        logger.debug(
                            "Kimi K3 layer %d routed experts: %s (group_size=%s) "
                            "from the checkpoint",
                            layer_idx,
                            cfg.quant_algo,
                            cfg.group_size,
                        )
                        return cfg
        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        # The CuTe DSL kernel sets K3 runs (context: CUTEDSL; generation: CUTEDSL_FC12 or the
        # fused-communication MEGAMOE_CUTEDSL).
        supported_backends = (*_K3_ROUTED_KERNEL_SETS, "MEGAMOE_CUTEDSL")
        if model_config.moe_backend not in supported_backends:
            raise ValueError(
                f"Kimi K3 SiTU routed experts run on {supported_backends}; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        if getattr(mapping, "_dwdp_size", 0) > 1:
            # DWDP does not hook the load path at all: setup_dwdp() scans the
            # model *after* every weight is loaded, reads w3_w1_weight /
            # w2_weight and their block scales straight off the backend
            # (dwdp/setup.py::collect_moe_params), copies them into fabric
            # memory, frees the originals, and rebinds param.data to a
            # composite VA view.  K3's packed-checkpoint streaming produces
            # exactly those tensors, so the two are compatible -- but only on
            # the NVFP4 CuteDSL leaves that declare supports_dwdp (CUTEDSL,
            # CUTEDSL_FC12).
            #
            # MEGAMOE_CUTEDSL is excluded deliberately rather than by
            # oversight: its process_weights_after_loading packs into the mega
            # buffers and shrinks the raw source params back to placeholders
            # (see maybe_finalize_layer), which has never been checked against
            # DWDP's copy-then-free of those same originals.
            if model_config.moe_backend not in ("CUTEDSL", "CUTEDSL_FC12"):
                raise NotImplementedError(
                    "Kimi K3 with DWDP requires moe_config.backend CUTEDSL or "
                    f"CUTEDSL_FC12; got {model_config.moe_backend!r}."
                )

        moe_tp, moe_ep = K3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
            )
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = model_config.moe_backend
        # MegaMoE uses this value as global DP SymmBuffer capacity, then
        # divides it by EP size for the per-rank allocation. Other backends
        # keep the user-configured value as their MoE chunking bound.
        # Preserve an explicitly larger capacity.
        if routed_model_config.moe_backend == "MEGAMOE_CUTEDSL":
            default_moe_max_num_tokens = routed_model_config.max_num_tokens * routed_mapping.dp_size
            configured_moe_max_num_tokens = int(routed_model_config.moe_max_num_tokens or 0)
            if configured_moe_max_num_tokens < default_moe_max_num_tokens:
                logger.info_once(
                    "Kimi K3 MegaMoE raises moe_max_num_tokens from "
                    f"{configured_moe_max_num_tokens} to {default_moe_max_num_tokens} "
                    "because the global DP SymmBuffer requires capacity for "
                    "max_num_tokens * dp_size.",
                    key=(
                        "kimi_k3_megamoe_capacity_override_"
                        f"{configured_moe_max_num_tokens}_{default_moe_max_num_tokens}"
                    ),
                )
            routed_model_config.moe_max_num_tokens = max(
                configured_moe_max_num_tokens,
                default_moe_max_num_tokens,
            )
        routed_model_config._frozen = True
        return routed_model_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        *,
        residual: Optional[torch.Tensor] = None,
        defer_residual: bool = False,
        rp_tail_ok: bool = False,
    ) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        With ``residual``, returns ``residual + moe_output``; with ``defer_residual`` as well, a
        ``_DeferredMoEResidual`` instead where that update would be the separate BF16 add, or a
        ``_DeferredRpTail`` where the row-parallel tail runs (the caller allows it with
        ``rp_tail_ok`` for generation-only batches).
        """
        identity = hidden_states
        rp_target = self._rp_tail_target(hidden_states, residual, rp_tail_ok)
        projected_input = None
        early_routed_in = None
        route_target = self._route_tail_target(hidden_states.shape[0])
        if self._can_fuse_router_down(hidden_states):
            router_logits, projected_input = torch.ops.trtllm.dsv3_router_latent_gemm_op(
                hidden_states, self.gate.weight, self.routed_expert_down_proj.weight
            )
        else:
            router_issued = False
            if self.latent_down_stream is not None and do_multi_stream():
                # Forked ahead of the router GEMM: the latent chain needs only
                # hidden_states.
                self.moe_latent_fork_event.record()
                if hidden_states.shape[0] <= _K3_ROUTER_FIRST_MAX_M:
                    # Router before latent_down in creation order (see _K3_ROUTER_FIRST_MAX_M).
                    router_logits = self._router_logits(hidden_states)
                    router_issued = True
                with torch.cuda.stream(self.latent_down_stream):
                    self.moe_latent_fork_event.wait()
                    early_routed_in = self._fused_latent_input(hidden_states)
                    if early_routed_in is None:
                        early_routed_in = self._routed_projection(
                            hidden_states, self.routed_expert_down_proj
                        )
                    self.moe_latent_ready_event.record()
                # No record_stream (under graph capture it holds every layer's
                # blocks until capture ends): both tensors are read on main only
                # after the join, and later work on this stream forks from main.
            if not router_issued:
                router_logits = self._router_logits(hidden_states)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None
        overlap_shared_allreduce = (
            self._defer_shared_allreduce
            and do_multi_stream()
            and self.shared_expert_stream is not None
        )
        num_tokens = hidden_states.shape[0]

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ=1 restores nn.Linear
            # (A/B escape hatch). When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a single fused GEMM (fp8_swap_ab_gemm).
            logits = router_logits
            moe_kwargs = {}
            if projected_input is not None:
                routed_in = projected_input
            elif early_routed_in is not None:
                routed_in = early_routed_in
                # The routed experts join the event before their first read of x.
                moe_kwargs = {"input_ready_event": self.moe_latent_ready_event}
            else:
                routed_in = self._fused_latent_input(hidden_states)
                if routed_in is None:
                    routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            if isinstance(routed_in, Fp4QuantizedTensor):
                moe_kwargs["output_dtype"] = hidden_states.dtype
            if isinstance(self.routed_experts, K3RoutedExperts):
                # Routed by the CuTe tail (route_target), the sorted routing kernel or
                # noaux_tc_op; then the kernels.
                y = self.routed_experts(
                    routed_in,
                    logits,
                    route=self._router(route_target),
                    input_ready_event=moe_kwargs.get("input_ready_event"),
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            else:
                y = self.routed_experts(
                    routed_in,
                    logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    **moe_kwargs,
                )
            if overlap_shared_allreduce:
                self.moe_routed_event.record()
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output():
            """The shared expert's (reduced) output."""
            final_params = (
                AllReduceParams(enable_allreduce=False) if self._defer_shared_allreduce else None
            )
            return _shared_reduce(
                self.shared_experts(identity, final_all_reduce_params=final_params)
            )

        def _shared_reduce(shared_out: torch.Tensor) -> torch.Tensor:
            """Reduce the shared expert's down-projection output."""
            if overlap_shared_allreduce:
                # Wait for all routed chunks, but overlap the norm/up tail.
                self.moe_routed_event.wait()
                shared_out = self._shared_all_reduce()(shared_out)
            return shared_out

        if rp_target is not None:
            return self._rp_tail_forward(
                _routed_output, identity, residual, rp_target, defer_residual
            )
        # The final join includes the shared all-reduce, keeping it ordered before any outer
        # routed reduction or the next layer's collectives.
        if (
            self._latent_ar_ws is not None
            and moe_all_reduce is not None
            and not self._defer_shared_allreduce
            and num_tokens <= self._latent_ar_ws["max_tokens"]
            and self.shared_expert_stream is not None
            and do_multi_stream()
        ):
            # The fork / join of fork_join, with the join left open until it is
            # known whether the latent all-reduce runs on its own workspace.
            self.moe_main_event.record()
            routed_out = _routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_out = _shared_output()
                self.moe_shared_event.record()
            if routed_out.dtype == torch.bfloat16:
                return self._own_workspace_latent_tail(
                    routed_out, shared_out, residual, defer_residual
                )
            self.moe_shared_event.wait()
        else:
            # Shared experts depend only on the block input, so overlap their GEMMs
            # with the routed dispatch/expert/combine chain. Multi-stream engages
            # only under CUDA graphs; otherwise both branches run in order on the
            # default stream. The shared expert includes its output all-reduce on
            # the auxiliary stream (deferred past the routed MoE for fused-comm
            # MegaMoE). The final join includes the shared all-reduce, so it
            # precedes the routed all-reduce: concurrent collectives on different
            # streams can corrupt SYMM_MEM all-reduce state.
            routed_out, shared_out = fork_join(
                _routed_output,
                _shared_output,
                self.moe_main_event,
                self.moe_shared_event,
                self.shared_expert_stream,
            )
        if self._defer_shared_allreduce and not overlap_shared_allreduce:
            shared_out = self._shared_all_reduce()(shared_out)
        if self._reduce_routed_output:
            # RMSNorm folded into the routed all-reduce (RMS_NORM fusion): the
            # MNNVL kernel normalizes straight out of the lamport buffer, so
            # the reduced row never round-trips through HBM for a norm kernel.
            # Strategies without the fusion fall back inside AllReduce.
            routed_latent = moe_all_reduce(
                routed_out,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM,
                    norm_weight=self.routed_expert_norm.weight,
                    eps=float(self.routed_expert_norm.variance_epsilon),
                ),
            )
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            return k3_moe_residual_add_op(routed_out, shared_out, residual)
        return routed_out + shared_out
