# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107): the MoE block.

``K3MoERuntime`` runs one layer's latent MoE: the router (``K3MoEGate``), the latent
down projection, the routed experts (the CuTe DSL kernel chain of ``moe.py``,
the MegaMoE module, or the small-batch kernels), the latent all-reduce,
``routed_expert_norm`` / ``routed_expert_up_proj``, and the shared experts
(``_K3SharedExpert``) on an auxiliary stream. Its decode tails leave the residual update to
the next layer's attention-residual mix (``attn_res.py``).

Also here: the routed-expert checkpoint specs (key families and the loader of each
quantization), the workspaces the latent all-reduce and the column-parallel tail share
across layers, and the MoE feature constants, read from ``features.active()`` at import.
"""

from __future__ import annotations

import copy
import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

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

from ..ops.comm.k3_lamport_push import k3_lamport_push
from ..ops.comm.k3_latent_ar_rmsnorm import k3_latent_ar_rmsnorm
from ..ops.gemm.k3_fp8w_lowm_gemm_push import k3_fp8w_lowm_gemm_push as k3_fp8w_lowm_gemm_push_op
from ..ops.gemm.k3_fp8w_lowm_mxfp8_push import k3_fp8w_lowm_mxfp8_push as k3_fp8w_lowm_mxfp8_push_op
from ..ops.gemm.k3_fp8w_mid_gemm import k3_fp8w_mid_gemm as k3_fp8w_mid_gemm_op
from ..ops.gemm.k3_latent_down_fp4 import k3_latent_down_fp4 as k3_latent_down_fp4_op
from ..ops.moe.k3_moe_residual_add import k3_moe_residual_add as k3_moe_residual_add_op
from ..ops.moe.k3_moe_tail1 import k3_moe_tail1 as k3_moe_tail1_op
from ..ops.moe.k3_route_tail import k3_route_tail as k3_route_tail_op
from ..ops.moe.k3_router_logits import k3_router_logits as k3_router_logits_op
from ..ops.moe.k3_shared_persist import k3_shared_persist as k3_shared_persist_op
from ..ops.moe.k3_ull_front import k3_ull_front as k3_ull_front_op
from .attn_res import _K3_ULL_TAIL_SANDWICH, _DeferredMoEResidual, _DeferredRpTail
from .features import active as _features
from .fp8_weight_read import (
    _K3_FP8W_LOWM_GEMM,
    _K3_LD_FUSED,
    _Fp8BlockScaleWeightReadLinear,
    _k3_lowm_gemm_can_run,
)
from .layers import (
    Bf16Linear,
    GatedMlp,
    LamportPending,
    LamportWorkspaceView,
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
from .moe import K3Nvfp4Experts, K3RoutedExperts, PublishedSortTables, SortedRouting, SortTables
from .text_config import _is_kda_layer, _resolve_kimi_situ_betas

_F = _features()

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

# c240: SM-count target of the shared expert's cuBLAS GEMMs (0: off). FC12's persistent grid needs
# whole SMs, so shared-expert GEMMs still resident when routing finishes delay its launch; with
# TRTLLM_K3_SHARED_SM=N they run on N SMs (torch SM carveout, captured into CUDA graphs) and the
# routed experts' backend keeps N SMs free of its persistent grids
# (``reserve_sms_for_concurrent_kernels``, K3MoERuntime).
_K3_SHARED_SM = _F.shared_sm
# Two-domain devices: die-major expert weights. Not available: the routed experts keep the loaded
# layout (``_k3_routed_backend_opt_ins`` logs it once).
_K3_FC12_DIE_MAJOR = _F.fc12_die_major


@contextmanager
def _k3_shared_sm_scope():
    """Run the enclosed cuBLAS GEMMs on ``TRTLLM_K3_SHARED_SM`` SMs (no-op when 0)."""
    if _K3_SHARED_SM <= 0:
        yield
        return
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    prev = torch._C._get_sm_carveout_experimental()
    torch._C._set_sm_carveout_experimental(max(0, num_sms - _K3_SHARED_SM))
    try:
        yield
    finally:
        torch._C._set_sm_carveout_experimental(prev)


# Small-batch MoE front end (b1-b4 decode), bit-identical to the path it replaces:
#   TRTLLM_K3_ROUTER_EXACT=kda: in the MoE blocks of KDA layers, the router logits of M <= 16
#     tokens come from k3_router_exact (bit-identical to dsv3_router_gemm_op; its weight is copied
#     into shared memory before griddepcontrol.wait). The faster router also starts the route tail
#     earlier, inside the burst of latent_down and the shared gate_up; after an MLA attention block
#     the route tail then stretches by ~3 us (b2 gen-only, GB300) and the routing chain gets longer
#     than with dsv3_router_gemm_op, while after a KDA attention block the front gets shorter.
_K3_ROUTER_EXACT_KDA = _F.router_exact_kda
# TRTLLM_K3_LATENT_AR_OWN_WS=1: for up to _K3_LATENT_AR_OWN_WS_MAX_M tokens, the routed latent
# all-reduce + RMSNorm (the same MNNVL Lamport kernels) runs on a workspace of its own instead of the
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
_K3_LATENT_AR_OWN_WS = _F.latent_ar_own_ws
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


def _router_exact_for_layer(cfg, layer_idx: int) -> bool:
    """Whether ``TRTLLM_K3_ROUTER_EXACT`` serves the router of this layer's MoE block."""
    return _K3_ROUTER_EXACT_KDA and _is_kda_layer(cfg, layer_idx)


# TRTLLM_K3_ULL2_COLL_CP_TAIL=1 (BF16 shared expert): the MoE tail of a
# generation-only forward splits latent_up by output columns instead of by latent columns. After
# the latent all-reduce + RMSNorm (k3_latent_ar_norm's row kernel with its MXFP8 output, as
# TRTLLM_K3_ULL_TAIL_SANDWICH) every rank computes its whole 128-row blocks of latent_up's output
# over the full latent (the MXFP8 low-M GEMM with the replicated projection's tactic) and multicasts
# them into the all-gather rows of the next call on a Lamport workspace of the tail's own
# (k3_lamport_ar "ag"); the shared expert's BF16 partial is pushed into the same call's slots on
# the shared stream as soon as it is ready. The next layer's pre-attention mix finishes the call
# (k3_attn_res ``finish_ag``): bf16(latent_up + bf16(all-reduced shared)), the replicated tail's
# bf16(routed + shared), so the result is bitwise the replicated tail (latent_up on every rank +
# the shared expert's MNNVL all-reduce) while each rank reads 1 / W of latent_up as the
# row-parallel tail does. Compared with the row-parallel tail, the critical path after the latent
# all-reduce carries 1 / W of the bytes (W x M x 7168 x 2 B of slots land on every rank, but the
# shared expert's land while the experts still run) and no two-shot step above the one-shot size.
# The latent all-reduce runs one-shot on a workspace of the tail's own, up to the C++ one-shot size
# (the sizes where the C++ all-reduce is one-shot too, so the sums keep its order); larger sizes keep
# the row-parallel tail. The mix polls the call before its griddepcontrol.wait (``early_poll``).
# The tail's latent all-reduce row kernel reads its workspace's rotation state, counts itself and
# re-arms the previous call's buffer before its griddepcontrol.wait (k3_latent_ar_norm
# ``early_state``: the previous call on that workspace, the previous MoE layer's, completed long
# before the routed experts launched). Same bits.
# With the routed experts' output push, the tail of a generation-only forward of at most 8 rows is
# one kernel (k3_moe_tail1). The routed experts (CUTEDSL_FC12's small-batch kernel) push this
# rank's latent partial rows, combined across experts with BF16 vector reds (whose order, and so
# the rows' bits, may change from run to run), into the slots of the latent
# workspace (``output_push``); k3_moe_tail1 polls them, sums them in the latent all-reduce's order,
# applies the RMSNorm after the products, multiplies by this rank's HIDDEN / W rows of latent_up and
# multicasts them into the column-parallel tail call's all-gather rows, which the next
# pre-attention mix finishes as before (``finish_ag``). It replaces the latent all-reduce row kernel
# and the column-parallel GEMM. Not bitwise (BF16 activation instead of MXFP8, the RMS scale applied
# to the FP32 products).
# On the column-parallel tail path of at most 8 rows, the shared expert (gate/up, SiTU, down and
# the push of its partial into the tail call's slots) is one persistent kernel on
# _K3_ULL_SHARED_PERSIST_CTAS CTAs (k3_shared_persist) on the shared stream, beside the routed
# experts, whose backend leaves it that many SMs (``reserve_sms_for_concurrent_kernels``).
# The next pre-attention mix reads the slots as before. Not bitwise (BF16 GEMM products in another
# summation order, tanh / sigmoid with ex2.approx). Needs the BF16 GatedMlp shard with SiTU on the
# tail's TP group; the weights are copied once into the kernel's stream order (16.5 MB per MoE layer
# and rank at TP16).
_K3_ULL2_COLL_CP_TAIL = _F.ull2_coll_cp_tail
_K3_ULL_SHARED_PERSIST_CTAS = 16

# The shared expert's part of the forward was pushed by the persistent kernel (no partial).
_SHARED_PUSHED = object()


# The column-parallel tail's Lamport workspaces, by (use, all-reduce mapping), the same workspaces
# as the routed experts' ``output_push`` takes them, and the multicast handles that own their memory.
_K3_CP_TAIL_WORKSPACES: Dict[Tuple[str, Mapping], Any] = {}
_K3_CP_TAIL_PUSH_VIEWS: Dict[Tuple[str, Mapping], LamportWorkspaceView] = {}
_K3_CP_TAIL_HANDLES: Dict[Tuple[str, Mapping], Any] = {}


def _k3_cp_tail_workspace(mapping: Mapping, hidden_size: int, max_tokens: int, use: str = "tail"):
    """``k3_lamport_ar.LamportWorkspace`` of the column-parallel MoE tail
    (``TRTLLM_K3_ULL2_COLL_CP_TAIL``) and the rotation state. ``use="tail"``: three Lamport buffers
    of ``max_tokens x (W + 1)`` BF16 rows of ``hidden_size`` (the one-shot slots plus the
    all-gather rows); ``"latent"``: ``max_tokens x W`` one-shot rows.
    Collective over the all-reduce group: every rank builds it at the same point of model
    construction. ``_K3_CP_TAIL_HANDLES[(use, mapping)]`` holds the multicast buffer."""
    workspace = _K3_CP_TAIL_WORKSPACES.get((use, mapping))
    if workspace is not None:
        return workspace
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    from ..kernels.k3r_comm.k3_lamport_ar import LamportWorkspace, required_bytes

    comm = _k3_mnnvl_mpi_comm(get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16))
    need = required_bytes(max_tokens, hidden_size, mapping.tp_size, True, ag=use == "tail")
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
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    flags = torch.tensor(
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    peers = [
        handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(mapping.tp_size)
    ]
    workspace = LamportWorkspace(
        world=mapping.tp_size,
        rank=mapping.tp_rank,
        uc_local=buffer.data_ptr(),
        mc_base=handle.get_mc_buffer((1,), torch.uint8, 0).data_ptr(),
        peer_ptrs=torch.tensor(peers, dtype=torch.int64, device=buffer.device),
        flags=flags,
        bytes_per_buffer=buffer_bytes,
    )
    _K3_CP_TAIL_WORKSPACES[(use, mapping)] = workspace
    _K3_CP_TAIL_PUSH_VIEWS[(use, mapping)] = LamportWorkspaceView(
        world=workspace.world,
        rank=workspace.rank,
        buffer=buffer,
        mc_base=workspace.mc_base,
        peer_ptrs=workspace.peer_ptrs,
        flags=flags,
        bytes_per_buffer=buffer_bytes,
    )
    _K3_CP_TAIL_HANDLES[(use, mapping)] = handle
    return workspace


# Row-parallel MoE tail for generation-only batches of up to _K3_RP_TAIL_MAX_M tokens (the low-M
# GEMM's range), where the routed latent is all-reduced by the model (MoE TP/EP over the all-reduce
# group, TP-sharded shared expert) and the layer's residual update is fused
# (TRTLLM_K3_FUSE_MOE_RESIDUAL). After the latent all-reduce + RMSNorm, every rank runs one low-M
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
# TRTLLM_K3_ULL_FRONT=1: MoE forwards of 1 .. _K3_ULL_FRONT_MAX_M (<= 32) rows run the MoE
# front of cute_dsl_kernels/blackwell/k3_ull_front.py on the routed all-reduce's TP group, two kernels
# on the main stream: each rank computes the router logits of every expert (k3_router_exact's
# arithmetic) and the latent_down columns of its 128-column tiles (the low-M GEMM of
# k3_latent_down_fp4) into an MNNVL workspace (the latent is multicast to every rank); a routing-tail
# cluster polls the logits and writes the top-16 and this rank's moe_sort tables, and gather CTAs
# write the NVFP4 latent. This replaces the router GEMM, the routing tail, the replicated latent_down
# + quantization and the latent stream's fork / join.
# Routing is bitwise (per 16-row tile of the router); the latent is bitwise up to 16 rows (above,
# k3_latent_down_fp4 runs the mid-M kernel, split 4, and the front the low-M kernel, split 4).
# The routing reaches the routed experts as ``routing``: where they offer a
# routing word for the call (``published_routing_word``: the small-batch kernel on MoE-TP slices),
# the route kernel publishes each token's top-k into it instead of writing the moe_sort tables and
# the experts build their tables from it while the route kernel still runs
# (``PublishedSortTables``; same tables, same bits); otherwise the front's tile-128 moe_sort tables
# (``SortTables``) replace the experts' own moe_sort.
# The A_mid that finishes the pushed o_proj all-reduce releases the front's head at kernel entry
# (k3_attn_res ``entry_trigger``) where the head follows it, so the head's CTAs become resident
# while A_mid polls the o_proj all-reduce and stream the router and latent_down weights into
# shared memory before their griddepcontrol.wait; after the wait only the activation loads, the
# MMAs and the pushes remain. Before its wait the head reads only weights and writes nothing, and
# it releases the route kernel only after its wait. The early-polling A_mid otherwise releases it
# after its stores. Same bits.
# Forwards of up to _K3_ULL_FRONT_FORK_AT_HEAD_MAX_M rows fork the shared expert after the head
# (between the head and the route kernel): under CUDA graph capture the shared GEMMs become
# programmatic dependents of the kernel the fork follows, so a fork before the head lets the
# entry-releasing A_mid release them with the head, and their CTAs take the SMs the head's CTAs
# need while A_mid polls (helix16 n1 / n2: -3.4 / -2.6% per step forking at the head vs -0.3 /
# -0.8% before it; at 24 / 32 rows forking before the head stays ahead, -1.5 / -1.3% vs -1.0 /
# +0.1%, where the larger gate_up competes with the routed experts instead).
_K3_ULL_FRONT = _F.ull_front
_K3_ULL_FRONT_MAX_M = 32
_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M = 16


def _ull_shared_fork(lead: bool, rows: int) -> Optional[str]:
    """Where the ULL front forks the shared expert's GEMMs on ``rows`` rows (``None``: no fork, ``lead`` is
    ``_shared_gemms_can_lead()``).

    * ``"head"`` (up to ``_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M`` rows): the event is recorded after the head, so
      the shared GEMMs depend on the head.
    * ``"after_head"`` (more rows): the event is recorded before the head, so the shared GEMMs depend on what
      precedes the head (the entry-releasing A_mid).

    The shared GEMMs themselves are issued after the front (``K3MoERuntime._ull_fork_shared``).
    """
    if not lead:
        return None
    return "head" if rows <= _K3_ULL_FRONT_FORK_AT_HEAD_MAX_M else "after_head"


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


def _k3_routed_backend_opt_ins(routed_experts, column_parallel_tail: bool) -> None:
    """The routed experts' opt-ins of a MoE layer, made once at construction.

    SMs the FC12 kernels' persistent grids leave to the kernels the shared stream runs beside them
    (the shared expert's GEMMs confined to ``_K3_SHARED_SM`` SMs; with the column-parallel tail,
    the persistent shared expert's ``_K3_ULL_SHARED_PERSIST_CTAS`` CTAs). The CUTEDSL kernels take
    no reserve.
    """
    if _K3_FC12_DIE_MAJOR:
        logger.info_once(
            "TRTLLM_K3_FC12_DIE_LAYOUT: die-major expert weights are not available; the routed "
            "experts keep the loaded layout",
            key="k3_fc12_die_major_off",
        )
    if getattr(routed_experts, "kernels", None) != "CUTEDSL_FC12":
        return
    reserve = _K3_SHARED_SM
    if column_parallel_tail:
        reserve = max(reserve, _K3_ULL_SHARED_PERSIST_CTAS)
    if reserve > 0:
        routed_experts.sm_reserve = reserve


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


class _CpTailOperands(NamedTuple):
    """One rank's operands of the column-parallel MoE tail (``TRTLLM_K3_ULL2_COLL_CP_TAIL``): the
    ``n`` output rows of latent_up from row ``c0`` (whole 128-row blocks; their FP8 codes read in
    place) and the DeepGEMM-packed UE8M0 scale of that ``[n, latent]`` slice."""

    c0: int
    n: int
    weight: torch.Tensor
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


class _NoSmallBatch:
    """The small-batch path of routed experts that have none."""

    max_tokens = 0
    counters = None

    @staticmethod
    def applies(num_tokens: int) -> bool:
        return False

    @staticmethod
    def published_routing_word(num_tokens: int) -> None:
        return None

    @staticmethod
    def post_load_weights() -> None:
        pass


_NO_SMALL_BATCH = _NoSmallBatch()


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

    # TRTLLM_K3_LATENT_AR_OWN_WS: the routed latent all-reduce's own MNNVL workspace (``None``: it
    # shares the one of the shared expert's all-reduce).
    _latent_ar_ws: Optional[Dict[str, Any]] = None
    # TRTLLM_K3_ULL2_COLL_CP_TAIL: the column-parallel tail's workspace and this rank's operands.
    _cp_ws: Optional[Any] = None
    _cp_latent_ws: Optional[Any] = None
    # The latent workspace as the routed experts' ``output_push`` (k3_moe_tail1).
    _cp_latent_push: Optional[LamportWorkspaceView] = None
    _cp_tail: Optional[_CpTailOperands] = None
    # k3_moe_tail1 (column-parallel tail): latent_up's UE8M0 block exponents
    # [HIDDEN / 128, LATENT / 128] (None: not built).
    _tail1_wexp: Optional[torch.Tensor] = None
    # k3_shared_persist (column-parallel tail): the shared expert's weights in the kernel's stream
    # order, its exchange workspace and SiTU's (beta, linear beta) (None: not built).
    _persist_frags: Optional[torch.Tensor] = None
    _persist_ws: Optional[Any] = None
    _persist_act: Optional[Tuple[float, Optional[float]]] = None
    # TRTLLM_K3_ROUTER_EXACT: whether this layer's router logits come from k3_router_exact (set
    # per layer in ``__init__``; off unless the switch selects the layer).
    _router_exact = False
    # The row-parallel tail: this rank's tail GEMM operands, built after loading (``None``: not
    # applicable to this layer).
    _rp_tail: Optional[_RpTailOperands] = None
    # TRTLLM_K3_ULL_FRONT: the front's MNNVL workspace of the routed all-reduce group (built at
    # construction) and this layer's front operands on this rank (built after loading).
    _ull_ws = None
    _ull_plan = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self._router_exact = _router_exact_for_layer(cfg, layer_idx)
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
        self.small_experts = getattr(self.routed_experts, "small_batch", None) or _NO_SMALL_BATCH
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
        if _K3_SHARED_SM > 0:
            shared_forward = self.shared_experts.forward

            def _shared_forward_confined(*args, **kwargs):
                with _k3_shared_sm_scope():
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
        # The shared expert's GEMMs issued ahead of the routed experts are done.
        self.moe_shared_gemm_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # FlashInfer RMSNorm on the reduced latent (or folded into the routed all-reduce).
        self.routed_expert_norm = RMSNormWeight(self.moe_hidden_size, cfg.rms_norm_eps, dtype=dtype)
        if _K3_LATENT_AR_OWN_WS and self._reduce_routed_output:
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                self._latent_ar_ws = _k3_latent_ar_workspace(
                    mnnvl.mapping, mnnvl.dtype, self.moe_hidden_size, _K3_LATENT_AR_OWN_WS_MAX_M
                )
        if _K3_ULL_FRONT and self._reduce_routed_output:
            from ..kernels.k3r_moe import k3_ull_front

            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and k3_ull_front.supports_world(mnnvl.mapping.tp_size):
                # Collective over the all-reduce group: every rank builds it here.
                self._ull_ws = k3_ull_front.ull_front_workspace(mnnvl.mapping)
        if _K3_ULL2_COLL_CP_TAIL and self._reduce_routed_output:
            # TRTLLM_K3_ULL2_COLL_CP_TAIL: the column-parallel tail's workspaces (collective).
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                world = mnnvl.mapping.tp_size
                row_bytes = self.moe_hidden_size * world * 2
                rows = min(_K3_RP_TAIL_MAX_M, self._cp_latent_oneshot_bytes() // row_bytes)
                if rows >= 1:
                    self._cp_ws = _k3_cp_tail_workspace(
                        mnnvl.mapping, self.hidden_size, _K3_RP_TAIL_MAX_M
                    )
                    self._cp_latent_ws = _k3_cp_tail_workspace(
                        mnnvl.mapping, self.moe_hidden_size, rows, use="latent"
                    )
                    self._cp_latent_push = _K3_CP_TAIL_PUSH_VIEWS[("latent", mnnvl.mapping)]
        _k3_routed_backend_opt_ins(self.routed_experts, self._cp_ws is not None)

    def _lu_mid_prequant_plan(self, latent: torch.Tensor):
        """``(tactic, sf_blocks)`` of the mid-M path that
        latent_up's :meth:`_Fp8BlockScaleWeightReadLinear.forward` runs on the BF16 rows ``latent``
        (SM107: the K3 low-M GEMM on ``lowm_weight_scale`` above the low-M cut-over),
        or ``None`` where it runs another GEMM."""
        up = self.routed_expert_up_proj
        if (
            not _K3_FP8W_LOWM_GEMM
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
        """FP32 router logits; ``TRTLLM_K3_ROUTER_EXACT`` serves ``M <= 16`` BF16 tokens with
        ``k3_router_exact`` (bit-identical to the ``dsv3_router_gemm_op`` it replaces) in the
        layers it selects."""
        if self._router_exact:
            from ..kernels.k3r_moe import k3_router_exact

            x = hidden_states.reshape(-1, self.gate.gating_dim)
            if k3_router_exact.can_run(x, self.gate.weight):
                return k3_router_logits_op(x, self.gate.weight)
        return self.gate.compute_logits(hidden_states)

    def _shared_gemms_can_lead(self) -> bool:
        """Whether this forward can issue the shared expert's GEMMs ahead of the routed experts
        (``_ull_fork_shared``) and its all-reduce separately."""
        return (
            self.shared_expert_stream is not None
            and self.moe_shared_gemm_event is not None
            and not self._defer_shared_allreduce
            and not getattr(self.shared_experts.down_proj, "use_fused_gemm_allreduce", False)
            and do_multi_stream()
        )

    def _shared_needs_all_reduce(self) -> bool:
        """Whether the shared expert's output is a row-parallel partial to all-reduce."""
        return bool(getattr(self.shared_experts.down_proj, "reduce_output", False))

    def _shared_all_reduce(self) -> TpAllReduce:
        """The shared expert's row-parallel reduction."""
        return self.shared_experts.down_proj.all_reduce

    def post_load_weights(self) -> None:
        self.small_experts.post_load_weights()
        self.prepare_rp_tail()
        self.prepare_ull_front()

    def cache_derived_state(self) -> None:
        self.small_experts.post_load_weights()
        self.prepare_rp_tail()
        self.prepare_ull_front()

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

    def _ull_front_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the sharded MoE front (``TRTLLM_K3_ULL_FRONT``), or
        ``None``."""
        if self._ull_ws is None:
            return "no MNNVL routed all-reduce over a supported TP group"
        moe = self.routed_experts
        if not isinstance(moe, K3RoutedExperts) or self._attention_dp:
            return "the routed experts are not the CuTe DSL kernels of a TP group"
        if moe.backend.num_slots != self.gate.num_experts:
            return "the routed experts have no static contiguous expert slice"
        down = self.routed_expert_down_proj
        if (
            not isinstance(down, _Fp8BlockScaleWeightReadLinear)
            or down._fused_projection is not None
        ):
            return "latent_down is not an FP8 weight-read projection"
        if self.gate.weight.dtype != torch.bfloat16:
            return "the router weight is not BF16"
        return None

    def prepare_ull_front(self) -> bool:
        """``TRTLLM_K3_ULL_FRONT``: build this layer's front operands on this rank (views of the
        router and latent_down weights, the packed scale of the rank's latent_down tiles). Rebuilt
        on every call (post-load), before CUDA graph capture."""
        self._ull_plan = None
        if not _K3_ULL_FRONT:
            return False
        reason = self._ull_front_unsupported()
        if reason is not None:
            logger.info_once(
                f"TRTLLM_K3_ULL_FRONT: MoE layers keep the replicated front: {reason}",
                key=f"k3_ull_front_off_{reason}",
            )
            return False
        from ..kernels.k3r_moe import k3_ull_front

        down = self.routed_expert_down_proj
        down.transform_weights()
        backend = self.routed_experts.backend
        ws = self._ull_ws
        with torch.no_grad():
            self._ull_plan = k3_ull_front.make_plan(
                down.weight,
                down.weight_scale,
                self.gate.weight,
                self.gate.e_score_correction_bias,
                float(self.gate.routed_scaling_factor),
                backend.fc31_input_scale,
                ws.world,
                ws.rank,
                backend.slot_start,
                backend.expert_size_per_partition,
            )
        logger.info_once(
            f"TRTLLM_K3_ULL_FRONT: sharded MoE front over {ws.world} ranks, routed experts "
            f"[{backend.slot_start}, {backend.slot_start + backend.expert_size_per_partition}) "
            f"on rank {ws.rank}",
            key="k3_ull_front_on",
        )
        return True

    def _ull_front_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether this forward runs the sharded MoE front."""
        plan = self._ull_plan
        if plan is None:
            return False
        if not 1 <= hidden_states.shape[0] <= min(_K3_ULL_FRONT_MAX_M, 32):
            return False
        from ..kernels.k3r_moe import k3_ull_front

        x = hidden_states.reshape(-1, self.hidden_size)
        if not k3_ull_front.can_run(x, plan):
            return False
        # The kernels compile on their first call, which must not fall into a CUDA graph capture.
        return not torch.cuda.is_current_stream_capturing() or k3_ull_front.is_compiled(
            x.shape[0], plan, self._ull_publish_word(x.shape[0]) is not None
        )

    def _ull_publish_word(self, m: int) -> Optional[torch.Tensor]:
        """The routed experts' routing word into which the front of an ``m``-row forward publishes
        the top-k (``K3SmallBatchExperts.published_routing_word``), or ``None`` where the front
        writes the ``moe_sort`` tables."""
        if not getattr(self.routed_experts, "consumes_sort_tables", False):
            return None
        return self.small_experts.published_routing_word(m)

    def ull_front_releases_early(self, like: torch.Tensor) -> bool:
        """Whether the mix that produces this forward's input may release the front's head at
        kernel entry, i.e. whether the next forward on rows shaped like ``like`` (a contiguous BF16
        ``[M, hidden]`` tensor, e.g. the prefix sum the mix updates) runs the sharded front, whose
        head is then the next launch on the stream. Conservative: ``False`` where ``like`` cannot
        stand for the mix's output."""
        if like.dim() != 2 or not like.is_contiguous():
            return False
        return self._ull_front_applies(like)

    def _ull_front(self, hidden_states: torch.Tensor, after_head=None):
        """Run the sharded MoE front: returns ``(router_logits placeholder, NVFP4 routed input,
        routing)``, ``routing`` the ``SortedRouting`` the routed experts take (with this rank's
        ``moe_sort`` tables, the published top-k, or no tables); ``after_head`` runs between the
        front's two launches."""
        from ..kernels.k3r_moe import k3_ull_front

        plan = self._ull_plan
        x = hidden_states.reshape(-1, self.hidden_size)
        if not x.is_contiguous():
            x = x.contiguous()
        moe = self.routed_experts
        backend = moe.backend
        # The global scale can be replaced after the plan was built (calibration, reload).
        plan.global_sf = backend.fc31_input_scale.reshape(-1)[:1]
        # Where the routed experts offer a routing word, the route kernel publishes each token's
        # top-k into it instead of writing the moe_sort tables; the experts build their tables.
        word = self._ull_publish_word(x.shape[0])
        routing, codes, scales = k3_ull_front_op(
            x,
            plan,
            self._ull_ws,
            after_head=after_head,
            publish=word.data_ptr() if word is not None else 0,
        )
        if word is not None:
            sorted_routing = SortedRouting(routing[0], routing[1], PublishedSortTables())
        elif moe.consumes_sort_tables and self.small_experts.applies(x.shape[0]):
            sorted_routing = SortedRouting(
                routing[0], routing[1], SortTables(k3_ull_front.TILE, tuple(routing[2:8]))
            )
        else:
            sorted_routing = SortedRouting(routing[0], routing[1], None)
        # The routing does not read the logits; they only carry the shape.
        logits = torch.empty(
            (x.shape[0], self.gate.num_experts), dtype=torch.float32, device=x.device
        )
        return logits, Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False), sorted_routing

    def _ull_fork_shared(
        self, identity: torch.Tensor, persist: bool, rows: int
    ) -> Tuple[Optional[Callable[[], None]], Optional[Callable[[], None]], List[torch.Tensor]]:
        """Fork the shared expert's GEMMs for a ULL front on ``rows`` rows where ``_ull_shared_fork`` puts it.

        Returns ``(after_head, after_front, forked)``: the callback the front runs between its head and route
        launches, the one the caller runs once the front is issued (``None``: nothing to run) and the list that
        holds the shared partial once the GEMMs are issued.

        The shared GEMMs are captured after the whole front, whichever kernel they depend on: each
        early-releasing kernel's first dependent in the CUDA graph then is the one its trigger is meant for
        (A_mid's the head, the head's the route kernel), and the executor keeps that one on the releasing
        kernel's stream. A shared gate_up captured before it would take that place whenever no
        other edge pins it to the shared stream, and the intended dependent would wait for the releasing kernel
        to finish.
        """
        forked: List[torch.Tensor] = []

        def run_shared():
            forked.append(self._ull_shared_first(identity, persist))

        fork = _ull_shared_fork(self._shared_gemms_can_lead(), rows)
        if fork == "head":
            return self.moe_main_event.record, run_shared, forked
        if fork == "after_head":
            self.moe_main_event.record()
            return None, run_shared, forked
        return None, None, forked

    def _ull_shared_first(self, identity: torch.Tensor, persist: bool = False) -> torch.Tensor:
        """The shared expert's GEMMs on the shared stream from ``moe_main_event`` (recorded around
        the sharded front's head, ``_ull_shared_fork``), without its all-reduce;
        ``moe_shared_gemm_event`` marks them done. ``persist`` (the column-parallel tail): the
        persistent kernel instead, which pushes the partial itself (returns ``_SHARED_PUSHED``)."""
        with torch.cuda.stream(self.shared_expert_stream):
            self.moe_main_event.wait()
            if persist:
                self._run_shared_persist(identity)
                partial = _SHARED_PUSHED
            else:
                partial = self.shared_experts(
                    identity, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
                )
            self.moe_shared_gemm_event.record()
        return partial

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
        self._cp_tail = None
        if self._cp_ws is not None:
            # TRTLLM_K3_ULL2_COLL_CP_TAIL: this rank's whole 128-row blocks of latent_up's output.
            c0, n = _rp_tail_latent_slice(hidden, mapping.tp_size, mapping.tp_rank)
            with torch.no_grad():
                cp_exps = k3_fp8w_lowm_gemm.block_exponents_of(
                    up.weight_scale, hidden, self.moe_hidden_size
                )[c0 // 128 : (c0 + n) // 128]
                cp_scale = k3_fp8w_lowm_gemm.pack_ue8m0_exponents(
                    cp_exps.contiguous(), n, self.moe_hidden_size
                )
            self._cp_tail = _CpTailOperands(c0, n, up.weight[c0 : c0 + n], cp_scale)
            logger.info_once(
                f"TRTLLM_K3_ULL2_COLL_CP_TAIL: column-parallel MoE tail, latent_up rows "
                f"[{c0}, {c0 + n}) (rank {mapping.tp_rank} of {mapping.tp_size})",
                key=f"k3_cp_tail_on_{n}",
            )
            self._prepare_tail1(up, hidden)
            self._prepare_shared_persist()
        logger.info_once(
            f"Kimi K3 row-parallel MoE tail, K = latent [{k0}, {k0 + k1}), BF16 shared "
            f"partial added (rank {mapping.tp_rank} of {mapping.tp_size})",
            key=f"k3_rp_tail_on_{k1}_{shard}",
        )
        return True

    def _prepare_tail1(self, up, hidden: int) -> None:
        """latent_up's block exponents for k3_moe_tail1 (the FP8 codes are read in place)."""
        self._tail1_wexp = None
        if self._cp_latent_push is None:
            return
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm
        from ..kernels.k3r_moe import k3_moe_tail1

        weight = up.weight
        if (
            weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (k3_moe_tail1.HIDDEN, k3_moe_tail1.LATENT)
            or hidden != k3_moe_tail1.HIDDEN
            or self.moe_hidden_size != k3_moe_tail1.LATENT
            or not weight.is_contiguous()
        ):
            logger.info_once(
                "k3_moe_tail1: latent_up is not the [7168, 3584] FP8 weight, tail unchanged",
                key="k3_tail1_off_weight",
            )
            return
        with torch.no_grad():
            wexp = k3_fp8w_lowm_gemm.block_exponents_of(
                up.weight_scale, hidden, self.moe_hidden_size
            ).contiguous()
        if int(wexp.min().item()) < 1 or int(wexp.max().item()) > 254:
            logger.info_once(
                "k3_moe_tail1: latent_up block scale outside the kernel's range, tail unchanged",
                key="k3_tail1_off_scale",
            )
            return
        self._tail1_wexp = wexp
        world, rank = self._cp_latent_ws.world, self._cp_latent_ws.rank
        c0, n = k3_moe_tail1.rank_rows(world, rank)
        logger.info_once(
            f"k3_moe_tail1: one-kernel MoE tail up to {k3_moe_tail1.MAX_M} rows, latent_up "
            f"rows [{c0}, {c0 + n}) (rank {rank} of {world})",
            key=f"k3_tail1_on_{n}",
        )

    def _prepare_shared_persist(self) -> None:
        """This rank's shared expert weights in k3_shared_persist's stream order and its
        workspace."""
        self._persist_frags = None
        if self._cp_ws is None:
            return
        from ..kernels.k3r_moe import k3_shared_persist

        ws = self._cp_ws
        mlp = self.shared_experts
        sms = _K3_ULL_SHARED_PERSIST_CTAS
        gate_up = getattr(getattr(mlp, "gate_up_proj", None), "weight", None)
        down = getattr(getattr(mlp, "down_proj", None), "weight", None)
        inter = k3_shared_persist.INTER // ws.world
        reason = None
        if getattr(mlp, "activation", None) != "situ":
            reason = "the shared expert's activation is not SiTU"
        elif (
            gate_up is None
            or down is None
            or gate_up.dtype != torch.bfloat16
            or down.dtype != torch.bfloat16
            or tuple(gate_up.shape) != (2 * inter, k3_shared_persist.HIDDEN)
            or tuple(down.shape) != (k3_shared_persist.HIDDEN, inter)
        ):
            reason = "the shared expert is not the BF16 [2 I, 7168] / [7168, I] shard"
        elif mlp.down_proj.tp_size != ws.world or mlp.down_proj.tp_rank != ws.rank:
            reason = "the shared expert's TP group is not the tail's"
        elif not k3_shared_persist.can_run(k3_shared_persist.MAX_M, ws.world, sms):
            reason = f"{sms} CTAs do not split the {inter}-column shard"
        if reason is not None:
            logger.info_once(
                f"k3_shared_persist: shared expert unchanged: {reason}",
                key=f"k3_persist_off_{reason}",
            )
            return
        with torch.no_grad():
            self._persist_frags = k3_shared_persist.persist_fragments(
                gate_up.data.contiguous(), down.data.contiguous(), sms
            )
        self._persist_ws = k3_shared_persist.PersistWorkspace(ws.world, gate_up.device)
        self._persist_act = (float(mlp.beta), mlp.linear_beta)
        logger.info_once(
            f"k3_shared_persist: the shared expert on {sms} CTAs beside the routed "
            f"experts up to {k3_shared_persist.MAX_M} rows (rank {ws.rank} of {ws.world})",
            key=f"k3_persist_on_{sms}",
        )

    def _persist_applies(self, num_tokens: int, target) -> bool:
        """Whether this forward's shared expert is the persistent kernel (the column-parallel
        tail)."""
        if (
            self._persist_frags is None
            or target is None
            or self.shared_expert_stream is None
            or not do_multi_stream()
            or not _K3_ULL2_COLL_CP_TAIL
            or not self._cp_tail_applies(num_tokens, target)
        ):
            return False
        from ..kernels.k3r_moe import k3_shared_persist

        return k3_shared_persist.can_run(num_tokens, self._cp_ws.world, _K3_ULL_SHARED_PERSIST_CTAS)

    def _run_shared_persist(self, identity: torch.Tensor) -> None:
        """The persistent shared expert of ``identity`` on the current (shared) stream, pushing
        into the tail call's slots."""
        ws = self._cp_ws
        x = identity if identity.is_contiguous() else identity.contiguous()
        beta, linear_beta = self._persist_act
        k3_shared_persist_op(
            x=x,
            frags=self._persist_frags,
            world=ws.world,
            rank=ws.rank,
            sms=_K3_ULL_SHARED_PERSIST_CTAS,
            ws=self._persist_ws,
            out_base=ws.mc_base,
            out_flags=ws.flags,
            beta=beta,
            linear_beta=linear_beta,
        )

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

    def _rp_tail_mx_applies(self, num_tokens: int) -> bool:
        """Whether the row-parallel tail GEMM can run as the MXFP8 low-M GEMM on the latent
        all-reduce's MXFP8 output (``TRTLLM_K3_ULL_TAIL_SANDWICH``; the tail GEMM has one source):
        the same tactic as the split-source BF16 GEMM, so the same bits."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        if num_tokens > k3_fp8w_lowm_mxfp8.MAX_M:
            return False
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(num_tokens, self.hidden_size, self._rp_tail.k1)
        except ValueError:
            return False
        return True

    def _rp_tail_latent_ar_early(
        self, routed_out: torch.Tensor, target, mxfp8: bool = False
    ) -> Optional[Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """``TRTLLM_K3_ULL_TAIL_SANDWICH``: the routed latent all-reduce + RMSNorm of the
        row-parallel tail by ``k3_latent_ar_norm`` (its row kernel where it serves the size, else
        the cluster kernel) on the MNNVL workspace the tail GEMM pushes into: bit-identical to the
        C++ one-shot kernel, but it releases the tail GEMM right after its own
        ``griddepcontrol.wait``, so the GEMM streams its weight slice while the all-reduce waits
        for the other ranks. Returns ``(latent, (xq, xs) with mxfp8 else None)``, or ``None``
        where it does not apply (the C++ kernel's two-shot sizes). ``mxfp8``: the kernel also
        stores the MXFP8 form of the latent, bitwise the low-M GEMM's own quantization of it."""
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

        from ..kernels.k3r_comm import k3_latent_ar_norm

        ws = target.workspace
        latent = routed_out.reshape(-1, self.moe_hidden_size)
        if not latent.is_contiguous():
            latent = latent.contiguous()
        gamma = self.routed_expert_norm.weight
        if latent.numel() * ws.world * latent.element_size() > _MNNVL_ONE_SHOT_THRESHOLD_BYTES:
            return None
        row = k3_latent_ar_norm.can_run_row(latent, gamma, ws.world, ws.bytes_per_buffer)
        if not (row or k3_latent_ar_norm.can_run(latent, gamma, ws.world)):
            return None
        out = torch.empty_like(latent)
        m, k = latent.shape
        latent_mx = None
        if mxfp8:
            latent_mx = (
                torch.empty((m, k), dtype=torch.float8_e4m3fn, device=latent.device),
                torch.empty((m, k // 32), dtype=torch.uint8, device=latent.device),
            )
        k3_latent_ar_rmsnorm(
            latent,
            gamma,
            float(self.routed_expert_norm.variance_epsilon),
            out,
            uc_addr=ws.uc_local,
            mc_addr=ws.mc_base,
            flags=ws.flags,
            rank=ws.rank,
            world_size=ws.world,
            mxfp8_out=latent_mx,
            row=row,
        )
        return out, latent_mx

    @staticmethod
    def _cp_latent_oneshot_bytes() -> int:
        """Largest one-shot latent all-reduce of the column-parallel tail (bytes of slots): the C++
        one-shot size."""
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

        return _MNNVL_ONE_SHOT_THRESHOLD_BYTES

    def _tail1_applies(self, num_tokens: int) -> bool:
        """Whether this forward's MoE tail runs k3_moe_tail1: its operands are built, the routed
        experts push their latent rows at ``num_tokens`` (``K3SmallBatchExperts.applies``: the
        small-batch kernel, no communication strategy) and the kernel serves the shape."""
        if (
            self._tail1_wexp is None
            or self._cp_latent_push is None
            or not self.small_experts.applies(num_tokens)
        ):
            return False
        from ..kernels.k3r_moe import k3_moe_tail1

        return k3_moe_tail1.can_run(num_tokens, self._cp_latent_push.world)

    def _cp_tail_applies(self, num_tokens: int, target) -> bool:
        """Whether this forward's MoE tail runs column-parallel (``TRTLLM_K3_ULL2_COLL_CP_TAIL``):
        the latent all-reduce runs one-shot by ``k3_latent_ar_norm``'s row kernel on the tail's
        latent workspace (its MXFP8 output feeds the tail GEMM), the replicated projection's MXFP8
        tactic serves ``num_tokens`` and the tail's workspaces hold them."""
        cp = self._cp_tail
        lws = self._cp_latent_ws
        if cp is None or lws is None or num_tokens > _K3_RP_TAIL_MAX_M:
            return False
        from ..kernels.k3r_comm import k3_latent_ar_norm
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        if (
            lws.world not in k3_latent_ar_norm.ROW_WORLD_SIZES
            or num_tokens > k3_latent_ar_norm.ROW_MAX_M
            or self.moe_hidden_size != k3_latent_ar_norm.HIDDEN
        ):
            return False
        nbytes = num_tokens * self.moe_hidden_size * lws.world * 2
        if nbytes > self._cp_latent_oneshot_bytes() or nbytes > lws.bytes_per_buffer:
            return False
        if num_tokens > k3_fp8w_lowm_mxfp8.MAX_M:
            return False
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(num_tokens, self.hidden_size, self.moe_hidden_size)
        except ValueError:
            return False
        return True

    def _cp_tail_forward(
        self,
        routed_output: Callable[..., torch.Tensor],
        identity: torch.Tensor,
        shared_part: Optional[torch.Tensor],
        residual: torch.Tensor,
        target,
        defer_residual: bool,
    ):
        """``TRTLLM_K3_ULL2_COLL_CP_TAIL``: the routed experts, the shared expert's BF16 partial
        pushed into the tail's call on the shared stream, the latent all-reduce + RMSNorm (MXFP8
        output), then this rank's latent_up output columns multicast into the call's all-gather
        rows; or, where k3_moe_tail1 applies, the routed experts push their latent rows
        (``routed_output(output_push=...)``) and one kernel does the rest. Returns a
        ``_DeferredRpTail`` whose pending call the next pre-attention mix finishes (``finish_ag``),
        or the updated residual."""
        from ..kernels.k3r_gemm.k3_fp8w_lowm_mxfp8 import default_mx_tactic

        cp = self._cp_tail
        cp_ws = self._cp_ws
        num_tokens = identity.shape[0]
        tail1 = self._tail1_applies(num_tokens)
        if tail1:
            routed_output = functools.partial(routed_output, output_push=self._cp_latent_push)

        def push_shared(part: torch.Tensor) -> None:
            part = part.reshape(num_tokens, -1)
            if not part.is_contiguous():
                part = part.contiguous()
            k3_lamport_push(cp_ws, part, True)

        side = self.shared_expert_stream is not None and do_multi_stream()
        if shared_part is not None:
            # The shared expert ran ahead on the shared stream (moe_shared_gemm_event); the
            # persistent kernel pushed its partial itself.
            routed_out = routed_output()
            if side:
                with torch.cuda.stream(self.shared_expert_stream):
                    self.moe_shared_gemm_event.wait()
                    if shared_part is not _SHARED_PUSHED:
                        push_shared(shared_part)
                    self.moe_shared_event.record()
            else:
                self.moe_shared_gemm_event.wait()
                push_shared(shared_part)
        elif side:
            self.moe_main_event.record()
            persist = self._persist_applies(num_tokens, target)
            routed_out = routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                if persist:
                    self._run_shared_persist(identity)
                else:
                    push_shared(self._rp_shared_output(identity))
                self.moe_shared_event.record()
        else:
            routed_out = routed_output()
            push_shared(self._rp_shared_output(identity))
        if tail1:
            # The whole tail in one kernel (the experts pushed their latent rows), into the call's
            # all-gather rows.
            from ..kernels.k3r_moe import k3_moe_tail1

            lws = self._cp_latent_ws
            c0, _ = k3_moe_tail1.rank_rows(lws.world, lws.rank)
            k3_moe_tail1_op(
                m=num_tokens,
                world=lws.world,
                lat_uc=lws.uc_local,
                lat_flags=lws.flags,
                weight=self.routed_expert_up_proj.weight,
                wexp=self._tail1_wexp,
                gamma=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
                ag_base=cp_ws.mc_base,
                ag_flags=cp_ws.flags,
                c0=c0,
            )
            return self._cp_tail_finish(cp_ws, num_tokens, residual, side, defer_residual)
        # The latent all-reduce + RMSNorm on the tail's latent workspace (one-shot, checked by
        # _cp_tail_applies), releasing the tail GEMM at once, with its MXFP8 output.
        latent = routed_out.reshape(-1, self.moe_hidden_size)
        if not latent.is_contiguous():
            latent = latent.contiguous()
        ws = self._cp_latent_ws
        latent_mx = (
            torch.empty(latent.shape, dtype=torch.float8_e4m3fn, device=latent.device),
            torch.empty(
                (num_tokens, self.moe_hidden_size // 32), dtype=torch.uint8, device=latent.device
            ),
        )
        k3_latent_ar_rmsnorm(
            latent,
            self.routed_expert_norm.weight,
            float(self.routed_expert_norm.variance_epsilon),
            torch.empty_like(latent),
            uc_addr=ws.uc_local,
            mc_addr=ws.mc_base,
            flags=ws.flags,
            rank=ws.rank,
            world_size=ws.world,
            mxfp8_out=latent_mx,
            early_state=True,
        )
        # The replicated projection's tactic: every 128-row output tile is computed as there.
        k3_fp8w_lowm_mxfp8_push_op(
            *latent_mx,
            cp.weight,
            cp.scale,
            cp_ws,
            col0=cp.c0,
            hidden=self.hidden_size,
            tactic=default_mx_tactic(num_tokens, self.hidden_size, self.moe_hidden_size),
        )
        return self._cp_tail_finish(cp_ws, num_tokens, residual, side, defer_residual)

    def _cp_tail_finish(
        self, cp_ws, num_tokens: int, residual: torch.Tensor, side: bool, defer_residual: bool
    ):
        """The column-parallel tail call (all-gather rows pushed, the shared expert's slots pushed
        on the shared stream, joined here) as a ``_DeferredRpTail`` the next pre-attention mix
        finishes, or the updated residual."""
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

        from .layers import LamportPushTarget

        if side:
            self.moe_shared_event.wait()
        # The call reproduces the shared expert's C++ all-reduce (its transport by size).
        cpp_one_shot = (
            num_tokens * self.hidden_size * cp_ws.world * 2 <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        )
        cp_target = LamportPushTarget(
            cp_ws, True, cpp_one_shot, torch.empty_like(residual.reshape(num_tokens, -1))
        )
        cp_target.pushed = True
        cp_target.deferred = True
        cp_target.ag = True
        deferred = _DeferredRpTail(residual.reshape(num_tokens, -1), LamportPending(cp_target))
        if defer_residual:
            return deferred
        return deferred.materialize().view(residual.shape)

    def _rp_tail_forward(
        self,
        routed_output: Callable[..., torch.Tensor],
        identity: torch.Tensor,
        shared_part: Optional[torch.Tensor],
        residual: torch.Tensor,
        target,
        defer_residual: bool,
    ):
        """The row-parallel tail: the routed experts (``routed_output``) and the shared expert's
        part (``_rp_shared_output``; ``shared_part`` if it already ran ahead of the routed experts),
        then the latent all-reduce + RMSNorm, then this rank's tail GEMM pushing its partial into
        ``target``. Returns a ``_DeferredRpTail`` (``defer_residual``) or the updated residual."""
        if _K3_ULL2_COLL_CP_TAIL and self._cp_tail_applies(
            residual.numel() // self.hidden_size, target
        ):
            return self._cp_tail_forward(
                routed_output, identity, shared_part, residual, target, defer_residual
            )
        if shared_part is not None:
            # The shared expert ran ahead on the shared stream and the routed experts waited for
            # it (moe_shared_gemm_event).
            routed_out = routed_output()
            self.moe_shared_gemm_event.wait()
        elif self.shared_expert_stream is not None and do_multi_stream():
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
        routed_latent = None
        latent_mx = None
        if _K3_ULL_TAIL_SANDWICH:
            num_rows = routed_out.numel() // self.moe_hidden_size
            early = self._rp_tail_latent_ar_early(
                routed_out, target, mxfp8=self._rp_tail_mx_applies(num_rows)
            )
            if early is not None:
                routed_latent, latent_mx = early
        if routed_latent is None:
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
        if latent_mx is not None:
            # TRTLLM_K3_ULL_TAIL_SANDWICH: the MXFP8 GEMM on the all-reduce's MXFP8 output (its
            # slice read in place), no quantization in the tail GEMM; the same bits.
            k3_fp8w_lowm_mxfp8_push_op(
                *latent_mx,
                self.routed_expert_up_proj.weight,
                rp.scale,
                target.workspace,
                target.one_shot,
                k0=lo,
                k1=rp.k1,
                addend=shared_part,
            )
        else:
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

    @staticmethod
    def _fused_projection_applies(x: torch.Tensor, projection: nn.Module) -> bool:
        """Whether the fused latent projection replaces ``projection(x)``."""
        if not _K3_LD_FUSED:
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
        if _K3_FP8W_LOWM_GEMM and _k3_lowm_gemm_can_run(x, projection.weight, packed_scale):
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
        """``TRTLLM_K3_LATENT_AR_OWN_WS``: the routed latent all-reduce + RMSNorm on its own MNNVL
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
        # The shared expert as the persistent kernel (forked with the ULL front's shared fork,
        # else in the column-parallel tail).
        persist = self._persist_applies(hidden_states.shape[0], rp_target)
        projected_input = None
        early_routed_in = None
        shared_partial = None
        front_routing = None
        route_target = self._route_tail_target(hidden_states.shape[0])
        if self._ull_front_applies(hidden_states):
            # TRTLLM_K3_ULL_FRONT: the shared expert forks where _ull_fork_shared puts it; the sharded
            # front routes (route_target is the front's) and produces the NVFP4 routed input on this stream.
            route_target = None
            after_head, after_front, forked = self._ull_fork_shared(
                identity, persist, hidden_states.shape[0]
            )
            router_logits, projected_input, front_routing = self._ull_front(
                hidden_states, after_head=after_head
            )
            if after_front is not None:
                after_front()
            if forked:
                shared_partial = forked[0]
        elif self._can_fuse_router_down(hidden_states):
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

        def _routed_output(output_push: Optional[LamportWorkspaceView] = None):
            # ``output_push``: also push the routed rows into that one-shot all-reduce's slots
            # (the column-parallel tail's k3_moe_tail1).
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
                # noaux_tc_op, unless the front routed already; then the kernels.
                y = self.routed_experts(
                    routed_in,
                    logits,
                    route=self._router(route_target),
                    routing=front_routing,
                    input_ready_event=moe_kwargs.get("input_ready_event"),
                    output_push=output_push,
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            else:
                if front_routing is not None or output_push is not None:
                    raise RuntimeError(
                        "caller routing or an output push needs the CuTe DSL experts"
                    )
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
                _routed_output, identity, shared_partial, residual, rp_target, defer_residual
            )
        # The final join includes the shared all-reduce, keeping it ordered before any outer
        # routed reduction or the next layer's collectives.
        if shared_partial is not None:
            routed_out = _routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                shared_out = shared_partial
                if self._shared_needs_all_reduce():
                    shared_out = self._shared_all_reduce()(shared_partial)
                self.moe_shared_event.record()
            self.moe_shared_event.wait()
        elif (
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
