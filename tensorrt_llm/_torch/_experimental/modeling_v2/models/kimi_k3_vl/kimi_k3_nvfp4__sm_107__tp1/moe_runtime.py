# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), tp1: the MoE block.

``K3MoERuntime`` runs one layer's latent MoE on one rank: the router (``K3MoEGate``, the BF16
gate GEMM with FP32 output), the latent down projection, the CUTEDSL routed experts
(``ops/moe/k3r_experts.py``), ``routed_expert_norm`` / ``routed_expert_up_proj``, and the shared
experts (``_K3SharedExpert``) on an auxiliary stream. One rank holds every expert, so no
collective runs and the layer's residual update is the caller's.

Also here: the routed-expert checkpoint specs (key families and the loader of each
quantization).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import ActivationType, AuxStreamType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from ..common.multi_stream import do_multi_stream
from ..ops.moe.k3_route_tail import k3_route_tail as k3_route_tail_op
from ..ops.moe.k3r_experts import K3RoutedExperts, SortedRouting, SortTables
from ..sm107_common.layers import (
    GatedMlp,
    RMSNormWeight,
    fork_join,
    gated_activation,
    subgroup_mapping,
)
from .text_config import _resolve_kimi_situ_betas

# The routed experts' kernel set: the context servers run CUTEDSL.
_K3_MOE_BACKEND = "CUTEDSL"

# Decode-batch routing tail of the CuteDSL MoE (see k3_route_tail): replaces noaux_tc_op ->
# moe_sort with one launch on a cluster of at most eight CTAs (same logits, same routing).
# Largest token count the routing tail serves: up to 256 tokens one cluster, above that (at most
# 2048 tokens, see k3_route_tail.wide_sort_fits) a grid of CTAs whose last CTA does the sort.
_K3_ROUTE_TAIL_MAX_M = 2048

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


class K3MoEGate(nn.Module):
    """Kimi K3 gate: the router weight, the score-correction bias and the routing kernels.

    Routing is ungrouped sigmoid top-k with the bias added for the selection and the selected
    scores renormalized and scaled (``noaux_tc``); :meth:`route` and :meth:`route_tail` are two
    kernels of the same routing.
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
    backend.load_expert(global_expert_id=expert_idx, local_slot_id=local_slot_id, **tensors)


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


class K3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block on one rank: the gate routes, the routed experts run the CUTEDSL
    kernel chain of ``ops/moe/k3r_experts``, the shared expert runs beside them."""

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

        # Routing scores stay fp32: the gate GEMM runs bf16xbf16 with fp32 accumulate/output (the
        # checkpoint stores the gate weight in bf16).
        self.gate = K3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config)
        self._moe_backend = routed_moe_model_config.moe_backend
        self._attention_dp = model_config.mapping.enable_attention_dp
        routed_mapping = routed_moe_model_config.mapping
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
        # One rank: the routed latent partial sums are complete, nothing to all-reduce.
        self.routed_experts.all_reduce = None
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
        # Neither the shared nor the routed output is a partial: no all-reduce to order.
        self._defer_shared_allreduce = False
        self._reduce_routed_output = False
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
            reduce_output=False,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # FlashInfer RMSNorm on the routed latent.
        self.routed_expert_norm = RMSNormWeight(self.moe_hidden_size, cfg.rms_norm_eps, dtype=dtype)

    def prepare_expert_load(self) -> None:
        """Before this layer's first routed expert streams in."""
        self.routed_experts.backend.prepare_streaming_load()

    def finalize_expert_load(self) -> None:
        """After this layer's last routed expert streamed in."""
        self.routed_experts.backend.finalize()

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

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits: the BF16 router GEMM."""
        return self.gate.compute_logits(hidden_states)

    def _router(self, tail_target: Optional[Tuple[int, int]]):
        """The routing of this forward's router logits, for the experts
        ``[local_expert_offset, local_expert_offset + local_num_experts)`` the routed experts ask
        for: the CuTe routing tail where ``tail_target`` is set (with its ``moe_sort`` tables where
        they are of the experts asked for), else ``noaux_tc_op`` (the experts then run
        ``moe_sort``)."""
        gate = self.gate
        consumes_tables = self.routed_experts.consumes_sort_tables

        def route(logits: torch.Tensor, local_expert_offset: int, local_num_experts: int):
            target = (local_expert_offset, local_num_experts)
            if tail_target is not None:
                routed = gate.route_tail(logits, *tail_target)
                if consumes_tables and target == tail_target:
                    return routed
                return SortedRouting(routed.token_selected_experts, routed.token_final_scales, None)
            return SortedRouting(*gate.route(logits), None)

        return route

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
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
        if model_config.moe_backend != _K3_MOE_BACKEND:
            raise ValueError(
                f"the Kimi K3 tp1 routed experts run on {_K3_MOE_BACKEND}; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        # DWDP (setup_dwdp after loading) collects the CUTEDSL leaves' expert weights as they are.
        mapping = model_config.mapping
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
        routed_model_config._frozen = True
        return routed_model_config

    def forward(self, hidden_states: torch.Tensor, all_rank_num_tokens=None) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16; returns the MoE output."""
        identity = hidden_states
        early_routed_in = None
        route_target = self._route_tail_target(hidden_states.shape[0])
        router_issued = False
        if self.latent_down_stream is not None and do_multi_stream():
            # Forked ahead of the router GEMM: the latent chain needs only hidden_states.
            self.moe_latent_fork_event.record()
            if hidden_states.shape[0] <= _K3_ROUTER_FIRST_MAX_M:
                # Router before latent_down in creation order (see _K3_ROUTER_FIRST_MAX_M).
                router_logits = self._router_logits(hidden_states)
                router_issued = True
            with torch.cuda.stream(self.latent_down_stream):
                self.moe_latent_fork_event.wait()
                early_routed_in = self._routed_projection(
                    hidden_states, self.routed_expert_down_proj
                )
                self.moe_latent_ready_event.record()
            # No record_stream (under graph capture it holds every layer's
            # blocks until capture ends): the tensor is read on main only
            # after the join, and later work on this stream forks from main.
        if not router_issued:
            router_logits = self._router_logits(hidden_states)

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a single fused GEMM (fp8_swap_ab_gemm).
            input_ready_event = None
            if early_routed_in is not None:
                routed_in = early_routed_in
                # The routed experts join the event before their first read of x.
                input_ready_event = self.moe_latent_ready_event
            else:
                routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            # Routed by the CuTe tail (route_target), the sorted routing kernel or
            # noaux_tc_op; then the kernels.
            y = self.routed_experts(
                routed_in,
                router_logits,
                route=self._router(route_target),
                input_ready_event=input_ready_event,
                zeroed_output=None,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output():
            """The shared expert's output."""
            return self.shared_experts(identity, final_all_reduce_params=None)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed expert chain. Multi-stream engages only under CUDA
        # graphs; otherwise both branches run in order on the default stream.
        routed_out, shared_out = fork_join(
            _routed_output,
            _shared_output,
            self.moe_main_event,
            self.moe_shared_event,
            self.shared_expert_stream,
        )
        return routed_out + shared_out
