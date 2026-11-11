# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), dep16: the target class and its model.

Serves generation with attention DP over 16 ranks (tp 16), the routed experts over EP16 on
MEGAMOE_CUTEDSL: the c383 recipe's generation servers. The target's constants
(``constants.py``) are folded into this directory's code; ``CONSTANTS`` stays as the record of
the values (and is what ``K3_CONSTANTS`` declares).

The checkpoint is the vision-language model's; this target is text only. It builds no vision
tower (its checkpoint tensors are never read, ``weights.py``) and refuses multimodal input. The
text model ``KimiK3DefaultForCausalLM``:

* 93 decoder layers (``KimiK3DefaultDecoderLayer``): 69 KDA (Kimi Delta Attention, linear
  attention) layers and 24 MLA (absorbed MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 has a dense SiTU MLP; layers 1..92 the latent MoE of ``moe_runtime.py``: 896 routed
  experts at top-16 (sigmoid + ``e_score_correction_bias`` routing, FP32 router GEMM), 2 shared
  experts, and the latent down / norm / up projections around the routed experts. The MoE layer
  hands its residual update to the next layer's pre-attention mix up to 64 tokens.
* Attention residuals (``attn_res.py``): before ``input_layernorm``, before
  ``post_attention_layernorm`` and at the model output each token mixes a bank of earlier
  sublayer outputs (the K3 PDL kernels up to 64 tokens); a snapshot is appended whenever
  ``layer_idx % attn_res_block_size == 0``.

Caching: per request, a KDA layer keeps a short-convolution window
``[3 * num_heads * head_dim, W]`` (bf16, sections ``[q | k | v]``) and a delta-rule recurrent
state ``[num_heads, head_dim, head_dim]`` (fp32, ``[H, V, K]``) in ``KimiK3HybridCacheManager``
(KV cache manager V2); an MLA layer uses the paged KV cache with one KV head of
``kv_lora_rank + qk_rope_head_dim`` (576).

Parallelism: attention is data-parallel (each rank owns its requests' tokens), the dense MLP and
the shared experts are replicated, the routed experts are the MegaMoE module's (EP over the 16
ranks, its own dispatch / combine). ``lm_head`` is vocab-parallel with a gather, so every rank
has the full logits. There is no Helix / context parallelism and no pipeline parallelism.

Speculative decoding: one-engine DSpark with the family's standalone MLA drafter and worker
(``dspark/``) on the fused spec tail (``spec_tail_fusion``); the decode head is not fused. KDA
and MLA verify several tokens per step and keep the accepted state. Chunked prefill continues a
request's KDA state and attends over its cached MLA prefix. KV cache block reuse is opt-in
(``kv_cache_config.enable_block_reuse``); periodic KDA state snapshots come from
``mamba_state_config.periodic_snapshot_interval``.

Engine settings the recipe carries, not this class: the model loader applies model defaults of the
checkpoint's built-in class before a Modeling V2 target is routed, so a target cannot set
``llm_args``. The c383 GEN recipe turns ``attention_dp_config.enable_kv_balanced_admission`` on,
which runs this target's planner (``kv_balance.py``).

Checkpoint loading is in ``weights.py``.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger

from ..common.weight_io import check_text_only
from ..dspark.drafter import K3DSparkPath
from ..dspark.shell import KimiK3SpecForCausalLM
from ..ops.attention import _fp4_mla_decode as fp4_mla_decode
from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.layers import EmbeddingTable, GatedMlp, RMSNormWeight, rms_norm, subgroup_mapping
from .attn_res import (
    _apply_attn_res,
    _apply_attn_res_add_and_rmsnorm,
    _apply_attn_res_and_rmsnorm,
    _defer_moe_residual,
    _DeferredMoEResidual,
    _k3_attn_res_pdl_moe_add,
    _materialize_prefix_sum,
)
from .constants import CONSTANTS
from .kda import K3KDALinearAttention, K3MambaMetadata
from .kv_balance import K3Dep16KvBalancedAdmission
from .mla import K3MLARuntime, resolve_attention_quant_config
from .moe_runtime import K3MoERuntime
from .text_config import _get_text_config, _is_kda_layer, _is_mla_layer
from .weights import KimiK3DefaultTextWeights, KimiK3DefaultWeights

# CUDA stream priority of the MoE input stream (latent_down chain, routed-output memset).
# Negative values rank above the main and shared-expert streams, so the latent chain that
# gates the routed GEMM1 gets free SMs first.
_K3_LATENT_STREAM_PRIORITY = 0

# CUDA stream priority of the stream shared by the attention-side aux work and the MoE shared
# expert. Negative values let the shared-expert chain take SMs ahead of the persistent routed
# GEMMs on the main stream, so its all-reduce does not land after them.
_K3_SHARED_STREAM_PRIORITY = 0

# The DFlash/DSpark hidden-state tap captures the pre-norm attn_res mixture (the value the next
# consumer reads), the convention of SGLang and of the RadixArk/Kimi-K3-DSpark drafter
# (``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py aggregate_stream``). A drafter
# distilled against the raw running prefix sum (vLLM's ``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``)
# would score lower here with nothing raised (K3 + RadixArk: AR 71.4% -> 66.9% the other way).


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiK3DefaultDecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16
        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            attention_config = copy.copy(model_config)
            attention_config._frozen = False
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(model_config, layer_idx, name)
                for name in projection_names
            }
            attention_config._frozen = model_config._frozen
            self.linear_attn = K3KDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping.mapping_for_full_rows(),
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = K3MLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
            )

        if self.is_moe:
            self.block_sparse_moe = K3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            # The dense MLP (layer 0): BF16 gate_up GEMM, SiTU-and-mul, down GEMM. Attention DP
            # replicates it (ranks own different tokens): no all-reduce.
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            self.mlp_tp_size = 1
            self.mlp = GatedMlp(
                cfg.hidden_size,
                cfg.intermediate_size,
                activation="situ",
                beta=getattr(cfg, "activation_situ_beta", None) or 1.0,
                linear_beta=getattr(cfg, "activation_situ_linear_beta", None),
                dtype=dtype,
                mapping=subgroup_mapping(mlp_mapping, self.mlp_tp_size),
            )

        # FlashInfer RMSNorm on the K3 weights: the input norm after the pre-attention mix and the
        # post-attention norm after A_mid (fused into the attention-residual kernels where they
        # serve, ``trtllm::flashinfer_rmsnorm`` otherwise).
        self.input_layernorm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNormWeight(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3): the score norms and the [1, H] score
        # projections the attention-residual kernels read (``.weight`` / ``.eps``).
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.mlp_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    @staticmethod
    def _capture_hidden_states(capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The DSpark tap of ``capture[1]``."""
        capture[0].maybe_capture_hidden_states(capture[1], value, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows. The prefix sum (input and output) can be a
        ``_DeferredMoEResidual`` (a MoE layer's residual update, up to 64 tokens).

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.
        """
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        if isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                )
                if fused is not None:
                    hidden_states, normed = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None:
            # The mixture tap needs the PRE-norm value (SGLang's
            # aggregate_stream is a "softmax mixture, no output norm"), which
            # the fused attn-res + RMSNorm kernel does not expose. Keep the two
            # steps split on captured layers only and fuse everywhere else.
            mixed = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if num_snapshots > 0
                else hidden_states
            )
            self._capture_hidden_states(capture, mixed)
            hidden_states = rms_norm(mixed, self.input_layernorm.weight, self.input_layernorm.eps)
        elif normed is not None:
            hidden_states = normed
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = rms_norm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.eps
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        if self.is_kda:
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata)

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            # The MoE layer applies the residual update itself (or defers it to the next mix).
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                all_rank_num_tokens,
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
            )
            return prefix_sum, num_snapshots
        hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op stand-in for ``forward``, matching ``DecoderLayer.skip_forward``.

        ``modeling_utils.skip_forward()`` only drops a module's weights when it
        finds this attribute, so without it the layer-wise benchmarks would
        allocate all 93 layers instead of the profiled slice.
        """
        return hidden_states, block_residual


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiK3DefaultModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream(priority=_K3_SHARED_STREAM_PRIORITY)]
        aux_stream_list += [torch.cuda.Stream() for _ in range(2)]
        aux_stream_list.append(torch.cuda.Stream(priority=_K3_LATENT_STREAM_PRIORITY))
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        # The whole vocabulary on every rank: a row gather (``F.embedding``).
        self.embed_tokens = EmbeddingTable(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiK3DefaultDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        # The final norm, fused into the output attention-residual mix where that kernel serves.
        self.norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode=attn_res_stream",
            key="kimi_k3_aux_capture_mode",
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = F.embedding(input_ids, self.embed_tokens.weight)
        hidden_states = inputs_embeds

        # The FP4 MLA decode plan of the previous step is stale.
        fp4_mla_decode.begin_step(attn_metadata)

        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            # DFlash/DSpark hidden-state capture. The drafter is distilled on
            # the aggregated stream value -- the pre-norm softmax mixture its
            # next consumer sees -- not on the raw prefix sum a layer returns,
            # which is SGLang's fallback for models without the
            # attention-residual scheme. Capturing the prefix sum costs 4.5pt
            # of draft acceptance on K3 + RadixArk DSpark (AR 66.9% -> 71.4%).
            # The tap fires inside layer i+1, which computes that tensor
            # anyway; see its forward docstring. Ground truth: SGLang
            # kimi_k3.py:2697 _dspark_capture_stream, attn_residual.py:313
            # aggregate_stream_torch.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
            )
        hidden_states = _materialize_prefix_sum(hidden_states)

        # The last layer has no successor, so this one recompute is
        # unavoidable -- output-side score weights, matching SGLang's
        # layer_idx + 1 >= end_layer branch. Unreachable for K3's capture set
        # against 93 layers; kept so a set that does include the final layer
        # gets the right tensor rather than the raw prefix sum.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        output = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        return output


class KimiK3DefaultForCausalLM(
    KimiK3DefaultTextWeights, KimiK3SpecForCausalLM[KimiK3DefaultModelingV2, Any]
):
    """The Kimi K3 text model (KDA + MLA + MoE decoder layers) of ``ModelingV2KimiK3Nvfp4Sm107Dep16``.

    Registered under no architecture: the target class builds it as its ``llm``. Weight loading
    is the ``KimiK3DefaultTextWeights`` mixin's (``weights.py``).

    Speculative decoding: one-engine DSpark with a standalone MLA drafter, built by the family's
    ``KimiK3SpecForCausalLM`` (``dspark/``): the fused spec tail (``spec_tail_fusion``, not the
    v2 tail), the full drafter of a generation server (not context-KV-only), and no fused decode
    head (no ``RuntimeHooks.decode_prologue``)."""

    mamba_metadata_cls = K3MambaMetadata

    @property
    def dspark_path(self) -> K3DSparkPath:
        """The DSpark deployment path: the fused spec tail, not the v2 tail."""
        return K3DSparkPath(spec_tail_fusion=True, tail_v2=False)

    @property
    def dspark_context_kv_only(self) -> bool:
        """A generation server: the drafter is not the context-KV-only one."""
        return False

    @property
    def spec_decode_prologue(self):
        """No fused decode head."""
        return None

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        if model_config.mapping.has_cp_helix():
            raise NotImplementedError("the Kimi K3 dep16 target runs no Helix context parallelism")
        # Speculative decoding is one-engine DSpark with a standalone drafter checkpoint;
        # KimiK3SpecForCausalLM rejects every other speculative config. The target side is the
        # hidden-state capture in KimiK3DefaultModelingV2.forward.
        super().__init__(
            KimiK3DefaultModelingV2(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )


_KV_BALANCED_ADMISSION = K3Dep16KvBalancedAdmission()


class ModelingV2KimiK3Nvfp4Sm107Dep16(KimiK3DefaultWeights, PreTrainedModel):
    """Kimi K3 NVFP4 on SM107, dep16.

    The text model ``KimiK3DefaultForCausalLM`` as ``llm`` behind the vision-language model's
    interface: text only, no vision tower (``mm_encoder`` is ``None``, as under multimodal
    disaggregation), and a step carrying multimodal input is refused. The speculative-decoding
    and weight-loading attributes the engine reads off the outer model (``draft_model``, ``lm_head``,
    ``runtime_hooks``, ...) are the text model's. Weight loading is the
    ``KimiK3DefaultWeights`` mixin's (``weights.py``).
    """

    # The values this target's code is folded from (``constants.py``).
    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    _LANG_PREFIX = "language_model."
    mamba_metadata_cls = KimiK3DefaultForCausalLM.mamba_metadata_cls

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["V2"]:
        """Prefer KV cache manager V2 (as Kimi K2.5)."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move
        the KDA recurrent state; the C++ transceiver has no KDA support.
        Adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL. An explicit non-Python runtime is
        rejected by ``get_kv_cache_manager_cls`` rather than silently
        routed to a path that cannot transfer the recurrent state.
        """
        return "PYTHON"

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself (TRT-LLM attention is wired
        # inside the sub-modules), so declare support to keep HF init happy.
        self._supports_sdpa = True
        PreTrainedModel.__init__(self, config)

        # Re-entry guard: the tail of this __init__ repoints
        # model_config.pretrained_config at the text_config and remaps the quant
        # exclude list, so running the body twice on the same instance would
        # fail the text_config assert below. If the text backbone already
        # exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        self._vlm_pretrained_config = config

        # Text only: no vision tower.
        self.mm_encoder = None

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3 config must have text_config"
        text_model_config._frozen = False
        text_model_config.pretrained_config = config.text_config

        # Remap quant exclude_modules: language_model.X -> model.X
        if text_model_config.quant_config.exclude_modules:
            text_model_config.quant_config = copy.copy(text_model_config.quant_config)
            p = self._LANG_PREFIX
            mapped = []
            for m in text_model_config.quant_config.exclude_modules:
                if m.startswith(p):
                    rest = m[len(p) :]
                    if rest.startswith("layers."):
                        rest = "model." + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            text_model_config.quant_config.exclude_modules = mapped

        if not text_model_config.skip_create_weights_in_init:
            text_model_config.skip_create_weights_in_init = True
        text_model_config._frozen = True

        self.llm = KimiK3DefaultForCausalLM(text_model_config)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone, matching K2.5.
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    # Forward spec-dec / weight-loading attrs to self.llm: this wrapper is not
    # itself a spec-dec model but is the outer model that those paths see.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The text model's step-loop hooks (the engine reads them off the outer model), 4 GiB of
        device memory left outside the PyTorch memory cap, and the KV-balanced admission planner
        the c383 GEN servers run (window 32, the KV load closest to the mean)."""
        return replace(
            self.llm.runtime_hooks or DEFAULT_RUNTIME_HOOKS,
            torch_mem_safety_margin=4 * 1024**3,
            kv_balanced_admission_planner=_KV_BALANCED_ADMISSION,
        )

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs,
    ) -> torch.Tensor:
        check_text_only(multimodal_params)
        # The text model embeds ``input_ids`` itself; the wrapper hands it no embeddings.
        inputs_embeds = None
        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            **kwargs,
        )


__all__ = ["KimiK3DefaultForCausalLM", "ModelingV2KimiK3Nvfp4Sm107Dep16"]
