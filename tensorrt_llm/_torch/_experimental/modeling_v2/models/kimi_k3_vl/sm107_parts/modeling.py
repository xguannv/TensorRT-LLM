# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107): the model.

``KimiK3Sm107ForConditionalGeneration`` serves every deployment ``routing.py`` admits on SM107:
one-rank DWDP context servers, generation over Helix CP8, over Helix CP16 with the routed
experts split MoE TP4 x EP4, and attention DP over 16 ranks. The layout comes from the engine's
``Mapping``; the deployment's optimizations come from its ``features.py`` profile, selected
before this module is imported.

It is the vision-language model: the vision tower (MoonViT3d) and its projector, and the text
model ``KimiK3DefaultForCausalLM``:

* 93 decoder layers (``KimiK3DefaultDecoderLayer``): 69 KDA (Kimi Delta Attention, linear
  attention) layers and 24 MLA (absorbed MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 has a dense SiTU MLP; layers 1..92 the latent MoE of ``moe_runtime.py``: 896 routed
  experts at top-16 (sigmoid + ``e_score_correction_bias`` routing), 2 shared experts, and the
  latent down / norm / up projections around the routed experts.
* Attention residuals (``attn_res.py``): before ``input_layernorm``, before
  ``post_attention_layernorm`` and at the model output each token mixes a bank of earlier
  sublayer outputs; a snapshot is appended whenever ``layer_idx % attn_res_block_size == 0``.

Caching: per request, a KDA layer keeps a short-convolution window
``[3 * num_heads * head_dim, W]`` (bf16, sections ``[q | k | v]``) and a delta-rule recurrent
state ``[num_heads, head_dim, head_dim]`` (fp32, ``[H, V, K]``) in ``KimiK3HybridCacheManager``
(KV cache manager V2); an MLA layer uses the paged KV cache with one KV head of
``kv_lora_rank + qk_rope_head_dim`` (576).

Parallelism: attention is head-split over the TP group, its KV sequence split over the CP
group under Helix, or data-parallel under attention DP. The routed experts are split MoE TP x
EP over the TP group (EP only unless ``moe_tensor_parallel_size`` /
``moe_expert_parallel_size`` are set). Routing is computed on every rank, and the routed
partial sums are all-reduced in the latent space, before ``routed_expert_norm`` /
``routed_expert_up_proj``. Without attention DP the shared experts are tensor-parallel over
the TP group and reduce on an auxiliary stream while the routed experts run on the main
stream; under attention DP they are replicated. ``lm_head`` is vocab-parallel with a gather,
so every rank has the full logits.

Speculative decoding: the one-engine DSpark / DFlash drafters of ``drafter.py`` / ``dflash.py``, and the suffix
automaton; KDA and MLA verify several tokens per step and keep the accepted state. Chunked
prefill continues a request's KDA state and attends over its cached MLA prefix. KV cache
block reuse is opt-in (``kv_cache_config.enable_block_reuse``); periodic KDA state snapshots
come from ``mamba_state_config.periodic_snapshot_interval``. Pipeline parallelism is not
supported.

Checkpoint loading is in ``weights.py``.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_k3_vl import KimiK3VisionModel
from tensorrt_llm._torch.models.modeling_kimi_k25 import (
    _MEDIA_PLACEHOLDER_TOKEN_ID,
    DISAGG,
    _has_meta_tensors,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, MetaInitException
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger

from ..kernels.k3r_spec.spec_decode_head import spec_decode_head
from ..ops.attention import _fp4_mla_decode as fp4_mla_decode
from .attn_res import (
    _K3_ULL_TAIL_SANDWICH,
    _apply_attn_res,
    _apply_attn_res_add_and_rmsnorm,
    _apply_attn_res_and_rmsnorm,
    _defer_moe_residual,
    _DeferredMoEResidual,
    _DeferredRpTail,
    _k3_attn_res_lamport_applies,
    _k3_attn_res_pdl_moe_add,
    _k3_attn_res_pdl_rp_tail,
    _materialize_prefix_sum,
)
from .drafter import install_k3_dspark
from .features import active as _features
from .kda import K3KDALinearAttention, K3MambaMetadata
from .layers import (
    EmbeddingTable,
    GatedMlp,
    RMSNormWeight,
    gather_embedding,
    rms_norm,
    subgroup_mapping,
)
from .mla import K3MLARuntime, resolve_attention_quant_config
from .moe_runtime import _K3_ULL2_COLL_CP_TAIL, K3MoERuntime
from .text_config import _get_text_config, _is_kda_layer, _is_mla_layer
from .weights import KimiK3DefaultTextWeights, KimiK3DefaultWeights

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

_F = _features()

_K3_FUSE_MOE_RESIDUAL = _F.fuse_moe_residual

# CUDA stream priority of the MoE input stream (latent_down chain, routed-output memset).
# Negative values rank above the main and shared-expert streams, so the latent chain that
# gates the routed GEMM1 gets free SMs first.
_K3_LATENT_STREAM_PRIORITY = 0

# CUDA stream priority of the stream shared by the attention-side aux work and the MoE shared
# expert. Negative values let the shared-expert chain take SMs ahead of the persistent routed
# GEMMs on the main stream, so its all-reduce does not land after them.
_K3_SHARED_STREAM_PRIORITY = 0

# BF16 shared/latent MLP weights are optionally quantized once after loading.
# Checkpoint-native attention FP8 is independent of this lossy conversion.
_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV = "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP"


KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"
"""Which residual-stream value the DFlash/DSpark hidden-state tap captures.

``1`` (default) captures the pre-norm attn_res mixture -- the value the next
consumer actually reads. ``0`` captures the raw running prefix sum instead.

Both conventions exist in the wild and a drafter distilled against one scores
lower on the other with nothing raised, so this is a property of the DRAFTER
checkpoint, not a performance knob. SGLang (and therefore RadixArk/Kimi-K3-DSpark)
uses the mixture: ``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py
aggregate_stream``. vLLM implements both and defaults to the prefix
(``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``, ``models/kimi_k3/nvidia/model.py
_capture_aux_hidden_stream``), which is what a TorchSpec-distilled drafter may
have been trained against. Measured cost of getting it wrong on K3 + RadixArk:
AR 71.4% -> 66.9%.

Per-checkpoint measurements on K3, GSM8K AL, n=200, TEP8:

===================  ==================  ============  ======
drafter              stream (default 1)  prefix (0)    delta
===================  ==================  ============  ======
RadixArk (GQA)       71.8%               --            --
Inferact (MLA)       65.7%               66.6%         +0.9pt
===================  ==================  ============  ======

So the default is right for RadixArk. For Inferact the prefix convention
matches its vLLM/TorchSpec lineage and measures better, but +0.9pt at n=200 is
inside this harness's noise band (it treats RadixArk's own 71-73% spread as
noise), so this is a direction, not a settled requirement -- unlike the 4.5pt
RadixArk case above, which was unambiguous. Both Inferact acc_len values (5.60
and 5.66) sit on its model card's 5.64, so neither convention is grossly wrong
for it.

Deriving this from checkpoint metadata is not possible today: neither published
drafter's config records which capture convention it was distilled against."""

_AUX_ATTN_RES_STREAM_ENABLED = os.environ.get(KIMI_K3_AUX_ATTN_RES_STREAM_ENV, "1") == "1"

# TRTLLM_K3_ATTN_RES_MXFP8=1: the pre-attention mix that folds in the previous MoE residual
# (k3_attn_res_moe_add_rmsnorm_mx) also stores the MXFP8 form of its output, and the
# attention's first projection (KDA fused QKVG, MLA fused q_a / kv_a) runs the low-M GEMM on it
# (k3_fp8w_lowm_mxfp8) instead of quantizing the BF16 rows itself: bitwise the same projection,
# generation-only batches where the projection runs the low-M kernel.
_K3_ATTN_RES_MXFP8 = _F.attn_res_mxfp8

# TRTLLM_K3_ULL_HYGIENE=1 (decode launch hygiene; the same bits): in CUDA-graph decode steps
# (multi-stream on)
# - the DSpark hidden-state taps (``maybe_capture_hidden_states``, a strided copy into the capture
#   buffer) run on a side stream forked after their source and joined at the end of the model
#   forward, instead of between the mix and the input norm of the tapped layer;
# - at an attention-residual block boundary served by the fused column-parallel / row-parallel tail
#   mix (``_k3_attn_res_pdl_rp_tail``), the mix writes the updated prefix sum straight into the
#   snapshot slot of the bank, so the snapshot copy (a memcpy graph node, which also cuts the PDL
#   edge to the next layer's first GEMM) is gone.
_K3_ULL_HYGIENE = _F.ull_hygiene


class _SideCopies:
    """``TRTLLM_K3_ULL_HYGIENE``: copies that nothing reads before the end of the forward run on a
    side stream. Each is forked after its source; the sources stay referenced until
    :meth:`join` makes the current stream wait for every pending copy, so the caching allocator
    cannot hand their memory to a later kernel while a copy may still read it."""

    def __init__(self, stream: torch.cuda.Stream):
        self.stream = stream
        self._pending: List[Tuple[torch.Tensor, torch.cuda.Event]] = []

    @staticmethod
    def active() -> bool:
        """Side copies only where the model forks side streams (CUDA graph capture)."""
        return do_multi_stream()

    def run(self, fn: Callable[[], None], source: torch.Tensor) -> None:
        """Run ``fn`` (a copy reading ``source``) on the side stream."""
        fork = torch.cuda.Event()
        fork.record()
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(fork)
            fn()
            done.record()
        self._pending.append((source, done))

    def join(self) -> None:
        current = torch.cuda.current_stream()
        for _, done in self._pending:
            current.wait_event(done)
        self._pending.clear()


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiK3DefaultDecoderLayer(nn.Module):
    # TRTLLM_K3_ULL_HYGIENE: the model's side-stream copies (``None``: off).
    _side_copies: Optional[_SideCopies] = None

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
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = K3MLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                # CP original stashed by _setup_helix_mappings; None outside helix.
                mapping_with_cp=getattr(model_config, "_helix_mapping_with_cp", None),
            )

        if self.is_moe:
            self.block_sparse_moe = K3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            # The dense MLP (layer 0): BF16 gate_up GEMM, SiTU-and-mul, down GEMM and, on a TP
            # group, the down projection's all-reduce. Attention DP replicates it (ranks own
            # different tokens); otherwise it is TP over the largest group of consecutive ranks
            # that divides the intermediate size and stays within one node.
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            if mlp_mapping.enable_attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, mlp_mapping.tp_size)
                if self.mlp_tp_size > mlp_mapping.gpus_per_node:
                    self.mlp_tp_size = math.gcd(self.mlp_tp_size, mlp_mapping.gpus_per_node)
            self.mlp = GatedMlp(
                cfg.hidden_size,
                cfg.intermediate_size,
                activation="situ",
                beta=getattr(cfg, "activation_situ_beta", None) or 1.0,
                linear_beta=getattr(cfg, "activation_situ_linear_beta", None),
                dtype=dtype,
                mapping=subgroup_mapping(mlp_mapping, self.mlp_tp_size),
                reduce_output=self.mlp_tp_size > 1,
                allreduce_strategy=model_config.allreduce_strategy,
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

    def _attn_takes_mxfp8(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Whether the pre-attention mix should also emit its MXFP8 form for the attention mixer
        (``TRTLLM_K3_ATTN_RES_MXFP8``; generation-only batches): the KDA fused QKVG projection
        (``qkvg_takes_mxfp8``) or the MLA fused q_a / kv_a projection (``q_a_takes_mxfp8``)."""
        if not _K3_ATTN_RES_MXFP8 or attn_metadata.num_contexts != 0:
            return False
        if self.is_kda:
            return self.linear_attn.qkvg_takes_mxfp8(num_rows)
        q_a_takes_mxfp8 = getattr(self.self_attn.mixer, "q_a_takes_mxfp8", None)
        return q_a_takes_mxfp8 is not None and q_a_takes_mxfp8(num_rows)

    def _attn_mx_atom(self, num_rows: int) -> bool:
        """Whether that MXFP8 form goes to the mid-M GEMM, i.e. the mix stores its scales in that
        GEMM's layout (k3_attn_res ``mx_atom``)."""
        if self.is_kda:
            mx_atom = getattr(self.linear_attn, "qkvg_mx_atom", None)
        else:
            mx_atom = getattr(self.self_attn.mixer, "q_a_mx_atom", None)
        return mx_atom is not None and bool(mx_atom(num_rows))

    def _snapshot_slot(
        self, block_residual: torch.Tensor, num_snapshots: int
    ) -> Optional[torch.Tensor]:
        """``TRTLLM_K3_ULL_HYGIENE``: on a block-boundary layer, the bank slot this layer's
        snapshot goes to, for the fused pre-attention mix to store the prefix sum into (no copy)."""
        if (
            not _K3_ULL_HYGIENE
            or self.layer_idx % self.attn_res_block_size != 0
            or num_snapshots >= block_residual.shape[0]
        ):
            return None
        return block_residual[num_snapshots]

    def _capture_hidden_states(self, capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The DSpark tap of ``capture[1]``; on the model's side stream with
        ``TRTLLM_K3_ULL_HYGIENE`` (joined at the end of the model forward, before the drafter)."""
        side = self._side_copies
        if side is None or not side.active():
            capture[0].maybe_capture_hidden_states(capture[1], value, None)
            return
        side.run(lambda: capture[0].maybe_capture_hidden_states(capture[1], value, None), value)

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
        ``_DeferredMoEResidual`` (``TRTLLM_K3_ATTN_RES_PDL``).

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
        attn_input_mxfp8 = None
        if isinstance(hidden_states, _DeferredRpTail):
            # The previous layer's row-parallel MoE tail: this mix finishes its all-reduce where
            # it can; the DSpark tap path materializes it first.
            if capture is None and num_snapshots > 0:
                rows = hidden_states.residual.shape[0]
                mxfp8 = self._attn_takes_mxfp8(rows, attn_metadata)
                fused = _k3_attn_res_pdl_rp_tail(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                    # TRTLLM_K3_ULL_TAIL_SANDWICH: release the staging KDA QKVG GEMM at kernel
                    # entry, as the MoE-residual mix does.
                    entry_trigger=(_K3_ULL_TAIL_SANDWICH or _K3_ULL2_COLL_CP_TAIL)
                    and mxfp8
                    and self.is_kda
                    and self.linear_attn.qkvg_stages_weights(rows),
                    prefix_out=self._snapshot_slot(block_residual, num_snapshots),
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        elif isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                rows = hidden_states.residual.shape[0]
                mxfp8 = self._attn_takes_mxfp8(rows, attn_metadata)
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                    # Release the staging KDA QKVG GEMM at kernel entry.
                    entry_trigger=mxfp8
                    and self.is_kda
                    and self.linear_attn.qkvg_stages_weights(rows),
                    mx_atom=mxfp8 and self._attn_mx_atom(rows),
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None and not _AUX_ATTN_RES_STREAM_ENABLED:
            # prefix_only wants the incoming running prefix, which is already in
            # hand -- no pre-norm intermediate is needed, so the tap is a plain
            # read and this layer keeps the fused attn_res+RMSNorm path like the
            # other 88. ``prefix_sum`` already has the previous layer's MLP
            # output folded in, i.e. the quantity vLLM builds as
            # ``prefix_sum + pending_mlp_out`` on its deferred-add path.
            self._capture_hidden_states(capture, prefix_sum)
            capture = None

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
            snapshot = block_residual[num_snapshots]
            if prefix_sum.data_ptr() != snapshot.data_ptr():
                # (TRTLLM_K3_ULL_HYGIENE: the fused mix above may have stored it there already.)
                snapshot.copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        # When A_mid below adds the attention output to the prefix sum, the attention block may
        # hand back its pushed all-reduce unfinished (a LamportPending) for A_mid to finish and
        # read from the workspace. A_mid must do so before the next all-reduce on the workspace
        # re-arms that buffer: the shared expert, whose down-projection all-reduce is the next
        # one, forks after A_mid, and its pushed finish kernel can only become resident once its
        # producer has passed its own griddepcontrol.wait (every grid before, A_mid included, has
        # then completed).
        attn = self.linear_attn if self.is_kda else self.self_attn
        attn.ar_finish_deferred = prefix_sum is not None and _k3_attn_res_lamport_applies(
            prefix_sum,
            valid_block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.post_attention_layernorm,
        )
        if self.is_kda:
            if attn_input_mxfp8:
                self.linear_attn.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            if attn_input_mxfp8:
                self.self_attn.mixer.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.self_attn(hidden_states, attn_metadata)
        attn.ar_finish_deferred = False

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
            # Where the MoE front's head is the next launch after A_mid, A_mid releases it at entry.
            releases_early = (
                getattr(self.block_sparse_moe, "ull_front_releases_early", None)
                if self.is_moe
                else None
            )
            front_early = releases_early is not None and releases_early(prefix_sum)
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
                entry_trigger=front_early,
            )
        if self.is_moe and _K3_FUSE_MOE_RESIDUAL:
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                all_rank_num_tokens,
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
                # The row-parallel tail serves generation-only batches.
                rp_tail_ok=getattr(attn_metadata, "num_contexts", 1) == 0,
            )
            return prefix_sum, num_snapshots
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states, all_rank_num_tokens)
        else:
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


# TRTLLM_K3_ULL4_MLA_PLAN_EARLY=1: in a generation-only step (multi-stream) the model writes the
# FP4 MLA step metadata (generation lengths + packed split plan, device page table) on a side
# stream at the step start; the first FP4 MLA layer joins it instead of launching those kernels on
# its critical path. The same kernels with the same inputs: bitwise.
_K3_ULL4_MLA_PLAN_EARLY = _F.ull4_mla_plan_early


class KimiK3DefaultModelingV2(DecoderModel):
    # TRTLLM_K3_ULL_HYGIENE: side-stream copies of the decode step (``None``: off).
    _side_copies: Optional[_SideCopies] = None

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

        # The whole vocabulary on every rank: a row gather (``embedding_gather`` with the profile's
        # ``embedding_gather``, ``F.embedding`` otherwise).
        self.embed_tokens = EmbeddingTable(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiK3DefaultDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        # TRTLLM_K3_ULL_HYGIENE: one side stream for the DSpark taps of every layer.
        self._side_copies = _SideCopies(torch.cuda.Stream()) if _K3_ULL_HYGIENE else None
        for layer in self.layers:
            layer._side_copies = self._side_copies
        # TRTLLM_K3_ULL4_MLA_PLAN_EARLY: the side stream and events of the step-metadata prelaunch.
        self._mla_plan_stream = torch.cuda.Stream() if _K3_ULL4_MLA_PLAN_EARLY else None
        self._mla_plan_events = (
            (torch.cuda.Event(), torch.cuda.Event()) if _K3_ULL4_MLA_PLAN_EARLY else None
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
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    def _prelaunch_mla_step_metadata(self, attn_metadata: AttentionMetadata) -> None:
        """TRTLLM_K3_ULL4_MLA_PLAN_EARLY: fork the step's FP4 MLA metadata kernels to a side stream
        here; the first FP4 MLA layer joins it (``step_metadata_ready``)."""
        if not do_multi_stream():
            return
        mixer = None
        for layer in self.layers:
            if layer.is_kda:
                continue
            candidate = getattr(layer.self_attn, "mixer", None)
            mqa = getattr(candidate, "mqa", None)
            if getattr(mqa, "uses_fp4_mla_attention", False) and hasattr(
                candidate, "prelaunch_step_metadata"
            ):
                mixer = candidate
                break
        if mixer is None:
            return
        fork, done = self._mla_plan_events
        fork.record()
        with torch.cuda.stream(self._mla_plan_stream):
            fork.wait()
            mixer.prelaunch_step_metadata(attn_metadata)
            done.record()
        mixer.step_metadata_ready = done

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
            weight = self.embed_tokens.weight
            if _F.embedding_gather and input_ids.dim() == 1 and weight.is_contiguous():
                # The F.embedding rows, copied with 16-byte accesses.
                inputs_embeds = gather_embedding(input_ids, weight)
            else:
                inputs_embeds = F.embedding(input_ids, weight)
        hidden_states = inputs_embeds

        # The FP4 MLA decode plan of the previous step is stale.
        fp4_mla_decode.begin_step(attn_metadata)
        if self._mla_plan_stream is not None and attn_metadata.num_contexts == 0:
            self._prelaunch_mla_step_metadata(attn_metadata)

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
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
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
        if self._side_copies is not None:
            # TRTLLM_K3_ULL_HYGIENE: the DSpark taps land before anything after the model (the
            # drafter) reads the capture buffer.
            self._side_copies.join()
        return output


class KimiK3DefaultForCausalLM(
    KimiK3DefaultTextWeights, SpecDecOneEngineForCausalLM[KimiK3DefaultModelingV2, Any]
):
    """The Kimi K3 text model (KDA + MLA + MoE decoder layers) of ``KimiK3Sm107ForConditionalGeneration``.

    Registered under no architecture: the target class builds it as its ``llm``. Weight loading
    is the ``KimiK3DefaultTextWeights`` mixin's (``weights.py``).

    Speculative decoding: ``SpecDecOneEngineForCausalLM`` builds the drafter and the worker;
    :meth:`_install_private_spec` replaces them with the target's DSpark drafter and worker
    (``drafter.py``, ``dflash.py``)."""

    mamba_metadata_cls = K3MambaMetadata

    @property
    def runtime_hooks(self) -> Optional[RuntimeHooks]:
        """The base hooks plus the ones the target's profile turns on: the speculative decode
        step's head in one kernel (``embedding_gather``, ``decode_prologue``) and the decode
        step's small input uploads, the one-model sampler's Philox seeds and offsets included,
        as one upload and one scatter at the graph replay (``pack_step_inputs``,
        ``pack_spec_rng_uploads``)."""
        features = _features()
        changes = {}
        if features.embedding_gather:
            changes["decode_prologue"] = spec_decode_head
        if features.pack_step_inputs:
            changes["pack_step_inputs"] = True
            changes["pack_spec_rng_uploads"] = True
        hooks = super().runtime_hooks
        if not changes:
            return hooks
        return replace(hooks or DEFAULT_RUNTIME_HOOKS, **changes)

    def __init__(self, model_config: ModelConfig):
        mlp_fp8 = os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV, "0")
        if mlp_fp8 not in ("0", "1"):
            raise ValueError(
                f"{_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV} must be 0 or 1; got {mlp_fp8!r}"
            )
        self._fp8_weight_read_moe_mlp = mlp_fp8 == "1"
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)

        # Helix: swap in the repurposed mapping; restored after super().__init__.
        self._setup_helix_mappings(model_config, cfg, spec_config)
        # Supported spec-dec modes:
        # - SA (suffix automaton): one-engine in-forward drafting, no draft
        #   weights; the KDA/MLA verify paths below implement multi-token
        #   verification for it.
        # - DFlash: external-drafter parallel drafting; the drafter is a
        #   separate dense checkpoint (K2.7-Code-DFlash schema) consumed by
        #   the generic DFlashForCausalLM wrapper, and the target only has
        #   to expose per-layer hidden states via maybe_capture_hidden_states
        #   (see KimiK3DefaultModelingV2.forward).
        # - DSpark: the same external-drafter flow with the Markov and
        #   confidence heads enabled (RadixArk/Kimi-K3-DSpark and friends).
        #   The target side is identical -- the capture in
        #   KimiK3DefaultModelingV2.forward is unconditional -- so this gate is the
        #   only place the mode has to be admitted.
        # Modes needing draft heads (MTP/Eagle) are blocked until a
        # draft-head checkpoint exists.
        assert (
            spec_config is None
            or spec_config.spec_dec_mode.is_sa()
            or spec_config.spec_dec_mode.is_dflash()
            or spec_config.spec_dec_mode.is_dspark()
        ), "Kimi K3 supports speculative decoding only with SA, DFlash or DSpark"
        super().__init__(
            KimiK3DefaultModelingV2(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The drafter and worker of main's shell are built (under the repurposed helix mapping,
        # which is still active here); the package's own replace them at this one point.
        self._install_private_spec(model_config)

        # Restore the CP original: executor-side helix bookkeeping keys off
        # has_cp_helix() at runtime.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def _install_private_spec(self, model_config: ModelConfig) -> None:
        """Swap in the target's DSpark drafter and worker (``drafter.py``, ``dflash.py``).

        Called once from ``__init__``, right after ``SpecDecOneEngineForCausalLM.__init__`` and
        before the helix CP mapping is restored, i.e. with the same mapping main's drafter was
        built under. At that point main's shell has set ``self.spec_config``,
        ``self.draft_config``, ``self.draft_model`` (from the draft-builder registry, or None for a
        separate draft engine), ``self.spec_worker`` (``mapping_for_full_rows()`` mapping,
        ``set_target_model(self)`` done), ``self.use_separate_draft_kv_cache`` and ``self.epilogue``.

        A private implementation must replace ``draft_model`` / ``spec_worker`` consistently:
        swap them in ``self.epilogue`` (main appends both there), call ``set_target_model(self)`` on the
        new worker, keep ``self.lm_head`` a stock ``LMHead`` and ``self.logits_processor`` exactly a
        ``LogitsProcessor`` (main's vocab-parallel selection checks the type), and keep the
        contract main's ``DFlashSpecMetadata.prepare()`` has with the worker (the attributes it
        reads and the slot methods it calls, listed in ``drafter.py``).

        One-engine DSpark with a standalone MLA drafter gets the package's drafter
        (``drafter.py``) and worker (``drafter.py`` on ``dflash.py``)
        through :func:`~.drafter.install_k3_dspark`, which keeps that contract; every other
        configuration keeps main's.
        """
        install_k3_dspark(self, model_config)

    def _setup_helix_mappings(
        self,
        model_config: ModelConfig,
        cfg: "PretrainedConfig",
        spec_config: Optional["DecodingBaseConfig"],
    ) -> None:
        """Validate helix preconditions and stage the dual-mapping swap.

        DeepseekV3 pattern: the MLA layers keep the CP original; everything
        else is built against the repurposed mapping (CP ranks become TP
        ranks). Sets ``mapping_with_cp`` (restored after construction) and
        ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
        None outside helix.
        """
        self.mapping_with_cp = None
        self._repurposed_tp_mapping = None
        if not model_config.mapping.has_cp_helix():
            return
        # The K3 helix targets run head-TP attention, the residual stream and the MoE on every
        # row of the CP group; there is no token-sliced (attention-DP) variant.
        if model_config.mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 helix runs without attention DP: KDA and MLA are head-TP over the CP "
                "group, so there is no request-level DP to add."
            )
        if spec_config is not None:
            # Helix supports only the standalone DSpark drafter (verified on
            # the V2 superblock ledger); reject everything else loudly rather
            # than let an unsupported spec mode run silently wrong.
            if not spec_config.spec_dec_mode.is_dspark():
                raise ValueError(
                    "Kimi K3 helix supports speculative decoding only with "
                    f"DSpark (standalone drafter); got "
                    f"{spec_config.decoding_type!r}."
                )
            # The SpeculationGate acceptance-rate trip permanently disables
            # speculation mid-flight while enable_spec_decode stays True;
            # in-flight helix requests then fall into the plain generation
            # loop whose position math (total_input_len_cp +
            # py_decoding_iter - 1) is stale once any draft token was
            # accepted -> silently wrong RoPE positions and KV slots. Reject
            # the trip wires until that loop is helix-group aware.
            if (
                spec_config.acceptance_rate_window_size is not None
                or spec_config.acceptance_rate_threshold is not None
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "acceptance-rate gate (acceptance_rate_window_size / "
                    "acceptance_rate_threshold): dynamically disabling "
                    "speculation mid-flight leaves helix requests on a "
                    "single-token position formula."
                )
            # max_concurrency is the same trip wire by another name: the
            # drafter re-evaluates should_use_spec_decode on every scheduling
            # iteration and flips enable_spec_decode off as soon as the active
            # batch exceeds the cap. In-flight helix requests then take the
            # plain generation loop, whose position formula counts ITERATIONS
            # (total_input_len_cp + py_decoding_iter - 1) rather than
            # committed tokens, so it is stale by however many draft tokens
            # were accepted -- a wrong RoPE position, and across a ledger page
            # boundary a KV write to the wrong CP rank. Mirror the drafter's
            # own "unset" test (Drafter.should_use_spec_decode returns True
            # when max_concurrency is None) so an unset value is not rejected.
            if spec_config.max_concurrency is not None:
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "concurrency cutoff (max_concurrency): disabling "
                    "speculation above the cap leaves in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration, which accepted draft "
                    "tokens break."
                )
            # draft_len_schedule is the user-facing alternative to
            # max_concurrency (llm_args rejects setting both) and reaches the
            # same end by a route that does not go through
            # should_use_spec_decode at all: py_executor turns speculation off
            # directly once the schedule yields draft_len 0 for the active
            # batch size. Guarding only max_concurrency would leave this door
            # open. Skip the schedule that llm_args synthesized from
            # max_concurrency, so a config that set only that field raises the
            # message above naming the field the user actually wrote.
            if (
                spec_config.draft_len_schedule is not None
                and not spec_config._translated_from_max_concurrency
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the dynamic draft-length "
                    "schedule (draft_len_schedule): a batch size past the "
                    "last entry drops the draft length to 0 and turns "
                    "speculation off mid-run, leaving in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration."
                )
        cp = model_config.mapping.cp_size
        repurposed_tp = model_config.mapping.tp_size * cp
        if cfg.num_attention_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
                f"to divide the MLA head count ({cfg.num_attention_heads})."
            )
        kda_heads = cfg.linear_attn_config["num_heads"]
        if kda_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) to "
                f"divide the KDA head count ({kda_heads})."
            )
        # MoE splits apply to the repurposed tp*cp group (helix
        # moe_world_size = tp*cp); default EP-only. The Mapping constructor
        # skips its product check when both sizes are 1, so validate here.
        moe_ep = repurposed_tp
        if model_config.mapping.moe_tp_ep_user_specified:
            moe_tp = model_config.mapping.moe_tp_size
            moe_ep = model_config.mapping.moe_ep_size
            if moe_tp * moe_ep != repurposed_tp:
                raise ValueError(
                    f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                    f"moe_expert_parallel_size ({moe_ep}) must equal "
                    f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                    "repurposed tp*cp group."
                )
        if cfg.num_experts and cfg.num_experts % moe_ep != 0:
            raise ValueError(
                f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
                f"divide the routed expert count ({cfg.num_experts}): each "
                "EP rank of the repurposed tp*cp group holds whole experts."
            )
        self.mapping_with_cp = copy.deepcopy(model_config.mapping)
        repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
        # repurpose passes resolved moe sizes, which the Mapping constructor
        # mistakes for user-specified values; restore the flag.
        repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
        # load_weights shard selection must use this tp_rank; the restored
        # CP original's tp_rank is 0 on every rank.
        self._repurposed_tp_mapping = repurposed
        model_config._frozen = False
        model_config.mapping = repurposed
        # Side-channel for the MLA layers (avoids threading a kwarg through
        # every intermediate signature).
        model_config._helix_mapping_with_cp = self.mapping_with_cp
        model_config._frozen = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - Keep block reuse opt-in because its combination with SA
        #   speculative decoding is not validated. Both settings use V2
        #   with the default use_kv_cache_manager_v2=True. The "auto"
        #   setting follows the model preference and can resolve to V1.
        # - tokens_per_block=64: keep the validated K3 page size. Earlier
        #   releases needed it to avoid a legacy MLA fallback that could not
        #   serve attention-DP's 96 query heads. The current FlashInfer
        #   TRTLLM-Gen backend declines H=96 at either page size, allowing
        #   the updated legacy TRTLLM-Gen autotuner to select a Q16 kernel.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            }
        }

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


class KimiK3Sm107ForConditionalGeneration(KimiK3DefaultWeights, PreTrainedModel):
    """Kimi K3 NVFP4 on SM107, sharded as the engine's mapping says.

    The vision-language model: the vision tower (``_VISION_MODEL_CLS``, built in
    :meth:`load_weights` when meta init deferred it) and the text model
    ``KimiK3DefaultForCausalLM`` as ``llm``. The speculative-decoding and weight-loading
    attributes the engine reads off the outer model (``draft_model``, ``lm_head``,
    ``runtime_hooks``, ...) are the text model's. Weight loading is the
    ``KimiK3DefaultWeights`` mixin's (``weights.py``).
    """

    _LANG_PREFIX = "language_model."
    _VISION_MODEL_CLS = KimiK3VisionModel
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
        """Default to the Python (v2) KV-cache transceiver (as Kimi K2.5).

        Only adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL; otherwise the C++ transceiver is used.
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

        # --- Vision encoder (deferred under MetaInitMode, recreated in
        # load_weights) ---
        self.mm_encoder = None
        if not DISAGG:
            try:
                mm_encoder = self._VISION_MODEL_CLS(model_config)
                if _has_meta_tensors(mm_encoder):
                    logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")
                else:
                    self.mm_encoder = mm_encoder
            except MetaInitException:
                logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")

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

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", _MEDIA_PLACEHOLDER_TOKEN_ID
        )
        # Backing storage for the ``mm_token_ids`` property below — built once
        # here so the executor doesn't re-allocate per step.
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

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
    def runtime_hooks(self) -> Optional[RuntimeHooks]:
        """The text model's step-loop hooks: the engine reads them off the outer model."""
        return self.llm.runtime_hooks

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "image.image_grid_thw",
            "video.pixel_values_videos",
            "video.video_grid_thw",
            "multimodal_embedding",
        ]

    @property
    def mm_token_ids(self) -> torch.Tensor:
        """Surface the in-vocab media placeholder to the model engine so
        ``runners.prepare_multimodal_indices`` selects the ``torch.isin`` predicate
        instead of the OOV (``>= vocab_size``) fallback (which would miss
        Kimi's placeholder and force ``fuse_input_embeds`` through the
        ``torch.where`` host-sync path on GPU input_ids).
        """
        return self._mm_token_ids

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
        multimodal_params = multimodal_params or []
        mm_embeds: List[torch.Tensor] = []
        mm_token_ids = None

        if len(multimodal_params) > 0:
            if DISAGG:
                raise NotImplementedError("Disaggregated inference not yet supported for K2.5.")
            mm_ctx_params = multimodal_params[: attn_metadata.num_contexts]
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self.mm_encoder.forward,
                multimodal_params=mm_ctx_params,
            )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_ctx_params)

            # ``runners.prepare_multimodal_indices`` sees Kimi's in-vocab
            # placeholder via ``self.mm_token_ids`` and emits indices that match
            # ``find_input_mm_embeds``'s active-chunk slice; a chunked-prefill /
            # KV-reuse mismatch surfaces as a row-count error inside
            # ``fuse_input_embeds`` itself.
            if len(mm_embeds) > 0:
                mm_token_ids = self.mm_token_ids.to(input_ids.device, dtype=input_ids.dtype)
        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=mm_token_ids,
            mm_token_indices=kwargs.get("mm_token_indices"),
            text_token_indices=kwargs.get("text_token_indices"),
        )

        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            **kwargs,
        )


__all__ = ["KimiK3DefaultForCausalLM", "KimiK3Sm107ForConditionalGeneration"]
