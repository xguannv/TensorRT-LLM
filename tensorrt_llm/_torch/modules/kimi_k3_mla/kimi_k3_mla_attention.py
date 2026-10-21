# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 specialization of the shared PyTorch-backend MLA module.

The base ``MLA`` class owns context attention, cached/chunked prefill,
absorbed generation, and paged-cache handling. This module only supplies the
K3 projection topology, NoPE identity table, KV-B checkpoint layout, and
gated output projection.
"""

from __future__ import annotations

import copy
import os
from typing import Callable, Optional, Tuple

import torch

from ....functional import PositionEmbeddingType
from ....logger import logger
from ....mapping import Mapping
from ....models.modeling_utils import QuantConfig
from ....quantization import QuantAlgo
from ...attention_backend import AttentionInputType, AttentionMetadata, TrtllmAttention
from ...attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from ...model_config import ModelConfig
from ...utils import AuxStreamType, is_nvfp4_marlin_enabled
from ..linear import Linear, TensorParallelMode
from ..mla import MLA, helix_gen_zero_kv_mask
from ..mla_fp8_q_absorb import mla_fp8_q_absorb_nope
from ..multi_stream_utils import do_multi_stream
from ..rms_norm import RMSNorm

_KIMI_K3_MLA_GEN_BACKEND_ENV = "TLLM_K3_MLA_GEN_BACKEND"

# TRTLLM_K3_MLA_GATE_FP8=1: when o_proj (a K3 FP8 weight-read projection) would quantize its BF16
# input 1x128 for its GEMM, the output-gate multiply emits that FP8 operand directly
# (output_gate_fp8.gate_mul_fp8_packed: sigmoid, multiply and quantization in one kernel, the same
# bytes), and o_proj runs on it.
_K3_MLA_GATE_FP8 = os.environ.get("TRTLLM_K3_MLA_GATE_FP8", "0") == "1"

# TRTLLM_K3_MLA_GATE_SIDE=1: in a generation-only Helix batch (V up-projection before the
# all-to-all) whose o_proj runs the low-M GEMM, the Helix zero-KV mask and the output gate
# sigmoid(g_proj(x)), which only depend on the step metadata and the layer input, run on a
# side stream; o_proj then multiplies the all-to-all's combined output by the gate while
# quantizing it (``k3_fp8w_lowm_gemm_gated``) and reads that output where the combine left
# it. After the combine only o_proj remains. Bit-exact.
_K3_MLA_GATE_SIDE = os.environ.get("TRTLLM_K3_MLA_GATE_SIDE", "0") == "1"
# TRTLLM_K3_MLA_Q_NORM_PDL=1: the q_a RMSNorm releases the q_b GEMM at launch instead of after
# its store (``q_norm_pdl``; same kernel code, bit-exact), so q_b streams its weights while
# q_a and the norm still run.
_K3_MLA_Q_NORM_PDL = os.environ.get("TRTLLM_K3_MLA_Q_NORM_PDL", "0") == "1"
# TRTLLM_K3_MLA_KV_SIDE=1 (with TRTLLM_K3_MLA_GATE_SIDE): in that generation path the KV half
# (kv_a RMSNorm, latent concat, KV append + decode scheduler buffers) runs on the attention aux
# stream from the q_a output to the decode, and the main stream (q_a RMSNorm, q_b, q absorb)
# joins it only before the decode; the generic path joins after each step, which keeps q_b
# from launching before the kv_a RMSNorm ends. Same kernels and arguments, bit-exact.
_K3_MLA_KV_SIDE = os.environ.get("TRTLLM_K3_MLA_KV_SIDE", "0") == "1"
# TRTLLM_K3_MLA_KV_SIDE_PRIO=1: run that KV half on a high-priority stream of its own, so its
# small kernels are dispatched ahead of grids that were launched early (PDL) and still wait
# for shared memory (the q absorb behind q_b).
_K3_MLA_KV_SIDE_PRIO = os.environ.get("TRTLLM_K3_MLA_KV_SIDE_PRIO", "0") == "1"
# TRTLLM_K3_MLA_FUSED_V_UP=1: with the V up-projection before the Helix all-to-all, the CuTe DSL
# FP8 decode applies it inside its balanced-split reduction (one kernel instead of the reduction
# and a BMM, bitwise the same output) up to 32 rows per head (batch * tokens per request, see
# mla_decode_fp8_v_up.MAX_ROWS); larger batches and the other attention backends keep the BMM.
_K3_MLA_FUSED_V_UP = os.environ.get("TRTLLM_K3_MLA_FUSED_V_UP", "0") == "1"
# TRTLLM_K3_MLA_KV_FUSED=1 (with TRTLLM_K3_MLA_KV_SIDE): the KV half of that path (kv_a RMSNorm,
# latent row, KV append with the decode's scale and scheduler buffers) is one kernel
# (``k3_mla_kv_append``) instead of the RMSNorm, the k_pe copy and ``mla_rope_generation``; it
# writes the same bytes.
_K3_MLA_KV_FUSED = os.environ.get("TRTLLM_K3_MLA_KV_FUSED", "0") == "1"
# TRTLLM_K3_MLA_GATE_EARLY=1 (with TRTLLM_K3_MLA_GATE_SIDE and the fused V up-projection): enqueue
# the output-gate GEMM where it is forked, before the all-to-all, instead of after the all-to-all
# and its combine. Scheduling only, bit-exact.
_K3_MLA_GATE_EARLY = os.environ.get("TRTLLM_K3_MLA_GATE_EARLY", "0") == "1"
# TRTLLM_K3_MLA_KV_JOIN_EARLY=1 (with TRTLLM_K3_MLA_KV_FUSED): the main stream joins the KV side
# stream before the q absorb instead of before the decode, so the decode's only dependency is the
# q absorb on its own stream (a programmatic dependent that also waits for another stream loses
# its early launch). The fused KV half ends long before q_b does. Scheduling only, bit-exact.
_K3_MLA_KV_JOIN_EARLY = os.environ.get("TRTLLM_K3_MLA_KV_JOIN_EARLY", "0") == "1"
# TRTLLM_K3_MLA_GATE_EXCLUSIVE=1 (with TRTLLM_K3_MLA_GATE_SIDE): the output-gate GEMM (low-M sigmoid
# GEMM, k3_fp8w_lowm_gemm_sigmoid) launches with the full per-block shared memory, so no CTA of
# another kernel shares an SM with one of its CTAs: it runs beside the Helix all-to-all on SMs of
# its own instead of on SMs that also hold all-to-all CTAs. The same GEMM, bit for bit. Meant for
# the fused all-to-all (TRTLLM_HELIX_A2A_FUSED, fewer CTAs than SMs); beside the FIFO all-to-all
# (one CTA on every SM) its CTAs would wait for the all-to-all to end.
_K3_MLA_GATE_EXCLUSIVE = os.environ.get("TRTLLM_K3_MLA_GATE_EXCLUSIVE", "0") == "1"
# TRTLLM_K3_MLA_GATE_SIDE_PRIO=1 (with TRTLLM_K3_MLA_GATE_SIDE): the zero-KV mask and the output
# gate run on a high-priority stream of their own, so the gate GEMM's CTAs are dispatched ahead
# of grids that are launched early (PDL) and wait for SMs (o_proj behind the all-to-all).
# Scheduling only: bit-exact.
_K3_MLA_GATE_SIDE_PRIO = os.environ.get("TRTLLM_K3_MLA_GATE_SIDE_PRIO", "0") == "1"
# TRTLLM_K3_MLA_PT_PREFETCH=1 (with TRTLLM_K3_MLA_KV_FUSED): the fused KV append also brings the
# page-table rows of the batch's KV and the KV lengths into L2 (bulk prefetches) for the decode,
# whose first loads read them. A hint only: bit-exact.
_K3_MLA_PT_PREFETCH = os.environ.get("TRTLLM_K3_MLA_PT_PREFETCH", "0") == "1"
# One side stream per device and role, shared by all MLA layers (their side work never
# overlaps).
_GATE_SIDE_STREAMS: dict[tuple[int, bool, str], torch.cuda.Stream] = {}


def _gate_side_stream(
    device: torch.device, high_priority: bool = False, role: str = "side"
) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, high_priority, role)
    stream = _GATE_SIDE_STREAMS.get(key)
    if stream is None:
        priority = torch.cuda.Stream.priority_range()[1] if high_priority else 0
        stream = torch.cuda.Stream(device=index, priority=priority)
        _GATE_SIDE_STREAMS[key] = stream
    return stream


# Deliberate divergence from upstream, which defaults this to "cute-dsl".
#
# On the FlashInfer/CuTe-DSL pair this branch currently ships, the CuTe-DSL MLA
# decode path cannot initialize at all. flashinfer 0.6.15 unpacks the MLA
# scheduler params one MLIR value per field
# (flashinfer/cute_dsl/attention/monolithic/mla_helpers.py), while the installed
# nvidia-cutlass-dsl-internal 0.3.0+20260529 serializes FastDivmodDivisor as two
# SSA values (cutlass issue #3243). Every rank dies during executor init with
#
#     ValueError: FastDivmodDivisor expects exactly 2 MLIR values
#                 (encoded divisor + scalar divisor SSA), got 1.
#
# which gives no hint that the MLA backend selection is what caused it.
#
# This is a package-pairing problem, not an architecture limit: nothing on that
# path inspects the SM version. So this is a default flip rather than a hard
# block -- set TLLM_K3_MLA_GEN_BACKEND=cute-dsl to opt back in wherever the
# FlashInfer and CuTe-DSL builds do agree (for example the K3 FlashInfer fork
# used on GB300). Restore the upstream default once the shipped pair is aligned.
_KIMI_K3_MLA_GEN_BACKEND_DEFAULT = "trtllm-gen"


def _select_mla_generation_backend(quant_config: Optional[QuantConfig]) -> str:
    """Select K3's absorbed-generation MLA backend.

    FP8 KV cache requires the native CuTe-DSL wrapper, which accepts its
    device scales; TRTLLM-Gen does not support K3's 96 query heads. Other
    KV-cache dtypes keep the existing configured/default backend selection
    unchanged.

    The non-FP8 default is TRTLLM-Gen on this branch; see
    ``_KIMI_K3_MLA_GEN_BACKEND_DEFAULT`` for why.
    """
    backend = os.environ.get(_KIMI_K3_MLA_GEN_BACKEND_ENV, _KIMI_K3_MLA_GEN_BACKEND_DEFAULT)
    has_fp8_kv_cache = bool(
        quant_config is not None and quant_config.layer_quant_mode.has_fp8_kv_cache()
    )
    if has_fp8_kv_cache and backend != "cute-dsl":
        logger.info(
            "Kimi K3 MLA: FP8 KV cache requires the CuTe-DSL MLA "
            f"generation backend; overriding '{backend}' -> 'cute-dsl'."
        )
        return "cute-dsl"
    return backend


def _kimi_k3_mla_decode_backend_policy(
    requested_backend: str,
    metadata,
    num_tokens: int,
) -> str:
    """Per-batch MLA decode backend selection for Kimi K3 with non-FP8 KV.

    Installed as ``mla_backend_policy`` on K3's generation attention backend
    only when the KV cache is not FP8 (see :class:`KimiK3MLAAttention`). The
    general attention code applies no such policy on its own.

    CuTe-DSL reuses one staged page table across MLA layers for a
    generation-only, one-token-per-request batch. A mixed context/generation
    batch cannot use that reuse key, so selecting CuTe-DSL would repeat the
    staging copies in every MLA layer and regress time to first token. Keep
    the requested CuTe-DSL fast path for plain decode and use TRTLLM-Gen for
    mixed batches and speculative multi-token verification. For H=96, the
    FlashInfer wrapper declines that effective TRTLLM-Gen backend so FMHA
    selection can continue to the legacy TRTLLM-Gen path.
    """
    is_single_token_generation = num_tokens == metadata.num_generations
    if requested_backend == "cute-dsl" and (
        metadata.num_contexts > 0 or not is_single_token_generation
    ):
        return "trtllm-gen"
    return requested_backend


def _make_pos_embd_params(
    *,
    qk_rope_head_dim: int,
    max_position_embeddings: int,
) -> PositionalEmbeddingParams:
    """Build a valid rope config so the backend allocates a real cache.

    We use rope_gpt_neox with default theta=10000 and ``duplicate_data
    =True`` (the same convention DeepSeek-V3-style MLA uses when
    ``qk_rope_head_dim`` is present). The resulting ``rotary_cos_sin``
    has the exact shape the C++ MLA rope kernel indexes. Immediately
    after backend construction we overwrite the tensor values with
    ``(cos=1, sin=0)`` — an identity rotation, matching K3's NoPE.
    """
    rope_params = RopeParams(
        dim=qk_rope_head_dim,
        theta=10000.0,
        max_positions=max_position_embeddings,
        original_max_positions=max_position_embeddings,
        duplicate_data=True,
    )
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        # Match the working DeepSeek-V3-style MLA reference test
        # (tests/unittest/_torch/attention/test_attention_mla.py) which
        # sets ``is_neox=False``. The MLA fused rope kernel is GPT-J
        # style regardless of this flag, but the C++ FMHA reads this bit
        # elsewhere and stability under identity-cos-sin depends on the
        # standard non-neox layout.
        is_neox=False,
    )


def _write_identity_rope_values(cos_sin: torch.Tensor) -> None:
    """Overwrite a rotary cos/sin table with identity values in place.

    Interleaved (cos, sin) pairs: index [::2] = cos, [1::2] = sin.
    Setting cos=1 and sin=0 per position makes the rotation the
    identity — a mathematical no-op — which preserves K3's NoPE
    semantics without patching the backend.
    """
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    # Ensure the identity write reaches CUDA memory before any kernel
    # launched from a different stream can read the table.
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)


def _install_identity_rope_table(backend: TrtllmAttention) -> None:
    """Install an identity rotary cos/sin table on ``backend``.

    The C++ MLA rope kernels (``mla_rope_generation`` and the context
    preprocess) read this table and apply the rotation; identity values
    make that a copy, preserving K3's NoPE.

    The tensor SHAPE produced by ``create_rope_const_params`` is kept
    intact so the C++ ``float2`` indexing stays valid. Only the values
    are overwritten in place. ``_ensure_rope_table_size`` is replaced
    with an identity-preserving resize: the table may GROW (so the
    fused rope-generation op can never index out of bounds for long
    sequences) but its values are always rewritten to identity right
    after a regeneration, so the real sinusoids never leak in.
    """
    cos_sin = backend.rotary_cos_sin
    if cos_sin is None:
        raise RuntimeError(
            "backend.rotary_cos_sin is None after construction; check "
            "pos_embd_params has a valid RopeParams with dim > 0."
        )
    _write_identity_rope_values(cos_sin)

    orig_resize = backend._ensure_rope_table_size  # bound method

    def _identity_preserving_resize(required_max_positions: int) -> None:
        if required_max_positions <= backend.rope_params.max_positions:
            return
        orig_resize(required_max_positions)
        _write_identity_rope_values(backend.rotary_cos_sin)

    backend._ensure_rope_table_size = _identity_preserving_resize


# ---------------------------------------------------------------------------
# KimiK3MLAAttention.
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(MLA):
    """Kimi K3 MLA implemented as a thin specialization of :class:`MLA`.

    K3 keeps the standard dense MLA attention/cache flow and only changes the
    checkpoint projection topology, positional encoding, KV-B runtime layout,
    and gated output projection.
    """

    # TRTLLM_K3_EXPERT_L2_PREFETCH: set by the decoder layer to its MoE block's
    # issue_expert_l2_prefetch, called with the token count once q_b and the q absorption are
    # issued (generation path with the KV half on a side stream).
    expert_l2_prefetch_hook: Optional[Callable[[int], None]] = None

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
    ) -> None:
        quant_config = model_config.get_quant_config()
        has_fp8_kv_cache = bool(
            quant_config is not None and quant_config.layer_quant_mode.has_fp8_kv_cache()
        )
        projection_configs = model_config.quant_config_dict or {}
        q_a_config = projection_configs.get("q_a_proj", model_config.quant_config)
        kv_a_config = projection_configs.get("kv_a_proj_with_mqa", model_config.quant_config)
        q_a_algo = q_a_config.quant_algo if q_a_config else None
        kv_a_algo = kv_a_config.quant_algo if kv_a_config else None
        fuse_qkv_a_proj = q_a_algo == kv_a_algo and (
            q_a_algo is None or (q_a_algo == QuantAlgo.FP8_BLOCK_SCALES and q_lora_rank % 128 == 0)
        )
        # Resolve each Linear's recipe before allocating weights, as in the
        # model's layerwise quantization post-init. Shared MLA stays generic.
        deferred_config = copy.copy(model_config)
        deferred_config._frozen = False
        deferred_config.skip_create_weights_in_init = True
        deferred_config._frozen = model_config._frozen
        pos_embd_params = _make_pos_embd_params(
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=False,
            config=deferred_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            reduce_output=False,
            fuse_qkv_a_proj=fuse_qkv_a_proj,
            rms_norm_eps=rms_norm_eps,
            flashinfer_mla_backend=_select_mla_generation_backend(quant_config),
        )
        # K3 calls forward_impl() directly to insert its output gate before
        # the base row-parallel o_proj. The original executor metadata remains
        # intact, so MLA performs its native mixed context/generation split.
        self.register_to_config = False

        self.use_output_gate = use_output_gate

        if use_output_gate:
            # The gate must match o_proj's input sharding (under helix the
            # post-all-to-all 1/cp head chunk); outside helix this equals
            # q_b_proj's head sharding, replicated under attention-DP.
            self.g_proj = Linear(
                hidden_size,
                num_heads * v_head_dim,
                bias=False,
                dtype=dtype,
                mapping=self.o_proj.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=(model_config.quant_config_dict or {}).get(
                    "g_proj", model_config.quant_config
                ),
                skip_create_weights_in_init=True,
                allreduce_strategy=model_config.allreduce_strategy,
                force_dynamic_quantization=model_config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

        for name, module in self.named_children():
            if isinstance(module, Linear):
                module.quant_config = projection_configs.get(name, model_config.quant_config)
        if not model_config.skip_create_weights_in_init:
            self.create_weights()
            for module in self.children():
                if isinstance(module, Linear):
                    module.create_weights()

        # K3 is NoPE. The base MLA backends still require real RoPE tables, so
        # retain their expected shape and replace every rotation with identity.
        assert isinstance(self.mha, TrtllmAttention)
        assert isinstance(self.mqa, TrtllmAttention)
        _install_identity_rope_table(self.mha)
        _install_identity_rope_table(self.mqa)
        # Only the absorbed-generation backend (mqa) requests CuTe-DSL, so
        # only it needs K3's per-batch fallback policy. FP8 KV must keep its
        # required CuTe-DSL backend because TRTLLM-Gen does not support K3's
        # 96 query heads, so leave the policy unset and preserve the static
        # backend for every batch composition. Use the config-derived flag:
        # TrtllmAttention has not populated has_fp8_kv_cache yet when weight
        # creation is deferred under MetaInitMode.
        if not has_fp8_kv_cache:
            self.mqa.mla_backend_policy = _kimi_k3_mla_decode_backend_policy
        self.rotary_emb = None
        self.apply_rotary_emb = False
        # NoPE: with an FP8 KV cache, build the FP8 decode query in one kernel
        # instead of a BF16 absorption BMM plus its FP8 re-read in the RoPE
        # kernel (opt-in: TRTLLM_K3_MLA_FP8_Q_ABSORB=1).
        self.fp8_nope_q_absorb = os.environ.get("TRTLLM_K3_MLA_FP8_Q_ABSORB", "0") == "1"
        # TRTLLM_K3_MLA_GATE_SIDE: side-stream fork / join of the zero-KV mask and the gate.
        self._gate_events = tuple(torch.cuda.Event() for _ in range(4))
        # TRTLLM_K3_MLA_KV_SIDE: fork of the KV half after q_a, join before the decode.
        self._kv_events = (torch.cuda.Event(), torch.cuda.Event())
        self.helix_v_up_in_attention = _K3_MLA_FUSED_V_UP
        # Set by the decoder layer for the next forward only: the MXFP8 form ``(xq, xs)`` of the
        # layer input (``trtllm::k3_attn_res_*_mx``), which the fused q_a / kv_a projection
        # consumes instead of quantizing the BF16 rows itself (see q_a_takes_mxfp8).
        self.next_input_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def q_a_takes_mxfp8(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a projection of ``num_rows`` rows can take their MXFP8 form
        (``next_input_mxfp8``): it runs the low-M kernel on them."""
        takes = getattr(self.kv_a_proj_with_mqa, "takes_mxfp8_input", None)
        return (
            self.fuse_qkv_a_proj
            and not self.is_lite
            and takes is not None
            and bool(takes(num_rows))
        )

    def _project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection; on ``next_input_mxfp8`` when the decoder layer
        handed it over (``forward_mxfp8``: bitwise the projection of the BF16 rows)."""
        mxfp8_input, self.next_input_mxfp8 = self.next_input_mxfp8, None
        if mxfp8_input is not None:
            return self.kv_a_proj_with_mqa.forward_mxfp8(hidden_states, *mxfp8_input)
        return super()._project_qkv_a(hidden_states)

    def _apply_output_gate_and_o_proj(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_output_gate:
            gate = self.g_proj(hidden_states)
            takes_fp8 = getattr(self.o_proj, "takes_fp8_packed_input", None)
            if (
                _K3_MLA_GATE_FP8
                and takes_fp8 is not None
                and attn_out.dim() == 2
                and attn_out.dtype == torch.bfloat16
                and attn_out.shape[-1] % 128 == 0
                and takes_fp8(attn_out.shape[0])
            ):
                from .output_gate_fp8 import gate_mul_fp8_packed

                return self.o_proj.forward_fp8_packed(*gate_mul_fp8_packed(attn_out, gate))
            attn_out = attn_out * gate.sigmoid()
        return self.o_proj(attn_out)

    @staticmethod
    def _runs_flashinfer_rmsnorm(norm) -> bool:
        """Whether ``norm(x)`` runs FlashInfer's BF16 RMSNorm (no residual, no fused quant)."""
        return (
            type(norm) is RMSNorm
            and not (norm.use_cuda_tile or norm.use_gemma or norm.is_nvfp4)
            and not norm.return_hp_output
            and IS_FLASHINFER_AVAILABLE
            and not is_nvfp4_marlin_enabled()
        )

    def _q_a_layernorm_maybe_fused(self, q: torch.Tensor, return_norm_out: bool = False):
        norm = self.q_a_layernorm
        if (
            _K3_MLA_Q_NORM_PDL
            and not return_norm_out
            and self._runs_flashinfer_rmsnorm(norm)
            and get_env_enable_pdl()
            and self._resolve_qa_fused_scale() is None
        ):
            # Where the base path runs FlashInfer's RMSNorm, run the same kernel with its PDL
            # trigger at launch (bit-exact; see ``q_norm_pdl``).
            from .q_norm_pdl import can_use, rmsnorm_release_at_launch

            if can_use(q, norm.weight):
                return rmsnorm_release_at_launch(q, norm.weight, norm.variance_epsilon)
        return super()._q_a_layernorm_maybe_fused(q, return_norm_out)

    def _side_gate_applies(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> bool:
        """Whether this forward takes the ``TRTLLM_K3_MLA_GATE_SIDE`` path."""
        if not (_K3_MLA_GATE_SIDE and self.use_output_gate):
            return False
        if (
            type(hidden_states) is not torch.Tensor
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.dim() != 2
            or hidden_states.shape[0] != attn_metadata.num_tokens
        ):
            return False
        if attn_metadata.num_contexts != 0 or self.sparse_attn_hooks is not None:
            return False
        if not self._helix_v_up_enabled() or not hasattr(self.o_proj, "gated_applies"):
            return False
        from ...cute_dsl_kernels.blackwell.k3_fp8w_lowm_gemm import low_m_max_for

        return hidden_states.shape[0] <= low_m_max_for(
            self.o_proj.out_features, self.o_proj.in_features
        )

    def _kv_side_applies(self, multi_stream: bool) -> bool:
        """Whether the generation-only forward can run :meth:`_forward_gen_kv_side`: the fused
        q_a / kv_a projection, an FP8 KV cache with the FP8 q absorb, NoPE, and the aux
        stream in use."""
        w_kc = getattr(self, "k_b_proj_trans", None)
        if w_kc is not None and w_kc.dtype != torch.bfloat16:
            w_kc = self.k_b_proj_trans_dequant
        return (
            _K3_MLA_KV_SIDE
            and multi_stream
            and self.aux_stream is not None
            and not self.is_lite
            and self.fuse_qkv_a_proj
            and self.fp8_nope_q_absorb
            and not self.apply_rotary_emb
            and not self.llama_4_scaling
            and self._helix_v_up_enabled()
            and isinstance(self.mqa, TrtllmAttention)
            and getattr(self.mqa, "has_fp8_kv_cache", False)
            and not self.mqa.uses_fp4_mla_attention
            and w_kc is not None
            and w_kc.dtype == torch.bfloat16
            and self._fp8_q_unit_scale is not None
        )

    def _latent_row(self, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        """``torch.concat([kv_a_layernorm(compressed_kv), k_pe], -1)``, bit for bit; where the
        norm is FlashInfer's, it writes straight into the concatenated row and ``k_pe`` is
        copied beside it (a tiny copy instead of the batched concat kernel)."""
        norm = self.kv_a_layernorm
        if not (self._runs_flashinfer_rmsnorm(norm) and compressed_kv.dtype == torch.bfloat16):
            return torch.concat([norm(compressed_kv), k_pe], dim=-1)
        from flashinfer.norm import rmsnorm

        lora = compressed_kv.shape[-1]
        latent = torch.empty(
            (compressed_kv.shape[0], lora + k_pe.shape[-1]),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        rmsnorm(
            compressed_kv,
            norm.weight,
            norm.variance_epsilon,
            out=latent[:, :lora],
            enable_pdl=get_env_enable_pdl(),
        )
        latent[:, lora:].copy_(k_pe)
        return latent

    @staticmethod
    def _flashinfer_cute_norm(device: torch.device) -> bool:
        """Whether FlashInfer's ``rmsnorm`` runs its CuTe DSL ``RMSNormKernel`` on ``device``
        (the arithmetic ``k3_mla_kv_append`` reproduces), not its CUDA JIT kernel."""
        try:
            from flashinfer.norm import _use_cuda_norm
        except ImportError:
            return False
        return not _use_cuda_norm(device)

    def _kv_fused_applies(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> bool:
        """Whether the KV half of :meth:`_forward_gen_kv_side` can run as one
        ``k3_mla_kv_append`` kernel (``TRTLLM_K3_MLA_KV_FUSED``): the norm the default path runs
        is FlashInfer's, the batch is generation only with whole requests, the cache is FP8
        without a quantization scale and the rotary table is the backend's FP32 one."""
        if not _K3_MLA_KV_FUSED or not self._runs_flashinfer_rmsnorm(self.kv_a_layernorm):
            return False
        if not self._flashinfer_cute_norm(compressed_kv.device):
            return False
        from ...cute_dsl_kernels.blackwell import k3_mla_kv_append as kv_append

        mqa = self.mqa
        num_gen = attn_metadata.num_generations
        return (
            kv_append.can_run(compressed_kv, k_pe, self.kv_a_layernorm.weight)
            and attn_metadata.num_contexts == 0
            and num_gen > 0
            and compressed_kv.shape[0] % num_gen == 0
            and self.kv_lora_rank == kv_append.KV_LORA_RANK
            and self.qk_rope_head_dim == kv_append.ROPE_DIM
            and getattr(mqa, "_nvfp4_mla_kv_scale_orig_quant", None) is None
            and getattr(mqa, "rotary_cos_sin", None) is not None
            and mqa.rotary_cos_sin.dtype == torch.float32
            and attn_metadata.kv_cache_manager is not None
        )

    def _kv_append_fused(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
    ) -> torch.Tensor:
        """``k3_mla_kv_append`` for :meth:`_forward_gen_kv_side`: returns the latent row and fills
        the KV cache and the scale / scheduler buffers as ``_latent_row`` followed by
        ``mla_rope_generation`` (``q_rope_applied``) do. The page pool and page table are viewed
        exactly as the CuTe DSL decode views them (``CuteDslMlaFmha._run_mla_decode``)."""
        from ...cute_dsl_kernels.blackwell import k3_mla_kv_append as kv_append

        mqa = self.mqa
        md = attn_metadata
        num_tokens = compressed_kv.shape[0]
        num_gen = md.num_generations
        mqa._ensure_rope_table_size(md.max_seq_len)
        kv_pool = md.kv_cache_manager.get_buffers(mqa.layer_idx)
        packed_block = 1
        for size in kv_pool.shape[1:]:
            packed_block *= size
        block_stride = kv_pool.stride(0)
        layers_in_pool = block_stride // packed_block if packed_block else 1
        page_offset = 0
        if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
            page_offset = kv_pool.storage_offset() // packed_block
            kv_pool = kv_pool.as_strided(
                (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
                (packed_block, *kv_pool.stride()[1:]),
                0,
            )
        kv_pages = kv_pool.view(torch.float8_e4m3fn)[
            :, 0, :, 0, : self.kv_lora_rank + self.qk_rope_head_dim
        ]
        pool_idx = int(md.host_kv_cache_pool_mapping[mqa.get_local_layer_idx(md), 0])
        page_table = md.kv_cache_block_offsets[pool_idx, :, 0, :]
        helix_slots = helix_inactive = None
        if getattr(md, "_helix_spec_tokens_valid", False):
            helix_slots = md.helix_local_slots
        else:
            helix_inactive = getattr(md, "helix_is_inactive_rank", None)
        latent = torch.empty(
            (num_tokens, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        kv_append.k3_mla_kv_append(
            compressed_kv,
            k_pe,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            latent=latent,
            kv_lens=md.kv_lens_cuda_runtime[:num_gen],
            seq_len=num_tokens // num_gen,
            num_heads=mqa.num_heads,
            cos_sin=mqa.rotary_cos_sin,
            positions=getattr(md, "helix_position_offsets", None),
            helix_slots=helix_slots,
            helix_inactive=helix_inactive,
            page_table=page_table,
            page_offset=page_offset,
            kv_pages=kv_pages,
            fmha_scheduler_counter=fmha_scheduler_counter,
            bmm1_scale=mla_bmm1_scale,
            bmm2_scale=mla_bmm2_scale,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            host_bmm1=kv_append.host_bmm1_scale(
                mqa.q_scaling, mqa.qk_nope_head_dim + mqa.qk_rope_head_dim
            ),
            prefetch_decode=_K3_MLA_PT_PREFETCH,
        )
        return latent

    def _forward_gen_kv_side(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Absorbed Helix generation with the KV half on the aux stream; returns the combined
        all-to-all output (``num_tokens x num_heads_tp_cp * v_head_dim``).

        The calls and their arguments are those of ``MLA._forward_impl`` and
        ``forward_absorption_generation`` for this configuration; only the stream order
        changes. The KV append (``mla_rope_generation`` with ``q_rope_applied``) reads only the
        latent row and the cache metadata; ``fused_q`` and ``q_pe`` are passed for their shape
        and strides, which is why a ``q_pe``-shaped placeholder stands in for the q_b output it
        precedes. Its scheduler / scale buffers are allocated on the aux stream that writes
        them, and the main stream joins before the decode reads them.
        """
        num_tokens = attn_metadata.num_tokens
        heads, lora, rope = self.num_heads_tp, self.kv_lora_rank, self.qk_rope_head_dim
        device = hidden_states.device
        w_kc_bf16 = (
            self.k_b_proj_trans
            if self.k_b_proj_trans.dtype == torch.bfloat16
            else self.k_b_proj_trans_dequant
        )
        q, compressed_kv, k_pe = self._project_qkv_a(hidden_states).split(
            [self.q_lora_rank, lora, rope], -1
        )
        fused_q = torch.empty(
            [num_tokens, heads, lora + rope], dtype=hidden_states.dtype, device=device
        )
        quant_q_buffer = torch.empty(
            num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
        )
        kv_fork, kv_ready = self._kv_events
        kv_fork.record()
        kv_stream = (
            _gate_side_stream(device, high_priority=True)
            if _K3_MLA_KV_SIDE_PRIO
            else self.aux_stream
        )
        with torch.cuda.stream(kv_stream):
            kv_fork.wait()
            fused_kv = self._kv_fused_applies(compressed_kv, k_pe, attn_metadata)
            if not fused_kv:
                latent_cache = self._latent_row(compressed_kv, k_pe)
            num_seqs = attn_metadata.num_seqs
            cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
            cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
            fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
            if fused_kv:
                latent_cache = self._kv_append_fused(
                    compressed_kv,
                    k_pe,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                )
            else:
                q_pe_shape = torch.empty(
                    [num_tokens, heads, self.qk_head_dim], dtype=hidden_states.dtype, device=device
                )[..., self.qk_nope_head_dim :]
                self.mqa.mla_rope_generation(
                    fused_q,
                    q_pe_shape,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                    quant_scale_qkv=self._fp8_q_unit_scale,
                    q_rope_applied=True,
                )
            kv_ready.record()
        join_early = fused_kv and _K3_MLA_KV_JOIN_EARLY
        q = self._q_a_layernorm_maybe_fused(q)
        q = self.q_b_proj(q)
        q_pe = q.view([-1, heads, self.qk_head_dim])[..., self.qk_nope_head_dim :]
        if join_early:
            kv_ready.wait()
        mla_fp8_q_absorb_nope(
            q.view([-1, heads, self.qk_head_dim]),
            w_kc_bf16,
            quant_q_buffer,
            self.qk_nope_head_dim,
            rope,
        )
        if self.expert_l2_prefetch_hook is not None:
            self.expert_l2_prefetch_hook(num_tokens)
        if not join_early:
            kv_ready.wait()
        attn_out = self._attn_forward_gen(
            self.mqa,
            fused_q.view([num_tokens, heads * (lora + rope)]),
            None,
            None,
            None,
            attn_metadata,
            helix_v_up=True,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,
            q_pe=q_pe,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
        )
        assert attn_out.shape == (num_tokens, self.num_heads_tp_cp * self.v_head_dim)
        return attn_out

    def _forward_side_gate(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Generation forward with the zero-KV mask and the output gate on a side stream.

        The mask (one small kernel) is forked once the layer input is written; the gate (one
        GEMM with a sigmoid epilogue) is forked when the all-to-all is enqueued, so it runs on
        the SMs the all-to-all leaves idle instead of beside the q_a / q_b GEMMs. When the
        attention applied the V up-projection itself (TRTLLM_K3_MLA_FUSED_V_UP), the gate GEMM
        is enqueued only after the all-to-all and its combine (still depending only on the
        attention): enqueued before them, it takes the SMs the fused reduction leaves free and
        the all-to-all launches later. TRTLLM_K3_MLA_GATE_EARLY enqueues it where it is forked
        in that case too. The main stream waits for the mask where the all-to-all reads it and
        for the gate just before o_proj, where both are long done.
        """
        num_tokens = hidden_states.shape[0]
        mask_fork, mask_ready, gate_fork, gate_ready = self._gate_events
        multi_stream = do_multi_stream()
        side = None
        if multi_stream:
            side = (
                _gate_side_stream(hidden_states.device, high_priority=True, role="gate")
                if _K3_MLA_GATE_SIDE_PRIO
                else _gate_side_stream(hidden_states.device)
            )
        gates = []
        mask_joined = [not multi_stream]

        def gate_of(x):
            forward_sigmoid = getattr(self.g_proj, "forward_sigmoid", None)
            if forward_sigmoid is not None:
                if _K3_MLA_GATE_EXCLUSIVE:
                    from ...cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

                    return forward_sigmoid(x, smem_request=k3_fp8w_lowm_gemm._SMEM_CAPACITY)
                return forward_sigmoid(x)
            return self.g_proj(x).sigmoid()

        if multi_stream:
            mask_fork.record()
            with torch.cuda.stream(side):
                mask_fork.wait()
                zero_kv_mask = helix_gen_zero_kv_mask(attn_metadata, num_tokens)
                mask_ready.record()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(attn_metadata, num_tokens)

        def zero_kv_mask_fn():
            if not mask_joined[0]:
                mask_ready.wait()
                mask_joined[0] = True
            return zero_kv_mask

        late_enqueue = [False]

        def enqueue_gate():
            with torch.cuda.stream(side):
                gate_fork.wait()
                gates.append(gate_of(hidden_states))
                gate_ready.record()

        def pre_exchange(v_up_in_attention: bool):
            if multi_stream:
                gate_fork.record()
                if v_up_in_attention and not _K3_MLA_GATE_EARLY:
                    late_enqueue[0] = True
                else:
                    enqueue_gate()
            else:
                gates.append(gate_of(hidden_states))

        attn_outputs = None
        self.helix_zero_kv_mask_fn = zero_kv_mask_fn
        self.helix_pre_exchange_fn = pre_exchange
        self.defer_helix_gen_output = True
        try:
            attn_out = None
            if self._kv_side_applies(multi_stream):
                attn_out = self._forward_gen_kv_side(hidden_states, attn_metadata)
            else:
                attn_outputs = self._create_outputs(hidden_states, attn_metadata)
                super().forward_impl(None, hidden_states, attn_metadata, attn_output=attn_outputs)
                attn_out = self.helix_gen_output
        finally:
            self.helix_zero_kv_mask_fn = None
            self.helix_pre_exchange_fn = None
            self.defer_helix_gen_output = False
            self.helix_gen_output = None
        if late_enqueue[0]:
            enqueue_gate()
        if attn_out is None:
            attn_out = attn_outputs[0]
        if not mask_joined[0]:
            mask_ready.wait()
        if gates:
            gate = gates[0]
            if multi_stream:
                gate_ready.wait()
        else:
            gate = gate_of(hidden_states)
        if self.o_proj.gated_applies(attn_out, gate):
            return self.o_proj.forward_gated(attn_out, gate)
        return self.o_proj(attn_out * gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        try:
            return self._forward(hidden_states, attn_metadata)
        finally:
            # consumed by the projection; never carried into another forward
            self.next_input_mxfp8 = None

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self._side_gate_applies(hidden_states, attn_metadata):
            return self._forward_side_gate(hidden_states, attn_metadata)
        # _create_outputs() rather than create_output(): the base implementation
        # takes a list so a sparse-attention backend can append its own buffers,
        # and it routes through the sparse hooks when they are installed. The
        # dense path this module uses is element 0.
        attn_outputs = self._create_outputs(hidden_states, attn_metadata)
        super().forward_impl(
            None,
            hidden_states,
            attn_metadata,
            attn_output=attn_outputs,
        )
        return self._apply_output_gate_and_o_proj(hidden_states, attn_outputs[0])
