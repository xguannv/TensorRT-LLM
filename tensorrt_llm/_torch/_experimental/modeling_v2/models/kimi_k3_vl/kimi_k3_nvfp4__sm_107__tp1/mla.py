# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA on Rubin: K3-owned weights and explicit kernel chains.

K3 MLA is DeepSeek-V3-style multi-latent attention with K3's deltas:

* **NoPE.** The MLA kernels' RoPE table is an identity table, so their rotation is a copy.
* **Output gate before ``o_proj``.** ``sigmoid(g_proj(hidden_states)) * attn_output``.
* **Softmax scale** ``(qk_nope + qk_rope) ** -0.5 = 192 ** -0.5`` (``q_scaling`` 1).

This module holds the weights and the forward of every phase (``K3MLAAttention``), the context
and generation attention chains, K3's split-K context FMHA, the fused Helix tail and the layer's
runtime (``K3MLARuntime``); ``ops/attention/_fp4_mla_decode`` holds K3's FP4 decode and
``kernels/mla/q_norm_pdl`` the PDL RMSNorm.
The engine owns the KV cache managers; the attention calls over them are the family ops of
``ops/attention/k3_mla_{attention,cache}``, which read the pools from the attention metadata.

Kimi K3 MLA: K3-owned weights and the explicit kernel chain of every phase.

Weights (``layers`` containers, the checkpoint loader's names): the fused q_a / kv_a projection
``kv_a_proj_with_mqa``, ``q_a_layernorm``, ``q_b_proj`` (head-sharded), ``kv_a_layernorm``,
``kv_b_proj`` (head-sharded, ``[all K | all V]`` rows), the output gate ``g_proj`` and ``o_proj``
(sharded over TP x CP), and the absorption weights the loader derives from kv_b
(``k_b_proj_trans`` / ``v_b_proj``, their FP8 scales and BF16 dequantized copies, and under Helix
``helix_v_b_proj_full``). FP8 projections are ``Fp8BlockScaleLinear`` until the loader swaps them
for its FP8 weight-read GEMM; BF16 projections are ``Bf16Linear``. The context and generation
attention roles are ``K3MlaAttnState`` (``ops/attention/k3_mla_state``; ``mha``: 192-dim context
heads, ``mqa``: the 576-dim absorbed latent head), whose RoPE table is an identity table (K3 is
NoPE).

One forward of ``T`` tokens (the first ``T_ctx`` context rows):

1. **Front**: the fused q_a / kv_a projection (on the MXFP8 layer input the decoder layer hands
   over, where its GEMM takes it; a BF16 one is ``trtllm::cublas_mm``), the q_a RMSNorm
   (FlashInfer, or with ``K3Sm107Constants.mla_q_norm_pdl`` its PDL-at-launch variant) beside the kv_a
   RMSNorm on the attention aux stream, then q_b (the context rows of a Helix rank: its head range
   only) beside the latent row ``concat(kv_a_norm, k_pe)``.
2. **Context** rows: ``forward_context``.
3. **Generation** rows: ``forward_absorption_generation``.
4. **Tail**: ``o_proj(attn * sigmoid(g_proj(x)))``; the TP / CP sum of o_proj's partials is the
   caller's (``K3MLARuntime``).

Generation-only Helix batches of at most 128 tokens take the side-gate forward (the zero-KV mask
and the output gate on a side stream, o_proj gated in its GEMM); FP4-KV layers of at most 16 tokens
the fused query (``forward_gen_q_fused``).

The attention part runs as :func:`mla_custom_op_inplace` under ``eager_on_graph`` (a breakable CUDA
graph breaks around it) where the layer is registered in the model's ``mla_layers``.

Kimi K3 MLA generation: the absorbed decode as one kernel chain per profile.

Every function takes the layer (:class:`~.kimi_k3_mla_attention.K3MLAAttention`) for its
weights and attention backends. The chain of one generation batch of ``T`` tokens:

* **W_UK and the cache update** (:func:`forward_absorption_generation`): the absorption BMM of the
  query's NoPE half into the latent query (BF16 ``trtllm::bmm_out`` on ``k_b_proj_trans``, or
  ``torch.bmm`` on its dequantized copy for FP8 KV-B) beside
  ``mla_rope_generation`` (identity RoPE, the cache append and, on an FP4 cache, the FP4 query
  quantization), on the attention aux stream for a BF16 cache.
* **Attention** (:func:`gen_attention`): K3's packed FP4 decode (``ops/attention/_fp4_mla_decode``) on an FP4
  cache with the CuTe DSL FP4 MLA backend; on an FP8 or BF16 cache the CuTe DSL MLA decode where it
  serves the batch, else the attention op's MLA generation (``mla_decode``).
* **Helix CP** (:func:`attn_forward_gen`): the V up-projection of this rank's partials
  (``trtllm::bmm_out`` on ``helix_v_b_proj_full``), then the fifo-v2 MNNVL all-to-all
  (``HelixAllToAllNative.alltoall_native``, which zeroes the rows this rank holds no KV for) and
  ``trtllm::helix_post_process_native``; or, where ``helix_tail`` serves the batch, the decode
  with its split reduction deferred and one ``k3_helix_tail`` kernel (fold, V up-projection,
  exchange, combine).
* **W_UV** without Helix: ``trtllm::bmm_out`` on ``v_b_proj`` (BF16) or ``torch.bmm`` on
  ``v_b_proj_dequant`` (FP8) into the layer output.

The fused query (:func:`forward_gen_q_fused`, generation-only Helix batches of FP4-KV layers): the
fused q_a / kv_a projection, then ``k3_mla_latent_row`` and the KV-only FP4 cache append
(``fp4_mla_kv_append``) on the aux stream beside ``k3_mla_q_fused`` (q_a RMSNorm, q_b, W_UK, RoPE
and the FP4 query quantization in one kernel), then the decode and the Helix tail as above.

The Kimi-K3 Helix generation tail as one kernel.

In a Helix generation batch whose V up-projection runs before the exchange,
:class:`K3HelixTail` replaces the packed split-KV reduction,
the V up-projection BMM, the FIFO all-to-all and the combine by one launch of
``cute_dsl_kernels/blackwell/k3_helix_tail.py``: the decode's FP32 split partials are folded by
the senders (the reduction is deferred with ``DeferredPackedReduce``), every head's rows are
projected and pushed into the head owner's Lamport receive buffer (MNNVL fabric memory), and the
receivers combine them into the ``[T, H / P * 128]`` o_proj input. Decode backends that store
their rows directly (no split partials, the FP8 layers) take the same kernel without the fold.

Sizes and features:

* ``_TAIL_MB``: receive buffer per rank (two halves), 16 MiB; a batch whose records do not fit a
  half takes the unfused path. The kernel runs one CTA per SM: every CTA folds, projects and
  sends, then combines tokens.
* o_proj's operand: where the side-stream output gate is joined before the tail, the receivers
  also store the MXFP8 form of ``bf16(out * gate)`` (``mx_out``), and o_proj runs the MXFP8
  low-M GEMM with the all-reduce push epilogue on it (bitwise the gated low-M GEMM + push).

Kimi K3 MLA context attention: one kernel chain per cache state.

Every function takes the layer (:class:`~.kimi_k3_mla_attention.K3MLAAttention`) for its
weights and its context attention role ``mha`` (``K3MlaAttnState``, ``H`` context heads of
192 / 128 dims). :func:`forward_context` picks the chain:

* **No cached prefix** (:func:`context_default`): kv_b over the batch's latent rows (a BF16 kv_b on
  SM100f: two per-head ``trtllm::bmm_out`` writing K-nope into the strided K buffer and V into a
  ``[K-nope | V]``-strided buffer; otherwise the kv_b GEMM and a copy of K-nope into K), then
  the context FMHA with ``latent_cache`` (``mla_context_attention``), which applies the identity
  RoPE, appends the cache and attends; on an FP4 cache ``fp4_mla_context_attention`` (the FP8 context
  FMHA over the step's FP8 scratch cache beside the FP4 scatter). Owner-only Helix prefill appends the KV it owns first
  (``mla_rope_append_paged_kv_assign_q``) and passes K with its K-pe and no ``latent_cache``.
* **Chunked prefill over an FP8 / BF16 cache** (:func:`context_chunked_prefill`):
  ``mla_rope_append_paged_kv_assign_q``; then per cached-prefix partition
  ``load_chunked_kv_cache_for_mla``, the K / V expansion and ``mla_context_attention`` with a full
  mask into a temporary output merged with ``merge_attention_for_mla``; the current chunk last,
  causal. On an FP8 cache with the FP8 context FMHA (``fp8_cached_kv_context``) Q is quantized once
  (``static_quantize_e4m3_per_tensor``) and each expansion is kv_b + ``mla_kv_to_fp8`` (FP8 K / V in
  one pass), handed to the op as already-quantized inputs; otherwise kv_b and the BF16 K-nope /
  K-pe concat.
* **Cached prefix without chunked prefill** (:func:`context_cached_kv`): the same expansion over the
  whole paged cache (``load_paged_kv_cache_for_mla``) in one ``mla_context_attention``.
* **Chunked prefill over an FP4 cache**: the Rubin FP8 split-K context FMHA
  (``forward_context_split_k_fp4``).

Kimi K3 MLA cached-prefix context on Rubin: the FP8 split-K context FMHA.

The chunked prefill of an MLA layer with an NVFP4 latent cache (:func:`forward_context_split_k_fp4`,
where :func:`uses_k3_split_k_context` holds: SM107, matching Q / KV heads, head dimensions
128 / 64 / 128, the CuTe DSL kernel imports, BF16 Q and output without an output scale):

* ``append_fp4_mla_context_cache`` rotates Q and the current K (identity RoPE)
  and appends the chunk to the FP4 cache;
* each cached-prefix partition, and then the current chunk (causal), runs the Rubin FP8 split-K
  context FMHA (``ops/attention/k3_context_fmha``, ``kernels/k3_context_fmha.py``), which reads
  K-nope ``[T, H, 128]``, the shared K-pe ``[T, 1, 64]`` and V as they are and merges the
  partition into the output in its epilogue (``merge_op_tensor`` selects copy / merge / skip per
  request);
* its inputs are unit-scale E4M3: Q once per layer call, each partition's kv_b output and K-pe.
  The Triton kernels of ``ops/attention/static_quantize_e4m3`` quantize Q and the K-pe rows (the
  current chunk's read in place from ``latent_cache``; ``static_quantize_e4m3_per_tensor`` for K-pe
  rows they do not support), with the bytes of ``static_quantize_e4m3_per_tensor``;
* where kv_b is the FP8 block-scale GEMM with its weight scale in the SM107 CuTe MXFP8 layout
  (:func:`kv_b_takes_mxfp8`), each cached-prefix partition is gathered from the FP4 cache straight
  into MXFP8 (``ops/attention/fp4_mla_gather_mxfp8``), the current chunk is quantized with
  ``fp8_quantize_1x128_packed_ue8m0``, and kv_b runs as ``cute_dsl_mxfp8_gemm_rubin`` writing FP8
  (the BF16 rounding at unit output scale); otherwise the BF16 gather and kv_b run and the output is
  quantized with ``static_quantize_e4m3_per_tensor``.

Kimi K3 NVFP4 on Rubin (SM107): the MLA layer.

``K3MLARuntime`` holds an MLA decoder layer's attention module (``mla.K3MLAAttention``)
and applies its TP output reduction. Also here: the quantization of a checkpoint attention
projection (``resolve_attention_quant_config``), the MLA layers whose KV cache is FP8 under
an NVFP4 KV cache, and the RoPE table size override.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Callable, Dict, Optional, Tuple

import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.attention.backends import AttentionMetadata, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    FP4_MLA_K_RESIDUAL_DIM,
    _fp4_mla_attention_backend,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _validate_fp4_mla_kv_storage_shape,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _materialize_fp4_mla_device_page_table_for_forward,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.distributed import HelixAllToAllNative
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from tensorrt_llm._torch.utils import maybe_compiled_cat, maybe_compiled_copy_
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger

from ..common.multi_stream import do_multi_stream
from ..kernels.k3r_mla_ctx import fp8_static_quant
from ..kernels.k3r_mla_ctx.mla_kv_fp8 import mla_kv_to_fp8
from ..ops.attention import _fp4_mla_decode as fp4_mla_decode
from ..ops.attention import k3_fp4_mla_decode as _decode_op
from ..ops.attention import k3_mla_attention as mla_attn
from ..ops.attention import k3_mla_cache as mla_cache
from ..ops.attention.k3_context_fmha import k3_context_fmha
from ..ops.attention.k3_mla_state import K3MlaAttnState
from ..ops.attention.static_quantize_e4m3 import static_quantize_e4m3
from ..sm107_parts.constants import K3Sm107Constants
from ..sm107_parts.layers import fork_join, rms_norm

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

import copy
import weakref

from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch import kimi_k3_cache_policy as _kimi_k3_cache_policy
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.custom_ops import BufferKind
from tensorrt_llm._torch.distributed.mnnvl_memory import HelixCpMnnvlMemory, MnnvlMemory
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common.parallel import TensorParallelMode
from ..sm107_parts.layers import (
    Bf16Linear,
    Fp8BlockScaleLinear,
    LamportPending,
    RMSNormWeight,
    TpAllReduce,
    finish_lamport_push,
    lamport_push_target,
    lamport_push_tensor,
)

# Head dimensions (K-nope, K-pe, V) the split-K context FMHA is built for.
_SPLIT_K_HEAD_DIMS = (128, 64, 128)


def split_k_context_unavailable_reason(attn) -> Optional[str]:
    """Why ``attn`` cannot run the Rubin FP8 split-K context FMHA, or None if it can."""
    sm_version = get_sm_version()
    if sm_version != 107:
        return f"requires SM107, got SM{sm_version}"
    if attn.num_heads <= 0 or attn.num_heads != attn.num_kv_heads:
        return (
            "requires matching positive Q/KV head counts per rank, "
            f"got {attn.num_heads}/{attn.num_kv_heads}"
        )
    head_dims = (attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim)
    if head_dims != _SPLIT_K_HEAD_DIMS:
        return f"requires head dimensions (K-nope, K-pe, V)={_SPLIT_K_HEAD_DIMS}, got {head_dims}"
    from ..kernels.k3r_mla_ctx import k3_context_fmha_op

    if k3_context_fmha_op.IS_K3_CONTEXT_FMHA_AVAILABLE:
        return None
    if k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR is not None:
        return f"CuTe DSL kernel import failed: {k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR!r}"
    return "CuTe DSL kernel lacks split-K epilogue merge support"


def supports_k3_fp8_split_k_context(attn) -> bool:
    """Whether the attention backend ``attn`` can run the Rubin FP8 split-K context FMHA."""
    reason = split_k_context_unavailable_reason(attn)
    if reason is None:
        return True
    logger.info_once(
        f"Kimi K3 MLA: Rubin FP8 split-K context FMHA disabled: {reason}; NVFP4 KV-cache layers "
        "cannot attend a cached prefix (chunked prefill raises).",
        key=("k3_mla_ctx_split_k_unavailable", reason),
    )
    return False


def run_k3_split_k_context_partition(
    attn,
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_stats: torch.Tensor,
    merge_op: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    *,
    max_q_len: int,
    max_kv_len: int,
    is_causal: bool,
) -> None:
    """Attend FP8 Q to one FP8 KV partition and merge it into ``output`` / ``softmax_stats``.

    ``q`` ``[T_q, H * 192]``, ``k_nope`` / ``v`` ``[T_kv, H * 128]`` with unit inner stride,
    ``k_pe`` ``[T_kv, 64]``, ``output`` ``[T_q, H * 128]`` BF16, ``softmax_stats``
    ``[T_q, H, 2]`` FP32; the softmax scale is the attention backend's.
    """
    from ..kernels.k3r_mla_ctx import k3_context_fmha_op

    sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + attn.qk_rope_head_dim) ** 0.5)
    num_heads = attn.num_heads
    hints = "on" if k3_context_fmha_op.K3_CONTEXT_FMHA_SCHEDULING_HINTS else "off"
    logger.info_once(
        "Kimi K3 MLA context FMHA: k3_context_fmha (Rubin FP8 split-K, epilogue "
        f"merge) serves {'causal current-chunk' if is_causal else 'cached-prefix'} partitions, "
        f"{num_heads} heads per rank, CuTe DSL scheduling hints {hints}.",
        key=("k3_mla_ctx_split_k_fmha", is_causal, num_heads),
    )
    k3_context_fmha(
        q.view(q.shape[0], num_heads, 192),
        k_nope.view(k_nope.shape[0], num_heads, 128),
        k_pe.view(k_pe.shape[0], 1, 64),
        v.view(v.shape[0], num_heads, 128),
        output.view(q.shape[0], num_heads, 128),
        cu_q_seqlens,
        cu_kv_seqlens,
        softmax_stats,
        merge_op,
        max_q_len,
        max_kv_len,
        sm_scale,
        is_causal,
    )


def load_fp4_mla_chunked_kv_cache_mxfp8(
    metadata,
    layer_idx: int,
    *,
    num_ctx_cached_tokens: int,
    cu_chunked_seq_len: torch.Tensor,
    chunked_global_offset: torch.Tensor,
    chunked_max_seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Gather one cached-prefix partition of dense FP4 MLA V2 storage as MXFP8 kv_b_proj input.

    The partition and metadata are those of ``load_fp4_mla_chunked_kv_cache``. Returns
    ``((E4M3 [M, 512], UE8M0 [pad_up(M, 128) * 16]), E4M3 [M, 64])``: the BF16-rounded latent
    quantized 1x128 with scales in the K32 R128c4 layout ``cute_dsl_mxfp8_gemm_rubin`` reads, and
    the BF16-rounded RoPE part plus its residual in unit-scale E4M3.
    """
    if kv_lora_rank != 512 or qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "MXFP8 gather requires kv_lora_rank=512 and "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {kv_lora_rank} and {qk_rope_head_dim}."
        )
    if num_ctx_cached_tokens < 0:
        raise ValueError("FP4 MLA chunk gather token count must be non-negative.")
    if chunked_max_seq_len < 0:
        raise ValueError("FP4 MLA chunk gather max sequence length must be non-negative.")
    if num_ctx_cached_tokens > 0 and chunked_max_seq_len == 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires a positive max length.")
    num_contexts = int(metadata.num_contexts)
    if num_ctx_cached_tokens > 0 and num_contexts <= 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires context requests.")
    tensors = (cu_chunked_seq_len, chunked_global_offset)
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA chunk gather metadata must use CUDA tensors.")
    if (
        cu_chunked_seq_len.dtype != torch.int64
        or cu_chunked_seq_len.ndim != 1
        or cu_chunked_seq_len.numel() < num_contexts + 1
        or not cu_chunked_seq_len.is_contiguous()
    ):
        raise ValueError(
            "FP4 MLA chunk gather requires a contiguous int64 cumulative-length tensor."
        )
    if (
        chunked_global_offset.dtype != torch.int64
        or chunked_global_offset.ndim != 1
        or chunked_global_offset.numel() < num_contexts
        or not chunked_global_offset.is_contiguous()
    ):
        raise ValueError("FP4 MLA chunk gather requires contiguous int64 global offsets.")
    if cu_chunked_seq_len.device != chunked_global_offset.device:
        raise ValueError("FP4 MLA chunk gather metadata tensors must share one device.")

    device = cu_chunked_seq_len.device
    compressed_kv = torch.empty(
        (num_ctx_cached_tokens, kv_lora_rank), dtype=torch.float8_e4m3fn, device=device
    )
    compressed_sf = torch.empty(
        (triton.cdiv(num_ctx_cached_tokens, 128) * 128 * (kv_lora_rank // 32),),
        dtype=torch.uint8,
        device=device,
    )
    k_pe = torch.empty(
        (num_ctx_cached_tokens, qk_rope_head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    if num_ctx_cached_tokens == 0:
        return (compressed_kv, compressed_sf), k_pe

    state = metadata.fp4_mla_state
    if not bool(getattr(state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA chunk gather requires fixed-stride device page metadata.")
    _materialize_fp4_mla_device_page_table_for_forward(metadata)
    page_table_stride = int(state.page_table_stride)
    page_ids = state._paged_kv_indices
    if (
        page_table_stride <= 0
        or not isinstance(page_ids, torch.Tensor)
        or page_ids.dtype != torch.int32
        or not page_ids.is_cuda
        or page_ids.device != device
        or page_ids.numel() < num_contexts * page_table_stride
    ):
        raise RuntimeError("FP4 MLA chunk gather received invalid device page metadata.")

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=_fp4_mla_attention_backend(),
    )
    if storage_head_dim != head_dim + FP4_MLA_K_RESIDUAL_DIM:
        raise RuntimeError(
            "FP4 MLA chunk gather requires K residual storage for BF16 reconstruction."
        )
    from ..ops.attention.fp4_mla_gather_mxfp8 import fp4_mla_gather_mxfp8

    logger.info_once(
        "Kimi K3 MLA context gather: cached-prefix chunks are read from the FP4 cache as MXFP8 "
        "latent + FP8 K-pe (fp4_mla_gather_mxfp8).",
        key="k3_mla_ctx_gather_mxfp8",
    )
    fp4_mla_gather_mxfp8(
        compressed_kv,
        compressed_sf,
        k_pe,
        kv_cache,
        sf_cache.view(torch.float8_e4m3fn),
        page_ids,
        cu_chunked_seq_len,
        chunked_global_offset,
        _get_fp4_mla_global_scale(metadata, kv_cache.device),
        chunked_max_seq_len,
        metadata.page_size,
        page_table_stride=page_table_stride,
        num_contexts=num_contexts,
    )
    return (compressed_kv, compressed_sf), k_pe


def uses_k3_split_k_context(layer, q: torch.Tensor, output: torch.Tensor) -> bool:
    """Whether the FP4-cache chunked prefill of ``layer`` runs on the split-K context FMHA: BF16 Q
    and output, an NVFP4 latent cache and :func:`supports_k3_fp8_split_k_context`."""
    mha = layer.mha
    return (
        q.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and layer.out_scale is None
        and isinstance(mha, K3MlaAttnState)
        and mha.uses_fp4_mla_attention
        and supports_k3_fp8_split_k_context(mha)
    )


def kv_b_takes_mxfp8(layer) -> bool:
    """Whether kv_b runs as the SM107 MXFP8 GEMM on an MXFP8 cached-prefix gather.

    Needs SM107 with the CuTe DSL Rubin kernels, the 512 / 64 latent / RoPE dimensions of the
    gather, and a bias-free FP8 block-scale kv_b that takes a prequantized CuTe-layout input, with
    its UE8M0 weight scale in the CuTe MXFP8 layout.
    """
    kv_b_proj = layer.kv_b_proj
    weight = getattr(kv_b_proj, "weight", None)
    weight_scale = getattr(kv_b_proj, "weight_scale", None)
    return (
        get_sm_version() == 107
        and IS_CUTLASS_DSL_RUBIN_AVAILABLE
        and layer.kv_lora_rank == 512
        and layer.qk_rope_head_dim == 64
        and getattr(kv_b_proj, "bias", None) is None
        and getattr(kv_b_proj, "supports_prequantized_input", False)
        and weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale is not None
        and weight_scale.dtype == torch.uint8
    )


def forward_context_split_k_fp4(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: TrtllmAttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP4 MLA chunked prefill of ``layer`` with every partition on the split-K context FMHA."""
    mha = layer.mha
    if latent_cache is None:
        raise RuntimeError("FP4 MLA chunked prefill requires latent_cache.")
    kv_b_mxfp8 = kv_b_takes_mxfp8(layer)
    logger.info_once(
        "Kimi K3 MLA cached-prefix context: NVFP4 KV-cache layers run the Rubin FP8 split-K "
        "context FMHA (k3_context_fmha); cached prefix: "
        + (
            "MXFP8 gather + FP8-output MXFP8 kv_b GEMM"
            if kv_b_mxfp8
            else "BF16 gather + kv_b_proj + FP8 quantization"
        )
        + "; Q FP8 quantization: static_quantize_e4m3 (Triton).",
        key=("k3_mla_ctx_split_k", kv_b_mxfp8),
    )
    lora, rope = layer.kv_lora_rank, layer.qk_rope_head_dim
    heads = layer.num_heads_tp

    # Rotate Q and the current K in place and append the chunk to the FP4 cache; the current
    # chunk's rotated K-pe is then read from latent_cache.
    mla_cache.append_fp4_mla_context_cache(mha, q, latent_cache, attn_metadata)
    current_k_pe = latent_cache.reshape(-1, lora + rope)[:, lora:]
    fp8_unit_scale = mha.kv_cache_scaling_factor
    q_fp8 = static_quantize_e4m3(q.contiguous(), fp8_unit_scale)
    context_cu_q_seqlens = attn_metadata.mla_prepare_ctx_cu_seqlens()
    if context_cu_q_seqlens is None:
        raise RuntimeError("K3 split-K context FMHA requires context requests.")

    # The first partition of every request writes the output and its stats (merge op 2).
    softmax_stats = torch.empty(
        (attn_metadata.num_ctx_tokens, heads, 2), dtype=torch.float, device=q.device
    )
    kv_split = [heads * layer.qk_nope_head_dim, heads * layer.v_head_dim]

    def expand_kv(
        partition_compressed_kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        partition_k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_b_mxfp8:
            if isinstance(partition_compressed_kv, tuple):
                activation, activation_scale = partition_compressed_kv
            else:
                activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
                    partition_compressed_kv.contiguous()
                )
            # The FP8 output keeps the GEMM's BF16 rounding (unit output scale).
            partition_kv = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                activation,
                layer.kv_b_proj.weight,
                activation_scale,
                layer.kv_b_proj.weight_scale,
                output_dtype=torch.float8_e4m3fn,
            )
        else:
            partition_kv, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                layer.kv_b_proj(partition_compressed_kv), fp8_unit_scale
            )
        if partition_k_pe.dtype == torch.float8_e4m3fn:
            k_pe_source = "FP8 from the gather"
        elif fp8_static_quant.supports_rows(partition_k_pe):
            k_pe_source = "static_quantize_e4m3_rows (Triton)"
            partition_k_pe = static_quantize_e4m3(partition_k_pe, fp8_unit_scale, rows=True)
        else:
            k_pe_source = "static_quantize_e4m3_per_tensor"
            partition_k_pe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                partition_k_pe.contiguous(), fp8_unit_scale
            )
        logger.info_once(
            "Kimi K3 MLA context expand: per-chunk kv_b_proj runs as "
            + (
                "cute_dsl_mxfp8_gemm_rubin (MXFP8 in, FP8 out)"
                if kv_b_mxfp8
                else "kv_b_proj (BF16) + static_quantize_e4m3_per_tensor"
            )
            + f"; K-pe: {k_pe_source}; K-nope / K-pe / V go to the split-K FMHA unconcatenated.",
            key=("k3_mla_ctx_split_k_expand", kv_b_mxfp8, k_pe_source),
        )
        partition_k_nope, partition_v = partition_kv.split(kv_split, -1)
        return partition_k_nope, partition_k_pe, partition_v

    def run_partition(
        partition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cu_kv_seqlens: torch.Tensor,
        max_kv_len: int,
        merge_op: torch.Tensor,
        is_causal: bool,
    ) -> None:
        partition_k_nope, partition_k_pe, partition_v = partition
        run_k3_split_k_context_partition(
            mha,
            q_fp8,
            partition_k_nope,
            partition_k_pe,
            partition_v,
            output,
            softmax_stats,
            merge_op,
            context_cu_q_seqlens,
            cu_kv_seqlens,
            max_q_len=attn_metadata.max_ctx_seq_len,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
        )

    for loop_idx in range(attn_metadata.chunked_loop_num):
        cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
        total_chunked_tokens = int(
            attn_metadata.host_cu_chunked_seq_len[loop_idx, attn_metadata.num_contexts]
        )
        chunked_max_seq_len = int(attn_metadata.max_chunk_len_per_loop[loop_idx])
        if kv_b_mxfp8:
            chunked_compressed_kv, chunked_k_pe = load_fp4_mla_chunked_kv_cache_mxfp8(
                attn_metadata,
                mha.layer_idx,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=chunked_max_seq_len,
                kv_lora_rank=lora,
                qk_rope_head_dim=rope,
            )
        else:
            chunked_compressed_kv, chunked_k_pe = mla_cache.load_chunked_kv_cache_for_mla(
                mha,
                attn_metadata,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype,
            )
        partition = expand_kv(chunked_compressed_kv, chunked_k_pe)
        chunked_compressed_kv = None
        chunked_k_pe = None
        run_partition(
            partition,
            cu_chunked_seq_len,
            chunked_max_seq_len,
            attn_metadata.merge_op_tensor[loop_idx],
            is_causal=False,
        )

    run_partition(
        expand_kv(compressed_kv, current_k_pe),
        context_cu_q_seqlens,
        attn_metadata.max_ctx_seq_len,
        attn_metadata.merge_op_tensor[attn_metadata.chunked_loop_num],
        is_causal=True,
    )
    return output


__all__ = [
    "forward_context_split_k_fp4",
    "kv_b_takes_mxfp8",
    "load_fp4_mla_chunked_kv_cache_mxfp8",
    "run_k3_split_k_context_partition",
    "split_k_context_unavailable_reason",
    "supports_k3_fp8_split_k_context",
    "uses_k3_split_k_context",
    "context_cached_kv",
    "context_chunked_prefill",
    "context_default",
    "forward_context",
    "project_context_kv",
    "HelixGenHooks",
    "attn_forward_gen",
    "forward_absorption_generation",
    "forward_gen_q_fused",
    "gen_attention",
    "helix_exchange",
    "helix_gen_zero_kv_mask",
    "helix_tail_applies",
    "latent_row",
    "project_helix_v_partials",
    "q_fused_applies",
    "tail_output_buffer",
    "K3MLAAttention",
]


def project_context_kv(layer, compressed_kv: torch.Tensor) -> torch.Tensor:
    """kv_b over the context heads: ``[T, H * qk_nope | H * v_head_dim]`` (one GEMM, or under Helix
    this rank's head range of the K and V halves, concatenated)."""
    if layer.num_context_heads == layer.num_heads_tp:
        return layer.kv_b_proj(compressed_kv)
    first_head = layer.mapping.cp_rank * layer.num_context_heads
    last_head = first_head + layer.num_context_heads
    key = layer.forward_slice(
        layer.kv_b_proj,
        compressed_kv,
        first_head * layer.qk_nope_head_dim,
        last_head * layer.qk_nope_head_dim,
    )
    value_start = layer.num_heads_tp * layer.qk_nope_head_dim
    value = layer.forward_slice(
        layer.kv_b_proj,
        compressed_kv,
        value_start + first_head * layer.v_head_dim,
        value_start + last_head * layer.v_head_dim,
    )
    # The [all K | all V] row stride the context FP8 preprocessing reads.
    return torch.cat((key, value), dim=-1)


def _expand_kv_bf16(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
    """kv_b, then K ``[T, H * 192]`` (K-nope and the shared K-pe, concatenated) and V."""
    heads = layer.num_context_heads
    kv = project_context_kv(layer, compressed_kv)
    k_nope, v = kv.split([heads * layer.qk_nope_head_dim, heads * layer.v_head_dim], -1)
    k_nope = k_nope.view(-1, heads, layer.qk_nope_head_dim)
    k_pe = k_pe.view(-1, 1, layer.qk_rope_head_dim)
    k = maybe_compiled_cat((k_nope, k_pe.expand(-1, heads, -1)), dim=-1)
    return k.view(-1, heads * layer.qk_head_dim), v


def _expand_kv_fp8(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
    """kv_b, then FP8 K ``[T, H * 192]`` and V ``[T, H * 128]`` in one pass: the bytes the
    attention op's context copy writes for the BF16 K and V."""
    return mla_kv_to_fp8(
        project_context_kv(layer, compressed_kv),
        k_pe.view(-1, layer.qk_rope_head_dim),
        layer.num_context_heads,
        layer.qk_nope_head_dim,
        layer.v_head_dim,
    )


def _fp8_q_args(layer, q_fp8):
    """Attention-op arguments that hand it ``q_fp8`` as the already-quantized FP8 Q."""
    if q_fp8 is None:
        return {}
    return {"quant_q_buffer": q_fp8, "quant_scale_qkv": layer.mha.kv_scale_orig_quant}


def _quantize_q_fp8(layer, q: torch.Tensor) -> torch.Tensor:
    """Q in FP8 as the attention op's context copy quantizes it (unit scale)."""
    q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        q, layer.mha.kv_scale_orig_quant
    )
    return q_fp8


@functools.cache
def _warmup_expand_kv_fp8(
    num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, device
):
    """Compile ``mla_kv_to_fp8`` before the first cached-KV context step (k_pe contiguous for
    loaded chunks, a ``latent_cache`` view for the current tokens)."""
    num_tokens = 2
    kv = torch.zeros(
        num_tokens, num_heads * (qk_nope_head_dim + v_head_dim), dtype=torch.bfloat16, device=device
    )
    latent = torch.zeros(
        num_tokens, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, device=device
    )
    for k_pe in (latent[:, kv_lora_rank:].contiguous(), latent[:, kv_lora_rank:]):
        mla_kv_to_fp8(kv, k_pe, num_heads, qk_nope_head_dim, v_head_dim)


@functools.cache
def _warmup_expand_kv_bf16(
    num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, dtype, device
):
    """Compile ``maybe_compiled_cat`` for both K layouts (a loaded chunk's contiguous K-pe and the
    current tokens' ``latent_cache`` view) with the token dim dynamic; one token count still
    specializes, so it is warmed separately."""

    def warmup(num_tokens):
        k_nope = torch.empty(
            num_tokens, num_heads * (qk_nope_head_dim + v_head_dim), dtype=dtype, device=device
        )[:, : num_heads * qk_nope_head_dim].view(num_tokens, num_heads, qk_nope_head_dim)
        chunked_k_pe = torch.empty(
            num_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device
        ).expand(-1, num_heads, -1)
        k_pe = torch.empty(
            num_tokens, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
        )[:, :, -qk_rope_head_dim:].expand(-1, num_heads, -1)
        torch._dynamo.maybe_mark_dynamic(k_nope, 0)
        torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
        torch._dynamo.maybe_mark_dynamic(k_pe, 0)
        maybe_compiled_cat((k_nope, chunked_k_pe), dim=-1)
        maybe_compiled_cat((k_nope, k_pe), dim=-1)

    warmup(2)
    warmup(1)


def _kv_b_strided_out(layer) -> bool:
    """Whether the no-prefix expansion writes K-nope / V straight into the attention inputs with
    per-head BMMs: a bias-free BF16 kv_b on SM100f."""
    w = layer.kv_b_proj.weight
    return (
        is_sm_100f()
        and w.dtype == torch.bfloat16
        and tuple(w.shape)
        == (layer.num_heads_tp * (layer.qk_nope_head_dim + layer.v_head_dim), layer.kv_lora_rank)
        and getattr(layer.kv_b_proj, "bias", None) is None
    )


def context_default(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention without a cached prefix (see the module docstring)."""
    heads = layer.num_context_heads
    nope, rope, vdim = layer.qk_nope_head_dim, layer.qk_rope_head_dim, layer.v_head_dim
    append_owned_kv = md.helix_context_local_slots is not None
    k_pe = None
    if append_owned_kv:
        mla_cache.mla_rope_append_paged_kv_assign_q(layer.mha, q, latent_cache, md)
        k_pe = latent_cache[:, layer.kv_lora_rank :]
    k = torch.empty_like(q).view(-1, heads, layer.qk_head_dim)
    if _kv_b_strided_out(layer):
        num_tokens = compressed_kv.shape[0]
        w = layer.kv_b_proj.weight
        nope_rows = layer.num_heads_tp * nope
        w_k = w[:nope_rows].view(layer.num_heads_tp, nope, layer.kv_lora_rank)
        w_v = w[nope_rows:].view(layer.num_heads_tp, vdim, layer.kv_lora_rank)
        if heads != layer.num_heads_tp:
            first = layer.mapping.cp_rank * heads
            w_k, w_v = w_k[first : first + heads], w_v[first : first + heads]
        ckv = compressed_kv.unsqueeze(0).expand(heads, num_tokens, layer.kv_lora_rank)
        # [H, T, nope] view of K with strides (192, H * 192, 1).
        k_nope_out = k[..., :nope].transpose(0, 1)
        # The context FP8 quantize kernel reads V with the [K-nope | V] row stride of the kv_b
        # output: V is the second half of such a buffer (its first half is never written).
        kv_buf = torch.empty(num_tokens, heads * (nope + vdim), dtype=q.dtype, device=q.device)
        v = kv_buf[:, heads * nope :]
        v_out = v.unflatten(-1, (heads, vdim)).transpose(0, 1)
        bmm_out(ckv, w_k.transpose(1, 2), k_nope_out)
        bmm_out(ckv, w_v.transpose(1, 2), v_out)
    else:
        kv = project_context_kv(layer, compressed_kv)
        k_nope, v = kv.split([heads * nope, heads * vdim], -1)
        maybe_compiled_copy_(k[..., :nope], k_nope.view(-1, heads, nope))
    # With the RoPE fused in the op the K-pe slot stays unwritten; the op reads latent_cache.
    if append_owned_kv:
        k[..., nope:] = k_pe.view(-1, 1, rope)
    k = k.view(-1, heads * layer.qk_head_dim)
    if layer.mha.uses_fp4_mla_attention:
        # The FP8 context FMHA over the step's FP8 scratch cache beside the FP4 scatter.
        return mla_cache.fp4_mla_context_attention(layer.mha, q, k, v, md, output, latent_cache)
    return mla_attn.mla_context_attention(
        layer.mha, q, k, v, md, output, latent_cache=None if append_owned_kv else latent_cache
    )


def context_cached_kv(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Context attention over the whole paged cache (cached prefix, no chunked prefill)."""
    mha = layer.mha
    mla_cache.mla_rope_append_paged_kv_assign_q(mha, q, latent_cache, md)
    full_compressed_kv, full_k_pe = mla_cache.load_paged_kv_cache_for_mla(mha, md, q.dtype)
    assert full_compressed_kv.shape == (
        md.num_ctx_cached_tokens + md.num_ctx_tokens,
        layer.kv_lora_rank,
    )
    assert full_k_pe.shape == (md.num_ctx_cached_tokens + md.num_ctx_tokens, layer.qk_rope_head_dim)
    assert full_compressed_kv.is_contiguous() and full_k_pe.is_contiguous()
    q_fp8 = None
    if layer.fp8_cached_kv_context:
        q_fp8 = _quantize_q_fp8(layer, q)
        full_k, full_v = _expand_kv_fp8(layer, full_compressed_kv, full_k_pe)
    else:
        full_k, full_v = _expand_kv_bf16(layer, full_compressed_kv, full_k_pe)
    full_compressed_kv = full_k_pe = None
    return mla_attn.mla_context_attention(
        mha, q, full_k, full_v, md, output, **_fp8_q_args(layer, q_fp8)
    )


def context_chunked_prefill(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Chunked-prefill context attention over an FP8 / BF16 cache (see the module docstring)."""
    mha = layer.mha
    heads = layer.num_context_heads
    mla_cache.mla_rope_append_paged_kv_assign_q(mha, q, latent_cache, md)
    fp8 = layer.fp8_cached_kv_context
    expand = _expand_kv_fp8 if fp8 else _expand_kv_bf16
    # With FP8 attention inputs, Q is quantized once for every partition.
    q_fp8 = _quantize_q_fp8(layer, q) if fp8 else None
    q_args = _fp8_q_args(layer, q_fp8)
    softmax_stats = torch.empty((md.num_ctx_tokens, heads, 2), dtype=torch.float, device="cuda")
    temp_softmax_stats = torch.empty(
        (md.num_ctx_tokens, heads, 2), dtype=torch.float, device="cuda"
    )
    temp_output = q.new_empty((q.size(0), heads * layer.v_head_dim), dtype=q.dtype)
    buffer_batch_size = md.runtime_features.chunked_prefill_buffer_batch_size

    # Each partition attends with its own KV lengths; the originals are restored at the end.
    origin_kv_lens_cuda_runtime = md.kv_lens_cuda_runtime
    origin_kv_lens_runtime = md.kv_lens_runtime
    origin_ctx_total_kv_len = md.host_total_kv_lens[0].item()

    for loop_idx in range(md.chunked_loop_num):
        total_chunked_tokens = md.host_cu_chunked_seq_len[loop_idx, md.num_contexts]
        chunked_compressed_kv, chunked_k_pe = mla_cache.load_chunked_kv_cache_for_mla(
            mha,
            md,
            num_ctx_cached_tokens=total_chunked_tokens,
            cu_chunked_seq_len=md.cu_chunked_seq_len[loop_idx],
            chunked_global_offset=md.chunked_global_offset[loop_idx],
            chunked_max_seq_len=md.max_chunk_len_per_loop[loop_idx],
            out_dtype=q.dtype,
            helix_plan=(
                md.helix_mla_chunk_plans[loop_idx] if md.helix_mla_chunk_plans is not None else None
            ),
        )
        chunked_k, chunked_v = expand(layer, chunked_compressed_kv, chunked_k_pe)
        chunked_compressed_kv = chunked_k_pe = None
        md.kv_lens_runtime = md.host_chunked_seq_len[loop_idx]
        md.kv_lens_cuda_runtime = md.chunked_seq_len[loop_idx]
        md.host_total_kv_lens[0] = total_chunked_tokens
        temp_output = mla_attn.mla_context_attention(
            mha,
            q,
            chunked_k,
            chunked_v,
            md,
            temp_output,
            full_mask=True,
            softmax_stats=temp_softmax_stats,
            chunked_prefill_buffer_batch_size=buffer_batch_size,
            **q_args,
        )
        mla_cache.merge_attention_for_mla(
            mha,
            output,
            temp_output,
            softmax_stats,
            temp_softmax_stats,
            md.merge_op_tensor[loop_idx],
            md,
        )

    # The current chunk, causal; its rotated K-pe is in latent_cache.
    _, k_pe = latent_cache.view([-1, layer.kv_lora_rank + layer.qk_rope_head_dim]).split(
        [layer.kv_lora_rank, layer.qk_rope_head_dim], -1
    )
    k, v = expand(layer, compressed_kv, k_pe)
    md.kv_lens_runtime = md.prompt_lens_cpu_runtime
    md.kv_lens_cuda_runtime = md.prompt_lens_cuda_runtime
    md.host_total_kv_lens[0] = md.prompt_lens_cpu_runtime[: md.num_contexts].sum().item()
    temp_output = mla_attn.mla_context_attention(
        mha,
        q,
        k,
        v,
        md,
        temp_output,
        softmax_stats=temp_softmax_stats,
        chunked_prefill_buffer_batch_size=buffer_batch_size,
        **q_args,
    )
    mla_cache.merge_attention_for_mla(
        mha,
        output,
        temp_output,
        softmax_stats,
        temp_softmax_stats,
        md.merge_op_tensor[md.chunked_loop_num],
        md,
    )
    md.kv_lens_runtime = origin_kv_lens_runtime
    md.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
    md.host_total_kv_lens[0] = origin_ctx_total_kv_len
    return output


def forward_context(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention of the context rows into ``output``: picks the chain by cache state."""
    mha = layer.mha
    if md.helix_context_local_slots is not None:
        if mha.uses_fp4_mla_attention:
            raise NotImplementedError(
                "Owner-only Helix prefill does not support dense FP4 MLA KV cache"
            )
        if md.num_ctx_cached_tokens > 0 and (
            not md.runtime_features.chunked_prefill or get_sm_version() < 100
        ):
            raise NotImplementedError(
                "Helix cached context requires chunked prefill on SM100 or newer"
            )
    if mha.has_cached_kv_for_mla_context_warmup(md):
        dims = (
            layer.num_context_heads,
            layer.qk_nope_head_dim,
            layer.qk_rope_head_dim,
            layer.kv_lora_rank,
            layer.v_head_dim,
        )
        if layer.fp8_cached_kv_context:
            _warmup_expand_kv_fp8(*dims, q.device)
        else:
            _warmup_expand_kv_bf16(*dims, q.dtype, q.device)
    chunked = mha.is_chunked_prefill_for_mla_context(md)
    if chunked and get_sm_version() >= 100:
        if mha.uses_fp4_mla_attention:
            if not uses_k3_split_k_context(layer, q, output):
                raise NotImplementedError(
                    "Kimi K3 MLA: an FP4 MLA cache attends a cached prefix only through the Rubin "
                    "FP8 split-K context FMHA (SM107, BF16 Q and output, see ctx_split_k)"
                )
            return forward_context_split_k_fp4(layer, q, compressed_kv, latent_cache, md, output)
        return context_chunked_prefill(layer, q, compressed_kv, latent_cache, md, output)
    if chunked or mha.has_cached_kv_for_mla_context(md):
        if mha.uses_fp4_mla_attention:
            raise RuntimeError(
                "FP4 MLA attends a cached prefix only through chunked-prefill partitions on "
                "SM100+; enable chunked prefill."
            )
        return context_cached_kv(layer, q, latent_cache, md, output)
    return context_default(layer, q, compressed_kv, md, output, latent_cache)


def _check_cu_result(cu_func_ret):
    """The values a CUDA driver call returned after its status, raising if it failed."""
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None


# Receive buffer per rank in MiB (two halves).
_TAIL_MB = 16


class K3HelixTail:
    """The fused Helix tail over the CP group of ``mapping`` (one receive workspace per mapping)."""

    _cache: Dict[Mapping, "K3HelixTail"] = {}

    def __init__(self, mapping: Mapping):
        from ..kernels.k3r_helix.k3_helix_tail import SENTINEL

        self.mapping = mapping
        MnnvlMemory.initialize()
        nbytes = _TAIL_MB << 20
        logger.info(f"Rank {mapping.cp_rank} allocating a {nbytes >> 20} MiB Helix tail buffer")
        self.workspace = HelixCpMnnvlMemory(mapping, nbytes)
        self.ws_addr = self.workspace.ptr
        self.rank_stride = self.workspace.rank_stride
        self.half_bytes = nbytes // 2 // 16 * 16
        # Armed through the driver: the kernel only uses raw addresses (see HelixAllToAllFused).
        _check_cu_result(
            cuda.cuMemsetD8Async(
                cuda.CUdeviceptr(self.ws_addr + mapping.cp_rank * self.rank_stride),
                SENTINEL & 0xFF,
                nbytes,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.max_ctas = num_sms
        self.use_pdl = get_env_enable_pdl()
        # Tests on one GPU: this rank's records go to its own buffer (k3_helix_tail loopback).
        self.loopback = False
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()

    @staticmethod
    def get(mapping: Mapping) -> "K3HelixTail":
        if mapping not in K3HelixTail._cache:
            K3HelixTail._cache[mapping] = K3HelixTail(mapping)
        return K3HelixTail._cache[mapping]

    @staticmethod
    def applies(
        mapping: Mapping, num_tokens: int, num_heads: int, latent_dim: int, v_head_dim: int
    ) -> bool:
        """Whether the fused tail serves a generation batch of ``num_tokens`` tokens."""
        from ..kernels.k3r_helix import k3_helix_tail as kernel

        if get_sm_version() not in (100, 103, 107) or num_tokens <= 0:
            return False
        if not mapping.has_cp_helix() or mapping.cp_config.get("use_nccl_for_alltoall", True):
            return False
        if not kernel.can_run(mapping.cp_size, num_heads, latent_dim, v_head_dim):
            return False
        nbytes = _TAIL_MB << 20
        record = kernel.record_bytes(num_heads // mapping.cp_size, v_head_dim)
        return num_tokens * mapping.cp_size * record <= nbytes // 2 // 16 * 16

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
        zero_kv_mask: Optional[torch.Tensor],
        fold=None,
        out: Optional[torch.Tensor] = None,
        mx_out=None,
        timestamps: Optional[torch.Tensor] = None,
        mx_atom: bool = False,
    ) -> torch.Tensor:
        """Fold, project, exchange and combine one layer's partials.

        Args:
            rows: BF16 ``[T, H * 512]`` decode output (latent rows ``(token, head)``); with
                ``fold`` only the rows the decode stored directly are read.
            stats: FP32 ``[T, H, 2]`` ``(max, sum)`` of ``rows``; read only without ``fold``.
            w_uv: BF16 ``[H, 128, 512]`` V up-projection of every head.
            zero_kv_mask: Optional bool ``[T]``, True where this rank owns no KV of the token.
            fold: ``k3_helix_tail.FoldInputs`` of the deferred split reduction, or None.
            out: Optional BF16 ``[T, H / P * 128]`` output.
            mx_out: Optional ``(gate, xq, xs)``: also store the MXFP8 form of
                ``bf16(out * gate)``, o_proj's gated operand (see ``k3_helix_tail``).
            timestamps: Optional int64 phase stamps (microbenchmarks, see ``k3_helix_tail``).
            mx_atom: (c240) ``xs`` in the mid-M GEMM's scale layout (see ``k3_helix_tail``).

        Returns:
            The combined BF16 ``[T, H / P * 128]`` output of this rank's heads.
        """
        from ..ops.attention.k3_helix_tail import k3_helix_tail

        num_tokens = rows.shape[0]
        num_heads = w_uv.shape[0]
        cp_size = self.mapping.cp_size
        if out is None:
            out = rows.new_empty((num_tokens, num_heads // cp_size * w_uv.shape[1]))
        if zero_kv_mask is not None:
            zero_kv_mask = zero_kv_mask[:num_tokens].contiguous()
        k3_helix_tail(
            rows.reshape(num_tokens * num_heads, -1),
            stats.reshape(num_tokens * num_heads, 2),
            w_uv,
            out,
            fold=fold,
            zero_kv_mask=zero_kv_mask,
            cp_rank=self.mapping.cp_rank,
            cp_size=cp_size,
            ws_addr=self.ws_addr,
            rank_stride=self.rank_stride,
            half_bytes=self.half_bytes,
            state=self.state,
            max_ctas=self.max_ctas,
            use_pdl=self.use_pdl,
            mx_out=mx_out,
            timestamps=timestamps,
            loopback=self.loopback,
            mx_atom=mx_atom,
        )
        return out


def fold_inputs_of(deferred, rows: torch.Tensor, stats: torch.Tensor):
    """``FoldInputs`` of the reduction a ``DeferredPackedReduce`` recorded for this decode (its
    output must be ``rows`` and its statistics ``stats``), or None if it recorded none. A recorded
    call on other buffers is launched instead (the caller then reads ``rows`` and ``stats``)."""
    from ..kernels.k3r_helix.k3_helix_tail import FoldInputs

    args = deferred.args
    if args is None:
        return None
    own_buffers = (
        args["output"].data_ptr() == rows.data_ptr()
        and args["output"].numel() == rows.numel()
        and args["stats"] is not None
        and args["stats"].data_ptr() == stats.data_ptr()
    )
    if not own_buffers:
        deferred.run()
        return None
    return FoldInputs(
        args["partial"],
        args["maxima"],
        args["sums"],
        args["plan"],
        args["q_scale"],
        args["kv_scale"],
        num_requests=args["num_requests"],
        valid_m=args["valid_m"],
        physical_m=args["physical_m"],
        sm_scale=args["sm_scale"],
        output_scale=args["output_scale"],
    )


# Largest generation batch (tokens) the fused query serves (the kernel takes up to 16).
_Q_FUSED_MAX_TOKENS = 16


class HelixGenHooks:
    """What the side-gate forward (``K3MLAAttention._forward_side_gate``) hands the Helix
    generation chain for one forward, and what the chain hands back.

    ``zero_kv_mask_fn`` returns the zero-local-KV mask computed ahead on the side stream (called
    where the exchange needs it); ``pre_exchange_fn`` forks the output gate; ``pre_tail_fn`` joins
    it before the fused tail and returns it (None if not forked yet). Back: ``tail_mx``, the MXFP8
    o_proj operand
    ``(xq, xs)`` the fused tail stored, and ``gen_output``, the combined all-to-all output."""

    __slots__ = (
        "zero_kv_mask_fn",
        "pre_exchange_fn",
        "pre_tail_fn",
        "tail_mx",
        "gen_output",
    )

    def __init__(
        self,
        zero_kv_mask_fn: Callable[[], Optional[torch.Tensor]],
        pre_exchange_fn: Callable[[], None],
        pre_tail_fn: Callable[[], Optional[torch.Tensor]],
    ) -> None:
        self.zero_kv_mask_fn = zero_kv_mask_fn
        self.pre_exchange_fn = pre_exchange_fn
        self.pre_tail_fn = pre_tail_fn
        self.tail_mx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.gen_output: Optional[torch.Tensor] = None


def helix_gen_zero_kv_mask(md: AttentionMetadata, num_tokens: int) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    With valid speculative verify groups KV ownership is per token (``helix_kv_bounds``);
    otherwise the per-sequence ``kv_lens_cuda == 0`` of the generation requests, expanded over
    their query lengths (``seq_lens_cuda``) with a static output size. None without those
    buffers."""
    helix_kv_bounds = getattr(md, "helix_kv_bounds", None)
    if helix_kv_bounds is not None and getattr(md, "_helix_spec_tokens_valid", False):
        return helix_kv_bounds[:num_tokens] == 0
    kv_lens = getattr(md, "kv_lens_cuda", None)
    seq_lens = getattr(md, "seq_lens_cuda", None)
    if kv_lens is None or seq_lens is None:
        return None
    seqs = slice(md.num_contexts, md.num_contexts + md.num_generations)
    return torch.repeat_interleave(kv_lens[seqs] == 0, seq_lens[seqs], output_size=num_tokens)


def helix_exchange(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    zero_kv_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """All-to-all of this rank's partials over the CP group (fifo v2 on the MNNVL workspace; the
    sender zeroes the ``zero_kv_mask`` rows) and the combine: ``[T, num_heads_tp_cp * value_dim]``
    BF16."""
    num_tokens = partial_o.shape[0]
    cp_size = mapping.cp_size
    partial_o_out, stats_out = HelixAllToAllNative.get(mapping).alltoall_native(
        partial_o.view(num_tokens, cp_size, num_heads_tp_cp * value_dim),
        softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp * 2),
        None if zero_kv_mask is None else zero_kv_mask[:num_tokens],
    )
    return torch.ops.trtllm.helix_post_process_native(
        partial_o_out.view(num_tokens, cp_size, num_heads_tp_cp, value_dim),
        stats_out.view(num_tokens, cp_size, num_heads_tp_cp, 2),
        1.0,
        1,
    )


def project_helix_v_partials(layer, partial_o: torch.Tensor) -> torch.Tensor:
    """Every tp-local head's V up-projection of this CP rank's partials (it commutes with the
    per-head weighted combine): ``[T, H * kv_lora_rank]`` -> ``[T, H * v_head_dim]``."""
    num_tokens = partial_o.shape[0]
    heads = layer.num_heads_tp
    partial_v = partial_o.new_empty((num_tokens, heads, layer.v_head_dim))
    bmm_out(
        partial_o.view(num_tokens, heads, layer.kv_lora_rank).transpose(0, 1),
        layer.helix_v_b_proj_full.transpose(1, 2),
        partial_v.transpose(0, 1),
    )
    return partial_v.view(num_tokens, heads * layer.v_head_dim)


def tail_output_buffer(
    buf: Optional[torch.Tensor], num_tokens: int, cols: int
) -> Optional[torch.Tensor]:
    """``buf`` as the fused tail's BF16 ``[T, cols]`` output, or None where the tail cannot write
    it as is."""
    if (
        buf is None
        or buf.dtype != torch.bfloat16
        or not buf.is_contiguous()
        or buf.numel() != num_tokens * cols
    ):
        return None
    return buf.view(num_tokens, cols)


def helix_tail_applies(layer, num_tokens: int) -> bool:
    """Whether the Helix tail of this batch is one ``k3_helix_tail`` kernel (``helix_tail``)."""
    w_uv = layer.helix_v_b_proj_full
    return (
        isinstance(w_uv, torch.Tensor)
        and w_uv.dtype == torch.bfloat16
        and w_uv.is_contiguous()
        and K3HelixTail.applies(
            layer.mapping, num_tokens, layer.num_heads_tp, layer.kv_lora_rank, layer.v_head_dim
        )
    )


def gen_attention(layer, q: torch.Tensor, md: AttentionMetadata, **kwargs) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``) into a new ``[T, H * kv_lora_rank]``: K3's
    packed FP4 decode on an FP4 cache with the CuTe DSL FP4 MLA backend (with the layer's
    ``constants``); on an FP8 or BF16 cache the CuTe DSL MLA decode where it serves the batch, else
    the attention op's MLA generation (``ops/attention/k3_mla_attention``). ``kwargs``: the buffers of
    the generation cache update (``mla_rope_generation``) and ``softmax_stats_tensor``."""
    if fp4_mla_decode.applies(layer.mqa):
        assert kwargs.get("out_scale") is None
        return _decode_op.k3_fp4_mla_decode(
            layer.mqa,
            q,
            md,
            softmax_stats_tensor=kwargs.get("softmax_stats_tensor"),
            page_plan=layer.constants.fp4_mla_page_plan,
            reduce_pdl=layer.constants.fp4_mla_reduce_pdl,
        )
    mqa = layer.mqa
    if mqa.uses_fp4_mla_attention:
        raise NotImplementedError(
            "Kimi K3 MLA: an FP4 MLA cache decodes on the CuTe DSL FP4 MLA backend (SM107)"
        )
    assert kwargs.pop("out_scale", None) is None
    softmax_stats = kwargs.pop("softmax_stats_tensor", None)
    output = q.new_empty((q.shape[0], mqa.num_heads * mqa.kv_lora_rank))
    return mla_attn.mla_decode(mqa, q, md, output, softmax_stats=softmax_stats, **kwargs)


def _helix_gen(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    helix_v_up: bool,
    hooks: Optional[HelixGenHooks],
    **kwargs,
) -> torch.Tensor:
    """Decode with softmax statistics, (V up-projection,) exchange and combine."""
    softmax_stats = torch.empty(
        (q.shape[0], layer.num_heads_tp, 2), device=q.device, dtype=torch.float32
    )
    kwargs["softmax_stats_tensor"] = softmax_stats
    partial_o = gen_attention(layer, q, md, **kwargs)
    assert partial_o.shape[-1] == layer.num_heads_tp * layer.kv_lora_rank
    if hooks is not None:
        zero_kv_mask = hooks.zero_kv_mask_fn()
    else:
        zero_kv_mask = helix_gen_zero_kv_mask(md, partial_o.shape[0])
    value_dim = layer.kv_lora_rank
    if helix_v_up:
        partial_o = project_helix_v_partials(layer, partial_o)
        value_dim = layer.v_head_dim
    if hooks is not None:
        hooks.pre_exchange_fn()
    return helix_exchange(
        partial_o, softmax_stats, layer.mapping, layer.num_heads_tp_cp, value_dim, zero_kv_mask
    )


def _helix_gen_fused_tail(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    output: Optional[torch.Tensor],
    hooks: Optional[HelixGenHooks],
    **kwargs,
) -> torch.Tensor:
    """The decode with its split reduction deferred, then one tail kernel (fold, V
    up-projection, exchange, combine) writing ``output`` where it can. With the side gate
    (``hooks``), forked before the decode and joined before the tail, the tail also stores
    o_proj's MXFP8 operand ``bf16(out * gate)`` where o_proj's gated MXFP8 GEMM takes it."""
    from ..kernels.k3r_fp4_mla.fp4_mla_packed_reduce import DeferredPackedReduce

    num_tokens = q.shape[0]
    softmax_stats = torch.empty(
        (num_tokens, layer.num_heads_tp, 2), device=q.device, dtype=torch.float32
    )
    kwargs["softmax_stats_tensor"] = softmax_stats
    if hooks is not None:
        hooks.pre_exchange_fn()
    with DeferredPackedReduce() as deferred:
        partial_o = gen_attention(layer, q, md, **kwargs)
    assert partial_o.shape[-1] == layer.num_heads_tp * layer.kv_lora_rank
    fold = fold_inputs_of(deferred, partial_o, softmax_stats)
    if hooks is not None:
        zero_kv_mask = hooks.zero_kv_mask_fn()
    else:
        zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)
    gate = hooks.pre_tail_fn() if hooks is not None else None
    cols = layer.num_heads_tp_cp * layer.v_head_dim
    out = tail_output_buffer(output, num_tokens, cols)
    mx_out = None
    mx_applies = getattr(layer.o_proj, "gated_mxfp8_applies", None)
    if gate is not None and mx_applies is not None:
        if out is None:
            out = partial_o.new_empty((num_tokens, cols))
        if mx_applies(out, gate):
            xq = torch.empty((num_tokens, cols), dtype=torch.float8_e4m3fn, device=q.device)
            xs = torch.empty((num_tokens, cols // 32), dtype=torch.uint8, device=q.device)
            mx_out = (gate, xq, xs)
            hooks.tail_mx = (xq, xs)
    return K3HelixTail.get(layer.mapping)(
        partial_o,
        softmax_stats,
        layer.helix_v_b_proj_full,
        zero_kv_mask,
        fold=fold,
        out=out,
        mx_out=mx_out,
    )


def attn_forward_gen(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    *,
    helix_v_up: bool,
    output: Optional[torch.Tensor] = None,
    hooks: Optional[HelixGenHooks] = None,
    **kwargs,
) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``), with the Helix exchange under Helix CP.

    Returns ``[T, H * kv_lora_rank]`` without Helix, ``[T, H / P * v_head_dim]`` with the V
    up-projection before the exchange (``helix_v_up``), ``[T, H / P * kv_lora_rank]`` otherwise.
    ``output``: the layer's buffer for the combined rows, which the fused tail writes."""
    num_tokens = q.shape[0]
    if helix_v_up and helix_tail_applies(layer, num_tokens):
        out = _helix_gen_fused_tail(layer, q, md, output, hooks, **kwargs)
    elif layer.mapping.has_cp_helix():
        out = _helix_gen(layer, q, md, helix_v_up, hooks, **kwargs)
    else:
        return gen_attention(layer, q, md, **kwargs)
    if helix_v_up and hooks is not None:
        hooks.gen_output = out
    return out


def forward_absorption_generation(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    hooks: Optional[HelixGenHooks] = None,
) -> torch.Tensor:
    """Absorbed generation of the ``q`` rows (``[T, H * 192]``, q_b's output) into ``output``
    (``[T, H / P * v_head_dim]``); ``latent_cache`` ``[T, 576]`` is the normalized latent and
    K-pe the cache update appends."""
    num_tokens = q.shape[0]
    heads, lora, rope = layer.num_heads_tp, layer.kv_lora_rank, layer.qk_rope_head_dim
    device = q.device
    q_nope, q_pe = q.view([-1, heads, layer.qk_head_dim]).split(
        [layer.qk_nope_head_dim, rope], dim=-1
    )
    num_seqs = md.num_seqs
    # The RoPE kernel writes seqQOffset[batch + 1] and scans total / per-sequence lengths over the
    # presented rows: one per token under the ragged (token-major) verify presentation.
    cu_rows = (
        max(num_seqs, num_tokens) + 1 if getattr(md, "is_ragged_verify", False) else num_seqs + 1
    )
    cu_q_seqlens = torch.empty(cu_rows, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(cu_rows, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = mla_bmm2_scale = quant_q_buffer = None
    if layer.quantized_kv:
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        quant_q_buffer = torch.empty(
            num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
        )
    # [T, H, kv_lora_rank | rope]: the W_UK output and the rotated q_pe the RoPE kernel writes.
    fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
    rope_kwargs = {"fuse_fp4_q_quant": True} if layer.fp4_kv else {}

    def rope_and_cache_update():
        mla_cache.mla_rope_generation(
            layer.mqa,
            fused_q,
            q_pe,
            latent_cache,
            md,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            **rope_kwargs,
        )

    # [H, T, qk_nope] x [H, qk_nope, kv_lora_rank] -> [H, T, kv_lora_rank], written into fused_q.
    q_nope_out = fused_q[..., :lora].transpose(0, 1)
    if layer.k_b_proj_trans.dtype == torch.bfloat16:

        def w_uk():
            bmm_out(q_nope.transpose(0, 1), layer.k_b_proj_trans.transpose(1, 2), q_nope_out)

    else:
        w_uk_bf16 = layer.k_b_proj_trans_dequant
        if w_uk_bf16 is None:
            raise NotImplementedError(
                "Kimi K3 MLA: the FP8 absorption BMM runs on the dequantized weight (SM100f)"
            )

        def w_uk():
            torch.bmm(q_nope.transpose(0, 1), w_uk_bf16.transpose(1, 2), out=q_nope_out)

    # The quantized-cache RoPE kernel writes the quantized query too: same stream as the decode.
    rope_stream = None if layer.quantized_kv else layer.aux_stream
    fork_join(w_uk, rope_and_cache_update, layer.ln_events[0], layer.ln_events[1], rope_stream)
    fused_q = fused_q.view([num_tokens, heads * (lora + rope)])

    helix_v_up = layer.helix_v_up_before_exchange
    attn_out_latent = attn_forward_gen(
        layer,
        fused_q,
        md,
        helix_v_up=helix_v_up,
        output=output if helix_v_up else None,
        hooks=hooks,
        out_scale=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
    )
    fused_q = None
    if helix_v_up:
        # The combine produced [T, H / P * v_head_dim]; the fused tail wrote it into output (the
        # copy is then a no-op on the same storage).
        output.view(attn_out_latent.shape).copy_(attn_out_latent)
        return output

    heads_out = layer.num_heads_tp_cp
    assert attn_out_latent.shape == (num_tokens, heads_out * lora)
    latent = attn_out_latent.view([-1, heads_out, lora])
    attn_output = output.view([num_tokens, heads_out, layer.v_head_dim])
    # [H, T, kv_lora_rank] x [H, kv_lora_rank, v_head_dim] -> [H, T, v_head_dim].
    if layer.v_b_proj.dtype == torch.bfloat16:
        bmm_out(latent.transpose(0, 1), layer.v_b_proj.transpose(1, 2), attn_output.transpose(0, 1))
    else:
        torch.bmm(
            latent.transpose(0, 1),
            layer.v_b_proj_dequant.transpose(1, 2),
            out=attn_output.transpose(0, 1),
        )
    return output


# ---------------------------------------------------------------------------------------------
# Fused query: generation-only Helix batches of FP4-KV layers.
# ---------------------------------------------------------------------------------------------


def _flashinfer_cute_norm(device: torch.device) -> bool:
    """Whether FlashInfer's ``rmsnorm`` runs its CuTe DSL ``RMSNormKernel`` on ``device`` (the
    arithmetic ``k3_mla_latent_row`` and ``k3_mla_q_fused`` reproduce), not its CUDA JIT kernel."""
    try:
        from flashinfer.norm import _use_cuda_norm
    except ImportError:
        return False
    return not _use_cuda_norm(device)


def q_fused_w_uk(layer) -> Optional[torch.Tensor]:
    """The BF16 absorption weight ``[H, kv_lora_rank, qk_nope_head_dim]`` the W_UK BMM reads (the
    weight, or its dequantized copy)."""
    w_kc = layer.k_b_proj_trans
    if w_kc is not None and w_kc.dtype != torch.bfloat16:
        w_kc = layer.k_b_proj_trans_dequant
    return w_kc


def q_fused_qb_exponents(layer) -> Optional[torch.Tensor]:
    """Block exponents of the q_b weight (``k3_mla_q_fused.block_exponents``), cached per weight
    scale; None if they would have to be built during CUDA graph capture."""
    scale = getattr(layer.q_b_proj, "weight_scale", None)
    if not isinstance(scale, torch.Tensor):
        return None
    cached = layer._q_fused_qb_exp
    if cached is not None and cached[0] is scale:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    from ..kernels.k3r_mla_q import k3_mla_q_fused

    weight = layer.q_b_proj.weight
    try:
        exponents = k3_mla_q_fused.block_exponents(scale, *weight.shape)
    except ValueError:
        return None
    layer._q_fused_qb_exp = (scale, exponents)
    return exponents


def q_fused_applies(layer, hidden_states: torch.Tensor, md: AttentionMetadata) -> bool:
    """Whether this forward takes :func:`forward_gen_q_fused`: a generation-only Helix batch of an
    FP4-KV layer (V up-projection before the exchange) whose query fits the fused kernel."""
    num_tokens = md.num_tokens
    if (
        hidden_states.dim() != 2
        or hidden_states.shape[0] != num_tokens
        or md.num_contexts != 0
        or md.num_generations <= 0
        or num_tokens % md.num_generations != 0
        or not layer.helix_v_up_before_exchange
        or not layer.mqa.uses_fp4_mla_attention
        or getattr(md, "helix_position_offsets", None) is None
        or getattr(md, "helix_is_inactive_rank", None) is None
        or layer.mqa.rotary_cos_sin is None
        or layer.mqa.rotary_cos_sin.dtype != torch.float32
        or not layer.mqa.rotary_cos_sin.is_contiguous()
    ):
        return False
    from ..kernels.k3r_mla_q import k3_mla_q_fused

    if not (
        layer.num_heads_tp == k3_mla_q_fused.NUM_HEADS
        and layer.q_lora_rank == k3_mla_q_fused.Q_LORA_RANK
        and layer.kv_lora_rank == k3_mla_q_fused.KV_LORA_RANK
        and layer.qk_nope_head_dim == k3_mla_q_fused.NOPE_DIM
        and layer.qk_rope_head_dim == k3_mla_q_fused.ROPE_DIM
        and num_tokens <= min(k3_mla_q_fused.MAX_TOKENS, _Q_FUSED_MAX_TOKENS)
    ):
        return False
    weight = layer.q_b_proj.weight
    w_uk = q_fused_w_uk(layer)
    if w_uk is None or not k3_mla_q_fused.weights_supported(weight, w_uk):
        return False
    return q_fused_qb_exponents(layer) is not None


def latent_row(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """``concat([kv_a RMSNorm(compressed_kv), k_pe], -1)``: FlashInfer's norm written straight into
    the row and ``k_pe`` copied beside it."""
    norm = layer.kv_a_layernorm
    if not (IS_FLASHINFER_AVAILABLE and compressed_kv.dtype == torch.bfloat16):
        return torch.concat(
            [rms_norm(compressed_kv, norm.weight, norm.variance_epsilon), k_pe], dim=-1
        )
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


def _q_fused_latent_row(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """:func:`latent_row` in one ``k3_mla_latent_row`` launch where FlashInfer's norm is its CuTe
    DSL kernel (whose arithmetic it repeats, bit for bit)."""
    from ..kernels.k3r_mla_q import k3_mla_q_fused
    from ..ops.attention.k3_mla_latent_row import k3_mla_latent_row

    norm = layer.kv_a_layernorm
    if (
        IS_FLASHINFER_AVAILABLE
        and _flashinfer_cute_norm(compressed_kv.device)
        and k3_mla_q_fused.latent_row_supported(compressed_kv, k_pe, norm.weight)
    ):
        return k3_mla_latent_row(
            compressed_kv,
            k_pe,
            norm.weight,
            norm.variance_epsilon,
            use_pdl=get_env_enable_pdl(),
        )
    return latent_row(layer, compressed_kv, k_pe)


def forward_gen_q_fused(
    layer,
    hidden_states: torch.Tensor,
    md: AttentionMetadata,
    hooks: Optional[HelixGenHooks] = None,
) -> torch.Tensor:
    """Absorbed Helix generation of an FP4-KV layer with the fused query; returns the combined
    all-to-all output ``[T, H / P * v_head_dim]``.

    After the fused q_a / kv_a projection the KV half (``k3_mla_latent_row``, then the FP4 cache
    update without its query rows) runs on the aux stream (with multi-stream on) and the main
    stream runs ``k3_mla_q_fused`` (on the raw q_a rows with the RMSNorm inside, or after the
    separate q_a RMSNorm), which writes the FP4 query rows into the Q buffers the cache update
    recorded on the metadata. The decode joins both; it reads only those buffers (the BF16
    ``fused_q`` it is handed provides shapes)."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.config import _fp4_mla_packed_rows_per_seq
    from tensorrt_llm._torch.attention.backends.fp4_mla.layout import _get_fp4_mla_q_global_scale

    from ..kernels.k3r_mla_q import k3_mla_q_fused as q_fused_kernel
    from ..ops.attention._k3_fp4_mla_cache import _prepare_fp4_mla_q_buffers
    from ..ops.attention.fp4_mla_kv_append import fp4_mla_kv_append
    from ..ops.attention.k3_mla_q_fused import k3_mla_q_fused

    num_tokens = md.num_tokens
    heads, lora, rope = layer.num_heads_tp, layer.kv_lora_rank, layer.qk_rope_head_dim
    device = hidden_states.device
    gen_len = num_tokens // md.num_generations
    # The Q buffers the KV half records on the metadata, created (zero-filled, once) on this
    # stream, which writes them.
    _prepare_fp4_mla_q_buffers(
        md, num_tokens, heads, device, rows_per_seq=_fp4_mla_packed_rows_per_seq(heads, gen_len)
    )
    q, compressed_kv, k_pe = layer.project_qkv_a(hidden_states).split(
        [layer.q_lora_rank, lora, rope], -1
    )
    kv_fork, kv_ready = layer._kv_events
    kv_stream = layer.aux_stream if do_multi_stream() and layer.aux_stream is not None else None
    if kv_stream is not None:
        kv_fork.record()
    with torch.cuda.stream(kv_stream) if kv_stream is not None else contextlib.nullcontext():
        if kv_stream is not None:
            kv_fork.wait()
        latent_cache = _q_fused_latent_row(layer, compressed_kv, k_pe)
        fp4_mla_kv_append(layer.mqa, latent_cache, md, heads)
        if kv_stream is not None:
            kv_ready.record()
    norm = layer.q_a_layernorm
    fuse_norm = (
        q_fused_kernel.NORM_FUSION_AVAILABLE
        and IS_FLASHINFER_AVAILABLE
        and _flashinfer_cute_norm(device)
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and q_fused_kernel.input_supported(q)
    )
    if not fuse_norm:
        q = layer.q_a_norm(q)
    assert q_fused_kernel.input_supported(q), "the fused query reads BF16 q_a rows"
    state = md.fp4_mla_state
    rows_per_seq = int(state.prequantized_q_rows_per_seq or 0)
    k3_mla_q_fused(
        q,
        layer.q_b_proj.weight,
        q_fused_qb_exponents(layer),
        q_fused_w_uk(layer),
        _get_fp4_mla_q_global_scale(md, device),
        layer.mqa.rotary_cos_sin,
        md.helix_position_offsets,
        state.prequantized_q,
        state.prequantized_q_sf.view(torch.uint8),
        gen_len=gen_len,
        rows_per_seq=rows_per_seq or gen_len * heads,
        use_pdl=get_env_enable_pdl(),
        norm=(norm.weight, norm.variance_epsilon) if fuse_norm else None,
    )
    if kv_stream is not None:
        kv_ready.wait()
    fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
    q_pe = fused_q[..., lora:]
    num_seqs = md.num_seqs
    attn_out = attn_forward_gen(
        layer,
        fused_q.view([num_tokens, heads * (lora + rope)]),
        md,
        helix_v_up=True,
        hooks=hooks,
        out_scale=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        cu_q_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
        cu_kv_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
        fmha_scheduler_counter=torch.empty(1, dtype=torch.uint32, device=device),
        mla_bmm1_scale=torch.empty(2, dtype=torch.float32, device=device),
        mla_bmm2_scale=torch.empty(1, dtype=torch.float32, device=device),
        quant_q_buffer=torch.empty(
            num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
        ),
    )
    assert attn_out.shape == (num_tokens, layer.num_heads_tp_cp * layer.v_head_dim)
    return attn_out


# Generation-only Helix batches of at most this many rows (V up-projection before the
# all-to-all) take the side-gate forward: the zero-KV mask and the output gate
# sigmoid(g_proj(x)) run on a side stream, and o_proj gates its input in its GEMM.
_K3_MLA_GATE_SIDE_MAX_M = 128
# One side stream per device, shared by all MLA layers (their side work never overlaps).
_GATE_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _gate_side_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _GATE_SIDE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index, priority=0)
        _GATE_SIDE_STREAMS[index] = stream
    return stream


# ---------------------------------------------------------------------------------------------
# The attention part under breakable CUDA graphs: the layer is found by its index string, so the
# break's frozen arguments are tensors and a string.
# ---------------------------------------------------------------------------------------------


def _registered_layer(layer_idx: str):
    """The step's attention metadata and the MLA layer registered under ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"
    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, TrtllmAttentionMetadata), "the MLA metadata must be TRTLLM's"
    mla_layers = extra_attrs.get("mla_layers", None)
    assert mla_layers is not None, "Attention layer is not registered"
    layer_ref = mla_layers.get(layer_idx, None)
    assert layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    layer = layer_ref()
    assert isinstance(layer, K3MLAAttention), "the registered MLA layer must be K3's"
    return metadata, layer


def create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    """The attention output buffer of the registered layer ``layer_idx`` for these rows."""
    _, layer = _registered_layer(layer_idx)
    return layer.create_output(hidden_states)


def mla_custom_op_inplace(
    hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor
) -> None:
    """The attention part of the registered layer ``layer_idx`` into ``output``, on the step's
    metadata registered with it."""
    metadata, layer = _registered_layer(layer_idx)
    layer.run_attention(hidden_states, metadata, output)


_bcg_mla_custom_op_inplace = eager_on_graph(mla_custom_op_inplace)


# ---------------------------------------------------------------------------------------------
# K3MLAAttention.
# ---------------------------------------------------------------------------------------------


class K3MLAAttention(nn.Module):
    """Kimi K3 MLA (see the module docstring)."""

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
        constants: K3Sm107Constants,
    ) -> None:
        super().__init__()
        config = model_config
        # The target's constants: ``mla_q_norm_pdl`` (q_a_norm) and the FP4 decode's
        # ``fp4_mla_page_plan`` / ``fp4_mla_reduce_pdl`` (gen_attention).
        self.constants = constants
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.use_output_gate = use_output_gate
        # The checkpoint's separate q_a / kv_a rows load into one fused projection.
        self.fuse_qkv_a_proj = True
        self.out_scale = None
        if rms_norm_eps is None:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            rms_norm_eps = 1e-6
        self.kv_cache_dtype = config.extra_attrs.get("kv_cache_dtype", "auto")

        # Each projection's checkpoint recipe: FP8 128 x 128 block scales or BF16.
        projection_configs = config.quant_config_dict or {}

        def is_fp8(name: str) -> bool:
            algo = getattr(projection_configs.get(name, config.quant_config), "quant_algo", None)
            if algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
                raise ValueError(f"Kimi K3 MLA: {name} has unsupported quantization {algo}")
            return algo == QuantAlgo.FP8_BLOCK_SCALES

        fp8 = {
            name: is_fp8(name)
            for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj")
            + (("g_proj",) if use_output_gate else ())
        }
        self._reject_unsupported(config, mapping_with_cp, fp8)

        # Registered for the custom ops; a later stack (one-model speculative decoding) with the
        # same local layer index registers as "<layer_idx>_<n>".
        self.register_to_config = True
        registry = config.extra_attrs.setdefault("mla_layers", {})
        suffix = 0
        while self.layer_idx_str in registry:
            self.layer_idx_str = f"{layer_idx}_{suffix}"
            suffix += 1
        registry[self.layer_idx_str] = weakref.ref(self)

        # Head sharding: q_b / kv_b over TP (attention DP: none), the post-exchange heads of
        # o_proj / g_proj over TP x CP.
        self.mapping = mapping_with_cp if mapping_with_cp is not None else config.mapping
        tp_size, pp_size, cp_size = self.mapping.tp_size, self.mapping.pp_size, self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size, tp_size = tp_size, 1
        world_size = pp_size * dp_size * tp_size * cp_size
        mapping = Mapping(
            world_size=world_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = Mapping(
            world_size=world_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        assert num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        # Helix context attends this CP rank's head range only.
        self.num_context_heads = (
            self.num_heads_tp_cp if self.mapping.has_cp_helix() else self.num_heads_tp
        )

        self.aux_stream = (aux_stream_dict or {}).get(AuxStreamType.Attention)
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        def projection(fp8: bool, in_features: int, out_features: int, mode, mp: Optional[Mapping]):
            shard = {} if mp is None else {"tp_size": mp.tp_size, "tp_rank": mp.tp_rank}
            if fp8:
                return Fp8BlockScaleLinear(in_features, out_features, tp_mode=mode, **shard)
            return Bf16Linear(in_features, out_features, dtype=dtype, tp_mode=mode, **shard)

        column, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW
        self.kv_a_proj_with_mqa = projection(
            fp8["kv_a_proj_with_mqa"],
            hidden_size,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
            None,
            None,
        )
        if fp8["kv_a_proj_with_mqa"]:
            # q_lora + kv_lora + rope rows (2112 for K3) leave a partial last 128-row tile,
            # which the FP8 low-M GEMM serves only on request.
            self.kv_a_proj_with_mqa.fp8w_lowm_partial_n_tile = True
        self.q_a_layernorm = RMSNormWeight(q_lora_rank, rms_norm_eps, dtype=dtype)
        self.q_b_proj = projection(
            fp8["q_b_proj"], q_lora_rank, num_heads * self.qk_head_dim, column, mapping
        )
        self.kv_a_layernorm = RMSNormWeight(kv_lora_rank, rms_norm_eps, dtype=dtype)
        self.kv_b_proj = projection(
            fp8["kv_b_proj"],
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            column,
            mapping,
        )
        self._kv_b_fp8 = isinstance(self.kv_b_proj, Fp8BlockScaleLinear)
        # This CP rank's post-exchange heads of the V absorption (bound by the loader).
        self.v_b_proj = nn.Parameter(
            torch.empty((self.num_heads_tp_cp, v_head_dim, kv_lora_rank), dtype=dtype),
            requires_grad=False,
        )
        # Helix only: the BF16 V weight of every tp-local head, applied to this rank's partials
        # before the all-to-all (``project_helix_v_partials``).
        self.register_buffer("helix_v_b_proj_full", None, persistent=False)
        self.helix_v_up_before_exchange = False
        self.o_proj = projection(
            fp8["o_proj"], num_heads * v_head_dim, hidden_size, row, self.mapping_o
        )
        self.attention_output_hidden_size = self.o_proj.in_features
        if use_output_gate:
            # o_proj's input sharding (under Helix the post-exchange 1/cp head chunk).
            self.g_proj = projection(
                fp8["g_proj"], hidden_size, num_heads * v_head_dim, column, self.mapping_o
            )

        quant_config = config.get_quant_config()
        self.quant_config = quant_config
        # The two attention roles over the engine's KV cache (``ops/attention/k3_mla_state``): the
        # absorbed generation (``mqa``: one 576-dim latent head per query head) and the context
        # (``mha``: 192 / 128-dim heads), each with the identity RoPE table (K3 is NoPE).
        role_dims = dict(
            layer_idx=layer_idx,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            skip_correction_threshold=config.skip_correction_threshold,
        )
        self.mqa = K3MlaAttnState.generation(
            num_heads=self.num_heads_tp,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            **role_dims,
        )
        self.mha = K3MlaAttnState.context(
            num_context_heads=self.num_context_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            **role_dims,
        )

        # Set in create_weights() from the backends' KV cache quantization.
        self.quantized_kv = False
        self.fp4_kv = False
        self.fp8_cached_kv_context = False
        self._weights_transformed = False
        # Side-gate forward: fork / join of the zero-KV mask and the gate on the side stream.
        self._gate_events = tuple(torch.cuda.Event() for _ in range(4))
        # Fused query: fork of the KV half after q_a, join before the decode.
        self._kv_events = (torch.cuda.Event(), torch.cuda.Event())
        # Set by the decoder layer for the next forward only: the MXFP8 form ``(xq, xs)`` of the
        # layer input, which the fused q_a / kv_a projection consumes (see q_a_takes_mxfp8).
        self.next_input_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # Fused query: (q_b weight scale, its block exponents), built on first use.
        self._q_fused_qb_exp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if not config.skip_create_weights_in_init:
            self.create_weights()

    @staticmethod
    def _reject_unsupported(
        config: ModelConfig, mapping_with_cp: Optional[Mapping], fp8: dict[str, bool]
    ) -> None:
        """Configurations whose kernel chain K3 does not carry."""
        mapping = mapping_with_cp if mapping_with_cp is not None else config.mapping
        unsupported = []
        if config.attn_backend.upper() != "TRTLLM":
            unsupported.append(f"attention backend {config.attn_backend} (TRTLLM only)")
        if config.sparse_attention_config is not None:
            unsupported.append("sparse attention")
        if fp8["q_a_proj"] != fp8["kv_a_proj_with_mqa"]:
            unsupported.append("q_a / kv_a with different recipes (no fused q_a / kv_a GEMM)")
        if config.extra_attrs.get("kv_cache_dtype", "auto") == "fp8_ds_mla":
            unsupported.append("the fp8_ds_mla KV cache")
        if hasattr(config.pretrained_config, "llama_4_scaling"):
            unsupported.append("Llama-4 attention scaling")
        # The CuTe DSL GEMM / BMM opt-ins of main's MLA, where they would select another kernel:
        # BF16 projections, BF16 absorption BMMs (BF16 kv_b, or the Helix V up-projection), FP8
        # absorption BMMs.
        if config.use_cute_dsl_bf16_gemm and not all(fp8.values()):
            unsupported.append("use_cute_dsl_bf16_gemm with BF16 projections")
        if config.use_cute_dsl_bf16_bmm and (mapping.has_cp_helix() or not fp8["kv_b_proj"]):
            unsupported.append("use_cute_dsl_bf16_bmm")
        if config.use_cute_dsl_blockscaling_bmm and fp8["kv_b_proj"]:
            unsupported.append("use_cute_dsl_blockscaling_bmm")
        if mapping.has_cp_ulysses():
            unsupported.append("Ulysses CP")
        if mapping.cp_size > 1 and not mapping.has_cp_helix():
            unsupported.append(f"CP type {mapping.cp_config.get('cp_type')}")
        if mapping.has_cp_helix():
            if mapping.enable_attention_dp:
                unsupported.append("Helix CP with attention DP")
            if mapping.cp_config.get("use_nccl_for_alltoall", True):
                unsupported.append("the NCCL Helix all-to-all (use_nccl_for_alltoall: false)")
            elif mapping.cp_config.get("fifo_version", 2) != 2:
                unsupported.append("the fifo-v1 Helix all-to-all")
        if unsupported:
            raise NotImplementedError("Kimi K3 MLA does not support: " + "; ".join(unsupported))

    def create_weights(self) -> None:
        """Backend KV cache quantization and the absorption weights (their dtype follows kv_b)."""
        self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)
        self.quantized_kv = bool(self.mqa.has_fp8_kv_cache or self.mqa.has_fp4_kv_cache)
        self.fp4_kv = bool(
            self.mqa.support_fp4_kv_cache()
            and self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp4_kv_cache()
        )
        if self.fp4_kv and not self.mqa.rope_duplicate_data:
            raise ValueError("FP4 MLA requires RopeParams.duplicate_data=True.")
        # Cached-prefix context on the FP8 context FMHA takes FP8 Q / K / V (plain conversions at
        # the op's unit KV scale).
        mha = self.mha
        scale = mha.kv_scale_orig_quant
        self.fp8_cached_kv_context = bool(
            mha.has_fp8_context_mla
            and not self.mapping.has_cp_helix()
            and (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) == (128, 64, 128)
            and isinstance(scale, torch.Tensor)
            and bool(torch.all(scale == 1.0).item())
        )
        heads, heads_cp = self.num_heads_tp, self.num_heads_tp_cp
        lora, nope, vdim = self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (heads, lora, nope), dtype=torch.float8_e4m3fn if self._kv_b_fp8 else self.dtype
            ),
            requires_grad=False,
        )
        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if self._kv_b_fp8:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty((heads, lora // 128, nope // 128), dtype=torch.float32),
                requires_grad=False,
            )
            self.v_b_proj_scale = nn.Parameter(
                torch.empty((heads_cp, vdim // 128, lora // 128), dtype=torch.float32),
                requires_grad=False,
            )
            if is_sm_100f():
                # The absorption BMMs run in BF16 on dequantized copies.
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty((heads, lora, nope), dtype=self.dtype), requires_grad=False
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty((heads_cp, vdim, lora), dtype=self.dtype), requires_grad=False
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
        self._weights_transformed = False

    # -----------------------------------------------------------------------------------------
    # Post-load.
    # -----------------------------------------------------------------------------------------

    def _bind_helix_v_up_weight(self, weight: torch.Tensor) -> None:
        """Apply the V up-projection with ``weight`` (BF16 ``[num_heads_tp, v_head_dim,
        kv_lora_rank]``) before the Helix all-to-all."""
        assert self.mapping.has_cp_helix()
        assert weight.dtype == torch.bfloat16
        assert tuple(weight.shape) == (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
        self.helix_v_b_proj_full = weight
        self.helix_v_up_before_exchange = True

    def _build_helix_v_up_weight(self) -> None:
        """Under Helix, every tp-local head's BF16 V weight from kv_b's loaded rows (BF16, or the
        raw FP8 codes with their FP32 128 x 128 block scales: this runs before kv_b's own
        transform, which the post-load walk visits after this module)."""
        if not self.mapping.has_cp_helix() or self.dtype != torch.bfloat16:
            return
        weight = self.kv_b_proj.weight
        nope_rows = self.num_heads_tp * self.qk_nope_head_dim
        num_rows = nope_rows + self.num_heads_tp * self.v_head_dim
        v_shape = (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
        if weight.is_meta or tuple(weight.shape) != (num_rows, self.kv_lora_rank):
            return
        if weight.dtype == torch.bfloat16:
            v_weight = weight[nope_rows:].view(v_shape)
        elif weight.dtype == torch.float8_e4m3fn:
            scale = self.kv_b_proj.weight_scale
            if scale.dtype != torch.float32 or tuple(scale.shape) != (
                -(-num_rows // 128),
                -(-self.kv_lora_rank // 128),
            ):
                return
            row_blocks = torch.arange(nope_rows, num_rows, device=weight.device) // 128
            v_scale = scale[row_blocks].repeat_interleave(128, dim=1)[:, : self.kv_lora_rank]
            v_weight = (weight[nope_rows:].float() * v_scale).to(self.dtype).view(v_shape)
            if self.v_b_proj_dequant is not None:
                # This rank's dequantized V chunk holds the same values.
                start = self.mapping.cp_rank * self.num_heads_tp_cp
                self.v_b_proj_dequant = nn.Parameter(
                    v_weight[start : start + self.num_heads_tp_cp], requires_grad=False
                )
        else:
            return
        self._bind_helix_v_up_weight(v_weight)

    def transform_weights(self) -> None:
        if self._weights_transformed:
            return
        self._build_helix_v_up_weight()
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        """The scales of the q_b / kv_b output slices a Helix rank's context heads read."""
        if self.num_context_heads != self.num_heads_tp:
            first_head = self.mapping.cp_rank * self.num_context_heads
            last_head = first_head + self.num_context_heads
            value_start = self.num_heads_tp * self.qk_nope_head_dim
            for proj, start, end in (
                (self.q_b_proj, first_head * self.qk_head_dim, last_head * self.qk_head_dim),
                (
                    self.kv_b_proj,
                    first_head * self.qk_nope_head_dim,
                    last_head * self.qk_nope_head_dim,
                ),
                (
                    self.kv_b_proj,
                    value_start + first_head * self.v_head_dim,
                    value_start + last_head * self.v_head_dim,
                ),
            ):
                if not isinstance(proj, Bf16Linear):
                    proj.prepare_output_slice(start, end)
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()

    # -----------------------------------------------------------------------------------------
    # Front.
    # -----------------------------------------------------------------------------------------

    def q_a_mx_atom(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a GEMM takes the MXFP8 rows with the scales in the mid-M
        GEMM's layout (``takes_mid_mxfp8_input``)."""
        takes_mid = getattr(self.kv_a_proj_with_mqa, "takes_mid_mxfp8_input", None)
        return takes_mid is not None and bool(takes_mid(num_rows))

    def q_a_takes_mxfp8(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a GEMM of ``num_rows`` rows can take their MXFP8 form
        (``next_input_mxfp8``)."""
        takes = getattr(self.kv_a_proj_with_mqa, "takes_mxfp8_input", None)
        return takes is not None and (bool(takes(num_rows)) or self.q_a_mx_atom(num_rows))

    def project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection ``[T, q_lora | kv_lora | rope]``: on the MXFP8 rows the
        decoder layer handed over for these rows (``forward_mxfp8``), else the GEMM on the BF16
        rows (a BF16 weight: ``cublas_mm``)."""
        mxfp8_input, self.next_input_mxfp8 = self.next_input_mxfp8, None
        proj = self.kv_a_proj_with_mqa
        if isinstance(proj, Bf16Linear):
            return cublas_mm(
                hidden_states,
                proj.weight.t(),
                None,
                out_dtype=None,
                output_buffer_kind=int(BufferKind.DEFAULT),
                group=None,
            )
        if mxfp8_input is not None and mxfp8_input[0].shape[0] == hidden_states.shape[0]:
            return proj.forward_mxfp8(hidden_states, *mxfp8_input)
        return proj(hidden_states)

    def q_a_norm(self, q: torch.Tensor) -> torch.Tensor:
        """The q_a RMSNorm (FlashInfer); with ``constants.mla_q_norm_pdl`` and PDL the same kernel
        releasing its dependent (q_b) at launch instead of after its store, so q_b streams its
        weights while q_a and the norm run."""
        norm = self.q_a_layernorm
        if self.constants.mla_q_norm_pdl and IS_FLASHINFER_AVAILABLE and get_env_enable_pdl():
            from ..kernels.k3r_mla_q.q_norm_pdl import can_use
            from ..ops.attention.rmsnorm_release_at_launch import rmsnorm_release_at_launch

            if can_use(q, norm.weight):
                return rmsnorm_release_at_launch(q, norm.weight, norm.variance_epsilon)
        return rms_norm(q, norm.weight, norm.variance_epsilon)

    @staticmethod
    def forward_slice(proj, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Output features ``[start, end)`` of a column-parallel projection, contiguous."""
        if isinstance(proj, Bf16Linear):
            return torch.nn.functional.linear(x, proj.weight[start:end])
        return proj.forward_slice(x, start, end)

    def _project_q_by_phase(
        self, q: torch.Tensor, num_ctx_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """q_b of the context rows (a Helix rank's context heads only) and of the generation rows."""
        if self.num_context_heads == self.num_heads_tp or num_ctx_tokens == 0:
            projected = self.q_b_proj(q)
            return projected[:num_ctx_tokens], projected[num_ctx_tokens:]
        first = self.mapping.cp_rank * self.num_context_heads * self.qk_head_dim
        last = first + self.num_context_heads * self.qk_head_dim
        ctx = self.forward_slice(self.q_b_proj, q[:num_ctx_tokens], first, last)
        gen = self.q_b_proj(q[num_ctx_tokens:]) if q.shape[0] > num_ctx_tokens else None
        return ctx, gen

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
        hooks: Optional[HelixGenHooks],
    ) -> None:
        """Front, context and generation attention of the batch's rows into ``output``."""
        num_ctx_tokens = md.num_ctx_tokens
        num_tokens = md.num_tokens
        hidden_states = hidden_states[:num_tokens, ...]
        q, compressed_kv, k_pe = self.project_qkv_a(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        kv_norm = self.kv_a_layernorm
        q, compressed_kv = fork_join(
            lambda: self.q_a_norm(q),
            lambda: rms_norm(compressed_kv, kv_norm.weight, kv_norm.variance_epsilon),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        (q_ctx, q_gen), latent_cache = fork_join(
            lambda: self._project_q_by_phase(q, num_ctx_tokens),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if md.num_contexts > 0:
            forward_context(
                self,
                q_ctx,
                compressed_kv[:num_ctx_tokens, ...],
                md,
                output[:num_ctx_tokens, :],
                latent_cache[:num_ctx_tokens, ...],
            )
        if md.num_generations > 0:
            assert q_gen is not None
            forward_absorption_generation(
                self,
                q_gen,
                latent_cache[num_ctx_tokens:, ...],
                md,
                output[num_ctx_tokens:num_tokens, :],
                hooks,
            )

    def run_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
        hooks: Optional[HelixGenHooks] = None,
    ) -> None:
        """The attention part of a forward (everything before the output gate) into ``output``,
        on the mixed-precision KV staging copy where the metadata has one."""
        with md.mixed_precision_kv_staging(self.layer_idx):
            self._forward_attention(hidden_states, md, output, hooks)

    def create_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The attention output ``[rows, H / P * v_head_dim]`` for the layer input's rows."""
        return hidden_states.new_empty(
            [hidden_states.shape[0], self.attention_output_hidden_size], dtype=hidden_states.dtype
        )

    # -----------------------------------------------------------------------------------------
    # Forward.
    # -----------------------------------------------------------------------------------------

    def _gate_and_o_proj(self, hidden_states: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        if self.use_output_gate:
            attn_out = attn_out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_out)

    def _side_gate_applies(self, hidden_states: torch.Tensor, md: AttentionMetadata) -> bool:
        """Whether this forward takes :meth:`_forward_side_gate`."""
        return (
            self.use_output_gate
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] == md.num_tokens
            and md.num_contexts == 0
            and self.helix_v_up_before_exchange
            and hasattr(self.o_proj, "gated_applies")
            and hidden_states.shape[0] <= _K3_MLA_GATE_SIDE_MAX_M
        )

    def _forward_side_gate(
        self, hidden_states: torch.Tensor, md: AttentionMetadata
    ) -> torch.Tensor:
        """Generation forward with the zero-KV mask and the output gate on a side stream.

        The mask (one small kernel) is forked once the layer input is written; the gate (one GEMM
        with a sigmoid epilogue) is forked when the all-to-all is enqueued (with the fused Helix
        tail: right before the decode), so it runs on the SMs the all-to-all leaves idle. The main
        stream waits for the mask where the exchange reads it and for the gate before o_proj; with
        the fused Helix tail, before the tail kernel, so o_proj depends on the tail alone and
        gates its input in its GEMM (``forward_gated(_mxfp8)``)."""
        num_tokens = hidden_states.shape[0]
        mask_fork, mask_ready, gate_fork, gate_ready = self._gate_events
        multi_stream = do_multi_stream()
        side = _gate_side_stream(hidden_states.device) if multi_stream else None
        gates = []
        mask_joined = [not multi_stream]
        gate_joined = [not multi_stream]

        def gate_of(x):
            forward_sigmoid = getattr(self.g_proj, "forward_sigmoid", None)
            if forward_sigmoid is not None:
                return forward_sigmoid(x)
            return self.g_proj(x).sigmoid()

        if multi_stream:
            mask_fork.record()
            with torch.cuda.stream(side):
                mask_fork.wait()
                zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)
                mask_ready.record()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)

        def zero_kv_mask_fn():
            if not mask_joined[0]:
                mask_ready.wait()
                mask_joined[0] = True
            return zero_kv_mask

        def pre_exchange():
            if multi_stream:
                gate_fork.record()
                with torch.cuda.stream(side):
                    gate_fork.wait()
                    gates.append(gate_of(hidden_states))
                    gate_ready.record()
            else:
                gates.append(gate_of(hidden_states))

        def pre_tail():
            if gates and not gate_joined[0]:
                gate_ready.wait()
                gate_joined[0] = True
            return gates[0] if gates else None

        hooks = HelixGenHooks(zero_kv_mask_fn, pre_exchange, pre_tail)
        if q_fused_applies(self, hidden_states, md):
            attn_out = forward_gen_q_fused(self, hidden_states, md, hooks)
        else:
            output = self.create_output(hidden_states)
            self.run_attention(hidden_states, md, output, hooks)
            attn_out = hooks.gen_output if hooks.gen_output is not None else output
        if not mask_joined[0]:
            mask_ready.wait()
        if gates:
            gate = gates[0]
            if not gate_joined[0]:
                gate_ready.wait()
        else:
            gate = gate_of(hidden_states)
        if hooks.tail_mx is not None:
            return self.o_proj.forward_gated_mxfp8(attn_out, gate, *hooks.tail_mx)
        if self.o_proj.gated_applies(attn_out, gate):
            return self.o_proj.forward_gated(attn_out, gate)
        return self.o_proj(attn_out * gate)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """o_proj's partial output of the layer for ``hidden_states`` (K3 is NoPE:
        ``position_ids`` is None)."""
        assert position_ids is None, "Kimi K3 MLA is NoPE"
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Kimi K3 MLA takes BF16 rows, got {type(hidden_states).__name__}")
        try:
            if self._side_gate_applies(hidden_states, attn_metadata):
                return self._forward_side_gate(hidden_states, attn_metadata)
            if q_fused_applies(self, hidden_states, attn_metadata):
                attn_out = forward_gen_q_fused(self, hidden_states, attn_metadata)
                return self._gate_and_o_proj(hidden_states, attn_out)
            if self.register_to_config:
                output = create_mla_outputs(hidden_states, self.layer_idx_str)
                _bcg_mla_custom_op_inplace(hidden_states, self.layer_idx_str, output)
            else:
                output = self.create_output(hidden_states)
                self.run_attention(hidden_states, attn_metadata, output)
            return self._gate_and_o_proj(hidden_states, output)
        finally:
            # Consumed by the projection; never carried into another forward.
            self.next_input_mxfp8 = None


# Identity-RoPE table positions for the MLA backends: max_position_embeddings. K3 is NoPE (the
# table holds cos=1/sin=0), but the chunked-context path indexes the table by absolute position,
# so it must cover max_position_embeddings (~512MB per backend for the 1M-position checkpoint); a
# smaller table is read out of bounds.


def _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo: Optional[QuantAlgo]) -> frozenset:
    """MLA layers whose KV cache falls back to FP8 under NVFP4 KV: the engine's cache policy
    decides (``kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids``, which the FP4 MLA cache
    manager reads too, so the layer and its cache agree; under NVFP4 the last two MLA layers)."""
    return _kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids(
        cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4
    )


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
        raise ValueError(
            f"Kimi K3 attention projection {names[0]} has unsupported checkpoint "
            f"quantization {selected.quant_algo}"
        )
    if selected.quant_algo == QuantAlgo.FP8_BLOCK_SCALES and selected.group_size not in (None, 128):
        raise ValueError(f"Kimi K3 attention requires 128x128 FP8 blocks for {names[0]}")
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


# ---------------------------------------------------------------------------
# MLA runtime.
# ---------------------------------------------------------------------------


class K3MLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
        *,
        constants: K3Sm107Constants,
    ) -> None:
        super().__init__()

        from .mla import K3MLAAttention

        max_positions = int(cfg.max_position_embeddings)
        self.layer_idx = layer_idx
        # K3MLAAttention owns MLA projection/head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        # Helix: mapping_with_cp (the CP original) activates the base MLA's
        # helix machinery; this wrapper's allreduce over the repurposed
        # mapping sums the base o_proj's tp*cp partials. reduce_output=False
        # leaves that reduction to the caller (helix_a2a reduce-scatter).
        mapping = model_config.mapping.mapping_for_full_rows()
        reduce_output = reduce_output and not mapping.enable_attention_dp and mapping.tp_size > 1
        self._o_allreduce = (
            TpAllReduce(mapping, torch.bfloat16, model_config.allreduce_strategy)
            if reduce_output
            else None
        )
        # Set by the decoder layer for one forward: a pushed o_proj all-reduce may be returned
        # unfinished, as a LamportPending that A_mid finishes.
        self.ar_finish_deferred = False
        attention_config = copy.copy(model_config)
        attention_config._frozen = False
        attention_config.quant_config_dict = {
            name: resolve_attention_quant_config(model_config, layer_idx, name)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "g_proj",
                "o_proj",
            )
        }
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        fp8_kv_layer_ids = _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo)
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    "The Kimi K3 FP8 KV-cache fallback layers require the model-wide KV cache "
                    "dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
            for projection_config in attention_config.quant_config_dict.values():
                projection_config.kv_cache_quant_algo = kv_quant_algo
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        attention_config._frozen = model_config._frozen
        self.mixer = K3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            constants=constants,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        o_proj = self.mixer.o_proj
        # o_proj stores its partial straight into the all-reduce's Lamport buffers where the
        # low-M GEMM serves it.
        ar_push = None
        if self._o_allreduce is not None and hasattr(o_proj, "next_ar_push"):
            ar_push = lamport_push_target(
                self._o_allreduce,
                hidden_states.shape[0],
                o_proj.out_features,
                deferred=self.ar_finish_deferred,
            )
            o_proj.next_ar_push = ar_push
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
        if ar_push is not None:
            o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # A_mid may finish the all-reduce (a LamportPending).
            out_shape = out.shape
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=self.ar_finish_deferred)
            if not isinstance(out, LamportPending):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            # o_proj ran without the push epilogue; A_mid finishes.
            lamport_push_tensor(self._o_allreduce, ar_push, out.reshape(ar_push.output.shape))
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=True)
        elif self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out
