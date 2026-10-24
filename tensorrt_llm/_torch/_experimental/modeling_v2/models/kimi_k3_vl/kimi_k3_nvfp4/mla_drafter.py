# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stock MLA DSpark drafter's block decode with its residual adds and RMSNorms in its all-reduces.

`K3MLADrafterNorms.install` replaces a ``MLADSparkForCausalLM``'s ``dflash_forward`` with
`K3MLADrafterNorms.forward`: the same absorbed-MLA block decode (projections, RoPE, the paged attention kernel the
drafter's backend picks), except that each layer's two row-parallel projections end in one collective each on the TP
group's decode state (``decode_comm.K3DecodeComm``):

* the output projection, its all-reduce, the residual add and the post-attention RMSNorm: ``comm/k3_sandwich_plain``;
* the MLP's SiLU-and-mul and down projection, its all-reduce, the residual add and the next layer's input RMSNorm
  (after the last layer, the final norm): ``comm/k3_sandwich_plain``'s SiLU-and-mul form, on gate / up from the
  ``drafter_gate_up`` decode GEMV.

Where the sandwich does not take a call (more rows than one token tile), the projection runs alone and
``comm/mnnvl_fusion_allreduce`` applies the add and the norm. A block the fused all-reduces do not hold (more rows than
the MNNVL workspace, an eager drafter backend, no page table) runs the stock forward.

With ``TRTLLM_K3LL_DRAFTER_GEMM`` the projections no decode GEMV site or sandwich takes at that row count (``[q_a |
kv_a]`` above one token tile, q_b at every row count, gate_up above one token tile) run on the low-M split-K GEMM at
pinned tactics (`mla_drafter_gemm`) instead of cuBLAS.

With ``TRTLLM_K3LL_DRAFTER_WIDE_AR`` the drafter's fused all-reduces of more rows than one token tile (the
``comm/mnnvl_fusion_allreduce`` calls here and the context projection's, `ar_ceiling`) send two-shot above the wide
decode steps' one-shot ceiling (``decode_comm.WIDE_AR_ONE_SHOT_MAX_BYTES``), not the decode path's.

With ``TRTLLM_K3LL_DRAFTER_PDL`` the drafter's own kernels launch with programmatic dependent launch:
``k3_drafter_mla_a_post`` (its norm weights loaded before the dependency wait), the query RoPE
(``cute_dsl_dspark_rope_into``) and the bf16 block decode's split and reduction (``k3_mla_decode_fp16_pdl``).

With ``TRTLLM_K3LL_DRAFTER_GLUE`` the block's layer-invariant state (`_fixup`) is built with fewer launches (cached
arange tensors and block mask, ``kv_bounds`` without ``repeat_interleave``, no int64 round trip of an int32 page
table) and the V up-projection writes its bmm straight into the o_proj input layout (`_v_up`). Both are the same
values.
"""

from typing import Callable, FrozenSet, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import (
    _build_mla_block_fixup,
    _mla_rope_into,
    _MLABlockFixup,
    _rmsnorm_rope_batched,
)
from tensorrt_llm.logger import logger

from . import census as _census
from . import decode_comm as _decode_comm
from . import flags as _flags
from . import mla_decode_pdl  # noqa: F401  (registers trtllm::k3_mla_decode_fp16_pdl)
from . import mla_drafter_gemm as _gemm
from . import mla_drafter_qkv as _qkv


def _plain_norm(norm: nn.Module, hidden: int) -> bool:
    """A plain bf16 RMSNorm of ``hidden`` columns: no Gemma offset, no quantized or high-precision output."""
    weight = getattr(norm, "weight", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (hidden,)
        and hasattr(norm, "variance_epsilon")
        and not getattr(norm, "use_gemma", True)
        and not getattr(norm, "is_nvfp4", True)
        and not getattr(norm, "return_hp_output", True)
    )


def ar_ceiling(rows: int) -> int:
    """The one-shot ceiling of the drafter's fused all-reduce of ``rows`` rows: the wide decode steps' with
    ``TRTLLM_K3LL_DRAFTER_WIDE_AR`` above one token tile, else the decode path's. Every rank decides alike."""
    if _flags.TRTLLM_K3LL_DRAFTER_WIDE_AR and rows > _decode_comm.SANDWICH_MAX_TOKENS:
        return _decode_comm.WIDE_AR_ONE_SHOT_MAX_BYTES
    return _decode_comm.DECODE_AR_ONE_SHOT_MAX_BYTES


def _plain_swiglu(mlp: nn.Module) -> bool:
    """A GatedMLP whose ``_apply_activation`` is ``trtllm::silu_and_mul`` on bf16 (plain SiLU, no limit / alpha / beta,
    a bf16 down projection) and whose gate_up is a plain bf16 projection."""
    down = getattr(mlp, "down_proj", None)
    return (
        getattr(mlp, "activation", None) is torch.nn.functional.silu
        and not getattr(mlp, "swiglu_limit", None)
        and getattr(mlp, "swiglu_alpha", None) in (None, 1.0)
        and getattr(mlp, "swiglu_beta", None) in (None, 0.0)
        and down is not None
        and not getattr(down, "has_fp8_qdq", False)
        and not getattr(down, "has_w4a8_nvfp4_fp8", False)
        and _plain_linear(getattr(mlp, "gate_up_proj", None))
    )


def _plain_linear(linear: nn.Module) -> bool:
    """A bias-free, unquantized bf16 projection, replicated or column-parallel without its output gather: its forward
    is ``F.linear(x, weight)``."""
    weight = getattr(linear, "weight", None)
    mode = getattr(getattr(linear, "tp_mode", None), "name", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and getattr(linear, "bias", None) is None
        and not getattr(linear, "has_any_quant", False)
        # reduce_output only acts on a row-parallel projection, gather_output only on a column-parallel one
        and mode in (None, "COLUMN")
        and not (mode == "COLUMN" and getattr(linear, "gather_output", False))
    )


def _reduces(linear: nn.Module) -> bool:
    """A bias-free row-parallel projection with its all-reduce."""
    return (
        getattr(getattr(linear, "tp_mode", None), "name", None) == "ROW"
        and getattr(linear, "reduce_output", False)
        and getattr(linear, "all_reduce", None) is not None
        and getattr(linear, "bias", None) is None
    )


class BlockPrep(NamedTuple):
    """The block decode's layer-invariant inputs, computed by the worker together with the drafter's inputs
    (``TRTLLM_K3LL_DRAFTER_PREP_FUSED``, `mla_drafter_prep.drafter_input_prep`) with the values `forward` would build
    from them: the fixup of ``num_ctx`` and the page-table rows ``[rows_start, rows_start + B)``, the pool slots of the
    block's latents for a pool of ``pool_strides`` (``(stride(0), stride(3))``) and the query RoPE rows
    ``freqs_cis[query_positions]``. `forward` takes it only for the call whose ``query_positions``,
    ``num_ctx_per_req`` and ``ctx_page_table`` are these very tensors."""

    query_positions: torch.Tensor
    num_ctx: torch.Tensor
    page_table: torch.Tensor
    rows_start: int
    pages: torch.Tensor
    offsets: torch.Tensor
    seq_lens_i32: torch.Tensor
    kv_bounds_i32: torch.Tensor
    slots: torch.Tensor
    pool_strides: Tuple[int, int]
    blk_freqs: torch.Tensor


class K3MLADrafterNorms:
    """A ``MLADSparkForCausalLM``'s block decode on the TP group's decode state (the module docstring).

    ``drafter``: the drafter; ``comm``: the group's state; ``gemvs_owner``: the module whose ``decode_gemvs`` (the
    target's decode GEMVs, ``drafter_gate_up``; None until built) is read at each call; ``stock``: the drafter's own
    ``dflash_forward``; ``forms``: the ``k3_sandwich_plain`` forms that compiled (``o_proj``, ``down``);
    ``a_weights``: with the fused query / latent path (`install`'s ``qkv``), each layer's ``[q_a_proj | kv_a_proj]``
    weight, else None; ``gemm``: with the launch value of ``TRTLLM_K3LL_DRAFTER_GEMM`` on, the pinned low-M GEMMs
    (`_gemm`, the switch read again at each call), else None."""

    block_prep: Optional[BlockPrep] = None

    def __init__(
        self,
        drafter: nn.Module,
        comm: _decode_comm.K3DecodeComm,
        gemvs_owner: nn.Module,
        stock: Callable[..., torch.Tensor],
        forms: FrozenSet[str],
        a_weights: Optional[List[torch.Tensor]] = None,
        gemm: Optional[_gemm.DrafterGemm] = None,
    ):
        self.drafter = drafter
        self.comm = comm
        self.gemvs_owner = gemvs_owner
        self.stock = stock
        self.forms = forms
        self.a_weights = a_weights
        self.gemm = gemm
        # TRTLLM_K3LL_DRAFTER_GLUE's per-shape constants (`_const`), built outside CUDA-graph capture.
        self._consts = {}
        # The next call's precomputed block state (`BlockPrep`), set by the worker; every call clears it.
        self.block_prep = None

    @staticmethod
    def layers_take(drafter: nn.Module) -> bool:
        """Whether every layer's norms and projections are ones the fused all-reduces reproduce: plain bf16 RMSNorms
        of the hidden width (input, post-attention, final) and row-parallel output and down projections with their
        all-reduces, the MLP a plain SiLU-and-mul one."""
        model = getattr(drafter, "model", None)
        layers = getattr(model, "layers", None)
        if not layers:
            return False
        hidden = layers[0].self_attn.hidden_size
        return _plain_norm(model.norm, hidden) and all(
            _plain_norm(layer.input_layernorm, hidden)
            and _plain_norm(layer.post_attention_layernorm, hidden)
            and _reduces(layer.self_attn.o_proj)
            and _reduces(layer.mlp.down_proj)
            and not getattr(layer.mlp, "split_gate_up", False)
            and getattr(getattr(layer.mlp, "gate_up_proj", None), "bias", 0) is None
            and layer.mlp.activation is torch.nn.functional.silu
            and getattr(layer.mlp, "swiglu_limit", None) is None
            for layer in layers
        )

    @classmethod
    def install(
        cls,
        drafter: nn.Module,
        comm: _decode_comm.K3DecodeComm,
        gemvs_owner: nn.Module,
        qkv: bool = False,
    ) -> Optional["K3MLADrafterNorms"]:
        """Bind the fused block decode as ``drafter.dflash_forward`` where its layers take it (`layers_take`); returns
        the installed object, or None (the drafter keeps its stock forward). Collective: every rank of the group calls
        it at the same point, before any CUDA-graph capture; it compiles each ``k3_sandwich_plain`` form every layer's
        shape shares with one call on a zero row of a zero weight, a form that does not compile is not used.

        ``qkv``: also run each layer's two a-projections as one GEMV over ``[q_a_proj | kv_a_proj]`` (the ``kv_a``
        decode site) followed by ``trtllm::k3_drafter_mla_a_post``, where the layers' shapes are the kernel's; the two
        projections' weights then become views of the concatenated one (no extra memory)."""
        if isinstance(getattr(drafter, "dflash_forward", None), _Bound):
            return drafter.dflash_forward.owner
        if not cls.layers_take(drafter):
            return None
        layers = drafter.model.layers
        forms = set()
        for form, swiglu, weights in (
            ("o_proj", False, [layer.self_attn.o_proj.weight for layer in layers]),
            ("down", True, [layer.mlp.down_proj.weight for layer in layers]),
        ):
            if len({tuple(w.shape) for w in weights}) == 1 and comm.compile_plain(
                weights[0], swiglu=swiglu
            ):
                forms.add(form)
        a_weights = cls._fuse_a_weights(layers) if qkv else None
        owner = cls(
            drafter,
            comm,
            gemvs_owner,
            drafter.dflash_forward,
            frozenset(forms),
            a_weights,
            gemm=_gemm.DrafterGemm() if _flags.TRTLLM_K3LL_DRAFTER_GEMM else None,
        )
        drafter.dflash_forward = _Bound(owner)
        return owner

    @staticmethod
    def _fuse_a_weights(layers) -> Optional[List[torch.Tensor]]:
        """Each layer's ``[q_a_proj | kv_a_proj_with_mqa]`` weight ``[2112, hidden]``, the two projections' weights
        re-pointed at its rows; None where a layer's shapes or norms are not the kernel's."""
        for layer in layers:
            attn = layer.self_attn
            if (
                attn.q_lora_rank != _qkv.Q_LORA
                or attn.kv_lora_rank != _qkv.KV_LORA
                or attn.qk_rope_head_dim != _qkv.ROPE
                or attn.qk_nope_head_dim != _qkv.NOPE
                or attn.v_head_dim != _qkv.NOPE
                or attn.q_a_proj.weight.dtype != torch.bfloat16
                or attn.kv_a_proj_with_mqa.weight.dtype != torch.bfloat16
                or getattr(attn.q_a_proj, "bias", None) is not None
                or getattr(attn.kv_a_proj_with_mqa, "bias", None) is not None
                or not attn.q_b_proj.weight.is_contiguous()
            ):
                return None
        fused = []
        for layer in layers:
            attn = layer.self_attn
            weight = torch.cat([attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight]).contiguous()
            attn.q_a_proj.weight.data = weight[: _qkv.Q_LORA]
            attn.kv_a_proj_with_mqa.weight.data = weight[_qkv.Q_LORA :]
            fused.append(weight)
        return fused

    def takes(
        self,
        noise_embedding: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor],
        ctx_page_table: Optional[torch.Tensor],
    ) -> bool:
        """Whether the fused block decode runs this block: a bf16 block on a paged kernel backend whose rows the
        MNNVL workspace holds."""
        drafter = self.drafter
        if (
            ctx_kv_cache is None
            or ctx_page_table is None
            or noise_embedding.dtype != torch.bfloat16
            or noise_embedding.dim() != 3
            or drafter._mla_block_decode_variant(paged=True) == "eager"
        ):
            return False
        rows = noise_embedding.shape[0] * noise_embedding.shape[1]
        return self.comm.takes_allreduce_norm(rows, noise_embedding.shape[2], ar_ceiling(rows))

    def forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: torch.Tensor,
        ctx_v_cache: Optional[torch.Tensor],
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor] = None,
        ctx_page_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The block's hidden states ``[B * block, hidden]`` after the final norm; arguments as the drafter's
        ``dflash_forward``. A `BlockPrep` the worker left for this call (``block_prep``) replaces the fixup, the pool
        slots and the query RoPE rows built here (the same values)."""
        prep, self.block_prep = self.block_prep, None
        if not self.takes(noise_embedding, ctx_kv_cache, ctx_page_table):
            return self.stock(
                noise_embedding,
                query_positions,
                num_ctx_per_req,
                ctx_k_cache,
                ctx_v_cache,
                ctx_cache_batch_idx,
                ctx_kv_cache,
                ctx_page_table,
            )
        drafter = self.drafter
        if drafter._fused_kv_weight is None:
            drafter._build_fused_kv_buffers()
        batch, block = noise_embedding.shape[:2]
        rows = batch * block
        page_size = ctx_kv_cache[0].shape[-2]
        if prep is not None and not (
            prep.query_positions is query_positions
            and prep.num_ctx is num_ctx_per_req
            and prep.page_table is ctx_page_table
            and ctx_page_table.dtype == torch.int32
            and tuple(prep.pages.shape) == (batch, block)
        ):
            prep = None
        if prep is not None:
            _census.count(f"dispatch:drafter_block_prep:{_census.mode()}")
            blk_freqs = prep.blk_freqs
            fixup = _MLABlockFixup(
                pages=prep.pages,
                offsets=prep.offsets,
                valid=self._const("valid", block, noise_embedding.device),
                page_tables_i32=ctx_page_table[prep.rows_start : prep.rows_start + batch],
                seq_lens_i32=prep.seq_lens_i32,
                kv_bounds_i32=prep.kv_bounds_i32,
            )
        else:
            blk_freqs = drafter._mla_freqs_cis(noise_embedding.device)[query_positions.long()]
            if _flags.TRTLLM_K3LL_DRAFTER_GLUE:
                _census.count(f"dispatch:drafter_glue:{_census.mode()}")
                fixup = self._fixup(
                    num_ctx_per_req[:batch], ctx_page_table, ctx_cache_batch_idx, block, page_size
                )
            else:
                page_tables = ctx_page_table.index_select(0, ctx_cache_batch_idx.long()).long()
                fixup = _build_mla_block_fixup(
                    num_ctx_per_req[:batch], page_tables, block, page_size
                )
        capacity = fixup.page_tables_i32.shape[1] * page_size
        if _flags.TRTLLM_K3LL_DRAFTER_PDL:
            _census.count(f"dispatch:drafter_pdl:{_census.mode()}")
        layers = drafter.model.layers
        qkv = None
        if (
            self.a_weights is not None
            and drafter._mla_block_decode_variant(paged=True) == "cute_dsl"
        ):
            pool = ctx_kv_cache[0]
            if (
                all(c.stride() == pool.stride() for c in ctx_kv_cache)
                and pool.dtype == torch.bfloat16
            ):
                if prep is not None and tuple(prep.pool_strides) == (
                    pool.stride(0),
                    pool.stride(3),
                ):
                    slots = prep.slots
                else:
                    slots = (fixup.pages * pool.stride(0) + fixup.offsets * pool.stride(3)).reshape(
                        -1
                    )
                qkv = (
                    slots,
                    torch.view_as_real(blk_freqs).reshape(rows, -1).contiguous(),
                    blk_freqs,
                )
        # The fused all-reduces read the residual and return the updated one: the block's input serves as the first
        # residual, uncopied.
        residual = noise_embedding.reshape(rows, -1)
        normed = layers[0].input_layernorm(residual)
        for layer_idx, layer in enumerate(layers):
            attn_out = None
            if qkv is not None:
                attn_out = self._attention_qkv(
                    layer.self_attn,
                    self.a_weights[layer_idx],
                    normed,
                    qkv,
                    ctx_kv_cache[layer_idx],
                    fixup,
                    batch,
                    block,
                )
            if attn_out is None:
                attn_out = self._attention(
                    layer.self_attn,
                    normed,
                    blk_freqs,
                    ctx_kv_cache[layer_idx],
                    fixup,
                    batch,
                    block,
                    capacity,
                )
            next_norm = (
                layers[layer_idx + 1].input_layernorm
                if layer_idx + 1 < len(layers)
                else drafter.model.norm
            )
            normed, residual = self._project_norm(
                attn_out, layer.self_attn.o_proj, residual, layer.post_attention_layernorm
            )
            normed, residual = self._mlp_norm(layer.mlp, normed, residual, next_norm)
        return normed

    def _attention_qkv(
        self,
        attn: nn.Module,
        a_weight: torch.Tensor,
        normed: torch.Tensor,
        qkv: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layer_cache: torch.Tensor,
        fixup,
        batch: int,
        block: int,
    ) -> Optional[torch.Tensor]:
        """`_attention` with both a-projections as one GEMV over ``[q_a | kv_a]`` and ``trtllm::k3_drafter_mla_a_post``
        (the q_a RMSNorm, and the block latent's RMSNorm + RoPE stored into the pool), then the q_b projection, the
        query RoPE and nope absorption, the cute-dsl decode and the V up-projection; None where the kernel does not take
        the operands."""
        drafter = self.drafter
        gemvs = getattr(self.gemvs_owner, "decode_gemvs", None)
        a = None if gemvs is None else gemvs.project("kv_a", normed, a_weight)
        if a is None:
            a = self._gemm(normed, a_weight, "q_kv_a")
        if a is None:
            a = torch.nn.functional.linear(normed, a_weight)
        if not _qkv.supports(a, layer_cache):
            return None
        slots, freqs, blk_freqs = qkv
        pdl = bool(_flags.TRTLLM_K3LL_DRAFTER_PDL)
        q_a = torch.ops.trtllm.k3_drafter_mla_a_post(
            a,
            attn.q_a_layernorm.weight,
            attn.kv_a_layernorm.weight,
            freqs,
            layer_cache,
            slots,
            attn.q_a_layernorm.variance_epsilon,
            attn.kv_a_layernorm.variance_epsilon,
            pdl,
        )
        q = self._gemm(q_a, attn.q_b_proj, "q_b")
        if q is None:
            q = attn.q_b_proj(q_a)
        query = self._query(attn, q, blk_freqs, batch, block)
        ctx_out = drafter._cute_dsl_block_decode(
            query,
            layer_cache,
            fixup,
            block,
            decode_op=self._decode_op(),
        )
        return self._v_up(attn, ctx_out, batch, block)

    def _v_up(self, attn: nn.Module, ctx_out: torch.Tensor, batch: int, block: int) -> torch.Tensor:
        """The V up-projection of the block attention's latent output ``[B, block, heads, kv_lora_rank]``:
        ``[B * block, heads x v_head_dim]``, the o_proj input. The stock arithmetic is an einsum, one bmm over the
        heads into a ``[heads, B * block, v]`` result that the reshape then copies into this layout. With
        ``TRTLLM_K3LL_DRAFTER_GLUE`` the same bmm (same operands, strides and shapes) writes this layout directly
        through a transposed view of the output (``trtllm::bmm_out``), so the copy goes; where ``ctx_out`` is
        contiguous (the einsum's operand views are then exactly these)."""
        v_b = attn._v_b_proj.to(ctx_out.dtype)
        if not (_flags.TRTLLM_K3LL_DRAFTER_GLUE and ctx_out.is_contiguous()):
            attn_out = torch.einsum("bshc,hvc->bshv", ctx_out, v_b)
            return attn_out.reshape(batch * block, -1)
        rows, heads = batch * block, ctx_out.shape[2]
        out = torch.empty((rows, heads, v_b.shape[1]), dtype=ctx_out.dtype, device=ctx_out.device)
        torch.ops.trtllm.bmm_out(
            ctx_out.reshape(rows, heads, -1).transpose(0, 1),
            v_b.transpose(1, 2),
            out.transpose(0, 1),
        )
        return out.view(rows, -1)

    def _const(self, name: str, block: int, device: torch.device) -> torch.Tensor:
        """A per-block-size constant of `_fixup`: ``arange`` (``arange(block)``, int64) or ``valid`` (the strict
        upper-triangle mask ``[1, block, 1, block]``, the stock fixup's), built once outside CUDA-graph capture
        (inside one it is built in the graph and not kept). Read only."""
        key = (name, block, device)
        t = self._consts.get(key)
        if t is None:
            idx = torch.arange(block, device=device)
            t = (
                idx
                if name == "arange"
                else (idx.view(1, block) > idx.view(block, 1)).view(1, block, 1, block)
            )
            if not torch.cuda.is_current_stream_capturing():
                self._consts[key] = t
        return t

    def _fixup(
        self,
        ctx_len: torch.Tensor,
        ctx_page_table: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
        block: int,
        page_size: int,
    ) -> _MLABlockFixup:
        """``_build_mla_block_fixup`` of the gathered page-table rows, each field the same values with fewer launches:
        the arange and the mask are cached (`_const`), ``kv_bounds`` repeats each request's length ``block`` times
        through an expanded view (one copy, no ``repeat_interleave`` scan), and an int32 table's gathered rows are the
        int32 table itself (no int64 round trip)."""
        device = ctx_len.device
        gathered = ctx_page_table.index_select(0, ctx_cache_batch_idx.long())
        page_tables = gathered.long()
        tables_i32 = gathered if gathered.dtype == torch.int32 else page_tables.to(torch.int32)
        pos = ctx_len.view(-1, 1) + self._const("arange", block, device)
        seq_lens = (ctx_len + block).to(torch.int32)
        return _MLABlockFixup(
            pages=page_tables.gather(1, pos // page_size),
            offsets=pos % page_size,
            valid=self._const("valid", block, device),
            page_tables_i32=tables_i32,
            seq_lens_i32=seq_lens,
            # Batch-major, as the stock repeat_interleave: the kernel reads it at b * seq_len_q + q. contiguous()
            # also for one request, whose expanded view would otherwise reshape to a stride-0 view.
            kv_bounds_i32=seq_lens.view(-1, 1).expand(-1, block).contiguous().view(-1),
        )

    def _query(
        self, attn: nn.Module, q: torch.Tensor, blk_freqs: torch.Tensor, batch: int, block: int
    ) -> torch.Tensor:
        """The absorbed query ``[B, block, heads, kv_lora_rank + rope]`` from q_b's output, the stock drafter's
        arithmetic: the rope half rotated into place, the nope half absorbed through ``k_b``."""
        drafter = self.drafter
        num_heads = drafter._num_heads
        nope, rope, latent = (
            drafter.qk_nope_head_dim,
            drafter.qk_rope_head_dim,
            drafter.kv_lora_rank,
        )
        q = q.view(batch, block, num_heads, nope + rope)
        query = torch.empty(
            (batch, block, num_heads, latent + rope), dtype=q.dtype, device=q.device
        )
        if not _mla_rope_into(
            q,
            blk_freqs,
            query,
            num_heads,
            rope,
            latent,
            use_pdl=bool(_flags.TRTLLM_K3LL_DRAFTER_PDL),
        ):
            q = _rmsnorm_rope_batched(
                q,
                None,
                0.0,
                rope,
                blk_freqs,
                num_heads=num_heads,
                apply_weight=False,
                apply_rmsnorm=False,
            )
            query[..., latent:] = q[..., nope:]
        torch.ops.trtllm.bmm_out(
            q[..., :nope].view(batch * block, num_heads, nope).transpose(0, 1),
            attn._k_b_proj.to(q.dtype),
            query[..., :latent].view(batch * block, num_heads, latent).transpose(0, 1),
        )
        return query

    def _attention(
        self,
        attn: nn.Module,
        normed: torch.Tensor,
        blk_freqs: torch.Tensor,
        layer_cache: torch.Tensor,
        fixup,
        batch: int,
        block: int,
        capacity: int,
    ) -> torch.Tensor:
        """One layer's absorbed-MLA block attention, the stock drafter's arithmetic: this rank's heads' output in
        value space, ``[B * block, heads x v_head_dim]``."""
        drafter = self.drafter
        rope, latent = drafter.qk_rope_head_dim, drafter.kv_lora_rank
        query = self._query(
            attn, attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(normed))), blk_freqs, batch, block
        )
        blk_kv = _rmsnorm_rope_batched(
            attn.kv_a_proj_with_mqa(normed).view(batch, block, -1),
            attn.kv_a_layernorm.weight,
            attn.kv_a_layernorm.variance_epsilon,
            rope,
            blk_freqs,
            norm_dim=latent,
        )
        ctx_out = drafter._mla_paged_attention(
            query, blk_kv, layer_cache, fixup, block, capacity, decode_op=self._decode_op()
        )
        return self._v_up(attn, ctx_out, batch, block)

    @staticmethod
    def _decode_op():
        """The bf16 block decode op: ``trtllm::k3_mla_decode_fp16_pdl`` with ``TRTLLM_K3LL_DRAFTER_PDL``, else None
        (the stock ``trtllm::cute_dsl_mla_decode_fp16_blackwell``)."""
        return torch.ops.trtllm.k3_mla_decode_fp16_pdl if _flags.TRTLLM_K3LL_DRAFTER_PDL else None

    def _project_norm(
        self, x: torch.Tensor, linear: nn.Module, residual: torch.Tensor, norm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm(updated), updated)``, ``updated = residual + linear(x)`` for the row-parallel output projection:
        ``comm/k3_sandwich_plain`` where it takes the rows, else the module without its all-reduce, then
        ``comm/mnnvl_fusion_allreduce``."""
        comm = self.comm
        if "o_proj" in self.forms and comm.takes_plain(x, linear.weight, residual, norm):
            return comm.sandwich_plain(x, linear.weight, residual, norm)
        partial = linear(x, all_reduce_params=_decode_comm.skip_all_reduce())
        return comm.allreduce_norm(partial, residual, norm, ar_ceiling(partial.shape[0]))

    def _mlp_norm(
        self, mlp: nn.Module, x: torch.Tensor, residual: torch.Tensor, norm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm(updated), updated)``, ``updated = residual + mlp(x)``: gate / up on ``drafter_gate_up``, then
        ``comm/k3_sandwich_plain``'s SiLU-and-mul form where it takes the rows; otherwise the module without its
        all-reduce, then ``comm/mnnvl_fusion_allreduce``."""
        comm = self.comm
        gemvs = getattr(self.gemvs_owner, "decode_gemvs", None)
        gate_up = (
            None if gemvs is None else gemvs.project("drafter_gate_up", x, mlp.gate_up_proj.weight)
        )
        if (
            gate_up is not None
            and "down" in self.forms
            and comm.takes_plain(gate_up, mlp.down_proj.weight, residual, norm, swiglu=True)
        ):
            return comm.sandwich_plain(gate_up, mlp.down_proj.weight, residual, norm, swiglu=True)
        act = self._gate_up_swiglu(x, mlp)
        if act is not None:
            # GatedMLP's unfused bf16 path with gate_up and the SwiGLU as one launch.
            partial = mlp.down_proj(act, all_reduce_params=_decode_comm.skip_all_reduce())
            return comm.allreduce_norm(partial, residual, norm, ar_ceiling(partial.shape[0]))
        h1 = self._gemm(x, mlp.gate_up_proj, "gate_up")
        if h1 is not None:
            # GatedMLP's unfused bf16 path with gate_up on the pinned GEMM.
            partial = mlp.down_proj(
                mlp._apply_activation(h1), all_reduce_params=_decode_comm.skip_all_reduce()
            )
        else:
            partial = mlp(x, final_all_reduce_params=_decode_comm.skip_all_reduce())
        return comm.allreduce_norm(partial, residual, norm, ar_ceiling(partial.shape[0]))

    def _gate_up_swiglu(self, x: torch.Tensor, mlp: nn.Module) -> Optional[torch.Tensor]:
        """``silu_and_mul(gate_up(x))`` as one launch (the gate_up pin's GEMM with the SwiGLU epilogue) where
        ``TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU`` and ``TRTLLM_K3LL_DRAFTER_GEMM`` are on, the MLP's activation is the
        plain bf16 SwiGLU and the kernel takes the call; else None."""
        if (
            self.gemm is None
            or not _flags.TRTLLM_K3LL_DRAFTER_GEMM
            or not _gemm.swiglu_enabled(x.device)
            or not _plain_swiglu(mlp)
        ):
            return None
        act = self.gemm.project_swiglu(x, mlp.gate_up_proj.weight)
        if act is not None:
            _census.count(f"dispatch:drafter_gemm_gate_up_swiglu:{_census.mode()}")
        return act

    def _gemm(self, x: torch.Tensor, weight, role: str) -> Optional[torch.Tensor]:
        """``x @ W.T`` on the pinned low-M GEMM for ``role`` (``weight``: the weight, or a plain column-parallel
        bf16 ``Linear`` whose forward is exactly that) where ``TRTLLM_K3LL_DRAFTER_GEMM`` is on (now and at launch) and
        it takes the call; else None."""
        if self.gemm is None or not _flags.TRTLLM_K3LL_DRAFTER_GEMM:
            return None
        if isinstance(weight, nn.Module):
            if not _plain_linear(weight):
                return None
            weight = weight.weight
        out = self.gemm.project(x, weight, role)
        if out is not None:
            _census.count(f"dispatch:drafter_gemm_{role}:{_census.mode()}")
        return out


class _Bound:
    """``owner.forward`` as the drafter's ``dflash_forward`` attribute, recognizable so a second install is a no-op."""

    def __init__(self, owner: K3MLADrafterNorms):
        self.owner = owner

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.owner.forward(*args, **kwargs)


def log_install(owner: Optional[K3MLADrafterNorms], qkv: bool = False) -> None:
    logger.info(
        "Kimi K3 MLA DSpark drafter: residual adds and RMSNorms "
        + (
            f"in the all-reduces (k3_sandwich_plain: {sorted(owner.forms) or 'none'}, else mnnvl_fusion_allreduce)"
            if owner is not None
            else "on the stock modules (a layer norm or projection the fused all-reduces do not reproduce)"
        )
        + (
            ""
            if not qkv
            else "; query and block latent "
            + (
                "on the [q_a | kv_a] GEMV and k3_drafter_mla_a_post (cute-dsl decode)"
                if owner is not None and owner.a_weights is not None
                else "on the stock projections (shapes the kernel does not take)"
            )
        )
    )
