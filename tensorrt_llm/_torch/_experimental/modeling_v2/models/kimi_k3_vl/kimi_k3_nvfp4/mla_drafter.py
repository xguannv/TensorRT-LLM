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
"""

from typing import Callable, FrozenSet, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import (
    _build_mla_block_fixup,
    _mla_rope_into,
    _rmsnorm_rope_batched,
)
from tensorrt_llm.logger import logger

from . import decode_comm as _decode_comm


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


def _reduces(linear: nn.Module) -> bool:
    """A bias-free row-parallel projection with its all-reduce."""
    return (
        getattr(getattr(linear, "tp_mode", None), "name", None) == "ROW"
        and getattr(linear, "reduce_output", False)
        and getattr(linear, "all_reduce", None) is not None
        and getattr(linear, "bias", None) is None
    )


class K3MLADrafterNorms:
    """A ``MLADSparkForCausalLM``'s block decode on the TP group's decode state (the module docstring).

    ``drafter``: the drafter; ``comm``: the group's state; ``gemvs_owner``: the module whose ``decode_gemvs`` (the
    target's decode GEMVs, ``drafter_gate_up``; None until built) is read at each call; ``stock``: the drafter's own
    ``dflash_forward``; ``forms``: the ``k3_sandwich_plain`` forms that compiled (``o_proj``, ``down``)."""

    def __init__(
        self,
        drafter: nn.Module,
        comm: _decode_comm.K3DecodeComm,
        gemvs_owner: nn.Module,
        stock: Callable[..., torch.Tensor],
        forms: FrozenSet[str],
    ):
        self.drafter = drafter
        self.comm = comm
        self.gemvs_owner = gemvs_owner
        self.stock = stock
        self.forms = forms

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
        cls, drafter: nn.Module, comm: _decode_comm.K3DecodeComm, gemvs_owner: nn.Module
    ) -> Optional["K3MLADrafterNorms"]:
        """Bind the fused block decode as ``drafter.dflash_forward`` where its layers take it (`layers_take`); returns
        the installed object, or None (the drafter keeps its stock forward). Collective: every rank of the group calls
        it at the same point, before any CUDA-graph capture; it compiles each ``k3_sandwich_plain`` form every layer's
        shape shares with one call on a zero row of a zero weight, a form that does not compile is not used."""
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
        owner = cls(drafter, comm, gemvs_owner, drafter.dflash_forward, frozenset(forms))
        drafter.dflash_forward = _Bound(owner)
        return owner

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
        return self.comm.takes_allreduce_norm(rows, noise_embedding.shape[2])

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
        ``dflash_forward``."""
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
        blk_freqs = drafter._mla_freqs_cis(noise_embedding.device)[query_positions.long()]
        page_tables = ctx_page_table.index_select(0, ctx_cache_batch_idx.long()).long()
        page_size = ctx_kv_cache[0].shape[-2]
        capacity = page_tables.shape[1] * page_size
        fixup = _build_mla_block_fixup(num_ctx_per_req[:batch], page_tables, block, page_size)
        layers = drafter.model.layers
        # The fused all-reduces read the residual and return the updated one: the block's input serves as the first
        # residual, uncopied.
        residual = noise_embedding.reshape(rows, -1)
        normed = layers[0].input_layernorm(residual)
        for layer_idx, layer in enumerate(layers):
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
        num_heads = drafter._num_heads
        nope, rope, latent = (
            drafter.qk_nope_head_dim,
            drafter.qk_rope_head_dim,
            drafter.kv_lora_rank,
        )
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(normed)))
        q = q.view(batch, block, num_heads, nope + rope)
        query = torch.empty(
            (batch, block, num_heads, latent + rope), dtype=q.dtype, device=q.device
        )
        if not _mla_rope_into(q, blk_freqs, query, num_heads, rope, latent):
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
        blk_kv = _rmsnorm_rope_batched(
            attn.kv_a_proj_with_mqa(normed).view(batch, block, -1),
            attn.kv_a_layernorm.weight,
            attn.kv_a_layernorm.variance_epsilon,
            rope,
            blk_freqs,
            norm_dim=latent,
        )
        ctx_out = drafter._mla_paged_attention(query, blk_kv, layer_cache, fixup, block, capacity)
        attn_out = torch.einsum("bshc,hvc->bshv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))
        return attn_out.reshape(batch * block, -1)

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
        return comm.allreduce_norm(partial, residual, norm)

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
        partial = mlp(x, final_all_reduce_params=_decode_comm.skip_all_reduce())
        return comm.allreduce_norm(partial, residual, norm)


class _Bound:
    """``owner.forward`` as the drafter's ``dflash_forward`` attribute, recognizable so a second install is a no-op."""

    def __init__(self, owner: K3MLADrafterNorms):
        self.owner = owner

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.owner.forward(*args, **kwargs)


def log_install(owner: Optional[K3MLADrafterNorms]) -> None:
    logger.info(
        "Kimi K3 MLA DSpark drafter: residual adds and RMSNorms "
        + (
            f"in the all-reduces (k3_sandwich_plain: {sorted(owner.forms) or 'none'}, else mnnvl_fusion_allreduce)"
            if owner is not None
            else "on the stock modules (a layer norm or projection the fused all-reduces do not reproduce)"
        )
    )
