# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix MLA decode with the V up-projection before the all-to-all (``TRTLLM_K3LL_HELIX_V_UP_BEFORE_A2A``).

The Helix combine of a head is a softmax-weighted sum of the CP ranks' partial latents, and the V up-projection
``W_UV[h]`` (512 -> 128) is linear, so it commutes with the combine: each rank applies every head's ``W_UV`` to its
own partials, and the all-to-all carries 128 values per head instead of the 512-wide latent::

    split read (all heads, this rank's pages) -> per head: combine the KV splits, x W_UV[h]   [T, 96, 128] + stats
    all-to-all + combine of this rank's head group, x sigmoid(gate)                          [T, 6 * 128]

``W_UV`` is the BF16 dequantization of the checkpoint's FP8 V-B codes and 128 x 128 block scales.

* ``trtllm::k3_mla_decode_fp8_v_up``: the CuTe DSL FP8 MLA decode of ``trtllm::cute_dsl_mla_decode_fp8_blackwell``
  (same runner, same tactic: the split kernel and its partials are that op's), with its split-KV reduction replaced
  by ``mla_decode_fp8_v_up.BlackwellMlaDecodeFP8VUp``'s reduction fused with the V up-projection, up to
  ``MAX_ROWS`` rows per head. Other shapes (more rows, or a tactic without split-KV partials) run the op itself and
  a BF16 BMM of its latent rows.
* ``trtllm::k3_helix_a2a_gated``: ``helix_a2a_fused`` at 128 values per head (the split read's ``(max, sum)`` per
  head as they are) with the gate in its combine (``cute_dsl_kernels/blackwell/helix_a2a_gated.py``).
* ``trtllm::k3_mla_gate``: ``values * sigmoid(gate)`` per head, the gate half of ``k3_fp8_mla_output`` (after the
  NCCL exchange, ``TRTLLM_K3LL_HELIX_A2A_FUSED=0``).
"""

from typing import List, Optional

import torch
import triton
import triton.language as tl

from . import mla_decode_pdl

#: Latent, rope and V widths of Kimi-K3's absorbed MLA.
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
V_HEAD_DIM = 128

_compiled: dict = {}


def max_fused_rows() -> int:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8_v_up import (
        MAX_ROWS,
    )

    return MAX_ROWS


def _run_fused(
    runner, inputs, tactic, softmax_scale, output_scale, v_weight, v_out, prewait=False
) -> None:
    """``runner.forward`` with the V up-projection fused into the reduction (``split_kv > 1``, rows <= MAX_ROWS).
    ``prewait``: the split kernel L2-prefetches ``v_weight`` before its PDL wait and releases the fused reduction
    after it (``TRTLLM_K3LL_MLA_PREWAIT_L2``; with the runner's PDL only; same bytes)."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8_v_up import (
        BlackwellMlaDecodeFP8VUp,
        v_up_rows_bucket,
    )

    (
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        cache_seqs,
        o,
        workspace,
        softmax_stats,
        kv_bounds,
    ) = inputs
    mma_qk, mma_pv, split_kv, is_persistent = tactic
    mma_qk, mma_pv, split_kv, is_persistent = (
        tuple(mma_qk),
        tuple(mma_pv),
        int(split_kv),
        bool(is_persistent),
    )
    heads, seq_len_q = runner.num_heads, runner.seq_len_q
    batch = cache_seqs.shape[0]
    rows = batch * seq_len_q
    v_rows = v_up_rows_bucket(rows)
    max_batch = max(batch, runner.max_batch_size)
    lse_offset, lse_size, split_offset, split_size, need = runner.get_workspace_layout(
        heads, seq_len_q, KV_LORA_RANK, max_batch, cutlass.Float32
    )
    ws = workspace.view(torch.uint8).reshape(-1)
    if ws.numel() < need:
        raise RuntimeError(f"k3_mla_decode_fp8_v_up: workspace {ws.numel()} B < {need} B")
    lse = (
        ws[lse_offset : lse_offset + lse_size]
        .view(torch.float32)
        .view(max_batch, seq_len_q, heads)[:batch]
        .permute(2, 1, 0)
    )
    split_ws = ws[split_offset : split_offset + split_size]
    stats_k = softmax_stats.view(batch, seq_len_q, heads, 2).permute(2, 1, 0, 3)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    key = runner.unique_id() + (
        mma_qk,
        mma_pv,
        split_kv,
        is_persistent,
        kv_bounds is not None,
        v_rows,
        runner.use_pdl,
    )
    prewait = bool(prewait and runner.use_pdl)
    if prewait:
        key = key + ("prewait",)
    compiled = _compiled.get(key)
    if compiled is None:
        from tensorrt_llm.logger import logger

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"k3_mla_decode_fp8_v_up: variant {key} not compiled before capture")
        logger.info(f"K3 MLA decode with the fused V up-projection: compiling {key} rows={rows}")
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
        mla = BlackwellMlaDecodeFP8VUp(
            cutlass.Float32,
            cutlass.Float32,
            mma_qk,
            mma_pv,
            clusters,
            runner.page_size,
            runner._SKIP_CORRECTION_THRESHOLD,
            is_persistent,
            runner._IS_VAR_SEQ,
            runner._IS_VAR_SPLIT_KV,
            num_heads=heads,
            seq_len_q=seq_len_q,
            reducer_max_splits=32,
            emit_softmax_stats=True,
            use_pdl=runner.use_pdl,
            v_rows=v_rows,
            prewait=prewait,
        )

        def dyn(t, lead, align=16):
            return cute.runtime.from_dlpack(t, assumed_align=align).mark_layout_dynamic(
                leading_dim=lead
            )

        args = [
            dyn(q_latent, 1),
            dyn(q_rope, 1),
            dyn(c_latent, 1),
            dyn(c_rope, 1),
            dyn(page_table, 0),
            cute.runtime.from_dlpack(o, assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, stride_order=(3, 2, 0, 1), divisibility=8),
            dyn(lse, 0),
            dyn(stats_k, 3),
            cute.runtime.from_dlpack(split_ws, assumed_align=32).mark_layout_dynamic(),
            split_kv,
            cute.runtime.from_dlpack(cache_seqs, assumed_align=16).mark_layout_dynamic(),
            None
            if kv_bounds is None
            else cute.runtime.from_dlpack(kv_bounds, assumed_align=4).mark_layout_dynamic(),
            cutlass.Float32(softmax_scale),
            cutlass.Float32(output_scale),
            cute.runtime.from_dlpack(v_weight, assumed_align=16),
            dyn(v_out, 2),
            stream,
        ]
        compiled = cute.compile(mla.run_with_softmax_stats_v_up, *args, options="--opt-level 2")
        _compiled[key] = compiled
    page_table_arg = page_table
    if page_table.shape[0] == 1 and page_table.shape[1] == 1:
        # a (1, 1) table has no extent > 1 to deduce its leading dimension from at call time
        page_table_arg = cute.runtime.from_dlpack(page_table, assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        )
    compiled(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table_arg,
        o,
        lse,
        stats_k,
        split_ws,
        split_kv,
        cache_seqs,
        kv_bounds,
        softmax_scale,
        output_scale,
        v_weight,
        v_out,
        stream,
    )


@torch.library.custom_op(
    "trtllm::k3_mla_decode_fp8_v_up",
    mutates_args=("latent", "workspace", "softmax_stats", "v_out"),
    device_types="cuda",
)
def mla_decode_fp8_v_up(
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqs: torch.Tensor,
    kv_bounds: Optional[torch.Tensor],
    latent: torch.Tensor,
    workspace: torch.Tensor,
    softmax_stats: torch.Tensor,
    v_weight: torch.Tensor,
    v_out: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    max_batch_size: int,
    use_pdl: bool,
    prewait: bool,
) -> None:
    """All-head FP8 MLA decode over this rank's pages followed by every head's V up-projection.

    Args:
        q: ``[batch, q_len, heads, 576]`` FP8 absorbed queries.
        kv_pages: ``[num_pages, page_size, >=576]`` FP8 page pool.
        page_table: ``[local_blocks, batch]`` int32 (dim-0-major view); ``cache_seqs``: ``[batch]`` int32;
            ``kv_bounds``: ``[batch * q_len]`` int32 or None (as ``trtllm::cute_dsl_mla_decode_fp8_blackwell``).
        latent: ``[batch, q_len, heads, 512]`` bf16 scratch: the decode's latent output where the fused reduction
            does not apply (else not written).
        softmax_stats: ``[batch * q_len, heads, 2]`` fp32 Helix ``(max, sum)`` per row (written).
        v_weight: ``[heads, 128, 512]`` bf16 ``W_UV`` of every head, contiguous.
        v_out: ``[batch * q_len, heads, 128]`` bf16 (written).
        use_pdl: the split kernel (and the unfused reduction) with programmatic dependent launch, as
            ``trtllm::k3_mla_decode_fp8_pdl`` (``TRTLLM_K3LL_MLA_DECODE_PDL``); the fused reduction always uses it.
        prewait: with ``use_pdl`` and the fused reduction, the split kernel L2-prefetches ``v_weight`` before its
            wait and releases the reduction after it (``TRTLLM_K3LL_MLA_PREWAIT_L2``; same bytes).
    """
    batch, q_len, heads, _ = q.shape
    page_size = kv_pages.shape[1]
    runner = mla_decode_pdl.runner(heads, q_len, page_size, max_batch_size, True, use_pdl)
    inputs = [
        q[..., :KV_LORA_RANK].permute(2, 3, 1, 0),
        q[..., KV_LORA_RANK:].permute(2, 3, 1, 0),
        kv_pages[..., :KV_LORA_RANK].permute(1, 2, 0),
        kv_pages[..., KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_DIM].permute(1, 2, 0),
        page_table,
        cache_seqs,
        latent.permute(2, 3, 1, 0),
        workspace,
        softmax_stats,
        kv_bounds,
    ]
    tactic = mla_decode_pdl.tactic(runner, inputs)
    if int(tactic[2]) > 1 and batch * q_len <= max_fused_rows():
        _run_fused(runner, inputs, tactic, softmax_scale, output_scale, v_weight, v_out, prewait)
        return
    runner(inputs, tactic=tactic, softmax_scale=softmax_scale, output_scale=output_scale)
    torch.ops.trtllm.bmm_out(
        latent.view(batch * q_len, heads, KV_LORA_RANK).transpose(0, 1),
        v_weight.transpose(1, 2),
        v_out.transpose(0, 1),
    )


@mla_decode_fp8_v_up.register_fake
def _(
    q,
    kv_pages,
    page_table,
    cache_seqs,
    kv_bounds,
    latent,
    workspace,
    softmax_stats,
    v_weight,
    v_out,
    softmax_scale,
    output_scale,
    max_batch_size,
    use_pdl,
    prewait,
) -> None:
    return None


def fused_applies(batch: int, q_len: int, heads: int, page_size: int, max_batch_size: int) -> bool:
    """Whether ``k3_mla_decode_fp8_v_up`` takes the fused reduction for this shape with the default tactic (for
    logs; the op decides on the tuner's tactic)."""
    if batch * q_len > max_fused_rows():
        return False
    runner = mla_decode_pdl.runner(heads, q_len, page_size, max_batch_size, True, False)
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    blocks = CuteDSLNVMlaDecodeBlackwellRunner._get_max_active_blocks()
    return runner.get_default_split_kv(batch, -(-heads * q_len // 128), blocks) > 1


@triton.jit
def _mla_gate(
    VALUES, GATE, OUT, H: tl.constexpr, GATE_STRIDE: tl.constexpr, GATE_COL: tl.constexpr
):
    row, head = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 128)
    values = tl.load(VALUES + (row * H + head) * 128 + k).to(tl.float32)
    gate = tl.sigmoid(tl.load(GATE + row * GATE_STRIDE + GATE_COL + head * 128 + k).to(tl.float32))
    tl.store(OUT + (row * H + head) * 128 + k, values * gate)


@torch.library.custom_op("trtllm::k3_mla_gate", mutates_args=())
def mla_gate(values: torch.Tensor, gate: torch.Tensor, gate_col: int) -> torch.Tensor:
    """``values * sigmoid(gate[:, gate_col : gate_col + H * 128])`` per head (the output gate of
    ``k3_fp8_mla_output``). ``values``: ``[T, H * 128]`` bf16, contiguous; ``gate``: ``[T, >= gate_col + H * 128]``
    bf16 rows."""
    tokens = values.shape[0]
    h = values.shape[1] // V_HEAD_DIM
    output = values.new_empty((tokens, h * V_HEAD_DIM))
    _mla_gate[(tokens, h)](values, gate, output, h, gate.stride(0), gate_col, num_warps=4)
    return output


@mla_gate.register_fake
def _gate_fake(values, gate, gate_col):
    return values.new_empty(values.shape)


@torch.library.custom_op(
    "trtllm::k3_helix_a2a_gated", mutates_args=("out", "state"), device_types="cuda"
)
def helix_a2a_gated_op(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
    gate: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    params: List[int],
) -> None:
    """One launch of ``helix_a2a_gated`` into ``out``; ``params`` as ``trtllm::helix_a2a_fused``'s (cp_rank,
    cp_size, num_heads, head_dim, ws_addr, rank_stride, half_bytes, max_send_ctas, max_recv_ctas, use_pdl)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.helix_a2a_gated import helix_a2a_gated

    cp_rank, cp_size, num_heads, head_dim, ws_addr, rank_stride, half_bytes, n_send, n_recv, pdl = (
        params
    )
    helix_a2a_gated(
        partial_o,
        softmax_stats,
        zero_kv_mask,
        out,
        gate,
        cp_rank=cp_rank,
        cp_size=cp_size,
        num_heads=num_heads,
        head_dim=head_dim,
        ws_addr=ws_addr,
        rank_stride=rank_stride,
        half_bytes=half_bytes,
        state=state,
        max_send_ctas=n_send,
        max_recv_ctas=n_recv,
        use_pdl=bool(pdl),
    )


@helix_a2a_gated_op.register_fake
def _(partial_o, softmax_stats, zero_kv_mask, gate, out, state, params) -> None:
    return None


def exchange_combine_gated(
    v_part: torch.Tensor,
    softmax_stats: torch.Tensor,
    empty: torch.Tensor,
    gate: torch.Tensor,
    fused,
    heads_per_rank: int,
) -> torch.Tensor:
    """The exchange of every head group's projected partials to its owner, their combine and the output gate, in one
    launch on ``fused``'s buffers (the CP group's ``HelixAllToAllFused``).

    Args:
        v_part: ``[tokens, cp * heads_per_rank * 128]`` bf16, the projected partials of every head, contiguous.
        softmax_stats: ``[tokens, cp * heads_per_rank, 2]`` fp32 ``(max, sum)`` per head, as the split read wrote them.
        empty: ``[tokens]`` bool, tokens without a key on this rank (the sender sends the combine's identity).
        gate: ``[tokens, heads_per_rank * 128]`` bf16 view, the pre-sigmoid gate of this rank's heads.
    Returns:
        ``[tokens, heads_per_rank * 128]`` bf16, ``combine * sigmoid(gate)`` of this rank's heads.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.helix_a2a_fused import record_bytes

    tokens = v_part.shape[0]
    cp = fused.mapping.cp_size
    if tokens * cp * record_bytes(heads_per_rank, V_HEAD_DIM) > fused.half_bytes:
        raise ValueError(f"{tokens} tokens do not fit one half of the fused exchange buffer")
    out = v_part.new_empty((tokens, heads_per_rank * V_HEAD_DIM))
    torch.ops.trtllm.k3_helix_a2a_gated(
        v_part,
        softmax_stats.reshape(tokens, -1),
        empty[:tokens].contiguous(),
        gate,
        out,
        fused.state,
        [
            fused.mapping.cp_rank,
            cp,
            heads_per_rank,
            V_HEAD_DIM,
            fused.ws_addr,
            fused.rank_stride,
            fused.half_bytes,
            fused.max_send_ctas,
            fused.max_recv_ctas,
            int(fused.use_pdl),
        ],
    )
    return out


def dequantize_v_b(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """BF16 ``W_UV`` ``[H, 128, 512]`` of FP8 V-B codes ``[H, 128, 512]`` and their 128 x 128 block scales
    ``[H, 1, 4]`` (fp32): ``bf16(code * scale)``."""
    expanded = scales.float().repeat_interleave(128, -2).repeat_interleave(128, -1)
    return (codes.float() * expanded).to(torch.bfloat16).contiguous()
