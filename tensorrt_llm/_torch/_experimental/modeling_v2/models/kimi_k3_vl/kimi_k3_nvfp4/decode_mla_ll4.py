# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8-cache MLA decode of the staircase on the ll4 CuTe DSL kernel (``cute_dsl_kernels/.../mla/ll4``).

The staircase's decode attention is a fixed shape: this rank's heads (6 under TP16) times ``q_len`` = 1 + draft
query tokens over each request's full FP8 latent cache (512 latent + 64 rope per token, page 64), 1-8 requests.
The shared op ``trtllm::cute_dsl_mla_decode_fp8_blackwell`` runs it with one uniform split-KV candidate, which at
batch 1 leaves most SMs idle. This runner drives the ll4 kernel directly:

* balanced split (default): ``max_active_clusters`` persistent 2-CTA clusters cut the batch's K tiles evenly, M
  tiles cut across clusters are combined by the kernel's own reduction;
* uniform split: ``split_kv`` clusters per M tile and the split-KV reduction kernel.

The operands are those of ``decode_mla_dcp.split_read_attention`` (FP8 queries ``[B, S, H, 576]``, the page pool
``[pages, 64, >=576]``, a dim-0-major ``[blocks, B]`` page table, ``[B]`` lengths including the new tokens, the
kernel's causal mask unless ``kv_bounds`` is given). The FP8 scales are device tensors: element 0 of
``softmax_scale`` is the natural-log softmax scale, element 0 of ``output_scale`` the output scale.

Host-side choices (split, kernel options, workspace layout) follow ll4's ``CuteDSLNVMlaDecodeBlackwellRunner``
for this shape only; nothing here touches the shared op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

#: Latent and rope widths of Kimi-K3's absorbed MLA.
KV_LORA_RANK = 512
QK_ROPE_DIM = 64

_MMA_TILE_M = 128
_TILERS = ((128, 128), (128, 256))
_CLUSTER_SIZE = 2
_WORKSPACE_ALIGN = 128
_ACC_BYTES = 4  # Float32 partial O / LSE
_MAX_SPLIT_KV = 32  # the kernel's can_implement bound for the uniform split
#: ll4's uniform split targets this many CTA waves (TRTLLM_MLA_SPLIT_KV_WAVES default).
_SPLIT_KV_WAVES = 8.0
#: Largest batch of the register-schedule (small-batch) balanced variant.
_SMALL_BATCH = 32
_TILE_REDUCTION_MAX_BATCH = 24
_DEEP_K_MAX_REQUESTS = 256

_max_active_clusters: Optional[int] = None


def _kernel_module():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.ll4 import mla_decode_fp8

    return mla_decode_fp8


def max_active_clusters() -> int:
    """Resident 2-CTA clusters of the decode kernel on this GPU (cached; query it before graph capture)."""
    global _max_active_clusters
    if _max_active_clusters is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MLA ll4 decode: max_active_clusters not cached before CUDA graph capture"
            )
        import cutlass

        _max_active_clusters = int(
            cutlass.utils.HardwareInfo().get_max_active_clusters(_CLUSTER_SIZE)
        )
    return _max_active_clusters


def fold_layout(num_heads: int, seq_len_q: int) -> Tuple[int, int]:
    """(rows per M tile, M tiles per request) of the folded query (6 heads x 8 tokens: 48 rows, 1 tile)."""
    kernel = _kernel_module().BlackwellMultiHeadLatentAttentionForwardFP8
    if num_heads >= _MMA_TILE_M or seq_len_q <= 1:
        return num_heads, seq_len_q
    return kernel.get_fold_layout(num_heads, seq_len_q, _MMA_TILE_M)


def default_uniform_split(batch: int, num_heads: int, seq_len_q: int) -> int:
    """ll4's uniform split_kv: about ``_SPLIT_KV_WAVES`` waves of CTAs, as a power of two, at most 32."""
    chunks = fold_layout(num_heads, seq_len_q)[1]
    target = _SPLIT_KV_WAVES * max_active_clusters() * _CLUSTER_SIZE / (2 * batch * chunks)
    return min(_MAX_SPLIT_KV, 1 << max(0, round(math.log2(target))))


def kernel_options(batch: int, balanced: bool, **overrides) -> Tuple[Tuple[str, object], ...]:
    """Keyword options of the ll4 FP8 kernel at ll4's defaults (``_fp8_kernel_options`` without its environment
    variables); ``overrides`` replaces any of them. Part of the compiled-kernel cache key."""
    small = balanced and batch <= _SMALL_BATCH
    tile_reduction = small and batch <= _TILE_REDUCTION_MAX_BATCH
    col_split = (2 if batch <= 4 else 1) if tile_reduction else 1
    if not balanced:
        unroll = 1
    elif small:
        unroll = min(8 if batch <= 2 else 4 if batch <= 4 else 2, 4 * col_split)
    else:
        unroll = 4 if batch <= 2 else 1
    opts: Dict[str, object] = {
        "use_pdl": True,
        "prefetch_next_work": True,
        "skip_correction_threshold": 0.0,
        "reduction_unroll": unroll,
        "kv_evict_first": "",
    }
    if balanced:
        kernel = _kernel_module().BlackwellMultiHeadLatentAttentionForwardFP8
        max_requests = (
            32
            if small
            else min(kernel.BALANCED_MAX_REQUESTS, max(64, 1 << (batch - 1).bit_length()))
        )
        deep_k = max_requests <= _DEEP_K_MAX_REQUESTS
        opts.update(
            balanced_max_requests=max_requests,
            reduction_col_split=col_split,
            aligned_segment_cost=-1,
            bf16_partials=False,
            k_stages=3 if deep_k else 2,
            p_stages=2 if deep_k else 3,
            tile_reduction=tile_reduction,
        )
    unknown = set(overrides) - set(opts)
    if unknown:
        raise ValueError(f"MLA ll4 decode: unknown kernel options {sorted(unknown)}")
    opts.update(overrides)
    return tuple(opts.items())


@dataclass(frozen=True)
class WorkspaceLayout:
    lse_offset: int
    lse_bytes: int
    split_offset: int
    split_bytes: int
    total_bytes: int


def workspace_layout(num_heads: int, seq_len_q: int, max_batch: int) -> WorkspaceLayout:
    """LSE ``[max_batch, S, H]`` fp32, then the split-KV region: the larger of the uniform split's partials at
    ``_MAX_SPLIT_KV`` splits for ``max_batch`` requests and the balanced split's partial slots. Fixed per
    (H, S, max_batch) so one buffer serves every batch and both modes (CUDA graphs bake its address)."""
    kernel = _kernel_module().BlackwellMultiHeadLatentAttentionForwardFP8
    rows, chunks = fold_layout(num_heads, seq_len_q)
    lse_bytes = max_batch * seq_len_q * num_heads * 4
    split_offset = _pad(lse_bytes)
    uniform = max_batch * _MAX_SPLIT_KV * rows * chunks * (KV_LORA_RANK + 1) * _ACC_BYTES
    import cutlass

    balanced = kernel.get_balanced_workspace_size(
        num_heads, seq_len_q, KV_LORA_RANK, max_active_clusters(), cutlass.Float32, _MMA_TILE_M
    )
    split_bytes = max(uniform, balanced)
    return WorkspaceLayout(
        0, lse_bytes, split_offset, split_bytes, split_offset + _pad(split_bytes)
    )


def _pad(n: int) -> int:
    return (n + _WORKSPACE_ALIGN - 1) // _WORKSPACE_ALIGN * _WORKSPACE_ALIGN


def _cute_tensor(t: torch.Tensor, leading_dim: int, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=leading_dim)


class MlaLl4Decode:
    """The ll4 FP8 MLA decode kernel for one (heads, q_len, page) shape.

    Args:
        balanced: balanced split (``split_kv`` ignored) or the uniform split.
        split_kv: uniform split; None picks ll4's default for the batch. 1 runs without split-KV partials.
        max_batch: requests the workspace is sized for.
        options: kernel option overrides (see ``kernel_options``).
    """

    _compiled: dict = {}

    def __init__(
        self,
        num_heads: int,
        seq_len_q: int,
        page_size: int = 64,
        max_batch: int = 8,
        balanced: bool = True,
        split_kv: Optional[int] = None,
        emit_softmax_stats: bool = False,
        options: Optional[dict] = None,
    ):
        if num_heads > _MMA_TILE_M:
            raise ValueError(f"MLA ll4 decode: at most {_MMA_TILE_M} heads, got {num_heads}")
        if split_kv is not None and not 1 <= split_kv <= _MAX_SPLIT_KV:
            raise ValueError(
                f"MLA ll4 decode: split_kv must be in [1, {_MAX_SPLIT_KV}], got {split_kv}"
            )
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        self.page_size = page_size
        self.max_batch = max_batch
        self.balanced = balanced
        self.split_kv = split_kv
        self.emit_softmax_stats = emit_softmax_stats
        self.options = dict(options or {})
        self.layout = workspace_layout(num_heads, seq_len_q, max_batch)

    def workspace_bytes(self) -> int:
        return self.layout.total_bytes

    def tactic(self, batch: int) -> Tuple[bool, int, bool]:
        """(balanced, split_kv, is_persistent) the kernel runs this batch with."""
        if self.balanced:
            if (
                batch
                > _kernel_module().BlackwellMultiHeadLatentAttentionForwardFP8.BALANCED_MAX_REQUESTS
            ):
                raise ValueError(
                    f"MLA ll4 decode: balanced split takes at most 1024 requests, got {batch}"
                )
            return True, 1, True
        split = (
            self.split_kv
            if self.split_kv is not None
            else default_uniform_split(batch, self.num_heads, self.seq_len_q)
        )
        return False, split, False

    def __call__(
        self,
        q: torch.Tensor,
        kv_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        softmax_scale: torch.Tensor,
        output_scale: torch.Tensor,
        kv_bounds: Optional[torch.Tensor] = None,
        softmax_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write ``out`` ``[B, S, H, 512]`` bf16 (and ``softmax_stats`` ``[B * S, H, 2]`` fp32 if enabled).

        Args:
            q: ``[B, S, H, 576]`` FP8 (e4m3) absorbed queries.
            kv_pages: ``[num_pages, page_size, >=576]`` FP8 page pool.
            page_table: ``[blocks, B]`` int32, a transposed view of a contiguous ``[B, blocks]`` table.
            cache_seqs: ``[B]`` int32 KV lengths including the ``S`` new tokens.
            workspace: contiguous, at least ``workspace_bytes()`` bytes.
            softmax_scale, output_scale: fp32 device tensors, element 0 used.
        """
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute

        batch, seq_len_q, heads, width = q.shape
        if (heads, seq_len_q) != (
            self.num_heads,
            self.seq_len_q,
        ) or width < KV_LORA_RANK + QK_ROPE_DIM:
            raise ValueError(
                f"MLA ll4 decode: runner is H={self.num_heads} S={self.seq_len_q}, q {tuple(q.shape)}"
            )
        if batch > self.max_batch:
            raise ValueError(f"MLA ll4 decode: batch {batch} exceeds max_batch {self.max_batch}")
        if (softmax_stats is not None) != self.emit_softmax_stats:
            raise ValueError("MLA ll4 decode: softmax_stats must be given iff emit_softmax_stats")
        ws = workspace.view(torch.uint8).reshape(-1)
        if not workspace.is_contiguous() or ws.numel() < self.layout.total_bytes:
            raise ValueError(
                f"MLA ll4 decode: need a contiguous workspace of {self.layout.total_bytes} bytes"
            )

        q_latent = q[..., :KV_LORA_RANK].permute(2, 3, 1, 0)
        q_rope = q[..., KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_DIM].permute(2, 3, 1, 0)
        c_latent = kv_pages[..., :KV_LORA_RANK].permute(1, 2, 0)
        c_rope = kv_pages[..., KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_DIM].permute(1, 2, 0)
        o = out.permute(2, 3, 1, 0)
        lay = self.layout
        lse = (
            ws[lay.lse_offset : lay.lse_offset + lay.lse_bytes]
            .view(torch.float32)
            .view(self.max_batch, seq_len_q, heads)[:batch]
            .permute(2, 1, 0)
        )
        split_ws = ws[lay.split_offset : lay.split_offset + lay.split_bytes]
        stats_k = None
        if softmax_stats is not None:
            stats_k = softmax_stats.view(batch, seq_len_q, heads, 2).permute(2, 1, 0, 3)

        balanced, split_kv, persistent = self.tactic(batch)
        use_ws = balanced or split_kv > 1
        opts = kernel_options(batch, balanced, **self.options)
        key = (heads, seq_len_q, self.page_size, self.emit_softmax_stats, kv_bounds is not None, balanced,
               use_ws, persistent, opts)  # fmt: skip
        compiled = self._compiled.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"MLA ll4 decode: kernel variant {key} not compiled before CUDA graph capture"
                )
            kw = dict(opts)
            threshold = kw.pop("skip_correction_threshold")
            if balanced:
                kw["balanced_split"] = True
            kernel = _kernel_module().BlackwellMultiHeadLatentAttentionForwardFP8(
                cutlass.Float32,
                cutlass.Float32,
                _TILERS[0],
                _TILERS[1],
                max_active_clusters(),
                self.page_size,
                threshold,
                persistent,
                True,  # is_var_seq
                False,  # is_var_split_kv
                num_heads=heads,
                seq_len_q=seq_len_q,
                fold_sq=heads < _MMA_TILE_M and seq_len_q > 1,
                emit_softmax_stats=self.emit_softmax_stats,
                **kw,
            )
            from cutlass.cute.runtime import from_dlpack

            o_ct = (
                from_dlpack(o, assumed_align=16)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, stride_order=(3, 2, 0, 1), divisibility=8)
            )
            args = [
                _cute_tensor(q_latent, 1),
                _cute_tensor(q_rope, 1),
                _cute_tensor(c_latent, 1),
                _cute_tensor(c_rope, 1),
                _cute_tensor(page_table, 0),
                o_ct,
                _cute_tensor(lse, 0),
            ]
            target = kernel
            if self.emit_softmax_stats:
                target = kernel.run_with_softmax_stats
                args.append(_cute_tensor(stats_k, 3))
            args += [
                from_dlpack(split_ws, assumed_align=32).mark_layout_dynamic() if use_ws else None,
                split_kv,
                from_dlpack(cache_seqs, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(kv_bounds, assumed_align=4).mark_layout_dynamic()
                if kv_bounds is not None
                else None,
                None,  # block_split_kvs
                _cute_tensor(softmax_scale, 0, 4),
                _cute_tensor(output_scale, 0, 4),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            ]
            compiled = cute.compile(target, *args, options="--opt-level 2")
            self._compiled[key] = compiled

        # Tensors whose leading dimension a bare runtime re-adapt cannot deduce (all extents 1) go in pre-marked.
        pt = (
            _cute_tensor(page_table, 0)
            if page_table.shape[0] == 1 and page_table.shape[1] == 1
            else page_table
        )
        rt = [q_latent, q_rope, c_latent, c_rope, pt, o, lse]
        if self.emit_softmax_stats:
            rt.append(stats_k)
        rt += [
            split_ws if use_ws else None,
            split_kv,
            cache_seqs,
            kv_bounds,
            None,
            _cute_tensor(softmax_scale, 0, 4),
            _cute_tensor(output_scale, 0, 4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        ]
        compiled(*rt)
        return out
