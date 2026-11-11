# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 routed NVFP4 SiTU experts of at most 32 tokens in one persistent kernel (FC1 + SiTU +
NVFP4 requantization + FC2 + weighted combine)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_moe import k3_small_batch_moe_op as _op
from .k3_route_tail import sort_tables_reference

_SF_VEC = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _fp4_to_f64(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    p = packed.contiguous().view(torch.uint8).long()
    return torch.stack([lut[p & 0xF], lut[p >> 4]], dim=-1).flatten(-2)


def _e4m3_to_f64(sf: torch.Tensor) -> torch.Tensor:
    return sf.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)


def _unswizzle_128x4(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Row-major ``[rows, cols]`` scale bytes from the 128x4-blocked layout."""
    padded_cols = -(-cols // 4) * 4
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    index = (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    return sf.reshape(-1)[index]


def _dequant_expert(w: torch.Tensor, sf: torch.Tensor, e: int) -> torch.Tensor:
    rows, cols = w.shape[1], w.shape[2] * 2
    lin = _unswizzle_128x4(sf[e].contiguous().view(torch.uint8), rows, cols // _SF_VEC)
    return _fp4_to_f64(w[e]) * _e4m3_to_f64(lin).repeat_interleave(_SF_VEC, dim=1)


def _nvfp4_roundtrip(h: torch.Tensor, global_sf: float) -> torch.Tensor:
    """``h`` quantized as the FC1 epilogue quantizes it (16-element blocks, E4M3 scales of
    ``amax / 6 * global_sf``), decoded in the scaled domain the FC2 alphas expect."""
    grid = torch.tensor(_E2M1[:8], dtype=torch.float64, device=h.device)
    blocks = h.view(h.size(0), -1, _SF_VEC) * global_sf
    amax = blocks.abs().amax(-1, keepdim=True)
    sf = (amax / 6.0).float().to(torch.float8_e4m3fn).double()
    v = torch.where(sf > 0, blocks / sf, torch.zeros_like(blocks)).clamp(-6.0, 6.0)
    q = grid[(v.abs()[..., None] - grid).abs().argmin(-1)] * v.sign()
    return (q * sf).view_as(h)


class K3SmallBatchMoe(OpWrapper):
    """Writes into every element of ``output [tokens, hidden]`` (BF16) the sum of this rank's
    experts' routing-weighted down-projections; returns None. ``counters`` is zero again after
    the launch.

    ``input`` / ``input_scale``: NVFP4 tokens and their linear E4M3 scales; ``fc1_*`` / ``fc2_*``:
    the local experts' NVFP4 weights ``[L, rows, cols / 2]``, 128x4-blocked E4M3 scales and FP32
    per-expert alphas (FC1 rows in ``[gate16, up16]`` groups); the ``moe_sort`` tables of tile 128;
    ``global_sf``: the FC2 input scale; ``counters``: ``new_small_batch_moe_counters``.
    ``token_indexed``: the token-indexed FC1 output layout (MoE-TP slices). ``published_routing``
    (token-indexed): ``expanded_idx_to_permuted_idx`` holds the top-k ids a routing kernel
    published into the routing word of ``counters``, the other four tables are None.
    ``push_world > 0``: the rows are also pushed into slot ``(token, push_rank)`` of the one-shot
    MNNVL all-reduce whose multicast base is ``push_mc`` and state ``push_flags``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: K3 routed experts (896 experts, top 16, latent 3584) per deployment, at decode rows:
    #: ``local`` experts from ``offset``, ``inter`` per rank, ``layout`` ``permuted`` or
    #: ``token`` (token-indexed), ``published``: routing published by the routing kernel.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep16 EP16: 56 experts with the full intermediate, one row",
            spec={"m": 1, "local": 56, "offset": 392, "inter": 3072, "layout": "permuted"},
        ),
        Cell(
            why="helix8 EP8: 112 experts, one request with seven draft tokens",
            spec={"m": 8, "local": 112, "offset": 336, "inter": 3072, "layout": "permuted"},
        ),
        Cell(
            why="helix16 MoE TP4 x EP4: 224 experts, a quarter of the intermediate, token-indexed, "
            "four requests",
            spec={"m": 32, "local": 224, "offset": 224, "inter": 768, "layout": "token"},
        ),
        Cell(
            why="helix16 ULL: the same window on the routing the front kernel publishes",
            spec={
                "m": 8,
                "local": 224,
                "offset": 672,
                "inter": 768,
                "layout": "token",
                "published": True,
            },
        ),
    )

    note = """
    The default combine adds the weighted rows with BF16 vector reductions, whose order (so the
    output's bits) can change from run to run; ``deterministic_combine`` sums the slices in a fixed
    order (bitwise independent of the CTA count). ``num_ctas`` > 0 overrides the persistent grid,
    which is otherwise one CTA per SM less ``sm_reserve`` and ``cute_dsl_utils.sm_reserve()``.
    Compiled kernels are cached per process by (token tile, shapes, top-k, SiTU betas, layout,
    CTA count, combine, routing, push world, PDL); the first call of each key compiles (outside
    CUDA graph capture). ``die_major`` raises (not supported).
    """

    def __call__(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_scale: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_scale: torch.Tensor,
        fc2_alpha: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_group_idx: Optional[torch.Tensor],
        tile_idx_to_mn_limit: Optional[torch.Tensor],
        permuted_idx_to_expanded_idx: Optional[torch.Tensor],
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_non_exiting_tiles: Optional[torch.Tensor],
        global_sf: torch.Tensor,
        token_final_scales: torch.Tensor,
        counters: torch.Tensor,
        top_k: int,
        situ_beta: float,
        situ_linear_beta: float,
        token_indexed: bool,
        num_ctas: int = 0,
        deterministic_combine: bool = False,
        die_major: bool = False,
        published_routing: bool = False,
        expert_offset: int = 0,
        push_mc: int = 0,
        push_flags: Optional[torch.Tensor] = None,
        push_rank: int = 0,
        push_world: int = 0,
        sm_reserve: int = 0,
    ) -> None:
        _op.k3r_nvfp4_small_batch_moe(
            input,
            input_scale,
            fc1_weight,
            fc1_weight_scale,
            fc1_alpha,
            fc2_weight,
            fc2_weight_scale,
            fc2_alpha,
            output,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            expanded_idx_to_permuted_idx,
            num_non_exiting_tiles,
            global_sf,
            token_final_scales,
            counters,
            top_k,
            situ_beta,
            situ_linear_beta,
            token_indexed,
            num_ctas,
            deterministic_combine,
            die_major,
            published_routing,
            expert_offset,
            push_mc,
            push_flags,
            push_rank,
            push_world,
            sm_reserve,
        )

    def reference(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_scale: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_scale: torch.Tensor,
        fc2_alpha: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_group_idx: Optional[torch.Tensor],
        tile_idx_to_mn_limit: Optional[torch.Tensor],
        permuted_idx_to_expanded_idx: Optional[torch.Tensor],
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_non_exiting_tiles: Optional[torch.Tensor],
        global_sf: torch.Tensor,
        token_final_scales: torch.Tensor,
        counters: torch.Tensor,
        top_k: int,
        situ_beta: float,
        situ_linear_beta: float,
        token_indexed: bool,
        num_ctas: int = 0,
        deterministic_combine: bool = False,
        die_major: bool = False,
        published_routing: bool = False,
        expert_offset: int = 0,
        push_mc: int = 0,
        push_flags: Optional[torch.Tensor] = None,
        push_rank: int = 0,
        push_world: int = 0,
        sm_reserve: int = 0,
    ) -> torch.Tensor:
        """The expected ``output`` in FP64 (round to BF16 to compare bits): per served slot, FC1
        of the decoded operands, SiTU, the FC1 epilogue's NVFP4 requantization, FC2, the routing
        weight, summed per token. The served slots are the tables' (with ``published_routing``,
        the top-k ids of the experts ``[expert_offset, expert_offset + L)``). The layout, combine,
        CTA count and push arguments do not change the value."""
        num_tokens = token_final_scales.shape[0]
        num_local = fc1_weight.shape[0]
        hidden = fc2_weight.shape[1]
        inter = fc2_weight.shape[2] * 2
        if published_routing:
            ids = expanded_idx_to_permuted_idx.reshape(num_tokens, top_k)
            t2g, t2lim, _, p2e, _, nnet = sort_tables_reference(ids, expert_offset, num_local, 128)
        else:
            t2g, t2lim, p2e, nnet = (
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
            )
        num_tiles = int(nnet[0])
        limits = t2lim[:num_tiles].tolist()
        rows = [(t, r) for t in range(num_tiles) for r in range(t * 128, limits[t])]
        out = torch.zeros(num_tokens, hidden, dtype=torch.float64, device=input.device)
        if not rows:
            return out
        dev = input.device
        row_idx = torch.tensor([r for _, r in rows], device=dev)
        tiles = torch.tensor([t for t, _ in rows], device=dev)
        expanded = p2e[row_idx].long()
        tokens, slots = expanded // top_k, expanded % top_k
        experts = t2g[tiles].long()
        x_sf = _e4m3_to_f64(input_scale).reshape(input.shape[0], -1)
        x = _fp4_to_f64(input) * x_sf.repeat_interleave(_SF_VEC, dim=1)
        h = torch.empty(len(rows), inter, dtype=torch.float64, device=dev)
        for e in experts.unique().tolist():
            sel = (experts == e).nonzero().flatten()
            w1 = _dequant_expert(fc1_weight, fc1_weight_scale, e)
            acc = (x[tokens[sel]] @ w1.t()) * float(fc1_alpha[e])
            acc = acc.view(len(sel), -1, 2, 16)  # [gate16, up16] row groups
            gate, up = acc[:, :, 0], acc[:, :, 1]
            act = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
            act = act * situ_linear_beta * torch.tanh(up / situ_linear_beta)
            h[sel] = act.reshape(len(sel), -1)
        h = _nvfp4_roundtrip(h, float(global_sf.reshape(-1)[0]))
        partial = torch.empty(len(rows), hidden, dtype=torch.float64, device=dev)
        for e in experts.unique().tolist():
            sel = (experts == e).nonzero().flatten()
            w2 = _dequant_expert(fc2_weight, fc2_weight_scale, e)
            partial[sel] = h[sel] @ w2.t() * float(fc2_alpha[e])
        weights = token_final_scales[tokens, slots].double()[:, None]
        out.index_add_(0, tokens, partial * weights)
        return out


k3_small_batch_moe = K3SmallBatchMoe()
