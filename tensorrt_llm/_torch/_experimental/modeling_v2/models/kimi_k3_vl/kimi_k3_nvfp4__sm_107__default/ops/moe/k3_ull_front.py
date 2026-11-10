# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MoE front of small decode steps over the TP group: router logits, routing tail and the
latent down-projection with its NVFP4 quantization, sharded over the ranks in two kernels."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ...kernels import k3_ull_front as _front
from ..gemm import _fp8_reference as gemm_ref
from .k3_route_tail import k3_route_tail
from .k3_router_logits import k3_router_logits


class K3UllFront(OpWrapper):
    """Returns ``(routing, codes, scales)`` for this rank's BF16 ``x [M, 7168]`` (``M <= 32``, the
    same rows on every rank of the group):

    * ``routing``: the 8 outputs of :class:`K3RouteTail` on the router logits of every expert, with
      the ``moe_sort`` tables of ``plan.local_offset`` / ``plan.local_num`` (with ``publish``, the
      address of a zero int32 word, the route kernel adds 1 to it per routed token instead and
      leaves the tables unwritten);
    * ``codes`` uint8 ``[M, 1792]`` / ``scales`` uint8 ``[M, 224]``: :class:`K3LatentDownFp4` of
      ``x`` (linear scales), each rank computing its ``plan`` tiles of latent_down and multicasting
      them through ``ws``.

    ``plan``: ``k3_ull_front.make_plan`` of the layer; ``ws``: the TP group's
    ``k3_ull_front.ull_front_workspace``. ``after_head`` runs between the two launches.
    """

    # No cells yet: the op needs an MNNVL multicast workspace over a TP group of ranks.
    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = ()

    note = """
    A collective over ``plan.world`` ranks; every rank calls it in the same order. Head kernel:
    the latent GEMM CTAs (the low-M split-K MXFP8 GEMM, one-shot push of this rank's tiles into
    slot ``(token, rank)`` of every rank's workspace half) and the router CTAs (``K3RouterLogits``'
    arithmetic on all 896 experts into this rank's own workspace). Route kernel, a PDL dependent of
    the head: the ``K3RouteTail`` cluster polling the logits and gather CTAs polling every rank's
    latent slots, then ``fp4_quantize``'s arithmetic. Every reader re-arms what it read;
    consecutive calls alternate between the workspace's two halves (``ws.calls`` is advanced on
    the host). The first call of each shape compiles, which must not happen inside CUDA graph
    capture (``k3_ull_front.is_compiled``).
    """

    def __call__(
        self,
        x: torch.Tensor,
        plan: "_front.UllFrontPlan",
        ws: "_front.UllFrontWorkspace",
        after_head: Optional[Callable[[], None]] = None,
        publish: int = 0,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        return _front.run_k3_ull_front(x, plan, ws, after_head=after_head, publish=publish)

    def reference(
        self,
        x: Sequence[torch.Tensor],
        plan: Sequence["_front.UllFrontPlan"],
        ws: "_front.UllFrontWorkspace",
        after_head: Optional[Callable[[], None]] = None,
        publish: int = 0,
    ) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]]:
        """A collective: ``x`` and ``plan`` are the group's in rank order; returns each rank's
        ``(routing, codes, scales)`` in rank order, composed from :class:`K3RouterLogits`,
        :class:`K3RouteTail` and the latent projection of the ranks' tiles together (activation
        MXFP8 1 x 32, then ``fp4_quantize`` with ``plan.global_sf``). ``ws``, ``after_head`` and
        ``publish`` are not read (with ``publish`` only the top-k is defined)."""
        latent_w = torch.cat([p.latent_w for p in plan])
        exps = torch.cat(
            [
                gemm_ref.packed_block_exponents(p.latent_ws, p.latent_w.shape[0], _front.HIDDEN)
                for p in plan
            ]
        )
        w = gemm_ref.dequant(latent_w, exps, 128)
        out = []
        for xr, p in zip(x, plan):
            logits = k3_router_logits.reference(xr, p.router_w)
            routing = k3_route_tail.reference(
                logits,
                p.bias,
                p.routed_scaling_factor,
                p.local_offset,
                p.local_num,
                _front.TILE,
            )
            codes, e = gemm_ref.mxfp8_quantize(xr, 32)
            y = gemm_ref.matmul_bf16(gemm_ref.dequant(codes, e, 32), w)
            q, sf = fp4_quantize.reference(y, p.global_sf, 16, False, False)
            out.append((routing, q, sf.view(xr.shape[0], -1)))
        return out


k3_ull_front = K3UllFront()
