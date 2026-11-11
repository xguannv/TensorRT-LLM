# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MoE back half of at most 8 rows in one kernel: the routed latent all-reduce, its
RMSNorm and this rank's output columns of latent_up, multicast into the tail call's all-gather
rows."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_moe import k3_moe_tail1 as _tail1


class K3MoeTail1(OpWrapper):
    """For ``m`` rows on ``world`` ranks: polls the BF16 latent partials ``[m, 3584]`` every rank's
    routed experts pushed into slot ``(token, rank)`` of the latent Lamport workspace (``lat_uc``
    local base, ``lat_flags`` its rotation state), sums them, applies the RMSNorm of ``gamma`` /
    ``eps`` and writes rows ``[c0, c0 + 7168 / world)`` of ``latent_up`` (``weight`` E4M3
    ``[7168, 3584]``, ``wexp`` its UE8M0 block exponents ``[56, 28]``) into the all-gather rows of
    the tail call (``ag_base``: multicast base, or a local one with ``mc_out=False``; ``ag_flags``
    its rotation state). Returns None; the next pre-attention mix finishes the call."""

    # No cells yet: the op reads the partials from a Lamport workspace of every rank.
    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = ()

    note = """
    A collective of the latent workspace's ranks. Clusters of 16 CTAs; each lane loads its
    latent_up fragments before polling. The latent sum is the C++ one-shot all-reduce's (FP32 in
    rank order, chunks of eight ranks, rounded to BF16); the RMS uses FP32 squares of it; the
    projection's activation is ``bf16(x * gamma)`` and ``rsqrt(mean + eps)`` scales the FP32
    product sum before its rounding, so the output is not bitwise the column-parallel tail.
    ``-0`` is stored as ``+0``. The kernel re-arms the previous call's latent buffer and, from the
    CTA that arrives last, rotates the latent workspace after its ``griddepcontrol.wait``; the
    tail call's state is only read. PDL: the dependent is released at entry. Compiled per
    (world, m, mc_out, device).
    """

    def __call__(
        self,
        *,
        m: int,
        world: int,
        lat_uc: int,
        lat_flags: torch.Tensor,
        weight: torch.Tensor,
        wexp: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        ag_base: int,
        ag_flags: torch.Tensor,
        c0: int,
        mc_out: bool = True,
    ) -> None:
        _tail1.run_k3_moe_tail1(
            m=m,
            world=world,
            lat_uc=lat_uc,
            lat_flags=lat_flags,
            weight=weight,
            wexp=wexp,
            gamma=gamma,
            eps=eps,
            ag_base=ag_base,
            ag_flags=ag_flags,
            c0=c0,
            mc_out=mc_out,
        )

    def reference(
        self,
        *,
        m: int,
        world: int,
        lat_uc: Sequence[torch.Tensor],
        lat_flags: torch.Tensor,
        weight: torch.Tensor,
        wexp: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        ag_base: int,
        ag_flags: torch.Tensor,
        c0: int,
        mc_out: bool = True,
    ) -> torch.Tensor:
        """A collective: ``lat_uc`` is the group's input, the ranks' BF16 latent partials
        ``[m, 3584]`` in rank order (what their experts pushed). Returns the BF16 ``[m, 7168]``
        all-gather rows once every rank ran: ``latent_up(rmsnorm(bf16(sum)))`` with the
        normalization applied in FP64. The workspace and output addresses are not read."""
        total = lat_uc[0].float()
        for part in lat_uc[1:]:
            total = total + part.float()
        latent = total.to(torch.bfloat16).double()
        rms = torch.rsqrt(latent.pow(2).mean(-1, keepdim=True) + eps)
        scale = torch.pow(2.0, wexp.double() - 127.0)
        scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        w = weight.to(torch.float64) * scale
        y = ((latent * gamma.double()) @ w.T) * rms
        return y.to(torch.bfloat16)


k3_moe_tail1 = K3MoeTail1()
