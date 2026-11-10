# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 shared expert of at most 8 rows on a few persistent CTAs beside the routed experts,
pushing its row-parallel partial into the MoE tail's Lamport call."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels import k3_shared_persist as _persist


class K3SharedPersist(OpWrapper):
    """This rank's shared-expert partial ``situ(x Wg^T, x Wu^T) Wd^T`` (BF16 ``[m, 7168]``) of BF16
    ``x [m, 7168]``, stored into slot ``(token, rank)`` of the tail call's current buffer
    (``out_base``: multicast base, or a local one with ``mc_out=False``; ``out_flags`` its rotation
    state); returns None. ``frags``: ``k3_shared_persist.persist_fragments`` of the rank's gate/up
    and down weights for ``sms`` CTAs; ``ws``: the exchange buffer and counters
    (``k3_shared_persist.PersistWorkspace``)."""

    # No cells yet: the op writes into a Lamport workspace the next pre-attention mix finishes.
    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = ()

    note = """
    ``sms`` persistent CTAs stream their weights once through a shared-memory ring, compute the
    gate/up products (BF16 ``mma.sync``, FP32 accumulation, outputs rounded to BF16), the SiTU
    activation with ``ex2.approx`` tanh / sigmoid (rounded to BF16), exchange the activation
    through ``ws`` and multiply their down m-tiles; ``-0`` is stored as ``+0``. Not bitwise the
    GEMM + activation-op path. Launched without PDL; consecutive launches on a stream reuse ``ws``
    (its counters are zero between launches). Compiled per (world, m, sms, mc_out, betas, device).
    """

    def __call__(
        self,
        *,
        x: torch.Tensor,
        frags: torch.Tensor,
        world: int,
        rank: int,
        sms: int,
        ws: "_persist.PersistWorkspace",
        out_base: int,
        out_flags: torch.Tensor,
        beta: float,
        linear_beta: Optional[float],
        mc_out: bool = True,
    ) -> None:
        _persist.run_k3_shared_persist(
            x=x,
            frags=frags,
            world=world,
            rank=rank,
            sms=sms,
            ws=ws,
            out_base=out_base,
            out_flags=out_flags,
            beta=beta,
            linear_beta=linear_beta,
            mc_out=mc_out,
        )

    def reference(
        self,
        *,
        x: torch.Tensor,
        frags: Tuple[torch.Tensor, torch.Tensor],
        world: int,
        rank: int,
        sms: int,
        ws: "_persist.PersistWorkspace",
        out_base: int,
        out_flags: torch.Tensor,
        beta: float,
        linear_beta: Optional[float],
        mc_out: bool = True,
    ) -> torch.Tensor:
        """The BF16 ``[m, 7168]`` slot rows. ``frags`` is the ``(gate_up [2 I, 7168], down
        [7168, I])`` BF16 pair ``persist_fragments`` packs (the packing is a permutation). FP32
        GEMMs rounded to BF16 like the op's, SiTU in FP32 rounded to BF16. The workspace and
        output addresses are not read."""
        gate_up, down = frags
        inter = gate_up.shape[0] // 2
        xf = x.float()
        g = (xf @ gate_up[:inter].float().T).to(torch.bfloat16).float()
        u = (xf @ gate_up[inter:].float().T).to(torch.bfloat16).float()
        act = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        act = (act * u).to(torch.bfloat16)
        return (act.float() @ down.float().T).to(torch.bfloat16)


k3_shared_persist = K3SharedPersist()
