# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's pre-attention sandwich: the row-parallel MoE tail (the latent up projection of the normed latent slice and
the shared experts' down projection), its TP all-reduce and the next layer's residual update (attention-residual
selection + RMSNorm) in one kernel, over a caller-owned :class:`K3SandwichWorkspace`; with a
:class:`K3SandwichLatentExchange`, the latent all-reduce before it as well."""

from typing import Optional, Tuple

import torch

# Importing the op module registers trtllm::k3_sandwich_*; the state types are the ones the ops' buffers come from.
from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich.op import (
    K3SandwichLatentExchange,
    K3SandwichWorkspace,
)

__all__ = ["K3SandwichLatentExchange", "K3SandwichWorkspace", "k3_sandwich_tail"]


def k3_sandwich_tail(
    latent: torch.Tensor,
    act: torch.Tensor,
    tail_weight: torch.Tensor,
    lo: int,
    lat_eps: float,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: K3SandwichWorkspace,
    tap: Optional[torch.Tensor] = None,
    tap_updated: bool = False,
    updated_out: Optional[torch.Tensor] = None,
    latent_exchange: Optional[K3SandwichLatentExchange] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed, updated)``: ``updated = prefix + allreduce([rmsnorm(latent)[:, lo:lo+224] | act] @
    tail_weight^T)`` over ``workspace``'s TP group (the sum alone without ``prefix``), ``normed`` =
    RMSNorm(attn_res(block_residual..., updated)). ``tap``: also store the pre-norm attention-residual mixture there
    (``updated`` with ``tap_updated``). ``updated_out``: store ``updated`` there; it is then the returned ``updated``.
    ``latent_exchange``: ``latent`` is this rank's unreduced latent partial ``[M, 3584]`` (M at most the exchange's
    ``max_rows``), which the kernel all-reduces over the exchange itself, bit for bit the one-shot MNNVL all-reduce.
    Advances ``workspace`` (and ``latent_exchange``) by one call: every rank of the group makes the same calls on it in
    the same order."""
    normed, updated = torch.ops.trtllm.k3_sandwich_tail(
        latent,
        act,
        tail_weight,
        lo,
        lat_eps,
        prefix,
        block_residual,
        res_weight,
        rms_weight,
        output_rms_weight,
        rms_eps,
        output_rms_eps,
        workspace.uc,
        workspace.mc,
        workspace.flags,
        workspace.rank,
        tap=tap,
        tap_updated=tap_updated,
        updated_out=updated_out,
        lat_uc=None if latent_exchange is None else latent_exchange.uc,
        lat_flags=None if latent_exchange is None else latent_exchange.flags,
        lat_mc=None if latent_exchange is None else latent_exchange.mc,
    )
    return normed, (updated if updated_out is None else updated_out)
