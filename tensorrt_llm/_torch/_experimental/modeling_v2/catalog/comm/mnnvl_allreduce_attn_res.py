# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot or two-shot MNNVL all-reduce with Kimi K3's residual update (attention-residual selection + RMSNorm)
as its epilogue, over a caller-owned :class:`MnnvlWorkspace`."""

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .mnnvl_workspace import MnnvlWorkspace


@torch.library.register_fake("trtllm::mnnvl_allreduce_attn_res")
def _(input, prefix_sum, block_residual, res_weight, rms_weight, output_rms_weight, rms_eps, output_rms_eps,
      comm_buffer, buffer_flags, two_shot=False):  # fmt: skip
    return [torch.empty_like(input), torch.empty_like(input)]


def mnnvl_allreduce_attn_res(
    input: torch.Tensor,
    prefix_sum: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: MnnvlWorkspace,
    two_shot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed, updated)``: ``updated = prefix_sum + allreduce(input)`` over ``workspace``'s TP group (the
    sum alone without ``prefix_sum``), ``normed`` = RMSNorm(attn_res(block_residual..., updated)). Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order. ``two_shot``
    sends each token's rows to one owner rank, which runs the epilogue and multicasts the results (less traffic for
    more tokens); the results are the one-shot exchange's, but for -0.0 returned as +0.0."""
    normed, updated = torch.ops.trtllm.mnnvl_allreduce_attn_res(
        input,
        prefix_sum,
        block_residual,
        res_weight,
        rms_weight,
        output_rms_weight,
        rms_eps,
        output_rms_eps,
        workspace.comm_buffer(input.dtype),
        workspace.buffer_flags,
        two_shot,
    )
    return normed, updated
