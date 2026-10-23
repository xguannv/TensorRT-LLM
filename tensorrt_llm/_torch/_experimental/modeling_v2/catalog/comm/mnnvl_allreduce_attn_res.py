# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot or two-shot MNNVL all-reduce with Kimi K3's residual update (attention-residual selection + RMSNorm)
as its epilogue, over a caller-owned :class:`MnnvlWorkspace`."""

import functools
from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .mnnvl_workspace import MnnvlWorkspace


@torch.library.register_fake("trtllm::mnnvl_allreduce_attn_res")
def _(input, prefix_sum, block_residual, res_weight, rms_weight, output_rms_weight, rms_eps, output_rms_eps,
      comm_buffer, buffer_flags, two_shot=False):  # fmt: skip
    return [torch.empty_like(input), torch.empty_like(input)]


@functools.cache
def has_mxfp8_out() -> bool:
    """Whether the loaded library has ``trtllm::mnnvl_allreduce_attn_res_mx`` (``mxfp8_out``); a library built before
    it has not, and then ``mxfp8_out`` must stay None."""
    return hasattr(torch.ops.trtllm, "mnnvl_allreduce_attn_res_mx")


if has_mxfp8_out():

    @torch.library.register_fake("trtllm::mnnvl_allreduce_attn_res_mx")
    def _(input, prefix_sum, block_residual, res_weight, rms_weight, output_rms_weight, rms_eps, output_rms_eps,
          comm_buffer, buffer_flags, mx_quant, mx_scale, two_shot=False):  # fmt: skip
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
    mxfp8_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed, updated)``: ``updated = prefix_sum + allreduce(input)`` over ``workspace``'s TP group (the
    sum alone without ``prefix_sum``), ``normed`` = RMSNorm(attn_res(block_residual..., updated)). Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order. ``two_shot``
    sends each token's rows to one owner rank, which runs the epilogue and multicasts the results (less traffic for
    more tokens); the results are the one-shot exchange's, but for -0.0 returned as +0.0. ``mxfp8_out``: ``(xq, xs)``,
    float8_e4m3fn [M, H] and uint8 [M, H / 32] (contiguous, 16-byte aligned): every rank also stores there the MXFP8
    form of ``normed`` that ``k3_fp8w_lowm_gemm`` quantizes it to inside its kernel (``k3_fp8w_lowm_mxfp8``'s input);
    ``normed`` and ``updated`` are the same bits with it and without. Needs ``has_mxfp8_out()``."""
    if mxfp8_out is not None:
        normed, updated = torch.ops.trtllm.mnnvl_allreduce_attn_res_mx(
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
            mxfp8_out[0],
            mxfp8_out[1],
            two_shot,
        )
        return normed, updated
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
