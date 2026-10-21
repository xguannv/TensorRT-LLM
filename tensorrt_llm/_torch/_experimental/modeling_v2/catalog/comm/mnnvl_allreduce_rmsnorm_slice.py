# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's column slice of ``RMSNorm(allreduce(x))`` over a caller-owned :class:`MnnvlWorkspace`, laid out as a
row-parallel GEMM's input (Kimi K3's wide MoE tail: the normalized latent slice, padding, the shared activation)."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .mnnvl_workspace import MnnvlWorkspace

__all__ = ["MnnvlWorkspace", "fits", "mnnvl_allreduce_rmsnorm_slice"]

ELTS_PER_PACKET = 8
MAX_SLICE_COLUMNS = 256


@torch.library.register_fake("trtllm::mnnvl_allreduce_rmsnorm_slice")
def _(input, tail, pad_cols, eps, world_size, comm_buffer, buffer_flags):
    num_tokens, hidden = input.shape
    tail_cols = 0 if tail is None else tail.shape[1]
    return input.new_empty((num_tokens, hidden // world_size + pad_cols + tail_cols))


def fits(
    num_tokens: int, hidden: int, pad_cols: int, tail_cols: int, workspace: MnnvlWorkspace
) -> bool:
    """Whether a call of these shapes is supported on ``workspace``: a world size of 4, 8 or 16, a slice of ``hidden /
    world`` columns that is a multiple of 8 and at most 256, padding and tail widths that are multiples of 8, and both
    stages (``num_tokens * hidden`` bf16, ``num_tokens * world`` fp32) within half a Lamport buffer."""
    world = workspace.world_size
    if world not in (4, 8, 16) or hidden % world or num_tokens <= 0:
        return False
    width = hidden // world
    if width % ELTS_PER_PACKET or width > MAX_SLICE_COLUMNS:
        return False
    if pad_cols % ELTS_PER_PACKET or tail_cols % ELTS_PER_PACKET or pad_cols < 0 or tail_cols < 0:
        return False
    stage = workspace.buffer_bytes // 2
    return num_tokens * hidden * 2 <= stage and num_tokens * world * 4 <= stage


def mnnvl_allreduce_rmsnorm_slice(
    input: torch.Tensor,
    tail: Optional[torch.Tensor],
    pad_cols: int,
    eps: float,
    workspace: MnnvlWorkspace,
) -> torch.Tensor:
    """``[RMSNorm(r)[:, s:s + w] | zeros(pad_cols) | tail]`` with ``r = bf16(sum over ranks of input)``, ``w = hidden /
    world``, ``s = rank * w`` and an RMSNorm without a weight: ``r * rsqrt(mean(r * r) + eps)``. ``input``: bf16
    ``[num_tokens, hidden]``, this rank's addend; ``tail``: bf16 ``[num_tokens, tail_cols]`` or None.

    ``r`` has the bits of the MNNVL two-shot all-reduce of ``input``; the mean of squares is summed per rank slice then
    over the ranks in rank order (the same bits on every rank), so the slice may differ from a whole-row RMSNorm of
    ``r`` in the last bit. Takes one turn of the workspace's Lamport rotation, as an all-reduce on it does: every rank
    of the group makes the same MNNVL calls on it in the same order. ``fits`` holds or the op raises."""
    num_tokens, hidden = input.shape
    tail_cols = 0 if tail is None else tail.shape[1]
    if not fits(num_tokens, hidden, pad_cols, tail_cols, workspace):
        raise ValueError(
            f"mnnvl_allreduce_rmsnorm_slice: unsupported call (tokens {num_tokens}, hidden {hidden}, pad {pad_cols}, "
            f"tail {tail_cols}, world {workspace.world_size}, buffer {workspace.buffer_bytes} bytes)"
        )
    return torch.ops.trtllm.mnnvl_allreduce_rmsnorm_slice(
        input.contiguous(),
        None if tail is None else tail.contiguous(),
        pad_cols,
        float(eps),
        workspace.world_size,
        workspace.comm_buffer(torch.bfloat16),
        workspace.buffer_flags,
    )
