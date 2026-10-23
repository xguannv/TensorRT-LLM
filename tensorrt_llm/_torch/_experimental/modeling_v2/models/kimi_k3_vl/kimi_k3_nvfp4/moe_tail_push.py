# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A wide decode step's MoE tail pushing its partial into the next all-reduce (``TRTLLM_K3LL_MOE_TAIL_PUSH``).

Without the switch a MoE layer of a wide decode step (`decode_moe_lowm.py`) ends with the row-parallel tail GEMV
``[RMSNorm(latent) slice | padding | shared activation] @ tail_weight.T`` on the decode GEMV site ``moe_tail``
(``k3_ctm_gemv_wide`` at 9-32 rows, ``F.linear`` above), which stores this rank's bf16 partial; the next layer's
pre-attention step (the final norm after the last layer) then runs ``K3DecodeComm.allreduce_attn_res(two_shot=True)``
on it: ``comm/mnnvl_allreduce_attn_res``'s two-shot kernel, which scatters the partial into the TP group's
``MnnvlWorkspace``, sums it at each row's owner and runs the residual update.

With the switch the layer hands on the tail's input instead (a `WideTail`), and the consumer runs, where
``K3DecodeComm.takes_tail_push`` holds, the step as three kernels on the workspace, as ``TRTLLM_K3LL_OPROJ_PUSH`` runs
the post-attention step (``oproj_push.py``):

* ``k3_ctm_gemv_wide``'s all-reduce push epilogue (``k3_ctm_gemv.op.run_k3_ctm_gemv_wide_ar_push``): the same GEMV
  (the same split-K config, sums and bf16 rounding) storing each token row into slot ``(t // W, rank)`` of the scatter
  stage of its owner ``t % W`` in the workspace's next Lamport buffer instead of into a tensor;
* ``K3LamportFinishKernel`` in place, in the C++ one-shot order (``K3OprojPush.finish_attn_res``);
* ``K3AttnResFastKernel`` with the Lamport addend and ``mnnvl_order``: the residual add, attn_res and RMSNorm.

Where it does not hold, the consumer runs the tail GEMV (`WideTail.partial`) and its own path, unchanged: the same
kernels in the same stream order as without the switch.

Which steps. A wide decode step (``step.wide``) whose tail the plain path runs on ``k3_ctm_gemv_wide`` (9 to the site's
``wide_rows``, 32: above it the plain path is ``F.linear`` and its cuBLAS sums, which the push cannot reproduce) and
whose consumer is the two-shot ``mnnvl_allreduce_attn_res`` (``TRTLLM_K3LL_WIDE_AR_ATTN_RES``, ``takes_wide``); the
attn_res kernel takes the snapshots and weights and a Lamport buffer holds the call. Steps of at most 8 tokens keep
the ``PendingTail`` / ``comm/k3_sandwich_tail`` path. The last MoE layer's consumer is the final norm (its
``output_attn_res_*`` modules and the model norm). A layer DSpark taps defers as before; its consumer taps the updated
prefix sum, which the attn_res kernel stores as a tensor (the same bits as the C++ op's). The decision reads the
switch, the step, shapes and load-time state only (the push kernel compiled at load, the plain kernel warmed), so
every rank of the group decides alike and makes the same calls on the workspace in the same order.

Numerics. The push epilogue stages exactly the bf16 values the plain kernel stages for its TMA store and sends ``-0``
as ``+0``, as the C++ two-shot kernel's scatter sanitizes its input; the finish kernel sums the W slots in that
kernel's order (``kRankChunk`` = 8 above 8 ranks, the one-shot order) with round-to-nearest; the attn_res kernel
with ``mnnvl_order`` runs that op's epilogue arithmetic. ``updated`` and ``normed`` are bit for bit the switch-off
path's.

Ordering. The consumer's three kernels run back to back on the layer's stream right where the switch-off path runs
the tail GEMV and the two-shot call (nothing runs between the MoE's last kernel and the consumer's pre-attention
step; the shared experts' side stream has joined before the tail input exists), so the attn_res kernel, which reads
the result from the workspace, lies on every path to the next call on the workspace (the next layer's
``mnnvl_allgather_split``, all-reduces or pushes), whose producer passes its ``griddepcontrol.wait`` only once the
attn_res kernel has completed. The push GEMV reads the rotation state after its own wait (the previous call, e.g. the
latent ``mnnvl_allreduce_rmsnorm_slice`` right before it, has completed) and releases the finish kernel only then.
With ``TRTLLM_K3LL_OPROJ_PUSH`` the layers' post-attention pushes are calls on the same workspace in the same stream
order on every rank.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from . import flags as _flags

#: The decode GEMV site of the tail (``decode_gemv.SITES``).
SITE = "moe_tail"


class WideTail(NamedTuple):
    """A wide decode step's MoE row-parallel tail handed on before its GEMV (``TRTLLM_K3LL_MOE_TAIL_PUSH``): the
    tail's input ``[T, K]`` (``[RMSNorm(latent) slice | padding | shared activation]``), the tail weight ``[H, K]``
    and the decode GEMVs' state that runs it."""

    x: torch.Tensor
    weight: torch.Tensor
    gemvs: object

    def partial(self) -> torch.Tensor:
        """This rank's unreduced share ``x @ weight.T``, ``[T, H]`` bf16, as the switch-off path computes it (the
        site's decode GEMV, else ``F.linear``)."""
        y = self.gemvs.project(SITE, self.x, self.weight)
        return F.linear(self.x, self.weight) if y is None else y


def enabled(step) -> bool:
    """Whether ``TRTLLM_K3LL_MOE_TAIL_PUSH`` (read at forward time) asks for ``step``: a wide decode step."""
    return bool(_flags.TRTLLM_K3LL_MOE_TAIL_PUSH) and step is not None and step.wide


__all__ = ["SITE", "WideTail", "enabled"]
