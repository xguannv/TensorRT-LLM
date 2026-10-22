# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode layer's attention output projection pushing its partial into the all-reduce (``TRTLLM_K3LL_OPROJ_PUSH``).

Without the switch a decode layer's post-attention step is o_proj on ``k3_fp8w_lowm_gemm``, which stores this rank's
bf16 partial, then one ``comm/mnnvl_allreduce_attn_res`` call (``decode_comm.py``) that sends the partial into the TP
group's ``MnnvlWorkspace``, sums it and runs the residual update (residual add, attention-residual selection and
RMSNorm). With it, `K3OprojPush.run` runs the same step as three kernels on that workspace:

* ``k3_fp8w_lowm_gemm``'s all-reduce push epilogue (``run_k3_fp8w_lowm_gemm_ar_push``): instead of storing the partial,
  the GEMM stores each token row into slot ``(t // W, rank)`` of the scatter stage of the row's owner ``t % W`` in the
  workspace's next Lamport buffer;
* ``K3LamportFinishKernel`` without its gather (``k3_lamport_ar.run_lamport_finish_in_place``): the owner polls its
  ``W`` slots, sums them and multicasts the row into the broadcast stage of every rank, where it stays (a
  `LamportResult`);
* ``K3AttnResFastKernel`` with the Lamport addend (``k3_attn_res.run_k3_attn_res(lamport_addend=...)``): every rank
  polls every row from its broadcast stage, adds it to the prefix sum (to zeros where the layer has just snapshotted
  it) and runs the attention-residual selection and the RMSNorm, the snapshot reads and their statistics before its
  PDL wait.

Programmatic dependent launch. The GEMM's weight warp streams its weights before the GEMM's ``griddepcontrol.wait``
as the plain GEMM's does; the push variant releases its dependent from the activation warp right after that wait
(the plain GEMM: from the weight warp at its start), since the finish kernel reads the rotation state as soon as it
is resident. The finish kernel's CTAs release theirs at their start, so the attn_res kernel is resident and has read
the snapshots while the rows travel. The attn_res kernel releases the next kernel (the MoE front or head) right after
its own wait (``early_trigger``), as ``mnnvl_allreduce_attn_res`` does after its ``cudaGridDependencySynchronize``,
so that kernel streams its weights while the attn_res kernel finishes; it becomes resident about one finish kernel
later than behind ``mnnvl_allreduce_attn_res``.

The kernels are Dongxu Yang's (``user/dongxuy/k3_lowlat``, ``TRTLLM_K3_AR_PUSH`` / ``TRTLLM_K3_AR_PUSH_AMID`` with
``TRTLLM_K3_ATTN_RES_PDL``); `LamportPushTarget`, `LamportResult`, `push_target` and `finish_lamport_push` are their
``distributed/ops.py`` plumbing on this target's caller-owned workspace.

Numerics. The GEMM rounds as the plain GEMM does and the finish kernel sums the ranks in the C++ one-shot kernel's
order (chunks of eight ranks above eight), which both transports of ``mnnvl_allreduce_attn_res`` use: the all-reduce
result and the updated prefix sum ``bf16(prefix + sum)`` are bit for bit those of the path without the switch. The
attention-residual statistics are summed in another order than that op's epilogue, so the normalized output can differ
from it by a bf16 rounding step or two in a small fraction of the elements (``k3_attn_res``). Every rank reads the same
rows and runs the same kernel on them, so the outputs are the same bits on every rank.

State and ordering. The Lamport rotation (current / dirty buffer, bytes to clear, CTA arrivals) lives in the
workspace's device flags, which the C++ MNNVL kernels on the same workspace read and advance alike: the push reads it
after its ``griddepcontrol.wait``, the finish kernel advances it, the attn_res kernel reads it after its own wait. A
CUDA graph captures no host state of the rotation, so graphs of different token counts and the calls of other
collectives on the workspace interleave freely, as long as every rank makes the same calls in the same order. The
result stays in the workspace until the next call on it re-arms that buffer: that call starts clearing only once its
producer (a C++ kernel itself, or a push GEMM) has passed its ``griddepcontrol.wait``, so every kernel between the
attn_res kernel and it must wait for its predecessor before it triggers or completes, as the decode path's kernels
do. The device view of the workspace (`K3OprojPush.create`) is a host-to-device copy, made once, eagerly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)

from . import flags as _flags

if TYPE_CHECKING:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_lamport_ar import LamportWorkspace

#: The most tokens of a step the switch takes: a wide decode step's (``decode_moe.WIDE_MAX_TOKENS``), the low-M GEMM's.
MAX_TOKENS = 64
#: The hidden size: o_proj's output rows and the attn_res kernel's specialization (``k3_attn_res.HIDDEN_SIZE``).
HIDDEN = 7168
#: ``TRTLLM_K3LL_OPROJ_PUSH`` values: every pure decode step of at most `MAX_TOKENS` tokens, or the wide ones only.
EVERY_DECODE_STEP, WIDE_STEPS = 1, 2
#: Programmatic dependent launch of the finish and gather kernels unless TRTLLM_ENABLE_PDL=0 (the stack-wide switch).
_USE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


# The kernels import the CuTe DSL; the decode collectives import this module on hosts that may not have it.
def _lamport():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_lamport_ar

    return k3_lamport_ar


def _lowm():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


def _attn_res():
    # Also registers trtllm::k3_attn_res_add_rmsnorm_lamport.
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    return k3_attn_res


class LamportPushTarget:
    """One all-reduce on a Lamport workspace whose input the producer pushes.

    Made by `push_target` for a single call. The producer stores its bf16 ``[num_tokens, hidden]`` output into the
    Lamport slots of ``workspace`` (a ``k3_lamport_ar.LamportWorkspace``) with the ``one_shot`` or two-shot transport
    and sets ``pushed``; `finish_lamport_push` then completes the all-reduce, reproducing the C++ one-shot kernel's
    order if ``cpp_one_shot``, else the two-shot kernel's. ``output``: where a gathered result goes (None for a call
    finished in place).
    """

    def __init__(
        self,
        workspace: LamportWorkspace,
        one_shot: bool,
        cpp_one_shot: bool,
        output: Optional[torch.Tensor],
        shape: Tuple[int, int],
    ):
        self.workspace = workspace
        self.one_shot = one_shot
        self.cpp_one_shot = cpp_one_shot
        self.output = output
        self.shape = shape
        self.pushed = False


class LamportResult:
    """The result of a pushed two-shot all-reduce left in the workspace (`finish_lamport_push` with ``in_place``).

    A consumer that runs next reads its rows with ``k3_lamport_ar.poll_result_v4(workspace.uc_local,
    workspace.flags, ...)`` after its own ``griddepcontrol.wait`` (``lamport_addend``: the address pair it takes);
    `materialize` copies them into a tensor instead. The rows live until the next all-reduce on the workspace re-arms
    their buffer, which happens once that call's producer has passed its own ``griddepcontrol.wait``: the consumer
    must lie on every dependency path from this call to the next one on the workspace, and every kernel between them
    must wait for its predecessor before it triggers or completes.
    """

    def __init__(self, target: LamportPushTarget):
        self.target = target
        self.workspace = target.workspace
        self.shape = tuple(target.shape)

    @property
    def lamport_addend(self) -> Tuple[int, int]:
        """``(uc_local, flags)``: the device addresses a consumer polls the rows with."""
        return self.workspace.uc_local, self.workspace.flags.data_ptr()

    def materialize(self) -> torch.Tensor:
        """The result as a bf16 tensor (a copy kernel on the current stream)."""
        out = torch.empty(self.shape, dtype=torch.bfloat16, device=self.workspace.flags.device)
        return _lamport().run_lamport_gather(self.workspace, out, use_pdl=_USE_PDL)


def lamport_view(workspace: MnnvlWorkspace) -> LamportWorkspace:
    """The device view (``k3_lamport_ar.LamportWorkspace``) of the TP group's ``MnnvlWorkspace``: every rank's
    unicast address of it, its multicast address, its flags and one buffer's size. Its peer table is a host-to-device
    copy, which CUDA-graph capture does not allow: make it eagerly, once."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("lamport_view copies a peer table to the device: make it before capture")
    return _lamport().LamportWorkspace.from_mnnvl(
        dict(
            handle=workspace.handle,
            uc_buffer=workspace.lamport,
            buffer_flags=workspace.buffer_flags,
            buffer_size_bytes=workspace.buffer_bytes,
        ),
        workspace.world_size,
        workspace.rank,
    )


def push_target(
    workspace: LamportWorkspace,
    num_tokens: int,
    hidden: int,
    in_place: bool = False,
    one_shot: bool = False,
    cpp_one_shot: bool = True,
) -> Optional[LamportPushTarget]:
    """A `LamportPushTarget` for an all-reduce of bf16 ``[num_tokens, hidden]`` on ``workspace``, or None where one
    Lamport buffer does not hold the call. ``in_place``: the caller finishes it in place, which takes the two-shot
    transport whatever ``one_shot`` says. ``cpp_one_shot``: the result is the C++ one-shot kernel's (the order of
    every ``mnnvl_allreduce_attn_res`` call), else the two-shot kernel's."""
    one_shot = one_shot and not in_place
    if num_tokens <= 0 or hidden % 8:
        return None
    need = _lamport().required_bytes(num_tokens, hidden, workspace.world, one_shot)
    if need > workspace.bytes_per_buffer:
        return None
    output = None
    if not in_place:
        output = torch.empty(
            num_tokens, hidden, dtype=torch.bfloat16, device=workspace.flags.device
        )
    return LamportPushTarget(workspace, one_shot, cpp_one_shot, output, (num_tokens, hidden))


def finish_lamport_push(
    target: LamportPushTarget, in_place: bool = False
) -> Union[torch.Tensor, LamportResult]:
    """Complete a pushed all-reduce: ``target.output``, or with ``in_place`` (two-shot transport only) a
    `LamportResult` the next kernel reads from the workspace."""
    if not target.pushed:
        raise RuntimeError("the producer did not push this all-reduce")
    if in_place:
        if target.one_shot:
            raise ValueError("only the two-shot transport leaves its result in the workspace")
        _lamport().run_lamport_finish_in_place(
            target.workspace,
            target.shape[0],
            target.shape[1],
            use_pdl=_USE_PDL,
            cpp_one_shot=target.cpp_one_shot,
        )
        return LamportResult(target)
    return _lamport().run_lamport_finish(
        target.workspace,
        target.output,
        target.one_shot,
        use_pdl=_USE_PDL,
        cpp_one_shot=target.cpp_one_shot,
    )


def _eps(norm: nn.Module) -> float:
    eps = norm.eps if hasattr(norm, "eps") else norm.variance_epsilon
    return float(torch.finfo(norm.weight.dtype).eps if eps is None else eps)


def _pushes(o_proj: nn.Module, rows: int, gated: bool) -> bool:
    """Whether ``o_proj`` is an FP8 module on ``k3_fp8w_lowm_gemm`` (``fp8w_lowm.install``) whose push epilogue serves
    ``rows`` tokens: the plain GEMM's checks on the weight side, ``HIDDEN`` output rows, and a low-M tactic without
    split-K (``ar_push_tactic``). Metadata only."""
    weight = getattr(o_proj, "weight", None)
    scale = getattr(o_proj, "weight_scale", None)
    if not getattr(o_proj, "_k3ll_fp8w_lowm", False) or getattr(o_proj, "bias", None) is not None:
        return False
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or not weight.is_contiguous():
        return False
    if not weight.is_cuda or weight.data_ptr() % 16:
        return False
    n, k = weight.shape
    if n != HIDDEN or n % _lowm().MMA_M or k % _lowm().CTA_K:
        return False
    words = -(-(k // _lowm().CTA_K) // 4)
    if (
        scale.dtype != torch.int32
        or tuple(scale.shape) != (n, words)
        or scale.stride(0) != 1
        or (words > 1 and scale.stride(1) != n)
    ):
        return False
    if not _lowm()._is_supported_device(weight.device.index):
        return False
    return _lowm().ar_push_tactic(rows, n, k, gated=gated) is not None


@dataclass(eq=False)
class K3OprojPush:
    """The switch's state on the TP group's ``MnnvlWorkspace``: its device view and a bf16 zero prefix ``[MAX_TOKENS,
    HIDDEN]`` (never written) for a layer that has just snapshotted its prefix sum. Built by `create` with the decode
    collectives (``K3DecodeComm.create``); the workspace's rotation state is its own (device side)."""

    lamport: LamportWorkspace
    zeros: torch.Tensor

    @classmethod
    def create(cls, workspace: MnnvlWorkspace) -> "K3OprojPush":
        """The state over ``workspace``. Rank-local (no collective), eager, before any CUDA-graph capture."""
        view = lamport_view(workspace)
        zeros = torch.zeros(
            MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=workspace.buffer_flags.device
        )
        return cls(view, zeros)

    @staticmethod
    def enabled(step) -> bool:
        """Whether ``TRTLLM_K3LL_OPROJ_PUSH`` (read at forward time) asks for ``step``: a pure decode step (any with 1,
        a wide one with 2)."""
        mode = _flags.TRTLLM_K3LL_OPROJ_PUSH
        if not mode or step is None or not step.decode:
            return False
        return mode == EVERY_DECODE_STEP or (mode == WIDE_STEPS and step.wide)

    def takes(
        self,
        step,
        rows: int,
        o_proj: nn.Module,
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_proj: nn.Module,
        res_norm: nn.Module,
        out_norm: nn.Module,
        gated: bool = False,
    ) -> bool:
        """Whether `run` takes a post-attention step of ``rows`` tokens with these operands: the switch asks for
        ``step`` (`enabled`), ``rows`` is the step's token count (at most `MAX_TOKENS`), o_proj's push epilogue serves
        it (with ``gated``, the output-gated GEMM's), one Lamport buffer holds the two-shot call, and the attn_res
        kernel takes the prefix sum (``None``: the zero prefix), the snapshots ``[S, rows, HIDDEN]`` and the weights.
        Read from shapes, the switch and load-time state only, so every rank of the group decides alike."""
        if not self.enabled(step) or rows != step.num_tokens or not 0 < rows <= MAX_TOKENS:
            return False
        if not _pushes(o_proj, rows, gated):
            return False
        if _lamport().required_bytes(rows, HIDDEN, self.lamport.world, False) > (
            self.lamport.bytes_per_buffer
        ):
            return False
        if _attn_res()._EXACT:
            return False
        prefix = self.zeros[:rows] if prefix_sum is None else prefix_sum
        weights = (res_proj.weight.reshape(-1), res_norm.weight, out_norm.weight)
        return _attn_res().can_use_k3_attn_res(prefix, (prefix,), block_residual, *weights)

    def run(
        self,
        x: torch.Tensor,
        o_proj: nn.Module,
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_proj: nn.Module,
        res_norm: nn.Module,
        out_norm: nn.Module,
        gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(normed, updated)`` of a post-attention step `takes` holds for: ``updated = prefix_sum + allreduce(x @
        o_proj.weight^T)`` (the sum alone without ``prefix_sum``), ``normed = out_norm(attn_res(block_residual...,
        updated))``, the attention residual selected with ``res_proj`` and ``res_norm``; with ``gate``, o_proj of
        ``bf16(x * gate)`` (the gated GEMM). ``x``: this rank's o_proj input, ``[rows, K]`` bf16."""
        x2d = x.reshape(-1, o_proj.weight.shape[1])
        if not x2d.is_contiguous() or x2d.data_ptr() % 16:
            x2d = x2d.clone(memory_format=torch.contiguous_format)
        rows = x2d.shape[0]
        target = push_target(self.lamport, rows, HIDDEN, in_place=True)
        _lowm().run_k3_fp8w_lowm_gemm_ar_push(
            x2d, o_proj.weight, o_proj.weight_scale, self.lamport, target.one_shot, gate=gate
        )
        target.pushed = True
        result = finish_lamport_push(target, in_place=True)
        prefix = self.zeros[:rows] if prefix_sum is None else prefix_sum
        normed = torch.empty_like(prefix)
        updated = torch.empty_like(prefix)
        # trtllm::k3_attn_res_add_rmsnorm_lamport, releasing the next kernel at its wait (see the module docstring).
        _attn_res().run_k3_attn_res(
            prefix,
            (),
            block_residual,
            res_proj.weight.reshape(-1),
            res_norm.weight,
            out_norm.weight,
            _eps(res_norm),
            _eps(out_norm),
            normed,
            updated,
            exact=False,
            lamport_addend=result.lamport_addend,
            early_trigger=True,
        )
        return normed, updated


__all__ = [
    "EVERY_DECODE_STEP",
    "HIDDEN",
    "K3OprojPush",
    "LamportPushTarget",
    "LamportResult",
    "MAX_TOKENS",
    "WIDE_STEPS",
    "finish_lamport_push",
    "lamport_view",
    "push_target",
]
