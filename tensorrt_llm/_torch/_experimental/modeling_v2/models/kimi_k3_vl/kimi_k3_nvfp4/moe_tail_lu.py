# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A wide decode step's MoE tail with a replicated FP8 latent_up (``TRTLLM_K3LL_MOE_TAIL_LU``). LOSSY.

Without the switch a MoE layer of a wide decode step (`decode_moe_lowm.py`, 9-32 tokens) runs the front, the routed
experts, ``comm/mnnvl_allreduce_rmsnorm_slice`` (the latent's reduce-scatter and RMSNorm: this rank keeps its 224
columns), and hands the row-parallel tail ``[latent slice | padding | shared activation] @ tail_weight.T`` on to its
consumer, which pushes it into one more all-reduce (``moe_tail_push.py``) and runs ``K3AttnResFastKernel``. So the
step has two collectives in a row after the experts.

With the switch, where `takes` holds, the layer runs:

* the front (``k3_moe_front``'s 16- or 32-token build), unchanged (the layer decides before it, `takes`);
* on a high-priority side stream forked right after it: the shared expert's down projection, row parallel, bf16, on
  the decode GEMV site `SHARED_SITE` (``k3_ctm_gemv_wide``) with its all-reduce push epilogue into the decode
  collectives' MNNVL workspace, then ``K3LamportFinishKernel`` gathering the sum into a tensor (C++ one-shot order);
  it runs under the routed experts;
* on the main stream: the routed experts, then ``K3LatentArRowKernel`` (``k3_latent_ar_norm``): the full latent's
  all-reduce and its RMSNorm on every rank, on an MNNVL workspace of its own (`K3TailLuState`, so it cannot race the
  side stream's all-reduce), storing the bf16 latent and its MXFP8 form;
* latent_up replicated: ``k3_fp8w_lowm_mxfp8`` on that MXFP8 form with this layer's full latent_up weight in FP8
  (128 x 128 blocks with power-of-two scales, quantized at load from the checkpoint's BF16, `quantize_block_fp8`): the
  ``[T, 7168]`` routed output, already reduced;
* the main stream joins the side stream, and the layer hands on a `LuTail` (routed, shared). Its consumer, the next
  layer's pre-attention step (the final norm after the last layer), runs ``K3AttnResFastKernel`` with the MoE add
  (``bf16(prefix + bf16(routed + shared))``) in the C++ MNNVL epilogue's arithmetic (``mnnvl_order``): `attn_res`.
  Where the kernel does not take the consumer's operands it adds the two in torch and runs the fused add + attn_res +
  RMSNorm (`LuTail.reduced`).

Numerics. The latent all-reduce and its RMSNorm are the C++ one-shot kernel's (``kARRMSNorm``), with the latent norm's
own weight (the layer keeps a copy taken before ``decode_moe.fold_latent_norm`` folds it into the bf16 tail weight;
the FP8 weight is quantized from the unfolded latent_up). latent_up is the FP8 GEMM of the MXFP8 latent (the low-M FP8
GEMM's in-kernel quantization) and the FP8 weight: lossy. The shared expert's output is its own bf16 GEMV, summed over
the ranks; the switch-off path sums it inside the tail GEMV, so it rounds differently too.

Which steps (`takes`; switch, step shape and load-time state only, so every rank of the group decides alike): the
switch is set, the layer built this state at load (``TRTLLM_K3LL_MOE_TAIL_LU`` at launch), 9 to 32 tokens, a front
build runs the step (``K3DecodeMoeNvfp4.front_head``; elsewhere the shared activation comes too late for the side
stream), the shared site's push GEMV takes the rows, both Lamport buffers hold the calls, and the MXFP8 GEMM compiled
for the rows at load. Elsewhere the layer runs the switch-off path, unchanged.

Ordering. The side stream forks after the front, which waits for the post-attention step (the last call on the
decode workspace) before it completes, so the shared push finds that call done; the main stream issues no call on
the decode workspace until it joins the side stream, right after latent_up, before the consumer's attn_res kernel;
the next call on that workspace follows the attn_res kernel. The latent workspace sees only the latent all-reduces,
one per engaged layer, in layer order on every rank. Outside CUDA-graph capture (``do_multi_stream`` off) the side
stream's kernels run first on the main stream: the same kernels and bits. The fork and join are events, captured
as graph edges.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream

from . import flags as _flags
from .oproj_push import LamportPushTarget, finish_lamport_push, lamport_view

#: The decode GEMV site of the shared expert's down projection (``decode_gemv.SITES``): TP16 per rank [7168, 384].
SHARED_SITE = "moe_shared_down"
#: The step sizes the switch takes: a wide step on the front's 16- or 32-token build.
MIN_ROWS, MAX_ROWS = 9, 32
LATENT = 3584
HIDDEN = 7168
#: One Lamport buffer of the latent workspace: a one-shot call of MAX_ROWS rows of 16 ranks is 3.5 MiB.
LATENT_BUFFER_BYTES = 4 << 20
#: CUDA priority of the side stream (lower is higher): its small kernels take SMs ahead of the routed experts'
#: pending persistent CTAs.
SIDE_STREAM_PRIORITY = -1
#: Weight rows quantized at a time at load (whole 128-row blocks).
_QUANT_CHUNK_ROWS = 1024


def _latent_ar():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_latent_ar_norm

    return k3_latent_ar_norm


def _mx():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8

    return k3_fp8w_lowm_mxfp8


def _lowm():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


def _attn_res():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    return k3_attn_res


def _lamport():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_lamport_ar

    return k3_lamport_ar


#: ``TRTLLM_K3LL_MOE_TAIL_LU``'s default (-1) turns it on only on these compute capabilities (measured on GB300 /
#: SM103). Its kernels (``k3_fp8w_lowm_mxfp8``, ``K3LatentArRowKernel``) run on SM100 / SM103 only; elsewhere the load
#: builds no state and the layer keeps the push tail, even when forced with 1.
MOE_TAIL_LU_CAPABILITIES: Tuple[Tuple[int, int], ...] = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _on_by_default() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in MOE_TAIL_LU_CAPABILITIES


def enabled() -> bool:
    """``TRTLLM_K3LL_MOE_TAIL_LU`` (read at load for the state, at forward time for engagement): 1 on, 0 off, -1
    (default) on where the device is in ``MOE_TAIL_LU_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_TAIL_LU)
    if value >= 0:
        return value > 0
    return _on_by_default()


def quantize_block_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scale)`` of a bf16 ``[N, K]`` weight (N and K multiples of 128) as the checkpoint's FP8 weights are
    stored: per 128 x 128 block the smallest power-of-two scale that keeps the block's absolute maximum within E4M3's
    448 (the maximum floored at 1e-4), E4M3 codes ``weight / scale`` (round to nearest). ``scale`` is fp32
    ``[N / 128, K / 128]``; ``dequantize_block_fp8`` inverts it up to the rounding."""
    n, k = weight.shape
    if weight.dim() != 2 or n % 128 or k % 128:
        raise ValueError(
            f"quantize_block_fp8: weight {tuple(weight.shape)} is not of whole 128 x 128 blocks"
        )
    codes = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty(n // 128, k // 128, dtype=torch.float32, device=weight.device)
    for lo in range(0, n, _QUANT_CHUNK_ROWS):
        hi = min(n, lo + _QUANT_CHUNK_ROWS)
        blocks = weight[lo:hi].float().view((hi - lo) // 128, 128, k // 128, 128)
        amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
        s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        codes[lo:hi] = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn).view(hi - lo, k)
        scale[lo // 128 : hi // 128] = s
    return codes, scale


def dequantize_block_fp8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The fp32 weight ``quantize_block_fp8``'s ``(codes, scale)`` stand for."""
    n, k = codes.shape
    blocks = codes.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]
    return blocks.view(n, k)


class LuTail(NamedTuple):
    """A wide decode step's MoE output handed on by ``TRTLLM_K3LL_MOE_TAIL_LU``: ``routed`` (latent_up of the reduced,
    normalized latent) and ``shared`` (the shared expert's reduced output), both bf16 ``[T, 7168]`` and the same on
    every rank. The consumer adds ``bf16(routed + shared)`` to its prefix sum. ``candidates`` / ``mx_candidates``: the
    candidate counts (snapshots + 1) the attn_res kernel with the MoE add compiled for at load, without / with the
    MXFP8 side output (``K3TailLuState``)."""

    routed: torch.Tensor
    shared: torch.Tensor
    candidates: frozenset = frozenset()
    #: The candidate counts whose MoE-add attn_res kernel with the MXFP8 side output (``TRTLLM_K3LL_ATTN_RES_MXFP8``)
    #: compiled at load.
    mx_candidates: frozenset = frozenset()

    def reduced(self) -> torch.Tensor:
        """The MoE output ``bf16(routed + shared)`` (one bf16 rounding, as the attn_res kernel's MoE add)."""
        return self.routed + self.shared


@dataclass(eq=False)
class K3TailLuState:
    """What every MoE layer's switch shares on one device: the latent all-reduce's own MNNVL workspace and its device
    view, the decode collectives' workspace view the shared expert's all-reduce pushes into (``K3OprojPush.lamport``),
    the side stream and its fork / join events, and what the load-time warm-up compiled."""

    latent_ws: MnnvlWorkspace
    latent: object  # k3_lamport_ar.LamportWorkspace
    push: object  # k3_lamport_ar.LamportWorkspace of the decode collectives' workspace
    stream: torch.cuda.Stream
    fork: torch.cuda.Event
    join: torch.cuda.Event
    #: Token counts whose MXFP8 latent_up GEMM ran at load.
    mx_rows: frozenset = frozenset()
    #: The latent all-reduce, the shared expert's push (16- and 32-token tiles) and finish ran at load.
    ready: bool = False
    #: Candidate counts (snapshots + 1) whose MoE-add attn_res kernel compiled at load.
    attn_res_candidates: frozenset = field(default_factory=frozenset)
    #: The same with the MXFP8 side output (``warm_up(..., mxfp8=True)``).
    attn_res_mx_candidates: frozenset = field(default_factory=frozenset)

    @classmethod
    def create(cls, mapping, push) -> "K3TailLuState":
        """The state for ``mapping``'s TP group over the decode collectives' workspace view ``push``. Collective (the
        latent workspace): every rank of the group calls it at the same point, eagerly, before any CUDA-graph
        capture."""
        workspace = MnnvlWorkspace.create(mapping, LATENT_BUFFER_BYTES)
        device = workspace.buffer_flags.device
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device, priority=SIDE_STREAM_PRIORITY)
        return cls(
            latent_ws=workspace,
            latent=lamport_view(workspace),
            push=push,
            stream=stream,
            fork=torch.cuda.Event(),
            join=torch.cuda.Event(),
        )


def layer_gaps(moe: nn.Module, tp_size: int) -> list:
    """Why the switch cannot serve MoE layer ``moe`` (empty when it can), read before the latent norm is folded."""
    up = moe.routed_expert_up_proj.weight
    norm = moe.routed_expert_norm
    down = moe.shared_experts.down_proj.weight
    if tuple(up.shape) != (HIDDEN, LATENT) or up.dtype != torch.bfloat16:
        return [f"latent_up {tuple(up.shape)} {up.dtype}"]
    if tuple(norm.weight.shape) != (LATENT,) or norm.weight.dtype != torch.bfloat16:
        return ["a latent norm other than bf16 [3584]"]
    if tp_size not in _latent_ar().ROW_WORLD_SIZES:
        return [
            f"TP {tp_size}: the latent all-reduce row kernel runs {_latent_ar().ROW_WORLD_SIZES}"
        ]
    if down.dim() != 2 or down.shape[0] != HIDDEN or not down.is_contiguous():
        return [f"shared down {tuple(down.shape)}"]
    return []


@dataclass(eq=False)
class K3TailLuLayer:
    """One MoE layer's switch weights: latent_up in FP8 (E4M3 ``[7168, 3584]`` and DeepGEMM's packed UE8M0 block
    scale, what ``k3_fp8w_lowm_mxfp8`` reads), the latent norm's weight and epsilon (before the fold), and the shared
    expert's bf16 down weight (the module's own, ``[7168, 384]`` per rank)."""

    up_weight: torch.Tensor
    up_scale: torch.Tensor
    gamma: torch.Tensor
    eps: float
    shared_down: torch.Tensor

    @classmethod
    def create(cls, moe: nn.Module) -> "K3TailLuLayer":
        """Build MoE layer ``moe``'s switch weights (``layer_gaps`` empty). Must run before
        ``decode_moe.fold_latent_norm`` folds the latent norm into latent_up."""
        with torch.no_grad():
            codes, scale = quantize_block_fp8(moe.routed_expert_up_proj.weight)
            packed = _lowm().pack_ue8m0_block_scale(scale, HIDDEN, LATENT)
            norm = moe.routed_expert_norm
            return cls(
                up_weight=codes,
                up_scale=packed,
                gamma=norm.weight.detach().clone().contiguous(),
                eps=float(norm.variance_epsilon),
                shared_down=moe.shared_experts.down_proj.weight.detach(),
            )

    @property
    def extra_bytes(self) -> int:
        """Device memory this layer adds: the FP8 weight, its scale and the norm weight's copy."""
        return sum(
            t.numel() * t.element_size() for t in (self.up_weight, self.up_scale, self.gamma)
        )


def takes(
    state: Optional[K3TailLuState], layer: Optional[K3TailLuLayer], rows: int, act_width: int, gemvs
) -> bool:
    """Whether `run` takes a wide step of ``rows`` tokens whose front build produces a shared activation ``act_width``
    wide (decided before the front runs): the switch is set, the load-time state exists and compiled the step's
    kernels, 9 to 32 tokens, the shared site's push takes the rows and both Lamport buffers hold the calls. Shapes,
    the switch and load-time state only."""
    if not enabled() or state is None or layer is None or gemvs is None or not state.ready:
        return False
    if not MIN_ROWS <= rows <= MAX_ROWS or rows not in state.mx_rows:
        return False
    latent, push = state.latent, state.push
    if rows > _latent_ar().ROW_MAX_M or rows * latent.world * LATENT * 2 > latent.bytes_per_buffer:
        return False
    if _lamport().required_bytes(rows, HIDDEN, push.world, False) > push.bytes_per_buffer:
        return False
    return gemvs.takes_push_rows(SHARED_SITE, rows, act_width, layer.shared_down, push.world)


def shared_all_reduce(state: K3TailLuState, gemvs, shared_down: torch.Tensor, act: torch.Tensor,
                      out: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """``out = allreduce(act @ shared_down.T)`` on the current stream: the push GEMV into the decode collectives'
    workspace, then the Lamport finish gathering the sum (the C++ one-shot kernel's order) into ``out``."""
    target = LamportPushTarget(state.push, False, True, out, (act.shape[0], HIDDEN))
    gemvs.project_push(SHARED_SITE, act, shared_down, state.push)
    target.pushed = True
    return finish_lamport_push(target)


def latent_all_reduce(
    state: K3TailLuState, layer: K3TailLuLayer, routed: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(latent, xq, xs)``: RMSNorm(gamma) of the sum over the ranks of ``routed`` (bf16 ``[T, 3584]``) on the latent
    workspace, and its MXFP8 form (E4M3 ``[T, 3584]``, UE8M0 ``[T, 112]``)."""
    rows = routed.shape[0]
    routed = routed.contiguous()
    latent = torch.empty_like(routed)
    xq = torch.empty(rows, LATENT, dtype=torch.float8_e4m3fn, device=routed.device)
    xs = torch.empty(rows, LATENT // 32, dtype=torch.uint8, device=routed.device)
    view = state.latent
    _latent_ar().run_k3_latent_ar_norm_row(
        routed, layer.gamma, layer.eps, latent, uc_addr=view.uc_local, mc_addr=view.mc_base, flags=view.flags,
        rank=view.rank, world_size=view.world, early_trigger=True, mxfp8_out=(xq, xs),
    )  # fmt: skip
    return latent, xq, xs


def latent_up(layer: K3TailLuLayer, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """latent_up of the MXFP8 latent ``(xq, xs)`` with the FP8 weight: bf16 ``[T, 7168]``."""
    out = torch.empty(xq.shape[0], HIDDEN, dtype=torch.bfloat16, device=xq.device)
    _mx().run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, layer.up_weight, layer.up_scale, out)
    return out


def run(
    state: K3TailLuState,
    layer: K3TailLuLayer,
    gemvs,
    shared_act: torch.Tensor,
    routed_fn: Callable[[], torch.Tensor],
) -> LuTail:
    """The tail of a step `takes` holds for, after the front: the shared expert's down projection and all-reduce on
    the side stream (in order on the current stream outside multi-stream capture), ``routed_fn()`` (this rank's routed
    experts' latent partial) on the current stream, the latent all-reduce + RMSNorm and latent_up; then the current
    stream joins the side stream."""
    rows = shared_act.shape[0]
    shared = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=shared_act.device)
    side = do_multi_stream() and not torch.compiler.is_compiling()
    if side:
        main = torch.cuda.current_stream()
        state.fork.record(main)
        with torch.cuda.stream(state.stream):
            state.stream.wait_event(state.fork)
            shared_all_reduce(state, gemvs, layer.shared_down, shared_act, shared)
            state.join.record(state.stream)
    else:
        shared_all_reduce(state, gemvs, layer.shared_down, shared_act, shared)
    _, xq, xs = latent_all_reduce(state, layer, routed_fn())
    routed = latent_up(layer, xq, xs)
    if side:
        main.wait_event(state.join)
    return LuTail(routed, shared, state.attn_res_candidates, state.attn_res_mx_candidates)


def warm_up(state: K3TailLuState, layer: K3TailLuLayer, gemvs, max_candidates: int, mxfp8: bool = False) -> None:
    """Compile every kernel of the switch before any capture, whatever the switch says at forward time. Collective:
    every rank makes the same calls (one shared all-reduce and one latent all-reduce on zeros, advancing both
    workspaces alike). ``mxfp8``: also the attn_res kernel with the MXFP8 side output (``TRTLLM_K3LL_ATTN_RES_MXFP8``
    on this device at load). Records what it compiled in ``state``."""
    device = layer.up_weight.device
    zeros = dict(dtype=torch.bfloat16, device=device)
    tiles = gemvs.compile_push(SHARED_SITE, layer.shared_down, state.push.world)
    if tiles != [16, 32]:
        torch.cuda.synchronize(device)
        return
    weight = torch.zeros_like(layer.shared_down)
    for rows in (16, 32):
        act = torch.zeros(rows, weight.shape[1], **zeros)
        # The plain wide GEMV of the site (``takes_push`` asks for it under capture) and the push + finish.
        gemvs.project(SHARED_SITE, act, weight)
        shared_all_reduce(state, gemvs, weight, act, torch.empty(rows, HIDDEN, **zeros))
    probe = K3TailLuLayer(
        layer.up_weight, layer.up_scale, torch.ones(LATENT, **zeros), 1e-6, weight
    )
    latent_all_reduce(state, probe, torch.zeros(MIN_ROWS, LATENT, **zeros))
    mx_rows = []
    up_zero = torch.zeros_like(layer.up_weight)
    for rows in range(MIN_ROWS, MAX_ROWS + 1):
        xq = torch.zeros(rows, LATENT, dtype=torch.float8_e4m3fn, device=device)
        xs = torch.full((rows, LATENT // 32), 127, dtype=torch.uint8, device=device)
        if _mx().can_run_mxfp8(xq, xs, up_zero, layer.up_scale):
            latent_up(K3TailLuLayer(up_zero, layer.up_scale, probe.gamma, 1e-6, weight), xq, xs)
            mx_rows.append(rows)
    candidates = []
    ones = torch.ones(HIDDEN, **zeros)
    row = torch.zeros(1, HIDDEN, **zeros)
    for n in range(_attn_res().MIN_CANDIDATES, min(max_candidates, _attn_res().MAX_CANDIDATES) + 1):
        block = torch.zeros(n - 1, 1, HIDDEN, **zeros)
        _attn_res().run_k3_attn_res(
            row, (row, row), block, ones, ones, ones, 1e-6, 1e-6, torch.empty_like(row), torch.empty_like(row),
            exact=False, early_trigger=True, mnnvl_order=True,
        )  # fmt: skip
        candidates.append(n)
    mx_candidates = []
    if mxfp8:
        xq = torch.empty(1, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
        xs = torch.empty(1, HIDDEN // 32, dtype=torch.uint8, device=device)
        for n in candidates:
            block = torch.zeros(n - 1, 1, HIDDEN, **zeros)
            _attn_res().run_k3_attn_res(
                row, (row, row), block, ones, ones, ones, 1e-6, 1e-6, torch.empty_like(row), torch.empty_like(row),
                exact=False, early_trigger=True, mnnvl_order=True, mxfp8_out=(xq, xs),
            )  # fmt: skip
            mx_candidates.append(n)
    torch.cuda.synchronize(device)
    state.mx_rows = frozenset(mx_rows)
    state.attn_res_candidates = frozenset(candidates)
    state.attn_res_mx_candidates = frozenset(mx_candidates)
    state.ready = True


def _eps(norm: nn.Module) -> float:
    eps = norm.eps if hasattr(norm, "eps") else norm.variance_epsilon
    return float(torch.finfo(norm.weight.dtype).eps if eps is None else eps)


def attn_res_takes(
    tail: LuTail,
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    res_proj: nn.Module,
    res_norm: nn.Module,
    out_norm: nn.Module,
) -> bool:
    """Whether `attn_res` takes the consumer's operands: the vectorized attn_res kernel (not
    ``TRTLLM_K3_ATTN_RES_PDL_EXACT``) compiled at load for the candidate count (``tail.candidates``) takes the prefix
    sum ``prefix``, the tail, the snapshots ``[S, T, 7168]`` and the weights."""
    if (
        _attn_res()._EXACT
        or block_residual.dim() != 3
        or block_residual.shape[0] + 1 not in tail.candidates
    ):
        return False
    weights = (res_proj.weight.reshape(-1), res_norm.weight, out_norm.weight)
    return _attn_res().can_use_k3_attn_res(
        prefix, (tail.routed, tail.shared), block_residual, *weights
    )


def attn_res(
    tail: LuTail,
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    res_proj: nn.Module,
    res_norm: nn.Module,
    out_norm: nn.Module,
    mxfp8_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(normed, updated)``: ``updated = bf16(prefix + bf16(tail.routed + tail.shared))``, ``normed =
    out_norm(attn_res(block_residual..., updated))`` (selected with ``res_proj`` and ``res_norm``), in one
    ``K3AttnResFastKernel`` launch in the C++ MNNVL epilogue's arithmetic, releasing the next kernel at its wait.
    ``mxfp8_out``: passed to ``run_k3_attn_res`` (the MXFP8 form of ``normed``)."""
    normed = torch.empty_like(prefix)
    updated = torch.empty_like(prefix)
    _attn_res().run_k3_attn_res(
        prefix, (tail.routed, tail.shared), block_residual, res_proj.weight.reshape(-1), res_norm.weight,
        out_norm.weight, _eps(res_norm), _eps(out_norm), normed, updated, exact=False, early_trigger=True,
        mnnvl_order=True, mxfp8_out=mxfp8_out,
    )  # fmt: skip
    return normed, updated


__all__ = [
    "HIDDEN",
    "LATENT",
    "MAX_ROWS",
    "MIN_ROWS",
    "SHARED_SITE",
    "K3TailLuLayer",
    "K3TailLuState",
    "LuTail",
    "attn_res",
    "attn_res_takes",
    "dequantize_block_fp8",
    "enabled",
    "latent_all_reduce",
    "latent_up",
    "layer_gaps",
    "quantize_block_fp8",
    "run",
    "shared_all_reduce",
    "takes",
    "warm_up",
]
