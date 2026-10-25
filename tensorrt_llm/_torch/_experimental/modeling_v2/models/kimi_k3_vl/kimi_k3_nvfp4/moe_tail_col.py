# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A wide decode step's MoE tail with a column-sharded BF16 latent_up (``TRTLLM_K3LL_MOE_TAIL_COL``). Lossless.

Where `takes` holds (the steps ``TRTLLM_K3LL_MOE_TAIL_LU`` takes: 9-32 tokens on the front's 16- / 32-token builds),
the layer runs, after the front:

* on ``K3TailLuState``'s high-priority side stream (forked after the front): the shared expert's down projection with
  its all-reduce, as ``TRTLLM_K3LL_MOE_TAIL_LU`` runs it (``moe_tail_lu.shared_all_reduce``: the push GEMV into the
  decode collectives' MNNVL workspace, then ``K3LamportFinishKernel`` gathering the bf16 ``[T, 7168]`` sum);
* on the main stream: the routed experts, then ``K3LatentArRowKernel`` (the full latent's all-reduce and RMSNorm on
  every rank, on ``K3TailLuState``'s own latent workspace), storing the bf16 normalized latent ``[T, 3584]`` (no
  MXFP8 form);
* the main stream joins the side stream, then the column publish (``k3_ctm_gemv_wide`` with its column-publish
  epilogue, ``k3_ctm_gemv.op.run_k3_ctm_gemv_wide_col_publish``): rank ``r`` computes only its ``W = 7168 / TP``
  output columns ``[W r, W r + W)`` (448 at TP16), ``latent @ up[W r : W r + W, :].T`` with fp32 sums from its row
  slice of the layer's bf16 latent_up (3.2 MB at TP16, against 25.9 MB of TAIL_LU's FP8 weight), adds the shared
  output's same columns, rounds once to bf16, and multicasts the slice into the broadcast stage of the decode
  workspace's next Lamport buffer on every rank. The 16 slices form the ``[T, 7168]`` rows a two-shot all-reduce's
  finish would leave there, so the layer hands on a `ColTail` naming them;
* its consumer, the next layer's pre-attention step (the final norm after the last layer), runs
  ``K3AttnResFastKernel`` with the Lamport addend (``attn_res``), exactly the kernel ``TRTLLM_K3LL_MOE_TAIL_PUSH`` /
  ``TRTLLM_K3LL_OPROJ_PUSH`` run after their finish: no finish kernel at all. Where the kernel does not take the
  consumer's operands, ``ColTail.materialize`` gathers the rows into a tensor (``K3LamportGatherKernel``) and the
  consumer runs the fused add + attn_res + RMSNorm on it.

The attn_res kernel polls every 16-byte vector of every row (``k3_lamport_ar.poll_result_v4``: until none of its
four 32-bit words is the Lamport sentinel), not a per-row flag, so the ranks' slices (whole 16-byte vectors: a
slice is a multiple of 8 columns) need no other signal.

Lamport rotation. The decode workspace's rotation state (current / dirty buffer, bytes to clear, CTA arrivals) lives
in its device flags, read and advanced by every call on it. The publish is one call (call ``n``): its producer and
its finish in one kernel. Every epilogue thread reads the state after its ``griddepcontrol.wait`` (the previous call,
the shared all-reduce, has completed: the main stream joined the side stream before the launch); each CTA counts
itself once its epilogue has read it, and the last CTA writes call ``n + 1``'s state (current + 1, dirty = this
buffer, two stages, ``T`` rows of the broadcast stage to clear), as the finish kernel's updater does; every CTA
re-arms its share of call ``n - 1``'s buffer (the shared all-reduce's, which its own gather has finished reading,
every rank's pushes into it having arrived) after its stores. The consumer reads the state after its own wait, finds
this call's buffer as the dirty one and polls its broadcast stage, as after a finish. Call ``n``'s buffer was re-armed
during call ``n - 2`` on every rank; a rank writes into a peer's buffer of call ``n`` only after its own call
``n - 1`` completed, which needed every peer's call ``n - 1`` data, which each peer sends after its call ``n - 2``
completed. Call ``n + 1`` (the next layer's first call on the workspace, e.g. its o_proj push) re-arms this buffer
only once its producer has passed its ``griddepcontrol.wait``, so the consumer must lie on every dependency path from
the publish to that call and every kernel between them must wait for its predecessor before it triggers or
completes: the decode path's kernels do (as for the push tail's attn_res). The publish releases its dependent (the
consumer) from its activation warp after its own wait, as the push GEMVs do, so the consumer's reads before its wait
(prefix sum, snapshots) see only completed grids. A CUDA graph captures no host state of the rotation; every rank
makes the same calls on both workspaces in the same order (`takes` reads the switch, shapes and load-time state).

Numerics. Lossless (the checkpoint's bf16 latent_up, fp32 sums). The latent all-reduce + RMSNorm are TAIL_LU's (the
C++ one-shot sums, ``kARRMSNorm``). Each output element is ``bf16(sum_k latent[t, k] up[c, k] + shared[t, c])``: one
rounding where the push tail (switch off) rounds the row-parallel partials and their sum (each rank's 224 latent
columns times its slice, the shared activation, a bf16 partial per rank, summed over the ranks) and TAIL_LU rounds the
FP8 GEMM, the shared sum and their sum. So it is not bit for bit either path; against the fp32 reference of the same
bf16 operands each element is within ``2^-8 |y| + 2^-12 sum_k |latent| |up|`` (the bf16 output rounding and the fp32
accumulation over 3584 terms), see the tests. The same bits on every rank (each column is computed once).

Precedence. Where both this switch and ``TRTLLM_K3LL_MOE_TAIL_LU`` hold for a step, this one runs (``decode_moe_lowm``).

The latent_up row slice and the latent norm's weight are taken together at build time (``K3TailColLayer.create``):
after ``decode_moe.fold_latent_norm`` the slice is of the folded weight and the norm's weight is ones, before it the
unfolded weight and the norm's own; either way the latent all-reduce + norm + slice computes the layer's function.
The target builds it after the fold, so the operands are the push tail's.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream

from . import flags as _flags
from . import moe_tail_lu as _tail_lu
from .moe_tail_lu import HIDDEN, LATENT, MAX_ROWS, MIN_ROWS, SHARED_SITE, K3TailLuState

#: Weight rows of one row tile of the column GEMV (``k3_ctm_gemv_wide``'s row tiles: one cluster each): TP16's 448
#: columns are 7 tiles. A slice that is not a multiple of it takes 128-row tiles.
TILE_ROWS = 64


def _ctm_op():
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op

    return op


def _ctm_kernel():
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import k3_ctm_gemv_kernel

    return k3_ctm_gemv_kernel


def _latent_ar():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_latent_ar_norm

    return k3_latent_ar_norm


def _attn_res():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    return k3_attn_res


def _lamport():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_lamport_ar

    return k3_lamport_ar


#: ``TRTLLM_K3LL_MOE_TAIL_COL``'s default (-1) turns it on only on these compute capabilities (measured on GB300 /
#: SM103). Its kernels (``k3_ctm_gemv_wide``'s column publish, ``K3LatentArRowKernel``) run on SM100 / SM103 only;
#: elsewhere the load builds no state and the layer keeps the push tail, even when forced with 1.
MOE_TAIL_COL_CAPABILITIES: Tuple[Tuple[int, int], ...] = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _on_by_default() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in MOE_TAIL_COL_CAPABILITIES
    )


def enabled() -> bool:
    """``TRTLLM_K3LL_MOE_TAIL_COL`` (read at load for the state, at forward time for engagement): 1 on, 0 off, -1
    (default) on where the device is in ``MOE_TAIL_COL_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_TAIL_COL)
    if value >= 0:
        return value > 0
    return _on_by_default()


def slice_width(world: int) -> int:
    """Output columns per rank."""
    return HIDDEN // world


def row_tiles(width: int) -> int:
    """The column GEMV's row tiles for a slice of ``width`` columns: tiles of `TILE_ROWS`, else 0 (128-row tiles)."""
    return width // TILE_ROWS if width % TILE_ROWS == 0 else 0


class ColTail(NamedTuple):
    """A wide decode step's MoE output handed on by ``TRTLLM_K3LL_MOE_TAIL_COL``: the ``[rows, 7168]`` bf16 result the
    column publish left in the broadcast stage of the decode workspace ``lamport`` (a ``k3_lamport_ar.
    LamportWorkspace``), the same on every rank. It lives until the next call on that workspace: the consumer reads it
    (``attn_res``) or copies it out (``materialize``) before anything else uses the workspace. ``candidates`` /
    ``mx_candidates``: the candidate counts (snapshots + 1) the Lamport attn_res kernel ran for at load, without / with
    the MXFP8 side output."""

    lamport: object
    rows: int
    candidates: frozenset = frozenset()
    mx_candidates: frozenset = frozenset()

    @property
    def lamport_addend(self) -> Tuple[int, int]:
        """``(uc_local, flags)``: the device addresses the attn_res kernel polls the rows with."""
        return self.lamport.uc_local, self.lamport.flags.data_ptr()

    def materialize(self) -> torch.Tensor:
        """The MoE output as a bf16 ``[rows, 7168]`` tensor (``K3LamportGatherKernel`` on the current stream)."""
        out = torch.empty(self.rows, HIDDEN, dtype=torch.bfloat16, device=self.lamport.flags.device)
        return _lamport().run_lamport_gather(self.lamport, out)


@dataclass(eq=False)
class K3TailColState:
    """What every MoE layer's switch shares on one device: ``base``, ``TRTLLM_K3LL_MOE_TAIL_LU``'s state (the latent
    workspace, the decode workspace's view, the side stream and its events; the same object as TAIL_LU's where both
    switches are set at load), and what the load-time warm-up ran."""

    base: K3TailLuState
    #: Token tiles (``k3_ctm_gemv_wide.wide_tile``) whose column publish ran at load.
    tiles: frozenset = frozenset()
    #: The shared all-reduce, the latent all-reduce and the publish ran at load.
    ready: bool = False
    #: Candidate counts whose Lamport attn_res kernel ran at load, without / with the MXFP8 side output.
    attn_res_candidates: frozenset = field(default_factory=frozenset)
    attn_res_mx_candidates: frozenset = field(default_factory=frozenset)


def layer_gaps(moe: nn.Module, tp_size: int) -> list:
    """Why the switch cannot serve MoE layer ``moe`` at TP ``tp_size`` (empty when it can): TAIL_LU's gaps (the
    latent_up, latent norm, shared down and latent all-reduce shapes), a slice of whole 16-byte vectors, and a column
    GEMV config for it at both token tiles."""
    gaps = _tail_lu.layer_gaps(moe, tp_size)
    if gaps:
        return gaps
    if HIDDEN % tp_size or slice_width(tp_size) % 8:
        return [f"TP {tp_size}: {HIDDEN} columns do not split into slices of whole 16-byte vectors"]
    width = slice_width(tp_size)
    sms = torch.cuda.get_device_properties(
        moe.routed_expert_up_proj.weight.device
    ).multi_processor_count
    for rows in (MIN_ROWS, MAX_ROWS):
        if _ctm_op().col_publish_config(width, LATENT, rows, sms, row_tiles(width), HIDDEN) is None:
            return [f"no k3_ctm_gemv_wide config for a [{width}, {LATENT}] slice at {rows} rows"]
    return []


@dataclass(eq=False)
class K3TailColLayer:
    """One MoE layer's switch weights: this rank's latent_up row slice ``up_slice`` (bf16 ``[W, 3584]``, output
    columns ``[col0, col0 + W)``), the latent norm's weight and epsilon taken at the same point, and the shared
    expert's bf16 down weight (the module's own, ``[7168, 384]`` per rank)."""

    up_slice: torch.Tensor
    col0: int
    tiles: int
    gamma: torch.Tensor
    eps: float
    shared_down: torch.Tensor

    @classmethod
    def create(cls, moe: nn.Module, tp_rank: int, tp_size: int) -> "K3TailColLayer":
        """Build MoE layer ``moe``'s switch weights (``layer_gaps`` empty), before or after
        ``decode_moe.fold_latent_norm`` (module docstring)."""
        width = slice_width(tp_size)
        col0 = tp_rank * width
        with torch.no_grad():
            up = moe.routed_expert_up_proj.weight.detach()
            norm = moe.routed_expert_norm
            return cls(
                up_slice=up[col0 : col0 + width].clone().contiguous(),
                col0=col0,
                tiles=row_tiles(width),
                gamma=norm.weight.detach().clone().contiguous(),
                eps=float(norm.variance_epsilon),
                shared_down=moe.shared_experts.down_proj.weight.detach(),
            )

    @property
    def extra_bytes(self) -> int:
        """Device memory this layer adds: the latent_up slice and the norm weight's copy."""
        return sum(t.numel() * t.element_size() for t in (self.up_slice, self.gamma))


def takes(
    state: Optional[K3TailColState],
    layer: Optional[K3TailColLayer],
    rows: int,
    act_width: int,
    gemvs,
) -> bool:
    """Whether `run` takes a wide step of ``rows`` tokens whose front build produces a shared activation ``act_width``
    wide (decided before the front runs): the switch is set, the load-time state exists and ran the step's kernels
    (the publish at the step's token tile), 9 to 32 tokens, the shared site's push takes the rows, the latent
    workspace's buffer holds the one-shot latent all-reduce and the decode workspace's the two-shot shared all-reduce
    (whose broadcast stage, the same size, holds the publish). Shapes, the switch and load-time state only."""
    if not enabled() or state is None or layer is None or gemvs is None or not state.ready:
        return False
    if not MIN_ROWS <= rows <= MAX_ROWS or _ctm_kernel().wide_tile(rows) not in state.tiles:
        return False
    latent, push = state.base.latent, state.base.push
    if rows > _latent_ar().ROW_MAX_M or rows * latent.world * LATENT * 2 > latent.bytes_per_buffer:
        return False
    if _lamport().required_bytes(rows, HIDDEN, push.world, False) > push.bytes_per_buffer:
        return False
    return gemvs.takes_push_rows(SHARED_SITE, rows, act_width, layer.shared_down, push.world)


def latent_all_reduce(
    base: K3TailLuState, layer: K3TailColLayer, routed: torch.Tensor
) -> torch.Tensor:
    """RMSNorm(gamma) of the sum over the ranks of ``routed`` (bf16 ``[T, 3584]``) on the latent workspace
    (``K3LatentArRowKernel``, TAIL_LU's kernel without the MXFP8 form)."""
    routed = routed.contiguous()
    latent = torch.empty_like(routed)
    view = base.latent
    _latent_ar().run_k3_latent_ar_norm_row(
        routed, layer.gamma, layer.eps, latent, uc_addr=view.uc_local, mc_addr=view.mc_base, flags=view.flags,
        rank=view.rank, world_size=view.world, early_trigger=True,
    )  # fmt: skip
    return latent


def publish(
    state: K3TailColState, layer: K3TailColLayer, latent: torch.Tensor, shared: torch.Tensor
) -> ColTail:
    """``bf16(latent @ up_slice.T + shared[:, col0:col0 + W])`` published as this rank's columns of the next call on
    the decode workspace (every rank must make the call); returns the `ColTail` naming the assembled rows."""
    workspace = state.base.push
    _ctm_op().run_k3_ctm_gemv_wide_col_publish(
        latent, layer.up_slice, shared, layer.col0, workspace, layer.tiles
    )
    return ColTail(
        workspace, latent.shape[0], state.attn_res_candidates, state.attn_res_mx_candidates
    )


def run(
    state: K3TailColState,
    layer: K3TailColLayer,
    gemvs,
    shared_act: torch.Tensor,
    routed_fn: Callable[[], torch.Tensor],
) -> ColTail:
    """The tail of a step `takes` holds for, after the front: the shared expert's down projection and all-reduce on
    the side stream (in order on the current stream outside multi-stream capture), ``routed_fn()`` (this rank's routed
    experts' latent partial) and the latent all-reduce + RMSNorm on the current stream, which then joins the side
    stream and publishes its columns."""
    base = state.base
    rows = shared_act.shape[0]
    shared = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=shared_act.device)
    side = do_multi_stream() and not torch.compiler.is_compiling()
    if side:
        main = torch.cuda.current_stream()
        base.fork.record(main)
        with torch.cuda.stream(base.stream):
            base.stream.wait_event(base.fork)
            _tail_lu.shared_all_reduce(base, gemvs, layer.shared_down, shared_act, shared)
            base.join.record(base.stream)
    else:
        _tail_lu.shared_all_reduce(base, gemvs, layer.shared_down, shared_act, shared)
    latent = latent_all_reduce(base, layer, routed_fn())
    if side:
        main.wait_event(base.join)
    return publish(state, layer, latent, shared)


def warm_up(
    state: K3TailColState,
    layer: K3TailColLayer,
    gemvs,
    max_candidates: int,
    mxfp8: bool = False,
) -> None:
    """Run every kernel of the switch once before any capture, whatever the switch says at forward time. Collective:
    every rank makes the same calls (shared all-reduces, one latent all-reduce and one publish per token tile on
    zeros, advancing both workspaces alike). ``mxfp8``: also the Lamport attn_res kernel with the MXFP8 side output
    (``TRTLLM_K3LL_ATTN_RES_MXFP8`` on this device at load). Records what ran in ``state``."""
    base = state.base
    device = layer.up_slice.device
    zeros = dict(dtype=torch.bfloat16, device=device)
    tiles = gemvs.compile_push(SHARED_SITE, layer.shared_down, base.push.world)
    if tiles != [16, 32]:
        torch.cuda.synchronize(device)
        return
    weight = torch.zeros_like(layer.shared_down)
    for rows in (16, 32):
        act = torch.zeros(rows, weight.shape[1], **zeros)
        # The plain wide GEMV of the site (``takes_push`` asks for it under capture) and the push + finish.
        gemvs.project(SHARED_SITE, act, weight)
        _tail_lu.shared_all_reduce(base, gemvs, weight, act, torch.empty(rows, HIDDEN, **zeros))
    probe = K3TailColLayer(
        layer.up_slice, layer.col0, layer.tiles, torch.ones(LATENT, **zeros), 1e-6, weight
    )
    latent_all_reduce(base, probe, torch.zeros(MIN_ROWS, LATENT, **zeros))
    ran = []
    tail = None
    for rows in (16, 32):
        if not _ctm_op().compile_wide_col_publish(layer.up_slice, rows, layer.tiles, HIDDEN):
            continue
        tail = publish(
            state, probe, torch.zeros(rows, LATENT, **zeros), torch.zeros(rows, HIDDEN, **zeros)
        )
        ran.append(_ctm_kernel().wide_tile(rows))
    candidates, mx_candidates = [], []
    if tail is not None:
        # The consumer's kernel on the last publish's rows (row 0; every rank polls the same rows).
        ones = torch.ones(HIDDEN, **zeros)
        row = torch.zeros(1, HIDDEN, **zeros)
        for n in range(
            _attn_res().MIN_CANDIDATES, min(max_candidates, _attn_res().MAX_CANDIDATES) + 1
        ):
            block = torch.zeros(n - 1, 1, HIDDEN, **zeros)
            outs = [torch.empty_like(row), torch.empty_like(row)]
            _attn_res().run_k3_attn_res(
                row, (), block, ones, ones, ones, 1e-6, 1e-6, *outs, exact=False,
                lamport_addend=tail.lamport_addend, early_trigger=True, mnnvl_order=True,
            )  # fmt: skip
            candidates.append(n)
            if mxfp8:
                xq = torch.empty(1, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
                xs = torch.empty(1, HIDDEN // 32, dtype=torch.uint8, device=device)
                _attn_res().run_k3_attn_res(
                    row, (), block, ones, ones, ones, 1e-6, 1e-6, *outs, exact=False,
                    lamport_addend=tail.lamport_addend, early_trigger=True, mnnvl_order=True,
                    mxfp8_out=(xq, xs),
                )  # fmt: skip
                mx_candidates.append(n)
    torch.cuda.synchronize(device)
    state.tiles = frozenset(ran)
    state.attn_res_candidates = frozenset(candidates)
    state.attn_res_mx_candidates = frozenset(mx_candidates)
    state.ready = bool(ran)


def _eps(norm: nn.Module) -> float:
    eps = norm.eps if hasattr(norm, "eps") else norm.variance_epsilon
    return float(torch.finfo(norm.weight.dtype).eps if eps is None else eps)


def attn_res_takes(
    tail: ColTail,
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    res_proj: nn.Module,
    res_norm: nn.Module,
    out_norm: nn.Module,
) -> bool:
    """Whether `attn_res` takes the consumer's operands: the vectorized attn_res kernel (not
    ``TRTLLM_K3_ATTN_RES_PDL_EXACT``), run at load for the candidate count (``tail.candidates``), takes the prefix sum
    ``prefix`` (``tail.rows`` rows), the snapshots ``[S, T, 7168]`` and the weights."""
    if (
        _attn_res()._EXACT
        or block_residual.dim() != 3
        or block_residual.shape[0] + 1 not in tail.candidates
        or prefix.dim() != 2
        or prefix.shape[0] != tail.rows
    ):
        return False
    weights = (res_proj.weight.reshape(-1), res_norm.weight, out_norm.weight)
    return _attn_res().can_use_k3_attn_res(prefix, (prefix,), block_residual, *weights)


def attn_res(
    tail: ColTail,
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    res_proj: nn.Module,
    res_norm: nn.Module,
    out_norm: nn.Module,
    mxfp8_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    tap: Optional[torch.Tensor] = None,
    updated_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(normed, updated)``: ``updated = bf16(prefix + moe)`` with ``moe`` the tail's rows polled from the decode
    workspace, ``normed = out_norm(attn_res(block_residual..., updated))`` (selected with ``res_proj`` and
    ``res_norm``), in one ``K3AttnResFastKernel`` launch with the Lamport addend in the C++ MNNVL epilogue's arithmetic
    (the push tail's consumer), releasing the next kernel at its wait. ``mxfp8_out``, ``tap`` (a bf16 ``[rows,
    7168]`` view the kernel also stores the pre-norm mixture into): passed to ``run_k3_attn_res``. ``updated_out``: a
    bf16 ``[rows, 7168]`` contiguous tensor the kernel stores ``updated`` into (e.g. the consumer's snapshot bank row),
    returned as ``updated``."""
    normed = torch.empty_like(prefix)
    updated = torch.empty_like(prefix) if updated_out is None else updated_out
    _attn_res().run_k3_attn_res(
        prefix, (), block_residual, res_proj.weight.reshape(-1), res_norm.weight, out_norm.weight, _eps(res_norm),
        _eps(out_norm), normed, updated, exact=False, lamport_addend=tail.lamport_addend, early_trigger=True,
        mnnvl_order=True, mxfp8_out=mxfp8_out, tap=tap,
    )  # fmt: skip
    return normed, updated


__all__ = [
    "TILE_ROWS",
    "ColTail",
    "K3TailColLayer",
    "K3TailColState",
    "attn_res",
    "attn_res_takes",
    "enabled",
    "latent_all_reduce",
    "layer_gaps",
    "publish",
    "row_tiles",
    "run",
    "slice_width",
    "takes",
    "warm_up",
]
