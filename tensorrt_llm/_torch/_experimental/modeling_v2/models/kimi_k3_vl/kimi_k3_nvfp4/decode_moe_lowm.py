# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode path's latent MoE over NVFP4 routed experts (the CuteDSL MoE backend's buffers).

The checkpoint keeps the router, the latent projections and the shared experts in bf16 and only the routed experts
in NVFP4, so the path is `decode_moe.py`'s with the experts swapped. **At most 8 tokens** (`MAX_TOKENS`), a MoE layer
runs:

* `moe/k3_moe_front`, one kernel, with the NVFP4 latent: this rank's slice of the MoE head as one GEMV, the slices'
  all-gather over the TP group's `K3MoeHeadWorkspace`, the top-16 routing, the latent quantized to NVFP4 on the routed
  experts' input global scale (``fp4_quantize``'s recipe), and the shared experts' gate_up + SiTU;
* ``moe_sort`` of the routing over this rank's experts (128-row tiles), then `trtllm::k3_moe_nvfp4_lowm`
  (`cute_dsl_kernels/k3_fused_moe/nvfp4_lowm_op.py`): FC1 + SiTU + FC2 + the routing-weighted combine of this rank's
  experts in one persistent kernel, its output zeroed by the kernel; with ``front_sort``
  (``TRTLLM_K3LL_MOE_FRONT_SORT``) the front writes the sorted routing itself (``sort_out``) and no ``moe_sort`` runs;
* the routed experts' latent all-reduce (one-shot, as `decode_comm.use_decode_one_shot` sets);
* the tail, as `decode_moe.py`'s: a `PendingTail` the next layer's pre-attention step reduces with its all-reduce and
  residual update in one `comm/k3_sandwich_tail` kernel, else the replicated tail.

With ``TRTLLM_K3LL_ROUTING_STATS_DIR`` each layer's routed ids (the front's, or ``k3_route_quant``'s on a wide step)
are also counted into rank 0's `routing_stats.RoutingStats` (``K3DecodeMoeNvfp4Layer.stats``).

Steps of more tokens keep the generic path. Only SM100 / SM103 take it (`nvfp4_lowm_op.supported`).

With ``TRTLLM_K3LL_MOE_LOWM_FC12`` (built at load when set at launch, `K3MoeFc12`; engagement read at forward time,
`fc12_takes`) a wide step of 17-32 tokens runs the routed experts as two kernels, ``K3MoeFc1LowMKernel`` then
``K3MoeFc2LowMKernel`` (`nvfp4_lowm_op.k3_moe_nvfp4_lowm_fc12`), into a persistent FC1 output buffer. FC2 adds into
the routed output, so the step zeroes that output first: under multi-stream capture on the state's side stream,
forked from the main stream before the MoE front and joined right before FC1, so the zeroing runs beside the front
(no kernel between the front and FC1); elsewhere on the main stream right before FC1. Every consumer of the routed
output follows FC2 on the main stream, as it followed the single kernel.

`K3DecodeMoeNvfp4` holds what every MoE layer shares (the head workspace, the expert kernel's counters);
`K3DecodeMoeNvfp4Layer` holds one layer's decode weights and handles. The target builds both in
`post_load_weights` when ``TRTLLM_K3LL_MOE_LOWM`` (flags.py) is set, before any CUDA-graph capture, and runs
every kernel build the path can select once there (`K3DecodeMoeNvfp4Layer.warm_up`), so none compiles under a
capture. A switch read at forward time selects among builds warm-up has already run: every build of each of its
values, not just the one its value at load picks.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
    mnnvl_allgather_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_rmsnorm_slice import (
    fits as rmsnorm_slice_fits,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_rmsnorm_slice import (
    mnnvl_allreduce_rmsnorm_slice,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe_front import (
    K3MoeHeadWorkspace,
    front_weight,
    k3_moe_front,
    sort_layout,
    sort_views,
    weight_supported,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_route_quant import k3_route_quant
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import nvfp4_lowm_op
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)

from . import flags as _flags
from . import moe_tail_col as _tail_col
from . import moe_tail_lu as _tail_lu
from .decode_comm import K3DecodeComm, PendingTail, wide_all_reduce
from .decode_moe import TAIL_K_TILE, _gemv
from .moe_tail_col import ColTail
from .moe_tail_lu import LuTail
from .moe_tail_push import WideTail
from .routing_stats import RoutingStats

MAX_TOKENS = 8
# The front's 16-token build (TRTLLM_K3LL_MOE_FRONT16): a wide step of at most this many tokens.
FRONT16_TOKENS = 16
# The front's 32-token build (TRTLLM_K3LL_MOE_FRONT32): a wide step of more than FRONT16_TOKENS, up to this many.
FRONT32_TOKENS = 32
# A wide decode step's most tokens: the expert kernel's limit.
WIDE_MAX_TOKENS = nvfp4_lowm_op.MAX_TOKENS
NUM_EXPERTS = 896
#: CUDA priority of ``TRTLLM_K3LL_MOE_LOWM_FC12``'s zeroing stream (lower is higher): its small kernel takes SMs
#: ahead of the front's pending CTAs.
FC12_STREAM_PRIORITY = -1


@dataclass(eq=False)
class K3MoeFc12:
    """``TRTLLM_K3LL_MOE_LOWM_FC12``'s state on one device: the FC1 output buffers every layer shares (the layers run
    one at a time on one stream) and the side stream that zeroes a step's routed output, with its fork and join
    events."""

    buffers: nvfp4_lowm_op.Fc12
    stream: torch.cuda.Stream
    fork: torch.cuda.Event
    join: torch.cuda.Event

    @classmethod
    def create(cls, device, n1: int, num_local: int, top_k: int) -> "K3MoeFc12":
        """Buffers for `WIDE_MAX_TOKENS` tokens over ``num_local`` experts of W1 ``[n1, k1]``. Eagerly, before any
        CUDA-graph capture."""
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device, priority=FC12_STREAM_PRIORITY)
        return cls(
            nvfp4_lowm_op.new_fc12(device, n1, num_local, top_k, WIDE_MAX_TOKENS),
            stream,
            torch.cuda.Event(),
            torch.cuda.Event(),
        )

    def begin(
        self, rows: int, cols: int, device, fc2_ceil_chunk: Optional[bool] = None
    ) -> "Fc12Step":
        """A step's routed output, bf16 ``[rows, cols]``, being zeroed: under multi-stream capture on the side stream
        forked here (call it before the front), else not yet (``Fc12Step.launch`` zeroes it on the main stream).
        ``fc2_ceil_chunk``: `Fc12Step`'s."""
        out = torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
        side = do_multi_stream() and not torch.compiler.is_compiling()
        if side:
            main = torch.cuda.current_stream()
            self.fork.record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.fork)
                out.zero_()
                self.join.record(self.stream)
        return Fc12Step(self, out, side, fc2_ceil_chunk)


@dataclass(eq=False)
class Fc12Step:
    """One wide step's ``TRTLLM_K3LL_MOE_LOWM_FC12`` launch (`K3MoeFc12.begin`): the routed output, whether the
    side stream zeroes it and FC2's ``fc2_ceil_chunk`` (None: read ``TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK`` at launch, as
    the forward path does)."""

    state: K3MoeFc12
    out: torch.Tensor
    side: bool
    fc2_ceil_chunk: Optional[bool] = None

    def launch(self) -> nvfp4_lowm_op.Fc12:
        """`nvfp4_lowm_op.run_sorted`'s ``fc12`` for this step, called on the main stream right before the experts:
        with the side stream, the main stream joins it here (the zeroing is done before FC1 starts)."""
        if self.side:
            torch.cuda.current_stream().wait_event(self.state.join)
        buffers = self.state.buffers
        ceil_chunk = self.fc2_ceil_chunk
        if ceil_chunk is None:
            ceil_chunk = fc2_ceil_chunk_enabled()
        if ceil_chunk:
            from . import census as _census

            _census.count(f"dispatch:moe_fc2_ceil_chunk:{_census.mode()}")
            # per step size: which captured graphs (rows = batch x (1 + draft)) take it
            _census.count(f"dispatch:moe_fc2_ceil_chunk_rows{self.out.shape[0]}:{_census.mode()}")
        return nvfp4_lowm_op.Fc12(
            buffers.h, buffers.h_sf, zeroed=self.side, fc2_ceil_chunk=ceil_chunk
        )


#: Devices where ``TRTLLM_K3LL_MOE_LOWM_FC12`` is on by default (-1): measured on GB300 (SM103) only.
MOE_LOWM_FC12_CAPABILITIES = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _fc12_on_by_default() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in MOE_LOWM_FC12_CAPABILITIES
    )


def fc12_enabled() -> bool:
    """``TRTLLM_K3LL_MOE_LOWM_FC12`` (read at load for the state, at forward time for engagement): 1 on, 0 off, -1
    (default) on where the device is in ``MOE_LOWM_FC12_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_LOWM_FC12)
    if value >= 0:
        return value > 0
    return _fc12_on_by_default()


#: Devices where ``TRTLLM_K3LL_MOE_SELF_SORT`` is on by default (-1): measured on GB300 (SM103) only.
MOE_SELF_SORT_CAPABILITIES = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _self_sort_on_by_default() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in MOE_SELF_SORT_CAPABILITIES
    )


def self_sort_enabled() -> bool:
    """``TRTLLM_K3LL_MOE_SELF_SORT`` (read at forward time): 1 on, 0 off, -1 (default) on where the device is in
    ``MOE_SELF_SORT_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_SELF_SORT)
    if value >= 0:
        return value > 0
    return _self_sort_on_by_default()


#: Devices where ``TRTLLM_K3LL_MOE_LOWM_DYN_FAST`` is on by default (-1): measured on GB300 (SM103) only.
MOE_LOWM_DYN_FAST_CAPABILITIES = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _lowm_dyn_fast_on_by_default() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in MOE_LOWM_DYN_FAST_CAPABILITIES
    )


def lowm_dyn_fast() -> bool:
    """``TRTLLM_K3LL_MOE_LOWM_DYN_FAST`` (read at forward time): `nvfp4_lowm_op.run`'s ``fast``. 1 on, 0 off, -1
    (default) on where the device is in ``MOE_LOWM_DYN_FAST_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_LOWM_DYN_FAST)
    if value >= 0:
        return value > 0
    return _lowm_dyn_fast_on_by_default()


def self_sort_fc12_enabled() -> bool:
    """``TRTLLM_K3LL_MOE_SELF_SORT_FC12`` (read at forward time): 1 on, 0 off, -1 (default) on where the device is in
    ``MOE_SELF_SORT_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_SELF_SORT_FC12)
    if value >= 0:
        return value > 0
    return _self_sort_on_by_default()


#: Devices where ``TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK`` is on by default (-1): measured on GB300 (SM103) only.
MOE_FC2_CEIL_CHUNK_CAPABILITIES = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _fc2_ceil_chunk_on_by_default() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in MOE_FC2_CEIL_CHUNK_CAPABILITIES
    )


def fc2_ceil_chunk_enabled() -> bool:
    """``TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK`` (read at forward time): 1 on, 0 off, -1 on where the device is in
    ``MOE_FC2_CEIL_CHUNK_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK)
    if value >= 0:
        return value > 0
    return _fc2_ceil_chunk_on_by_default()


def fc12_takes(state, rows: int) -> bool:
    """Whether a wide step of ``rows`` tokens runs the routed experts as FC1 + FC2: ``TRTLLM_K3LL_MOE_LOWM_FC12``
    (read at forward time) is set, ``rows`` is 17 to 32 (``nvfp4_lowm_op.FC12_MIN_TOKENS`` / ``FC12_MAX_TOKENS``;
    at most 16 rows keep ``K3MoeLowMDynKernel``), and the load built ``state.fc12``. The switch and shapes only, so
    every rank decides alike."""
    return (
        fc12_enabled()
        and nvfp4_lowm_op.FC12_MIN_TOKENS <= rows <= nvfp4_lowm_op.FC12_MAX_TOKENS
        and state.fc12 is not None
    )


_PRESORTED_LOGGED = False


def _log_presorted(rows: int) -> None:
    """One line per process where ``TRTLLM_K3LL_MOE_PRESORTED`` first engages (an A/B's proof the arm runs it)."""
    global _PRESORTED_LOGGED
    _PRESORTED_LOGGED = True
    from tensorrt_llm.logger import logger

    logger.info(f"K3 decode MoE: TRTLLM_K3LL_MOE_PRESORTED engaged (first step: {rows} tokens)")


@dataclass(eq=False)
class K3DecodeMoeNvfp4:
    """What every MoE layer's NVFP4 decode path shares on one device: the TP group's ``K3MoeHeadWorkspace`` (the
    front's all-gather), the expert kernel's counter buffer and, with the front's sort, its counter word (the layers
    run one at a time on one stream)."""

    head: K3MoeHeadWorkspace
    counters: torch.Tensor
    mnnvl: MnnvlWorkspace
    sort_ctr: Optional[torch.Tensor] = None
    #: The 16-token front build's own head workspace (a step of 9-16 tokens), or None: such steps keep the wide head.
    head16: Optional[K3MoeHeadWorkspace] = None
    #: The 32-token front build's own head workspace (a step of 17-32 tokens), or None: such steps keep the wide head.
    head32: Optional[K3MoeHeadWorkspace] = None
    #: ``TRTLLM_K3LL_MOE_TAIL_LU``'s shared state (``moe_tail_lu.py``), built where the switch is set at load.
    lu: Optional[_tail_lu.K3TailLuState] = None
    #: ``TRTLLM_K3LL_MOE_LOWM_FC12``'s state (`K3MoeFc12`), built by the first layer's ``warm_up`` where the switch is
    #: set at launch.
    fc12: Optional[K3MoeFc12] = None
    #: ``TRTLLM_K3LL_MOE_TAIL_COL``'s shared state (``moe_tail_col.py``), built where the switch is set at load.
    col: Optional[_tail_col.K3TailColState] = None
    #: ``TRTLLM_K3LL_MOE_PRESORTED``'s FC1 token-scale image (zero where no scale lands), with the front's sort.
    presorted_image: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        mapping,
        device,
        mnnvl: MnnvlWorkspace,
        front_sort: bool = False,
        front16: bool = False,
        front32: bool = False,
    ) -> "K3DecodeMoeNvfp4":
        """Collective (the head workspace): every rank of the TP group calls it at the same point, eagerly, before
        any CUDA-graph capture. ``mnnvl``: the TP group's MNNVL workspace, for a wide step's head all-gather.
        ``front_sort``: the front sorts the routing over the local experts (``k3_moe_front``'s ``sort_out``).
        ``front16``: steps of 9-16 tokens run the front's 16-token build (its own head workspace).
        ``front32``: steps of 17-32 tokens run the front's 32-token build (its own head workspace)."""
        sort_ctr = torch.zeros(1, dtype=torch.int32, device=device) if front_sort else None
        head16 = K3MoeHeadWorkspace.create(mapping, m_max=FRONT16_TOKENS) if front16 else None
        head32 = K3MoeHeadWorkspace.create(mapping, m_max=FRONT32_TOKENS) if front32 else None
        state = cls(
            K3MoeHeadWorkspace.create(mapping),
            nvfp4_lowm_op.new_counters(device),
            mnnvl,
            sort_ctr,
            head16,
            head32,
        )
        if front_sort and front32:
            from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import (
                k3_moe_front as _front_kernel,
            )

            # one image for every layer: a layer's front writes it only after every earlier kernel has completed
            # (each kernel's grid-dependency wait), so the previous layer's FC1 has read it
            state.presorted_image = torch.zeros(
                _front_kernel.IMAGE_BYTES, dtype=torch.uint8, device=device
            )
        return state

    def front_head(self, rows: int, front32: Optional[bool] = None) -> Optional[K3MoeHeadWorkspace]:
        """The head workspace of the front build that runs a step of ``rows`` tokens, or None: the wide head runs
        it. ``front32``: whether a step of 17-32 tokens takes the 32-token build (None: ``TRTLLM_K3LL_MOE_FRONT32``,
        read at forward time, so a one-process A/B can switch it; the build's workspace exists whenever the layer
        supports it)."""
        if rows <= MAX_TOKENS:
            return self.head
        if rows <= FRONT16_TOKENS:
            return self.head16
        if rows > FRONT32_TOKENS:
            return None
        if front32 is None:
            front32 = bool(_flags.TRTLLM_K3LL_MOE_FRONT32)
        return self.head32 if front32 else None


def _front_supported(moe: nn.Module, tp_size: int, m_max: int) -> bool:
    gate_up = moe.shared_experts.gate_up_proj.weight
    return weight_supported(
        tp_size, gate_up.shape[0] // 2, gate_up.shape[1], gate_up.device, m_max=m_max
    )


def front16_supported(moe: nn.Module, tp_size: int) -> bool:
    """Whether the front's 16-token build runs MoE layer ``moe`` (whose 8-token path ``layout_gaps`` takes)."""
    return _front_supported(moe, tp_size, FRONT16_TOKENS)


def front32_supported(moe: nn.Module, tp_size: int) -> bool:
    """Whether the front's 32-token build runs MoE layer ``moe`` (whose 8-token path ``layout_gaps`` takes)."""
    return _front_supported(moe, tp_size, FRONT32_TOKENS)


def _experts(backend) -> tuple:
    """The routed experts' NVFP4 buffers and global scales the expert kernel reads in place, in
    ``nvfp4_lowm_op.run``'s order."""
    scales = backend.quant_scales
    return (
        backend.w3_w1_weight,
        scales.fc1_weight_block,
        scales.fc1_global,
        backend.w2_weight,
        scales.fc2_weight_block,
        scales.fc2_global,
    )


def _scalar(value) -> float:
    """A SiTU constant of the backend (a float, or a tensor holding one value per expert, all equal) as a float."""
    if not torch.is_tensor(value):
        return float(value)
    flat = value.detach().reshape(-1).float()
    if not bool((flat == flat[0]).all()):
        raise ValueError("the NVFP4 decode MoE takes one SiTU constant for every expert")
    return float(flat[0])


def layout_gaps(moe: nn.Module, tp_size: int, max_snapshots: int) -> list:
    """Why this decode path does not take MoE layer ``moe`` (empty when it takes it), read once the routed experts'
    buffers exist: the checks in order, up to the first that fails."""
    gate, backend = moe.gate, moe.routed_experts.backend
    shared = moe.shared_experts
    gate_up, down = shared.gate_up_proj.weight, shared.down_proj.weight
    if not moe._reduce_routed_output:
        return ["a routed output the model does not reduce"]
    if (moe.num_experts, moe.top_k, moe.moe_hidden_size, moe.hidden_size) != (
        NUM_EXPERTS,
        16,
        3584,
        7168,
    ):
        return [
            f"experts / top-k / latent / hidden {(moe.num_experts, moe.top_k, moe.moe_hidden_size)}"
        ]
    if (gate.num_expert_group, gate.topk_group) != (1, 1):
        return ["grouped routing"]
    if moe.moe_hidden_size % (8 * tp_size) or moe.num_experts % (4 * tp_size):
        return [f"a head slice of TP {tp_size}"]
    latent = (gate.weight, moe.routed_expert_down_proj.weight, moe.routed_expert_up_proj.weight)
    if not isinstance(moe.routed_expert_up_proj, nn.Linear) or any(
        w.dtype != torch.bfloat16 for w in latent
    ):
        return ["latent projections or a router other than bf16"]
    if (
        gate_up.dtype != torch.bfloat16
        or down.dtype != torch.bfloat16
        or shared.gate_up_proj.bias is not None
        or shared.down_proj.bias is not None
        or down.shape[0] != moe.hidden_size
    ):
        return ["a shared expert other than bf16 and unbiased"]
    names = ("w3_w1_weight", "w2_weight", "quant_scales", "fc31_input_scale", "fc2_input_scale", "slot_start",
             "act_alpha", "act_beta")  # fmt: skip
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE

    # The expert kernel reads the CuteDSL backend's buffers; another backend (TRTLLM-Gen: other layouts, quant_scales
    # a tuple) keeps the generic path.
    if not isinstance(backend, CuteDslFusedMoE):
        return [f"routed experts on {type(backend).__name__}, not the CuteDSL backend"]
    scale_fields = ("fc1_weight_block", "fc1_global", "fc2_weight_block", "fc2_global")
    if (
        not getattr(backend, "has_nvfp4", False)
        or not all(hasattr(backend, name) for name in names)
        or not all(hasattr(backend.quant_scales, f) for f in scale_fields)
    ):
        return ["routed experts other than the CuteDSL NVFP4 backend"]
    if getattr(backend, "scaling_vector_size", 16) != 16:
        return ["NVFP4 experts with a scale group other than 16"]
    if any(
        t is None or t.numel() != 1 or t.dtype != torch.float32
        for t in (backend.fc31_input_scale, backend.fc2_input_scale)
    ):
        return ["NVFP4 experts without one fp32 input / FC2 global scale each"]
    w1 = backend.w3_w1_weight
    ok, why = nvfp4_lowm_op.supported(w1.device, w1.shape[1], backend.w2_weight.shape[1])
    if not ok:
        return [f"the NVFP4 expert kernel: {why}"]
    shared_cols, width = gate_up.shape[0] // 2, moe.moe_hidden_size // tp_size
    if not weight_supported(tp_size, shared_cols, gate_up.shape[1], gate_up.device):
        return ["a MoE front of this TP size and shared width"]
    tail_cols = width + (-width % TAIL_K_TILE) + shared_cols
    probe = gate_up.new_empty
    if not K3DecodeComm.takes_tail(
        probe(1, moe.moe_hidden_size),
        probe(1, shared_cols),
        probe(moe.hidden_size, tail_cols),
        max_snapshots,
    ):
        return ["a row-parallel tail the sandwich tail kernel does not take"]
    return []


@dataclass(eq=False)
class K3DecodeMoeNvfp4Layer:
    """One MoE layer's NVFP4 decode path: the front weight (this rank's head slice padded to whole tiles, then the
    shared gate_up re-ordered), the row-parallel tail weight ``[latent up columns lo:lo+width | padding | shared
    down]``, the routing bias, and the routed experts' buffers and global scales. Built by `create` once the weights
    are final (and the latent norm folded)."""

    state: K3DecodeMoeNvfp4
    front_weight: torch.Tensor
    head_weight: torch.Tensor
    tail_weight: torch.Tensor
    tail_pad: Optional[torch.Tensor]
    bias: torch.Tensor
    lo: int
    width: int
    shared_cols: int
    experts: tuple
    input_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    offset: int
    situ: tuple
    #: ``TRTLLM_K3LL_MOE_TAIL_LU``'s weights of this layer (``moe_tail_lu.py``), built where the switch is set at load.
    lu: Optional[_tail_lu.K3TailLuLayer] = None
    #: ``TRTLLM_K3LL_MOE_TAIL_COL``'s weights of this layer (``moe_tail_col.py``), built where the switch is set at
    #: load.
    col: Optional[_tail_col.K3TailColLayer] = None
    #: ``TRTLLM_K3LL_ROUTING_STATS_DIR``: the recorder this layer's routing is counted into (rank 0), else None.
    stats: Optional[RoutingStats] = None
    layer_idx: int = -1

    @classmethod
    def create(
        cls, moe: nn.Module, state: K3DecodeMoeNvfp4, tp_rank: int, tp_size: int
    ) -> "K3DecodeMoeNvfp4Layer":
        """Build MoE layer ``moe``'s decode weights (``layout_gaps`` empty)."""
        width = moe.moe_hidden_size // tp_size
        experts = moe.num_experts // tp_size
        gate_up = moe.shared_experts.gate_up_proj.weight
        shared_down = moe.shared_experts.down_proj.weight
        up = moe.routed_expert_up_proj.weight
        backend = moe.routed_experts.backend
        lo = tp_rank * width
        pad = -width % TAIL_K_TILE
        with torch.no_grad():
            head = torch.cat(
                [
                    moe.routed_expert_down_proj.weight[lo : lo + width],
                    moe.gate.weight[tp_rank * experts : (tp_rank + 1) * experts],
                ]
            )
            front = front_weight(head, gate_up)
            parts = [up[:, lo : lo + width]]
            if pad:
                parts.append(up.new_zeros(up.shape[0], pad))
            parts.append(shared_down)
            tail = torch.cat(parts, dim=1).contiguous()
        return cls(
            state=state,
            front_weight=front,
            head_weight=front[: head.shape[0]],
            tail_weight=tail,
            tail_pad=up.new_zeros(WIDE_MAX_TOKENS, pad) if pad else None,
            bias=moe.gate.e_score_correction_bias.detach(),
            lo=lo,
            width=width,
            shared_cols=gate_up.shape[0] // 2,
            experts=_experts(backend),
            input_scale=backend.fc31_input_scale.detach().reshape(1),
            fc2_input_scale=backend.fc2_input_scale.detach().reshape(1),
            offset=int(backend.slot_start),
            situ=(_scalar(backend.act_alpha), _scalar(backend.act_beta)),
            layer_idx=getattr(moe, "layer_idx", -1),
        )

    def warm_up(self, moe: nn.Module) -> None:
        """One call of every kernel build the path can select, on zero inputs (M = 1), so none compiles under a
        capture, whatever the switches read at forward time say then: the front (collective: every rank makes the
        same call), ``k3_route_quant`` and the expert kernel's builds for 8, 16 and 32 tokens, each with and without
        the expert kernel's own sort and ``TRTLLM_K3LL_MOE_LOWM_DYN_FAST`` (both read at forward time). With
        ``TRTLLM_K3LL_MOE_LOWM_FC12`` set at launch, also builds the shared `K3MoeFc12` and runs the FC1 + FC2 launch
        at 17 and 32 tokens (its 32-token build), each with and without ``TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK`` (read at
        forward time)."""
        device = self.front_weight.device
        x = torch.zeros(1, moe.hidden_size, dtype=torch.bfloat16, device=device)
        # Both sorts (the front's or moe_sort, and the expert kernel's own, TRTLLM_K3LL_MOE_SELF_SORT) and both
        # expert-kernel builds of TRTLLM_K3LL_MOE_LOWM_DYN_FAST, so either switch can change at forward time.
        for fast in (False, True):
            for self_sort in (False, True):
                self._front_routed(moe, x, self_sort=self_sort, fast=fast)
                if self.state.head16 is not None:
                    self._front_routed(
                        moe, x.new_zeros(MAX_TOKENS + 1, x.shape[1]), self_sort=self_sort, fast=fast
                    )
            if self.state.head32 is not None:
                self._front_routed(
                    moe, x.new_zeros(FRONT16_TOKENS + 1, x.shape[1]), front32=True, fast=fast
                )
            for rows in (9, WIDE_MAX_TOKENS):
                logits = torch.zeros(rows, moe.num_experts, dtype=torch.float32, device=device)
                latent = torch.zeros(rows, moe.moe_hidden_size, dtype=torch.bfloat16, device=device)
                for self_sort in (False, True):
                    self._routed(
                        *self._route_quant(moe, logits, latent), self_sort=self_sort, fast=fast
                    )
        if fc12_enabled() and self.state.fc12 is None:
            w1 = self.experts[0]
            self.state.fc12 = K3MoeFc12.create(device, w1.shape[1], self.num_local, moe.top_k)
        if self.state.fc12 is not None:
            cols = self.experts[3].shape[1]
            # both FC2 builds of TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK, so the switch can change at forward time
            for ceil_chunk in (False, True):
                for rows in (nvfp4_lowm_op.FC12_MIN_TOKENS, nvfp4_lowm_op.FC12_MAX_TOKENS):
                    logits = torch.zeros(rows, moe.num_experts, dtype=torch.float32, device=device)
                    latent = torch.zeros(
                        rows, moe.moe_hidden_size, dtype=torch.bfloat16, device=device
                    )
                    # both sorts (moe_sort, and FC1 / FC2's own, TRTLLM_K3LL_MOE_SELF_SORT_FC12)
                    for self_sort in (False, True):
                        step = self.state.fc12.begin(rows, cols, device, fc2_ceil_chunk=ceil_chunk)
                        self._routed(
                            *self._route_quant(moe, logits, latent), fc12=step, self_sort=self_sort
                        )
                if self.state.head32 is not None:
                    # the 32-token front without its sort (the FC1 / FC2 self-sort's front)
                    rows = FRONT16_TOKENS + 1
                    step = self.state.fc12.begin(rows, cols, device, fc2_ceil_chunk=ceil_chunk)
                    self._front_routed(
                        moe, x.new_zeros(rows, x.shape[1]), front32=True, fc12=step, self_sort=True
                    )
        torch.cuda.synchronize(device)

    def _route_quant(self, moe: nn.Module, logits: torch.Tensor, latent: torch.Tensor):
        return k3_route_quant(
            logits, self.bias, latent.contiguous(), float(moe.gate.routed_scaling_factor),
            nvfp4_gsf=self.input_scale,
        )  # fmt: skip

    def _front(
        self,
        moe: nn.Module,
        x: torch.Tensor,
        sort_out: Optional[torch.Tensor] = None,
        front32: Optional[bool] = None,
        presorted: bool = False,
    ):
        head = self.state.front_head(x.shape[0], front32)
        return k3_moe_front(
            x.contiguous(),
            self.front_weight,
            self.bias,
            float(moe.gate.routed_scaling_factor),
            self.shared_cols,
            *moe._situ_betas,
            head,
            nvfp4_gsf=self.input_scale,
            sort_out=sort_out,
            sort_ctr=self.state.sort_ctr if sort_out is not None else None,
            sort_offset=self.offset,
            sort_local=self.num_local,
            sf_image=self.state.presorted_image if presorted else None,
            sort_weights=not presorted,
            sort_fence=not presorted,
        )

    def presorted_takes(self, rows: int, fc12: bool) -> bool:
        """``TRTLLM_K3LL_MOE_PRESORTED`` (read at forward time) on a wide FC1 + FC2 step of 17-32 tokens with the
        front's sort and the 32-token build's image."""
        return (
            fc12
            and int(_flags.TRTLLM_K3LL_MOE_PRESORTED) > 0
            and self.state.sort_ctr is not None
            and self.state.presorted_image is not None
            and FRONT16_TOKENS < rows <= FRONT32_TOKENS
        )

    @property
    def num_local(self) -> int:
        """This rank's routed experts."""
        return self.experts[0].shape[0]

    def self_sort_takes(
        self, rows: int, self_sort: Optional[bool] = None, fc12: bool = False
    ) -> bool:
        """Whether the expert kernel sorts a step of ``rows`` tokens itself (``TRTLLM_K3LL_MOE_SELF_SORT``, or with
        ``fc12`` FC1 / FC2 each per CTA, ``TRTLLM_K3LL_MOE_SELF_SORT_FC12``; read at forward time unless
        ``self_sort`` says): then neither the front nor ``moe_sort`` sorts it."""
        if self_sort is None:
            self_sort = self_sort_fc12_enabled() if fc12 else self_sort_enabled()
        return self_sort and nvfp4_lowm_op.self_sort_takes(rows, self.num_local, fc12=fc12)

    def _routed(
        self,
        ids,
        weights,
        x_q,
        x_sf,
        fc12: Optional[Fc12Step] = None,
        self_sort: Optional[bool] = None,
        fast: Optional[bool] = None,
    ) -> torch.Tensor:
        """``moe_sort`` and the routed experts (the expert kernel's own sort where `self_sort_takes`); ``fc12``: as
        two kernels into ``fc12.out`` (`Fc12Step`). ``fast``: the single kernel's `lowm_dyn_fast` (None: read the
        switch, as the forward path does)."""
        if fc12 is not None:
            self_sort = self.self_sort_takes(ids.shape[0], self_sort, fc12=True)
            if self_sort:
                from . import census as _census

                _census.count(f"dispatch:moe_self_sort_fc12:{_census.mode()}")
            return nvfp4_lowm_op.run(
                x_q, x_sf, ids, weights, self.experts, self.offset, NUM_EXPERTS, self.fc2_input_scale,
                self.state.counters, *self.situ, out=fc12.out, fc12=fc12.launch(), self_sort=self_sort,
            )  # fmt: skip
        self_sort = self.self_sort_takes(ids.shape[0], self_sort)
        fast = self._dyn_fast(fast)
        if self_sort:
            from . import census as _census

            _census.count(f"dispatch:moe_self_sort:{_census.mode()}")
        return nvfp4_lowm_op.run(
            x_q, x_sf, ids, weights, self.experts, self.offset, NUM_EXPERTS, self.fc2_input_scale,
            self.state.counters, *self.situ, self_sort=self_sort, fast=fast,
        )  # fmt: skip

    @staticmethod
    def _dyn_fast(fast: Optional[bool] = None) -> bool:
        """`lowm_dyn_fast` (or ``fast`` where given), counted (``dispatch:moe_lowm_dyn_fast``) where it engages: the
        single expert kernel."""
        if fast is None:
            fast = lowm_dyn_fast()
        if fast:
            from . import census as _census

            _census.count(f"dispatch:moe_lowm_dyn_fast:{_census.mode()}")
        return fast

    def _front_routed(
        self,
        moe: nn.Module,
        x: torch.Tensor,
        front32: Optional[bool] = None,
        fc12: Optional[Fc12Step] = None,
        self_sort: Optional[bool] = None,
        fast: Optional[bool] = None,
    ):
        """The front and this rank's routed partial of ``x`` (at most `MAX_TOKENS` rows, or a wide step a front build
        takes, ``K3DecodeMoeNvfp4.front_head``): ``(routed, shared_act)``.
        With the front's sort the expert kernel reads the front's sorted routing; else ``moe_sort`` sorts it.
        ``fc12``: `_routed_after`'s."""
        routing, shared_act = self._front_only(moe, x, front32, self_sort, fc12=fc12 is not None)
        return self._routed_after(routing, fc12=fc12, self_sort=self_sort, fast=fast), shared_act

    def _front_only(
        self,
        moe: nn.Module,
        x: torch.Tensor,
        front32: Optional[bool] = None,
        self_sort: Optional[bool] = None,
        fc12: bool = False,
    ):
        """``_front_routed``'s front: ``(routing, shared_act)``, ``routing`` what `_routed_after` takes. Where
        `self_sort_takes` the front does not sort (the expert kernel does); ``fc12``: the experts run as FC1 + FC2."""
        rows = x.shape[0]
        if (
            self.presorted_takes(rows, fc12)
            and self.state.front_head(rows, front32) is self.state.head32
        ):
            from . import census as _census

            _census.count(f"dispatch:moe_presorted:{_census.mode()}")
            if not _PRESORTED_LOGGED:
                _log_presorted(rows)
            sort_out = torch.empty(
                sort_layout(rows, self.num_local)[3], dtype=torch.int32, device=x.device
            )
            ids, weights, x_q, _, shared_act = self._front(
                moe, x, sort_out, front32, presorted=True
            )
            if self.stats is not None:
                self.stats.record(self.layer_idx, ids)
            # the bf16 weights in the routing's third field, the image in place of the linear scales
            return (sort_out, rows, weights, x_q, self.state.presorted_image), shared_act
        if self.state.sort_ctr is None or self.self_sort_takes(x.shape[0], self_sort, fc12=fc12):
            ids, weights, x_q, x_sf, shared_act = self._front(moe, x, front32=front32)
            if self.stats is not None:
                self.stats.record(self.layer_idx, ids)
            return (None, ids, weights, x_q, x_sf), shared_act
        rows = x.shape[0]
        sort_out = torch.empty(
            sort_layout(rows, self.num_local)[3], dtype=torch.int32, device=x.device
        )
        ids, _, x_q, x_sf, shared_act = self._front(moe, x, sort_out, front32)
        if self.stats is not None:
            self.stats.record(self.layer_idx, ids)
        return (sort_out, rows, None, x_q, x_sf), shared_act

    def _routed_after(
        self,
        routing,
        fc12: Optional[Fc12Step] = None,
        self_sort: Optional[bool] = None,
        fast: Optional[bool] = None,
    ) -> torch.Tensor:
        """``_front_routed``'s routed experts on the front's ``routing`` (`_front_only`). ``fc12``
        (``TRTLLM_K3LL_MOE_LOWM_FC12``, `K3MoeFc12.begin`): FC1 + FC2 into ``fc12.out`` instead of the single
        kernel."""
        sort_out, a, b, x_q, x_sf = routing
        if sort_out is None:
            return self._routed(a, b, x_q, x_sf, fc12=fc12, self_sort=self_sort, fast=fast)
        if fc12 is not None:
            views = sort_views(sort_out, a, self.num_local)
            if (
                b is not None
            ):  # TRTLLM_K3LL_MOE_PRESORTED (`_front_only`): the bf16 weights, x_sf the image
                views = (*views[:4], b)
            return nvfp4_lowm_op.run_sorted(
                x_q, x_sf, views, self.experts, self.fc2_input_scale,
                self.state.counters, *self.situ, out=fc12.out, fc12=fc12.launch(), presorted_layout=b is not None,
            )  # fmt: skip
        return nvfp4_lowm_op.run_sorted(
            x_q, x_sf, sort_views(sort_out, a, self.num_local), self.experts, self.fc2_input_scale,
            self.state.counters, *self.situ, fast=self._dyn_fast(fast),
        )  # fmt: skip

    def takes(self, hidden_states: torch.Tensor, step, partial_tail: bool) -> bool:
        """Whether this decode path runs the layer on ``step`` (bf16 rows ``hidden_states``): any step of at most
        `MAX_TOKENS` tokens; with ``partial_tail``, also a wide decode step of at most `WIDE_MAX_TOKENS`."""
        rows = hidden_states.shape[0]
        if hidden_states.dtype != torch.bfloat16:
            return False
        return 0 < rows <= MAX_TOKENS or (partial_tail and step.wide and rows <= WIDE_MAX_TOKENS)

    def forward(
        self,
        moe: nn.Module,
        hidden_states: torch.Tensor,
        gemvs,
        partial_tail: bool,
        push: bool = False,
    ) -> Union[torch.Tensor, PendingTail, WideTail, LuTail, ColTail]:
        """The MoE output of ``hidden_states`` (``takes`` holds): with ``partial_tail``, this rank's unreduced share
        (a ``PendingTail``; on a wide step a tensor, a ``WideTail`` with ``TRTLLM_K3LL_MOE_TAIL_PUSH``, a ``LuTail``
        with ``TRTLLM_K3LL_MOE_TAIL_LU``, or a ``ColTail`` with ``TRTLLM_K3LL_MOE_TAIL_COL``); else the
        reduced output. ``push`` is ignored: the latent all-reduce is always the routed
        experts' all-reduce."""
        if hidden_states.shape[0] > MAX_TOKENS:
            return self._wide(moe, hidden_states, gemvs)
        routed, shared_act = self._front_routed(moe, hidden_states)
        latent = moe.routed_experts.all_reduce(routed)
        if partial_tail:
            return PendingTail(
                latent.contiguous(),
                shared_act.contiguous(),
                self.tail_weight,
                self.lo,
                float(moe.routed_expert_norm.variance_epsilon),
            )
        shared = moe.shared_experts
        shared_out = shared.down_proj(shared_act, layer_idx=shared.layer_idx)
        up = _gemv(gemvs, "moe_up", latent, moe.routed_expert_up_proj.weight, out_fp32=True)
        scale = torch.rsqrt(
            latent.float().pow(2).mean(-1, keepdim=True) + moe.routed_expert_norm.variance_epsilon
        )
        return (up * scale + shared_out.float()).bfloat16()

    def _wide_head(self, moe: nn.Module, x: torch.Tensor, gemvs, fc12: Optional[Fc12Step] = None):
        """A wide step's routed partial and shared activation on M-general ops: the head GEMV, the MNNVL all-gather,
        ``k3_route_quant`` and the experts, and the shared gate_up GEMV + activation on the shared-expert stream.
        ``fc12``: `_routed`'s."""
        head = _gemv(gemvs, "moe_head", x, self.head_weight, out_fp32=True)
        routed_in, router_logits = mnnvl_allgather_split(head, self.width, self.state.mnnvl)
        shared = moe.shared_experts

        def _routed_partial():
            routing = self._route_quant(moe, router_logits, routed_in)
            if self.stats is not None:
                self.stats.record(self.layer_idx, routing[0])
            return self._routed(*routing, fc12=fc12)

        def _shared_activation():
            return shared._apply_activation(
                _gemv(gemvs, "moe_shared_gate_up", x, shared.gate_up_proj.weight)
            )

        return maybe_execute_in_parallel(
            _routed_partial,
            _shared_activation,
            moe.moe_main_event,
            moe.moe_shared_event,
            moe.shared_expert_stream,
            disable_on_compile=True,
        )

    def _wide(
        self, moe: nn.Module, hidden_states: torch.Tensor, gemvs
    ) -> Union[torch.Tensor, WideTail, LuTail, ColTail]:
        """A wide decode step's MoE (`MAX_TOKENS` < M <= `WIDE_MAX_TOKENS`): this rank's unreduced share of the
        output, ``[M, hidden]`` bf16, as `decode_moe.py`'s wide step with the NVFP4 latent and experts. With
        ``TRTLLM_K3LL_MOE_TAIL_PUSH`` (read at forward time) the tail's input and weight instead (a `WideTail`), whose
        consumer runs the tail GEMV (``moe_tail_push.py``). With ``TRTLLM_K3LL_MOE_TAIL_LU`` (read at forward time),
        where ``moe_tail_lu.takes`` holds, the reduced routed and shared outputs (a `LuTail`,
        ``moe_tail_lu.py``). With ``TRTLLM_K3LL_MOE_TAIL_COL`` (read at forward time), where ``moe_tail_col.takes``
        holds, the reduced output published into the decode workspace (a `ColTail`, ``moe_tail_col.py``); it takes
        precedence over ``TRTLLM_K3LL_MOE_TAIL_LU``. With ``TRTLLM_K3LL_MOE_LOWM_FC12`` (read at forward time), where
        `fc12_takes` holds, the routed experts run as FC1 + FC2 (`K3MoeFc12`), their output zeroed from before the
        front."""
        x = hidden_states.contiguous()
        fc12 = None
        if fc12_takes(self.state, x.shape[0]):
            from . import census as _census

            _census.count(f"dispatch:moe_lowm_fc12:{_census.mode()}")
            fc12 = self.state.fc12.begin(x.shape[0], self.experts[3].shape[1], x.device)
        front32 = bool(_flags.TRTLLM_K3LL_MOE_FRONT32)
        if (
            front32
            and FRONT16_TOKENS < x.shape[0] <= FRONT32_TOKENS
            and self.state.head32 is not None
        ):
            from . import census as _census

            _census.count(f"moe_front32:{_census.mode()}")
        if self.state.front_head(x.shape[0], front32) is not None:
            # The front's 16- or 32-token build: head GEMV, all-gather, routing, NVFP4 latent and shared gate_up + SiTU.
            # TRTLLM_K3LL_MOE_TAIL_COL first: where both tails take the step, the lossless one runs.
            if _tail_col.takes(self.state.col, self.col, x.shape[0], self.shared_cols, gemvs):
                from . import census as _census

                _census.count(f"dispatch:moe_tail_col:{_census.mode()}")
                routing, shared_act = self._front_only(moe, x, front32, fc12=fc12 is not None)
                return _tail_col.run(
                    self.state.col,
                    self.col,
                    gemvs,
                    shared_act,
                    lambda: self._routed_after(routing, fc12=fc12),
                )
            if _tail_lu.takes(self.state.lu, self.lu, x.shape[0], self.shared_cols, gemvs):
                from . import census as _census

                _census.count(f"dispatch:moe_tail_lu:{_census.mode()}")
                routing, shared_act = self._front_only(moe, x, front32, fc12=fc12 is not None)
                return _tail_lu.run(
                    self.state.lu,
                    self.lu,
                    gemvs,
                    shared_act,
                    lambda: self._routed_after(routing, fc12=fc12),
                )
            routed, shared_act = self._front_routed(moe, x, front32, fc12=fc12)
        else:
            routed, shared_act = self._wide_head(moe, x, gemvs, fc12=fc12)
        # The latent norm's weight is folded into the tail weight: normalize the whole latent row, keep this rank's
        # columns. With TRTLLM_K3LL_WIDE_LATENT_TAIL one MNNVL call builds the tail's input: each rank reduces only its
        # own latent columns, and the ranks exchange the per-token sums of squares.
        pad = 0 if self.tail_pad is None else self.tail_pad.shape[1]
        if _flags.TRTLLM_K3LL_WIDE_LATENT_TAIL and rmsnorm_slice_fits(
            x.shape[0], routed.shape[1], pad, shared_act.shape[1], self.state.mnnvl
        ):
            tail_in = mnnvl_allreduce_rmsnorm_slice(
                routed,
                shared_act,
                pad,
                float(moe.routed_expert_norm.variance_epsilon),
                self.state.mnnvl,
            )
        else:
            normed = moe.routed_expert_norm(wide_all_reduce(moe.routed_experts.all_reduce, routed))
            parts = [normed[:, self.lo : self.lo + self.width]]
            if self.tail_pad is not None:
                parts.append(self.tail_pad[: x.shape[0]])
            parts.append(shared_act)
            tail_in = torch.cat(parts, dim=1)
        if _flags.TRTLLM_K3LL_MOE_TAIL_PUSH and gemvs is not None:
            # The consumer (the next pre-attention step or the final norm) runs the GEMV, pushing where it can.
            return WideTail(tail_in, self.tail_weight, gemvs)
        return _gemv(gemvs, "moe_tail", tail_in, self.tail_weight)
