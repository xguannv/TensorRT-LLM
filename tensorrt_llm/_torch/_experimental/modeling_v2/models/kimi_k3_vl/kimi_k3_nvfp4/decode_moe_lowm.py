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

A wide decode step (9 to `WIDE_MAX_TOKENS` = 32 tokens; to `WIDE_MAX_TOKENS_M64` = 64 with ``TRTLLM_K3LL_MOE_LOWM_M64``,
read at forward time, on the expert kernel's 64-token build) runs the M-general head (`_wide_head`), the same expert
kernel and the row-parallel tail. Steps of more tokens keep the generic path. Only SM100 / SM103 take it
(`nvfp4_lowm_op.supported`).

`K3DecodeMoeNvfp4` holds what every MoE layer shares (the head workspace, the expert kernel's counters);
`K3DecodeMoeNvfp4Layer` holds one layer's decode weights and handles. The target builds both in
`post_load_weights` when ``TRTLLM_K3LL_MOE_LOWM`` (flags.py) is set, before any CUDA-graph capture, and runs
every kernel once there so none compiles under a capture.
"""

from __future__ import annotations

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
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

from . import flags as _flags
from .decode_comm import K3DecodeComm, PendingTail, wide_all_reduce
from .decode_moe import TAIL_K_TILE, _gemv

MAX_TOKENS = 8
# The front's 16-token build (TRTLLM_K3LL_MOE_FRONT16): a wide step of at most this many tokens.
FRONT16_TOKENS = 16
# A wide decode step's most tokens: the expert kernel's default limit, and its limit with TRTLLM_K3LL_MOE_LOWM_M64.
WIDE_MAX_TOKENS = nvfp4_lowm_op.MAX_TOKENS
WIDE_MAX_TOKENS_M64 = nvfp4_lowm_op.MAX_TOKENS_WIDE
NUM_EXPERTS = 896


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

    @classmethod
    def create(
        cls,
        mapping,
        device,
        mnnvl: MnnvlWorkspace,
        front_sort: bool = False,
        front16: bool = False,
    ) -> "K3DecodeMoeNvfp4":
        """Collective (the head workspace): every rank of the TP group calls it at the same point, eagerly, before
        any CUDA-graph capture. ``mnnvl``: the TP group's MNNVL workspace, for a wide step's head all-gather.
        ``front_sort``: the front sorts the routing over the local experts (``k3_moe_front``'s ``sort_out``).
        ``front16``: steps of 9-16 tokens run the front's 16-token build (its own head workspace)."""
        sort_ctr = torch.zeros(1, dtype=torch.int32, device=device) if front_sort else None
        head16 = K3MoeHeadWorkspace.create(mapping, m_max=FRONT16_TOKENS) if front16 else None
        return cls(
            K3MoeHeadWorkspace.create(mapping),
            nvfp4_lowm_op.new_counters(device),
            mnnvl,
            sort_ctr,
            head16,
        )


def front16_supported(moe: nn.Module, tp_size: int) -> bool:
    """Whether the front's 16-token build runs MoE layer ``moe`` (whose 8-token path ``layout_gaps`` takes)."""
    gate_up = moe.shared_experts.gate_up_proj.weight
    return weight_supported(
        tp_size, gate_up.shape[0] // 2, gate_up.shape[1], gate_up.device, m_max=FRONT16_TOKENS
    )


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
    #: Whether the expert kernel's 64-token build serves this layer's experts (``TRTLLM_K3LL_MOE_LOWM_M64``).
    m64: bool = False

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
            # sized for the 64-token steps whatever the switch (free when unused)
            tail_pad=up.new_zeros(WIDE_MAX_TOKENS_M64, pad) if pad else None,
            bias=moe.gate.e_score_correction_bias.detach(),
            lo=lo,
            width=width,
            shared_cols=gate_up.shape[0] // 2,
            experts=_experts(backend),
            input_scale=backend.fc31_input_scale.detach().reshape(1),
            fc2_input_scale=backend.fc2_input_scale.detach().reshape(1),
            offset=int(backend.slot_start),
            situ=(_scalar(backend.act_alpha), _scalar(backend.act_beta)),
            m64=nvfp4_lowm_op.supported(
                backend.w3_w1_weight.device,
                backend.w3_w1_weight.shape[1],
                backend.w2_weight.shape[1],
                WIDE_MAX_TOKENS_M64,
                max_tokens=WIDE_MAX_TOKENS_M64,
            )[0],
        )

    def warm_up(self, moe: nn.Module) -> None:
        """One call of every kernel of the path on zero inputs (M = 1), so none compiles under a capture: the front
        (collective: every rank makes the same call), ``k3_route_quant`` and the expert kernel's builds for 8, 16 and
        32 tokens, and for 64 when ``TRTLLM_K3LL_MOE_LOWM_M64`` is 1 at load (set later, the 64-token build compiles
        on the first eager call, e.g. a CUDA graph's warm-up run)."""
        device = self.front_weight.device
        x = torch.zeros(1, moe.hidden_size, dtype=torch.bfloat16, device=device)
        self._front_routed(moe, x)
        if self.state.head16 is not None:
            self._front_routed(moe, x.new_zeros(MAX_TOKENS + 1, x.shape[1]))
        rows_list = (9, WIDE_MAX_TOKENS)
        if _flags.TRTLLM_K3LL_MOE_LOWM_M64 and self.m64:
            rows_list += (WIDE_MAX_TOKENS_M64,)
        for rows in rows_list:
            logits = torch.zeros(rows, moe.num_experts, dtype=torch.float32, device=device)
            latent = torch.zeros(rows, moe.moe_hidden_size, dtype=torch.bfloat16, device=device)
            self._routed(*self._route_quant(moe, logits, latent))
        torch.cuda.synchronize(device)

    def _route_quant(self, moe: nn.Module, logits: torch.Tensor, latent: torch.Tensor):
        return k3_route_quant(
            logits, self.bias, latent.contiguous(), float(moe.gate.routed_scaling_factor),
            nvfp4_gsf=self.input_scale,
        )  # fmt: skip

    def _front(self, moe: nn.Module, x: torch.Tensor, sort_out: Optional[torch.Tensor] = None):
        head = self.state.head16 if x.shape[0] > MAX_TOKENS else self.state.head
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
        )

    @property
    def num_local(self) -> int:
        """This rank's routed experts."""
        return self.experts[0].shape[0]

    def _routed(self, ids, weights, x_q, x_sf) -> torch.Tensor:
        return nvfp4_lowm_op.run(
            x_q, x_sf, ids, weights, self.experts, self.offset, NUM_EXPERTS, self.fc2_input_scale,
            self.state.counters, *self.situ,
        )  # fmt: skip

    def _front_routed(self, moe: nn.Module, x: torch.Tensor):
        """The front and this rank's routed partial of ``x`` (at most `MAX_TOKENS` rows): ``(routed, shared_act)``.
        With the front's sort the expert kernel reads the front's sorted routing; else ``moe_sort`` sorts it."""
        if self.state.sort_ctr is None:
            ids, weights, x_q, x_sf, shared_act = self._front(moe, x)
            return self._routed(ids, weights, x_q, x_sf), shared_act
        rows = x.shape[0]
        sort_out = torch.empty(
            sort_layout(rows, self.num_local)[3], dtype=torch.int32, device=x.device
        )
        _, _, x_q, x_sf, shared_act = self._front(moe, x, sort_out)
        routed = nvfp4_lowm_op.run_sorted(
            x_q, x_sf, sort_views(sort_out, rows, self.num_local), self.experts, self.fc2_input_scale,
            self.state.counters, *self.situ,
        )  # fmt: skip
        return routed, shared_act

    def wide_max_tokens(self) -> int:
        """A wide decode step's most tokens on this path: `WIDE_MAX_TOKENS_M64` with ``TRTLLM_K3LL_MOE_LOWM_M64`` (read
        here, at forward time) where the 64-token build serves the layer, else `WIDE_MAX_TOKENS`."""
        if _flags.TRTLLM_K3LL_MOE_LOWM_M64 and self.m64:
            return WIDE_MAX_TOKENS_M64
        return WIDE_MAX_TOKENS

    def takes(self, hidden_states: torch.Tensor, step, partial_tail: bool) -> bool:
        """Whether this decode path runs the layer on ``step`` (bf16 rows ``hidden_states``): any step of at most
        `MAX_TOKENS` tokens; with ``partial_tail``, also a wide decode step of at most ``wide_max_tokens()``."""
        rows = hidden_states.shape[0]
        if hidden_states.dtype != torch.bfloat16:
            return False
        return 0 < rows <= MAX_TOKENS or (
            partial_tail and step.wide and rows <= self.wide_max_tokens()
        )

    def forward(
        self,
        moe: nn.Module,
        hidden_states: torch.Tensor,
        gemvs,
        partial_tail: bool,
        push: bool = False,
    ) -> Union[torch.Tensor, PendingTail]:
        """The MoE output of ``hidden_states`` (``takes`` holds): with ``partial_tail``, this rank's unreduced share
        (a ``PendingTail``); else the reduced output. ``push`` is ignored: the latent all-reduce is always the routed
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

    def _wide_head(self, moe: nn.Module, x: torch.Tensor, gemvs):
        """A wide step's routed partial and shared activation on M-general ops: the head GEMV, the MNNVL all-gather,
        ``k3_route_quant`` and the experts, and the shared gate_up GEMV + activation on the shared-expert stream."""
        head = _gemv(gemvs, "moe_head", x, self.head_weight, out_fp32=True)
        routed_in, router_logits = mnnvl_allgather_split(head, self.width, self.state.mnnvl)
        shared = moe.shared_experts

        def _routed_partial():
            return self._routed(*self._route_quant(moe, router_logits, routed_in))

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

    def _wide(self, moe: nn.Module, hidden_states: torch.Tensor, gemvs) -> torch.Tensor:
        """A wide decode step's MoE (`MAX_TOKENS` < M <= ``wide_max_tokens()``): this rank's unreduced share of the
        output, ``[M, hidden]`` bf16, as `decode_moe.py`'s wide step with the NVFP4 latent and experts."""
        x = hidden_states.contiguous()
        if x.shape[0] > WIDE_MAX_TOKENS:
            from . import census as _census

            _census.count(f"dispatch:moe_lowm_m64:{_census.mode()}")
        if self.state.head16 is not None and x.shape[0] <= FRONT16_TOKENS:
            # The front's 16-token build: head GEMV, all-gather, routing, NVFP4 latent and shared gate_up + SiTU.
            routed, shared_act = self._front_routed(moe, x)
        else:
            routed, shared_act = self._wide_head(moe, x, gemvs)
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
            return _gemv(gemvs, "moe_tail", tail_in, self.tail_weight)
        normed = moe.routed_expert_norm(wide_all_reduce(moe.routed_experts.all_reduce, routed))
        parts = [normed[:, self.lo : self.lo + self.width]]
        if self.tail_pad is not None:
            parts.append(self.tail_pad[: x.shape[0]])
        parts.append(shared_act)
        return _gemv(gemvs, "moe_tail", torch.cat(parts, dim=1), self.tail_weight)
