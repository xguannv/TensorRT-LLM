# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode path's latent MoE on NVFP4 experts (switch ``TRTLLM_K3LL_NVFP4_DECODE_MOE``).

For every decode step of at most `WIDE_MAX_TOKENS` tokens a MoE layer runs:

* the head: this rank's slice of the latent down projection and of the router (``[width + experts / TP, hidden]``) as
  one GEMM with an fp32 output, then ``comm/mnnvl_allgather_split`` over the TP group, which returns the full bf16
  latent and the full fp32 router logits;
* the routing (``moe/k3_route_quant``: sigmoid, top-16 of sigmoid + bias, renormalized times the routed scaling
  factor; its MXFP8 outputs are not used) and the latent quantized to NVFP4 with the layer's ``fc31_input_scale``
  (``trtllm::fp4_quantize``, linear scales);
* ``trtllm::k3_moe_nvfp4``: this rank's routed partial over its NVFP4 experts (the M <= 8 build up to 8 tokens, the
  wide build above), beside the shared experts' gate_up + SiTU on the auxiliary stream;
* the latent all-reduce and the latent norm (its weight folded into the latent up projection), then the row-parallel
  tail ``[normed latent columns lo:lo+width | padding | shared activation] @ [latent up columns | padding | shared
  down]``: this rank's unreduced share of the MoE output. Where the next layer's pre-attention step reduces it, the
  layer hands it on; otherwise the layer reduces it itself.

So the router and the latent projections are read 1/TP per rank instead of replicated, the shared experts' all-reduce
merges into the output's, and the routed experts are one kernel. Built by the target in ``post_load_weights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
    mnnvl_allgather_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_route_quant import k3_route_quant
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op_nvfp4
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

from .decode_comm import wide_all_reduce
from .decode_moe import TAIL_K_TILE, _gemv, fold_latent_norm

MAX_TOKENS = op_nvfp4.MAX_TOKENS
WIDE_MAX_TOKENS = op_nvfp4.WIDE_MAX_TOKENS
SF_VEC = 16

__all__ = ["K3DecodeMoeNvfp4", "K3DecodeMoeNvfp4Layer", "fold_latent_norm", "layout_gaps"]


def _experts(moe: nn.Module) -> tuple:
    backend = moe.routed_experts.backend
    return (
        backend.w3_w1_weight,
        backend.w3_w1_weight_scale,
        backend.w2_weight,
        backend.w2_weight_scale,
    )


def _scalars(moe: nn.Module) -> Optional[tuple]:
    """``(fc31_input_scale, fc31_alpha, fc2_alpha, fc2_input_scale)`` of the routed experts, None if any is missing."""
    backend = moe.routed_experts.backend
    names = ("fc31_input_scale", "fc31_alpha", "fc2_alpha", "fc2_input_scale")
    values = tuple(getattr(backend, name, None) for name in names)
    if any(v is None for v in values):
        return None
    return tuple(v.detach() if isinstance(v, torch.Tensor) else v for v in values)


def layout_gaps(moe: nn.Module, tp_size: int) -> list:
    """Why the NVFP4 decode path does not take MoE layer ``moe`` (empty when it takes it), read once the routed experts'
    buffers and scales are final: the checks in order, up to the first that fails."""
    gate, backend = moe.gate, moe.routed_experts.backend
    shared = moe.shared_experts
    gate_up, down = shared.gate_up_proj.weight, shared.down_proj.weight
    if not moe._reduce_routed_output:
        return ["a routed output the model does not reduce"]
    if (moe.num_experts, moe.top_k, moe.moe_hidden_size, moe.hidden_size) != (896, 16, 3584, 7168):
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
    if not all(
        hasattr(backend, n)
        for n in ("w3_w1_weight", "w2_weight", "expert_size_per_partition", "slot_start")
    ):
        return ["routed experts without TRTLLM-Gen buffers"]
    ok, why = op_nvfp4.is_supported(*_experts(moe), backend.expert_size_per_partition)
    if not ok:
        return [f"routed-expert buffers k3_moe_nvfp4 does not read ({why})"]
    scalars = _scalars(moe)
    if scalars is None:
        return ["routed experts without the NVFP4 global scales"]
    up_alpha = getattr(backend, "fc31_up_alpha", None)
    if up_alpha is not None and not torch.equal(up_alpha.detach(), scalars[1]):
        return ["gate and up halves with different global scales"]
    return []


@dataclass(eq=False)
class K3DecodeMoeNvfp4:
    """What every MoE layer's NVFP4 decode path shares on one device: the ``k3_moe_nvfp4`` builds for up to 8 and up to
    64 tokens (their scratch; the layers run one at a time on one stream) and the TP group's MNNVL workspace for the
    head all-gather."""

    small: op_nvfp4.K3MoeNvfp4State
    wide: op_nvfp4.K3MoeNvfp4WideState
    mnnvl: MnnvlWorkspace

    @classmethod
    def create(cls, device, i_tp: int, num_local: int, mnnvl: MnnvlWorkspace) -> "K3DecodeMoeNvfp4":
        return cls(
            op_nvfp4.K3MoeNvfp4State(device, i_tp, num_local),
            op_nvfp4.K3MoeNvfp4WideState(device, i_tp, num_local),
            mnnvl,
        )


@dataclass(eq=False)
class K3DecodeMoeNvfp4Layer:
    """One MoE layer's NVFP4 decode path: the head weight (this rank's latent-down rows, then its router rows), the
    row-parallel tail weight ``[latent up columns lo:lo+width | padding | shared down]``, the routing bias, the NVFP4
    scales, and its ``k3_moe_nvfp4`` handles. Built by `create` once the weights are final (latent norm folded)."""

    state: K3DecodeMoeNvfp4
    head_weight: torch.Tensor
    tail_weight: torch.Tensor
    tail_pad: Optional[torch.Tensor]
    bias: torch.Tensor
    fc31_input_scale: torch.Tensor
    lo: int
    width: int
    small: op_nvfp4.K3MoeNvfp4Layer
    wide: op_nvfp4.K3MoeNvfp4Layer

    @classmethod
    def create(
        cls, moe: nn.Module, state: K3DecodeMoeNvfp4, tp_rank: int, tp_size: int
    ) -> "K3DecodeMoeNvfp4Layer":
        width = moe.moe_hidden_size // tp_size
        experts = moe.num_experts // tp_size
        shared_down = moe.shared_experts.down_proj.weight
        up = moe.routed_expert_up_proj.weight
        lo = tp_rank * width
        pad = -width % TAIL_K_TILE
        with torch.no_grad():
            head = torch.cat(
                [
                    moe.routed_expert_down_proj.weight[lo : lo + width],
                    moe.gate.weight[tp_rank * experts : (tp_rank + 1) * experts],
                ]
            ).contiguous()
            parts = [up[:, lo : lo + width]]
            if pad:
                parts.append(up.new_zeros(up.shape[0], pad))
            parts.append(shared_down)
            tail = torch.cat(parts, dim=1).contiguous()
        fc31_input_scale, fc31_alpha, fc2_alpha, fc2_input_scale = _scalars(moe)
        scales = (
            fc31_alpha.float().contiguous(),
            fc2_alpha.float().contiguous(),
            fc2_input_scale.float().reshape(1).contiguous(),
        )
        weights = _experts(moe)
        return cls(
            state=state,
            head_weight=head,
            tail_weight=tail,
            tail_pad=up.new_zeros(WIDE_MAX_TOKENS, pad) if pad else None,
            bias=moe.gate.e_score_correction_bias.detach(),
            fc31_input_scale=fc31_input_scale.float().reshape(1).contiguous(),
            lo=lo,
            width=width,
            small=state.small.layer(*weights, *scales),
            wide=state.wide.layer(*weights, *scales),
        )

    def takes(self, hidden_states: torch.Tensor, step, partial_tail: bool) -> bool:
        """Any bf16 step of at most `WIDE_MAX_TOKENS` tokens."""
        return (
            hidden_states.dtype == torch.bfloat16 and 0 < hidden_states.shape[0] <= WIDE_MAX_TOKENS
        )

    def warm_up(self, moe: nn.Module) -> None:
        """One call of each ``k3_moe_nvfp4`` build on zero inputs (M = 1), so none compiles under a capture."""
        device = self.head_weight.device
        latent = torch.zeros(1, moe.moe_hidden_size, dtype=torch.bfloat16, device=device)
        logits = torch.zeros(1, moe.num_experts, dtype=torch.float32, device=device)
        for build in (self.small, self.wide):
            self._routed(moe, latent, logits, build)
        torch.cuda.synchronize(device)

    def _routed(
        self, moe: nn.Module, latent: torch.Tensor, logits: torch.Tensor, build
    ) -> torch.Tensor:
        rows = latent.shape[0]
        ids, weights, _, _ = k3_route_quant(
            logits, self.bias, latent, float(moe.gate.routed_scaling_factor), early_trigger=False
        )
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
            latent, self.fc31_input_scale, SF_VEC, False, False
        )
        x_fp4 = x_fp4.view(torch.uint8).reshape(rows, moe.moe_hidden_size // 2)
        x_sf = x_sf.view(torch.uint8).reshape(-1)[: rows * (moe.moe_hidden_size // SF_VEC)]
        return build(x_fp4, x_sf, ids, weights, moe.routed_experts.backend.slot_start)

    def forward(self, moe: nn.Module, hidden_states: torch.Tensor, gemvs, partial_tail: bool,
                push: bool = False) -> torch.Tensor:  # fmt: skip
        """The MoE output of ``hidden_states`` (``takes`` holds): with ``partial_tail`` this rank's unreduced share
        ``[M, hidden]`` (the next layer's pre-attention step reduces it), else the reduced output. ``push`` is unused
        (no latent exchange on this path)."""
        x = hidden_states.contiguous()
        rows = x.shape[0]
        head = _gemv(gemvs, "moe_head", x, self.head_weight, out_fp32=True)
        routed_in, router_logits = mnnvl_allgather_split(head, self.width, self.state.mnnvl)
        shared = moe.shared_experts
        build = self.small if rows <= MAX_TOKENS else self.wide

        def _routed_partial():
            return self._routed(moe, routed_in.contiguous(), router_logits.contiguous(), build)

        def _shared_activation():
            return shared._apply_activation(
                _gemv(gemvs, "moe_shared_gate_up", x, shared.gate_up_proj.weight)
            )

        routed, shared_act = maybe_execute_in_parallel(
            _routed_partial,
            _shared_activation,
            moe.moe_main_event,
            moe.moe_shared_event,
            moe.shared_expert_stream,
            disable_on_compile=True,
        )
        # The latent norm's weight is folded into the tail weight: normalize the whole latent row, keep this rank's
        # columns.
        normed = moe.routed_expert_norm(wide_all_reduce(moe.routed_experts.all_reduce, routed))
        parts = [normed[:, self.lo : self.lo + self.width]]
        if self.tail_pad is not None:
            parts.append(self.tail_pad[:rows])
        parts.append(shared_act)
        partial = _gemv(gemvs, "moe_tail", torch.cat(parts, dim=1), self.tail_weight)
        if partial_tail:
            return partial
        return wide_all_reduce(shared.down_proj.all_reduce, partial)
