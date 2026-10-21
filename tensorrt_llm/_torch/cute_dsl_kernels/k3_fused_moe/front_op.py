# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``trtllm::k3_moe_front``: the Kimi K3 MoE front at decode size in one kernel (``k3_moe_front.py``).

Replaces, per MoE layer: the sharded head GEMV, the head all-gather, the top-16 routing and the MXFP8 latent, and, on
the shared-expert stream, the shared gate_up GEMV and SiTU-and-mul. The head all-gather runs over the TP group's
:class:`~.op.K3MoeHeadWorkspace` (the caller's, created collectively before CUDA-graph capture), with the buffer
layout and protocol of ``k3_route_quant_ag.py``. The kernel compiles on the first call for each configuration, which
must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Dict, Optional, Tuple

import torch

TOP_K = 16
HIDDEN_SIZE = 3584
NUM_EXPERTS = 896
SF_VEC = 32
NVFP4_SF_VEC = 16
MAX_TOKENS = 8
# The 16-token build (2 requests x 8 tokens): calls of 9 to 16 rows. Its head workspace is laid out for 16 tokens
# (``K3MoeHeadWorkspace.create(..., m_max=16)``); a workspace serves one build only.
WIDE_TOKENS = 16
DEFAULT_RING = 4

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _kernel():
    from . import k3_moe_front as kernel

    return kernel


def front_weight(
    head_weight: torch.Tensor, gate_up_weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The front's one weight: ``head_weight`` (``[3584/W + 896/W, K]``: this rank's latent-down rows, then its router
    rows) zero-padded to whole 128-row tiles, then ``gate_up_weight`` (``[2 I, K]``: gate rows, then up rows) with
    every 32 rows holding 16 gate rows and the 16 up rows of the same columns. Without ``gate_up_weight`` the front
    computes the head alone (``shared_cols`` 0)."""
    rows, k_in = head_weight.shape
    if gate_up_weight is None:
        gate_up_weight = head_weight[:0]
    inter = gate_up_weight.shape[0] // 2
    if (
        gate_up_weight.shape[1] != k_in
        or inter % 64 != 0
        or head_weight.dtype != gate_up_weight.dtype
    ):
        raise ValueError(
            f"k3_moe_front weight: head {tuple(head_weight.shape)}, gate_up {tuple(gate_up_weight.shape)} "
            f"(needs equal K and dtype and a multiple of 64 shared columns)"
        )
    padded = -(-rows // 128) * 128
    head = torch.zeros(padded, k_in, dtype=head_weight.dtype, device=head_weight.device)
    head[:rows].copy_(head_weight)
    gate = gate_up_weight[:inter].reshape(inter // 16, 16, k_in)
    up = gate_up_weight[inter:].reshape(inter // 16, 16, k_in)
    shared = torch.stack([gate, up], dim=1).reshape(2 * inter, k_in)
    return torch.cat([head, shared]).contiguous()


@functools.lru_cache(maxsize=None)
def _max_clusters_of(device_index: int) -> int:
    from cutlass.utils.hardware_info import HardwareInfo

    return HardwareInfo(device_index).get_max_active_clusters(_kernel().SPLIT)


def m_build(num_tokens: int) -> int:
    """The kernel build of a call of ``num_tokens`` rows: 8, or 16 above 8."""
    return MAX_TOKENS if num_tokens <= MAX_TOKENS else WIDE_TOKENS


def max_clusters(device: torch.device) -> int:
    """Clusters of the kernel's SPLIT CTAs, one CTA per SM, that ``device`` holds at once."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _max_clusters_of(index)


def weight_supported(
    world: int,
    shared_cols: int,
    k_in: int,
    device: torch.device,
    ring: int = DEFAULT_RING,
    m_max: int = MAX_TOKENS,
) -> bool:
    """Whether the ``m_max``-token build runs this TP world, shared activation width and hidden size (any M up to
    ``m_max``)."""
    return _kernel().supports(world, shared_cols, max_clusters(device), k_in, ring, m_max)


def supports(
    x: torch.Tensor, w_front: torch.Tensor, world: int, shared_cols: int, ring: int = DEFAULT_RING
) -> bool:
    kernel = _kernel()
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= WIDE_TOKENS
        and x.dtype == torch.bfloat16
        and w_front.dtype == torch.bfloat16
        and x.shape[1] == w_front.shape[1]
        and w_front.shape[0] == (kernel.head_tiles(world) * 128 + 2 * shared_cols)
        and kernel.supports(
            world, shared_cols, max_clusters(x.device), x.shape[1], ring, m_build(x.shape[0])
        )
    )


def sort_layout(num_tokens: int, num_local: int) -> Tuple[int, int, int, int]:
    """``(tiles, weight offset, row offset, words)`` of ``sort_out`` (int32) for ``num_tokens`` tokens over
    ``num_local`` local experts: ``[0, tiles)`` the tiles' local experts, ``[tiles, 2 tiles)`` their mn limits,
    ``[2 tiles]`` the tile count, fp32 ``[M, 16]`` routing weights from the weight offset, ``tiles * 128`` permuted
    rows from the row offset (``moe_sort``'s outputs for 128-row tiles)."""
    return _kernel().sort_layout(num_tokens, num_local)


def sort_views(sort_out: torch.Tensor, num_tokens: int, num_local: int) -> Tuple[torch.Tensor, ...]:
    """``(tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles,
    token_final_scales)`` views of ``sort_out`` (`sort_layout`)."""
    tiles, w0, p0, words = sort_layout(num_tokens, num_local)
    flat = sort_out.view(-1)[:words]
    return (
        flat[:tiles],
        flat[tiles : 2 * tiles],
        flat[p0:words],
        flat[2 * tiles : 2 * tiles + 1],
        flat[w0:p0].view(torch.float32).view(num_tokens, TOP_K),
    )


@torch.library.custom_op(
    "trtllm::k3_moe_front",
    mutates_args=("ag_uc", "ag_mc", "ag_flags", "ag_ready", "sort_out", "sort_ctr"),
)
def k3_moe_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    ag_uc: torch.Tensor,
    ag_mc: torch.Tensor,
    ag_flags: torch.Tensor,
    ag_rank: int,
    ag_world: int,
    ring: int = DEFAULT_RING,
    ag_ready: Optional[torch.Tensor] = None,
    nvfp4_gsf: Optional[torch.Tensor] = None,
    sort_out: Optional[torch.Tensor] = None,
    sort_ctr: Optional[torch.Tensor] = None,
    sort_offset: int = 0,
    sort_local: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``x`` bf16 ``[M <= 16, 7168]`` (the MoE input, the same on every rank), ``w_front`` from ``front_weight``,
    ``bias`` the routing bias fp32 ``[896]``, ``ag_*`` the fields of the TP group's ``K3MoeHeadWorkspace``. Returns
    ``(topk_ids, topk_weights, quantized, scales, shared)``: what ``trtllm::k3_route_quant`` returns for the gathered
    head's router logits and latent, and the shared experts' activation bf16 ``[M, shared_cols]``. With ``ag_ready``
    it also releases the per-token ready words a ``head_flags`` build of k3_moe acquires.

    With ``nvfp4_gsf`` (fp32 ``[1]``, the routed experts' input global scale) the latent is NVFP4 instead of MXFP8:
    ``quantized`` uint8 ``[M, 1792]`` (E2M1 pairs, low nibble first) and ``scales`` uint8 ``[M, 224]`` (E4M3, linear),
    ``fp4_quantize``'s recipe with ``sf_vec_size`` 16 on the gathered bf16 latent.

    With ``sort_out`` (int32, at least `sort_layout`'s words; with ``nvfp4_gsf``) the kernel also writes ``moe_sort``
    of the routing over the local experts ``[sort_offset, sort_offset + sort_local)`` with 128-row tiles and the
    routing weights in fp32 there (`sort_views`). ``sort_ctr``: int32 ``[1]``, zero, which every call leaves zero;
    calls that share it must be stream-ordered."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    kernel = _kernel()
    if not supports(x, w_front, ag_world, shared_cols, ring):
        raise ValueError(
            f"k3_moe_front: unsupported call x {tuple(x.shape)} {x.dtype}, w_front {tuple(w_front.shape)} "
            f"{w_front.dtype}, world {ag_world}, shared_cols {shared_cols}, ring {ring}"
        )
    from . import k3_route_quant_ag as layout

    m_max = m_build(x.shape[0])
    ag_words = layout.workspace_words(ag_world, m_max)
    if ag_uc.numel() < ag_words or ag_mc.numel() < ag_words:
        raise ValueError(
            f"k3_moe_front: the head workspace holds {ag_uc.numel()} words, the front needs {ag_words} "
            "(K3MoeHeadWorkspace: the all-gather's buffers, then the router partials)"
        )
    num_tokens, k_in = x.shape
    device = x.device
    topk_ids = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(num_tokens, TOP_K, dtype=torch.bfloat16, device=device)
    nvfp4 = nvfp4_gsf is not None
    if nvfp4 and (
        nvfp4_gsf.dtype != torch.float32 or nvfp4_gsf.numel() != 1 or nvfp4_gsf.device != device
    ):
        raise ValueError("k3_moe_front: nvfp4_gsf must be one fp32 value on the input's device")
    if nvfp4 and ag_ready is not None:
        raise ValueError("k3_moe_front: the NVFP4 latent has no head_flags consumer (ag_ready)")
    if m_max != MAX_TOKENS and ag_ready is not None:
        raise ValueError("k3_moe_front: the 16-token build has no head_flags consumer (ag_ready)")
    if nvfp4:
        quantized = torch.empty(num_tokens, HIDDEN_SIZE // 2, dtype=torch.uint8, device=device)
        scales = torch.empty(
            num_tokens, HIDDEN_SIZE // NVFP4_SF_VEC, dtype=torch.uint8, device=device
        )
    else:
        quantized = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
        scales = torch.empty(num_tokens, HIDDEN_SIZE // SF_VEC, dtype=torch.uint8, device=device)
    sort = sort_out is not None
    if sort and (
        not nvfp4
        or sort_ctr is None
        or sort_ctr.dtype != torch.int32
        or sort_out.dtype != torch.int32
        or not sort_out.is_contiguous()
        or not 0 < sort_local <= NUM_EXPERTS
        or sort_out.numel() < sort_layout(num_tokens, sort_local)[3]
    ):
        raise ValueError(
            "k3_moe_front: the sort needs the NVFP4 latent, an int32 sort_ctr and a contiguous int32 sort_out of "
            "sort_layout's words over 1 to 896 local experts"
        )
    shared = torch.empty(num_tokens, shared_cols, dtype=torch.bfloat16, device=device)
    # Head-only fronts (shared_cols 0) still hand the kernel an aligned shared-output argument it never writes.
    shared_arg = shared if shared_cols else ag_flags.view(torch.int16).view(-1)

    def arg2(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def arg(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)

    args = (
        arg2(w_front),
        arg2(x.contiguous()),
        arg(bias.contiguous().view(-1)),
        arg(ag_uc.view(-1)),
        arg(ag_mc.view(-1)),
        arg(ag_flags.view(-1)),
        arg(ag_ready.view(-1) if ag_ready is not None else ag_flags.view(-1)),
        arg(topk_ids.view(-1)),
        arg(topk_weights.view(-1).view(torch.int16)),
        arg(quantized.view(-1).view(torch.int32)),
        arg(scales.view(-1)),
        arg(shared_arg.view(-1).view(torch.int16)),
        # Without nvfp4 the kernel never reads it: any aligned fp32 word.
        arg(nvfp4_gsf.view(-1) if nvfp4 else bias.view(-1)[:1]),
        # Without the sort the kernel never touches them: any aligned int32 words.
        arg(sort_out.view(-1) if sort else ag_flags.view(-1)),
        arg(sort_ctr.view(-1) if sort else ag_flags.view(-1)),
    )
    capacity = max_clusters(device)
    # One round with the head in 64-row half-tiles when it fits (TP16): every head k-tile on chip before the wait.
    half = kernel.half_geometry(ag_world, shared_cols, capacity, k_in, ring, m_max)
    head_half = half is not None
    ht, st, clusters = (
        half if head_half else kernel.geometry(ag_world, shared_cols, capacity, m_max)
    )
    publish = ag_ready is not None
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (
        device.index if device.index is not None else torch.cuda.current_device(),
        ag_world,
        shared_cols,
        ht + st,
        clusters,
        k_in,
        ring,
        float(gate_cap),
        float(linear_cap),
        publish,
        use_pdl,
        head_half,
        nvfp4,
        sort,
        m_max,
    )
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_moe_front must run once per configuration outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_moe_front, *args, num_tokens, ag_rank, float(routed_scaling_factor), sort_offset,
                    sort_local, ag_world, shared_cols, ht, ht + st, clusters, k_in, ring, float(gate_cap),
                    float(linear_cap), publish, use_pdl, head_half, nvfp4, sort, m_max, stream,
                )  # fmt: skip
    fn(*args, num_tokens, ag_rank, float(routed_scaling_factor), sort_offset, sort_local, stream)
    return topk_ids, topk_weights, quantized, scales, shared


@k3_moe_front.register_fake
def _(x, w_front, bias, routed_scaling_factor, shared_cols, gate_cap, linear_cap, ag_uc, ag_mc, ag_flags, ag_rank,
      ag_world, ring=DEFAULT_RING, ag_ready=None, nvfp4_gsf=None, sort_out=None, sort_ctr=None, sort_offset=0,
      sort_local=0):  # fmt: skip
    num_tokens = x.shape[0]
    if nvfp4_gsf is not None:
        quantized = x.new_empty((num_tokens, HIDDEN_SIZE // 2), dtype=torch.uint8)
        scales = x.new_empty((num_tokens, HIDDEN_SIZE // NVFP4_SF_VEC), dtype=torch.uint8)
    else:
        quantized = x.new_empty((num_tokens, HIDDEN_SIZE), dtype=torch.float8_e4m3fn)
        scales = x.new_empty((num_tokens, HIDDEN_SIZE // SF_VEC), dtype=torch.uint8)
    return (
        x.new_empty((num_tokens, TOP_K), dtype=torch.int32),
        x.new_empty((num_tokens, TOP_K), dtype=torch.bfloat16),
        quantized,
        scales,
        x.new_empty((num_tokens, shared_cols), dtype=torch.bfloat16),
    )
