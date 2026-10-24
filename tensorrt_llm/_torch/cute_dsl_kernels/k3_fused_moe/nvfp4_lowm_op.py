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
"""``trtllm::k3_moe_nvfp4_lowm``: Kimi K3's NVFP4 routed experts at decode size in one kernel.

The persistent CuTe DSL kernel ``blackwell/k3_moe_lowm_dyn.py`` (FC1 + SiTU + NVFP4 quantization + FC2 + the
routing-weighted combine, work units handed out dynamically, the output zeroed by the kernel itself) over the CuteDSL
MoE backend's NVFP4 buffers, read in place, for at most 32 tokens. Its inputs are the NVFP4 latent with linear E4M3
scales (``[M, 3584 / 16]``), the routing weights in fp32 and the six ``moe_sort`` outputs of this rank's experts with
128-row tiles.

The weight layouts are the CuteDSL backend's on SM100 / SM103: W1 rows interleaved 64 up rows then the matching 64
gate rows per 128-row block, W1 / W2 scales in the 128x4 blocked layout. SM107 stores W1 with another interleave, so
`supported` refuses every other architecture.

Each configuration compiles on its first call, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Dict, Optional, Tuple

import torch

MAX_TOKENS = 32
TILE = 128
SF_VEC = 16
#: Tiles a counter buffer holds.
MAX_TILES = 4096
#: Weight ring stages for token tiles of 8 and 16 (fewer bytes in flight per SM: FC2 finds its tiles sooner); 32
#: tokens take as many as fit.
_DEFAULT_STAGES = {8: 6, 16: 6}
_SUPPORTED_SM = ((10, 0), (10, 3))

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _lowm():
    from ..blackwell import k3_moe_lowm

    return k3_moe_lowm


def _dyn():
    from ..blackwell import k3_moe_lowm_dyn

    return k3_moe_lowm_dyn


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    from cutlass.utils.hardware_info import HardwareInfo

    return HardwareInfo(device_index).get_max_active_clusters(1)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def supported(device: torch.device, n1: int, k1: int, num_tokens: int = 1) -> Tuple[bool, str]:
    """Whether the kernel serves ``num_tokens`` tokens over experts of W1 ``[n1, k1]`` on ``device``, and why not."""
    if torch.cuda.get_device_capability(device) not in _SUPPORTED_SM:
        return (
            False,
            "needs sm_100 or sm_103 (the CuteDSL backend's W1 interleave differs on other architectures)",
        )
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    if n1 % 256 or (n1 // 2) % 256 or k1 % 256:
        return False, f"W1 [{n1}, {k1}] is not a multiple of 256 rows / columns (and half rows)"
    if not 0 < num_tokens <= MAX_TOKENS:
        return False, f"{num_tokens} tokens (1 to {MAX_TOKENS})"
    return True, ""


def new_counters(device: torch.device) -> torch.Tensor:
    """A zeroed counter buffer (work, exit and per-tile counters) for one caller. Every launch leaves it zero again;
    launches that share it must be stream-ordered."""
    return torch.zeros(_dyn().counter_words(MAX_TILES), dtype=torch.int32, device=device)


@torch.library.custom_op(
    "trtllm::k3_moe_nvfp4_lowm", mutates_args=("output", "counters", "h", "h_sf")
)
def k3_moe_nvfp4_lowm(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    alpha1: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    alpha2: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    token_final_scales: torch.Tensor,
    counters: torch.Tensor,
    top_k: int,
    situ_beta: float,
    situ_linear_beta: float,
    num_ctas: int = 0,
    h: Optional[torch.Tensor] = None,
    h_sf: Optional[torch.Tensor] = None,
) -> None:
    """This rank's routed partial into ``output`` bf16 ``[M, k1]`` (zeroed by the kernel) for ``M <= 32`` tokens.

    ``x_fp4`` uint8 ``[M, k1 / 2]`` (E2M1 pairs) and ``x_sf`` uint8 ``[M, k1 / 16]`` (E4M3, linear): the NVFP4 latent;
    ``w1`` / ``w1_sf`` / ``alpha1`` / ``w2`` / ``w2_sf`` / ``alpha2``: the CuteDSL backend's NVFP4 buffers and global
    scales of the local experts; the four ``moe_sort`` outputs for 128-row tiles; ``fc2_input_scale`` fp32 ``[1]``
    (the FC1 output's NVFP4 global scale); ``token_final_scales`` fp32 ``[M, top_k]``; ``counters`` from
    `new_counters`. ``num_ctas``: persistent CTAs (0: one per SM). ``h`` / ``h_sf``: uint8 buffers for the FC1 output
    (``[m, n1 / 4]`` E2M1 pairs and ``m * n1 / 32`` scale bytes in the 128x4 blocked layout, ``m`` the permuted rows),
    allocated here when not given."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    from ..blackwell.utils import make_ptr

    lowm, dyn = _lowm(), _dyn()
    orig_m = x_fp4.shape[0]
    num_local, n1 = w1.shape[0], w1.shape[1]
    k1 = w2.shape[1]
    k2 = n1 // 2
    m = permuted_idx_to_expanded_idx.shape[0]
    if m // TILE > MAX_TILES or counters.numel() < dyn.counter_words(m // TILE):
        raise ValueError(f"k3_moe_nvfp4_lowm: {m // TILE} tiles exceed the counter buffer")
    n_tok = lowm.n_tok_for(orig_m)
    device = x_fp4.device
    sms = _num_sms(_device_index(device))
    ctas = min(num_ctas, sms) if num_ctas > 0 else sms
    stages = _DEFAULT_STAGES.get(n_tok, 16)
    if h is None:
        h = torch.empty(m, k2 // 2, dtype=torch.uint8, device=device)
    if h_sf is None:
        h_sf = torch.empty(m * k2 // SF_VEC, dtype=torch.uint8, device=device)
    if h.numel() < m * k2 // 2 or h_sf.numel() < m * k2 // SF_VEC:
        raise ValueError(
            "k3_moe_nvfp4_lowm: h / h_sf hold fewer than the permuted rows' FC1 output"
        )

    def ptr(dtype, t, align=None):
        if align is None:
            return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem)
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    ptrs = [
        ptr(cutlass.Float4E2M1FN, x_fp4, 32),
        ptr(cutlass.Float4E2M1FN, w1, 32),
        ptr(cutlass.Float8E4M3FN, x_sf, 16),
        ptr(cutlass.Float8E4M3FN, w1_sf, 16),
        ptr(cutlass.Float4E2M1FN, h, 32),
        ptr(cutlass.Float8E4M3FN, h_sf, 16),
        ptr(cutlass.Float32, alpha1),
        ptr(cutlass.Float4E2M1FN, w2, 32),
        ptr(cutlass.Float8E4M3FN, w2_sf, 16),
        ptr(cutlass.Float32, alpha2),
        ptr(cutlass.BFloat16, output, 16),
        ptr(cutlass.Int32, tile_idx_to_group_idx),
        ptr(cutlass.Int32, tile_idx_to_mn_limit),
        ptr(cutlass.Int32, permuted_idx_to_expanded_idx),
        ptr(cutlass.Int32, num_non_exiting_tiles),
        ptr(cutlass.Float32, fc2_input_scale),
        ptr(cutlass.Float32, token_final_scales),
        ptr(cutlass.Int32, counters),
    ]
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (_device_index(device), n_tok, n1, k1, top_k, float(situ_beta), float(situ_linear_beta), ctas, stages,
           use_pdl)  # fmt: skip
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_moe_nvfp4_lowm must run once per configuration outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                kernel = dyn.K3MoeLowMDynKernel(
                    n_tok, n1, k1, top_k, float(situ_beta), float(situ_linear_beta), fc2_chunk=1,
                    max_ab_stage=stages, use_pdl=use_pdl,
                )  # fmt: skip
                fn = _compiled[key] = cute.compile(
                    kernel.wrapper, *ptrs, orig_m, m, num_local, ctas, stream
                )
    fn(*ptrs, orig_m, m, num_local, stream)


@k3_moe_nvfp4_lowm.register_fake
def _(x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, output, tile_idx_to_group_idx, tile_idx_to_mn_limit,
      permuted_idx_to_expanded_idx, num_non_exiting_tiles, fc2_input_scale, token_final_scales, counters, top_k,
      situ_beta, situ_linear_beta, num_ctas=0, h=None, h_sf=None):  # fmt: skip
    return None


def run(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights,
    local_expert_offset: int,
    num_experts: int,
    fc2_input_scale: torch.Tensor,
    counters: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    out: Optional[torch.Tensor] = None,
    num_ctas: int = 0,
    local_num_experts: Optional[int] = None,
) -> torch.Tensor:
    """``moe_sort`` of ``topk_ids`` over this rank's experts, then `k3_moe_nvfp4_lowm`: this rank's routed partial
    bf16 ``[M, k1]``. ``weights``: ``(w1, w1_sf, alpha1, w2, w2_sf, alpha2)`` (an ``NvFp4WeightView``'s order);
    ``topk_weights`` any float dtype ``[M, top_k]`` (converted to fp32). ``local_num_experts``: the experts
    ``[local_expert_offset, local_expert_offset + local_num_experts)`` map to the first weight slots (default: all
    of them, ``w1.shape[0]``)."""
    w1, w1_sf, alpha1, w2, w2_sf, alpha2 = weights
    num_tokens, top_k = topk_ids.shape
    scales = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        _expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = torch.ops.trtllm.moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        local_num_experts=w1.shape[0] if local_num_experts is None else local_num_experts,
        tile_tokens_dim=TILE,
    )
    return run_sorted(
        x_fp4, x_sf, (tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx,
                      num_non_exiting_tiles, scales),
        weights, fc2_input_scale, counters, situ_beta, situ_linear_beta, out, num_ctas,
    )  # fmt: skip


def run_sorted(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    sorted_routing: tuple,
    weights,
    fc2_input_scale: torch.Tensor,
    counters: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    out: Optional[torch.Tensor] = None,
    num_ctas: int = 0,
) -> torch.Tensor:
    """`k3_moe_nvfp4_lowm` on a routing already sorted: ``sorted_routing`` = ``(tile_idx_to_group_idx,
    tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales)``, ``moe_sort``'s
    outputs for 128-row tiles and the fp32 ``[M, top_k]`` weights (e.g. ``k3_moe_front``'s ``sort_views``)."""
    w1, w1_sf, alpha1, w2, w2_sf, alpha2 = weights
    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        scales,
    ) = sorted_routing
    num_tokens, top_k = scales.shape
    if out is None:
        out = torch.empty(num_tokens, w2.shape[1], dtype=torch.bfloat16, device=x_fp4.device)
    torch.ops.trtllm.k3_moe_nvfp4_lowm(
        x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, out, tile_idx_to_group_idx, tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx, num_non_exiting_tiles, fc2_input_scale, scales, counters, top_k,
        float(situ_beta), float(situ_linear_beta), num_ctas,
    )  # fmt: skip
    return out
