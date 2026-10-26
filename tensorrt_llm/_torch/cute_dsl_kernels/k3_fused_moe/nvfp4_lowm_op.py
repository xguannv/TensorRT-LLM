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

``trtllm::k3_moe_nvfp4_lowm_fc12`` runs the same experts as two persistent kernels of ``blackwell/k3_moe_lowm.py``
on the same buffers: ``K3MoeFc1LowMKernel`` (FC1 + SiTU + NVFP4 quantization into ``h`` / ``h_sf``, triggering its
dependents right after its wait) and then ``K3MoeFc2LowMKernel`` (FC2 + the routing-weighted combine, streaming its
weights before its wait), one CTA per SM each. FC2 adds into its output, so the output must be zero before FC2
(`run_sorted` with ``fc12``). The FC1 output and every FC2 partial are bit for bit the single kernel's; the bf16 sum
of a token's partials (``red.global.add``) is in hardware order in both, so the output is not bitwise.

Each configuration compiles on its first call, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Dict, NamedTuple, Optional, Tuple

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
#: The token counts the two-kernel launch (``k3_moe_nvfp4_lowm_fc12``) serves in the decode path: the 32-token tile.
FC12_MIN_TOKENS = 17
FC12_MAX_TOKENS = MAX_TOKENS

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


def max_permuted_rows(num_tokens: int, top_k: int, num_local: int) -> int:
    """The most permuted rows a sort of ``num_tokens`` tokens' top ``top_k`` over ``num_local`` local experts in
    128-row tiles produces: ``moe_sort``'s allocation (``getMaxPermutedPaddedCount``: one tile per touched expert,
    plus one per 128 expanded rows beyond the first per expert), which also bounds ``k3_moe_front``'s sort (one tile
    per touched expert)."""
    expanded = num_tokens * top_k
    filled = min(num_local, expanded)
    return (filled + (expanded - filled) // TILE) * TILE


class Fc12(NamedTuple):
    """`run` / `run_sorted`'s two-kernel launch (``k3_moe_nvfp4_lowm_fc12``): the FC1 output buffers (`new_fc12`) and
    whether the caller already zeroed the output (stream-ordered before the call; else the call zeroes it first)."""

    h: torch.Tensor
    h_sf: torch.Tensor
    zeroed: bool = False


def new_fc12(
    device: torch.device, n1: int, num_local: int, top_k: int, num_tokens: int = MAX_TOKENS
) -> Fc12:
    """Persistent FC1 output buffers for ``k3_moe_nvfp4_lowm_fc12`` on up to ``num_tokens`` tokens over
    ``num_local`` experts of W1 ``[n1, k1]``: ``h`` uint8 ``[rows, n1 / 4]`` and ``h_sf`` uint8 ``rows * n1 / 32``
    (``rows`` = `max_permuted_rows`). Launches that share them must be stream-ordered."""
    rows = max_permuted_rows(num_tokens, top_k, num_local)
    k2 = n1 // 2
    return Fc12(
        torch.zeros(rows, k2 // 2, dtype=torch.uint8, device=device),
        torch.zeros(rows * k2 // SF_VEC, dtype=torch.uint8, device=device),
    )


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
    topk_ids: Optional[torch.Tensor] = None,
    local_offset: int = 0,
    static_first: bool = False,
    fast_tok: bool = False,
    late_atom: bool = False,
) -> None:
    """This rank's routed partial into ``output`` bf16 ``[M, k1]`` (zeroed by the kernel) for ``M <= 32`` tokens.

    ``x_fp4`` uint8 ``[M, k1 / 2]`` (E2M1 pairs) and ``x_sf`` uint8 ``[M, k1 / 16]`` (E4M3, linear): the NVFP4 latent;
    ``w1`` / ``w1_sf`` / ``alpha1`` / ``w2`` / ``w2_sf`` / ``alpha2``: the CuteDSL backend's NVFP4 buffers and global
    scales of the local experts; the four ``moe_sort`` outputs for 128-row tiles; ``fc2_input_scale`` fp32 ``[1]``
    (the FC1 output's NVFP4 global scale); ``token_final_scales`` fp32 or bf16 ``[M, top_k]`` (bf16 widened in the
    kernel, exactly); ``counters`` from `new_counters`. ``num_ctas``: persistent CTAs (0: one per SM). ``h`` /
    ``h_sf``: uint8 buffers for the FC1 output
    (``[m, n1 / 4]`` E2M1 pairs and ``m * n1 / 32`` scale bytes in the 128x4 blocked layout, ``m`` the permuted rows),
    allocated here when not given. ``topk_ids`` int32 ``[M, top_k]`` (at most `SELF_SORT_MAX_TOKENS` tokens): the
    kernel sorts the routing over the local experts ``[local_offset, local_offset + E)`` itself (`run`'s
    ``self_sort``) and the four sort arguments are not read. ``static_first``: CTA ``b`` takes unit ``b`` first
    (``K3MoeLowMDynKernel``'s static first unit; same results). ``fast_tok``: the kernel's fast token operand
    (no per-FC2-unit counter round trip; same results). ``late_atom`` (with ``static_first``): the CTA's second unit
    taken from the counter after its first unit's stages are issued (same results)."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    from ..blackwell.utils import make_ptr

    lowm, dyn = _lowm(), _dyn()
    orig_m = x_fp4.shape[0]
    num_local, n1 = w1.shape[0], w1.shape[1]
    k1 = w2.shape[1]
    k2 = n1 // 2
    self_sort = topk_ids is not None
    if self_sort:
        if (
            orig_m > SELF_SORT_MAX_TOKENS
            or topk_ids.dtype != torch.int32
            or not topk_ids.is_contiguous()
        ):
            raise ValueError(
                "k3_moe_nvfp4_lowm: self-sort takes contiguous int32 ids of at most 16 tokens"
            )
        m = max_permuted_rows(orig_m, top_k, num_local)
    else:
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

    if token_final_scales.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"k3_moe_nvfp4_lowm: token_final_scales must be fp32 or bf16, not {token_final_scales.dtype}"
        )
    scales_bf16 = token_final_scales.dtype == torch.bfloat16
    scales_dtype = cutlass.BFloat16 if scales_bf16 else cutlass.Float32
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
        ptr(scales_dtype, token_final_scales),
        ptr(cutlass.Int32, counters),
        ptr(cutlass.Int32, topk_ids if self_sort else tile_idx_to_group_idx),
    ]
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (_device_index(device), n_tok, n1, k1, top_k, float(situ_beta), float(situ_linear_beta), ctas, stages,
           use_pdl, num_local if self_sort else 0, scales_bf16, bool(static_first), bool(fast_tok),
           bool(late_atom))  # fmt: skip
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
                    max_ab_stage=stages, use_pdl=use_pdl, self_sort_local=num_local if self_sort else 0,
                    static_first=bool(static_first), fast_tok=bool(fast_tok), late_atom=bool(late_atom),
                )  # fmt: skip
                fn = _compiled[key] = cute.compile(
                    kernel.wrapper, *ptrs, orig_m, m, num_local, local_offset, ctas, stream
                )
    fn(*ptrs, orig_m, m, num_local, local_offset, stream)


@k3_moe_nvfp4_lowm.register_fake
def _(x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, output, tile_idx_to_group_idx, tile_idx_to_mn_limit,
      permuted_idx_to_expanded_idx, num_non_exiting_tiles, fc2_input_scale, token_final_scales, counters, top_k,
      situ_beta, situ_linear_beta, num_ctas=0, h=None, h_sf=None, topk_ids=None, local_offset=0,
      static_first=False, fast_tok=False, late_atom=False):  # fmt: skip
    return None


def _gmem_ptr(dtype, t: torch.Tensor, align: Optional[int] = None):
    import cutlass.cute as cute

    from ..blackwell.utils import make_ptr

    if align is None:
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem)
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@torch.library.custom_op("trtllm::k3_moe_nvfp4_lowm_fc12", mutates_args=("output", "h", "h_sf"))
def k3_moe_nvfp4_lowm_fc12(
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
    h: torch.Tensor,
    h_sf: torch.Tensor,
    top_k: int,
    situ_beta: float,
    situ_linear_beta: float,
    topk_ids: Optional[torch.Tensor] = None,
    local_offset: int = 0,
) -> None:
    """`k3_moe_nvfp4_lowm`'s routed partial as two kernels: ``K3MoeFc1LowMKernel`` into ``h`` / ``h_sf`` (required:
    `new_fc12`), then ``K3MoeFc2LowMKernel`` adding into ``output``, which must be zero (this op does not zero it).
    Same operands as `k3_moe_nvfp4_lowm` but the counters. Both grids one CTA per SM. With PDL
    (``TRTLLM_ENABLE_PDL``, as `k3_moe_nvfp4_lowm`): FC1 waits on the previous grid and then triggers FC2 at once;
    FC2 streams its weights before its wait (it reads only the tile metadata, which the kernel before FC1 wrote) and
    loads FC1's output, so also computes and adds, after it; FC2 triggers its dependents after all of its warps are
    done. ``topk_ids`` int32 ``[M, top_k]``: both kernels sort the routing over the local experts
    ``[local_offset, local_offset + E)`` themselves, per CTA (`run`'s ``self_sort``; the front's tiles and rows), the
    four sort arguments are not read and ``token_final_scales`` may be bf16."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    lowm = _lowm()
    orig_m = x_fp4.shape[0]
    num_local, n1 = w1.shape[0], w1.shape[1]
    k1 = w2.shape[1]
    k2 = n1 // 2
    self_sort = topk_ids is not None
    if self_sort:
        if (
            orig_m > MAX_TOKENS
            or num_local > SELF_SORT_MAX_LOCAL
            or topk_ids.dtype != torch.int32
            or not topk_ids.is_contiguous()
        ):
            raise ValueError(
                "k3_moe_nvfp4_lowm_fc12: self-sort takes contiguous int32 ids of at most 32 tokens"
            )
        m = max_permuted_rows(orig_m, top_k, num_local)
    else:
        m = permuted_idx_to_expanded_idx.shape[0]
    if h.numel() < m * k2 // 2 or h_sf.numel() < m * k2 // SF_VEC:
        raise ValueError(
            "k3_moe_nvfp4_lowm_fc12: h / h_sf hold fewer than the permuted rows' FC1 output"
        )
    if token_final_scales.dtype not in (torch.float32, torch.bfloat16) or (
        token_final_scales.dtype == torch.bfloat16 and not self_sort
    ):
        raise ValueError(
            "k3_moe_nvfp4_lowm_fc12: token_final_scales must be fp32 (or bf16 with topk_ids), "
            f"not {token_final_scales.dtype}"
        )
    scales_bf16 = token_final_scales.dtype == torch.bfloat16
    n_tok = lowm.n_tok_for(orig_m)
    device = x_fp4.device
    ctas = _num_sms(_device_index(device))
    P = _gmem_ptr
    fc1_ptrs = [
        P(cutlass.Float4E2M1FN, x_fp4, 32),
        P(cutlass.Float4E2M1FN, w1, 32),
        P(cutlass.Float8E4M3FN, x_sf, 16),
        P(cutlass.Float8E4M3FN, w1_sf, 16),
        P(cutlass.Float4E2M1FN, h, 32),
        P(cutlass.Float8E4M3FN, h_sf, 16),
        P(cutlass.Float32, alpha1),
        P(cutlass.Int32, tile_idx_to_group_idx),
        P(cutlass.Int32, tile_idx_to_mn_limit),
        P(cutlass.Int32, permuted_idx_to_expanded_idx),
        P(cutlass.Int32, num_non_exiting_tiles),
        P(cutlass.Float32, fc2_input_scale),
    ]
    fc2_ptrs = [
        P(cutlass.Float4E2M1FN, h, 32),
        P(cutlass.Float4E2M1FN, w2, 32),
        P(cutlass.Float8E4M3FN, h_sf, 16),
        P(cutlass.Float8E4M3FN, w2_sf, 16),
        P(cutlass.BFloat16, output, 16),
        P(cutlass.Float32, alpha2),
        P(cutlass.Int32, tile_idx_to_group_idx),
        P(cutlass.Int32, tile_idx_to_mn_limit),
        P(cutlass.Int32, permuted_idx_to_expanded_idx),
        P(cutlass.Int32, num_non_exiting_tiles),
        P(cutlass.BFloat16 if scales_bf16 else cutlass.Float32, token_final_scales),
    ]
    # the optional (tile_ready, trace) pointers stay None; the ids pointer only with self-sort
    tail = (None, None, P(cutlass.Int32, topk_ids)) if self_sort else ()
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    base = (_device_index(device), n_tok, n1, k1, top_k, ctas, use_pdl)
    if self_sort:
        base = (*base, "self_sort", num_local, int(local_offset))
    key1 = ("fc1", *base, float(situ_beta), float(situ_linear_beta))
    key2 = ("fc2", *base, scales_bf16)
    fc1, fc2 = _compiled.get(key1), _compiled.get(key2)
    if fc1 is None or fc2 is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_moe_nvfp4_lowm_fc12 must run once per configuration outside CUDA-graph capture first"
            )
        with _lock:
            fc1 = _compiled.get(key1)
            if fc1 is None:
                kernel = lowm.K3MoeFc1LowMKernel(
                    n_tok, n1, k1, top_k, float(situ_beta), float(situ_linear_beta), use_pdl=use_pdl,
                    late_trigger=False, self_sort_local=num_local if self_sort else 0,
                    self_sort_offset=int(local_offset) if self_sort else 0,
                )  # fmt: skip
                fc1 = _compiled[key1] = cute.compile(
                    kernel.wrapper, *fc1_ptrs, orig_m, m, num_local, ctas, stream, *tail
                )
            fc2 = _compiled.get(key2)
            if fc2 is None:
                kernel = lowm.K3MoeFc2LowMKernel(
                    n_tok, k1, k2, top_k, use_pdl=use_pdl, early_weights=True,
                    self_sort_local=num_local if self_sort else 0,
                    self_sort_offset=int(local_offset) if self_sort else 0,
                )  # fmt: skip
                fc2 = _compiled[key2] = cute.compile(
                    kernel.wrapper, *fc2_ptrs, m, num_local, orig_m, ctas, stream, *tail
                )
    fc1(*fc1_ptrs, orig_m, m, num_local, stream, *tail)
    fc2(*fc2_ptrs, m, num_local, orig_m, stream, *tail)


@k3_moe_nvfp4_lowm_fc12.register_fake
def _(x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, output, tile_idx_to_group_idx, tile_idx_to_mn_limit,
      permuted_idx_to_expanded_idx, num_non_exiting_tiles, fc2_input_scale, token_final_scales, h, h_sf, top_k,
      situ_beta, situ_linear_beta, topk_ids=None, local_offset=0):  # fmt: skip
    return None


#: The most tokens the expert kernel's own sort takes (``K3MoeLowMDynKernel(self_sort_local=...)``: its scratch
#: leaves the 8- and 16-token builds' rings as deep as without it).
SELF_SORT_MAX_TOKENS = 16
#: The most local experts it takes (bitmap words of its scratch).
SELF_SORT_MAX_LOCAL = 256


def self_sort_takes(num_tokens: int, num_local: int, fc12: bool = False) -> bool:
    """Whether `run`'s ``self_sort`` serves ``num_tokens`` tokens over ``num_local`` local experts (``fc12``: with
    `run`'s ``fc12``, i.e. FC1 + FC2 each sorting per CTA, up to `MAX_TOKENS` tokens)."""
    max_tokens = MAX_TOKENS if fc12 else SELF_SORT_MAX_TOKENS
    return 0 < num_tokens <= max_tokens and 0 < num_local <= SELF_SORT_MAX_LOCAL


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
    fc12: Optional[Fc12] = None,
    self_sort: bool = False,
    fast: bool = False,
) -> torch.Tensor:
    """``moe_sort`` of ``topk_ids`` over this rank's experts, then `k3_moe_nvfp4_lowm`: this rank's routed partial
    bf16 ``[M, k1]``. ``weights``: ``(w1, w1_sf, alpha1, w2, w2_sf, alpha2)`` (an ``NvFp4WeightView``'s order);
    ``topk_weights`` any float dtype ``[M, top_k]`` (converted to fp32). ``fc12``: `run_sorted`'s. ``self_sort``
    (where `self_sort_takes` holds, with ``fc12`` its ``fc12`` form): no ``moe_sort``; the expert kernel(s) sort the
    ids themselves (the same tiles and rows). ``fast``: `run_sorted`'s."""
    w1, w1_sf, alpha1, w2, w2_sf, alpha2 = weights
    num_tokens, top_k = topk_ids.shape
    if self_sort and fc12 is not None and self_sort_takes(num_tokens, w1.shape[0], fc12=True):
        scales = (
            topk_weights
            if topk_weights.dtype in (torch.float32, torch.bfloat16)
            else topk_weights.float()
        )
        ids = (topk_ids if topk_ids.dtype == torch.int32 else topk_ids.int()).contiguous()
        if out is None:
            if fc12.zeroed:
                raise ValueError("run: fc12.zeroed without the zeroed output")
            out = torch.empty(num_tokens, w2.shape[1], dtype=torch.bfloat16, device=x_fp4.device)
        if not fc12.zeroed:
            out.zero_()
        torch.ops.trtllm.k3_moe_nvfp4_lowm_fc12(
            x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, out, ids, ids, ids, ids, fc2_input_scale, scales,
            fc12.h, fc12.h_sf, top_k, float(situ_beta), float(situ_linear_beta), topk_ids=ids,
            local_offset=int(local_expert_offset),
        )  # fmt: skip
        return out
    if self_sort and fc12 is None and self_sort_takes(num_tokens, w1.shape[0]):
        # The kernel reads fp32 or bf16 weights itself: no conversion kernel between the front and the experts.
        scales = (
            topk_weights
            if topk_weights.dtype in (torch.float32, torch.bfloat16)
            else topk_weights.float()
        )
        ids = (topk_ids if topk_ids.dtype == torch.int32 else topk_ids.int()).contiguous()
        if out is None:
            out = torch.empty(num_tokens, w2.shape[1], dtype=torch.bfloat16, device=x_fp4.device)
        torch.ops.trtllm.k3_moe_nvfp4_lowm(
            x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, out, ids, ids, ids, ids, fc2_input_scale, scales,
            counters, top_k, float(situ_beta), float(situ_linear_beta), num_ctas, topk_ids=ids,
            local_offset=int(local_expert_offset), static_first=fast, fast_tok=fast,
        )  # fmt: skip
        return out
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
        local_num_experts=w1.shape[0],
        tile_tokens_dim=TILE,
    )
    return run_sorted(
        x_fp4, x_sf, (tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx,
                      num_non_exiting_tiles, scales),
        weights, fc2_input_scale, counters, situ_beta, situ_linear_beta, out, num_ctas, fc12, fast,
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
    fc12: Optional[Fc12] = None,
    fast: bool = False,
) -> torch.Tensor:
    """`k3_moe_nvfp4_lowm` on a routing already sorted: ``sorted_routing`` = ``(tile_idx_to_group_idx,
    tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales)``, ``moe_sort``'s
    outputs for 128-row tiles and the fp32 ``[M, top_k]`` weights (e.g. ``k3_moe_front``'s ``sort_views``).
    ``fc12`` (`new_fc12`): `k3_moe_nvfp4_lowm_fc12` instead, into its buffers, after zeroing ``out`` on the current
    stream unless ``fc12.zeroed`` (then ``out`` must be given, already zero); ``num_ctas`` and ``fast`` do not apply.
    ``fast`` (the single kernel only): `k3_moe_nvfp4_lowm`'s ``static_first`` and ``fast_tok``, the same results."""
    w1, w1_sf, alpha1, w2, w2_sf, alpha2 = weights
    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        scales,
    ) = sorted_routing
    num_tokens, top_k = scales.shape
    if fc12 is not None:
        if out is None:
            if fc12.zeroed:
                raise ValueError("run_sorted: fc12.zeroed without the zeroed output")
            out = torch.empty(num_tokens, w2.shape[1], dtype=torch.bfloat16, device=x_fp4.device)
        if not fc12.zeroed:
            out.zero_()
        torch.ops.trtllm.k3_moe_nvfp4_lowm_fc12(
            x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, out, tile_idx_to_group_idx, tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx, num_non_exiting_tiles, fc2_input_scale, scales, fc12.h, fc12.h_sf, top_k,
            float(situ_beta), float(situ_linear_beta),
        )  # fmt: skip
        return out
    if out is None:
        out = torch.empty(num_tokens, w2.shape[1], dtype=torch.bfloat16, device=x_fp4.device)
    torch.ops.trtllm.k3_moe_nvfp4_lowm(
        x_fp4, x_sf, w1, w1_sf, alpha1, w2, w2_sf, alpha2, out, tile_idx_to_group_idx, tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx, num_non_exiting_tiles, fc2_input_scale, scales, counters, top_k,
        float(situ_beta), float(situ_linear_beta), num_ctas, static_first=fast, fast_tok=fast,
    )  # fmt: skip
    return out
