# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's FP8 attention projections on ``trtllm::k3_fp8w_lowm_gemm`` at decode size.

With ``TRTLLM_K3LL_FP8W_LOWM`` (flags.py) the target hands the modules of its decoder layers to `install`. Two kinds
of module hold an FP8 block-scale weight that DeepGEMM's ``fp8_swap_ab_gemm`` multiplies (the UE8M0-resmoothed E4M3
weight and the packed UE8M0 scale):

* ``_Fp8BlockScaleWeightReadLinear`` (``modeling_kimi_linear.py``), which carries the K3 attention projections when
  their ``Linear`` is created with ``use_cute_dsl_blockscaling_mm`` off (the switch makes the target build its
  attention modules that way whatever the engine sets, so the rest of the model keeps the engine's GEMM backend).
  Its ``forward`` calls ``fp8_swap_ab_gemm``;
* a ``Linear`` with ``FP8BlockScalesLinearMethod`` whose GEMM backend is DeepGEMM (``use_cute_dsl_blockscaling_mm``
  and ``disable_deep_gemm`` off). Only its ``apply_linear`` is replaced, so its all-reduce is untouched.

Each routed module computes its GEMM, for the inputs ``k3_fp8w_lowm_gemm.can_run`` takes (2D bf16, ``M`` up to the
measured cut-over for the weight's shape, SM100 / SM103), with the low-M kernel: swap-AB, the activation quantized
to MXFP8 inside the kernel, cluster split-K, the weight streamed before the grid-dependency wait. Everything else
runs the module's own GEMM. Code that reads a module's weight directly (the fused KDA / MLA kernels) is unaffected.

`install` runs from the target's ``post_load_weights``, which the model loader calls before the ones of the target's
children (``model.modules()`` is pre-order): each weight may still hold the checkpoint's raw FP32 128x128 scales,
which the module's own ``post_load_weights`` resmooths and packs afterwards. So `rejection` accepts either scale and
the packed INT32 scale is checked on every call (``can_run``), where a module still holding raw scales keeps its
own GEMM.

The kernel (``cute_dsl_kernels/blackwell/k3_fp8w_lowm_gemm.py``) comes from Dongxu Yang's ``user/dongxuy/k3_lowlat``,
where it replaces ``fp8_swap_ab_gemm`` in ``_Fp8BlockScaleWeightReadLinear.forward`` in the ll4 AgentX point. It
compiles on its first call for each shape and token tile, which must happen outside CUDA-graph capture (the
engine's warm-up steps do it).

A pre-quantized MXFP8 input. A producer of a GEMM input can also write the MXFP8 form of it that the low-M kernel
would quantize it to inside its kernel; the GEMM then runs ``k3_fp8w_lowm_mxfp8`` on it, which skips that
quantization and keeps the low-M kernel's tiling, split-K, K-tile order, scale folding and reduction order: its output
is bit for bit the low-M kernel's on the bf16 input. The API (stable; the other call sites build on it):

* Layout. ``xq``: ``torch.float8_e4m3fn [M, K]``, ``xs``: ``torch.uint8 [M, K / 32]``, both row-major, contiguous and
  16-byte aligned; ``xs[r, g]`` is the UE8M0 (biased exponent) scale of columns ``[32 g, 32 g + 32)`` of row ``r``.
  The values are ``k3_fp8w_lowm_mxfp8.mxfp8_quantize_reference(x)`` of the bf16 input ``x`` (exponent of
  ``2^ceil(log2(amax / 448))``, amax floored at bf16 ``0x38D2``; codes ``satfinite_rn_e4m3(x * 2^(127 - e))``); a
  CuTe DSL producer computes them with that module's helpers (``amax_bf16x8_words``, ``amax_pair``,
  ``ue8m0_of_amax``, ``bf16x2_inv_scale``, ``quant_bf16x8_to_e4m3``: the low-M kernel's own PTX), four threads of 8
  consecutive columns per group.
* Buffers. The producer's caller allocates them (`new_mxfp8_input`) on the producer's stream, before the producer
  runs; the producer still writes the bf16 ``x`` (every fallback reads it).
* `takes_mxfp8_input(target, rows)`: whether a consumer would run on an MXFP8 input of ``rows`` rows; ask it before
  having the producer write one. ``target`` is a module this path routes (`install`) or a ``(weight, packed_scale,
  partial_n_tile)`` triple (a fused weight called through `gemm`).
* `forward_mxfp8(module, x, xq, xs)`: a routed module's output for ``x``, from ``(xq, xs)`` where `takes_mxfp8_input`
  holds for it at this call, else ``module(x)``; the same bits either way. `gemm_mxfp8` is the same for a weight
  (`gemm`'s).
* `offer_mxfp8(x, xq, xs)` / `withdraw_mxfp8()`: the implicit form, for consumers whose code is not touched. While
  ``(xq, xs)`` is offered, a routed GEMM (`_forward`, `_apply`, `gemm`) whose input is the offered tensor (the same
  memory, width and strides, its leading rows, on the stream it was offered on) runs as `forward_mxfp8`. The offer
  holds ``x``, so its memory cannot be reused while offered; the caller withdraws it once the consumers have run.

``TRTLLM_K3LL_MX_SF_EARLY_RELEASE`` (read at every call) picks the ``k3_fp8w_lowm_mxfp8`` build that releases each K
tile to the MMA once its scales are folded (the same bits). ``k3_fp8w_lowm_mxfp8`` compiles on its first call of each
shape (outside a capture: the engine's warm-up steps run first); a capture that finds it not compiled keeps the low-M
kernel (the same bits) and logs it. ``MX_STATS`` counts what happened (tests).

``TRTLLM_K3LL_QKVG_BS1_SPLIT`` (read at every call): `gemm` runs the fused KDA weight (`QKVG_BS1_SHAPE`) at 8 or
fewer rows at the split-K the switch forces (one of `QKVG_BS1_SPLITS`, `qkvg_bs1_tactic`) where no MXFP8 input is
offered for it; 0 keeps the default tactic, -1 (default) resolves by `qkvg_bs1_split` (4 on SM103, else 0). Not the
same bits as the default split. The KDA layer compiles every forced split at load (`compile_qkvg_bs1_splits`); a
capture that finds one not compiled keeps the default split (census ``fallback:qkvg_bs1_split_not_compiled``).
Census: every forced-split call counts ``dispatch:qkvg_bs1_split<S>_rows<M>``, a call of the fused shape above 8 rows
with a split in force ``qkvg_bs1_split_skip_rows<M>``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections import Counter
from typing import Callable, Iterable, NamedTuple, Optional, Tuple

import torch
from torch import nn

from . import flags as _flags


def _kernel():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


def _mx_kernel():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8

    return k3_fp8w_lowm_mxfp8


class _MxOffer(NamedTuple):
    x: torch.Tensor
    xq: torch.Tensor
    xs: torch.Tensor
    stream: int


# The offered MXFP8 input (one at a time: the layers run one after the other on one stream), and how many GEMMs took
# it.
_OFFER: list = [None, 0]
#: How many GEMMs ran on an MXFP8 input (``mxfp8``), offers made / taken by no GEMM (``offered`` / ``unconsumed``), and
#: calls that kept the low-M kernel (``declined``: the MXFP8 kernel does not take them; ``not_compiled``: in a capture).
MX_STATS: Counter = Counter()
_MX_LOGGED: set = set()

# TRTLLM_K3LL_MX_SF_EARLY_RELEASE's default (-1) picks the early-release build only on these compute capabilities
# (measured on GB300 / SM103). k3_fp8w_lowm_mxfp8 itself runs only where can_take_rows passes (SM100 / SM103).
MX_SF_EARLY_RELEASE_CAPABILITIES: Tuple[Tuple[int, int], ...] = ((10, 3),)


#: TRTLLM_K3LL_QKVG_BS1_SPLIT: the KDA fused [q | k | v | g | f_a | b] weight (``(N, K)``, built at load by
#: append_rows) at the 8-row tile (bs1 at MTP7) and the split-K values the switch may force. The heuristic's split 5
#: makes 130 CTAs in clusters of 5; behind bs1's k3_sandwich (56 SMs, launching its dependents ~6 us before it ends)
#: more than half of those calls finish ~2-3 us later than the rest. Splits 4 (104 CTAs) and 3 (78) do not.
QKVG_BS1_SHAPE: Tuple[int, int] = (3216, 7168)
QKVG_BS1_MMA_N = 8
QKVG_BS1_SPLITS: Tuple[int, ...] = (3, 4)
#: TRTLLM_K3LL_QKVG_BS1_SPLIT's default (-1) forces `QKVG_BS1_DEFAULT_SPLIT` only on these compute capabilities
#: (measured on GB300 / SM103); elsewhere it keeps the heuristic's split.
QKVG_BS1_SPLIT_CAPABILITIES: Tuple[Tuple[int, int], ...] = ((10, 3),)
QKVG_BS1_DEFAULT_SPLIT = 4
#: ``(device index, N, K, split)`` of the forced-split kernels compiled in this process (`compile_qkvg_bs1_splits`,
#: or an eager call).
_QKVG_BS1_COMPILED: set = set()


@functools.lru_cache(maxsize=None)
def qkvg_bs1_tactic(n: int, k: int, split: int):
    """``k3_fp8w_lowm_gemm.default_tactic``'s heuristic branch for the 8-row tile at split-K ``split``: every
    activation tile in flight if the weight ring keeps at least 6 stages, else 8 / 4 activation stages; the weight
    ring takes the rest of shared memory."""
    kernel = _kernel()
    base = kernel.default_tactic(QKVG_BS1_MMA_N, n, k)
    kt_max = -(-(k // kernel.CTA_K) // split)
    tactic = None
    for x_stages in (kt_max, 8, 4):
        candidate = kernel.K3LowMTactic(
            mma_n=QKVG_BS1_MMA_N,
            split_k=split,
            a_stages=1,
            x_stages=min(x_stages, kt_max),
            b_stages=min(4, kt_max),
            quant_warps=base.quant_warps,
            quant_groups=4,
        )
        try:
            kernel.validate_tactic(n, k, candidate)
        except ValueError:
            continue
        tactic = dataclasses.replace(candidate, a_stages=kernel.max_a_stages(n, k, candidate))
        if tactic.a_stages >= min(6, kt_max):
            break
    if tactic is None:
        raise ValueError(f"no split-{split} tactic for N={n}, K={k}")
    return tactic


def _device_index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def compile_qkvg_bs1_splits(weight: torch.Tensor) -> bool:
    """Compile (no launch) every `QKVG_BS1_SPLITS` kernel of `TRTLLM_K3LL_QKVG_BS1_SPLIT` for ``weight`` (the fused KDA
    weight) if its shape is `QKVG_BS1_SHAPE`, whatever the switch says, so a capture after flipping it finds them.
    Returns whether it did."""
    n, k = weight.shape
    if (n, k) != QKVG_BS1_SHAPE:
        return False
    index = _device_index(weight)
    with torch.cuda.device(index):
        for split in QKVG_BS1_SPLITS:
            if (index, n, k, split) not in _QKVG_BS1_COMPILED:
                _kernel()._compiled_kernel(n, k, qkvg_bs1_tactic(n, k, split), True)
                _QKVG_BS1_COMPILED.add((index, n, k, split))
    return True


@functools.lru_cache(maxsize=None)
def _qkvg_bs1_split_by_default(device_index: int) -> int:
    on = torch.cuda.get_device_capability(device_index) in QKVG_BS1_SPLIT_CAPABILITIES
    return QKVG_BS1_DEFAULT_SPLIT if on else 0


def qkvg_bs1_split(device: torch.device) -> int:
    """``TRTLLM_K3LL_QKVG_BS1_SPLIT`` on ``device``: the forced split-K (0 = the heuristic's). A value >= 0 is used as
    given; -1 (default) is `QKVG_BS1_DEFAULT_SPLIT` where ``device`` is in `QKVG_BS1_SPLIT_CAPABILITIES`, else 0."""
    value = int(_flags.TRTLLM_K3LL_QKVG_BS1_SPLIT)
    if value >= 0:
        return value
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _qkvg_bs1_split_by_default(index)


def _qkvg_bs1_split(
    x2d: torch.Tensor, weight: torch.Tensor, packed_scale: torch.Tensor
) -> Optional[torch.Tensor]:
    """`TRTLLM_K3LL_QKVG_BS1_SPLIT`: the fused KDA GEMM at the forced split-K where the switch is set, the weight is
    `QKVG_BS1_SHAPE` and ``x2d`` has at most `QKVG_BS1_MMA_N` rows (the caller checked ``can_run``); else None. In
    a capture an uncompiled kernel keeps the default split (census ``fallback:qkvg_bs1_split_not_compiled``)."""
    split = qkvg_bs1_split(x2d.device)
    if not split:
        return None
    if split not in QKVG_BS1_SPLITS:
        raise ValueError(
            f"TRTLLM_K3LL_QKVG_BS1_SPLIT={split}: expected 0 or one of {QKVG_BS1_SPLITS}"
        )
    m = x2d.shape[0]
    n, k = weight.shape
    if (n, k) != QKVG_BS1_SHAPE or m > QKVG_BS1_MMA_N:
        return None
    key = (_device_index(x2d), n, k, split)
    if torch.cuda.is_current_stream_capturing() and key not in _QKVG_BS1_COMPILED:
        _census_count("fallback:qkvg_bs1_split_not_compiled")
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x2d.device)
    _kernel().run_k3_fp8w_lowm_gemm(
        x2d, weight, packed_scale, out, tactic=qkvg_bs1_tactic(n, k, split)
    )
    _QKVG_BS1_COMPILED.add(key)
    _census_count(f"dispatch:qkvg_bs1_split{split}_rows{m}")
    return out


@functools.lru_cache(maxsize=None)
def _sf_early_by_default(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in MX_SF_EARLY_RELEASE_CAPABILITIES


def _sf_early_release(device: torch.device) -> bool:
    """``TRTLLM_K3LL_MX_SF_EARLY_RELEASE`` on ``device``: 1 on, 0 off, -1 (default) on where the device is in
    ``MX_SF_EARLY_RELEASE_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_MX_SF_EARLY_RELEASE)
    if value >= 0:
        return value > 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _sf_early_by_default(index)


def _census_count(key: str) -> None:
    from . import census as _census

    _census.count(f"{key}:{_census.mode()}")


def new_mxfp8_input(rows: int, k: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """New ``(xq, xs)`` for the MXFP8 form of a ``[rows, k]`` GEMM input: float8_e4m3fn ``[rows, k]`` codes and uint8
    ``[rows, k / 32]`` UE8M0 scales, on the current stream."""
    return (
        torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(rows, k // 32, dtype=torch.uint8, device=device),
    )


def _target(target) -> Optional[Tuple[torch.Tensor, torch.Tensor, bool]]:
    if isinstance(target, tuple):
        return target
    if target is None or not getattr(target, "_k3ll_fp8w_lowm", False):
        return None
    return target.weight, target.weight_scale, _partial(target)


def takes_mxfp8_input(target, rows: int) -> bool:
    """Whether ``target`` (a module `install` routed, or a ``(weight, packed_scale, partial_n_tile)`` triple) would
    run on an MXFP8 input of ``rows`` rows (``k3_fp8w_lowm_mxfp8``) at this moment."""
    triple = _target(target)
    return triple is not None and _mx_kernel().can_take_rows(
        rows, triple[0], triple[1], allow_partial_n_tile=triple[2]
    )


def gemm_mxfp8(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    packed_scale: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    partial_n_tile: bool = True,
) -> Optional[torch.Tensor]:
    """``x2d @ dequant(weight)^T`` from ``(xq, xs)``, the MXFP8 form of the bf16 ``x2d [M, K]`` (module docstring),
    bit for bit the low-M kernel on ``x2d``; None where the low-M kernel or ``k3_fp8w_lowm_mxfp8`` does not take the
    call (the caller then runs its bf16 GEMM)."""
    if x2d.dtype != torch.bfloat16 or not _kernel().can_run(
        x2d, weight, packed_scale, allow_partial_n_tile=partial_n_tile
    ):
        return None
    return _run_mx(xq[: x2d.shape[0]], xs[: x2d.shape[0]], weight, packed_scale, partial_n_tile)


def forward_mxfp8(
    module: nn.Module, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
) -> torch.Tensor:
    """``module(x)`` for a module `install` routed, from ``(xq, xs)`` (the MXFP8 form of ``x``'s rows) where its
    low-M GEMM and ``k3_fp8w_lowm_mxfp8`` take the call, else ``module(x)``: the same bits either way."""
    x2d = x.reshape(-1, x.shape[-1])
    if getattr(module, "_k3ll_fp8w_lowm", False) and _lowm(module, x2d):
        out = _run_mx(
            xq[: x2d.shape[0]],
            xs[: x2d.shape[0]],
            module.weight,
            module.weight_scale,
            _partial(module),
        )
        if out is not None:
            return out.reshape(*x.shape[:-1], module.weight.shape[0])
    return module(x)


def offer_mxfp8(x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> None:
    """Offer ``(xq, xs)``, the MXFP8 form of the bf16 ``x`` (module docstring), to the routed GEMMs whose input is
    ``x`` on the current stream, until `withdraw_mxfp8` (or the next offer)."""
    _OFFER[0] = _MxOffer(x, xq, xs, torch.cuda.current_stream(x.device).cuda_stream)
    _OFFER[1] = 0
    MX_STATS["offered"] += 1


def withdraw_mxfp8() -> None:
    """End the current offer (if any); an offer no GEMM took is counted (``MX_STATS["unconsumed"]``, census
    ``mxfp8_offer_unconsumed``)."""
    if _OFFER[0] is not None and _OFFER[1] == 0:
        MX_STATS["unconsumed"] += 1
        _census_count("mxfp8_offer_unconsumed")
    _OFFER[0] = None
    _OFFER[1] = 0


def _offered(x2d: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The offered ``(xq, xs)`` rows of ``x2d`` where ``x2d`` is (leading rows of) the offered tensor on its stream."""
    offer = _OFFER[0]
    if offer is None:
        return None
    x = offer.x.reshape(-1, offer.x.shape[-1])
    if (
        x2d.data_ptr() != x.data_ptr()
        or x2d.dtype != x.dtype
        or x2d.dim() != 2
        or x2d.shape[1] != x.shape[1]
        or x2d.shape[0] > x.shape[0]
        or x2d.stride() != x.stride()
        or torch.cuda.current_stream(x2d.device).cuda_stream != offer.stream
    ):
        return None
    m = x2d.shape[0]
    return offer.xq[:m], offer.xs[:m]


def _offered_gemm(
    x2d: torch.Tensor, weight: torch.Tensor, packed_scale: torch.Tensor, partial: bool
) -> Optional[torch.Tensor]:
    """`_run_mx` on the offered form of ``x2d``; None where nothing is offered for it. Only for a call the low-M kernel
    takes (the caller's check)."""
    mx = _offered(x2d)
    if mx is None:
        return None
    out = _run_mx(mx[0], mx[1], weight, packed_scale, partial)
    if out is not None:
        _OFFER[1] += 1
    return out


def _run_mx(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    packed_scale: torch.Tensor,
    partial: bool,
) -> Optional[torch.Tensor]:
    """``k3_fp8w_lowm_mxfp8`` on ``(xq, xs)`` (``TRTLLM_K3LL_MX_SF_EARLY_RELEASE``'s build), or None where it does not
    take the call or, in a capture, has not been compiled."""
    kernel = _mx_kernel()
    if not kernel.can_run_mxfp8(xq, xs, weight, packed_scale, allow_partial_n_tile=partial):
        MX_STATS["declined"] += 1
        return None
    m, n, k = xq.shape[0], weight.shape[0], weight.shape[1]
    early = _sf_early_release(xq.device)
    if torch.cuda.is_current_stream_capturing() and not kernel.is_compiled(m, n, k, True, early):
        # Compiling under capture is not allowed; the low-M kernel gives the same bits. Warm-up runs eagerly first.
        # Counted in the census too, so an A/B whose B arm recaptured into this fallback shows it (tools/gates).
        MX_STATS["not_compiled"] += 1
        _census_count("fallback:mx_not_compiled")
        key = (m, n, k, early)
        if key not in _MX_LOGGED:
            _MX_LOGGED.add(key)
            from tensorrt_llm.logger import logger

            logger.warning(
                f"k3_fp8w_lowm_mxfp8 M={m} N={n} K={k} sf_early_release={int(early)} not compiled before this "
                "CUDA-graph capture: the capture keeps the low-M kernel (same bits)"
            )
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    kernel.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, weight, packed_scale, out, sf_early_release=early)
    MX_STATS["mxfp8"] += 1
    _census_count("dispatch:mxfp8_gemm")
    if early:
        _census_count("dispatch:mx_sf_early_release")
    return out


def _is_weight_read(module: nn.Module) -> bool:
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    return isinstance(module, _Fp8BlockScaleWeightReadLinear)


def rejection(module: nn.Module, partial_n_tile: bool = False) -> Optional[str]:
    """Why this path does not serve ``module`` (None when it does). ``partial_n_tile`` also accepts an FP8 weight-read
    module whose N is not a multiple of the 128-row MMA tile (the kernel zero-fills the last tile's missing rows)."""
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return "no weight / weight_scale"
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return f"weight {weight.dtype} {weight.dim()}D"
    if not _is_weight_read(module):
        if not callable(getattr(module, "apply_linear", None)):
            return "neither an FP8 weight-read module nor a Linear"
        from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

        method = getattr(module, "quant_method", None)
        if not isinstance(method, FP8BlockScalesLinearMethod):
            return f"quant_method {type(method).__name__}"
        if getattr(module, "use_cute_dsl_blockscaling_mm", False):
            return "use_cute_dsl_blockscaling_mm"
        if getattr(module, "disable_deep_gemm", False):
            return "disable_deep_gemm"
        if getattr(module, "_locality_domain_weight_shards", None) is not None:
            return "locality-domain shards"
    n, k = weight.shape
    # Raw 128x128 block scales (not yet transformed) or DeepGEMM's packed UE8M0 scale.
    raw = scale.dtype == torch.float32 and tuple(scale.shape) == (-(-n // 128), -(-k // 128))
    if not raw and scale.dtype != torch.int32:
        return f"weight_scale {scale.dtype} {tuple(scale.shape)} for weight {n}x{k}"
    kernel = _kernel()
    partial = partial_n_tile and _is_weight_read(module) and n % kernel.N_ALIGN == 0
    if (n % kernel.MMA_M and not partial) or k % kernel.CTA_K:
        return f"shape {n}x{k}"
    return None


def takes(module: nn.Module) -> bool:
    """Whether this path serves ``module`` (`rejection` is None)."""
    return rejection(module) is None


def _lowm(module: nn.Module, x2d: torch.Tensor) -> bool:
    # A module routed with partial_n_tile takes the kernel's partial last row tile only while
    # TRTLLM_K3LL_FP8W_LOWM_FUSED_A is 1 (read here, at every call); otherwise its N rejects it and it runs its own
    # GEMM.
    return (
        x2d.dtype == torch.bfloat16
        and getattr(module, "_locality_domain_weight_shards", None) is None
        and _kernel().can_run(
            x2d, module.weight, module.weight_scale, allow_partial_n_tile=_partial(module)
        )
    )


def _apply(module: nn.Module, stock: Callable, input, bias, lora_params=None, layer_idx=None):
    if (
        bias is None
        and not lora_params
        and isinstance(input, torch.Tensor)
        and _lowm(module, input)
    ):
        out = _offered_gemm(input, module.weight, module.weight_scale, _partial(module))
        if out is not None:
            return out
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(input, module.weight, module.weight_scale)
    return stock(input, bias, lora_params, layer_idx)


def _partial(module: nn.Module) -> bool:
    """Whether ``module`` takes the kernels' partial last row tile at this call (`_lowm`)."""
    return bool(
        getattr(module, "_k3ll_fp8w_partial_n_tile", False) and _flags.TRTLLM_K3LL_FP8W_LOWM_FUSED_A
    )


def _forward(module: nn.Module, stock: Callable, x: torch.Tensor, **kwargs):
    x2d = x.reshape(-1, x.shape[-1])
    if _lowm(module, x2d):
        out = _offered_gemm(x2d, module.weight, module.weight_scale, _partial(module))
        if out is None:
            out = torch.ops.trtllm.k3_fp8w_lowm_gemm(x2d, module.weight, module.weight_scale)
        return out.reshape(*x.shape[:-1], module.weight.shape[0])
    return stock(x, **kwargs)


def install(
    modules: Iterable[nn.Module], skipped: Optional[Counter] = None, partial_n_tile: bool = False
) -> int:
    """Route each module `takes` accepts through the low-M kernel (idempotent); returns how many it routed.

    ``skipped`` (optional) counts the `rejection` reasons of the modules with an FP8 weight that were not routed;
    ``partial_n_tile`` is `rejection`'s. A module whose N needs the partial last row tile takes the kernel only while
    ``TRTLLM_K3LL_FP8W_LOWM_FUSED_A`` is 1 at its call (`_lowm`); with it 0 it runs its own GEMM."""
    count = 0
    for module in modules:
        if getattr(module, "_k3ll_fp8w_lowm", False):
            continue
        why = rejection(module, partial_n_tile)
        if why is not None:
            weight = getattr(module, "weight", None)
            if (
                skipped is not None
                and isinstance(weight, torch.Tensor)
                and weight.dtype == torch.float8_e4m3fn
            ):
                skipped[f"{type(module).__name__}: {why}"] += 1
            continue
        if _is_weight_read(module):
            module.forward = functools.partial(_forward, module, module.forward)
        else:
            module.apply_linear = functools.partial(_apply, module, module.apply_linear)
        module._k3ll_fp8w_lowm = True
        module._k3ll_fp8w_partial_n_tile = partial_n_tile
        count += 1
    return count


def append_rows(
    weight: torch.Tensor, packed_scale: torch.Tensor, rows: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """``(weight', packed_scale')``: an FP8 weight-read weight (``N`` a multiple of 128) and its DeepGEMM-packed UE8M0
    scale with the bf16 ``rows`` appended below it, quantized like the checkpoint's FP8 weights: one power-of-two
    scale per 128x128 block (the smallest that keeps the block's absolute maximum within E4M3's 448), E4M3 codes.
    The rows are zero-padded to a multiple of `N_ALIGN` (a partial last 128-row tile). New tensors; the inputs are
    not modified, and the new weight's first ``N`` rows hold ``weight``'s bytes (a caller may replace ``weight`` by
    that view). None where the inputs are not in that form."""
    kernel = _kernel()
    n, k = weight.shape if weight.dim() == 2 else (-1, -1)
    if (
        weight.dtype != torch.float8_e4m3fn
        or n % kernel.MMA_M
        or k % kernel.CTA_K
        or packed_scale.dtype != torch.int32
        or packed_scale.dim() != 2
        or packed_scale.shape[0] != n
        or rows.dtype != torch.bfloat16
        or rows.dim() != 2
        or rows.shape[1] != k
    ):
        return None
    extra = -(-rows.shape[0] // kernel.N_ALIGN) * kernel.N_ALIGN
    padded = torch.zeros(
        -(-extra // kernel.MMA_M) * kernel.MMA_M, k, dtype=torch.float32, device=rows.device
    )
    padded[: rows.shape[0]] = rows.float()
    blocks = padded.view(-1, kernel.MMA_M, k // kernel.CTA_K, kernel.CTA_K)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    codes = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(-1, k)[:extra]
    packed_rows = kernel.pack_ue8m0_block_scale(scale, extra, k)
    if packed_rows.shape[1] != packed_scale.shape[1]:
        return None
    fused_w = torch.cat((weight, codes), dim=0).contiguous()
    words = torch.cat((packed_scale, packed_rows), dim=0)
    # DeepGEMM's MN-major layout: int32 [N, ceil(K / 512)] with strides (1, N).
    fused_s = torch.empty(
        (words.shape[1], words.shape[0]), dtype=torch.int32, device=words.device
    ).t()
    fused_s.copy_(words)
    return fused_w, fused_s


def set_weight(module: nn.Module, weight: torch.Tensor) -> None:
    """Point ``module.weight`` (a buffer, a parameter or a plain attribute) at ``weight``, keeping its kind; the
    module no longer references its previous weight tensor."""
    if "weight" in module._buffers:
        module._buffers["weight"] = weight
    elif "weight" in module._parameters:
        module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
    else:
        module.weight = weight


def gemm(
    x2d: torch.Tensor, weight: torch.Tensor, packed_scale: torch.Tensor
) -> Optional[torch.Tensor]:
    """``x2d @ dequant(weight)^T`` on the low-M kernel (a partial last row tile allowed), or None where it does not
    take the call."""
    if x2d.dtype != torch.bfloat16 or not _kernel().can_run(
        x2d, weight, packed_scale, allow_partial_n_tile=True
    ):
        return None
    if (
        tuple(weight.shape) == QKVG_BS1_SHAPE
        and x2d.shape[0] > QKVG_BS1_MMA_N
        and qkvg_bs1_split(x2d.device)
    ):
        _census_count(f"qkvg_bs1_split_skip_rows{x2d.shape[0]}")
    out = _offered_gemm(x2d, weight, packed_scale, True)
    if out is not None:
        return out
    # An offered MXFP8 input (AttnResFast writes one; bs1's k3_sandwich does not) keeps its kernel above.
    out = _qkvg_bs1_split(x2d, weight, packed_scale)
    if out is not None:
        return out
    return torch.ops.trtllm.k3_fp8w_lowm_gemm(x2d, weight, packed_scale)
