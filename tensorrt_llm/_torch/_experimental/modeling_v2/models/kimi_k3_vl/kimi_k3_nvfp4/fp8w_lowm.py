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
"""

from __future__ import annotations

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
    early = bool(_flags.TRTLLM_K3LL_MX_SF_EARLY_RELEASE)
    if torch.cuda.is_current_stream_capturing() and not kernel.is_compiled(m, n, k, True, early):
        # Compiling under capture is not allowed; the low-M kernel gives the same bits. Warm-up runs eagerly first.
        MX_STATS["not_compiled"] += 1
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
    out = _offered_gemm(x2d, weight, packed_scale, True)
    if out is not None:
        return out
    return torch.ops.trtllm.k3_fp8w_lowm_gemm(x2d, weight, packed_scale)
