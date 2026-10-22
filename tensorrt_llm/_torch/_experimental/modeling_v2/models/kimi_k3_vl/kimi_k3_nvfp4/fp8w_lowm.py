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
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import Callable, Iterable, Optional

import torch
from torch import nn


def _kernel():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


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
    return (
        x2d.dtype == torch.bfloat16
        and getattr(module, "_locality_domain_weight_shards", None) is None
        and _kernel().can_run(
            x2d,
            module.weight,
            module.weight_scale,
            allow_partial_n_tile=getattr(module, "_k3ll_fp8w_partial_n_tile", False),
        )
    )


def _apply(module: nn.Module, stock: Callable, input, bias, lora_params=None, layer_idx=None):
    if (
        bias is None
        and not lora_params
        and isinstance(input, torch.Tensor)
        and _lowm(module, input)
    ):
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(input, module.weight, module.weight_scale)
    return stock(input, bias, lora_params, layer_idx)


def _forward(module: nn.Module, stock: Callable, x: torch.Tensor, **kwargs):
    x2d = x.reshape(-1, x.shape[-1])
    if _lowm(module, x2d):
        out = torch.ops.trtllm.k3_fp8w_lowm_gemm(x2d, module.weight, module.weight_scale)
        return out.reshape(*x.shape[:-1], module.weight.shape[0])
    return stock(x, **kwargs)


def install(
    modules: Iterable[nn.Module], skipped: Optional[Counter] = None, partial_n_tile: bool = False
) -> int:
    """Route each module `takes` accepts through the low-M kernel (idempotent); returns how many it routed.

    ``skipped`` (optional) counts the `rejection` reasons of the modules with an FP8 weight that were not routed;
    ``partial_n_tile`` is `rejection`'s."""
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
    not modified. None where the inputs are not in that form."""
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


def gemm(
    x2d: torch.Tensor, weight: torch.Tensor, packed_scale: torch.Tensor
) -> Optional[torch.Tensor]:
    """``x2d @ dequant(weight)^T`` on the low-M kernel (a partial last row tile allowed), or None where it does not
    take the call."""
    if x2d.dtype != torch.bfloat16 or not _kernel().can_run(
        x2d, weight, packed_scale, allow_partial_n_tile=True
    ):
        return None
    return torch.ops.trtllm.k3_fp8w_lowm_gemm(x2d, weight, packed_scale)
