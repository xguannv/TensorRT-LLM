# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's FP8 attention projections on ``trtllm::k3_fp8w_lowm_gemm`` at decode size.

With ``TRTLLM_K3LL_FP8W_LOWM`` (flags.py) the target hands every FP8 block-scale ``Linear`` of its decoder layers
whose weight is in DeepGEMM's layout (``use_cute_dsl_blockscaling_mm`` off: the UE8M0-resmoothed E4M3 weight and the
packed UE8M0 scale ``fp8_swap_ab_gemm`` reads) to `install`. Each such module then computes its GEMM, for the inputs
``k3_fp8w_lowm_gemm.can_run`` takes (2D bf16, ``M`` up to the measured cut-over for the weight's shape, SM100 /
SM103), with the low-M kernel: swap-AB, the activation quantized to MXFP8 inside the kernel, cluster split-K, the
weight streamed before the grid-dependency wait. Everything else (bias, LoRA, larger ``M``, the fused GEMM +
all-reduce and NCCL-window paths, which never call ``apply_linear``) runs the module's own GEMM. The module's
all-reduce, where it has one, is untouched: only ``apply_linear`` is replaced.

`install` runs from the target's ``post_load_weights``, which the model loader calls before the ones of the target's
child ``Linear`` modules (``model.modules()`` is pre-order): at that point each weight is still the checkpoint's E4M3
with raw FP32 block scales, and the module's own ``post_load_weights`` resmooths it to UE8M0 and packs the scale
afterwards. So `takes` only checks what the transform keeps (dtype, shape, the GEMM backend choice) and the packed
INT32 scale is checked on every call (``can_run``), where a module still holding raw scales keeps its own GEMM.

The kernel (``cute_dsl_kernels/blackwell/k3_fp8w_lowm_gemm.py``) comes from Dongxu Yang's ``user/dongxuy/k3_lowlat``,
where it replaces ``fp8_swap_ab_gemm`` on the same weights in the ll4 AgentX point. It compiles on its first call
for each shape and token tile, which must happen outside CUDA-graph capture (the engine's warm-up steps do it).
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


def rejection(module: nn.Module) -> Optional[str]:
    """Why this path does not serve ``module`` (None when it does): an FP8 block-scale projection whose GEMM is (or,
    after its own ``post_load_weights``, will be) DeepGEMM's ``fp8_swap_ab_gemm``, with a shape this path can serve."""
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    if weight is None or scale is None or not callable(getattr(module, "apply_linear", None)):
        return "no weight / weight_scale / apply_linear"
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return f"weight {weight.dtype} {weight.dim()}D"
    method = getattr(module, "quant_method", None)
    if method is not None:
        from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

        if not isinstance(method, FP8BlockScalesLinearMethod):
            return f"quant_method {type(method).__name__}"
    n, k = weight.shape
    # Raw 128x128 block scales (not yet transformed) or DeepGEMM's packed UE8M0 scale.
    raw = scale.dtype == torch.float32 and tuple(scale.shape) == (-(-n // 128), -(-k // 128))
    if not raw and scale.dtype != torch.int32:
        return f"weight_scale {scale.dtype} {tuple(scale.shape)} for weight {n}x{k}"
    if getattr(module, "use_cute_dsl_blockscaling_mm", False):
        return "use_cute_dsl_blockscaling_mm"
    if getattr(module, "disable_deep_gemm", False):
        return "disable_deep_gemm"
    if getattr(module, "_locality_domain_weight_shards", None) is not None:
        return "locality-domain shards"
    kernel = _kernel()
    if n % kernel.MMA_M or k % kernel.CTA_K:
        return f"shape {n}x{k}"
    return None


def takes(module: nn.Module) -> bool:
    """Whether this path serves ``module`` (`rejection` is None)."""
    return rejection(module) is None


def _apply(module: nn.Module, stock: Callable, input, bias, lora_params=None, layer_idx=None):
    kernel = _kernel()
    if (
        bias is None
        and not lora_params
        and getattr(module, "_locality_domain_weight_shards", None) is None
        and isinstance(input, torch.Tensor)
        and input.dtype == torch.bfloat16
        and kernel.can_run(input, module.weight, module.weight_scale)
    ):
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(input, module.weight, module.weight_scale)
    return stock(input, bias, lora_params, layer_idx)


def install(modules: Iterable[nn.Module], skipped: Optional[Counter] = None) -> int:
    """Route each module `takes` accepts through the low-M kernel (idempotent); returns how many it routed.

    ``skipped`` (optional) counts the `rejection` reasons of the modules with an FP8 weight that were not routed."""
    count = 0
    for module in modules:
        if getattr(module, "_k3ll_fp8w_lowm", False):
            continue
        why = rejection(module)
        if why is not None:
            weight = getattr(module, "weight", None)
            if (
                skipped is not None
                and isinstance(weight, torch.Tensor)
                and weight.dtype == torch.float8_e4m3fn
            ):
                skipped[f"{type(module).__name__}: {why}"] += 1
            continue
        module.apply_linear = functools.partial(_apply, module, module.apply_linear)
        module._k3ll_fp8w_lowm = True
        count += 1
    return count
