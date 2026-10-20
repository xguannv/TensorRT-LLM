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

The kernel (``cute_dsl_kernels/blackwell/k3_fp8w_lowm_gemm.py``) comes from Dongxu Yang's ``user/dongxuy/k3_lowlat``,
where it replaces ``fp8_swap_ab_gemm`` on the same weights in the ll4 AgentX point. It compiles on its first call
for each shape and token tile, which must happen outside CUDA-graph capture (the engine's warm-up steps do it).
"""

from __future__ import annotations

import functools
from typing import Callable, Iterable

import torch
from torch import nn


def _kernel():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm


def takes(module: nn.Module) -> bool:
    """Whether ``module`` is an FP8 block-scale projection in DeepGEMM's layout this path can serve."""
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    if weight is None or scale is None or not callable(getattr(module, "apply_linear", None)):
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or scale.dtype != torch.int32:
        return False
    if getattr(module, "use_cute_dsl_blockscaling_mm", False) or getattr(
        module, "disable_deep_gemm", False
    ):
        return False
    if getattr(module, "_locality_domain_weight_shards", None) is not None:
        return False
    kernel = _kernel()
    n, k = weight.shape
    return n % kernel.MMA_M == 0 and k % kernel.CTA_K == 0


def _apply(module: nn.Module, stock: Callable, input, bias, lora_params=None, layer_idx=None):
    kernel = _kernel()
    if (
        bias is None
        and not lora_params
        and isinstance(input, torch.Tensor)
        and input.dtype == torch.bfloat16
        and kernel.can_run(input, module.weight, module.weight_scale)
    ):
        return torch.ops.trtllm.k3_fp8w_lowm_gemm(input, module.weight, module.weight_scale)
    return stock(input, bias, lora_params, layer_idx)


def install(modules: Iterable[nn.Module]) -> int:
    """Route each module `takes` accepts through the low-M kernel (idempotent); returns how many it routed."""
    count = 0
    for module in modules:
        if getattr(module, "_k3ll_fp8w_lowm", False) or not takes(module):
            continue
        module.apply_linear = functools.partial(_apply, module, module.apply_linear)
        module._k3ll_fp8w_lowm = True
        count += 1
    return count
