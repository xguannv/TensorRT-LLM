# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA DSpark drafter's BF16 projections on the low-M split-K tensor-core GEMM (``TRTLLM_K3LL_DRAFTER_GEMM``).

At the drafter's 7 to 32 rows the projections that neither a decode GEMV site nor the sandwich takes run on cuBLAS;
``cute_dsl_kernels/blackwell/low_m_bf16_splitk`` (each cluster rank accumulates an exact K slice in fp32, rank 0 sums
them) at the tactics pinned per projection and row count below runs them in fewer microseconds on GB300 with the
weights cold in L2. The split-K partial sums change the accumulation order: draft tokens only.

`DrafterGemm.project` returns ``x @ weight.T`` or None (the caller keeps its path) where the switch is off, the role
has no pin at this row count, the operands do not fit the kernel, or, under CUDA-graph capture, the build has not
run eagerly.
"""

import functools
from typing import Optional, Set, Tuple

import torch

from . import flags as _flags

# role -> ((max rows, split-K tactic (mma_m, mma_n, split_k, ab_stages)), ...), first entry whose max rows holds.
PINS = {
    # [q_a | kv_a | k_pe]: 2112 x 7168 (replicated).
    "q_kv_a": (
        (8, (64, 16, 4, 8)),
        (16, (64, 16, 4, 9)),
        (32, (64, 16, 2, 11)),
    ),
    # q_b: 4 heads x 192 = 768 x 1536 per rank.
    "q_b": ((16, (64, 8, 4, 10)), (32, (64, 16, 3, 10))),
    # gate_up: [gate 896 | up 896] = 1792 x 7168 per rank.
    "gate_up": ((16, (64, 16, 4, 8)), (32, (64, 32, 4, 8))),
}


#: Devices where ``TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU`` is on by default (-1): the gate_up pins above are GB300 (SM103)
#: measurements.
GATE_UP_SWIGLU_CAPABILITIES = ((10, 3),)


@functools.lru_cache(maxsize=None)
def _swiglu_by_default(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in GATE_UP_SWIGLU_CAPABILITIES


def swiglu_enabled(device: torch.device) -> bool:
    """``TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU`` (read per call): 1 on, 0 off, -1 (default) on where the device is in
    ``GATE_UP_SWIGLU_CAPABILITIES``."""
    value = int(_flags.TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU)
    if value >= 0:
        return value > 0
    return device.type == "cuda" and _swiglu_by_default(
        device.index if device.index is not None else torch.cuda.current_device()
    )


class DrafterGemm:
    """The builds (role, rows tactic, PDL) that ran eagerly: the only ones a capture may take."""

    def __init__(self) -> None:
        self._ran: Set[Tuple] = set()

    def project(self, x: torch.Tensor, weight: torch.Tensor, role: str) -> Optional[torch.Tensor]:
        k = x.shape[-1]
        m = x.numel() // k if k else 0
        pin = next((p for p in PINS.get(role, ()) if 0 < m <= p[0]), None)
        if (
            pin is None
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.shape[1] != k
            or not weight.is_contiguous()
            or not x.is_cuda
            or k % 128
            or torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3))
        ):
            return None
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        x2d = x.detach().reshape(m, k)
        weight = weight.detach()
        if not x2d.is_contiguous() or x2d.data_ptr() % 32 or weight.data_ptr() % 32:
            return None
        pdl = get_env_enable_pdl()
        key = (role, pin[0], tuple(weight.shape), pdl)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in self._ran:
            return None
        out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
        run_splitk_dense(x2d, weight.t(), None, out, pdl, SplitKTactic(*pin[1]))
        if not capturing:
            self._ran.add(key)
        return out.view(*x.shape[:-1], weight.shape[0])

    def project_swiglu(self, x: torch.Tensor, gate_up: torch.Tensor) -> Optional[torch.Tensor]:
        """``silu_and_mul(x @ gate_up.T)`` (``gate_up``: ``[gate I | up I]``) as one launch: the gate_up pin's GEMM with
        the SwiGLU epilogue, bit for bit that GEMM followed by ``trtllm::silu_and_mul``; or None (the caller runs the
        GEMM and the activation) where the pin, the operands or, under capture, an eager run are missing."""
        k = x.shape[-1]
        m = x.numel() // k if k else 0
        pin = next((p for p in PINS["gate_up"] if 0 < m <= p[0]), None)
        if (
            pin is None
            or x.dtype != torch.bfloat16
            or gate_up.dtype != torch.bfloat16
            or not x.is_cuda
            or torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3))
        ):
            return None
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
            SplitKTactic,
            run_splitk_dense_swiglu,
            swiglu_supported,
        )
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        x2d = x.detach().reshape(m, k)
        gate_up = gate_up.detach()
        tactic = SplitKTactic(*pin[1])
        if (
            not x2d.is_contiguous()
            or x2d.data_ptr() % 32
            or gate_up.data_ptr() % 32
            or not swiglu_supported(x2d, gate_up, tactic)
        ):
            return None
        pdl = get_env_enable_pdl()
        key = ("gate_up_swiglu", pin[0], tuple(gate_up.shape), pdl)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in self._ran:
            # Not compiled for this shape: the caller's GEMM + silu_and_mul (the same bits). Counted, so an A / B of
            # TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU can tell that it measured the fallback, not the switch.
            from . import census as _census

            _census.count(f"fallback:drafter_swiglu_not_compiled:{_census.mode()}")
            return None
        out = torch.empty((m, gate_up.shape[0] // 2), dtype=x.dtype, device=x.device)
        run_splitk_dense_swiglu(x2d, gate_up, out, tactic, pdl)
        if not capturing:
            self._ran.add(key)
        return out.view(*x.shape[:-1], gate_up.shape[0] // 2)
