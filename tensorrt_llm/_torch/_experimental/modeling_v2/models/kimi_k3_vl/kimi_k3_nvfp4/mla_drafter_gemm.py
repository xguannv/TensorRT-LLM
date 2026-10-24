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

from typing import Optional, Set, Tuple

import torch

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

# TRTLLM_K3LL_DRAFTER_GEMM_ROW: o_proj (row-parallel, before its all-reduce) and a shallower q_b ring at up to 16 rows.
# Measured with a PDL predecessor and cold weights (oak/dgemm/bench_drafter_gemm.py, GB300), exposed us against cuBLAS /
# the GEMM pin: o_proj 2.56 / 3.22 (16 rows), 2.77 / 3.43 (32); q_b 2.22 / 2.77 (16 rows). down stays on cuBLAS: on
# the low-M GEMM it measured +0.2..+0.3 us per layer in the decode graph.
ROW_PINS = {
    # o_proj: 7168 x (4 heads x 128) per rank.
    "o_proj": ((16, (64, 16, 1, 4)), (32, (64, 32, 1, 4))),
    "q_b": ((16, (64, 8, 4, 4)), (32, (64, 16, 3, 10))),
}


class DrafterGemm:
    """The builds (role, rows tactic, PDL) that ran eagerly: the only ones a capture may take."""

    def __init__(self) -> None:
        self._ran: Set[Tuple] = set()

    def project(
        self, x: torch.Tensor, weight: torch.Tensor, role: str, row_pins: bool = False
    ) -> Optional[torch.Tensor]:
        k = x.shape[-1]
        m = x.numel() // k if k else 0
        table = ROW_PINS if row_pins and role in ROW_PINS else PINS
        pin = next((p for p in table.get(role, ()) if 0 < m <= p[0]), None)
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
        key = (role, pin[0], tuple(weight.shape), pdl, pin[1])
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in self._ran:
            return None
        out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
        run_splitk_dense(x2d, weight.t(), None, out, pdl, SplitKTactic(*pin[1]))
        if not capturing:
            self._ran.add(key)
        return out.view(*x.shape[:-1], weight.shape[0])
