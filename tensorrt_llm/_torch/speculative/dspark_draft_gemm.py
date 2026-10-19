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
"""Pinned GEMM kernels for the DSpark MLA drafter's BF16 projections.

The drafter's projections run at 7 to 32 token rows. The generic low-M BF16
dispatcher picks a kernel per shape from an isolated timing race, which does
not see what the decode graph sees: cold weights (the drafter streams ~80 MB
per layer) and a PDL head start from the predecessor. Under those conditions
(measured on GB300 as predecessor RMSNorm + GEMM in a CUDA graph with PDL,
weights rotated beyond L2) the low-M split-K tensor-core kernel at the tactics
below beats the dispatcher's in-graph choices -- the SIMT direct kernel for
q_b runs 6.8 us in the decode graph against 2.2 us here, and the fused
[q_a | kv_a] projection, created after the dispatcher labelled the modules,
falls back to cuBLAS (8-10 us against 6.6 us).

``TLLM_DSPARK_DRAFT_GEMM=1`` selects them. They change the drafter's GEMM
accumulation order (split-K partial sums), i.e. draft-token rounding only; the
target's verification keeps the output distribution unchanged.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

DRAFT_GEMM_ENV_VAR = "TLLM_DSPARK_DRAFT_GEMM"

# role -> ((max rows, kernel, split-K tactic (mma_m, mma_n, split_k, ab_stages)), ...)
_PINS = {
    # [q_a | kv_a]: 2112 x 7168
    "q_kv_a": (
        (8, "splitk", (64, 16, 4, 8)),
        (16, "splitk", (64, 16, 4, 9)),
        (32, "splitk", (64, 16, 2, 11)),
    ),
    # q_b: 768 x 1536 per rank
    "q_b": ((16, "splitk", (64, 8, 4, 10)), (32, "splitk", (64, 16, 3, 10))),
    # gate_up: 1792 x 7168 per rank
    "gate_up": ((16, "splitk", (64, 16, 4, 8)), (32, "splitk", (64, 32, 4, 8))),
    # down: 7168 x 896 per rank
    "down": ((32, "cublas", None),),
    # context latents of all layers: 2880 x 7168
    "ctx_latent": (
        (8, "splitk", (128, 8, 4, 6)),
        (16, "splitk", (64, 16, 2, 10)),
        (32, "splitk", (64, 32, 2, 9)),
    ),
    # the draft block's LM head shard: 10240 x 7168
    "lm_head": ((16, "splitk", (128, 16, 1, 6)), (32, "splitk", (128, 32, 1, 5))),
}


def draft_gemm_enabled() -> bool:
    """Whether ``TLLM_DSPARK_DRAFT_GEMM=1`` pins the drafter's GEMM kernels."""
    return os.environ.get(DRAFT_GEMM_ENV_VAR, "0") == "1"


def draft_gemm(x: torch.Tensor, weight: torch.Tensor, role: str) -> Optional[torch.Tensor]:
    """``x @ weight.T`` with the kernel pinned for ``role`` at this row count.

    Returns None (the caller keeps its module path) when the switch is off, the
    row count is above the pinned range, or the operands do not fit the kernel.
    """
    if not draft_gemm_enabled() or role not in _PINS:
        return None
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    pin = next((p for p in _PINS[role] if 0 < m <= p[0]), None)
    if (
        pin is None
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or weight.shape[1] != k
        or not weight.is_contiguous()
        or not x.is_cuda
    ):
        return None
    _, kind, tactic = pin
    if kind == "cublas":
        return F.linear(x, weight)
    from ..cute_dsl_kernels.blackwell.low_m_bf16_splitk import SplitKTactic, run_splitk_dense
    from ..flashinfer_utils import get_env_enable_pdl

    x2d = x.reshape(m, k)
    if not x2d.is_contiguous() or k % 128 or x2d.data_ptr() % 32 or weight.data_ptr() % 32:
        return None
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
    run_splitk_dense(x2d, weight.t(), None, out, get_env_enable_pdl(), SplitKTactic(*tactic))
    return out.view(*x.shape[:-1], weight.shape[0])
