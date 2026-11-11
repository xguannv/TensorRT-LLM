# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M BF16 / FP16 GEMM with an in-kernel cluster split-K reduction (CuTe DSL), at the
tactics the DSpark drafter pins per projection."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_gemm.low_m_bf16_splitk import SplitKTactic, run_splitk_dense


class K3SplitkDense(OpWrapper):
    """``out = a [M, K] @ b [K, N] (+ bias [N])`` in ``out``'s dtype (BF16 / FP16), FP32
    accumulation: each cluster rank sums an exact K slice, rank 0 reduces the partials through
    DSMEM, casts and stores once. Returns ``out``.

    ``b`` is column-major (``weight.t()`` of a contiguous ``[N, K]`` weight); ``a`` row-major,
    or with ``strided_a`` a row-major view with a 16-byte-aligned row stride. All 32-byte
    aligned; ``tactic`` is ``(mma_m, mma_n, split_k, ab_stages)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The helix16 ULL drafter's projections at its block row counts (1 / 2 / 4 requests x 8),
    # each with the tactic ``dspark_draft_gemm`` pins for that shape and row bound.
    CELLS = (
        Cell(
            why="helix16 ULL c4 [q_a | kv_a] 2112 x 7168 at 32 rows",
            spec=dict(n=2112, k=7168, rows=32, tactic=(64, 32, 4, 8)),
        ),
        Cell(
            why="helix16 ULL c1 [q_a | kv_a] 2112 x 7168 at 8 rows",
            spec=dict(n=2112, k=7168, rows=8, tactic=(64, 8, 4, 12)),
        ),
        Cell(
            why="helix16 ULL c2 context latents of the 5 layers, 2880 x 7168 at 16 rows",
            spec=dict(n=2880, k=7168, rows=16, tactic=(64, 16, 4, 10)),
        ),
        Cell(
            why="helix16 ULL c4 q_b, 64 heads x 192 = 12288 x 1536 at 32 rows",
            spec=dict(n=12288, k=1536, rows=32, tactic=(64, 32, 1, 9)),
        ),
        Cell(
            why="helix16 ULL c4 MLP gate_up shard of the TP16 group, 1792 x 7168 at 32 rows",
            spec=dict(n=1792, k=7168, rows=32, tactic=(64, 32, 4, 8)),
        ),
        Cell(
            why="helix16 ULL c2 MLP down shard, 7168 x 896 at 16 rows",
            spec=dict(n=7168, k=896, rows=16, tactic=(64, 16, 1, 11)),
        ),
        Cell(
            why="helix16 ULL c1 draft LM head shard, 10240 x 7168 at 8 rows",
            spec=dict(n=10240, k=7168, rows=8, tactic=(128, 8, 2, 6)),
        ),
    )

    note = """
    ``dspark_draft_gemm.draft_gemm`` pins it for the drafter's projections at up to 32 rows (the
    tactics per (N, K) and row bound in ``_PINS`` / ``_PINS_SM107``). Changes the drafter's
    accumulation order only. The launcher validates tensors and tactic and raises ValueError. A
    process-wide compile cache holds one kernel per (dtype, tactic, PDL, bias, leading dims);
    layouts are dynamic.
    """

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor],
        out: torch.Tensor,
        pdl: bool,
        tactic: SplitKTactic,
        strided_a: bool = False,
    ) -> torch.Tensor:
        return run_splitk_dense(a, b, bias, out, pdl, tactic, strided_a=strided_a)

    def reference(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor],
        out: torch.Tensor,
        pdl: bool,
        tactic: SplitKTactic,
        strided_a: bool = False,
    ) -> torch.Tensor:
        """``out`` after the call (new): FP64 product, rounded once to ``out``'s dtype."""
        y = a.double() @ b.double()
        if bias is not None:
            y = y + bias.double()
        return y.to(out.dtype)


k3_splitk_dense = K3SplitkDense()
