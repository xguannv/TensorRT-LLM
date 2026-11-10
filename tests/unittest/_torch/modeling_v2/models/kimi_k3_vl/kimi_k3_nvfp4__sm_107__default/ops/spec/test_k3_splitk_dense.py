# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the K3 SM107 target's k3_splitk_dense entry: the certified cells against the
entry's FP64 reference."""

import zlib

import pytest
import torch

pytest.importorskip("cutlass")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.kernels.dspark_draft_gemm_push import (  # noqa: E402, E501
    ensure_cute_runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.kernels.low_m_bf16_splitk import (  # noqa: E402, E501
    SplitKTactic,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default.ops.spec import (  # noqa: E402, E501
    k3_splitk_dense as entry,
)

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402, F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

op = entry.k3_splitk_dense


def _on_archs() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in op.ARCHS}


pytestmark = pytest.mark.skipif(not _on_archs(), reason="not an ARCHS architecture")


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS.

    One BF16 rounding on each side: the kernel sums FP32 partial products per split-K slice and
    then the slices, the reference is an FP64 product. With unit-scale operands the FP32
    accumulation over K <= 7168 is off by far less than a BF16 step of the output, so the two
    differ only where the FP32 value sits at a BF16 rounding midpoint, by one step of that
    element: at most 2 units of ``ULP[bf16]`` against the row scale. Those flips are rare enough
    that the relative RMS stays far below half an ulp. A dropped or doubled K slice or a
    transposed tile misses by O(1).
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    a = torch.randn(spec["rows"], spec["k"], device="cuda", generator=gen).to(torch.bfloat16)
    weight = (torch.randn(spec["n"], spec["k"], device="cuda", generator=gen) / 16).to(
        torch.bfloat16
    )
    before = a.clone()
    out = torch.full((spec["rows"], spec["n"]), 5.0, dtype=torch.bfloat16, device="cuda")
    ensure_cute_runtime()
    tactic = SplitKTactic(*spec["tactic"])
    result = op(a, weight.t(), None, out, True, tactic)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(a, before), "the op must not write its input"
    _compare(out, op.reference(a, weight.t(), None, out, True, tactic))
