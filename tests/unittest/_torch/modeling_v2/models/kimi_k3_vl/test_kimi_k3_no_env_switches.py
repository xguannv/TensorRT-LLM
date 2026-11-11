# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family reads no environment switch of its own.

What a target runs is in its constants (``K3Sm107Constants``), not in the environment. The only
environment variables the family code may read are the engine-wide ones every Modeling V2 family
honours (PDL, forced MNNVL all-reduce, the MNNVL test override), read as literals. The two MXFP4
sm100 targets predate the rule and are not scanned.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_FAMILY_DIR = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl"
)
_EXEMPT_TARGETS = {
    "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1",
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4",
}
ENGINE_WIDE = {"TRTLLM_ENABLE_PDL", "TRTLLM_FORCE_MNNVL_AR", "TLLM_TEST_MNNVL"}

_READ = re.compile(r"\b(?:os\.environ|os\.getenv|getenv)\b")
_LITERAL = re.compile(
    r"""(?:os\.environ\.get|os\.environ\[|os\.getenv)\(?\s*["']([A-Z0-9_]+)["']"""
)


def _reads():
    for path in sorted(_FAMILY_DIR.rglob("*.py")):
        rel = path.relative_to(_FAMILY_DIR)
        if rel.parts[0] in _EXEMPT_TARGETS or "__pycache__" in rel.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _READ.search(line):
                yield f"{rel}:{number}", line, _LITERAL.findall(line)


def test_only_engine_wide_environment_variables():
    bad = [
        f"{where}: {line.strip()}"
        for where, line, names in _reads()
        if not names or not set(names) <= ENGINE_WIDE
    ]
    assert not bad, "environment switches in the Kimi K3 family:\n" + "\n".join(bad)
