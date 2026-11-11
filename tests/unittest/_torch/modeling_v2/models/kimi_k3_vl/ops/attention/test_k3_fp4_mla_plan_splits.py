# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3's generation KV update plans the FP4 MLA backend's packed split plan with the decode's 8 KV splits.

The first FP4 MLA kernel of a generation step (``kernels/k3r_fp4_mla/fp4_mla_kv_append``) writes the
backend's packed split plan fused with the generation lengths. K3's packed decode always runs 8 task slots
per request (``ops/attention/_fp4_mla_decode._SPLITS``), and the reference recipes' generation servers
planned with 8 as well. The KV update hands that count to the backend's geometry
(``_fp4_mla_packed_plan_geometry(num_splits=...)``), so no environment variable decides it. Host side, with
the kernel modules stubbed.
"""

import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_REDUCE = "tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce"
_SPLIT_KV_ENV = "TRTLLM_FP4_MLA_SPLIT_KV"


def _plan_module(monkeypatch):
    from tensorrt_llm._torch.attention.backends.fp4_mla import packed_plan

    kernel = SimpleNamespace(_select_runtime_kv_profile=lambda kv: 1 << (kv - 1).bit_length())
    monkeypatch.setattr(packed_plan, "_fp4_mla_cutedsl_kernel_module", lambda: kernel)
    monkeypatch.setattr(packed_plan, "_get_sm_count", lambda device: 212)
    monkeypatch.setattr(
        packed_plan,
        "_fp4_mla_packed_rows_per_seq",
        lambda heads, query_len: (
            0 if heads * query_len > 1024 else -(-heads * query_len // 128) * 128
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        _REDUCE,
        SimpleNamespace(packed_plan_length=lambda requests, tasks: 3 + requests + 3 * tasks),
    )
    return packed_plan


def _metadata(num_requests):
    state = SimpleNamespace(
        device_page_table=True,
        page_table_stride=3904,
        generation_kv_lens=torch.zeros(num_requests, dtype=torch.int32),
        workspaces={"_fp4_mla_packed_plan": torch.empty(1 << 16, dtype=torch.int32)},
    )
    return SimpleNamespace(page_size=128, is_cuda_graph=True, fp4_mla_state=state)


def _splits(packed_plan, num_requests, num_heads, query_len, **kwargs):
    geometry = packed_plan._fp4_mla_packed_plan_geometry(
        _metadata(num_requests),
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        device=torch.device("cpu"),
        helix_kv_bounds=None,
        **kwargs,
    )
    return geometry.splits


# (requests, heads, query length): helix8 MTP7 (96 heads x 8), dep16 MTP5 (6 heads x 6) at its batch sizes.
_SHAPES = [(1, 96, 8), (2, 96, 8), (4, 96, 8), (16, 96, 8), (1, 6, 6), (8, 6, 6), (16, 6, 6)]


@pytest.mark.parametrize("env", [None, "auto", "3"])
def test_declared_splits_win_over_the_environment(monkeypatch, env):
    packed_plan = _plan_module(monkeypatch)
    if env is None:
        monkeypatch.delenv(_SPLIT_KV_ENV, raising=False)
    else:
        monkeypatch.setenv(_SPLIT_KV_ENV, env)
    for shape in _SHAPES:
        assert _splits(packed_plan, *shape, num_splits=8) == 8, shape


def test_without_a_declaration_the_selection_is_not_always_8(monkeypatch):
    """What the K3 generation servers would plan with the engine's own selection: the declaration
    matters."""
    packed_plan = _plan_module(monkeypatch)
    monkeypatch.delenv(_SPLIT_KV_ENV, raising=False)
    selected = {shape: _splits(packed_plan, *shape) for shape in _SHAPES}
    assert any(n != 8 for n in selected.values()), selected
    assert _splits(packed_plan, 1, 96, 8, num_splits=64) == 16  # capped to the kernel's slots


def test_kv_update_plans_with_the_decode_splits():
    import importlib

    decode = importlib.import_module(f"{_FAMILY}.ops.attention._fp4_mla_decode")
    append = importlib.import_module(f"{_FAMILY}.kernels.k3r_fp4_mla.fp4_mla_kv_append")
    assert decode._SPLITS == 8
    assert append._DECODE_SPLITS is decode._SPLITS
    source = inspect.getsource(append._scatter_generation_kv_only_2d)
    assert source.count("_fp4_mla_packed_plan_geometry(") == 1
    assert "num_splits=_DECODE_SPLITS" in source
