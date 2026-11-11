# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kimi_k3_nvfp4__sm_107__helix8_moeep8: its constants are 084745cde4's ``helix8`` profile (the values
932149's c24 run exported), it reads no environment
variable, and its model defaults are what its recipes run."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_TARGET = "kimi_k3_nvfp4__sm_107__helix8_moeep8"
_CLASS = "ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8"
_TARGET_DIR = Path(__file__).resolve().parents[6] / _FAMILY.replace(".", "/") / _TARGET

_EXPECTED = {
    "spec_head_fusion": True,
    "fuse_moe_residual": True,
    "fp8w_lowm_gemm": True,
    "ld_fused": True,
    "shared_sm": 16,
    "fc12_die_major": True,
    "router_exact_kda": True,
    "latent_ar_own_ws": True,
    "attn_res_pdl": True,
    "attn_res_pdl_max_m": 128,
    "attn_res_mxfp8": True,
    "attn_res_early_trigger": True,
    "kda_prepad_prefill": False,
    "kda_fused_conv_prefill": False,
    "kda_l2_prefetch": True,
    "mla_q_norm_pdl": True,
    "fp4_mla_page_plan": True,
    "fp4_mla_reduce_pdl": True,
    "spec_tail_fusion": True,
    "dspark_tail_v2": True,
    "dspark_context_kv_only": False,
    "torch_mem_safety_margin": 4 * 1024**3,
}

_MODEL_DEFAULTS = {"kv_cache_config": {"enable_block_reuse": False, "tokens_per_block": 128}}


def test_constants_are_the_reference_profile():
    import importlib

    CONSTANTS = importlib.import_module(f"{_FAMILY}.{_TARGET}.constants").CONSTANTS

    assert dataclasses.asdict(CONSTANTS) == _EXPECTED


def test_target_reads_no_environment_variable():
    readers = re.compile(r"os\.environ|getenv\(")
    hits = [
        f"{path.name}:{lineno}"
        for path in sorted(_TARGET_DIR.rglob("*.py"))
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if readers.search(line)
    ]
    assert not hits, hits


def test_target_class_declares_its_constants_and_model_defaults():
    import importlib

    module = importlib.import_module(f"{_FAMILY}.{_TARGET}.modeling")
    family = importlib.import_module(f"{_FAMILY}.sm107_parts.modeling")
    cls = getattr(module, _CLASS)
    assert issubclass(cls, family.KimiK3Sm107ForConditionalGeneration)
    assert cls.K3_CONSTANTS is importlib.import_module(f"{_FAMILY}.{_TARGET}.constants").CONSTANTS
    # The engine reads model defaults off the routed class; the text model declares none.
    assert "get_model_defaults" in vars(cls)
    assert "get_model_defaults" not in vars(family.KimiK3DefaultForCausalLM)
    assert cls.get_model_defaults(None) == _MODEL_DEFAULTS


def test_model_defaults_apply_under_the_user_config(tmp_path):
    import importlib

    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

    cls = getattr(importlib.import_module(f"{_FAMILY}.{_TARGET}.modeling"), _CLASS)
    args = TorchLlmArgs(model=str(tmp_path))
    applied = apply_model_defaults_to_llm_args(args, cls.get_model_defaults(args))
    assert applied
    assert args.kv_cache_config.tokens_per_block == 128
    assert (
        args.kv_cache_config.enable_block_reuse
        is _MODEL_DEFAULTS["kv_cache_config"]["enable_block_reuse"]
    )
    # A value the recipe sets wins over the default.
    user = not _MODEL_DEFAULTS["kv_cache_config"]["enable_block_reuse"]
    args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(enable_block_reuse=user))
    apply_model_defaults_to_llm_args(args, cls.get_model_defaults(args))
    assert args.kv_cache_config.enable_block_reuse is user
    assert args.kv_cache_config.tokens_per_block == 128
