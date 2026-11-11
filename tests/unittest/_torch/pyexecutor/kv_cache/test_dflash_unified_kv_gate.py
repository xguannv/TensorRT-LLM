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
"""Selection of the unified target+DFlash KV cache layout."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.model_config import KVCacheLayerSpec
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.mode import QuantMode

pytestmark = pytest.mark.cpu_only


def _model_config(*, kv_heads, head_dim, kv_quant_algo=None):
    pretrained = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=32 * head_dim,
        torch_dtype=torch.bfloat16,
    )
    quant_mode = QuantMode.from_quant_algo(kv_cache_quant_algo=kv_quant_algo)
    return SimpleNamespace(
        pretrained_config=pretrained,
        quant_config=SimpleNamespace(quant_mode=quant_mode),
    )


class _V1LikeManager:
    """A non-V2 manager: predates the appended-draft layer specs."""


def _mla_model_config(*, kv_quant_algo=None):
    """An MLA target: one K-only head per rank holding the latent."""
    pretrained = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hidden_size=8192,
        torch_dtype=torch.bfloat16,
    )
    quant_mode = QuantMode.from_quant_algo(kv_cache_quant_algo=kv_quant_algo)
    return SimpleNamespace(
        pretrained_config=pretrained,
        quant_config=SimpleNamespace(quant_mode=quant_mode),
    )


def _creator(*, is_disagg, target, draft, tp_size=1, draft_windows=None):
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(
            is_dflash=lambda: True, is_dspark=lambda: False, is_external_drafter=lambda: True
        ),
        attention_backend="TRTLLM",
        _allow_separate_draft_kv_cache=True,
        draft_len_schedule=None,
    )
    creator._draft_config = draft
    creator._mapping = SimpleNamespace(
        pp_size=1,
        cp_size=1,
        tp_size=tp_size,
        enable_attention_dp=False,
        pp_layers=lambda num_layers: list(range(num_layers)),
    )
    creator._max_seq_len = 4096
    creator._get_draft_max_attention_window = lambda max_seq_len, kv_cache_config: draft_windows
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=target, config=target.pretrained_config)
    )
    creator._kv_cache_manager_cls = KVCacheManagerV2
    creator._kv_cache_config = SimpleNamespace(max_attention_window=None)
    creator._tokens_per_block = 32
    creator._unification_max_seq_len = 4096
    creator._is_disagg = is_disagg
    creator._get_draft_kv_model_config = lambda: draft
    return creator


@pytest.mark.parametrize("is_disagg", [False, True])
def test_matching_draft_layout_unifies(is_disagg):
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert creator._should_unify_dflash_kv_cache()
    specs = creator._build_unified_dflash_layer_specs(creator._kv_cache_config)
    assert [spec.global_layer_id for spec in specs] == [0, 1, 2, 3]
    assert [spec.dtype for spec in specs] == [DataType.BF16] * 4


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("shift_label, expected_width", [(True, 7), (False, 8)])
def test_dspark_checkpoint_width_reaches_shared_cache_cost_and_construction(
    monkeypatch, is_disagg, shift_label, expected_width
):
    from tensorrt_llm._torch.pyexecutor import _util

    layout = _model_config(kv_heads=8, head_dim=128)
    layout.is_generation = True
    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout)
    creator._speculative_config.spec_dec_mode = SimpleNamespace(
        is_dflash=lambda: False, is_dspark=lambda: True, is_external_drafter=lambda: True
    )
    creator._speculative_config.max_draft_len = 7
    creator._speculative_config.draft_is_embedded_in_target = False
    creator._model_engine.model.draft_model = SimpleNamespace(
        shift_label=shift_label, reads_managed_ctx_pages=lambda tokens_per_block: True
    )
    creator._kv_cache_config = KvCacheConfig()
    creator._tokens_per_block = 32
    creator._max_batch_size = 4
    creator._max_num_tokens = 4096
    creator._max_beam_width = 1
    creator._sparse_attention_config = None
    creator._kv_connector_manager = None
    creator._execution_stream = None
    creator._disable_overlap_scheduler = True
    creator._skip_est = True
    creator._llm_args.kv_cache_config = creator._kv_cache_config
    monkeypatch.setattr(creator, "_get_model_kv_cache_manager_cls", lambda *args: KVCacheManagerV2)
    monkeypatch.setattr(creator, "_enable_kv_cache_stats", lambda: False)

    class RecordingManager(KVCacheManagerV2):
        get_cache_size_per_token = Mock(return_value=(1, 0))
        get_cache_bytes_per_request = Mock(return_value=1)
        get_cache_quota_per_request = Mock(return_value=1)

    assert creator._should_unify_dflash_kv_cache()
    assert not creator._should_create_separate_draft_kv_cache()
    creator._per_manager_cache_cost(RecordingManager, layout, include_request_cost=True)
    create_manager = Mock(return_value=SimpleNamespace(max_seq_len=4096))
    monkeypatch.setattr(_util, "_create_kv_cache_manager", create_manager)
    creator._create_kv_cache_manager()

    runtime_specs = create_manager.call_args.kwargs["layer_specs"]
    for call in (
        RecordingManager.get_cache_size_per_token,
        RecordingManager.get_cache_bytes_per_request,
        RecordingManager.get_cache_quota_per_request,
        create_manager,
    ):
        assert call.call_count == 1
        assert call.call_args.kwargs["shared_draft_block_width"] == expected_width
        assert call.call_args.kwargs["layer_specs"] == runtime_specs


def test_mismatched_draft_kv_heads_unify_only_in_aggregated_serving() -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=4, head_dim=128)

    assert _creator(is_disagg=False, target=target, draft=draft)._should_unify_dflash_kv_cache()
    assert not _creator(is_disagg=True, target=target, draft=draft)._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_mismatched_draft_head_dim_unifies(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=64)

    assert _creator(is_disagg=is_disagg, target=target, draft=draft)._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_fp8_target_with_bf16_draft_unifies(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128, kv_quant_algo=QuantAlgo.FP8)
    draft = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert creator._should_unify_dflash_kv_cache()
    specs = creator._build_unified_dflash_layer_specs(creator._kv_cache_config)
    assert [spec.dtype for spec in specs] == [DataType.FP8] * 2 + [DataType.BF16] * 2


@pytest.mark.parametrize("is_disagg", [False, True])
def test_non_v2_target_manager_keeps_separate_cache(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)
    creator._kv_cache_manager_cls = _V1LikeManager

    assert not creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("target_is_mla", [True, False])
@pytest.mark.parametrize("is_disagg", [False, True])
def test_mismatched_draft_k_only_keeps_separate_cache(is_disagg: bool, target_is_mla: bool) -> None:
    """A drafter's K-only flag never unifies, aggregated or disaggregated.

    Unlike the head-count and dtype mismatches, aggregated serving has no
    transfer step: the drafter's layers would land in the target's pool, whose
    storage holds one of the two shapes (a K-only Kimi K3 target with its GQA
    DSpark drafter), so both the mismatch directions keep a separate draft KV
    cache instead of failing the appended-layer validation at construction.
    """
    gqa = _model_config(kv_heads=1, head_dim=128)
    if target_is_mla:
        target, draft = _mla_model_config(), gqa
    else:
        target, draft = gqa, _mla_model_config()

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert not creator._should_unify_dflash_kv_cache()


def test_layout_compares_rank_local_heads():
    # 8 target heads and 2 draft heads differ per rank at every TP size; at
    # TP4 the draft is duplicated to 1 head per rank while the target has 2.
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=2, head_dim=128)
    creator = _creator(is_disagg=True, target=target, draft=draft, tp_size=4)

    assert creator._dflash_kv_layout(target, torch.bfloat16)[0] == 2
    assert creator._dflash_kv_layout(draft, torch.bfloat16)[0] == 1
    assert not creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("attention_dp", [False, True])
def test_attention_dp_changes_the_disagg_layout_decision(is_disagg, attention_dp):
    # At TP4 a 4-head target and a 1-head draft both hold one head per rank, so
    # they transfer together. Under attention DP each rank keeps every head (4
    # and 1), so the layouts differ and disaggregated serving keeps a separate cache.
    target = _model_config(kv_heads=4, head_dim=128)
    draft = _model_config(kv_heads=1, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=target, draft=draft, tp_size=4)
    creator._mapping.enable_attention_dp = attention_dp

    heads = (
        creator._dflash_kv_layout(target, torch.bfloat16)[0],
        creator._dflash_kv_layout(draft, torch.bfloat16)[0],
    )
    assert heads == ((4, 1) if attention_dp else (1, 1))
    assert creator._should_unify_dflash_kv_cache() == (not (is_disagg and attention_dp))


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("per_layer_side", ["target", "draft"])
def test_per_layer_kv_heads_keep_separate_cache(is_disagg, per_layer_side):
    uniform = _model_config(kv_heads=8, head_dim=128)
    per_layer = _model_config(kv_heads=[8, 4], head_dim=128)
    target, draft = (per_layer, uniform) if per_layer_side == "target" else (uniform, per_layer)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert not creator._should_unify_dflash_kv_cache()
    assert creator._build_unified_dflash_layer_specs(kv_cache_config=None) == ()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_embedded_dspark_draft_keeps_its_own_window(is_disagg: bool) -> None:
    """An embedded DSpark draft (DeepSeek-V4) is not a DFlashForCausalLM: the
    gate reports why it keeps its own window instead of calling hooks it lacks."""
    layout = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout)
    creator._speculative_config.spec_dec_mode = SimpleNamespace(
        is_dflash=lambda: False, is_dspark=lambda: True, is_external_drafter=lambda: False
    )
    creator._speculative_config.draft_is_embedded_in_target = True
    creator._model_engine.model.draft_model = object()

    assert creator._loaded_standalone_dspark_drafter() is None
    assert creator._dflash_unify_blocker() is not None
    assert not creator._should_unify_dflash_kv_cache()


def test_disagg_sharing_follows_the_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disaggregated serving shares whenever the loaded models allow it; the
    removed TRTLLM_DISAGG_SHARED_DRAFT_KV variable no longer opts out."""
    monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", "0")
    layout = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=True, target=layout, draft=layout)
    assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize(
    ("is_disagg", "tp_size", "shared"),
    [
        (False, 4, True),
        (True, 4, False),
        (True, 8, True),
    ],
)
def test_disagg_compares_the_draft_and_target_heads_per_rank(is_disagg, tp_size, shared):
    """8 target and 4 draft KV heads: 2 and 1 per rank at TP4, 1 and 1 at TP8.

    The KV transfer needs one rank-local layout for both; a colocated manager
    holds layers of different head counts.
    """
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=4, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft, tp_size=tp_size)
    assert creator._should_unify_dflash_kv_cache() is shared


@pytest.mark.parametrize("is_disagg", [False, True])
def test_draft_layers_with_different_windows_keep_separate_cache(is_disagg):
    layout = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout, draft_windows=[128, 4096])

    assert not creator._should_unify_dflash_kv_cache()
    assert creator._build_unified_dflash_layer_specs(kv_cache_config=None) == ()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_draft_layers_with_one_window_still_unify(is_disagg):
    layout = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout, draft_windows=[128, 128])

    assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize(
    ("tokens_per_block", "draft_tokens_per_block"),
    [(8, 32), (16, 32), (32, 32), (64, 64), (128, 32)],
)
def test_unification_requires_supported_draft_page_size(
    is_disagg: bool, tokens_per_block: int, draft_tokens_per_block: int
) -> None:
    creator = _creator(
        is_disagg=is_disagg,
        target=_model_config(kv_heads=8, head_dim=128),
        draft=_model_config(kv_heads=8, head_dim=128),
    )
    creator._tokens_per_block = tokens_per_block

    assert creator._should_unify_dflash_kv_cache() is (tokens_per_block in (16, 32, 64))
    assert creator._get_draft_tokens_per_block() == draft_tokens_per_block


@pytest.mark.parametrize("is_disagg", [False, True])
def test_variable_model_geometry_keeps_separate_cache(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    target.kv_cache_layer_specs = (
        KVCacheLayerSpec(head_dim=128, num_kv_heads=8, num_q_heads=32),
        KVCacheLayerSpec(head_dim=256, num_kv_heads=4, num_q_heads=32),
    )
    draft = _model_config(kv_heads=8, head_dim=128)

    assert not _creator(
        is_disagg=is_disagg, target=target, draft=draft
    )._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_retained_token_mixed_precision_keeps_separate_cache(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)
    creator._kv_cache_config.mixed_precision_kv = object()

    assert not creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_attention_dp_unifies_with_unsharded_heads(is_disagg):
    # Under attention DP each rank keeps every KV head, so the TP4 layouts of
    # an 8-head target and a 2-head draft are not resharded to 2 and 1.
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=target, draft=draft, tp_size=4)
    creator._mapping.enable_attention_dp = True

    assert creator._dflash_kv_layout(target, torch.bfloat16)[0] == 8
    assert (
        creator._dflash_kv_layout(_model_config(kv_heads=2, head_dim=128), torch.bfloat16)[0] == 2
    )
    assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize(
    "backend,shared",
    [("TRTLLM", True), ("VANILLA", False), ("FA4", False), (None, False)],
)
def test_disagg_worker_shares_only_with_trtllm_draft_attention(
    backend: str | None, shared: bool
) -> None:
    creator = _creator(
        is_disagg=True,
        target=_model_config(kv_heads=8, head_dim=128),
        draft=_model_config(kv_heads=8, head_dim=128),
    )
    creator._speculative_config.attention_backend = backend or "VANILLA"
    assert creator._should_unify_dflash_kv_cache() is shared


@pytest.mark.parametrize(
    "backend, tokens_per_block, reads",
    [("TRTLLM", 32, True), ("TRTLLM", 64, True), ("TRTLLM", 128, False), ("FA4", 32, False)],
)
def test_dflash_drafter_reads_managed_pages_only_at_trtllm_gen_page_sizes(
    backend: str, tokens_per_block: int, reads: bool
) -> None:
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    # GQA DSpark drafters inherit this; at 128-token pages they keep their own pool.
    drafter = SimpleNamespace(dflash_attention_backend=backend)
    assert DFlashForCausalLM.reads_managed_ctx_pages(drafter, tokens_per_block) is reads
