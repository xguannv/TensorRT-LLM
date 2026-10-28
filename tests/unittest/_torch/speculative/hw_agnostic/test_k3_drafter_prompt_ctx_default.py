# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_DRAFTER_PROMPT_CTX``: its spelling, its default and where it is resolved (CPU only).

* Registry: 0 (the default, off), -1, 1 and a window W >= 2 are accepted; -2 is rejected.
* One resolver (``k3ll_flags.resolve_drafter_prompt_window``): -1 is W = 16384 on SM103 only (off on SM100 / SM107),
  1 is 16384 on any arch, W is W, 0 is 0.
* Serving (``serve._carry_disagg_role_to_model_ranks``): only a server with a disaggregated role and a DSpark
  ``speculative_config`` is touched. On (-1 on SM103, 1, W): the role reaches the model ranks. Off (0, the default,
  or -1 off SM103) on a context server: the DSpark ``speculative_config`` is dropped, so the context server's
  ``llm_args`` are exactly those of one without it -- no drafter, no draft KV cache manager, the same KV cache
  manager arguments. A context server without a DSpark ``speculative_config`` is untouched at any value.
* Transceiver wiring (``_util._drafter_prompt_window_transfer_kwargs``): off, no draft argument, so the transceiver
  is built as without the switch; at -1 a deployment that cannot take part stays off rather than raising.
* Context draft pool (``KvCacheCreator._drafter_prompt_tail_window``): tail-only (a W + 1 sliding window) exactly
  where the capture-only drafter runs on a joint-reuse KVCacheManagerV2 pair.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import k3ll_flags

NAME = "TRTLLM_K3LL_DRAFTER_PROMPT_CTX"
SM100, SM103, SM107 = (10, 0), (10, 3), (10, 7)
_FLAGS_PATH = pathlib.Path(k3ll_flags.__file__).parents[1] / (
    "_experimental/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4/flags.py"
)
DSPARK = {
    "decoding_type": "DSpark",
    "block_size": 7,
    "max_draft_len": 7,
    "speculative_model": "/models/Kimi-K3-DSpark",
}

pytestmark = pytest.mark.cpu_only


def _flags():
    return importlib.import_module(k3ll_flags.K3LL_FLAGS_MODULE)


def _load_flags_copy(monkeypatch, value):
    """A private copy of flags.py read with the switch at ``value`` (None: unset)."""
    if value is None:
        monkeypatch.delenv(NAME, raising=False)
    else:
        monkeypatch.setenv(NAME, str(value))
    spec = importlib.util.spec_from_file_location("_k3_flags_drafter_ctx", _FLAGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _switch(monkeypatch, value):
    """Set the switch as the flags module read it (it is read once, at import)."""
    monkeypatch.setattr(k3ll_flags, "k3ll_flag_any_target", lambda name: value, raising=True)


def test_registry_takes_minus_one_zero_one_and_a_window(monkeypatch):
    flags = _flags()
    _, default, doc = flags.flags()[NAME]
    assert default == 0
    assert flags.accepted_values()[NAME] == flags.Range(-1)
    assert "-1: W = 16384 on SM103 only" in doc and "0 (default): off" in doc
    assert getattr(_load_flags_copy(monkeypatch, None), NAME) == 0
    for value in (-1, 0, 1, 2, 16384, 1048576):
        flags.check_value(NAME, value)
        assert getattr(_load_flags_copy(monkeypatch, value), NAME) == value
    with pytest.raises(ValueError, match=NAME):
        flags.check_value(NAME, -2)
    with pytest.raises(ValueError, match=NAME):
        _load_flags_copy(monkeypatch, -2)


@pytest.mark.parametrize(
    "value,capability,window",
    [
        (-1, SM103, 16384),
        (-1, SM100, 0),
        (-1, SM107, 0),
        (-1, None, 0),
        (1, SM103, 16384),
        (1, SM100, 16384),
        (1, SM107, 16384),
        (8192, SM107, 8192),
        (16384, SM103, 16384),
        (2, SM100, 2),
        (0, SM103, 0),
        (0, SM107, 0),
    ],
)
def test_resolution(value, capability, window):
    assert k3ll_flags.resolve_drafter_prompt_window(value, capability) == window


def test_resolution_rejects_below_minus_one():
    with pytest.raises(ValueError, match=NAME):
        k3ll_flags.resolve_drafter_prompt_window(-2, SM103)


def test_resolver_reads_the_device_only_at_the_default(monkeypatch):
    reads = []
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: reads.append(1) or SM103)
    _switch(monkeypatch, 4096)
    assert k3ll_flags.drafter_prompt_window() == 4096 and not reads
    _switch(monkeypatch, -1)
    assert k3ll_flags.drafter_prompt_window() == 16384 and reads == [1]
    assert k3ll_flags.drafter_prompt_window(SM107) == 0 and reads == [1]


def _serve(monkeypatch, role, value, args, capability):
    from tensorrt_llm.commands import serve

    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    _switch(monkeypatch, value)
    args = dict(args)
    serve._carry_disagg_role_to_model_ranks(args, capability)
    return args


@pytest.mark.parametrize(
    "value,capability",
    [(-1, SM103), (1, SM107), (16384, SM100)],
    ids=["minus1-SM103", "1-SM107", "W-SM100"],
)
def test_serve_on_carries_the_role_and_keeps_the_spec(monkeypatch, value, capability):
    for role in ("context", "generation"):
        args = _serve(
            monkeypatch,
            role,
            value,
            {"speculative_config": DSPARK, "env_overrides": None},
            capability,
        )
        assert args == {"speculative_config": DSPARK, "env_overrides": {"TRTLLM_DISAGG_ROLE": role}}
    # A role already in env_overrides is kept.
    args = _serve(
        monkeypatch,
        "context",
        value,
        {"speculative_config": DSPARK, "env_overrides": {"A": "1", "TRTLLM_DISAGG_ROLE": "x"}},
        capability,
    )
    assert args["env_overrides"] == {"A": "1", "TRTLLM_DISAGG_ROLE": "x"}


@pytest.mark.parametrize(
    "value,capability",
    [(0, SM103), (0, SM107), (-1, SM100), (-1, SM107)],
    ids=["0-SM103", "0-SM107", "minus1-SM100", "minus1-SM107"],
)
def test_serve_off_context_equals_a_context_without_the_spec(monkeypatch, value, capability):
    """0 (the default; or -1 off SM103) with a DSpark spec on a context server: the server's llm_args equal
    those of a context server without the spec config and the switch unset -- what KvCacheCreator, the drafter
    build and the transceiver wiring then see is the same (no drafter, no draft KV cache manager, the same KV
    cache manager arguments)."""
    base = {"model": "/models/Kimi-K3-NVFP4", "kv_cache_config": {"free_gpu_memory_fraction": 0.67}}
    off = _serve(monkeypatch, "context", value, {**base, "speculative_config": DSPARK}, capability)
    reference = _serve(monkeypatch, "context", -1, base, SM103)
    assert off == reference == base


def test_serve_off_generation_keeps_its_drafter(monkeypatch):
    """Off on a generation server: its own drafter stays (generated tokens only), no role is carried."""
    args = {"speculative_config": DSPARK, "env_overrides": None}
    for value, capability in ((0, SM103), (-1, SM107)):
        assert _serve(monkeypatch, "generation", value, args, capability) == args


def test_serve_without_a_dspark_spec_or_role_is_untouched(monkeypatch):
    """A context server without a DSpark speculative_config is today's context at any value, 0 (the default)
    included; so is a server without a disaggregated role (aggregated serving)."""
    eagle = {"decoding_type": "Eagle3", "max_draft_len": 3}
    for value in (-1, 0, 1, 16384):
        for args in ({}, {"env_overrides": None}, {"speculative_config": eagle}):
            assert _serve(monkeypatch, "context", value, args, SM103) == args
        assert _serve(monkeypatch, None, value, {"speculative_config": DSPARK}, SM103) == {
            "speculative_config": DSPARK
        }


def test_serve_recognizes_a_dspark_config_object():
    from tensorrt_llm.commands import serve

    assert serve._is_dspark_speculative_config(SimpleNamespace(decoding_type="DSpark"))
    assert not serve._is_dspark_speculative_config(SimpleNamespace(decoding_type="MTP"))
    assert not serve._is_dspark_speculative_config(None)


class _Renumber:
    row_elems = 576


def _wiring(monkeypatch, value, capability, drafter, draft_manager, runtime="PYTHON", pp_size=1):
    from tensorrt_llm._torch.pyexecutor import _util

    _switch(monkeypatch, value)
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: capability)
    return _util._drafter_prompt_window_transfer_kwargs(
        draft_manager,
        object(),
        SimpleNamespace(backend="NIXL", transceiver_runtime=runtime),
        SimpleNamespace(model=SimpleNamespace(draft_model=drafter)),
        SimpleNamespace(pp_size=pp_size),
    )


def _v2_draft_manager():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    return KVCacheManagerV2.__new__(KVCacheManagerV2)


def test_wiring_off_builds_the_transceiver_without_the_switch(monkeypatch):
    drafter = SimpleNamespace(context_renumber=_Renumber, CAPTURE_ONLY=True)
    for value, capability in ((0, SM103), (-1, SM100), (-1, SM107)):
        draft = _v2_draft_manager()
        assert _wiring(monkeypatch, value, capability, drafter, draft) == {}
        assert draft.drafter_prompt_window == 0
    # No transceiver, or no separate draft manager: nothing either.
    from tensorrt_llm._torch.pyexecutor import _util

    _switch(monkeypatch, 16384)
    target = object()
    assert _util._drafter_prompt_window_transfer_kwargs(None, target, None, None, None) == {}
    assert _util._drafter_prompt_window_transfer_kwargs(target, target, None, None, None) == {}


@pytest.mark.parametrize("value", [-1, 1, 16384])
def test_wiring_on(monkeypatch, value):
    drafter = SimpleNamespace(context_renumber=_Renumber, CAPTURE_ONLY=True)
    draft = _v2_draft_manager()
    kwargs = _wiring(monkeypatch, value, SM103, drafter, draft)
    assert kwargs["draft_kv_cache_manager"] is draft
    assert kwargs["draft_prompt_window"] == draft.drafter_prompt_window == 16384
    assert isinstance(kwargs["draft_context_renumber"], _Renumber)
    assert kwargs["draft_capture_only"] is True


def test_wiring_default_stays_off_where_it_cannot_take_part(monkeypatch):
    """At -1, a drafter that is not the MLA DSpark drafter, a non-V2 draft manager, the C++ transceiver
    or PP > 1 keep the switch off (other speculative models on SM103 are unchanged); set explicitly, the missing
    context_renumber raises."""
    mla = SimpleNamespace(context_renumber=_Renumber)
    assert _wiring(monkeypatch, -1, SM103, SimpleNamespace(), _v2_draft_manager()) == {}
    assert _wiring(monkeypatch, -1, SM103, mla, object()) == {}
    assert _wiring(monkeypatch, -1, SM103, mla, _v2_draft_manager(), runtime="CPP") == {}
    assert _wiring(monkeypatch, -1, SM103, mla, _v2_draft_manager(), pp_size=2) == {}
    with pytest.raises(ValueError, match="context_renumber"):
        _wiring(monkeypatch, 16384, SM103, SimpleNamespace(), _v2_draft_manager())


def test_capture_only_build_follows_the_resolved_window(monkeypatch):
    from tensorrt_llm._torch.models import modeling_dspark

    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: SM103)
    _switch(monkeypatch, -1)
    assert modeling_dspark.dspark_capture_only()
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: SM107)
    assert not modeling_dspark.dspark_capture_only()
    _switch(monkeypatch, 0)
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: SM103)
    assert not modeling_dspark.dspark_capture_only()


def _tail_creator(monkeypatch, *, v2=True, joint=True, dspark=True, embedded=False, mla=True):
    """A KvCacheCreator carrying only what ``_drafter_prompt_tail_window`` reads."""
    from tensorrt_llm._torch.pyexecutor import _util

    creator = _util.KvCacheCreator.__new__(_util.KvCacheCreator)
    creator._is_kv_cache_manager_v2 = v2
    creator._joint_kv_cache_reuse = joint
    creator._tokens_per_block = 32
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: dspark),
        draft_is_embedded_in_target=embedded,
    )
    creator._get_effective_draft_config = lambda: SimpleNamespace(pretrained_config=object())
    monkeypatch.setattr(_util.config_utils, "is_mla", lambda config: mla)
    return creator


def test_context_draft_pool_is_tail_only_where_the_capture_only_drafter_runs(monkeypatch):
    """The capture-only context drafter's joint V2 draft pool is a W + 1 sliding window (W plus at most one page
    per request); every other draft pool keeps its full length."""
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: SM103)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    _switch(monkeypatch, -1)
    assert _tail_creator(monkeypatch)._drafter_prompt_tail_window(1 << 20) == 16384 + 1
    _switch(monkeypatch, 4096)
    assert _tail_creator(monkeypatch)._drafter_prompt_tail_window(1 << 20) == 4096 + 1
    # The window reaching max_seq_len is a full-length pool.
    assert _tail_creator(monkeypatch)._drafter_prompt_tail_window(4097) == 0
    for kwargs in (
        dict(v2=False),
        dict(joint=False),
        dict(dspark=False),
        dict(embedded=True),
        dict(mla=False),
    ):
        assert _tail_creator(monkeypatch, **kwargs)._drafter_prompt_tail_window(1 << 20) == 0, (
            kwargs
        )
    # Off, or not a context worker: no capture-only drafter, a full-length pool.
    _switch(monkeypatch, 0)
    assert _tail_creator(monkeypatch)._drafter_prompt_tail_window(1 << 20) == 0
    _switch(monkeypatch, -1)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    assert _tail_creator(monkeypatch)._drafter_prompt_tail_window(1 << 20) == 0


def _draft_cost_kwargs(monkeypatch, creator):
    """The keyword arguments ``_get_draft_cache_cost`` costs the external drafter's pool with."""
    from tensorrt_llm._torch.pyexecutor import _util

    recorded = {}
    creator._draft_model_engine = None
    creator._mapping = SimpleNamespace(has_cp_helix=lambda: False)
    creator._max_seq_len = 1 << 20
    creator._is_disagg = True
    creator._cache_transceiver_config = None
    creator._speculative_config.spec_dec_mode.is_external_drafter = lambda: True
    creator._get_draft_kv_model_config = lambda: SimpleNamespace(pretrained_config=object())
    creator._get_one_model_draft_kv_cache_config = lambda config, max_seq_len: config
    creator._validate_or_fallback_kv_cache_manager_v2 = lambda cls, *args: cls
    monkeypatch.setattr(_util, "get_kv_cache_manager_cls", lambda *args, **kwargs: object)

    def cost(manager_cls, model_config, kv_cache_config, **kwargs):
        recorded.update(kwargs)
        return _util.CacheCost.from_raw((1, 1))

    creator._per_manager_cache_cost = cost
    creator._get_draft_cache_cost(
        object(), use_separate_draft_kv_cache=True, include_request_cost=True
    )
    return recorded


def test_tail_only_draft_pool_is_budgeted_by_its_window(monkeypatch):
    """The budget split's per-request estimates cost the tail-only context draft pool by its window
    (``draft_retention_windowed``), not as a full-length mirror; every other draft pool keeps the
    full-length estimate."""
    monkeypatch.setattr(k3ll_flags, "_device_capability", lambda: SM103)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    _switch(monkeypatch, -1)
    assert _draft_cost_kwargs(monkeypatch, _tail_creator(monkeypatch)).get(
        "draft_retention_windowed"
    )
    _switch(monkeypatch, 0)
    assert "draft_retention_windowed" not in _draft_cost_kwargs(
        monkeypatch, _tail_creator(monkeypatch)
    )
    _switch(monkeypatch, -1)
    assert "draft_retention_windowed" not in _draft_cost_kwargs(
        monkeypatch, _tail_creator(monkeypatch, joint=False)
    )
