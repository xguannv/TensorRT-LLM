# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-deployment feature profiles of the Kimi K3 SM107 target (``features.py``)."""

import dataclasses
import subprocess
import sys
import textwrap
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__default import (
    features,
)

pytestmark = pytest.mark.cpu_only

_TARGETS = {"tp1", "helix8", "helix16_moetp4ep4", "dep16"}


@pytest.fixture
def fresh(monkeypatch):
    """A process in which no profile is selected and no other module of the target is imported."""
    monkeypatch.setattr(features, "_ACTIVE", features.K3Features())
    monkeypatch.setattr(features, "_SELECTED", None)
    monkeypatch.setattr(features, "_PACKAGE", "k3_features_test_no_such_package")
    warnings = []
    from tensorrt_llm.logger import logger

    monkeypatch.setattr(logger, "warning", lambda msg, *args, **kwargs: warnings.append(msg))
    return warnings


def test_one_profile_per_routing_target():
    assert set(features.PROFILES) == _TARGETS
    assert all(isinstance(p, features.K3Features) for p in features.PROFILES.values())


def test_every_field_replaces_one_retired_variable():
    fields = {f.name for f in dataclasses.fields(features.K3Features)}
    mapped = [f for f in features.RETIRED_ENV.values() if f is not None]
    assert sorted(mapped) == sorted(fields)


def test_context_profile_only_sets_context_features():
    assert set(features.PROFILES["tp1"].changed()) == {
        "kda_fused_conv_prefill",
        "kda_prepad_prefill",
    }


def test_every_field_is_set_by_some_target_and_left_off_by_another():
    for f in dataclasses.fields(features.K3Features):
        values = {repr(getattr(p, f.name)) for p in features.PROFILES.values()}
        assert len(values) > 1, f"{f.name}: every target uses {values}; the code should not ask"


def test_active_is_the_plain_path_before_selection(fresh):
    assert features.active() == features.K3Features()
    assert features.active().changed() == {}
    assert features.selected() is None


def test_select_is_once_per_process(fresh):
    assert features.select("helix8") is features.PROFILES["helix8"]
    assert features.active() is features.PROFILES["helix8"]
    assert features.selected() == "helix8"
    assert features.select("helix8") is features.PROFILES["helix8"]
    with pytest.raises(RuntimeError, match="helix8"):
        features.select("dep16")


def test_select_rejects_unknown_targets(fresh):
    with pytest.raises(KeyError):
        features.select("tp8")


def test_select_after_a_k3_rubin_import_raises(fresh, monkeypatch):
    monkeypatch.setitem(sys.modules, f"{features._PACKAGE}.modeling", types.ModuleType("modeling"))
    with pytest.raises(RuntimeError, match="modeling"):
        features.select("tp1")
    assert features.selected() is None


def test_select_warns_about_retired_variables(fresh, monkeypatch):
    for name in features.RETIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    features.select("dep16")
    assert fresh == []
    monkeypatch.setattr(features, "_SELECTED", None)
    monkeypatch.setenv("TRTLLM_K3_ROUTE_TAIL", "1")
    features.select("dep16")
    assert len(fresh) == 1 and "TRTLLM_K3_ROUTE_TAIL" in fresh[0]


def test_select_names_the_variable_main_reads_for_a_renamed_one(fresh, monkeypatch):
    """The helix recipes set the MNNVL one-shot limit under a name nothing reads any more; the
    warning names the variable main's all-reduce reads instead."""
    for name in features.RETIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES", "2097152")
    features.select("helix8")
    assert len(fresh) == 1
    assert "TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES (set TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES" in fresh[0]


# What the routing does, in a process of its own: select the deployment's profile, then import the target.
_PACKAGE = features.__name__.rpartition(".")[0]
# The target's modules that read the profile at import (each holds the profile as ``_F``).
_PROFILE_READERS = ("modeling", "moe_runtime", "attn_res", "fp8_weight_read")


def _python(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]
    return result.stdout


@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_select_then_import_the_target(target):
    """``select`` and then ``modeling``, as ``route()`` does: the modules import without a cycle, every
    module that reads the profile at import holds the selected one, and each weight-loading mixin's
    ``load_weights`` is the one its class resolves."""
    out = _python(
        f"""
        import importlib
        from {_PACKAGE} import features

        profile = features.select({target!r})
        modeling = importlib.import_module("{_PACKAGE}.modeling")
        weights = importlib.import_module("{_PACKAGE}.weights")
        for name in {_PROFILE_READERS!r}:
            module = importlib.import_module("{_PACKAGE}." + name)
            assert module._F is profile, name
        assert importlib.import_module("{_PACKAGE}.moe_runtime")._K3_SHARED_SM == profile.shared_sm
        target_cls, text_cls = modeling.ModelingV2KimiK3Nvfp4Sm107Default, modeling.KimiK3DefaultForCausalLM
        assert target_cls.__mro__[1] is weights.KimiK3DefaultWeights
        assert text_cls.__mro__[1] is weights.KimiK3DefaultTextWeights
        assert target_cls.load_weights is weights.KimiK3DefaultWeights.load_weights
        assert text_cls.load_weights is weights.KimiK3DefaultTextWeights.load_weights
        print("imported", features.selected())
        """
    )
    assert f"imported {target}" in out


@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_import_before_select_is_rejected(target):
    """Importing ``modeling`` first would leave its import-time settings at the plain path's: the
    profile's selection must refuse such a process."""
    out = _python(
        f"""
        import importlib
        from {_PACKAGE} import features

        importlib.import_module("{_PACKAGE}.modeling")
        try:
            features.select({target!r})
        except RuntimeError as error:
            assert "{_PACKAGE}.modeling" in str(error), error
            assert features.selected() is None
            print("rejected")
        """
    )
    assert "rejected" in out
