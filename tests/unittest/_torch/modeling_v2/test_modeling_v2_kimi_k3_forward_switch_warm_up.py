# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every Kimi K3 switch whose default is -1 (on where the device is in the switch's capability list), and the
read-at-forward-time switches listed in ``NON_DEFAULT``, must say how each build any of its values selects is
compiled before a CUDA-graph capture uses it. No GPU needed.

A switch read at forward time can change between load and a capture (the gates flip switches at runtime and
recapture), so a build its new value selects may meet its first call inside a capture. ``COVERAGE`` holds, per such
switch, one entry per build family its value selects, each of one of these kinds:

* ``WARM_UP``: the model's warm-up runs every value's build, and the named unit test asserts it.
* ``LAZY_RAISE``: the build compiles on its first eager call (the CUDA-graph runner's eager forward before every
  capture makes it with the same value), and a miss under capture raises at the named site.
* ``FALLBACK_COUNTED``: a miss under capture falls back to another build with the same bits and counts it at the
  named site (a census ``fallback:*`` key, or a stats counter), which the gates require to stay 0 in an A / B
  segment, so the A / B measures the switch and not the fallback.
* ``TRITON_EAGER``: a direct Triton call, compiled by Triton on its first call with no capture check; covered only
  because its listed constexprs depend only on shapes the runner's eager forward before every capture uses
  identically, so that forward compiles the same specialization.
* ``NO_BUILD``: no value selects a separately compiled build (pure dispatch, or a build made at load).
* ``NOT_FORWARD``: read once, at load or model construction, never at forward time.
* ``PENDING``: a known gap and its owner, until the owner's fix makes it one of the kinds above.

A new -1 switch fails `test_every_default_minus_one_switch_is_covered` until it has an entry.

Every value a switch accepts must map to a registered family (`test_every_accepted_value_maps_to_a_family`). The
accepted values are flags.py's (``flags.accepted_values()``: a tuple, or a ``flags.Range`` for a count; -1 resolves to
one of them per device), every registered switch has them, and each of its entries names the values whose builds it
covers (``Values``), so a family that only a default moved per arch selects cannot go unregistered. A value outside
the set raises when the switch is read (`test_every_registered_switch_rejects_an_out_of_set_value`).
"""

import ast
import importlib
import importlib.util
import pathlib

import pytest

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MODELING_V2_TESTS = pathlib.Path(__file__).resolve().parent
_NVFP4 = "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4"

WARM_UP, LAZY_RAISE, FALLBACK_COUNTED, TRITON_EAGER, NO_BUILD, NOT_FORWARD, PENDING = (
    "warm_up",
    "lazy_raise",
    "fallback_counted",
    "triton_eager",
    "no_build",
    "not_forward",
    "pending",
)
KINDS = (WARM_UP, LAZY_RAISE, FALLBACK_COUNTED, TRITON_EAGER, NO_BUILD, NOT_FORWARD, PENDING)


class Values(tuple):
    """The switch values whose builds a ``COVERAGE`` entry covers."""


class AllValues(Values):
    """Every value the switch accepts (a count switch's whole ``flags.Range``: one build family for any count)."""


_FLAGS_PATH = _ROOT / _NVFP4 / "flags.py"
_ON_OFF = Values((0, 1))

#: Switches covered here although their default is not -1 (read at forward time, on by default).
NON_DEFAULT = ("TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE", "TRTLLM_K3LL_MLA_PARTIAL_F16")

_COUNTS_KERNELS = (
    TRITON_EAGER,
    f"{_NVFP4}/decode_mla_dcp.py::_var_split_counts_tile_kernel",
    ("B", "BLOCK_B", "GRID", "BLOCK_C", "BUDGET", "K_TILE", "TALLY_AT", "STATIC_GRID"),
    _ON_OFF,
)
_COUNTS_KERNEL = (
    TRITON_EAGER,
    f"{_NVFP4}/decode_mla_dcp.py::_var_split_counts_kernel",
    ("B", "BLOCK_B", "GRID", "BUDGET", "K_TILE"),
    _ON_OFF,
)
_V_UP = (LAZY_RAISE, f"{_NVFP4}/mla_v_up.py::not compiled before capture", _ON_OFF)

#: switch -> entries, each (kind, evidence) or (TRITON_EAGER, evidence, constexprs), followed by ``Values`` (the values
#: of ``flags.accepted_values()`` whose builds the entry covers, or ``AllValues``). WARM_UP: "<test file in
#: modeling_v2>::<test function>"; LAZY_RAISE / FALLBACK_COUNTED: "<repo-relative source file>::<text of the raise
#: message / counter in it>"; TRITON_EAGER: "<repo-relative source file>::<kernel name>" and the kernel's constexpr
#: parameters; the others: the reason (PENDING: "<owner>: <the gap>").
COVERAGE = {
    # MoE (decode_moe_lowm.py)
    "TRTLLM_K3LL_MOE_SELF_SORT": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_dyn_fast_builds",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MOE_LOWM_DYN_FAST": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_dyn_fast_builds",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MOE_SELF_SORT_FC12": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_fc2_ceil_chunk_builds",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_fc2_ceil_chunk_builds",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MOE_LOWM_FC12": (
        (
            NO_BUILD,
            "K3MoeFc12 is built at load only when on and fc12_takes requires it, so a 0 -> 1 flip without it never "
            "engages; 1 -> 0 falls back to the 32-token dyn build, which warm_up runs at 32 rows (x self-sort x fast)",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MOE_TAIL_COL": (
        (
            NO_BUILD,
            "state and layers built at load only when on; the forward takes it only where layer.col exists; the off "
            "path's kernels compile unconditionally at load (comm.compile_tail, gemvs.compile_push)",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_LAR_CLEAR_SPLIT": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_warm_up_compiles_both_latent_ar_builds",
            _ON_OFF,
        ),
    ),
    # attention residual / MX
    "TRTLLM_K3LL_ATTN_RES_SNAPSHOT_INPLACE": (
        (
            NO_BUILD,
            "no separate build: only the updated pointer changes; k3_attn_res._compiled's key has no updated / "
            "out term",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_ATTN_RES_TAP": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_warm_up_runs_the_tap_and_mxfp8_attn_res_builds",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_ATTN_RES_MXFP8": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_warm_up_runs_the_tap_and_mxfp8_attn_res_builds",
            _ON_OFF,
        ),
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_push_attn_res_compiles_at_load",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MX_SF_EARLY_RELEASE": (
        (FALLBACK_COUNTED, f"{_NVFP4}/fp8w_lowm.py::fallback:mx_not_compiled", _ON_OFF),
    ),
    # MLA
    "TRTLLM_K3LL_MLA_KV_JOIN_EARLY": (
        (
            NO_BUILD,
            "pure dispatch: where the KV side's run and join are enqueued; no kernel build depends on it",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY": (_V_UP,),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT": (_V_UP,),
    "TRTLLM_K3LL_MLA_PARTIAL_F16": (_V_UP,),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS": (_V_UP, _COUNTS_KERNELS, _COUNTS_KERNEL),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL": (_V_UP, _COUNTS_KERNELS, _COUNTS_KERNEL),
    "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED": (
        (
            TRITON_EAGER,
            f"{_NVFP4}/decode_mla_dcp.py::_helix_plan_kernel",
            ("Q_LEN", "PAGE", "CP", "RANK", "BLOCK", "Q_BLOCK"),
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE": (_V_UP, _COUNTS_KERNELS, _COUNTS_KERNEL),
    # KDA fused QKVG at bs1: splits 3 and 4 compile at load whatever the value (fp8w_lowm.compile_qkvg_bs1_splits);
    # test_modeling_v2_kimi_k3_qkvg_bs1_split.py::test_capture covers the fallback
    "TRTLLM_K3LL_QKVG_BS1_SPLIT": (
        # every value's build (split 5 default tactic for 0, forced splits 3 / 4) compiles at load
        # (fp8w_lowm.compile_qkvg_bs1_splits, recorded in _QKVG_BS1_COMPILED)
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_qkvg_bs1_split.py::test_capture",
            Values((0, 3, 4)),
        ),
        # a forced split not compiled for this device falls back to the default tactic, counted
        (
            FALLBACK_COUNTED,
            f"{_NVFP4}/fp8w_lowm.py::fallback:qkvg_bs1_split_not_compiled",
            Values((3, 4)),
        ),
    ),
    # KDA verify
    "TRTLLM_K3LL_KDA_VERIFY_HOIST": (
        (
            LAZY_RAISE,
            f"{_NVFP4}/verify.py::must run once per configuration outside CUDA-graph capture first",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_KDA_VERIFY_ODD_CLAMP": (
        (
            LAZY_RAISE,
            f"{_NVFP4}/verify.py::must run once per configuration outside CUDA-graph capture first",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS": (
        (
            LAZY_RAISE,
            f"{_NVFP4}/verify.py::must run once per configuration outside CUDA-graph capture first",
            _ON_OFF,
        ),
    ),
    # speculative decoding / drafter
    "TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE": (
        (
            LAZY_RAISE,
            "tensorrt_llm/_torch/cute_dsl_kernels/k3_spec_accept/op.py::must run once per shape outside CUDA-graph "
            "capture first",
            _ON_OFF,
        ),
    ),
    "TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT": (
        (
            LAZY_RAISE,
            "tensorrt_llm/_torch/cute_dsl_kernels/k3_spec_accept/op.py::must run once per shape outside CUDA-graph "
            "capture first",
            AllValues(()),
        ),
    ),
    "TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU": (
        (
            FALLBACK_COUNTED,
            f"{_NVFP4}/mla_drafter_gemm.py::fallback:drafter_swiglu_not_compiled",
            _ON_OFF,
        ),
    ),
    # launch configuration
    "TRTLLM_K3LL_MAXCARVE": (
        (
            NOT_FORWARD,
            "read once at import by cute_dsl_kernels/k3_carveout.py (other values than -1 / 0 / 1 raise there); "
            "k3_carveout.apply sets the kernel attribute on each vectorized AttnResFast / ctm_gemv wide build when "
            "it is compiled, before any capture; 0 and 1 compile the same kernels",
            Values((0, 1)),
        ),
    ),
    # executor
    "TRTLLM_K3LL_INPUT_CHAIN": (
        (
            NOT_FORWARD,
            "read once at model construction (input_chain.py sets the executor's input-chain defaults)",
            _ON_OFF,
        ),
    ),
}


def _flags():
    import importlib

    return importlib.import_module(f"{_PKG}.flags").flags()


def _default_minus_one():
    return sorted(name for name, (_value, default, _doc) in _flags().items() if default == -1)


def test_every_default_minus_one_switch_is_covered():
    flags = _flags()
    switches = _default_minus_one()
    missing = [name for name in switches if name not in COVERAGE]
    stale = [name for name in COVERAGE if name not in switches and name not in NON_DEFAULT]
    assert not missing, (
        f"-1 switches without a COVERAGE entry (say how each value's build is compiled before capture): {missing}"
    )
    assert not stale, (
        f"COVERAGE entries for switches that are gone or no longer default -1: {stale}"
    )
    for name in NON_DEFAULT:
        assert name in flags and name in COVERAGE, name


def _function_names(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _kernel_constexprs(path: pathlib.Path, kernel: str):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == kernel:
            args = node.args.args + node.args.kwonlyargs
            return {
                a.arg
                for a in args
                if a.annotation is not None and "constexpr" in ast.unparse(a.annotation)
            }
    return None


@pytest.mark.parametrize("name", sorted(COVERAGE))
def test_coverage_evidence_resolves(name):
    entries = COVERAGE[name]
    assert entries, name
    for entry in entries:
        kind, evidence = entry[0], entry[1]
        assert kind in KINDS, (name, kind)
        if kind == WARM_UP:
            file, func = evidence.split("::")
            path = _MODELING_V2_TESTS / file
            assert path.is_file(), (name, evidence)
            assert func in _function_names(path), (name, evidence)
        elif kind in (LAZY_RAISE, FALLBACK_COUNTED):
            file, text = evidence.split("::", 1)
            path = _ROOT / file
            assert path.is_file(), (name, evidence)
            assert text in path.read_text(), (name, evidence)
        elif kind == TRITON_EAGER:
            file, kernel = evidence.split("::")
            constexprs = _kernel_constexprs(_ROOT / file, kernel)
            assert constexprs is not None, (name, evidence)
            # the listed constexprs are the kernel's: a new one must be checked against the eager forward's shapes
            assert set(entry[2]) == constexprs, (name, kernel, sorted(constexprs))
        else:
            assert (
                len([x for x in entry[2:] if not isinstance(x, Values)]) == 0 and evidence.strip()
            ), (
                name,
                kind,
            )


def _entry_values(entry):
    for x in entry[2:]:
        if isinstance(x, Values):
            return x
    return None


def _flags_module():
    return importlib.import_module(f"{_PKG}.flags")


def test_every_registered_switch_has_accepted_values():
    accepted = _flags_module().accepted_values()
    registered = set(_default_minus_one()) | set(NON_DEFAULT)
    assert registered <= set(accepted), (
        f"switches without flags.py accepted values: {sorted(registered - set(accepted))}"
    )


@pytest.mark.parametrize("name", sorted(COVERAGE))
def test_every_accepted_value_maps_to_a_family(name):
    accepted = _flags_module().accepted_values()[name]
    is_range = isinstance(accepted, _flags_module().Range)
    covered = set()
    for entry in COVERAGE[name]:
        values = _entry_values(entry)
        assert values is not None, (
            f"{name}: every entry names the values it covers (Values or AllValues)"
        )
        if isinstance(values, AllValues):
            covered = None
            continue
        assert not is_range, f"{name}: a count switch's entries cover its whole range (AllValues)"
        assert set(values) <= set(accepted), (name, sorted(set(values) - set(accepted)))
        if covered is not None:
            covered |= set(values)
    assert covered is None or set(accepted) <= covered, (
        f"{name}: values {sorted(set(accepted) - covered)} select a family with no entry"
    )


def _load_flags_copy(monkeypatch, env):
    """A private copy of flags.py, executed with ``env`` (name -> value) and every other registered switch unset."""
    for name in _flags_module().flags():
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    spec = importlib.util.spec_from_file_location("_k3_flags_copy", _FLAGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out_of_set(default, accepted, range_type):
    """Values just outside ``accepted`` (and a typo-like 7 / -2), none of them the default."""
    if isinstance(accepted, range_type):
        candidates = {accepted.lo - 1, accepted.lo - 2}
        if accepted.hi is not None:
            candidates.add(accepted.hi + 1)
        inside = accepted.contains
    else:
        candidates = {max(accepted) + 1, min(accepted) - 1, -2, 7}
        inside = accepted.__contains__
    return sorted(v for v in candidates if v != default and not inside(v))


@pytest.mark.parametrize("name", sorted(_flags_module().accepted_values()))
def test_every_registered_switch_rejects_an_out_of_set_value(monkeypatch, name):
    flags = _flags_module()
    accepted = flags.accepted_values()[name]
    default = flags.flags()[name][1]
    bad_values = _out_of_set(default, accepted, flags.Range)
    assert bad_values, name
    for bad in bad_values:
        with pytest.raises(ValueError, match=name):
            _load_flags_copy(monkeypatch, {name: bad})
        with pytest.raises(ValueError, match=name):
            flags.check_value(name, bad)


@pytest.mark.parametrize("name", sorted(_flags_module().accepted_values()))
def test_accepted_values_and_the_default_read_unchanged(monkeypatch, name):
    """Unset, the default; set to the default or any accepted value, that value as given (what every resolver read
    before the check)."""
    flags = _flags_module()
    accepted = flags.accepted_values()[name]
    default = flags.flags()[name][1]
    assert _load_flags_copy(monkeypatch, {}).flags()[name][0] == default
    if isinstance(accepted, flags.Range):
        values = [
            accepted.lo,
            accepted.lo + 1,
            accepted.hi if accepted.hi is not None else accepted.lo + 8,
        ]
    else:
        values = list(accepted)
    for value in [default, *values]:
        assert getattr(_load_flags_copy(monkeypatch, {name: value}), name) == value
        flags.check_value(name, value)


def test_unset_environment_reads_every_default(monkeypatch):
    copy = _load_flags_copy(monkeypatch, {})
    for name, (value, default, _doc) in copy.flags().items():
        assert value == default, name
