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

Every value a switch accepts must map to a registered family (`test_every_accepted_value_maps_to_a_family`): a
binary switch's entries cover 0 and 1 (-1 resolves to one of them per device); a switch in ``ACCEPTED_VALUES`` takes
more values, and each of its entries names the values whose builds it covers (``Values``), so a family that only a
default moved per arch selects cannot go unregistered.
"""

import ast
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
    """The switch values whose builds a ``COVERAGE`` entry covers (an entry without one covers every value)."""


#: Switches that accept more than 0 / 1, and the values they accept.
ACCEPTED_VALUES = {
    "TRTLLM_K3LL_QKVG_BS1_SPLIT": (0, 3, 4),
}

#: Switches covered here although their default is not -1 (read at forward time, on by default).
NON_DEFAULT = ("TRTLLM_K3LL_MLA_VAR_SPLIT_COUNTS_TILE",)

_COUNTS_KERNELS = (
    TRITON_EAGER,
    f"{_NVFP4}/decode_mla_dcp.py::_var_split_counts_tile_kernel",
    ("B", "BLOCK_B", "GRID", "BLOCK_C", "BUDGET", "K_TILE", "TALLY_AT", "STATIC_GRID"),
)
_COUNTS_KERNEL = (
    TRITON_EAGER,
    f"{_NVFP4}/decode_mla_dcp.py::_var_split_counts_kernel",
    ("B", "BLOCK_B", "GRID", "BUDGET", "K_TILE"),
)
_V_UP = (LAZY_RAISE, f"{_NVFP4}/mla_v_up.py::not compiled before capture")

#: switch -> entries, each (kind, evidence) or (TRITON_EAGER, evidence, constexprs), optionally followed by ``Values``
#: (required for every entry of an ``ACCEPTED_VALUES`` switch). WARM_UP: "<test file in
#: modeling_v2>::<test function>"; LAZY_RAISE / FALLBACK_COUNTED: "<repo-relative source file>::<text of the raise
#: message / counter in it>"; TRITON_EAGER: "<repo-relative source file>::<kernel name>" and the kernel's constexpr
#: parameters; the others: the reason (PENDING: "<owner>: <the gap>").
COVERAGE = {
    # MoE (decode_moe_lowm.py)
    "TRTLLM_K3LL_MOE_SELF_SORT": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_dyn_fast_builds",
        ),
    ),
    "TRTLLM_K3LL_MOE_LOWM_DYN_FAST": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_dyn_fast_builds",
        ),
    ),
    "TRTLLM_K3LL_MOE_SELF_SORT_FC12": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_fc2_ceil_chunk_builds",
        ),
    ),
    "TRTLLM_K3LL_MOE_FC2_CEIL_CHUNK": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_self_sort.py::test_warm_up_compiles_both_fc2_ceil_chunk_builds",
        ),
    ),
    "TRTLLM_K3LL_MOE_LOWM_FC12": (
        (
            NO_BUILD,
            "K3MoeFc12 is built at load only when on and fc12_takes requires it, so a 0 -> 1 flip without it never "
            "engages; 1 -> 0 falls back to the 32-token dyn build, which warm_up runs at 32 rows (x self-sort x fast)",
        ),
    ),
    "TRTLLM_K3LL_MOE_TAIL_COL": (
        (
            NO_BUILD,
            "state and layers built at load only when on; the forward takes it only where layer.col exists; the off "
            "path's kernels compile unconditionally at load (comm.compile_tail, gemvs.compile_push)",
        ),
    ),
    "TRTLLM_K3LL_LAR_CLEAR_SPLIT": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_warm_up_compiles_both_latent_ar_builds",
        ),
    ),
    # attention residual / MX
    "TRTLLM_K3LL_ATTN_RES_SNAPSHOT_INPLACE": (
        (
            NO_BUILD,
            "no separate build: only the updated pointer changes; k3_attn_res._compiled's key has no updated / "
            "out term",
        ),
    ),
    "TRTLLM_K3LL_ATTN_RES_TAP": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_warm_up_runs_the_tap_and_mxfp8_attn_res_builds",
        ),
    ),
    "TRTLLM_K3LL_ATTN_RES_MXFP8": (
        (
            WARM_UP,
            "test_modeling_v2_kimi_k3_moe_tail_col.py::test_warm_up_runs_the_tap_and_mxfp8_attn_res_builds",
        ),
        (WARM_UP, "test_modeling_v2_kimi_k3_moe_tail_col.py::test_push_attn_res_compiles_at_load"),
    ),
    "TRTLLM_K3LL_MX_SF_EARLY_RELEASE": (
        (FALLBACK_COUNTED, f"{_NVFP4}/fp8w_lowm.py::fallback:mx_not_compiled"),
    ),
    # MLA
    "TRTLLM_K3LL_MLA_KV_JOIN_EARLY": (
        (
            NO_BUILD,
            "pure dispatch: where the KV side's run and join are enqueued; no kernel build depends on it",
        ),
    ),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_SKIP_EMPTY": (_V_UP,),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_FLAT": (_V_UP,),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_WIDE_WS": (_V_UP, _COUNTS_KERNELS, _COUNTS_KERNEL),
    "TRTLLM_K3LL_MLA_VAR_SPLIT_SMALL": (_V_UP, _COUNTS_KERNELS, _COUNTS_KERNEL),
    "TRTLLM_K3LL_MLA_HELIX_PLAN_FUSED": (
        (
            TRITON_EAGER,
            f"{_NVFP4}/decode_mla_dcp.py::_helix_plan_kernel",
            ("Q_LEN", "PAGE", "CP", "RANK", "BLOCK", "Q_BLOCK"),
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
        ),
    ),
    "TRTLLM_K3LL_KDA_VERIFY_ODD_CLAMP": (
        (
            LAZY_RAISE,
            f"{_NVFP4}/verify.py::must run once per configuration outside CUDA-graph capture first",
        ),
    ),
    "TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS": (
        (
            LAZY_RAISE,
            f"{_NVFP4}/verify.py::must run once per configuration outside CUDA-graph capture first",
        ),
    ),
    # speculative decoding / drafter
    "TRTLLM_K3LL_SPEC_ACCEPT_WIDE_TABLE": (
        (
            LAZY_RAISE,
            "tensorrt_llm/_torch/cute_dsl_kernels/k3_spec_accept/op.py::must run once per shape outside CUDA-graph "
            "capture first",
        ),
    ),
    "TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT": (
        (
            LAZY_RAISE,
            "tensorrt_llm/_torch/cute_dsl_kernels/k3_spec_accept/op.py::must run once per shape outside CUDA-graph "
            "capture first",
        ),
    ),
    "TRTLLM_K3LL_DRAFTER_GATE_UP_SWIGLU": (
        (
            FALLBACK_COUNTED,
            f"{_NVFP4}/mla_drafter_gemm.py::fallback:drafter_swiglu_not_compiled",
        ),
    ),
    # executor
    "TRTLLM_K3LL_INPUT_CHAIN": (
        (
            NOT_FORWARD,
            "read once at model construction (input_chain.py sets the executor's input-chain defaults)",
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
            return set(x)
    return None


@pytest.mark.parametrize("name", sorted(COVERAGE))
def test_every_accepted_value_maps_to_a_family(name):
    accepted = set(ACCEPTED_VALUES.get(name, (0, 1)))
    covered = set()
    for entry in COVERAGE[name]:
        values = _entry_values(entry)
        if name in ACCEPTED_VALUES:
            assert values is not None, (
                f"{name}: every entry of a multi-valued switch names the values it covers"
            )
        if values is None:
            values = accepted
        assert values <= accepted, (name, sorted(values - accepted))
        covered |= values
    assert accepted <= covered, (
        f"{name}: values {sorted(accepted - covered)} select a family with no entry"
    )


def test_accepted_values_include_the_default():
    flags = _flags()
    for name, accepted in ACCEPTED_VALUES.items():
        assert name in flags and name in COVERAGE, name
        default = flags[name][1]
        assert default == -1 or default in accepted, (name, default)
