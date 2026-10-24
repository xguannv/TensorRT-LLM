# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MOE_TAIL_COL`` (``kimi_k3_nvfp4/moe_tail_col.py``) on one process: the switch's default and doc, the
slice geometry and the column GEMV's config, the engagement decision and the wide MoE step's dispatch (and its
precedence over ``TRTLLM_K3LL_MOE_TAIL_LU``) with stand-ins, no GPU needed. The collectives and the error bound are
checked by ``comm/_kimi_k3_moe_tail_col_op_matrix.py`` and ``comm/_kimi_k3_lowm_switches_op_matrix.py``.
"""

from types import SimpleNamespace

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{_PKG}.{name}")


def test_switch_default_sm103_and_documented():
    flags = _mod("flags")
    value, default, doc = flags.flags()["TRTLLM_K3LL_MOE_TAIL_COL"]
    assert default == -1
    assert (
        "Lossless" in doc
        and "MB per layer" in doc
        and "TRTLLM_K3LL_MOE_TAIL_LU" in doc
        and "SM103" in doc
    )
    assert _mod("moe_tail_col").MOE_TAIL_COL_CAPABILITIES == ((10, 3),)
    # The lossy replicated tail is off by default: this switch takes its steps.
    assert flags.flags()["TRTLLM_K3LL_MOE_TAIL_LU"][1] == 0


@pytest.mark.parametrize(
    "value,capability,on",
    [
        (0, (10, 3), False),
        (1, (10, 3), True),
        (2, (10, 7), True),
        (1, (10, 7), True),
        (-1, (10, 3), True),
        (-1, (10, 7), False),
        (-1, (10, 0), False),
        (-1, (9, 0), False),
    ],
)
def test_enabled_value(monkeypatch, value, capability, on):
    col = _mod("moe_tail_col")
    monkeypatch.setattr(col._flags, "TRTLLM_K3LL_MOE_TAIL_COL", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    col._on_by_default.cache_clear()
    try:
        assert col.enabled() is on
    finally:
        col._on_by_default.cache_clear()


def test_slice_geometry():
    col = _mod("moe_tail_col")
    assert col.slice_width(16) == 448 and col.slice_width(8) == 896 and col.slice_width(4) == 1792
    # TP16: 7 row tiles of 64 rows (one cluster each); other widths in tiles of 64 too, else 128-row tiles.
    assert col.row_tiles(448) == 7 and col.row_tiles(896) == 14 and col.row_tiles(1792) == 28
    assert col.row_tiles(440) == 0


def test_col_publish_config():
    """The column GEMV's config at TP16 on 148 SMs: 7 row tiles x 8-CTA clusters (56 CTAs, 3.2 MB of weight), a
    3-stage weight ring (28 k-tiles over 8 ranks: 3 or 4 each, the rest prefetched to L2 before the grid wait)."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op

    assert op.col_publish_config(448, 3584, 16, 148, 7, 7168) == (8, 3, 3, 16)
    assert op.col_publish_config(448, 3584, 9, 148, 7, 7168) == (8, 3, 3, 16)
    assert op.col_publish_config(448, 3584, 32, 148, 7, 7168) == (8, 3, 3, 32)
    # 128-row tiles: 4 clusters (the last one half empty).
    assert op.col_publish_config(448, 3584, 32, 148, 0, 7168) == (8, 3, 3, 32)
    # Tiles of 34 rows are not whole 16-byte vectors; a slice wider than the row; too many tokens.
    assert op.col_publish_config(136, 7168, 16, 148, 4, 7168) is None
    assert op.col_publish_config(448, 3584, 16, 148, 7, 256) is None
    assert op.col_publish_config(448, 3584, 65, 148, 7, 7168) is None


class _Gemvs:
    def __init__(self, ok: bool):
        self.ok, self.asked = ok, []

    def takes_push_rows(self, site, rows, width, weight, world):
        self.asked.append((site, rows, width, world))
        return self.ok


def _state(ready=True, tiles=(16, 32), buffer=4 << 20, push_buffer=4 << 20, world=16):
    latent = SimpleNamespace(world=world, bytes_per_buffer=buffer)
    push = SimpleNamespace(world=world, bytes_per_buffer=push_buffer)
    return SimpleNamespace(
        ready=ready, tiles=frozenset(tiles), base=SimpleNamespace(latent=latent, push=push)
    )


def test_takes_decision(monkeypatch):
    """The decision reads the switch, the step's rows and load-time state only (before the front runs)."""
    pytest.importorskip("cutlass")
    col = _mod("moe_tail_col")
    layer = SimpleNamespace(shared_down=torch.zeros(1))
    monkeypatch.setattr(col._flags, "TRTLLM_K3LL_MOE_TAIL_COL", 0)
    assert not col.takes(_state(), layer, 16, 384, _Gemvs(True))
    monkeypatch.setattr(col._flags, "TRTLLM_K3LL_MOE_TAIL_COL", 1)
    gemvs = _Gemvs(True)
    for rows in (8, 9, 16, 17, 32, 33):
        assert col.takes(_state(), layer, rows, 384, gemvs) == (9 <= rows <= 32), rows
    assert gemvs.asked[0] == (col.SHARED_SITE, 9, 384, 16)
    assert not col.takes(_state(), layer, 16, 384, _Gemvs(False)), "the shared push refused"
    assert not col.takes(_state(ready=False), layer, 16, 384, gemvs), "not warmed at load"
    # The publish ran at load for the 16-token tile only: 17-32 rows refused, 9-16 taken.
    assert not col.takes(_state(tiles=(16,)), layer, 24, 384, gemvs)
    assert col.takes(_state(tiles=(16,)), layer, 12, 384, gemvs)
    assert not col.takes(None, layer, 16, 384, gemvs) and not col.takes(
        _state(), None, 16, 384, gemvs
    )
    assert not col.takes(_state(), layer, 16, 384, None)
    # 32 rows x 16 ranks x 3584 x 2 B = 3.5 MiB of latent: a 3 MiB latent buffer refuses them, keeps 16.
    assert not col.takes(_state(buffer=3 << 20), layer, 32, 384, gemvs)
    assert col.takes(_state(buffer=3 << 20), layer, 16, 384, gemvs)
    # The shared all-reduce's two-shot call of 32 rows of 7168 at 16 ranks needs 917504 B (its broadcast stage, half
    # of it, holds the publish).
    assert not col.takes(_state(push_buffer=1 << 19), layer, 32, 384, gemvs)


class _Stop(Exception):
    pass


def _wide_layer(dm, col_takes: bool, lu_takes: bool, monkeypatch):
    """A ``K3DecodeMoeNvfp4Layer`` whose front and experts are stand-ins; records what ``_wide`` calls."""
    calls = []
    layer = object.__new__(dm.K3DecodeMoeNvfp4Layer)
    layer.state = SimpleNamespace(
        front_head=lambda rows, front32=None: object(), lu="lu_state", col="col_state"
    )
    layer.lu, layer.col = "lu_layer", "col_layer"
    layer.shared_cols = 384
    shared_act = torch.zeros(16, 384, dtype=torch.bfloat16)
    layer._front_only = lambda moe, x, front32=None: (
        calls.append("front") or ("routing", shared_act)
    )

    def front_routed(moe, x, front32=None, fc12=None):
        calls.append("front_routed")
        raise _Stop

    def routed_after(routing, fc12=None):
        calls.append(("routed", routing))
        raise _Stop

    layer._front_routed = front_routed
    layer._routed_after = routed_after

    def takes(name, result):
        def fn(st, ly, rows, width, gemvs):
            calls.append((name, st, ly, rows, width))
            return result

        return fn

    monkeypatch.setattr(dm._tail_col, "takes", takes("col_takes", col_takes))
    monkeypatch.setattr(dm._tail_lu, "takes", takes("lu_takes", lu_takes))

    def run(name):
        def fn(st, ly, gemvs, act, routed_fn):
            calls.append((name, st, ly, act is shared_act))
            try:
                routed_fn()
            except _Stop:
                pass
            return name

        return fn

    monkeypatch.setattr(dm._tail_col, "run", run("col_run"))
    monkeypatch.setattr(dm._tail_lu, "run", run("lu_run"))
    return layer, calls


@pytest.mark.parametrize(
    "col_takes,lu_takes", [(True, True), (True, False), (False, True), (False, False)]
)
def test_wide_dispatch_precedence(col_takes, lu_takes, monkeypatch):
    """The wide step decides before the front. ``moe_tail_col.takes`` is asked first and, where it holds, ``run``
    gets the front's shared activation and the routed experts, TAIL_LU not even asked; else TAIL_LU's decision as
    before; else ``_front_routed``."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    layer, calls = _wide_layer(dm, col_takes, lu_takes, monkeypatch)
    x = torch.zeros(16, 7168, dtype=torch.bfloat16)
    col_decision = ("col_takes", "col_state", "col_layer", 16, 384)
    lu_decision = ("lu_takes", "lu_state", "lu_layer", 16, 384)
    if col_takes:
        assert dm.K3DecodeMoeNvfp4Layer._wide(layer, None, x, gemvs=None) == "col_run"
        assert calls == [
            col_decision,
            "front",
            ("col_run", "col_state", "col_layer", True),
            ("routed", "routing"),
        ]
    elif lu_takes:
        assert dm.K3DecodeMoeNvfp4Layer._wide(layer, None, x, gemvs=None) == "lu_run"
        assert calls == [
            col_decision,
            lu_decision,
            "front",
            ("lu_run", "lu_state", "lu_layer", True),
            ("routed", "routing"),
        ]
    else:
        with pytest.raises(_Stop):
            dm.K3DecodeMoeNvfp4Layer._wide(layer, None, x, gemvs=None)
        assert calls == [col_decision, lu_decision, "front_routed"]


def test_col_tail_addend_and_consumer_decision(monkeypatch):
    """The handed-on rows are named by the decode workspace's addresses; the consumer refuses another workspace, a
    candidate count the warm-up did not run, and a prefix of another row count, before asking the kernel."""
    col = _mod("moe_tail_col")
    dc = _mod("decode_comm")
    flags_t = torch.zeros(9, dtype=torch.int32)
    ws = SimpleNamespace(uc_local=0x1000, flags=flags_t)
    tail = col.ColTail(ws, 16, frozenset((2, 3, 4)), frozenset())
    assert tail.lamport_addend == (0x1000, flags_t.data_ptr())
    asked = []
    monkeypatch.setattr(
        col, "attn_res_takes", lambda t, prefix, block, *mods: asked.append(prefix.shape) or True
    )
    zeros = torch.zeros(64, 7168, dtype=torch.bfloat16)
    comm = object.__new__(dc.K3DecodeComm)
    comm.oproj_push = SimpleNamespace(lamport=ws, zeros=zeros)
    block = torch.zeros(3, 16, 7168, dtype=torch.bfloat16)
    mods = (None, None, None)
    assert comm.takes_tail_col(tail, None, block, *mods)
    assert asked[-1] == (16, 7168), "the zero prefix of the tail's rows"
    other = col.ColTail(SimpleNamespace(uc_local=0, flags=flags_t), 16)
    assert not comm.takes_tail_col(other, None, block, *mods), "another workspace"
    assert not comm.takes_tail_col(tail._replace(rows=65), None, block, *mods), (
        "beyond the zero prefix"
    )
    comm.oproj_push = None
    assert not comm.takes_tail_col(tail, None, block, *mods)


def test_attn_res_takes_shapes(monkeypatch):
    """``attn_res_takes`` refuses, before the kernel's own predicate, a candidate count not run at load and a prefix
    of another row count."""
    col = _mod("moe_tail_col")
    kernel = SimpleNamespace(_EXACT=False, can_use_k3_attn_res=lambda *a: True)
    monkeypatch.setattr(col, "_attn_res", lambda: kernel)
    w = SimpleNamespace(weight=torch.zeros(7168, dtype=torch.bfloat16))
    tail = col.ColTail(None, 16, frozenset((4,)))
    prefix = torch.zeros(16, 7168, dtype=torch.bfloat16)
    assert col.attn_res_takes(tail, prefix, torch.zeros(3, 16, 7168), w, w, w)
    assert not col.attn_res_takes(tail, prefix, torch.zeros(2, 16, 7168), w, w, w)
    assert not col.attn_res_takes(tail, prefix[:8], torch.zeros(3, 8, 7168), w, w, w)
    kernel._EXACT = True
    assert not col.attn_res_takes(tail, prefix, torch.zeros(3, 16, 7168), w, w, w)


def test_published_value_rounds_once():
    """The publish's arithmetic on one element, emulated in torch: ``bf16(fp32(sum) + shared)`` is one rounding of
    the exact value of the bf16 operands, within the error bound the op matrix holds the kernel to."""
    g = torch.Generator().manual_seed(0)
    latent = torch.randn(32, 3584, generator=g).bfloat16()
    up = (torch.randn(448, 3584, generator=g) * 0.02).bfloat16()
    shared = torch.randn(32, 448, generator=g).bfloat16()
    exact = latent.double() @ up.double().T + shared.double()
    once = ((latent.float() @ up.float().T) + shared.float()).bfloat16()
    terms = latent.double().abs() @ up.double().abs().T + shared.double().abs()
    diff = (once.double() - exact).abs()
    assert bool((diff <= 2.0**-8 * exact.abs() + 2.0**-10 * terms).all())
