# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3LL_MOE_TAIL_LU`` (``kimi_k3_nvfp4/moe_tail_lu.py``) on one process: the switch's default and doc, the
FP8 weight build, the engagement decision and the wide MoE step's dispatch with stand-ins (no GPU needed), and the
replicated FP8 latent_up on its MXFP8 input against ``F.linear`` (SM100 / SM103). The collectives are checked by
``comm/_kimi_k3_moe_tail_lu_op_matrix.py``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4"


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{_PKG}.{name}")


def _sm10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


def test_switch_default_sm103_and_documented():
    flags = _mod("flags")
    value, default, doc = flags.flags()["TRTLLM_K3LL_MOE_TAIL_LU"]
    assert default == -1
    assert "LOSSY" in doc and "MB per layer" in doc and "SM103" in doc
    assert _mod("moe_tail_lu").MOE_TAIL_LU_CAPABILITIES == ((10, 3),)


@pytest.mark.parametrize(
    "value,capability,on",
    [
        (1, (10, 3), True),
        (1, (10, 7), True),
        (0, (10, 3), False),
        (-1, (10, 3), True),
        (-1, (10, 7), False),
        (-1, (10, 0), False),
    ],
)
def test_enabled_value(monkeypatch, value, capability, on):
    lu = _mod("moe_tail_lu")
    monkeypatch.setattr(lu._flags, "TRTLLM_K3LL_MOE_TAIL_LU", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    lu._on_by_default.cache_clear()
    try:
        assert lu.enabled() is on
    finally:
        lu._on_by_default.cache_clear()


def test_shared_site_is_optional():
    dg = _mod("decode_gemv")
    lu = _mod("moe_tail_lu")
    assert lu.SHARED_SITE in dg.SITES and lu.SHARED_SITE in dg.OPTIONAL_SITES
    assert dg.DEFAULT_SITES == tuple(s for s in dg.SITES if s != lu.SHARED_SITE)
    spec = dg.SITES[lu.SHARED_SITE]
    assert (spec.n, spec.k, spec.wide, spec.wide_rows) == (7168, 384, True, lu.MAX_ROWS)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantize_block_fp8(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    lu = _mod("moe_tail_lu")
    g = torch.Generator(device=device).manual_seed(0)
    w = (torch.randn(384, 512, generator=g, device=device) * 0.02).bfloat16()
    w[:128, :128] *= 300
    w[128:256, 256:384] = 0
    codes, scale = lu.quantize_block_fp8(w)
    assert codes.dtype == torch.float8_e4m3fn and tuple(scale.shape) == (3, 4)
    assert bool((torch.log2(scale) == torch.round(torch.log2(scale))).all())
    blocks = w.float().view(3, 128, 4, 128).abs().amax(dim=(1, 3)).clamp(min=1e-4)
    # The smallest power of two keeping the block within 448.
    assert bool((blocks / scale <= 448).all()) and bool((blocks / (scale / 2) > 448).all())
    deq = lu.dequantize_block_fp8(codes, scale)
    full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert bool(((deq - w.float()).abs() <= w.float().abs() * 2.0**-4 + full * 2.0**-9).all())
    assert bool((deq[128:256, 256:384] == 0).all())
    with pytest.raises(ValueError):
        lu.quantize_block_fp8(w[:, :100])


def test_lu_tail_reduced_rounds_once():
    lu = _mod("moe_tail_lu")
    a = torch.tensor([[1.0, 2.0**-8, 3.0]], dtype=torch.bfloat16)
    b = torch.tensor([[2.0**-9, 1.0, -3.0]], dtype=torch.bfloat16)
    want = (a.float() + b.float()).bfloat16()
    assert torch.equal(lu.LuTail(a, b).reduced(), want)


class _Gemvs:
    def __init__(self, ok: bool):
        self.ok, self.asked = ok, []

    def takes_push_rows(self, site, rows, width, weight, world):
        self.asked.append((site, rows, width, world))
        return self.ok


def _state(ready=True, rows=range(9, 33), buffer=4 << 20, push_buffer=4 << 20, world=16):
    view = SimpleNamespace(world=world, bytes_per_buffer=buffer)
    push = SimpleNamespace(world=world, bytes_per_buffer=push_buffer)
    return SimpleNamespace(ready=ready, mx_rows=frozenset(rows), latent=view, push=push)


def test_takes_decision(monkeypatch):
    """The decision reads the switch, the step's rows and load-time state only (before the front runs)."""
    pytest.importorskip("cutlass")
    lu = _mod("moe_tail_lu")
    layer = SimpleNamespace(shared_down=torch.zeros(1))
    monkeypatch.setattr(lu._flags, "TRTLLM_K3LL_MOE_TAIL_LU", 0)
    assert not lu.takes(_state(), layer, 16, 384, _Gemvs(True))
    monkeypatch.setattr(lu._flags, "TRTLLM_K3LL_MOE_TAIL_LU", 1)
    assert lu._flags.TRTLLM_K3LL_MOE_TAIL_LU_MAX_ROWS == 16, "default: bs2 at MTP7 only"
    for rows in (8, 9, 16, 17, 32):
        assert lu.takes(_state(), layer, rows, 384, _Gemvs(True)) == (9 <= rows <= 16), rows
    monkeypatch.setattr(lu._flags, "TRTLLM_K3LL_MOE_TAIL_LU_MAX_ROWS", 32)
    gemvs = _Gemvs(True)
    for rows in (8, 9, 16, 17, 32, 33):
        assert lu.takes(_state(), layer, rows, 384, gemvs) == (9 <= rows <= 32), rows
    assert gemvs.asked[0] == (lu.SHARED_SITE, 9, 384, 16)
    assert not lu.takes(_state(), layer, 16, 384, _Gemvs(False)), "the shared push refused"
    assert not lu.takes(_state(ready=False), layer, 16, 384, gemvs), "not warmed at load"
    assert not lu.takes(_state(rows=range(9, 17)), layer, 24, 384, gemvs), (
        "MXFP8 GEMM not compiled for the rows"
    )
    assert not lu.takes(None, layer, 16, 384, gemvs) and not lu.takes(
        _state(), None, 16, 384, gemvs
    )
    assert not lu.takes(_state(), layer, 16, 384, None)
    # 32 rows x 16 ranks x 3584 x 2 B = 3.5 MiB: a 3 MiB latent buffer refuses them, keeps 16.
    assert not lu.takes(_state(buffer=3 << 20), layer, 32, 384, gemvs)
    assert lu.takes(_state(buffer=3 << 20), layer, 16, 384, gemvs)
    # The shared all-reduce's two-shot call of 32 rows of 7168 at 16 ranks needs 917504 B.
    assert not lu.takes(_state(push_buffer=1 << 19), layer, 32, 384, gemvs)


class _Stop(Exception):
    pass


def _wide_layer(dm, takes: bool, monkeypatch):
    """A ``K3DecodeMoeNvfp4Layer`` whose front and experts are stand-ins; records what ``_wide`` calls."""
    calls = []
    layer = object.__new__(dm.K3DecodeMoeNvfp4Layer)
    layer.state = SimpleNamespace(front_head=lambda rows, front32=None: object(), lu="state")
    layer.lu = "layer"
    layer.shared_cols = 384
    shared_act = torch.zeros(16, 384, dtype=torch.bfloat16)
    layer._front_only = lambda moe, x, front32=None: (
        calls.append("front") or ("routing", shared_act)
    )

    def front_routed(moe, x, front32=None):
        calls.append("front_routed")
        raise _Stop

    def routed_after(routing):
        calls.append(("routed", routing))
        raise _Stop

    layer._front_routed = front_routed
    layer._routed_after = routed_after

    def lu_takes(st, ly, rows, width, gemvs):
        calls.append(("takes", st, ly, rows, width))
        return takes

    monkeypatch.setattr(dm._tail_lu, "takes", lu_takes)

    def run(st, ly, gemvs, act, routed_fn):
        calls.append(("run", st, ly, act is shared_act))
        try:
            routed_fn()
        except _Stop:
            pass
        return "tail"

    monkeypatch.setattr(dm._tail_lu, "run", run)
    return layer, calls


@pytest.mark.parametrize("takes", [False, True])
def test_wide_dispatch(takes, monkeypatch):
    """The wide step decides before the front. With ``moe_tail_lu.takes`` it hands ``run`` the front's shared
    activation and the routed experts (after the front, once); without it the step runs ``_front_routed`` as before."""
    pytest.importorskip("cutlass")
    dm = _mod("decode_moe_lowm")
    layer, calls = _wide_layer(dm, takes, monkeypatch)
    x = torch.zeros(16, 7168, dtype=torch.bfloat16)
    decision = ("takes", "state", "layer", 16, 384)
    if takes:
        assert dm.K3DecodeMoeNvfp4Layer._wide(layer, None, x, gemvs=None) == "tail"
        assert calls == [decision, "front", ("run", "state", "layer", True), ("routed", "routing")]
    else:
        with pytest.raises(_Stop):
            dm.K3DecodeMoeNvfp4Layer._wide(layer, None, x, gemvs=None)
        assert calls == [decision, "front_routed"]


@pytest.mark.skipif(not _sm10x(), reason="needs an SM100/SM103 GPU")
@pytest.mark.parametrize("rows", [9, 16, 17, 24, 32])
def test_latent_up_mxfp8(rows):
    """The replicated FP8 latent_up on the MXFP8 latent: ``F.linear`` of the dequantized operands within the fp32
    summation order and bf16 output, and the bf16 latent_up within FP8's rounding."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    lu = _mod("moe_tail_lu")
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = (torch.randn(lu.HIDDEN, lu.LATENT, generator=g, device="cuda") * 0.02).bfloat16()
    w[:128, :128] *= 16
    latent = torch.randn(rows, lu.LATENT, generator=g, device="cuda").bfloat16()
    codes, scale = lu.quantize_block_fp8(w)
    layer = SimpleNamespace(
        up_weight=codes, up_scale=lowm.pack_ue8m0_block_scale(scale, lu.HIDDEN, lu.LATENT)
    )
    xq, xs = mx.mxfp8_quantize_reference(latent)
    assert mx.can_run_mxfp8(xq, xs, layer.up_weight, layer.up_scale)
    out = lu.latent_up(layer, xq, xs)
    x_deq = (xq.float().view(rows, -1, 32) * torch.exp2(xs.float() - 127)[..., None]).view(rows, -1)
    ref = F.linear(x_deq, lu.dequantize_block_fp8(codes, scale))
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 1e-2, float(err)
    exact = F.linear(latent.float(), w.float())
    err = (out.float() - exact).abs().max() / exact.abs().max()
    assert err < 1e-1, float(err)
    # The low-M GEMM on the bf16 latent is the same bits (the MXFP8 kernel's contract).
    assert torch.equal(
        out, torch.ops.trtllm.k3_fp8w_lowm_gemm(latent, layer.up_weight, layer.up_scale)
    )


@pytest.mark.skipif(not _sm10x(), reason="needs an SM100/SM103 GPU")
@pytest.mark.parametrize("rows,snapshots", [(16, 1), (16, 3), (32, 5)])
def test_attn_res_mxfp8_side_output(rows, snapshots):
    """``moe_tail_lu.attn_res`` with ``mxfp8_out`` (``TRTLLM_K3LL_ATTN_RES_MXFP8`` on a MoE-tail-LU consumer):
    ``normed`` and ``updated`` the same bits as without it, and the MXFP8 copy equal to the reference quantizer of
    ``normed`` (what the attention's first FP8 projection would quantize itself)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx

    lu = _mod("moe_tail_lu")
    fp8w = _mod("fp8w_lowm")
    g = torch.Generator(device="cuda").manual_seed(rows + snapshots)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).bfloat16()

    tail = lu.LuTail(rnd(rows, lu.HIDDEN, scale=0.5), rnd(rows, lu.HIDDEN, scale=0.5))
    prefix = rnd(rows, lu.HIDDEN)
    block = rnd(snapshots, rows, lu.HIDDEN)

    def norm(w):
        return SimpleNamespace(weight=w, eps=1e-6)

    res_proj = SimpleNamespace(weight=rnd(1, lu.HIDDEN, scale=0.05))
    res_norm, out_norm = norm(1 + rnd(lu.HIDDEN, scale=0.1)), norm(1 + rnd(lu.HIDDEN, scale=0.1))
    normed0, updated0 = lu.attn_res(tail, prefix, block, res_proj, res_norm, out_norm)
    xq, xs = fp8w.new_mxfp8_input(rows, lu.HIDDEN, prefix.device)
    normed1, updated1 = lu.attn_res(
        tail, prefix, block, res_proj, res_norm, out_norm, mxfp8_out=(xq, xs)
    )
    torch.cuda.synchronize()
    assert torch.equal(normed0, normed1)
    assert torch.equal(updated0, updated1)
    ref_q, ref_s = mx.mxfp8_quantize_reference(normed1)
    assert torch.equal(xs, ref_s)
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
