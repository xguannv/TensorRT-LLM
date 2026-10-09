# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL FP8 MLA decode: flat (q_tok, head) fold vs the unfolded kernel.

K3 has 96 MLA heads and verifies 1 + 3 draft tokens per request. 96 heads fill
only 3/4 of the 128-row M tile and no integer token fold fits, so the kernel
used to run S_q = 4 tiles of 96 valid rows per request. The flat fold packs the
384 (q_tok, head) rows into 3 full M tiles; each row's math is unchanged, so the
output must be BITWISE identical to the unfolded kernel (TRTLLM_MLA_FLAT_FOLD=0)
at any split_kv, and agree with an fp32 reference of the MTP-causal attention.

When H * S_q is not a multiple of 128 (K3 with MTP6: S_q = 7, 672 rows) the flat
fold pads the last M tile (6 tiles instead of 7). The padded rows must never be
stored: the output, LSE and Helix softmax stats past each request stay untouched
and the valid rows stay bitwise identical to the unfolded kernel.

Requires 1 SM100/SM103 GPU and nvidia-cutlass-dsl.
"""

import math

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
    )
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_sm100_or_sm103():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_sm100_or_sm103(), reason="needs sm100/sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

D, R, PAGE = 512, 64, 64


def _make_inputs(lens, S, H, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B = len(lens)
    pages_per = [math.ceil(L / PAGE) for L in lens]
    max_blocks = max(pages_per) + 1
    num_pages = sum(pages_per) + 8
    perm = torch.randperm(num_pages, device=dev).to(torch.int32)
    pt = torch.zeros(B, max_blocks, dtype=torch.int32, device=dev)
    off = 0
    for b, n in enumerate(pages_per):
        pt[b, :n] = perm[off : off + n]
        off += n
    kv = (torch.randn(num_pages, 1, PAGE, 1, D + R, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv_pages = kv[:, 0, :, 0, :]
    q = (torch.randn(B, S, H, D + R, device=dev) * 0.5).to(torch.float8_e4m3fn)
    o = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=dev)
    return dict(
        q=q,
        kv_pages=kv_pages,
        pt=pt,
        o=o,
        inputs=[
            q[..., :D].permute(2, 3, 1, 0),
            q[..., D:].permute(2, 3, 1, 0),
            kv_pages[..., :D].permute(1, 2, 0),
            kv_pages[..., D:].permute(1, 2, 0),
            pt.transpose(0, 1),
            torch.tensor(lens, dtype=torch.int32, device=dev),
            o.permute(2, 3, 1, 0),
            torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
            None,
            None,
            torch.tensor([1.0 / math.sqrt(192)], dtype=torch.float32, device=dev),
            torch.tensor([1.0], dtype=torch.float32, device=dev),
        ],
    )


def _run(x, H, S, split_kv, flat_fold, monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1" if flat_fold else "0")
    # the compiled-kernel cache is keyed by shape/tactic only
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", {})
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN, num_heads=H, seq_len_q=S, page_size=PAGE, max_batch_size=64
    )
    runner.forward(x["inputs"], ((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()
    return x["o"].clone()


def _reference(x, lens, S, H):
    q = x["q"].float()
    kv = x["kv_pages"].float()
    out = torch.empty(len(lens), S, H, D, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(192)
    for b, L in enumerate(lens):
        pages = x["pt"][b, : math.ceil(L / PAGE)].long()
        c = kv[pages].reshape(-1, D + R)[:L]
        for s in range(S):
            bound = L - (S - 1) + s
            logits = (q[b, s] @ c[:bound].T) * scale  # [H, bound]
            p = torch.softmax(logits, dim=-1)
            out[b, s] = p @ c[:bound, :D]
    return out


def test_k3_shape_takes_the_flat_fold(monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1")
    k = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=96,
        seq_len_q=4,
        fold_sq=True,
    )
    assert k.fold_rows == 128 and k.flat_fold
    # integer folds are unchanged
    k32 = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=32,
        seq_len_q=4,
        fold_sq=True,
    )
    assert k32.fold_rows == 128 and not k32.flat_fold


@pytest.mark.parametrize("split_kv", [1, 2, 4])
@pytest.mark.parametrize("H,S", [(96, 4), (48, 8)])
def test_flat_fold_is_bitwise_identical(H, S, split_kv, monkeypatch):
    lens = [3000, 70, 1500, 8191, 64, 513, 2048, 4500]
    x = _make_inputs(lens, S, H, seed=7)
    ref = _run(x, H, S, split_kv, flat_fold=False, monkeypatch=monkeypatch)
    new = _run(x, H, S, split_kv, flat_fold=True, monkeypatch=monkeypatch)
    assert torch.equal(new, ref)
    gold = _reference(x, lens, S, H)
    err = (new.float() - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


@pytest.mark.parametrize("flat_fold", [True, False])
def test_fold_layout(flat_fold, monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1" if flat_fold else "0")
    layout = BlackwellMultiHeadLatentAttentionForwardFP8.get_fold_layout
    # (rows per M tile, M tiles per request) for K3's 96 heads
    expected = {
        2: (96, 2),
        3: (96, 3),
        4: (128, 3),
        5: (128, 4),
        6: (128, 5),
        7: (128, 6),
        8: (128, 6),
    }
    for S, flat in expected.items():
        assert layout(96, S, 128) == (flat if flat_fold else (96, S)), S
    # integer folds that fill the tile are unchanged; a partial one goes flat
    assert layout(32, 4, 128) == (128, 1)
    assert layout(48, 3, 128) == ((128, 2) if flat_fold else (48, 3))
    assert layout(16, 3, 128) == (48, 1)

    rows = CuteDSLNVMlaDecodeBlackwellRunner.get_split_kv_rows_per_request
    assert rows(96, 7) == (768 if flat_fold else 672)
    torch.zeros(1, device="cuda")  # HardwareInfo needs a current CUDA context
    assert rows(96, 4) == 384 and rows(128, 2) == 256 and rows(96, 1) == 96
    # the split-KV workspace bound grows only by the padding rows
    acc = CuteDSLNVMlaDecodeBlackwellRunner.get_max_split_kv_acc_rows
    old = CuteDSLNVMlaDecodeBlackwellRunner.get_max_split_kv_rows
    assert acc(96, 4, 128) == old(96, 4, 128) * 96
    assert acc(96, 7, 128) >= old(96, 7, 128) * 96


def test_k3_mtp6_takes_the_padded_flat_fold(monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1")
    k = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=96,
        seq_len_q=7,
        fold_sq=True,
    )
    assert (k.fold_rows, k.fold_chunks) == (128, 6)
    assert k.flat_fold and k.flat_fold_padded


# Compiled kernels per TRTLLM_MLA_FLAT_FOLD value, shared across the tests below
# (the runner's cache key does not see the environment variable).
_KERNEL_CACHES = {"0": {}, "1": {}}
_MAX_BATCH = 64


def _padded_inputs(lens, S, H, helix, seed):
    torch.manual_seed(seed)
    x = _make_inputs(lens, S, H, seed=seed)
    B = len(lens)
    x["lens"] = lens
    if helix:
        # Per-token rank-local bounds of a speculative verify group: the rank
        # holds K local entries, of which the group's S_q tokens own at most
        # S_q, so each token sees between K - S_q and K of them (none at all
        # for a request whose K <= S_q).
        bounds = []
        for L in lens:
            for _ in range(S):
                bounds.append(int(torch.randint(max(0, L - S), L + 1, (1,)).item()))
        bounds[0] = lens[0] - S
        bounds[-S] = 0
        bounds[-1] = lens[-1]
        x["kv_bounds"] = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        x["stats_shape"] = (B * S, H, 2)
    return x


def _run_guarded(x, H, S, split_kv, flat_fold, monkeypatch, persistent=False):
    """Run the runner with NaN-filled outputs plus one request's worth of
    guard rows behind O and the softmax stats; returns (o, lse, stats).

    Requests without KV are skipped by the finiteness checks: the uniform
    split leaves them unstored with split_kv == 1."""
    flag = "1" if flat_fold else "0"
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", flag)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", _KERNEL_CACHES[flag])
    B = len(x["lens"])
    helix = "kv_bounds" in x
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=H,
        seq_len_q=S,
        page_size=PAGE,
        max_batch_size=_MAX_BATCH,
        emit_softmax_stats=helix,
    )
    n_o = B * S * H * D
    o_buf = torch.full((n_o + S * H * D,), float("nan"), dtype=torch.bfloat16, device="cuda")
    o = o_buf[:n_o].view(B, S, H, D)
    lse_off, lse_size, _, _, ws_size = runner.get_workspace_layout(
        H, S, D, _MAX_BATCH, cutlass.Float32
    )
    ws = torch.full(
        (ws_size,), 0xFF, dtype=torch.uint8, device="cuda"
    )  # 0xFFFFFFFF is a float32 NaN
    inputs = list(x["inputs"])
    inputs[6] = o.permute(2, 3, 1, 0)
    inputs[7] = ws
    stats_buf = None
    if helix:
        stats_buf = torch.full((B * S + S, H, 2), float("nan"), dtype=torch.float32, device="cuda")
        inputs[8] = stats_buf[: B * S]
        inputs[9] = x["kv_bounds"]
    runner.forward(inputs, ((128, 128), (128, 256), split_kv, persistent))
    torch.cuda.synchronize()

    # Nothing is stored past the last request, and every real row is stored.
    has_kv = torch.tensor([L > 0 for L in x["lens"]], device="cuda")
    assert torch.isnan(o_buf[n_o:].float()).all()
    assert torch.isfinite(o[has_kv].float()).all()
    lse_all = ws[lse_off : lse_off + lse_size].view(torch.float32).view(_MAX_BATCH, S, H)
    assert torch.isnan(lse_all[B:]).all()
    assert torch.isfinite(lse_all[:B][has_kv]).all()
    stats = None
    if helix:
        assert torch.isnan(stats_buf[B * S :]).all()
        stats = stats_buf[: B * S].clone()
        assert not torch.isnan(stats.view(B, S, H, 2)[has_kv]).any()
    return o.clone(), lse_all[:B].clone(), stats


def _reference_bounded(x, S, H):
    """fp32 reference; bound[b][s] is the number of KV entries token s sees."""
    q = x["q"].float()
    kv = x["kv_pages"].float()
    lens = x["lens"]
    out = torch.zeros(len(lens), S, H, D, dtype=torch.float32, device="cuda")
    valid = torch.zeros(len(lens), S, dtype=torch.bool)
    scale = 1.0 / math.sqrt(192)
    for b, L in enumerate(lens):
        pages = x["pt"][b, : math.ceil(L / PAGE)].long()
        c = kv[pages].reshape(-1, D + R)[:L]
        for s in range(S):
            if "kv_bounds" in x:
                bound = int(x["kv_bounds"][b * S + s].item())
            else:
                bound = L - (S - 1) + s
            if bound <= 0:
                continue
            logits = (q[b, s] @ c[:bound].T) * scale  # [H, bound]
            out[b, s] = torch.softmax(logits, dim=-1) @ c[:bound, :D]
            valid[b, s] = True
    return out, valid


def _effective_split(H, S, B, split_kv, monkeypatch):
    """Requested split, capped like forward() caps it, for both foldings, so
    the folded and unfolded runs use the same split."""
    torch.zeros(1, device="cuda")  # HardwareInfo needs a current CUDA context
    fits = []
    for flag in ("0", "1"):
        monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", flag)
        runner = CuteDSLNVMlaDecodeBlackwellRunner(
            in_dtype=cutlass.Float8E4M3FN,
            num_heads=H,
            seq_len_q=S,
            page_size=PAGE,
            max_batch_size=_MAX_BATCH,
        )
        fits.append(runner.get_fitting_split_kv(B, _MAX_BATCH))
    return max(1, min([split_kv] + fits))


_LENS = {
    # mixed lengths incl. one KV tile, page and tile boundaries, < 2 tiles
    "mixed8": [3000, 70, 1500, 8191, 64, 513, 2048, 4500],
    # K3 Helix CP16 per-rank KV (7.3k / 10k) and a request barely past S_q
    "k3": [7300, 10000, 9],
    # one request: the last tile's padding rows would run past every tensor
    "single": [7300],
}


@pytest.mark.parametrize("lens_name", list(_LENS))
@pytest.mark.parametrize("split_kv", [1, 2, 4, 8])
@pytest.mark.parametrize("H,S", [(96, 7), (96, 6), (96, 5), (48, 3)])
def test_padded_flat_fold_is_bitwise_identical(H, S, split_kv, lens_name, monkeypatch):
    lens = _LENS[lens_name]
    split_kv = _effective_split(H, S, len(lens), split_kv, monkeypatch)
    x = _padded_inputs(lens, S, H, helix=False, seed=11)
    ref_o, ref_lse, _ = _run_guarded(x, H, S, split_kv, flat_fold=False, monkeypatch=monkeypatch)
    new_o, new_lse, _ = _run_guarded(x, H, S, split_kv, flat_fold=True, monkeypatch=monkeypatch)
    assert torch.equal(new_o, ref_o)
    assert torch.equal(new_lse, ref_lse)
    gold, _ = _reference_bounded(x, S, H)
    err = (new_o.float() - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


@pytest.mark.parametrize("split_kv", [1, 4])
@pytest.mark.parametrize("H,S", [(96, 7), (96, 5), (48, 3)])
def test_padded_flat_fold_helix_bounds_and_stats(H, S, split_kv, monkeypatch):
    lens = _LENS["k3"] + [700, 3000, S]
    split_kv = _effective_split(H, S, len(lens), split_kv, monkeypatch)
    x = _padded_inputs(lens, S, H, helix=True, seed=5)
    ref_o, ref_lse, ref_stats = _run_guarded(
        x, H, S, split_kv, flat_fold=False, monkeypatch=monkeypatch
    )
    new_o, new_lse, new_stats = _run_guarded(
        x, H, S, split_kv, flat_fold=True, monkeypatch=monkeypatch
    )
    assert torch.equal(new_o, ref_o)
    assert torch.equal(new_lse, ref_lse)
    assert torch.equal(new_stats, ref_stats)
    gold, valid = _reference_bounded(x, S, H)
    v = valid.to("cuda")
    err = (new_o.float() - gold).abs()[v].max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


@pytest.mark.parametrize("S", [2, 3])
def test_k3_short_mtp_keeps_the_unfolded_tiling(S, monkeypatch):
    # 96 * S rows need S M tiles either way, so the tiling stays per token.
    lens = _LENS["mixed8"]
    x = _padded_inputs(lens, S, 96, helix=False, seed=3)
    new_o, _, _ = _run_guarded(x, 96, S, 2, flat_fold=True, monkeypatch=monkeypatch)
    gold, _ = _reference_bounded(x, S, 96)
    err = (new_o.float() - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


# Balanced split (tactic split_kv == BALANCED_SPLIT_TACTIC): persistent
# clusters, each given an equal share of the batch's K tiles.


def _balanced_split_tiles(lens, num_chunks, num_clusters, min_units, k_tile=128):
    """(request, chunk) M tiles the balanced split cuts across clusters, by
    the partition of MLABalancedTileScheduler."""
    kt = [max(0, math.ceil(L / k_tile)) for L in lens]
    prefix = [0]
    for t in kt:
        prefix.append(prefix[-1] + t)
    G, C, V = num_clusters, num_chunks, prefix[-1]
    W = max(math.ceil(V * C / G), min_units)
    n = G // C
    s = G - n * C
    nW = min(W * n, V)
    Lw = V - nW
    ranges = []  # (chunk, first position, end position) in the chunk's units
    for g in range(G):
        if g < n * C:
            lane = g // C
            ranges.append((g % C, min(lane * W, V), min((lane + 1) * W, V)))
            continue
        s_tot = Lw * C
        Ws = math.ceil(s_tot / s)
        q, qb = min((g - n * C) * Ws, s_tot), min((g - n * C + 1) * Ws, s_tot)
        while q < qb:
            c = q // Lw
            p = nW + q - c * Lw
            p_end = nW + min(Lw, qb - c * Lw)
            ranges.append((c, p, p_end))
            q += p_end - p
    split = set()
    for c, p, p_end in ranges:
        while p < p_end:
            b = max(i for i in range(len(lens)) if prefix[i] <= p)
            end = min(prefix[b + 1], p_end)
            if p > prefix[b] or end < prefix[b + 1]:
                split.add((b, c))
            p = end
    return split


_BALANCED_LENS = {
    "mixed8": _LENS["mixed8"],
    "k3": _LENS["k3"],
    "single": _LENS["single"],
    # skewed per-rank KV of a Helix batch, with requests that hold no KV here
    "skew": [600, 0, 61000, 2048, 9000, 0, 130, 15000, 7300, 64],
}


@pytest.mark.parametrize("lens_name", list(_BALANCED_LENS))
@pytest.mark.parametrize(
    "H,S,helix",
    [(96, 7, True), (96, 7, False), (96, 6, False), (96, 8, True), (48, 3, False), (128, 1, False)],
)
def test_balanced_split_matches_uniform_split(H, S, helix, lens_name, monkeypatch):
    """M tiles the balanced split keeps whole are bitwise identical to the
    uniform split_kv = 1 kernel; M tiles it cuts across clusters match it to
    split-KV rounding and the fp32 reference; requests without KV get the
    empty result (O = 0, LSE = +inf, stats (-inf, 0))."""
    lens = _BALANCED_LENS[lens_name]
    x = _padded_inputs(lens, S, H, helix=helix, seed=13)
    ref_o, ref_lse, ref_stats = _run_guarded(x, H, S, 1, True, monkeypatch)
    new_o, new_lse, new_stats = _run_guarded(
        x,
        H,
        S,
        CuteDSLNVMlaDecodeBlackwellRunner.BALANCED_SPLIT_TACTIC,
        True,
        monkeypatch,
        persistent=True,
    )

    if H < 128 and S > 1:
        rows, chunks = BlackwellMultiHeadLatentAttentionForwardFP8.get_fold_layout(H, S, 128)
    else:
        rows, chunks = H, S
    split = _balanced_split_tiles(
        lens,
        chunks,
        CuteDSLNVMlaDecodeBlackwellRunner._get_max_active_blocks() // 2,
        BlackwellMultiHeadLatentAttentionForwardFP8.BALANCED_MIN_UNITS,
    )
    gold, valid = _reference_bounded(x, S, H)
    tol = 0.05 * gold.abs().max().item()
    flat = lambda t: t.reshape(len(lens), S * H, *t.shape[3:])  # noqa: E731
    new_rows, ref_rows, gold_rows = flat(new_o), flat(ref_o), flat(gold)
    new_lse_rows, ref_lse_rows = flat(new_lse), flat(ref_lse)
    if helix:
        new_st = new_stats.view(len(lens), S * H, 2)
        ref_st = ref_stats.view(len(lens), S * H, 2)
    for b, L in enumerate(lens):
        if L <= 0:
            assert (new_o[b] == 0).all()
            assert torch.isinf(new_lse[b]).all() and (new_lse[b] > 0).all()
            if helix:
                st = new_stats.view(len(lens), S, H, 2)[b]
                assert torch.isinf(st[..., 0]).all() and (st[..., 1] == 0).all()
            continue
        for c in range(chunks):
            r = slice(c * rows, min((c + 1) * rows, S * H))
            if (b, c) not in split:
                assert torch.equal(new_rows[b, r], ref_rows[b, r]), (b, c)
                assert torch.equal(new_lse_rows[b, r], ref_lse_rows[b, r]), (b, c)
                if helix:
                    assert torch.equal(new_st[b, r], ref_st[b, r]), (b, c)
            else:
                # Each part requantizes P to fp8 against its own running max,
                # so a cut tile differs from the uncut one at fp8-P noise.
                d = (new_rows[b, r].float() - ref_rows[b, r].float()).abs().max().item()
                assert d <= tol, (b, c, d)
                dl = (new_lse_rows[b, r] - ref_lse_rows[b, r]).abs().max().item()
                assert dl <= 1e-4, (b, c, dl)
        v = valid[b].to("cuda").repeat_interleave(H)
        err = (new_rows[b].float() - gold_rows[b])[v].abs().max().item() if v.any() else 0.0
        assert err <= tol, (b, err)
