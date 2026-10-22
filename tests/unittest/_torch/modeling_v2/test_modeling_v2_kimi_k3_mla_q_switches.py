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
"""The query-side switches of the Kimi K3 NVFP4 MLA decode (``kimi_k3_nvfp4/flags.py``) on one GPU, at the Helix16
per-rank shapes (96 heads, this rank's group of 6), 8 / 16 / 32 rows (bs1 / bs2 / bs4 with 7 draft tokens):
``TRTLLM_K3LL_HELIX_Q_ABSORB``, ``TRTLLM_K3LL_MLA_GATE_SIDE``, ``TRTLLM_K3LL_FP8W_LOWM_FUSED_A`` and
``TRTLLM_K3LL_MLA_QKV_PDL``.

What runs. ``K3DecodeMLA._quantized_decode`` itself on a stand-in module that holds what it reads: the fused q_a / kv_a,
gate and Q-B projections as ``_Fp8BlockScaleWeightReadLinear`` built from raw FP8 128x128 block weights and routed by
``fp8w_lowm.install`` (the fused q_a / kv_a with ``partial_n_tile``, as a launch with FUSED_A=1 builds it), the
stock RMSNorms, every head's FP8 K-B, this rank's FP8 V-B, the FP8 Q-B rows of its decode group from
``K3DecodeMLA.prepare_fp8_decode_weights`` and the BF16 W_UK from ``K3DecodeMLA.prepare_helix_q_absorb`` (the state a
launch with every switch at 1 builds). The projections, ``attention.mla_project_qkv`` / ``mla_latent`` / ``mla_output``,
``k3_fp8w_lowm_gemm`` and ``K3MlaQAbsorb`` are the real kernels. The two ops that need a KV cache and the Helix group
are recording stand-ins: the backend's ``mla_rope_generation`` (records the latent row; quantizes the BF16 query to
FP8 at unit scale as the cache kernel does unless ``q_rope_applied``) and the split read (records the FP8 query bytes
and returns a fixed function of this rank's heads of them).

Checks (eagerly and from a captured CUDA graph replayed with new inputs, each replay bit for bit the eager call on the
same inputs):
  * all switches 0 (with every switch's load-time state built): the output, the recorded query bytes and latent row,
    bit for bit the pre-change chain (``_pre_change``: the concatenated ``[kv_a | g]``, ``mla_project_qkv``, the
    cache kernel, the split read with the all-gather, ``mla_output``) on the module's stock fused q_a / kv_a GEMM;
  * GATE_SIDE 1: output bit for bit the switch-off output (the gate is the same projection, read in place);
  * MLA_QKV_PDL 1: output, query bytes and latent row bit for bit the switch-off ones (the three query kernels with
    programmatic dependent launch; in the graph test they replay after the fused q_a / kv_a GEMM's PDL launch);
  * FUSED_A 1: the fused q_a / kv_a on ``k3_fp8w_lowm_gemm``; its error against FP32 at most ``FUSED_A_ERR_RATIO``
    times the stock GEMM's; the output within ``OUT_TOL`` (relative L2) of the switch-off output;
  * HELIX_Q_ABSORB 1 on a GPU it does not build on: no W_UK, everything bit for bit the switch-off decode;
  * HELIX_Q_ABSORB 1: every head's FP8 query on this rank (96 heads into the split read, ``all_heads``; the cache
    kernel called with ``q_rope_applied``), the latent row bit for bit the switch-off one; the query's error against
    an FP32 reference at most ``Q_ERR_RATIO`` times the current path's (``mla_project_qkv`` with every head's Q-B / K-B,
    which is bit for bit the gathered head groups, then FP8); this rank's heads within ``Q_TOL`` of the switch-off
    query; the output within ``OUT_TOL`` of the switch-off output.
"""

import contextlib
import types
from unittest import mock

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="k3_fp8w_lowm_gemm and K3MlaQAbsorb run on SM100 / SM103",
)

HEADS, CP, CP_RANK = 96, 16, 5
HP = HEADS // CP
HIDDEN, QA, KVL, NOPE, ROPE, V = 7168, 1536, 512, 128, 64, 128
FUSED_A = QA + KVL + ROPE  # 2112
GATE = HP * V
TOKENS_PER_REQUEST = 8
ROWS = (8, 16, 32)
REPLAYS = 2
EPS = 1e-6
# Tolerances (stated as measured on GB300 with these inputs, with margin).
FUSED_A_ERR_RATIO = 1.25
Q_ERR_RATIO = 1.05
Q_TOL = 0.06
OUT_TOL = 0.05
SWITCHES = (
    "TRTLLM_K3LL_HELIX_Q_ABSORB",
    "TRTLLM_K3LL_MLA_GATE_SIDE",
    "TRTLLM_K3LL_FP8W_LOWM_FUSED_A",
    "TRTLLM_K3LL_MLA_QKV_PDL",
)


def _pkg():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
        attention,
        flags,
        fp8w_lowm,
        implementation,
    )

    return types.SimpleNamespace(
        attention=attention, flags=flags, fp8w_lowm=fp8w_lowm, impl=implementation
    )


def _block_fp8(n, k, std, gen):
    """A checkpoint-style FP8 weight: N(0, std) quantized per 128 x 128 block with scale amax / 448 (FP32 scales)."""
    n_pad = -(-n // 128) * 128
    w = torch.randn((n_pad, k), device="cuda", generator=gen) * std
    blocks = w.view(n_pad // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-6) / 448.0
    codes = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n_pad, k)
    return codes[:n].contiguous(), scale[: -(-n // 128)].float().contiguous()


def _weight_read(codes, scale, partial=False):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    module = _Fp8BlockScaleWeightReadLinear(codes.clone(), scale.clone(), codes.shape[0])
    module._weights_transformed = False
    return module


def _norm(weight):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    norm = RMSNorm(hidden_size=weight.numel(), eps=EPS, dtype=torch.bfloat16, device="cuda")
    norm.weight.data.copy_(weight)
    return norm


class _Backend:
    """The MLA backend's ``mla_rope_generation`` as a recording stand-in (see the module docstring)."""

    def __init__(self, heads, record):
        self.num_heads, self.layer_idx, self.record = heads, 3, record
        self.has_fp8_kv_cache, self.has_fp4_kv_cache = True, False

    def mla_rope_generation(self, fused_q, q_pe, latent, metadata, cu_q, cu_kv, scheduler, bmm1, bmm2,
                            quant_q, fuse_fp4_q_quant=False, q_rope_applied=False):  # fmt: skip
        assert not fuse_fp4_q_quant
        self.record.append(("rope", latent.clone(), q_rope_applied, fused_q.shape[1]))
        bmm1.fill_(0.0625)
        bmm2.fill_(1.0)
        if not q_rope_applied:
            quant_q.view(torch.float8_e4m3fn).copy_(
                fused_q.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            )


class _SplitRead:
    """``MlaSplitRead.forward`` as a recording stand-in: returns this rank's heads' NoPE query bytes, scaled."""

    def __init__(self, record):
        self.record = record

    def forward(self, mqa, quant_q, meta, q_len, bmm1, bmm2, all_heads=False):
        tokens = meta.num_generations * q_len
        heads = HEADS if all_heads else HP
        q = quant_q[:tokens].view(torch.float8_e4m3fn).view(tokens, heads, KVL + ROPE)
        self.record.append(("split", q.view(torch.uint8).clone(), all_heads))
        if all_heads:
            q = q[:, CP_RANK * HP : (CP_RANK + 1) * HP]
        return (q[..., :KVL].float() * (1.0 / 16)).bfloat16().reshape(tokens, HP * KVL)


def _random_weights(gen):
    """Checkpoint-like random weights of one MLA layer (this rank's gate / V-B group only)."""
    kb = (torch.randn(HEADS, KVL, NOPE, device="cuda", generator=gen) * 64).clamp(-440, 440)
    vb = (torch.randn(HP, V, KVL, device="cuda", generator=gen) * 64).clamp(-440, 440)
    return {
        "fused": _block_fp8(FUSED_A, HIDDEN, 0.02, gen),
        "gate": _block_fp8(GATE, HIDDEN, 0.02, gen),
        "qb": _block_fp8(HEADS * (NOPE + ROPE), QA, 0.03, gen),
        "kb": kb.to(torch.float8_e4m3fn),
        "kb_scale": torch.rand(HEADS, KVL // 128, 1, device="cuda", generator=gen) * 1e-3 + 5e-4,
        "vb": vb.to(torch.float8_e4m3fn),
        "vb_scale": torch.rand(HP, 1, KVL // 128, device="cuda", generator=gen) * 1e-3 + 5e-4,
        "qa_norm": (1 + 0.1 * torch.randn(QA, device="cuda", generator=gen)).bfloat16(),
        "kv_norm": (1 + 0.1 * torch.randn(KVL, device="cuda", generator=gen)).bfloat16(),
    }


def _checkpoint_weights(path, layer):
    """One MLA layer of the K3 NVFP4 checkpoint at ``path`` (FP8 128x128 block projections), laid out as the target's
    loader lays it out for Helix rank ``CP_RANK``: [q_a ; kv_a] fused, every head's Q-B and K-B (``kv_b_proj``'s key
    rows, transposed), this rank's gate rows and V-B heads."""
    import json
    import os

    from safetensors import safe_open

    weight_map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    prefix = next(
        k[: -len("q_b_proj.weight")]
        for k in weight_map
        if k.endswith(f"layers.{layer}.self_attn.q_b_proj.weight")
    )

    def get(name):
        with safe_open(os.path.join(path, weight_map[prefix + name]), "pt", device="cuda") as f:
            return f.get_tensor(prefix + name)

    def fp8(name):
        # The checkpoint stores the 128x128 block scales as [N / 128, 1, K / 128, 1].
        weight = get(name + ".weight")
        scale = get(name + ".weight_scale").float()
        return weight, scale.reshape(
            -(-weight.shape[0] // 128), weight.shape[1] // 128
        ).contiguous()

    qa, qa_s = fp8("q_a_proj")
    kva, kva_s = fp8("kv_a_proj_with_mqa")
    gate, gate_s = fp8("g_proj")
    kvb, kvb_s = fp8("kv_b_proj")
    group = slice(CP_RANK * HP, (CP_RANK + 1) * HP)
    kvb = kvb.view(HEADS, NOPE + V, KVL)
    kvb_s = kvb_s.view(HEADS, (NOPE + V) // 128, KVL // 128)
    return {
        "fused": (torch.cat((qa, kva)).contiguous(), torch.cat((qa_s, kva_s)).contiguous()),
        "gate": (gate[group.start * V : group.stop * V].contiguous(),
                 gate_s[group.start * V // 128 : group.stop * V // 128].contiguous()),
        "qb": fp8("q_b_proj"),
        "kb": kvb[:, :NOPE].transpose(1, 2).contiguous(),
        "kb_scale": kvb_s[:, : NOPE // 128].transpose(1, 2).contiguous(),
        "vb": kvb[group, NOPE:].contiguous(),
        "vb_scale": kvb_s[group, NOPE // 128 :].contiguous(),
        "qa_norm": get("q_a_layernorm.weight").bfloat16(),
        "kv_norm": get("kv_a_layernorm.weight").bfloat16(),
    }  # fmt: skip


def _standin(w):
    """A K3DecodeMLA holding the decode path's state on weights ``w``, with every switch's load-time state built."""
    p = _pkg()
    mla = p.impl.K3DecodeMLA.__new__(p.impl.K3DecodeMLA)
    nn.Module.__init__(mla)
    record = []
    mla.record = record
    mla.q_lora_rank, mla.kv_lora_rank = QA, KVL
    mla.qk_nope_head_dim, mla.qk_rope_head_dim = NOPE, ROPE
    mla.qk_head_dim, mla.v_head_dim = NOPE + ROPE, V
    mla.num_heads_tp, mla.num_heads_tp_cp = HEADS, HP
    mla.mapping = types.SimpleNamespace(has_cp_helix=lambda: True, cp_rank=CP_RANK)
    mla.fused_a_raw, mla.q_b_raw = w["fused"], w["qb"]
    mla.kv_a_proj_with_mqa = _weight_read(*w["fused"])
    mla.g_proj = _weight_read(*w["gate"])
    mla.q_b_proj = _weight_read(*w["qb"])
    # The pre-change chain's fused q_a / kv_a: the same weight, not routed (FUSED_A had no routing).
    mla.stock_fused_a = _weight_read(*w["fused"])
    mla.q_a_layernorm, mla.kv_a_layernorm = _norm(w["qa_norm"]), _norm(w["kv_norm"])
    mla.k_b_proj_trans, mla.k_b_proj_trans_scale = w["kb"], w["kb_scale"]
    mla.v_b_proj, mla.v_b_proj_scale = w["vb"], w["vb_scale"]
    # Load order of the target: the decode Q-B copy from the raw scales, the low-M routing (FP8W_LOWM=1, FUSED_A=1 at
    # launch), then each projection's own post_load_weights; then K-B views, the backends, the split read, W_UK.
    mla.prepare_fp8_decode_weights()
    p.fp8w_lowm.install([mla.kv_a_proj_with_mqa, mla.g_proj, mla.q_b_proj], partial_n_tile=True)
    for module in (mla.kv_a_proj_with_mqa, mla.g_proj, mla.q_b_proj, mla.stock_fused_a):
        module.post_load_weights()
    assert all(m._k3ll_fp8w_lowm for m in (mla.kv_a_proj_with_mqa, mla.g_proj, mla.q_b_proj))
    heads = mla._decode_head_rows()
    mla._decode_k_b = mla.k_b_proj_trans[heads]
    mla._decode_k_b_scale = mla.k_b_proj_trans_scale[heads]
    mla.mqa, mla._decode_mqa = _Backend(HEADS, record), _Backend(HP, record)
    mla._split_read = _SplitRead(record)
    mla._fp8_decode_ready = True
    mla._q_absorb_w_kc = None
    mla._gate_side = p.impl._decode_mla_q.GateSide()
    # W_UK as a launch on a GPU the switch builds on (HELIX_SWITCH_CAPABILITIES) builds it.
    with mock.patch.object(
        p.impl, "HELIX_SWITCH_CAPABILITIES", (torch.cuda.get_device_capability(),)
    ):
        assert mla.prepare_helix_q_absorb() is None and mla._q_absorb_w_kc.dtype == torch.bfloat16
    return mla


def _pre_change(mla, x, meta):
    """``K3DecodeMLA``'s quantized-cache split-read decode before the query-side switches, on the stock fused GEMM."""
    p = _pkg()
    ag = torch.cat((mla.stock_fused_a(x), mla.g_proj(x)), dim=-1)
    heads = HP
    query, latent_cache = p.attention.mla_project_qkv(
        ag, mla.q_a_layernorm.weight, mla._fp8_qb_weight, mla._fp8_qb_scale, mla._decode_k_b,
        mla._decode_k_b_scale, mla.kv_a_layernorm.weight, float(mla.q_a_layernorm.variance_epsilon),
        float(mla.kv_a_layernorm.variance_epsilon), mla._fp8_qb_ue8m0,
    )  # fmt: skip
    q_pe = query.view(-1, heads, 576)[..., 512:]
    cu_q = torch.empty(meta.num_seqs + 1, dtype=torch.int32, device=ag.device)
    cu_kv = torch.empty_like(cu_q)
    scheduler = torch.empty(1, dtype=torch.uint32, device=ag.device)
    bmm1 = torch.empty(2, dtype=torch.float32, device=ag.device)
    bmm2 = torch.empty(1, dtype=torch.float32, device=ag.device)
    quant_q = torch.empty((ag.shape[0], heads, 576), dtype=torch.uint8, device=ag.device)
    mla._decode_mqa.mla_rope_generation(query.view(-1, heads, 576), q_pe, latent_cache, meta, cu_q, cu_kv,
                                        scheduler, bmm1, bmm2, quant_q, fuse_fp4_q_quant=False)  # fmt: skip
    latent = mla._split_read.forward(
        mla.mqa, quant_q, meta, ag.shape[0] // meta.num_generations, bmm1, bmm2
    )
    return p.attention.mla_output(
        latent, mla.v_b_proj, mla.v_b_proj_scale, ag, mla.q_lora_rank + mla.kv_lora_rank + ROPE
    )


@contextlib.contextmanager
def _switches(monkeypatch, **values):
    flags = _pkg().flags
    with monkeypatch.context() as m:
        for name in SWITCHES:
            m.setattr(flags, name, int(values.get(name, 0)))
        yield


def _meta(rows):
    reqs = rows // TOKENS_PER_REQUEST
    return types.SimpleNamespace(num_generations=reqs, num_seqs=reqs)


def _x(rows, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, HIDDEN, device="cuda", generator=gen).bfloat16()


def _run(mla, fn, x, meta):
    """``fn(x)``'s output and its records (rope latent, split query bytes)."""
    mla.record.clear()
    out = fn(x, meta)
    torch.cuda.synchronize()
    rope = [r for r in mla.record if r[0] == "rope"]
    split = [r for r in mla.record if r[0] == "split"]
    assert len(rope) == 1 and len(split) == 1
    return out, rope[0], split[0]


def _decode(mla):
    return lambda x, meta: mla._quantized_decode(x, None, meta)


def _bits(a, b) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    )


def _rel(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _graph(mla, fn, rows, monkeypatch, values):
    """Capture ``fn`` at ``rows`` with the side stream on; returns (static x, output, records, graph)."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    meta = _meta(rows)
    x = _x(rows, 1)
    with _switches(monkeypatch, **values):
        fn(x, meta)  # compiles every kernel outside the capture
        torch.cuda.synchronize()
        mla.record.clear()
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.graph(graph):
            out = fn(x, meta)
    records = list(mla.record)
    return x, out, records, graph, meta


@pytest.fixture(scope="module")
def mla():
    torch.manual_seed(0)
    return _standin(_random_weights(torch.Generator(device="cuda").manual_seed(1234)))


@pytest.fixture(scope="module")
def mla_checkpoint():
    """K3LL_MLA_CKPT=<K3 NVFP4 checkpoint dir> [K3LL_MLA_LAYER=<an MLA layer, 3>]: the same stand-in on that layer."""
    import os

    path = os.environ.get("K3LL_MLA_CKPT")
    if not path:
        pytest.skip("K3LL_MLA_CKPT not set")
    return _standin(_checkpoint_weights(path, int(os.environ.get("K3LL_MLA_LAYER", "3"))))


@pytest.mark.parametrize("rows", ROWS)
def test_switches_off_equal_pre_change(mla, monkeypatch, rows):
    meta = _meta(rows)
    for seed in (10, 11):
        x = _x(rows, seed + rows)
        with _switches(monkeypatch):
            out, rope, split = _run(mla, _decode(mla), x, meta)
        ref, ref_rope, ref_split = _run(mla, lambda x, m: _pre_change(mla, x, m), x, meta)
        assert _bits(out, ref)
        assert _bits(rope[1], ref_rope[1]) and rope[2] is False and rope[3] == HP
        assert _bits(split[1], ref_split[1]) and split[2] is False


@pytest.mark.parametrize("rows", ROWS)
def test_q_absorb_unsupported_gpu_equals_off(mla, monkeypatch, rows):
    """HELIX_Q_ABSORB at 1 on a GPU it does not build on (``HELIX_SWITCH_CAPABILITIES`` without this one): no W_UK,
    and the output, query bytes and latent row bit for bit the switch-off ones; with W_UK built the same inputs take
    the all-head path."""
    p = _pkg()
    meta = _meta(rows)
    x = _x(rows, 20 + rows)
    built = mla._q_absorb_w_kc
    try:
        mla._q_absorb_w_kc = None
        with monkeypatch.context() as m:
            m.setattr(p.impl, "HELIX_SWITCH_CAPABILITIES", ())
            why = mla.prepare_helix_q_absorb()
        assert why is not None and "SM103" in why and mla._q_absorb_w_kc is None
        with _switches(monkeypatch):
            off, off_rope, off_split = _run(mla, _decode(mla), x, meta)
        with _switches(monkeypatch, TRTLLM_K3LL_HELIX_Q_ABSORB=1):
            on, on_rope, on_split = _run(mla, _decode(mla), x, meta)
        assert _bits(on, off) and on_split[2] is False
        assert _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])
    finally:
        mla._q_absorb_w_kc = built
    with _switches(monkeypatch, TRTLLM_K3LL_HELIX_Q_ABSORB=1):
        _, _, split = _run(mla, _decode(mla), x, meta)
    assert split[2] is True


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize(
    "values",
    [
        {},
        {"TRTLLM_K3LL_MLA_GATE_SIDE": 1},
        {"TRTLLM_K3LL_HELIX_Q_ABSORB": 1},
        {"TRTLLM_K3LL_FP8W_LOWM_FUSED_A": 1},
        {"TRTLLM_K3LL_MLA_QKV_PDL": 1},
        {
            "TRTLLM_K3LL_MLA_QKV_PDL": 1,
            "TRTLLM_K3LL_MLA_GATE_SIDE": 1,
            "TRTLLM_K3LL_FP8W_LOWM_FUSED_A": 1,
        },
        {name: 1 for name in SWITCHES},
    ],
    ids=["off", "gate_side", "q_absorb", "fused_a", "qkv_pdl", "head_group_on", "all"],
)
def test_graph_replay_equals_eager(mla, monkeypatch, rows, values):
    """Each arm captured with the side stream on, replayed with new inputs: bit for bit its eager call; the off arm's
    replay also bit for bit the pre-change chain."""
    x, out, records, graph, meta = _graph(mla, _decode(mla), rows, monkeypatch, values)
    for replay in range(REPLAYS):
        x.copy_(_x(rows, 100 + replay))
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        replay_split = next(r for r in records if r[0] == "split")[1].clone()
        with _switches(monkeypatch, **values):
            eager, _, eager_split = _run(mla, _decode(mla), x.clone(), meta)
        assert _bits(replayed, eager)
        assert _bits(replay_split, eager_split[1])
        if not values:
            ref, _, _ = _run(mla, lambda x, m: _pre_change(mla, x, m), x.clone(), meta)
            assert _bits(replayed, ref)


@pytest.mark.parametrize("rows", ROWS)
def test_gate_side_bitwise(mla, monkeypatch, rows):
    meta = _meta(rows)
    x = _x(rows, 20 + rows)
    with _switches(monkeypatch):
        off, off_rope, off_split = _run(mla, _decode(mla), x, meta)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_GATE_SIDE=1):
        on, on_rope, on_split = _run(mla, _decode(mla), x, meta)
    assert _bits(on, off) and _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])


@pytest.mark.parametrize("rows", ROWS)
def test_qkv_pdl_bitwise(mla, monkeypatch, rows):
    meta = _meta(rows)
    x = _x(rows, 25 + rows)
    with _switches(monkeypatch):
        off, off_rope, off_split = _run(mla, _decode(mla), x, meta)
    with _switches(monkeypatch, TRTLLM_K3LL_MLA_QKV_PDL=1):
        on, on_rope, on_split = _run(mla, _decode(mla), x, meta)
    assert _bits(on, off) and _bits(on_rope[1], off_rope[1]) and _bits(on_split[1], off_split[1])


@pytest.mark.parametrize("weights", ["random", "checkpoint"])
@pytest.mark.parametrize("rows", ROWS)
def test_fused_a_numerics(request, weights, monkeypatch, rows):
    mla = request.getfixturevalue("mla" if weights == "random" else "mla_checkpoint")
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm

    meta = _meta(rows)
    x = _x(rows, 30 + rows)
    module = mla.kv_a_proj_with_mqa
    assert k3_fp8w_lowm_gemm.can_run(
        x, module.weight, module.weight_scale, allow_partial_n_tile=True
    )
    from tensorrt_llm.quantization.utils.fp8_utils import resmooth_to_fp8_e8m0

    codes, scale = resmooth_to_fp8_e8m0(mla.fused_a_raw[0].clone(), mla.fused_a_raw[1].clone())
    reference = x.float() @ _dequant(codes, scale.float()).t()
    with _switches(monkeypatch):
        stock = module(x)
        off, _, _ = _run(mla, _decode(mla), x, meta)
    with _switches(monkeypatch, TRTLLM_K3LL_FP8W_LOWM_FUSED_A=1):
        lowm = module(x)
        on, _, _ = _run(mla, _decode(mla), x, meta)
    assert _bits(stock, mla.stock_fused_a(x))
    assert _bits(lowm, torch.ops.trtllm.k3_fp8w_lowm_gemm(x, module.weight, module.weight_scale))
    err_stock, err_lowm = _rel(stock, reference), _rel(lowm, reference)
    print(
        f"fused_a {weights} rows={rows}: rel err stock {err_stock:.3e} lowm {err_lowm:.3e}; out {_rel(on, off):.3e}"
    )
    assert err_lowm <= FUSED_A_ERR_RATIO * err_stock
    assert _rel(on, off) < OUT_TOL


def _dequant(codes, scale):
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[
        : codes.shape[0], : codes.shape[1]
    ]
    return codes.float() * expanded


def _q_reference(mla, ag):
    """FP32 every-head query ``[rows, heads, 576]`` from the resmoothed Q-B (the projection's weight) and K-B."""
    from tensorrt_llm.quantization.utils.fp8_utils import resmooth_to_fp8_e8m0

    qa = ag[:, :QA].float()
    qn = qa * torch.rsqrt(qa.pow(2).mean(-1, keepdim=True) + EPS) * mla.q_a_layernorm.weight.float()
    qb, qbs = resmooth_to_fp8_e8m0(mla.q_b_raw[0].clone(), mla.q_b_raw[1].clone())
    q = (qn @ _dequant(qb, qbs.float()).t()).view(-1, HEADS, NOPE + ROPE)
    kb = mla.k_b_proj_trans.float() * mla.k_b_proj_trans_scale.repeat_interleave(128, 1)
    absorbed = torch.einsum("tbn,bln->tbl", q[..., :NOPE], kb)
    return torch.cat((absorbed, q[..., NOPE:]), -1)


@pytest.mark.parametrize("weights", ["random", "checkpoint"])
@pytest.mark.parametrize("rows", ROWS)
def test_q_absorb_numerics(request, weights, monkeypatch, rows):
    mla = request.getfixturevalue("mla" if weights == "random" else "mla_checkpoint")
    p = _pkg()
    meta = _meta(rows)
    x = _x(rows, 40 + rows)
    with _switches(monkeypatch):
        off, off_rope, off_split = _run(mla, _decode(mla), x, meta)
    with _switches(monkeypatch, TRTLLM_K3LL_HELIX_Q_ABSORB=1):
        on, on_rope, on_split = _run(mla, _decode(mla), x, meta)
    assert on_rope[2] is True and on_split[2] is True and on_split[1].shape == (rows, HEADS, 576)
    assert _bits(on_rope[1], off_rope[1])
    # The current path for every head: mla_project_qkv with every head's Q-B / K-B (bit for bit the 16 head groups
    # gathered), then the cache kernel's FP8.
    ag = torch.cat((mla.stock_fused_a(x), mla.g_proj(x)), dim=-1)
    qb_all, qbs_all = mla.q_b_raw[0].clone(), mla.q_b_raw[1].clone()
    from tensorrt_llm.quantization.utils.fp8_utils import resmooth_to_fp8_e8m0

    qb_all, qbs_all = resmooth_to_fp8_e8m0(qb_all, qbs_all)
    query, _ = p.attention.mla_project_qkv(
        ag, mla.q_a_layernorm.weight, qb_all, qbs_all, mla.k_b_proj_trans, mla.k_b_proj_trans_scale,
        mla.kv_a_layernorm.weight, EPS, EPS, True,
    )  # fmt: skip
    current = query.view(rows, HEADS, 576).float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    group = slice(CP_RANK * HP, (CP_RANK + 1) * HP)
    assert _bits(current[:, group].to(torch.float8_e4m3fn), off_split[1].view(torch.float8_e4m3fn))
    new = on_split[1].view(torch.float8_e4m3fn).float()
    reference = _q_reference(mla, ag)
    err_cur, err_new = _rel(current, reference), _rel(new, reference)
    err_group = _rel(new[:, group], off_split[1].view(torch.float8_e4m3fn).float())
    print(f"q {weights} rows={rows}: rel err vs fp32 current {err_cur:.3e} new {err_new:.3e}; "
          f"group vs off {err_group:.3e}; out {_rel(on, off):.3e}")  # fmt: skip
    assert err_new <= Q_ERR_RATIO * err_cur
    assert err_group < Q_TOL
    assert _rel(on, off) < OUT_TOL
