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
"""trtllm::k3_moe_nvfp4 (the NVFP4 port of the persistent k3_moe kernel, M <= 8 and the wide build) on one GPU, at the
Kimi K3 TP16 deployment's routed-expert rank layout (experts TP4 x EP4: 224 local experts, intermediate 768 per rank),
on random checkpoint-format NVFP4 experts put through TRT-LLM's own TRTLLM-Gen NVFP4 loader: against the stock path
(trtllm::fp4_block_scale_moe_runner pre-routed with SiTU, on the same buffers and scalars the TRTLLM backend passes)
and an fp64 reference of the formula (op_nvfp4's docstring), for random routing and for routing that puts 0, 4 and 16
of each token's experts on this rank. The NVFP4 intermediate rounds differently from the stock cubin's near block
boundaries, so the gate is relative RMS against the fp64 reference, no worse than a small multiple of the stock path's;
plus run-to-run identical bits and the scratch re-armed after every call."""

import functools
from types import SimpleNamespace

import pytest
import torch


def _is_sm10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _is_sm10x(), reason="k3_moe_nvfp4 needs an sm_10x GPU")

H, TOP_K, NUM_EXPERTS, SV = 3584, 16, 896, 16
I_TP, E_LOCAL, MOE_TP, TP_RANK, EP_RANK = 768, 224, 4, 1, 1
OFFSET = EP_RANK * E_LOCAL
GATE_CAP, LINEAR_CAP = 4.0, 25.0
X_AMAX = 6.0  # the calibrated latent amax behind fc31_input_scale
A_AMAX = 40.0  # the calibrated intermediate amax behind fc2_input_scale
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _ops():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op_nvfp4  # noqa: F401

    return torch.ops.trtllm


def _rand_nvfp4(rows, k, gen):
    """Random checkpoint-format NVFP4 codes (packed [rows, k / 2], low nibble = even k) and UE4M3 scales [rows, k / 16]
    in [0.25, 1)."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    sf = (torch.rand(rows, k // SV, device="cuda", generator=gen) * 0.75 + 0.25).to(
        torch.float8_e4m3fn
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), sf


@functools.lru_cache(maxsize=None)
def _experts(seed: int = 20261020):
    """This rank's experts through NVFP4TRTLLMGenFusedMoEMethod's loader (the buffers both kernels read), their global
    scales as the TRTLLM backend derives them, and the rank's logical slices of the checkpoint tensors."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod

    i_full = I_TP * MOE_TP
    method = NVFP4TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=MOE_TP, tp_rank=TP_RANK, intermediate_size=i_full,
                             intermediate_size_per_partition=I_TP, hidden_size=H, scaling_vector_size=SV,
                             intermediate_size_expand_ratio=2, is_gated_activation=True,
                             expert_size_per_partition=E_LOCAL, bias=False)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    proc = dict(
        w31=torch.empty(E_LOCAL, 2 * I_TP, H // 2, **kw),
        w31s=torch.empty(E_LOCAL, 2 * I_TP, H // SV, dtype=torch.float8_e4m3fn, device="cuda"),
        w2=torch.empty(E_LOCAL, H, I_TP // 2, **kw),
        w2s=torch.empty(E_LOCAL, H, I_TP // SV, dtype=torch.float8_e4m3fn, device="cuda"),
    )
    raw = {name: [] for name in ("up", "up_s", "gate", "gate_s", "down", "down_s")}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = TP_RANK * I_TP, (TP_RANK + 1) * I_TP
    # Per-expert weight_scale_2, sized so FC1's dequantized outputs land around std 2 (SiTU's interesting range).
    ws2_1 = (torch.rand(E_LOCAL, device="cuda", generator=gen) * 0.5 + 0.75) * 2.5e-3
    ws2_2 = (torch.rand(E_LOCAL, device="cuda", generator=gen) * 0.5 + 0.75) * 1.0e-3
    for e in range(E_LOCAL):
        w1, w1s = _rand_nvfp4(i_full, H, gen)  # gate
        w3, w3s = _rand_nvfp4(i_full, H, gen)  # up
        w2, w2s = _rand_nvfp4(H, i_full, gen)  # down
        method.load_expert_w3_w1_weight(module, w1, w3, proc["w31"][e])
        method.load_expert_w2_weight(module, w2, proc["w2"][e])
        method.load_expert_w3_w1_weight_scale_nvfp4(module, w1s, w3s, proc["w31s"][e])
        method.load_expert_w2_weight_scale_nvfp4(module, w2s, proc["w2s"][e])
        method._shuffle_w3_w1_weight(module, proc["w31"][e])
        method._shuffle_w2_weight(proc["w2"][e])
        method._shuffle_and_interleave_w3_w1_weight_scale(proc["w31s"][e])
        method._shuffle_and_interleave_w2_weight_scale(proc["w2s"][e])
        raw["up"].append(w3[lo:hi])
        raw["up_s"].append(w3s[lo:hi])
        raw["gate"].append(w1[lo:hi])
        raw["gate_s"].append(w1s[lo:hi])
        raw["down"].append(w2[:, lo // 2 : hi // 2].contiguous())
        raw["down_s"].append(w2s[:, lo // SV : hi // SV].contiguous())
    torch.cuda.synchronize()
    # The backend's scalars: input scales are the checkpoint's amax / (6 * 448); fc31/fc2_input_scale their
    # reciprocals (the global encode scales), alpha = weight_scale_2 * input_scale.
    gs1 = 6.0 * 448.0 / X_AMAX
    gs2 = 6.0 * 448.0 / A_AMAX
    scales = dict(
        fc31_input_scale=torch.tensor([gs1], dtype=torch.float32, device="cuda"),
        fc2_input_scale=torch.tensor([gs2], dtype=torch.float32, device="cuda"),
        fc31_alpha=(ws2_1 / gs1).float().contiguous(),
        fc2_alpha=(ws2_2 / gs2).float().contiguous(),
    )
    return proc, raw, scales


def _deq_w(packed, sf):
    lut = torch.tensor(_E2M1, device=packed.device, dtype=torch.float64)
    vals = torch.empty(
        packed.shape[0], packed.shape[1] * 2, device=packed.device, dtype=torch.float64
    )
    vals[:, 0::2] = lut[(packed & 0xF).long()]
    vals[:, 1::2] = lut[(packed >> 4).long()]
    return vals * sf.double().repeat_interleave(SV, dim=1)


def _round_e2m1(v):
    """Round to the nearest E2M1 value (ties to even code), saturated at 6."""
    a = v.abs().clamp(max=6.0)
    grid = _E2M1_POS.to(v.device)
    d = (a.unsqueeze(-1) - grid).abs()
    best = d.min(dim=-1, keepdim=True).values
    # ties: take the even code index (0, 1.0, 2.0, 4.0 have even codes)
    cand = (d == best) & (torch.arange(8, device=v.device) % 2 == 0)
    idx = torch.where(cand.any(-1), cand.float().argmax(-1), d.argmin(-1))
    return grid[idx] * torch.sign(v)


def _q4(act):
    """NVFP4 per 16 columns: UE4M3 scale of amax / 6 (round to nearest), E2M1 codes; dequantized."""
    rows, cols = act.shape
    blocks = act.reshape(rows, cols // SV, SV)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = (amax / 6.0).float().to(torch.float8_e4m3fn).double()
    q = torch.where(sf > 0, _round_e2m1(blocks / sf.clamp_min(1e-30)), torch.zeros_like(blocks))
    return (q * sf).reshape(rows, cols)


def _reference(raw, scales, x_deq, ids, weights):
    """fp64 routed MoE over this rank's experts from the checkpoint tensors (op_nvfp4's formula)."""
    a1 = scales["fc31_alpha"].double()
    a2 = scales["fc2_alpha"].double()
    gs2 = scales["fc2_input_scale"].double().item()
    out = torch.zeros(x_deq.shape[0], H, device="cuda", dtype=torch.float64)
    for e in range(E_LOCAL):
        tok, slot = (ids == OFFSET + e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = x_deq[tok]
        up = (xe @ _deq_w(raw["up"][e], raw["up_s"][e]).t()) * a1[e]
        gate = (xe @ _deq_w(raw["gate"][e], raw["gate_s"][e]).t()) * a1[e]
        act = GATE_CAP * torch.tanh(gate / GATE_CAP) * torch.sigmoid(gate)
        act = act * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP)) * gs2
        y = (_q4(act) @ _deq_w(raw["down"][e], raw["down_s"][e]).t()) * a2[e]
        out.index_add_(0, tok, y * weights[tok, slot].double().unsqueeze(1))
    return out


def _deq_x(x_fp4, x_sf):
    m = x_fp4.shape[0]
    return _deq_w(x_fp4, x_sf.view(torch.float8_e4m3fn).reshape(m, H // SV))


def _quant_x(x, scales):
    q, sf = _ops().fp4_quantize(x, scales["fc31_input_scale"], SV, False, False)
    m = x.shape[0]
    sf = sf.view(torch.uint8).reshape(-1)[: m * (H // SV)].reshape(m, H // SV).contiguous()
    return q.view(torch.uint8).reshape(m, H // 2).contiguous(), sf


def _stock(proc, scales, x_fp4, x_sf, ids, w):
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    alpha = torch.full((E_LOCAL,), GATE_CAP, dtype=torch.float32, device="cuda")
    beta = (LINEAR_CAP / scales["fc31_alpha"]).float().contiguous()
    # SiTU is not linear in the gate, so trtllm-gen's FC1 scaleC is the intermediate's quantization scale alone
    # (fc31_alpha reaches the activation through the gate scalar); see quantization._fc31_scale_c_omits_dequant.
    scale_c = scales["fc2_input_scale"].float().reshape(1).expand(E_LOCAL).contiguous()
    w31s = proc["w31s"].view(torch.float8_e4m3fn)
    w2s = proc["w2s"].view(torch.float8_e4m3fn)
    return _ops().fp4_block_scale_moe_runner(
        None, None, x_fp4, x_sf.view(torch.float8_e4m3fn).reshape(-1), proc["w31"], w31s, None, alpha,
        beta, None, proc["w2"], w2s, None, scale_c, scales["fc31_alpha"], scales["fc2_alpha"], NUM_EXPERTS,
        TOP_K, 1, 1, I_TP, OFFSET, E_LOCAL, 1.0, int(RoutingMethodType.DeepSeekV3), True,
        int(ActType_TrtllmGen.SiTu), topk_weights=w, topk_ids=ids,
    )[0]  # fmt: skip


@functools.lru_cache(maxsize=None)
def _layer(wide: bool):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op_nvfp4

    _ops()
    proc, _, scales = _experts()
    dev = torch.device("cuda", torch.cuda.current_device())
    state = (op_nvfp4.K3MoeNvfp4WideState if wide else op_nvfp4.K3MoeNvfp4State)(dev, I_TP, E_LOCAL)
    return state.layer(proc["w31"], proc["w31s"], proc["w2"], proc["w2s"], scales["fc31_alpha"],
                       scales["fc2_alpha"], scales["fc2_input_scale"])  # fmt: skip


def _scratch_rearmed(layer):
    st = layer.state
    mod = st.mod
    cs = st.cs.view(mod.G_CAP, 8, mod.K2_TILES, mod.SFB_GROUP_BYTES)
    return (
        bool((st.c == mod.C_SENTINEL_I8).all())
        and bool((cs[..., :4] == -1).all())
        and bool((layer.counters == 0).all())
    )


def _tokens(case: str, m: int, seed: int = 7):
    gen = torch.Generator(device="cuda").manual_seed(seed + m)
    x = (torch.randn(m, H, device="cuda", generator=gen) * 1.2).to(torch.bfloat16)
    if case == "random":
        logits = torch.randn(m, NUM_EXPERTS, device="cuda", generator=gen)
        ids = logits.topk(TOP_K, dim=-1).indices.int()
    else:
        n_local = {"none_local": 0, "4_local": 4, "16_local": 16}[case]
        ids = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
        for t in range(m):
            local = OFFSET + torch.randperm(E_LOCAL, device="cuda", generator=gen)[:n_local]
            remote = torch.randperm(NUM_EXPERTS - E_LOCAL, device="cuda", generator=gen)[
                : TOP_K - n_local
            ]
            remote = torch.where(remote >= OFFSET, remote + E_LOCAL, remote)
            ids[t] = torch.cat([local, remote]).int()
    w = torch.rand(m, TOP_K, device="cuda", generator=gen)
    w = (w / w.sum(-1, keepdim=True) * 2.5).to(torch.bfloat16)
    return x, ids.contiguous(), w.contiguous()


def _rel_rms(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()


CASES = ["random", "4_local", "16_local", "none_local"]


@pytest.mark.parametrize(
    "wide,m", [(False, 1), (False, 3), (False, 8), (True, 8), (True, 16), (True, 32)]
)
@pytest.mark.parametrize("case", CASES)
def test_k3_moe_nvfp4(case, wide, m):
    proc, raw, scales = _experts()
    layer = _layer(wide)
    x, ids, w = _tokens(case, m)
    x_fp4, x_sf = _quant_x(x, scales)
    y = layer(x_fp4, x_sf, ids, w, OFFSET)
    y2 = layer(x_fp4, x_sf, ids, w, OFFSET)
    torch.cuda.synchronize()
    assert torch.isfinite(y.float()).all()
    assert torch.equal(y.view(torch.int16), y2.view(torch.int16)), "not run-to-run identical"
    assert _scratch_rearmed(layer), "scratch not re-armed"
    if case == "none_local":
        assert bool((y == 0).all())
        return
    ref = _reference(raw, scales, _deq_x(x_fp4, x_sf), ids, w)
    stock = _stock(proc, scales, x_fp4, x_sf, ids, w)
    e_k = _rel_rms(y, ref)
    e_s = _rel_rms(stock, ref)
    e_ks = _rel_rms(y, stock)
    print(f"[k3_moe_nvfp4] case={case} wide={wide} m={m}: rel_rms kernel/ref {e_k:.4g} stock/ref {e_s:.4g} "
          f"kernel/stock {e_ks:.4g}")  # fmt: skip
    assert e_k <= max(2.0 * e_s, 1e-3) and e_k < 0.05, (e_k, e_s)
