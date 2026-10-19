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
"""KDA verify with ``TRTLLM_K3_KDA_L2_PREFETCH``: the mixer prefetches the next KDA layer's
projection weights from the projection aux stream after its own projections.

The prefetch is a hint, so the verify results (outputs, committed state, conv windows, replay
caches) must be bitwise those of the run without it, eagerly and under CUDA graph replay; the
prefetch must target the next KDA layer of the same model and nothing for the last one.
"""

from types import SimpleNamespace

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cutlass  # noqa: F401
    from fla.ops.kda import fused_recurrent_kda  # noqa: F401

    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="needs an SM100/SM103 GPU",
    ),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

HIDDEN = 512
H = 8
K = 128
W = 4
NUM_SPEC = 2
B = 2
MUTATED = (
    "ssm",
    "kda_conv_q",
    "kda_conv_k",
    "kda_conv_v",
    "kda_k_cache",
    "kda_g_cache",
    "kda_v_cache",
    "kda_beta_cache",
)


def _cfg():
    return KimiLinearConfig(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            kda_layers=[1],
            full_attn_layers=[],
            num_heads=H,
            head_dim=K,
            short_conv_kernel_size=W,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )


@torch.no_grad()
def _runtime(cfg, layer_idx, aux, seed):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig

    rt = KimiKDALinearAttention(
        cfg,
        layer_idx=layer_idx,
        aux_stream=aux,
        model_config=ModelConfig(quant_config=QuantConfig()),
    ).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for name, p in rt.named_parameters():
        if name.endswith("A_log"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.5)
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.1)
        else:
            p.copy_((torch.randn(p.shape, generator=gen, device="cuda") * 0.03).to(p.dtype))
    rt.finalize_decode_weights()
    assert rt._qkvg_proj_weight is not None and rt._bfa_proj_weight is not None
    return rt


def _state(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    d = H * K
    s = W - 1 + NUM_SPEC

    def conv():
        return (torch.randn(B, s, d, generator=gen, device="cuda") * 0.5).bfloat16().transpose(1, 2)

    cache = SimpleNamespace(
        kda_conv_q=conv(),
        kda_conv_k=conv(),
        kda_conv_v=conv(),
        kda_k_cache=(torch.randn(B, NUM_SPEC, d, generator=gen, device="cuda") * 0.08).bfloat16(),
        kda_g_cache=-torch.rand(B, NUM_SPEC, d, generator=gen, device="cuda") * 0.5,
        kda_v_cache=torch.randn(B, NUM_SPEC, d, generator=gen, device="cuda").bfloat16(),
        kda_beta_cache=torch.rand(B, NUM_SPEC, H, generator=gen, device="cuda").bfloat16(),
        prev_num_accepted_tokens=torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        intermediate_conv_window=None,
        intermediate_ssm=None,
    )
    ssm = torch.randn(B, H, K, K, generator=gen, device="cuda") * 0.05
    x = (torch.randn(B * (NUM_SPEC + 1), HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
    return cache, ssm, x


def _snapshot(cache, ssm):
    snap = {name: getattr(cache, name).clone() for name in MUTATED if name != "ssm"}
    snap["ssm"] = ssm.clone()
    return snap


def _restore(cache, ssm, snap):
    for name, value in snap.items():
        (ssm if name == "ssm" else getattr(cache, name)).copy_(value)


@pytest.mark.parametrize("graph", [False, True])
@torch.no_grad()
def test_l2_prefetch_is_a_hint(graph, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_l2_prefetch
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_FUSED_FAB", True)
    monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_PDL", True)
    cfg = _cfg()
    aux = torch.cuda.Stream()
    first, second = _runtime(cfg, 3, aux, seed=1), _runtime(cfg, 7, aux, seed=2)
    other_model = _runtime(_cfg(), 5, aux, seed=3)
    # the next KDA layer of the same model only
    assert first._next_kda_layer() is second
    assert second._next_kda_layer() is None
    assert other_model._next_kda_layer() is None
    ranges = second.l2_prefetch_ranges()
    targets = {addr for addr, _ in ranges}
    for t in (second._qkvg_proj_weight, second._bfa_proj_weight, second.f_b_proj.weight):
        assert t.data_ptr() in targets
    assert all(nbytes > 0 and addr % 16 == 0 and nbytes % 16 == 0 for addr, nbytes in ranges)

    calls = []
    run = k3_l2_prefetch.run_k3_l2_prefetch

    def recording_run(rngs, **kwargs):
        calls.append(list(rngs))
        return run(rngs, **kwargs)

    monkeypatch.setattr(k3_l2_prefetch, "run_k3_l2_prefetch", recording_run)
    cache, ssm, x = _state(seed=4)
    snap = _snapshot(cache, ssm)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    main = torch.cuda.Stream()

    def verify(rt):
        out = rt.forward_verify(x, NUM_SPEC + 1, cache, None, ssm, slots)
        # the model's MoE block joins the aux stream back; here the test does
        torch.cuda.current_stream().wait_stream(aux)
        return out

    def run_once(prefetch, rt=first):
        monkeypatch.setattr(kimi_kda_mixer, "_KDA_L2_PREFETCH", prefetch)
        _restore(cache, ssm, snap)
        with torch.cuda.stream(main), with_multi_stream(True):
            if not graph:
                out = verify(rt).clone()
            else:
                verify(rt)  # warm-up (compiles, allocates the fab state)
                _restore(cache, ssm, snap)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=main):
                    out_g = verify(rt)
                _restore(cache, ssm, snap)
                g.replay()
                out = out_g.clone()
        torch.cuda.synchronize()
        return out, _snapshot(cache, ssm)

    ref_out, ref_state = run_once(False)
    assert not calls
    out, state = run_once(True)
    assert calls and all(c == ranges for c in calls)
    assert torch.equal(out, ref_out)
    for name in MUTATED:
        assert torch.equal(state[name], ref_state[name]), name
    # the last KDA layer prefetches nothing
    calls.clear()
    run_once(True, rt=second)
    assert not calls
