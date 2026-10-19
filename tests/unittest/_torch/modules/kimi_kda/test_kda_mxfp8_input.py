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
"""KDA verify with an MXFP8 projection input (``TRTLLM_K3_ATTN_RES_MXFP8``).

The decoder layer hands the mixer the MXFP8 form of its rows (``next_input_mxfp8``); the fused FP8
QKVG projection then runs the low-M GEMM on it. The verify results (output, committed state, conv
windows, replay caches) must be bitwise those of the BF16 input, and the hand-off is consumed by
one forward.
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
def _runtime_with_fp8_qkvg(aux, seed):
    """A KDA mixer whose fused QKVG projection is the FP8 weight-read linear (low-M GEMM)."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm.models.modeling_utils import QuantConfig

    rt = KimiKDALinearAttention(
        _cfg(), layer_idx=1, aux_stream=aux, model_config=ModelConfig(quant_config=QuantConfig())
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
    weight = rt._qkvg_proj_weight
    linear = kimi._Fp8BlockScaleWeightReadLinear(
        *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight), weight.shape[0]
    )
    rt.qkvg_proj = linear
    rt.qkvg_split_sizes = [H * K] * 4
    rt._qkvg_proj_weight = None
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
def test_mxfp8_input_is_bitwise(graph, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_mxfp8 as mx
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    monkeypatch.setattr(kimi, "_K3_FP8W_LOWM_GEMM", True)
    monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_FUSED_FAB", True)
    monkeypatch.setattr(kimi_kda_mixer, "_KDA_DECODE_PDL", True)
    aux = torch.cuda.Stream()
    rt = _runtime_with_fp8_qkvg(aux, seed=1)
    cache, ssm, x = _state(seed=4)
    assert rt.qkvg_takes_mxfp8(x.shape[0])
    snap = _snapshot(cache, ssm)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    main = torch.cuda.Stream()
    calls = []
    run_mx = mx.run_k3_fp8w_lowm_gemm_mxfp8

    def recording_run(*args, **kwargs):
        calls.append(args[0].shape)
        return run_mx(*args, **kwargs)

    monkeypatch.setattr(mx, "run_k3_fp8w_lowm_gemm_mxfp8", recording_run)

    def verify(mxfp8_input):
        if mxfp8_input is not None:
            rt.next_input_mxfp8 = mxfp8_input
        out = rt.forward_verify(x, NUM_SPEC + 1, cache, None, ssm, slots)
        assert rt.next_input_mxfp8 is None
        torch.cuda.current_stream().wait_stream(aux)
        return out

    def run_once(use_mxfp8):
        _restore(cache, ssm, snap)
        with torch.cuda.stream(main), with_multi_stream(True):
            mxfp8_input = mx.mxfp8_quantize_reference(x) if use_mxfp8 else None
            if not graph:
                out = verify(mxfp8_input).clone()
            else:
                verify(mxfp8_input)  # warm-up (compiles, allocates the fab state)
                _restore(cache, ssm, snap)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=main):
                    out_g = verify(mxfp8_input)
                _restore(cache, ssm, snap)
                g.replay()
                out = out_g.clone()
        torch.cuda.synchronize()
        return out, _snapshot(cache, ssm)

    ref_out, ref_state = run_once(False)
    assert not calls
    out, state = run_once(True)
    assert calls and all(shape == (x.shape[0], HIDDEN) for shape in calls)
    assert torch.equal(out, ref_out)
    for name in MUTATED:
        assert torch.equal(state[name], ref_state[name]), name
