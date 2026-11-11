# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix8 glue's FP8 weight read (the K3 low-M GEMM folded in) against the family one swapped
in with the helix8 constants: the same scales and, on every row count, the same bits.

The attention projections and the MoE MLPs of the helix8 targets all run through it, so the folded
dispatch (no generic low-M GEMM, ``lowm_weight_scale`` wherever the Rubin MXFP8 path is) must pick
the kernels the family module picks with ``fp8w_lowm_gemm`` (and, on ``routed_expert_down_proj``,
``ld_fused``) set.
"""

import pytest
import torch
from torch import nn

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs a block-scaled FP8 GPU (SM100 / SM103 / SM107)",
)

ROWS = (1, 2, 3, 8, 16, 17, 31, 32, 64, 97, 100, 128, 129, 256, 1000)
# K3's FP8 projection shapes: q_a + kv_a fused, q_b, kv_b, o (MLA), KDA o / q, the latent down /
# up and the shared expert's gate_up / down.
SHAPES = [
    (2112, 7168),
    (1152, 1536),
    (2304, 512),
    (7168, 1024),
    (4096, 7168),
    (3584, 7168),
    (7168, 3584),
    (6144, 7168),
    (7168, 3072),
]


def _modules():
    import importlib

    layers = importlib.import_module(f"{_FAMILY}.sm107_parts.layers")
    family = importlib.import_module(f"{_FAMILY}.sm107_parts.fp8_weight_read")
    helix8 = importlib.import_module(f"{_FAMILY}.sm107_helix8_parts.fp8_weight_read")
    constants = importlib.import_module(
        f"{_FAMILY}.kimi_k3_nvfp4__sm_107__helix8_moeep8.constants"
    ).CONSTANTS
    return layers, family, helix8, constants


def _checkpoint_fp8(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    blocks = torch.nn.functional.pad(weight, (0, -k % 128, 0, -n % 128)).view(
        -(-n // 128), 128, -(-k // 128), 128
    )
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
    expanded = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return (weight / expanded).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("attr", ["o_proj", "routed_expert_down_proj"])
@pytest.mark.parametrize("n,k", SHAPES)
@torch.inference_mode()
def test_folded_weight_read_matches_family_with_helix8_constants(n, k, attr):
    layers, family, helix8, constants = _modules()
    codes, scale = _checkpoint_fp8(n, k, seed=n * 7 + k)
    built = {}
    for name, module in (("family", family), ("helix8", helix8)):
        parent = nn.Module()
        source = layers.Fp8BlockScaleLinear(k, n, device="cuda")
        source.load(codes.clone(), scale.clone())
        setattr(parent, attr, source)
        if name == "family":
            module._swap_linear_to_fp8_weight_read(
                parent, attr, (layers.Fp8BlockScaleLinear,), constants=constants
            )
        else:
            module._swap_linear_to_fp8_weight_read(parent, attr, (layers.Fp8BlockScaleLinear,))
        proj = getattr(parent, attr)
        proj.post_load_weights()
        built[name] = proj
    ref, got = built["family"], built["helix8"]
    assert ref.fp8w_lowm_gemm and ref._low_m_max == 0 and ref._low_m_weight_scale is None
    assert torch.equal(got.weight.view(torch.uint8), ref.weight.view(torch.uint8))
    assert torch.equal(got.weight_scale, ref.weight_scale)
    assert (got.lowm_weight_scale is None) == (ref.lowm_weight_scale is None)
    if ref.lowm_weight_scale is not None:
        assert torch.equal(got.lowm_weight_scale, ref.lowm_weight_scale)
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    for m in ROWS:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        assert got._runs_low_m_kernel(m) == ref._runs_low_m_kernel(m), m
        assert got.mid_operand_tactic(m) == ref.mid_operand_tactic(m), m
        assert torch.equal(got(x), ref(x)), m
