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
"""Kimi K3 routed experts for decode on NVFP4 experts: ``trtllm::k3_moe_nvfp4``, the persistent CuTe DSL kernel
``k3_moe_nvfp4_kernel.py`` (the NVFP4 port of ``k3_moe``), over the TRTLLM-Gen NVFP4 MoE buffers read in place.

Inputs: the latent quantized to NVFP4 with the layer's ``fc31_input_scale`` (``x_fp4`` packed E2M1 ``[M, 3584 / 2]``
uint8, ``x_sf`` UE4M3 ``[M, 224]`` in linear layout: ``trtllm::fp4_quantize`` with ``sfUseUE8M0=False`` and
``isSfSwizzledLayout=False``), the routing (``topk_ids`` int32, ``topk_weights`` bf16, ``[M, 16]``), and per layer
the experts' ``fc31_alpha`` and ``fc2_alpha`` (fp32 ``[num_local]``) and ``fc2_input_scale`` (fp32, one element): the
scalars the TRTLLM-Gen runner takes as ``output1_scale_gate_scalar``, ``output2_scale_scalar`` and (for SiTU, whose
``fc31_scale_c`` is ``fc2_input_scale`` times the up alpha, equal to ``fc31_alpha`` on Kimi K3) the quantization scale
of the intermediate. Per token ``t``::

    y[t] = bf16( sum over t's local experts e of  w[t, e] * fc2_alpha[e] * (q4(fc2_input_scale *
                 SiTU(fc31_alpha[e] * x @ W_gate[e]^T, fc31_alpha[e] * x @ W_up[e]^T)) @ W_down[e]^T) )

with ``x`` the dequantized NVFP4 latent without its global scale, ``q4`` NVFP4 per 16 intermediate columns (UE4M3 scale
of amax / 6, round to nearest), and the weights' E2M1 codes times their UE4M3 scales. The state, call-order and
re-arming rules are those of ``trtllm::k3_moe`` (``op.py``); the head-flags, push and fold builds are not built.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

import torch

from .op import (
    _OPTIONAL_ARGS,
    _TOKEN_SLOTS,
    HIDDEN_SIZE,
    MAX_TOKENS,
    NUM_EXPERTS,
    TOP_K,
    WIDE_MAX_TOKENS,
    _compile_key,
    _part_rows,
    _view,
)

_SF_VEC = 16
_KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_nvfp4_kernel.py")
_modules: Dict[tuple, object] = {}
_compiled: Dict[tuple, object] = {}

# (alignment, leading dim) of every tensor argument, in the kernel's order: the 18 of k3_moe, its 11 optional ones
# (stand-ins here), then fc31_alpha, fc2_alpha and fc2_input_scale.
_ALIGNS = [16, 16, 16, 16, 16, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 4, 4, 4]
_ALIGNS += [16] * _OPTIONAL_ARGS + [4, 4, 4]
_LEADING = [0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
_LEADING += [0] * _OPTIONAL_ARGS + [0, 0, 0]


def _kernel_module(config: dict):
    """One kernel module per configuration: shapes are trace-time constants."""
    key = tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        tag = "_".join(f"{k}{v}" for k, v in key)
        name = f"{__name__}_kernel_{tag}"
        spec = importlib.util.spec_from_file_location(name, _KERNEL_PATH)
        mod = importlib.util.module_from_spec(spec)
        mod.K3_CONFIG = dict(config)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _modules[key] = mod
    return mod


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def is_supported(w3_w1_weight: torch.Tensor, w3_w1_weight_scale: torch.Tensor, w2_weight: torch.Tensor,
                 w2_weight_scale: torch.Tensor, local_num_experts: int) -> Tuple[bool, str]:  # fmt: skip
    """Whether these TRTLLM-Gen NVFP4 MoE buffers fit the kernel. Only metadata is read (the buffers may be on the
    meta device)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False, "needs an sm_10x GPU (tcgen05)"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    if w3_w1_weight.dim() != 3:
        return False, "w3_w1_weight is not [E, 2I, H/2]"
    _, two_i, _ = w3_w1_weight.shape
    i_tp = two_i // 2
    expected = {
        "w3_w1_weight": (
            w3_w1_weight,
            (local_num_experts, two_i, HIDDEN_SIZE // 2),
            (torch.uint8,),
        ),
        "w3_w1_weight_scale": (
            w3_w1_weight_scale,
            (local_num_experts, two_i, HIDDEN_SIZE // _SF_VEC),
            (torch.uint8, torch.float8_e4m3fn),
        ),
        "w2_weight": (w2_weight, (local_num_experts, HIDDEN_SIZE, i_tp // 2), (torch.uint8,)),
        "w2_weight_scale": (
            w2_weight_scale,
            (local_num_experts, HIDDEN_SIZE, i_tp // _SF_VEC),
            (torch.uint8, torch.float8_e4m3fn),
        ),
    }
    for name, (t, shape, dtypes) in expected.items():
        if t.dtype not in dtypes or tuple(t.shape) != shape or not t.is_contiguous():
            return (
                False,
                f"{name} is {t.dtype} {tuple(t.shape)}, expected contiguous {dtypes} {shape}",
            )
    if i_tp % 128 != 0:
        return False, f"intermediate size {i_tp} is not a multiple of 128"
    if local_num_experts > NUM_EXPERTS:
        return False, f"{local_num_experts} local experts"
    return True, ""


def _config(i_tp: int, num_ctas: int, num_local: int, m_max: int, use_pdl: bool) -> dict:
    return {
        "i_tp": i_tp,
        "num_ctas": num_ctas,
        "num_local": num_local,
        "m_max": m_max,
        "pdl": int(use_pdl),
        "head_flags": 0,
        "lat_slab": 0,
    }


class _K3MoeNvfp4Scratch:
    """The scratch of one build on one device, shared by its layers: the FC1 -> FC2 intermediate slab (armed between
    calls: 0x88 value bytes, UE4M3 NaN in bytes 0..3 of each 16-byte scale group) and the FC2 partial rows."""

    def __init__(self, device: torch.device, i_tp: int, num_local: int, m_max: int, use_pdl: bool,
                 num_ctas: Optional[int]):  # fmt: skip
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{type(self).__name__} allocates its scratch: build it outside CUDA-graph capture"
            )
        if num_ctas is None:
            num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.config = _config(i_tp, num_ctas, num_local, m_max, use_pdl)
        self.mod = mod = _kernel_module(self.config)
        self.device = device
        self.i_tp = i_tp
        self.num_local = num_local
        self.m_max = m_max
        self.num_ctas = num_ctas
        self.use_pdl = bool(use_pdl)
        g_cap = mod.G_CAP
        kw = dict(device=device)
        self.c = torch.full(
            (g_cap, _TOKEN_SLOTS, i_tp // 2), mod.C_SENTINEL_I8, dtype=torch.int8, **kw
        )
        self.cs = torch.zeros(g_cap, _TOKEN_SLOTS, mod.SF_STRIDE0, dtype=torch.int8, **kw)
        self.cs.view(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)[..., :4] = (
            mod.SF_SENTINEL_I8
        )
        self.part = torch.zeros(_part_rows(mod), HIDDEN_SIZE, dtype=torch.float32, **kw)

    def layer(self, w3_w1_weight: torch.Tensor, w3_w1_weight_scale: torch.Tensor, w2_weight: torch.Tensor,
              w2_weight_scale: torch.Tensor, fc31_alpha: torch.Tensor, fc2_alpha: torch.Tensor,
              fc2_input_scale: torch.Tensor) -> "K3MoeNvfp4Layer":  # fmt: skip
        """A layer's handle: its experts' TRTLLM-Gen NVFP4 buffers and global scales (read in place), its counters."""
        return K3MoeNvfp4Layer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, fc31_alpha,
                               fc2_alpha, fc2_input_scale)  # fmt: skip

    @property
    def compiled(self) -> bool:
        return _compile_key(self.device, self.config) in _compiled


class K3MoeNvfp4State(_K3MoeNvfp4Scratch):
    """``k3_moe_nvfp4`` for 1..8 decode tokens on one device (see ``op.K3MoeState`` for the rules)."""

    def __init__(self, device: torch.device, i_tp: int, num_local: int, use_pdl: Optional[bool] = None,
                 num_ctas: Optional[int] = None):  # fmt: skip
        if use_pdl is None:
            use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
        super().__init__(device, i_tp, num_local, MAX_TOKENS, use_pdl, num_ctas)


class K3MoeNvfp4WideState(_K3MoeNvfp4Scratch):
    """``k3_moe_nvfp4`` for 1..64 tokens on one device (see ``op.K3MoeWideState``)."""

    def __init__(self, device: torch.device, i_tp: int, num_local: int, use_pdl: bool = True):
        super().__init__(device, i_tp, num_local, WIDE_MAX_TOKENS, use_pdl, None)


class K3MoeNvfp4Layer:
    """One MoE layer on a state: its experts' buffers and global scales, and its counters (zero between calls)."""

    def __init__(self, state: _K3MoeNvfp4Scratch, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale,
                 fc31_alpha, fc2_alpha, fc2_input_scale):  # fmt: skip
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeNvfp4Layer allocates its counters: build it outside CUDA-graph capture"
            )
        ok, why = is_supported(
            w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, state.num_local
        )
        if not ok or w3_w1_weight.shape[1] != 2 * state.i_tp:
            raise ValueError(
                f"k3_moe_nvfp4 layer: {why or 'intermediate size differs from the state'}"
            )
        for name, t, n in (("fc31_alpha", fc31_alpha, state.num_local), ("fc2_alpha", fc2_alpha, state.num_local),
                           ("fc2_input_scale", fc2_input_scale, 1)):  # fmt: skip
            if t.dtype != torch.float32 or t.numel() != n or not t.is_contiguous():
                raise ValueError(
                    f"k3_moe_nvfp4 layer: {name} must be contiguous fp32 with {n} elements"
                )
        self.state = state
        self.weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)
        self.scales = (fc31_alpha.view(-1), fc2_alpha.view(-1), fc2_input_scale.view(-1))
        self.counters = torch.zeros(state.mod.NUM_STATE, dtype=torch.int32, device=state.device)

    def __call__(self, x_fp4: torch.Tensor, x_sf: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                 local_expert_offset: int, out: Optional[torch.Tensor] = None) -> torch.Tensor:  # fmt: skip
        st = self.state
        y = torch.ops.trtllm.k3_moe_nvfp4(
            x_fp4, x_sf, topk_ids, topk_weights, *self.weights, *self.scales, st.c, st.cs, st.part, self.counters,
            local_expert_offset, st.num_local, st.num_ctas, st.m_max, st.use_pdl, out,
        )  # fmt: skip
        return y if out is None else out[: topk_ids.shape[0]]


def _compile(mod, args, scalars):
    import cutlass.cute as cute

    assert len(args) == len(_ALIGNS)
    signature = [_view(t, a, d) for t, a, d in zip(args, _ALIGNS, _LEADING)]
    return cute.compile(
        mod.k3_moe,
        *signature,
        *scalars,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::k3_moe_nvfp4", mutates_args=("c", "cs", "part", "counters", "out")
)
def k3_moe_nvfp4(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    fc31_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    c: torch.Tensor,
    cs: torch.Tensor,
    part: torch.Tensor,
    counters: torch.Tensor,
    local_expert_offset: int,
    num_local: int,
    num_ctas: int,
    m_max: int,
    use_pdl: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's routed partial ``[M, 3584]`` bf16 over NVFP4 experts (module docstring). ``c``, ``cs``, ``part``:
    the scratch of a :class:`K3MoeNvfp4State` (``m_max`` 8) or :class:`K3MoeNvfp4WideState` (``m_max`` 64);
    ``counters``: the layer's. ``out``: as ``trtllm::k3_moe``'s."""
    num_tokens = topk_ids.shape[0]
    if m_max not in (MAX_TOKENS, WIDE_MAX_TOKENS):
        raise ValueError(f"k3_moe_nvfp4: m_max is {MAX_TOKENS} or {WIDE_MAX_TOKENS}, got {m_max}")
    if not 0 < num_tokens <= m_max:
        raise ValueError(f"k3_moe_nvfp4: M must be in [1, {m_max}], got {num_tokens}")
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape) != (num_tokens, TOP_K)
        or topk_weights.dtype != torch.bfloat16
        or tuple(topk_weights.shape) != (num_tokens, TOP_K)
        or x_fp4.dtype != torch.uint8
        or tuple(x_fp4.shape) != (num_tokens, HIDDEN_SIZE // 2)
        or x_sf.numel() != num_tokens * (HIDDEN_SIZE // _SF_VEC)
        or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        or not (x_fp4.is_contiguous() and x_sf.is_contiguous())
    ):
        raise ValueError(
            "k3_moe_nvfp4: expects the NVFP4 latent (linear scales) and the routing of M tokens"
        )
    ok, why = is_supported(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, num_local)
    if not ok:
        raise ValueError(f"k3_moe_nvfp4: {why}")
    for name, t, n in (("fc31_alpha", fc31_alpha, num_local), ("fc2_alpha", fc2_alpha, num_local),
                       ("fc2_input_scale", fc2_input_scale, 1)):  # fmt: skip
        if t.dtype != torch.float32 or t.numel() != n or not t.is_contiguous():
            raise ValueError(f"k3_moe_nvfp4: {name} must be contiguous fp32 with {n} elements")
    e, two_i, _ = w3_w1_weight.shape
    i_tp = two_i // 2
    config = _config(i_tp, num_ctas, num_local, m_max, use_pdl)
    mod = _kernel_module(config)
    g_cap = mod.G_CAP
    if (
        c.dtype != torch.int8
        or tuple(c.shape) != (g_cap, _TOKEN_SLOTS, i_tp // 2)
        or cs.dtype != torch.int8
        or tuple(cs.shape) != (g_cap, _TOKEN_SLOTS, mod.SF_STRIDE0)
        or part.dtype != torch.float32
        or tuple(part.shape) != (_part_rows(mod), HIDDEN_SIZE)
        or counters.dtype != torch.int32
        or tuple(counters.shape) != (mod.NUM_STATE,)
        or not all(t.is_contiguous() for t in (c, cs, part, counters))
    ):
        raise ValueError(
            "k3_moe_nvfp4: c / cs / part / counters are not this build's scratch and layer counters"
        )
    if out is None:
        y = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=x_fp4.device)
    else:
        if (
            out.dtype != torch.bfloat16
            or out.dim() != 2
            or out.shape[0] < num_tokens
            or out.shape[1] != HIDDEN_SIZE
            or not out.is_contiguous()
        ):
            raise ValueError("k3_moe_nvfp4: out must be contiguous bf16 [>= M, 3584]")
        y = out[:num_tokens]
    sfb2 = (
        cs.view(torch.uint8)
        .reshape(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)
        .permute(3, 2, 1, 0)
    )
    unused = counters
    args = (
        _u8(w3_w1_weight).permute(2, 1, 0),
        x_fp4.permute(1, 0),
        _u8(w3_w1_weight_scale).view(e, two_i // 128, HIDDEN_SIZE // 64, 512).permute(3, 2, 1, 0),
        _u8(x_sf).view(num_tokens, HIDDEN_SIZE // _SF_VEC),
        c, cs, c.view(-1).view(torch.int32), cs.view(-1).view(torch.int32),
        _u8(w2_weight).permute(2, 1, 0),
        c.view(torch.uint8).permute(2, 1, 0),
        _u8(w2_weight_scale).view(e, HIDDEN_SIZE // 128, i_tp // 64, 512).permute(3, 2, 1, 0),
        sfb2, y, y.view(torch.int32), part, topk_ids, topk_weights, counters,
        *((unused,) * _OPTIONAL_ARGS),
        fc31_alpha.view(-1), fc2_alpha.view(-1), fc2_input_scale.view(-1),
    )  # fmt: skip
    scalars = (num_tokens, local_expert_offset, num_local, 0, 1.0, 0, 0)
    key = _compile_key(x_fp4.device, config)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_moe_nvfp4 compiles on its first call for each build: call it once before CUDA-graph capture"
            )
        fn = _compiled[key] = _compile(mod, args, scalars)
    fn(*args, *scalars, torch.cuda.current_stream(x_fp4.device).cuda_stream)
    if out is not None:
        return y.new_empty((0, HIDDEN_SIZE))
    return y


@k3_moe_nvfp4.register_fake
def _(x_fp4, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, fc31_alpha,
      fc2_alpha, fc2_input_scale, c, cs, part, counters, local_expert_offset, num_local, num_ctas, m_max, use_pdl,
      out=None):  # fmt: skip
    rows = 0 if out is not None else topk_ids.shape[0]
    return x_fp4.new_empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16)
