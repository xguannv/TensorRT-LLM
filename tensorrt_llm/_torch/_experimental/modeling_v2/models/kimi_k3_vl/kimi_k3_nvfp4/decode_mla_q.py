# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query side of the Kimi K3 NVFP4 MLA decode: every head's FP8 query on every Helix rank, and the gate side stream.

``TRTLLM_K3LL_HELIX_Q_ABSORB`` (flags.py). Under Helix every rank holds every head's Q-B (``q_b_proj``, an FP8
weight-read projection that ``fp8w_lowm`` routes to ``k3_fp8w_lowm_gemm`` at decode size) and K-B. `all_head_queries`
computes the FP8 query of all heads from the fused q_a / kv_a output, in three launches:

* ``q_a_layernorm`` on the q_a columns (the module's RMSNorm, BF16 out);
* ``q_b_proj`` on that, BF16 ``[tokens, heads * (nope + rope)]``;
* the absorption ``q_nope @ W_UK`` with W_UK in BF16 (`absorb_weight`: the FP8 K-B codes times their block scales),
  written as FP8 E4M3 together with the copied ``q_pe`` (NoPE), at unit scale: ``K3MlaQAbsorb``
  (``cute_dsl_kernels/blackwell/k3_mla_q_absorb.py``, SM100 / SM103) or the Triton kernel of
  ``modules/mla_fp8_q_absorb.py``, which give the same bytes.

The split read then attends with these queries directly instead of all-gathering every rank's head group. The
numerics differ from the head-group path (``attention.mla_project_qkv``): Q-B's activation is quantized to MXFP8 per 32
columns inside the low-M kernel instead of FP8 per 128 columns, and the absorption multiplies the BF16 ``q_nope`` with
BF16 W_UK in FP32 and rounds once to FP8, instead of an FP8 ``q_nope`` (one scale per row) with the FP8 K-B and a BF16
result that the cache kernel rounds to FP8.

``TRTLLM_K3LL_MLA_GATE_SIDE``: `GateSide` runs the output gate projection on a per-device side stream, forked where
the attention starts and joined before the V-B / gate kernel, which reads it in place.

The kernels come from Dongxu Yang's ``user/dongxuy/k3_lowlat`` (``TRTLLM_K3_MLA_FP8_Q_ABSORB(_CUTE)``,
``TRTLLM_K3_MLA_GATE_SIDE`` in ``modules/kimi_k3_mla/kimi_k3_mla_attention.py``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

_SIDE_STREAMS: Dict[int, torch.cuda.Stream] = {}


def side_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device, shared by every MLA layer (one layer's side work ends before the next forks)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SIDE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _SIDE_STREAMS[index] = stream
    return stream


def absorb_weight(k_b: torch.Tensor, k_b_scale: Optional[torch.Tensor]) -> torch.Tensor:
    """BF16 W_UK ``[heads, kv_lora_rank, nope]`` from the module's K-B (``k_b_proj_trans``, FP8 codes with
    ``k_b_proj_trans_scale`` ``[heads, kv_lora_rank / 128, nope / 128]``, or already BF16): the codes times their
    block scales, rounded once to BF16 (the KV-B loader's ``k_b_proj_trans_dequant``)."""
    if k_b.dtype == torch.bfloat16:
        return k_b.detach().contiguous()
    if k_b_scale is None:
        raise ValueError(f"K-B {k_b.dtype} without block scales")
    expanded = k_b_scale.detach().float().repeat_interleave(128, -2).repeat_interleave(128, -1)
    expanded = expanded[:, : k_b.shape[1], : k_b.shape[2]]
    return (k_b.detach().float() * expanded).to(torch.bfloat16).contiguous()


def absorb_queries(
    q: torch.Tensor, w_kc: torch.Tensor, out: torch.Tensor, nope: int, rope: int
) -> str:
    """``out[t, h] = [fp8(q[t, h, :nope] @ w_kc[h].T) | fp8(q[t, h, nope:])]`` at unit scale; returns the kernel."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb

    if k3_mla_q_absorb.can_run(q, w_kc, out, nope, rope):
        k3_mla_q_absorb.k3_mla_q_absorb(q, w_kc, out, nope, rope)
        return "K3MlaQAbsorb"
    from tensorrt_llm._torch.modules import mla_fp8_q_absorb

    mla_fp8_q_absorb.mla_fp8_q_absorb_nope(q, w_kc, out, nope, rope)
    return "triton"


def all_head_queries(
    q_a: torch.Tensor,
    q_a_layernorm: Callable[[torch.Tensor], torch.Tensor],
    q_b_proj: Callable[[torch.Tensor], torch.Tensor],
    w_kc: torch.Tensor,
    nope: int,
    rope: int,
) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
    """Every head's FP8 query from the q_a columns ``q_a`` (``[tokens, q_lora_rank]`` bf16, any row stride).

    Returns ``(quant_q, q, kernel, q_norm)``: ``quant_q`` ``[tokens, heads, kv_lora_rank + rope]`` uint8 (FP8 bytes,
    what the split read takes), ``q`` the BF16 Q-B output viewed ``[tokens, heads, nope + rope]``, the absorption
    kernel's name and Q-B's input (the normalized q_a)."""
    heads, lora = w_kc.shape[0], w_kc.shape[1]
    q_norm = q_a_layernorm(q_a)
    q = q_b_proj(q_norm).view(q_a.shape[0], heads, nope + rope)
    quant_q = torch.empty((q_a.shape[0], heads, lora + rope), dtype=torch.uint8, device=q_a.device)
    kernel = absorb_queries(q, w_kc, quant_q, nope, rope)
    return quant_q, q, kernel, q_norm


class GateSide:
    """``TRTLLM_K3LL_MLA_GATE_SIDE``: the gate projection of one forward on the side stream.

    `fork` records the main stream's position and enqueues ``g_proj(x)`` on the side stream behind it; `join` makes the
    main stream wait for it and returns its output. Without a side stream (``multi_stream`` False) `fork` runs the
    projection in place."""

    def __init__(self) -> None:
        self._fork_event = torch.cuda.Event()
        self._ready_event = torch.cuda.Event()
        self._gate: Optional[torch.Tensor] = None
        self._side = False

    def fork(
        self, g_proj: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, multi_stream: bool
    ) -> None:
        if not multi_stream:
            self._gate, self._side = g_proj(x), False
            return
        self._fork_event.record()
        with torch.cuda.stream(side_stream(x.device)):
            self._fork_event.wait()
            self._gate = g_proj(x)
            self._ready_event.record()
        self._side = True

    def join(self) -> torch.Tensor:
        gate, self._gate = self._gate, None
        if gate is None:
            raise RuntimeError("GateSide.join without fork")
        # The gate's block belongs to the side stream's pool. The side stream reuses it only after its next fork, which
        # the main stream records after this forward's consumer of the gate.
        if self._side:
            self._ready_event.wait()
        return gate
