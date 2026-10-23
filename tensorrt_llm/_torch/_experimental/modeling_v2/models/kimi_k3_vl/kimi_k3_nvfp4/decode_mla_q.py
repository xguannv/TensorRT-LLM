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

Two bitwise switches act on these launches (forward time):

* ``TRTLLM_K3LL_MLA_QNORM_PDL``: the q_a norm is ``modules/kimi_k3_mla/q_norm_pdl.py``'s copy of FlashInfer's
  ``RMSNormKernel`` that releases its dependent (q_b) when it starts and reads its weight before its PDL wait, so q_b
  streams its weights while q_a and the norm run (`q_a_norm`). Where that copy is not the module's kernel's bitwise
  twin (another norm path, PDL off, the CUDA JIT FlashInfer norm) the module's norm runs.
* ``TRTLLM_K3LL_MLA_PREWAIT_L2``: ``K3MlaQAbsorb`` with ``prewait`` (every W_UK tile of the CTA staged or
  L2-prefetched before its PDL wait, dependents released after it).
* ``TRTLLM_K3LL_MLA_QNORM_MXFP8`` (with ``TRTLLM_K3LL_MLA_QNORM_PDL``): that released norm also writes the MXFP8 form
  of its output (``q_norm_pdl.rmsnorm_mxfp8_release_at_launch``) and Q-B runs ``k3_fp8w_lowm_mxfp8`` on it
  (``fp8w_lowm.forward_mxfp8``) instead of quantizing its activation inside ``k3_fp8w_lowm_gemm`` after its wait
  (`q_norm_and_q_b`). Where Q-B's MXFP8 GEMM does not take the rows (or is not compiled under a capture) Q-B reads
  the BF16 norm as before.

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
    q: torch.Tensor,
    w_kc: torch.Tensor,
    out: torch.Tensor,
    nope: int,
    rope: int,
    prewait: bool = False,
) -> str:
    """``out[t, h] = [fp8(q[t, h, :nope] @ w_kc[h].T) | fp8(q[t, h, nope:])]`` at unit scale; returns the kernel.
    ``prewait``: ``K3MlaQAbsorb``'s weight-before-wait variant (same bytes; ``TRTLLM_K3LL_MLA_PREWAIT_L2``)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb

    if k3_mla_q_absorb.can_run(q, w_kc, out, nope, rope):
        k3_mla_q_absorb.k3_mla_q_absorb(q, w_kc, out, nope, rope, prewait=prewait)
        return "K3MlaQAbsorb+prewait" if prewait else "K3MlaQAbsorb"
    from tensorrt_llm._torch.modules import mla_fp8_q_absorb

    mla_fp8_q_absorb.mla_fp8_q_absorb_nope(q, w_kc, out, nope, rope)
    return "triton"


def _release_norm_applies(q_a: torch.Tensor, norm) -> bool:
    """Whether ``norm(q_a)`` runs FlashInfer's CuTe DSL ``RMSNormKernel`` with PDL, whose bitwise twin with the early
    trigger (``q_norm_pdl.rmsnorm_release_at_launch``) can then replace it (``RMSNorm.forward`` without a residual)."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    return bool(
        isinstance(norm, RMSNorm)
        and IS_FLASHINFER_AVAILABLE
        and not norm.use_gemma
        and not norm.use_cuda_tile
        and not norm.return_hp_output
        and not (norm.is_nvfp4 and norm.nvfp4_scale is not None)
        and get_env_enable_pdl()
        and torch.cuda.get_device_capability(q_a.device)[0] >= 9
        and q_norm_pdl.can_use(q_a, norm.weight)
    )


def q_a_norm(q_a: torch.Tensor, q_a_layernorm, release: bool) -> Tuple[torch.Tensor, bool]:
    """``q_a_layernorm(q_a)``; with ``release`` (``TRTLLM_K3LL_MLA_QNORM_PDL``) through the PDL-release twin of its
    kernel where that is bitwise the same. Returns the norm and whether the twin ran."""
    if release and _release_norm_applies(q_a, q_a_layernorm):
        from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

        out = q_norm_pdl.rmsnorm_release_at_launch(
            q_a, q_a_layernorm.weight, float(q_a_layernorm.variance_epsilon)
        )
        return out, True
    return q_a_layernorm(q_a), False


def q_norm_and_q_b(
    q_a: torch.Tensor, q_a_layernorm, q_b_proj, release: bool, mxfp8: bool
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """``q_b_proj(q_a_layernorm(q_a))`` (bitwise), with ``release`` (``TRTLLM_K3LL_MLA_QNORM_PDL``) through `q_a_norm`
    and, with ``mxfp8`` too (``TRTLLM_K3LL_MLA_QNORM_MXFP8``), the norm's MXFP8 form handed to Q-B where Q-B's MXFP8
    GEMM takes these rows. Returns ``(q_norm, q, suffix)``: ``suffix`` ``"+qnorm_mxfp8"`` when Q-B ran on the MXFP8
    form, ``"+qnorm_pdl"`` when the released norm ran, else ``""``."""
    from . import fp8w_lowm as _fp8w_lowm

    if (
        release
        and mxfp8
        and _fp8w_lowm.takes_mxfp8_input(q_b_proj, q_a.shape[0])
        and _release_norm_applies(q_a, q_a_layernorm)
    ):
        from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

        if q_norm_pdl.can_use_mxfp8(q_a, q_a_layernorm.weight):
            xq, xs = _fp8w_lowm.new_mxfp8_input(q_a.shape[0], q_a.shape[1], q_a.device)
            q_norm = q_norm_pdl.rmsnorm_mxfp8_release_at_launch(
                q_a, q_a_layernorm.weight, float(q_a_layernorm.variance_epsilon), xq, xs
            )
            before = _fp8w_lowm.MX_STATS["mxfp8"]
            q = _fp8w_lowm.forward_mxfp8(q_b_proj, q_norm, xq, xs)
            took = _fp8w_lowm.MX_STATS["mxfp8"] > before
            return q_norm, q, "+qnorm_mxfp8" if took else "+qnorm_pdl"
    q_norm, released = q_a_norm(q_a, q_a_layernorm, release)
    return q_norm, q_b_proj(q_norm), "+qnorm_pdl" if released else ""


def all_head_queries(
    q_a: torch.Tensor,
    q_a_layernorm: Callable[[torch.Tensor], torch.Tensor],
    q_b_proj: Callable[[torch.Tensor], torch.Tensor],
    w_kc: torch.Tensor,
    nope: int,
    rope: int,
    release_norm: bool = False,
    absorb_prewait: bool = False,
    qnorm_mxfp8: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
    """Every head's FP8 query from the q_a columns ``q_a`` (``[tokens, q_lora_rank]`` bf16, any row stride).

    ``release_norm`` / ``absorb_prewait`` / ``qnorm_mxfp8``: ``TRTLLM_K3LL_MLA_QNORM_PDL`` /
    ``TRTLLM_K3LL_MLA_PREWAIT_L2`` / ``TRTLLM_K3LL_MLA_QNORM_MXFP8`` (bitwise). Returns ``(quant_q, q, kernel,
    q_norm)``: ``quant_q`` ``[tokens, heads, kv_lora_rank + rope]`` uint8 (FP8 bytes, what the split read takes),
    ``q`` the BF16 Q-B output viewed ``[tokens, heads, nope + rope]``, the kernels' names (absorption, then
    ``+qnorm_pdl`` when the released norm ran or ``+qnorm_mxfp8`` when Q-B also ran on its MXFP8 form) and Q-B's
    input (the normalized q_a)."""
    heads, lora = w_kc.shape[0], w_kc.shape[1]
    q_norm, q, suffix = q_norm_and_q_b(q_a, q_a_layernorm, q_b_proj, release_norm, qnorm_mxfp8)
    q = q.view(q_a.shape[0], heads, nope + rope)
    quant_q = torch.empty((q_a.shape[0], heads, lora + rope), dtype=torch.uint8, device=q_a.device)
    kernel = absorb_queries(q, w_kc, quant_q, nope, rope, prewait=absorb_prewait) + suffix
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
