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


_KV_STREAMS: Dict[int, torch.cuda.Stream] = {}

# TRTLLM_K3LL_MLA_KV_SIDE applies to decode steps of 1 to KV_SIDE_MAX_REQUESTS requests; from 5 requests the latent row
# and the KV append run in order on the main stream (the side stream measured slower at 8 requests).
KV_SIDE_MAX_REQUESTS = 4


# TRTLLM_K3LL_MLA_QNORM_PDL applies to decode steps of 1 to QNORM_PDL_MAX_REQUESTS requests; above it the q_a norm is
# the module's own kernel, as with the switch at 0.
QNORM_PDL_MAX_REQUESTS = 4


def qnorm_pdl_applies(requests: int) -> bool:
    """``TRTLLM_K3LL_MLA_QNORM_PDL``: whether a decode step of ``requests`` requests releases q_b from the q_a norm at
    launch (1 to `QNORM_PDL_MAX_REQUESTS`)."""
    return 1 <= requests <= QNORM_PDL_MAX_REQUESTS


def kv_side_applies(requests: int) -> bool:
    """``TRTLLM_K3LL_MLA_KV_SIDE``: whether a decode step of ``requests`` requests puts the latent row and the KV append
    on `kv_side_stream` (1 to `KV_SIDE_MAX_REQUESTS`)."""
    return 1 <= requests <= KV_SIDE_MAX_REQUESTS


def kv_side_stream(device: torch.device) -> torch.cuda.Stream:
    """``TRTLLM_K3LL_MLA_KV_SIDE``: one stream per device for the latent row and the KV append, shared by every MLA
    layer and used by nothing else (not the gate's `side_stream`, whose projection would queue behind the append)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _KV_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _KV_STREAMS[index] = stream
    return stream


class KvSide:
    """``TRTLLM_K3LL_MLA_KV_SIDE``: the latent row and KV append of one forward on `kv_side_stream`.

    `mark` records the main stream's position (after the fused q_a / kv_a); `run` enqueues ``fn`` on the side stream
    behind that position (so the query kernels the main stream enqueues in between run beside it) and returns True;
    `join` makes the main stream wait for it. Without a side stream (``multi_stream`` False) `run` calls ``fn`` in
    place and returns False."""

    def __init__(self) -> None:
        self._fork_event = torch.cuda.Event()
        self._done_event = torch.cuda.Event()
        self._side = False
        self._marked = False

    def mark(self, multi_stream: bool) -> None:
        self._side = multi_stream
        if multi_stream:
            self._fork_event.record()
        self._marked = True

    def run(self, fn: Callable[[], None], device: torch.device) -> bool:
        """Whether ``fn`` ran on `kv_side_stream` (False: in place on the current stream)."""
        if not self._marked:
            raise RuntimeError("KvSide.run without mark")
        if not self._side:
            fn()
            return False
        with torch.cuda.stream(kv_side_stream(device)):
            self._fork_event.wait()
            fn()
            self._done_event.record()
        return True

    def join(self) -> None:
        if not self._marked:
            raise RuntimeError("KvSide.join without mark")
        if self._side:
            self._done_event.wait()
        self._marked = self._side = False


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


def all_head_queries(
    q_a: torch.Tensor,
    q_a_layernorm: Callable[[torch.Tensor], torch.Tensor],
    q_b_proj: Callable[[torch.Tensor], torch.Tensor],
    w_kc: torch.Tensor,
    nope: int,
    rope: int,
    release_norm: bool = False,
    absorb_prewait: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
    """Every head's FP8 query from the q_a columns ``q_a`` (``[tokens, q_lora_rank]`` bf16, any row stride).

    ``release_norm`` / ``absorb_prewait``: ``TRTLLM_K3LL_MLA_QNORM_PDL`` / ``TRTLLM_K3LL_MLA_PREWAIT_L2`` (bitwise).
    Returns ``(quant_q, q, kernel, q_norm)``: ``quant_q`` ``[tokens, heads, kv_lora_rank + rope]`` uint8 (FP8 bytes,
    what the split read takes), ``q`` the BF16 Q-B output viewed ``[tokens, heads, nope + rope]``, the kernels'
    names (absorption, and ``+qnorm_pdl`` when the released norm ran) and Q-B's input (the normalized q_a)."""
    heads, lora = w_kc.shape[0], w_kc.shape[1]
    q_norm, released = q_a_norm(q_a, q_a_layernorm, release_norm)
    q = q_b_proj(q_norm).view(q_a.shape[0], heads, nope + rope)
    quant_q = torch.empty((q_a.shape[0], heads, lora + rope), dtype=torch.uint8, device=q_a.device)
    kernel = absorb_queries(q, w_kc, quant_q, nope, rope, prewait=absorb_prewait)
    if released:
        kernel += "+qnorm_pdl"
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
