# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-call primitives of the Kimi K3 Rubin dep16 model: K3-owned weights and direct kernel calls.

Each container is an ``nn.Module`` that registers the parameter names the checkpoint and the
kernels read (``weight``, ``weight_scale``, ...), loads its rank's shard with an explicit
``load`` (and ``prepare`` where the kernel wants another layout), and runs one fixed kernel chain:

* ``linear``: BF16 ``F.linear``; FP8 128 x 128 block scales on the low-M GEMM, the SM107 MXFP8
  GEMM (``mxfp8_gemm``) or the SM100 / SM103 swap-AB GEMM.
* ``norm``: FlashInfer RMSNorm (plain and fused residual add) on a K3 weight.
* ``mlp``: gate-up GEMM, SiTU / SiLU-and-mul, down GEMM.
* ``embedding``: the whole-table lookup.
* ``conv``: the short-convolution weight the KDA kernels read.
* ``streams``: two branches on the current and an auxiliary stream.

Under attention DP every rank owns its own tokens: the dense MLP, the shared experts and the
embedding are replicated, and nothing here reduces across ranks.

Projections on K3-owned weights: tensor-parallel shards, BF16 GEMM and FP8 block-scale GEMM.

Weight containers register the parameters under the names the checkpoint loader looks up and keep
this rank's shard. Their forward is one explicit kernel chain.

**Sharding** (:func:`shard_range` / :func:`load_shard`): rank ``r`` of ``tp`` takes a contiguous
range of the split dim (output features for ``COLUMN``, input features for ``ROW``), the first
``width % tp`` ranks one unit more; for FP8 block-scale weights the unit is a 128-wide block and
the block-scale grid follows the weight's range (``scale_span=128``).

**BF16** (:class:`Bf16Linear`): ``F.linear`` (cuBLASLt).

**FP8 block scales** (:class:`Fp8BlockScaleLinear`): E4M3 ``weight [N, K]`` with one FP32 scale per
128 x 128 block (``weight_scale [ceil(N/128), ceil(K/128)]``). :meth:`Fp8BlockScaleLinear.prepare`
rounds the scales to UE8M0 (``resmooth_to_fp8_e8m0`` rescales the codes to match) and lays them
out for the GEMM of this GPU; :func:`fp8_block_scale_gemm` then runs, on BF16 rows:

* SM107 (CuTe DSL Rubin): ``fp8_quantize_1x128_cutedsl_ue8m0`` (activation to E4M3 with K32
  R128c4 UE8M0 scales) and ``ops.gemm.k3_mxfp8_gemm`` (``ops/gemm/_mxfp8_gemm.py``), the weight
  scale in the K32 R128c4 layout;
* SM100 / SM103: ``fp8_swap_ab_gemm(disable_ue8m0_cast=True)`` (DeepGEMM; activation quantized in
  the kernel), the weight scale packed for DeepGEMM;
* either, where ``low_m_max`` covers the rows: ``trtllm::fp8_block_scale_low_m_gemm`` on the
  DeepGEMM-packed scale (on SM107 a copy built from the K32 R128c4 one).

RMSNorm on a K3-owned weight: the FlashInfer RMSNorm kernels through their catalog entries.

``rms_norm(x, w, eps)`` is the catalog's ``flashinfer_rmsnorm`` (``w * x / rms(x)``, FP32
statistics, the weight applied to the BF16-rounded normalized value); ``fused_add_rms_norm(x, r, w,
eps)`` is the catalog's ``flashinfer_fused_add_rmsnorm``, which writes ``r += x`` and
``x = rms_norm(r)`` in place. Both launch with PDL where ``TRTLLM_ENABLE_PDL`` is set (the op reads
it).

:class:`RMSNormWeight` holds the ``[hidden]`` weight under the parameter name ``weight`` and the
epsilon as ``variance_epsilon`` (also readable as ``eps``), the names the checkpoint loader and the
kernels that take a norm's weight read. The KDA output norm is one too: its kernel is
``rms_norm_gated_token_major`` (``modules/mamba/layernorm_gated.py``), called by the mixer on
``weight`` / ``eps``.

Depthwise causal short-convolution weight of a KDA q / k / v stream.

:class:`ShortConvWeight` registers ``weight`` ``[channels, 1, kernel_size]`` and no bias, the
parameter a depthwise ``nn.Conv1d`` (``groups=channels``) registers, so the checkpoint's
``*_conv1d.weight`` loads by name. The convolution itself is the caller's kernel call on that
weight (``causal_conv1d_fn``, the KDA prefill / decode / verify kernels), with ``activation``
(SiLU) applied inside it.

Two launches on two CUDA streams, joined on the calling stream.

``fork_join(fn0, fn1, fork, join, aux)`` runs ``fn0`` on the calling stream and ``fn1`` on ``aux``:

    fork.record()                      # calling stream
    r0 = fn0()                         # calling stream
    with torch.cuda.stream(aux):
        fork.wait(); r1 = fn1(); join.record()
    join.wait()                        # calling stream, before anything reads r1

Both run on the calling stream, in that order, unless ``do_multi_stream()`` (the engine's
per-thread flag, set while CUDA graphs are captured) holds and ``aux`` is given. Outside graph
capture the CUDA tensors ``fn1`` returns (directly or in nested tuples / lists / dict values) are
recorded on the calling stream, so the caching allocator does not hand their memory to later
``aux`` work while the calling stream still reads them.

Token embedding on a K3-owned table.

:class:`EmbeddingTable` holds ``weight`` ``[V, hidden]``, the whole vocabulary. Its forward is
the masked lookup ``F.embedding`` with ids outside the table (the rejection sampler pads with -1)
embedding to zeros.

Gated MLP on K3-owned BF16 weights of one rank (replicated: TP 1): gate/up GEMM, gated
activation, down GEMM.

    h = x @ gate_up_proj.weight.T            # F.linear, [M, 2 * I] = [gate | up]
    a = act(h)                               # family situ_and_mul or trtllm::silu_and_mul
    y = a @ down_proj.weight.T               # F.linear, [M, H]

Activations:

* ``situ``: the family ``situ_and_mul(h, beta, linear_beta)`` entry, the Triton SiTU kernel
  (``beta * tanh(gate / beta) * sigmoid(gate) * (linear_beta * tanh(up / linear_beta) or up)``,
  FP32 math);
* ``silu``: ``trtllm::silu_and_mul(h)``, the Triton SwiGLU kernel.

:func:`subgroup_mapping` is the mapping of a TP subgroup of ``tp_size`` ranks: consecutive ranks
group together and the remaining factor is folded into PP.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.situ_and_mul import (
    situ_and_mul,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

from ..common.multi_stream import do_multi_stream
from ..common.parallel import TensorParallelMode

FP8_BLOCK = 128


def _rubin_mxfp8() -> bool:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _split(width: int, parts: int, index: int) -> int:
    return (width // parts) * index + min(width % parts, index)


def shard_range(width: int, tp_size: int, tp_rank: int, alignment: int = 1) -> Tuple[int, int]:
    """``[start, end)`` of rank ``tp_rank`` over a split dim of ``width`` elements."""
    if alignment <= 1:
        return _split(width, tp_size, tp_rank), _split(width, tp_size, tp_rank + 1)
    if width % alignment:
        raise ValueError(f"split dim {width} is not a multiple of {alignment}")
    blocks = width // alignment
    return (
        _split(blocks, tp_size, tp_rank) * alignment,
        _split(blocks, tp_size, tp_rank + 1) * alignment,
    )


def load_shard(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    mode: Optional[TensorParallelMode],
    *,
    features: Optional[int] = None,
    alignment: int = 1,
    scale_span: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """This rank's slice of a checkpoint tensor (a weight, or with ``scale_span`` its block-scale
    grid) on ``device``. ``features``: the full split-dim width of the weight (default: the
    tensor's). A 1-D tensor of a ``ROW`` projection, or a split dim of width 1, is not split."""
    device = device if device is not None else tensor.device
    if mode is None or tp_size <= 1:
        return tensor.to(device)
    dim = TensorParallelMode.split_dim(mode)
    if (tensor.dim() == 1 and dim == 1) or tensor.shape[dim] == 1:
        return tensor.to(device)
    start, end = shard_range(features or tensor.shape[dim], tp_size, tp_rank, alignment)
    if scale_span:
        if start % scale_span or end % scale_span:
            raise ValueError(f"shard [{start}, {end}) does not align with {scale_span}-wide scales")
        start, end = start // scale_span, end // scale_span
    return tensor.narrow(dim, start, end - start).to(device)


class _ShardedLinear(nn.Module):
    alignment = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_mode: Optional[TensorParallelMode],
    ) -> None:
        super().__init__()
        self.tp_size, self.tp_rank, self.tp_mode = tp_size, tp_rank, tp_mode
        self.full_in_features, self.full_out_features = in_features, out_features
        self.in_features, self.out_features = in_features, out_features
        if tp_size > 1 and tp_mode is not None:
            dim_width = in_features if tp_mode == TensorParallelMode.ROW else out_features
            start, end = shard_range(dim_width, tp_size, tp_rank, self.alignment)
            if tp_mode == TensorParallelMode.ROW:
                self.in_features = end - start
            else:
                self.out_features = end - start

    def split_features(self) -> int:
        return (
            self.full_in_features
            if self.tp_mode == TensorParallelMode.ROW
            else self.full_out_features
        )

    def shard(self, tensor: torch.Tensor, scale_span: Optional[int] = None) -> torch.Tensor:
        """This rank's slice of a full checkpoint tensor of the projection, on its device."""
        return load_shard(
            tensor,
            self.tp_size,
            self.tp_rank,
            self.tp_mode,
            features=self.split_features(),
            alignment=self.alignment,
            scale_span=scale_span,
            device=self.weight.device,
        )


def bf16_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ weight.T (+ bias)``: cuBLASLt via ``F.linear``."""
    return F.linear(x, weight, bias)


class Bf16Linear(_ShardedLinear):
    """BF16 (or FP16 / FP32) ``weight [out, in]`` (and ``bias [out]``) of one rank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def load(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Copy this rank's shard of the full checkpoint ``weight`` (and ``bias``)."""
        self.weight.data.copy_(self.shard(weight).to(self.weight.dtype))
        if bias is not None:
            self.bias.data.copy_(self.shard(bias).to(self.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bias on rank 0 only for a row-parallel projection (its partial sums add up).
        bias = self.bias if self.tp_mode != TensorParallelMode.ROW or self.tp_rank == 0 else None
        return bf16_linear(x, self.weight, bias)


def prepare_fp8_block_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scale)`` for :func:`fp8_block_scale_gemm` from checkpoint E4M3 codes and their FP32
    128 x 128 grid: scales rounded to UE8M0 with the codes rescaled to match (in place), then the
    scale laid out for this GPU's GEMM (K32 R128c4 UE8M0 bytes on SM107, DeepGEMM-packed UE8M0
    otherwise)."""
    from tensorrt_llm.quantization.utils.fp8_utils import (
        resmooth_to_fp8_e8m0,
        transform_k128_scales_to_cutedsl_mxfp8_layout,
        transform_sf_into_required_layout,
    )

    weight, weight_scale = resmooth_to_fp8_e8m0(
        weight.contiguous(), weight_scale.contiguous().float()
    )
    n, k = weight.shape
    if _rubin_mxfp8():
        return weight, transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
    return weight, transform_sf_into_required_layout(
        weight_scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )


def low_m_gemm_rows(n: int, k: int) -> int:
    """Largest M ``trtllm::fp8_block_scale_low_m_gemm`` serves for an ``[n, k]`` weight here (0:
    none)."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return 0
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.max_m_for(n, k)


def low_m_gemm_scale(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from a :func:`prepare_fp8_block_weight`
    scale (itself on SM100 / SM103, repacked from K32 R128c4 on SM107)."""
    if not _rubin_mxfp8():
        return weight_scale
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


def fp8_block_scale_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    low_m_max: int = 0,
    low_m_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 ``x [M, K] @ weight.T`` on :func:`prepare_fp8_block_weight` operands (see the module
    docstring for the chain per GPU)."""
    if (
        low_m_max
        and 1 <= x.shape[0] <= low_m_max
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    ):
        return torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, weight, low_m_scale)
    if _rubin_mxfp8():
        from ..ops.gemm.k3_mxfp8_gemm import k3_mxfp8_gemm

        act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
        return k3_mxfp8_gemm(act, weight, act_scale, weight_scale)
    return torch.ops.trtllm.fp8_swap_ab_gemm(
        x, weight, weight_scale, output_dtype=x.dtype, disable_ue8m0_cast=True
    )


class Fp8BlockScaleLinear(_ShardedLinear):
    """FP8 block-scale ``weight`` / ``weight_scale`` of one rank (plus the unit ``input_scale`` /
    ``inv_input_scale`` checkpoint FP8 projections also register), bias-free."""

    alignment = FP8_BLOCK

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        n, k = self.out_features, self.in_features
        self.weight = nn.Parameter(
            torch.empty(n, k, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(-(-n // FP8_BLOCK), -(-k // FP8_BLOCK), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.inv_input_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.register_parameter("bias", None)
        self.prepared = False
        self.low_m_max = 0
        self.low_m_scale: Optional[torch.Tensor] = None

    def load(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        """Copy this rank's shard of the full checkpoint codes and their FP32 block-scale grid
        (a modelopt ``[N/128, 1, K/128, 1]`` grid is squeezed first)."""
        if weight_scale.dim() == 4:
            weight_scale = weight_scale.squeeze(1).squeeze(-1)
        self.weight.data.copy_(self.shard(weight))
        self.weight_scale.data.copy_(self.shard(weight_scale, scale_span=FP8_BLOCK).float())
        self.prepared = False

    def prepare(self, low_m: bool = True) -> None:
        """Once after loading: :func:`prepare_fp8_block_weight` in place of the parameters, and
        with ``low_m`` the rows the low-M GEMM serves and its scale."""
        if self.prepared:
            return
        weight, scale = prepare_fp8_block_weight(self.weight.data, self.weight_scale.data)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        n, k = self.out_features, self.in_features
        self.low_m_max = low_m_gemm_rows(n, k) if low_m else 0
        self.low_m_scale = low_m_gemm_scale(scale, n, k) if self.low_m_max else None
        self.prepared = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.prepared:
            raise RuntimeError("Fp8BlockScaleLinear.prepare() must run after loading")
        x_flat = x.reshape(-1, x.shape[-1])
        out = fp8_block_scale_gemm(
            x_flat,
            self.weight,
            self.weight_scale,
            low_m_max=self.low_m_max,
            low_m_scale=self.low_m_scale,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


_FLASHINFER_DTYPES = (torch.bfloat16, torch.float16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``weight * x / sqrt(mean(x^2) + eps)`` over the last dim of a BF16 / FP16 ``x``."""
    if x.dtype not in _FLASHINFER_DTYPES:
        raise TypeError(f"rms_norm takes BF16 / FP16 activations, got {x.dtype}")
    return flashinfer_rmsnorm(x, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In place: ``residual += x``, then ``x = rms_norm(residual)``. Returns ``(x, residual)``."""
    if x.dtype not in _FLASHINFER_DTYPES:
        raise TypeError(f"fused_add_rms_norm takes BF16 / FP16 activations, got {x.dtype}")
    return flashinfer_fused_add_rmsnorm(x, residual, weight, eps)


class RMSNormWeight(nn.Module):
    """The ``[hidden]`` RMSNorm weight (``weight``, initialized to ones) and its epsilon."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    @property
    def eps(self) -> float:
        """``variance_epsilon``, under the name the gated (KDA ``o_norm``) and attention-residual
        norm readers use."""
        return self.variance_epsilon

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.variance_epsilon)
        return fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)


class ShortConvWeight(nn.Module):
    """``weight [channels, 1, kernel_size]`` of one rank's channels."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        activation: Optional[str] = "silu",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size, self.kernel_size, self.activation = channels, kernel_size, activation
        self.weight = nn.Parameter(
            torch.empty(channels, 1, kernel_size, dtype=dtype, device=device)
        )
        self.register_parameter("bias", None)


def _record_on(obj: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            try:
                obj.record_stream(stream)
            except RuntimeError as error:
                # cudaMallocAsync asserts on memory it did not allocate (an external buffer wrapped
                # as a tensor), which it never frees: nothing to record.
                if (
                    torch.cuda.memory.get_allocator_backend() != "cudaMallocAsync"
                    or "ptr not found in ptr_info" not in str(error)
                ):
                    raise
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _record_on(item, stream)
    elif isinstance(obj, dict):
        for item in obj.values():
            _record_on(item, stream)


def fork(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    join_event: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any, Optional[torch.cuda.Event]]:
    """``fn0()`` and ``fn1()`` and the event the calling stream must wait on before it reads
    ``fn1()``'s outputs (None when both ran on the calling stream)."""
    if not (do_multi_stream() and aux_stream is not None):
        return fn0(), fn1(), None
    fork_event.record()
    result0 = fn0()
    with torch.cuda.stream(aux_stream):
        fork_event.wait()
        result1 = fn1()
        join_event.record()
    if not torch.cuda.is_current_stream_capturing():
        _record_on(result1, torch.cuda.current_stream())
    return result0, result1, join_event


def fork_join(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    join_event: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any]:
    """``(fn0(), fn1())``, ``fn1`` on ``aux_stream`` where :func:`fork` forks; joined."""
    result0, result1, join = fork(fn0, fn1, fork_event, join_event, aux_stream)
    if join is not None:
        join.wait()
    return result0, result1


def masked_embedding(ids: torch.Tensor, weight: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Rows ``ids - start`` of ``weight`` where ``start <= ids < end``, zeros elsewhere."""
    hit = (ids >= start) & (ids < end)
    out = F.embedding(hit * (ids - start * hit), weight)
    out.masked_fill_(~hit.unsqueeze(-1), 0)
    return out


class EmbeddingTable(nn.Module):
    """``weight`` of the whole vocabulary and the masked lookup above."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
            requires_grad=False,
        )

    def load(self, weight: torch.Tensor) -> None:
        """Copy the full checkpoint table."""
        self.weight.data.copy_(weight[: self.num_embeddings])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return masked_embedding(ids, self.weight, 0, self.weight.shape[0])


def subgroup_mapping(mapping: Mapping, tp_size: int) -> Mapping:
    """``mapping`` regrouped into TP groups of ``tp_size`` consecutive ranks."""
    if mapping.tp_size % tp_size:
        raise ValueError(f"TP {mapping.tp_size} does not split into groups of {tp_size}")
    pp_size = mapping.pp_size * mapping.tp_size // tp_size
    return Mapping(
        world_size=tp_size * pp_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        tp_size=tp_size,
        pp_size=pp_size,
        enable_attention_dp=mapping.enable_attention_dp,
    )


def gated_activation(
    h: torch.Tensor, activation: str, beta: float = 1.0, linear_beta: Optional[float] = None
) -> torch.Tensor:
    """``act(gate) * up`` of a packed ``[..., 2 * I]`` = ``[gate | up]`` tensor."""
    if activation == "situ":
        out = situ_and_mul(h.reshape(-1, h.shape[-1]), beta, linear_beta)
        return out.reshape(*h.shape[:-1], h.shape[-1] // 2)
    if activation == "silu":
        return torch.ops.trtllm.silu_and_mul(
            h, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
        )
    raise ValueError(f"unknown gated activation {activation!r}")


class GatedMlp(nn.Module):
    """``gate_up_proj`` / ``down_proj`` BF16 weights of one rank and the chain above."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: str,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Optional[Mapping] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
        if mapping.tp_size != 1:
            raise ValueError(
                f"the Kimi K3 dep16 MLPs are replicated (TP 1); got TP {mapping.tp_size}"
            )
        self.mapping = mapping
        self.activation, self.beta, self.linear_beta = activation, beta, linear_beta
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        tp, rank = mapping.tp_size, mapping.tp_rank
        self.inter_start, self.inter_end = shard_range(intermediate_size, tp, rank)
        local = self.inter_end - self.inter_start
        self.gate_up_proj = Bf16Linear(hidden_size, 2 * local, dtype=dtype, device=device)
        self.down_proj = Bf16Linear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            tp_size=tp,
            tp_rank=rank,
            tp_mode=TensorParallelMode.ROW,
            device=device,
        )

    def load(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        """Copy this rank's rows of the full checkpoint gate / up ``[I, H]`` and columns of down
        ``[H, I]``."""
        start, end = self.inter_start, self.inter_end
        local = end - start
        weight = self.gate_up_proj.weight
        weight.data[:local].copy_(gate[start:end].to(weight.device, weight.dtype))
        weight.data[local:].copy_(up[start:end].to(weight.device, weight.dtype))
        self.down_proj.load(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = bf16_linear(x, self.gate_up_proj.weight)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a)
