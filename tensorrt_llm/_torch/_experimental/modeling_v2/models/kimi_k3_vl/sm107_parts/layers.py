# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-call primitives of the Kimi K3 Rubin model: K3-owned weights and direct kernel calls.

Each container is an ``nn.Module`` that registers the parameter names the checkpoint and the
kernels read (``weight``, ``weight_scale``, ...), loads its rank's shard with an explicit
``load`` (and ``prepare`` where the kernel wants another layout), and runs one fixed kernel chain:

* ``linear``: BF16 ``F.linear``; FP8 128 x 128 block scales on the low-M GEMM, the SM107 MXFP8
  GEMM (``mxfp8_gemm``) or the SM100 / SM103 swap-AB GEMM.
* ``ops/gemm/_mxfp8_gemm``: the SM107 MXFP8 GEMM over K3's tactics and its autotuner runner.
* ``norm``: FlashInfer RMSNorm (plain and fused residual add) on a K3 weight.
* ``mlp``: gate-up GEMM, SiTU / SiLU-and-mul, down GEMM and its all-reduce.
* ``embedding``: masked (vocab-parallel) lookup and its all-reduce.
* ``conv``: the short-convolution weight the KDA kernels read.
* ``allreduce``: the TP all-reduce on the MNNVL / trtllm all-reduce op.
* ``streams``: two branches on the current and an auxiliary stream.

Tensor-parallel all-reduce of K3's row-parallel projections, as one fixed kernel call.

:class:`TpAllReduce` decides its transport once, at construction, from the mapping and the requested
strategy (``ModelConfig.allreduce_strategy``, ``AUTO`` in every K3 recipe):

* **MNNVL**: ``AUTO`` / ``MNNVL`` on a group MNNVL serves (BF16 / FP16 / FP32, no CP, aarch64 with
  every NVLink up, and the group spans nodes, an explicit ``MNNVL``, or ``TLLM_TEST_MNNVL=1``). Each
  call is ``trtllm::mnnvl_fusion_allreduce`` on the Lamport workspace of
  ``get_or_scale_allreduce_mnnvl_workspace`` (one per mapping, shared with every other MNNVL
  all-reduce and the pushed all-reduces of ``layers.py``; allocated here, a TP-wide collective,
  and grown on the first eager call that needs more). The kernel is one-shot up to
  ``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES`` of ``tokens * hidden * ranks * element size``, two-shot above.
  Every K3 all-reduce runs here on Rubin / GB300 NVL72: helix GEN groups span nodes, and the dense
  MLP's TP-4 subgroup is declared over the whole world.
* **The trtllm all-reduce op** (the catalog's ``allreduce`` entry: ``trtllm::allreduce`` on the IPC
  workspace of ``get_allreduce_workspace``) for any other group, with the strategy fixed: an
  explicit one as given, ``AUTO`` as the op's own size heuristic
  (``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1`` behaviour of main's ``AllReduce``; main otherwise
  autotunes the strategy there).
* **Identity** with one rank or under attention DP (no TP group to reduce over).

Fusions: ``NONE`` and the residual / plain RMSNorm epilogues the MNNVL kernel takes
(``AllReduceParams``), with main's output contract (a tensor, or a tuple for fusions with several
outputs).

For ``layers.py`` an MNNVL instance also answers ``mnnvl_allreduce`` (itself), ``mapping``,
``dtype``, ``symm_mem_allreduce`` (None) and ``get_required_workspace_size``.

Projections on K3-owned weights: tensor-parallel shards, BF16 GEMM and FP8 block-scale GEMM.

Weight containers register the parameters under the names the checkpoint loader looks up and keep
this rank's shard. Their forward is one explicit kernel chain. A row-parallel BF16 projection given
an all-reduce (``layers.TpAllReduce``, as ``all_reduce``) reduces its partial with it, with
the caller's ``all_reduce_params`` (``enable_allreduce=False``: the partial; an RMSNorm fusion:
the fused outputs), as main's ``Linear(ROW, reduce_output=True)``.

**Sharding** (:func:`shard_range` / :func:`load_shard`): rank ``r`` of ``tp`` takes a contiguous
range of the split dim (output features for ``COLUMN``, input features for ``ROW``), the first
``width % tp`` ranks one unit more; for FP8 block-scale weights the unit is a 128-wide block and
the block-scale grid follows the weight's range (``scale_span=128``). :func:`load_weight_shard`
is main's ceil-divided split (``ceil(width / tp)`` per rank), which the NVFP4 experts load with.

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

:class:`EmbeddingTable` holds ``weight`` ``[rows, hidden]``: the whole vocabulary, or with a
vocab-parallel mapping (``tp_size > 1``, no attention DP) rank ``r``'s rows
``[r * ceil(V / tp), min((r + 1) * ceil(V / tp), V))``, zero-padded to ``ceil(V / tp)`` rows. Its
forward is

    hit = (ids >= start) & (ids < end)                  # start, end: the rows this rank holds
    out = F.embedding(hit * (ids - start * hit), weight)
    out.masked_fill_(~hit, 0)
    out = all_reduce(out)                               # vocab-parallel, unless the caller reduces

so an id outside the table (the rejection sampler pads with -1) embeds to zeros and every vocab
row comes from exactly one rank. :func:`gather_embedding` is the one-launch lookup of a whole table
(``ops/embedding/k3_embedding_gather.py``, bitwise the masked lookup).

Gated MLP on K3-owned BF16 weights: gate/up GEMM, gated activation, down GEMM, all-reduce.

    h = x @ gate_up_proj.weight.T            # F.linear, [M, 2 * I_local] = [gate | up]
    a = act(h)                               # family situ_and_mul or trtllm::silu_and_mul
    y = a @ down_proj.weight.T               # F.linear, [M, H] (row-parallel partial)
    y = down_proj.all_reduce(y, params)      # layers.TpAllReduce, tp > 1 only

``gate_up_proj.weight`` holds this rank's rows ``[start, end)`` of the gate weight followed by the
same rows of the up weight (``start, end`` from :func:`layers.shard_range` over the
intermediate size); ``down_proj.weight`` the same columns of the down weight. ``params`` is the
caller's ``final_all_reduce_params`` (``enable_allreduce=False`` returns the partial). Activations:

* ``situ``: the family ``situ_and_mul(h, beta, linear_beta)`` entry, the Triton SiTU kernel
  (``beta * tanh(gate / beta) * sigmoid(gate) * (linear_beta * tanh(up / linear_beta) or up)``,
  FP32 math);
* ``silu``: ``trtllm::silu_and_mul(h)``, the Triton SwiGLU kernel.

:func:`subgroup_mapping` is the mapping of a TP subgroup of ``tp_size`` ranks: consecutive ranks
reduce together and the remaining factor is folded into PP, the subgroup the all-reduce runs over
when the MLP's TP size is smaller than the model's.

Pushed MNNVL all-reduce, private to the Kimi K3 Rubin package.

A producer kernel may store its BF16 output straight into the Lamport buffers of the MNNVL
all-reduce that follows it (the K3 low-M FP8 GEMM's push epilogue, ``kernels/k3_lamport_ar.py``).
:func:`lamport_push_target` hands the buffers to the producer and :func:`finish_lamport_push`
completes the all-reduce, bitwise like the C++ kernel the call replaces.

This is the push API of the Rubin agentic branch (``189fe67b81``, ``distributed/ops.py``:
``LamportPushTarget``/``LamportPending`` and the ``AllReduce`` /
``MNNVLAllReduce`` methods ``lamport_push_target``, ``lamport_push_tensor`` and
``finish_lamport_push``), as free functions that take the all-reduce: ``layers.TpAllReduce``
(or main's ``AllReduce``), read through ``mnnvl_allreduce`` (the MNNVL transport: its ``mapping``,
``dtype`` and ``get_required_workspace_size``) and ``symm_mem_allreduce``. They work on the shared
MNNVL workspace of ``get_or_scale_allreduce_mnnvl_workspace`` (same dict keys and the same 9-word
``buffer_flags`` layout).

Differences from the branch, all forced by main:

* The C++ one-/two-shot switch is ``distributed.ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES``
  (``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES``): the pushed call must reproduce the kernel main's
  ``MNNVLAllReduce`` would have run.
* The device view of a workspace is built on the first eager call; during CUDA graph capture a
  workspace without a view, or one that would have to grow, gives no target (the caller falls
  back to the plain all-reduce). The eager warm-up forward that precedes every capture builds it.
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import allreduce
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.situ_and_mul import (
    situ_and_mul,
)
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    get_allreduce_workspace,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._utils import get_sm_version, is_device_integrated
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

MNNVL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
#: Fusions ``trtllm::mnnvl_fusion_allreduce`` runs (besides ``NONE``).
MNNVL_FUSION_OPS = frozenset(
    {
        AllReduceFusionOp.RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
    }
)


def mnnvl_serves(mapping: Mapping, dtype: torch.dtype, strategy: AllReduceStrategy) -> bool:
    """Whether the all-reduce of ``mapping`` runs on MNNVL for ``strategy`` (``AUTO`` / ``MNNVL``)."""
    if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        return False
    if dtype not in MNNVL_DTYPES or mapping.has_cp():
        return False
    if os.environ.get("TLLM_TEST_MNNVL", "0") == "1":
        return True
    supported = MnnvlMemory.supports_mnnvl() and "aarch64" in platform.machine().lower()
    return supported and (strategy == AllReduceStrategy.MNNVL or mapping.is_multi_node())


def mnnvl_workspace_bytes(num_tokens: int, hidden: int, world: int, dtype: torch.dtype) -> int:
    """Lamport buffer bytes one MNNVL call of ``[num_tokens, hidden]`` needs."""
    elem = torch.tensor([], dtype=dtype).element_size()
    if num_tokens * hidden * world * elem <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES:
        return num_tokens * hidden * world * elem
    return 2 * math.ceil(num_tokens / world) * world * hidden * elem


class TpAllReduce:
    """The all-reduce over ``mapping``'s TP group; ``self(x)`` returns the reduced tensor."""

    def __init__(
        self,
        mapping: Mapping,
        dtype: torch.dtype,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.strategy = strategy
        self.symm_mem_allreduce = None
        self.transport = "identity"
        self.workspace = None
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return
        if mnnvl_serves(mapping, dtype, strategy):
            self.transport = "mnnvl"
            get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)
            return
        if strategy in (
            AllReduceStrategy.MNNVL,
            AllReduceStrategy.SYMM_MEM,
            AllReduceStrategy.UB,
            AllReduceStrategy.LOWPRECISION,
        ):
            raise ValueError(f"K3 all-reduce: strategy {strategy.name} is not served here")
        self.transport = "op"
        if strategy not in (AllReduceStrategy.NCCL, AllReduceStrategy.NCCL_SYMMETRIC):
            self.workspace = get_allreduce_workspace(mapping)

    get_required_workspace_size = staticmethod(mnnvl_workspace_bytes)

    @property
    def mnnvl_allreduce(self) -> Optional["TpAllReduce"]:
        """Self on the MNNVL transport (what ``layers.py`` reads), else None."""
        return self if self.transport == "mnnvl" else None

    def __call__(
        self, x: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        params = all_reduce_params
        if self.transport == "identity" or (params is not None and not params.enable_allreduce):
            return x
        x = x.contiguous()
        params = params or AllReduceParams()
        if self.transport == "mnnvl":
            return self._mnnvl(x, params)
        return allreduce(
            x,
            params.residual,
            params.norm_weight,
            params.scale,
            params.bias,
            self.workspace,
            self.mapping.tp_group,
            self.strategy,
            params.fusion_op,
            params.eps,
            params.trigger_completion_at_end,
        )

    def _mnnvl(
        self, x: torch.Tensor, params: AllReduceParams
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        fusion_op = params.fusion_op
        is_fusion = fusion_op != AllReduceFusionOp.NONE
        if is_fusion and fusion_op not in MNNVL_FUSION_OPS:
            raise ValueError(f"the MNNVL all-reduce does not fuse {fusion_op!r}")
        hidden = x.shape[-1]
        num_tokens = x.numel() // hidden
        need = mnnvl_workspace_bytes(num_tokens, hidden, self.mapping.tp_size, self.dtype)
        if need >= 2**32 - 1:
            raise ValueError(
                f"MNNVL all-reduce of ({num_tokens}, {hidden}) over {self.mapping.tp_size} ranks "
                f"needs {need} workspace bytes, past the kernel's uint32 limit"
            )
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, self.dtype, buffer_size_bytes=need
        )
        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            x,
            params.norm_weight,
            params.residual,
            params.eps,
            workspace["uc_buffer"].view(self.dtype).view(3, -1),
            workspace["buffer_flags"],
            is_fusion,
            params.scale,
            int(fusion_op),
            fp8_payload=False,
        )
        if is_fusion and len(outputs) > 1:
            return tuple(outputs)
        return outputs[0]


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


def load_weight_shard(
    weight,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    tensor_parallel_mode: Optional[TensorParallelMode] = None,
    device: torch.device = torch.device("cpu"),
    return_slice_indices: bool = False,
) -> torch.Tensor:
    """This rank's ceil-divided slice of ``weight`` (a tensor or a safetensors slice) on ``device``:
    ``ceil(width / tp)`` elements of the split dim per rank, the last rank the remainder. A 1-D
    tensor of a ``ROW`` projection, or a split dim of width 1, is not split. On an integrated GPU
    the tensor stays on its own device. ``return_slice_indices`` (tensors only): also the slices."""
    # Skip device transfers on integrated GPUs to conserve shared memory
    if weight.device.type != device.type and is_device_integrated():
        # For integrated GPU systems (e.g., DGX Spark), CPU and GPU share limited physical memory.
        # Avoiding device transfers reduces memory consumption and unnecessary data copies,
        # enabling support for larger models on memory-constrained systems.
        logger.warning_once(
            f"[load_weight_shard] Skipping device transfer from {weight.device} to {device} "
            "on integrated GPU to conserve shared memory.",
            key="load_weight_shard_skip_device_transfer_with_integrated_gpu",
        )
        device = weight.device
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor, indices: list[slice] | None = None):
            if indices is None:
                # Avoid unnecessary copy
                result = (tensor.to(device), [slice(d) for d in tensor.shape])
            else:
                result = (tensor[indices].to(device), indices)
            return result if return_slice_indices else result[0]

    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to
    # the module; safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)
        ):
            return tensor[indices].to(device)
    else:
        raise ValueError(f"unsupported weight type: {type(weight)}")
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(d) for d in tensor_shape]
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


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
        all_reduce: Optional[Callable[..., torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        if all_reduce is not None and tp_mode != TensorParallelMode.ROW:
            raise ValueError("only a row-parallel projection reduces its output")
        self.all_reduce = all_reduce
        self.reduce_output = all_reduce is not None
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

    def forward(
        self, x: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        # Bias on rank 0 only for a row-parallel projection (its partial sums are all-reduced).
        bias = self.bias if self.tp_mode != TensorParallelMode.ROW or self.tp_rank == 0 else None
        out = bf16_linear(x, self.weight, bias)
        if self.all_reduce is not None:
            out = self.all_reduce(out, all_reduce_params=all_reduce_params)
        return out


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


def gather_embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """:func:`masked_embedding` of a whole table in one Triton launch (PDL where enabled)."""
    from ..ops.embedding.k3_embedding_gather import k3_embedding_gather

    return k3_embedding_gather(ids, weight)


class EmbeddingTable(nn.Module):
    """``weight`` of one rank and the lookup above."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        vocab_parallel: bool = False,
        reduce_output: bool = True,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.vocab_parallel = (
            vocab_parallel and mapping.tp_size > 1 and not mapping.enable_attention_dp
        )
        if self.vocab_parallel:
            rows = math.ceil(num_embeddings / mapping.tp_size)
            self.vocab_start = mapping.tp_rank * rows
            self.vocab_end = min(self.vocab_start + rows, num_embeddings)
            self.all_reduce = (
                TpAllReduce(mapping, dtype or torch.bfloat16, allreduce_strategy)
                if reduce_output
                else None
            )
        else:
            rows = num_embeddings
            self.vocab_start, self.vocab_end = 0, num_embeddings
            self.all_reduce = None
        self.weight = nn.Parameter(
            torch.empty(rows, embedding_dim, dtype=dtype, device=device), requires_grad=False
        )

    def load(self, weight: torch.Tensor) -> None:
        """Copy this rank's rows of the full checkpoint table (zero padding rows past the vocab)."""
        rows = self.vocab_end - self.vocab_start
        self.weight.data[:rows].copy_(weight[self.vocab_start : self.vocab_end])
        self.weight.data[rows:].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = masked_embedding(
            ids,
            self.weight,
            self.vocab_start,
            self.vocab_end if self.vocab_parallel else self.weight.shape[0],
        )
        if self.all_reduce is not None:
            out = self.all_reduce(out)
        return out


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
        reduce_output: bool = True,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
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
            all_reduce=(
                TpAllReduce(mapping, dtype, allreduce_strategy)
                if reduce_output and tp > 1
                else None
            ),
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

    def forward(
        self, x: torch.Tensor, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        h = bf16_linear(x, self.gate_up_proj.weight)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a, all_reduce_params=final_all_reduce_params)


# Pushed all-reduces of at most this many bytes (num_tokens * hidden_dim * group_size * 2) use the
# one-shot transport, larger ones the two-shot transport: the transport of the C++ kernel the
# call replaces. The result always reproduces the C++ kernel main selects.
_K3_AR_PUSH_ONE_SHOT_MAX_BYTES = int(_MNNVL_ONE_SHOT_THRESHOLD_BYTES)
# Pushed all-reduces whose consumer finishes them (``deferred``: the K3 attn_res kernel reduces
# the slots itself, after o_proj and after the row-parallel MoE tail) use the one-shot transport up
# to this many bytes, the two-shot transport above.
_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES = _K3_AR_PUSH_ONE_SHOT_MAX_BYTES


@dataclasses.dataclass(frozen=True)
class LamportWorkspaceView:
    """Device addresses of one MNNVL all-reduce workspace for pushed calls, as the routed experts'
    output push takes them.

    ``buffer``: this rank's unicast mapping of the workspace; ``mc_base``: its multicast mapping;
    ``peer_ptrs``: int64 CUDA tensor with every rank's unicast mapping (this rank's own at index
    ``rank``); ``flags``: the uint32 rotation state shared with the C++ kernels;
    ``bytes_per_buffer``: size of one of the three Lamport buffers.
    """

    world: int
    rank: int
    buffer: torch.Tensor
    mc_base: int
    peer_ptrs: torch.Tensor
    flags: torch.Tensor
    bytes_per_buffer: int

    @property
    def uc_local(self) -> int:
        """Device address of ``buffer``."""
        return self.buffer.data_ptr()


# Device views of the workspaces for pushed all-reduces, by the address of the workspace's buffer
# (a scaled workspace is a new buffer).
_LAMPORT_VIEWS: Dict[int, object] = {}
# The latest shared MNNVL workspace an eager call here got, by all-reduce mapping (workspaces only
# grow: a later one is at least as large).
_SEEN_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


class LamportPushTarget:
    """One MNNVL all-reduce whose input the producer pushes.

    Made by :func:`lamport_push_target` for a single call. The producer stores its BF16
    ``[num_tokens, hidden]`` output into the Lamport slots of ``workspace`` (a
    ``k3_lamport_ar.LamportWorkspace``) with the ``one_shot`` or two-shot transport and sets
    ``pushed``; :func:`finish_lamport_push` then writes the all-reduce result into ``output``,
    reproducing the C++ one-shot kernel if ``cpp_one_shot``, else the two-shot kernel.
    """

    def __init__(self, workspace, one_shot: bool, cpp_one_shot: bool, output: torch.Tensor):
        self.workspace = workspace
        self.one_shot = one_shot
        self.cpp_one_shot = cpp_one_shot
        self.output = output
        self.pushed = False
        # Set by ``lamport_push_target(deferred=True)``: the one-shot transport, finished by the
        # consumer (``LamportPending``).
        self.deferred = False
        # The call carries all-gather rows (``k3_lamport_ar`` "ag", one-shot transport): its
        # result is ``bf16(y + bf16(sum of the slots))``.
        self.ag = False


class LamportPending:
    """A pushed all-reduce whose finish is left to its consumer
    (``finish_lamport_push(..., deferred=True)``).

    Every rank has stored its partial into the slots of the call (one-shot or two-shot
    transport), but no kernel has reduced them or advanced the workspace's rotation state: the
    consumer's kernel does both (``k3_attn_res_lamport``), or ``materialize`` runs
    the finish kernel into a tensor. Exactly one of the two must run, before the next all-reduce on
    the workspace.
    """

    def __init__(self, target: LamportPushTarget):
        self.target = target
        self.workspace = target.workspace
        self.shape = tuple(target.output.shape)

    @property
    def finish_args(self) -> Tuple[int, int, bool, int, int]:
        """``(finish_world, finish_chunk, finish_two_shot, mc_base, rank)`` of
        ``k3_attn_res_lamport``: the ranks per partial sum of the C++ kernel the result
        reproduces, the transport and the workspace addresses."""
        from ..kernels.k3r_comm import k3_lamport_ar

        ws = self.workspace
        return (
            ws.world,
            k3_lamport_ar.reduce_chunk(ws.world, self.target.cpp_one_shot),
            not self.target.one_shot,
            ws.mc_base,
            ws.rank,
        )

    @property
    def finish_ag(self) -> bool:
        """Whether the call carries all-gather rows (``finish_ag`` of ``k3_attn_res_lamport``)."""
        return getattr(self.target, "ag", False)

    def materialize(self) -> torch.Tensor:
        """Finish the all-reduce into a BF16 tensor (the finish kernel on the current stream)."""
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from ..ops.comm.k3_lamport_finish import k3_lamport_finish

        return k3_lamport_finish(
            self.workspace,
            self.target.output,
            self.target.one_shot,
            use_pdl=get_env_enable_pdl(),
            cpp_one_shot=self.target.cpp_one_shot,
            ag=self.finish_ag,
        )


def _lamport_view(mnnvl: Any, workspace: dict):
    """The ``k3_lamport_ar.LamportWorkspace`` of an MNNVL workspace (built once per buffer)."""
    key = workspace["uc_buffer"].data_ptr()
    view = _LAMPORT_VIEWS.get(key)
    if view is None:
        from ..kernels.k3r_comm import k3_lamport_ar

        view = k3_lamport_ar.LamportWorkspace.from_mnnvl(
            workspace, mnnvl.mapping.tp_size, mnnvl.mapping.tp_rank
        )
        _LAMPORT_VIEWS[key] = view
        logger.info_once(
            f"[MNNVL AllReduce] pushed all-reduce: Lamport workspace view "
            f"for tp_size {mnnvl.mapping.tp_size} ({view.bytes_per_buffer} bytes per buffer)",
            key=("k3_ar_push_view", mnnvl.mapping.tp_size),
        )
    return view


def mnnvl_lamport_push_target(
    mnnvl: Any,
    num_tokens: int,
    hidden_dim: int,
    one_shot: Optional[bool] = None,
) -> Optional[LamportPushTarget]:
    """``MNNVLAllReduce.lamport_push_target`` of the branch: a target for a plain all-reduce of
    BF16 ``[num_tokens, hidden_dim]`` on the workspace of ``mnnvl`` (an all-reduce's MNNVL
    transport, ``mnnvl_allreduce``), or None. ``one_shot`` forces the transport instead of the
    size rule."""
    from ..kernels.k3r_comm import k3_lamport_ar

    world = mnnvl.mapping.tp_size
    nbytes = num_tokens * hidden_dim * world * 2
    cpp_one_shot = nbytes <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    if one_shot is None:
        one_shot = nbytes <= _K3_AR_PUSH_ONE_SHOT_MAX_BYTES
    need = max(
        mnnvl.get_required_workspace_size(num_tokens, hidden_dim, world, mnnvl.dtype),
        k3_lamport_ar.required_bytes(num_tokens, hidden_dim, world, one_shot),
    )
    if need >= 2**32 - 1:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        # Main raises when a workspace is created or grows during capture, and building the view
        # is a host-to-device copy: both must have happened in the eager warm-up of this shape.
        seen = _SEEN_WORKSPACES.get(mnnvl.mapping)
        if (
            seen is None
            or seen["buffer_size_bytes"] < need
            or seen["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS
        ):
            return None
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mnnvl.mapping, mnnvl.dtype, buffer_size_bytes=need
    )
    if capturing:
        if workspace["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS:
            # Another all-reduce grew the workspace after the warm-up's push here.
            return None
    else:
        _SEEN_WORKSPACES[mnnvl.mapping] = workspace
    view = _lamport_view(mnnvl, workspace)
    output = torch.empty(
        num_tokens, hidden_dim, dtype=mnnvl.dtype, device=workspace["uc_buffer"].device
    )
    return LamportPushTarget(view, one_shot, cpp_one_shot, output)


def mnnvl_workspace_mpi_comm(workspace: Dict[str, object]):
    """The MPI communicator of a shared MNNVL all-reduce workspace
    (``get_or_scale_allreduce_mnnvl_workspace``), for the package's private Lamport workspaces:
    ``McastGPUBuffer`` takes its Fortran handle. The workspace keys it ``"comm"``; under a non-MPI
    orchestrator that entry is the TP ProcessGroup, which these workspaces do not support."""
    comm = workspace.get("comm")
    if comm is None or not hasattr(comm, "py2f"):
        raise RuntimeError(
            "Kimi K3 private MNNVL workspaces need the MPI communicator of the shared MNNVL "
            "workspace (MPI orchestrator, workspace attached)"
        )
    return comm


def _mnnvl_finish(
    target: LamportPushTarget, deferred: bool = False
) -> Union[torch.Tensor, LamportPending]:
    """``MNNVLAllReduce.finish_lamport_push`` of the branch."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    from ..ops.comm.k3_lamport_finish import k3_lamport_finish

    if deferred:
        return LamportPending(target)
    return k3_lamport_finish(
        target.workspace,
        target.output,
        target.one_shot,
        use_pdl=get_env_enable_pdl(),
        cpp_one_shot=target.cpp_one_shot,
    )


def lamport_push_target(
    allreduce: Any,
    num_tokens: int,
    hidden_dim: int,
    all_reduce_params: Optional[AllReduceParams] = None,
    deferred: bool = False,
) -> Optional[LamportPushTarget]:
    """A ``LamportPushTarget`` if ``allreduce`` would run the MNNVL all-reduce on a BF16
    ``[num_tokens, hidden_dim]`` input with these parameters (plain sum, no fusion); else None.

    A producer that stores its output into the target (and sets ``pushed``) replaces the input of
    this all-reduce; :func:`finish_lamport_push` then returns the same result the all-reduce would.
    Targets must be finished in the order they were pushed, by every rank. ``deferred`` sets
    ``target.deferred`` for ``finish_lamport_push(deferred=True)``, whose consumer finishes the
    call; it takes the one-shot transport where the call is at most
    ``_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES`` (``num_tokens * hidden_dim * world * 2``), the
    two-shot one above.
    """
    mnnvl = getattr(allreduce, "mnnvl_allreduce", None)
    if (
        mnnvl is None
        or allreduce.mapping.tp_size == 1
        or getattr(allreduce, "symm_mem_allreduce", None) is not None
    ):
        return None
    if all_reduce_params is not None and (
        not all_reduce_params.enable_allreduce
        or all_reduce_params.fusion_op != AllReduceFusionOp.NONE
    ):
        return None
    if mnnvl.dtype != torch.bfloat16 or num_tokens <= 0 or hidden_dim % 8:
        return None
    if not deferred:
        return mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim)
    one_shot = (
        num_tokens * hidden_dim * allreduce.mapping.tp_size * 2
        <= _K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
    )
    target = mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim, one_shot=one_shot)
    if target is not None:
        target.deferred = True
    return target


def lamport_push_tensor(allreduce: Any, target: LamportPushTarget, x: torch.Tensor) -> None:
    """Push the contiguous BF16 ``[num_tokens, hidden_dim]`` tensor ``x`` into ``target`` (a copy
    kernel), for a producer without the push epilogue."""
    del allreduce  # the target carries its workspace; kept for the branch's call shape
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    from ..ops.comm.k3_lamport_push import k3_lamport_push

    k3_lamport_push(target.workspace, x.contiguous(), target.one_shot, use_pdl=get_env_enable_pdl())
    target.pushed = True


def finish_lamport_push(
    allreduce: Any,
    target: LamportPushTarget,
    deferred: bool = False,
) -> Union[torch.Tensor, LamportPending]:
    """The all-reduce result of a target the producer has pushed: a tensor, or with ``deferred``
    (where ``target.deferred``) a ``LamportPending`` that the next kernel finishes."""
    del allreduce  # the target carries its workspace; kept for the branch's call shape
    if not target.pushed:
        raise RuntimeError("the producer did not push this all-reduce")
    return _mnnvl_finish(target, deferred and target.deferred)


__all__ = [
    "LamportPending",
    "LamportPushTarget",
    "finish_lamport_push",
    "lamport_push_target",
    "lamport_push_tensor",
    "mnnvl_lamport_push_target",
]
