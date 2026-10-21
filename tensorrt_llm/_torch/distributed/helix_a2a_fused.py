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
"""Helix CP all-to-all fused with the combine of the partial attention outputs.

``TRTLLM_HELIX_A2A_FUSED=1`` replaces the MNNVL FIFO all-to-all (``alltoall_helix_native``) and
``helix_post_process_native`` of ``_helix_post_process`` by one CuTe DSL kernel
(``cute_dsl_kernels/blackwell/helix_a2a_fused.py``) for the shapes it supports (SM100 / SM103,
BF16 partials, ``cp_size <= 8`` or 16, an even number of heads per rank, 128 or 256 values per
head); the result is bitwise the same. Senders write each record directly into the destination
rank's Lamport receive buffer, receivers poll the data itself and combine it in place. The
FIFO workspace of the replaced path is not allocated.

Knobs (environment):

* ``TRTLLM_HELIX_A2A_FUSED_MB``: receive buffer per rank (two halves), default 128 MiB. A call
  whose records do not fit a half runs in several launches.
* ``TRTLLM_HELIX_A2A_FUSED_SEND_CTAS`` / ``_RECV_CTAS``: CTA caps per role (default: twice the
  SM count).
"""

import os
from typing import Dict, List, Optional

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import (
    HelixCpMnnvlMemory,
    MnnvlMemory,
    _check_cu_result,
    cuda,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..flashinfer_utils import get_env_enable_pdl

HELIX_A2A_FUSED = os.environ.get("TRTLLM_HELIX_A2A_FUSED", "0") == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class HelixAllToAllFused:
    """Fused Helix all-to-all + combine over the CP group of ``mapping`` (one per mapping)."""

    _cache: Dict[Mapping, "HelixAllToAllFused"] = {}

    def __init__(self, mapping: Mapping):
        from ..cute_dsl_kernels.blackwell.helix_a2a_fused import SENTINEL

        self.mapping = mapping
        MnnvlMemory.initialize()
        nbytes = _env_int("TRTLLM_HELIX_A2A_FUSED_MB", 128) << 20
        logger.info(
            f"Rank {mapping.cp_rank} allocating a {nbytes >> 20} MiB fused Helix all-to-all buffer"
        )
        self.workspace = HelixCpMnnvlMemory(mapping, nbytes)
        self.ws_addr = self.workspace.ptr
        self.rank_stride = self.workspace.rank_stride
        self.half_bytes = nbytes // 2 // 16 * 16
        # Arm this rank's segment with the sentinel bytes through the driver: the kernel only uses
        # raw addresses, and a DLPack tensor view of fabric memory must never be freed while the
        # process runs (its storage deleter dereferences the capsule of the base tensor).
        _check_cu_result(
            cuda.cuMemsetD8Async(
                cuda.CUdeviceptr(self.ws_addr + mapping.cp_rank * self.rank_stride),
                SENTINEL & 0xFF,
                nbytes,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )
        # Launch counter (ticket) of the kernel.
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.max_send_ctas = _env_int("TRTLLM_HELIX_A2A_FUSED_SEND_CTAS", 2 * num_sms)
        self.max_recv_ctas = _env_int("TRTLLM_HELIX_A2A_FUSED_RECV_CTAS", 2 * num_sms)
        self.use_pdl = get_env_enable_pdl()
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()

    @staticmethod
    def get(mapping: Mapping) -> "HelixAllToAllFused":
        if mapping not in HelixAllToAllFused._cache:
            HelixAllToAllFused._cache[mapping] = HelixAllToAllFused(mapping)
        return HelixAllToAllFused._cache[mapping]

    @staticmethod
    def supports(mapping: Mapping, partial_o: torch.Tensor, num_heads: int, head_dim: int) -> bool:
        """Whether the fused kernel handles this call (else use the FIFO path)."""
        from ..cute_dsl_kernels.blackwell.helix_a2a_fused import supports

        return (
            get_sm_version() in (100, 103)
            and partial_o.dtype == torch.bfloat16
            and supports(mapping.cp_size, num_heads, head_dim)
        )

    def __call__(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
        num_heads: int,
        head_dim: int,
        zero_kv_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exchange and combine.

        Args:
            partial_o: BF16 ``[T, cp_size * num_heads * head_dim]``: this rank's partial output
                of every head, destination-rank major.
            softmax_stats: FP32 ``[T, cp_size * num_heads, 2]`` ``(max, sum)`` per head.
            num_heads: Heads per rank after the exchange.
            head_dim: Values per head.
            zero_kv_mask: Optional bool ``[T]``, True where this rank owns no KV of the token;
                those rows contribute nothing.
            out: Optional BF16 ``[T, num_heads * head_dim]`` output.

        Returns:
            The combined BF16 ``[T, num_heads * head_dim]`` output of this rank's heads.
        """
        from ..cute_dsl_kernels.blackwell.helix_a2a_fused import record_bytes

        cp_size = self.mapping.cp_size
        num_tokens = partial_o.shape[0]
        if out is None:
            out = partial_o.new_empty((num_tokens, num_heads * head_dim))
        partial_o = partial_o.reshape(num_tokens, -1)
        softmax_stats = softmax_stats.reshape(num_tokens, -1)
        assert partial_o.is_contiguous() and softmax_stats.is_contiguous()
        assert partial_o.shape[1] == cp_size * num_heads * head_dim
        assert softmax_stats.shape[1] == cp_size * num_heads * 2
        if zero_kv_mask is not None:
            zero_kv_mask = zero_kv_mask[:num_tokens].contiguous()
        chunk = self.half_bytes // (cp_size * record_bytes(num_heads, head_dim))
        if chunk < 1:
            raise ValueError("TRTLLM_HELIX_A2A_FUSED_MB is too small for one token")
        for start in range(0, num_tokens, chunk):
            end = min(num_tokens, start + chunk)
            torch.ops.trtllm.helix_a2a_fused(
                partial_o[start:end],
                softmax_stats[start:end],
                None if zero_kv_mask is None else zero_kv_mask[start:end],
                out[start:end],
                self.state,
                [
                    self.mapping.cp_rank,
                    cp_size,
                    num_heads,
                    head_dim,
                    self.ws_addr,
                    self.rank_stride,
                    self.half_bytes,
                    self.max_send_ctas,
                    self.max_recv_ctas,
                    int(self.use_pdl),
                ],
            )
        return out


@torch.library.custom_op("trtllm::helix_a2a_fused", mutates_args=("out", "state"))
def _helix_a2a_fused_op(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
    out: torch.Tensor,
    state: torch.Tensor,
    params: List[int],
) -> None:
    """One launch of the fused kernel into ``out``; ``params`` = (cp_rank, cp_size, num_heads,
    head_dim, ws_addr, rank_stride, half_bytes, max_send_ctas, max_recv_ctas, use_pdl)."""
    from ..cute_dsl_kernels.blackwell.helix_a2a_fused import helix_a2a_fused

    cp_rank, cp_size, num_heads, head_dim, ws_addr, rank_stride, half_bytes, n_send, n_recv, pdl = (
        params
    )
    helix_a2a_fused(
        partial_o,
        softmax_stats,
        zero_kv_mask,
        out,
        cp_rank=cp_rank,
        cp_size=cp_size,
        num_heads=num_heads,
        head_dim=head_dim,
        ws_addr=ws_addr,
        rank_stride=rank_stride,
        half_bytes=half_bytes,
        state=state,
        max_send_ctas=n_send,
        max_recv_ctas=n_recv,
        use_pdl=bool(pdl),
    )


@_helix_a2a_fused_op.register_fake
def _(partial_o, softmax_stats, zero_kv_mask, out, state, params) -> None:
    return None
