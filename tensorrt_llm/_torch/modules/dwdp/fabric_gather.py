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

"""Host-side access to the DWDP fabric-gather kernel (``trtllm::dwdp_fabric_gather``).

Opt-in via ``TRTLLM_K3LL_DWDP_GATHER=1`` (default 0): an SM kernel out-runs the copy engine but occupies
SMs for the whole step, so the compute kernels are told to leave those SMs free
(``apply_compute_side_defaults``).
``TRTLLM_DWDP_GATHER_LIB`` loads the op from a prebuilt library; ``TRTLLM_DWDP_GATHER_BLOCKS`` and
``TRTLLM_DWDP_GATHER_THREADS`` size the grid; the launcher reads ``TRTLLM_DWDP_GATHER_CLUSTER`` and
``TRTLLM_DWDP_GATHER_KERNEL`` (``tma`` unless set).  ``TRTLLM_DWDP_GATHER_ROWS`` (default 8) re-cuts
a layer's plan into that many equal-byte grid rows instead of one row per peer, so the grid can be
smaller than two blocks per peer (``TRTLLM_DWDP_GATHER_BLOCKS`` = rows x blocks per row; the SM
reserves follow it).  See ``compute_side_defaults`` for the compute-side knobs.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import torch

try:
    from tensorrt_llm.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

_ENABLE_ENV = "TRTLLM_K3LL_DWDP_GATHER"
_LIB_ENV = "TRTLLM_DWDP_GATHER_LIB"
_BLOCKS_ENV = "TRTLLM_DWDP_GATHER_BLOCKS"
_THREADS_ENV = "TRTLLM_DWDP_GATHER_THREADS"
_ROWS_ENV = "TRTLLM_DWDP_GATHER_ROWS"

#: Row cut granularity: the TMA kernel moves a descriptor in 16 KiB chunks.
ROW_CUT_BYTES = 16384

#: Load width of the kernel; every descriptor pointer and byte count must be a multiple of it.
LINE_BYTES = 16

#: Block size the op accepts: a positive multiple of 32 up to 1024 (the kernel's launch bound).
MAX_THREADS = 1024


def _int_env(name: str, default: int) -> int:
    """Integer env knob; an unparsable value logs a warning and keeps ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[DWDP] {name}={raw!r} is not an integer; using {default}")
        return default


def _threads() -> int:
    threads = _int_env(_THREADS_ENV, MAX_THREADS)
    if threads <= 0 or threads % 32 or threads > MAX_THREADS:
        logger.warning(
            f"[DWDP] {_THREADS_ENV}={threads} must be a positive multiple of 32 up to "
            f"{MAX_THREADS}; using {MAX_THREADS}"
        )
        return MAX_THREADS
    return threads


#: Threads per block (``TRTLLM_DWDP_GATHER_THREADS`` overrides).  Bytes in flight scale with
#: blocks x threads, so 1024 keeps the fabric saturated from as few SMs as possible.
THREADS = _threads()

#: Total blocks, split evenly across the grid rows of one layer.  With the default 8 rows this is 8 x 2
#: blocks (one 2-CTA cluster per row): GB300, 16.8 GB pulled from the 3 other GPUs of a node, 8 x 2
#: 779 GB/s vs 15 x 2 792 GB/s (a465ed5296), so 16 SMs instead of 30.
TARGET_BLOCKS = 16

#: Grid rows a layer's plan is re-cut into when ``TRTLLM_DWDP_GATHER_ROWS`` is unset.
TARGET_ROWS = 8

_op: Optional[Callable] = None
_probed = False


def is_enabled() -> bool:
    """True if the env knob asks for the gather transport."""
    return os.environ.get(_ENABLE_ENV, "0") not in ("", "0", "false", "False")


def total_blocks() -> int:
    """Total gather blocks, from ``TRTLLM_DWDP_GATHER_BLOCKS`` or ``TARGET_BLOCKS``."""
    return max(1, _int_env(_BLOCKS_ENV, TARGET_BLOCKS))


def blocks_per_row(rows: int) -> int:
    """Blocks to give each group so the whole grid lands near ``TARGET_BLOCKS``."""
    return max(1, total_blocks() // max(rows, 1))


def compute_side_defaults() -> List[Tuple[str, str]]:
    """The knobs a resident gather needs on the compute side, as ``(name, default)``.

    - ``TRTLLM_K3LL_CUTEDSL_SM_RESERVE`` / ``TRTLLM_K3LL_DEEPGEMM_SM_RESERVE``: keep the gather's
      block count of SMs out of the CuTeDSL and DeepGEMM persistent grids, whose CTAs would
      otherwise land on the occupied SMs and run their static share as a second wave.
    - ``TRTLLM_K3LL_KDA_PREFILL_SM_RESERVE=1``: the KDA prefill grids keep the same reserve.
    - ``TRTLLM_K3LL_ROUTING_DISABLE_COOP_LAUNCH=1``: the grid-cooperative routing launch waits
      until its whole grid fits, i.e. behind the gather; route without it.
    - ``TRTLLM_DWDP_GATHER_KERNEL=tma``: the shared-memory-staged bulk-copy kernel.
    """
    reserve = str(total_blocks())
    return [
        ("TRTLLM_K3LL_CUTEDSL_SM_RESERVE", reserve),
        ("TRTLLM_K3LL_DEEPGEMM_SM_RESERVE", reserve),
        ("TRTLLM_K3LL_KDA_PREFILL_SM_RESERVE", "1"),
        ("TRTLLM_K3LL_ROUTING_DISABLE_COOP_LAUNCH", "1"),
        ("TRTLLM_DWDP_GATHER_KERNEL", "tma"),
    ]


def apply_compute_side_defaults() -> None:
    """Set ``compute_side_defaults`` in the environment; call before any kernel compiles or launches.

    These knobs are read only from here (or from an explicit setting, which wins), so a process
    without the gather never reserves SMs.  The C++ readers cache their value at first use.
    """
    applied = [f"{k}={v}" for k, v in compute_side_defaults() if os.environ.setdefault(k, v) == v]
    if applied:
        logger.info(f"[DWDP] SM gather on: {' '.join(applied)} (set an explicit value to override)")


def gather_op() -> Optional[Callable]:
    """Resolve the op once, honoring ``TRTLLM_DWDP_GATHER_LIB``; None if unavailable."""
    global _op, _probed
    if _probed:
        return _op
    _probed = True

    lib = os.environ.get(_LIB_ENV)
    if lib:
        try:
            torch.ops.load_library(lib)
        except OSError as exc:
            logger.warning(f"[DWDP] {_LIB_ENV}={lib!r} could not be loaded: {exc}")

    try:
        _op = torch.ops.trtllm.dwdp_fabric_gather
    except (AttributeError, RuntimeError):
        _op = None
    return _op


def regroup_rows(
    groups: List[List[Tuple[int, int, int]]], rows: int
) -> List[List[Tuple[int, int, int]]]:
    """Re-cut descriptor groups into ``rows`` groups of (nearly) equal bytes, keeping the order.

    Descriptors are split at ``ROW_CUT_BYTES`` multiples, so every piece keeps the 16-byte
    alignment of its parent.  Returns ``groups`` unchanged if ``rows`` is not smaller.
    """
    if rows <= 0 or rows >= len(groups):
        return groups
    flat = [d for group in groups for d in group]
    total = sum(size for _, _, size in flat)
    if total == 0:
        return groups
    per_row = -(-total // rows)
    per_row = -(-per_row // ROW_CUT_BYTES) * ROW_CUT_BYTES
    out: List[List[Tuple[int, int, int]]] = [[]]
    room = per_row
    for dst, src, size in flat:
        offset = 0
        while offset < size:
            if room == 0:
                out.append([])
                room = per_row
            take = min(size - offset, room)
            out[-1].append((dst + offset, src + offset, take))
            offset += take
            room -= take
    return [group for group in out if group]


def build_plan(
    groups: List[List[Tuple[int, int, int]]], device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Turn per-group ``(dst_ptr, src_ptr, size_bytes)`` descriptors into launch arguments.

    Each group becomes one grid row walked by the same blocks.  Returns None (caller keeps the copy
    engine) if any descriptor is not ``LINE_BYTES``-aligned; the kernel has no unaligned path.

    Returns:
        ``(dst_ptrs, src_ptrs, lines, row_starts, blocks_per_row)`` with device-resident int64
        tensors, or None if the plan cannot be gathered.
    """
    dst_ptrs: List[int] = []
    src_ptrs: List[int] = []
    lines: List[int] = []
    row_starts: List[int] = [0]

    groups = regroup_rows(groups, _int_env(_ROWS_ENV, TARGET_ROWS))
    for descriptors in groups:
        for dst_ptr, src_ptr, size in descriptors:
            if dst_ptr % LINE_BYTES or src_ptr % LINE_BYTES or size % LINE_BYTES:
                return None
            dst_ptrs.append(dst_ptr)
            src_ptrs.append(src_ptr)
            lines.append(size // LINE_BYTES)
        row_starts.append(len(dst_ptrs))

    if not dst_ptrs:
        return None

    def as_tensor(values: List[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    rows = len(row_starts) - 1
    return (
        as_tensor(dst_ptrs),
        as_tensor(src_ptrs),
        as_tensor(lines),
        as_tensor(row_starts),
        blocks_per_row(rows),
    )
