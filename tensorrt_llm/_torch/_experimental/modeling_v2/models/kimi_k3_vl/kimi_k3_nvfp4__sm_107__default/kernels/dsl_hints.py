# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instruction-scheduling hints of the internal CuTe DSL build, for the K3 Rubin kernels.

The internal ``nvidia-cutlass-dsl`` build provides four scheduling hints under ``cute.nvgpu``; the
public package provides none of them. Kernel code calls the wrappers below while it is traced.
When the installed DSL provides all four hints (``HAS_SCHEDULING_HINTS``, fixed at import), each
wrapper emits its hint at the call site; otherwise it emits nothing:

- ``cfence()`` emits ``cute.nvgpu.cfence()``, a compiler code-motion fence.
- ``warp_switch()`` emits ``cute.nvgpu.warp_switch()``, a warp-switch hint.
- ``sched_res_busy_xu64()`` emits ``cute.nvgpu.sched_res_busy_xu64()`` and
  ``reset_sched_res_busy_xu64()`` emits ``cute.nvgpu.reset_sched_res_busy_xu64()``, the MUFU (XU)
  resource hints.

The hints only shape the SASS schedule. The orderings the kernels rely on come from mbarriers,
named barriers, tcgen05 / proxy fences and ``.sync`` instructions, so a kernel computes the same
result with or without them.
"""

import cutlass.cute as cute

_CFENCE = getattr(cute.nvgpu, "cfence", None)
_WARP_SWITCH = getattr(cute.nvgpu, "warp_switch", None)
_SCHED_RES_BUSY_XU64 = getattr(cute.nvgpu, "sched_res_busy_xu64", None)
_RESET_SCHED_RES_BUSY_XU64 = getattr(cute.nvgpu, "reset_sched_res_busy_xu64", None)
HAS_SCHEDULING_HINTS = None not in (
    _CFENCE,
    _WARP_SWITCH,
    _SCHED_RES_BUSY_XU64,
    _RESET_SCHED_RES_BUSY_XU64,
)


def cfence() -> None:
    """Emit ``cute.nvgpu.cfence()`` with the internal build's hints (traced in place)."""
    if HAS_SCHEDULING_HINTS:
        _CFENCE()


def warp_switch() -> None:
    """Emit ``cute.nvgpu.warp_switch()`` with the internal build's hints (traced in place)."""
    if HAS_SCHEDULING_HINTS:
        _WARP_SWITCH()


def sched_res_busy_xu64() -> None:
    """Emit ``cute.nvgpu.sched_res_busy_xu64()`` with the internal build's hints."""
    if HAS_SCHEDULING_HINTS:
        _SCHED_RES_BUSY_XU64()


def reset_sched_res_busy_xu64() -> None:
    """Emit ``cute.nvgpu.reset_sched_res_busy_xu64()`` with the internal build's hints."""
    if HAS_SCHEDULING_HINTS:
        _RESET_SCHED_RES_BUSY_XU64()
