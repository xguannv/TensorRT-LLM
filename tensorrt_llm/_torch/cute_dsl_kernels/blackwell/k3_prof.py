# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic in-kernel timer marks (prototype, never for trunk): ``TRTLLM_K3LL_PROF_KINDS=a2a,vup`` at process
start compiles the marked variants of the kernels that read it; on this branch the TAIL_COL column publish
(``pub``, k3_ctm_gemv_wide with the publish epilogue) and the Lamport-addend K3AttnResFastKernel (``tar``: the MoE
tail's consumer and the o_proj push's finish, told apart by time).

Each kind owns one int64 device buffer: ``HEADER`` words of per-CTA call counters, then ``SLOTS`` call slots of
``ctas x marks`` %globaltimer words. A marked kernel writes its marks at ``base + 8 * (HEADER + (slot * ctas + cta) *
marks + k)``, ``slot`` = its call number modulo ``SLOTS`` (here a per-CTA counter in the header). The base address
is baked into the compiled kernel, so the buffer is allocated once and never moved.

A daemon thread copies every buffer to ``$TRTLLM_K3LL_PROF_DIR/k3prof_rank<R>.pt`` on its own stream, in relaxed
capture mode, once no Python call of a marked op has run for 15 s (warm-up and capture done; graph replays only);
an atexit hook writes a last copy.
"""

from __future__ import annotations

import atexit
import os
import threading
import time

import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

HEADER = 512
SLOTS = int(os.environ.get("TRTLLM_K3LL_PROF_SLOTS", "1536"))
#: kind -> (max CTAs per launch, marks per CTA)
SHAPES = {"a2a": (320, 9), "vup": (128, 5), "lar": (64, 12), "pub": (64, 9), "tar": (256, 7)}
_st = dict(bufs={}, thread=None, last_call=0.0, lock=threading.Lock(), device=None)


def kinds() -> set:
    return {k for k in os.environ.get("TRTLLM_K3LL_PROF_KINDS", "").split(",") if k}


def addr(kind: str) -> int:
    """The device address of ``kind``'s buffer (allocated on the current device at first use), 0 when off."""
    if kind not in kinds():
        return 0
    with _st["lock"]:
        buf = _st["bufs"].get(kind)
        if buf is None:
            ctas, marks = SHAPES[kind]
            dev = torch.device("cuda", torch.cuda.current_device())
            buf = _st["bufs"][kind] = torch.zeros(
                HEADER + SLOTS * ctas * marks, dtype=torch.int64, device=dev
            )
            _st["device"] = dev
            if _st["thread"] is None:
                _st["thread"] = threading.Thread(target=_dump_loop, daemon=True)
                _st["thread"].start()
                atexit.register(_dump_once)
    return buf.data_ptr()


def touch() -> None:
    """A Python call of a marked op (warm-up or capture): the dump thread waits."""
    _st["last_call"] = time.time()


def _path() -> str:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
    return os.path.join(os.environ.get("TRTLLM_K3LL_PROF_DIR", "/tmp"), f"k3prof_rank{rank}.pt")


def _dump_once(stream=None) -> None:
    if not _st["bufs"]:
        return
    out = {}
    s = stream or torch.cuda.Stream(device=_st["device"])
    with torch.cuda.stream(s):
        for kind, buf in list(_st["bufs"].items()):
            out[kind] = buf.to("cpu", non_blocking=True)
    s.synchronize()
    path = _path()
    torch.save(dict(bufs=out, shapes=dict(SHAPES), header=HEADER, slots=SLOTS, host_ns=time.time_ns()),
               path + ".tmp")  # fmt: skip
    os.replace(path + ".tmp", path)


def _dump_loop() -> None:
    import cuda.bindings.driver as cuda_driver

    torch.cuda.set_device(_st["device"])
    cuda_driver.cuThreadExchangeStreamCaptureMode(
        cuda_driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
    )
    stream = torch.cuda.Stream(device=_st["device"])
    while True:
        time.sleep(2.0)
        if time.time() - _st["last_call"] < 15.0:
            continue
        _dump_once(stream)


# ---------------------------------------------------------------------------------------------------------------------
# Device side: one marking thread per CTA writes its marks straight into the kind's buffer (fire-and-forget stores).
# ---------------------------------------------------------------------------------------------------------------------
def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res, operands, text, constraints, has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def globaltimer(*, loc=None, ip=None):
    """%globaltimer, ordered after the preceding instructions (memory clobber)."""
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l,~{memory}", loc=loc, ip=ip))


@dsl_user_op
def globaltimer_after(value, *, loc=None, ip=None):
    """%globaltimer read once the 32-bit register ``value`` is available (the read is predicated on it)."""
    return Int64(
        _asm(
            T.i64(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0x7fc00001;\nmov.u64 $0, 0;\n"
            "@p mov.u64 $0, %globaltimer;\n}",
            "=&l,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_u64(addr, value, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
            "atom.global.add.u64 $0, [$1], $2;",
            "=l,l,l",
            loc=loc,
            ip=ip,
        )
    )


def slot_base(base: int, kind: str, cta):
    """(device code) The address of CTA ``cta``'s marks of this call in ``kind``'s buffer at ``base``: the CTA's call
    counter (header word ``cta``) taken and advanced, modulo ``SLOTS``."""
    ctas, marks = SHAPES[kind]
    calls = atom_add_u64(Int64(base) + Int64(8) * Int64(cta), Int64(1))
    slot = calls % Int64(SLOTS)
    return Int64(base) + Int64(8) * (
        Int64(HEADER) + (slot * Int64(ctas) + Int64(cta)) * Int64(marks)
    )
