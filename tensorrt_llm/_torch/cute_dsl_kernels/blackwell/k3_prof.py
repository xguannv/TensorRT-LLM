# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic in-kernel timer marks (prototype, never for trunk): ``TRTLLM_K3LL_PROF_KINDS=a2a,vup`` at process
start compiles the marked variants of the Helix gated all-to-all (``a2a``) and the fused V-up reduction (``vup``).

Each kind owns one int64 device buffer: ``HEADER`` words of per-CTA call counters, then ``SLOTS`` call slots of
``ctas x marks`` %globaltimer words. A marked kernel writes its marks at ``base + 8 * (HEADER + (slot * ctas + cta) *
marks + k)``, ``slot`` = its call number modulo ``SLOTS`` (the a2a: its launch number from the ticket, the same on
every rank; the reduction: a per-head counter). The base address is baked into the compiled kernel, so the buffer is
allocated once and never moved.

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

HEADER = 512
SLOTS = int(os.environ.get("TRTLLM_K3LL_PROF_SLOTS", "1536"))
#: kind -> (max CTAs per launch, marks per CTA)
SHAPES = {"a2a": (320, 9), "vup": (128, 5)}
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
