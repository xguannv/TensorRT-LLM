# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import contextlib
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Tuple, Union

import torch

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None

from tensorrt_llm.logger import logger

if psutil is None:
    logger.warning(
        "A required package 'psutil' is not installed. Will not "
        "monitor the host memory usages. Please install the package "
        "first, e.g, 'pip install psutil'."
    )

if pynvml is None:
    logger.warning(
        "A required package 'pynvml' is not installed. Will not "
        "monitor the device memory usages. Please install the package "
        "first, e.g, 'pip install nvidia-ml-py>=12'."
    )


class Timer:
    def __init__(self):
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag):
        self._start_times[tag] = time.time()

    def stop(self, tag) -> float:
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag) -> float:
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self, tag=None) -> None:
        if tag is None:
            self._start_times.clear()
            self._total_elapsed_times.clear()
        else:
            self._start_times.pop(tag, None)
            self._total_elapsed_times.pop(tag, None)

    def summary(self):
        logger.info("Profile Results")
        for tag, elapsed_time in self._total_elapsed_times.items():
            logger.info(f" - {tag.ljust(30, '.')}: {elapsed_time:.6f} (sec)")


_default_timer = Timer()


def start(tag):
    _default_timer.start(tag)


def stop(tag):
    return _default_timer.stop(tag)


def elapsed_time_in_sec(tag):
    return _default_timer.elapsed_time_in_sec(tag)


def reset(tag=None):
    _default_timer.reset(tag=tag)


def summary():
    _default_timer.summary()


MemUnitType = Literal["GiB", "MiB", "KiB"]


@contextlib.contextmanager
def pynvml_context():
    has_pynvml = pynvml is not None
    if has_pynvml:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            has_pynvml = False

    try:
        yield
    finally:
        if has_pynvml:
            pynvml.nvmlShutdown()


if pynvml is not None:
    with pynvml_context():
        _device_get_memory_info_fn = partial(
            pynvml.nvmlDeviceGetMemoryInfo,
            version=pynvml.nvmlMemory_v2,
        )


def host_memory_info(pid: Optional[int] = None) -> Tuple[int, int, int]:
    if psutil is not None:
        process = psutil.Process(pid)
        # USS reports the amount of memory that would be freed if the process
        # was terminated right now.
        #   https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_full_info
        vmem = psutil.virtual_memory()
        total_mem = vmem.total
        free_mem = vmem.available
        alloc_mem = process.memory_full_info().uss
        return alloc_mem, free_mem, total_mem
    return 0, 0, 0  # used, free, total


def device_memory_info(device: Optional[Union[torch.device, int]] = None) -> Tuple[int, int, int]:
    if pynvml is not None:
        if device is None:
            device = torch.cuda.current_device()
        index = device.index if isinstance(device, torch.device) else device
        with pynvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem_info = _device_get_memory_info_fn(handle)
        return mem_info.used, mem_info.free, mem_info.total
    return 0, 0, 0  # used, free, total


@dataclass(frozen=True)
class DeviceProcessUsage:
    pid: int
    used_bytes: Optional[int]


@dataclass(frozen=True)
class DeviceProcessInfoStatus:
    source_available: bool
    processes: Tuple[DeviceProcessUsage, ...]
    error: Optional[str] = None


def _safe_error_text(error: BaseException) -> str:
    try:
        return str(error)
    except Exception:
        return f"<{type(error).__name__}: message unavailable>"


def device_process_info_status(
    device: Optional[Union[torch.device, int]] = None,
) -> DeviceProcessInfoStatus:
    """Return a status-preserving NVML compute-process query.

    Query failure, an empty process table, unavailable per-process bytes, and
    a known zero-byte value remain distinct.  The function never raises.
    """
    if pynvml is None:
        return DeviceProcessInfoStatus(
            source_available=False,
            processes=(),
            error="pynvml is unavailable",
        )
    try:
        if device is None:
            device = torch.cuda.current_device()
        index = device.index if isinstance(device, torch.device) else device

        device_uuid = None
        try:
            raw_device_uuid = torch.cuda.get_device_properties(index).uuid
            if raw_device_uuid is not None:
                device_uuid = str(raw_device_uuid)
        except (AttributeError, AssertionError, RuntimeError):
            # Older torch may not expose .uuid; fall back to NVML index below.
            pass

        with pynvml_context():
            if device_uuid is not None:
                nvml_uuid = (
                    device_uuid
                    if device_uuid.startswith(("GPU-", "MIG-"))
                    else f"GPU-{device_uuid}"
                )
                handle = pynvml.nvmlDeviceGetHandleByUUID(nvml_uuid.encode())
            elif os.environ.get("CUDA_VISIBLE_DEVICES"):
                return DeviceProcessInfoStatus(
                    source_available=False,
                    processes=(),
                    error=(
                        "CUDA device UUID is unavailable while CUDA_VISIBLE_DEVICES "
                        "is set; refusing an ambiguous NVML index lookup"
                    ),
                )
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            result = []
            for process in procs:
                used = getattr(process, "usedGpuMemory", None)
                # Older pynvml uses a large sentinel when accounting is not
                # available for a visible process.
                if used is not None and used >= (1 << 63):
                    used = None
                result.append(
                    DeviceProcessUsage(
                        pid=int(process.pid),
                        used_bytes=None if used is None else int(used),
                    )
                )
        return DeviceProcessInfoStatus(
            source_available=True,
            processes=tuple(result),
        )
    except Exception as error:
        # This is a diagnostic boundary used from OOM paths.  NVML and CUDA
        # version mismatches can raise several package-specific exception
        # types, none of which may escape to the caller.
        return DeviceProcessInfoStatus(
            source_available=False,
            processes=(),
            error=_safe_error_text(error),
        )


def bytes_to_target_unit(mem_bytes: int, unit: MemUnitType) -> float:
    units = {"GiB": 1 << 30, "MiB": 1 << 20, "KiB": 1 << 10}
    _rename_map = {"GB": "GiB", "MB": "MiB", "KB": "KiB"}
    if unit not in units:
        unit = _rename_map[unit]
    return float(mem_bytes) / units[unit]


def _format(mem_bytes: int, unit: MemUnitType) -> str:
    mem_usage = bytes_to_target_unit(mem_bytes, unit)
    return f"{mem_usage:.4f} ({unit})"


def _print_mem_message(msg: str, tag: Optional[str] = None):
    if tag:
        msg = f"{tag} - {msg}"
    logger.info(f"[MemUsage] {msg}")


def print_host_memory_usage(tag: Optional[str] = None, unit: MemUnitType = "GiB"):
    if psutil is None:
        return
    alloc_mem, _, _ = host_memory_info()
    msg = f"Allocated Host Memory {_format(alloc_mem, unit)}"
    _print_mem_message(msg, tag)


def print_device_memory_usage(
    tag: Optional[str] = None,
    unit: MemUnitType = "GiB",
    device: Optional[Union[torch.device, int]] = None,
):
    alloc_mem, _, _ = device_memory_info(device)
    msg = f"Allocated Device Memory {_format(alloc_mem, unit)}"
    _print_mem_message(msg, tag)


def print_memory_usage(
    tag: Optional[str] = None,
    unit: MemUnitType = "GiB",
    device: Optional[Union[torch.device, int]] = None,
):
    alloc_host_mem, _, _ = host_memory_info()
    alloc_device_mem, _, _ = device_memory_info(device=device)
    msg = (
        f"Allocated Memory: Host {_format(alloc_host_mem, unit)} "
        f"Device {_format(alloc_device_mem, unit)}"
    )
    _print_mem_message(msg, tag)
