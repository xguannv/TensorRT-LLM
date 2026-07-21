# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort GPU memory snapshots and OOM diagnostics.

This module is intentionally independent of pyexecutor.  Callers provide
stage- and owner-specific context, while collectors only read counters that
are safe to query without synchronizing CUDA or allocating GPU memory.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, Optional

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "GpuProcessSnapshot",
    "GpuProcessUsage",
    "MemorySnapshot",
    "MemoryTrace",
    "OomFinding",
    "OomReportResult",
    "ProcessRelation",
    "SnapshotContext",
    "SnapshotDetail",
    "StageRecord",
    "capture_process_snapshot",
    "capture_snapshot",
    "format_snapshot",
    "is_gpu_oom",
    "log_oom_report",
    "log_snapshot",
    "parse_requested_bytes",
]

_SCHEMA_VERSION = 1
_PROFILE_ENV = "TLLM_LOG_MEM_PROFILE"
_OOM_REFRESH_NVML_ENV = "TLLM_MEM_OOM_REFRESH_NVML"
_MAX_ERROR_MESSAGE_LENGTH = 1024
_MAX_PROCESS_RECORDS = 16
_REQUESTED_BYTES_PATTERN = re.compile(
    r"(?:tried to allocate|trying to allocate)\s+"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>bytes?|[kmgt](?:i)?b)",
    re.IGNORECASE,
)
_BYTE_MULTIPLIERS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1000,
    "kib": 1 << 10,
    "mb": 1000**2,
    "mib": 1 << 20,
    "gb": 1000**3,
    "gib": 1 << 30,
    "tb": 1000**4,
    "tib": 1 << 40,
}


class SnapshotDetail(str, Enum):
    """Label the routine and failure-path snapshot contracts in logs."""

    FAST = "fast"
    OOM_SAFE = "oom_safe"


class ProcessRelation(str, Enum):
    """Relationship between a visible GPU process and the current process."""

    SELF = "self"
    NON_SELF = "non_self"


@dataclass(frozen=True)
class CollectorError:
    source: str
    message: str


@dataclass(frozen=True)
class GpuProcessUsage:
    pid: int
    used_bytes: Optional[int]
    relation: ProcessRelation


@dataclass(frozen=True)
class GpuProcessSnapshot:
    captured_at_ns: int
    source_available: bool
    processes: tuple[GpuProcessUsage, ...]
    error: Optional[str] = None


ProcessSnapshotSource = Literal["current", "baseline"]


@dataclass(frozen=True)
class OomFinding:
    """One evidence-backed interpretation of a recognized GPU OOM."""

    code: str
    confidence: str
    action: str
    primary: bool = False
    requested_bytes: Optional[int] = None
    device_free_bytes: Optional[int] = None
    device_total_bytes: Optional[int] = None
    non_self_processes: tuple[GpuProcessUsage, ...] = ()
    process_snapshot_captured_at_ns: Optional[int] = None
    process_snapshot_source: Optional[ProcessSnapshotSource] = None


@dataclass(frozen=True)
class OomReportResult:
    """Structured result shared by OOM logging and user-facing guidance."""

    findings: tuple[OomFinding, ...]

    @property
    def primary_finding(self) -> Optional[OomFinding]:
        for finding in self.findings:
            if finding.primary:
                return finding
        return None


@dataclass(frozen=True)
class SnapshotContext:
    """Caller-owned values needed to identify a snapshot."""

    rank: Optional[int] = None
    device_index: Optional[int] = None
    prefetched_device_free_bytes: Optional[int] = None
    prefetched_device_total_bytes: Optional[int] = None


@dataclass(frozen=True)
class MemorySnapshot:
    schema_version: int
    tag: str
    detail: SnapshotDetail
    capture_phase: str
    timestamp_ns: int
    capture_duration_us: int

    pid: int
    rank: Optional[int]
    device_index: Optional[int]
    device_uuid: Optional[str]

    device_free_bytes: Optional[int]
    device_total_bytes: Optional[int]

    torch_allocated_bytes: Optional[int]
    torch_reserved_bytes: Optional[int]
    torch_allocated_peak_since_reset_bytes: Optional[int]
    torch_reserved_peak_since_reset_bytes: Optional[int]

    cpp_gpu_live_requested_bytes: Optional[int]
    collector_errors: tuple[CollectorError, ...]

    @property
    def device_used_bytes(self) -> Optional[int]:
        if self.device_free_bytes is None or self.device_total_bytes is None:
            return None
        return self.device_total_bytes - self.device_free_bytes

    @property
    def known_owner_estimate_bytes(self) -> int:
        # MemoryCounters is process-wide. TensorRT-LLM normally runs one rank
        # per GPU, so this is a device estimate rather than a strict ledger.
        values = (
            self.torch_reserved_bytes,
            self.cpp_gpu_live_requested_bytes,
        )
        return sum(value for value in values if value is not None)

    @property
    def device_gap_estimate_bytes(self) -> Optional[int]:
        used = self.device_used_bytes
        if used is None:
            return None
        return used - self.known_owner_estimate_bytes


@dataclass(frozen=True)
class StageRecord:
    stage: str
    timestamp_ns: int
    device_free_bytes_pre: Optional[int]
    device_free_bytes_post: Optional[int]
    device_total_bytes: Optional[int]
    process_snapshot: Optional[GpuProcessSnapshot]


class MemoryTrace:
    """Executor-local startup baseline and bounded healthy-stage history."""

    def __init__(self, trace_id: Optional[str] = None, max_entries: int = 16) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.trace_id = trace_id or uuid.uuid4().hex
        self._baseline: Optional[StageRecord] = None
        self._recent_healthy: deque[StageRecord] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def record_baseline(self, entry: StageRecord) -> None:
        with self._lock:
            if self._baseline is not None:
                raise RuntimeError("startup baseline has already been recorded")
            self._baseline = entry

    def record_healthy(self, entry: StageRecord) -> None:
        with self._lock:
            self._recent_healthy.append(entry)

    def diagnostic_history(
        self,
    ) -> tuple[Optional[StageRecord], tuple[StageRecord, ...]]:
        with self._lock:
            return self._baseline, tuple(self._recent_healthy)


def _truncate_error(error: BaseException | str) -> str:
    try:
        message = str(error)
    except Exception:
        message = f"<{type(error).__name__}: message unavailable>"
    return message[:_MAX_ERROR_MESSAGE_LENGTH]


def capture_process_snapshot(
    device: Optional[torch.device | int] = None,
) -> GpuProcessSnapshot:
    """Capture the current GPU process table without raising."""

    captured_at_ns = time.time_ns()
    try:
        from tensorrt_llm import profiler

        result = profiler.device_process_info_status(device)
        current_pid = os.getpid()
        processes = tuple(
            GpuProcessUsage(
                pid=process.pid,
                used_bytes=process.used_bytes,
                relation=(
                    ProcessRelation.SELF if process.pid == current_pid else ProcessRelation.NON_SELF
                ),
            )
            for process in result.processes
        )
        return GpuProcessSnapshot(
            captured_at_ns=captured_at_ns,
            source_available=result.source_available,
            processes=processes,
            error=_truncate_error(result.error) if result.error is not None else None,
        )
    except Exception as error:
        # Diagnostics are called from failure paths; optional NVML plumbing
        # must never replace the original exception.
        return GpuProcessSnapshot(
            captured_at_ns=captured_at_ns,
            source_available=False,
            processes=(),
            error=_truncate_error(error),
        )


def _capture_device_identity(
    context: SnapshotContext,
) -> tuple[Optional[int], Optional[str]]:
    device_index = context.device_index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is not None:
        device_uuid = str(device_uuid)
    return device_index, device_uuid


def _capture_device_memory(
    context: SnapshotContext,
    device_index: Optional[int],
) -> tuple[int, int]:
    if (
        context.prefetched_device_free_bytes is not None
        and context.prefetched_device_total_bytes is not None
    ):
        return (
            context.prefetched_device_free_bytes,
            context.prefetched_device_total_bytes,
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return int(free_bytes), int(total_bytes)


def _capture_torch_counters(device_index: Optional[int]) -> tuple[int, int, int, int]:
    return (
        int(torch.cuda.memory_allocated(device_index)),
        int(torch.cuda.memory_reserved(device_index)),
        int(torch.cuda.max_memory_allocated(device_index)),
        int(torch.cuda.max_memory_reserved(device_index)),
    )


def _capture_cpp_gpu_counter() -> int:
    from tensorrt_llm.bindings import MemoryCounters

    return int(MemoryCounters.instance().gpu)


def capture_snapshot(
    tag: str,
    *,
    detail: SnapshotDetail = SnapshotDetail.FAST,
    capture_phase: str = "normal",
    context: Optional[SnapshotContext] = None,
) -> MemorySnapshot:
    """Read a best-effort memory snapshot.

    Collection is intentionally not gated so tests and OOM paths can invoke it
    directly.  Each source fails independently and records a collector error.
    """

    context = context or SnapshotContext()
    start_ns = time.perf_counter_ns()
    timestamp_ns = time.time_ns()
    errors: list[CollectorError] = []

    device_index: Optional[int] = context.device_index
    device_uuid: Optional[str] = None
    try:
        device_index, device_uuid = _capture_device_identity(context)
    except Exception as error:
        errors.append(CollectorError("device_identity", _truncate_error(error)))

    device_free_bytes: Optional[int] = None
    device_total_bytes: Optional[int] = None
    try:
        device_free_bytes, device_total_bytes = _capture_device_memory(context, device_index)
    except Exception as error:
        errors.append(CollectorError("device_memory", _truncate_error(error)))

    torch_values: tuple[Optional[int], ...] = (None,) * 4
    try:
        torch_values = _capture_torch_counters(device_index)
    except Exception as error:
        errors.append(CollectorError("torch_allocator", _truncate_error(error)))

    cpp_gpu_bytes: Optional[int] = None
    try:
        cpp_gpu_bytes = _capture_cpp_gpu_counter()
    except Exception as error:
        # Binding availability and nanobind failures vary by build.  This
        # optional collector must not block the remaining snapshot.
        errors.append(CollectorError("cpp_memory_counters", _truncate_error(error)))

    return MemorySnapshot(
        schema_version=_SCHEMA_VERSION,
        tag=tag,
        detail=detail,
        capture_phase=capture_phase,
        timestamp_ns=timestamp_ns,
        capture_duration_us=(time.perf_counter_ns() - start_ns) // 1000,
        pid=os.getpid(),
        rank=context.rank,
        device_index=device_index,
        device_uuid=device_uuid,
        device_free_bytes=device_free_bytes,
        device_total_bytes=device_total_bytes,
        torch_allocated_bytes=torch_values[0],
        torch_reserved_bytes=torch_values[1],
        torch_allocated_peak_since_reset_bytes=torch_values[2],
        torch_reserved_peak_since_reset_bytes=torch_values[3],
        cpp_gpu_live_requested_bytes=cpp_gpu_bytes,
        collector_errors=tuple(errors),
    )


def _format_optional(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)


def format_snapshot(snapshot: MemorySnapshot) -> str:
    """Format a snapshot as one append-only, machine-friendly log line."""

    error_sources = ",".join(error.source for error in snapshot.collector_errors)
    return (
        f"[mem-profile/{snapshot.tag}] "
        f"schema={snapshot.schema_version} detail={snapshot.detail.value} "
        f"phase={snapshot.capture_phase} pid={snapshot.pid} "
        f"rank={snapshot.rank} device={snapshot.device_index} "
        f"device_free_bytes={_format_optional(snapshot.device_free_bytes)} "
        f"device_total_bytes={_format_optional(snapshot.device_total_bytes)} "
        f"device_used_bytes={_format_optional(snapshot.device_used_bytes)} "
        f"torch_allocated_bytes={_format_optional(snapshot.torch_allocated_bytes)} "
        f"torch_reserved_bytes={_format_optional(snapshot.torch_reserved_bytes)} "
        f"torch_allocated_peak_bytes="
        f"{_format_optional(snapshot.torch_allocated_peak_since_reset_bytes)} "
        f"torch_reserved_peak_bytes="
        f"{_format_optional(snapshot.torch_reserved_peak_since_reset_bytes)} "
        f"cpp_gpu_bytes={_format_optional(snapshot.cpp_gpu_live_requested_bytes)} "
        f"device_gap_estimate_bytes="
        f"{_format_optional(snapshot.device_gap_estimate_bytes)} "
        f"capture_duration_us={snapshot.capture_duration_us} "
        f"collector_errors={error_sources or 'none'}"
    )


def log_snapshot(
    tag: str,
    *,
    detail: SnapshotDetail = SnapshotDetail.FAST,
    capture_phase: str = "normal",
    context: Optional[SnapshotContext] = None,
) -> None:
    """Log a routine snapshot when ``TLLM_LOG_MEM_PROFILE=1``."""

    if os.environ.get(_PROFILE_ENV, "") != "1":
        return
    try:
        snapshot = capture_snapshot(
            tag,
            detail=detail,
            capture_phase=capture_phase,
            context=context,
        )
        logger.info(format_snapshot(snapshot))
    except Exception:
        # Optional profiling must never block executor creation or serving.
        return


def _iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _matched_gpu_oom(error: BaseException) -> Optional[BaseException]:
    torch_oom_type = getattr(torch, "OutOfMemoryError", None)
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    for candidate in _iter_exception_chain(error):
        if torch_oom_type is not None and isinstance(candidate, torch_oom_type):
            return candidate
        if cuda_oom_type is not None and isinstance(candidate, cuda_oom_type):
            return candidate
        message = _truncate_error(candidate).lower()
        if "cuda out of memory" in message or "cuda error: out of memory" in message:
            return candidate
    return None


def is_gpu_oom(error: BaseException) -> bool:
    """Return whether an exception chain contains explicit CUDA OOM evidence."""

    return _matched_gpu_oom(error) is not None


def parse_requested_bytes(error: BaseException | str) -> Optional[int]:
    """Best-effort parse of the requested allocation from a Torch OOM."""

    match = _REQUESTED_BYTES_PATTERN.search(_truncate_error(error))
    if match is None:
        return None
    multiplier = _BYTE_MULTIPLIERS.get(match.group("unit").lower())
    if multiplier is None:
        return None
    return int(float(match.group("value")) * multiplier)


def _emit_oom_record(report_id: str, event: str, **fields: object) -> None:
    record = {
        "schema": _SCHEMA_VERSION,
        "report_id": report_id,
        "event": event,
        **fields,
    }
    logger.error("[mem-oom] " + json.dumps(record, separators=(",", ":"), sort_keys=True))


def _snapshot_record(snapshot: MemorySnapshot) -> dict[str, object]:
    return {
        "capture_phase": snapshot.capture_phase,
        "pid": snapshot.pid,
        "rank": snapshot.rank,
        "timestamp_ns": snapshot.timestamp_ns,
        "device_index": snapshot.device_index,
        "device_uuid": snapshot.device_uuid,
        "device_free_bytes": snapshot.device_free_bytes,
        "device_total_bytes": snapshot.device_total_bytes,
        "device_used_bytes": snapshot.device_used_bytes,
        "torch_allocated_bytes": snapshot.torch_allocated_bytes,
        "torch_reserved_bytes": snapshot.torch_reserved_bytes,
        "torch_allocated_peak_bytes": snapshot.torch_allocated_peak_since_reset_bytes,
        "torch_reserved_peak_bytes": snapshot.torch_reserved_peak_since_reset_bytes,
        "cpp_gpu_bytes": snapshot.cpp_gpu_live_requested_bytes,
        "device_gap_estimate_bytes": snapshot.device_gap_estimate_bytes,
        "collector_errors": [
            {"source": error.source, "message": error.message}
            for error in snapshot.collector_errors
        ],
    }


def _sort_processes(
    processes: tuple[GpuProcessUsage, ...],
) -> tuple[GpuProcessUsage, ...]:
    return tuple(
        sorted(
            processes,
            key=lambda process: (
                process.used_bytes is not None,
                process.used_bytes if process.used_bytes is not None else -1,
            ),
            reverse=True,
        )
    )


def _build_oom_findings(
    snapshot: Optional[MemorySnapshot],
    requested_bytes: Optional[int],
    process_snapshot: Optional[GpuProcessSnapshot],
    process_snapshot_source: Optional[ProcessSnapshotSource],
) -> tuple[OomFinding, ...]:
    device_free_bytes = snapshot.device_free_bytes if snapshot is not None else None
    device_total_bytes = snapshot.device_total_bytes if snapshot is not None else None

    request_finding: Optional[OomFinding] = None
    if (
        requested_bytes is not None
        and device_free_bytes is not None
        and requested_bytes > device_free_bytes
    ):
        action = "Reduce the failing allocation or free device memory."
        if device_total_bytes is not None and requested_bytes > device_total_bytes:
            action = "Reduce the failing allocation; it exceeds total device capacity."
        request_finding = OomFinding(
            code="REQUEST_EXCEEDS_FREE",
            confidence="high",
            action=action,
            requested_bytes=requested_bytes,
            device_free_bytes=device_free_bytes,
            device_total_bytes=device_total_bytes,
        )

    non_self_finding: Optional[OomFinding] = None
    non_self_processes: tuple[GpuProcessUsage, ...] = ()
    if process_snapshot is not None and process_snapshot.source_available:
        non_self_processes = _sort_processes(
            tuple(
                process
                for process in process_snapshot.processes
                if process.relation is ProcessRelation.NON_SELF
                and (process.used_bytes is None or process.used_bytes > 0)
            )
        )
    if non_self_processes:
        if process_snapshot_source == "baseline":
            confidence = "medium"
            action = (
                "Recheck non-self GPU processes; this evidence comes from startup "
                "and may be stale. Determine their ownership before reclaiming memory."
            )
        elif any(process.used_bytes is not None for process in non_self_processes):
            confidence = "medium"
            action = (
                "Inspect non-self GPU processes and determine their ownership; their "
                "presence alone does not prove that they caused this OOM."
            )
        else:
            confidence = "medium"
            action = (
                "Inspect non-self GPU processes; per-process bytes and ownership are unavailable."
            )
        non_self_finding = OomFinding(
            code="NON_SELF_PROCESS_PRESSURE",
            confidence=confidence,
            action=action,
            non_self_processes=non_self_processes,
            process_snapshot_captured_at_ns=process_snapshot.captured_at_ns,
            process_snapshot_source=process_snapshot_source,
        )

    supporting: list[OomFinding] = []
    if request_finding is not None:
        primary = replace(request_finding, primary=True)
    else:
        primary = OomFinding(
            code="UNKNOWN",
            confidence="low",
            action="Compare the last healthy stage with the current snapshot.",
            primary=True,
        )
    if non_self_finding is not None:
        supporting.append(non_self_finding)

    return (primary, *supporting)


def _finding_record(finding: OomFinding) -> dict[str, object]:
    record: dict[str, object] = {
        "code": finding.code,
        "confidence": finding.confidence,
        "action": finding.action,
        "primary": finding.primary,
    }
    if finding.requested_bytes is not None:
        record["requested_bytes"] = finding.requested_bytes
    if finding.device_free_bytes is not None:
        record["device_free_bytes"] = finding.device_free_bytes
    if finding.device_total_bytes is not None:
        record["device_total_bytes"] = finding.device_total_bytes
    if finding.non_self_processes:
        record["non_self_process_count"] = len(finding.non_self_processes)
        record["non_self_process_used_bytes"] = sum(
            process.used_bytes
            for process in finding.non_self_processes
            if process.used_bytes is not None and process.used_bytes > 0
        )
        record["non_self_process_pids"] = [process.pid for process in finding.non_self_processes]
        record["process_snapshot_captured_at_ns"] = finding.process_snapshot_captured_at_ns
        record["process_snapshot_source"] = finding.process_snapshot_source
    return record


def log_oom_report(
    *,
    stage: str,
    error: BaseException,
    trace: Optional[MemoryTrace] = None,
    capture_phase: str,
    context: Optional[SnapshotContext] = None,
    requested_bytes: Optional[int] = None,
) -> Optional[OomReportResult]:
    """Emit a no-throw, line-oriented report for a recognized GPU OOM."""

    try:
        matched_error = _matched_gpu_oom(error)
    except Exception:
        return None
    if matched_error is None:
        return None

    try:
        report_id = uuid.uuid4().hex
        baseline: Optional[StageRecord] = None
        history: tuple[StageRecord, ...] = ()
        if trace is not None:
            baseline, history = trace.diagnostic_history()

        cached_process_snapshot = baseline.process_snapshot if baseline is not None else None
        context = context or SnapshotContext()
        refresh_processes = os.environ.get(_OOM_REFRESH_NVML_ENV, "") == "1"

        requested_bytes_source = "caller"
        if requested_bytes is None:
            requested_bytes = parse_requested_bytes(matched_error)
            requested_bytes_source = "exception_text" if requested_bytes is not None else "unknown"

        _emit_oom_record(
            report_id,
            "summary",
            stage=stage,
            trace_id=trace.trace_id if trace is not None else None,
            rank=context.rank,
            pid=os.getpid(),
            exception_type=type(error).__name__,
            exception_message=_truncate_error(error),
            requested_bytes=requested_bytes,
            requested_bytes_source=requested_bytes_source,
        )

        snapshot: Optional[MemorySnapshot]
        snapshot_error: Optional[str] = None
        try:
            snapshot = capture_snapshot(
                f"oom/{stage}",
                detail=SnapshotDetail.OOM_SAFE,
                capture_phase=capture_phase,
                context=context,
            )
        except Exception as capture_error:
            snapshot = None
            snapshot_error = _truncate_error(capture_error)

        refreshed_process_snapshot: Optional[GpuProcessSnapshot] = None
        if refresh_processes:
            device_index = snapshot.device_index if snapshot is not None else context.device_index
            refreshed_process_snapshot = capture_process_snapshot(device_index)

        process_snapshot: Optional[GpuProcessSnapshot]
        process_snapshot_source: Optional[ProcessSnapshotSource]
        if refreshed_process_snapshot is not None and refreshed_process_snapshot.source_available:
            process_snapshot = refreshed_process_snapshot
            process_snapshot_source = "current"
        elif cached_process_snapshot is not None:
            process_snapshot = cached_process_snapshot
            process_snapshot_source = "baseline"
        elif refreshed_process_snapshot is not None:
            process_snapshot = refreshed_process_snapshot
            process_snapshot_source = "current"
        else:
            process_snapshot = None
            process_snapshot_source = None

        current_record = (
            _snapshot_record(snapshot)
            if snapshot is not None
            else {
                "capture_phase": capture_phase,
                "pid": os.getpid(),
                "rank": context.rank,
                "timestamp_ns": time.time_ns(),
                "collector_errors": [{"source": "snapshot", "message": snapshot_error}],
            }
        )
        current_record.update(
            {
                "process_snapshot_source": process_snapshot_source,
                "process_source_available": (
                    process_snapshot.source_available if process_snapshot is not None else None
                ),
                "process_error": (process_snapshot.error if process_snapshot is not None else None),
                "process_count": (
                    len(process_snapshot.processes) if process_snapshot is not None else None
                ),
                "process_refresh_error": (
                    refreshed_process_snapshot.error
                    if refreshed_process_snapshot is not None
                    and not refreshed_process_snapshot.source_available
                    else None
                ),
            }
        )
        _emit_oom_record(report_id, "current", **current_record)

        if baseline is not None:
            _emit_oom_record(
                report_id,
                "baseline",
                rank=context.rank,
                stage=baseline.stage,
                timestamp_ns=baseline.timestamp_ns,
                device_free_bytes=baseline.device_free_bytes_post,
                device_total_bytes=baseline.device_total_bytes,
            )

        for entry in history:
            _emit_oom_record(
                report_id,
                "history",
                rank=context.rank,
                stage=entry.stage,
                timestamp_ns=entry.timestamp_ns,
                device_free_bytes_pre=entry.device_free_bytes_pre,
                device_free_bytes_post=entry.device_free_bytes_post,
                device_total_bytes=entry.device_total_bytes,
            )

        if process_snapshot is not None:
            for process in process_snapshot.processes[:_MAX_PROCESS_RECORDS]:
                _emit_oom_record(
                    report_id,
                    "process",
                    rank=context.rank,
                    process_pid=process.pid,
                    used_bytes=process.used_bytes,
                    relation=process.relation,
                    process_snapshot_captured_at_ns=process_snapshot.captured_at_ns,
                    process_snapshot_source=process_snapshot_source,
                )
            if len(process_snapshot.processes) > _MAX_PROCESS_RECORDS:
                _emit_oom_record(
                    report_id,
                    "process_truncated",
                    rank=context.rank,
                    omitted_count=len(process_snapshot.processes) - _MAX_PROCESS_RECORDS,
                )

        findings = _build_oom_findings(
            snapshot,
            requested_bytes,
            process_snapshot,
            process_snapshot_source,
        )
        for finding in findings:
            _emit_oom_record(
                report_id,
                "finding",
                rank=context.rank,
                **_finding_record(finding),
            )
        return OomReportResult(findings=findings)
    except Exception:
        # Never let diagnostics replace the user's original OOM.
        return None
