# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch import mem_probe
from tensorrt_llm._torch.pyexecutor import trace_log_utils


def _mock_cuda(monkeypatch):
    monkeypatch.setattr(mem_probe.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        mem_probe.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid="test-uuid"),
    )
    monkeypatch.setattr(
        mem_probe.torch.cuda,
        "mem_get_info",
        lambda device=None: (700, 1000),
    )
    monkeypatch.setattr(mem_probe.torch.cuda, "memory_allocated", lambda device=None: 100)
    monkeypatch.setattr(mem_probe.torch.cuda, "memory_reserved", lambda device=None: 200)
    monkeypatch.setattr(mem_probe.torch.cuda, "max_memory_allocated", lambda device=None: 150)
    monkeypatch.setattr(mem_probe.torch.cuda, "max_memory_reserved", lambda device=None: 250)
    monkeypatch.setattr(
        mem_probe,
        "_capture_cpp_gpu_counter",
        lambda: 30,
    )


def _stage_record(
    trace: mem_probe.MemoryTrace,
    stage: str,
    *,
    process_snapshot: mem_probe.GpuProcessSnapshot | None = None,
) -> mem_probe.StageRecord:
    return mem_probe.StageRecord(
        stage=stage,
        timestamp_ns=1,
        device_free_bytes_pre=800,
        device_free_bytes_post=700,
        device_total_bytes=1000,
        process_snapshot=process_snapshot,
    )


def test_log_snapshot_gate_off_does_not_capture(monkeypatch):
    monkeypatch.delenv("TLLM_LOG_MEM_PROFILE", raising=False)
    capture = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "capture_snapshot", capture)

    assert mem_probe.log_snapshot("disabled") is None
    capture.assert_not_called()


def test_log_snapshot_failure_never_raises(monkeypatch):
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    monkeypatch.setattr(
        mem_probe,
        "capture_snapshot",
        mock.MagicMock(side_effect=RuntimeError("collector bug")),
    )

    assert mem_probe.log_snapshot("broken") is None


def test_existing_warmup_logger_delegates_to_mem_probe(monkeypatch):
    log = mock.MagicMock()
    monkeypatch.setattr(trace_log_utils, "log_snapshot", log)
    monkeypatch.setattr(trace_log_utils.logger, "rank", 3)

    trace_log_utils.log_mem_snapshot("warmup/test")

    log.assert_called_once_with("warmup/test", context=mem_probe.SnapshotContext(rank=3))


def test_capture_snapshot_maps_fast_sources(monkeypatch):
    _mock_cuda(monkeypatch)

    snapshot = mem_probe.capture_snapshot("test")

    assert snapshot.device_index == 0
    assert snapshot.device_uuid == "test-uuid"
    assert snapshot.device_free_bytes == 700
    assert snapshot.device_total_bytes == 1000
    assert snapshot.device_used_bytes == 300
    assert snapshot.torch_allocated_bytes == 100
    assert snapshot.torch_reserved_bytes == 200
    assert snapshot.torch_allocated_peak_since_reset_bytes == 150
    assert snapshot.cpp_gpu_live_requested_bytes == 30
    assert snapshot.device_gap_estimate_bytes == 70
    assert snapshot.collector_errors == ()
    line = mem_probe.format_snapshot(snapshot)
    assert line.startswith("[mem-profile/test] schema=1")
    assert "device_used_bytes=300" in line
    assert "torch_allocated_peak_bytes=150" in line
    assert "cpp_gpu_bytes=30" in line


def test_prefetched_device_info_avoids_mem_get_info(monkeypatch):
    _mock_cuda(monkeypatch)
    mem_get_info = mock.MagicMock(side_effect=AssertionError("unexpected query"))
    monkeypatch.setattr(mem_probe.torch.cuda, "mem_get_info", mem_get_info)

    snapshot = mem_probe.capture_snapshot(
        "prefetched",
        context=mem_probe.SnapshotContext(
            prefetched_device_free_bytes=600,
            prefetched_device_total_bytes=1000,
        ),
    )

    assert snapshot.device_free_bytes == 600
    assert snapshot.device_total_bytes == 1000
    mem_get_info.assert_not_called()


def test_snapshot_uses_one_device_for_all_cuda_counters(monkeypatch):
    _mock_cuda(monkeypatch)
    counters = {}
    for name, value in (
        ("memory_allocated", 100),
        ("memory_reserved", 200),
        ("max_memory_allocated", 150),
        ("max_memory_reserved", 250),
    ):
        counters[name] = mock.MagicMock(return_value=value)
        monkeypatch.setattr(mem_probe.torch.cuda, name, counters[name])
    mem_get_info = mock.MagicMock(return_value=(700, 1000))
    monkeypatch.setattr(mem_probe.torch.cuda, "mem_get_info", mem_get_info)

    snapshot = mem_probe.capture_snapshot(
        "device", context=mem_probe.SnapshotContext(device_index=2)
    )

    assert snapshot.device_index == 2
    mem_get_info.assert_called_once_with(2)
    for counter in counters.values():
        counter.assert_called_once_with(2)


def test_collector_failure_does_not_hide_other_sources(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(
        mem_probe,
        "_capture_device_identity",
        mock.MagicMock(side_effect=RuntimeError("identity failed")),
    )
    monkeypatch.setattr(
        mem_probe,
        "_capture_cpp_gpu_counter",
        mock.MagicMock(side_effect=ImportError("bindings unavailable")),
    )

    snapshot = mem_probe.capture_snapshot("partial")

    assert snapshot.device_free_bytes == 700
    assert snapshot.torch_reserved_bytes == 200
    assert snapshot.cpp_gpu_live_requested_bytes is None
    assert {error.source for error in snapshot.collector_errors} == {
        "device_identity",
        "cpp_memory_counters",
    }


def test_signed_gap_is_not_clamped(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(mem_probe.torch.cuda, "memory_reserved", lambda device=None: 500)
    monkeypatch.setattr(mem_probe, "_capture_cpp_gpu_counter", lambda: 100)

    snapshot = mem_probe.capture_snapshot("negative-gap")

    assert snapshot.device_used_bytes == 300
    assert snapshot.device_gap_estimate_bytes == -300


def test_process_relation_only_distinguishes_self(monkeypatch):
    from tensorrt_llm import profiler

    monkeypatch.setattr(
        profiler,
        "device_process_info_status",
        lambda device=None: SimpleNamespace(
            source_available=True,
            processes=(
                SimpleNamespace(pid=os.getpid(), used_bytes=100),
                SimpleNamespace(pid=os.getpid() + 1, used_bytes=200),
            ),
            error=None,
        ),
    )

    snapshot = mem_probe.capture_process_snapshot()

    assert [process.relation for process in snapshot.processes] == [
        mem_probe.ProcessRelation.SELF,
        mem_probe.ProcessRelation.NON_SELF,
    ]


def test_memory_trace_keeps_baseline_outside_ring():
    trace = mem_probe.MemoryTrace(trace_id="trace", max_entries=2)
    baseline = _stage_record(trace, "startup/baseline")
    trace.record_baseline(baseline)

    trace.record_healthy(_stage_record(trace, "one"))
    trace.record_healthy(_stage_record(trace, "two"))
    trace.record_healthy(_stage_record(trace, "three"))

    recorded_baseline, history = trace.diagnostic_history()
    assert recorded_baseline is baseline
    assert [entry.stage for entry in history] == ["two", "three"]


def test_memory_trace_rejects_duplicate_baseline():
    trace = mem_probe.MemoryTrace(trace_id="trace")
    trace.record_baseline(_stage_record(trace, "startup/baseline"))

    with pytest.raises(RuntimeError, match="already been recorded"):
        trace.record_baseline(_stage_record(trace, "other-baseline"))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.50 GiB", int(2.5 * (1 << 30))),
        ("trying to allocate 512 MiB", 512 * (1 << 20)),
        ("CUDA out of memory", None),
    ],
)
def test_parse_requested_bytes(message, expected):
    assert mem_probe.parse_requested_bytes(message) == expected


def test_is_gpu_oom_is_conservative():
    assert mem_probe.is_gpu_oom(RuntimeError("CUDA out of memory"))
    assert not mem_probe.is_gpu_oom(RuntimeError("host out of memory"))
    assert not mem_probe.is_gpu_oom(RuntimeError("NCCL unhandled CUDA error"))


def test_exception_with_broken_string_conversion_is_safe():
    class BrokenStringError(RuntimeError):
        def __str__(self):
            raise RuntimeError("cannot format")

    error = BrokenStringError()

    assert not mem_probe.is_gpu_oom(error)
    assert mem_probe.parse_requested_bytes(error) is None


def test_non_oom_report_does_not_capture(monkeypatch):
    capture = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "capture_snapshot", capture)

    assert (
        mem_probe.log_oom_report(
            stage="event_loop",
            error=RuntimeError("ordinary failure"),
            capture_phase="runtime_post_unwind",
        )
        is None
    )
    capture.assert_not_called()


def test_oom_report_is_parseable_and_uses_cached_processes(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.delenv("TLLM_MEM_OOM_REFRESH_NVML", raising=False)
    fake_logger = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "logger", fake_logger)
    process_refresh = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "capture_process_snapshot", process_refresh)

    process_snapshot = mem_probe.GpuProcessSnapshot(
        captured_at_ns=1,
        source_available=True,
        processes=(
            mem_probe.GpuProcessUsage(
                pid=123,
                used_bytes=400,
                relation=mem_probe.ProcessRelation.NON_SELF,
            ),
        ),
    )
    trace = mem_probe.MemoryTrace(trace_id="trace")
    trace.record_baseline(
        _stage_record(trace, "startup/baseline", process_snapshot=process_snapshot)
    )
    trace.record_healthy(_stage_record(trace, "MODEL_ENGINE_MAIN"))

    result = mem_probe.log_oom_report(
        stage="KV_CACHE",
        error=RuntimeError("CUDA out of memory. Tried to allocate 800 bytes"),
        trace=trace,
        capture_phase="startup_error",
        context=mem_probe.SnapshotContext(rank=0),
    )

    records = [
        json.loads(call.args[0].split(" ", 1)[1]) for call in fake_logger.error.call_args_list
    ]
    assert result is not None
    assert len({record["report_id"] for record in records}) == 1
    assert all(record["rank"] == 0 for record in records)
    assert {record["event"] for record in records} >= {
        "summary",
        "current",
        "baseline",
        "history",
        "process",
        "finding",
    }
    findings = [record for record in records if record["event"] == "finding"]
    assert findings[0]["code"] == "REQUEST_EXCEEDS_FREE"
    assert findings[0]["primary"] is True
    assert findings[1]["code"] == "NON_SELF_PROCESS_PRESSURE"
    assert findings[1]["primary"] is False
    assert findings[1]["process_snapshot_source"] == "baseline"
    assert result.primary_finding.code == "REQUEST_EXCEEDS_FREE"
    assert result.findings[1].non_self_processes == process_snapshot.processes
    process_refresh.assert_not_called()
    _, history = trace.diagnostic_history()
    assert [entry.stage for entry in history] == ["MODEL_ENGINE_MAIN"]
    current = next(record for record in records if record["event"] == "current")
    assert current["process_snapshot_source"] == "baseline"
    assert "self_gap_estimate_bytes" not in current


def test_failed_process_refresh_falls_back_to_baseline_as_supporting(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.setenv("TLLM_MEM_OOM_REFRESH_NVML", "1")
    fake_logger = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "logger", fake_logger)
    process_refresh = mock.MagicMock(
        return_value=mem_probe.GpuProcessSnapshot(
            captured_at_ns=2,
            source_available=False,
            processes=(),
            error="NVML unavailable",
        )
    )
    monkeypatch.setattr(mem_probe, "capture_process_snapshot", process_refresh)

    baseline_process_snapshot = mem_probe.GpuProcessSnapshot(
        captured_at_ns=1,
        source_available=True,
        processes=(
            mem_probe.GpuProcessUsage(
                pid=123,
                used_bytes=400,
                relation=mem_probe.ProcessRelation.NON_SELF,
            ),
        ),
    )
    trace = mem_probe.MemoryTrace(trace_id="trace")
    trace.record_baseline(
        _stage_record(
            trace,
            "startup/baseline",
            process_snapshot=baseline_process_snapshot,
        )
    )

    result = mem_probe.log_oom_report(
        stage="MODEL_ENGINE_MAIN",
        error=RuntimeError("CUDA out of memory. Tried to allocate 800 bytes"),
        trace=trace,
        capture_phase="startup_error",
    )

    assert result is not None
    assert result.primary_finding.code == "REQUEST_EXCEEDS_FREE"
    assert result.findings[1].code == "NON_SELF_PROCESS_PRESSURE"
    assert result.findings[1].process_snapshot_source == "baseline"
    process_refresh.assert_called_once_with(0)
    records = [
        json.loads(call.args[0].split(" ", 1)[1]) for call in fake_logger.error.call_args_list
    ]
    current = next(record for record in records if record["event"] == "current")
    assert current["process_snapshot_source"] == "baseline"
    assert current["process_refresh_error"] == "NVML unavailable"


def test_current_non_self_process_remains_supporting(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.setenv("TLLM_MEM_OOM_REFRESH_NVML", "1")
    fake_logger = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "logger", fake_logger)

    process_snapshot = mem_probe.GpuProcessSnapshot(
        captured_at_ns=123,
        source_available=True,
        processes=(
            mem_probe.GpuProcessUsage(
                pid=321,
                used_bytes=400,
                relation=mem_probe.ProcessRelation.NON_SELF,
            ),
        ),
    )
    process_refresh = mock.MagicMock(return_value=process_snapshot)
    monkeypatch.setattr(mem_probe, "capture_process_snapshot", process_refresh)

    result = mem_probe.log_oom_report(
        stage="MODEL_ENGINE_MAIN",
        error=RuntimeError("CUDA out of memory. Tried to allocate 800 bytes"),
        capture_phase="startup_error",
        context=mem_probe.SnapshotContext(rank=0),
    )

    assert result is not None
    assert [finding.code for finding in result.findings] == [
        "REQUEST_EXCEEDS_FREE",
        "NON_SELF_PROCESS_PRESSURE",
    ]
    primary = result.primary_finding
    assert primary.code == "REQUEST_EXCEEDS_FREE"
    assert result.findings[1].process_snapshot_source == "current"
    assert result.findings[1].non_self_processes == process_snapshot.processes
    process_refresh.assert_called_once_with(0)

    records = [
        json.loads(call.args[0].split(" ", 1)[1]) for call in fake_logger.error.call_args_list
    ]
    findings = [record for record in records if record["event"] == "finding"]
    assert findings[0]["primary"] is True
    assert findings[1]["primary"] is False
    assert findings[1]["non_self_process_pids"] == [321]
    assert findings[1]["non_self_process_used_bytes"] == 400
    assert findings[1]["process_snapshot_captured_at_ns"] == 123
    assert findings[1]["process_snapshot_source"] == "current"


@pytest.mark.parametrize(
    (
        "requested_bytes",
        "non_self_bytes",
        "expected_primary",
        "expected_confidence",
        "primary_action_fragment",
    ),
    [
        (1200, 2000, "REQUEST_EXCEEDS_FREE", "medium", "exceeds total device capacity"),
        (800, 50, "REQUEST_EXCEEDS_FREE", "medium", None),
        (800, None, "REQUEST_EXCEEDS_FREE", "medium", None),
        (None, 400, "UNKNOWN", "medium", None),
    ],
)
def test_oom_finding_matrix(
    monkeypatch,
    requested_bytes,
    non_self_bytes,
    expected_primary,
    expected_confidence,
    primary_action_fragment,
):
    _mock_cuda(monkeypatch)
    snapshot = mem_probe.capture_snapshot("oom")
    process_snapshot = mem_probe.GpuProcessSnapshot(
        captured_at_ns=123,
        source_available=True,
        processes=(
            mem_probe.GpuProcessUsage(
                pid=321,
                used_bytes=non_self_bytes,
                relation=mem_probe.ProcessRelation.NON_SELF,
            ),
        ),
    )

    findings = mem_probe._build_oom_findings(snapshot, requested_bytes, process_snapshot, "current")

    assert findings[0].code == expected_primary
    assert findings[0].primary is True
    assert findings[1].code == "NON_SELF_PROCESS_PRESSURE"
    assert findings[1].primary is False
    assert findings[1].confidence == expected_confidence
    assert findings[1].non_self_processes[0].used_bytes == non_self_bytes
    assert findings[1].process_snapshot_source == "current"
    if primary_action_fragment is not None:
        assert primary_action_fragment in findings[0].action


def test_oom_report_failure_never_raises(monkeypatch):
    fake_logger = mock.MagicMock()
    monkeypatch.setattr(mem_probe, "logger", fake_logger)
    monkeypatch.setattr(
        mem_probe,
        "capture_snapshot",
        mock.MagicMock(side_effect=RuntimeError("collector bug")),
    )

    result = mem_probe.log_oom_report(
        stage="event_loop",
        error=RuntimeError("CUDA out of memory"),
        capture_phase="runtime_post_unwind",
    )

    assert result is not None
    assert result.primary_finding.code == "UNKNOWN"
    records = [
        json.loads(call.args[0].split(" ", 1)[1]) for call in fake_logger.error.call_args_list
    ]
    assert {record["event"] for record in records} == {
        "summary",
        "current",
        "finding",
    }
