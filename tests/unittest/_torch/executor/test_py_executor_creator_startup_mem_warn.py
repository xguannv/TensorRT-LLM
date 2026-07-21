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
"""Unit tests for the startup "GPU not empty" preflight warning in
``_ExecutorMemoryMonitor``.

No GPU is required: the monitor is built via ``__new__`` (bypassing the
CUDA-touching ``__init__``), free/total memory is injected directly, and the
per-PID attribution is mocked. This is how we exercise arbitrary cluster
memory states without ever controlling real GPU memory.
"""

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm import profiler
from tensorrt_llm._torch.mem_probe import (
    GpuProcessSnapshot,
    GpuProcessUsage,
    MemoryTrace,
    OomFinding,
    OomReportResult,
    ProcessRelation,
    StageRecord,
)
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

GIB = 1 << 30


def _make_monitor(total_gib: float):
    """Build a monitor without running __init__ (which reads real GPU memory)."""
    monitor = py_executor_creator._ExecutorMemoryMonitor.__new__(
        py_executor_creator._ExecutorMemoryMonitor
    )
    monitor._rank = 0
    monitor._trace = MemoryTrace(trace_id="test-trace")
    monitor._test_total_gpu_memory_bytes = int(total_gib * GIB)
    monitor._trace.record_baseline(
        StageRecord(
            stage="startup/baseline",
            timestamp_ns=1,
            device_free_bytes_pre=int(total_gib * GIB),
            device_free_bytes_post=int(total_gib * GIB),
            device_total_bytes=int(total_gib * GIB),
            process_snapshot=None,
        )
    )
    return monitor


@pytest.fixture
def fake_logger(monkeypatch):
    fake = mock.MagicMock()
    monkeypatch.setattr(py_executor_creator, "logger", fake)
    return fake


def _mock_processes(monkeypatch, procs):
    """Make process capture return the given ``(pid, used_bytes)`` pairs."""
    _mock_process_usages(
        monkeypatch,
        tuple(
            GpuProcessUsage(
                pid=pid,
                used_bytes=used_bytes,
                relation=(ProcessRelation.SELF if pid == os.getpid() else ProcessRelation.NON_SELF),
            )
            for pid, used_bytes in procs
        ),
    )


def _mock_process_usages(monkeypatch, processes):
    process_snapshot = GpuProcessSnapshot(
        captured_at_ns=1,
        source_available=True,
        processes=tuple(processes),
    )
    monkeypatch.setattr(
        py_executor_creator,
        "capture_process_snapshot",
        lambda *args, **kwargs: process_snapshot,
    )


def _warn(monitor, free_gpu_memory_bytes):
    return monitor._warn_if_gpu_not_empty(
        free_gpu_memory_bytes,
        monitor._test_total_gpu_memory_bytes,
    )


# --- tiering: silent / info / warning -------------------------------------


def test_clean_gpu_stays_silent(monkeypatch, fake_logger):
    # free/total = 0.975 -> above info ratio -> nothing logged.
    _mock_processes(monkeypatch, [])
    _warn(_make_monitor(80), int(78 * GIB))
    fake_logger.warning.assert_not_called()
    fake_logger.info.assert_not_called()


def test_below_absolute_floor_stays_silent(monkeypatch, fake_logger):
    # Small GPU: ratio 0.85 is low, but only 1.5 GiB used (< 2 GiB floor).
    # This is the fixed-overhead false-positive guard.
    _mock_processes(monkeypatch, [])
    _warn(_make_monitor(10), int(8.5 * GIB))
    fake_logger.warning.assert_not_called()
    fake_logger.info.assert_not_called()


def test_info_band_logs_info_only(monkeypatch, fake_logger):
    # ratio 0.8 in [warn=0.75, info=0.9), 16 GiB used, no other process.
    _mock_processes(monkeypatch, [])
    _warn(_make_monitor(80), int(64 * GIB))
    fake_logger.info.assert_called_once()
    fake_logger.warning.assert_not_called()
    assert "not empty at startup" in fake_logger.info.call_args[0][0]


def test_low_free_ratio_logs_warning(monkeypatch, fake_logger):
    # ratio 0.5 < warn ratio -> warning, with the actionable nvidia-smi hint.
    _mock_processes(monkeypatch, [])
    _warn(_make_monitor(80), int(40 * GIB))
    fake_logger.warning.assert_called_once()
    fake_logger.info.assert_not_called()
    assert "nvidia-smi" in fake_logger.warning.call_args[0][0]


def test_zero_total_is_guarded(monkeypatch, fake_logger):
    _mock_processes(monkeypatch, [])
    _warn(_make_monitor(0), 0)
    fake_logger.warning.assert_not_called()
    fake_logger.info.assert_not_called()


# --- per-PID attribution ---------------------------------------------------


def test_non_self_pid_escalates_info_to_warning(monkeypatch, fake_logger):
    # In the info band (ratio 0.8) a named non-self process still escalates to a
    # warning and lists the culprit PID.
    non_self_pid = os.getpid() + 1
    _mock_processes(monkeypatch, [(non_self_pid, int(15 * GIB))])
    _warn(_make_monitor(80), int(64 * GIB))
    fake_logger.warning.assert_called_once()
    fake_logger.info.assert_not_called()
    msg = fake_logger.warning.call_args[0][0]
    assert f"PID {non_self_pid}" in msg
    assert "15.00 GiB" in msg


def test_own_pid_is_filtered_out(monkeypatch, fake_logger):
    # Only our own process shows up -> treated as "no other process" -> info.
    _mock_processes(monkeypatch, [(os.getpid(), int(1 * GIB))])
    _warn(_make_monitor(80), int(64 * GIB))
    fake_logger.info.assert_called_once()
    fake_logger.warning.assert_not_called()
    assert "No non-self compute process" in fake_logger.info.call_args[0][0]


def test_multiple_pids_sorted_desc(monkeypatch, fake_logger):
    p1, p2 = os.getpid() + 1, os.getpid() + 2
    _mock_processes(monkeypatch, [(p1, int(3 * GIB)), (p2, int(20 * GIB))])
    _warn(_make_monitor(80), int(40 * GIB))
    msg = fake_logger.warning.call_args[0][0]
    # Larger consumer listed first.
    assert msg.index(f"PID {p2}") < msg.index(f"PID {p1}")


# --- env-var overrides -----------------------------------------------------


def test_env_override_tightens_warn_ratio(monkeypatch, fake_logger):
    # Raise the warn ratio so an otherwise-info state (0.8) becomes a warning.
    _mock_processes(monkeypatch, [])
    monkeypatch.setenv("TLLM_STARTUP_FREE_MEM_WARN_RATIO", "0.95")
    _warn(_make_monitor(80), int(64 * GIB))
    fake_logger.warning.assert_called_once()


def test_env_override_relaxes_info_ratio(monkeypatch, fake_logger):
    # Lower the info ratio below the actual ratio so nothing is logged.
    _mock_processes(monkeypatch, [])
    monkeypatch.setenv("TLLM_STARTUP_FREE_MEM_INFO_RATIO", "0.5")
    _warn(_make_monitor(80), int(64 * GIB))  # ratio 0.8 >= 0.5
    fake_logger.warning.assert_not_called()
    fake_logger.info.assert_not_called()


def test_warn_ratio_above_info_ratio_still_warns(monkeypatch, fake_logger):
    # Regression: warn_ratio > info_ratio. A free ratio in (info, warn) must
    # still warn, not be silenced by the logging gate (max(info, warn)).
    _mock_processes(monkeypatch, [])
    monkeypatch.setenv("TLLM_STARTUP_FREE_MEM_WARN_RATIO", "0.95")  # info=0.9
    _warn(_make_monitor(80), int(0.92 * 80 * GIB))  # ratio 0.92
    fake_logger.warning.assert_called_once()
    assert "not empty at startup" in fake_logger.warning.call_args[0][0]


@pytest.mark.parametrize("invalid_value", ["not-a-number", "nan", "inf", "-1", "2"])
def test_invalid_env_falls_back_without_raising(monkeypatch, fake_logger, invalid_value):
    _mock_processes(monkeypatch, [])
    monkeypatch.setenv("TLLM_STARTUP_FREE_MEM_WARN_RATIO", invalid_value)
    _warn(_make_monitor(80), int(40 * GIB))  # ratio 0.5 < 0.75
    assert any("not empty at startup" in c.args[0] for c in fake_logger.warning.call_args_list)


def test_extreme_used_floor_suppresses_warning_without_overflow(monkeypatch, fake_logger):
    _mock_processes(monkeypatch, [])
    monkeypatch.setenv("TLLM_STARTUP_FREE_MEM_MIN_GIB", "1e308")

    _warn(_make_monitor(80), int(40 * GIB))

    fake_logger.warning.assert_not_called()
    fake_logger.info.assert_not_called()


# --- profiler.device_process_info graceful degradation ---------------------


def test_device_process_info_without_pynvml(monkeypatch):
    monkeypatch.setattr(profiler, "pynvml", None)
    status = profiler.device_process_info_status()
    assert status.source_available is False
    assert status.processes == ()


def test_device_process_info_cuda_unavailable(monkeypatch):
    # pynvml present but CUDA unavailable: current_device() raises -> [] (the
    # whole device resolve is inside the try, so it never propagates).
    if profiler.pynvml is None:
        pytest.skip("pynvml not available in this environment")
    monkeypatch.setattr(
        profiler.torch.cuda, "current_device", mock.MagicMock(side_effect=RuntimeError("no CUDA"))
    )
    status = profiler.device_process_info_status()
    assert status.source_available is False
    assert status.processes == ()


def test_device_process_info_handles_unprintable_error(monkeypatch):
    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("broken __str__")

    monkeypatch.setattr(profiler, "pynvml", object())
    monkeypatch.setattr(
        profiler.torch.cuda,
        "current_device",
        mock.MagicMock(side_effect=UnprintableError()),
    )

    status = profiler.device_process_info_status()

    assert status.source_available is False
    assert status.processes == ()
    assert status.error == "<UnprintableError: message unavailable>"


def test_device_process_info_preserves_prefixed_uuid(monkeypatch):
    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByUUID=mock.MagicMock(return_value="handle"),
        nvmlDeviceGetComputeRunningProcesses=mock.MagicMock(return_value=[]),
    )
    monkeypatch.setattr(profiler, "pynvml", fake_nvml)
    monkeypatch.setattr(profiler, "pynvml_context", lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        profiler.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid="GPU-test-uuid"),
    )

    status = profiler.device_process_info_status(0)

    assert status.source_available is True
    fake_nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(b"GPU-test-uuid")


@pytest.mark.parametrize("properties", [SimpleNamespace(), SimpleNamespace(uuid=None)])
def test_device_process_info_rejects_ambiguous_masked_index(monkeypatch, properties):
    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByIndex=mock.MagicMock(),
        nvmlDeviceGetComputeRunningProcesses=mock.MagicMock(return_value=[]),
    )
    monkeypatch.setattr(profiler, "pynvml", fake_nvml)
    monkeypatch.setattr(profiler, "pynvml_context", lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        profiler.torch.cuda,
        "get_device_properties",
        lambda device: properties,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    status = profiler.device_process_info_status(0)

    assert status.source_available is False
    assert "ambiguous NVML index" in status.error
    fake_nvml.nvmlDeviceGetHandleByIndex.assert_not_called()


def test_successful_creation_stage_records_only_memory_trace(monkeypatch):
    monitor = _make_monitor(80)
    monkeypatch.setattr(
        py_executor_creator.torch.cuda,
        "mem_get_info",
        mock.MagicMock(side_effect=[(70 * GIB, 80 * GIB), (60 * GIB, 80 * GIB)]),
    )
    monkeypatch.setattr(py_executor_creator, "log_snapshot", lambda *args, **kwargs: None)

    with monitor.observe_creation_stage(ExecutorMemoryType.SAMPLER):
        pass

    _, records = monitor.trace.diagnostic_history()
    assert len(records) == 1
    assert records[0].stage == ExecutorMemoryType.SAMPLER.value
    assert records[0].device_free_bytes_pre == 70 * GIB
    assert records[0].device_free_bytes_post == 60 * GIB


def test_creation_oom_reports_without_overwriting_healthy_history(monkeypatch):
    monitor = _make_monitor(80)
    monkeypatch.setattr(
        py_executor_creator.torch.cuda,
        "mem_get_info",
        mock.MagicMock(
            side_effect=[
                (70 * GIB, 80 * GIB),
                (60 * GIB, 80 * GIB),
                (10 * GIB, 80 * GIB),
            ]
        ),
    )
    monkeypatch.setattr(py_executor_creator, "log_snapshot", lambda *args, **kwargs: None)
    report = mock.MagicMock(return_value=None)
    monkeypatch.setattr(py_executor_creator, "log_oom_report", report)
    original = torch.OutOfMemoryError("CUDA out of memory")

    with monitor.observe_creation_stage(ExecutorMemoryType.SAMPLER):
        pass
    _, healthy_history = monitor.trace.diagnostic_history()

    with pytest.raises(RuntimeError) as raised:
        with monitor.observe_creation_stage(ExecutorMemoryType.MODEL_ENGINE_MAIN):
            raise original

    assert raised.value.__cause__ is original
    assert "sampler: 70.00 / 60.00" in str(raised.value)
    _, history_after_oom = monitor.trace.diagnostic_history()
    assert history_after_oom == healthy_history
    report.assert_called_once()


def test_creation_oom_keeps_tuning_advice_with_non_self_process_evidence():
    monitor = _make_monitor(80)
    non_self_process = GpuProcessUsage(
        pid=123,
        used_bytes=40 * GIB,
        relation=ProcessRelation.NON_SELF,
    )
    diagnostic = OomReportResult(
        findings=(
            OomFinding(
                code="REQUEST_EXCEEDS_FREE",
                confidence="high",
                action="Reduce the failing allocation or free device memory.",
                primary=True,
                requested_bytes=2 * GIB,
                device_free_bytes=1 * GIB,
                device_total_bytes=80 * GIB,
            ),
            OomFinding(
                code="NON_SELF_PROCESS_PRESSURE",
                confidence="medium",
                action="Inspect non-self GPU processes and determine their ownership.",
                non_self_processes=(non_self_process,),
                process_snapshot_captured_at_ns=1,
                process_snapshot_source="baseline",
            ),
        ),
    )

    explanation = monitor._maybe_explain_if_oom(
        torch.OutOfMemoryError("CUDA out of memory"),
        current_stage=ExecutorMemoryType.MODEL_ENGINE_MAIN,
        free_gpu_memory_bytes_pre=1 * GIB,
        diagnostic=diagnostic,
    )

    assert explanation is not None
    assert "Supporting non-self GPU-process evidence (baseline)" in explanation
    assert "PID 123 (40.00 GiB)" in explanation
    assert "requested=2.00, current_free=1.00, shortfall=1.00" in explanation
    assert "reduce max_num_tokens" in explanation


def test_host_oom_does_not_get_gpu_tuning_guidance():
    monitor = _make_monitor(80)

    explanation = monitor._maybe_explain_if_oom(
        RuntimeError("host out of memory"),
        current_stage=ExecutorMemoryType.MODEL_ENGINE_MAIN,
        free_gpu_memory_bytes_pre=1 * GIB,
    )

    assert explanation is None
