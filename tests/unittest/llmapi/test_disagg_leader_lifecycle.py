# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the disaggregated leader's proxy ownership with model-free processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

_THIS_FILE = Path(__file__).resolve()
_PROCESS_GUARD = _THIS_FILE.parents[3] / "tensorrt_llm" / "llmapi" / "_llmapi_process_guard.py"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def _record_process(directory: Path, role: str) -> None:
    _write_json(directory / f"{role}.pid.json", {"pid": os.getpid(), "pgid": os.getpgrp()})


def _wait_for_file(
    path: Path, timeout: float = 90, process: subprocess.Popen | None = None
) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Owner exited with {process.returncode} before {path.name}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Process did not publish {path.name}")
        time.sleep(0.02)


def _live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        # Darwin has no /proc; ps also distinguishes an exited zombie from a survivor.
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2
        )
        state = result.stdout.strip()
    return bool(state) and not state.startswith(("Z", "X"))


def _records(directory: Path) -> dict[str, dict]:
    return {path.name: json.loads(path.read_text()) for path in directory.glob("*.pid.json")}


def _assert_exited(directory: Path) -> None:
    deadline = time.monotonic() + 15
    while True:
        survivors = {role: row for role, row in _records(directory).items() if _live(row["pid"])}
        if not survivors or time.monotonic() >= deadline:
            _write_json(directory / "survivors.json", survivors)
            assert not survivors, f"Disaggregated leader left live processes: {survivors}"
            return
        time.sleep(0.05)


def _cleanup(leader: subprocess.Popen, directory: Path) -> None:
    # This fallback runs only after the ownership assertions have observed the result.
    groups = {leader.pid} if leader.poll() is None else set()
    groups.update(row["pgid"] for row in _records(directory).values() if _live(row["pid"]))
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # On Darwin, a group that just became zombie-only can report EPERM.
            assert not any(
                row["pgid"] == group and _live(row["pid"]) for row in _records(directory).values()
            )
    leader.wait(timeout=5)


def _run_proxy(directory: Path, scenario: str) -> int:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    subprocess.Popen([sys.executable, str(_THIS_FILE), "descendant", str(directory), scenario])
    _wait_for_file(directory / "descendant.pid.json")
    _record_process(directory, "proxy")
    _write_json(
        directory / "proxy-env.json",
        {
            name: os.environ.get(name)
            for name in (
                "OMPI_H7_TEST_MARKER",
                "TLLM_DISAGG_INSTANCE_IDX",
                "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT",
                "TRTLLM_NO_USAGE_STATS",
            )
        },
    )
    (directory / "proxy-ready").touch()
    if scenario.startswith("proxy_exit_"):
        return int(scenario.rsplit("_", 1)[1])
    time.sleep(180)
    return 99


def _run_leader(serve: ModuleType, directory: Path, scenario: str) -> int:
    from tensorrt_llm.llmapi import mgmn_leader_node

    original_popen = subprocess.Popen
    os.environ["OMPI_H7_TEST_MARKER"] = "remove-before-proxy-exec"
    os.environ["TRTLLM_NO_USAGE_STATS"] = "1"
    serve._child_p_global = None

    def launch_proxy(command: list[str], **kwargs: object) -> subprocess.Popen:
        # Keep the real guard command, environment, session and stream wiring.
        # Only replace the model-serving payload with a CPU-only stand-in.
        index = command.index("disaggregated_mpi_worker")
        proxy_command = command[: index - 2] + [
            sys.executable,
            str(_THIS_FILE),
            "proxy",
            str(directory),
            scenario,
        ]
        child = original_popen(proxy_command, **kwargs)
        _write_json(directory / "owner-child.pid.json", {"pid": child.pid, "pgid": child.pid})
        _write_json(
            directory / "launch.json",
            {
                "command": command,
                "start_new_session": kwargs["start_new_session"],
                "stdout_inherited": kwargs["stdout"] is sys.stdout,
                "stderr_inherited": kwargs["stderr"] is sys.stderr,
            },
        )
        return child

    def run_server(_comm: object) -> None:
        _wait_for_file(directory / "proxy-ready")
        (directory / "server-ready").touch()
        if scenario == "abort":
            os._exit(23)
        if scenario == "exception":
            raise RuntimeError("original disaggregated server failure")
        if scenario in ("sigterm", "sigint"):
            os.kill(os.getpid(), signal.SIGTERM if scenario == "sigterm" else signal.SIGINT)
        if scenario == "owner_sigkill":
            time.sleep(180)
        if scenario.startswith("proxy_exit_"):
            serve._child_p_global.wait(timeout=20)

    serve.subprocess.Popen = launch_proxy
    mgmn_leader_node.launch_server_main = run_server
    code = 0
    error = None
    try:
        serve._launch_disaggregated_leader(
            SimpleNamespace(Get_rank=lambda: 0), 2, "fake.yaml", "info"
        )
    except serve._command_telemetry.SignalExit as caught:
        code = caught.code
        error = str(caught)
    except RuntimeError as caught:
        code = 42
        error = str(caught)
    _write_json(
        directory / "leader-result.json",
        {"code": code, "error": error, "proxy_status": serve._child_p_global.poll()},
    )
    return code


@pytest.mark.parametrize(
    ("scenario", "expected_status"),
    [
        ("abort", 23),
        ("owner_sigkill", -signal.SIGKILL),
        ("exception", 42),
        ("sigterm", 143),
        ("sigint", 130),
        ("return", 0),
        ("proxy_exit_0", 0),
        ("proxy_exit_7", 0),
    ],
)
def test_disaggregated_leader_reclaims_proxy_and_descendant(
    tmp_path: Path, scenario: str, expected_status: int
) -> None:
    """Both returning cleanup and nonreturning owner death reclaim the proxy group."""
    with (tmp_path / "leader.log").open("w") as log:
        leader = subprocess.Popen(
            [sys.executable, str(_THIS_FILE), "leader", str(tmp_path), scenario],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            _wait_for_file(tmp_path / "server-ready", process=leader)
            if scenario == "owner_sigkill":
                leader.kill()
            status = leader.wait(timeout=45)
            assert status == expected_status, (tmp_path / "leader.log").read_text()
            _assert_exited(tmp_path)
            records = _records(tmp_path)
            assert {
                "proxy.pid.json",
                "descendant.pid.json",
                "owner-child.pid.json",
            } <= records.keys()
            assert records["proxy.pid.json"]["pgid"] == records["descendant.pid.json"]["pgid"]
            assert records["proxy.pid.json"]["pid"] != records["owner-child.pid.json"]["pid"]
            launch = json.loads((tmp_path / "launch.json").read_text())
            assert (
                launch["start_new_session"]
                and launch["stdout_inherited"]
                and launch["stderr_inherited"]
            )
            assert "--autonomous" in launch["command"]
            assert json.loads((tmp_path / "proxy-env.json").read_text()) == {
                "OMPI_H7_TEST_MARKER": None,
                "TLLM_DISAGG_INSTANCE_IDX": "2",
                "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT": "1",
                "TRTLLM_NO_USAGE_STATS": "1",
            }
            if scenario == "exception":
                assert json.loads((tmp_path / "leader-result.json").read_text())["error"] == (
                    "original disaggregated server failure"
                )
            if scenario.startswith("proxy_exit_"):
                assert json.loads((tmp_path / "leader-result.json").read_text())[
                    "proxy_status"
                ] == int(scenario.rsplit("_", 1)[1])
        finally:
            _cleanup(leader, tmp_path)


def test_autonomous_guard_does_not_start_payload_for_missing_owner(tmp_path: Path) -> None:
    marker = tmp_path / "payload-started"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(_PROCESS_GUARD),
            "--parent-pid",
            str(os.getpid() + 1),
            "--term-grace",
            "1",
            "--autonomous",
            "--role",
            "proxy",
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(marker),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 143, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("outcome", ["return", "exception", "signal"])
def test_disaggregated_leader_cleanup_timeout_retains_guard(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A delayed guard remains alive to finish cleanup without masking the server error."""
    from tensorrt_llm.commands import serve
    from tensorrt_llm.llmapi import mgmn_leader_node

    child = mock.MagicMock(pid=12345)
    child.poll.return_value = None
    child.wait.side_effect = subprocess.TimeoutExpired("process-guard", 30)
    original_error = (
        serve._command_telemetry.SignalExit(signal.SIGTERM)
        if outcome == "signal"
        else RuntimeError("original server failure")
    )

    def run_server(_comm: object) -> None:
        if outcome != "return":
            raise original_error

    monkeypatch.setattr(serve, "_child_p_global", None)
    # The leader writes these directly; restore their original state at teardown.
    for name in (
        serve.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS.value,
        serve.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR.value,
        serve.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value,
        serve.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.value,
        serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX.value,
    ):
        monkeypatch.setenv(name, "placeholder")
    monkeypatch.setattr(serve, "find_free_ipc_addr", lambda: "ipc://fake-proxy")
    monkeypatch.setattr(serve.subprocess, "Popen", mock.Mock(return_value=child))
    monkeypatch.setattr(mgmn_leader_node, "launch_server_main", run_server)
    if outcome == "return":
        with pytest.raises(RuntimeError, match="cleanup is still pending after 30s"):
            serve._launch_disaggregated_leader(
                SimpleNamespace(Get_rank=lambda: 0), 2, "fake", "info"
            )
    else:
        with pytest.raises(type(original_error)) as caught:
            serve._launch_disaggregated_leader(
                SimpleNamespace(Get_rank=lambda: 0), 2, "fake", "info"
            )
        assert caught.value is original_error
    child.terminate.assert_called_once_with()
    child.wait.assert_called_once_with(timeout=30)
    child.kill.assert_not_called()


@pytest.mark.parametrize(
    ("cuda_device", "visible"),
    [(0, None), (0, "2"), (1, "3,2,1,0"), (2, None), (None, "2,3")],
    ids=["device0", "single-visible", "reordered", "all-visible", "no-device"],
)
def test_disaggregated_leader_hands_its_cuda_device_to_the_proxy(
    monkeypatch: pytest.MonkeyPatch, cuda_device: int | None, visible: str | None
) -> None:
    """The proxy gets the leader's logical device and the same visible-device list.

    Device 0 is passed on like any other device; a leader that selected none passes none on.
    """
    from tensorrt_llm.commands import serve
    from tensorrt_llm.llmapi import mgmn_leader_node

    device_env = serve.DisaggLauncherEnvs.TLLM_DISAGG_PROXY_CUDA_DEVICE.value
    child = mock.MagicMock(pid=12345)
    child.poll.return_value = 0
    popen = mock.Mock(return_value=child)
    monkeypatch.setattr(serve, "_child_p_global", None)
    for name in (
        serve.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS.value,
        serve.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR.value,
        serve.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value,
        serve.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.value,
        serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX.value,
        device_env,
    ):
        monkeypatch.setenv(name, "7")
    monkeypatch.setenv("SLURM_PROCID", "2")
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(serve, "find_free_ipc_addr", lambda: "ipc://fake-proxy")
    monkeypatch.setattr(serve.subprocess, "Popen", popen)
    monkeypatch.setattr(mgmn_leader_node, "launch_server_main", lambda comm: None)

    serve._launch_disaggregated_leader(
        SimpleNamespace(Get_rank=lambda: 0), 2, "fake", "info", cuda_device=cuda_device
    )

    proxy_env = popen.call_args.kwargs["env"]
    # The proxy has no rank variables to derive a device from, and the device is an ordinal of the
    # visible-device list both processes share.
    assert "SLURM_PROCID" not in proxy_env
    assert proxy_env.get("CUDA_VISIBLE_DEVICES") == visible
    if cuda_device is None:
        assert device_env not in proxy_env
    else:
        assert proxy_env[device_env] == str(cuda_device)


@pytest.mark.parametrize(
    ("rank", "visible_count", "expected"),
    [(0, 4, 0), (3, 4, 3), (6, 4, 2), (5, 1, 0)],
    ids=["rank0", "rank3", "second-node-rank", "single-visible"],
)
def test_set_cuda_device_returns_the_logical_device_it_selects(
    monkeypatch: pytest.MonkeyPatch, rank: int, visible_count: int, expected: int
) -> None:
    """The leader hands its proxy exactly the ordinal it selected for itself."""
    from tensorrt_llm.commands import serve

    set_device = mock.Mock()
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", str(rank))
    monkeypatch.setattr(serve, "device_count", lambda: visible_count)
    monkeypatch.setattr(serve.torch.cuda, "set_device", set_device)

    assert serve.set_cuda_device() == expected
    set_device.assert_called_once_with(expected)


def _write_disagg_config(tmp_path: Path) -> Path:
    import yaml

    path = tmp_path / "disagg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "hostname": "localhost",
                "port": 8000,
                "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
                "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
            }
        )
    )
    return path


def _stub_worker_entry(monkeypatch: pytest.MonkeyPatch, serve: ModuleType) -> None:
    monkeypatch.setattr(
        serve._command_telemetry, "apply_disaggregated_telemetry_config", lambda *a, **k: None
    )
    monkeypatch.setattr(serve, "_build_llm_args_from_disagg_server_cfg", lambda args: {})
    monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", "0")


def test_disaggregated_mpi_worker_passes_the_leader_device_to_the_proxy_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A leader rank launches its proxy with the device set_cuda_device() selected."""
    from tensorrt_llm import _utils
    from tensorrt_llm.commands import serve
    from tensorrt_llm.llmapi import disagg_utils

    path = _write_disagg_config(tmp_path)
    _stub_worker_entry(monkeypatch, serve)
    monkeypatch.delenv(
        serve.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.value, raising=False
    )
    monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX.value, "placeholder")
    sub_comm = object()
    launch_leader = mock.Mock()
    monkeypatch.setattr(serve, "set_cuda_device", lambda: 3)
    monkeypatch.setattr(disagg_utils, "split_world_comm", lambda configs: (True, 1, sub_comm))
    monkeypatch.setattr(_utils, "set_mpi_comm", lambda comm: None)
    monkeypatch.setattr(serve, "_launch_disaggregated_leader", launch_leader)

    serve.disaggregated_mpi_worker.callback(
        config_file=str(path), log_level="info", telemetry=False
    )

    launch_leader.assert_called_once_with(sub_comm, 1, str(path), "info", cuda_device=3)


def test_disaggregated_mpi_worker_non_leader_rank_launches_no_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-leader rank still selects its own device and only serves its leader's MPI tasks."""
    import mpi4py.futures

    from tensorrt_llm import _utils
    from tensorrt_llm.commands import serve
    from tensorrt_llm.llmapi import disagg_utils

    path = _write_disagg_config(tmp_path)
    _stub_worker_entry(monkeypatch, serve)
    device_env = serve.DisaggLauncherEnvs.TLLM_DISAGG_PROXY_CUDA_DEVICE.value
    monkeypatch.delenv(
        serve.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.value, raising=False
    )
    monkeypatch.delenv(device_env, raising=False)
    sub_comm = object()
    executor = mock.MagicMock()
    executor.return_value.__enter__.return_value = None
    set_cuda_device = mock.Mock(return_value=1)
    launch_leader = mock.Mock()
    monkeypatch.setattr(serve, "set_cuda_device", set_cuda_device)
    monkeypatch.setattr(disagg_utils, "split_world_comm", lambda configs: (False, 0, sub_comm))
    monkeypatch.setattr(_utils, "set_mpi_comm", lambda comm: None)
    monkeypatch.setattr(mpi4py.futures, "MPICommExecutor", executor)
    monkeypatch.setattr(serve, "_launch_disaggregated_leader", launch_leader)

    serve.disaggregated_mpi_worker.callback(
        config_file=str(path), log_level="info", telemetry=False
    )

    set_cuda_device.assert_called_once_with()
    executor.assert_called_once_with(sub_comm)
    launch_leader.assert_not_called()
    assert device_env not in os.environ


@pytest.mark.parametrize("device", ["0", "3", None])
def test_disaggregated_proxy_selects_the_device_its_leader_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str | None
) -> None:
    """The proxy branch selects the leader's device, not the rank-derived one or device 0."""
    from tensorrt_llm.commands import serve

    path = _write_disagg_config(tmp_path)
    _stub_worker_entry(monkeypatch, serve)
    device_env = serve.DisaggLauncherEnvs.TLLM_DISAGG_PROXY_CUDA_DEVICE.value
    monkeypatch.setenv(
        serve.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.value, "1"
    )
    monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX.value, "1")
    if device is None:
        monkeypatch.delenv(device_env, raising=False)
    else:
        monkeypatch.setenv(device_env, device)
    set_cuda_device = mock.Mock()
    set_device = mock.Mock()
    launch = mock.Mock()
    monkeypatch.setattr(serve, "set_cuda_device", set_cuda_device)
    monkeypatch.setattr(serve.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(serve, "_launch_disaggregated_server", launch)

    serve.disaggregated_mpi_worker.callback(
        config_file=str(path), log_level="info", telemetry=False
    )

    set_cuda_device.assert_not_called()
    if device is None:
        set_device.assert_not_called()
    else:
        set_device.assert_called_once_with(int(device))
    launch.assert_called_once_with(str(path), {})


if __name__ == "__main__":
    mode, directory, scenario = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if mode == "descendant":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _record_process(directory, "descendant")
        time.sleep(180)
    elif mode == "proxy":
        sys.exit(_run_proxy(directory, scenario))
    elif mode == "leader":
        from tensorrt_llm.commands import serve

        sys.exit(_run_leader(serve, directory, scenario))
