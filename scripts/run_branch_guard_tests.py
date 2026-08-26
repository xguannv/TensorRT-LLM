#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Submit the DSV4 and Kimi K3 branch guard tests through Slurm.

The runner deliberately keeps the Slurm allocation and container launch in this
file so that the same command can be reviewed, reproduced, and run from a
checked-out branch.

Examples:
    # Editable repository guard (explicit; also the default).
    python3 scripts/run_branch_guard_tests.py \
        --scope all \
        --install-from-repo \
        --image /path/to/trtllm.sqsh \
        --partition batch-xdr \
        --account coreai_comparch_aarwlt \
        --model-root /path/to/llm-models

    # Image-only guard.
    python3 scripts/run_branch_guard_tests.py \
        --scope all \
        --skip-install \
        --image /path/to/trtllm.sqsh \
        --partition batch-xdr \
        --account coreai_comparch_aarwlt \
        --model-root /path/to/llm-models

The guards install this checkout into the container in editable mode by
default. Use ``--skip-install`` to exercise the image as-is.

Set TRTLLM_TEST_IMAGE, TRTLLM_SLURM_PARTITION, TRTLLM_SLURM_ACCOUNT, and
LLM_MODELS_ROOT to avoid repeating those options.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

_PYTEST_SUMMARY_RE = re.compile(
    r"(?P<summary>\d+\s+(?:passed|failed|skipped|xfailed|xpassed|error|errors|warnings|deselected)"
    r"(?:,\s*\d+\s+\w+)*(?:\s+in\s+\S+(?:\s+\([^)]*\))?)?)"
)
_MONITOR_INTERVAL_SECONDS = 5
_TERMINAL_SLURM_STATES = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
}
_EARLY_FAILURE_RE = re.compile(
    r"prte has detected an attempt to run as root|MPI_ERR_[A-Z_]+|OPAL ERROR|"
    r"Traceback \(most recent call last\)|CUDA error|CUDA out of memory|"
    r"OutOfMemoryError|NCCL (?:WARN|ERROR)|AssertionError|RuntimeError:|"
    r"Connection refused|Address already in use|"
    r"(?:ctx|context|gen|generation|disagg|disaggregated)[^\n]{0,120}"
    r"(?:failed|failure|error|exception|not ready|unavailable)|"
    r"(?:failed|failure|error|exception|not ready|unavailable)[^\n]{0,120}"
    r"(?:ctx|context|gen|generation|disagg|disaggregated)",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"\b(?:ctx|context|gen|generation|disagg|disaggregated)\b",
    re.IGNORECASE,
)


def _is_trtllm_repo_root(path: Path) -> bool:
    return all(
        (path / marker).exists()
        for marker in (
            "tensorrt_llm",
            "tests",
            "pyproject.toml",
        )
    )


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _detect_trtllm_repo_root() -> Path:
    configured_root = os.environ.get("TRTLLM_REPO_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not _is_trtllm_repo_root(root):
            raise RuntimeError(f"TRTLLM_REPO_ROOT is not a TensorRT-LLM repository: {root}")
        return root

    if _is_trtllm_repo_root(SCRIPT_REPO_ROOT):
        return SCRIPT_REPO_ROOT

    raise RuntimeError(
        "Unable to detect a TensorRT-LLM repository two levels above the script; "
        "pass --repo-root or set TRTLLM_REPO_ROOT."
    )


REPO_ROOT = _detect_trtllm_repo_root()

SINGLE_NODE_SELECTORS = (
    "tests/integration/defs/accuracy/test_llm_api_pytorch.py"
    "::TestDeepSeekV4ProDSpark::test_gsm8k_dep4_megamoe_cutedsl",
)

K3_AGGREGATE_SELECTOR = (
    "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestKimiK3::test_gsm8k_full_accuracy"
)
K3_CHECKPOINT_PATH = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/nvidia_kimi-k3-nvfp4/hf/hf-1ec10fa_orig"
)
DSV4_DSPARK_CHECKPOINT_PATH = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/"
    "nvidia_deepseek-v4-pro-nvfp4-dspark/hf/hf-318bf60_orig"
)
BRANCH_GUARD_MODEL_ROOT = "/lustre/share/coreai_comparch_trtllm/lizhiz/branch_guard_models"
K3_DISAGGREGATED_ACCURACY_CONFIG = "examples/kimi_k3/disagg/benchmark_kimi_k3_dep8_gsm8k.yaml"

MULTI_NODE_TESTS: dict[str, int] = {
    "tests/integration/defs/accuracy/test_disaggregated_multinode.py"
    "::TestDeepSeekV4ProDSparkMultinode::test_gsm8k_1p1d_dep4": 2,
}


@dataclass(frozen=True)
class GuardJob:
    name: str
    scope: str
    selectors: tuple[str, ...]
    nodes: int
    ntasks: int
    ntasks_per_node: int
    gpus_per_node: int
    use_pmix: bool
    launcher: str = "pytest"
    model_name: str | None = None
    checkpoint_path: str | None = None
    model_path_env: str | None = None


GUARD_JOBS = (
    GuardJob(
        name="dsv4-aggregate",
        scope="single-node",
        selectors=SINGLE_NODE_SELECTORS,
        nodes=1,
        ntasks=4,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=True,
        model_name="DeepSeek-V4-Pro-DSpark",
        checkpoint_path=DSV4_DSPARK_CHECKPOINT_PATH,
    ),
    GuardJob(
        name="k3-aggregate-gsm8k",
        scope="multi-node",
        selectors=(K3_AGGREGATE_SELECTOR,),
        nodes=2,
        ntasks=8,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=True,
        model_name="Kimi-K3",
        checkpoint_path=K3_CHECKPOINT_PATH,
        model_path_env="KIMI_K3_CKPT",
    ),
    GuardJob(
        name="k3-disaggregated-gsm8k",
        scope="multi-node",
        selectors=(K3_DISAGGREGATED_ACCURACY_CONFIG,),
        nodes=4,
        ntasks=16,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=False,
        launcher="benchmark",
    ),
    *(
        GuardJob(
            name=f"dsv4-multi-node-{test_name.rsplit('::', maxsplit=1)[-1]}",
            scope="multi-node",
            selectors=(test_name,),
            nodes=num_nodes,
            ntasks=num_nodes,
            ntasks_per_node=1,
            gpus_per_node=4,
            use_pmix=False,
            model_name="DeepSeek-V4-Pro-DSpark",
            checkpoint_path=DSV4_DSPARK_CHECKPOINT_PATH,
        )
        for test_name, num_nodes in MULTI_NODE_TESTS.items()
    ),
)


def _default(value: str, environment_name: str) -> str:
    return os.environ.get(environment_name, value)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DSV4 and Kimi K3 branch guards on Slurm.")
    parser.add_argument(
        "--scope",
        choices=("all", "single-node", "multi-node"),
        default="all",
        help="Which Slurm guard scope to submit (default: all).",
    )
    parser.add_argument(
        "--job",
        choices=tuple(job.name for job in GUARD_JOBS),
        default=None,
        help="Submit one named guard job instead of a whole scope.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("TRTLLM_TEST_IMAGE", ""),
        help="Pyxis container image (or set TRTLLM_TEST_IMAGE).",
    )
    parser.add_argument(
        "--partition",
        default=os.environ.get("TRTLLM_SLURM_PARTITION", ""),
        help="Slurm partition (or set TRTLLM_SLURM_PARTITION).",
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("TRTLLM_SLURM_ACCOUNT", ""),
        help="Slurm account (or set TRTLLM_SLURM_ACCOUNT).",
    )
    parser.add_argument(
        "--model-root",
        "--llm-models-root",
        dest="model_root",
        default=os.environ.get("LLM_MODELS_ROOT", ""),
        help="LLM model root visible inside the container.",
    )
    parser.add_argument(
        "--mounts",
        default=_default(
            "/lustre:/lustre",
            "TRTLLM_CONTAINER_MOUNTS",
        ),
        help="Pyxis container mounts (default: /lustre:/lustre).",
    )
    parser.add_argument(
        "--repo-root",
        "--trtllm-repo-root",
        type=Path,
        default=Path(os.environ.get("TRTLLM_REPO_ROOT", str(REPO_ROOT))),
        help=(
            "TensorRT-LLM repository path visible inside the container "
            "(default: auto-detected; or set TRTLLM_REPO_ROOT)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TRTLLM_GUARD_LOG_DIR",
                str(REPO_ROOT / "build_images" / "branch_guard_logs"),
            )
        ),
        help="Directory for Slurm stdout/stderr logs.",
    )
    parser.add_argument(
        "--single-node-time",
        default="04:00:00",
        help="Time limit for the single-node job.",
    )
    parser.add_argument(
        "--multi-node-time",
        default="04:00:00",
        help="Time limit for the multi-node job.",
    )
    parser.add_argument("--qos", default=os.environ.get("TRTLLM_SLURM_QOS", ""))
    parser.add_argument(
        "--constraint",
        default=os.environ.get("TRTLLM_SLURM_CONSTRAINT", ""),
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment assignment inside the container.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed to pytest; may be repeated.",
    )
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--install-from-repo",
        dest="install_from_repo",
        action="store_true",
        default=True,
        help="Install the detected repository with pip install -e inside the container.",
    )
    install_group.add_argument(
        "--skip-install",
        dest="install_from_repo",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Submit the next scope even if the previous job fails.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after submission instead of waiting for each job.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the guards without validating Slurm arguments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact sbatch commands without submitting jobs.",
    )
    return parser.parse_args(argv)


def _selected_jobs(scope: str, job_name: str | None = None) -> tuple[GuardJob, ...]:
    if job_name is not None:
        return tuple(job for job in GUARD_JOBS if job.name == job_name)
    if scope == "all":
        return GUARD_JOBS
    return tuple(job for job in GUARD_JOBS if job.scope == scope)


def _selector_path(repo_root: Path, selector: str) -> Path:
    return repo_root / selector.split("::", maxsplit=1)[0]


def _validate_env_assignments(assignments: Sequence[str]) -> None:
    for assignment in assignments:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", assignment):
            raise ValueError(f"Invalid --env assignment: {assignment!r}")


def _validate_args(args: argparse.Namespace, jobs: Sequence[GuardJob]) -> None:
    args.repo_root = args.repo_root.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    if not _is_trtllm_repo_root(args.repo_root):
        raise ValueError(f"Not a TensorRT-LLM repository: {args.repo_root}")

    required = {
        "--image": args.image,
        "--partition": args.partition,
        "--account": args.account,
        "--model-root": args.model_root,
    }
    missing = [name for name, value in required.items() if not value]
    if missing and not args.dry_run:
        raise ValueError("Missing required Slurm inputs: " + ", ".join(missing))

    if args.image and not Path(args.image).is_file() and not args.dry_run:
        raise ValueError(f"Container image does not exist: {args.image}")
    if args.model_root and not Path(args.model_root).is_dir() and not args.dry_run:
        raise ValueError(f"Model root does not exist: {args.model_root}")

    _validate_env_assignments(args.env)
    for job in jobs:
        for selector in job.selectors:
            if not _selector_path(args.repo_root, selector).is_file():
                raise ValueError(f"Test file does not exist: {selector}")

    if not args.dry_run:
        commands = ["sbatch", "srun"]
        if not args.no_wait:
            commands.extend(["squeue", "sacct", "scancel"])
        for command in commands:
            if shutil.which(command) is None:
                raise ValueError(f"Required Slurm command is not on PATH: {command}")


def _shell_env_lines(args: argparse.Namespace) -> list[str]:
    values = {
        "LLM_MODELS_ROOT": args.model_root,
        "HF_HOME": os.environ.get(
            "HF_HOME",
            "/lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache",
        ),
        "TLLM_LOG_LEVEL": "INFO",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "PRTE_ALLOW_RUN_AS_ROOT": "1",
        "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "PMIX_MCA_gds": "hash",
        "PYTHONPATH": "",
    }
    lines = [f"export {key}={shlex.quote(value)}" for key, value in values.items()]
    lines.append("unset TRTLLM_MOE_A2A_DISABLE_CFT_COUNTED_WRITES")
    lines.extend(f"export {assignment}" for assignment in args.env)
    return lines


def _install_lines(args: argparse.Namespace) -> list[str]:
    commands = [*_shell_env_lines(args)]
    if args.install_from_repo:
        commands.append(
            shlex.join(
                [
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    str(args.repo_root),
                ]
            )
        )
    commands.append(
        shlex.join(
            [
                "python3",
                "-m",
                "pip",
                "uninstall",
                "--yes",
                "nvidia-cutlass-dsl",
            ]
        )
    )
    install_shell = "; ".join(commands)
    return [
        f'flock "/tmp/trtllm_branch_guard_install_"$SLURM_JOB_ID".lock" '
        f"bash -lc {shlex.quote(install_shell)}",
    ]


def _model_setup_lines(args: argparse.Namespace, job: GuardJob) -> list[str]:
    if job.model_name is None or job.checkpoint_path is None:
        return []

    lines = [
        f"guard_model_root={shlex.quote(BRANCH_GUARD_MODEL_ROOT)}/{shlex.quote(job.name)}/job_${{SLURM_JOB_ID}}",
        f"guard_checkpoint={shlex.quote(job.checkpoint_path)}",
        f"guard_dataset_root={shlex.quote(str(Path(args.model_root) / 'datasets'))}",
        'test -d "$guard_checkpoint"',
        'test -d "$guard_dataset_root"',
        'mkdir -p "$guard_model_root"',
        f'ln -sfn "$guard_checkpoint" "$guard_model_root/{job.model_name}"',
        'ln -sfn "$guard_dataset_root" "$guard_model_root/datasets"',
        'export LLM_MODELS_ROOT="$guard_model_root"',
    ]
    if job.model_path_env is not None:
        lines.append(f'export {job.model_path_env}="$guard_model_root/{job.model_name}"')
    return lines


def _pytest_command(
    args: argparse.Namespace,
    job: GuardJob,
) -> tuple[list[str], str]:
    pytest_args = ["-x"] if not args.continue_on_failure else []
    pytest_args.extend(args.pytest_arg)
    pytest_args.extend(str(args.repo_root / selector) for selector in job.selectors)

    if job.use_pmix:
        return (
            [
                str(args.repo_root / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch"),
                # The launcher scrubs OMPI_* before starting pytest.
                # Re-add the explicit container-root override after that scrub.
                "env",
                "OMPI_ALLOW_RUN_AS_ROOT=1",
                "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
                "python3",
                "-m",
                "pytest",
                "-v",
                "-s",
                "--rootdir=/tmp",
                *pytest_args,
            ],
            "/tmp",
        )

    return (
        [
            "env",
            "PRTE_ALLOW_RUN_AS_ROOT=1",
            "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1",
            "python3",
            "-m",
            "pytest",
            "-q",
            "-s",
            "--rootdir=/tmp",
            *pytest_args,
        ],
        str(args.repo_root),
    )


def _container_shell(args: argparse.Namespace, job: GuardJob) -> str:
    if job.launcher != "pytest":
        raise ValueError(f"{job.name} is launched by its benchmark harness")
    command, workdir = _pytest_command(args, job)
    lines = ["set -eu", *_shell_env_lines(args)]
    lines.extend(_install_lines(args))
    lines.extend(_model_setup_lines(args, job))
    lines.append(f"cd {shlex.quote(workdir)}")
    if job.scope == "multi-node":
        lines.append(
            'if [ "$(id -u)" -eq 0 ] && '
            '{ [ "${PRTE_ALLOW_RUN_AS_ROOT:-}" != "1" ] || '
            '[ "${PRTE_ALLOW_RUN_AS_ROOT_CONFIRM:-}" != "1" ]; }; then '
            "echo 'MPI root-spawn environment is not configured; refusing to start the model.' >&2; "
            "exit 1; "
            "fi"
        )
    lines.append(f"exec {shlex.join(command)}")
    return "\n".join(lines)


def _srun_command(args: argparse.Namespace, job: GuardJob) -> list[str]:
    command = [
        "srun",
        "--kill-on-bad-exit=1",
        f"--ntasks={job.ntasks}",
        f"--ntasks-per-node={job.ntasks_per_node}",
        f"--container-image={args.image}",
        f"--container-mounts={args.mounts}",
        f"--container-workdir={args.repo_root}",
    ]
    if job.use_pmix:
        command.append("--mpi=pmix")
    command.extend(["bash", "-lc", _container_shell(args, job)])
    return command


def _sbatch_command(args: argparse.Namespace, job: GuardJob) -> list[str]:
    time_limit = args.single_node_time if job.scope == "single-node" else args.multi_node_time
    log_dir = str(args.log_dir)
    command = [
        "sbatch",
        "--parsable",
    ]
    command.extend(
        [
            f"--job-name=coreai_comparch_trtllm-{job.name}",
            f"--partition={args.partition}",
            f"--account={args.account}",
            f"--nodes={job.nodes}",
            f"--ntasks={job.ntasks}",
            f"--ntasks-per-node={job.ntasks_per_node}",
            f"--time={time_limit}",
            f"--output={log_dir}/job-%j.log",
            f"--error={log_dir}/job-%j.log",
        ]
    )
    if args.qos:
        command.append(f"--qos={args.qos}")
    if args.constraint:
        command.append(f"--constraint={args.constraint}")
    command.extend(["--wrap", shlex.join(_srun_command(args, job))])
    return command


def _print_jobs(jobs: Sequence[GuardJob]) -> None:
    for job in jobs:
        mpi = "pmix" if job.use_pmix else "none"
        selectors = ", ".join(job.selectors)
        print(
            f"{job.name}: scope={job.scope}, nodes={job.nodes}, "
            f"tasks={job.ntasks} ({job.ntasks_per_node}/node), "
            f"gpus/node={job.gpus_per_node}, mpi={mpi}, launcher={job.launcher}"
        )
        print(f"  tests: {selectors}")


def _job_id_from_output(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        submitted = re.search(r"Submitted batch job (\d+)", stripped)
        if submitted is not None:
            return submitted.group(1)
        job_id = stripped.split(";", maxsplit=1)[0]
        if job_id.isdigit():
            return job_id
    return None


def _write_benchmark_slurm_script(repo_root: Path, log_dir: Path) -> Path:
    source = repo_root / "examples/disaggregated/slurm/benchmark/disaggr_torch.slurm"
    output = log_dir / "disaggr_torch_branch_guard.slurm"
    marker = "# Get node lists and replace the placeholder with the actual node names"
    uninstall = """# Remove the image dependency after TensorRT-LLM installation.
echo "Uninstalling nvidia-cutlass-dsl..."
uninstall_log=${full_logdir}/3_uninstall_cutlass_dsl.log
if ! srun --container-name=${container_name} \\
        --container-mounts=${container_mount} --no-container-mount-home \\
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \\
        bash -c "python3 -m pip uninstall --yes nvidia-cutlass-dsl" \\
        &> "${uninstall_log}"; then
    cleanup_on_failure "nvidia-cutlass-dsl uninstall failed. Check ${uninstall_log} for details"
fi
echo "nvidia-cutlass-dsl uninstall completed successfully"

"""
    script = source.read_text()
    if script.count(marker) != 1:
        raise ValueError(f"Expected exactly one install-completion marker in {source}")
    output.write_text(script.replace(marker, uninstall + marker))
    output.chmod(source.stat().st_mode)
    return output


def _benchmark_config(args: argparse.Namespace, job: GuardJob) -> tuple[Path, Path]:
    source = args.repo_root / job.selectors[0]
    config = deepcopy(yaml.safe_load(source.read_text()))
    log_dir = args.log_dir / job.name
    log_dir.mkdir(parents=True, exist_ok=True)
    config["slurm"]["script_file"] = str(
        _write_benchmark_slurm_script(args.repo_root, log_dir)
    )

    config["slurm"]["partition"] = args.partition
    config["slurm"]["account"] = args.account
    config["slurm"]["job_time"] = args.multi_node_time
    config["slurm"]["job_name"] = job.name
    # batch-xdr allocates one GPU per task for this guard; the example config
    # uses --gpus-per-node, which this partition rejects as an invalid GRES.
    config["slurm"]["extra_args"] = ""
    config["environment"]["container_image"] = args.image
    config["environment"]["model_path"] = K3_CHECKPOINT_PATH
    config["environment"]["trtllm_repo"] = str(args.repo_root) if args.install_from_repo else ""
    config["environment"]["build_wheel"] = False
    config["environment"]["trtllm_wheel_path"] = ""
    config["environment"].pop("work_dir", None)
    config["environment"]["log_dir"] = str(log_dir)
    # The Rubin compute nodes do not expose a host /scratch directory to
    # Pyxis. All guard inputs, logs, and the checkout are on shared Lustre.
    config["environment"]["container_mount"] = (
        f"{args.repo_root}:{args.repo_root}:rw,/lustre:/lustre")

    # The branch guard is an accuracy run, not the optional synthetic serving
    # benchmark.  Keeping the benchmark disabled also avoids requiring its
    # unrelated 8K/1K JSONL dataset on the compute nodes.
    config["benchmark"]["enable_benchmark"] = False
    accuracy_env = config.setdefault("accuracy", {}).setdefault("env_var", {})
    accuracy_env["HF_HOME"] = os.environ.get(
        "HF_HOME", "/lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache"
    )
    accuracy_env["HF_HUB_OFFLINE"] = "1"
    accuracy_env["HF_DATASETS_OFFLINE"] = "1"

    # The example config names a developer checkout/venv.  The guard mounts
    # this repository and installs it from source, so remove those stale
    # path prepends while retaining the K3/NIXL environment settings.
    for key in ("worker_env_var", "server_env_var", "ctx_worker_env_var", "gen_worker_env_var"):
        value = config["environment"].get(key, "")
        config["environment"][key] = " ".join(
            token
            for token in value.split()
            if not token.startswith(("TRTLLM_PATH_PREPEND=", "TRTLLM_PYTHONPATH_PREPEND="))
        )

    runtime_config = args.log_dir / f"{job.name}-config-{int(time.time())}.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return runtime_config, log_dir


def _submit_benchmark_job(args: argparse.Namespace, job: GuardJob) -> tuple[str, str]:
    runtime_config, log_dir = _benchmark_config(args, job)
    submitter = args.repo_root / "examples/disaggregated/slurm/benchmark/submit.py"
    command = [
        sys.executable,
        str(submitter),
        "--config",
        str(runtime_config),
        "--log-dir",
        str(log_dir),
    ]
    result = subprocess.run(
        command, cwd=args.repo_root, check=False, capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output.strip() or f"benchmark submitter exited {result.returncode}")
    job_id = _job_id_from_output(output)
    if job_id is None:
        raise RuntimeError(f"benchmark submitter returned no Slurm job id:\n{output}")
    print(output, end="" if output.endswith("\n") else "\n")
    return job_id, output


def _test_result_summaries(log_path: Path) -> list[str]:
    try:
        log_text = log_path.read_text(errors="replace")
    except OSError:
        return []

    counts: dict[str, int] = {}
    for line in log_text.splitlines():
        match = _PYTEST_SUMMARY_RE.search(line)
        if match is not None:
            summary = " ".join(match.group("summary").split())
            counts[summary] = counts.get(summary, 0) + 1

    return [
        summary if count == 1 else f"{summary} (reported {count} times)"
        for summary, count in counts.items()
    ]


def _job_log_path(
    args: argparse.Namespace,
    sbatch_output: str,
    started_at: float,
) -> Path | None:
    job_id = _job_id_from_output(sbatch_output)
    if job_id is not None:
        log_path = args.log_dir / f"job-{job_id}.log"
        if log_path.exists():
            return log_path

    fresh_logs: list[Path] = []
    for log_path in args.log_dir.glob("job-*.log"):
        try:
            if log_path.stat().st_mtime >= started_at - 5:
                fresh_logs.append(log_path)
        except OSError:
            continue
    if not fresh_logs:
        return None
    return max(fresh_logs, key=lambda path: path.stat().st_mtime)


def _job_log_paths(args: argparse.Namespace, job_id: str, started_at: float) -> list[Path]:
    primary = args.log_dir / f"job-{job_id}.log"
    paths = [primary]
    role_names = ("ctx", "context", "gen", "generation", "disagg")
    for path in (
        *args.log_dir.rglob("*.log"),
        *args.log_dir.rglob("*.out"),
        *args.log_dir.rglob("*.err"),
    ):
        if path in paths:
            continue
        try:
            recent = path.stat().st_mtime >= started_at - 5
        except OSError:
            continue
        if recent and (
            job_id in path.name or any(role in path.stem.lower() for role in role_names)
        ):
            paths.append(path)
    return paths


def _read_new_log(path: Path, offset: int) -> tuple[int, list[str]]:
    try:
        size = path.stat().st_size
        if size < offset:
            offset = 0
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offset)
            text = stream.read()
            offset = stream.tell()
    except OSError:
        return offset, []
    return offset, text.splitlines()


def _failure_context(line: str) -> str | None:
    if _EARLY_FAILURE_RE.search(line) is None:
        return None
    role_map = {
        "ctx": "ctx",
        "context": "ctx",
        "gen": "gen",
        "generation": "gen",
        "disagg": "disagg",
        "disaggregated": "disagg",
    }
    roles: list[str] = []
    for raw_role in _ROLE_RE.findall(line):
        role = role_map[raw_role.lower()]
        if role not in roles:
            roles.append(role)
    label = "/".join(roles) if roles else "disagg/ctx/gen"
    return f"{label}: {line.strip()}"


def _scan_new_logs(paths: Sequence[Path], offsets: dict[Path, int]) -> str | None:
    for path in paths:
        offset, lines = _read_new_log(path, offsets.get(path, 0))
        offsets[path] = offset
        for line in lines:
            context = _failure_context(line)
            if context is not None:
                return f"{path}: {context}"
    return None


def _slurm_job_state(job_id: str) -> str | None:
    for command in (
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
    ):
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            state = line.strip().split("|", 1)[0].split("+", 1)[0]
            if state:
                return state
    return None


def _cancel_slurm_job(job_id: str) -> None:
    result = subprocess.run(
        ["scancel", job_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"[job {job_id}] scancel failed: {result.stderr.strip()}",
            file=sys.stderr,
        )


def _monitor_job(args: argparse.Namespace, job: GuardJob, job_id: str, started_at: float) -> bool:
    offsets: dict[Path, int] = {}
    print(f"[{job.name}] MONITORING job {job_id} for ctx/gen/disagg failures")
    while True:
        failure = _scan_new_logs(_job_log_paths(args, job_id, started_at), offsets)
        if failure is not None:
            print(f"[{job.name}] EARLY FAILURE: {failure}", file=sys.stderr)
            _cancel_slurm_job(job_id)
            return False

        state = _slurm_job_state(job_id)
        if state in _TERMINAL_SLURM_STATES:
            failure = _scan_new_logs(_job_log_paths(args, job_id, started_at), offsets)
            if failure is not None:
                print(f"[{job.name}] FAILURE: {failure}", file=sys.stderr)
                return False
            if state != "COMPLETED":
                print(f"[{job.name}] Slurm terminal state: {state}", file=sys.stderr)
                return False
            return True
        time.sleep(_MONITOR_INTERVAL_SECONDS)


def _print_test_results(
    args: argparse.Namespace,
    job: GuardJob,
    sbatch_output: str,
    started_at: float,
) -> None:
    job_id = _job_id_from_output(sbatch_output)
    if args.no_wait:
        if job_id is None:
            print(f"[{job.name}] SUBMITTED; Slurm output: {sbatch_output.strip()}")
        else:
            print(f"[{job.name}] SUBMITTED job {job_id}")
        return

    log_path = _job_log_path(args, sbatch_output, started_at)
    if log_path is None:
        print(f"[{job.name}] TEST RESULTS: no completed Slurm log found")
        return
    summaries = _test_result_summaries(log_path)
    print(f"[{job.name}] LOG: {log_path}")
    if not summaries:
        print(f"[{job.name}] TEST RESULTS: no pytest summary found")
        return

    print(f"[{job.name}] TEST RESULTS:")
    for summary in summaries:
        print(f"  {summary}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    jobs = _selected_jobs(args.scope, args.job)
    if args.list:
        _print_jobs(jobs)
        return 0

    try:
        _validate_args(args, jobs)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not args.dry_run:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for job in jobs:
        print(f"\n[{job.name}]")
        if job.launcher == "benchmark":
            if args.dry_run:
                runtime_config = args.repo_root / job.selectors[0]
                submitter = args.repo_root / "examples/disaggregated/slurm/benchmark/submit.py"
                print(shlex.join([sys.executable, str(submitter), "--config", str(runtime_config)]))
                continue
            started_at = time.time()
            try:
                job_id, sbatch_output = _submit_benchmark_job(args, job)
            except RuntimeError as error:
                failures += 1
                print(f"[{job.name}] FAIL ({error})", file=sys.stderr)
                if not args.continue_on_failure:
                    break
                continue
            print(f"[{job.name}] SUBMITTED job {job_id}")
            if args.no_wait:
                continue
            passed = _monitor_job(args, job, job_id, started_at)
            if passed:
                print(f"[{job.name}] PASS")
                continue
            failures += 1
            print(f"[{job.name}] FAIL (benchmark job did not complete)", file=sys.stderr)
            if not args.continue_on_failure:
                break
            continue

        command = _sbatch_command(args, job)
        print(shlex.join(command))
        if args.dry_run:
            continue

        started_at = time.time()
        result = subprocess.run(
            command,
            cwd=args.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )
        sbatch_output = result.stdout if result.stdout.strip() else result.stderr
        failure_message = f"sbatch exit {result.returncode}"
        passed = False
        if result.returncode == 0:
            job_id = _job_id_from_output(sbatch_output)
            if args.no_wait:
                _print_test_results(args, job, sbatch_output, started_at)
                continue
            if job_id is None:
                failure_message = "sbatch returned no job id"
            else:
                passed = _monitor_job(args, job, job_id, started_at)
            _print_test_results(args, job, sbatch_output, started_at)
            if passed:
                print(f"[{job.name}] PASS")
                continue
            if job_id is not None:
                failure_message = "early log monitor detected a failure"

        failures += 1
        print(
            f"[{job.name}] FAIL ({failure_message})",
            file=sys.stderr,
        )
        if not args.continue_on_failure:
            break

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
