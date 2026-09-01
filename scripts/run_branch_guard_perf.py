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
"""Run a pinned TensorRT-LLM branch performance guard through srt-slurm.

``--preset`` selects one Python-defined bundle containing an immutable recipe,
topology, expected runtime backends, workload shape, reference metrics, and gate
policy. The recipe is read from its pinned Git revision and SHA-256 checked, so
working-tree edits in the shared benchmark checkout cannot affect a run.

Normal operation prepares an image-specific recipe, validates it with srtctl,
submits it, waits for Slurm, then prints measured metrics beside the preset's
references and derived pass/fail guards. Useful modes are:

* ``--list-presets``: show available presets without requiring runtime inputs.
* ``--dry-run``: prepare and validate without submitting.
* ``--result-csv FILE``: evaluate an existing AIPerf result without submitting.

The benchmark checkout, srtctl, and setup hook must live under shared Lustre so
login-node temporary clones cannot silently become runtime dependencies.

Examples::

    # Show every measured Pareto preset.
    python3 scripts/run_branch_guard_perf.py --list-presets

    # Validate one preset without submitting it.
    python3 scripts/run_branch_guard_perf.py \
        --preset 1p6d-dep4-tep4-c12-b4-mtp \
        --image build_images/501fe74ae4/trtllm.sqsh \
        --aiperf-cache /lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache/aiperf/mmap \
        --dry-run

    # Submit the same preset by omitting --dry-run.
    python3 scripts/run_branch_guard_perf.py \
        --preset 1p6d-dep4-tep4-c12-b4-mtp \
        --image build_images/501fe74ae4/trtllm.sqsh \
        --aiperf-cache /lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache/aiperf/mmap

    # Re-evaluate an existing result against its matching preset.
    python3 scripts/run_branch_guard_perf.py \
        --preset 1p6d-dep4-tep4-c6-b4-mtp \
        --result-csv /path/to/profile_export_aiperf.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_LUSTRE_ROOT = Path("/lustre/share")
SHARED_TOOLS_ROOT = SHARED_LUSTRE_ROOT / "coreai_comparch_trtllm" / "lizhiz" / "tools"
DEFAULT_BENCHMARK_REPO = SHARED_TOOLS_ROOT / "bench-trtllm-disagg-perf-dsv4"
DEFAULT_SRTCTL = SHARED_TOOLS_ROOT / "srt-slurm" / ".venv-submit" / "bin" / "srtctl"
DEFAULT_SETUP_SCRIPTS_DIR = SHARED_TOOLS_ROOT / "srt-slurm" / "configs"
DEFAULT_WAIT_TIMEOUT_SECONDS = 4 * 60 * 60
CUTLASS_DSL_UNINSTALL_COMMAND = ""
TERMINAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class PerfPreset:
    """Immutable recipe identity, workload shape, and performance reference.

    A preset deliberately keeps these values together. Selecting a performance
    reference without its matching recipe, topology, and workload would produce
    a plausible-looking but invalid comparison.
    """

    name: str
    description: str
    recipe: Path
    reference_revision: str
    reference_sha256: str
    setup_script: str
    gpu_count: int
    gpus_per_node: int
    prefill_nodes: int
    decode_nodes: int
    prefill_backend: str
    decode_backend: str
    benchmark_duration_seconds: float
    benchmark_concurrency: int
    reference_tps_per_user: float
    reference_tps_per_gpu: float
    reference_p90_itl_ms: float
    reference_p50_ttft_ms: float
    minimum_duration_slack_seconds: float = 10.0
    minimum_throughput_ratio: float = 0.95
    maximum_latency_ratio: float = 1.10
    maximum_p50_ttft_ms: float = 10_000.0


DSV4_AGENTX_RECIPE_DIR = Path("srt-slurm-configs/deepseek-V4-Pro/AgentX-VRNVL72-20260814/recipes")
DSV4_AGENTX_REFERENCE_REVISION = "fe181981ef5118356b2ff11783c303b1c544a0a8"


def _dsv4_agentx_preset(
    name: str,
    recipe_sha256: str,
    gpu_count: int,
    prefill_nodes: int,
    decode_nodes: int,
    prefill_backend: str,
    decode_backend: str,
    concurrency: int,
    tps_per_user: float,
    tps_per_gpu: float,
    p90_itl_ms: float,
    p50_ttft_ms: float,
) -> PerfPreset:
    """Construct one measured point from the authoritative AgentX Pareto table."""
    return PerfPreset(
        name=name,
        description=f"DeepSeek V4 Pro AgentX VRNVL72 {name}",
        recipe=DSV4_AGENTX_RECIPE_DIR / f"disagg-VRNVL72-{name}.yaml",
        reference_revision=DSV4_AGENTX_REFERENCE_REVISION,
        reference_sha256=recipe_sha256,
        setup_script="reinstall-tensorrt-llm-repo.sh",
        gpu_count=gpu_count,
        gpus_per_node=4,
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        prefill_backend=prefill_backend,
        decode_backend=decode_backend,
        benchmark_duration_seconds=3600.0,
        benchmark_concurrency=concurrency,
        reference_tps_per_user=tps_per_user,
        reference_tps_per_gpu=tps_per_gpu,
        reference_p90_itl_ms=p90_itl_ms,
        reference_p50_ttft_ms=p50_ttft_ms,
    )


# These are all ten points in the measured TTFT-gated Pareto frontier documented
# by AgentX-VRNVL72-20260814/PERF.md. Add a new key instead of changing an
# existing point when the recipe, workload, or reference measurements change.
PERF_PRESETS = {
    preset.name: preset
    for preset in (
        _dsv4_agentx_preset(
            "1p6d-dep4-tep4-c6-b4-mtp",
            "4ffa1ef3d2321d41fd6f57aa9705e3309259b0d20485312f6a481fe10b833058",
            28,
            1,
            6,
            "CUTEDSL",
            "TRTLLM",
            6,
            263.9,
            1319.0,
            3.79,
            673.0,
        ),
        _dsv4_agentx_preset(
            "1p6d-dep4-tep4-c12-b4-mtp",
            "63afdee87bd5b1d49d4db22dac943d0454c4abfb4307f31cb98046d2fab5bf10",
            28,
            1,
            6,
            "CUTEDSL",
            "TRTLLM",
            12,
            244.5,
            2285.0,
            4.09,
            590.0,
        ),
        _dsv4_agentx_preset(
            "1p6d-dep4-tep4-c24-b4-mtp",
            "bca332b66ac60c7350cdfee9934c19a2103b6bba8f66d69b2f0224dd18d64308",
            28,
            1,
            6,
            "CUTEDSL",
            "TRTLLM",
            24,
            218.3,
            5683.0,
            4.58,
            659.0,
        ),
        _dsv4_agentx_preset(
            "1p1d-dep8-dep32-c160-b1-mtp",
            "83fbf9e3e451005d069b43f8a9ad2e29148d59bc15681aae666d7009e3a0d4aa",
            40,
            2,
            8,
            "CUTEDSL",
            "CUTEDSL",
            160,
            202.0,
            21138.0,
            4.95,
            5161.0,
        ),
        _dsv4_agentx_preset(
            "1p1d-dep8-dep32-c306-b2-mtp",
            "2fe5d9cb239f5f714beae81653ed60fe32d2e4867ed046e03d8653124935029a",
            40,
            2,
            8,
            "CUTEDSL",
            "CUTEDSL",
            306,
            181.8,
            39215.0,
            5.50,
            7020.0,
        ),
        _dsv4_agentx_preset(
            "2p1d-dep8-dep32-c561-b4-mtp",
            "7604f0ed006b83f59e85dfa4a822abbf97c0f5885372cb8ecbd5ae55edc08db1",
            48,
            4,
            8,
            "CUTEDSL",
            "CUTEDSL",
            561,
            170.6,
            61758.0,
            5.86,
            5092.0,
        ),
        _dsv4_agentx_preset(
            "2p1d-dep8-dep16-c934-b16-mtp",
            "1d9b67fdf578c8afe6d9f60abeab686d7150bdce3a70d56d15f2ba6f2585fa61",
            32,
            4,
            4,
            "CUTEDSL",
            "CUTEDSL",
            934,
            130.7,
            139237.0,
            7.65,
            6046.0,
        ),
        _dsv4_agentx_preset(
            "2p1d-dep8-dep32-c1061-b8-mtp",
            "95e8d36ea37f9a1b468d3fd0282baa4d039e9616ed2c0ab816232fde5cc18bcc",
            48,
            4,
            8,
            "CUTEDSL",
            "CUTEDSL",
            1061,
            144.5,
            104322.0,
            6.92,
            7456.0,
        ),
        _dsv4_agentx_preset(
            "3p1d-dep8-dep16-c1533-b32-mtp",
            "b86a5101b2c1e4f564a476eee4c19e565c31e8e4913c0840b373d212340e7e42",
            40,
            6,
            4,
            "CUTEDSL",
            "CUTEDSL",
            1533,
            100.0,
            170104.0,
            10.00,
            6390.0,
        ),
        _dsv4_agentx_preset(
            "4p1d-dep8-dep16-c2572-b64-mtp",
            "9630688f74d0bf2f7e99b746b1b3deeedde342bf4d68ec1a0d0adc12e9a9f885",
            48,
            8,
            4,
            "CUTEDSL",
            "CUTEDSL",
            2572,
            76.9,
            214622.0,
            13.01,
            6706.0,
        ),
    )
}
DEFAULT_PERF_PRESET = "1p6d-dep4-tep4-c6-b4-mtp"


@dataclass(frozen=True)
class PerfMetrics:
    duration_seconds: float
    request_count: int
    error_count: int
    error_rate: float
    total_tps: float
    tps_per_user: float
    tps_per_gpu: float
    p90_itl_ms: float
    p50_ttft_ms: float


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    limit: float
    minimum: bool

    @property
    def passed(self) -> bool:
        if self.minimum:
            return self.value >= self.limit
        return self.value <= self.limit


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PERF_PRESETS),
        default=DEFAULT_PERF_PRESET,
        help="Recipe, topology, workload, reference metrics, and guard policy to use.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available presets and exit.",
    )
    parser.add_argument(
        "--result-csv",
        type=Path,
        help="Evaluate an existing aiperf CSV with --preset; do not submit a job.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("TRTLLM_TEST_IMAGE", ""),
        help="TensorRT-LLM SquashFS image (or set TRTLLM_TEST_IMAGE).",
    )
    parser.add_argument(
        "--legacy-block-reuse-policy",
        action="store_true",
        help="Keep the reference recipe block_reuse_policy field for legacy images.",
    )
    parser.add_argument(
        "--nsys-prefill-window",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="Capture only prefill iterations START through STOP with Nsys.",
    )
    parser.add_argument(
        "--nsys-bin",
        type=Path,
        help="Nsys executable (or wrapper) passed to srtctl through SRTCTL_NSYS_BIN.",
    )
    parser.add_argument(
        "--benchmark-repo",
        type=Path,
        default=Path(os.environ.get("TRTLLM_PERF_BENCH_REPO", DEFAULT_BENCHMARK_REPO)),
        help="Current bench-trtllm-disagg checkout containing the reference recipe.",
    )
    parser.add_argument(
        "--aiperf-cache",
        type=Path,
        default=(
            Path(os.environ["AIPERF_DATASET_MMAP_CACHE_DIR"])
            if os.environ.get("AIPERF_DATASET_MMAP_CACHE_DIR")
            else None
        ),
        help="Writable shared AgentX dataset cache.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TRTLLM_PERF_GUARD_DIR",
                SCRIPT_REPO_ROOT / "build_images" / "branch_guard_perf",
            )
        ),
        help="Shared directory for generated configs and srtctl outputs.",
    )
    parser.add_argument(
        "--srtctl",
        default=os.environ.get("TRTLLM_SRTCTL", str(DEFAULT_SRTCTL)),
        help="Shared-Lustre srtctl executable (or set TRTLLM_SRTCTL).",
    )
    parser.add_argument(
        "--setup-scripts-dir",
        type=Path,
        default=Path(os.environ.get("TRTLLM_SETUP_SCRIPTS_DIR", DEFAULT_SETUP_SCRIPTS_DIR)),
        help="Shared-Lustre directory containing the preset setup hook.",
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("TRTLLM_SLURM_ACCOUNT", "coreai_comparch_trtllm"),
    )
    parser.add_argument(
        "--partition",
        default=os.environ.get("TRTLLM_SLURM_PARTITION", "batch-xdr"),
    )
    parser.add_argument(
        "--constraint",
        default=os.environ.get("TRTLLM_SLURM_CONSTRAINT", "cr"),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between Slurm state checks.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for the submitted job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate with srtctl without submitting.",
    )
    return parser.parse_args(argv)


def _replace_one(text: str, pattern: str, replacement: str, field: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {field} field in the reference recipe")
    return updated


def _read_pinned_recipe(benchmark_repo: Path, preset: PerfPreset) -> str:
    """Read and authenticate a recipe from Git, ignoring working-tree edits."""
    result = subprocess.run(
        ["git", "show", f"{preset.reference_revision}:{preset.recipe.as_posix()}"],
        cwd=benchmark_repo,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"Cannot read preset {preset.name!r} from {preset.reference_revision}: {detail}"
        )
    digest = hashlib.sha256(result.stdout).hexdigest()
    if digest != preset.reference_sha256:
        raise ValueError(
            f"Recipe hash mismatch for preset {preset.name!r}: "
            f"expected {preset.reference_sha256}, got {digest}"
        )
    return result.stdout.decode()


def _recipe_integer(recipe: str, field: str) -> int:
    matches = re.findall(rf"^  {re.escape(field)}: (\d+)$", recipe, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one resources.{field} field in the pinned recipe")
    return int(matches[0])


def _validate_reference_recipe(recipe: str, preset: PerfPreset) -> None:
    """Fail before submission if the recipe no longer matches its preset."""
    expected_resources = {
        "gpus_per_node": preset.gpus_per_node,
        "prefill_nodes": preset.prefill_nodes,
        "decode_nodes": preset.decode_nodes,
    }
    for field, expected in expected_resources.items():
        actual = _recipe_integer(recipe, field)
        if actual != expected:
            raise ValueError(f"Preset expects resources.{field}={expected}, got {actual}")
    configured_gpus = (preset.prefill_nodes + preset.decode_nodes) * preset.gpus_per_node
    if configured_gpus != preset.gpu_count:
        raise ValueError(
            f"Preset topology describes {configured_gpus} GPUs but gpu_count={preset.gpu_count}"
        )

    role_sections = {
        "prefill": r"(?ms)^    prefill:\n(.*?)(?=^    decode:\n)",
        "decode": r"(?ms)^    decode:\n(.*?)(?=^[a-z])",
    }
    for role, expected_backend in (
        ("prefill", preset.prefill_backend),
        ("decode", preset.decode_backend),
    ):
        section_match = re.search(role_sections[role], recipe)
        if section_match is None:
            raise ValueError(f"Cannot find the {role} runtime configuration")
        backend_matches = re.findall(
            r"^      moe_config:\n        backend: (\S+)$",
            section_match.group(1),
            flags=re.MULTILINE,
        )
        if backend_matches != [expected_backend]:
            raise ValueError(
                f"Preset expects the {role} MoE backend to be {expected_backend}; "
                f"found {backend_matches or 'none'}"
            )

    benchmark_pattern = (
        rf"--benchmark-duration {preset.benchmark_duration_seconds:g} "
        rf"--concurrency {preset.benchmark_concurrency}(?:\s|\\)"
    )
    if re.search(benchmark_pattern, recipe) is None:
        raise ValueError(
            "Pinned recipe benchmark duration/concurrency does not match the selected preset"
        )


def _prepare_recipe(
    source: str,
    output: Path,
    image: Path,
    cache: Path,
    preset: PerfPreset,
    legacy_block_reuse_policy: bool,
    nsys_prefill_window: tuple[int, int] | None,
) -> None:
    _validate_reference_recipe(source, preset)
    recipe = _replace_one(
        source,
        r"^(name: .+)$",
        rf"\1\nsetup_script: {preset.setup_script}",
        "name",
    )
    recipe = _replace_one(
        recipe,
        r"^  container: .+$",
        f"  container: {image}",
        "model.container",
    )
    recipe = _replace_one(
        recipe,
        r"^    image: .+$",
        f"    image: {image}",
        "identity.container.image",
    )
    recipe = _replace_one(
        recipe,
        r"^    AIPERF_DATASET_MMAP_CACHE_DIR: .+$",
        f"    AIPERF_DATASET_MMAP_CACHE_DIR: {cache}",
        "benchmark.env.AIPERF_DATASET_MMAP_CACHE_DIR",
    )
    if not legacy_block_reuse_policy:
        recipe = _replace_one(
            recipe,
            r"^(\s*)block_reuse_policy: (.+)$",
            r"\1policy: \2",
            "model.trtllm_config.prefill.kv_cache_config.block_reuse_config.policy",
        )
    if nsys_prefill_window:
        start_step, stop_step = nsys_prefill_window
        if start_step <= 0 or stop_step < start_step:
            raise ValueError("--nsys-prefill-window requires 0 < START <= STOP")
        recipe += (
            "\nprofiling:\n"
            "  type: nsys\n"
            "  prefill:\n"
            f"    start_step: {start_step}\n"
            f"    stop_step: {stop_step}\n"
            "  decode: {}\n"
        )
    _validate_reference_recipe(recipe, preset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(recipe)


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _write_cluster_config(
    output: Path,
    account: str,
    partition: str,
    constraint: str,
) -> None:
    config = "\n".join(
        (
            f"default_account: {_yaml_string(account)}",
            f"default_partition: {_yaml_string(partition)}",
            'default_time_limit: "04:00:00"',
            f"default_bash_preamble: {_yaml_string(CUTLASS_DSL_UNINSTALL_COMMAND)}",
            "gpus_per_node: 4",
            "use_gpus_per_node_directive: false",
            "use_segment_sbatch_directive: true",
            "use_exclusive_sbatch_directive: false",
            "use_het_jobs: false",
            "default_sbatch_directives:",
            f"  constraint: {_yaml_string(constraint)}",
            "default_mounts:",
            '  "/lustre": "/lustre"',
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(config)


def _resolve_executable(value: str) -> Path:
    resolved = shutil.which(value)
    path = Path(resolved or value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Executable does not exist or is not executable: {value}")
    return path


def _require_shared_lustre_path(path: Path, description: str) -> None:
    try:
        path.relative_to(SHARED_LUSTRE_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must be under {SHARED_LUSTRE_ROOT}: {path}") from error


def _validate_inputs(args: argparse.Namespace, preset: PerfPreset) -> tuple[Path, str, Path, Path]:
    if not args.image:
        raise ValueError("--image or TRTLLM_TEST_IMAGE is required")
    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise ValueError(f"Container image does not exist: {image}")

    benchmark_repo = args.benchmark_repo.expanduser().resolve()
    _require_shared_lustre_path(benchmark_repo, "Benchmark repository")
    if not (benchmark_repo / ".git").exists():
        raise ValueError(f"Benchmark repository is not a Git checkout: {benchmark_repo}")
    source_recipe = _read_pinned_recipe(benchmark_repo, preset)
    _validate_reference_recipe(source_recipe, preset)

    setup_scripts_dir = args.setup_scripts_dir.expanduser().resolve()
    _require_shared_lustre_path(setup_scripts_dir, "Setup scripts directory")
    setup_script = setup_scripts_dir / preset.setup_script
    if not setup_script.is_file():
        raise ValueError(f"Preset setup hook does not exist: {setup_script}")

    if args.aiperf_cache is None:
        raise ValueError("--aiperf-cache or AIPERF_DATASET_MMAP_CACHE_DIR is required")
    cache = args.aiperf_cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if not os.access(cache, os.W_OK):
        raise ValueError(f"AgentX dataset cache is not writable: {cache}")

    if not args.srtctl:
        raise ValueError("--srtctl or TRTLLM_SRTCTL is required")
    srtctl = _resolve_executable(args.srtctl)
    _require_shared_lustre_path(srtctl, "srtctl")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be positive")
    if args.wait_timeout <= 0:
        raise ValueError("--wait-timeout must be positive")
    if not args.dry_run:
        for command in ("squeue", "sacct", "scancel"):
            _resolve_executable(command)
    return image, source_recipe, cache, srtctl


def _run_srtctl(
    args: argparse.Namespace,
    srtctl: Path,
    recipe: Path,
    cluster_config: Path,
    output_base: Path,
) -> dict[str, str] | None:
    command = [
        str(srtctl),
        "dry-run" if args.dry_run else "apply",
        "-f",
        str(recipe),
        "-o",
        str(output_base),
        "-y",
    ]
    if not args.dry_run:
        command.append("--json")
    environment = os.environ.copy()
    environment["SRTSLURM_CONFIG"] = str(cluster_config)
    if args.nsys_bin:
        environment["SRTCTL_NSYS_BIN"] = str(args.nsys_bin)
    print("Running:", " ".join(command))
    result = subprocess.run(
        command,
        cwd=recipe.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if args.dry_run:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.returncode != 0:
            raise RuntimeError(f"srtctl dry-run failed with exit code {result.returncode}")
        return None

    if result.returncode != 0:
        raise RuntimeError(f"srtctl apply failed with exit code {result.returncode}")
    try:
        submission = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError("srtctl apply did not return its --json submission record") from error
    if submission.get("status") != "submitted":
        raise RuntimeError(f"srtctl submission failed: {submission}")
    return submission


def _normalize_slurm_state(value: str) -> str:
    return value.strip().split("|", maxsplit=1)[0].split("+", maxsplit=1)[0]


def _slurm_state(job_id: str) -> str | None:
    queue = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in queue.stdout.splitlines():
        state = _normalize_slurm_state(line)
        if state:
            return state

    accounting = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in accounting.stdout.splitlines():
        state = _normalize_slurm_state(line)
        if state:
            return state
    return None


def _wait_for_job(job_id: str, poll_interval: float, timeout: float) -> None:
    started_at = time.monotonic()
    previous_state = None
    while True:
        state = _slurm_state(job_id)
        if state != previous_state:
            print(f"Job {job_id}: {state or 'state unavailable'}")
            previous_state = state
        if state in TERMINAL_SLURM_STATES:
            if state != "COMPLETED":
                raise RuntimeError(f"Slurm job {job_id} finished in state {state}")
            return
        if time.monotonic() - started_at >= timeout:
            subprocess.run(["scancel", job_id], check=False)
            raise RuntimeError(f"Timed out waiting for Slurm job {job_id}; cancelled it")
        time.sleep(poll_interval)


def _find_result_csv(output_dir: Path, preset: PerfPreset) -> Path:
    expected = (
        output_dir
        / "logs"
        / "agentic"
        / f"concurrency_{preset.benchmark_concurrency}"
        / "profile_export_aiperf.csv"
    )
    if expected.is_file():
        return expected
    matches = list(output_dir.rglob("profile_export_aiperf.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one profile_export_aiperf.csv under {output_dir}")
    return matches[0]


def _read_metric_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "Metric" not in rows[0]:
        raise ValueError(f"Invalid aiperf CSV schema: {csv_path}")
    return {
        row["Metric"]: row
        for row in rows
        if row.get("Metric") and "(error-adjusted)" not in row["Metric"]
    }


def _metric_value(
    rows: dict[str, dict[str, str]],
    metric_prefix: str,
    column: str,
) -> float:
    matches = [name for name in rows if name.startswith(metric_prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one aiperf metric starting with {metric_prefix!r}")
    raw_value = rows[matches[0]].get(column, "")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing {column} value for aiperf metric {matches[0]!r}") from error


def _optional_metric_value(
    rows: dict[str, dict[str, str]],
    metric_prefix: str,
    column: str,
    default: float,
) -> float:
    matches = [name for name in rows if name.startswith(metric_prefix)]
    if not matches:
        return default
    return _metric_value(rows, metric_prefix, column)


def _parse_metrics(csv_path: Path, preset: PerfPreset) -> PerfMetrics:
    """Parse AIPerf output, including its omitted zero-error summary row."""
    rows = _read_metric_rows(csv_path)
    duration = _metric_value(rows, "Benchmark Duration", "avg")
    completed = _metric_value(rows, "Request Count", "avg")
    # AIPerf emits Error Request Count only when it is nonzero.
    errors = _optional_metric_value(rows, "Error Request Count", "avg", 0.0)
    request_count = completed + errors
    error_rate = errors / request_count if request_count else 1.0
    total_tps = _metric_value(rows, "Total Token Throughput", "avg")
    p90_itl = _metric_value(rows, "Full-Response Inter Token Latency", "p90")
    if p90_itl <= 0:
        raise ValueError(f"Full-Response p90 ITL must be positive, got {p90_itl}")
    p50_ttft = _metric_value(rows, "Time to First Token", "p50")
    return PerfMetrics(
        duration_seconds=duration,
        request_count=int(request_count),
        error_count=int(errors),
        error_rate=error_rate,
        total_tps=total_tps,
        tps_per_user=1000.0 / p90_itl,
        tps_per_gpu=total_tps / preset.gpu_count,
        p90_itl_ms=p90_itl,
        p50_ttft_ms=p50_ttft,
    )


def _gate_results(metrics: PerfMetrics, preset: PerfPreset) -> tuple[GateResult, ...]:
    return (
        GateResult(
            "duration_seconds",
            metrics.duration_seconds,
            preset.benchmark_duration_seconds - preset.minimum_duration_slack_seconds,
            True,
        ),
        GateResult("error_rate", metrics.error_rate, 0.0, False),
        GateResult(
            "tps_per_user",
            metrics.tps_per_user,
            preset.reference_tps_per_user * preset.minimum_throughput_ratio,
            True,
        ),
        GateResult(
            "tps_per_gpu",
            metrics.tps_per_gpu,
            preset.reference_tps_per_gpu * preset.minimum_throughput_ratio,
            True,
        ),
        GateResult(
            "p90_itl_ms",
            metrics.p90_itl_ms,
            preset.reference_p90_itl_ms * preset.maximum_latency_ratio,
            False,
        ),
        GateResult(
            "p50_ttft_ms",
            metrics.p50_ttft_ms,
            min(
                preset.reference_p50_ttft_ms * preset.maximum_latency_ratio,
                preset.maximum_p50_ttft_ms,
            ),
            False,
        ),
    )


def _check_result(csv_path: Path, preset: PerfPreset) -> bool:
    """Print measured values beside references and derived pass/fail limits."""
    metrics = _parse_metrics(csv_path, preset)
    results = _gate_results(metrics, preset)
    references = {
        "duration_seconds": preset.benchmark_duration_seconds,
        "error_rate": 0.0,
        "tps_per_user": preset.reference_tps_per_user,
        "tps_per_gpu": preset.reference_tps_per_gpu,
        "p90_itl_ms": preset.reference_p90_itl_ms,
        "p50_ttft_ms": preset.reference_p50_ttft_ms,
    }
    print(f"Preset: {preset.name} - {preset.description}")
    print(f"Results: {csv_path}")
    print(f"Requests: {metrics.request_count} total, {metrics.error_count} errors")
    print(f"Total token throughput: {metrics.total_tps:.2f} tokens/sec")
    print("Metric                    Actual    Reference       Guard  Result")
    print("--------------------  ----------  -----------  ----------  ------")
    for result in results:
        comparison = ">=" if result.minimum else "<="
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.name:20}  {result.value:10.3f}  "
            f"{references[result.name]:11.3f}  {comparison}{result.limit:8.3f}  {status:>6}"
        )
    return all(result.passed for result in results)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.list_presets:
            for name, available_preset in PERF_PRESETS.items():
                print(f"{name}: {available_preset.description}")
            return 0

        preset = PERF_PRESETS[args.preset]
        if args.result_csv is not None:
            csv_path = args.result_csv.expanduser().resolve()
            if not csv_path.is_file():
                raise ValueError(f"AIPerf result CSV does not exist: {csv_path}")
            return 0 if _check_result(csv_path, preset) else 1

        image, source_recipe, cache, srtctl = _validate_inputs(args, preset)
        image_run_id = image.parent.name
        run_dir = args.work_dir.expanduser().resolve() / preset.name / image_run_id
        recipe = run_dir / preset.recipe.name
        cluster_config = run_dir / "srtslurm-hecate.yaml"
        output_base = run_dir / "outputs"

        _prepare_recipe(
            source_recipe,
            recipe,
            image,
            cache,
            preset,
            args.legacy_block_reuse_policy,
            tuple(args.nsys_prefill_window) if args.nsys_prefill_window else None,
        )
        _write_cluster_config(
            cluster_config,
            account=args.account,
            partition=args.partition,
            constraint=args.constraint,
        )
        print(f"Prepared recipe: {recipe}")
        print(f"Cluster config: {cluster_config}")
        print(f"Preset: {preset.name} - {preset.description}")
        print(
            f"Pinned reference: {preset.reference_revision}:{preset.recipe} "
            f"(sha256 {preset.reference_sha256})"
        )

        submission = _run_srtctl(
            args,
            srtctl,
            recipe,
            cluster_config,
            output_base,
        )
        if submission is None:
            print("DRY RUN PASS")
            return 0

        job_id = submission["slurm_job_id"]
        output_dir = Path(submission["output_dir"]).resolve()
        print(f"Submitted job {job_id}; output: {output_dir}")
        _wait_for_job(job_id, args.poll_interval, args.wait_timeout)
        return 0 if _check_result(_find_result_csv(output_dir, preset), preset) else 1
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
