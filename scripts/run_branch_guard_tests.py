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
# Lines that label themselves INFO or DEBUG cannot be the failure, whatever
# they happen to contain. On job 557546 the scanner's first hit was
# "[TRT-LLM] [I] [_torch][RANK 2] Fallback to regular model init: Traceback
# (most recent call last):" -- an expected fast-init fallback, matched only
# because the pattern accepts a bare "Traceback (most recent call last)". In
# wait mode that would have cancelled the job 6m17s before the real cause (an
# OOM) was written, and then reported the fallback as the failure: the wrong
# diagnosis, with the right one destroyed. 16 further benign matches preceded
# the genuine one. WARNING is deliberately not suppressed -- NCCL WARN is in
# the pattern on purpose.
_BENIGN_SEVERITY_RE = re.compile(
    r"\[TRT-LLM\] \[[ID]\]|\b(?:INFO|DEBUG)\b",
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


def _first_available_path(primary: str, *fallbacks: str) -> str:
    for candidate in (primary, *fallbacks):
        if Path(candidate).is_dir():
            return candidate
    return primary


def _path_for_this_cluster(primary: str, *fallbacks: tuple[str, str]) -> str:
    """Pick a path by which cluster we are on, given ``(marker, path)`` pairs.

    ``_first_available_path`` probes the leaf, which is correct for a read-only
    checkpoint that must already be present. It is wrong for a directory the
    guard creates on demand -- a writable staging root or an HF cache -- and
    wrong for a container mount string, which is not a local path at all. In
    both of those cases probing the leaf would fall through to another
    cluster's value and then fail at mkdir or at container start.
    """
    for marker, candidate in fallbacks:
        if Path(marker).is_dir():
            return candidate
    return primary


SINGLE_NODE_SELECTORS = (
    "tests/integration/defs/accuracy/test_llm_api_pytorch.py"
    "::TestDeepSeekV4ProDSpark::test_gsm8k_dep4_megamoe_cutedsl",
)

K3_AGGREGATE_SELECTOR = (
    "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestKimiK3::test_gsm8k_full_accuracy"
)
K3_NVFP4_KV_AGGREGATE_SELECTOR = (
    "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestKimiK3::test_gsm8k_nvfp4_kv"
)
K3_DSPARK_AGGREGATE_SELECTOR = (
    "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestKimiK3::test_gsm8k_tep8_dspark"
)
K3_CHECKPOINT_PATH = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/nvidia_kimi-k3-nvfp4/hf/hf-1ec10fa_orig"
)
K3_DSPARK_CHECKPOINT_PATH = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/radixark_kimi-k3-dspark/hf/hf-3c5bac3_orig"
)
# Resolved here rather than behind a flag: this one is baked into a GuardJob as
# draft_checkpoint_path, which takes precedence over --k3-dspark-checkpoint, so
# there is no command-line way to correct it on a cluster where it is absent.
K3_HELIX_DSPARK_CHECKPOINT_PATH = _first_available_path(
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/inferact_kimi-k3-dspark/hf/hf-cf6b824_orig",
    "/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/"
    "inferact_kimi-k3-dspark/hf/hf-cf6b824_orig",
)
FLASHINFER_CUBINS_REPOSITORY = (
    "https://${URM_USER}:${URM_TOKEN}@urm.nvidia.com/"
    "artifactory/sw-kernelinferencelibrary-generic-local/"
)
DSV4_DSPARK_CHECKPOINT_PATH = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/"
    "nvidia_deepseek-v4-pro-nvfp4-dspark/hf/hf-318bf60_orig"
)
OCI_AGA_K3_CHECKPOINT_PATH = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/"
    "nvidia_kimi-k3-nvfp4/hf/hf-1ec10fa_orig"
)
OCI_AGA_DSV4_DSPARK_CHECKPOINT_PATH = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/"
    "nvidia_deepseek-v4-pro-nvfp4-dspark/hf/hf-318bf60_orig"
)
OCI_AGA_K3_DSPARK_CHECKPOINT_PATH = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/"
    "radixark_kimi-k3-dspark/hf/hf-3c5bac3_orig"
)
BRANCH_GUARD_MODEL_ROOT = "/lustre/share/coreai_comparch_trtllm/lizhiz/branch_guard_models"
OCI_AGA_BRANCH_GUARD_MODEL_ROOT = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/lizhiz/"
    "branch_guard_models"
)
HF_CACHE = "/lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache"
OCI_AGA_HF_CACHE = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/lizhiz/hf_cache"
)
CONTAINER_MOUNTS = "/lustre:/lustre"
OCI_AGA_CONTAINER_MOUNTS = "/scratch:/scratch"

# Directories that exist on exactly one cluster, used to choose between the
# path sets above. Deliberately NOT "/lustre": on oci-jhb /lustre is a symlink
# to /scratch, so /lustre and /lustre/fsw exist there too and discriminate
# nothing. /lustre/share and /scratch/fsw/portfolios do.
HECATE_MARKER = "/lustre/share"
OCI_AGA_MARKER = "/scratch/fsw/portfolios"
# lm-eval --seed, "python,numpy,torch,fewshot". The fewshot field is what the
# K3 reference numbers were measured with; its lm-eval default of 1234 is not.
_LM_EVAL_SEED = "0,1234,1234,0"
K3_DISAGGREGATED_ACCURACY_CONFIG = "examples/kimi_k3/disagg/benchmark_kimi_k3_dep8_gsm8k.yaml"
K3_DSPARK_SPECULATIVE_CONFIG: dict[str, str | int] = {
    "attention_backend": "TRTLLM",
    "block_size": 7,
    "decoding_type": "DSpark",
    "max_draft_len": 7,
    "speculative_model": K3_DSPARK_CHECKPOINT_PATH,
}
K3_DEP16_DISAGGREGATED_ACCURACY_CONFIG = (
    "tests/scripts/perf/disaggregated/"
    "gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml"
)

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
    draft_model_name: str | None = None
    draft_checkpoint_path: str | None = None
    speculative_config: dict[str, str | int] | None = None


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
        # Smallest gate that reaches FP4 MLA generation: the checkpoint
        # declares kv_cache_quant_algo null, so only an explicit nvfp4 KV
        # cache selects that path, and the disaggregated perf presets that
        # otherwise cover it need six nodes.
        name="k3-aggregate-nvfp4-kv-gsm8k",
        scope="multi-node",
        selectors=(K3_NVFP4_KV_AGGREGATE_SELECTOR,),
        nodes=4,
        ntasks=16,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=True,
        model_name="Kimi-K3",
        checkpoint_path=K3_CHECKPOINT_PATH,
        model_path_env="KIMI_K3_CKPT",
    ),
    GuardJob(
        name="k3-aggregate-dspark-gsm8k",
        scope="multi-node",
        selectors=(K3_DSPARK_AGGREGATE_SELECTOR,),
        nodes=2,
        ntasks=8,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=True,
        model_name="Kimi-K3",
        checkpoint_path=K3_CHECKPOINT_PATH,
        model_path_env="KIMI_K3_CKPT",
        draft_model_name="Kimi-K3-DSpark",
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
    GuardJob(
        name="k3-disaggregated-dspark-gsm8k",
        scope="multi-node",
        selectors=(K3_DISAGGREGATED_ACCURACY_CONFIG,),
        nodes=4,
        ntasks=16,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=False,
        launcher="benchmark",
        speculative_config=K3_DSPARK_SPECULATIVE_CONFIG,
    ),
    GuardJob(
        name="k3-disaggregated-dspark-helix-cp8-gsm8k",
        scope="multi-node",
        selectors=(K3_DISAGGREGATED_ACCURACY_CONFIG,),
        nodes=4,
        ntasks=16,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=False,
        launcher="benchmark",
        speculative_config=K3_DSPARK_SPECULATIVE_CONFIG,
        draft_checkpoint_path=K3_HELIX_DSPARK_CHECKPOINT_PATH,
    ),
    GuardJob(
        name="k3-disaggregated-dep16-gsm8k",
        scope="multi-node",
        selectors=(K3_DEP16_DISAGGREGATED_ACCURACY_CONFIG,),
        nodes=8,
        ntasks=32,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=False,
        launcher="benchmark",
    ),
    # DSpark on the EP16 topology. The EP8 DSpark job above cannot be made to
    # fit on GB300: it OOMs during model creation, before KV-cache sizing, at
    # 252.59 GB/rank of weights alone, so no free_gpu_memory_fraction helps.
    # Expert-parallel width is the only measured lever -- EP16 holds 56 experts
    # per rank per layer against EP8's 112 and was measured at 163.44 GB/rank
    # on the same cluster -- so this is the variant that actually exercises
    # DSpark there.
    GuardJob(
        name="k3-disaggregated-dspark-dep16-gsm8k",
        scope="multi-node",
        selectors=(K3_DEP16_DISAGGREGATED_ACCURACY_CONFIG,),
        nodes=8,
        ntasks=32,
        ntasks_per_node=4,
        gpus_per_node=4,
        use_pmix=False,
        launcher="benchmark",
        speculative_config=K3_DSPARK_SPECULATIVE_CONFIG,
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
        "--k3-checkpoint",
        default=os.environ.get(
            "KIMI_K3_CKPT",
            _first_available_path(K3_CHECKPOINT_PATH, OCI_AGA_K3_CHECKPOINT_PATH),
        ),
        help="Kimi K3 checkpoint path (or set KIMI_K3_CKPT).",
    )
    parser.add_argument(
        "--k3-dspark-checkpoint",
        default=os.environ.get(
            "KIMI_K3_DSPARK_CKPT",
            _first_available_path(
                K3_DSPARK_CHECKPOINT_PATH,
                OCI_AGA_K3_DSPARK_CHECKPOINT_PATH,
            ),
        ),
        help="Kimi K3 DSpark checkpoint path (or set KIMI_K3_DSPARK_CKPT).",
    )
    parser.add_argument(
        "--dsv4-dspark-checkpoint",
        default=os.environ.get(
            "DSV4_DSPARK_CKPT",
            _first_available_path(
                DSV4_DSPARK_CHECKPOINT_PATH,
                OCI_AGA_DSV4_DSPARK_CHECKPOINT_PATH,
            ),
        ),
        help="DeepSeek V4 Pro DSpark checkpoint path (or set DSV4_DSPARK_CKPT).",
    )
    parser.add_argument(
        "--guard-model-root",
        default=os.environ.get(
            "TRTLLM_GUARD_MODEL_ROOT",
            _path_for_this_cluster(
                BRANCH_GUARD_MODEL_ROOT,
                (OCI_AGA_MARKER, OCI_AGA_BRANCH_GUARD_MODEL_ROOT),
            ),
        ),
        help="Writable directory used to stage model and dataset symlinks.",
    )
    parser.add_argument(
        "--mounts",
        default=_default(
            _path_for_this_cluster(
                CONTAINER_MOUNTS,
                (OCI_AGA_MARKER, OCI_AGA_CONTAINER_MOUNTS),
            ),
            "TRTLLM_CONTAINER_MOUNTS",
        ),
        help=(
            "Pyxis container mounts (default: /lustre:/lustre on hecate, "
            "/scratch:/scratch on oci-jhb, where the lustre mount is /scratch/fsw)."
        ),
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
        default="00:50:00",
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


def _requires_flashinfer_cubins(job: GuardJob) -> bool:
    return (
        job.model_name == "Kimi-K3" and job.draft_model_name is not None
    ) or job.speculative_config is not None


def _validate_args(args: argparse.Namespace, jobs: Sequence[GuardJob]) -> None:
    args.repo_root = args.repo_root.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    if args.image:
        args.image = str(Path(args.image).expanduser().resolve())
    if args.model_root:
        args.model_root = str(Path(args.model_root).expanduser().resolve())
    if args.k3_checkpoint:
        args.k3_checkpoint = str(Path(args.k3_checkpoint).expanduser().resolve())
    if args.k3_dspark_checkpoint:
        args.k3_dspark_checkpoint = str(Path(args.k3_dspark_checkpoint).expanduser().resolve())
    if args.dsv4_dspark_checkpoint:
        args.dsv4_dspark_checkpoint = str(Path(args.dsv4_dspark_checkpoint).expanduser().resolve())
    args.guard_model_root = str(Path(args.guard_model_root).expanduser().resolve())
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

    if (
        not args.dry_run
        and any(_requires_flashinfer_cubins(job) for job in jobs)
        and not os.environ.get("URM_TOKEN")
    ):
        raise ValueError("URM_TOKEN must be set before submitting a K3 D-Spark test")

    if args.image and not Path(args.image).is_file() and not args.dry_run:
        raise ValueError(f"Container image does not exist: {args.image}")
    if args.model_root and not Path(args.model_root).is_dir() and not args.dry_run:
        raise ValueError(f"Model root does not exist: {args.model_root}")
    if any(job.name.startswith("k3-") for job in jobs):
        if not args.k3_checkpoint:
            raise ValueError("Missing required Kimi K3 checkpoint: --k3-checkpoint")
        if not Path(args.k3_checkpoint).is_dir() and not args.dry_run:
            raise ValueError(f"Kimi K3 checkpoint does not exist: {args.k3_checkpoint}")
    if any(job.draft_model_name is not None or job.speculative_config is not None for job in jobs):
        if not args.k3_dspark_checkpoint:
            raise ValueError("Missing required Kimi K3 DSpark checkpoint: --k3-dspark-checkpoint")
        if not args.dry_run and not Path(args.k3_dspark_checkpoint).is_dir():
            raise ValueError(
                f"Kimi K3 DSpark checkpoint does not exist: {args.k3_dspark_checkpoint}"
            )
    if any(job.model_name == "DeepSeek-V4-Pro-DSpark" for job in jobs):
        if not args.dsv4_dspark_checkpoint:
            raise ValueError(
                "Missing required DeepSeek V4 Pro DSpark checkpoint: --dsv4-dspark-checkpoint"
            )
        if not Path(args.dsv4_dspark_checkpoint).is_dir() and not args.dry_run:
            raise ValueError(
                f"DeepSeek V4 Pro DSpark checkpoint does not exist: {args.dsv4_dspark_checkpoint}"
            )

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


def _shell_env_lines(
    args: argparse.Namespace, include_flashinfer_repository: bool = False
) -> list[str]:
    values = {
        "LLM_MODELS_ROOT": args.model_root,
        "HF_HOME": os.environ.get(
            "HF_HOME",
            _path_for_this_cluster(HF_CACHE, (OCI_AGA_MARKER, OCI_AGA_HF_CACHE)),
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
    if include_flashinfer_repository:
        lines.extend(
            [
                'test -n "${URM_TOKEN:-}"',
                'export URM_USER="${USER}"',
                f'export FLASHINFER_CUBINS_REPOSITORY="{FLASHINFER_CUBINS_REPOSITORY}"',
            ]
        )
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
    install_shell = "; ".join(commands)
    # The lock must sit on shared storage, not /tmp. enroot mounts /tmp as a
    # per-node tmpfs, so a /tmp lock only serialises the ranks within one node
    # while every node still runs "pip install -e" against the same shared
    # checkout at once. Concurrent editable installs then clobber each other's
    # build metadata in the repo and the losers die with "is not a valid
    # editable requirement" and a working directory that no longer exists.
    lock_path = f"{args.repo_root}/.branch_guard_install_$SLURM_JOB_ID.lock"
    return [
        f'flock "{lock_path}" bash -lc {shlex.quote(install_shell)}',
    ]


def _model_setup_lines(args: argparse.Namespace, job: GuardJob) -> list[str]:
    if job.model_name is None or job.checkpoint_path is None:
        return []

    if job.name.startswith("k3-"):
        checkpoint_path = args.k3_checkpoint
    elif job.model_name == "DeepSeek-V4-Pro-DSpark":
        checkpoint_path = args.dsv4_dspark_checkpoint
    else:
        checkpoint_path = job.checkpoint_path
    lines = [
        f"guard_model_root={shlex.quote(args.guard_model_root)}/"
        f"{shlex.quote(job.name)}/job_${{SLURM_JOB_ID}}",
        f"guard_checkpoint={shlex.quote(checkpoint_path)}",
        f"guard_dataset_root={shlex.quote(str(Path(args.model_root) / 'datasets'))}",
        'test -d "$guard_checkpoint"',
        'test -d "$guard_dataset_root"',
        'mkdir -p "$guard_model_root"',
        f'ln -sfn "$guard_checkpoint" "$guard_model_root/{job.model_name}"',
        'ln -sfn "$guard_dataset_root" "$guard_model_root/datasets"',
        'export LLM_MODELS_ROOT="$guard_model_root"',
    ]
    if job.draft_model_name is not None:
        lines.extend(
            [
                f"guard_draft_checkpoint={shlex.quote(job.draft_checkpoint_path or args.k3_dspark_checkpoint)}",
                'test -d "$guard_draft_checkpoint"',
                f'ln -sfn "$guard_draft_checkpoint" "$guard_model_root/{job.draft_model_name}"',
            ]
        )
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
    include_flashinfer_repository = _requires_flashinfer_cubins(job)
    lines = [
        "set -eu",
        *_shell_env_lines(args, include_flashinfer_repository),
    ]
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
    if args.partition != "batch-xdr":
        command.append(f"--gpus-per-node={job.gpus_per_node}")
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
    output = log_dir.parent / f"{log_dir.name}-disaggr_torch_branch_guard.slurm"
    script = source.read_text()
    install_command = "pip install -e .[devel]"
    if script.count(install_command) != 1:
        raise RuntimeError(
            "Unable to locate the TensorRT-LLM editable-install command "
            "in the benchmark Slurm script."
        )
    script = script.replace(
        install_command,
        (
            f"{install_command} && python3 -m pip install "
            "flashinfer_python==0.6.18+cf3c3a3e.nvinternal.rubin.0.8dev.devel.cu.64035134 "
            "--index-url https://gitlab-master.nvidia.com/api/v4/projects/179191/packages/pypi/simple "
            "--no-deps --force-reinstall --break-system-packages"
        ),
    )
    output.write_text(script)
    output.chmod(source.stat().st_mode)
    return output


def _benchmark_config(args: argparse.Namespace, job: GuardJob) -> tuple[Path, Path]:
    source = args.repo_root / job.selectors[0]
    config = deepcopy(yaml.safe_load(source.read_text()))
    run_id = int(time.time())
    log_dir = args.log_dir / job.name / f"run-{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    config["slurm"]["script_file"] = str(_write_benchmark_slurm_script(args.repo_root, log_dir))

    config["slurm"]["partition"] = args.partition
    config["slurm"]["account"] = args.account
    config["slurm"]["job_time"] = args.multi_node_time
    config["slurm"]["job_name"] = job.name
    # batch-xdr allocates one GPU per task and rejects a GPU GRES request.
    # Other GPU partitions need the per-node request explicitly.
    config["slurm"]["extra_args"] = (
        "" if args.partition == "batch-xdr" else f"--gpus-per-node={job.gpus_per_node}"
    )
    config["environment"]["container_image"] = args.image
    config["environment"]["model_path"] = args.k3_checkpoint
    config["environment"]["trtllm_repo"] = str(args.repo_root) if args.install_from_repo else ""
    config["environment"]["build_wheel"] = False
    config["environment"]["trtllm_wheel_path"] = ""
    config["environment"].pop("work_dir", None)
    config["environment"]["log_dir"] = str(log_dir)
    mounts = [f"{args.repo_root}:{args.repo_root}:rw"]
    mounts.extend(mount for mount in args.mounts.split(",") if mount)
    config["environment"]["container_mount"] = ",".join(dict.fromkeys(mounts))

    # The branch guard is an accuracy run, not the optional synthetic serving
    # benchmark.  Keeping the benchmark disabled also avoids requiring its
    # unrelated 8K/1K JSONL dataset on the compute nodes.
    config["benchmark"]["enable_benchmark"] = False
    config.setdefault("accuracy", {})["enable_accuracy_test"] = True
    # lm-eval's --seed is "python,numpy,torch,fewshot" and its fewshot default
    # is 1234, which selects different 5-shot exemplars than every K3 reference
    # number was measured with. benchmark_kimi_k3_dep8_gsm8k.yaml pins
    # "0,1234,1234,0" and records the cost of not doing so: fewshot seed 0 gave
    # 96.36 / 96.40 / 96.47 against seed 1234's 95.07 / 95.68 / 95.68 / 95.68,
    # i.e. about 0.8 points, with the questions identical and all 1319 prompts
    # different. Job 559983 ran a perf recipe, whose accuracy block is disabled
    # boilerplate carrying no seed, and scored 95.83 -- squarely in the
    # wrong-seed band and not comparable to any anchor.
    #
    # Enabling the accuracy test therefore has to carry the seed with it. A
    # perf recipe will never supply one, and nothing in the output says the
    # number is incomparable, so leaving this to the recipe is a trap that
    # reads as a model regression.
    for task_config in config["accuracy"].setdefault("tasks", {}).values():
        task_config.setdefault("extra_kwargs", {}).setdefault("seed", _LM_EVAL_SEED)
    if job.speculative_config is not None:
        speculative_config = deepcopy(job.speculative_config)
        speculative_config["speculative_model"] = (
            job.draft_checkpoint_path or args.k3_dspark_checkpoint
        )
        config["worker_config"]["gen"]["speculative_config"] = speculative_config
    # #19179 flipped KIMI_K3_FP8_WEIGHT_READ_MOE_MLP from on to off, so the
    # replicated shared-expert and latent MLP projections stay BF16 unless
    # something asks for FP8. They are not TP-sharded, so that is twice the
    # bytes on every rank. The perf guard has set this deliberately since
    # 76e0ef7212; this guard never did, which is the inconsistency being fixed
    # here.
    #
    # It does not fix the GB300 accuracy-preset OOM, which was measured as a
    # ~68 GiB transient during model creation, dominated by raw BF16 expert
    # staging -- nothing this variable touches.
    config["environment"]["worker_env_var"] = " ".join(
        [
            "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP=1",
            config["environment"].get("worker_env_var", ""),
        ]
    ).strip()
    if _requires_flashinfer_cubins(job):
        flashinfer_repository = FLASHINFER_CUBINS_REPOSITORY.replace(
            "${URM_USER}", os.environ["USER"]
        )
        config["environment"]["worker_env_var"] = " ".join(
            [
                f"URM_USER={os.environ['USER']}",
                f"FLASHINFER_CUBINS_REPOSITORY={flashinfer_repository}",
                config["environment"].get("worker_env_var", ""),
            ]
        ).strip()
    if job.name in {
        "k3-disaggregated-dspark-gsm8k",
        "k3-disaggregated-dspark-helix-cp8-gsm8k",
        "k3-disaggregated-dspark-dep16-gsm8k",
    }:
        for role in ("ctx", "gen"):
            config["worker_config"][role]["kv_cache_config"]["dtype"] = "fp8"
    if job.name == "k3-disaggregated-dspark-helix-cp8-gsm8k":
        gen_config = config["worker_config"]["gen"]
        gen_config.update(
            {
                "tensor_parallel_size": 1,
                "enable_attention_dp": False,
                "context_parallel_size": 8,
                "cp_config": {
                    "cp_type": "HELIX",
                    "tokens_per_block": 64,
                },
                "max_batch_size": 16,
                "max_num_tokens": 256,
            }
        )
    if job.name in {
        "k3-disaggregated-dep16-gsm8k",
        "k3-disaggregated-dspark-dep16-gsm8k",
    }:
        # The normal SBSA GB300 image does not provide the MegaMoE DeepGEMM
        # implementation selected by the performance recipe. Use K3's
        # supported TRTLLM MoE backend for this correctness guard.
        for role in ("ctx", "gen"):
            config["worker_config"][role].setdefault("moe_config", {})["backend"] = "TRTLLM"
        config["environment"]["server_health_timeout"] = 5400
    accuracy_env = config.setdefault("accuracy", {}).setdefault("env_var", {})
    accuracy_env["HF_HOME"] = os.environ.get(
        "HF_HOME", _path_for_this_cluster(HF_CACHE, (OCI_AGA_MARKER, OCI_AGA_HF_CACHE))
    )
    accuracy_env["LLM_MODELS_ROOT"] = args.model_root
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

    runtime_config = args.log_dir / f"{job.name}-config-{run_id}.yaml"
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
    if _BENIGN_SEVERITY_RE.search(line) is not None:
        return None
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


def _slurm_batch_step_completed(job_id: str) -> bool:
    """Whether the job's batch script ran to completion.

    The allocation's own state is not a verdict. The benchmark harness tears
    its allocation down once the workload finishes, so a clean run ends
    CANCELLED -- both a successful reference run and our own runs do. Grading
    on ``state == "COMPLETED"`` reports those as failures. The batch step is
    the thing that actually ran the work, and it records COMPLETED whatever
    happens to the allocation afterwards.
    """
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", f"{job_id}.batch", "--format=State"],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        state = line.strip().split("|", 1)[0].split("+", 1)[0]
        if state:
            return state == "COMPLETED"
    return False


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
            if state != "COMPLETED" and not _slurm_batch_step_completed(job_id):
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
