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
"""Run a pinned Kimi-K3 disaggregated performance guard through bench-trtllm-disagg.

This is the K3 counterpart of ``run_branch_guard_perf.py``. The two are kept
separate on purpose: the DSV4 guard drives ``srtctl`` against SHA-pinned
srt-slurm recipes and reads AIPerf CSV, whereas the measured K3 points live in
the ``bench-trtllm-disagg`` harness (``trtllm_config.yaml`` + ``submit.py``) and
report through two different result formats. Folding both into one script would
have meant a mode flag on essentially every function.

``--preset`` selects one Python-defined bundle containing an immutable reference
workdir, topology, workload shape, reference metrics, and gate policy. The
reference ``trtllm_config.yaml`` is SHA-256 checked before use, so edits under
the shared reference tree cannot silently move the baseline.

Each preset pins one measured reference run:

* ``k3-genonly-helix8-c16-mtp5`` - GEN-only, helix8 / concurrency 16 / MTP5.
  Reference: job 605293 (``.../k3-nsys-fp4-vs-fp8/fp4-job605293-mr10675fix``),
  the post-MR!10675 FP4 point. Results are read from the ``trtllm`` bench
  client's ``result.json`` plus the per-iteration ``gen_only_0.txt`` log.
* ``k3-ctxonly-dep16-c96`` - CTX-only, dep16 / concurrency 96.
  Reference: job 586472 (``.../k3-ctx-baseline-job586472``), the 7.66 req/s
  baseline. Results are read from AIPerf's ``profile_export_aiperf.csv``.
* ``k3-ctxonly-dwdp16-c128`` - CTX-only with DWDP16 expert prefetch: 16 x TP1
  context servers plus one helix8 decode worker, concurrency 128.
  Reference: job 624450 (``.../k3_dwdp_ima_20260922/624450_oldimg_PASS_control``),
  the one PASSING control in taizhongw's DWDP illegal-memory-access archive.
  EVALUATION-ONLY -- the bench revision pinned here has no DWDP launcher, so
  this preset grades result directories and cannot submit. See its
  ``single_string`` comment for what would have to land first.

Reference metrics are per cluster, not per preset. The same config on R200
(hecate) and on GB300 (oci-jhb) produces different absolute numbers, so each
preset carries one ``ClusterReference`` per cluster it has been measured on and
the guard picks the set matching wherever it is running. The cluster is detected
from Slurm's ``ClusterName``, falling back to a filesystem marker unique to each
site; detection never guesses, because grading a GB300 run against R200 numbers
would report a confident verdict about the wrong hardware. ``--cluster`` forces
it. Cluster-dependent defaults -- account, partition, constraint, guard home,
benchmark repo -- follow the detected cluster too.

Useful modes:

* ``--list-presets``: show available presets without requiring runtime inputs.
* ``--result-dir DIR``: evaluate an existing result directory without submitting.
* ``--dry-run``: prepare and validate the config without submitting.
* ``--cluster NAME``: force the cluster instead of detecting it.
* ``--establish-reference``: run and report metrics without gating them, for a
  cluster that has no baseline yet. Prints a pasteable ``ClusterReference``.

Examples::

    # Show both presets.
    python3 scripts/run_branch_guard_perf_k3.py --list-presets

    # Re-evaluate the reference run itself (self-check; every gate must pass).
    python3 scripts/run_branch_guard_perf_k3.py \
        --preset k3-genonly-helix8-c16-mtp5 --self-check

    # Prepare and validate without submitting.
    python3 scripts/run_branch_guard_perf_k3.py \
        --preset k3-genonly-helix8-c16-mtp5 \
        --image /lustre/share/coreai_comparch_trtllm/lizhiz/rubin_k3/540bf9e6e6/trtllm.sqsh \
        --dry-run

    # Submit, wait, and gate.
    python3 scripts/run_branch_guard_perf_k3.py \
        --preset k3-genonly-helix8-c16-mtp5 \
        --image /lustre/share/coreai_comparch_trtllm/lizhiz/rubin_k3/540bf9e6e6/trtllm.sqsh

    # Gate an already-finished run directory.
    python3 scripts/run_branch_guard_perf_k3.py \
        --preset k3-ctxonly-dep16-c96 --result-dir /path/to/run

    # First run on a cluster with no baseline: measure, do not gate.
    python3 scripts/run_branch_guard_perf_k3.py \
        --preset k3-ctxonly-dep16-c96 --establish-reference \
        --image /path/to/base.sqsh --wheel /path/to/tensorrt_llm.whl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_LUSTRE_ROOT = Path("/lustre/share")
K3_REFERENCE_ROOT = SHARED_LUSTRE_ROOT / "coreai_comparch_infbench" / "taizhongw"
# These presets need 6 nodes at once. On a congested batch-xdr (measured
# 2026-09-18: 1173 queued jobs, 66 running, 20 idle nodes) a 6 h wait
# expired before the job was ever scheduled -- Elapsed 00:00:00, Start
# None -- and the guard then cancelled its own still-queued job. Wait a
# full day by default; use --wait-timeout to shorten it.
DEFAULT_WAIT_TIMEOUT_SECONDS = 24 * 60 * 60
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

GEN_ONLY = "gen_only"
CTX_ONLY = "ctx_only"

HECATE = "hecate"
OCI_JHB = "oci-jhb"


@dataclass(frozen=True)
class Cluster:
    """One cluster the guard knows how to submit to and grade against.

    Reference numbers are hardware-specific, so the cluster identity is what
    selects them. Everything that differs between the two sites lives in one
    record rather than being sprinkled through argparse defaults, because a
    half-applied cluster switch -- oci-jhb paths with R200 reference numbers --
    produces a plausible-looking but meaningless verdict.
    """

    name: str
    # Decoder/compute part name. Used for labelling and for the harness's
    # log-directory naming; the harness derives nothing functional from it.
    gpu_name: str
    # AIHub-style clusters reject a GPU job that does not ask for an explicit
    # --gres; the harness adds one when this is set. Measured on oci-jhb:
    # "srun: error: Cannot find GPU specification, you may not submit a job not
    # requesting GPUs in a non-CPU partition, partition: batch".
    on_oci: bool
    # Slurm's own ClusterName, the primary detection signal.
    slurm_cluster_name: str
    # A directory that exists on this cluster and on no other known one.
    # Deliberately NOT "/lustre": on oci-jhb /lustre is a symlink to /scratch,
    # so /lustre and /lustre/fsw exist on both sites and discriminate nothing.
    # /lustre/share exists only on hecate; /scratch/fsw only on oci-jhb.
    marker: Path
    account: str
    partition: str
    # Empty means "submit without --constraint"; the harness treats a falsy
    # CONSTRAINT as absent.
    constraint: str
    # Bind mounts for the benchmark containers, in the harness's
    # CONTAINER_MOUNTS form. Left to its default this is "/lustre/", which on
    # oci-jhb names a symlink to /scratch rather than the lustre mount itself,
    # so every /scratch/fsw path in the config would be missing inside the
    # container.
    container_mounts: str
    guard_home: Path
    benchmark_repo: Path


CLUSTERS: dict[str, Cluster] = {
    cluster.name: cluster
    for cluster in (
        Cluster(
            name=HECATE,
            gpu_name="R200",
            on_oci=False,
            slurm_cluster_name="hecate",
            marker=Path("/lustre/share/coreai_comparch_trtllm"),
            account="coreai_comparch_trtllm",
            partition="batch-xdr",
            constraint="cr",
            container_mounts="/lustre/",
            guard_home=Path("/lustre/share/coreai_comparch_trtllm/lizhiz/k3_guard_home"),
            benchmark_repo=Path("/lustre/fsw/coreai_comparch_trtllm/lizhiz/bench-trtllm-disagg-k3"),
        ),
        # No preset has a GB300 reference yet. Five runs have been made here and
        # each one moved the failure further in, so the chain is recorded rather
        # than rediscovered. Every fixed item below is confirmed by a later run
        # observing zero of its signature.
        #
        # 1. FIXED (b3d3ad30d4). Jobs 530847 / 530848: "FP4 MLA fused Q
        #    quantization eligibility changed before launch" --
        #    can_fuse_fp4_mla_q_quant carried a redundant
        #    get_sm_version() == 107 while uses_fp4_mla_attention carried no SM
        #    check at all. Zero occurrences from job 531579 on.
        #
        # 2. FIXED (K3_GEN_MOE_BACKEND=CUTEDSL, see this preset's overrides).
        #    Jobs 531579 / 531580: the GEN worker asked for MoE backend
        #    CUTEDSL_FC12, which CuteDslFc12FusedMoE.can_implement() rejects on
        #    SM103. Zero occurrences from job 532131 on.
        #
        # 3. FIXED (ctx_gpu_memory_fraction 0.53, gen-only preset). Job 531579's
        #    context worker OOMed at 0.8. Job 532131 measured a 63.5849 GiB
        #    quota against the 63.58 GiB the override was derived for, and zero
        #    OOMs on that worker.
        #
        # 4. OPEN, gen-only preset, not configurable around. Job 532131's GEN
        #    worker raised, ten times,
        #      NotImplementedError: FP4 MLA Helix softmax stats require the
        #      cutedsl attention backend.
        #    from fp4_mla/__init__.py:4032-4035, which raises whenever
        #    softmax_stats_tensor is not None and the selected backend is not
        #    cutedsl. Helix needs those stats to combine softmax across ranks,
        #    and SM103 selects "triton". This is not a gate someone tightened:
        #    grepping softmax_stats in fp4_mla_kernels.py returns nothing, so
        #    the Triton path has no such kernel to call. Relaxing the check
        #    would reach an implementation that does not exist. Until that
        #    kernel is written, k3-genonly-helix8-c16-mtp5 cannot run on GB300,
        #    and it cannot be configured around either -- helix8 is the point of
        #    the preset.
        #
        # 5. OPEN, ctx-only preset, needs a lever the guard does not have. Job
        #    532135 got its CONTEXT server up and serving -- the harness printed
        #    "Waiting for servers, context: []" -- and then its GEN worker OOMed
        #    at gen-side free_gpu_memory_fraction 0.8. This was invisible until
        #    4 and 2 were fixed, because the GEN worker used to die earlier.
        #    Ledger, per rank: 67.32 GiB free at the kv_cache stage, quota
        #    min(0.8 x 67.32, max_tokens 56.69) = 53.85 GiB -- so unlike case 3
        #    the fraction already binds here and lowering it does do something.
        #    "Additional executor resources" then wanted 27.53 GiB against 20.10
        #    GiB free on the best rank and 19.42 GiB on the worst: deficits of
        #    7.43 and 8.11 GiB. 0.60 would return 13.46 GiB and clear the worst
        #    rank by 5.35 GiB; 0.68 lands 0.03 GiB short of it. The larger
        #    footprint is plausibly a direct cost of 2: CUTEDSL is the two-op
        #    backend and materialises intermediates the fused FC12 kernel does
        #    not. --ctx-gpu-memory-fraction rewrites ctx_config only, by
        #    design, so there is no gen-side equivalent to set today.
        #
        # Verified working here regardless: detection, submission with the right
        # partition and gres, the reference CTX cost triple and
        # KIMI_K3_FP8_WEIGHT_READ_MOE_MLP=1 reaching the generated config and
        # env_vars.json, and the branch wheel force-reinstalling on all six
        # nodes.
        Cluster(
            name=OCI_JHB,
            gpu_name="GB300",
            on_oci=True,
            slurm_cluster_name="oci-jhb-slurm-1",
            marker=Path("/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm"),
            account="coreai_comparch_trtllm",
            partition="batch",
            constraint="",
            # /scratch, not /scratch/fsw: the harness mounts this path as-is and
            # the guard home, checkpoints and datasets all live under it.
            container_mounts="/scratch/",
            guard_home=Path(
                "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm"
                "/users/lizhiz/k3_guard_home"
            ),
            benchmark_repo=Path(
                "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm"
                "/users/lizhiz/bench-trtllm-disagg-k3"
            ),
        ),
    )
}


def _slurm_cluster_name() -> str | None:
    """Slurm's configured ClusterName, or None if Slurm cannot be reached.

    This works from a login node, which is the point: the submitting host has
    no GPU, so torch.cuda and nvidia-smi cannot say what hardware the job will
    land on. ClusterName is served by the local controller and is authoritative
    for "which cluster am I submitting into".
    """
    result = subprocess.run(
        ["scontrol", "show", "config"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "ClusterName":
            return value.strip() or None
    return None


def _detect_cluster(explicit: str | None = None) -> Cluster:
    """Resolve which cluster this invocation is running on.

    Order: explicit request, then Slurm's ClusterName, then a unique
    filesystem marker -- the same filesystem-existence idiom
    ``scripts/run_branch_guard_tests.py`` uses in ``_first_available_path``.

    Never guesses. Grading a GB300 run against R200 references would report a
    confident number for the wrong hardware, which is strictly worse than
    refusing to run, so an unrecognised environment raises.
    """
    if explicit:
        if explicit not in CLUSTERS:
            raise ValueError(
                f"Unknown cluster {explicit!r}; known clusters: {', '.join(sorted(CLUSTERS))}"
            )
        return CLUSTERS[explicit]

    slurm_name = _slurm_cluster_name()
    if slurm_name is not None:
        for cluster in CLUSTERS.values():
            if cluster.slurm_cluster_name == slurm_name:
                return cluster

    matches = [cluster for cluster in CLUSTERS.values() if cluster.marker.is_dir()]
    if len(matches) == 1:
        return matches[0]

    probed = ", ".join(
        f"{cluster.name} (ClusterName {cluster.slurm_cluster_name!r}, marker {cluster.marker})"
        for cluster in CLUSTERS.values()
    )
    detail = (
        f"Slurm ClusterName {slurm_name!r}"
        if slurm_name is not None
        else "Slurm ClusterName unavailable"
    )
    if len(matches) > 1:
        detail += f"; markers matched {', '.join(cluster.name for cluster in matches)}"
    else:
        detail += "; no cluster marker matched"
    raise ValueError(
        f"Cannot identify the cluster ({detail}). Probed: {probed}. "
        f"Pass --cluster explicitly. Refusing to guess: reference metrics are "
        f"hardware-specific and grading against the wrong set is worse than not "
        f"running at all."
    )


@dataclass(frozen=True)
class Gate:
    """One pass/fail comparison against a reference-derived limit.

    ``minimum`` selects the direction: throughput-like metrics must stay at or
    above their limit, latency-like metrics at or below it.
    """

    metric: str
    reference: float
    minimum: bool
    ratio: float
    absolute_limit: float | None = None

    def limit(self) -> float:
        scaled = self.reference * self.ratio
        if self.absolute_limit is None:
            return scaled
        if self.minimum:
            return max(scaled, self.absolute_limit)
        return min(scaled, self.absolute_limit)


@dataclass(frozen=True)
class ClusterOverrides:
    """Operating-point changes a preset needs before it can run on a cluster.

    Deliberately separate from ClusterReference. A reference records what was
    measured; this records what had to change to measure anything at all, and
    it has to exist *before* the first run, which is exactly when there is no
    reference yet.

    Every entry here moves the preset away from its R200 operating point, so
    each carries the reason inline and the reason is echoed at submission time
    and into the pinned reference's provenance. A GB300 number produced under
    these overrides is not comparable to the R200 number for the same preset,
    and nothing about that should have to be reconstructed later.
    """

    cluster: str
    reason: str
    # Merged over the preset's own generator_env, so a name here wins and every
    # other name the preset pins is preserved.
    generator_env: tuple[tuple[str, str], ...] = ()
    # Applied unless --ctx-gpu-memory-fraction says otherwise.
    ctx_gpu_memory_fraction: float | None = None
    # Likewise for the decode worker. Separate from the ctx value because the
    # two workers OOM on different things and at different fractions.
    gen_gpu_memory_fraction: float | None = None


@dataclass(frozen=True)
class ClusterReference:
    """One cluster's measured baseline for a preset, plus its gate policy.

    Reference metrics are a property of (preset, hardware), not of the preset
    alone: the same config on R200 and on GB300 produces different absolute
    numbers, so a single set of gates cannot serve both. Keeping the numbers,
    their provenance, and the pass/fail policy in one record is what makes it
    possible to add a cluster without silently reusing another one's baseline.
    """

    cluster: str
    # Where the numbers came from: job ids, date, image, wheel. Prose, but it
    # is the only thing that lets a future reader decide whether a baseline is
    # still comparable to what they are about to run.
    provenance: str
    benchmark_duration_seconds: float
    gates: tuple[Gate, ...]
    minimum_duration_ratio: float = 0.95
    caveat: str = ""


@dataclass(frozen=True)
class K3Preset:
    """Immutable reference identity, topology, workload, and gate policy.

    A preset deliberately keeps these together. A reference number without its
    matching config, topology, and workload would produce a plausible-looking
    but invalid comparison -- which is exactly the failure mode the CTX-only
    reference's ``caveat`` guards against.
    """

    name: str
    description: str
    mode: str
    reference_dir: Path
    reference_config_sha256: str
    # The *source* agent config in bench-trtllm-disagg, pinned by revision and
    # digest. submit.py consumes this input form; the reference workdir's
    # trtllm_config.yaml is the expanded output form and is not submittable.
    source_config: Path
    source_revision: str
    source_sha256: str
    # The gen_configs / ctx_configs row selecting the measured point, in
    # submit.py's -s format. None means the preset cannot be submitted.
    single_string: str | None
    sweep_key: str
    # submit.py resolves the worker-config generator from the *config file's*
    # directory (<dir>/gen_worker_config.py, then <dir>/../gen_worker_config.py,
    # then the repo root). Only the model-specific one understands helix, so it
    # is pinned and staged beside the prepared config.
    generator_config: Path
    generator_sha256: str
    # The generator reads these at import time and they change the emitted
    # worker config, so they are part of the preset's identity rather than
    # ambient environment. K3_MAMBA_SSM_DTYPE is what puts
    # mamba_ssm_cache_dtype into the config at all; without it the KDA
    # recurrent state silently falls back to the runtime default.
    generator_env: tuple[tuple[str, str], ...]
    ctx_nodes: int
    gen_nodes: int
    gpus_per_node: int
    ctx_gpus: int
    gen_gpus: int
    benchmark_concurrency: int
    input_length: int
    output_length: int
    # Measured baselines, one per cluster the preset has actually been run on.
    # A cluster absent from here has no baseline and cannot be graded; see
    # reference_for().
    references: tuple[ClusterReference, ...]
    # Operating-point changes needed to run at all on a given cluster. Empty
    # for a cluster whose hardware supports the preset as written.
    overrides: tuple[ClusterOverrides, ...] = ()
    # Worker env the generator does not know how to emit. It builds
    # worker_envs_vars as a literal with no passthrough, so a name it does not
    # mention cannot otherwise reach the workers from a preset.
    worker_env: tuple[tuple[str, str], ...] = ()

    @property
    def total_gpus(self) -> int:
        return self.ctx_gpus + self.gen_gpus

    @property
    def reference_config(self) -> Path:
        return self.reference_dir / "trtllm_config.yaml"

    def overrides_for(self, cluster: str) -> ClusterOverrides | None:
        """The operating-point overrides for ``cluster``, if it needs any."""
        for override in self.overrides:
            if override.cluster == cluster:
                return override
        return None

    def generator_env_for(self, cluster: str) -> tuple[tuple[str, str], ...]:
        """The preset's generator env with this cluster's overrides applied."""
        merged = dict(self.generator_env)
        override = self.overrides_for(cluster)
        if override is not None:
            merged.update(override.generator_env)
        return tuple(merged.items())

    def reference_for(self, cluster: str) -> ClusterReference:
        """The baseline measured on ``cluster``.

        Raises rather than falling back to any other cluster's numbers. The
        fallback would be a silent hardware mismatch, and the whole point of
        per-cluster references is that such a comparison is meaningless.
        """
        for reference in self.references:
            if reference.cluster == cluster:
                return reference
        known = ", ".join(reference.cluster for reference in self.references) or "none"
        raise ValueError(
            f"Preset {self.name!r} has no reference measured on cluster "
            f"{cluster!r} (has: {known}). Establish one first with "
            f"--establish-reference, then pin it into this script."
        )


# ---------------------------------------------------------------------------
# Presets
#
# Reference values below were read directly out of the two reference runs, not
# transcribed from prose. The GEN-only device_step_time_ms reference (28.929 ms)
# reproduces the 28.93 ms quoted in that run's README using the derivation
# implemented in _gen_device_step_time_ms().
# ---------------------------------------------------------------------------

GENONLY_REFERENCE_DIR = K3_REFERENCE_ROOT / "k3-nsys-fp4-vs-fp8" / "fp4-job605293-mr10675fix"
CTXONLY_REFERENCE_DIR = K3_REFERENCE_ROOT / "k3-ctx-baseline-job586472"
# taizhongw's GB300 sweep, on oci-jhb rather than the hecate reference share.
# taizhongw's GB300 gen-only sweep. Same batch as the CTX-only FP8 reference.
GENONLY_GB300_REFERENCE_DIR = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_aarwlt/users/taizhongw"
    "/bench-trtllm-disagg/kimi-k3-nvfp4-agent"
    "/bm_kimi-k3-nvfp4-genonly-turnbatch-sol-996608-512-ratio1"
    "-dd78bbce-mr26-mr35-20260913-GB300"
)
# The DWDP CTX-only reference is the PASSING control of taizhongw's 2026-09-22
# archive, not one of its illegal-memory-access reproductions. That archive's
# README is explicit that 626177 / 626311 / 628753 / 629318 all die in MR!10698's
# CuTeDSL ``gather_mxfp8`` with CUDA_ERROR_ILLEGAL_ADDRESS (700), so a reference
# taken from any of them would be a reference for a broken state. 624450 is the
# only run in the archive that produced numbers at all.
DWDP_CTXONLY_REFERENCE_DIR = (
    K3_REFERENCE_ROOT / "k3_dwdp_ima_20260922" / "624450_oldimg_PASS_control"
)
CTXONLY_FP8_REFERENCE_DIR = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_aarwlt/users/taizhongw"
    "/bench-trtllm-disagg/kimi-k3-nvfp4-agent"
    "/bm_kimi-k3-nvfp4-agentx-ctxonly-e2e-996608-2-dd78bbce-mr26-mr35-20260913-GB300"
)

# bench-trtllm-disagg revision carrying the kimi-k3-nvfp4-agent configs.
K3_BENCH_REVISION = "0c7dcd10fa32e9ea45bd5acdd2b2a5d049ca1b59"
K3_AGENT_CONFIG_DIR = Path("kimi-k3-nvfp4-agent/agent_configs")

# Both reference runs logged "reading 368 MoE-layer MLP projections
# (shared-expert + latent) at FP8 block-scale", because the code they ran
# defaulted KIMI_K3_FP8_WEIGHT_READ_MOE_MLP to "1". #19179 flipped that default
# to "0" (modeling_kimi_linear.py: os.environ.get(..., "0")), so the branch
# keeps those replicated projections in BF16 -- twice the bytes, on every rank,
# since they are not TP-sharded. Nothing in the harness sets the name, and the
# nvfp4 generator deliberately does not ("both default ON" per its comment), so
# the guard has to.
#
# This does not restore the whole gap: #19179 also deleted
# _convert_mla_projections_to_fp8_weight_read outright, and the 96 MLA
# q_a/q_b/o/g projections it converted (~2.4 GiB/rank at DEP8 by its own
# comment) cannot be brought back by any environment variable.
_REFERENCE_WORKER_ENV = (("KIMI_K3_FP8_WEIGHT_READ_MOE_MLP", "1"),)

K3_PRESETS: dict[str, K3Preset] = {
    preset.name: preset
    for preset in (
        K3Preset(
            name="k3-genonly-helix8-c16-mtp5",
            description=(
                "Kimi-K3 NVFP4 GEN-only, ctx dep16 + gen helix8, concurrency 16, "
                "MTP5 (job 605293, post-MR!10675)"
            ),
            mode=GEN_ONLY,
            reference_dir=GENONLY_REFERENCE_DIR,
            reference_config_sha256="b6f34426c0be7bfba2d4c127f3eff1cba091f89f12cac7ec9c3c55277ac705be",
            source_config=K3_AGENT_CONFIG_DIR / "genonly_turnbatch_c16.yaml",
            source_revision=K3_BENCH_REVISION,
            source_sha256="6182e88e8b1ebad093cf6cdd426fed0624302412837814dee487200640e0c9ad",
            # gen_configs row "helix8 lbs16 mtp5":
            # [ctx_num, gen_num, rank_count, max_batch, max_num_tokens,
            #  adp|helix, gmem, mtp, eplb, concurrency]
            single_string="1,1,8,16,96,helix,0.8,5,0,16",
            sweep_key="gen_configs",
            generator_config=Path("kimi-k3-nvfp4-agent/gen_worker_config.py"),
            generator_sha256=("3c7d08a94dfc7e112ee1e52e1fb4fb0df8f1e9651e80dd9818a64301d642c9c3"),
            # Same CTX cost triple as the CTX-only preset: job 605293 set it
            # explicitly too, and the generator's defaults are different. The
            # gen-side gates are far less sensitive to CTX scheduling than
            # request throughput is, but the reference is the reference.
            generator_env=(
                ("K3_MAMBA_SSM_DTYPE", "float32"),
                ("K3_CTX_COST_KV_OFFSET", "35000"),
                ("K3_CTX_COST_KV_DEPTH_THRESHOLD", "250000"),
                ("K3_CTX_COST_PER_CHUNK_DEPTH_TOKENS", "3000"),
            ),
            ctx_nodes=4,
            gen_nodes=2,
            gpus_per_node=4,
            ctx_gpus=16,
            gen_gpus=8,
            benchmark_concurrency=16,
            input_length=996608,
            output_length=512,
            references=(
                ClusterReference(
                    cluster=HECATE,
                    provenance=(
                        "R200 / hecate. Job 605293, measured 2026-09, "
                        "image /lustre/share/coreai_comparch_infbench/taizhongw "
                        "post-MR!10675 FP4 build, no wheel overlay "
                        "(the run was made from a saved container)."
                    ),
                    benchmark_duration_seconds=116.28,
                    gates=(
                        Gate("output_throughput", 70.4478, minimum=True, ratio=0.95),
                        Gate("user_throughput", 4.40540, minimum=True, ratio=0.95),
                        Gate("mean_tpot_ms", 6.37775, minimum=False, ratio=1.10),
                        # Guards an MTP acceptance regression. Note the reference
                        # run pins TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=2.62,
                        # so this measures that the forced path still holds, not
                        # real acceptance.
                        Gate("avg_decoded_tokens_per_iter", 3.59331, minimum=True, ratio=0.95),
                        # Sharpest single signal on this point: the MR!10675
                        # KDA-replay fix moved it from 43.97 ms to 28.93 ms.
                        Gate("device_step_time_ms", 28.929, minimum=False, ratio=1.10),
                    ),
                ),
            ),
            overrides=(
                ClusterOverrides(
                    cluster=OCI_JHB,
                    reason=(
                        "GEN MoE backend CUTEDSL_FC12 does not exist on SM103: "
                        "CuteDslFc12FusedMoE.can_implement() rejects it "
                        "('targets Rubin (SM107), got SM103') because the "
                        "FC1+FC2-fused kernel is not ported. The factory "
                        "falls back to CutlassFusedMoE and the model then "
                        "refuses to run, which is correct -- that silent "
                        "fallback voided an earlier GB300 batch. "
                        "gen_worker_config.py names CUTEDSL, the two-op "
                        "parent of the same family, as the GB300 value. "
                        "GB300 therefore measures a DIFFERENT MoE kernel "
                        "than R200 and the two are not comparable."
                        " Separately, the CONTEXT worker OOMs here at "
                        "gpu_memory_fraction 0.8."
                    ),
                    generator_env=(("K3_GEN_MOE_BACKEND", "CUTEDSL"),),
                    # Derived from job 531579's own memory ledger, not picked.
                    # Per rank: 276.62 GiB total, 119.97 GiB free after the
                    # model, and the KV estimation pool took 68.12 GiB of it.
                    # The quota is min(fraction x free, what max_tokens needs)
                    # = min(95.98, 68.11), so max_tokens bound it and the
                    # fraction did not -- 0.8 -> 0.7 would have been a pure
                    # no-op, since 0.7 x 119.97 = 83.98 is still above 68.11.
                    # The fraction only starts binding below 0.5677.
                    # The failure is one stage later, in
                    # _no_capture_init_extra_resources: 12.04 GiB wanted
                    # against 10.56 GiB free on the best rank and 9.81 GiB on
                    # the worst, so the deficit is 1.48 GiB and 2.23 GiB
                    # respectively. 0.53 gives a 63.58 GiB quota, which binds
                    # and returns 4.53 GiB, clearing the worst rank by 2.30 GiB
                    # -- about twice its deficit. 0.55 would clear the best
                    # rank (12.69) and still fail the worst (11.94), and 0.54
                    # leaves only 1.10 GiB of margin.
                    ctx_gpu_memory_fraction=0.53,
                ),
            ),
            worker_env=_REFERENCE_WORKER_ENV,
        ),
        K3Preset(
            name="k3-ctxonly-dep16-c96",
            description=(
                "Kimi-K3 NVFP4 CTX-only, dep16, concurrency 96 "
                "(job 586472, the 7.66 req/s baseline)"
            ),
            mode=CTX_ONLY,
            reference_dir=CTXONLY_REFERENCE_DIR,
            reference_config_sha256="6723dd82438ce9ebc2f97d741cdb74508f54f7996d1ec884827b855aff8a5a04",
            source_config=K3_AGENT_CONFIG_DIR / "agentx_disagg_ctxonly_ctxswept_rubin.yaml",
            source_revision=K3_BENCH_REVISION,
            source_sha256="cb2f379ec27a942669a1d7d2d2c6923fdb2c2322aa671ec7c4ca80a53c215153",
            # Deliberately not submittable: every ctx_configs row at this
            # revision runs gpu_memory_fraction 0.8, but this baseline was
            # measured at 0.7. Submitting would silently produce an
            # incomparable number, so this preset evaluates results only.
            # [ctx_num, tp, pp, max_batch, max_num_tokens, adp, gmem, concurrency]
            # Deliberately not one of the upstream ctx_configs rows: those all
            # run gpu_memory_fraction 0.8 while this baseline was measured at
            # 0.7 with mamba_ssm_cache_dtype bfloat16. The preset exists to
            # reproduce job 586472, so the row mirrors that run's own
            # ctx_config.yaml. Running frac 0.8 against a frac-0.7 reference is
            # what would produce the incomparable number.
            single_string="1,16,1,64,8192,true,0.7,96",
            sweep_key="ctx_configs",
            generator_config=Path("kimi-k3-nvfp4-agent/gen_worker_config.py"),
            generator_sha256=("3c7d08a94dfc7e112ee1e52e1fb4fb0df8f1e9651e80dd9818a64301d642c9c3"),
            # The CTX cost triple is not a default. gen_worker_config.py ships
            # 114000/143000/0 and says of the reference values "Put it back per
            # submission rather than as a default", so a submission that omits
            # them measures a different scheduler. Job 586472 ran
            # 35000/250000/3000; leaving the defaults in place cost 26% of
            # request throughput and 31% of p90 TTFT (job 618999).
            generator_env=(
                ("K3_MAMBA_SSM_DTYPE", "bfloat16"),
                ("K3_CTX_COST_KV_OFFSET", "35000"),
                ("K3_CTX_COST_KV_DEPTH_THRESHOLD", "250000"),
                ("K3_CTX_COST_PER_CHUNK_DEPTH_TOKENS", "3000"),
            ),
            ctx_nodes=4,
            gen_nodes=2,
            gpus_per_node=4,
            ctx_gpus=16,
            gen_gpus=8,
            benchmark_concurrency=96,
            input_length=996608,
            output_length=2,
            references=(
                ClusterReference(
                    cluster=HECATE,
                    provenance=(
                        "R200 / hecate. Job 586472, the 7.66 req/s baseline, "
                        "measured on an image WITHOUT MR!10657 / MR!10675, "
                        "no wheel overlay."
                    ),
                    benchmark_duration_seconds=3629.32,
                    gates=(
                        Gate("request_throughput", 7.66, minimum=True, ratio=0.95),
                        Gate("input_token_throughput", 1144254.57, minimum=True, ratio=0.95),
                        Gate("p50_ttft_ms", 12771.25, minimum=False, ratio=1.10),
                        Gate("p90_ttft_ms", 49506.77, minimum=False, ratio=1.10),
                        # A block-reuse regression would change what "prefill"
                        # even means here, so comparing throughput without it is
                        # meaningless.
                        Gate("prompt_cache_read_pct", 94.54, minimum=True, ratio=0.98),
                    ),
                    caveat=(
                        "This baseline was measured with free_gpu_memory_fraction=0.7, "
                        "mamba_ssm_cache_dtype=bfloat16, and an image WITHOUT MR!10657 / "
                        "MR!10675. The current CTX-only convention is frac 0.8. Treat it "
                        "as a historical reference; do not reuse these numbers as the "
                        "denominator for a frac-0.8 run."
                    ),
                ),
            ),
            overrides=(
                ClusterOverrides(
                    cluster=OCI_JHB,
                    reason=(
                        "GEN MoE backend CUTEDSL_FC12 does not exist on SM103: "
                        "CuteDslFc12FusedMoE.can_implement() rejects it "
                        "('targets Rubin (SM107), got SM103') because the "
                        "FC1+FC2-fused kernel is not ported. The factory "
                        "falls back to CutlassFusedMoE and the model then "
                        "refuses to run, which is correct -- that silent "
                        "fallback voided an earlier GB300 batch. "
                        "gen_worker_config.py names CUTEDSL, the two-op "
                        "parent of the same family, as the GB300 value. "
                        "GB300 therefore measures a DIFFERENT MoE kernel "
                        "than R200 and the two are not comparable."
                        " No fraction override: this preset's context worker "
                        "runs at 0.7 and came up serving on GB300."
                    ),
                    generator_env=(("K3_GEN_MOE_BACKEND", "CUTEDSL"),),
                    # Derived from job 532135's GEN ledger, not picked. Per
                    # rank: 67.32 GiB free at the kv_cache stage and a quota of
                    # min(0.8 x 67.32, max_tokens 56.69) = 53.85 GiB, so unlike
                    # the context case the fraction already binds at 0.8 and
                    # lowering it does real work. "Additional executor
                    # resources" then wanted 27.53 GiB against 20.10 GiB free on
                    # the best rank and 19.42 GiB on the worst -- deficits of
                    # 7.43 and 8.11 GiB. 0.60 returns 13.46 GiB and clears the
                    # worst rank by 5.35 GiB; 0.68 lands 0.03 GiB short of it,
                    # which is not a margin to spend a 6-node run on.
                    #
                    # Cheap to give away here: this preset's reference README
                    # (k3-ctx-baseline-job586472/README.md) records that the
                    # GEN side is a placeholder tep8 unrelated to the baseline,
                    # present only because the harness needs a decode worker to
                    # retire requests at osl=2. Shrinking its KV cache cannot
                    # move the measured CTX numbers.
                    gen_gpu_memory_fraction=0.60,
                ),
            ),
            worker_env=_REFERENCE_WORKER_ENV,
        ),
        K3Preset(
            name="k3-ctxonly-dep16-c96-fp8",
            description=(
                "Kimi-K3 CTX-only with an FP8 KV cache, dep16, concurrency 96 "
                "(the GB300 operating point)"
            ),
            mode=CTX_ONLY,
            reference_dir=CTXONLY_FP8_REFERENCE_DIR,
            reference_config_sha256="",
            source_config=K3_AGENT_CONFIG_DIR / "agentx_disagg_ctxonly_ctxswept_rubin.yaml",
            source_revision=K3_BENCH_REVISION,
            source_sha256="cb2f379ec27a942669a1d7d2d2c6923fdb2c2322aa671ec7c4ca80a53c215153",
            # taizhongw's GB300 sweep ran this shape at fraction 0.8, and an
            # FP8 KV cache is roughly half the tokens per byte of NVFP4, so
            # there is no reason to hold the 0.7 the NVFP4 preset inherited
            # from its own R200 reference.
            single_string="1,16,1,64,8192,true,0.8,96",
            sweep_key="ctx_configs",
            generator_config=Path("kimi-k3-nvfp4-agent/gen_worker_config.py"),
            generator_sha256=("3c7d08a94dfc7e112ee1e52e1fb4fb0df8f1e9651e80dd9818a64301d642c9c3"),
            # K3_KV_DTYPE is the whole point of this preset. NVFP4 KV is a win
            # on Rubin, where FP4 MLA runs the CuTeDSL kernel, and a loss on
            # GB300, where the backend selector can only reach the Triton
            # fallback: job 532868 held 3,893,478 KV tokens against
            # taizhongw's 1,727,118 and was still more than 3x slower through
            # warmup. FP8 keeps the attention path off FP4 MLA entirely.
            #
            # The shape otherwise reproduces job 351572, read from that run's
            # own env_vars.json and configs: MoE backends, both fractions and
            # tokens_per_block.
            #
            # The CTX cost triple is the deliberate exception. The reference
            # ran the generator defaults (114000/143000/0); this preset keeps
            # the tuned 35000/250000/3000 that every other K3 preset pins,
            # because the defaults are a known 26% request-throughput loss and
            # a guard should not gate on a scheduler nobody intends to ship.
            # It makes this preset faster than its own reference by
            # construction, which the reference's caveat records.
            generator_env=(
                ("K3_KV_DTYPE", "fp8"),
                ("K3_CTX_MOE_BACKEND", "TRTLLM"),
                ("K3_GEN_MOE_BACKEND", "CUTEDSL"),
                ("K3_CTX_COST_KV_OFFSET", "35000"),
                ("K3_CTX_COST_KV_DEPTH_THRESHOLD", "250000"),
                ("K3_CTX_COST_PER_CHUNK_DEPTH_TOKENS", "3000"),
            ),
            ctx_nodes=4,
            gen_nodes=2,
            gpus_per_node=4,
            ctx_gpus=16,
            gen_gpus=8,
            benchmark_concurrency=96,
            input_length=996608,
            output_length=2,
            references=(
                ClusterReference(
                    cluster=OCI_JHB,
                    provenance=(
                        "GB300 / oci-jhb. taizhongw job 351572, image "
                        "dd78bbce + MR!10626 / MR!10635, no wheel overlay. "
                        "14,318 requests, 0 errors, 3600 s."
                    ),
                    benchmark_duration_seconds=3600.0,
                    gates=(
                        Gate("request_throughput", 3.93, minimum=True, ratio=0.95),
                        Gate("input_token_throughput", 557525.47, minimum=True, ratio=0.95),
                        Gate("p50_ttft_ms", 20007.83, minimum=False, ratio=1.10),
                        # 95.04 is "Overall Usage Prompt Cache Read %", which is
                        # what _parse_ctx_only_metrics reads. The reference CSV
                        # also carries "Theoretical Prefix Cache Hit %" at 96.94;
                        # taking that one instead failed this gate on a run whose
                        # cache behaviour was in fact unchanged.
                        Gate("prompt_cache_read_pct", 95.04, minimum=True, ratio=0.98),
                    ),
                    caveat=(
                        "The reference ran the generator's DEFAULT CTX cost triple "
                        "(114000/143000/0) while this preset pins the tuned "
                        "35000/250000/3000, a known 26% request-throughput "
                        "difference, so a healthy run should clear these gates with "
                        "room rather than land on them. No p90 TTFT gate: that "
                        "percentile was not recorded for the reference run and is "
                        "not worth inventing."
                    ),
                ),
            ),
            # No cluster overrides: the MoE backends are already pinned to the
            # reference's own values in generator_env, and the reference ran
            # both fractions at 0.8.
            overrides=(),
            worker_env=_REFERENCE_WORKER_ENV,
        ),
        K3Preset(
            name="k3-genonly-helix16-c16-mtp5-fp8",
            description=(
                "Kimi-K3 GEN-only with an FP8 KV cache, ctx dep16 + gen helix16, "
                "concurrency 16, MTP5 (the GB300 operating point)"
            ),
            mode=GEN_ONLY,
            reference_dir=GENONLY_GB300_REFERENCE_DIR,
            reference_config_sha256="",
            source_config=K3_AGENT_CONFIG_DIR / "genonly_turnbatch_c16.yaml",
            source_revision=K3_BENCH_REVISION,
            source_sha256="6182e88e8b1ebad093cf6cdd426fed0624302412837814dee487200640e0c9ad",
            # gen_configs row "helix16 lbs16 mtp5".
            single_string="1,1,16,16,96,helix,0.8,5,0,16",
            sweep_key="gen_configs",
            generator_config=Path("kimi-k3-nvfp4-agent/gen_worker_config.py"),
            generator_sha256=("3c7d08a94dfc7e112ee1e52e1fb4fb0df8f1e9651e80dd9818a64301d642c9c3"),
            # FP8 KV for the same reason the CTX-only FP8 preset exists: NVFP4
            # KV puts SM103 on the Triton FP4 MLA fallback, and for helix it is
            # worse than slow -- Helix softmax stats are implemented only by the
            # CuTeDSL backend, so helix + NVFP4 raises NotImplementedError on
            # Blackwell. FP8 keeps the attention path off FP4 MLA and helix runs.
            # The reference did not set mamba_ssm_cache_dtype; neither does this.
            generator_env=(
                ("K3_KV_DTYPE", "fp8"),
                # Both MoE backends are pinned to what the reference job
                # actually ran, read out of its own ctx_config.yaml and
                # gen_config.yaml: ctx TRTLLM, gen CUTEDSL. Without them the
                # generator's defaults apply, which are CUTEDSL for ctx and
                # CUTEDSL_FC12 for gen -- and CUTEDSL_FC12 does not exist on
                # SM103, so can_implement() rejects it, the factory falls back
                # to CutlassFusedMoE, and modeling_kimi_linear.py refuses to
                # run. That is a dead GEN worker rather than a wrong number,
                # but it is still a multi-node run spent on a known failure.
                ("K3_CTX_MOE_BACKEND", "TRTLLM"),
                ("K3_GEN_MOE_BACKEND", "CUTEDSL"),
                ("K3_CTX_COST_KV_OFFSET", "35000"),
                ("K3_CTX_COST_KV_DEPTH_THRESHOLD", "250000"),
                ("K3_CTX_COST_PER_CHUNK_DEPTH_TOKENS", "3000"),
            ),
            ctx_nodes=4,
            gen_nodes=4,
            gpus_per_node=4,
            ctx_gpus=16,
            gen_gpus=16,
            benchmark_concurrency=16,
            input_length=996608,
            output_length=512,
            references=(
                ClusterReference(
                    cluster=OCI_JHB,
                    provenance=(
                        "GB300 / oci-jhb. taizhongw job 354153, image "
                        "dd78bbce + MR!10626 / MR!10635, no wheel overlay. "
                        "16/16 completed, 174.889 s."
                    ),
                    benchmark_duration_seconds=174.889,
                    gates=(
                        Gate("output_throughput", 46.84116, minimum=True, ratio=0.95),
                        Gate("user_throughput", 2.92866, minimum=True, ratio=0.95),
                        Gate("mean_tpot_ms", 6.53533, minimum=False, ratio=1.10),
                        Gate("avg_decoded_tokens_per_iter", 3.59331, minimum=True, ratio=0.95),
                        Gate("device_step_time_ms", 29.487, minimum=False, ratio=1.10),
                    ),
                    caveat=(
                        "FP8 KV, so this measures a different attention path than "
                        "the R200 helix8 preset's NVFP4. Not comparable to it. The "
                        "reference also ran the generator's default CTX cost triple "
                        "while this preset pins the tuned one."
                    ),
                ),
            ),
            overrides=(),
            worker_env=_REFERENCE_WORKER_ENV,
        ),
        K3Preset(
            name="k3-genonly-tep8-c2-mtp5-fp8",
            description=(
                "Kimi-K3 GEN-only with an FP8 KV cache, ctx dep16 + gen tep8, "
                "concurrency 2, MTP5 (the GB300 low-concurrency point)"
            ),
            mode=GEN_ONLY,
            reference_dir=GENONLY_GB300_REFERENCE_DIR,
            reference_config_sha256="",
            source_config=K3_AGENT_CONFIG_DIR / "genonly_turnbatch_c2.yaml",
            source_revision=K3_BENCH_REVISION,
            source_sha256="937bb2f4e5100195f2e142e941907c41c0cf5609f337e57ec64168c4a2504d79",
            # gen_configs row "tep8 lbs2 mtp5". The topology field is false
            # rather than helix or true: plain tensor parallelism, no
            # attention DP on the gen side.
            single_string="1,1,8,2,12,false,0.8,5,0,2",
            sweep_key="gen_configs",
            generator_config=Path("kimi-k3-nvfp4-agent/gen_worker_config.py"),
            generator_sha256=("3c7d08a94dfc7e112ee1e52e1fb4fb0df8f1e9651e80dd9818a64301d642c9c3"),
            generator_env=(
                ("K3_KV_DTYPE", "fp8"),
                # Both MoE backends are pinned to what the reference job
                # actually ran, read out of its own ctx_config.yaml and
                # gen_config.yaml: ctx TRTLLM, gen CUTEDSL. Without them the
                # generator's defaults apply, which are CUTEDSL for ctx and
                # CUTEDSL_FC12 for gen -- and CUTEDSL_FC12 does not exist on
                # SM103, so can_implement() rejects it, the factory falls back
                # to CutlassFusedMoE, and modeling_kimi_linear.py refuses to
                # run. That is a dead GEN worker rather than a wrong number,
                # but it is still a multi-node run spent on a known failure.
                ("K3_CTX_MOE_BACKEND", "TRTLLM"),
                ("K3_GEN_MOE_BACKEND", "CUTEDSL"),
                ("K3_CTX_COST_KV_OFFSET", "35000"),
                ("K3_CTX_COST_KV_DEPTH_THRESHOLD", "250000"),
                ("K3_CTX_COST_PER_CHUNK_DEPTH_TOKENS", "3000"),
            ),
            ctx_nodes=4,
            gen_nodes=2,
            gpus_per_node=4,
            ctx_gpus=16,
            gen_gpus=8,
            benchmark_concurrency=2,
            input_length=996608,
            output_length=512,
            references=(
                ClusterReference(
                    cluster=OCI_JHB,
                    provenance=(
                        "GB300 / oci-jhb. taizhongw job 354126, image "
                        "dd78bbce + MR!10626 / MR!10635, no wheel overlay. "
                        "2/2 completed, 147.008 s."
                    ),
                    benchmark_duration_seconds=147.008,
                    gates=(
                        Gate("output_throughput", 6.96559, minimum=True, ratio=0.95),
                        Gate("user_throughput", 3.48313, minimum=True, ratio=0.95),
                        Gate("mean_tpot_ms", 5.58552, minimum=False, ratio=1.10),
                        Gate("avg_decoded_tokens_per_iter", 3.61842, minimum=True, ratio=0.95),
                        Gate("device_step_time_ms", 26.632, minimum=False, ratio=1.10),
                    ),
                    caveat=(
                        "Concurrency 2, so this is a latency point rather than a "
                        "throughput one -- two requests decide every number here, "
                        "and the gate ratios are the same 5% written for a "
                        "16-request point. FP8 KV, and the reference ran the "
                        "generator's default CTX cost triple."
                    ),
                ),
            ),
            overrides=(),
            worker_env=_REFERENCE_WORKER_ENV,
        ),
        K3Preset(
            name="k3-ctxonly-dwdp16-c128",
            description=(
                "Kimi-K3 NVFP4 CTX-only with DWDP16 expert prefetch, 16 x TP1 "
                "context servers + helix8 gen, concurrency 128 (job 624450, the "
                "PASSING control of the 2026-09-22 DWDP IMA archive). "
                "EVALUATION-ONLY: the pinned bench revision has no DWDP launcher."
            ),
            mode=CTX_ONLY,
            reference_dir=DWDP_CTXONLY_REFERENCE_DIR,
            reference_config_sha256=(
                "c98961a8a46ac64a713c7731019b89321d3f66fccea9536378b5127183a4bcee"
            ),
            # Pinned for identity, not for submission. The ctx-only agent config
            # is the nearest analogue at K3_BENCH_REVISION and it is emphatically
            # NOT what produced job 624450: no config at that revision can
            # express a DWDP topology at all.
            source_config=K3_AGENT_CONFIG_DIR / "agentx_disagg_ctxonly_ctxswept_rubin.yaml",
            source_revision=K3_BENCH_REVISION,
            source_sha256="cb2f379ec27a942669a1d7d2d2c6923fdb2c2322aa671ec7c4ca80a53c215153",
            # Not submittable, and unlike k3-ctxonly-dep16-c96 this is not a
            # gpu_memory_fraction mismatch that a different row would fix: the
            # bench revision pinned here has no DWDP path whatsoever. Checked at
            # 0c7dcd10fa and against every other ref present in that clone --
            # `git ls-files | grep -i dwdp` is empty, the inner
            # trtllm-disagg-benchmark/submit.py contains zero occurrences of
            # "dwdp", gen_worker_config.py cannot emit a dwdp_config block, and
            # `git log --all -S dwdp` finds only an unrelated Qwen3.5 patch file.
            #
            # The topology itself is expressible -- ctx_num is field 0 of a
            # ctx_configs row, so "16 TP1 context servers" is just
            # [16, 1, 1, ...]. What is missing is dwdp_config, which the worker
            # config generator has no code to emit, and the per-instance
            # TRTLLM_DWDP_RANK / _MASTER_ADDR that the 16 independent size-1
            # srun tasks need in order to find each other through MR!10640's
            # TCPStore rendezvous.
            #
            # A DWDP-capable perf harness does exist -- job 624450 ran under one
            # -- but at the time of writing only as an uncommitted working-tree
            # modification in a developer checkout. _read_pinned_file
            # authenticates its inputs with `git show <revision>:<path>`, which
            # cannot reach a working-tree-only change, and no source_sha256
            # could authenticate it. Making this preset submittable means
            # landing DWDP support on a bench-trtllm-disagg commit first, then
            # bumping K3_BENCH_REVISION and filling in the row here.
            single_string=None,
            sweep_key="ctx_configs",
            generator_config=Path("kimi-k3-nvfp4-agent/gen_worker_config.py"),
            generator_sha256=("3c7d08a94dfc7e112ee1e52e1fb4fb0df8f1e9651e80dd9818a64301d642c9c3"),
            # Inert while single_string is None. Recorded as the operating point
            # a submittable version has to reproduce, read out of job 624450's own
            # ctx_config.yaml / gen_config.yaml / env_vars.json rather than copied
            # from a neighbouring preset. Two entries differ from every other K3
            # preset and the difference is the point:
            #
            #  * The CTX cost triple is 0/0/0, not the tuned 35000/250000/3000
            #    that k3-ctxonly-dep16-c96 pins. The control's ctx_worker_env_var
            #    sets TLLM_V2_CTX_COST_KV_OFFSET, _KV_DEPTH_THRESHOLD and
            #    _PER_CHUNK_DEPTH_TOKENS all to 0, i.e. the cost model is off.
            #    DWDP16 runs 16 independent tp1 context servers behind a
            #    conversation router, so the depth-aware chunk costing that earns
            #    26% of request throughput on one tp16 server is not what
            #    schedules this point. Copying 35000/250000/3000 in here would
            #    have measured a scheduler the reference never ran.
            #  * The GEN MoE backend is TRTLLM, not the CUTEDSL_FC12 the generator
            #    defaults to on R200, because the control's gen_config.yaml says
            #    moe_config.backend: TRTLLM.
            generator_env=(
                ("K3_MAMBA_SSM_DTYPE", "bfloat16"),
                ("K3_CTX_MOE_BACKEND", "CUTEDSL"),
                ("K3_GEN_MOE_BACKEND", "TRTLLM"),
                ("K3_CTX_COST_KV_OFFSET", "0"),
                ("K3_CTX_COST_KV_DEPTH_THRESHOLD", "0"),
                ("K3_CTX_COST_PER_CHUNK_DEPTH_TOKENS", "0"),
            ),
            # The same 6-node / 24-GPU envelope as the other R200 presets, packed
            # differently: 16 ctx GPUs as 16 tp1 servers 4-to-a-node, plus one
            # helix8 decode worker on 2 nodes. Job 624450's server_config.yaml
            # lists 16 context URLs across hecate0241-0244 and one gen URL.
            ctx_nodes=4,
            gen_nodes=2,
            gpus_per_node=4,
            ctx_gpus=16,
            gen_gpus=8,
            benchmark_concurrency=128,
            input_length=996608,
            output_length=2,
            references=(
                ClusterReference(
                    cluster=HECATE,
                    provenance=(
                        "R200 / hecate. Job 624450, measured 2026-09-21, the "
                        "PASSING control of taizhongw's k3_dwdp_ima_20260922 "
                        "archive. Every number below was read from that run's own "
                        "concurrency_128/profile_export_aiperf.csv, not from the "
                        "archive README. Image "
                        "/lustre/share/coreai_comparch_trtllm/lizhiz/rubin_k3/"
                        "923d189d87/trtllm.sqsh -- the OLD image -- plus a "
                        "PYTHONPATH overlay onto laliao's cutedsl-runtime and "
                        "tekit-k3-dwdp-10683-kvgather-smreserve-clc trees, and an "
                        "external gather kernel loaded through "
                        "TRTLLM_DWDP_GATHER_LIB. 41458 requests, 0 errors, "
                        "3619.82 s. Cross-check: 11.39 req/s is what the archive "
                        "README quotes, and 1970926.13 x (1 - 0.9516) / 16 ctx "
                        "GPUs = 5962 uncached tok/s per ctx GPU against the "
                        "README's 5998, so the CSV and the prose describe the "
                        "same run."
                    ),
                    benchmark_duration_seconds=3619.82,
                    gates=(
                        Gate("request_throughput", 11.39, minimum=True, ratio=0.95),
                        Gate("input_token_throughput", 1970926.13, minimum=True, ratio=0.95),
                        Gate("p50_ttft_ms", 12343.63, minimum=False, ratio=1.10),
                        Gate("p90_ttft_ms", 45111.47, minimum=False, ratio=1.10),
                        # 95.16 is "Overall Usage Prompt Cache Read %", which is
                        # what _parse_ctx_only_metrics reads. The same CSV also
                        # carries "Theoretical Prefix Cache Hit %" at 97.55;
                        # taking that one instead failed this gate on the FP8
                        # preset for a run whose cache behaviour was unchanged.
                        Gate("prompt_cache_read_pct", 95.16, minimum=True, ratio=0.98),
                    ),
                    caveat=(
                        "This reference did not run the branch, and the point it "
                        "measures is known to be broken on current code. Three "
                        "things follow. (1) The run took its Python from a "
                        "PYTHONPATH overlay onto two of laliao's trees; PYTHONPATH "
                        "precedes site-packages, so it would shadow anything "
                        "--wheel or --trtllm-repo installed, and reproducing this "
                        "operating point on the branch means dropping the overlay. "
                        "(2) Its DWDP gather came from an external .so named by "
                        "TRTLLM_DWDP_GATHER_LIB, whereas the in-tree path selects "
                        "the registered trtllm::dwdp_fabric_gather op and "
                        "TRTLLM_DWDP_GATHER_KERNEL; those are different kernels, "
                        "so these numbers gate the DWDP operating point rather "
                        "than the in-tree gather implementation. (3) The archive "
                        "exists because this exact configuration raises "
                        "CUDA_ERROR_ILLEGAL_ADDRESS (700) on the ToT image "
                        "(rubin-advance 6a457e391e + MR!10696/!10697/!10698), in "
                        "MR!10698's CuTeDSL gather_mxfp8 at "
                        "cute_dsl_kernels/rubin/fp4_mla_context_gather.py:336, "
                        "reached from forward_context_with_chunked_prefill_fp4. "
                        "Four reproductions; the stale-autotune-cache and "
                        "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP hypotheses are both "
                        "already ruled out, by 626311 and 628753 respectively. "
                        "Until that is fixed, a submission at this point is "
                        "expected to die in warmup around step 53/512 rather than "
                        "produce a number. Finally, the CTX cost model is disabled "
                        "here (0/0/0) while every other K3 preset pins "
                        "35000/250000/3000, so these numbers are not comparable to "
                        "k3-ctxonly-dep16-c96 even though both are CTX-only at "
                        "ISL 996608."
                    ),
                ),
            ),
            overrides=(),
            # Job 624450 ran the old image, whose modeling_kimi_linear.py still
            # defaulted KIMI_K3_FP8_WEIGHT_READ_MOE_MLP to "1" -- the archive
            # README confirms it by recording 628753 as the run that set it to
            # "0" on purpose. #19179 flipped that default, so the guard has to
            # set it back for a branch run to match this baseline.
            worker_env=_REFERENCE_WORKER_ENV,
        ),
    )
}
DEFAULT_K3_PRESET = "k3-genonly-helix8-c16-mtp5"


@dataclass(frozen=True)
class PerfMetrics:
    """Parsed results, keyed by the metric names the gates reference."""

    duration_seconds: float
    request_count: int
    error_count: int
    values: dict[str, float] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        if not self.request_count:
            return 1.0
        return self.error_count / self.request_count


@dataclass(frozen=True)
class GateResult:
    metric: str
    value: float
    reference: float
    limit: float
    minimum: bool

    @property
    def passed(self) -> bool:
        if self.minimum:
            return self.value >= self.limit
        return self.value <= self.limit


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _read_aiperf_metric_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    """Read an AIPerf CSV.

    The file holds two stacked tables: a wide per-metric table
    (``Metric,avg,min,max,...``) followed by a narrow summary table
    (``Metric,Value``). ``csv.DictReader`` applies the first header to both, so
    summary values land in the ``avg`` column. That is intentional and is what
    the DSV4 guard relies on too.
    """
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "Metric" not in rows[0]:
        raise ValueError(f"Invalid aiperf CSV schema: {csv_path}")
    return {
        row["Metric"]: row
        for row in rows
        if row.get("Metric") and "(error-adjusted)" not in row["Metric"]
    }


def _aiperf_value(
    rows: dict[str, dict[str, str]],
    metric_prefix: str,
    column: str,
    default: float | None = None,
) -> float:
    matches = [name for name in rows if name.startswith(metric_prefix)]
    if not matches:
        if default is not None:
            return default
        raise ValueError(f"No aiperf metric starts with {metric_prefix!r}")
    if len(matches) != 1:
        raise ValueError(
            f"Expected one aiperf metric starting with {metric_prefix!r}, got {matches}"
        )
    raw_value = rows[matches[0]].get(column, "")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing {column} value for aiperf metric {matches[0]!r}") from error


def _find_one(result_dir: Path, name: str) -> Path:
    direct = result_dir / name
    if direct.is_file():
        return direct
    matches = sorted(result_dir.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {result_dir}, found {len(matches)}")
    return matches[0]


def _parse_ctx_only_metrics(result_dir: Path) -> PerfMetrics:
    csv_path = _find_one(result_dir, "profile_export_aiperf.csv")
    rows = _read_aiperf_metric_rows(csv_path)
    request_count = _aiperf_value(rows, "Request Count", "avg")
    errors = _aiperf_value(rows, "Error Request Count", "avg", default=0.0)
    values = {
        "request_throughput": _aiperf_value(rows, "Request Throughput", "avg"),
        "input_token_throughput": _aiperf_value(rows, "Input Token Throughput", "avg"),
        "p50_ttft_ms": _aiperf_value(rows, "Time to First Token (ms)", "p50"),
        "p90_ttft_ms": _aiperf_value(rows, "Time to First Token (ms)", "p90"),
        "prompt_cache_read_pct": _aiperf_value(rows, "Overall Usage Prompt Cache Read %", "avg"),
    }
    return PerfMetrics(
        duration_seconds=_aiperf_value(rows, "Benchmark Duration", "avg"),
        request_count=int(request_count),
        error_count=int(errors),
        values=values,
    )


_ITER_PATTERN = re.compile(r"num_scheduled_requests = (\d+).*?prev_device_step_time = ([0-9.]+)ms")


def _gen_device_step_time_ms(log_path: Path, concurrency: int) -> float:
    """Mean saturated per-iteration device step time, outlier-filtered.

    Reproduces the reference run's ``elapsed_time_avg``: keep only iterations
    where every client slot is occupied (``num_scheduled_requests ==
    concurrency``), drop samples more than 20% away from the median, then take
    the mean of what remains. Verified to return 28.929 ms on job 605293, which
    matches the 28.93 ms that run's README reports.
    """
    samples = []
    with log_path.open(errors="replace") as stream:
        for line in stream:
            match = _ITER_PATTERN.search(line)
            if match and int(match.group(1)) == concurrency:
                samples.append(float(match.group(2)))
    if not samples:
        raise ValueError(
            f"No saturated iterations (num_scheduled_requests == {concurrency}) in {log_path}"
        )
    median = statistics.median(samples)
    kept = [value for value in samples if abs(value - median) <= 0.2 * median]
    if not kept:
        raise ValueError(f"Every saturated iteration was an outlier in {log_path}")
    return statistics.mean(kept)


def _parse_gen_only_metrics(result_dir: Path, preset: K3Preset) -> PerfMetrics:
    result_path = _find_one(result_dir, "result.json")
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid gen-only result.json: {result_path}") from error

    required = (
        "duration",
        "completed",
        "num_prompts",
        "output_throughput",
        "user_throughput",
        "mean_tpot_ms",
        "mean_avg_decoded_tokens_per_iter",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"result.json is missing {missing}: {result_path}")

    iteration_log = _find_one(result_dir, "gen_only_0.txt")
    values = {
        "output_throughput": float(payload["output_throughput"]),
        "user_throughput": float(payload["user_throughput"]),
        "mean_tpot_ms": float(payload["mean_tpot_ms"]),
        "avg_decoded_tokens_per_iter": float(payload["mean_avg_decoded_tokens_per_iter"]),
        "device_step_time_ms": _gen_device_step_time_ms(
            iteration_log, preset.benchmark_concurrency
        ),
    }
    completed = int(payload["completed"])
    requested = int(payload["num_prompts"])
    return PerfMetrics(
        duration_seconds=float(payload["duration"]),
        request_count=requested,
        error_count=max(requested - completed, 0),
        values=values,
    )


def _parse_metrics(result_dir: Path, preset: K3Preset) -> PerfMetrics:
    if preset.mode == GEN_ONLY:
        return _parse_gen_only_metrics(result_dir, preset)
    return _parse_ctx_only_metrics(result_dir)


# ---------------------------------------------------------------------------
# Gating and reporting
# ---------------------------------------------------------------------------


def _gate_results(metrics: PerfMetrics, reference: ClusterReference) -> tuple[GateResult, ...]:
    results = [
        GateResult(
            "duration_seconds",
            metrics.duration_seconds,
            reference.benchmark_duration_seconds,
            reference.benchmark_duration_seconds * reference.minimum_duration_ratio,
            True,
        ),
        # The CTX-only reference itself recorded one client error out of 27871,
        # so an exact-zero gate would fail on its own baseline. Gate the rate.
        GateResult("error_rate", metrics.error_rate, 0.0, 0.001, False),
    ]
    for gate in reference.gates:
        if gate.metric not in metrics.values:
            raise ValueError(f"Parsed results do not contain {gate.metric!r}")
        results.append(
            GateResult(
                gate.metric,
                metrics.values[gate.metric],
                gate.reference,
                gate.limit(),
                gate.minimum,
            )
        )
    return tuple(results)


_PIP_SKIPPED_WHEEL = "already installed with the same version as the provided wheel"
_PIP_INSTALLED_WHEEL = "Successfully installed tensorrt"


def _assert_branch_build_used(result_dir: Path) -> None:
    """Refuse to gate a run that silently used the base image's build.

    The harness installs the branch wheel with a plain ``pip install``. When
    the wheel's version equals the one already in the image -- which it does
    whenever the image was built from a nearby commit -- pip skips it and the
    run measures the image instead, reporting entirely plausible numbers
    against the wrong build. Observed once: a run on a pre-MR!10675 image
    reproduced that image's known 43.97 ms step time and read as a branch
    regression. Fail loudly rather than publish a number for unknown code.
    """
    for install_log in sorted(result_dir.rglob("2_install.log")):
        text = install_log.read_text(errors="replace")
        # The recipe installs twice: --force-reinstall, then the [devel]
        # extras. The second pass legitimately reports the wheel as already
        # installed, so the skip message alone means nothing -- only its
        # presence *without* any successful install indicates the branch
        # build never landed.
        if _PIP_SKIPPED_WHEEL in text and _PIP_INSTALLED_WHEEL not in text:
            raise ValueError(
                f"pip skipped the branch wheel in {install_log}: the image "
                f"already had that version, so this run measured the image, "
                f"not the branch. Reinstall with --force-reinstall."
            )


def _verify_reference_config(result_dir: Path, preset: K3Preset) -> None:
    """Authenticate the pinned reference workdir before grading against it.

    ``reference_config_sha256`` was recorded on every preset from the start but
    never read back, so the pin documented a baseline without protecting it --
    an edit under the shared reference tree would have moved the denominator
    silently, which is the one thing pinning a digest is for.

    Two deliberate narrowings. It runs only when the directory being graded IS
    the preset's own reference, because a fresh result directory has no
    reference config to authenticate and the digest would be meaningless there.
    And it runs only when the preset actually carries a digest: the three GB300
    presets pin the empty string, and minting digests for them is a separate
    change that needs their own reference trees checked first.
    """
    if not preset.reference_config_sha256:
        return
    try:
        if result_dir.resolve() != preset.reference_dir.resolve():
            return
    except OSError:
        return
    config = preset.reference_config
    if not config.is_file():
        raise ValueError(f"Pinned reference config is missing: {config}")
    digest = _sha256_bytes(config.read_bytes())
    if digest != preset.reference_config_sha256:
        raise ValueError(
            f"Pinned reference config {config} hashes to {digest}, expected "
            f"{preset.reference_config_sha256}. The baseline this preset grades "
            f"against has moved under it; re-measure rather than re-pin."
        )


def _print_run_header(result_dir: Path, preset: K3Preset, cluster: Cluster) -> None:
    print(f"Preset:  {preset.name} - {preset.description}")
    print(f"Mode:    {preset.mode}")
    print(f"Cluster: {cluster.name} ({cluster.gpu_name})")
    print(f"Results: {result_dir}")
    print(
        f"Topology: {preset.total_gpus} GPUs "
        f"(ctx {preset.ctx_gpus} / gen {preset.gen_gpus}), "
        f"concurrency {preset.benchmark_concurrency}, "
        f"ISL {preset.input_length} / OSL {preset.output_length}"
    )


def _establish_reference(result_dir: Path, preset: K3Preset, cluster: Cluster) -> bool:
    """Report a run's metrics without grading them, for a cluster with no baseline.

    The first run on new hardware has nothing to compare against, and inventing
    a denominator from another cluster's numbers is exactly the mistake
    per-cluster references exist to prevent. So this grades only on the run
    having completed cleanly -- artifacts present, parseable, branch wheel
    actually installed -- and prints the measured values as a ClusterReference
    literal ready to paste into K3_PRESETS.

    Gate directions and ratios are copied from an existing reference for the
    same preset: which metrics matter and which way is "worse" are properties
    of the workload, not of the hardware. Only the numbers are new.
    """
    _verify_reference_config(result_dir, preset)
    _assert_branch_build_used(result_dir)
    metrics = _parse_metrics(result_dir, preset)

    _print_run_header(result_dir, preset, cluster)
    print(f"Requests: {metrics.request_count} total, {metrics.error_count} errors")
    print()
    print("No reference for this cluster; reporting measured values only.")
    print()
    print("Metric                         Measured")
    print("--------------------------  ------------")
    print(f"{'duration_seconds':26}  {metrics.duration_seconds:12.3f}")
    print(f"{'error_rate':26}  {metrics.error_rate:12.6f}")
    for metric, value in metrics.values.items():
        print(f"{metric:26}  {value:12.5f}")
    print()
    print("Paste into this preset's references= tuple:")
    print()
    print(_render_reference_block(metrics, preset, cluster))
    return True


def _render_reference_block(metrics: PerfMetrics, preset: K3Preset, cluster: Cluster) -> str:
    """Render measured metrics as a pasteable ``ClusterReference`` literal."""
    if not preset.references:
        raise ValueError(
            f"Preset {preset.name!r} has no existing reference to copy a gate "
            f"policy from; add one by hand before using --establish-reference."
        )
    template = preset.references[0]
    lines = [
        "                ClusterReference(",
        f"                    cluster={cluster.name.upper().replace('-', '_')},",
        "                    provenance=(",
        f'                        "{cluster.gpu_name} / {cluster.name}. '
        f'Job <job id>, measured {time.strftime("%Y-%m-%d")}, "',
        '                        "image <image>, wheel <wheel>."',
        "                    ),",
        f"                    benchmark_duration_seconds={metrics.duration_seconds:.2f},",
        "                    gates=(",
    ]
    for gate in template.gates:
        value = metrics.values[gate.metric]
        lines.append(
            f'                        Gate("{gate.metric}", {value:.5f}, '
            f"minimum={gate.minimum}, ratio={gate.ratio}),"
        )
    lines.extend(
        [
            "                    ),",
            "                ),",
        ]
    )
    return "\n".join(lines)


def _check_result(result_dir: Path, preset: K3Preset, cluster: Cluster) -> bool:
    """Print measured values beside references and derived pass/fail limits."""
    _verify_reference_config(result_dir, preset)
    _assert_branch_build_used(result_dir)
    reference = preset.reference_for(cluster.name)
    metrics = _parse_metrics(result_dir, preset)
    results = _gate_results(metrics, reference)

    _print_run_header(result_dir, preset, cluster)
    print(f"Requests: {metrics.request_count} total, {metrics.error_count} errors")
    print(f"Baseline: {reference.provenance}")
    if reference.caveat:
        print(f"CAVEAT:  {reference.caveat}")
    print()
    print("Metric                         Actual      Reference          Guard  Result")
    print("--------------------------  ----------  -------------  -------------  ------")
    for result in results:
        comparison = ">=" if result.minimum else "<="
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.metric:26}  {result.value:10.3f}  {result.reference:13.3f}  "
            f"{comparison}{result.limit:12.3f}  {status:>6}"
        )
    return all(result.passed for result in results)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_pinned_file(
    benchmark_repo: Path, revision: str, path: Path, expected_sha256: str, label: str
) -> bytes:
    """Read and authenticate one file from the pinned revision.

    Reading through the VCS rather than the working tree means local edits in
    the shared benchmark checkout cannot change what the guard submits.
    """
    result = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        cwd=benchmark_repo,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"Cannot read {label} from {revision}: {detail}")
    digest = _sha256_bytes(result.stdout)
    if digest != expected_sha256:
        raise ValueError(f"{label} hash mismatch: expected {expected_sha256}, got {digest}")
    return result.stdout


def _read_pinned_source_config(benchmark_repo: Path, preset: K3Preset) -> str:
    return _read_pinned_file(
        benchmark_repo,
        preset.source_revision,
        preset.source_config,
        preset.source_sha256,
        f"preset {preset.name!r} source config",
    ).decode()


def _config_scalar(config: str, key: str) -> str:
    pattern = rf"^{re.escape(key)}: *(.+?) *(?:#.*)?$"
    matches = re.findall(pattern, config, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one top-level {key} in the source config")
    return matches[0].strip().strip("'\"")


def _validate_source_config(config: str, preset: K3Preset) -> None:
    """Fail before submission if the pinned source config drifted from the preset."""
    expected_mode = "gen_only" if preset.mode == GEN_ONLY else "e2e"
    actual_mode = _config_scalar(config, "benchmark_mode")
    if actual_mode != expected_mode:
        raise ValueError(
            f"Preset expects benchmark_mode={expected_mode}, source config has {actual_mode}"
        )
    for key, expected in (("isl", preset.input_length), ("osl", preset.output_length)):
        actual = int(_config_scalar(config, key))
        if actual != expected:
            raise ValueError(f"Preset expects {key}={expected}, source config has {actual}")
    if preset.single_string is None:
        raise ValueError(f"Preset {preset.name!r} is evaluation-only and cannot be submitted.")
    # The sweep list has to exist; its rows are replaced by the preset's own,
    # which reproduces the reference run rather than the upstream sweep.
    if re.search(rf"^{re.escape(preset.sweep_key)}: *$", config, flags=re.MULTILINE) is None:
        raise ValueError(
            f"Pinned source config has no {preset.sweep_key} list for preset {preset.name!r}"
        )


def _select_pinned_row(config: str, preset: K3Preset) -> str:
    """Reduce the sweep list to the preset's row, for submission via ``-m all``.

    submit.py's ``-m single -s`` path re-parses the row as ten positional
    fields and reads field 5 only as an attention-DP boolean, so the topology
    keyword is lost: a ``helix`` row silently degrades to plain tensor
    parallelism (verified -- it produced tensor_parallel_size 8 with no
    cp_config, against the reference's context_parallel_size 8 + HELIX, and
    also flipped the MoE backend and the speculative decoding type). Filtering
    the sweep list and letting ``-m all`` consume the row keeps it exactly as
    the reference run did.
    """
    if preset.single_string is None:
        raise ValueError(f"Preset {preset.name!r} has no row to select")

    values = [value.strip() for value in preset.single_string.split(",")]

    def _quote(index: int, value: str) -> str:
        # Mirror how the upstream rows are written: plain integers bare, the
        # topology keyword and booleans bare, gmem quoted, and the trailing
        # concurrency field always quoted since it may carry a list.
        if index == len(values) - 1:
            return f"'{value}'"
        if value.isdigit() or value in ("true", "false", "helix"):
            return value
        return f"'{value}'"

    row = "- [" + ", ".join(_quote(i, v) for i, v in enumerate(values)) + "]"

    output: list[str] = []
    in_sweep = False
    replaced = False
    for line in config.splitlines():
        if re.match(rf"^{re.escape(preset.sweep_key)}: *$", line):
            in_sweep = True
            replaced = True
            output.append(line)
            output.append(row)
            continue
        if in_sweep:
            stripped = line.strip()
            if line.startswith("- [") or stripped.startswith("#") or not stripped:
                continue
            in_sweep = False
        output.append(line)

    if not replaced:
        raise ValueError(
            f"Pinned source config has no {preset.sweep_key} list for preset {preset.name!r}"
        )
    return "\n".join(output) + "\n"


_GENERATOR_HELPERS_ANCHOR = (
    "# ----------------------------------------------------------------- helpers --"
)


def _inject_worker_env(generator: bytes, entries: tuple[tuple[str, str], ...]) -> bytes:
    """Splice worker env assignments into the staged generator copy.

    gen_worker_config.py builds ``worker_envs_vars`` as a dict literal plus a
    few conditional mutations and offers no passthrough, so a name it does not
    mention cannot reach the workers from a preset. The staged copy is
    sha256-verified before this runs, so the anchor is a known line: the
    helpers banner, which sits after the last mutation and well before the
    dict is first read.
    """
    if not entries:
        return generator
    text = generator.decode()
    if text.count(_GENERATOR_HELPERS_ANCHOR) != 1:
        raise ValueError("Expected exactly one helpers banner in the worker-config generator")
    injected = "".join(f"worker_envs_vars[{key!r}] = {value!r}\n" for key, value in entries)
    return text.replace(
        _GENERATOR_HELPERS_ANCHOR,
        f"# Injected by the branch perf guard.\n{injected}\n\n{_GENERATOR_HELPERS_ANCHOR}",
        1,
    ).encode()


def _override_ctx_gpu_memory_fraction(config: str, fraction: float) -> str:
    """Rewrite ctx_config.gpu_memory_fraction in the prepared source config.

    The fraction sizes the temporary pool the KV cache size estimation builds
    (_util.py computes max_gpu_total_bytes as fraction * free memory), so
    lowering it is the lever for getting a context worker past an estimation
    OOM. It also moves the operating point away from the reference, which
    measured at 0.8, so a run using this is not directly comparable to the
    reference numbers.
    """
    pattern = re.compile(
        r"^(ctx_config:\n(?:[ \t]+.*\n)*?[ \t]+gpu_memory_fraction: )[0-9.]+", re.MULTILINE
    )
    updated, count = pattern.subn(rf"\g<1>{fraction}", config, count=1)
    if count != 1:
        raise ValueError("Expected exactly one ctx_config.gpu_memory_fraction to override")
    return updated


def _override_gen_gpu_memory_fraction(config: str, fraction: float) -> str:
    """Rewrite gen_config.gpu_memory_fraction in the prepared source config.

    Strictly symmetric to _override_ctx_gpu_memory_fraction: it anchors on the
    gen_config block, replaces exactly one value, and raises if the file does
    not contain exactly one. Touching only the block it names is the whole
    value of these overrides -- a fraction that silently moved both sides would
    change the measured point while looking like it had not.
    """
    pattern = re.compile(
        r"^(gen_config:\n(?:[ \t]+.*\n)*?[ \t]+gpu_memory_fraction: )[0-9.]+", re.MULTILINE
    )
    updated, count = pattern.subn(rf"\g<1>{fraction}", config, count=1)
    if count != 1:
        raise ValueError("Expected exactly one gen_config.gpu_memory_fraction to override")
    return updated


def _override_job_time(config: str, job_time: str) -> str:
    """Rewrite slurm.job_time in the prepared source config.

    Applied to the in-memory copy after the reference config's SHA-256 has been
    verified, exactly like the two gpu_memory_fraction overrides, so the pinned
    file on the shared reference tree is never touched and the pin still means
    what it says.

    This moves only the Slurm wall limit, not the workload, so unlike a memory
    fraction it does not move the measured point away from the reference. What
    it can do is truncate a run that needed the original limit -- and a job
    killed at the wall yields no result rather than a wrong one, so that
    failure is loud rather than silent.
    """
    pattern = re.compile(r"^([ \t]*job_time: ).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{job_time}", config, count=1)
    if count != 1:
        raise ValueError("Expected exactly one slurm.job_time to override")
    return updated


def _write_server_config(
    output: Path,
    preset: K3Preset,
    cluster: Cluster,
    image: Path,
    account: str,
    partition: str,
    constraint: str,
    home_dir: Path,
) -> Path:
    """Write the cluster config submit.py executes to resolve image and paths.

    FLASHINFER_CUBIN_DIR is set so downloaded cubins persist and ranks do not
    contend on an in-container cache lock. FLASHINFER_CUBINS_REPOSITORY and
    URM_NETRC are deliberately left unset: submit.py records that forcing the
    internal repository killed every mtp>0 GEN worker on hecate, while leaving
    flashinfer at its default resolved the same cubin hash cleanly.

    ON_OCI and GPU_NAME come from the detected cluster. ON_OCI is not cosmetic:
    with it unset the harness omits --gres, and oci-jhb then rejects the job
    outright ("Cannot find GPU specification ... partition: batch").
    """
    content = "\n".join(
        (
            f'PARTITION="{partition}"',
            f"ON_OCI={cluster.on_oci}",
            f"GPUS_PER_NODE={preset.gpus_per_node}",
            f'ACCOUNT="{account}"',
            f'IMAGE_NAME="{image}"',
            f'DATE_STR="{time.strftime("%Y%m%d")}"',
            f'GPU_NAME="{cluster.gpu_name}"',
            f'HOME_DIR="{home_dir}"',
            f'CONSTRAINT="{constraint}"',
            f'CONTAINER_MOUNTS="{cluster.container_mounts}"',
            f'FLASHINFER_CUBIN_DIR="{home_dir / "flashinfer_cubins"}"',
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return output


def _validate_guard_home(home_dir: Path, preset: K3Preset) -> None:
    """Check the ${home_dir} tree the source config templates against."""
    required = [home_dir / "Kimi-K3-NVFP4", home_dir / "hf_cache"]
    if preset.mode == GEN_ONLY:
        required.append(home_dir / "Kimi-K3-DSpark")
        required.append(
            home_dir
            / "datasets"
            / "k3-turnbatch"
            / f"k3-turnbatch-c{preset.benchmark_concurrency}_for_serve.json"
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Guard home {home_dir} is missing: {', '.join(missing)}")


def _prepare_inputs(
    benchmark_repo: Path,
    preset: K3Preset,
    work_dir: Path,
    wheel: Path | None,
    ctx_gpu_memory_fraction: float | None = None,
    gen_gpu_memory_fraction: float | None = None,
    job_time: str | None = None,
    trtllm_repo: Path | None = None,
) -> Path:
    """Stage the config and its generator in the layout the harness expects.

    Two lookups have to line up, which is why this mirrors the upstream tree
    rather than dumping both files side by side:

    * submit.py resolves the worker-config generator from the config file's
      directory, then its parent. Only the model-specific generator understands
      helix; the repo-root fallback does not and would run the point as plain
      TEP.
    * That generator then does
      ``sys.path.insert(0, __file__/../../lib)`` to import ``config_utils``, so
      it must sit exactly one level below a directory containing ``lib``.

    Result::

        <work_dir>/lib -> <benchmark_repo>/lib
        <work_dir>/<model>/gen_worker_config.py
        <work_dir>/<model>/agent_configs/<config>.yaml
    """
    source = _read_pinned_source_config(benchmark_repo, preset)
    _validate_source_config(source, preset)

    generator = _read_pinned_file(
        benchmark_repo,
        preset.source_revision,
        preset.generator_config,
        preset.generator_sha256,
        f"preset {preset.name!r} worker-config generator",
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    lib_link = work_dir / "lib"
    if lib_link.is_symlink() or lib_link.exists():
        lib_link.unlink()
    lib_link.symlink_to(benchmark_repo / "lib")

    model_dir = work_dir / preset.generator_config.parent.name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "gen_worker_config.py").write_bytes(
        _inject_worker_env(generator, preset.worker_env)
    )

    config = _select_pinned_row(source, preset)
    if ctx_gpu_memory_fraction is not None:
        config = _override_ctx_gpu_memory_fraction(config, ctx_gpu_memory_fraction)
    if gen_gpu_memory_fraction is not None:
        config = _override_gen_gpu_memory_fraction(config, gen_gpu_memory_fraction)
    if job_time is not None:
        config = _override_job_time(config, job_time)
    if trtllm_repo is not None:
        # Editable install straight from the checkout. The harness supports
        # this (submit.py forwards --trtllm-repo, and submit_ctx_only.py gates
        # on `trtllm_repo or trtllm_wheel_path`); the guard simply never used
        # it. It is the only way a Python-level change reaches a perf run
        # without a rebuild, which matters when iterating on one.
        #
        # It replaces the wheel rather than joining it: the harness would
        # otherwise have to pick, and a run whose Python came from one source
        # and whose binaries came from another is not attributable to either.
        # Only the Python layer is replaced -- the C++/CUDA objects still come
        # from the image, so the checkout must already carry built artifacts.
        config += f"\ntrtllm_install:\n  trtllm_repo: {trtllm_repo}\n"
    elif wheel is not None:
        # Install the branch wheel over the base image instead of shipping a
        # whole saved container. pyxis --container-save cannot export an image
        # that has bind mounts on this cluster (the enroot export cannot
        # traverse the mountpoints), and gating a wheel against a fixed base
        # image isolates the branch more cleanly anyway.
        config += f"\ntrtllm_install:\n  trtllm_wheel_path: {wheel}\n"

    output = model_dir / "agent_configs" / preset.source_config.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(config)
    return output


def _submit(
    args: argparse.Namespace,
    preset: K3Preset,
    cluster: Cluster,
    config: Path,
    server_config: Path,
    work_dir: Path,
) -> str | None:
    benchmark_repo = args.benchmark_repo.expanduser().resolve()
    submit_script = benchmark_repo / "submit.py"
    if not submit_script.is_file():
        raise ValueError(f"Benchmark submit script does not exist: {submit_script}")
    assert preset.single_string is not None  # _validate_source_config enforces this
    # -m all over a config already filtered to the single pinned row; see
    # _select_pinned_row for why -m single cannot be used here.
    command = [
        sys.executable,
        str(submit_script),
        "-m",
        "all",
        "-c",
        str(config),
        "-w",
        str(work_dir),
        "--server-config",
        str(server_config),
    ]
    if args.dry_run:
        command.append("--dry-run")
    print(f"Running: {' '.join(command)}")
    # Cluster-merged, not preset-raw: the generator reads these at import time
    # and a preset that cannot run as written on this hardware says so through
    # its overrides.
    environment = {**os.environ, **dict(preset.generator_env_for(cluster.name))}
    result = subprocess.run(
        command,
        cwd=benchmark_repo,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"submit.py failed with exit code {result.returncode}")
    if args.dry_run:
        return None
    job_ids = re.findall(r"Submitted batch job (\d+)", result.stdout)
    if not job_ids:
        raise RuntimeError("Could not find a submitted Slurm job id in submit.py output")
    return job_ids[-1]


def _normalize_slurm_state(line: str) -> str | None:
    fields = line.split("|")
    if len(fields) < 2:
        return None
    return fields[1].strip().split()[0] if fields[1].strip() else None


def _slurm_state(job_id: str) -> str | None:
    accounting = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-P", "-o", "JobID,State"],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in accounting.stdout.splitlines():
        state = _normalize_slurm_state(line)
        if state:
            return state
    return None


def _harness_finished(work_dir: Path, job_id: str) -> bool:
    """Whether the benchmark harness got far enough to write its done marker.

    The harness tears its own allocation down once the benchmark returns, so
    the job's Slurm state is CANCELLED even on a clean run -- both the
    reference reproduction (616181) and the branch run (618836) ended that
    way with complete artifacts. Gating on COMPLETED would therefore reject
    every run. The last thing the harness writes is 8_done_<job_id>.txt, so
    use that as the signal instead and let _check_result judge the numbers.
    """
    return any(work_dir.rglob(f"8_done_{job_id}.txt"))


def _wait_for_job(job_id: str, poll_interval: float, timeout: float, work_dir: Path) -> None:
    started_at = time.monotonic()
    previous_state = None
    while True:
        state = _slurm_state(job_id)
        if state != previous_state:
            print(f"Job {job_id}: {state or 'state unavailable'}")
            previous_state = state
        if state in TERMINAL_SLURM_STATES:
            if not _harness_finished(work_dir, job_id):
                raise RuntimeError(
                    f"Slurm job {job_id} finished in state {state} without the "
                    f"harness writing 8_done_{job_id}.txt"
                )
            return
        if time.monotonic() - started_at >= timeout:
            subprocess.run(["scancel", job_id], check=False)
            raise RuntimeError(f"Timed out waiting for Slurm job {job_id}; cancelled it")
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _resolve_cluster_defaults(args: argparse.Namespace, cluster: Cluster) -> None:
    """Fill the cluster-dependent arguments the caller left unset.

    Precedence is explicit flag, then the matching environment variable (both
    already applied by argparse), then the cluster record. Nothing here has a
    site-independent default: hecate's batch-xdr/cr and oci-jhb's batch/no
    constraint are not interchangeable, and a stale default silently submitting
    to the wrong partition is the failure this avoids.
    """
    if args.account is None:
        args.account = cluster.account
    if args.partition is None:
        args.partition = cluster.partition
    if args.constraint is None:
        args.constraint = cluster.constraint
    if args.benchmark_repo is None:
        args.benchmark_repo = cluster.benchmark_repo
    if args.home_dir is None:
        args.home_dir = cluster.guard_home


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(K3_PRESETS),
        default=DEFAULT_K3_PRESET,
        help="Reference run, topology, workload, reference metrics, and gate policy.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available presets and exit.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Evaluate an existing result directory with --preset; do not submit.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Evaluate the preset's own reference run; every gate must pass.",
    )
    parser.add_argument(
        "--cluster",
        choices=tuple(CLUSTERS),
        default=os.environ.get("TRTLLM_K3_GUARD_CLUSTER") or None,
        help=(
            "Force the cluster instead of detecting it. Selects the reference "
            "metrics, so getting this wrong grades against the wrong hardware."
        ),
    )
    parser.add_argument(
        "--establish-reference",
        action="store_true",
        help=(
            "Run the preset and report its metrics without gating them, for a "
            "cluster that has no baseline yet. Succeeds if the run completed "
            "cleanly; prints a pasteable ClusterReference block."
        ),
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("TRTLLM_TEST_IMAGE", ""),
        help="TensorRT-LLM SquashFS image (or set TRTLLM_TEST_IMAGE).",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help=(
            "Install this TensorRT-LLM wheel over --image at job start instead "
            "of requiring a branch-specific saved container."
        ),
    )
    parser.add_argument(
        "--trtllm-repo",
        type=Path,
        help=(
            "Install TensorRT-LLM editable from this checkout instead of a "
            "wheel. Replaces --wheel. Python layer only: the checkout must "
            "already contain built artifacts, and the C++/CUDA objects still "
            "come from the image."
        ),
    )
    parser.add_argument(
        "--job-time",
        help=(
            "Override the Slurm wall limit from the pinned source config "
            "(HH:MM:SS). Moves only the time limit, not the measured point."
        ),
    )
    parser.add_argument(
        "--ctx-gpu-memory-fraction",
        type=float,
        help=(
            "Override ctx_config.gpu_memory_fraction. Lowering it shrinks the "
            "KV cache size estimation pool, which is what a context worker "
            "OOMs on; the run then sits at a different operating point than "
            "the reference and its numbers are not directly comparable."
        ),
    )
    parser.add_argument(
        "--gen-gpu-memory-fraction",
        type=float,
        help=(
            "Override gen_config.gpu_memory_fraction. The decode worker sizes "
            "its KV cache from it, so lowering it is the lever for a generation "
            "worker that OOMs after the cache is allocated."
        ),
    )
    parser.add_argument(
        "--benchmark-repo",
        type=Path,
        default=_optional_path(os.environ.get("TRTLLM_K3_BENCH_REPO")),
        help="bench-trtllm-disagg checkout providing submit.py (default: per-cluster).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TRTLLM_K3_PERF_GUARD_DIR",
                SCRIPT_REPO_ROOT / "build_images" / "branch_guard_perf_k3",
            )
        ),
        help="Directory for the generated config and harness outputs.",
    )
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=_optional_path(os.environ.get("TRTLLM_K3_GUARD_HOME")),
        help=(
            "Shared home the source config templates ${home_dir} against; must "
            "hold Kimi-K3-NVFP4, Kimi-K3-DSpark, datasets/, and hf_cache "
            "(default: per-cluster)."
        ),
    )
    # These three default to the detected cluster's values rather than to
    # hecate's; see _resolve_cluster_defaults.
    parser.add_argument(
        "--account",
        default=os.environ.get("TRTLLM_SLURM_ACCOUNT"),
    )
    parser.add_argument(
        "--partition",
        default=os.environ.get("TRTLLM_SLURM_PARTITION"),
    )
    parser.add_argument(
        "--constraint",
        default=os.environ.get("TRTLLM_SLURM_CONSTRAINT"),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
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
        help="Prepare and validate the config without submitting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.list_presets:
            for name, preset in K3_PRESETS.items():
                print(f"{name}: {preset.description}")
                for override in preset.overrides:
                    print(f"    overrides[{override.cluster}]: {override.reason}")
                for reference in preset.references:
                    print(f"    reference[{reference.cluster}]: {reference.provenance}")
                    if reference.caveat:
                        print(f"        caveat: {reference.caveat}")
            return 0

        preset = K3_PRESETS[args.preset]
        cluster = _detect_cluster(args.cluster)
        _resolve_cluster_defaults(args, cluster)
        grade = _establish_reference if args.establish_reference else _check_result

        if args.self_check:
            # The stored reference run is a hecate artifact, so it is graded
            # against hecate's numbers regardless of where this script runs.
            print("Self-check against the preset's own reference run.\n")
            return 0 if _check_result(preset.reference_dir, preset, CLUSTERS[HECATE]) else 1

        if args.result_dir is not None:
            return 0 if grade(args.result_dir.expanduser().resolve(), preset, cluster) else 1

        if not args.establish_reference:
            # Fail before burning a 6-node allocation, not after: without a
            # baseline for this cluster the run cannot be graded at the end.
            preset.reference_for(cluster.name)

        if not args.image:
            raise ValueError("--image or TRTLLM_TEST_IMAGE is required")
        image = Path(args.image).expanduser().resolve()
        if not image.is_file():
            raise ValueError(f"Container image does not exist: {image}")

        benchmark_repo = args.benchmark_repo.expanduser().resolve()
        home_dir = args.home_dir.expanduser().resolve()
        _validate_guard_home(home_dir, preset)

        work_dir = args.work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        wheel = None
        if args.wheel is not None:
            wheel = args.wheel.expanduser().resolve()
            if not wheel.is_file():
                raise ValueError(f"Wheel does not exist: {wheel}")
        override = preset.overrides_for(cluster.name)
        ctx_gpu_memory_fraction = args.ctx_gpu_memory_fraction
        gen_gpu_memory_fraction = args.gen_gpu_memory_fraction
        if override is not None:
            # Explicit beats the preset's own override, so a one-off
            # investigation can still move a fraction by hand.
            if ctx_gpu_memory_fraction is None:
                ctx_gpu_memory_fraction = override.ctx_gpu_memory_fraction
            if gen_gpu_memory_fraction is None:
                gen_gpu_memory_fraction = override.gen_gpu_memory_fraction
            print(f"Cluster overrides:      {override.reason}")
            for name, value in override.generator_env:
                print(f"  generator env:        {name}={value}")
        for label, value, explicit in (
            ("ctx", ctx_gpu_memory_fraction, args.ctx_gpu_memory_fraction),
            ("gen", gen_gpu_memory_fraction, args.gen_gpu_memory_fraction),
        ):
            if value is not None:
                source = (
                    f"--{label}-gpu-memory-fraction" if explicit is not None else "preset override"
                )
                print(f"  {label} gpu mem fraction: {value} ({source})")
        config = _prepare_inputs(
            benchmark_repo,
            preset,
            work_dir,
            wheel,
            ctx_gpu_memory_fraction,
            gen_gpu_memory_fraction,
            args.job_time,
            args.trtllm_repo,
        )
        server_config = _write_server_config(
            work_dir / "server-branchguard.config",
            preset,
            cluster,
            image,
            args.account,
            args.partition,
            args.constraint,
            home_dir,
        )
        print(f"Cluster:                {cluster.name} ({cluster.gpu_name})")
        print(f"Prepared config:        {config}")
        print(f"Prepared server config: {server_config}")

        job_id = _submit(args, preset, cluster, config, server_config, work_dir)
        if args.dry_run:
            print("Dry run complete; nothing submitted.")
            return 0

        print(f"Submitted Slurm job {job_id}")
        _wait_for_job(job_id, args.poll_interval, args.wait_timeout, work_dir)
        return 0 if grade(work_dir, preset, cluster) else 1
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
