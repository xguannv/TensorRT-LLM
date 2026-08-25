# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNNER_PATH = REPO_ROOT / "scripts" / "run_branch_guard_perf.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_branch_guard_perf", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_runner()


def test_all_measured_pareto_presets_are_registered() -> None:
    assert len(guard.PERF_PRESETS) == 10
    assert len({preset.recipe for preset in guard.PERF_PRESETS.values()}) == 10
    assert len({preset.reference_sha256 for preset in guard.PERF_PRESETS.values()}) == 10
    for name, preset in guard.PERF_PRESETS.items():
        assert preset.name == name
        assert (
            preset.gpu_count == (preset.prefill_nodes + preset.decode_nodes) * preset.gpus_per_node
        )
        assert preset.reference_tps_per_user == pytest.approx(
            1000.0 / preset.reference_p90_itl_ms, abs=0.2
        )


def _reference_recipe() -> str:
    return """name: test-recipe
model:
  container: /old/image.sqsh
identity:
  container:
    image: /old/image.sqsh
resources:
  gpus_per_node: 4
  prefill_nodes: 1
  decode_nodes: 6
backend:
  trtllm_config:
    prefill:
      kv_cache_config:
        block_reuse_config:
          block_reuse_policy: per_conversation
      moe_config:
        backend: CUTEDSL
    decode:
      moe_config:
        backend: TRTLLM
benchmark:
  env:
    AIPERF_DATASET_MMAP_CACHE_DIR: /old/cache
  command: |
    aiperf profile --benchmark-duration 3600 --concurrency 6 \\
      --output-artifact-dir /logs/agentic/concurrency_6
"""


def test_prepare_recipe_preserves_preset_backends(tmp_path: Path) -> None:
    preset = guard.PERF_PRESETS[guard.DEFAULT_PERF_PRESET]
    output = tmp_path / "recipe.yaml"

    guard._prepare_recipe(
        _reference_recipe(),
        output,
        Path("/new/image.sqsh"),
        Path("/shared/cache"),
        preset,
    )

    generated = output.read_text()
    assert "setup_script: reinstall-tensorrt-llm-repo.sh" in generated
    assert "backend: CUTEDSL" in generated
    assert "backend: TRTLLM" in generated
    assert "backend: MEGAMOE_CUTEDSL" not in generated
    assert generated.count("/new/image.sqsh") == 2
    assert "AIPERF_DATASET_MMAP_CACHE_DIR: /shared/cache" in generated
    assert "policy: per_conversation" in generated


def test_validate_reference_recipe_rejects_backend_drift() -> None:
    preset = guard.PERF_PRESETS[guard.DEFAULT_PERF_PRESET]
    recipe = _reference_recipe().replace("backend: CUTEDSL", "backend: MEGAMOE_CUTEDSL")

    with pytest.raises(ValueError, match="prefill MoE backend"):
        guard._validate_reference_recipe(recipe, preset)


def test_parse_metrics_treats_missing_error_row_as_zero(tmp_path: Path) -> None:
    result_csv = tmp_path / "profile_export_aiperf.csv"
    result_csv.write_text(
        "Metric,avg,p50,p90\n"
        "Benchmark Duration (sec),3601.98,,\n"
        "Request Count,748,,\n"
        "Total Token Throughput (tokens/sec),33685.34,,\n"
        "Full-Response Inter Token Latency (ms),,,3.79\n"
        "Time to First Token (ms),,575.97,\n"
    )
    preset = guard.PERF_PRESETS[guard.DEFAULT_PERF_PRESET]

    metrics = guard._parse_metrics(result_csv, preset)

    assert metrics.request_count == 748
    assert metrics.error_count == 0
    assert metrics.error_rate == 0.0
    assert metrics.tps_per_gpu == pytest.approx(33685.34 / 28)
