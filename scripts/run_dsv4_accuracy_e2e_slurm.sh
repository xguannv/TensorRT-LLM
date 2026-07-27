#!/usr/bin/env bash
#
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

set -euo pipefail

readonly SCRIPT_PATH="${DSV4_LAUNCHER_PATH:-$(readlink -f "${BASH_SOURCE[0]}")}"
readonly DEFAULT_REPO_ROOT="${DSV4_REPO_ROOT:-$(git -C "$(dirname "${SCRIPT_PATH}")" rev-parse --show-toplevel)}"
readonly DEFAULT_IMAGE="${DEFAULT_REPO_ROOT}/build_images/0b0beeba0c/trtllm.sqsh"
readonly TEST_TARGET="tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestDeepSeekV4Pro::test_gsm8k_full_accuracy"

usage() {
    echo "Usage: $0 [tp4-mtp0|dep4-mtp3] [IMAGE]"
}

run_rank() {
    flock "${DSV4_PIP_LOCK}" \
        python3 -m pip install --no-deps --no-build-isolation \
        -e "${DSV4_REPO_ROOT}"

    export LLM_MODELS_ROOT=/lustre/fsw/coreai_comparch_trtllm/common/llm-models
    export PMIX_MCA_gds=hash
    export TLLM_LOG_LEVEL=INFO
    export HF_HOME=/lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache
    export PYTHONPATH=
    export TRTLLM_ACCURACY_NO_REFERENCE=1

    case "${DSV4_TEST_MODE}" in
    tp4-mtp0)
        export DSV4_DIAG_DISABLE_ATTN_DP=1
        export DSV4_DIAG_MTP_NEXTN=0
        ;;
    dep4-mtp3)
        export DSV4_DIAG_MTP_NEXTN=3
        export TRTLLM_MOE_A2A_DISABLE_CFT_COUNTED_WRITES=1
        ;;
    *)
        echo "Unsupported mode: ${DSV4_TEST_MODE}" >&2
        exit 2
        ;;
    esac

    cd /tmp
    exec trtllm-llmapi-launch python3 -m pytest -vs --rootdir=/tmp \
        "${DSV4_REPO_ROOT}/${TEST_TARGET}"
}

run_batch() {
    exec srun --kill-on-bad-exit=1 --mpi=pmix --ntasks=4 --ntasks-per-node=4 \
        --container-image="${DSV4_IMAGE}" \
        --container-mounts=/lustre:/lustre \
        "${SCRIPT_PATH}" --rank
}

submit_test() {
    local mode="${1:-tp4-mtp0}"
    local image="${2:-${DEFAULT_IMAGE}}"

    case "${mode}" in
    tp4-mtp0 | dep4-mtp3) ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
    esac

    if [[ ! -f "${image}" ]]; then
        echo "Image does not exist: ${image}" >&2
        exit 2
    fi

    local image_dir log_dir log_pattern job_id log_path job_mode
    image="$(readlink -f "${image}")"
    image_dir="$(dirname "${image}")"
    log_dir="${image_dir}/test_logs"
    mkdir -p "${log_dir}"
    job_mode="${mode//-/_}"
    log_pattern="${log_dir}/dsv4_acc_${job_mode}_%j.log"

    job_id="$(
        sbatch --parsable \
            --job-name="coreai_comparch_trtllm-dsv4.acc_${job_mode}" \
            --partition=batch-xdr \
            --account=coreai_comparch_trtllm \
            --qos=normal \
            --nodes=1 \
            --ntasks=4 \
            --ntasks-per-node=4 \
            --exclusive \
            --time=05:00:00 \
            --output="${log_pattern}" \
            --error="${log_pattern}" \
            --export="ALL,DSV4_IMAGE=${image},DSV4_LAUNCHER_PATH=${SCRIPT_PATH},DSV4_PIP_LOCK=${image_dir}/pip_editable.lock,DSV4_REPO_ROOT=${DEFAULT_REPO_ROOT},DSV4_TEST_MODE=${mode}" \
            "${SCRIPT_PATH}" --batch
    )"
    job_id="${job_id%%;*}"
    log_path="${log_pattern//%j/${job_id}}"

    echo "job_id=${job_id}"
    echo "error_log=${log_path}"
}

case "${1:-}" in
--batch)
    run_batch
    ;;
--rank)
    run_rank
    ;;
*)
    submit_test "$@"
    ;;
esac
