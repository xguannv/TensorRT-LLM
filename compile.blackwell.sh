#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uex

PARTITION=${1:-batch}
CLEAN=${2:---clean}

ACCOUNT=${ACCOUNT:-coreai_comparch_inferencex}
REPO_DIR=$(readlink -f "$PWD")
MOUNT="/lustre:/lustre,${REPO_DIR}:/code/tensorrt_llm"
WORKDIR="/code/tensorrt_llm"

source "${REPO_DIR}/jenkins/current_image_tags.properties"
IMAGE=${LLM_SBSA_DOCKER_IMAGE}

HASH=$(git -C "${REPO_DIR}" rev-parse --short HEAD)
BUILD_DIR="${REPO_DIR}/build_images/${HASH}-blackwell"
JOB_NAME="${ACCOUNT}-blackwell-build.${HASH}"

if [[ -d ${BUILD_DIR} ]]; then
    rm -r "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

rm "${REPO_DIR}"/build/tensorrt_llm-*.whl || true

srun -A "${ACCOUNT}" -p "${PARTITION}" --job-name="${JOB_NAME}" --container-image="${IMAGE}" -G 4 \
    --container-mounts="${MOUNT}" \
    --container-workdir="${WORKDIR}" \
    --container-save="${BUILD_DIR}/trtllm.sqsh" \
    -N 1 --ntasks=1 --ntasks-per-node=1 -t 2:00:00 \
    bash -c "python3 -m pip install --upgrade -r requirements-dev.txt && python scripts/build_wheel.py -G Ninja -a '103-real' ${CLEAN} && pip install build/tensorrt_llm-*.whl"

cp "${REPO_DIR}"/build/tensorrt_llm-*.whl "${BUILD_DIR}/"
