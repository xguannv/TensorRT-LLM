#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uex

PARTITION=${1:-batch-xdr}
CLEAN=${2:-""}

ACCOUNT=${ACCOUNT:-coreai_comparch_aarwlt}
REPO_DIR=`readlink -f $PWD`
MOUNT="/lustre:/lustre,${REPO_DIR}:/code/tensorrt_llm"
WORKDIR="/code/tensorrt_llm"

source ${REPO_DIR}/jenkins/current_image_tags.properties
IMAGE=${LLM_SBSA_RUBIN_DOCKER_IMAGE}

HASH=$(git -C ${REPO_DIR} rev-parse --short HEAD)
BUILD_DIR="${REPO_DIR}/build_images/${HASH}"
PUBLISH_ROOT=${PUBLISH_ROOT:-/lustre/share/coreai_comparch_trtllm/lizhiz/rubin_k3}
PUBLISH_DIR="${PUBLISH_ROOT}/${HASH}"
JOB_NAME="${ACCOUNT}-rubin-build.${HASH}"
EXTRA_FLAGS="--extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl --nvrtc_dynamic_linking"

if [[ -d ${BUILD_DIR} ]]; then
    rm -r ${BUILD_DIR}
fi
mkdir -p ${BUILD_DIR}

rm ${REPO_DIR}/build/tensorrt_llm-*.whl || true

NSYS_VERSION="2026.6.0.224-266038638214v0"
NSYS_TARGET="${NSYS_VERSION%%-*}"
NSYS_URL="http://nsys/dvs-nightlies/QuadDBringup/${NSYS_VERSION}/nsight-systems-cli-sbsa-DVS.tar.gz"
NSYS_CACHE="${PUBLISH_ROOT}/nsys/nsight-systems-cli-sbsa-DVS-${NSYS_VERSION}.tar.gz"
#INSTALL_NSYS="NSYS_ROOT=/opt/nvidia/nsight-systems/${NSYS_TARGET}; \
#  NSYS_ARCHIVE=/tmp/nsight-systems-cli-sbsa-DVS-${NSYS_TARGET}.tar.gz; \
#  NSYS_CACHE='${NSYS_CACHE}'; \
#  mkdir -p \"\${NSYS_ROOT}\" \"\$(dirname \"\${NSYS_CACHE}\")\"; \
#  if test -s \"\${NSYS_CACHE}\"; then cp \"\${NSYS_CACHE}\" \"\${NSYS_ARCHIVE}\"; \
#  else wget --progress=dot:giga --tries=3 '${NSYS_URL}' -O \"\${NSYS_ARCHIVE}\" && cp \"\${NSYS_ARCHIVE}\" \"\${NSYS_CACHE}\"; fi; \
#  tar -xzf \"\${NSYS_ARCHIVE}\" -C \"\${NSYS_ROOT}\"; \
#  NSYS_BIN=\$(find \"\${NSYS_ROOT}\" -type f -path '*/target-linux-sbsa-armv8/nsys' -print -quit); \
#  test -n \"\${NSYS_BIN}\"; \
#  ln -sf \"\${NSYS_BIN}\" /usr/local/bin/nsys; \
#  ln -sf \"\${NSYS_BIN}\" /usr/local/cuda/bin/nsys; \
#  hash -r; \
#  nsys --version | tee /tmp/nsys-version.txt; \
#  grep -q '${NSYS_VERSION}' /tmp/nsys-version.txt"

INSTALL_NSYS="echo 'skip nsys'"

srun -A ${ACCOUNT} -p ${PARTITION} --job-name=${JOB_NAME} --container-image=${IMAGE} \
    --container-mounts=${MOUNT} \
    --container-workdir=${WORKDIR} \
    --container-save=${BUILD_DIR}/trtllm.sqsh \
    -N 1 --ntasks=1 --ntasks-per-node=1  -t 2:00:00 \
    bash -c "python3 -m pip install --upgrade -r requirements-dev.txt && python scripts/build_wheel.py -G Ninja -a '107-real' $EXTRA_FLAGS $CLEAN  && pip install build/tensorrt_llm-*.whl && ${INSTALL_NSYS} "
cp ${REPO_DIR}/build/tensorrt_llm-*.whl ${BUILD_DIR}/
mkdir -p "${PUBLISH_DIR}"
cp "${BUILD_DIR}/trtllm.sqsh" "${PUBLISH_DIR}/"
cp "${BUILD_DIR}"/tensorrt_llm-*.whl "${PUBLISH_DIR}/"
