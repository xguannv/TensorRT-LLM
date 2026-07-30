#!/bin/bash
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#
set -u
set -e
set -x

role=${1}
instance_id=${2}
model_path=${3}
port=${4}
numa_bind=${5}
log_dir=${6}
enable_nsys=${7}
config_file=${8}
# CUDA_VISIBLE_DEVICES selection:
#   - Default packing (no gpu_map file): each node is dedicated to one
#     worker, so SLURM_LOCALID maps directly to the physical GPU id.
#   - Compact packing (gpu_map file emitted by submit.py): two workers may
#     share a node and would both see LOCALID=0, so look up the per-worker
#     gpu_map "<rank> <host> <local_gpu_id>" by SLURM_PROCID. srun
#     --distribution=arbitrary assigns PROCID in hostfile order, so it
#     indexes directly into the map.
gpu_map_file="${log_dir}/gpu_map_${role}_${instance_id}.txt"
if [ -f "${gpu_map_file}" ]; then
    gpu_id=$(awk -v p="${SLURM_PROCID}" '$1==p {print $3; exit}' "${gpu_map_file}")
    if [ -z "${gpu_id}" ]; then
        echo "ERROR: no GPU mapping for SLURM_PROCID=${SLURM_PROCID} in ${gpu_map_file}" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=${gpu_id}
elif [ "${TRTLLM_DISABLE_GPU_MASK:-0}" = "1" ]; then
    # MegaMoE opt-out. Torch symmetric-memory rendezvous compares device
    # ordinals across ranks, and MegaMoEDeepGemm picks its device as
    #   local_rank % torch.cuda.device_count()
    # so masking each rank to a single GPU makes device_count 1, every rank
    # resolve to cuda:0, and the rendezvous abort with
    #   CUDASymmetricMemoryAllocator::rendezvous: detected allocations from
    #   overlapping devices from different ranks
    # Leaving every node GPU visible restores distinct ordinals. Safe only for
    # default packing (one worker per node); the gpu_map branch above still
    # masks, because there two workers share a node and would collide.
    echo "TRTLLM_DISABLE_GPU_MASK=1: leaving all node GPUs visible (CUDA_VISIBLE_DEVICES unset)"
else
    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
fi

# Container runtimes (pyxis/enroot) reset image-defined variables like PATH
# at container start, so values passed via srun --export are lost for them.
# Allow the launcher config to prepend entries from inside the container.
if [ -n "${TRTLLM_PATH_PREPEND:-}" ]; then
    export PATH="${TRTLLM_PATH_PREPEND}:${PATH}"
fi
if [ -n "${TRTLLM_PYTHONPATH_PREPEND:-}" ]; then
    export PYTHONPATH="${TRTLLM_PYTHONPATH_PREPEND}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Clear UCX_TLS for specific clusters. Some clusters instead need an
# explicit transport list (e.g. NVL72 nodes whose verbs transports cannot
# initialize): set TRTLLM_WORKER_UCX_TLS in worker_env_var to re-pin
# UCX_TLS here, after the container-provided value is cleared.
if [ -n "${TRTLLM_WORKER_UCX_TLS:-}" ]; then
    export UCX_TLS="${TRTLLM_WORKER_UCX_TLS}"
else
    unset UCX_TLS
fi

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200/GB300 NVL72, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

echo "config_file: ${config_file}"

nsys_prefix=()
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_bin="${NSYS_BIN:-nsys}"
    if ! command -v "${nsys_bin}" >/dev/null 2>&1; then
        echo "ERROR: Nsight Systems executable not found: ${nsys_bin}" >&2
        exit 1
    fi
    echo "Using Nsight Systems executable: $(command -v "${nsys_bin}")"
    "${nsys_bin}" --version

    nsys_file=${log_dir}/nsys_worker_proc_${role}_${instance_id}_${SLURM_PROCID}
    echo "nsys is enabled on ${role} GPUs, TLLM_PROFILE_START_STOP=${TLLM_PROFILE_START_STOP}"
    nsys_prefix=(
        "${nsys_bin}" profile
        -o "${nsys_file}"
        -f true
        -t cuda,nvtx,python-gil
        -c cudaProfilerApi
        --cuda-graph-trace node
        --capture-range-end=stop
        --gpu-metrics-devices=none
    )
fi

# In-place (.pth-style) TRT-LLM installs may lack the trtllm-serve console
# script; fall back to the module entry point in that case.
trtllm_serve_cmd="trtllm-serve"
if ! command -v trtllm-serve >/dev/null 2>&1; then
    trtllm_serve_cmd="python3 -m tensorrt_llm.commands.serve"
fi

"${nsys_prefix[@]}" trtllm-llmapi-launch ${numa_bind_cmd} \
    ${trtllm_serve_cmd} ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file}
