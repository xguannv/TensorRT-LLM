/*
 * Copyright 1993-2022 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_device_h__
#define __cuda_etbl_tools_device_h__

#include "cuda.h"
#include "g_nvconfig.h"

#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "cuda_uuid.h"

#include "tools_common_types.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    // =============================================================================
    // DEVICE
    // =============================================================================

    // CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE
    // Base enumeration value for attributes that can be retrieved from the CUDA API
    // using a method other than cuDeviceGetAttribute.  (e.g. cuDeviceGetName)
    //
    // CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE
    // Base enumeration value for attributes that cannot be retrieved using the
    // CUDA Driver API.
    //
    // CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE
    // Base enumeration value for directly mapping to the internal CUdeviceLimits.
    //
    // These base numbers should be sufficiently high as to ensure that there is
    // never a conflict between CUdevice_attribute and CUtools_device_attribute.
    //
    // All enumeration values should be explicitly set.  Do not rely on auto-increment.

    enum
    {
        CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE = 0x10000000UL,
        CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE = 0x20000000UL,
        CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE = 0x30000000UL,
    };

// Add attributes that map to CUdevice_attribute_enum.  Blank lines are just to match cuda.h, for side-by-side
// comparison.
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_01                                                                           \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),                             \
                                                                                                                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(TOTAL_CONSTANT_MEMORY, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(WARP_SIZE, CU_DEVICE_ATTRIBUTE_WARP_SIZE),                                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_PITCH, CU_DEVICE_ATTRIBUTE_MAX_PITCH),                                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAX_REGISTERS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),        \
                                                                                                                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CLOCK_RATE, CU_DEVICE_ATTRIBUTE_CLOCK_RATE),                                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(TEXTURE_ALIGNMENT, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_OVERLAP, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP),                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),              \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(KERNEL_EXEC_TIMEOUT, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(INTEGRATED, CU_DEVICE_ATTRIBUTE_INTEGRATED),                                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CAN_MAP_HOST_MEMORY, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(COMPUTE_MODE, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE),                              \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_TEXTURE1D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_TEXTURE2D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_TEXTURE2D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_TEXTURE3D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_TEXTURE3D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_TEXTURE3D_DEPTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS),                   \
                                                                                                                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(SURFACE_ALIGNMENT, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CONCURRENT_KERNELS, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(ECC_ENABLED, CU_DEVICE_ATTRIBUTE_ECC_ENABLED),                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(PCI_BUS_ID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID),                                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(PCI_DEVICE_ID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID),                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(TCC_DRIVER, CU_DEVICE_ATTRIBUTE_TCC_DRIVER),                                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MEMORY_CLOCK_RATE, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GLOBAL_MEMORY_BUS_WIDTH, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(L2_CACHE_SIZE, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_THREADS_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(ASYNC_ENGINE_COUNT, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(UNIFIED_ADDRESSING, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE1D_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE1D_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CAN_TEX2D_GATHER, CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER),                      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_GATHER_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_GATHER_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE),               \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(PCI_DOMAIN_ID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID),                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(TEXTURE_PITCH_ALIGNMENT, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURECUBEMAP_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS),         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_SURFACE1D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_SURFACE2D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_SURFACE2D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_SURFACE3D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_SURFACE3D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MAXIMUM_SURFACE3D_DEPTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACE1D_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACE1D_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACE2D_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACE2D_LAYERED_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACE2D_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACECUBEMAP_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS),         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE1D_LINEAR_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_LINEAR_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_LINEAR_PITCH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT),               \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            STREAM_PRIORITIES_SUPPORTED, CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GLOBAL_L1_CACHE_SUPPORTED, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED),    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LOCAL_L1_CACHE_SUPPORTED, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_REGISTERS_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MANAGED_MEMORY, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MULTI_GPU_BOARD, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MULTI_GPU_BOARD_GROUP_ID, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            HOST_NATIVE_ATOMIC_SUPPORTED, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO),         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(PAGEABLE_MEMORY_ACCESS, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CONCURRENT_MANAGED_ACCESS, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS),    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            COMPUTE_PREEMPTION_SUPPORTED, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM),     \
                                                                                                                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(COOPERATIVE_LAUNCH, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            COOPERATIVE_MULTI_DEVICE_LAUNCH, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CAN_FLUSH_REMOTE_WRITES, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(HOST_REGISTER_SUPPORTED, CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,                                 \
            CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES),                                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST),             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,                                  \
            CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED),                                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED),               \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED),       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_BLOCKS_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GENERIC_COMPRESSION_SUPPORTED, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_PERSISTING_L2_CACHE_SIZE, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_ACCESS_POLICY_WINDOW_SIZE, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED),     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            RESERVED_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            SPARSE_CUDA_ARRAY_SUPPORTED, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MEMORY_POOLS_SUPPORTED, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_DIRECT_RDMA_SUPPORTED, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED),    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_DIRECT_RDMA_WRITES_ORDERING, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MEMPOOL_SUPPORTED_HANDLE_TYPES, CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CLUSTER_LAUNCH, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED, CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED),         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IPC_EVENT_SUPPORTED, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(NUMA_CONFIG, CU_DEVICE_ATTRIBUTE_NUMA_CONFIG),                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(NUMA_ID, CU_DEVICE_ATTRIBUTE_NUMA_ID),

#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_01                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CAN_USE_STREAM_MEM_OPS_V1, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            CAN_USE_64_BIT_STREAM_MEM_OPS_V1, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            CAN_USE_STREAM_WAIT_VALUE_NOR_V1, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            CAN_USE_64_BIT_STREAM_MEM_OPS, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            CAN_USE_STREAM_WAIT_VALUE_NOR, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR),
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_02                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(DMA_BUF_SUPPORTED, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED),
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_03                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MEM_SYNC_DOMAIN_COUNT, CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT),
#if NVCFG(GLOBAL_FEATURE_CTK3152_TMA_ENCODER_SUPPORT)
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_04                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(TENSOR_MAP_ACCESS_SUPPORTED, CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED),
#else
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_04
#endif
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_05                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(HANDLE_TYPE_FABRIC_SUPPORTED, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED),
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_06                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(UNIFIED_FUNCTION_POINTERS, CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS),
#if NVCFG(GLOBAL_FEATURE_MULTICAST_FABRIC)
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_08                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MULTICAST_SUPPORTED, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED),
#else
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_08
#endif
#if NVCFG(GLOBAL_FEATURE_CTK4771_MPS_STATUS_DEVICE_PROP)
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_09                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(MPS_ENABLED, CU_DEVICE_ATTRIBUTE_MPS_ENABLED),
#else
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_09
#endif
#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_10                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(HOST_NUMA_ID, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID),

// Add attributes that can be queried through functions in cuda.h
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PUBLIC                                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(DISPLAY_NAME, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 0),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            COMPUTE_CAPABILITY_MAJOR, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 1),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            COMPUTE_CAPABILITY_MINOR, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 2),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(TOTAL_MEMORY, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 3),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(RAM_TYPE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 4),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(RAM_LOCATION, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 5),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_PCI_DEVICE_ID, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 6),       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_PCI_SUB_SYSTEM_ID, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 7),   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_PCI_REVISION_ID, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 8),     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_PCI_EXT_DEVICE_ID, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 9),   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_PCI_EXT_GEN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 10),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(GPU_PCI_EXT_GPU_GEN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 11),    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_PCI_EXT_GPU_LINK_RATE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 12),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_PCI_EXT_GPU_LINK_WIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 13),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_PCI_EXT_DOWNSTREAM_LINK_RATE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 14),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            GPU_PCI_EXT_DOWNSTREAM_LINK_WIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 15),

// Add private attributes
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_01                                                                                                              \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(DEVICE_INDEX, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 0),                                                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(ARCHITECTURE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 1),                                                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(CHIP, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 2),                                                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(BUS_TYPE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 3),                                                            \
        /* DEPRECATED:                   COMPUTE_MAJOR, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 4), */ /* DEPRECATED:                                     \
                                                                                                                    COMPUTE_MINOR,                                  \
                                                                                                                    CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE \
                                                                                                                    +                                               \
                                                                                                                    5),                                             \
                                                                                                                  */                                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(ASYNC_ENGINE,                                                                                                              \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE                                                                                                         \
                + 6), /* DEPRECATED:                   ASYNC_ENGINE_COUNT,                                                                                          \
                         CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 7), */                                                                                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(NAME, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 8),                                                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(ISA_VERSION, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 9),                                                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_QUADRO, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 10),                                                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_TESLA, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 11),                                                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(PTIMER_SCALE_FACTOR, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 12),                                                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_TEGRA, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 13),                                                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(VIRTUALIZATION_MODE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 14),

#if NVCFG(GLOBAL_FEATURE_INCLUDE_CUDA_WDDM)
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_NVCFG_01                                                           \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_WDDM_HARDWARE_SCHEDULING, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 15),
#else
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_NVCFG_01
#endif

#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_02                                                                 \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LMEM_STACK_POINTER, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 16),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LMEM_THREAD_STATE_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 17),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LMEM_SAVED_STACK_POINTER, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 18),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LMEM_SAVED_RETURN_PC, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 19),  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            VGPU_DIRECT_RDMA_SUPPORTED, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 20),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            VGPU_TSG_TIMESLICE_OVERRIDE_SUPPORTED, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 21),

#if NVCFG(GLOBAL_FEATURE_INCLUDE_CUDA_WDDM)
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_NVCFG_02                                                           \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_WDDM_CLIENT_VIRTUALIZED, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 22),
#else
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_NVCFG_02
#endif

#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_03                                                                 \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_CMP_SKU, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 23),                \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            PCIE_SUPPORTED_GPU_ATOMICS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 24),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            SELF_HOSTED_CAPABILITY, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 25),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(IS_SLI, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE + 26),

// Add private attributes that map to device limits
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS_01                                                          \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_NUM_GPCS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 0),     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_NUM_TPCS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 1), \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SM_PER_TPC, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 2),                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_NUM_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 3),   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_MAX_TPCS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 4), \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_SM_PER_TPC, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 5),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_THRID_PER_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 6),                          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_WARPS_PER_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 6),                          \
        /* ALIAS OF THRID_PER_SM */ /* REMOVED:                       LIMITS_SM_SIG */ /* REMOVED:                     \
                                                                                          LIMITS_PM_REGS               \
                                                                                        */                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_LOCAL_REGS_PER_BLOCK, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 9),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_PER_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 10),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_REF_CNTS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 11),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_GMEM_REGIONS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 12),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_CONST_POOLS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 13),                      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_CONST_MEM_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 14),                      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_TEXTURES, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 15),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_PARAMS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 16),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_PARAMS_OVERFLOW, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 17),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_SPECIAL_GRID_PARAM_SIZE,                                               \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 18), /* REMOVED: LIMITS_CONST_TABLE_SIZE */       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TOTAL_CONST_MEM_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 20),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_THREAD_DIMENSION, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 21),             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_THREAD_DIM0, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 22),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_THREAD_DIM1, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 23),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_THREAD_DIM2, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 24),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_GRID_WIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 25),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CTA_GRID_HEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 26),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_MAX_CTA_GRID_DEPTH,                                                    \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 27), /* REMOVED: LIMITS_GOB_SIZE_BYTES */         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_GOB_WIDTH_BYTES, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 29),                      \
        /* UNUSED: LIMITS_GOB_HEIGHT, */ /* UNUSED:                                                                    \
                                            LIMITS_GOB_DEPTH,                                                          \
                                          */                                                                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_TEXHDR_SLOTS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 30),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_NUM_TEXSMP_SLOTS,                                                      \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 31), /* REMOVED: LIMITS_TEXHDR_SIZE_BYTES */      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_TEXHDR_SIZE_DWORDS,                                                    \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 33), /* REMOVED: LIMITS_TEXSAMPLER_SIZE_BYTES */  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXSAMPLER_SIZE_DWORDS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 35),               \
        /* REMOVED: LIMITS_NUM_TEXHDR_INDEX */ /* REMOVED:                                                             \
                                                  LIMITS_NUM_TEXSMP_INDEX                                              \
                                                */                                                                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_SHARED_MEM_SIZE_PER_BLOCK, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 38),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SHARED_MEM_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 39),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_MEM2MEM_PITCH,                                                         \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 40), /* REMOVED: LIMITS_MEM2MEM_STRIDELG */       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MEM2MEM_WIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 42),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MEM2MEM_HEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 43),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MEM2MEM_BUFFER_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 44),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MEM2MEM_BUFFER_COUNT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 45),                 \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TWOD_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 46),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TWOD_PITCH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 47),                           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TWOD_PITCH_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 48),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_WARP_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 49),                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_LMEM_VECTOR_BYTES, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 50),                    \
        /* REMOVED: LIMITS_LMEM_MAX_SIZE */ /* REMOVED:                                                                \
                                               LIMITS_LMEM_MAX_THEADS                                                  \
                                             */                                                                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 53),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE1D_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 54),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 55),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 56),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE3D_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 57),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE3D_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 58),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE3D_MAXDEPTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 59),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 60),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_SURFACES, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 61),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_CONSTANT_BANKS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 62),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_FUNC_MEM_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 63),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MEMBAR_ALLOCATION_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 64),               \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_LAYERED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 65),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_LAYERED_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 66),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_LAYERED_MAXLAYERS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 67),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_GLOBAL_L1_CACHE_MIN_SIZE_PER_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 68),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_GLOBAL_L1_CACHE_LINE_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 69),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_COPY_ENGINE_PITCH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 70),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE1D_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 71),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE2D_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 72),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE2D_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 73),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE3D_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 74),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE3D_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 75),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE3D_MAXDEPTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 76),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_FB_BUS_WIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 77),                         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_L2_CACHE_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 78),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_FBPS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 79),                             \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE1D_LAYERED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 80),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE1D_LAYERED_MAXLAYERS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 81),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE_PITCH_ALIGN, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 82),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURECUBEMAP_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 83),              \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURECUBEMAP_LAYERED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 84),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURECUBEMAP_LAYERED_MAXLAYERS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 85),     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE1D_LAYERED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 86),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE1D_LAYERED_MAXLAYERS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 87),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE2D_LAYERED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 88),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE2D_LAYERED_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 89),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACE2D_LAYERED_MAXLAYERS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 90),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACECUBEMAP_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 91),              \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACECUBEMAP_LAYERED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 92),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_SURFACECUBEMAP_LAYERED_MAXLAYERS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 93),     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE1D_LINEAR_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 94),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_LINEAR_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 95),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_LINEAR_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 96),           \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_LINEAR_MAXPITCH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 97),            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_MIPMAPPED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 98),         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_MIPMAPPED_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 99),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE1D_MIPMAPPED_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 100),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_SHARED_MEM_SIZE_PER_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 101),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_LOCAL_REGS_PER_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 102),                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_TPCS_PER_GPC, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 103),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_MAX_REGS_PER_THREAD,                                                   \
            CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE                                                     \
                + 104), /* REMOVED:                      LIMITS_SPECIAL_GRID_RESERVED_PARAM_SIZE  */                   \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_MAX_GATHER_WIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 106),          \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE2D_MAX_GATHER_HEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 107),         \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE3D_ALTERNATE_MAXWIDTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 108),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE3D_ALTERNATE_MAXHEIGHT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 109),       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_TEXTURE3D_ALTERNATE_MAXDEPTH, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 110),        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_FUNCTION_PADDING, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 111),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_NUM_L2_SLICES, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 112),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_L2_SLICES_PER_L2_CACHE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 113),              \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_L2_CACHE_PER_FBP, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 114),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_PARTITION_AS_SM, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 115),                     \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_MAX_FBPS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 116),                            \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            LIMITS_FBP_DISABLE_MASK, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 117),                    \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(LIMITS_NUM_L2, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 118), \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            INSTRUCTION_SIZE_IN_BYTES, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 119),                  \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_RUN_QUEUE_COUNT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 120),                        \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            ASYNC_COPY_MAPPING_INDEX_MAP, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 121),               \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_SUBCONTEXT_COUNT, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 122),                       \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            MAX_COMPUTE_CHANNELS, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 123),

#if NVCFG(GLOBAL_ARCH_HOPPER)
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS_NVCFG01                                                     \
    CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(DSMEM_STRIDE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 124),      \
        CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(                                                                              \
            PORTABLE_CLUSTER_SIZE, CU_TOOLS_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE + 125),
#else
#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS_NVCFG01
#endif

#define CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC                                                                              \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_01                                                                               \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_01                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_02                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_03                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_04                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_05                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_06                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_08                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_09                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC_NVCFG_10

#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE                                                                    \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_01                                                                     \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_NVCFG_01                                                               \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_02                                                                     \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_NVCFG_02                                                               \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_03

#define CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS                                                             \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS_01                                                              \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS_NVCFG01

// CU_TOOLS_DEVICE_ATTRIBUTE_TABLE
// This macros combines the attribute tables.
#define CU_TOOLS_DEVICE_ATTRIBUTE_TABLE                                                                                \
    CU_TOOLS_DEVICE_ATTRIBUTES_PUBLIC                                                                                  \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PUBLIC                                                                         \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE                                                                        \
    CU_TOOLS_DEVICE_ATTRIBUTES_INTERNAL_PRIVATE_LIMITS

// Generate CUtools_device_attribute enumeration.  This is done by
// defining CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE to generate the line
//      CU_TOOLS_DEVICE_ATTRIBUTE_<NAME> = <VALUE>
// The comma is provided by the CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE definition in
// the previous tables.
#ifdef CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE
#undef CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE
#endif
#define CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE(NAME, VALUE) CU_TOOLS_DEVICE_ATTRIBUTE_##NAME = (VALUE)

    /// Device Attributes
    typedef enum CUtools_device_attribute_enum
    {
        CU_TOOLS_DEVICE_ATTRIBUTE_TABLE CU_TOOLS_DEVICE_ATTRIBUTE_FORCE_INT = 0x7fffffff
    } CUtools_device_attribute;

#undef CU_TOOLS_DEVICE_DEFINE_ATTRIBUTE

    /// Device Attribute Properties.
    typedef enum CUtools_device_attribute_property_enum
    {
        CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_VALUE = 0,      /// Value of the attribute
        CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_VISIBILITY = 1, /// Visibility (public or private) or the attribute
        // --- always add new constants to the end here ---
        CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_SIZE,
        CU_TOOLS_DEVICE_ATTRIBUTE_PROPERTY_FORCE_INT = 0x7fffffff
    } CUtools_device_attribute_property;

    typedef enum CUtools_device_attribute_visibility_enum
    {
        CU_TOOLS_DEVICE_ATTRIBUTE_VISIBILITY_PUBLIC = 0,  /// Visible through the CUDA driver API
        CU_TOOLS_DEVICE_ATTRIBUTE_VISIBILITY_PRIVATE = 1, /// Only visible through the tools API
        // --- always add new constants to the end here ---
        CU_TOOLS_DEVICE_ATTRIBUTE_VISIBILITY_SIZE,
        CU_TOOLS_DEVICE_ATTRIBUTE_VISIBILITY_FORCE_INT = 0x7fffffff
    } CUtools_device_attribute_visibility;

    /// Type of the value returned in a CUtools_variant_type.
    typedef enum CUtools_variant_type_enum
    {
        CU_TOOLS_VARIANT_TYPE_INVALID = 0,
        CU_TOOLS_VARIANT_TYPE_S64 = 1,    /// signed 64-bit integer
        CU_TOOLS_VARIANT_TYPE_U64 = 2,    /// unsigned 64-bit integer
        CU_TOOLS_VARIANT_TYPE_STRING = 3, /// const char* with a lifetime equal to the driver
        CU_TOOLS_VARIANT_TYPE_F64 = 4,    /// 64-bit float
        // --- always add new constants to the end here ---
        CU_TOOLS_VARIANT_TYPE_SIZE,
        CU_TOOLS_VARIANT_TYPE_FORCE_INT = 0x7fffffff
    } CUtools_variant_type;

    /// Flexible structure for returning variable data types.
    /// \p type contains a CUtools_variant_type defining which member of the
    /// \p data union the value is contained in.
    /// \p data contains the value.
    // NOTE: pchar is only valid for the lifetime of the CUDA driver.  It is
    // recommended that the caller immediately copy strings returned by the
    // DeviceGetAttributeProperty function.
    typedef struct CUtoolsVariant_st
    {
        CUtools_variant_type type;
        uint32_t reserved;

        union variantData
        {
            int64_t s64;
            uint64_t u64;
            char const* pchar;
            double f64;
        } data;
    } CUtoolsVariant;

    /// Type of the FB RAM of a device
    typedef enum CUtools_fb_ram_type_enum
    {
        CU_TOOLS_FB_RAM_TYPE_UNKNOWN = 0x0,
        CU_TOOLS_FB_RAM_TYPE_SDRAM = 0x1,
        CU_TOOLS_FB_RAM_TYPE_DDR1 = 0x2,
        CU_TOOLS_FB_RAM_TYPE_DDR2 = 0x3,
        CU_TOOLS_FB_RAM_TYPE_GDDR2 = 0x4,
        CU_TOOLS_FB_RAM_TYPE_GDDR3 = 0x5,
        CU_TOOLS_FB_RAM_TYPE_GDDR4 = 0x6,
        CU_TOOLS_FB_RAM_TYPE_DDR3 = 0x7,
        CU_TOOLS_FB_RAM_TYPE_GDDR5 = 0x8,
        CU_TOOLS_FB_RAM_TYPE_LPDDR2 = 0x9,
        // --- ignore GDDR3 GBA values since they're not used in the API (0xA and 0xB)
        CU_TOOLS_FB_RAM_TYPE_SDDR4 = 0xC,
        CU_TOOLS_FB_RAM_TYPE_LPDDR4 = 0xD,
        CU_TOOLS_FB_RAM_TYPE_HBM1 = 0xE,
        CU_TOOLS_FB_RAM_TYPE_HBM2 = 0xF,
        CU_TOOLS_FB_RAM_TYPE_GDDR5X = 0x10,
        CU_TOOLS_FB_RAM_TYPE_GDDR6 = 0x11,
        // --- always add new constants to the end here ---
        CU_TOOLS_FB_RAM_TYPE_SIZE,
        CU_TOOLS_FB_RAM_TYPE_FORCE_INT = 0x7fffffff
    } CUtools_fb_ram_type;

    /// Location of the FB RAM of a device
    typedef enum CUtools_fb_ram_location_enum
    {
        CU_TOOLS_FB_RAM_LOCATION_UNKNOWN = 0,
        CU_TOOLS_FB_RAM_LOCATION_GPU_DEDICATED = 1,
        CU_TOOLS_FB_RAM_LOCATION_SYS_SHARED = 2,
        CU_TOOLS_FB_RAM_LOCATION_SYS_DEDICATED = 3,
        // --- always add new constants to the end here ---
        CU_TOOLS_FB_RAM_LOCATION_SIZE,
        CU_TOOLS_FB_RAM_LOCATION_FORCE_INT = 0x7fffffff
    } CUtools_fb_ram_location;

    /// Location of the pcie generation of a device
    typedef enum CUtools_pcie_gen_enum
    {
        CU_TOOLS_PCIE_GEN_GEN1 = 0,
        CU_TOOLS_PCIE_GEN_GEN2 = 1,
        CU_TOOLS_PCIE_GEN_GEN3 = 2,
        CU_TOOLS_PCIE_GEN_GEN4 = 3,
        CU_TOOLS_PCIE_GEN_GEN5 = 4,
#if NVCFG(GLOBAL_ARCH_BLACKWELL)
        CU_TOOLS_PCIE_GEN_GEN6 = 5,
#endif
        // --- always add new constants to the end here ---
        CU_TOOLS_PCIE_GEN_SIZE,
        CU_TOOLS_PCIE_GEN_FORCE_INT = 0x7fffffff
    } CUtools_pcie_gen;

    /// Bus type simplementation of a device
    typedef enum CUtools_bus_type_enum
    {
        CU_TOOLS_BUS_TYPE_UNKNOWN = 0,
        CU_TOOLS_BUS_TYPE_PCI = 1,
        CU_TOOLS_BUS_TYPE_AGP = 2,
        CU_TOOLS_BUS_TYPE_FPCI = 3,
        CU_TOOLS_BUS_TYPE_AXI = 4,
        CU_TOOLS_BUS_TYPE_PCI_EXPRESS = 8, // CUPTI has PCI_EXPRESS hard-coded as 8. See bug 1365250
        // --- always add new constants to the end here ---
        CU_TOOLS_BUS_TYPE_SIZE,
        CU_TOOLS_BUS_TYPE_SIZE_INT = 0x7fffffff
    } CUtools_bus_type;

    /// Virtualization mode of a device
    typedef enum CUtools_virtualization_mode_enum
    {
        CU_TOOLS_VIRTUALIZATION_MODE_NONE = 0,
        CU_TOOLS_VIRTUALIZATION_MODE_NMOS = 1,
        CU_TOOLS_VIRTUALIZATION_MODE_VGX = 2,
        CU_TOOLS_VIRTUALIZATION_MODE_HOST_VGPU = 3,
        CU_TOOLS_VIRTUALIZATION_MODE_HOST_VSGA = 4,
        // --- always add new constants to the end here ---
        CU_TOOLS_VIRTUALIZATION_MODE_SIZE,
        CU_TOOLS_VIRTUALIZATION_MODE_SIZE_INT = 0x7fffffff
    } CUtools_virtualization_mode;

#if NVCFG(GLOBAL_FEATURE_INCLUDE_CUDA_WDDM)
    /// Windows shared backing store support info.
    /// Mapped to nvlPrivate.h - NVL_SHARED_BACKINGSTORE_SUPPORT_TYPE enum.
    /// Shared backing store is an allocation feature added by Microsoft that allows both user mode and kernel mode
    /// to access memory associated with WDDM allocations.
    typedef enum CUtools_shared_backingstore_support_type_enum
    {
        CU_TOOLS_SHARED_BACKINGSTORE_SUPPORT_TYPE_NONE = 0,
        // supported only for sharing memory between user mode and kernel mode
        CU_TOOLS_SHARED_BACKINGSTORE_SUPPORT_TYPE_KERNEL_ONLY = 1,
        // supported for sharing memory between user mode, kernel mode, and hardware
        CU_TOOLS_SHARED_BACKINGSTORE_SUPPORT_TYPE_KERNEL_HW = 2,
        // --- always add new constants to the end here ---
        CU_TOOLS_SHARED_BACKINGSTORE_SUPPORT_TYPE_SIZE,
        CU_TOOLS_SHARED_BACKINGSTORE_SUPPORT_TYPE_SIZE_INT = 0x7fffffff
    } CUtools_shared_backingstore_support_type;
#endif

    /// Information about BAR0 registers of a device
    /// Used for CUetblToolsDevice::GetDeviceBar0Info
    typedef struct CUtoolsDeviceBar0Info_st
    {
        uint32_t struct_size;
        uint32_t reserved0;
        uint64_t size;   // Size of BAR0
        uint64_t offset; // Address of BAR0
    } CUtoolsDeviceBar0Info;

    /// Information about SMC partitioning of a device
    /// Used for CUetblToolsDevice::DeviceGetSmcInfo
    typedef struct CUtoolsDeviceSmcInfo_st
    {
        uint32_t struct_size;
        uint8_t inSmcMode; // True when subscription to a compute instance succeeded
        uint8_t reserved0[3];
        uint32_t gpuInstId;
#if defined(NV_MODS)
        uint32_t smcEngId;
#else
    uint32_t computeInstId;
#endif
        uint8_t migEnabled;   // True when GPU entered MIG mode
        uint8_t reserved1[7]; // --- end of v1, check struct_size for presence of newer fields ---
        CUuuid migUuid;
    } CUtoolsDeviceSmcInfo;

    /// \brief The device table provides functions to retrieve state from CUdevice
    CU_DEFINE_UUID(CU_ETID_ToolsDevice, 0xe14105b1, 0xc7f7, 0x4ac7, 0x9f, 0x64, 0xf2, 0x23, 0xbe, 0x99, 0xf1, 0xe2);

    typedef struct CUetblToolsDevice_st
    {
        // This export table supports versioning by adding to the end without changing
        // the ETID.  The struct_size field will always be set to the size in bytes of
        // the entire export table structure.
        size_t struct_size;

        /// \brief Returns a handle to a compute device
        /// Returns in \p *device a device handle given an ordinal in the range <b>[0,
        /// ::cuDeviceGetCount()-1]</b>.
        ///
        /// This is a wrapper for cuDeviceGet.
        ///
        /// \param device  - Returned device handle
        /// \param ordinal - Device number to get handle for
        /// \sa ::cuDeviceGet
        CUresult(CUDAAPI* DeviceGet)(CUdevice* device, uint32_t ordinal);

        /// \brief Returns the number of compute-capable devices
        ///
        /// /Returns in \p *count the number of devices with compute capability greater
        /// than or equal to 2.0 that are available for execution. If there is no such
        /// device, ::cuDeviceGetCount() returns 0.
        ///
        /// This is a wrapper for cuDeviceGetCount.
        ///
        /// \param count - Returns number of compute-capable devices
        /// \sa ::cuDeviceGetCount
        CUresult(CUDAAPI* DeviceGetCount)(uint32_t* count);

        /// \brief Returns the number of attributes for the device \p dev in \p *count.
        /// \param dev   - device handle
        /// \param count - Returns the number of attributes for the device.
        CUresult(CUDAAPI* DeviceGetAttributeCount)(CUdevice dev, size_t* count);

        /// \brief Returns a list of device attribute keys
        /// On input \p count contains the number of items allocated in \p *keys.
        /// \param dev   - device handle
        /// \param count - specifies the number of elements in \p *keys
        /// \param keys  - caller allocated memory in which \p *count elements
        ///                keys are returned.
        ///
        /// /Returns in \p *keys an array of \p count keys
        CUresult(CUDAAPI* DeviceGetAttributeKeys)(CUdevice dev, size_t count, CUtools_device_attribute* keys);

        /// \brief Returns a property of the attribute
        /// \param dev   - device handle
        /// \param key   - attribute identifier
        /// \param prop  - property identifier
        /// \param value - value of property
        CUresult(CUDAAPI* DeviceGetAttributeProperty)(
            CUdevice dev, CUtools_device_attribute key, CUtools_device_attribute_property prop, CUtoolsVariant* value);

        /// \brief Acquire a timestamp (nanoseconds since reset) from a specific CUDA device.
        CUresult(CUDAAPI* DeviceGetTimestamp)(CUdevice dev, uint64_t* timestamp);

        /// \brief Get the RM GPUID for a specified CUdevice.
        /// Currently this is only implemented on Windows platforms.
        CUresult(CUDAAPI* DeviceGetGpuId)(uint32_t* pGpuId, CUdevice dev);

        /// \brief Get the type of driver for a specified CUdevice.
        CUresult(CUDAAPI* DeviceGetDriverType)(CUtools_driver_type* pDriverType, CUdevice dev);

        /// Get the GR engine class for the specified device.
        CUresult(CUDAAPI* DeviceGetGrEngineClass)(CUtools_gr_engine_class* pEngineClass, CUdevice dev);

        /// Get information about BAR0 for the given device.
        /// May only be called if this device is using the RM driver model.
        CUresult(CUDAAPI* GetDeviceBar0Info)(CUtoolsDeviceBar0Info* bar0Info, CUdevice dev);

        /// \brief Get the uuid for a specified CUdevice.
        CUresult(CUDAAPI* DeviceGetUuid)(CUuuid* uuid, CUdevice dev);

        /// \brief Get the luid for a specified CUdevice.
        /// Returns CUDA_ERROR_NOT_SUPPORTED for platforms where there's no LUID.
        CUresult(CUDAAPI* DeviceGetLuid)(uint32_t* deviceNodeMask, uint64_t* luid, CUdevice dev);

        /// \brief Query if the specified CUdevice has TTU.
        CUresult(CUDAAPI* DeviceQueryTTU)(uint8_t* supportsTTU, CUdevice dev);

        /// \brief Query if the specified CUdevice supports F2FP instruction.
        CUresult(CUDAAPI* DeviceQueryF2FP)(uint8_t* supportsF2FP, CUdevice dev);

        /// \brief Get simulation type(fmodel, emulation..) for a specified CUdevice.
        CUresult(CUDAAPI* DeviceGetSimulationType)(CUtools_simulation_type* pSimulationType, CUdevice dev);

        /// \brief Get information about SMC for a specified CUdevice.
        /// Usage:
        /// CUtoolsDeviceSmcInfo smcInfo = {0};
        /// smcInfo.struct_size = sizeof(smcInfo);
        /// CUresult smcQueryResult = GetSmcInfo(&smcInfo);
        CUresult(CUDAAPI* DeviceGetSmcInfo)(CUtoolsDeviceSmcInfo* smcInfo, CUdevice dev);

        /// \brief Query the default memory pool associated with this device
        CUresult(CUDAAPI* DeviceGetDefaultMemoryPool)(CUmemoryPool* memoryPool, CUdevice dev);

        /// \brief Set the memory pool associated with this device
        CUresult(CUDAAPI* DeviceSetMemoryPool)(CUdevice dev, CUmemoryPool memoryPool);

        /// \brief Query the currently set memory pool associated with this device
        /// Creates the default pool, if it was not initialized yet.
        CUresult(CUDAAPI* DeviceGetMemoryPool)(CUmemoryPool* memoryPool, CUdevice dev);

        /// \brief Query if the specified CUdevice supports debugging.
        CUresult(CUDAAPI* DeviceQueryDebuggingCapability)(uint8_t* supportsDebugging, CUdevice dev);

        /// \brief Returns the number of invisible compute-capable devices
        ///
        /// /Returns in \p *count the number of invisible devices with compute capability
        /// greater than or equal to 2.0 that are available for execution. If there is no such
        /// device, it  returns 0.    ///
        ///
        /// \param count - Returns number of invisible compute-capable devices
        CUresult(CUDAAPI* InvisibleDeviceGetCount)(uint32_t* count);

        /// \brief Returns a handle to an invisible compute device
        /// Returns in \p *device a device handle given an ordinal in the range of invisible devices
        ///
        /// \param device  - Returned device handle
        /// \param ordinal - Device number to get handle for
        CUresult(CUDAAPI* InvisibleDeviceGet)(CUdevice* device, uint32_t ordinal);

        /// \brief Returns a property of the attribute for an invisible device
        /// \param dev   - device handle
        /// \param key   - attribute identifier
        /// \param prop  - property identifier
        /// \param value - value of property
        CUresult(CUDAAPI* InvisibleDeviceGetAttributeProperty)(
            CUdevice dev, CUtools_device_attribute key, CUtools_device_attribute_property prop, CUtoolsVariant* value);

        /// \brief Get the uuid for a specified invisible CUdevice.
        CUresult(CUDAAPI* InvisibleDeviceGetUuid)(CUuuid* uuid, CUdevice dev);

        /// \brief Query the currently set memory pool associated with this device
        /// Returns NULL, if the default pool was not yet created.
        CUresult(CUDAAPI* DeviceGetDefaultMemoryPoolNoInit)(CUmemoryPool* memoryPool, CUdevice dev);

#if NVCFG(GLOBAL_FEATURE_INCLUDE_CUDA_WDDM)
        /// \brief Get the WDDM version for a specified CUdevice.
        /// Returns an error CUDA_ERROR_NOT_SUPPORTED if the device is not running on WDDM.
        CUresult(CUDAAPI* DeviceGetWddmVersion)(uint32_t* pWddmVersion, CUdevice dev);

        /// \brief Returns shared backing store support type CUtools_shared_backingstore_support_type.
        /// Returns an error CUDA_ERROR_NOT_SUPPORTED if the device is not running on WDDM.
        CUresult(CUDAAPI* GetSharedBackingStoreSupportType)(
            CUtools_shared_backingstore_support_type* supportedType, CUdevice dev);

        /// \brief Returns whether the device is using WDDM in (compute only) MCDM mode.
        CUresult(CUDAAPI* DeviceHasMcdmEnabled)(uint8_t* enabled, CUdevice dev);
#else
    void* legacy_0;
    void* legacy_1;
    void* legacy_2;
#endif

        /// \brief Get the uuid for a specified CUdevice. In case of MIG device, it returns the CI UUID of the MIG
        /// device.
        CUresult(CUDAAPI* DeviceGetUuid_V2)(CUuuid* uuid, CUdevice dev);

#if NVCFG(GLOBAL_FEATURE_CTK7320_MPS_UGPU_STEERING)
        /// \brief Returns whether the specified CUdevice uses MPS-based uGPU localization
        CUresult(CUDAAPI* DeviceUsesUGpuLocalization)(uint8_t* isLocalized, CUdevice dev);
#else
    void* reserved_0;
#endif

        /// \brief Returns the HW SM version for a specified CUdevice.
        /// See https://p4hw-swarm.nvidia.com/view/hw/doc/gpu/rubin/rubin/design/IAS/SM/ISA/opcodes.htm
        CUresult(CUDAAPI* GetHwSmVersion)(CUdevice dev, uint32_t* pVersion);
    } CUetblToolsDevice;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
