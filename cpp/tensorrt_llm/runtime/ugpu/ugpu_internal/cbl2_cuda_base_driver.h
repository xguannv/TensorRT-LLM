/*
 * Copyright 2022-2023 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cbl2_cuda_base_driver_h__
#define __cbl2_cuda_base_driver_h__

#include "cbl2.h"
#include "cuda.h"

#include "cuda_uuid.h"
#include "g_nvconfig.h"
#include "nvtypes.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    CU_DEFINE_UUID(
        CU_ETID_CBL2CudaBaseDriver, 0xec3bb529, 0xda14, 0x456c, 0xa2, 0xa6, 0x6c, 0xf7, 0xfe, 0xe5, 0x46, 0x7d);

#define CBL_CUDA_TPCINFO_ARRAY_SIZE 256

    typedef void* cblCudaContextDestroyCallback;

    typedef enum cblCudaMembarType_t
    {
        CBL_CUDA_MEMBAR_NONE = 0x0,
        CBL_CUDA_MEMBAR_GL = 0x1,
        CBL_CUDA_MEMBAR_SYS = 0x2,
        // --- always add new constants to the end here ---
        CBL_CUDA_MEMBAR_SIZE,
        CBL_CUDA_MEMBAR_FORCE_INT = 0x7fffffff
    } cblCudaMembarType;

    typedef enum cblCudaMemoryLocation_t
    {
        CBL_CUDA_LOCATION_DEVICE = 0x0,
        CBL_CUDA_LOCATION_HOST = 0x1,
        // --- always add new constants to the end here ---
        CBL_CUDA_LOCATION_SIZE,
        CBL_CUDA_LOCATION_FORCE_INT = 0x7fffffff
    } cblCudaMemoryLocation;

    typedef enum cblCudaMemoryUsageFlags_t
    {
        CBL_CUDA_MEMORY_USAGE_FLAGS_HOST_ENGINE_ACCESSIBLE
        = 0x1, // Allocation needs to be accessible by the HOST engine, which limits what virtual addresses are allowed
        CBL_CUDA_MEMORY_USAGE_FLAGS_READ_ONLY
        = 0x2, // Allocation should be marked read only, faulting if any writes occur
        CBL_CUDA_MEMORY_USAGE_FLAGS_CACHE_DEVICE = 0x4,  // Allocation should be cacheable on the device
        CBL_CUDA_MEMORY_USAGE_FLAGS_CACHE_HOST = 0x8,    // Allocation should be cacheable on the host
        CBL_CUDA_MEMORY_USAGE_FLAGS_MAP_HOST = 0x10,     // Allocation should be mapped to the HOST
        CBL_CUDA_MEMORY_USAGE_FLAGS_SHARABLE = 0x20,     // Allocation can be shared to other entities
        CBL_CUDA_MEMORY_USAGE_FLAGS_PUSHBUFFER = 0x1000, // Allocation is meant to be used as a pushbuffer
        // --- always add new constants to the end here ---
        CBL_CUDA_MEMORY_USAGE_FLAGS_SIZE,
        CBL_CUDA_MEMORY_USAGE_FLAGS_MASK = ((CBL_CUDA_MEMORY_USAGE_FLAGS_SIZE - 1ULL) << 1ULL) - 1ULL
    } cblCudaMemoryUsageFlags;

    typedef enum cblCudaSemaphoreAcquireOp_t
    {
        CBL_CUDA_SEMA_ACQUIRE_EQ,
        CBL_CUDA_SEMA_ACQUIRE_GEQ,
    } cblCudaSemaphoreAcquireOp;

    typedef struct cblCudaCreateBaseDriverParams_st
    {
        /// @name Inputs
        /// @{
        // The CUDA Context attached that base driver. Only the objects attached
        // to that context (Streams, Events ...) can be used with the created
        // hBaseDriver backend
        CUcontext hContext;
        cblCudaContextDestroyCallback cudaContextDestroyCallback;
        /// @}

        /// @name Outputs
        /// @{
        // Pass this parameter to cblCreateContextParams.pBaseDriverCallback
        cblBaseDriverCallbacks baseDriverCallbacks;
        // Pass this parameter to cblCreateContextParams.hBaseDriver
        cblBaseDriverHandle hBaseDriver;
        /// @}
    } cblCudaCreateBaseDriverParams;

    typedef struct cblCudaEncodeMemcpyInlineParams_st
    {
        cblBaseDriverHandle hBaseDriver;

        NvU32 size;              // Size of the copy (max 2^16)
        void* srcHostPointer;    // pointer to pageable host memory that contains the contents to copy
        NvU64 dstDevicePointer;  // pointer to device memory to store the contents
        void* pushbufferPointer; // IN/OUT Pushbuffer pointer, the address will be updated to reflect theused by the
                                 // newly encoded launch
                                 // @note The size of this buffer should be roughly 10% more than size of the request in
                                 // order to account for the message headers
        NvU32 pushbufferMaxSize; // IN/OUT Maximum size of pushbuffer given. Updated to requirement if not enough
    } cblCudaEncodeMemcpyInlineParams;

    typedef struct cblCudaEncodeLaunchParams_st
    {
        cblBaseDriverHandle hBaseDriver;

        cblQmdPcasAction pcasAction; // Type of scheduling action to perform on the given QMD
        NvU64 qmdDevicePointer;      // pointer to device memory containing the uploaded QMD
        void* pushbufferPointer;     // IN/OUT Pushbuffer pointer, the address will be updated to reflect theused by the
                                     // newly encoded launch
                                     // @note The size of this buffer should be at least 16B in size
        NvU32 pushbufferMaxSize;     // IN/OUT Maximum size of pushbuffer given. Updated to requirement if not enough
    } cblCudaEncodeLaunchParams;

    typedef struct cblCudaEncodeMembarParams_st
    {
        cblBaseDriverHandle hBaseDriver;

        cblCudaMembarType membarType;

        void* pushbufferPointer;
        NvU32 pushbufferMaxSize; // IN/OUT Maximum size of pushbuffer given. Updated to requirement if not enough
    } cblCudaEncodeMembarParams;

    typedef void* cblSubmission;

    typedef struct cblCudaStreamBeginPushParams_st
    {
        /// @name Inputs
        /// @{
        cblBaseDriverHandle hBaseDriver; // The CBL context associated with the CUDA backing
        CUstream hStream;                // The stream targeting the push
        /// @}

        /// @name Outputs
        /// @{
        cblSubmission hSubmission;             // The submission handle to be specified in cblCudaStreamEndPush
        NvU64 completionSemaphoreCurrentValue; // The current value of the completion semaphore.  The pushbuffer
                                               // specified in cblCudaStreamEndPush must release a value larger than
                                               // this to completionSemaphoreAddress
        NvU64 completionSemaphoreAddress;      // The address of the completion semaphore to release.
        /// @}
    } cblCudaStreamBeginPushParams;

    typedef struct cblCudaStreamEndPushParams_st
    {
        cblSubmission hSubmission;              // The submission handle for that submission
        NvU64 pushbufferDeviceAddress;          // The device address of the pushbuffer submitted to the GPU
        NvU32 pushbufferSize;                   // Size of the pushbuffer to submit
        NvU64 completionSemaphoreExpectedValue; // The value released by the submission in the completion semaphore. All
                                                // work must complete before that value is released
        NvBool hostPrefetch; // If set, allows HW to prefetch the pushbuffer immediately for execution.  If there are
                             // on-going memory operations on the pushbuffer at the time of the call, this should be set
                             // to false.
    } cblCudaStreamEndPushParams;

    typedef struct cblCudaAllocatememoryParams_st
    {
        /// @name Inputs
        /// @{
        cblBaseDriverHandle hBaseDriver;
        cblCudaMemoryLocation location;
        cblCudaMemoryUsageFlags usage;
        void* win32MetaData;
        NvU64 size;
        /// @}

        /// @name Outputs
        /// @{
        NvU64 deviceAddress;
        void* hostAddress;
        NvU64 alignment; // Guaranteed alignment for this allocation
        /// @}
    } cblCudaAllocateMemoryParams;

    typedef struct cblCudaReconfigureLocalMemoryParams_st
    {
        /// @name Inputs
        /// @{
        cblBaseDriverHandle hBaseDriver;
        NvU64
            localMemoryTotalSize; // Total local memory size for the entire device @sa cblCudaGetTotalLocalMemoryParams
        /// @}
    } cblCudaReconfigureLocalMemoryParams;

    typedef struct cblCudaPatchQmdSemaphoreReleaseParams_st
    {
        /// @name Inputs
        /// @{
        cblBaseDriverHandle hBaseDriver; // Base driver matters since QMD layout varies between devices
        void* hostQmd;                   // Host QMD buffer to patch
        NvU64 semaphoreAddress;
        NvU64 semaphorePayload;          // The semaphore value to release to
        /// @}
    } cblCudaPatchQmdSemaphoreReleaseParams;

    typedef struct cblCudaEncodeSemaphoreAcquireParams_st
    {
        cblBaseDriverHandle hBaseDriver;
        void* pushbufferPointer;
        NvU32 pushbufferMaxSize; // IN/OUT Maximum size of pushbuffer given. Updated to requirement if not enough
        NvU64 semaphoreAddress;
        NvU64 semaphorePayload;
        cblCudaSemaphoreAcquireOp op;
    } cblCudaEncodeSemaphoreAcquireParams;

    typedef struct cblCudaGetTpcInfoParams_st
    {
        cblBaseDriverHandle hBaseDriver;
        NvU32 numSmsPerTpc;
        NvU32 numTpcsPerGpc;
        NvU32 numGpcs;
        NvU32 numTpcInfo;

        struct
        {
            NvU16 gpcId;
            NvU16 virtualGpcId;
            NvU16 migratableTpcId;
            NvU16 globalTpcId;
        } tpcInfoArray[CBL_CUDA_TPCINFO_ARRAY_SIZE];
    } cblCudaGetTpcInfoParams;

    typedef struct CUetblCBL2CudaBaseDriver_st
    {
        // This export table supports versioning by adding to the end without
        // changing the ETID.  The struct_size field will always be set to the
        // size in bytes of the entire export table structure.
        size_t struct_size;
        CUresult(CUDAAPI* cblCudaCreateBaseDriver)(cblCudaCreateBaseDriverParams* pParams);
        CUresult(CUDAAPI* cblCudaDestroyBaseDriver)(cblBaseDriverHandle hBaseDriver);
        CUresult(CUDAAPI* cblEncodeMemcpyInline)(cblCudaEncodeMemcpyInlineParams* pParams);
        CUresult(CUDAAPI* cblEncodeLaunch)(cblCudaEncodeLaunchParams* pParams);
        CUresult(CUDAAPI* cblEncodeMembar)(cblCudaEncodeMembarParams* pParams);
        // @note This call will prevent any CUDA API calls (including other CBL calls unless otherwise documented)
        // targeting the same CUDA context from making forward progress until cblCudaStreamEndPush is called
        CUresult(CUDAAPI* cblCudaStreamBeginPush)(cblCudaStreamBeginPushParams* pParams);
        // @note This call must be called after cblCudaStreamBeginPush and before any CUDA API call targeting the same
        // cuda context (including other CBL calls unless otherwise documented), otherwise the caller may recursively
        // deadlock.
        CUresult(CUDAAPI* cblCudaStreamEndPush)(cblCudaStreamEndPushParams* pParams);
        CUresult(CUDAAPI* cblCudaAllocateMemory)(cblCudaAllocateMemoryParams* pParams);
        CUresult(CUDAAPI* cblCudaFreeMemory)(cblBaseDriverHandle hBaseDriver, NvU64 deviceAddress);
        CUresult(CUDAAPI* cblCudaReconfigureLocalMemory)(cblCudaReconfigureLocalMemoryParams* pParams);
        // @note This call assumes QMD already has semaphore 0 enable flag set
        CUresult(CUDAAPI* cblCudaPatchQmdSemaphoreRelease)(cblCudaPatchQmdSemaphoreReleaseParams* pParams);
        CUresult(CUDAAPI* cblEncodeSemaphoreAcquire)(cblCudaEncodeSemaphoreAcquireParams* pParams);
        CUresult(CUDAAPI* cblCudaGetTpcInfo)(cblCudaGetTpcInfoParams* pParams);
    } CUetblCBL2CudaBaseDriver;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __cbl2_cuda_base_driver_h__
