/*
 * Copyright 2022 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_green_context_h__
#define __cuda_etbl_green_context_h__

#include "cuda.h"
#include "cuda_uuid.h"

CU_DEFINE_UUID(CU_ETID_GreenContext, 0x60883e26, 0xd27c, 0x4361, 0x92, 0xf6, 0xbb, 0xd5, 0x0, 0x6d, 0xfa, 0x7e);

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

#define CU_RESOURCE_AFFINITY_TYPE_SM 0
#define CU_ETBL_GC_MAX_VGPCS 30

    typedef struct CUresourceAffinityParam_st
    {
        unsigned int type;

        struct
        {
            struct
            {
                unsigned int minCount; /**< The number of SMs the context is limited to use. */
            } sm;
        } resource;
    } CUresourceAffinityParam;

    typedef enum CUcontextType_enum
    {
        CUI_CTX_TYPE_LEGACY = 0,
        CUI_CTX_TYPE_GREEN = 1
    } CUcontextType;

    typedef struct CUetblMockDeviceDescriptor_st
    {
        int struct_size;
        //
        uint32_t numSmsPerTpc;
        uint32_t minSmGranularity;
        uint32_t minSmCount;
        uint32_t gpcCount;
        uint32_t tpcCount;
        uint32_t smCount;
        uint16_t tpcMask[CU_ETBL_GC_MAX_VGPCS];
        uint32_t physicalTpcMask[CU_ETBL_GC_MAX_VGPCS];
        uint32_t physicalTpcCount;
        uint32_t physicalTpcsPerGpc;
        uint32_t physicalGpcCount;
        uint32_t physicalGpcMask;
    } CUetblMockDeviceDescriptor;

    typedef struct CUetblGreenContext_st
    {
        // This export table supports versioning by adding to the end without
        // changing the ETID.  The struct_size field will always be set to the
        // size in bytes of the entire export table structure.
        size_t struct_size;

        /*
         * DO NOT USE!
         * This is here for compatibility with old tests.
         * - numParams is always assumed to be 1
         */
        CUresult(CUDAAPI* cuGreenCtxCreate)(
            CUcontext* pCtx, CUresourceAffinityParam* param, int numParams, int flags, CUdevice dev);

        /**
         * \brief Returns a context's type
         *
         * Returns the context type of \p ctx in \p type. Additionally, if the specified
         * context \p ctx is green, its primary context is returned in \p primaryCtx.
         * If the specified \p ctx is NULL, the type of the current context is returned.
         *
         * \param ctx - Context to query
         * \param type - Pointer to return the context type
         * \param primaryCtx - Pointer to return the primary context if \p ctx is green
         *
         * \return
         * ::CUDA_SUCCESS,
         * ::CUDA_ERROR_DEINITIALIZED,
         * ::CUDA_ERROR_NOT_INITIALIZED,
         * ::CUDA_ERROR_INVALID_CONTEXT,
         * ::CUDA_ERROR_CONTEXT_IS_DESTROYED,
         * ::CUDA_ERROR_INVALID_VALUE
         * \notefnerr
         *
         * \sa
         * ::cuGreenCtxCreate,
         * ::cuCtxCreate
         */
        CUresult(CUDAAPI* cuCtxGetType)(CUcontext ctx, CUcontextType* type, CUcontext* primaryCtx);

        // Return whether the primary context is a green context (for testing purposes!)
        int(CUDAAPI* primaryCtxIsGreenCtxSupportEnabled)(void);

        // cuEtblSetMockDeviceDescriptor
        CUresult(CUDAAPI* cuEtblExchangeMockDeviceDesc)(
            CUdevice device, CUetblMockDeviceDescriptor* old, CUetblMockDeviceDescriptor* desired, int struct_size);

        /// \brief Splits SM resources by uGPU
        /// \param result - Output array of \p CUdevResource resources. Can be NULL to query the number of groups.
        /// \param nbGroups - This is a pointer, specifying the number of groups that would be or should be created as
        /// described below. \param input - Input SM resource to be split. Must be a valid \p CU_DEV_RESOURCE_TYPE_SM
        /// resource. \param remaining - If the input resource cannot be cleanly split among \p nbGroups, the remaining
        /// is placed in here. Can be omitted (NULL) if the user does not need the remaining set.
        CUresult(CUDAAPI* cuDevSmResourceSplitByUgpu)(
            CUdevResource* result, unsigned int* nbGroups, CUdevResource const* input, CUdevResource* remaining);

        CUresult(CUDAAPI* cuDeviceGetDesc)(CUdevice device, CUdevResourceDesc* phDesc);

        CUresult(CUDAAPI* cuCtxGetDesc)(CUcontext userCtx, CUdevResourceDesc* phDesc);

        CUresult(CUDAAPI* cuGreenCtxGetDesc)(CUgreenCtx hGreenCtx, CUdevResourceDesc* phDesc);

        CUresult(CUDAAPI* cuEtblStreamGetDesc)(CUstream hStream, CUdevResourceDesc* hDesc);

        CUresult(CUDAAPI* cuEtblStreamSetDesc)(CUstream hStream, CUdevResourceDesc hDesc);

        CUresult(CUDAAPI* cuDevResourceDescGet)(
            CUdevResourceDesc phDesc, CUdevResource* resource, CUdevResourceType type);

        CUresult(CUDAAPI* cuEtblDevResourceDescOccupancyMaxPotentialBlockSize)(CUdevResourceDesc phDesc,
            CUfunction function, CUlaunchConfig* launchConfig, size_t (*blockSizeToDynamicSMemSize)(int),
            int maxBlockSize);

        CUresult(CUDAAPI* cuDevResourceDescOccupancyGetMaxPotentialClusterSize)(
            CUdevResourceDesc phDesc, CUfunction func, CUlaunchConfig const* config, int* maxClusterSize);

        CUresult(CUDAAPI* cuDevResourceDescOccupancyGetMaxActiveClusters)(
            CUdevResourceDesc phDesc, CUfunction func, CUlaunchConfig const* config, int* numClusters);

    } CUetblGreenContext;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
