/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_common_types_h__
#define __cuda_etbl_tools_common_types_h__

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

#include "g_nvconfig.h"

#include "cuda_stdint.h"

    //  Types used by callbacks and export tables.  This file is for defining
    //  types that are used in multiple Tools API headers.

    typedef struct CUtoolsStreamHandle_st* CUtoolsStreamHandle;
    typedef struct CUtoolsTaskHandle_st* CUtoolsTaskHandle;
    typedef struct CUtoolsMemObjHandle_st* CUtoolsMemObjHandle;
    typedef struct CUtoolsMemBlockHandle_st* CUtoolsMemBlockHandle;
    typedef struct CUtoolsChannelHandle_st* CUtoolsChannelHandle;
    typedef struct CUtoolsMem2memBufferHandle_st* CUtoolsMem2memBufferHandle;
    typedef struct CUtoolsLaunchHandle_st* CUtoolsLaunchHandle;
    typedef struct CUtoolsMemmgrHandle_st* CUtoolsMemmgrHandle;
    typedef struct CUtoolsCmdListHandle_st* CUtoolsCmdListHandle;

    typedef struct CUtoolsQmd_st* CUtoolsQmd;
    typedef struct CUtoolsNvCurrent_st* CUtoolsNvCurrent;

    typedef void* CUtoolsContextMarkerHandle;

    typedef enum CUtools_channel_use_type_enum
    {
        CU_TOOLS_CHANNEL_USE_COMPUTE = 0,
        CU_TOOLS_CHANNEL_USE_ASYNC_MEMORY_H_TO_D = 1,
        CU_TOOLS_CHANNEL_USE_ASYNC_MEMORY_D_TO_H = 2,
        CU_TOOLS_CHANNEL_USE_SEC2 = 3,
#if NVCFG(GLOBAL_FEATURE_CONFIDENTIAL_COMPUTE)
        CU_TOOLS_CHANNEL_USE_SECURE_WORK_LAUNCH = 4,
        CU_TOOLS_CHANNEL_USE_SECURE_WORK_COMPLETION = 5,
#endif
#if NVCFG(GLOBAL_FEATURE_CTK5022_CE_DECOMPRESS)
        CU_TOOLS_CHANNEL_USE_DECOMP = 6,
#endif
#if NVCFG(GLOBAL_FEATURE_CTK7136_CUDA_DX12_CIG_STREAMS)
        CU_TOOLS_CHANNEL_USE_VIRTUAL_CIG_COMPUTE = 7,
#endif
        // --- always add new constants to the end here ---
        CU_TOOLS_CHANNEL_USE_SIZE,
        CU_TOOLS_CHANNEL_USE_INT = 0x7fffffff
    } CUtools_channel_use_type;

    typedef enum CUtools_channel_type_enum
    {
        CU_TOOLS_CHANNEL_COMPUTE = 0,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_0 = 1,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_1 = 2,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_2 = 3,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_3 = 4,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_4 = 5,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_5 = 6,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_6 = 7,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_7 = 8,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_8 = 9,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_9 = 10,
        CU_TOOLS_CHANNEL_TYPE_SEC2 = 11,
#if NVCFG(GLOBAL_ARCH_BLACKWELL)
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_10 = 12,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_11 = 13,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_12 = 14,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_13 = 15,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_14 = 16,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_15 = 17,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_16 = 18,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_17 = 19,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_18 = 20,
        CU_TOOLS_CHANNEL_ASYNC_MEMCPY_19 = 21,
#endif
#if NVCFG(GLOBAL_FEATURE_CTK5022_CE_DECOMPRESS)
        CU_TOOLS_CHANNEL_DECOMP_0 = 22,
        CU_TOOLS_CHANNEL_DECOMP_1 = 23,
        CU_TOOLS_CHANNEL_DECOMP_2 = 24,
        CU_TOOLS_CHANNEL_DECOMP_3 = 25,
        CU_TOOLS_CHANNEL_DECOMP_4 = 26,
        CU_TOOLS_CHANNEL_DECOMP_5 = 27,
        CU_TOOLS_CHANNEL_DECOMP_6 = 28,
        CU_TOOLS_CHANNEL_DECOMP_7 = 29,
        CU_TOOLS_CHANNEL_DECOMP_8 = 30,
        CU_TOOLS_CHANNEL_DECOMP_9 = 31,
        CU_TOOLS_CHANNEL_DECOMP_10 = 32,
        CU_TOOLS_CHANNEL_DECOMP_11 = 33,
        CU_TOOLS_CHANNEL_DECOMP_12 = 34,
        CU_TOOLS_CHANNEL_DECOMP_13 = 35,
        CU_TOOLS_CHANNEL_DECOMP_14 = 36,
        CU_TOOLS_CHANNEL_DECOMP_15 = 37,
        CU_TOOLS_CHANNEL_DECOMP_16 = 38,
        CU_TOOLS_CHANNEL_DECOMP_17 = 39,
        CU_TOOLS_CHANNEL_DECOMP_18 = 40,
        CU_TOOLS_CHANNEL_DECOMP_19 = 41,
#endif
#if NVCFG(GLOBAL_FEATURE_CTK7136_CUDA_DX12_CIG_STREAMS)
        CU_TOOLS_CHANNEL_VIRTUAL_CIG_COMPUTE = 42,
#endif
        // --- always add new constants to the end here ---
        CU_TOOLS_CHANNEL_SIZE,
        CU_TOOLS_CHANNEL_INT = 0x7fffffff
    } CUtools_channel_type;

    typedef enum CUtools_engine_type_enum
    {
        CU_TOOLS_ENGINE_COMPUTE = 0,
        CU_TOOLS_ENGINE_TWOD = 1,
        CU_TOOLS_ENGINE_MEM2MEM = 2,
        CU_TOOLS_ENGINE_ASYNC_MEMCPY = 3,
        CU_TOOLS_ENGINE_SEC2 = 4,
        CU_TOOLS_ENGINE_SW = 5,
#if NVCFG(GLOBAL_FEATURE_CTK5022_CE_DECOMPRESS)
        CU_TOOLS_ENGINE_DECOMP = 6,
#endif
        // --- always add new constants to the end here ---
        CU_TOOLS_ENGINE_SIZE,
        CU_TOOLS_ENGINE_FORCE_INT = 0x7fffffff
    } CUtools_engine_type;

    /// Type of driver active on a device.
    typedef enum CUtools_driver_type_enum
    {
        CU_TOOLS_DRIVER_TYPE_INVALID = 0,
        CU_TOOLS_DRIVER_TYPE_RM = 1,
#if NVCFG(GLOBAL_FEATURE_INCLUDE_CUDA_WDDM)
        CU_TOOLS_DRIVER_TYPE_WDDM = 2,
#else
    CU_TOOLS_DRIVER_TYPE_RESERVED = 2,
#endif
        CU_TOOLS_DRIVER_TYPE_GLK = 3,
        CU_TOOLS_DRIVER_TYPE_AMODEL = 4,
        CU_TOOLS_DRIVER_TYPE_MPS = 5,
        CU_TOOLS_DRIVER_TYPE_MRM = 6,
        CU_TOOLS_DRIVER_TYPE_REMOVED_1 = 7,
        // --- always add new constants to the end here ---
        CU_TOOLS_DRIVER_TYPE_SIZE,
        CU_TOOLS_DRIVER_TYPE_FORCE_INT = 0x7fffffff
    } CUtools_driver_type;

    typedef enum CUtools_gr_engine_type_enum
    {
        CU_TOOLS_GR_ENGINE_TYPE_INVALID = 0,
        CU_TOOLS_GR_ENGINE_TYPE_GRAPHICS = 1,
        CU_TOOLS_GR_ENGINE_TYPE_COMPUTE = 2,
        CU_TOOLS_GR_ENGINE_TYPE_SIZE,
        CU_TOOLS_GR_ENGINE_TYPE_forceint = 0x7fffffff
    } CUtools_gr_engine_type;

    /// HW Engine class.
    typedef enum CUtools_gr_engine_class_enum
    {
        CU_TOOLS_GR_ENGINE_CLASS_INVALID = 0x0000,
        CU_TOOLS_GR_ENGINE_CLASS_FERMI_A = 0x0003,
        CU_TOOLS_GR_ENGINE_CLASS_FERMI_B = 0x0004,
        CU_TOOLS_GR_ENGINE_CLASS_FERMI_C = 0x0005,
        CU_TOOLS_GR_ENGINE_CLASS_FERMI_COMPUTE_A = 0x0006,
        CU_TOOLS_GR_ENGINE_CLASS_FERMI_COMPUTE_B = 0x0007,
        CU_TOOLS_GR_ENGINE_CLASS_KEPLER_A = 0x0008,
        CU_TOOLS_GR_ENGINE_CLASS_KEPLER_B = 0x0009,
        CU_TOOLS_GR_ENGINE_CLASS_KEPLER_COMPUTE_A = 0x000A,
        CU_TOOLS_GR_ENGINE_CLASS_KEPLER_COMPUTE_B = 0x000B,
        CU_TOOLS_GR_ENGINE_CLASS_KEPLER_C = 0x0011,
        CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_A = 0x0012,
        CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_COMPUTE_A = 0x0013,
        CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_COMPUTE_B = 0x0014,
        CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_B = 0x0015,
        CU_TOOLS_GR_ENGINE_CLASS_PASCAL_A = 0x0016,
        CU_TOOLS_GR_ENGINE_CLASS_PASCAL_COMPUTE_A = 0x0017,
        CU_TOOLS_GR_ENGINE_CLASS_PASCAL_B = 0x0018,
        CU_TOOLS_GR_ENGINE_CLASS_PASCAL_COMPUTE_B = 0x0019,
        // --- always add new constants to the end here ---
        CU_TOOLS_GR_ENGINE_CLASS_SIZE,
        CU_TOOLS_GR_ENGINE_CLASS_forceint = 0x7fffffff
    } CUtools_gr_engine_class;

    typedef enum CUtools_debug_event_notify_index_enum
    {
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_INVALID = 0x0000,
        /// NV9097_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_A = 0x0003,
        /// NV9197_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_B = 0x0004,
        /// NV9297_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_C = 0x0005,
        /// NV90C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_COMPUTE_A = 0x0006,
        /// NV91C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_COMPUTE_B = 0x0007,
        /// NVA097_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_A = 0x0008,
        /// NVA197_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_B = 0x0009,
        /// NVA0C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_COMPUTE_A = 0x000A,
        /// NVA1C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_COMPUTE_B = 0x000B,
        // NVA297_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_C = 0x0011,
        // NVB097_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_MAXWELL_A = 0x0012,
        // NVB0C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_MAXWELL_COMPUTE_A = 0x0013,
        // NVB1C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_MAXWELL_COMPUTE_B = 0x0014,
        // NVC097_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_PASCAL_A = 0x0015,
        // NVC0C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_PASCAL_COMPUTE_A = 0x0016,
        // NVC197_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_PASCAL_B = 0x0017,
        // NVC1C0_NOTIFIERS_DEBUG_INTR
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_PASCAL_COMPUTE_B = 0x0018,
        // --- always add new constants to the end here ---
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_SIZE,
        CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_forceint = 0x7fffffff
    } CUtools_debug_event_notify_index;

    typedef enum CUtoolsModuleOwner_enum
    {
        CU_TOOLS_MODULE_OWNER_INVALID = 0x000,
        CU_TOOLS_MODULE_OWNER_DRIVER = 0x001,
        CU_TOOLS_MODULE_OWNER_USER = 0x002,
        // --- always add new constants to the end here ---
        CU_TOOLS_MODULE_OWNER_SIZE,
        CU_TOOLS_MODULE_OWNER_FORCE_INT = 0x7fffffff
    } CUtoolsModuleOwner;

    typedef enum CUtoolsModuleVisibility_enum
    {
        CU_TOOLS_MODULE_VISIBILITY_VISIBLE = 0x000,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_SYSCALL = 0x001,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_TRAPHANDLER = 0x002,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_TRAMPOLINE = 0x003,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_ATENTRY = 0x004,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_TOOLS = 0x005,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_KEPLER_MEMBAR_WAR = 0x006,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_EXITFUNCTION = 0x007,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_USER = 0x008,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_MEMBAR_OPTI_WAR = 0x009,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_NANOSLEEP_WAR = 0x00A,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_LDC_WAR = 0x00B,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_BARSYNC_WAR = 0x010,
        // --- always add new constants to the end here ---
        CU_TOOLS_MODULE_VISIBILITY_SIZE,
        CU_TOOLS_MODULE_VISIBILITY_HIDDEN_UNKNOWN = 0x7fffffff,
        CU_TOOLS_MODULE_VISIBILITY_FORCE_INT = 0x7fffffff
    } CUtoolsModuleVisibility;

    typedef enum CUtools_cnp_support_enum
    {
        CU_TOOLS_CNP_NOT_SUPPORTED = 0x00,
        CU_TOOLS_CNP_SUPPORTED = 0x01,
        // --- always add new constants to the end here ---
        CU_TOOLS_CNP_SIZE,
        CU_TOOLS_CNP_FORCE_INT = 0x7fffffff
    } CUtools_cnp_support;

    typedef enum CUtoolsMemType_enum
    {
        CU_TOOLS_MEM_TYPE_INVALID = 0x0000,
        CU_TOOLS_MEM_TYPE_GENERIC = 0x0001,
        CU_TOOLS_MEM_TYPE_IMAGE = 0x0002,
        CU_TOOLS_MEM_TYPE_PUSHBUFFER = 0x0003,
        CU_TOOLS_MEM_TYPE_GPFIFOBUFFER = 0x0004,
        CU_TOOLS_MEM_TYPE_FUNCTION = 0x0005,
        CU_TOOLS_MEM_TYPE_CONTEXT_SAVE = 0x0006,
        CU_TOOLS_MEM_TYPE_TEXTURE = 0x0007,
        CU_TOOLS_MEM_TYPE_CLH_OOL = 0x0008,
        CU_TOOLS_MEM_TYPE_CONSTANT = 0x0009,
        CU_TOOLS_MEM_TYPE_VIRTUAL_CHANNEL = 0x000a,
        CU_TOOLS_MEM_TYPE_NOTIFIER = 0x000b,
        CU_TOOLS_MEM_TYPE_SKED_REFLECTED = 0x000c,
        CU_TOOLS_MEM_TYPE_SHARED_SEMAPHORE = 0x000d,
        CU_TOOLS_MEM_TYPE_QMD = 0x000e,
        CU_TOOLS_MEM_TYPE_MANAGED = 0x000f,
        CU_TOOLS_MEM_TYPE_UVM_SEMAPHORE = 0x0010,
        CU_TOOLS_MEM_TYPE_USERD = 0x0011,
        CU_TOOLS_MEM_TYPE_CONSTANT_BANKS = 0X0012,

        // --- always add new constants to the end here ---
        CU_TOOLS_MEM_TYPE_SIZE,
    } CUtoolsMemType;

    typedef enum CUtoolsMemOwner_enum
    {
        CU_TOOLS_MEM_OWNER_NONE = 0x0,
        CU_TOOLS_MEM_OWNER_DRIVER = 0x1,
        CU_TOOLS_MEM_OWNER_USER = 0x2,

        // --- always add new constants to the end here ---
        CU_TOOLS_MEM_OWNER_SIZE,
    } CUtoolsMemOwner;

    typedef enum CUtoolsMemApi_enum
    {
        // driver-internal allocation
        CU_TOOLS_MEM_API_NOT_VISIBLE = 0x0,

        // user-visible user allocation
        CU_TOOLS_MEM_API_VISIBLE = 0x1,

        //
        // allocations that can be shared out using P2P or portable host memory
        //

        // user allocation created using cuMemAlloc
        // - can be freed using cuMemFree
        CU_TOOLS_MEM_API_MEM_ALLOC = 0x2,

        // user array allocation using cuArrayCreate
        // - can be freed using cuArrayDestroy
        // - freeing will remove all portable instances
        CU_TOOLS_MEM_API_ARRAY_CREATE = 0x3,

        // user host allocation created using cuMemAllocHost
        // or cuMemHostAlloc
        // - can be freed using cuMemFreeHost
        // - freeing will remove all portable instances
        CU_TOOLS_MEM_API_HOST_ALLOC = 0x4,

        // user host allocation created using cuMemHostRegister
        // - can be freed using cuMemHostUnregister
        // - freeing will remove all portable instances
        CU_TOOLS_MEM_API_HOST_REGISTER = 0x5,

        //
        // allocations created from other allocations via P2P or portable host memory
        //

        // user allocation created in another context, shared
        // with this context via P2P automatically
        CU_TOOLS_MEM_API_MEM_ALLOC_PORTABLE = 0x6,

        // user array allocation using cuArrayCreate
        // - can be freed using cuArrayDestroy
        // - freeing will remove all portable instances
        CU_TOOLS_MEM_API_ARRAY_CREATE_PORTABLE = 0x7,

        // user host allocation created as portable in another
        // context using cuMemHostAlloc, shared to this
        // context
        // - can be freed using cuMemFreeHost
        // - freeing will remove all portable instances
        //   (including in the owning context)
        CU_TOOLS_MEM_API_HOST_ALLOC_PORTABLE = 0x8,

        // user host allocation created as portable in another
        // context using cuMemHostRegister, shared to this
        // context
        // - can be freed using cuMemHostUnregister
        // - freeing will remove all portable instances
        //   (including in the owning context)
        CU_TOOLS_MEM_API_HOST_REGISTER_PORTABLE = 0x9,

        // user allocation created in another process, shared
        // with this process via P2P if necessary. Created with
        // cuMemSharedOpen and freed with cuMemSharedClose
        CU_TOOLS_MEM_API_IPC = 0xa,

        // user managed allocation created using cuMemAllocManaged
        // - can be freed using cuMemFree
        CU_TOOLS_MEM_API_MANAGED_ALLOC = 0xb,

        // user managed allocation created in another context, shared
        // with this context via P2P automatically
        CU_TOOLS_MEM_API_MANAGED_ALLOC_PORTABLE = 0xc,

        CU_TOOLS_MEM_API_MEM_MAP = 0xd,

        // --- always add new constants to the end here ---
        CU_TOOLS_MEM_API_SIZE,
    } CUtoolsMemApi;

    typedef enum CUtoolsMemSharing_enum
    {
        CU_TOOLS_MEM_SHARING_NONE = 0x0,

        // Memory is shared by populating the DM-specific handles (e.g,
        // RM allocation handle, Global Shareable handle) from another API (e.g, GL)
        CU_TOOLS_MEM_SHARING_EXTERNAL_HANDLE = 0x1,

        // Memory is shared by specifying a host address allocated by
        // another driver or the user, and having the driver lock a
        // physical backing for that address range
        CU_TOOLS_MEM_SHARING_EXTERNAL_HOST_ADDRESS = 0x2,

        // Memory is shared by specifying a CUmemobj from another
        // CUcontext
        CU_TOOLS_MEM_SHARING_CUDA_MEMOBJ = 0x3,

        // Memory is shared by specifying a shmName from another
        // process
        CU_TOOLS_MEM_SHARING_SHM = 0x4,
    } CUtoolsMemSharing;

    typedef enum CUtoolsMemMapHost_enum
    {
        CU_TOOLS_MEM_MAP_HOST_NONE = 0x0,
        CU_TOOLS_MEM_MAP_HOST_VA = 0x1,

        // --- always add new constants to the end here ---
        CU_TOOLS_MEM_MAP_HOST_SIZE,
    } CUtoolsMemMapHost;

    typedef enum CUtoolsMemMapDevice_enum
    {
        CU_TOOLS_MEM_MAP_DEVICE_NONE = 0x0,
        CU_TOOLS_MEM_MAP_DEVICE_VA = 0x1,
        CU_TOOLS_MEM_MAP_DEVICE_PTR_FORCE_32_BIT
        = 0x2, // Force to the lower 4GB of the address space, even in 64-bit contexts
        CU_TOOLS_MEM_MAP_DEVICE_PTR_FORCE_64_BIT = 0x3, // Obsolete, do not use.
        CU_TOOLS_MEM_MAP_DEVICE_PTR
        = 0x4, // Allow to be anywhere in context's device pointer space (which may be 32-bit or 64-bit)
        CU_TOOLS_MEM_MAP_DEVICE_RANGE = 0x5,
        CU_TOOLS_MEM_MAP_DEVICE_SPECIAL_NO_VA = 0x6, // Special allocations that run through most of the "map" code, but
                                                     // don't get a VADDR (context-save/virtual-channel only)

        // --- always add new constants to the end here ---
        CU_TOOLS_MEM_MAP_DEVICE_SIZE,
    } CUtoolsMemMapDevice;

    typedef enum CUtoolsMemLocation_enum
    {
        CU_TOOLS_MEM_LOCATION_INVALID = 0x0,
        CU_TOOLS_MEM_LOCATION_HOST = 0x1,
        CU_TOOLS_MEM_LOCATION_DEVICE = 0x2,
    } CUtoolsMemLocation;

    typedef enum CUtoolsUvmLiteOwnerType_enum
    {
        CU_TOOLS_UVM_LITE_OWNER_TYPE_INVALID = 0,
        CU_TOOLS_UVM_LITE_OWNER_TYPE_ALL_STREAMS = 1,
        CU_TOOLS_UVM_LITE_OWNER_TYPE_ONE_STREAM = 2,
        CU_TOOLS_UVM_LITE_OWNER_TYPE_NO_STREAM = 3,

        // --- always add new constants to the end here ---
        CU_TOOLS_UVM_LITE_OWNER_TYPE_SIZE,
        CU_TOOLS_UVM_LITE_OWNER_TYPE_FORCE_INT = 0x7fffffff
    } CUtoolsUvmLiteOwnerType;

    typedef enum CUtools_allocation_format_enum
    {
        CU_TOOLS_ALLOCATION_FORMAT_INVALID = 0,
        CU_TOOLS_ALLOCATION_FORMAT_PITCHED_LINEAR = 1,
        CU_TOOLS_ALLOCATION_FORMAT_BLOCK_LINEAR = 2,
        // --- always add new constants to the end here ---
        CU_TOOLS_ALLOCATION_FORMAT_SIZE,
        CU_TOOLS_ALLOCATION_FORMAT_FORCE_INT = 0x7fffffff
    } CUtools_allocation_format;

    typedef enum CUtools_allocation_name_enum
    {
        CU_TOOLS_ALLOCATION_NAME_INVALID = 0,
        CU_TOOLS_ALLOCATION_NAME_USER = 1,
        CU_TOOLS_ALLOCATION_NAME_DRIVER_INTERNAL = 2,
        CU_TOOLS_ALLOCATION_NAME_DEVICE_HEAP = 3,
        CU_TOOLS_ALLOCATION_NAME_SURFACE_POOL = 4,
        CU_TOOLS_ALLOCATION_NAME_CNP_LAUNCH_QUEUE_HEAD = 5,
        CU_TOOLS_ALLOCATION_NAME_GL_INTEROP = 6,
        CU_TOOLS_ALLOCATION_NAME_CG_RUNTIME = 7,
        CU_TOOLS_ALLOCATION_NAME_HIDDEN_USER = 8,

        // --- always add new constants to the end here ---
        CU_TOOLS_ALLOCATION_NAME_SIZE,
        CU_TOOLS_ALLOCATION_NAME_FORCE_INT = 0x7fffffff
    } CUtools_allocation_name;

    typedef struct CUtoolsMemoryAllocationDescriptor_st
    {
        uint32_t struct_size;

        uint8_t allocationLocation; // one of CU_MEMORYTYPE_* defined in cuda.h
        uint8_t allocationFormat;   // one of CU_TOOLS_ALLOCATION_FORMAT_*
        uint16_t reserved0;
        uint32_t memHostAllocFlags; // flags: CU_MEMHOSTALLOC_* defined in cuda.h

        //  3D allocation parameters -- only a subset may be active depending on the allocation
        uint32_t dimensionality; // {1, 2, 3}
        uint64_t pitch;          // the 'width' dimension
        uint64_t height;
        uint64_t depth;

        //  Array-specific attributes
        uint32_t arrayFormat;      // one of CUarray_format
        uint32_t numChannels;      // channels per array element
        uint32_t ownedByDriver;    // boolean, non-zero means internal driver allocation
        uint32_t reserved1;
        uint32_t isDeviceHeap;     // boolean, non-zero means this is device heap
        uint32_t reserved2;
        uint32_t isUVMManaged;     // boolean, non zero means this is a UVM managed allocation
        uint32_t reserved3;
        uint32_t allocationName;   // one of CU_TOOLS_ALLOCATION_NAME_*
        uint32_t reserved4;
        uint32_t uvmLiteOwnerType; // one of CU_TOOLS_UVM_LITE_OWNER_TYPE
        uint32_t reserved5;
    } CUtoolsMemoryAllocationDescriptor;

    typedef enum CUtoolsMembarType_enum
    {
        CU_TOOLS_MEMBAR_TYPE_NONE = 0,
        CU_TOOLS_MEMBAR_TYPE_GL = 1,
        CU_TOOLS_MEMBAR_TYPE_SYS = 2,
    } CUtoolsMembarType;

    /// Enum describing what type of simulator the HW is running on
    typedef enum CUtools_simulation_type_enum
    {
        CU_TOOLS_SIMULATION_TYPE_NONE = 0,      // The device is actual hardware, not a simulation
        CU_TOOLS_SIMULATION_TYPE_EMULATION = 1, // The device is running on the emulator
        CU_TOOLS_SIMULATION_TYPE_FMODEL = 2,    // The device is running on Fmodel
        CU_TOOLS_SIMULATION_TYPE_AMODEL = 3,    // The device is running on Amodel
        // --- always add new constants to the end here ---
        CU_TOOLS_SIMULATION_TYPE_SIZE,
        CU_TOOLS_SIMULATION_TYPE_FORCE_INT = 0x7fffffff
    } CUtools_simulation_type;

    typedef enum CUtoolsAccess_enum
    {
        CU_TOOLS_ACCESS_PROT_NONE = 0,
        CU_TOOLS_ACCESS_PROT_READ = 1,
        CU_TOOLS_ACCESS_PROT_READWRITE = 2,
        // --- always add new constants to the end here ---
        CU_TOOLS_ACCESS_SIZE,
        CU_TOOLS_ACCESS_FORCE_INT = 0x7fffffff
    } CUtoolsAccess;

    typedef enum CUtoolsContextCigMode_enum
    {
        CU_TOOLS_CONTEXT_CIG_MODE_REGULAR = 0x0,
        CU_TOOLS_CONTEXT_CIG_MODE_CIG = 0x1,
        CU_TOOLS_CONTEXT_CIG_MODE_CIG_FALLBACK = 0x2,
        // --- always add new constants to the end here ---
        CU_TOOLS_CONTEXT_CIG_MODE_SIZE,
        CU_TOOLS_CONTEXT_CIG_MODE_FORCE_INT = 0x7fffffff
    } CUtoolsContextCigMode;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
