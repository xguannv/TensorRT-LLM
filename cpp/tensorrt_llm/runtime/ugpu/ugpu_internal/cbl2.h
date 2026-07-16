/*
 * Copyright 2022-2023 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cbl2_h__
#define __cbl2_h__

#include "cuda_uuid.h"
#include "g_nvconfig.h"
#include "nvtypes.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    CU_DEFINE_UUID(CU_ETID_CBL2, 0xd94cf65d, 0x1fb1, 0x4a23, 0xaa, 0x72, 0x30, 0xb3, 0x49, 0x65, 0x98, 0xe6);

    CU_DEFINE_UUID(CU_ETID_CBL2_STABLE, 0xfb889824, 0x9e8a, 0x4b64, 0x88, 0xec, 0x11, 0x7b, 0x2, 0x3e, 0x2c, 0x1f);

    typedef int cblStatus;

    enum cblStatus_enum
    {
        CBL_SUCCESS = 0,
        CBL_ERROR_INVALID_ARGUMENT = 1,
        CBL_ERROR_OUT_OF_MEMORY = 2,
        CBL_ERROR_ALREDY_EXISTS = 3,
        CBL_ERROR_INVALID_DEVICE = 4,
        CBL_ERROR_ALIGNMENT = 5,
        CBL_ERROR_NOT_SUPPORTED = 6,
        // The CUresult value will be stored in the upper 16 bits
        CBL_ERROR_CUDA = 0xffff,
    };

    typedef enum cblClientId_t
    {
        CBL_CLIENT_ID_UNKNOWN
        = 0x00000000, // No client should use it in production and it should be reserved for early prototyping
        CBL_CLIENT_ID_TEST = 0x00000001,   // Used for unit tests and functional tests on CBL
        CBL_CLIENT_ID_RTCORE = 0x00000002, // Used for RTCore
        CBL_CLIENT_ID_DLSS = 0x00000004,   // Used for DLSS
        CBL_CLIENT_ID_WINML = 0x00000006,  // Used for WinML
        // --- always add new constants to the end here ---
        CBL_CLIENT_ID_SIZE,
        CBL_CLIENT_ID_FORCE_INT = 0x7fffffff
    } cblClientId;

    typedef enum cblDeviceIdType_t
    {
        CBL_DEVICE_ID_TYPE_UUID = 0x00000000,    // Indicates the device id specifies a UUID
        CBL_DEVICE_ID_TYPE_LUID = 0x00000001,    // Indicates the device id specifies a LUID+index pair
        CBL_DEVICE_ID_TYPE_CUDA = 0x00000002,    // Indicates the device id specifies a cuda index
        CBL_DEVICE_ID_TYPE_RMGPUID = 0x00000003, // Indicates the device id specifies an RmGpuId
        // --- always add new constants to the end here ---
        CBL_DEVICE_ID_TYPE_SIZE,
        CBL_DEVICE_ID_TYPE_FORCE_INT = 0x7fffffff
    } cblDeviceIdType;

    typedef enum cblContextProperty_t
    {
        CBL_CONTEXT_PROPERTY_QMD_SIZE = 0x00000000, // Query the QMD Size that needs to be allocated for this context
        CBL_CONTEXT_PROPERTY_QMD_ALIGNMENT
        = 0x00000001,                               // Query the QMD Alignment that needs to be used for this context
        CBL_CONTEXT_PROPERTY_CONSTBANK_ALIGNMENT
        = 0x00000002, // Query the constant bank alignment that needs to be used for this context
        CBL_CONTEXT_PROPERTY_SYSCALL_CONSTBANK_ADDRESS
        = 0x00000003, // Query the syscall (printf only) constant bank address
        CBL_CONTEXT_PROPERTY_SYSCALL_CONSTBANK_SIZE = 0x00000004, // Query the syscall (printf only) constant bank size
        CBL_CONTEXT_PROPERTY_SYSCALL_CONSTBANK_INDEX
        = 0x00000005,                                             // Query the syscall (printf only) constant bank index
        CBL_CONTEXT_PROPERTY_SHARED_MEM_ALIGNMENT = 0x00000006,   // Query the shared memory alignment requirement
        CBL_CONTEXT_PROPERTY_LOCAL_MEM_THREAD_ALIGNMENT
        = 0x00000007,                                       // Query the per-thread local memory alignment requirement
        CBL_CONTEXT_PROPERTY_NUM_WARPS_PER_SM = 0x00000008, // Query the number of warps per sm
        CBL_CONTEXT_PROPERTY_NUM_SMS_PER_TPC = 0x00000009,  // Query the number of sms per tpc
        CBL_CONTEXT_PROPERTY_NUM_TPCS = 0x0000000a,         // Query the number of tpcs available
        CBL_CONTEXT_PROPERTY_GRID_PARAM_SIZE
        = 0x0000000b, // Query byte size of basic grid parameters in the constant bank 0
        CBL_CONTEXT_PROPERTY_USER_PARAM_OFFSET
        = 0x0000000c, // Query byte offset for kernel parameters in the constant bank 0
        CBL_CONTEXT_PROPERTY_MIN_SHARED_MEMORY_PER_SM
        = 0x0000000d, // Query minimum shared memory size per multiprocessor
        CBL_CONTEXT_PROPERTY_MAX_SHARED_MEMORY_PER_SM
        = 0x0000000e, // Query maximum shared memory size per multiprocessor
        CBL_CONTEXT_PROPERTY_COMPUTE_CAPABILITY_MAJOR = 0x0000000f, // Query major compute capability version number
        CBL_CONTEXT_PROPERTY_COMPUTE_CAPABILITY_MINOR = 0x00000010, // Query minor compute capability version number
        CBL_CONTEXT_PROPERTY_MAX_SHARED_MEMORY_PER_BLOCK
        = 0x00000011,                                               // Query the soft limit on per block shared memory
        CBL_CONTEXT_PROPERTY_WARP_SIZE = 0x00000012,                // Query the thread count per warp
        CBL_CONTEXT_PROPERTY_MAX_THREADS_PER_BLOCK = 0x00000013,    // Query the thread limit per CTA
        CBL_CONTEXT_PROPERTY_MAX_THREADS_PER_SM = 0x00000014,       // Query the thread limit per SM
        CBL_CONTEXT_PROPERTY_MAX_REGISTERS_PER_BLOCK = 0x00000015,  // Query the register limit per CTA
        CBL_CONTEXT_PROPERTY_MAX_REGISTERS_PER_SM = 0x00000016,     // Query the register limit per SM
        CBL_CONTEXT_PROPERTY_MAX_GRID_DIM_X = 0x00000017,           // Query maximum grid dimension on X
        CBL_CONTEXT_PROPERTY_LOCAL_MEM_HIGH_INTERNAL_SIZE = 0x00000018, // Query the internal reserved lmem high size
        // --- always add new constants to the end here ---
        CBL_CONTEXT_PROPERTY_SIZE,
        CBL_CONTEXT_PROPERTY_FORCE_INT = 0x7fffffff
    } cblContextProperty;

    typedef enum cblFunctionProperty_t
    {
        CBL_FUNCTION_PROPERTY_NAME
        = 0x00000000, // Function name as stored in the module (as pointer to string in NvU64)
        CBL_FUNCTION_PROPERTY_MIN_LOCAL_MEM_HI_SIZE
        = 0x00000001, // Minimum local memory high size per thread (in bytes)
        CBL_FUNCTION_PROPERTY_MIN_LOCAL_MEM_LO_SIZE = 0x00000002, // Minimum local memory low size per thread (in bytes)
        CBL_FUNCTION_PROPERTY_MIN_SHARED_MEM_SIZE = 0x00000003, // Minimum shared memory size needed per CTA (in bytes)
        CBL_FUNCTION_PROPERTY_MIN_NUM_REGISTERS = 0x00000004,   // Minimum number of general registers needed per thread
        CBL_FUNCTION_PROPERTY_MIN_NUM_BARRIERS = 0x00000005,    // Minimum number of barrier registers needed per CTA
        CBL_FUNCTION_PROPERTY_MIN_CRS_SIZE = 0x00000006,    // Minimum local memory CRS size per warp (in bytes, <Volta)
        CBL_FUNCTION_PROPERTY_USER_PARAM_SIZE = 0x00000007, // Query byte size of kernel user parameters
        CBL_FUNCTION_PROPERTY_CODE_ADDRESS = 0x00000008,    // Query code address (LaunchPc).
        CBL_FUNCTION_PROPERTY_CODE_SIZE = 0x00000009,       // Query code size in bytes.
        CBL_FUNCTION_PROPERTY_MAX_THREADS_PER_BLOCK
        = 0x0000000a, // Query the thread limit per CTA, accounting for function register count
        CBL_FUNCTION_PROPERTY_CACHE_MODE_CA = 0x0000000b, // Query compile time specified global cache mode
        CBL_FUNCTION_PROPERTY_NUM_SYSCALL_TYPES
        = 0x0000000c, // Query the number of syscall types used by the kernel (only printf supported)
        CBL_FUNCTION_PROPERTY_CONSTBANK0_SIZE = 0x00000010,    // Query byte size of ABI constant bank 0
        CBL_FUNCTION_PROPERTY_CONSTBANK1_SIZE = 0x00000011,    // Query byte size of constant bank 1
        CBL_FUNCTION_PROPERTY_CONSTBANK2_SIZE = 0x00000012,    // Query byte size of constant bank 2
        CBL_FUNCTION_PROPERTY_CONSTBANK3_SIZE = 0x00000013,    // Query byte size of constant bank 3
        CBL_FUNCTION_PROPERTY_CONSTBANK4_SIZE = 0x00000014,    // Query byte size of constant bank 4
        CBL_FUNCTION_PROPERTY_CONSTBANK5_SIZE = 0x00000015,    // Query byte size of constant bank 5
        CBL_FUNCTION_PROPERTY_CONSTBANK6_SIZE = 0x00000016,    // Query byte size of constant bank 6
        CBL_FUNCTION_PROPERTY_CONSTBANK7_SIZE = 0x00000017,    // Query byte size of constant bank 7
        CBL_FUNCTION_PROPERTY_CONSTBANK0_ADDRESS = 0x00000020, // Query address of ABI constant bank 0
        CBL_FUNCTION_PROPERTY_CONSTBANK1_ADDRESS = 0x00000021, // Query address of constant bank 1
        CBL_FUNCTION_PROPERTY_CONSTBANK2_ADDRESS = 0x00000022, // Query address of constant bank 2
        CBL_FUNCTION_PROPERTY_CONSTBANK3_ADDRESS = 0x00000023, // Query address of constant bank 3
        CBL_FUNCTION_PROPERTY_CONSTBANK4_ADDRESS = 0x00000024, // Query address of constant bank 4
        CBL_FUNCTION_PROPERTY_CONSTBANK5_ADDRESS = 0x00000025, // Query address of constant bank 5
        CBL_FUNCTION_PROPERTY_CONSTBANK6_ADDRESS = 0x00000026, // Query address of constant bank 6
        CBL_FUNCTION_PROPERTY_CONSTBANK7_ADDRESS = 0x00000027, // Query address of constant bank 7
        CBL_FUNCTION_PROPERTY_SYMBOL_INDEX = 0x00000028,       // Query the symbol index of the function
        CBL_FUNCTION_PROPERTY_INSTRUCTIONS
        = 0x00000029, // Query the instructions of the function, returns a host pointer
        CBL_FUNCTION_PROPERTY_IS_HIDDEN
        = 0x0000002a, // Query if the function is hidden (equivalent to etiFunctionIsHidden)

        // --- always add new constants to the end here ---
        CBL_FUNCTION_PROPERTY_SIZE,
        CBL_FUNCTION_PROPERTY_FORCE_INT = 0x7fffffff
    } cblFunctionProperty;

    typedef enum cblModuleInputType_t
    {
        CBL_MODULE_INPUT_TYPE_CUBIN = 0x00000000,     // Input specifies a pointer to a loaded cubin
        CBL_MODULE_INPUT_TYPE_PTX = 0x00000001,       // Input specifies a pointer to a null-terminated PTX ascii string
        CBL_MODULE_INPUT_TYPE_FATBINARY = 0x00000002, // Input specifies a pointer to a loaded fatbinary
        // --- always add new constants to the end here ---
        CBL_MODULE_INPUT_TYPE_SIZE,
        CBL_MODULE_INPUT_TYPE_FORCE_INT = 0x7fffffff
    } cblModuleInputType;

    typedef enum cblModuleLoadOptionName_t
    {
        CBL_MODULE_LOAD_OPTION_MAX_REGISTERS
        = 0x00000000, // Option value specifies the maximum number of registers allowed (JIT)
        CBL_MODULE_LOAD_OPTION_THREADS_PER_BLOCK
        = 0x00000001, // Option value specifies the threads per block for occupancy calculations (JIT)
        CBL_MODULE_LOAD_OPTION_WALL_TIME
        = 0x00000002, // Option value will have the wall time for the JIT operation (JIT)
        CBL_MODULE_LOAD_OPTION_INFO_LOG_BUFFER = 0x00000003, // Option value specifies a buffer that will have INFO
                                                             // messages written to by the JIT compiler (JIT)
        CBL_MODULE_LOAD_OPTION_INFO_LOG_BUFFER_SIZE_BYTES
        = 0x00000004, // Option value specifies the size of the INFO buffer (JIT) \sa
                      // CBL_MODULE_LOAD_OPTION_INFO_LOG_BUFFER
        CBL_MODULE_LOAD_OPTION_ERROR_LOG_BUFFER = 0x00000005, // Option value specifies a buffer that will have ERROR
                                                              // messages written to by the JIT compiler (JIT)
        CBL_MODULE_LOAD_OPTION_ERROR_LOG_BUFFER_SIZE_BYTES
        = 0x00000006, // Option value specifies the size of the ERROR buffer (JIT) \sa
                      // CBL_MODULE_LOAD_OPTION_ERROR_LOG_BUFFER
        CBL_MODULE_LOAD_OPTION_OPTIMIZATION_LEVEL
        = 0x00000007, // Option value specifies the level of optimization (JIT)
        CBL_MODULE_LOAD_OPTION_GENERATE_DEBUG_INFO
        = 0x00000008, // Option value specifies that debug info sections be generated for debugging (JIT)
        CBL_MODULE_LOAD_OPTION_LOG_VERBOSE
        = 0x00000009, // Option value specifies the verbosity level of specified logs (JIT)
        CBL_MODULE_LOAD_OPTION_GENERATE_LINE_INFO
        = 0x0000000A, // Option value specifies that line info sections be generated for debugging (JIT)
        CBL_MODULE_LOAD_OPTION_CACHE_MODE = 0x0000000B, // Option value specifies the cache mode used
        CBL_MODULE_LOAD_OPTION_GLOBAL_SYMBOL_NAMES = 0x0000000C,
        CBL_MODULE_LOAD_OPTION_GLOBAL_SYMBOL_ADDRESSES = 0x0000000D,
        CBL_MODULE_LOAD_OPTION_GLOBAL_SYMBOL_COUNT = 0x0000000E,
        CBL_MODULE_LOAD_OPTION_LTO = 0x0000000F, // Option value specifies if link-time optimization should be enabled
        CBL_MODULE_LOAD_OPTION_FTZ = 0x00000010,
        CBL_MODULE_LOAD_OPTION_PREC_DIV = 0x00000011,
        CBL_MODULE_LOAD_OPTION_PREC_SQRT = 0x00000012,
        CBL_MODULE_LOAD_OPTION_FMA = 0x00000013,
        CBL_MODULE_LOAD_OPTION_REFERENCED_KERNEL_NAMES = 0x00000014,
        CBL_MODULE_LOAD_OPTION_REFERENCED_KERNEL_COUNT = 0x00000015,
        CBL_MODULE_LOAD_OPTION_REFERENCED_VARIABLE_NAMES = 0x00000016,
        CBL_MODULE_LOAD_OPTION_REFERENCED_VARIABLE_COUNT = 0x00000017,
        CBL_MODULE_LOAD_OPTION_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 0x00000018,
        CBL_MODULE_LOAD_OPTION_SW4575628 = 0x00000019,
        CBL_MODULE_LOAD_OPTION_TEXMODE_RAW = 0x0000001A,
#if NVCFG(GLOBAL_FEATURE_CTK8050_CBL_MODULE_PTX_CACHE_CONTROL)
        CBL_MODULE_LOAD_OPTION_PTX_CACHE
        = 0x0000001B, // Option value specified whether the result of PTX JIT should be emplaced into the ComputeCache
#else
    CBL_MODULE_LOAD_OPTION_RESERVED_1B = 0x0000001B,
#endif
        // --- always add new constants to the end here ---
        CBL_MODULE_LOAD_OPTION_SIZE,
        CBL_MODULE_LOAD_OPTION_FORCE_INT = 0x7fffffff
    } cblModuleLoadOptionName;

#if NVCFG(GLOBAL_FEATURE_CTK8050_CBL_MODULE_PTX_CACHE_CONTROL)
    typedef enum cblModuleLoadOptionJitCache_t
    {
        /**
         * Tells the driver to not emplace JIT results into the compute cache.
         */
        CBL_MODULE_LOAD_OPTION_PTX_CACHE_DISABLE = 0,
        /**
         * Tells the driver to cache JIT results into a shared user cache. This
         * _may_ displace existing cache entries from other applications sharing
         * the same compute cache.
         */
        CBL_MODULE_LOAD_OPTION_PTX_CACHE_ENABLE = 1,
        /**
         * Tells the driver to cache JIT results into a shared user cache. This
         * will unconditionally cache the result regardless of size. It will not
         * displace existing cache entries.
         */
        CBL_MODULE_LOAD_OPTION_PTX_CACHE_ALWAYS = 2
    } cblModuleLoadOptionJitCache;
#endif

    typedef enum cblModuleProperty_t
    {
        CBL_MODULE_PROPERTY_USER_CONSTBANK_DPTR = 0,
        CBL_MODULE_PROPERTY_USER_CONSTBANK_SIZE = 1,
        // --- always add new constants to the end here ---
        CBL_MODULE_PROPERTY_SIZE,
        CBL_MODULE_PROPERTY_FORCE_INT = 0x7fffffff
    } cblModuleProperty;

    // Flags for controlling the behavior of semaphore reductions
    // No need to make them bit based - we dont expect them to be used to be used all at same time
    typedef enum cblSemaphoreReductionFlags_enum
    {
        CBL_SEMAPHORE_REDUCTION_FLAGS_MIN = 0x00,      // Perform atomic min reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_MAX = 0x01,      // Perform atomic max reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_XOR = 0x02,      // Perform atomic xor reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_AND = 0x03,      // Perform atomic and reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_OR = 0x04,       // Perform atomic or reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_ADD = 0x05,      // Perform atomic add reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_INC = 0x06,      // Perform atomic inc reduction
        CBL_SEMAPHORE_REDUCTION_FLAGS_DEC = 0x07,      // Perform atomic dec reduction

        CBL_SEMAPHORE_REDUCTION_FLAGS_SIGNED = 0x10,   // Perform a reduction on signed values
        CBL_SEMAPHORE_REDUCTION_FLAGS_UNSIGNED = 0x20, // Perform a reduction on unsigned values

        CBL_SEMAPHORE_REDUCTION_FLAGS_INTERRUPT
        = 0x100, // Indicate that the semaphore reduction will also trigger a non-stalling interrupt
    } cblSemaphoreReductionFlags;

    // Flags for controlling the behavior of compute semaphore releases
    typedef enum cblSemaphoreReleaseComputeFlags_enum
    {
        CBL_SEMAPHORE_RELEASE_COMPUTE_FLAGS_NONE = 0,
        CBL_SEMAPHORE_RELEASE_COMPUTE_FLAGS_ONE_WORD
        = 1, // One word (payload only) release instead of standard four word.
        CBL_SEMAPHORE_RELEASE_COMPUTE_FLAGS_NO_MEMBAR = 2,
        CBL_SEMAPHORE_RELEASE_COMPUTE_FLAGS_INTERRUPT
        = 4, // Indicate that the semaphore release will also trigger a non-stalling interrupt
    } cblSemaphoreReleaseComputeFlags;

    typedef enum cblQmdPcasAction_t
    {
        CBL_QMD_PCAS_ACTION_INVALIDATE = 0x1,
        CBL_QMD_PCAS_ACTION_SCHEDULE = 0x2,
        CBL_QMD_PCAS_ACTION_COPY_HW_FIELDS = 0x4,
        CBL_QMD_PCAS_ACTION_PREFETCH = 0x8,
        CBL_QMD_PCAS_ACTION_DECREMENT = 0x10,
        // --- always add new constants to the end here ---
        CBL_QMD_PCAS_ACTION_SIZE,
        CBL_QMD_PCAS_ACTION_FORCE_INT = 0x7fffffff
    } cblQmdPcasAction;

    typedef enum cblQmdMembarType_t
    {
        CBL_QMD_MEMBAR_FE_MASK = 0xF0,
        CBL_QMD_MEMBAR_FE_NONE = 0x00,
        CBL_QMD_MEMBAR_FE_SYS = 0x10,

        CBL_QMD_CWD_MEMBAR_CWD_MASK = 0xF00,
        CBL_QMD_CWD_MEMBAR_CWD_NONE = 0x000,
        CBL_QMD_MEMBAR_CWD_L1_SYS = 0x100,
        CBL_QMD_MEMBAR_CWD_L1 = 0x200,
        // --- always add new constants to the end here ---
        CBL_QMD_MEMBAR_SIZE,
        CBL_QMD_MEMBAR_FORCE_INT = 0x7fffffff
    } cblQmdMembarType;

    typedef enum cblFuncCachePreferType_t
    {
        CBL_FUNC_CACHE_PREFER_NONE = 0x00,   // no preference for shared memory or L1 (default)
        CBL_FUNC_CACHE_PREFER_SHARED = 0x01, // prefer larger shared memory and smaller L1 cache
        CBL_FUNC_CACHE_PREFER_L1 = 0x02,     // prefer larger L1 cache and smaller shared memory
        CBL_FUNC_CACHE_PREFER_EQUAL = 0x03,  // prefer equal sized L1 cache and shared memory
        // --- always add new constants to the end here ---
        CBL_FUNC_CACHE_PREFER_SIZE,
        CBL_FUNC_CACHE_PREFER_INT = 0x7fffffff
    } cblFuncCachePreferType;

    typedef enum
    {
        CBL_DEVTOOLS_PROP_DEV_CB_ON_QMD_ENCODED,
    } cblDevtoolsProperty;

#define CBL_MAX_QMD_SEMAPHORE_RELEASE_COUNT (3)
#define CBL_MAX_QMD_DEPENDENT_QMD_COUNT (2)
#define CBL_MAX_SM_DISABLE_MASK_DWORDS (8)
#define CBL_MAX_CONSTANT_BUFFER_COUNT (8)

    typedef struct cblDeviceId_st
    {
        // Type of device id
        cblDeviceIdType type;

        union Id
        {
            // UUID of device of interest
            NvU8 uuid[16];

            struct LUIDAndPhysicalAdapter
            {
                // LUID of adapter of interest
                NvU64 luid;
                // Set to the physical device within the Linked Display Adapter
                NvU32 physicalAdapterIndex;
            } luidInfo;

            // cuda device id of interest
            NvS32 cudaDeviceId;
            // RmGpuId
            NvU32 rmGpuId;
        } id;
    } cblDeviceId;

    // context handle
    typedef void* cblContext;
    // base driver handle
    typedef struct cblBaseDriver_st* cblBaseDriverHandle;

    // base driver callback structure, retrieved from base driver implementation
    typedef struct cblBaseDriverCallbacks_st const* cblBaseDriverCallbacks;
    // devtools attach callback structure
    typedef void (*cblDevtoolsAttachCallback)(void* userData);

    // CBL Context management API
    typedef struct cblCreateContextParams_st
    {
        /// @name Inputs
        /// @{

        // Client information
        cblClientId clientId;
        NvU32 clientVersion;

        // Base driver information
        cblBaseDriverCallbacks baseDriverCallbacks;

        // An opaque handle that will be passed to all the callbacks in pBaseDriverCallback.
        cblBaseDriverHandle hBaseDriver;

        // Device information
        cblDeviceId deviceId;

        // Devtools information
        // A callback used to inform the client that devtools is attached/detached and they need to start/stop notify
        // them from now on
        cblDevtoolsAttachCallback devtoolsAttachCallback;
        void* callbackUserData;

        /// @}

        /// @name Outputs
        /// @{

        cblContext hContext;

        // Device Caps
        NvBool supportsHostAllocs; // Whether or not the base driver callbacks supports host allocs

        /// @}
    } cblCreateContextParams;

    typedef void* cblModule;

    typedef struct cblModuleInput_st
    {
        cblModuleInputType type; // Type of module input
        void* data;              // Pointer to module input data
        NvU64 size;              // Size of buffer pointed to by data (not necessary for PTX or FATBINARY types)
    } cblModuleInput;

    typedef struct cblModuleLoadOption_st
    {
        cblModuleLoadOptionName option; // Option specified
        void* value;                    // IN/OUT Pointer to value for option
    } cblModuleLoadOption;

    typedef struct cblLoadModuleParams_st
    {
        /// @name Inputs
        /// @{
        cblContext hContext;

        NvU64 inputCount;
        cblModuleInput* inputs;

        NvU64 optionCount;
        cblModuleLoadOption* options;

        NvU64 allowUnresolvedSymbols : 1;
        /// @}

        /// @name Outputs
        /// @{

        cblModule hModule;

        /// @}
    } cblLoadModuleParams;

    typedef void* cblFunction;

    typedef struct cblModuleGetFunctionsParams_st
    {
        /// @name Inputs
        /// @{
        size_t structSize;
        cblModule hModule;
        /// @}

        /// @name Input/Outputs
        /// @{
        NvU64 functionCount;    // Size of the function array.  This will be updated to the number of valid entries in
                                // after the call
        cblFunction* functions; // Caller allocated array that is big enough to hold functionCount or Null (to query the
                                // function count)
        /// @}
    } cblModuleGetFunctionsParams;

    typedef struct cblEncodeQmdParams_st
    {
        // Context in which to encode the QMD
        cblContext hContext;

        // Buffer to hold the encoded QMD, must be at least CBL_CONTEXT_PROPERTY_QMD_SIZE bytes
        void* buffer;

        // Grid dimension (number of blocks)
        NvU32 gridWidth;
        NvU32 gridHeight;
        NvU32 gridDepth;

        // Number of thread per block
        NvU32 threadBlockWidth;
        NvU32 threadBlockHeight;
        NvU32 threadBlockDepth;

        // Pointer to the first address of the program
        NvU64 programLaunchPc;

        // Information on how to prefetch program data into the instruction cache (Turing+)
        NvU64 programPrefetchAddress;
        NvU32 programPrefetchSize;

        // Constant buffers info
        struct cblQmdConstantBuffer_st
        {
            NvBool valid;
            NvU64 address;
            NvU64 size;
            NvBool invalidate;
        } constantBuffer[CBL_MAX_CONSTANT_BUFFER_COUNT];

        // Number of per-thread registers allocated for the program.
        // @sa CBL_FUNCTION_PROPERTY_MIN_NUM_REGISTERS
        NvU16 registerCount;
        // Number of barrier per CTA
        // @sa CBL_FUNCTION_PROPERTY_MIN_NUM_BARRIERS
        NvU16 barrierCount;

        // Bitmask of disabled logical SM (post-floorswept) or TPC (2 SMs groups) for Volta+.
        NvU32 smDisableMask[CBL_MAX_SM_DISABLE_MASK_DWORDS];

        // Total size of shared memory
        // @sa CBL_FUNCTION_PROPERTY_MIN_SHARED_MEM_SIZE
        NvU32 sharedMemorySize;

        // size of the per-thread memory for variable storage
        // @sa CBL_FUNCTION_PROPERTY_MIN_LOCAL_MEM_HI_SIZE
        NvU64 localMemoryHighSize;
        // size of the per-thread memory for data stack
        // @sa CBL_FUNCTION_PROPERTY_MIN_LOCAL_MEM_LO_SIZE
        NvU64 localMemoryLowSize;
        // size of the per-thread crs (Call Return Stack) for Pascal-
        // @sa CBL_FUNCTION_PROPERTY_MIN_CRS_SIZE
        NvU64 localMemoryCrsSize;

        // set the priority group associated to this QMD.
        NvU8 groupId;
        // when set to true the QMD will be put at the top of the queue
        // for its specified priority group (i.e. it should be the next
        // one to be scheduled when its priority group will be picked)
        NvBool addToHeadOfGroupList;

        // The following fields indicates which caches must be invalidated
        // when the task is launched by SKED
        NvBool invalidateTextureHeaderCacheAtLaunch;
        NvBool invalidateTextureSamplerCacheAtLaunch;
        NvBool invalidateTextureDataCacheAtLaunch;
        NvBool invalidateShaderDataCacheAtLaunch;
        NvBool invalidateInstructionCacheAtLaunch;
        NvBool invalidateShaderConstantCacheAtLaunch;

        // When set to true only Scheduling PCAS can put this QMD in the scheduling queue.
        // This means that other PCAS like the copy PCAS that happens in an IQ2M will put
        // the QMD in the sched cache but will not add the QMD to the scheduling queue.
        NvBool requireSchedulingPcas;
        // When set to true, HW will issue a PCAS COPY on the QMD at completion,
        // allowing a subsequent Scheduling PCAS to trigger a re-launch of this QMD
        // immediately after it's completion
        NvBool selfCopyAtCompletion;

        // This is the initial value of the counter for this QMD that is updated via
        // PCAS increments and decrements.  Once the counter is zero, SKED issues a
        // scheduling pcas and adds the QMD to the scheduling queue.  This requires
        // "requireSchedulingPcas" to be set.
        NvU16 dependenceCounter;

        // CBL defaults to independent and CUDA defaults to combined texture sampler
        NvBool isCombinedTexSampler;

        // Enforce API Visible Call Limit of 32
        NvBool enforceApiVisibleCallLimit;

        // Used to configure QMD chaining
        struct cblQmdDependentQmd_st
        {
            NvBool enabled;
            NvU64 address;
            cblQmdPcasAction action;
        } dependentQmd[CBL_MAX_QMD_DEPENDENT_QMD_COUNT];

        // Indicates what kind of semaphore must be released when this QMD will completes
        struct cblQmdSemaphoreRelease_st
        {
            NvBool enabled;
            NvU64 address;
            NvU64 payload;
            NvBool useReduction;
            NvU32 reductionFlags; // : cblSemaphoreReductionFlags
            NvU32 flags;          // : cblSemaphoreReleaseComputeFlags
        } semaphoreRelease[CBL_MAX_QMD_SEMAPHORE_RELEASE_COUNT];

        // Performance hint on some architecture:
        //    HW will assume that this specified number of CTA can fit
        //    at once on an empty SM and this without waiting for feedback
        //    to determine the free slots remaining on the SM.
        NvU64 hintFreeCtaSlotsOnAnEmptySm;

        NvBool smGlobalCaching;

        // For Volta+ indicates how to select and
        // how to configure an SM for shared memory / L1 usage.
        // The hardware will first look for an SM that has a configuration
        // that matches the range [smConfigurationMinSharedMemorySize, smConfigurationMaxSharedMemorySize]
        // if Hardware can't find any it will provision a new SM with using smConfigurationTargetSharedMemorySize
        NvU32 smConfigurationMinSharedMemorySize;
        NvU32 smConfigurationTargetSharedMemorySize;
        NvU32 smConfigurationMaxSharedMemorySize;

        // Scoped membar issued at completion of the QMD (when the last kernel completes)
        cblQmdMembarType membar;

        // For Ampere GA10X+ It is possible to use a feature called
        // SM simultaneous compute and compute usually called SMSCC.
        // This feature is mainly designed to help with GPU utilization
        // and raytracing and more detail on the full behavior can be found
        // in bug: 2509750
        NvU32 ctaLaunchQueue;
        NvU32 occupancyThresholdWarp;
        NvU32 occupancyThresholdRegister;
        NvU32 occupancyThresholdSharedMem;
        NvU32 occupancyMaxWarp;
        NvU32 occupancyMaxRegister;
        NvU32 occupancyMaxSharedMem;

        NvBool demotePersistingL2LinesAtCompletion;

        NvU32 correlationID; // Correlation ID for Blackwell GB20X+ HES
    } cblEncodeQmdParams;

    typedef struct cblEncodeParametersConstantBankParams_st
    {
        // @name Input
        // @{
        // Context in which to encode the constant bank
        cblFunction hFunction;

        void* buffer;
        NvU32 bufferSize;

        // @name grid dimension information (this should match the gridWidth and threadBlock fields in the QMD)
        // @{
        NvU32 gridDimX;
        NvU32 gridDimY;
        NvU32 gridDimZ;
        NvU32 blockDimX;
        NvU32 blockDimY;
        NvU32 blockDimZ;
        // @}

        // Amount of dynamic shared memory in bytes (does not include static or reserved shared memory)
        NvU32 dynamicSharedMemBytes;

        // Device address of each constant bank
        NvU64 constantBankDevicePointer[CBL_MAX_CONSTANT_BUFFER_COUNT];
        // @}

        // @name Output
        // @{
        // Offset in buffer in which to copy packed parameters
        void* pPackedParameters;
        // Expected size of the packed parameters
        NvU32 packedParametersSize;
        // @}
    } cblEncodeParametersConstantBankParams;

    typedef struct cblEncodeParametersConstantBankParamsV2_st
    {
        // @name Input
        // @{
        // Context in which to encode the constant bank
        cblFunction hFunction;

        void* buffer;
        NvU32 bufferSize;

        // @name grid dimension information (this should match the gridWidth and threadBlock fields in the QMD)
        // @{
        NvU32 gridDimX;
        NvU32 gridDimY;
        NvU32 gridDimZ;
        NvU32 blockDimX;
        NvU32 blockDimY;
        NvU32 blockDimZ;
        // @}

        // Amount of dynamic shared memory in bytes (does not include static or reserved shared memory)
        NvU32 dynamicSharedMemBytes;

        // Device address of each constant bank
        NvU64 constantBankDevicePointer[CBL_MAX_CONSTANT_BUFFER_COUNT];

        // Layout of the per-thread stack. This adds up to the total lmem high size.
        struct
        {
            // Reserved stack size by the driver
            NvU32 internalSize;    // Constant
            NvU32 systemStackSize; // State managed by the base driver
            // Available as kernel thread stack
            NvU32 dataStackSize; // Constant function property
        } lmemHighLayout;

        // @}

        // @name Output
        // @{
        // Offset in buffer in which to copy packed parameters
        void* pPackedParameters;
        // Expected size of the packed parameters
        NvU32 packedParametersSize;
        // @}
    } cblEncodeParametersConstantBankParamsV2;

    typedef struct cblGetTotalLocalMemory_st
    {
        /// @name Inputs
        /// @{
        size_t structSize;
        cblFunction hFunction;
        /// @}
        /// @name Outputs
        /// @{
        NvU32 localMemoryHighSize; // The calculated per-thread size of local memory high (for encoding in QMDs)
        NvU32 localMemoryLowSize;  // The calculated per-thread size of local memory low (for encoding in QMDs)
        NvU32 localMemoryCrsSize;  // The calculated per-warp size of the CRS (for encoding in QMDs)
        NvU64
            localMemoryTotalSize; // The total local memory size for the entire device @sa cblCudaReconfigureLocalMemory
        /// @}
    } cblGetTotalLocalMemoryParams;

    typedef struct cblFunctionOptimalShmem_st
    {
        /// @name Inputs
        /// @{
        cblFunction hFunction;
        NvU32 blockSize;
        NvU32 sharedMemorySize;
        cblFuncCachePreferType cacheConfig;
        /// @}
        /// @name Outputs
        /// @{
        NvU32 optimalShmem; // The calculated optimal shared memory size
        /// @}
    } cblFunctionOptimalShmemParams;

    typedef struct cblModuleGetGlobalParams_st
    {
        /// @name Inputs
        /// @{
        cblModule hModule;
        char const* name;
        /// @}

        /// @name Outputs
        /// @{
        NvU64 dptr;
        size_t size;
        /// @}
    } cblModuleGetGlobalParams;

    typedef struct cblModuleGetImageParams_st
    {
        /// @name Inputs
        /// @{
        size_t structSize;
        cblModule hModule;
        /// @}

        /// @name Outputs
        /// @{
        void const* image;
        size_t imageSize;
        /// @}
    } cblModuleGetImageParams;

    typedef struct cblEncodeDevtoolsConstantsParams_st
    {
        /// @name Inputs
        /// @{
        cblContext hContext;
        void* pConstBank0Host;
        NvU64 gridId;
        NvU64 selfQmdVaddr;
        NvU64 startPc;
        NvU32 totalShmemSize;
        /// @}
    } cblEncodeDevtoolsConstantsParams;

    typedef struct cblSyscallOnKernelLaunchParams_st
    {
        /// @name Inputs
        /// @{
        cblContext hContext;
        // Optional handle to the launched function. If NULL, printf will be setup regardless.
        cblFunction hFunction;
        /// @}
    } cblSyscallOnKernelLaunchParams;

    typedef struct CUetblCBL2_st
    {
        // This export table supports versioning by adding to the end without
        // changing the ETID.  The struct_size field will always be set to the
        // size in bytes of the entire export table structure.
        size_t struct_size;

        cblStatus(CUDAAPI* cblContextCreate)(cblCreateContextParams* pParams);
        cblStatus(CUDAAPI* cblContextDestroy)(cblContext hContext);
        cblStatus(CUDAAPI* cblContextQueryProperties)(
            cblContext hContext, NvU32 propertyCount, cblContextProperty* pProperty, NvU64* pOutProperties);
        cblStatus(CUDAAPI* cblModuleLoad)(cblLoadModuleParams* pParams);
        cblStatus(CUDAAPI* cblModuleUnload)(cblModule hModule);
        cblStatus(CUDAAPI* cblModuleQueryProperties)(
            cblModule hModule, NvU32 propertyCount, cblModuleProperty* pProperty, NvU64* pOutProperties);
        cblStatus(CUDAAPI* cblModuleGetFunctions)(cblModuleGetFunctionsParams* pParams);
        cblStatus(CUDAAPI* cblEncodeQmd)(cblEncodeQmdParams* pParams);
        cblStatus(CUDAAPI* cblEncodeParametersConstantBank)(cblEncodeParametersConstantBankParams* pParams);
        cblStatus(CUDAAPI* cblFunctionQueryProperties)(
            cblFunction hFunction, NvU32 propertyCount, cblFunctionProperty* pProperty, NvU64* pOutProperties);
        cblStatus(CUDAAPI* cblGetTotalLocalMemory)(cblGetTotalLocalMemoryParams* pParams);
        cblStatus(CUDAAPI* cblFunctionOptimalShmem)(cblFunctionOptimalShmemParams* pParams);
        cblStatus(CUDAAPI* cblDevtoolsQuery)(
            cblContext hContext, NvU32 propertyCount, cblDevtoolsProperty* pProperty, NvU64* pOutProperties);
        cblStatus(CUDAAPI* cblDevtoolsNotify)(NvU32 callbackId, void* payload);
        cblStatus(CUDAAPI* cblModuleGetGlobal)(cblModuleGetGlobalParams* pParams);
        cblStatus(CUDAAPI* cblModuleGetImage)(cblModuleGetImageParams* pParams);
        cblStatus(CUDAAPI* cblEncodeDevtoolsConstants)(cblEncodeDevtoolsConstantsParams* pParams);
        cblStatus(CUDAAPI* cblSyscallOnKernelLaunch)(cblSyscallOnKernelLaunchParams* pParams);
        cblStatus(CUDAAPI* cblEncodeParametersConstantBankV2)(cblEncodeParametersConstantBankParamsV2* pParams);
    } CUetblCBL2;

    // A subset of the CBL 2 export table that is ABI stable. For use by tools that cannot be updated with the driver.
    typedef struct CUetblCBL2Stable_st
    {
        size_t struct_size;

        cblStatus(CUDAAPI* cblModuleGetFunctions)(cblModuleGetFunctionsParams* pParams);
        cblStatus(CUDAAPI* cblModuleGetImage)(cblModuleGetImageParams* pParams);
        cblStatus(CUDAAPI* cblModuleQueryProperties)(
            cblModule hModule, NvU32 propertyCount, cblModuleProperty* pProperty, NvU64* pOutProperties);
        cblStatus(CUDAAPI* cblFunctionQueryProperties)(
            cblFunction hFunction, NvU32 propertyCount, cblFunctionProperty* pProperty, NvU64* pOutProperties);
    } CUetblCBL2Stable;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __cbl2_h__
