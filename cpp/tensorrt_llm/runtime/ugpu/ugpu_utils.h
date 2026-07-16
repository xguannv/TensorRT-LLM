/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>

namespace tensorrt_llm
{

namespace ugpu
{

class UgpuLocalization;

class UgpuLocalizationHandle
{
public:
    UgpuLocalizationHandle();
    ~UgpuLocalizationHandle();

    // Delete copy constructor and copy assignment
    UgpuLocalizationHandle(UgpuLocalizationHandle const&) = delete;
    UgpuLocalizationHandle& operator=(UgpuLocalizationHandle const&) = delete;

    // Allow move constructor and move assignment
    UgpuLocalizationHandle(UgpuLocalizationHandle&&) noexcept;
    UgpuLocalizationHandle& operator=(UgpuLocalizationHandle&&) noexcept;

    bool supportsUgpuLocalization() const;

    void ugpuMalloc(void** localizedDevPtr, size_t size, int ugpuId);
    void ugpuFree(void* localizedDevPtr);
    CUmemGenericAllocationHandle createUgpuLocalizedAllocationHandle(
        size_t size, int ugpuId, unsigned int requestedHandleTypes, bool gpuDirectRDMACapable);

    CUstream createUgpuLocalizedStream(int ugpuId);

private:
    UgpuLocalization* mImpl;
};

} // namespace ugpu

} // namespace tensorrt_llm
