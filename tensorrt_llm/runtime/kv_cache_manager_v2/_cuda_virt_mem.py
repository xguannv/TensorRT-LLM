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


import enum
import os
from typing import Final, Optional

import cuda.bindings.driver as drv

from ._common import MemAddress
from ._exceptions import CuError
from ._utils import (
    ItemHolderWithSharedPool,
    PooledFactoryBase,
    _unwrap,
    div_up,
    get_current_device_id,
)

MEM_DEBUG: Final[int] = int(os.environ.get("V2_KV_CACHE_MEM_DEBUG", "0")) == 1


class LocalizationMode(str, enum.Enum):
    AUTO = "auto"
    OFF = "off"
    MOCK = "mock"


def _get_localization_mode() -> LocalizationMode:
    """Return the configured localization mode.

    ``TRT_LLM_MOCK_LOCALIZATION_SUPPORT`` controls the mode:
      - unset: probe hardware support
      - ``0``: force non-localized behavior
      - ``1``: force localized logic while using non-localized allocations
    """
    mode = os.environ.get("TRT_LLM_MOCK_LOCALIZATION_SUPPORT")
    if mode is None:
        return LocalizationMode.AUTO
    if mode == "0":
        return LocalizationMode.OFF
    if mode == "1":
        return LocalizationMode.MOCK
    raise ValueError("TRT_LLM_MOCK_LOCALIZATION_SUPPORT must be '0' or '1' when set")


def _get_ugpu_localization_handle():
    import tensorrt_llm.bindings.internal.runtime as _tbr

    return _tbr.UgpuLocalizationHandle()


def _location_to_ugpu_id(location: int) -> int:
    if location == _UGPU_LOC_UGPU0:
        return 0
    if location == _UGPU_LOC_UGPU1:
        return 1
    raise ValueError(f"Unsupported uGPU localization location: {location}")


def _create_localized_allocation_handle(
    size: int, prop: drv.CUmemAllocationProp, ugpu_id: int
) -> drv.CUmemGenericAllocationHandle:
    handle = _get_ugpu_localization_handle().create_ugpu_localized_allocation_handle(
        size,
        ugpu_id,
        int(prop.requestedHandleTypes),
        bool(prop.allocFlags.gpuDirectRDMACapable),
    )
    return drv.CUmemGenericAllocationHandle(handle)


def is_device_localization_supported(device_id: int) -> bool:
    """Return whether the given device supports uGPU memory localization.

    Callers still need to opt in through ``GpuCacheTierConfig.enable_ugpu``.
    ``TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1`` forces the localized path for
    tests. ``TRT_LLM_MOCK_LOCALIZATION_SUPPORT=0`` forces the non-localized
    path even on localization-capable hardware. Unset probes real hardware
    support.
    """
    mode = _get_localization_mode()
    if mode is LocalizationMode.OFF:
        return False
    if mode is LocalizationMode.MOCK:
        return True

    current_device_id = get_current_device_id()
    if device_id != current_device_id:
        if MEM_DEBUG:
            print(
                "is_device_localization_supported called for non-current device: "
                f"requested={device_id} current={current_device_id}"
            )
        return False

    try:
        return _get_ugpu_localization_handle().supports_ugpu_localization()
    except Exception:
        return False


def _is_prop_supported(prop: drv.CUmemAllocationProp, location: Optional[int] = None) -> bool:
    mode = _get_localization_mode()
    if location is not None and mode is not LocalizationMode.MOCK:
        try:
            handle = _create_localized_allocation_handle(
                2 << 20, prop, _location_to_ugpu_id(location)
            )
        except Exception as exc:
            if MEM_DEBUG:
                print(
                    "localized allocation prop unsupported: "
                    f"requestedHandleTypes={int(prop.requestedHandleTypes)} "
                    f"gpuDirectRDMACapable={int(prop.allocFlags.gpuDirectRDMACapable)} "
                    f"location={location} error={exc}"
                )
            return False
        _unwrap(drv.cuMemRelease(handle))
        return True

    err, handle = drv.cuMemCreate(2 << 20, prop, 0)
    err_int = int(err)
    if err_int == int(drv.CUresult.CUDA_SUCCESS):
        _unwrap(drv.cuMemRelease(handle))
        return True
    # Note: OOM is intentionally not caught here — OOM on a 2 MiB probe
    # indicates a fundamental resource problem, not an unsupported property.
    elif err_int in (
        int(drv.CUresult.CUDA_ERROR_NOT_PERMITTED),
        int(drv.CUresult.CUDA_ERROR_NOT_SUPPORTED),
        int(drv.CUresult.CUDA_ERROR_INVALID_DEVICE),
        int(drv.CUresult.CUDA_ERROR_INVALID_VALUE),
    ):
        return False
    else:
        raise CuError(err)


# Physical memory
class NativePhysMemAllocator:
    __slots__ = (
        "_device_id",
        "_size",
        "_prop",
        "_outstanding_handles",
        "_outstanding_handles_location",
        "_support_localization",
        "_location",
    )

    _device_id: int
    _size: int
    _prop: drv.CUmemAllocationProp
    _outstanding_handles: set[int]  # allocated but not released
    _outstanding_handles_location: dict[int, Optional[int]]  # handle_int -> location
    _support_localization: bool
    _location: Optional[int]  # uGPU location baked in at construction; None for non-localized

    def __init__(
        self,
        size: int,
        support_localization: bool = False,
        location: int | None = None,
    ) -> None:
        self._device_id = get_current_device_id()
        self._size = size
        self._support_localization = support_localization
        self._location = location

        if self._support_localization:
            localization_supported = is_device_localization_supported(self._device_id)
            if not localization_supported:
                raise ValueError("Localization is not supported on this device")
            if MEM_DEBUG:
                print(
                    f"[] Localization mode: device_id={self._device_id} "
                    f"chunk_size={size} location={location}"
                )
        else:
            if MEM_DEBUG:
                print(f"[] device_id={self._device_id} chunk_size={size}")

        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device_id
        prop.allocFlags.gpuDirectRDMACapable = 1
        prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        if not _is_prop_supported(prop, location if self._support_localization else None):
            prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            if MEM_DEBUG:
                print(
                    "CU_MEM_HANDLE_TYPE_FABRIC not supported, "
                    "falling back to CU_MEM_HANDLE_TYPE_NONE"
                )
            if not _is_prop_supported(prop, location if self._support_localization else None):
                prop.allocFlags.gpuDirectRDMACapable = 0
                if MEM_DEBUG:
                    print("gpuDirectRDMACapable not supported, disabling")
                if not _is_prop_supported(prop, location if self._support_localization else None):
                    raise ValueError("Failed to create physical memory allocation property")
        if MEM_DEBUG:
            print(
                f"allocation prop: "
                f"requestedHandleTypes={prop.requestedHandleTypes} "
                f"gpuDirectRDMACapable={prop.allocFlags.gpuDirectRDMACapable}"
            )
        self._prop = prop
        self._outstanding_handles = set()
        self._outstanding_handles_location = {}

    def _default_allocate(self) -> drv.CUmemGenericAllocationHandle:
        # Replace drv.cuMemCreate with the non-localized Rubin API when available.
        if MEM_DEBUG:
            print(f"_default_allocate: location={self._location} size={self._size}")
        return _unwrap(drv.cuMemCreate(self._size, self._prop, 0))

    def _localized_allocate(self) -> drv.CUmemGenericAllocationHandle:
        if MEM_DEBUG:
            print(f"_localized_allocate: location={self._location} size={self._size}")
        if _get_localization_mode() is LocalizationMode.MOCK:
            return _unwrap(drv.cuMemCreate(self._size, self._prop, 0))

        assert self._location is not None
        return _create_localized_allocation_handle(
            self._size, self._prop, _location_to_ugpu_id(self._location)
        )

    def allocate(self) -> drv.CUmemGenericAllocationHandle:
        if self._support_localization:
            handle = self._localized_allocate()
        else:
            handle = self._default_allocate()

        int_handle = int(handle)  # pyright: ignore
        assert (int_handle not in self._outstanding_handles) and int_handle != 0
        self._outstanding_handles.add(int_handle)
        self._outstanding_handles_location[int_handle] = self._location
        if MEM_DEBUG:
            print(
                f"allocated handle={int_handle:#x} location={self._location} "
                f"num_outstanding={len(self._outstanding_handles)}"
            )
        return handle

    def release(self, handle: drv.CUmemGenericAllocationHandle) -> None:
        if handle == drv.CUmemGenericAllocationHandle(0):
            return
        int_handle = int(handle)  # pyright: ignore
        assert int_handle in self._outstanding_handles
        location = self._outstanding_handles_location.pop(int_handle)
        self._outstanding_handles.remove(int_handle)
        if MEM_DEBUG:
            print(
                f"release: handle={int_handle:#x} location={location} "
                f"num_outstanding={len(self._outstanding_handles)}"
            )
        try:
            _unwrap(drv.cuMemRelease(handle))
        except:
            print(
                f"failed to release handle={int_handle:#x} "
                f"location={location} num_outstanding={len(self._outstanding_handles)}"
            )
            raise

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def size(self) -> int:
        return self._size


class PhysMem(ItemHolderWithSharedPool[drv.CUmemGenericAllocationHandle]):
    __slots__ = ()


# uGPU location enum values matching CUetblUGpuLocalizationMemoryLocation in
# pytorch-localization/etbl/ugpu_localization.h.
# typedef enum {
#     CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_ANY   = 0,
#     CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU0 = 1,
#     CU_UGPU_LOCALIZATION_LOCALIZED_MEMORY_UGPU1 = 2,
# } CUetblUGpuLocalizationMemoryLocation;
_UGPU_LOC_UGPU0: int = 1
_UGPU_LOC_UGPU1: int = 2
_UGPU_LOCATIONS: tuple[int, ...] = (_UGPU_LOC_UGPU0, _UGPU_LOC_UGPU1)


class _SinglePool(PooledFactoryBase[drv.CUmemGenericAllocationHandle, PhysMem]):
    """Pooled physical memory factory for one uGPU."""

    _Holder = PhysMem

    def __init__(self, raw_alloc: NativePhysMemAllocator) -> None:
        super().__init__(lambda: raw_alloc.allocate(), lambda handle: raw_alloc.release(handle))


class PooledPhysMemAllocator:
    """Pooled physical memory allocator supporting 1 or N uGPUs.

    Default construction creates a single pool (``num_ugpus == 1``) for
    non-localized hardware.  Use :meth:`create_localized` to create an
    N-pool allocator (``num_ugpus == N``) where each pool targets a
    specific uGPU via ``NativePhysMemAllocator(location=...)``.

    All public operations accept an optional ``ugpu_id`` (default 0) so
    that existing single-pool call sites remain unchanged.
    """

    __slots__ = ("device_id", "phys_mem_size", "_pools")
    device_id: int
    phys_mem_size: int
    _pools: list[_SinglePool]

    def __init__(self, phys_mem_size: int) -> None:
        """Create a single-pool (non-localized) allocator."""
        raw_alloc = NativePhysMemAllocator(phys_mem_size)
        self.device_id = raw_alloc.device_id
        self.phys_mem_size = phys_mem_size
        self._pools = [_SinglePool(raw_alloc)]

    @classmethod
    def create_localized(cls, phys_mem_size: int) -> "PooledPhysMemAllocator":
        """Create a two-pool allocator with per-uGPU memory localization."""
        obj = cls.__new__(cls)
        raw_allocs = [
            NativePhysMemAllocator(
                size=phys_mem_size,
                support_localization=True,
                location=loc,
            )
            for loc in _UGPU_LOCATIONS
        ]
        obj.device_id = raw_allocs[0].device_id
        obj.phys_mem_size = phys_mem_size
        obj._pools = [_SinglePool(ra) for ra in raw_allocs]
        return obj

    @property
    def num_ugpus(self) -> int:
        return len(self._pools)

    def create(self, ugpu_id: int = 0) -> PhysMem:
        """Allocate a physical memory chunk from the specified uGPU's pool."""
        assert 0 <= ugpu_id < len(self._pools), (
            f"ugpu_id {ugpu_id} out of range [0, {len(self._pools)})"
        )
        return self._pools[ugpu_id].create()

    def clear(self) -> None:
        """Release all cached physical memory handles from all uGPU pools."""
        for pool in self._pools:
            pool.clear()


# Large fixed VA offset separating uGPU regions within a multi-uGPU VirtMem.
# The first pool's mapped bytes can never realistically reach this value.
LOCALIZATION_OFFSET: Final[int] = 1 << 40  # 1 TiB


# Virtual memory
class VirtMem:
    """Virtual memory supporting 1 or N uGPUs within a single VA reservation.

    Each uGPU owns an independent region of the VA space with its own physical
    memory stack.  The number of uGPUs is derived from the
    ``PooledPhysMemAllocator`` passed at construction time
    (``allocator.num_ugpus``).

    VA layout (N uGPUs)::

        ugpu 0: [base,                              base + vm_sizes[0])
        ugpu 1: [base + 1 * LOCALIZATION_OFFSET,    base + 1 * LOCALIZATION_OFFSET + vm_sizes[1])
        ugpu k: [base + k * LOCALIZATION_OFFSET,    base + k * LOCALIZATION_OFFSET + vm_sizes[k])

    For a single uGPU (the default, N == 1) this simplifies to a contiguous
    region ``[base, base + vm_sizes[0])`` with no gap.

    All public operations accept an optional ``ugpu_id`` (default 0) so that
    existing single-uGPU call sites remain unchanged.
    """

    __slots__ = ("_vm_sizes", "_allocator", "_address", "_pm_stacks", "_access_desc")
    _vm_sizes: list[int]  # len == num_ugpus
    _allocator: PooledPhysMemAllocator
    _address: drv.CUdeviceptr
    _pm_stacks: list[list[PhysMem]]  # len == num_ugpus
    _access_desc: drv.CUmemAccessDesc

    def __init__(
        self,
        vm_sizes: int | list[int],
        phys_mem_allocator: PooledPhysMemAllocator,
        init_num_phys_mem: int | list[int] = 0,
    ) -> None:
        num_ugpus = phys_mem_allocator.num_ugpus
        # Normalise scalar args to per-uGPU lists.
        if isinstance(vm_sizes, int):
            vm_sizes = [vm_sizes]
        if isinstance(init_num_phys_mem, int):
            init_num_phys_mem = [init_num_phys_mem] * num_ugpus
        assert len(vm_sizes) == num_ugpus and len(init_num_phys_mem) == num_ugpus, (
            f"Expected {num_ugpus} elements (num_ugpus={num_ugpus}), "
            f"got vm_sizes={len(vm_sizes)}, init_num_phys_mem={len(init_num_phys_mem)}"
        )

        phys_mem_size = phys_mem_allocator.phys_mem_size
        for vs in vm_sizes:
            assert vs % phys_mem_size == 0
        if num_ugpus > 1:
            for vs in vm_sizes:
                assert vs <= LOCALIZATION_OFFSET

        self._allocator = phys_mem_allocator
        device_id = phys_mem_allocator.device_id
        # Reserve VA: single uGPU gets a tight reservation; multi-uGPU uses
        # LOCALIZATION_OFFSET gaps to place each region at offset k * LOCALIZATION_OFFSET.
        if num_ugpus == 1:
            self._address = _unwrap(drv.cuMemAddressReserve(vm_sizes[0], phys_mem_size, 0, 0))
        else:
            total_va = (num_ugpus - 1) * LOCALIZATION_OFFSET + vm_sizes[-1]
            self._address = _unwrap(drv.cuMemAddressReserve(total_va, 0, 0, 0))

        self._vm_sizes = list(vm_sizes)
        self._pm_stacks = [[] for _ in range(num_ugpus)]
        self._access_desc = drv.CUmemAccessDesc()
        self._access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access_desc.location.id = device_id
        self._access_desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        for uid, n in enumerate(init_num_phys_mem):
            if n:
                self.extend(n, ugpu_id=uid)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_ugpus(self) -> int:
        return len(self._pm_stacks)

    @property
    def phys_mem_size(self) -> int:
        return self._allocator.phys_mem_size

    @property
    def address(self) -> MemAddress:
        """Base address of the entire reservation (== ugpu_address(0))."""
        return MemAddress(int(self._address))

    def ugpu_address(self, ugpu_id: int = 0) -> MemAddress:
        """Base virtual address for the given uGPU's region."""
        return MemAddress(int(self._address) + ugpu_id * LOCALIZATION_OFFSET)

    def virtual_bytes(self, ugpu_id: int = 0) -> int:
        """VA region size for the given uGPU."""
        return self._vm_sizes[ugpu_id]

    def num_phys_mem(self, ugpu_id: int = 0) -> int:
        return len(self._pm_stacks[ugpu_id])

    def mapped_bytes(self, ugpu_id: int = 0) -> int:
        return self.phys_mem_size * self.num_phys_mem(ugpu_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if all(vs == 0 for vs in self._vm_sizes):
            return
        _unwrap(drv.cuCtxSynchronize())
        for uid in range(self.num_ugpus):
            while self._pm_stacks[uid]:
                self._pop(uid).close()
        if self.num_ugpus == 1:
            total_va = self._vm_sizes[0]
        else:
            total_va = (self.num_ugpus - 1) * LOCALIZATION_OFFSET + self._vm_sizes[-1]
        _unwrap(drv.cuMemAddressFree(self._address, total_va))
        self._address = drv.CUdeviceptr(0)
        self._vm_sizes = [0] * self.num_ugpus

    def __del__(self) -> None:
        self.destroy()

    # ------------------------------------------------------------------
    # Core operations (all take ugpu_id, defaulting to 0)
    # ------------------------------------------------------------------

    def _push(self, phy_mem: PhysMem, ugpu_id: int = 0) -> None:
        phys_mem_size = self.phys_mem_size
        pm_stack = self._pm_stacks[ugpu_id]
        assert phys_mem_size * (len(pm_stack) + 1) <= self._vm_sizes[ugpu_id]
        vm_ptr = drv.CUdeviceptr(self.ugpu_address(ugpu_id) + phys_mem_size * len(pm_stack))
        _unwrap(drv.cuMemMap(vm_ptr, phys_mem_size, 0, phy_mem.handle, 0))
        _unwrap(drv.cuMemSetAccess(vm_ptr, phys_mem_size, (self._access_desc,), 1))
        pm_stack.append(phy_mem)

    def _pop(self, ugpu_id: int = 0) -> PhysMem:
        pm_stack = self._pm_stacks[ugpu_id]
        assert pm_stack
        phys_mem_size = self.phys_mem_size
        vm_ptr = drv.CUdeviceptr(self.ugpu_address(ugpu_id) + phys_mem_size * (len(pm_stack) - 1))
        _unwrap(drv.cuMemUnmap(vm_ptr, phys_mem_size))
        return pm_stack.pop()

    def extend(self, num_phys_mem: int, ugpu_id: int = 0) -> None:
        assert 0 <= ugpu_id < self.num_ugpus, (
            f"ugpu_id {ugpu_id} out of range [0, {self.num_ugpus})"
        )
        pm_stack = self._pm_stacks[ugpu_id]
        old_num = len(pm_stack)
        try:
            for _ in range(num_phys_mem):
                self._push(self._allocator.create(ugpu_id), ugpu_id)
        except Exception:
            # Rollback to make realloc behave like normal realloc on OOM.
            while len(self._pm_stacks[ugpu_id]) > old_num:
                self._pop(ugpu_id).close()
            raise

    def shrink(self, num_phys_mem: int, ugpu_id: int = 0) -> None:
        assert 0 <= ugpu_id < self.num_ugpus, (
            f"ugpu_id {ugpu_id} out of range [0, {self.num_ugpus})"
        )
        _unwrap(drv.cuCtxSynchronize())
        for _ in range(num_phys_mem):
            self._pop(ugpu_id).close()

    # Different from normal realloc, this function never changes the pointer.
    def realloc(self, num_bytes: int, ugpu_id: int = 0) -> None:
        required_num_phys_mem = div_up(num_bytes, self.phys_mem_size)
        current = self.num_phys_mem(ugpu_id)
        if required_num_phys_mem > current:
            self.extend(required_num_phys_mem - current, ugpu_id)
        elif required_num_phys_mem < current:
            self.shrink(current - required_num_phys_mem, ugpu_id)
