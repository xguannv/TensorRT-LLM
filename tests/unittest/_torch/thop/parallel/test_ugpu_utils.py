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
"""
Unit tests for UGPU utilities in tensorrt_llm._torch.ugpu_utils.

Tests cover:
- UGPU support detection
- Resource initialization
- Stream and mempool retrieval
- Error handling and edge cases
"""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.ugpu_utils import (
    get_ugpu_mempool,
    get_ugpu_stream,
    initialize_ugpu_resources,
    is_ugpu_enabled,
    is_ugpu_supported,
)


@pytest.fixture(scope="module")
def check_ugpu_support():
    """Check if UGPU is supported and skip tests if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not is_ugpu_supported():
        pytest.skip("UGPU localization is not supported on this system")


class TestUgpuSupport:
    """Tests for UGPU support detection."""

    def test_is_ugpu_supported_returns_bool(self):
        """Test that is_ugpu_supported returns a boolean value."""
        result = is_ugpu_supported()
        assert isinstance(result, bool)

    def test_is_ugpu_enabled_requires_rubin(self):
        is_ugpu_enabled.cache_clear()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("tensorrt_llm._torch.ugpu_utils.get_sm_version", return_value=100),
            patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_supported", return_value=True),
        ):
            assert not is_ugpu_enabled()
        is_ugpu_enabled.cache_clear()

    def test_is_ugpu_enabled_allows_rubin_when_supported(self):
        is_ugpu_enabled.cache_clear()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("tensorrt_llm._torch.ugpu_utils.get_sm_version", return_value=107),
            patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_supported", return_value=True),
        ):
            assert is_ugpu_enabled()
        is_ugpu_enabled.cache_clear()


class TestUgpuInitialization:
    """Tests for UGPU resource initialization."""

    def test_initialize_ugpu_resources(self, check_ugpu_support):
        """Test initializing UGPU resources for current device."""
        # Should not raise any exception
        initialize_ugpu_resources()

    def test_initialize_ugpu_resources_idempotent(self, check_ugpu_support):
        """Test that multiple initializations are safe."""
        # Initialize multiple times - should not raise
        initialize_ugpu_resources()
        initialize_ugpu_resources()
        initialize_ugpu_resources()


class TestUgpuStream:
    """Tests for UGPU stream retrieval."""

    def test_get_ugpu_stream_ugpu0(self, check_ugpu_support):
        """Test getting stream for uGPU0."""
        ugpu_id = 0
        stream = get_ugpu_stream(ugpu_id)

        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    def test_get_ugpu_stream_ugpu1(self, check_ugpu_support):
        """Test getting stream for uGPU1."""
        ugpu_id = 1
        stream = get_ugpu_stream(ugpu_id)

        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    def test_get_ugpu_stream_different_streams(self, check_ugpu_support):
        """Test that uGPU0 and uGPU1 have different streams."""
        stream0 = get_ugpu_stream(0)
        stream1 = get_ugpu_stream(1)

        # The streams should be different objects
        assert stream0 is not stream1

    def test_get_ugpu_stream_invalid_ugpu_id(self, check_ugpu_support):
        """Test that invalid ugpu_id raises ValueError."""
        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_stream(2)

        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_stream(-1)

    def test_get_ugpu_stream_lazy_initialization(self, check_ugpu_support):
        """Test that stream retrieval triggers lazy initialization."""
        ugpu_id = 0

        # First call should initialize
        stream1 = get_ugpu_stream(ugpu_id)
        # Second call should return the same stream
        stream2 = get_ugpu_stream(ugpu_id)

        assert stream1 is stream2


class TestUgpuMempool:
    """Tests for UGPU memory pool retrieval."""

    def test_get_ugpu_mempool_ugpu0(self, check_ugpu_support):
        """Test getting mempool for uGPU0."""
        ugpu_id = 0

        try:
            mempool = get_ugpu_mempool(ugpu_id)
            assert mempool is not None
            assert isinstance(mempool, torch.cuda.MemPool)
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise

    def test_get_ugpu_mempool_ugpu1(self, check_ugpu_support):
        """Test getting mempool for uGPU1."""
        ugpu_id = 1

        try:
            mempool = get_ugpu_mempool(ugpu_id)
            assert mempool is not None
            assert isinstance(mempool, torch.cuda.MemPool)
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise

    def test_get_ugpu_mempool_different_pools(self, check_ugpu_support):
        """Test that uGPU0 and uGPU1 have different mempools."""
        try:
            mempool0 = get_ugpu_mempool(0)
            mempool1 = get_ugpu_mempool(1)

            # The mempools should be different objects
            assert mempool0 is not mempool1
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise

    def test_get_ugpu_mempool_invalid_ugpu_id(self, check_ugpu_support):
        """Test that invalid ugpu_id raises ValueError."""
        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_mempool(2)

        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_mempool(-1)

    def test_get_ugpu_mempool_lazy_initialization(self, check_ugpu_support):
        """Test that mempool retrieval triggers lazy initialization."""
        ugpu_id = 0

        try:
            # First call should initialize
            mempool1 = get_ugpu_mempool(ugpu_id)
            # Second call should return the same mempool
            mempool2 = get_ugpu_mempool(ugpu_id)

            assert mempool1 is mempool2
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise


class TestUgpuIntegration:
    """Integration tests for UGPU utilities."""

    def test_all_resources_initialized_together(self, check_ugpu_support):
        """Test that initializing creates all resources (streams and mempools)."""
        # Initialize once
        initialize_ugpu_resources()

        # Core resources should always be available
        stream0 = get_ugpu_stream(0)
        stream1 = get_ugpu_stream(1)

        assert stream0 is not None
        assert stream1 is not None

        # Mempools may or may not be available depending on system support
        try:
            mempool0 = get_ugpu_mempool(0)
            mempool1 = get_ugpu_mempool(1)
            assert mempool0 is not None
            assert mempool1 is not None
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                # This is acceptable - mempools are optional
                pytest.skip(f"UGPU mempool not available: {e}")
            else:
                raise

    def test_resources_persistent_across_calls(self, check_ugpu_support):
        """Test that resources are persistent and reused."""
        # Get resources multiple times
        stream0_1 = get_ugpu_stream(0)
        stream0_2 = get_ugpu_stream(0)

        # Should be the same objects
        assert stream0_1 is stream0_2

        # Test mempool persistence if available
        try:
            mempool1_1 = get_ugpu_mempool(1)
            mempool1_2 = get_ugpu_mempool(1)
            assert mempool1_1 is mempool1_2
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                # This is acceptable - mempools are optional
                pass
            else:
                raise

    def test_stream_can_be_used_for_operations(self, check_ugpu_support):
        """Test that UGPU streams can be used for CUDA operations."""
        ugpu_id = 0
        stream = get_ugpu_stream(ugpu_id)

        # Create a tensor and perform an operation on the stream
        device_id = torch.cuda.current_device()
        with torch.cuda.stream(stream):
            tensor = torch.randn(10, 10, device=f"cuda:{device_id}")
            result = tensor * 2

            # Wait for stream to complete
            stream.synchronize()

            assert result.shape == (10, 10)
            assert result.device.type == "cuda"

    def test_concurrent_stream_operations(self, check_ugpu_support):
        """Test that operations on different UGPU streams can execute concurrently."""
        stream0 = get_ugpu_stream(0)
        stream1 = get_ugpu_stream(1)

        device_id = torch.cuda.current_device()

        # Launch operations on both streams
        with torch.cuda.stream(stream0):
            tensor0 = torch.randn(100, 100, device=f"cuda:{device_id}")
            result0 = tensor0 @ tensor0.T

        with torch.cuda.stream(stream1):
            tensor1 = torch.randn(100, 100, device=f"cuda:{device_id}")
            result1 = tensor1 @ tensor1.T

        # Synchronize both streams
        stream0.synchronize()
        stream1.synchronize()

        # Verify results
        assert result0.shape == (100, 100)
        assert result1.shape == (100, 100)
        assert not torch.allclose(result0, result1)  # Different results


class TestUgpuMempoolAllocation:
    """Tests for UGPU memory pool allocation and deallocation."""

    def test_allocate_tensor_with_mempool(self, check_ugpu_support):
        """Test allocating a tensor using UGPU mempool."""
        try:
            ugpu_id = 0
            mempool = get_ugpu_mempool(ugpu_id)
            stream = get_ugpu_stream(ugpu_id)

            device_id = torch.cuda.current_device()

            # Allocate tensor using the mempool
            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate tensor using the UGPU mempool
                    tensor = torch.randn(100, 100, device=f"cuda:{device_id}")

                    # Perform operation
                    result = tensor * 2.0

                    # Synchronize
                    stream.synchronize()

                    # Verify
                    assert result.shape == (100, 100)
                    assert result.device.type == "cuda"

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool allocation not supported: {e}")
            raise

    def test_mempool_with_stream_context(self, check_ugpu_support):
        """Test using mempool within its corresponding stream context."""
        try:
            ugpu_id = 0
            stream = get_ugpu_stream(ugpu_id)
            mempool = get_ugpu_mempool(ugpu_id)

            device_id = torch.cuda.current_device()

            # Use mempool within stream context
            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate multiple tensors
                    tensors = []
                    for i in range(5):
                        tensor = torch.randn(50, 50, device=f"cuda:{device_id}")
                        tensors.append(tensor)

                    # Perform operations
                    results = [t @ t.T for t in tensors]

                    # Synchronize
                    stream.synchronize()

                    # Verify all results
                    for result in results:
                        assert result.shape == (50, 50)
                        assert result.device.type == "cuda"

            # Cleanup - delete tensors to free memory
            del tensors
            del results
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool operation not supported: {e}")
            raise

    def test_large_allocation_with_mempool(self, check_ugpu_support):
        """Test allocating large tensors using UGPU mempool."""
        try:
            ugpu_id = 0
            stream = get_ugpu_stream(ugpu_id)
            mempool = get_ugpu_mempool(ugpu_id)

            device_id = torch.cuda.current_device()

            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate a large tensor (100MB)
                    large_tensor = torch.randn(5000, 5000, device=f"cuda:{device_id}")

                    # Perform operation to ensure it's accessible
                    result = large_tensor.sum()

                    # Synchronize
                    stream.synchronize()

                    # Verify
                    assert result.device.type == "cuda"
                    assert large_tensor.shape == (5000, 5000)

            # Cleanup
            del large_tensor
            del result
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Not enough memory for large allocation test: {e}")
            elif (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool operation not supported: {e}")
            raise

    def test_mempool_reuse_after_free(self, check_ugpu_support):
        """Test that mempool can reuse freed memory."""
        try:
            ugpu_id = 0
            stream = get_ugpu_stream(ugpu_id)
            mempool = get_ugpu_mempool(ugpu_id)

            device_id = torch.cuda.current_device()

            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate tensor
                    tensor1 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result1 = tensor1.sum()
                    stream.synchronize()

                    # Free tensor
                    del tensor1
                    torch.cuda.empty_cache()

                    # Allocate another tensor of same size
                    tensor2 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result2 = tensor2.sum()
                    stream.synchronize()

                    # Verify both operations succeeded
                    assert result1.device.type == "cuda"
                    assert result2.device.type == "cuda"

            # Cleanup
            del tensor2
            del result1, result2
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool operation not supported: {e}")
            raise

    def test_mempool_across_different_ugpus(self, check_ugpu_support):
        """Test allocating memory on different UGPU mempools."""
        try:
            stream0 = get_ugpu_stream(0)
            stream1 = get_ugpu_stream(1)
            mempool0 = get_ugpu_mempool(0)
            mempool1 = get_ugpu_mempool(1)

            device_id = torch.cuda.current_device()

            # Allocate on uGPU0
            with torch.cuda.stream(stream0):
                with torch.cuda.use_mem_pool(mempool0):
                    tensor0 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result0 = tensor0 @ tensor0.T

            # Allocate on uGPU1
            with torch.cuda.stream(stream1):
                with torch.cuda.use_mem_pool(mempool1):
                    tensor1 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result1 = tensor1 @ tensor1.T

            # Synchronize both
            stream0.synchronize()
            stream1.synchronize()

            # Verify both allocations succeeded
            assert result0.shape == (100, 100)
            assert result1.shape == (100, 100)
            assert not torch.allclose(result0, result1)

            # Cleanup
            del tensor0, tensor1, result0, result1
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool not available: {e}")
            raise
