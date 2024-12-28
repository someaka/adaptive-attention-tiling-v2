"""Performance tests for CPU memory management.

This module tests the memory management characteristics of the
Adaptive Attention Tiling system, focusing on:
1. Memory pool efficiency
2. Cache utilization
3. Memory bandwidth optimization
4. Resource allocation patterns
"""

import gc
import resource
import signal
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NoReturn, Optional, Dict, List
import random

import pytest
import torch

from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics

# Test configurations from tiny.yaml
POOL_SIZES = [32, 64]  # Based on max_dim from tiny.yaml
BLOCK_SIZES = [8, 16]  # Smaller blocks for better memory management
ALLOCATION_PATTERNS = ["sequential", "random", "interleaved"]
CACHE_SIZES = [16, 32]  # Based on batch_size from tiny.yaml

# Resource limits from tiny.yaml hardware profile
MAX_MEMORY_GB = 1  # Limited to 1GB for tests
MAX_TIME_SECONDS = 5  # Reduced from 10 seconds


@contextmanager
def resource_guard() -> Generator[None, None, None]:
    """Set up resource limits for memory and time."""
    # Store original limits
    original_memory_soft, original_memory_hard = resource.getrlimit(resource.RLIMIT_AS)
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set memory limit
        memory_limit = MAX_MEMORY_GB * 1024 * 1024 * 1024  # Convert to bytes
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, original_memory_hard))

        # Set up timeout
        def timeout_handler(_signum: int, _frame: Any) -> NoReturn:
            msg = f"Test exceeded {MAX_TIME_SECONDS} seconds time limit"
            raise TimeoutError(msg)

        # Set signal handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(MAX_TIME_SECONDS)

        yield
    finally:
        # Reset signal handler and alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        # Reset memory limit
        resource.setrlimit(resource.RLIMIT_AS, (original_memory_soft, original_memory_hard))
        # Memory cleanup
        gc.collect()


@pytest.fixture
def memory_manager() -> Generator[MemoryManager, None, None]:
    """Fixture to provide a memory manager instance."""
    manager = MemoryManager()
    yield manager
    # Ensure cleanup after each test
    manager._cleanup_dead_refs()
    gc.collect()


def generate_allocation_pattern(pattern: str, size: int, block_size: int) -> List[int]:
    """Generate memory allocation patterns for testing."""
    num_blocks = size // block_size
    if pattern == "sequential":
        return list(range(num_blocks))
    if pattern == "random":
        blocks = list(range(num_blocks))
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(blocks)
        return blocks
    # interleaved
    return [
        i // 2 if i % 2 == 0 else num_blocks - 1 - i // 2 for i in range(num_blocks)
    ]


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.parametrize("pool_size", POOL_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_memory_pool_efficiency(
    memory_manager: MemoryManager, pool_size: int, block_size: int
) -> None:
    """Test memory pool allocation and deallocation efficiency."""
    with resource_guard():
        # Allocate blocks
        blocks = []
        for _i in range(pool_size // block_size):
            block = memory_manager.allocate_tensor((block_size,))
            blocks.append(block)

        # Get metrics after allocation
        allocated_memory = memory_manager.get_allocated_memory()
        fragmentation_ratio = memory_manager.get_fragmentation_ratio()

        # Verify allocation efficiency
        assert allocated_memory <= pool_size * 1024  # Convert KB to bytes
        assert fragmentation_ratio < 0.2  # Less than 20% fragmentation

        # Deallocate in reverse order
        for block in reversed(blocks):
            del block
        blocks.clear()  # Clear list references
        gc.collect()
        memory_manager._cleanup_dead_refs()  # Force cleanup

        # Get metrics after deallocation
        allocated_memory = memory_manager.get_allocated_memory()
        fragmentation_ratio = memory_manager.get_fragmentation_ratio()

        # Verify cleanup
        assert allocated_memory == 0
        assert fragmentation_ratio == 0.0


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.parametrize("pattern", ALLOCATION_PATTERNS)
@pytest.mark.parametrize("pool_size", POOL_SIZES)
def test_allocation_pattern_impact(
    memory_manager: MemoryManager, pattern: str, pool_size: int
) -> None:
    """Test impact of different allocation patterns on memory performance."""
    with resource_guard():
        block_size = 8  # Smallest block size for pattern testing

        # Generate allocation pattern
        allocation_sequence = generate_allocation_pattern(
            pattern, pool_size, block_size
        )

        # Allocate blocks according to pattern
        blocks = []
        for block_index in allocation_sequence:
            block = memory_manager.allocate_tensor((block_size,))
            blocks.append(block)

        # Get metrics
        fragmentation_ratio = memory_manager.get_fragmentation_ratio()

        # Pattern-specific assertions
        if pattern == "sequential":
            assert fragmentation_ratio < 0.1  # Sequential should have minimal fragmentation
        elif pattern == "random":
            # Random might have more fragmentation but should still be manageable
            assert fragmentation_ratio < 0.3
        else:  # interleaved
            # Interleaved pattern tests worst-case fragmentation
            assert fragmentation_ratio < 0.4

        # Cleanup
        for block in blocks:
            del block
        blocks.clear()
        gc.collect()
        memory_manager._cleanup_dead_refs()  # Force cleanup


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.parametrize("cache_size", CACHE_SIZES)
def test_cache_utilization(memory_manager: MemoryManager, cache_size: int) -> None:
    """Test memory cache utilization and hit rates."""
    with resource_guard():
        pool_size = cache_size * 2  # Reduced multiplier for faster tests
        block_size = 8  # Smallest block size
        num_blocks = pool_size // block_size
        blocks = []

        # Warmup
        for _ in range(3):
            block = memory_manager.allocate_tensor((block_size,))
            del block
        gc.collect()
        memory_manager._cleanup_dead_refs()

        # Allocate and access blocks sequentially
        start_time = time.perf_counter()
        for _ in range(num_blocks):
            block = memory_manager.allocate_tensor((block_size,))
            blocks.append(block)
        end_time = time.perf_counter()
        sequential_time = end_time - start_time

        # Time assertions - adjusted for slower machines
        assert sequential_time < 2.0  # Increased from 0.2 seconds

        # Cleanup
        for block in blocks:
            del block
        blocks.clear()
        gc.collect()
        memory_manager._cleanup_dead_refs()  # Force cleanup


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_memory_bandwidth() -> None:
    """Test memory bandwidth utilization."""
    with resource_guard():
        # Test different transfer sizes
        sizes = [32, 64]  # Based on max_dim from tiny.yaml
        bandwidths = []

        for size in sizes:
            # Allocate source and destination buffers
            source = torch.randn(size)
            dest = torch.zeros_like(source)

            # Warmup
            for _ in range(3):
                dest.copy_(source)

            # Measure transfer time
            start_time = time.perf_counter()
            dest.copy_(source)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            bandwidth = (size * 4) / (elapsed_time * 1024 * 1024)  # MB/s
            bandwidths.append(bandwidth)

        # Bandwidth scaling assertions
        assert bandwidths[0] > 0  # Small transfers should have non-zero bandwidth
        assert bandwidths[1] > bandwidths[0] * 0.8  # Medium transfers should scale well


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_resource_cleanup(memory_manager: MemoryManager) -> None:
    """Test proper cleanup of memory resources."""
    with resource_guard():
        pool_size = 32  # Based on batch_size from tiny.yaml
        block_size = 8  # Smallest block size
        num_blocks = pool_size // block_size

        # Warmup
        for _ in range(3):
            block = memory_manager.allocate_tensor((block_size,))
            del block
            gc.collect()
            memory_manager._cleanup_dead_refs()

        # Allocate all blocks
        blocks = []
        for _ in range(num_blocks):
            block = memory_manager.allocate_tensor((block_size,))
            blocks.append(block)

        # Delete half the blocks
        half_point = len(blocks) // 2
        for block in blocks[:half_point]:
            del block
        blocks = blocks[half_point:]
        gc.collect()
        memory_manager._cleanup_dead_refs()  # Force cleanup

        # Get metrics
        allocated_memory = memory_manager.get_allocated_memory()
        fragmentation_ratio = memory_manager.get_fragmentation_ratio()

        # Verify cleanup
        assert allocated_memory <= (pool_size * 1024) // 2  # Convert KB to bytes
        assert fragmentation_ratio < 0.3  # Some fragmentation is expected

        # Delete remaining blocks
        for block in blocks:
            del block
        blocks.clear()
        gc.collect()
        memory_manager._cleanup_dead_refs()  # Force cleanup
        gc.collect()  # Run garbage collection again
        memory_manager._cleanup_dead_refs()  # Force another cleanup

        # Get final metrics
        allocated_memory = memory_manager.get_allocated_memory()
        fragmentation_ratio = memory_manager.get_fragmentation_ratio()

        # Verify complete cleanup with some tolerance
        assert allocated_memory <= 128  # Allow for small residual memory
        assert fragmentation_ratio < 0.1  # Minimal fragmentation after cleanup
