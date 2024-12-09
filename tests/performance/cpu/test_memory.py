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
from typing import Any, NoReturn

import numpy as np
import pytest
import torch

from src.core.performance.cpu.memory import MemoryManager

# Test configurations - reduced sizes for safety
POOL_SIZES = [1024, 4096]  # Removed 16384 KB
BLOCK_SIZES = [32, 128]  # Removed 512 KB
ALLOCATION_PATTERNS = ["sequential", "random", "interleaved"]
CACHE_SIZES = [32, 256]  # Removed 1024 KB

# Resource limits
MAX_MEMORY_GB = 4  # Maximum memory limit in GB
MAX_TIME_SECONDS = 30  # Maximum time limit per test in seconds


@contextmanager
def resource_guard() -> Generator[None, None, None]:
    """Set up resource limits for memory and time."""
    # Set memory limit
    memory_limit = MAX_MEMORY_GB * 1024 * 1024 * 1024  # Convert to bytes
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))

    # Set up timeout
    def timeout_handler(_signum: int, _frame: Any) -> NoReturn:
        msg = f"Test exceeded {MAX_TIME_SECONDS} seconds time limit"
        raise TimeoutError(msg)

    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_TIME_SECONDS)

    try:
        yield
    finally:
        # Reset signal handler and alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # Reset memory limit
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def memory_manager():
    """Create a MemoryManager instance for testing."""
    return MemoryManager(enable_monitoring=True)


def generate_allocation_pattern(pattern: str, size: int, block_size: int) -> list[int]:
    """Generate memory allocation patterns for testing."""
    num_blocks = size // block_size
    if pattern == "sequential":
        return list(range(num_blocks))
    if pattern == "random":
        blocks = list(range(num_blocks))
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        rng.shuffle(blocks)
        return blocks
    # interleaved
    return [i // 2 if i % 2 == 0 else num_blocks - 1 - i // 2 for i in range(num_blocks)]


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("pool_size", POOL_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_memory_pool_efficiency(memory_manager: MemoryManager, pool_size: int, block_size: int):
    """Test memory pool allocation and deallocation efficiency."""
    with resource_guard():
        # Initialize pool
        pool = memory_manager.create_pool(pool_size)

        # Allocate blocks
        blocks = []
        for _i in range(pool_size // block_size):
            block = memory_manager.allocate(block_size, pool)
            blocks.append(block)

        # Get metrics after allocation
        metrics = memory_manager.get_metrics()

        # Verify allocation efficiency
        assert metrics.allocated_memory <= pool_size
        assert metrics.fragmentation < 0.2  # Less than 20% fragmentation

        # Deallocate in reverse order
        for block in reversed(blocks):
            memory_manager.deallocate(block, pool)

        # Get metrics after deallocation
        metrics = memory_manager.get_metrics()

        # Verify cleanup
        assert metrics.allocated_memory == 0
        assert metrics.fragmentation == 0.0


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("pattern", ALLOCATION_PATTERNS)
@pytest.mark.parametrize("pool_size", POOL_SIZES)
def test_allocation_pattern_impact(memory_manager: MemoryManager, pattern: str, pool_size: int):
    """Test impact of different allocation patterns on memory performance."""
    with resource_guard():
        block_size = 32  # Fixed block size for pattern testing
        pool = memory_manager.create_pool(pool_size)

        # Generate allocation pattern
        allocation_sequence = generate_allocation_pattern(pattern, pool_size, block_size)

        # Allocate blocks according to pattern
        blocks = [None] * len(allocation_sequence)
        for _i, block_index in enumerate(allocation_sequence):
            blocks[block_index] = memory_manager.allocate(block_size, pool)

        # Get metrics
        metrics = memory_manager.get_metrics()

        # Pattern-specific assertions
        if pattern == "sequential":
            assert metrics.fragmentation < 0.1  # Sequential should have minimal fragmentation
        elif pattern == "random":
            # Random might have more fragmentation but should still be manageable
            assert metrics.fragmentation < 0.3
        else:  # interleaved
            # Interleaved pattern tests worst-case fragmentation
            assert metrics.cache_misses < metrics.total_accesses * 0.3


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("cache_size", CACHE_SIZES)
def test_cache_utilization(memory_manager: MemoryManager, cache_size: int):
    """Test memory cache utilization and hit rates."""
    with resource_guard():
        pool_size = cache_size * 4  # Pool size relative to cache size
        pool = memory_manager.create_pool(pool_size)

        # Sequential access pattern
        block_size = 32
        num_blocks = pool_size // block_size
        blocks = []

        # Allocate and access blocks sequentially
        start_time = memory_manager.get_time()
        for _ in range(num_blocks):
            block = memory_manager.allocate(block_size, pool)
            blocks.append(block)
        sequential_time = memory_manager.get_time() - start_time

        # Get cache metrics
        metrics = memory_manager.get_metrics()

        # Cache utilization assertions
        assert metrics.cache_hits > 0
        assert metrics.cache_misses > 0
        assert sequential_time < metrics.baseline_access_time * 1.5


@pytest.mark.benchmark(min_rounds=5)
def test_memory_bandwidth():
    """Test memory bandwidth utilization."""
    with resource_guard():
        # Test different transfer sizes
        sizes = [1024, 4096]  # Reduced sizes
        bandwidths = []

        for size in sizes:
            # Allocate source and destination buffers
            source = torch.randn(size)
            dest = torch.zeros_like(source)

            # Measure transfer time
            start_time = (
                torch.cuda.Event(enable_timing=True)
                if torch.cuda.is_available()
                else time.perf_counter()
            )
            dest.copy_(source)
            end_time = (
                torch.cuda.Event(enable_timing=True)
                if torch.cuda.is_available()
                else time.perf_counter()
            )

            # Calculate bandwidth
            if torch.cuda.is_available():
                start_time.record()
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                elapsed_time = end_time - start_time

            bandwidth = (size * 4) / (elapsed_time * 1024 * 1024)  # MB/s
            bandwidths.append(bandwidth)

        # Bandwidth scaling assertions
        assert bandwidths[0] > 0  # Small transfers should have non-zero bandwidth
        assert bandwidths[1] > bandwidths[0] * 0.8  # Medium transfers should scale well


@pytest.mark.benchmark(min_rounds=5)
def test_resource_cleanup(memory_manager: MemoryManager):
    """Test proper cleanup of memory resources."""
    with resource_guard():
        pool_size = 1024  # KB
        block_size = 32  # KB
        num_blocks = pool_size // block_size

        # Create pool and allocate all blocks
        pool = memory_manager.create_pool(pool_size)
        blocks = []
        for _ in range(num_blocks):
            block = memory_manager.allocate(block_size, pool)
            blocks.append(block)

        # Delete half the blocks
        for block in blocks[: num_blocks // 2]:
            memory_manager.deallocate(block, pool)

        # Get metrics
        metrics = memory_manager.get_metrics()

        # Verify partial cleanup
        assert metrics.allocated_memory == pool_size // 2
        assert metrics.active_pools == 1

        # Cleanup remaining blocks
        for block in blocks[num_blocks // 2 :]:
            memory_manager.deallocate(block, pool)

        # Delete pool
        memory_manager.delete_pool(pool)

        # Verify complete cleanup
        metrics = memory_manager.get_metrics()
        assert metrics.allocated_memory == 0
        assert metrics.active_pools == 0
