"""Performance tests for CPU memory management.

This module tests the memory management characteristics of the
Adaptive Attention Tiling system, focusing on:
1. Memory pool efficiency
2. Cache utilization
3. Memory bandwidth optimization
4. Resource allocation patterns
"""

import numpy as np
import pytest
import torch

from src.core.performance.cpu.memory import MemoryManager

# Test configurations
POOL_SIZES = [1024, 4096, 16384]  # KB
BLOCK_SIZES = [32, 128, 512]  # KB
ALLOCATION_PATTERNS = ["sequential", "random", "interleaved"]
CACHE_SIZES = [32, 256, 1024]  # KB


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
        rng = np.random.default_rng()
        rng.shuffle(blocks)
        return blocks
    # interleaved
    return [i // 2 if i % 2 == 0 else num_blocks - 1 - i // 2 for i in range(num_blocks)]


@pytest.mark.parametrize("pool_size", POOL_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_memory_pool_efficiency(memory_manager: MemoryManager, pool_size: int, block_size: int):
    """Test memory pool allocation and deallocation efficiency."""
    # Initialize pool
    pool = memory_manager.create_pool(pool_size)

    # Pre-calculate maximum blocks that can fit
    max_blocks = pool_size // block_size
    allocations = []

    # Allocate blocks up to maximum
    for _ in range(max_blocks):
        block = memory_manager.allocate(block_size, pool)
        allocations.append(block)

    # Verify pool utilization
    metrics = memory_manager.get_metrics()
    assert metrics.pool_utilization > 0.9  # At least 90% utilization
    assert metrics.fragmentation < 0.1  # Less than 10% fragmentation

    # Test deallocation
    for block in allocations:
        memory_manager.deallocate(block)

    metrics = memory_manager.get_metrics()
    assert metrics.available_memory == pool_size
    assert metrics.fragmentation == 0.0


@pytest.mark.parametrize("pattern", ALLOCATION_PATTERNS)
@pytest.mark.parametrize("pool_size", POOL_SIZES)
def test_allocation_pattern_impact(memory_manager: MemoryManager, pattern: str, pool_size: int):
    """Test impact of different allocation patterns on memory performance."""
    block_size = 64  # KB
    pool = memory_manager.create_pool(pool_size)

    # Generate allocation sequence
    allocation_sequence = generate_allocation_pattern(pattern, pool_size, block_size)

    # Perform allocations
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for block_idx in allocation_sequence:
        offset = block_idx * block_size
        _ = memory_manager.allocate(block_size, pool, preferred_offset=offset)
    end_time.record()

    torch.cuda.synchronize()
    allocation_time = start_time.elapsed_time(end_time)

    metrics = memory_manager.get_metrics()

    # Performance assertions based on pattern
    if pattern == "sequential":
        assert allocation_time < metrics.baseline_allocation_time * 1.2
    elif pattern == "random":
        # Random access might be slower due to fragmentation
        assert metrics.fragmentation < 0.2
    else:  # interleaved
        assert metrics.cache_misses < metrics.total_accesses * 0.3


@pytest.mark.parametrize("cache_size", CACHE_SIZES)
def test_cache_utilization(memory_manager: MemoryManager, cache_size: int):
    """Test memory cache utilization and hit rates."""
    # Configure cache
    memory_manager.configure_cache(cache_size)

    # Perform cache-intensive operations
    data_size = cache_size * 2  # Ensure some cache misses
    data = torch.randn(data_size // 4)  # 4 bytes per float

    # Sequential access
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(10):  # Multiple passes
        _ = torch.sum(data)
    end_time.record()

    torch.cuda.synchronize()
    sequential_time = start_time.elapsed_time(end_time)

    metrics = memory_manager.get_metrics()

    # Cache performance assertions
    assert metrics.cache_hit_rate > 0.7  # At least 70% hit rate
    assert sequential_time < metrics.baseline_access_time * 1.5


def test_memory_bandwidth():
    """Test memory bandwidth utilization."""
    # Test different transfer sizes
    sizes = [1024, 4096, 16384]  # KB
    bandwidths = []

    for size in sizes:
        data = torch.randn(size * 256)  # Create data block

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        for _ in range(100):  # Multiple transfers
            # Simulate memory transfer
            _ = torch.clone(data)
        end_time.record()

        torch.cuda.synchronize()
        transfer_time = start_time.elapsed_time(end_time)

        bandwidth = (size * 100) / (transfer_time / 1000)  # MB/s
        bandwidths.append(bandwidth)

    # Bandwidth should increase with transfer size up to a point
    assert bandwidths[1] > bandwidths[0]  # Medium > Small
    assert bandwidths[1] > bandwidths[2] * 0.8  # Medium ~= Large


def test_resource_cleanup(memory_manager: MemoryManager):
    """Test proper cleanup of memory resources."""
    pool_size = 1024  # KB
    block_size = 64  # KB

    # Create pool and allocate blocks
    pool = memory_manager.create_pool(pool_size)
    blocks = []

    for _ in range(pool_size // block_size // 2):  # Fill half the pool
        block = memory_manager.allocate(block_size, pool)
        blocks.append(block)

    # Delete some blocks
    for block in blocks[: len(blocks) // 2]:
        memory_manager.deallocate(block)

    metrics = memory_manager.get_metrics()

    # Verify cleanup
    assert metrics.leaked_memory == 0
    assert metrics.available_memory == pool_size - (len(blocks) // 2 * block_size)

    # Cleanup remaining blocks
    for block in blocks[len(blocks) // 2 :]:
        memory_manager.deallocate(block)

    metrics = memory_manager.get_metrics()
    assert metrics.available_memory == pool_size
    assert metrics.active_allocations == 0
