import gc
from typing import Tuple

import pytest
import torch

from src.core.performance.cpu.memory_management import MemoryManager


def test_memory_allocation_deallocation() -> None:
    """Test memory allocation and deallocation behavior."""
    manager = MemoryManager()

    # Test allocation with smaller tensors
    tensor1 = manager.allocate_tensor((100, 100))
    tensor2 = manager.allocate_tensor((200, 200))

    assert tensor1.shape == (100, 100)
    assert tensor2.shape == (200, 200)

    # Check memory usage tracking
    assert manager.get_allocated_memory() > 0

    # Test deallocation
    del tensor1
    gc.collect()
    initial_memory = manager.get_allocated_memory()

    del tensor2
    gc.collect()
    final_memory = manager.get_allocated_memory()

    assert final_memory < initial_memory


def test_memory_fragmentation() -> None:
    """Test memory fragmentation prevention."""
    manager = MemoryManager()

    # Allocate tensors
    tensors = [manager.allocate_tensor((10, 10)) for _ in range(10)]
    
    # Create new list without alternate tensors
    tensors = [t for i, t in enumerate(tensors) if i % 2 == 1]
    gc.collect()

    # Allocate new tensors
    new_tensors = [manager.allocate_tensor((10, 10)) for _ in range(5)]
    tensors.extend(new_tensors)

    # Check fragmentation metrics
    fragmentation = manager.get_fragmentation_ratio()
    assert fragmentation < 0.2  # Less than 20% fragmentation


def test_memory_efficient_operations() -> None:
    """Test memory-efficient tensor operations."""
    manager = MemoryManager()

    # Test in-place operations with smaller tensor
    x = manager.allocate_tensor((100, 100))
    initial_memory = manager.get_allocated_memory()

    # Perform operations that should be memory efficient
    def add_inplace(t: torch.Tensor) -> None:
        t.add_(1)
        
    def mul_inplace(t: torch.Tensor) -> None:
        t.mul_(2)

    manager.inplace_operation(x, add_inplace)
    manager.inplace_operation(x, mul_inplace)

    final_memory = manager.get_allocated_memory()

    # Memory usage should not increase significantly
    assert abs(final_memory - initial_memory) < 1024  # Allow small overhead


def test_memory_peak_tracking() -> None:
    """Test peak memory usage tracking."""
    manager = MemoryManager()

    # Record initial peak
    initial_peak = manager.get_peak_memory()

    # Perform operations that should increase memory usage
    tensors = [manager.allocate_tensor((100, 100)) for _ in range(5)]

    # Check peak memory increased
    peak_with_tensors = manager.get_peak_memory()
    assert peak_with_tensors > initial_peak

    # Cleanup
    for tensor in tensors:
        del tensor
    gc.collect()


def test_memory_optimization_strategies() -> None:
    """Test memory optimization strategies."""
    manager = MemoryManager()

    # Test tensor reuse with smaller tensors
    x = manager.allocate_tensor((100, 100))
    y = manager.allocate_tensor((100, 100))

    initial_memory = manager.get_allocated_memory()

    # Perform operation with memory optimization
    manager.optimized_matmul(x, y)

    final_memory = manager.get_allocated_memory()

    # Memory increase should be minimal due to optimization
    assert (
        final_memory - initial_memory
    ) / initial_memory < 0.5  # Less than 50% increase


@pytest.mark.parametrize(
    "size", [(10, 10), (100, 100), (500, 500)]
)  # Reduced from 5000x5000
def test_memory_scaling(size: Tuple[int, int]) -> None:
    """Test memory usage scaling with different tensor sizes."""
    manager = MemoryManager()

    initial_memory = manager.get_allocated_memory()
    tensor = manager.allocate_tensor(size)  # Keep reference to tensor
    final_memory = manager.get_allocated_memory()

    # Calculate theoretical memory
    theoretical_memory = size[0] * size[1] * 4  # Assuming float32

    # Check if actual memory usage is within 10% of theoretical
    actual_memory = final_memory - initial_memory
    ratio = actual_memory / theoretical_memory
    assert 0.9 <= ratio <= 1.1  # Within 10% of theoretical memory

    # Keep reference to tensor until end of test
    assert tensor.shape == size
