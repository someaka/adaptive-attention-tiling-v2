"""Tests for GPU memory management system."""

import gc

import pytest
import torch

from src.core.performance.gpu.memory_management import GPUMemoryManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestGPUMemoryManager:
    """Test suite for GPU memory management."""

    @pytest.fixture
    def manager(self):
        """Create a GPU memory manager for testing."""
        manager = GPUMemoryManager()
        # Clear GPU memory before each test
        torch.cuda.empty_cache()
        gc.collect()
        return manager

    def test_tensor_allocation(self, manager):
        """Test basic tensor allocation on GPU."""
        initial_memory = manager.get_allocated_memory()

        # Allocate tensor
        tensor = manager.allocate_tensor((100, 100))

        # Check tensor properties
        assert tensor.device.type == "cuda"
        assert tensor.shape == (100, 100)
        assert manager.get_allocated_memory() > initial_memory

        # Verify memory tracking
        assert manager.get_peak_memory() >= manager.get_allocated_memory()
        assert manager.get_fragmentation_ratio() >= 0.0

    def test_memory_tracking(self, manager):
        """Test memory usage tracking during operations."""
        # Allocate multiple tensors
        tensors = [manager.allocate_tensor((50, 50)) for _ in range(5)]
        peak_memory = manager.get_peak_memory()

        # Delete some tensors
        del tensors[::2]
        gc.collect()
        torch.cuda.empty_cache()

        # Check memory changes
        assert manager.get_allocated_memory() < peak_memory
        assert manager.get_peak_memory() == peak_memory

    def test_cpu_to_gpu_transfer(self, manager):
        """Test CPU to GPU tensor transfer."""
        # Create CPU tensor
        cpu_tensor = torch.randn(100, 100)
        initial_memory = manager.get_allocated_memory()

        # Transfer to GPU
        gpu_tensor = manager.transfer_to_gpu(cpu_tensor)

        # Verify transfer
        assert gpu_tensor.device.type == "cuda"
        assert torch.all(cpu_tensor == gpu_tensor.cpu())
        assert manager.get_allocated_memory() > initial_memory

    def test_memory_optimization(self, manager):
        """Test memory layout optimization."""
        # Allocate tensors of different sizes
        [
            manager.allocate_tensor((10, 10)),
            manager.allocate_tensor((50, 50)),
            manager.allocate_tensor((20, 20)),
        ]

        # Record fragmentation
        initial_fragmentation = manager.get_fragmentation_ratio()

        # Optimize memory layout
        manager.optimize_memory_layout()

        # Check if fragmentation improved or stayed the same
        assert manager.get_fragmentation_ratio() <= initial_fragmentation * 1.1

    def test_cache_management(self, manager):
        """Test GPU memory cache management."""
        # Record initial cache
        initial_cache = manager.get_cache_memory()

        # Allocate and free some tensors
        tensor = manager.allocate_tensor((1000, 1000))
        del tensor
        gc.collect()

        # Clear cache
        manager.clear_cache()

        # Verify cache was cleared
        assert manager.get_cache_memory() <= initial_cache

    def test_memory_limits(self, manager):
        """Test handling of memory limits."""
        # Try to allocate a very large tensor
        large_size = (1000, 1000, 1000)  # 4GB for float32

        # This should raise an out of memory error
        with pytest.raises(RuntimeError, match="out of memory"):
            manager.allocate_tensor(large_size)

    @pytest.mark.parametrize("size", [(10, 10), (100, 100), (500, 500)])
    def test_memory_scaling(self, manager, size):
        """Test memory usage scaling with different tensor sizes."""
        initial_memory = manager.get_allocated_memory()

        # Allocate tensor
        manager.allocate_tensor(size)
        final_memory = manager.get_allocated_memory()

        # Calculate theoretical memory (assuming float32)
        theoretical_memory = size[0] * size[1] * 4

        # Check if actual memory usage is within 10% of theoretical
        actual_memory = final_memory - initial_memory
        ratio = actual_memory / theoretical_memory
        assert 0.9 <= ratio <= 1.1  # Within 10% of theoretical memory
