import torch
import time
import pytest
from typing import Tuple, List
import numpy as np

def setup_test_tensors(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create test tensors of specified size."""
    return torch.randn(size, size), torch.randn(size, size)

def measure_transfer_time(func) -> float:
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            result.cpu()  # Force sync
        return time.time() - start
    return wrapper

class TestVulkanMemory:
    @pytest.fixture(autouse=True)
    def setup(self):
        assert torch.is_vulkan_available(), "Vulkan is not available"
        self.sizes = [512, 1024, 2048]
        self.iterations = 5

    def test_pinned_memory_performance(self):
        """Test pinned memory transfer performance."""
        for size in self.sizes:
            # Create pinned memory tensor
            a_cpu = torch.randn(size, size, pin_memory=True)
            
            # Measure transfer time
            times = []
            for _ in range(self.iterations):
                start = time.time()
                a_vulkan = a_cpu.to("vulkan")
                a_vulkan.cpu()  # Force sync
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            print(f"\nPinned Memory Transfer {size}x{size}")
            print(f"Average transfer time: {avg_time:.4f}s")
            
            # Compare with non-pinned memory
            a_nonpin = torch.randn(size, size)
            start = time.time()
            a_vulkan = a_nonpin.to("vulkan")
            a_vulkan.cpu()
            nonpin_time = time.time() - start
            
            print(f"Non-pinned transfer time: {nonpin_time:.4f}s")
            print(f"Speedup: {nonpin_time/avg_time:.2f}x")
            
            assert avg_time < nonpin_time, "Pinned memory should be faster"

    def test_staging_buffer_efficiency(self):
        """Test staging buffer transfer efficiency."""
        for size in self.sizes:
            a, b = setup_test_tensors(size)
            
            # Test different staging buffer sizes
            staging_sizes = [size//4, size//2, size]
            times = []
            
            for staging_size in staging_sizes:
                start = time.time()
                # Simulate staged transfer
                for i in range(0, size, staging_size):
                    end = min(i + staging_size, size)
                    slice_vulkan = a[i:end].to("vulkan")
                    slice_vulkan.cpu()  # Force sync
                times.append(time.time() - start)
            
            print(f"\nStaging Buffer Test {size}x{size}")
            for staging_size, t in zip(staging_sizes, times):
                print(f"Staging size {staging_size}: {t:.4f}s")
            
            # Optimal staging size should be faster than smallest
            assert min(times) < times[0], "Larger staging buffers should be more efficient"

    def test_zero_copy_operations(self):
        """Test zero-copy operation performance."""
        for size in self.sizes:
            # Create page-aligned memory for zero-copy
            a = torch.randn(size, size, pin_memory=True)
            
            # Measure zero-copy operation time
            start = time.time()
            a_vulkan = a.to("vulkan")
            result = torch.relu(a_vulkan)
            result.cpu()
            zero_copy_time = time.time() - start
            
            # Compare with standard copy
            b = torch.randn(size, size)
            start = time.time()
            b_vulkan = b.to("vulkan")
            result = torch.relu(b_vulkan)
            result.cpu()
            standard_time = time.time() - start
            
            print(f"\nZero-copy Test {size}x{size}")
            print(f"Zero-copy time: {zero_copy_time:.4f}s")
            print(f"Standard time: {standard_time:.4f}s")
            print(f"Speedup: {standard_time/zero_copy_time:.2f}x")
            
            assert zero_copy_time < standard_time, "Zero-copy should be faster"

    def test_memory_barriers(self):
        """Test memory barrier overhead and optimization."""
        for size in self.sizes:
            a, b = setup_test_tensors(size)
            a_vulkan = a.to("vulkan")
            b_vulkan = b.to("vulkan")
            
            # Test with explicit barriers
            start = time.time()
            for _ in range(self.iterations):
                c = a_vulkan + b_vulkan
                c = c * a_vulkan
                c = torch.relu(c)
                c.cpu()  # Force sync
            barrier_time = time.time() - start
            
            # Test with fused operations
            start = time.time()
            for _ in range(self.iterations):
                c = torch.relu(a_vulkan * (a_vulkan + b_vulkan))
                c.cpu()
            fused_time = time.time() - start
            
            print(f"\nBarrier Test {size}x{size}")
            print(f"With barriers: {barrier_time:.4f}s")
            print(f"Fused ops: {fused_time:.4f}s")
            print(f"Overhead: {(barrier_time-fused_time)/fused_time*100:.1f}%")
            
            assert fused_time < barrier_time, "Fused operations should be faster"

    def test_memory_pool_efficiency(self):
        """Test memory pool allocation and recycling."""
        allocation_sizes = [32, 64, 128, 256]  # KB
        allocations: List[torch.Tensor] = []
        
        # Test allocation speed
        start = time.time()
        for size in allocation_sizes:
            tensor = torch.randn(size*256, pin_memory=True).to("vulkan")
            allocations.append(tensor)
        alloc_time = time.time() - start
        
        # Test reuse speed
        start = time.time()
        for tensor in allocations:
            tensor.zero_()
        reuse_time = time.time() - start
        
        print("\nMemory Pool Test")
        print(f"Allocation time: {alloc_time:.4f}s")
        print(f"Reuse time: {reuse_time:.4f}s")
        print(f"Reuse speedup: {alloc_time/reuse_time:.2f}x")
        
        assert reuse_time < alloc_time, "Memory reuse should be faster"
        
        # Cleanup
        for tensor in allocations:
            del tensor

    def test_defragmentation(self):
        """Test memory defragmentation impact."""
        base_size = 64  # Base allocation size in KB
        
        # Create fragmented memory state
        fragments = []
        for i in range(10):
            size = base_size * (2 ** (i % 3))  # Various sizes
            tensor = torch.randn(size*256, pin_memory=True).to("vulkan")
            fragments.append(tensor)
        
        # Delete every other allocation to create gaps
        for i in range(0, len(fragments), 2):
            del fragments[i]
        
        # Measure allocation time in fragmented state
        start = time.time()
        new_tensor = torch.randn(base_size*256, pin_memory=True).to("vulkan")
        frag_time = time.time() - start
        
        # Clean up and defrag
        del fragments
        torch.cuda.empty_cache()  # Similar concept for Vulkan
        
        # Measure allocation time after cleanup
        start = time.time()
        new_tensor2 = torch.randn(base_size*256, pin_memory=True).to("vulkan")
        clean_time = time.time() - start
        
        print("\nDefragmentation Test")
        print(f"Fragmented allocation: {frag_time:.4f}s")
        print(f"Clean allocation: {clean_time:.4f}s")
        print(f"Fragmentation overhead: {(frag_time-clean_time)/clean_time*100:.1f}%")
        
        assert clean_time < frag_time, "Defragmented memory should be faster"
