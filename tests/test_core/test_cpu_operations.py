import pytest
import numpy as np
import torch
import time

from src.infrastructure.base import CPUDevice

def test_cpu_basic_operations():
    """Test basic arithmetic operations on CPU."""
    device = CPUDevice()
    
    # Test tensor creation
    x = device.create_tensor([1, 2, 3, 4])
    y = device.create_tensor([5, 6, 7, 8])
    
    # Test addition
    result = device.add(x, y)
    expected = torch.tensor([6, 8, 10, 12])
    assert torch.allclose(result, expected)
    
    # Test multiplication
    result = device.multiply(x, y)
    expected = torch.tensor([5, 12, 21, 32])
    assert torch.allclose(result, expected)

def test_cpu_memory_management():
    """Test memory allocation and deallocation on CPU."""
    device = CPUDevice()
    
    # Test large tensor allocation
    large_tensor = device.create_tensor(np.random.randn(1000, 1000))
    assert large_tensor.shape == (1000, 1000)
    
    # Test memory cleanup
    del large_tensor
    torch.cuda.empty_cache()  # Force cleanup

def test_cpu_thread_pool():
    """Test thread pool operations."""
    device = CPUDevice()
    
    # Test parallel operations
    def parallel_op(x):
        return x * 2
    
    data = device.create_tensor(np.random.randn(1000, 100))
    result = device.parallel_map(parallel_op, data)
    expected = data * 2
    assert torch.allclose(result, expected)

def test_cpu_cache_efficiency():
    """Test cache-friendly operations."""
    device = CPUDevice()
    
    # Test operations with different memory layouts
    row_major = device.create_tensor(np.random.randn(1000, 1000))
    col_major = row_major.t().contiguous()
    
    # Test row-wise operations (should be faster)
    row_sum = device.sum(row_major, dim=1)
    col_sum = device.sum(col_major, dim=0)
    
    assert row_sum.shape == (1000,)
    assert col_sum.shape == (1000,)

def test_cpu_performance_benchmarks():
    """Test performance metrics."""
    device = CPUDevice()
    
    # Test computation speed
    data = device.create_tensor(np.random.randn(1000, 1000))
    
    start_time = time.perf_counter()
    result = device.matmul(data, data.t())
    compute_time = time.perf_counter() - start_time
    
    assert result.shape == (1000, 1000)
    assert compute_time > 0  # Basic sanity check