import torch
import time
import pytest
from typing import Tuple, List
import numpy as np

def setup_pattern_data(batch_size: int, pattern_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create pattern and flow test data."""
    pattern = torch.randn(batch_size, pattern_size, pattern_size)
    flow = torch.randn(batch_size, pattern_size, pattern_size)
    return pattern, flow

def setup_flow_data(batch_size: int, manifold_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create metric and connection test data."""
    metric = torch.randn(batch_size, manifold_dim, manifold_dim)
    connection = torch.randn(batch_size, manifold_dim, manifold_dim, manifold_dim)
    return metric, connection

class TestVulkanShaders:
    @pytest.fixture(autouse=True)
    def setup(self):
        assert torch.is_vulkan_available(), "Vulkan is not available"
        self.batch_sizes = [1, 4, 16]
        self.pattern_sizes = [128, 256, 512]
        self.manifold_dims = [16, 32, 64]
        self.iterations = 3

    def test_pattern_evolution_performance(self):
        """Test pattern evolution shader performance."""
        for batch_size in self.batch_sizes:
            for pattern_size in self.pattern_sizes:
                pattern, flow = setup_pattern_data(batch_size, pattern_size)
                
                # Move to Vulkan
                pattern_vulkan = pattern.to("vulkan")
                flow_vulkan = flow.to("vulkan")
                
                # Warmup
                _ = torch.tanh(pattern_vulkan + flow_vulkan * 0.1)
                
                # Test with different evolution rates
                rates = [0.01, 0.1, 0.5]
                times = []
                
                for rate in rates:
                    start = time.time()
                    for _ in range(self.iterations):
                        result = torch.tanh(pattern_vulkan + flow_vulkan * rate)
                        result.cpu()  # Force sync
                    times.append((time.time() - start) / self.iterations)
                
                print(f"\nPattern Evolution {batch_size}x{pattern_size}x{pattern_size}")
                for rate, t in zip(rates, times):
                    print(f"Rate {rate}: {t:.4f}s")
                print(f"Batch processing efficiency: {times[0]/times[-1]:.2f}x")

    def test_flow_computation_performance(self):
        """Test flow computation shader performance."""
        for batch_size in self.batch_sizes:
            for dim in self.manifold_dims:
                metric, connection = setup_flow_data(batch_size, dim)
                
                # Move to Vulkan
                metric_vulkan = metric.to("vulkan")
                connection_vulkan = connection.to("vulkan")
                
                # Warmup
                _ = torch.matmul(metric_vulkan, connection_vulkan.view(batch_size, dim, -1))
                
                # Test computation
                start = time.time()
                for _ in range(self.iterations):
                    result = torch.matmul(metric_vulkan, connection_vulkan.view(batch_size, dim, -1))
                    result.cpu()
                avg_time = (time.time() - start) / self.iterations
                
                print(f"\nFlow Computation {batch_size}x{dim}x{dim}")
                print(f"Average time: {avg_time:.4f}s")
                print(f"GFLOPS: {2*batch_size*dim*dim*dim/avg_time/1e9:.2f}")

    def test_workgroup_impact(self):
        """Test impact of different workgroup sizes."""
        pattern_size = 256
        batch_size = 4
        pattern, flow = setup_pattern_data(batch_size, pattern_size)
        
        # Move to Vulkan
        pattern_vulkan = pattern.to("vulkan")
        flow_vulkan = flow.to("vulkan")
        
        # Test different effective workgroup sizes by padding
        paddings = [0, 8, 16]  # Simulates different workgroup alignments
        times = []
        
        for padding in paddings:
            if padding > 0:
                pattern_pad = torch.nn.functional.pad(pattern_vulkan, (0, padding, 0, padding))
                flow_pad = torch.nn.functional.pad(flow_vulkan, (0, padding, 0, padding))
            else:
                pattern_pad = pattern_vulkan
                flow_pad = flow_vulkan
            
            start = time.time()
            for _ in range(self.iterations):
                result = torch.tanh(pattern_pad + flow_pad * 0.1)
                result.cpu()
            times.append((time.time() - start) / self.iterations)
        
        print("\nWorkgroup Size Impact")
        for padding, t in zip(paddings, times):
            print(f"Padding {padding}: {t:.4f}s")
        print(f"Alignment impact: {(times[-1]-times[0])/times[0]*100:.1f}%")

    def test_push_constant_performance(self):
        """Test push constant vs descriptor performance."""
        pattern_size = 256
        batch_size = 4
        pattern, flow = setup_pattern_data(batch_size, pattern_size)
        
        # Move to Vulkan
        pattern_vulkan = pattern.to("vulkan")
        flow_vulkan = flow.to("vulkan")
        
        # Test with different parameter passing methods
        # Method 1: Push constants (implicit in operation)
        start = time.time()
        for _ in range(self.iterations):
            result = torch.tanh(pattern_vulkan + flow_vulkan * 0.1)
            result.cpu()
        push_time = (time.time() - start) / self.iterations
        
        # Method 2: Descriptor set (using additional buffer)
        param_tensor = torch.tensor([0.1], device="vulkan")
        start = time.time()
        for _ in range(self.iterations):
            result = torch.tanh(pattern_vulkan + flow_vulkan * param_tensor)
            result.cpu()
        desc_time = (time.time() - start) / self.iterations
        
        print("\nParameter Passing Performance")
        print(f"Push constant time: {push_time:.4f}s")
        print(f"Descriptor set time: {desc_time:.4f}s")
        print(f"Overhead: {(desc_time-push_time)/push_time*100:.1f}%")

    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        pattern_size = 256
        pattern, flow = setup_pattern_data(1, pattern_size)
        
        times_single = []
        times_batch = []
        
        for batch_size in self.batch_sizes:
            # Single processing
            start = time.time()
            for _ in range(batch_size):
                pattern_vulkan = pattern.to("vulkan")
                flow_vulkan = flow.to("vulkan")
                result = torch.tanh(pattern_vulkan + flow_vulkan * 0.1)
                result.cpu()
            times_single.append(time.time() - start)
            
            # Batch processing
            pattern_batch = pattern.repeat(batch_size, 1, 1)
            flow_batch = flow.repeat(batch_size, 1, 1)
            pattern_vulkan = pattern_batch.to("vulkan")
            flow_vulkan = flow_batch.to("vulkan")
            
            start = time.time()
            result = torch.tanh(pattern_vulkan + flow_vulkan * 0.1)
            result.cpu()
            times_batch.append(time.time() - start)
        
        print("\nBatch Processing Efficiency")
        for batch_size, t_single, t_batch in zip(self.batch_sizes, times_single, times_batch):
            print(f"Batch size {batch_size}:")
            print(f"  Single: {t_single:.4f}s")
            print(f"  Batch:  {t_batch:.4f}s")
            print(f"  Speedup: {t_single/t_batch:.2f}x")
