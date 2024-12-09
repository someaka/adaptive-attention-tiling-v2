"""Vulkan-specific benchmarks for the Adaptive Attention Tiling system.

This module provides comprehensive benchmarks for Vulkan operations including:
1. Memory operations and transfer patterns
2. Resource management and pool efficiency
3. Lifecycle analysis and optimization
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple
from src.core.benchmarks import BenchmarkMetrics
from src.core.vulkan import VulkanMemory, VulkanResources

class TestVulkanBenchmarks:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        assert torch.is_vulkan_available(), "Vulkan is not available"
        self.memory = VulkanMemory()
        self.resources = VulkanResources()
        self.sizes = [512, 1024, 2048, 4096]
        self.iterations = 10
        self.metrics = BenchmarkMetrics()

    def test_pool_efficiency(self):
        """Benchmark memory pool allocation efficiency."""
        for size in self.sizes:
            # Test different pool configurations
            pool_sizes = [size*4, size*8, size*16]  # KB
            
            for pool_size in pool_sizes:
                self.memory.create_pool(pool_size)
                
                # Measure allocation performance
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                allocations = []
                for _ in range(10):
                    buffer = self.memory.allocate(size)
                    allocations.append(buffer)
                end_time.record()
                
                torch.cuda.synchronize()
                alloc_time = start_time.elapsed_time(end_time)
                
                # Measure fragmentation
                fragmentation = self.memory.get_fragmentation()
                
                # Cleanup
                for buffer in allocations:
                    self.memory.free(buffer)
                
                self.metrics.record_operation(
                    name="pool_efficiency",
                    pool_size=pool_size,
                    allocation_size=size,
                    allocation_time=alloc_time,
                    fragmentation=fragmentation
                )
                
                self.memory.destroy_pool()

    def test_fragmentation_analysis(self):
        """Analyze memory fragmentation patterns."""
        for size in self.sizes:
            # Create mixed-size allocations
            allocation_sizes = [size//4, size//2, size]
            allocations = []
            
            # Initial allocations
            for alloc_size in allocation_sizes:
                buffer = self.memory.allocate(alloc_size)
                allocations.append(buffer)
            
            # Measure fragmentation after mixed deallocations
            fragmentation_points = []
            for i in range(len(allocations)):
                self.memory.free(allocations[i])
                fragmentation_points.append(self.memory.get_fragmentation())
            
            # Attempt defragmentation
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            self.memory.defragment()
            end_time.record()
            
            torch.cuda.synchronize()
            defrag_time = start_time.elapsed_time(end_time)
            
            self.metrics.record_operation(
                name="fragmentation_analysis",
                size=size,
                initial_fragmentation=fragmentation_points[0],
                peak_fragmentation=max(fragmentation_points),
                defrag_time=defrag_time,
                final_fragmentation=self.memory.get_fragmentation()
            )

    def test_resource_lifecycle(self):
        """Test resource lifecycle management efficiency."""
        for size in self.sizes:
            # Create resources of different types
            resource_types = ["buffer", "image", "sampler"]
            creation_times = []
            destruction_times = []
            
            for res_type in resource_types:
                # Measure creation time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                resource = self.resources.create(res_type, size)
                end_time.record()
                
                torch.cuda.synchronize()
                creation_times.append(start_time.elapsed_time(end_time))
                
                # Use resource
                self.resources.bind(resource)
                self.resources.unbind(resource)
                
                # Measure destruction time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.resources.destroy(resource)
                end_time.record()
                
                torch.cuda.synchronize()
                destruction_times.append(start_time.elapsed_time(end_time))
            
            # Record metrics
            for i, res_type in enumerate(resource_types):
                self.metrics.record_operation(
                    name=f"resource_lifecycle_{res_type}",
                    size=size,
                    creation_time=creation_times[i],
                    destruction_time=destruction_times[i],
                    total_time=creation_times[i] + destruction_times[i]
                )

    def test_memory_operations(self):
        """Benchmark memory operations performance."""
        for size in self.sizes:
            # Test different transfer patterns
            host_data = torch.randn(size, size)
            
            # Host to device transfer
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            device_data = self.memory.transfer_to_device(host_data)
            end_time.record()
            
            torch.cuda.synchronize()
            h2d_time = start_time.elapsed_time(end_time)
            
            # Device to host transfer
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            result = self.memory.transfer_to_host(device_data)
            end_time.record()
            
            torch.cuda.synchronize()
            d2h_time = start_time.elapsed_time(end_time)
            
            # Record metrics
            self.metrics.record_operation(
                name="memory_operations",
                size=size,
                h2d_time=h2d_time,
                d2h_time=d2h_time,
                bandwidth=size * size * 4 / (h2d_time + d2h_time)  # bytes/ms
            )
