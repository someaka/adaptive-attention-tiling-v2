"""
Unit tests for infrastructure components.

Tests cover:
1. CPU optimization
2. Memory management
3. Vulkan integration
4. Parallel processing
5. Resource allocation
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure import (
    CPUOptimizer,
    MemoryManager,
    VulkanIntegration,
    ParallelProcessor,
    ResourceAllocator,
    InfrastructureMetrics
)

class TestInfrastructure:
    @pytest.fixture
    def batch_size(self) -> int:
        return 32

    @pytest.fixture
    def data_dim(self) -> int:
        return 1024

    @pytest.fixture
    def num_threads(self) -> int:
        return mp.cpu_count()

    @pytest.fixture
    def cpu_optimizer(self) -> CPUOptimizer:
        return CPUOptimizer(
            cache_size=2**20,
            prefetch_distance=4,
            thread_affinity='compact'
        )

    def test_cpu_optimization(self, cpu_optimizer: CPUOptimizer,
                            batch_size: int, data_dim: int):
        """Test CPU optimization features."""
        # Generate test data
        data = torch.randn(batch_size, data_dim)
        
        # Test cache optimization
        with cpu_optimizer.optimize_cache():
            result = torch.matmul(data, data.t())
        assert result.shape == (batch_size, batch_size)
        
        # Test thread affinity
        thread_map = cpu_optimizer.get_thread_mapping()
        assert len(thread_map) == mp.cpu_count()
        assert all(isinstance(core, int) for core in thread_map.values())
        
        # Test SIMD optimization
        def compute_intensive_task(x: torch.Tensor) -> torch.Tensor:
            return torch.fft.fft2(x)
            
        optimized_result = cpu_optimizer.optimize_simd(
            compute_intensive_task, data
        )
        assert optimized_result.shape == data.shape
        
        # Test performance metrics
        metrics = cpu_optimizer.get_performance_metrics()
        assert 'cache_hits' in metrics
        assert 'cache_misses' in metrics
        assert 'flops' in metrics

    def test_memory_management(self, batch_size: int, data_dim: int):
        """Test memory management features."""
        # Create memory manager
        manager = MemoryManager(
            max_memory=2**30,  # 1GB
            allocation_strategy='dynamic'
        )
        
        # Test memory allocation
        data = manager.allocate((batch_size, data_dim))
        assert isinstance(data, torch.Tensor)
        assert data.shape == (batch_size, data_dim)
        
        # Test memory tracking
        usage = manager.get_memory_usage()
        assert usage > 0
        assert usage <= manager.max_memory
        
        # Test memory optimization
        def memory_intensive_task():
            tensors = [torch.randn(1000, 1000) for _ in range(10)]
            return torch.stack(tensors)
            
        with manager.optimize_memory():
            result = memory_intensive_task()
        assert isinstance(result, torch.Tensor)
        
        # Test garbage collection
        manager.collect_garbage()
        assert manager.get_memory_usage() < usage

    def test_vulkan_integration(self, batch_size: int, data_dim: int):
        """Test Vulkan integration features."""
        # Create Vulkan integration
        vulkan = VulkanIntegration(
            device_index=0,
            compute_queue_count=2
        )
        
        # Test device capabilities
        capabilities = vulkan.get_device_capabilities()
        assert 'compute_shader' in capabilities
        assert 'shared_memory' in capabilities
        
        # Test memory transfer
        cpu_data = torch.randn(batch_size, data_dim)
        gpu_data = vulkan.transfer_to_device(cpu_data)
        assert gpu_data.shape == cpu_data.shape
        
        # Test compute shader
        shader_code = """
        #version 450
        layout(local_size_x = 256) in;
        layout(std430, binding = 0) buffer Data {
            float data[];
        };
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx < data.length()) {
                data[idx] = data[idx] * data[idx];
            }
        }
        """
        result = vulkan.run_compute_shader(
            shader_code, gpu_data, local_size=256
        )
        assert torch.allclose(result.cpu(), cpu_data * cpu_data)
        
        # Test synchronization
        vulkan.synchronize()
        assert not vulkan.is_busy()

    def test_parallel_processing(self, batch_size: int, data_dim: int,
                               num_threads: int):
        """Test parallel processing features."""
        # Create parallel processor
        processor = ParallelProcessor(num_threads=num_threads)
        
        # Test data parallelism
        data = torch.randn(batch_size, data_dim)
        chunks = processor.split_data(data, num_chunks=num_threads)
        assert len(chunks) == num_threads
        
        def process_chunk(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(x)
            
        results = processor.map(process_chunk, chunks)
        assert len(results) == num_threads
        combined = processor.combine_results(results)
        assert combined.shape == data.shape
        
        # Test thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(process_chunk, chunk) 
                      for chunk in chunks]
            results = [f.result() for f in futures]
        assert len(results) == num_threads
        
        # Test load balancing
        load_stats = processor.get_load_statistics()
        assert 'thread_utilization' in load_stats
        assert 'load_balance' in load_stats

    def test_resource_allocation(self, batch_size: int, data_dim: int):
        """Test resource allocation features."""
        # Create resource allocator
        allocator = ResourceAllocator(
            memory_limit=2**30,
            compute_limit=0.8
        )
        
        # Test resource monitoring
        resources = allocator.get_available_resources()
        assert 'memory' in resources
        assert 'compute' in resources
        
        # Test allocation strategy
        strategy = allocator.get_optimal_strategy(
            data_size=batch_size * data_dim * 4,
            compute_intensity=0.5
        )
        assert 'batch_size' in strategy
        assert 'num_threads' in strategy
        
        # Test resource limits
        def resource_intensive_task():
            return torch.randn(10000, 10000)
            
        with pytest.raises(ResourceError):
            allocator.execute_with_limits(
                resource_intensive_task,
                memory_limit=1000,
                compute_limit=0.1
            )
            
        # Test resource cleanup
        allocator.release_resources()
        assert allocator.get_memory_usage() == 0

    def test_infrastructure_integration(self, batch_size: int, data_dim: int,
                                     num_threads: int):
        """Test integrated infrastructure components."""
        # Create infrastructure components
        cpu_opt = CPUOptimizer()
        mem_mgr = MemoryManager()
        vulkan = VulkanIntegration()
        parallel = ParallelProcessor(num_threads=num_threads)
        allocator = ResourceAllocator()
        
        # Generate test data
        data = torch.randn(batch_size, data_dim)
        
        # Test integrated computation pipeline
        def compute_pipeline(x: torch.Tensor) -> torch.Tensor:
            # CPU optimization
            with cpu_opt.optimize_cache():
                x = torch.matmul(x, x.t())
                
            # Memory management
            with mem_mgr.optimize_memory():
                x = torch.fft.fft2(x)
                
            # Vulkan acceleration
            if vulkan.is_available():
                x = vulkan.accelerate_computation(x)
                
            # Parallel processing
            chunks = parallel.split_data(x, num_chunks=num_threads)
            results = parallel.map(torch.nn.functional.relu, chunks)
            x = parallel.combine_results(results)
            
            return x
            
        # Execute pipeline with resource limits
        result = allocator.execute_with_limits(
            lambda: compute_pipeline(data),
            memory_limit=2**30,
            compute_limit=0.8
        )
        
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()
        
        # Test performance metrics
        metrics = InfrastructureMetrics.collect_all(
            cpu_opt, mem_mgr, vulkan, parallel, allocator
        )
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in [
            'cpu_efficiency',
            'memory_efficiency',
            'gpu_utilization',
            'parallel_efficiency',
            'resource_utilization'
        ])
