"""
Unit tests for infrastructure components.

Tests cover:
1. CPU optimization
2. Memory management
3. Vulkan integration
4. Parallel processing
5. Resource allocation
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from src.infrastructure.base import (
    CPUOptimizer,
    InfrastructureMetrics,
    MemoryManager,
    ParallelProcessor,
    ResourceAllocator,
    VulkanIntegration,
    ResourceAllocationError,
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
        return CPUOptimizer(enable_profiling=True)

    def test_cpu_optimization(
        self, cpu_optimizer: CPUOptimizer, batch_size: int, data_dim: int
    ):
        """Test CPU optimization features."""
        # Generate test data
        data = torch.randn(batch_size, data_dim)

        # Test cache optimization
        with cpu_optimizer.profile():
            result = torch.matmul(data, data.t())
        assert result.shape == (batch_size, batch_size)

        # Test thread affinity
        thread_info = cpu_optimizer.get_thread_info()
        assert len(thread_info) > 0
        assert all(isinstance(info, dict) for info in thread_info)

        # Test SIMD optimization
        def compute_intensive_task(x: torch.Tensor) -> torch.Tensor:
            return torch.fft.fft2(x)

        optimized_result = cpu_optimizer.optimize(compute_intensive_task, data)
        assert optimized_result.shape == data.shape

        # Test performance metrics
        metrics = cpu_optimizer.get_metrics()
        assert metrics is not None
        assert "execution_time" in metrics
        assert "memory_usage" in metrics

    def test_memory_management(self, batch_size: int, data_dim: int):
        """Test memory management features."""
        # Create memory manager
        manager = MemoryManager(pool_size=2**30)  # 1GB

        # Test memory allocation
        data = torch.empty(batch_size, data_dim, dtype=torch.float32)
        managed_data = manager.manage_tensor(data)
        assert isinstance(managed_data, torch.Tensor)
        assert managed_data.shape == (batch_size, data_dim)

        # Test memory tracking
        stats = manager.get_memory_stats()
        assert stats is not None
        assert stats.total_allocated > 0
        assert stats.total_allocated <= manager.pool_size

        # Test memory optimization
        def memory_intensive_task():
            tensors = [torch.randn(1000, 1000) for _ in range(10)]
            return torch.stack(tensors)

        with manager.optimize():
            result = memory_intensive_task()
        assert isinstance(result, torch.Tensor)

        # Test cleanup
        manager.cleanup()
        new_stats = manager.get_memory_stats()
        assert new_stats.total_allocated < stats.total_allocated

    def test_vulkan_integration(self, batch_size: int, data_dim: int):
        """Test Vulkan integration features."""
        # Create Vulkan integration
        vulkan = VulkanIntegration()

        # Test device capabilities
        info = vulkan.get_device_info()
        assert info is not None
        assert info.compute_support
        assert len(info.memory_types) > 0

        # Test memory transfer
        cpu_data = torch.randn(batch_size, data_dim)
        gpu_buffer = vulkan.create_buffer(cpu_data)
        assert gpu_buffer is not None

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
        result = vulkan.compute(shader_code, gpu_buffer, workgroup_size=256)
        assert result is not None
        result_data = vulkan.download_buffer(result)
        assert torch.allclose(result_data, cpu_data * cpu_data)

        # Test synchronization
        vulkan.wait_idle()
        assert vulkan.get_queue_status() == "idle"

    def test_parallel_processing(
        self, batch_size: int, data_dim: int, num_threads: int
    ):
        """Test parallel processing features."""
        # Create parallel processor
        processor = ParallelProcessor(num_threads=num_threads)

        # Test data parallelism
        data = torch.randn(batch_size, data_dim)
        partitions = processor.partition_data(data)
        assert len(partitions) == num_threads

        def process_chunk(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(x)

        results = processor.process_parallel(process_chunk, partitions)
        assert len(results) == num_threads
        combined = processor.merge_results(results)
        assert combined.shape == data.shape

        # Test thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(process_chunk, chunk) for chunk in partitions]
            results = [f.result() for f in futures]
        assert len(results) == num_threads

        # Test performance monitoring
        stats = processor.get_stats()
        assert stats is not None
        assert "thread_usage" in stats
        assert "processing_time" in stats

    def test_resource_allocation(self, batch_size: int, data_dim: int):
        """Test resource allocation features."""
        # Create resource allocator
        allocator = ResourceAllocator(memory_limit=2**30, compute_limit=80)  # 80% CPU limit

        # Test resource monitoring
        status = allocator.get_status()
        assert status is not None
        assert "available_memory" in status
        assert "cpu_usage" in status

        # Test allocation planning
        plan = allocator.plan_allocation(
            memory_size=batch_size * data_dim * 4,
            compute_intensity=50  # 50% CPU intensity
        )
        assert plan is not None
        assert "recommended_batch_size" in plan
        assert "thread_count" in plan

        # Test resource limits
        def resource_intensive_task():
            return torch.randn(10000, 10000)

        with pytest.raises(ResourceAllocationError):
            allocator.run_with_limits(
                resource_intensive_task,
                memory_limit=1000,
                cpu_limit=10
            )

        # Test resource cleanup
        allocator.cleanup()
        assert allocator.get_status()["available_memory"] == allocator.memory_limit

    def test_infrastructure_integration(
        self, batch_size: int, data_dim: int, num_threads: int
    ):
        """Test integrated infrastructure components."""
        # Create infrastructure components
        cpu_opt = CPUOptimizer(enable_profiling=True)
        mem_mgr = MemoryManager(pool_size=2**30)
        vulkan = VulkanIntegration()
        parallel = ParallelProcessor(num_threads=num_threads)
        allocator = ResourceAllocator(memory_limit=2**30, compute_limit=80)

        # Generate test data
        data = torch.randn(batch_size, data_dim)

        # Test integrated computation pipeline
        def compute_pipeline(x: torch.Tensor) -> torch.Tensor:
            # CPU optimization
            with cpu_opt.profile():
                x = torch.matmul(x, x.t())

            # Memory management
            with mem_mgr.optimize():
                x = torch.fft.fft2(x)

            # Vulkan acceleration
            if vulkan.get_device_info() is not None:
                gpu_buffer = vulkan.create_buffer(x)
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
                x = vulkan.download_buffer(vulkan.compute(shader_code, gpu_buffer))

            # Parallel processing
            partitions = parallel.partition_data(x)
            results = parallel.process_parallel(torch.nn.functional.relu, partitions)
            return parallel.merge_results(results)

        # Execute pipeline with resource limits
        result = allocator.run_with_limits(
            lambda: compute_pipeline(data),
            memory_limit=2**30,
            cpu_limit=80
        )

        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

        # Test performance metrics
        metrics = InfrastructureMetrics.collect(
            cpu_optimizer=cpu_opt,
            memory_manager=mem_mgr,
            vulkan_integration=vulkan,
            parallel_processor=parallel,
            resource_allocator=allocator
        )
        assert isinstance(metrics, dict)
        assert all(
            key in metrics
            for key in [
                "cpu_metrics",
                "memory_metrics",
                "vulkan_metrics",
                "parallel_metrics",
                "resource_metrics"
            ]
        )
