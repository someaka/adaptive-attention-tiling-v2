"""Vulkan-specific benchmarks for the Adaptive Attention Tiling system.

This module provides comprehensive benchmarks for Vulkan operations including:
1. Memory operations and transfer patterns
2. Resource management and pool efficiency
3. Lifecycle analysis and optimization
"""

import pytest
import time
import numpy as np
import vulkan as vk
import ctypes
from ctypes import c_void_p, c_float, byref, c_uint32, cast, POINTER, c_size_t, create_string_buffer

from src.core.benchmarks import BenchmarkMetrics
from src.core.vulkan import VulkanMemory, VulkanResources


class TestVulkanBenchmarks:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Create instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"VulkanBenchmarks",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName=b"No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )
        
        self.instance = c_void_p()
        result = vk.vkCreateInstance(byref(create_info), None, byref(self.instance))
        assert result == vk.VK_SUCCESS
        
        # Get physical device
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("Failed to find GPUs with Vulkan support")
        self.physical_device = devices[0]
        
        # Create device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=0,
            queueCount=1,
            pQueuePriorities=(c_float * 1)(1.0)
        )
        
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=queue_create_info
        )
        
        self.device = c_void_p()
        result = vk.vkCreateDevice(self.physical_device, byref(device_create_info), None, byref(self.device))
        assert result == vk.VK_SUCCESS
        
        # Initialize managers
        device_handle = int(cast(self.device, c_void_p).value or 0)
        physical_device_handle = int(cast(self.physical_device, c_void_p).value or 0)
        self.memory = VulkanMemory(device=device_handle, physical_device=physical_device_handle)
        self.resources = VulkanResources(device=device_handle, memory_manager=self.memory)
        self.sizes = [512, 1024, 2048, 4096]
        self.iterations = 10
        self.metrics = BenchmarkMetrics()

    def test_pool_efficiency(self):
        """Benchmark memory pool allocation efficiency."""
        for size in self.sizes:
            # Test different pool configurations
            pool_sizes = [size * 4, size * 8, size * 16]  # KB

            for pool_size in pool_sizes:
                # Create pool
                memory_type_bits = 0xFFFFFFFF  # All memory types
                properties = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                pool = self.memory.allocate(pool_size, memory_type_bits, properties)

                # Create command pool and buffer for timing
                command_pool_info = vk.VkCommandPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    queueFamilyIndex=0,
                    flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
                )
                command_pool = c_void_p()
                vk.vkCreateCommandPool(self.device, byref(command_pool_info), None, byref(command_pool))

                # Create query pool for timing
                query_pool_info = vk.VkQueryPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
                    queryCount=2
                )
                query_pool = c_void_p()
                vk.vkCreateQueryPool(self.device, byref(query_pool_info), None, byref(query_pool))

                # Measure allocation performance
                start_time = time.perf_counter()
                allocations = []
                for _ in range(10):
                    buffer = self.resources.create_buffer(
                        size=size,
                        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        memory_properties=properties
                    )
                    allocations.append(buffer)
                end_time = time.perf_counter()
                alloc_time = (end_time - start_time) * 1000  # Convert to milliseconds

                # Get pool stats
                stats = self.memory.get_stats()
                fragmentation = stats["fragmentation"]

                # Cleanup
                for buffer in allocations:
                    self.resources.destroy_buffer(buffer)

                self.metrics.record_operation(
                    name="pool_efficiency",
                    pool_size=pool_size,
                    allocation_size=size,
                    allocation_time=alloc_time,
                    fragmentation=fragmentation,
                )

                # Cleanup Vulkan resources
                vk.vkDestroyQueryPool(self.device, query_pool, None)
                vk.vkDestroyCommandPool(self.device, command_pool, None)
                self.memory.free(pool)

    def test_memory_operations(self):
        """Benchmark memory operations performance."""
        for size in self.sizes:
            # Create host-visible buffer
            memory_type_bits = 0xFFFFFFFF
            properties = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            
            buffer = self.resources.create_buffer(
                size=size * size * 4,  # float32 size
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                memory_properties=properties
            )

            # Map memory
            data_ptr = c_void_p()
            result = vk.vkMapMemory(self.device, buffer.memory.memory, buffer.memory.offset, buffer.size, data_ptr)
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to map memory: {result}")

            # Host to device transfer
            host_data = np.random.randn(size, size).astype(np.float32)
            start_time = time.perf_counter()
            # Copy data to mapped memory
            buffer_data = host_data.tobytes()
            ctypes.memmove(data_ptr, buffer_data, host_data.nbytes)
            end_time = time.perf_counter()
            h2d_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Device to host transfer
            start_time = time.perf_counter()
            # Copy data from mapped memory
            device_data = create_string_buffer(host_data.nbytes)
            ctypes.memmove(device_data, data_ptr, host_data.nbytes)
            end_time = time.perf_counter()
            d2h_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Unmap memory
            vk.vkUnmapMemory(self.device, buffer.memory.memory)

            # Cleanup
            self.resources.destroy_buffer(buffer)

            # Record metrics
            self.metrics.record_operation(
                name="memory_operations",
                size=size,
                h2d_time=h2d_time,
                d2h_time=d2h_time,
                bandwidth=size * size * 4 / (h2d_time + d2h_time),  # bytes/ms
            )

    def teardown_method(self):
        """Cleanup after each test method."""
        self.memory.cleanup()
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)
