"""Performance tests for Vulkan synchronization primitives.

This module tests the performance characteristics of various Vulkan synchronization
mechanisms in the Adaptive Attention Tiling system, focusing on:
1. Fence operations
2. Semaphore operations
3. Event operations
4. Memory barriers and pipeline barriers
5. Queue synchronization
"""

import time
from ctypes import c_void_p, c_char_p, byref, c_float, c_uint32

import pytest
import torch
import vulkan as vk

from src.core.performance.vulkan.sync import VulkanSync


class TestVulkanSync:
    """Test suite for Vulkan synchronization primitives performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        assert torch.is_vulkan_available(), "Vulkan is not available"

        # Create instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"VulkanSyncTest",
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
        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not physical_devices:
            raise RuntimeError("No Vulkan devices found")
        self.physical_device = physical_devices[0]

        # Find queue family
        queue_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        self.queue_family_index = next(
            i for i, prop in enumerate(queue_props)
            if prop.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
        )

        # Create device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
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

        # Get queue
        self.queue = c_void_p()
        vk.vkGetDeviceQueue(self.device, self.queue_family_index, 0, byref(self.queue))

        # Create sync manager
        self.sync = VulkanSync(device=self.device, enable_profiling=True)
        self.sync.set_queue(self.queue)

        self.sizes = [512, 1024, 2048]
        self.iterations = 10

    def teardown_method(self):
        """Cleanup after each test."""
        if hasattr(self, 'sync'):
            self.sync.cleanup()
        if hasattr(self, 'device'):
            vk.vkDestroyDevice(self.device, None)
        if hasattr(self, 'instance'):
            vk.vkDestroyInstance(self.instance, None)

    def test_fence_performance(self):
        """Test fence creation and wait performance."""
        fence_counts = [1, 10, 100]

        for count in fence_counts:
            # Create fences
            start = time.time()
            fences = [self.sync.create_fence() for _ in range(count)]
            creation_time = time.time() - start

            # Signal and wait
            start = time.time()
            for fence in fences:
                self.sync.signal_fence(fence)
            signal_time = time.time() - start

            start = time.time()
            for fence in fences:
                self.sync.wait_fence(fence)
            wait_time = time.time() - start

            print(f"\nFence Test ({count} fences)")
            print(f"Creation time: {creation_time:.4f}s")
            print(f"Signal time: {signal_time:.4f}s")
            print(f"Wait time: {wait_time:.4f}s")

            # Cleanup
            for fence in fences:
                self.sync.destroy_fence(fence)

            # Performance assertions
            assert creation_time < 0.001 * count, "Fence creation too slow"
            assert signal_time < 0.001 * count, "Fence signaling too slow"
            assert wait_time < 0.001 * count, "Fence waiting too slow"

    def test_semaphore_performance(self):
        """Test semaphore operations performance."""
        for size in self.sizes:
            data = torch.randn(size, size)

            # Test binary semaphore
            times = []
            for _ in range(self.iterations):
                start = time.time()
                semaphore = self.sync.create_binary_semaphore()
                self.sync.queue_submit(
                    data, wait_semaphore=None, signal_semaphore=semaphore
                )
                self.sync.queue_submit(
                    data, wait_semaphore=semaphore, signal_semaphore=None
                )
                self.sync.wait_idle()
                times.append(time.time() - start)
                self.sync.destroy_semaphore(semaphore)

            avg_binary_time = sum(times) / len(times)

            # Test timeline semaphore
            times = []
            for _ in range(self.iterations):
                start = time.time()
                semaphore = self.sync.create_timeline_semaphore()
                self.sync.queue_submit(data, timeline_value=1)
                self.sync.wait_semaphore(semaphore, 1)
                times.append(time.time() - start)
                self.sync.destroy_semaphore(semaphore)

            avg_timeline_time = sum(times) / len(times)

            print(f"\nSemaphore Test {size}x{size}")
            print(f"Binary semaphore avg time: {avg_binary_time:.4f}s")
            print(f"Timeline semaphore avg time: {avg_timeline_time:.4f}s")

            assert (
                avg_timeline_time < avg_binary_time * 1.5
            ), "Timeline semaphores should be competitive"

    def test_event_performance(self):
        """Test event operations performance."""
        event_counts = [1, 10, 100]

        for count in event_counts:
            # Create events
            start = time.time()
            events = [self.sync.create_event() for _ in range(count)]
            creation_time = time.time() - start

            # Set events
            start = time.time()
            for event in events:
                self.sync.set_event(event)
            set_time = time.time() - start

            # Reset events
            start = time.time()
            for event in events:
                self.sync.reset_event(event)
            reset_time = time.time() - start

            print(f"\nEvent Test ({count} events)")
            print(f"Creation time: {creation_time:.4f}s")
            print(f"Set time: {set_time:.4f}s")
            print(f"Reset time: {reset_time:.4f}s")

            # Cleanup
            for event in events:
                self.sync.destroy_event(event)

            # Performance assertions
            assert creation_time < 0.001 * count, "Event creation too slow"
            assert set_time < 0.0005 * count, "Event setting too slow"
            assert reset_time < 0.0005 * count, "Event reset too slow"

    def test_barrier_overhead(self):
        """Test memory and pipeline barrier overhead."""
        for size in self.sizes:
            data = torch.randn(size, size)

            # Test without barriers
            start = time.time()
            for _ in range(self.iterations):
                result = self.sync.compute_operation(data)
                result = self.sync.compute_operation(result)
                self.sync.wait_idle()
            no_barrier_time = time.time() - start

            # Test with memory barriers
            start = time.time()
            for _ in range(self.iterations):
                result = self.sync.compute_operation(data)
                self.sync.memory_barrier()
                result = self.sync.compute_operation(result)
                self.sync.wait_idle()
            memory_barrier_time = time.time() - start

            # Test with pipeline barriers
            start = time.time()
            for _ in range(self.iterations):
                result = self.sync.compute_operation(data)
                self.sync.pipeline_barrier()
                result = self.sync.compute_operation(result)
                self.sync.wait_idle()
            pipeline_barrier_time = time.time() - start

            print(f"\nBarrier Test {size}x{size}")
            print(f"No barriers: {no_barrier_time:.4f}s")
            print(f"Memory barriers: {memory_barrier_time:.4f}s")
            print(f"Pipeline barriers: {pipeline_barrier_time:.4f}s")

            # Calculate overhead
            memory_overhead = (
                (memory_barrier_time - no_barrier_time) / no_barrier_time * 100
            )
            pipeline_overhead = (
                (pipeline_barrier_time - no_barrier_time) / no_barrier_time * 100
            )

            print(f"Memory barrier overhead: {memory_overhead:.1f}%")
            print(f"Pipeline barrier overhead: {pipeline_overhead:.1f}%")

            assert memory_overhead < 50, "Memory barrier overhead too high"
            assert pipeline_overhead < 75, "Pipeline barrier overhead too high"

    def test_queue_sync(self):
        """Test queue synchronization performance."""
        for size in self.sizes:
            data = torch.randn(size, size)

            # Test single queue
            start = time.time()
            for _ in range(self.iterations):
                self.sync.compute_operation(data)
                self.sync.wait_idle()
            single_queue_time = time.time() - start

            # Test multiple queues with synchronization
            start = time.time()
            for _ in range(self.iterations):
                semaphore = self.sync.create_binary_semaphore()
                self.sync.queue_submit(data, queue_index=0, signal_semaphore=semaphore)
                self.sync.queue_submit(data, queue_index=1, wait_semaphore=semaphore)
                self.sync.wait_idle()
                self.sync.destroy_semaphore(semaphore)
            multi_queue_time = time.time() - start

            print(f"\nQueue Sync Test {size}x{size}")
            print(f"Single queue time: {single_queue_time:.4f}s")
            print(f"Multi queue time: {multi_queue_time:.4f}s")

            # Calculate overhead
            sync_overhead = (
                (multi_queue_time - single_queue_time) / single_queue_time * 100
            )
            print(f"Queue sync overhead: {sync_overhead:.1f}%")

            assert sync_overhead < 100, "Queue synchronization overhead too high"
