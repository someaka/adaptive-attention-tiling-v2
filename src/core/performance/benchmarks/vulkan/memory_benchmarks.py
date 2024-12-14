"""Vulkan Memory Management Benchmarks.

This module provides comprehensive benchmarks for Vulkan memory operations,
including transfer speeds, allocation patterns, and memory barrier overhead.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from ctypes import c_void_p, c_uint32, c_int, Structure, POINTER, byref, c_char_p, c_size_t

import matplotlib.pyplot as plt
import numpy as np
import vulkan as vk

from ....performance.vulkan.memory.barrier_manager import AccessPattern, BarrierManager
from ....performance.vulkan.memory.buffer_manager import BufferManager
from ....performance.vulkan.memory.memory_pool import MemoryPoolManager

# Vulkan type definitions
VkDevice = c_void_p
VkPhysicalDevice = c_void_p
VkCommandPool = c_void_p
VkQueue = c_void_p
VkBuffer = c_void_p
VkFence = c_void_p
VkCommandBuffer = c_void_p

# Vulkan constants
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
VK_STRUCTURE_TYPE_SUBMIT_INFO = 4

VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001

class VkCommandBufferAllocateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("commandPool", c_void_p),
        ("level", c_uint32),
        ("commandBufferCount", c_uint32)
    ]

class VkCommandBufferBeginInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pInheritanceInfo", c_void_p)
    ]

class VkSubmitInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("waitSemaphoreCount", c_uint32),
        ("pWaitSemaphores", c_void_p),
        ("pWaitDstStageMask", POINTER(c_uint32)),
        ("commandBufferCount", c_uint32),
        ("pCommandBuffers", POINTER(c_void_p)),
        ("signalSemaphoreCount", c_uint32),
        ("pSignalSemaphores", c_void_p)
    ]

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    metrics: Dict[str, float]
    timestamps: List[float]

    def plot(self, output_dir: str = "benchmark_results") -> None:
        """Plot benchmark results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        for metric, value in self.metrics.items():
            if isinstance(value, (int, float)):
                plt.plot(
                    self.timestamps,
                    [value] * len(self.timestamps),
                    label=f"{metric}: {value:.2f}",
                )

        plt.title(f"Benchmark Results: {self.name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{output_dir}/{self.name.lower().replace(" ", "_")}.png')
        plt.close()


class VulkanMemoryBenchmark:
    """Benchmarks for Vulkan memory operations."""

    def __init__(
        self,
        device: VkDevice,
        physical_device: VkPhysicalDevice,
        command_pool: VkCommandPool,
        queue: VkQueue,
    ):
        self.device = device
        self.physical_device = physical_device
        self.command_pool = command_pool
        self.queue = queue

        # Initialize managers
        self.buffer_manager = BufferManager(device, physical_device)
        self.barrier_manager = BarrierManager()
        self.memory_pool = MemoryPoolManager(device, physical_device)

        # Benchmark parameters
        self.sizes = [
            1024,  # 1KB
            1024 * 1024,  # 1MB
            10 * 1024 * 1024,  # 10MB
            100 * 1024 * 1024,  # 100MB
        ]
        self.iterations = 100

    def _measure_transfer_speed(self, size: int) -> Tuple[float, float]:
        """Measure host to device transfer speed."""
        # Create test data
        data = np.random.rand(size // 8).astype(np.float64)

        # Create device buffer
        device_buffer = self.buffer_manager.create_buffer(
            size=size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        # Measure transfer time
        start_time = time.perf_counter()
        self.buffer_manager.copy_to_device(
            data=data,
            device_buffer=device_buffer,
            command_pool=self.command_pool,
            queue=self.queue,
        )
        end_time = time.perf_counter()

        transfer_time = end_time - start_time
        transfer_speed = size / (transfer_time * 1024 * 1024)  # MB/s

        return transfer_time, transfer_speed

    def _measure_allocation_overhead(self, size: int) -> Tuple[float, int]:
        """Measure memory allocation overhead and fragmentation."""
        start_time = time.perf_counter()

        # Perform multiple allocations
        blocks = []
        for _ in range(10):
            block = self.memory_pool.allocate(
                size=size, memory_type_index=0  # Assuming first memory type
            )
            blocks.append(block)

        # Free half of the blocks
        for block in blocks[:5]:
            self.memory_pool.free(block)

        # Allocate new blocks
        for _ in range(5):
            block = self.memory_pool.allocate(size=size, memory_type_index=0)
            blocks.append(block)

        end_time = time.perf_counter()
        allocation_time = end_time - start_time

        # Calculate fragmentation
        total_size = sum(block.size for block in blocks)
        used_size = sum(block.size for block in blocks if not block.is_free)
        fragmentation = (total_size - used_size) / total_size * 100

        return allocation_time, int(fragmentation)

    def _measure_barrier_overhead(self) -> float:
        """Measure memory barrier insertion overhead."""
        # Create command buffer
        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        command_buffer = c_void_p()
        result = vk.vkAllocateCommandBuffers(self.device, byref(alloc_info), byref(command_buffer))
        if result != 0:
            raise RuntimeError(f"Failed to allocate command buffer: {result}")

        # Begin command buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None
        )
        
        result = vk.vkBeginCommandBuffer(command_buffer, byref(begin_info))
        if result != 0:
            raise RuntimeError(f"Failed to begin command buffer: {result}")

        # Measure barrier insertion time
        start_time = time.perf_counter()

        for _ in range(100):
            self.barrier_manager.global_barrier(
                command_buffer=command_buffer, pattern=AccessPattern.COMPUTE_SHADER
            )

        end_time = time.perf_counter()

        # End and submit command buffer
        result = vk.vkEndCommandBuffer(command_buffer)
        if result != 0:
            raise RuntimeError(f"Failed to end command buffer: {result}")

        command_buffers = (c_void_p * 1)(command_buffer)
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=None,
            waitSemaphoreCount=0,
            pWaitSemaphores=None,
            pWaitDstStageMask=None,
            commandBufferCount=1,
            pCommandBuffers=command_buffers,
            signalSemaphoreCount=0,
            pSignalSemaphores=None
        )

        result = vk.vkQueueSubmit(self.queue, 1, byref(submit_info), None)
        if result != 0:
            raise RuntimeError(f"Failed to submit queue: {result}")

        vk.vkQueueWaitIdle(self.queue)

        # Cleanup
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, command_buffers)

        return (end_time - start_time) / 100  # Average time per barrier

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all memory benchmarks."""
        results = []
        timestamps = []

        # Transfer speed benchmarks
        for size in self.sizes:
            transfer_times = []
            transfer_speeds = []

            for _ in range(self.iterations):
                time_taken, speed = self._measure_transfer_speed(size)
                transfer_times.append(time_taken)
                transfer_speeds.append(speed)
                timestamps.append(time.perf_counter())

            results.append(
                BenchmarkResult(
                    name=f"Transfer Speed {size/1024/1024}MB",
                    metrics={
                        "avg_time_ms": float(np.mean(transfer_times) * 1000),
                        "avg_speed_mbps": float(np.mean(transfer_speeds)),
                        "min_speed_mbps": float(np.min(transfer_speeds)),
                        "max_speed_mbps": float(np.max(transfer_speeds)),
                    },
                    timestamps=timestamps,
                )
            )

        # Allocation overhead benchmarks
        for size in self.sizes:
            alloc_times = []
            fragmentations = []

            for _ in range(self.iterations):
                time_taken, fragmentation = self._measure_allocation_overhead(size)
                alloc_times.append(time_taken)
                fragmentations.append(fragmentation)
                timestamps.append(time.perf_counter())

            results.append(
                BenchmarkResult(
                    name=f"Allocation Overhead {size/1024/1024}MB",
                    metrics={
                        "avg_time_ms": float(np.mean(alloc_times) * 1000),
                        "avg_fragmentation": float(np.mean(fragmentations)),
                        "max_fragmentation": float(np.max(fragmentations)),
                    },
                    timestamps=timestamps,
                )
            )

        # Barrier overhead benchmarks
        barrier_times = []
        for _ in range(self.iterations):
            barrier_time = self._measure_barrier_overhead()
            barrier_times.append(barrier_time)
            timestamps.append(time.perf_counter())

        results.append(
            BenchmarkResult(
                name="Barrier Overhead",
                metrics={
                    "avg_time_us": float(np.mean(barrier_times) * 1_000_000),
                    "min_time_us": float(np.min(barrier_times) * 1_000_000),
                    "max_time_us": float(np.max(barrier_times) * 1_000_000),
                },
                timestamps=timestamps,
            )
        )

        return results

    def run_and_report(self, output_dir: str = "benchmark_results") -> None:
        """Run benchmarks and generate reports."""
        results = self.run_benchmarks()
        for result in results:
            result.plot(output_dir)
            logging.info(f"Benchmark: {result.name}")
            for metric, value in result.metrics.items():
                logging.info(f"  {metric}: {value:.2f}")
