"""Vulkan Memory Management Benchmarks.

This module provides comprehensive benchmarks for Vulkan memory operations,
including transfer speeds, allocation patterns, and memory barrier overhead.
"""

import vulkan as vk
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from ...vulkan.memory.buffer_manager import BufferManager
from ...vulkan.memory.barrier_manager import BarrierManager, AccessPattern
from ...vulkan.memory.memory_pool import MemoryPoolManager

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    metrics: Dict[str, float]
    timestamps: List[float]
    
    def plot(self, output_dir: str = 'benchmark_results') -> None:
        """Plot benchmark results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        for metric, value in self.metrics.items():
            if isinstance(value, (int, float)):
                plt.plot(self.timestamps, [value] * len(self.timestamps), 
                        label=f'{metric}: {value:.2f}')
        
        plt.title(f'Benchmark Results: {self.name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'{output_dir}/{self.name.lower().replace(" ", "_")}.png')
        plt.close()

class VulkanMemoryBenchmark:
    """Benchmarks for Vulkan memory operations."""
    
    def __init__(self, 
                 device: vk.Device,
                 physical_device: vk.PhysicalDevice,
                 command_pool: vk.CommandPool,
                 queue: vk.Queue):
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
            1024,           # 1KB
            1024 * 1024,    # 1MB
            10 * 1024 * 1024,  # 10MB
            100 * 1024 * 1024  # 100MB
        ]
        self.iterations = 100
    
    def _measure_transfer_speed(self, size: int) -> Tuple[float, float]:
        """Measure host to device transfer speed."""
        # Create test data
        data = np.random.rand(size // 8).astype(np.float64)
        
        # Create device buffer
        device_buffer = self.buffer_manager.create_buffer(
            size=size,
            usage=vk.BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.BUFFER_USAGE_TRANSFER_DST_BIT,
            properties=vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        
        # Measure transfer time
        start_time = time.perf_counter()
        self.buffer_manager.copy_to_device(
            data=data,
            device_buffer=device_buffer,
            command_pool=self.command_pool,
            queue=self.queue
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
                size=size,
                memory_type_index=0  # Assuming first memory type
            )
            blocks.append(block)
        
        # Free half of the blocks
        for block in blocks[:5]:
            self.memory_pool.free(block)
        
        # Allocate new blocks
        for _ in range(5):
            block = self.memory_pool.allocate(
                size=size,
                memory_type_index=0
            )
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
        alloc_info = vk.CommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        command_buffer = vk.AllocateCommandBuffers(self.device, alloc_info)[0]
        
        # Begin command buffer
        begin_info = vk.CommandBufferBeginInfo(
            flags=vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.BeginCommandBuffer(command_buffer, begin_info)
        
        # Measure barrier insertion time
        start_time = time.perf_counter()
        
        for _ in range(100):
            self.barrier_manager.global_barrier(
                command_buffer=command_buffer,
                pattern=AccessPattern.COMPUTE_SHADER
            )
        
        end_time = time.perf_counter()
        
        # End and submit command buffer
        vk.EndCommandBuffer(command_buffer)
        submit_info = vk.SubmitInfo(commandBuffers=[command_buffer])
        vk.QueueSubmit(self.queue, 1, submit_info, vk.Fence(0))
        vk.QueueWaitIdle(self.queue)
        
        # Cleanup
        vk.FreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])
        
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
            
            results.append(BenchmarkResult(
                name=f'Transfer Speed {size/1024/1024}MB',
                metrics={
                    'avg_time_ms': np.mean(transfer_times) * 1000,
                    'avg_speed_mbps': np.mean(transfer_speeds),
                    'min_speed_mbps': np.min(transfer_speeds),
                    'max_speed_mbps': np.max(transfer_speeds)
                },
                timestamps=timestamps
            ))
        
        # Allocation overhead benchmarks
        for size in self.sizes:
            alloc_times = []
            fragmentations = []
            
            for _ in range(self.iterations):
                time_taken, fragmentation = self._measure_allocation_overhead(size)
                alloc_times.append(time_taken)
                fragmentations.append(fragmentation)
                timestamps.append(time.perf_counter())
            
            results.append(BenchmarkResult(
                name=f'Allocation Overhead {size/1024/1024}MB',
                metrics={
                    'avg_time_ms': np.mean(alloc_times) * 1000,
                    'avg_fragmentation': np.mean(fragmentations),
                    'max_fragmentation': np.max(fragmentations)
                },
                timestamps=timestamps
            ))
        
        # Barrier overhead benchmarks
        barrier_times = []
        for _ in range(self.iterations):
            barrier_time = self._measure_barrier_overhead()
            barrier_times.append(barrier_time)
            timestamps.append(time.perf_counter())
        
        results.append(BenchmarkResult(
            name='Barrier Overhead',
            metrics={
                'avg_time_us': np.mean(barrier_times) * 1_000_000,
                'min_time_us': np.min(barrier_times) * 1_000_000,
                'max_time_us': np.max(barrier_times) * 1_000_000
            },
            timestamps=timestamps
        ))
        
        return results
    
    def run_and_report(self, output_dir: str = 'benchmark_results') -> None:
        """Run benchmarks and generate reports."""
        logging.info("Starting Vulkan memory benchmarks...")
        
        results = self.run_benchmarks()
        
        # Generate plots
        for result in results:
            result.plot(output_dir)
            
            # Log results
            logging.info(f"\nBenchmark: {result.name}")
            for metric, value in result.metrics.items():
                logging.info(f"{metric}: {value}")
        
        logging.info("Vulkan memory benchmarks completed.")
