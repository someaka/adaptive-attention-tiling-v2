"""Continuous benchmark monitoring system for Adaptive Attention Tiling.

This module provides a framework for continuous performance monitoring,
focusing on:
1. Automated benchmark execution
2. Performance regression detection
3. Resource utilization tracking
4. Historical trend analysis
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import psutil
import torch
import vulkan as vk
import ctypes


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    test_name: str
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    gpu_utilization: Optional[float]
    timestamp: datetime
    metrics: dict[str, float]


class BenchmarkMonitor:
    """Continuous benchmark monitoring system."""

    def __init__(
        self,
        results_dir: Union[str, Path],
        history_length: int = 100,
        alert_threshold: float = 0.2,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history_length = history_length
        self.alert_threshold = alert_threshold
        self.current_results: list[BenchmarkResult] = []

        # Set up logging
        logging.basicConfig(
            filename=self.results_dir / "benchmark.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vulkan resources
        self._init_vulkan()
        
    def _init_vulkan(self) -> None:
        """Initialize Vulkan instance, device and command buffer."""
        try:
            import vulkan as vk
            
            # Required extensions
            instance_extensions = [
                vk.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
            ]
            device_extensions = [
                vk.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
            ]
            
            # Create instance
            app_info = vk.VkApplicationInfo(
                pApplicationName="BenchmarkMonitor",
                applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                pEngineName="No Engine",
                engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                apiVersion=0x401000  # Vulkan 1.1
            )
            
            instance_info = vk.VkInstanceCreateInfo(
                pApplicationInfo=app_info,
                enabledExtensionCount=len(instance_extensions),
                ppEnabledExtensionNames=instance_extensions,
                enabledLayerCount=0,
                ppEnabledLayerNames=None
            )
            
            self.instance = vk.vkCreateInstance(instance_info, None)
            
            # Get physical device
            physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
            if not physical_devices:
                raise RuntimeError("No Vulkan devices found")
            self.physical_device = physical_devices[0]
            
            # Create logical device
            queue_info = vk.VkDeviceQueueCreateInfo(
                queueFamilyIndex=0,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            
            device_info = vk.VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=[queue_info],
                enabledExtensionCount=len(device_extensions),
                ppEnabledExtensionNames=device_extensions,
                enabledLayerCount=0,
                ppEnabledLayerNames=None
            )
            
            self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
            
            # Get queue
            self.queue = vk.vkGetDeviceQueue(self.device, 0, 0)
            
            # Create command pool
            pool_info = vk.VkCommandPoolCreateInfo(
                queueFamilyIndex=0,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
            )
            self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
            
            # Allocate command buffer
            alloc_info = vk.VkCommandBufferAllocateInfo(
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            self.command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Vulkan: {e}")
            self.instance = None
            self.device = None
            self.command_buffer = None
            
    def __del__(self):
        """Cleanup Vulkan resources."""
        try:
            if hasattr(self, 'device') and self.device is not None:
                import vulkan as vk
                if hasattr(self, 'command_pool'):
                    vk.vkDestroyCommandPool(self.device, self.command_pool, None)
                vk.vkDestroyDevice(self.device, None)
            if hasattr(self, 'instance') and self.instance is not None:
                vk.vkDestroyInstance(self.instance, None)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup Vulkan resources: {e}")

    def start_benchmark(self, test_name: str) -> None:
        """Start timing a benchmark."""
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        self.start_cpu = psutil.cpu_percent()
        self.current_test = test_name

        self.logger.info(f"Starting benchmark: {test_name}")

    def end_benchmark(
        self, additional_metrics: Optional[dict[str, float]] = None
    ) -> BenchmarkResult:
        """End timing and record benchmark results."""
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()

        # Calculate metrics
        execution_time = end_time - self.start_time
        memory_usage = (end_memory - self.start_memory) / 1024 / 1024  # MB
        cpu_utilization = (self.start_cpu + end_cpu) / 2

        # Get GPU utilization if available
        try:
            gpu_util = self.get_gpu_utilization()
        except (AttributeError, RuntimeError):
            gpu_util = None

        result = BenchmarkResult(
            test_name=self.current_test,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_utilization=cpu_utilization,
            gpu_utilization=gpu_util,
            timestamp=datetime.now(),
            metrics=additional_metrics or {},
        )

        self.current_results.append(result)
        self._save_result(result)
        self._check_regression(result)

        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to disk."""
        result_dict = {
            "test_name": result.test_name,
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "cpu_utilization": result.cpu_utilization,
            "gpu_utilization": result.gpu_utilization,
            "timestamp": result.timestamp.isoformat(),
            "metrics": result.metrics,
        }

        result_file = (
            self.results_dir
            / f"{result.test_name}_{result.timestamp:%Y%m%d_%H%M%S}.json"
        )
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2)

    def _check_regression(self, result: BenchmarkResult) -> None:
        """Check for performance regression."""
        # Load historical results
        historical_results = self._load_historical_results(result.test_name)

        if not historical_results:
            return

        # Calculate baseline statistics
        baseline_times = [r.execution_time for r in historical_results]
        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)

        # Check for regression
        z_score = (result.execution_time - baseline_mean) / (baseline_std + 1e-10)
        if z_score > self.alert_threshold:
            self.logger.warning(
                f"Performance regression detected in {result.test_name}:\n"
                f"Current time: {result.execution_time:.4f}s\n"
                f"Historical mean: {baseline_mean:.4f}s\n"
                f"Z-score: {z_score:.2f}"
            )

    def _load_historical_results(self, test_name: str) -> list[BenchmarkResult]:
        """Load historical benchmark results."""
        results = []
        result_files = sorted(
            self.results_dir.glob(f"{test_name}_*.json"), reverse=True
        )[: self.history_length]

        for file in result_files:
            with open(file) as f:
                data = json.load(f)
                result = BenchmarkResult(
                    test_name=data["test_name"],
                    execution_time=data["execution_time"],
                    memory_usage=data["memory_usage"],
                    cpu_utilization=data["cpu_utilization"],
                    gpu_utilization=data["gpu_utilization"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metrics=data["metrics"],
                )
                results.append(result)

        return results

    def generate_report(self) -> dict[str, dict[str, float]]:
        """Generate a summary report of all benchmarks."""
        report = {}

        for result in self.current_results:
            if result.test_name not in report:
                report[result.test_name] = {
                    "mean_time": result.execution_time,
                    "mean_memory": result.memory_usage,
                    "mean_cpu": result.cpu_utilization,
                    "count": 1,
                }
            else:
                stats = report[result.test_name]
                stats["mean_time"] = (
                    stats["mean_time"] * stats["count"] + result.execution_time
                ) / (stats["count"] + 1)
                stats["mean_memory"] = (
                    stats["mean_memory"] * stats["count"] + result.memory_usage
                ) / (stats["count"] + 1)
                stats["mean_cpu"] = (
                    stats["mean_cpu"] * stats["count"] + result.cpu_utilization
                ) / (stats["count"] + 1)
                stats["count"] += 1

        return report

    def plot_trends(self, test_name: str) -> None:
        """Plot performance trends for a specific test."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed. Cannot generate plots.")
            return

        results = self._load_historical_results(test_name)
        if not results:
            return

        timestamps = [r.timestamp for r in results]
        times = [r.execution_time for r in results]
        memories = [r.memory_usage for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Execution time trend
        ax1.plot(timestamps, times, "b-")
        ax1.set_title(f"{test_name} - Execution Time Trend")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Time (s)")
        ax1.grid(True)

        # Memory usage trend
        ax2.plot(timestamps, memories, "r-")
        ax2.set_title(f"{test_name} - Memory Usage Trend")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Memory (MB)")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / f"{test_name}_trends.png")
        plt.close()

    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage using Vulkan memory and queue tracking.
        
        Returns:
            float: GPU utilization percentage between 0 and 100.
            
        This implementation uses VK_EXT_device_memory_report for memory tracking
        and queue timestamps for workload tracking.
        """
        try:
            # Get memory utilization
            memory_util = self._get_vulkan_memory_utilization()
            
            # Get queue/compute utilization
            queue_util = self._get_vulkan_queue_utilization()
            
            # Combine metrics (weighted average)
            total_util = 0.7 * memory_util + 0.3 * queue_util
            return float(total_util)
            
        except Exception as e:
            self.logger.warning(f"Failed to get GPU utilization: {e}")
            return 0.0
            
    def _get_vulkan_memory_utilization(self) -> float:
        """Get Vulkan memory utilization using VK_EXT_device_memory_report.
        
        Returns:
            float: Memory utilization percentage (0-100)
        """
        try:
            # Get physical device memory properties
            memory_props = vk.VkPhysicalDeviceMemoryProperties()
            vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, memory_props)
            
            # Track heap utilization
            total_memory = 0
            used_memory = 0
            
            # Access memory heaps through the proper structure
            for i in range(memory_props.memoryHeapCount):
                heap = memory_props.memoryHeaps[i]
                if heap.flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
                    total_memory += heap.size
                    
                    # Get current allocation using memory report
                    # Note: We need to enable VK_EXT_device_memory_report extension first
                    memory_info = vk.VkPhysicalDeviceMemoryBudgetPropertiesEXT()
                    memory_info.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
                    memory_info.pNext = None
                    
                    device_props = vk.VkPhysicalDeviceProperties2()
                    device_props.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
                    device_props.pNext = memory_info
                    
                    vk.vkGetPhysicalDeviceProperties2(self.physical_device, device_props)
                    used_memory += memory_info.heapUsage[i]
                    
            return (used_memory / total_memory) * 100 if total_memory > 0 else 0
            
        except Exception as e:
            self.logger.debug(f"Memory utilization tracking failed: {e}")
            return 0.0
            
    def _get_vulkan_queue_utilization(self) -> float:
        """Get Vulkan queue utilization using timestamp queries.
        
        Returns:
            float: Queue utilization percentage (0-100)
        """
        try:
            # Create timestamp query pool
            query_pool_info = vk.VkQueryPoolCreateInfo(
                queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
                queryCount=2
            )
            query_pool = vk.vkCreateQueryPool(self.device, query_pool_info, None)
            
            # Reset query pool
            vk.vkCmdResetQueryPool(self.command_buffer, query_pool, 0, 2)
            
            # Write timestamps
            vk.vkCmdWriteTimestamp(
                self.command_buffer,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                query_pool,
                0
            )
            
            # Small delay to measure activity
            time.sleep(0.1)
            
            vk.vkCmdWriteTimestamp(
                self.command_buffer,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                query_pool,
                1
            )
            
            # Get results
            results = (ctypes.c_uint64 * 2)()
            vk.vkGetQueryPoolResults(
                self.device,
                query_pool,
                0,
                2,
                ctypes.sizeof(results),
                results,
                ctypes.sizeof(ctypes.c_uint64),
                vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT
            )
            
            # Calculate utilization based on timestamp delta
            delta_ns = results[1] - results[0]
            total_ns = 100_000_000  # 100ms in ns
            
            # Cleanup
            vk.vkDestroyQueryPool(self.device, query_pool, None)
            
            return (delta_ns / total_ns) * 100
            
        except Exception as e:
            self.logger.debug(f"Queue utilization tracking failed: {e}")
            return 0.0
