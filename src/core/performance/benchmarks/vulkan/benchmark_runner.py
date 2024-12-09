"""Vulkan Benchmark Runner.

This module provides tools for running and analyzing Vulkan benchmarks,
including memory operations and compute shader performance.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import vulkan as vk

from .memory_benchmarks import VulkanMemoryBenchmark


class VulkanBenchmarkRunner:
    """Runner for Vulkan benchmarks."""

    def __init__(
        self, instance: Optional[vk.Instance] = None, device: Optional[vk.Device] = None
    ):
        self.instance = instance or self._create_instance()
        self.device = device or self._create_device()

        # Create command pool and queue
        queue_family_index = self._find_queue_family()
        self.queue = vk.GetDeviceQueue(self.device, queue_family_index, 0)

        pool_info = vk.CommandPoolCreateInfo(
            queueFamilyIndex=queue_family_index,
            flags=vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.CreateCommandPool(self.device, pool_info, None)

        # Initialize benchmarks
        self.memory_benchmark = VulkanMemoryBenchmark(
            device=self.device,
            physical_device=self._get_physical_device(),
            command_pool=self.command_pool,
            queue=self.queue,
        )

    def _create_instance(self) -> vk.Instance:
        """Create Vulkan instance."""
        app_info = vk.ApplicationInfo(
            applicationName="Vulkan Benchmarks",
            applicationVersion=vk.MAKE_VERSION(1, 0, 0),
            engineName="No Engine",
            engineVersion=vk.MAKE_VERSION(1, 0, 0),
            apiVersion=vk.API_VERSION_1_0,
        )

        create_info = vk.InstanceCreateInfo(
            pApplicationInfo=app_info, enabledLayerNames=["VK_LAYER_KHRONOS_validation"]
        )

        return vk.CreateInstance(create_info, None)

    def _get_physical_device(self) -> vk.PhysicalDevice:
        """Get physical device."""
        return vk.EnumeratePhysicalDevices(self.instance)[0]

    def _find_queue_family(self) -> int:
        """Find suitable queue family index."""
        physical_device = self._get_physical_device()
        queue_families = vk.GetPhysicalDeviceQueueFamilyProperties(physical_device)

        for i, family in enumerate(queue_families):
            if family.queueFlags & vk.QUEUE_COMPUTE_BIT:
                return i

        raise RuntimeError("No compute queue family found")

    def _create_device(self) -> vk.Device:
        """Create logical device."""
        physical_device = self._get_physical_device()
        queue_family_index = self._find_queue_family()

        queue_info = vk.DeviceQueueCreateInfo(
            queueFamilyIndex=queue_family_index, queueCount=1, pQueuePriorities=[1.0]
        )

        device_features = vk.PhysicalDeviceFeatures()
        device_create_info = vk.DeviceCreateInfo(
            queueCreateInfos=[queue_info], enabledFeatures=device_features
        )

        return vk.CreateDevice(physical_device, device_create_info, None)

    def run_all_benchmarks(self, output_dir: str = "benchmark_results") -> None:
        """Run all Vulkan benchmarks."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(output_path / "benchmark.log"),
                logging.StreamHandler(),
            ],
        )

        try:
            logging.info("Starting Vulkan benchmarks...")

            # Run memory benchmarks
            self.memory_benchmark.run_and_report(str(output_path))

            # Save benchmark metadata
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "vulkan_version": vk.API_VERSION_1_0,
                "device_name": vk.GetPhysicalDeviceProperties(
                    self._get_physical_device()
                ).deviceName.decode("utf-8"),
            }

            with open(output_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logging.info("All benchmarks completed successfully.")

        except Exception as e:
            logging.error(f"Benchmark failed: {e!s}", exc_info=True)
            raise

    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        vk.DestroyCommandPool(self.device, self.command_pool, None)
        vk.DestroyDevice(self.device, None)
        vk.DestroyInstance(self.instance, None)


def run_benchmarks(output_dir: str = "benchmark_results") -> None:
    """Convenience function to run all Vulkan benchmarks."""
    runner = VulkanBenchmarkRunner()
    try:
        runner.run_all_benchmarks(output_dir)
    finally:
        runner.cleanup()
