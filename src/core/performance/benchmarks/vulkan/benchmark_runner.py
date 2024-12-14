"""Vulkan Benchmark Runner.

This module provides tools for running and analyzing Vulkan benchmarks,
including memory operations and compute shader performance.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Optional, List
from ctypes import c_void_p, c_uint32, c_int, Structure, POINTER, byref, c_char_p, c_float, c_bool, Array

import vulkan as vk

from .memory_benchmarks import VulkanMemoryBenchmark

# Vulkan type definitions
VkInstance = c_void_p
VkPhysicalDevice = c_void_p
VkDevice = c_void_p
VkQueue = c_void_p
VkCommandPool = c_void_p

# Vulkan constants
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39

VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
VK_QUEUE_COMPUTE_BIT = 0x00000002

class VkExtent3D(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("depth", c_uint32)
    ]

class VkQueueFamilyProperties(Structure):
    _fields_ = [
        ("queueFlags", c_uint32),
        ("queueCount", c_uint32),
        ("timestampValidBits", c_uint32),
        ("minImageTransferGranularity", VkExtent3D)
    ]

class VkApplicationInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("pApplicationName", c_char_p),
        ("applicationVersion", c_uint32),
        ("pEngineName", c_char_p),
        ("engineVersion", c_uint32),
        ("apiVersion", c_uint32)
    ]

class VkInstanceCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pApplicationInfo", POINTER(VkApplicationInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p))
    ]

class VkDeviceQueueCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueFamilyIndex", c_uint32),
        ("queueCount", c_uint32),
        ("pQueuePriorities", POINTER(c_float))
    ]

class VkPhysicalDeviceFeatures(Structure):
    _fields_ = [
        ("robustBufferAccess", c_bool),
        # Add other feature fields as needed
    ]

class VkDeviceCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueCreateInfoCount", c_uint32),
        ("pQueueCreateInfos", POINTER(VkDeviceQueueCreateInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
        ("pEnabledFeatures", POINTER(VkPhysicalDeviceFeatures))
    ]

class VkCommandPoolCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueFamilyIndex", c_uint32)
    ]

class VulkanBenchmarkRunner:
    """Runner for Vulkan benchmarks."""

    def __init__(
        self, instance: Optional[VkInstance] = None, device: Optional[VkDevice] = None
    ):
        self.instance = instance or self._create_instance()
        self.device = device or self._create_device()

        # Create command pool and queue
        queue_family_index = self._find_queue_family()
        self.queue = c_void_p()
        vk.vkGetDeviceQueue(self.device, queue_family_index, 0, byref(self.queue))

        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext=None,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=queue_family_index
        )
        
        self.command_pool = c_void_p()
        result = vk.vkCreateCommandPool(self.device, byref(pool_info), None, byref(self.command_pool))
        if result != 0:
            raise RuntimeError(f"Failed to create command pool: {result}")

        # Initialize benchmarks
        self.memory_benchmark = VulkanMemoryBenchmark(
            device=self.device,
            physical_device=self._get_physical_device(),
            command_pool=self.command_pool,
            queue=self.queue,
        )

    def _create_instance(self) -> VkInstance:
        """Create Vulkan instance."""
        app_name = b"Vulkan Benchmarks"
        engine_name = b"No Engine"
        
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext=None,
            pApplicationName=app_name,
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName=engine_name,
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 0, 0)
        )

        layer_name = b"VK_LAYER_KHRONOS_validation"
        layer_names = (c_char_p * 1)(layer_name)
        
        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=None,
            flags=0,
            pApplicationInfo=byref(app_info),
            enabledLayerCount=1,
            ppEnabledLayerNames=layer_names,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None
        )

        instance = c_void_p()
        result = vk.vkCreateInstance(byref(create_info), None, byref(instance))
        if result != 0:
            raise RuntimeError(f"Failed to create instance: {result}")
        
        return instance

    def _get_physical_device(self) -> VkPhysicalDevice:
        """Get physical device."""
        # Get devices directly
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan physical devices found")
        
        return devices[0]

    def _find_queue_family(self) -> int:
        """Find suitable queue family index."""
        physical_device = self._get_physical_device()
        
        # Get queue families directly
        properties = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)

        for i, family in enumerate(properties):
            if family.queueFlags & VK_QUEUE_COMPUTE_BIT:
                return i

        raise RuntimeError("No compute queue family found")

    def _create_device(self) -> VkDevice:
        """Create logical device."""
        physical_device = self._get_physical_device()
        queue_family_index = self._find_queue_family()

        priorities = (c_float * 1)(1.0)
        queue_info = VkDeviceQueueCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext=None,
            flags=0,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=priorities
        )

        device_features = VkPhysicalDeviceFeatures()
        queue_infos = (VkDeviceQueueCreateInfo * 1)(queue_info)
        
        device_create_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=None,
            flags=0,
            queueCreateInfoCount=1,
            pQueueCreateInfos=queue_infos,
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
            pEnabledFeatures=byref(device_features)
        )

        device = c_void_p()
        result = vk.vkCreateDevice(physical_device, byref(device_create_info), None, byref(device))
        if result != 0:
            raise RuntimeError(f"Failed to create device: {result}")
        
        return device

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
            physical_device = self._get_physical_device()
            props = vk.VkPhysicalDeviceProperties()
            vk.vkGetPhysicalDeviceProperties(physical_device, byref(props))

            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "vulkan_version": vk.VK_API_VERSION_1_0,
                "device_name": props.deviceName.decode("utf-8"),
            }

            with open(output_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logging.info("All benchmarks completed successfully.")

        except Exception as e:
            logging.error(f"Benchmark failed: {e!s}", exc_info=True)
            raise

    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)


def run_benchmarks(output_dir: str = "benchmark_results") -> None:
    """Convenience function to run all Vulkan benchmarks."""
    runner = VulkanBenchmarkRunner()
    try:
        runner.run_all_benchmarks(output_dir)
    finally:
        runner.cleanup()
