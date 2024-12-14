"""Vulkan device management and initialization."""

from dataclasses import dataclass
from typing import Optional, List
from ctypes import (
    c_char_p, c_uint32, c_void_p, POINTER, Structure, c_int, c_uint64, 
    c_char, byref, cast, c_float, Array, pointer
)

import vulkan as vk

# Constants
VK_MAX_MEMORY_TYPES = 32
VK_MAX_MEMORY_HEAPS = 16
VK_MAX_EXTENSION_NAME_SIZE = 256
VK_MAX_DESCRIPTION_SIZE = 256


class VkMemoryType(Structure):
    """Vulkan memory type structure."""
    _fields_ = [
        ("propertyFlags", c_int),
        ("heapIndex", c_uint32),
    ]


class VkMemoryHeap(Structure):
    """Vulkan memory heap structure."""
    _fields_ = [
        ("size", c_uint64),
        ("flags", c_int),
    ]


class VkPhysicalDeviceMemoryProperties(Structure):
    """Vulkan physical device memory properties structure."""
    _fields_ = [
        ("memoryTypeCount", c_uint32),
        ("memoryTypes", VkMemoryType * 32),  # VK_MAX_MEMORY_TYPES
        ("memoryHeapCount", c_uint32),
        ("memoryHeaps", VkMemoryHeap * 16),  # VK_MAX_MEMORY_HEAPS
    ]


class VkQueueFamilyProperties(Structure):
    """Vulkan queue family properties structure."""
    _fields_ = [
        ("queueFlags", c_uint32),
        ("queueCount", c_uint32),
        ("timestampValidBits", c_uint32),
        ("minImageTransferGranularity", c_uint32 * 3),  # VkExtent3D
    ]


class VkLayerProperties(Structure):
    """Vulkan layer properties structure."""
    _fields_ = [
        ("layerName", c_char * 256),  # VK_MAX_EXTENSION_NAME_SIZE
        ("specVersion", c_uint32),
        ("implementationVersion", c_uint32),
        ("description", c_char * 256),  # VK_MAX_DESCRIPTION_SIZE
    ]


class VkApplicationInfo(Structure):
    """Vulkan application info structure."""
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("pApplicationName", c_char_p),
        ("applicationVersion", c_uint32),
        ("pEngineName", c_char_p),
        ("engineVersion", c_uint32),
        ("apiVersion", c_uint32),
    ]


class VkInstanceCreateInfo(Structure):
    """Vulkan instance create info structure."""
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pApplicationInfo", POINTER(VkApplicationInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
    ]


class VkDeviceQueueCreateInfo(Structure):
    """Vulkan device queue create info structure."""
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueFamilyIndex", c_uint32),
        ("queueCount", c_uint32),
        ("pQueuePriorities", POINTER(c_float)),
    ]


class VkPhysicalDeviceFeatures(Structure):
    """Vulkan physical device features structure."""
    _fields_ = [
        ("robustBufferAccess", c_uint32),
        # Add other feature fields as needed
    ]


class VkDeviceCreateInfo(Structure):
    """Vulkan device create info structure."""
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
        ("pEnabledFeatures", POINTER(VkPhysicalDeviceFeatures)),
    ]


@dataclass
class QueueFamilyIndices:
    """Queue family indices for required queues."""

    compute: Optional[int] = None
    transfer: Optional[int] = None

    def is_complete(self) -> bool:
        """Check if all required queues are available."""
        return self.compute is not None and self.transfer is not None


class VulkanDevice:
    """Manages Vulkan device initialization and queues."""

    def __init__(self):
        self.instance = None
        self.physical_device = None
        self.device = None
        self.compute_queue = None
        self.transfer_queue = None
        self.queue_family_indices = QueueFamilyIndices()

    def initialize(self) -> bool:
        """Initialize Vulkan device and queues."""
        try:
            self._create_instance()
            self._select_physical_device()
            self._create_logical_device()
            return True
        except Exception as e:
            print(f"Failed to initialize Vulkan: {e}")
            self.cleanup()
            return False

    def _create_instance(self):
        """Create Vulkan instance."""
        app_name = b"Adaptive Attention Tiling"
        engine_name = b"No Engine"

        app_info = VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext=None,
            pApplicationName=app_name,
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName=engine_name,
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0)  # VK_API_VERSION_1_2
        )

        layers = []
        if self._check_validation_layer_support():
            layers = [b"VK_LAYER_KHRONOS_validation"]

        layer_count = len(layers)
        layer_names = (c_char_p * layer_count)(*layers) if layer_count > 0 else None

        create_info = VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=None,
            flags=0,
            pApplicationInfo=byref(app_info),
            enabledLayerCount=layer_count,
            ppEnabledLayerNames=layer_names,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None
        )

        instance = c_void_p()
        result = vk.vkCreateInstance(byref(create_info), None, byref(instance))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create instance: {result}")
        
        self.instance = instance

    def _select_physical_device(self):
        """Select suitable physical device."""
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("Failed to find GPUs with Vulkan support")

        # Select first suitable device
        for device in devices:
            if self._is_device_suitable(device):
                self.physical_device = device
                self.queue_family_indices = self._find_queue_families(device)
                break

        if not self.physical_device:
            raise RuntimeError("Failed to find a suitable GPU")

    def _create_logical_device(self):
        """Create logical device and queues."""
        indices = self.queue_family_indices
        if not indices.is_complete():
            raise RuntimeError("Queue family indices not initialized")

        unique_queue_families = {indices.compute, indices.transfer}
        queue_create_infos = []
        queue_priorities = (c_float * 1)(1.0)

        for queue_family in unique_queue_families:
            queue_info = VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext=None,
                flags=0,
                queueFamilyIndex=queue_family,
                queueCount=1,
                pQueuePriorities=queue_priorities
            )
            queue_create_infos.append(queue_info)

        features = VkPhysicalDeviceFeatures()
        queue_create_array = (VkDeviceQueueCreateInfo * len(queue_create_infos))(*queue_create_infos)
        
        create_info = VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=None,
            flags=0,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_array,
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
            pEnabledFeatures=byref(features)
        )

        device = c_void_p()
        result = vk.vkCreateDevice(self.physical_device, byref(create_info), None, byref(device))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create logical device: {result}")

        self.device = device

        # Get queue handles
        compute_queue = c_void_p()
        transfer_queue = c_void_p()
        vk.vkGetDeviceQueue(self.device, indices.compute, 0)
        vk.vkGetDeviceQueue(self.device, indices.transfer, 0)
        
        self.compute_queue = compute_queue
        self.transfer_queue = transfer_queue

    def _find_queue_families(self, device: c_void_p) -> QueueFamilyIndices:
        """Find required queue families."""
        indices = QueueFamilyIndices()

        count = c_uint32(0)
        properties = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)

        for i, prop in enumerate(properties):
            # Look for compute queue
            if prop.queueFlags & vk.VK_QUEUE_COMPUTE_BIT and indices.compute is None:
                indices.compute = i

            # Look for transfer queue
            if prop.queueFlags & vk.VK_QUEUE_TRANSFER_BIT and indices.transfer is None:
                indices.transfer = i

            if indices.is_complete():
                break

        return indices

    def _is_device_suitable(self, device: c_void_p) -> bool:
        """Check if device is suitable for compute operations."""
        indices = self._find_queue_families(device)
        return indices.is_complete()

    def _check_validation_layer_support(self) -> bool:
        """Check if validation layers are available."""
        try:
            available_layers = vk.vkEnumerateInstanceLayerProperties()
            required_layers = [b"VK_LAYER_KHRONOS_validation"]

            for layer in required_layers:
                found = False
                for available in available_layers:
                    if layer == available.layerName:
                        found = True
                        break
                if not found:
                    return False
            return True
        except:
            return False

    def get_memory_properties(self) -> VkPhysicalDeviceMemoryProperties:
        """Get memory properties of physical device."""
        props = VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, byref(props))
        return props

    def cleanup(self):
        """Clean up Vulkan resources."""
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)

        self.device = None
        self.instance = None
        self.physical_device = None
        self.compute_queue = None
        self.transfer_queue = None
