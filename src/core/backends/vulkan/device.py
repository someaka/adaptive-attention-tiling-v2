"""Vulkan device management and initialization."""

from dataclasses import dataclass
from typing import Optional

import vulkan as vk


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
        self.queue_family_indices = None

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
        app_info = vk.ApplicationInfo(
            application_name="Adaptive Attention Tiling",
            application_version=vk.make_version(1, 0, 0),
            engine_name="No Engine",
            engine_version=vk.make_version(1, 0, 0),
            api_version=vk.API_VERSION_1_2,
        )

        create_info = vk.InstanceCreateInfo(
            application_info=app_info,
            enabled_layer_names=(
                ["VK_LAYER_KHRONOS_validation"]
                if self._check_validation_layer_support()
                else []
            ),
        )

        self.instance = vk.create_instance(create_info, None)

    def _select_physical_device(self):
        """Select suitable physical device."""
        devices = vk.enumerate_physical_devices(self.instance)
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
        unique_queue_families = {indices.compute, indices.transfer}

        queue_create_infos = []
        for queue_family in unique_queue_families:
            queue_create_infos.append(
                vk.DeviceQueueCreateInfo(
                    queue_family_index=queue_family, queue_priorities=[1.0]
                )
            )

        # Enable features needed for compute
        features = vk.PhysicalDeviceFeatures()

        device_create_info = vk.DeviceCreateInfo(
            queue_create_infos=queue_create_infos,
            enabled_features=features,
            enabled_extension_names=[],
        )

        self.device = vk.create_device(self.physical_device, device_create_info, None)

        # Get queue handles
        self.compute_queue = vk.get_device_queue(self.device, indices.compute, 0)
        self.transfer_queue = vk.get_device_queue(self.device, indices.transfer, 0)

    def _find_queue_families(self, device: vk.PhysicalDevice) -> QueueFamilyIndices:
        """Find required queue families."""
        indices = QueueFamilyIndices()

        properties = vk.get_physical_device_queue_family_properties(device)
        for i, prop in enumerate(properties):
            # Look for compute queue
            if prop.queue_flags & vk.QueueFlagBits.COMPUTE and indices.compute is None:
                indices.compute = i

            # Look for transfer queue
            if (
                prop.queue_flags & vk.QueueFlagBits.TRANSFER
                and indices.transfer is None
            ):
                indices.transfer = i

            if indices.is_complete():
                break

        return indices

    def _is_device_suitable(self, device: vk.PhysicalDevice) -> bool:
        """Check if device is suitable for compute operations."""
        indices = self._find_queue_families(device)

        return indices.is_complete()

    def _check_validation_layer_support(self) -> bool:
        """Check if validation layers are available."""
        try:
            available_layers = vk.enumerate_instance_layer_properties()
            required_layers = ["VK_LAYER_KHRONOS_validation"]

            for layer in required_layers:
                found = False
                for available in available_layers:
                    if layer == available.layer_name:
                        found = True
                        break
                if not found:
                    return False
            return True
        except:
            return False

    def get_memory_properties(self) -> vk.PhysicalDeviceMemoryProperties:
        """Get memory properties of physical device."""
        return vk.get_physical_device_memory_properties(self.physical_device)

    def cleanup(self):
        """Clean up Vulkan resources."""
        if self.device:
            vk.destroy_device(self.device, None)
        if self.instance:
            vk.destroy_instance(self.instance, None)

        self.device = None
        self.instance = None
        self.physical_device = None
        self.compute_queue = None
        self.transfer_queue = None
