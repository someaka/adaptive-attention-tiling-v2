"""Vulkan compute interface for tensor operations."""

from typing import Optional
from ctypes import c_void_p, c_uint32, POINTER, cast

import torch

from .tensor_ops import VulkanTensorOps, TensorDescriptor
from .device import VulkanDevice
from .command_pool import CommandPoolType


def handle_to_int(handle: c_void_p) -> int:
    """Convert a Vulkan handle to integer."""
    if not handle:
        return 0
    # Use ctypes cast to safely convert to integer
    ptr = cast(handle, POINTER(c_uint32))
    if not ptr:
        return 0
    return ptr.contents.value


class VulkanCompute:
    """High-level interface for Vulkan compute operations."""

    def __init__(self):
        # Initialize device
        self.device = VulkanDevice()
        if not self.device.initialize():
            raise RuntimeError("Failed to initialize Vulkan device")

        # Convert device handle to int first
        raw_device = self.device.device
        raw_queue = self.device.compute_queue
        
        device_int = int(raw_device) if raw_device is not None else 0
        queue_int = int(raw_queue) if raw_queue is not None else 0
        
        # Ensure queue family index is not None
        queue_family_index = self.device.queue_family_indices.compute
        if queue_family_index is None:
            raise RuntimeError("Compute queue family index is None")

        # Initialize tensor operations
        self.tensor_ops = VulkanTensorOps(
            device=device_int,
            queue=queue_int,
            queue_family_index=queue_family_index
        )

    def register_tensor(self, tensor: torch.Tensor) -> TensorDescriptor:
        """Register a tensor for Vulkan operations."""
        return self.tensor_ops.register_tensor(tensor)

    def compute_tile(
        self, input_tensor: torch.Tensor, resolution: float, d_state: int
    ) -> torch.Tensor:
        """Process a tile using Vulkan compute shader."""
        # Register input tensor
        input_descriptor = self.register_tensor(input_tensor)
            
        # Create output tensor
        output_tensor = torch.empty_like(input_tensor)
        output_descriptor = self.register_tensor(output_tensor)
        
        # Copy input data to device
        self.tensor_ops._copy_to_device(input_tensor, input_descriptor)
        
        # Record and execute compute commands
        command_buffer = self.tensor_ops.command_manager.allocate_command_buffer(
            reuse=True
        )
        
        # TODO: Implement compute shader dispatch
        # For now, just copy input to output
        self.tensor_ops._copy_from_device(input_descriptor, output_tensor)
        
        # Free command buffer for reuse
        self.tensor_ops.command_manager._reusable_buffers.append(command_buffer)
        
        return output_tensor

    def cleanup(self):
        """Clean up Vulkan resources."""
        self.tensor_ops.cleanup()
        self.device.cleanup() 