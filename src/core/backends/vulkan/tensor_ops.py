"""PyTorch tensor operations for Vulkan backend."""

from dataclasses import dataclass
import ctypes
import struct
from typing import Dict, List, Tuple, Optional
from ctypes import c_void_p, c_uint32, cast, POINTER

import torch
import vulkan as vk

from .command_buffer import CommandBufferManager, CommandConfig
from .memory_manager import BufferUsage, VulkanMemoryManager
from .pipeline import PipelineType, VulkanPipeline
from .shader_manager import ShaderManager


@dataclass
class TensorDescriptor:
    """Descriptor for tensor layout and memory."""

    shape: torch.Size
    dtype: torch.dtype
    strides: Tuple[int, ...]
    buffer: int  # VkBuffer handle
    memory: int  # VkDeviceMemory handle
    offset: int
    size: int
    is_contiguous: bool


def handle_to_int(handle: c_void_p) -> int:
    """Convert a Vulkan handle (CData) to integer."""
    if handle is None:
        return 0
    return cast(handle, POINTER(c_uint32)).contents.value


class VulkanTensorOps:
    """High-performance tensor operations using Vulkan compute."""

    def __init__(self, device: int, queue: int, queue_family_index: int):  # device: VkDevice, queue: VkQueue
        # Store original device handle for managers that need int
        self.device_int = device
        
        # Convert handles for Vulkan API calls
        device_handle = c_void_p(device)
        queue_handle = c_void_p(queue)
        
        self.device = device_handle
        self.queue = queue_handle
        
        # Initialize managers with appropriate handle types
        self.memory_manager = VulkanMemoryManager(device_handle)  # Needs c_void_p
        self.command_manager = CommandBufferManager(device_handle, queue_family_index)  # Needs c_void_p
        self.pipeline_manager = VulkanPipeline(device)  # Needs int
        self.shader_manager = ShaderManager(device)  # Needs int

        # Cache for tensor descriptors
        self._tensor_cache: Dict[torch.Tensor, TensorDescriptor] = {}

        # Initialize pipelines
        self._init_pipelines()

    def _init_pipelines(self) -> None:
        """Initialize compute pipelines."""
        # TODO: Initialize compute pipelines when needed
        pass

    def register_tensor(self, tensor: torch.Tensor) -> TensorDescriptor:
        """Register a PyTorch tensor for Vulkan operations."""
        if tensor in self._tensor_cache:
            return self._tensor_cache[tensor]

        # Get tensor info
        size_bytes = tensor.nelement() * tensor.element_size()
        is_contiguous = tensor.is_contiguous()

        # Allocate Vulkan buffer
        buffer, memory, offset = self.memory_manager.allocate_tile_state(size_bytes)

        # Create descriptor
        descriptor = TensorDescriptor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            strides=tensor.stride(),
            buffer=handle_to_int(buffer),  # Convert c_void_p to int
            memory=handle_to_int(memory),  # Convert c_void_p to int
            offset=offset,
            size=size_bytes,
            is_contiguous=is_contiguous,
        )

        # Copy data to device
        self._copy_to_device(tensor, descriptor)
        self._tensor_cache[tensor] = descriptor

        return descriptor

    def _copy_to_device(self, tensor: torch.Tensor, descriptor: TensorDescriptor) -> None:
        """Copy tensor data to Vulkan buffer."""
        # Map memory
        data_ptr = c_void_p()
        result = vk.vkMapMemory(
            self.device,
            c_void_p(descriptor.memory),  # Convert back to c_void_p
            descriptor.offset,
            descriptor.size,
            0  # flags
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to map memory: {result}")

        # Copy data
        tensor_data = tensor.contiguous().data_ptr()
        ctypes.memmove(data_ptr, tensor_data, descriptor.size)

        # Unmap memory
        vk.vkUnmapMemory(self.device, c_void_p(descriptor.memory))  # Convert back to c_void_p

    def _copy_from_device(self, descriptor: TensorDescriptor, tensor: torch.Tensor) -> None:
        """Copy data from Vulkan buffer to tensor."""
        # Map memory
        data_ptr = c_void_p()
        result = vk.vkMapMemory(
            self.device,
            c_void_p(descriptor.memory),  # Convert back to c_void_p
            descriptor.offset,
            descriptor.size,
            0  # flags
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to map memory: {result}")

        # Copy data
        tensor_data = tensor.data_ptr()
        ctypes.memmove(tensor_data, data_ptr, descriptor.size)

        # Unmap memory
        vk.vkUnmapMemory(self.device, c_void_p(descriptor.memory))  # Convert back to c_void_p

    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        for descriptor in self._tensor_cache.values():
            self.memory_manager.free_buffer(c_void_p(descriptor.buffer), BufferUsage.TILE_STATE)
        self._tensor_cache.clear()
        self.memory_manager.cleanup()
