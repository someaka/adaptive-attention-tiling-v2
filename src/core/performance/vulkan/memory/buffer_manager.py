"""Vulkan Buffer Management.

This module provides efficient buffer management for Vulkan memory operations,
including staging buffers, memory pools, and zero-copy operations.
"""

import ctypes
from ctypes import c_void_p, c_uint32, c_int, byref, cast, POINTER, Structure
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import vulkan as vk


@dataclass
class BufferInfo:
    """Information about a Vulkan buffer."""

    buffer: c_void_p  # VkBuffer handle
    memory: c_void_p  # VkDeviceMemory handle
    size: int
    usage: int  # VkBufferUsageFlags
    properties: int  # VkMemoryPropertyFlags
    is_mapped: bool = False
    mapped_ptr: Optional[int] = None


class BufferManager:
    """Manages Vulkan buffers and memory allocation."""

    def __init__(self, device: c_void_p, physical_device: c_void_p):
        self.device = device
        self.physical_device = physical_device
        self.buffers: Dict[str, BufferInfo] = {}
        self.staging_pools: Dict[int, List[BufferInfo]] = (
            {}
        )  # size -> available buffers

    def _find_memory_type(
        self, type_filter: int, properties: int
    ) -> int:
        """Find suitable memory type index."""
        mem_properties = vk.VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, byref(mem_properties))

        for i in range(mem_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                mem_properties.memoryTypes[i].propertyFlags & properties
            ) == properties:
                return i

        raise RuntimeError("Failed to find suitable memory type")

    def create_buffer(
        self,
        size: int,
        usage: int,  # VkBufferUsageFlags
        properties: int,  # VkMemoryPropertyFlags
        name: str = "",
    ) -> BufferInfo:
        """Create a Vulkan buffer with specified properties."""
        # Create buffer
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext=None,
            flags=0,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            pQueueFamilyIndices=None
        )
        buffer = c_void_p()
        result = vk.vkCreateBuffer(self.device, byref(buffer_info), None, byref(buffer))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create buffer: {result}")

        # Get memory requirements
        mem_requirements = vk.VkMemoryRequirements()
        vk.vkGetBufferMemoryRequirements(self.device, buffer, byref(mem_requirements))

        # Allocate memory
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=self._find_memory_type(
                mem_requirements.memoryTypeBits, properties
            ),
        )
        memory = c_void_p()
        result = vk.vkAllocateMemory(self.device, byref(alloc_info), None, byref(memory))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate memory: {result}")

        # Bind buffer memory
        result = vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to bind buffer memory: {result}")

        buffer_info = BufferInfo(
            buffer=buffer,
            memory=memory,
            size=size,
            usage=usage,
            properties=properties
        )

        if name:
            self.buffers[name] = buffer_info

        return buffer_info

    def get_staging_buffer(self, size: int) -> BufferInfo:
        """Get a staging buffer from pool or create new one."""
        # Round up size to nearest power of 2 for better reuse
        pool_size = 1 << (size - 1).bit_length()

        # Check if we have available buffer in pool
        if self.staging_pools.get(pool_size):
            return self.staging_pools[pool_size].pop()

        # Create new staging buffer
        return self.create_buffer(
            size=pool_size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties=vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                      vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def return_staging_buffer(self, buffer_info: BufferInfo) -> None:
        """Return staging buffer to pool."""
        if buffer_info.is_mapped:
            self.unmap_memory(buffer_info)

        self.staging_pools.setdefault(buffer_info.size, []).append(buffer_info)

    def map_memory(self, buffer_info: BufferInfo) -> int:
        """Map buffer memory for CPU access."""
        if not buffer_info.is_mapped:
            data_ptr = c_void_p()
            result = vk.vkMapMemory(
                self.device,
                buffer_info.memory,
                0,  # offset
                buffer_info.size,
                0   # flags
            )
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to map memory: {result}")
                
            # Get the mapped pointer from the data_ptr
            mapped_ptr = data_ptr.value
            if mapped_ptr is None:
                raise RuntimeError("Failed to map memory: null pointer")
                
            buffer_info.mapped_ptr = mapped_ptr
            buffer_info.is_mapped = True

        if buffer_info.mapped_ptr is None:
            raise RuntimeError("Failed to map memory: null pointer")
        return buffer_info.mapped_ptr

    def unmap_memory(self, buffer_info: BufferInfo) -> None:
        """Unmap buffer memory."""
        if buffer_info.is_mapped:
            vk.vkUnmapMemory(self.device, buffer_info.memory)
            buffer_info.is_mapped = False
            buffer_info.mapped_ptr = None

    def copy_to_device(
        self,
        data: Union[bytes, np.ndarray],
        device_buffer: BufferInfo,
        command_pool: c_void_p,  # VkCommandPool
        queue: c_void_p,  # VkQueue
    ) -> None:
        """Copy data to device using staging buffer."""
        data_size = len(data.tobytes() if isinstance(data, np.ndarray) else data)

        # Get staging buffer
        staging_buffer = self.get_staging_buffer(data_size)

        try:
            # Map and copy to staging buffer
            ptr = self.map_memory(staging_buffer)
            if isinstance(data, np.ndarray):
                ctypes.memmove(ptr, data.ctypes.data, data_size)
            else:
                ctypes.memmove(ptr, data, data_size)
            self.unmap_memory(staging_buffer)

            # Create command buffer
            alloc_info = vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext=None,
                commandPool=command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            command_buffer = c_void_p()
            result = vk.vkAllocateCommandBuffers(self.device, byref(alloc_info), byref(command_buffer))
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to allocate command buffer: {result}")

            # Record copy command
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext=None,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo=None
            )
            result = vk.vkBeginCommandBuffer(command_buffer, byref(begin_info))
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to begin command buffer: {result}")

            copy_region = vk.VkBufferCopy(
                srcOffset=0,
                dstOffset=0,
                size=data_size
            )
            vk.vkCmdCopyBuffer(
                command_buffer,
                staging_buffer.buffer,
                device_buffer.buffer,
                1,
                byref(copy_region)
            )

            result = vk.vkEndCommandBuffer(command_buffer)
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to end command buffer: {result}")

            # Submit command buffer
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext=None,
                waitSemaphoreCount=0,
                pWaitSemaphores=None,
                pWaitDstStageMask=None,
                commandBufferCount=1,
                pCommandBuffers=byref(command_buffer),
                signalSemaphoreCount=0,
                pSignalSemaphores=None
            )
            result = vk.vkQueueSubmit(queue, 1, byref(submit_info), None)
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to submit queue: {result}")
            
            vk.vkQueueWaitIdle(queue)

            # Cleanup command buffer
            vk.vkFreeCommandBuffers(self.device, command_pool, 1, byref(command_buffer))

        finally:
            # Return staging buffer to pool
            self.return_staging_buffer(staging_buffer)

    def cleanup(self) -> None:
        """Clean up all buffers and memory."""
        # Clean up named buffers
        for buffer_info in self.buffers.values():
            if buffer_info.is_mapped:
                self.unmap_memory(buffer_info)
            vk.vkDestroyBuffer(self.device, buffer_info.buffer, None)
            vk.vkFreeMemory(self.device, buffer_info.memory, None)

        # Clean up staging pools
        for pool in self.staging_pools.values():
            for buffer_info in pool:
                if buffer_info.is_mapped:
                    self.unmap_memory(buffer_info)
                vk.vkDestroyBuffer(self.device, buffer_info.buffer, None)
                vk.vkFreeMemory(self.device, buffer_info.memory, None)

        self.buffers.clear()
        self.staging_pools.clear()
