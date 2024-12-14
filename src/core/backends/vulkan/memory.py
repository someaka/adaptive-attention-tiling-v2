"""
Memory Manager for Vulkan-based Adaptive Attention Tiling.
Handles efficient memory allocation, transfers, and pooling.
"""

from typing import Dict, Tuple, Optional
from ctypes import c_void_p, c_uint32, c_size_t, byref, POINTER, Structure, cast

import numpy as np
import vulkan as vk

# Vulkan type definitions
VkDevice = c_void_p
VkPhysicalDevice = c_void_p
VkDeviceMemory = c_void_p
VkBuffer = c_void_p
VkCommandPool = c_void_p
VkQueue = c_void_p

# Vulkan memory property flags
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001

# Vulkan buffer usage flags
VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001
VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002

# Vulkan sharing mode
VK_SHARING_MODE_EXCLUSIVE = 0

# Vulkan command buffer flags
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001


class MemoryManager:
    def __init__(self, device: VkDevice, physical_device: VkPhysicalDevice):
        self.device = device
        self.physical_device = physical_device
        
        # Pool sizes
        self.staging_pool_size = 64 * 1024 * 1024  # 64MB staging pool
        self.device_pool_size = 512 * 1024 * 1024  # 512MB device pool

        # Memory type indices
        self.host_visible_index = self._find_memory_type(
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        self.device_local_index = self._find_memory_type(
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )

        # Memory pools
        self.staging_pool = self._create_memory_pool(
            size=self.staging_pool_size,
            memory_type_index=self.host_visible_index,
        )

        self.device_pool = self._create_memory_pool(
            size=self.device_pool_size,
            memory_type_index=self.device_local_index,
        )

        # Buffer tracking
        self.buffer_allocations: Dict[VkBuffer, Tuple[int, int]] = {}  # offset, size

    def _find_memory_type(self, properties: int) -> int:
        """Find suitable memory type index."""
        mem_props = vk.VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, byref(mem_props))
        
        for i in range(mem_props.memoryTypeCount):
            if (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i

        raise RuntimeError("Failed to find suitable memory type")

    def _create_memory_pool(self, size: int, memory_type_index: int) -> VkDeviceMemory:
        """Create memory pool."""
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=size,
            memoryTypeIndex=memory_type_index
        )
        
        memory = c_void_p()
        result = vk.vkAllocateMemory(self.device, byref(alloc_info), None, byref(memory))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate memory: {result}")
        
        return memory

    def allocate_buffer(
        self, size: int, usage: int, device_local: bool = True
    ) -> Tuple[VkBuffer, int]:
        """Allocate buffer from pool."""
        # Create buffer
        create_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext=None,
            flags=0,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            pQueueFamilyIndices=None
        )
        
        buffer = c_void_p()
        result = vk.vkCreateBuffer(self.device, byref(create_info), None, byref(buffer))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create buffer: {result}")

        # Get memory requirements
        mem_reqs = vk.VkMemoryRequirements()
        vk.vkGetBufferMemoryRequirements(self.device, buffer, byref(mem_reqs))

        # Find space in pool
        pool = self.device_pool if device_local else self.staging_pool
        offset = self._find_pool_space(pool, mem_reqs.size, mem_reqs.alignment)

        # Bind memory
        result = vk.vkBindBufferMemory(self.device, buffer, pool, offset)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to bind buffer memory: {result}")

        # Track allocation
        self.buffer_allocations[buffer] = (offset, size)

        return buffer, offset

    def _find_pool_space(self, pool: VkDeviceMemory, size: int, alignment: int) -> int:
        """Find free space in memory pool."""
        # Simple first-fit allocation strategy
        # In production, use more sophisticated allocation strategy
        used_ranges = sorted(
            [
                (offset, offset + size)
                for offset, size in self.buffer_allocations.values()
            ]
        )

        current_offset = 0
        for start, end in used_ranges:
            if current_offset + size <= start:
                # Found free space
                aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1)
                if aligned_offset + size <= start:
                    return aligned_offset
            current_offset = end

        # Try end of used space
        aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1)
        pool_size = self.staging_pool_size if pool == self.staging_pool else self.device_pool_size
        if aligned_offset + size <= pool_size:
            return aligned_offset

        raise RuntimeError("Out of memory in pool")

    def copy_to_device(
        self,
        data: np.ndarray,
        device_buffer: VkBuffer,
        command_pool: VkCommandPool,
        queue: VkQueue,
    ):
        """Copy data to device buffer using staging buffer."""
        data_size = data.nbytes

        # Allocate staging buffer
        staging_buffer, staging_offset = self.allocate_buffer(
            size=data_size,
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            device_local=False
        )

        # Map memory and copy data
        data_ptr = c_void_p()
        result = vk.vkMapMemory(
            self.device,
            self.staging_pool,
            staging_offset,
            data_size,
            0  # flags
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to map memory: {result}")

        # Copy data to mapped memory
        src_ptr = data.ctypes.data_as(c_void_p)
        if data_ptr.value is None:
            raise RuntimeError("Failed to map memory: null pointer")
        
        # Create a ctypes array from the mapped memory
        dst_array = (c_uint32 * (data_size // 4)).from_address(data_ptr.value or 0)
        # Create a ctypes array from the source data
        src_array = (c_uint32 * (data_size // 4)).from_address(src_ptr.value or 0)
        # Copy the data
        for i in range(data_size // 4):
            dst_array[i] = src_array[i]
        
        vk.vkUnmapMemory(self.device, self.staging_pool)

        # Create command buffer for transfer
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        command_buffer = c_void_p()
        result = vk.vkAllocateCommandBuffers(self.device, byref(alloc_info), byref(command_buffer))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate command buffer: {result}")

        # Begin command buffer
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None
        )
        
        result = vk.vkBeginCommandBuffer(command_buffer, byref(begin_info))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to begin command buffer: {result}")

        # Record copy command
        copy_region = vk.VkBufferCopy(
            srcOffset=staging_offset,
            dstOffset=self.buffer_allocations[device_buffer][0],
            size=data_size
        )
        
        vk.vkCmdCopyBuffer(command_buffer, staging_buffer, device_buffer, 1, byref(copy_region))

        # End command buffer
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
            pCommandBuffers=[command_buffer],
            signalSemaphoreCount=0,
            pSignalSemaphores=None
        )
        
        result = vk.vkQueueSubmit(queue, 1, byref(submit_info), None)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to submit queue: {result}")
        
        vk.vkQueueWaitIdle(queue)

        # Cleanup
        vk.vkFreeCommandBuffers(self.device, command_pool, 1, byref(command_buffer))
        self.free_buffer(staging_buffer)

    def copy_from_device(
        self,
        device_buffer: VkBuffer,
        size: int,
        command_pool: VkCommandPool,
        queue: VkQueue,
    ) -> np.ndarray:
        """Copy data from device buffer using staging buffer."""
        # Allocate staging buffer
        staging_buffer, staging_offset = self.allocate_buffer(
            size=size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            device_local=False
        )

        # Create command buffer for transfer
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        command_buffer = c_void_p()
        result = vk.vkAllocateCommandBuffers(self.device, byref(alloc_info), byref(command_buffer))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate command buffer: {result}")

        # Begin command buffer
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None
        )
        
        result = vk.vkBeginCommandBuffer(command_buffer, byref(begin_info))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to begin command buffer: {result}")

        # Record copy command
        copy_region = vk.VkBufferCopy(
            srcOffset=self.buffer_allocations[device_buffer][0],
            dstOffset=staging_offset,
            size=size
        )
        
        vk.vkCmdCopyBuffer(command_buffer, device_buffer, staging_buffer, 1, byref(copy_region))

        # End command buffer
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
            pCommandBuffers=[command_buffer],
            signalSemaphoreCount=0,
            pSignalSemaphores=None
        )
        
        result = vk.vkQueueSubmit(queue, 1, byref(submit_info), None)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to submit queue: {result}")
        
        vk.vkQueueWaitIdle(queue)

        # Map memory and read data
        data_ptr = c_void_p()
        result = vk.vkMapMemory(
            self.device,
            self.staging_pool,
            staging_offset,
            size,
            0  # flags
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to map memory: {result}")

        if data_ptr.value is None:
            raise RuntimeError("Failed to map memory: null pointer")

        # Create a buffer view of the mapped memory
        buffer = (c_uint32 * (size // 4)).from_address(data_ptr.value or 0)
        # Convert to numpy array
        data = np.frombuffer(buffer, dtype=np.float16)
        
        vk.vkUnmapMemory(self.device, self.staging_pool)

        # Cleanup
        vk.vkFreeCommandBuffers(self.device, command_pool, 1, byref(command_buffer))
        self.free_buffer(staging_buffer)

        return data

    def free_buffer(self, buffer: VkBuffer):
        """Free buffer and its memory allocation."""
        if buffer in self.buffer_allocations:
            del self.buffer_allocations[buffer]
            vk.vkDestroyBuffer(self.device, buffer, None)

    def __del__(self):
        """Cleanup memory pools."""
        vk.vkFreeMemory(self.device, self.staging_pool, None)
        vk.vkFreeMemory(self.device, self.device_pool, None)
