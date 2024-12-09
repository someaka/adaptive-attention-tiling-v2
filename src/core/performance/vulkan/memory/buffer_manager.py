"""Vulkan Buffer Management.

This module provides efficient buffer management for Vulkan memory operations,
including staging buffers, memory pools, and zero-copy operations.
"""

import vulkan as vk
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import ctypes

@dataclass
class BufferInfo:
    """Information about a Vulkan buffer."""
    buffer: vk.Buffer
    memory: vk.DeviceMemory
    size: int
    usage: vk.BufferUsageFlags
    properties: vk.MemoryPropertyFlags
    is_mapped: bool = False
    mapped_ptr: Optional[int] = None

class BufferManager:
    """Manages Vulkan buffers and memory allocation."""
    
    def __init__(self, device: vk.Device, physical_device: vk.PhysicalDevice):
        self.device = device
        self.physical_device = physical_device
        self.buffers: Dict[str, BufferInfo] = {}
        self.staging_pools: Dict[int, List[BufferInfo]] = {}  # size -> available buffers
        
    def _find_memory_type(self, type_filter: int, properties: vk.MemoryPropertyFlags) -> int:
        """Find suitable memory type index."""
        mem_properties = vk.GetPhysicalDeviceMemoryProperties(self.physical_device)
        
        for i in range(mem_properties.memoryTypeCount):
            if ((type_filter & (1 << i)) and 
                (mem_properties.memoryTypes[i].propertyFlags & properties) == properties):
                return i
        
        raise RuntimeError("Failed to find suitable memory type")
    
    def create_buffer(self,
                     size: int,
                     usage: vk.BufferUsageFlags,
                     properties: vk.MemoryPropertyFlags,
                     name: str = "") -> BufferInfo:
        """Create a Vulkan buffer with specified properties."""
        # Create buffer
        buffer_info = vk.BufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=vk.SHARING_MODE_EXCLUSIVE
        )
        buffer = vk.CreateBuffer(self.device, buffer_info, None)
        
        # Get memory requirements
        mem_requirements = vk.GetBufferMemoryRequirements(self.device, buffer)
        
        # Allocate memory
        alloc_info = vk.MemoryAllocateInfo(
            allocationSize=mem_requirements.size,
            memoryTypeIndex=self._find_memory_type(
                mem_requirements.memoryTypeBits,
                properties
            )
        )
        memory = vk.AllocateMemory(self.device, alloc_info, None)
        
        # Bind buffer memory
        vk.BindBufferMemory(self.device, buffer, memory, 0)
        
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
        if pool_size in self.staging_pools and self.staging_pools[pool_size]:
            return self.staging_pools[pool_size].pop()
        
        # Create new staging buffer
        return self.create_buffer(
            size=pool_size,
            usage=vk.BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties=vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                      vk.MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
    
    def return_staging_buffer(self, buffer_info: BufferInfo) -> None:
        """Return staging buffer to pool."""
        if buffer_info.is_mapped:
            self.unmap_memory(buffer_info)
            
        self.staging_pools.setdefault(buffer_info.size, []).append(buffer_info)
    
    def map_memory(self, buffer_info: BufferInfo) -> int:
        """Map buffer memory for CPU access."""
        if not buffer_info.is_mapped:
            buffer_info.mapped_ptr = vk.MapMemory(
                self.device,
                buffer_info.memory,
                0,
                buffer_info.size,
                0
            )
            buffer_info.is_mapped = True
            
        return buffer_info.mapped_ptr
    
    def unmap_memory(self, buffer_info: BufferInfo) -> None:
        """Unmap buffer memory."""
        if buffer_info.is_mapped:
            vk.UnmapMemory(self.device, buffer_info.memory)
            buffer_info.is_mapped = False
            buffer_info.mapped_ptr = None
    
    def copy_to_device(self,
                      data: Union[bytes, np.ndarray],
                      device_buffer: BufferInfo,
                      command_pool: vk.CommandPool,
                      queue: vk.Queue) -> None:
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
            alloc_info = vk.CommandBufferAllocateInfo(
                commandPool=command_pool,
                level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            command_buffer = vk.AllocateCommandBuffers(self.device, alloc_info)[0]
            
            # Record copy command
            begin_info = vk.CommandBufferBeginInfo(
                flags=vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            )
            vk.BeginCommandBuffer(command_buffer, begin_info)
            
            copy_region = vk.BufferCopy(
                srcOffset=0,
                dstOffset=0,
                size=data_size
            )
            vk.CmdCopyBuffer(
                command_buffer,
                staging_buffer.buffer,
                device_buffer.buffer,
                1,
                [copy_region]
            )
            
            vk.EndCommandBuffer(command_buffer)
            
            # Submit command buffer
            submit_info = vk.SubmitInfo(
                commandBuffers=[command_buffer]
            )
            vk.QueueSubmit(queue, 1, submit_info, vk.Fence(0))
            vk.QueueWaitIdle(queue)
            
            # Cleanup command buffer
            vk.FreeCommandBuffers(self.device, command_pool, 1, [command_buffer])
            
        finally:
            # Return staging buffer to pool
            self.return_staging_buffer(staging_buffer)
    
    def cleanup(self) -> None:
        """Clean up all buffers and memory."""
        # Clean up named buffers
        for buffer_info in self.buffers.values():
            if buffer_info.is_mapped:
                self.unmap_memory(buffer_info)
            vk.DestroyBuffer(self.device, buffer_info.buffer, None)
            vk.FreeMemory(self.device, buffer_info.memory, None)
        
        # Clean up staging pools
        for pool in self.staging_pools.values():
            for buffer_info in pool:
                if buffer_info.is_mapped:
                    self.unmap_memory(buffer_info)
                vk.DestroyBuffer(self.device, buffer_info.buffer, None)
                vk.FreeMemory(self.device, buffer_info.memory, None)
        
        self.buffers.clear()
        self.staging_pools.clear()
