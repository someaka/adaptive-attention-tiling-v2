"""
Memory Manager for Vulkan-based Adaptive Attention Tiling.
Handles efficient memory allocation, transfers, and pooling.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import vulkan as vk

class MemoryManager:
    def __init__(self, device: vk.Device, physical_device: vk.PhysicalDevice):
        self.device = device
        self.physical_device = physical_device
        
        # Memory type indices
        self.host_visible_index = self._find_memory_type(
            vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        self.device_local_index = self._find_memory_type(
            vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        
        # Memory pools
        self.staging_pool = self._create_memory_pool(
            size=64 * 1024 * 1024,  # 64MB staging pool
            memory_type_index=self.host_visible_index
        )
        
        self.device_pool = self._create_memory_pool(
            size=512 * 1024 * 1024,  # 512MB device pool
            memory_type_index=self.device_local_index
        )
        
        # Buffer tracking
        self.buffer_allocations: Dict[vk.Buffer, Tuple[int, int]] = {}  # offset, size
        
    def _find_memory_type(self, properties: int) -> int:
        """Find suitable memory type index."""
        memory_properties = self.physical_device.getMemoryProperties()
        
        for i in range(memory_properties.memoryTypeCount):
            if ((memory_properties.memoryTypes[i].propertyFlags & properties) == 
                properties):
                return i
        
        raise RuntimeError("Failed to find suitable memory type")
    
    def _create_memory_pool(self, size: int, memory_type_index: int) -> vk.DeviceMemory:
        """Create memory pool."""
        alloc_info = vk.MemoryAllocateInfo(
            allocationSize=size,
            memoryTypeIndex=memory_type_index
        )
        return self.device.allocateMemory(alloc_info)
    
    def allocate_buffer(self,
                       size: int,
                       usage: int,
                       device_local: bool = True) -> Tuple[vk.Buffer, int]:
        """Allocate buffer from pool."""
        # Create buffer
        buffer_info = vk.BufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=vk.SHARING_MODE_EXCLUSIVE
        )
        buffer = self.device.createBuffer(buffer_info)
        
        # Get memory requirements
        mem_reqs = self.device.getBufferMemoryRequirements(buffer)
        
        # Find space in pool
        pool = self.device_pool if device_local else self.staging_pool
        offset = self._find_pool_space(pool, mem_reqs.size, mem_reqs.alignment)
        
        # Bind memory
        self.device.bindBufferMemory(buffer, pool, offset)
        
        # Track allocation
        self.buffer_allocations[buffer] = (offset, size)
        
        return buffer, offset
    
    def _find_pool_space(self, pool: vk.DeviceMemory, size: int, alignment: int) -> int:
        """Find free space in memory pool."""
        # Simple first-fit allocation strategy
        # In production, use more sophisticated allocation strategy
        used_ranges = sorted(
            [(offset, offset + size) 
             for offset, size in self.buffer_allocations.values()]
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
        if aligned_offset + size <= pool.size:
            return aligned_offset
        
        raise RuntimeError("Out of memory in pool")
    
    def copy_to_device(self,
                      data: np.ndarray,
                      device_buffer: vk.Buffer,
                      command_pool: vk.CommandPool,
                      queue: vk.Queue):
        """Copy data to device buffer using staging buffer."""
        data_size = data.nbytes
        
        # Allocate staging buffer
        staging_buffer, staging_offset = self.allocate_buffer(
            size=data_size,
            usage=vk.BUFFER_USAGE_TRANSFER_SRC_BIT,
            device_local=False
        )
        
        # Map memory and copy data
        memory_ptr = self.device.mapMemory(
            self.staging_pool,
            staging_offset,
            data_size
        )
        memory_ptr.write(data.tobytes())
        self.device.unmapMemory(self.staging_pool)
        
        # Create command buffer for transfer
        alloc_info = vk.CommandBufferAllocateInfo(
            commandPool=command_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        command_buffer = self.device.allocateCommandBuffers(alloc_info)[0]
        
        # Record transfer command
        begin_info = vk.CommandBufferBeginInfo(
            flags=vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        command_buffer.begin(begin_info)
        
        copy_region = vk.BufferCopy(
            srcOffset=staging_offset,
            dstOffset=self.buffer_allocations[device_buffer][0],
            size=data_size
        )
        command_buffer.copyBuffer(staging_buffer, device_buffer, [copy_region])
        
        command_buffer.end()
        
        # Submit transfer command
        submit_info = vk.SubmitInfo(
            commandBuffers=[command_buffer]
        )
        queue.submit([submit_info])
        queue.waitIdle()
        
        # Cleanup
        self.device.freeCommandBuffers(command_pool, [command_buffer])
        self.free_buffer(staging_buffer)
    
    def copy_from_device(self,
                        device_buffer: vk.Buffer,
                        size: int,
                        command_pool: vk.CommandPool,
                        queue: vk.Queue) -> np.ndarray:
        """Copy data from device buffer using staging buffer."""
        # Allocate staging buffer
        staging_buffer, staging_offset = self.allocate_buffer(
            size=size,
            usage=vk.BUFFER_USAGE_TRANSFER_DST_BIT,
            device_local=False
        )
        
        # Create command buffer for transfer
        alloc_info = vk.CommandBufferAllocateInfo(
            commandPool=command_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        command_buffer = self.device.allocateCommandBuffers(alloc_info)[0]
        
        # Record transfer command
        begin_info = vk.CommandBufferBeginInfo(
            flags=vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        command_buffer.begin(begin_info)
        
        copy_region = vk.BufferCopy(
            srcOffset=self.buffer_allocations[device_buffer][0],
            dstOffset=staging_offset,
            size=size
        )
        command_buffer.copyBuffer(device_buffer, staging_buffer, [copy_region])
        
        command_buffer.end()
        
        # Submit transfer command
        submit_info = vk.SubmitInfo(
            commandBuffers=[command_buffer]
        )
        queue.submit([submit_info])
        queue.waitIdle()
        
        # Map memory and read data
        memory_ptr = self.device.mapMemory(
            self.staging_pool,
            staging_offset,
            size
        )
        data = np.frombuffer(memory_ptr.read(), dtype=np.float16)
        self.device.unmapMemory(self.staging_pool)
        
        # Cleanup
        self.device.freeCommandBuffers(command_pool, [command_buffer])
        self.free_buffer(staging_buffer)
        
        return data
    
    def free_buffer(self, buffer: vk.Buffer):
        """Free buffer and its memory allocation."""
        if buffer in self.buffer_allocations:
            del self.buffer_allocations[buffer]
            self.device.destroyBuffer(buffer)
    
    def __del__(self):
        """Cleanup memory pools."""
        self.device.freeMemory(self.staging_pool)
        self.device.freeMemory(self.device_pool)
