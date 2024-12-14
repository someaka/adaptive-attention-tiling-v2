"""Vulkan memory management implementation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import ctypes
import numpy as np
import torch
import vulkan as vk
from ctypes import c_void_p, byref, cast, POINTER, c_uint32, memmove

from .memory_base import MemoryManagerBase, MemoryError


@dataclass
class VulkanBuffer:
    """Vulkan buffer information."""
    
    buffer: c_void_p  # VkBuffer handle
    memory: c_void_p  # VkDeviceMemory handle
    size: int  # Size in bytes
    offset: int  # Offset in memory
    shape: Optional[Tuple[int, ...]] = None  # Buffer shape if tensor
    dtype: Any = None  # Buffer data type if tensor


class VulkanMemoryManager(MemoryManagerBase):
    """Memory manager for Vulkan operations."""

    def __init__(self, device: c_void_p, physical_device: c_void_p):
        """Initialize Vulkan memory manager.
        
        Args:
            device: VkDevice handle
            physical_device: VkPhysicalDevice handle
        """
        super().__init__()
        self.device = device
        self.physical_device = physical_device
        self._buffer_pool: Dict[int, VulkanBuffer] = {}  # buffer_id -> buffer info
        self._buffer_pool_size = 1024 * 1024 * 128  # 128MB default pool size

    def allocate_tensor(self, size: Union[Tuple[int, ...], torch.Size], dtype: Any = np.float32) -> VulkanBuffer:
        """Allocate a Vulkan buffer for tensor data.
        
        Args:
            size: Tensor dimensions
            dtype: Data type
            
        Returns:
            Allocated buffer
        """
        try:
            # Calculate memory size
            element_size = np.dtype(dtype).itemsize
            memory_size = int(element_size * np.prod(size))  # Convert to int
            
            # Check if we need to clear buffer pool
            if self._allocated_memory + memory_size > self._peak_memory + self._buffer_pool_size:
                self._clear_buffer_pool()
            
            # Create buffer
            buffer_create_info = vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=memory_size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            )
            
            buffer = c_void_p()
            result = vk.vkCreateBuffer(self.device, buffer_create_info, None, byref(buffer))
            if result != vk.VK_SUCCESS:
                raise MemoryError(f"Failed to create buffer: {result}")
                
            try:
                # Get memory requirements
                mem_requirements = vk.VkMemoryRequirements()
                vk.vkGetBufferMemoryRequirements(self.device, buffer, byref(mem_requirements))
                
                # Allocate memory
                alloc_info = vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=mem_requirements.size,
                    memoryTypeIndex=self._find_memory_type(
                        int(mem_requirements.memoryTypeBits),  # Convert to int
                        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    )
                )
                
                memory = c_void_p()
                result = vk.vkAllocateMemory(self.device, alloc_info, None, byref(memory))
                if result != vk.VK_SUCCESS:
                    raise MemoryError(f"Failed to allocate memory: {result}")
                    
                # Bind memory to buffer
                result = vk.vkBindBufferMemory(self.device, buffer, memory, 0)
                if result != vk.VK_SUCCESS:
                    raise MemoryError(f"Failed to bind buffer memory: {result}")
                    
            except Exception as e:
                # Clean up on error
                vk.vkDestroyBuffer(self.device, buffer, None)
                raise e
                
            # Create buffer info
            buffer_info = VulkanBuffer(
                buffer=buffer,
                memory=memory,
                size=memory_size,
                offset=0,
                shape=size,
                dtype=dtype
            )
            
            # Update tracking
            buffer_id = id(buffer)
            self._buffer_pool[buffer_id] = buffer_info
            self._allocated_memory += memory_size
            self._peak_memory = max(self._peak_memory, int(self._allocated_memory))  # Convert to int
            
            self.record_metric("allocate")
            return buffer_info
            
        except Exception as e:
            raise MemoryError(f"Failed to allocate buffer: {e}")

    def free_tensor(self, buffer: VulkanBuffer) -> None:
        """Free a Vulkan buffer.
        
        Args:
            buffer: Buffer to free
        """
        try:
            buffer_id = id(buffer.buffer)
            if buffer_id in self._buffer_pool:
                # Free Vulkan resources
                vk.vkFreeMemory(self.device, buffer.memory, None)
                vk.vkDestroyBuffer(self.device, buffer.buffer, None)
                
                # Update tracking
                self._allocated_memory -= buffer.size
                del self._buffer_pool[buffer_id]
                
                self.record_metric("free")
                
        except Exception as e:
            raise MemoryError(f"Failed to free buffer: {e}")

    def copy_to_device(self, src: np.ndarray, dst: VulkanBuffer) -> None:
        """Copy data to Vulkan buffer.
        
        Args:
            src: Source NumPy array
            dst: Destination buffer
        """
        try:
            # Map memory
            data_ptr = c_void_p()
            result = vk.vkMapMemory(self.device, dst.memory, dst.offset, dst.size, data_ptr)
            if result != vk.VK_SUCCESS:
                raise MemoryError(f"Failed to map memory: {result}")
                
            try:
                # Copy data using memmove
                src_ptr = src.ctypes.data_as(c_void_p)
                memmove(data_ptr, src_ptr, dst.size)
                
            finally:
                # Always unmap
                vk.vkUnmapMemory(self.device, dst.memory)
                
            self.record_metric("copy")
            
        except Exception as e:
            raise MemoryError(f"Failed to copy to device: {e}")

    def copy_from_device(self, src: VulkanBuffer, dst: np.ndarray) -> None:
        """Copy data from Vulkan buffer.
        
        Args:
            src: Source buffer
            dst: Destination NumPy array
        """
        try:
            # Map memory
            data_ptr = c_void_p()
            result = vk.vkMapMemory(self.device, src.memory, src.offset, src.size, data_ptr)
            if result != vk.VK_SUCCESS:
                raise MemoryError(f"Failed to map memory: {result}")
                
            try:
                # Copy data using memmove
                dst_ptr = dst.ctypes.data_as(c_void_p)
                memmove(dst_ptr, data_ptr, src.size)
                
            finally:
                # Always unmap
                vk.vkUnmapMemory(self.device, src.memory)
                
            self.record_metric("copy")
            
        except Exception as e:
            raise MemoryError(f"Failed to copy from device: {e}")

    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self._allocated_memory:
            return 0.0
            
        # Calculate fragmentation based on buffer allocation patterns
        total_gaps = 0
        sorted_allocations = sorted(b.size for b in self._buffer_pool.values())
        
        for i in range(len(sorted_allocations) - 1):
            gap = sorted_allocations[i + 1] - sorted_allocations[i]
            if gap > 0:
                total_gaps += gap
                
        return total_gaps / self._allocated_memory if self._allocated_memory else 0.0

    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        """Find suitable memory type index.
        
        Args:
            type_filter: Memory type bits
            properties: Required memory properties
            
        Returns:
            Memory type index
        """
        try:
            mem_properties = vk.VkPhysicalDeviceMemoryProperties()
            vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, byref(mem_properties))
            
            for i in range(int(mem_properties.memoryTypeCount)):
                if (type_filter & (1 << i)) and (
                    mem_properties.memoryTypes[i].propertyFlags & properties
                ) == properties:
                    return i
                    
            raise MemoryError("Failed to find suitable memory type")
            
        except Exception as e:
            raise MemoryError(f"Failed to find memory type: {e}")

    def _clear_buffer_pool(self) -> None:
        """Clear buffer pool."""
        try:
            # Free all buffers
            for buffer_info in self._buffer_pool.values():
                vk.vkFreeMemory(self.device, buffer_info.memory, None)
                vk.vkDestroyBuffer(self.device, buffer_info.buffer, None)
                
            # Reset tracking
            self._buffer_pool.clear()
            self._allocated_memory = 0
            
            self.record_metric("clear_pool")
            
        except Exception as e:
            raise MemoryError(f"Failed to clear buffer pool: {e}")

    def cleanup(self) -> None:
        """Clean up memory resources."""
        try:
            self._clear_buffer_pool()
            self.record_metric("cleanup")
            
        except Exception as e:
            raise MemoryError(f"Failed to cleanup: {e}")

    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.cleanup()
        except:
            pass 