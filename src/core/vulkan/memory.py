"""Vulkan memory management."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from ctypes import c_void_p, cast, POINTER, Structure

import vulkan as vk


@dataclass
class MemoryBlock:
    """Block of allocated memory."""
    
    memory: Any  # VkDeviceMemory handle as CData
    size: int
    offset: int
    type_index: int


class VulkanMemory:
    """Manages Vulkan memory allocation."""
    
    def __init__(self, device: int, physical_device: int):
        """Initialize memory manager.
        
        Args:
            device: Vulkan device handle
            physical_device: Physical device handle
        """
        self.device = device
        self.physical_device = physical_device
        
        # Get memory properties
        memory_props = vk.VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(
            physicalDevice=physical_device,
            pMemoryProperties=memory_props
        )
        self.memory_properties = memory_props
        
        # Track allocations
        self.allocations: Dict[int, MemoryBlock] = {}
        
    def allocate(
        self,
        size: int,
        memory_type_bits: int,
        properties: int,
    ) -> MemoryBlock:
        """Allocate memory block.
        
        Args:
            size: Size in bytes
            memory_type_bits: Memory type bits
            properties: Memory property flags
            
        Returns:
            Allocated memory block
        """
        # Find memory type
        type_index = self._find_memory_type(memory_type_bits, properties)
        
        # Allocate memory
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=size,
            memoryTypeIndex=type_index,
        )
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        
        # Create block
        block = MemoryBlock(
            memory=memory,  # CData object from Vulkan
            size=size,
            offset=0,
            type_index=type_index,
        )
        
        # Track allocation
        self.allocations[id(memory)] = block
        
        return block
        
    def free(self, block: MemoryBlock):
        """Free memory block.
        
        Args:
            block: Memory block to free
        """
        vk.vkFreeMemory(self.device, block.memory, None)
        del self.allocations[id(block.memory)]
        
    def _find_memory_type(self, type_bits: int, properties: int) -> int:
        """Find suitable memory type.
        
        Args:
            type_bits: Memory type bits
            properties: Required properties
            
        Returns:
            Memory type index
        """
        # Access memory properties directly
        for i in range(self.memory_properties.memoryTypeCount):
            if (type_bits & (1 << i)) and (
                self.memory_properties.memoryTypes[i].propertyFlags & properties
            ) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")
