"""Vulkan Memory Pool Management.

This module provides efficient memory allocation and pooling for Vulkan resources,
implementing best practices for memory management including:
- Memory type selection based on requirements
- Efficient pooling and reuse
- Proper alignment handling
- Host-visible memory mapping
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from ctypes import c_void_p

import vulkan as vk
from vulkan.ext_types import VkDevice, VkPhysicalDevice, VkDeviceMemory, VkBuffer


@dataclass
class MemoryBlock:
    """Represents a block of GPU memory."""

    memory: VkDeviceMemory
    size: int
    offset: int
    is_free: bool
    alignment: int
    memory_type: int
    mapped_ptr: Optional[c_void_p] = None


class MemoryPool:
    """Manages a pool of Vulkan memory blocks for efficient reuse."""

    def __init__(
        self,
        device: VkDevice,
        physical_device: VkPhysicalDevice,
        initial_size: int = 1024 * 1024,
    ):
        self.device = device
        self.physical_device = physical_device
        self.initial_size = initial_size

        # Get memory properties
        self.memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)

        # Memory pools for different usage patterns
        self.pools: Dict[int, List[MemoryBlock]] = {
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: [],
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: [],
        }

        # Statistics for adaptive sizing
        self.allocation_stats = {
            "hits": 0,
            "misses": 0,
            "fragmentation": 0.0,
            "mapped_blocks": 0,
        }

    def allocate(
        self,
        size: int,
        alignment: int,
        memory_properties: int,
        memory_type_bits: int,
    ) -> Tuple[VkDeviceMemory, int]:
        """Allocate memory from the pool.
        
        Args:
            size: Size in bytes
            alignment: Required alignment
            memory_properties: Desired memory properties flags
            memory_type_bits: Memory type bits from requirements
            
        Returns:
            Tuple of (memory handle, offset)
        """
        # Find suitable memory type
        memory_type = self._find_memory_type(memory_type_bits, memory_properties)
        if memory_type == -1:
            raise RuntimeError("Failed to find suitable memory type")

        # Adjust size for alignment
        aligned_size = (size + alignment - 1) & ~(alignment - 1)

        # Try to find existing block
        block = self._find_free_block(aligned_size, alignment, memory_properties)
        if block is not None:
            self.allocation_stats["hits"] += 1
            block.is_free = False
            return block.memory, block.offset

        self.allocation_stats["misses"] += 1

        # Create new block if none found
        allocation_size = max(aligned_size, self.initial_size)
        try:
            memory = self._allocate_memory(allocation_size, memory_type)
        except Exception as e:
            raise RuntimeError(f"Failed to allocate memory: {e}")

        block = MemoryBlock(
            memory=memory,
            size=allocation_size,
            offset=0,
            is_free=False,
            alignment=alignment,
            memory_type=memory_type,
        )

        # Map memory if host visible
        if memory_properties & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
            try:
                block.mapped_ptr = self._map_memory(block)
                self.allocation_stats["mapped_blocks"] += 1
            except Exception as e:
                vk.vkFreeMemory(self.device, memory, None)
                raise RuntimeError(f"Failed to map memory: {e}")

        self.pools[memory_properties].append(block)
        return memory, 0

    def free(self, memory: VkDeviceMemory, offset: int):
        """Return memory block to the pool."""
        for blocks in self.pools.values():
            for block in blocks:
                if block.memory == memory and block.offset == offset:
                    if block.mapped_ptr is not None:
                        vk.vkUnmapMemory(self.device, block.memory)
                        block.mapped_ptr = None
                        self.allocation_stats["mapped_blocks"] -= 1
                    block.is_free = True
                    self._maybe_merge_blocks(blocks)
                    self._update_fragmentation()
                    return

    def bind_buffer(
        self, buffer: VkBuffer, memory: VkDeviceMemory, offset: int
    ) -> None:
        """Bind buffer to allocated memory."""
        try:
            vk.vkBindBufferMemory(self.device, buffer, memory, offset)
        except Exception as e:
            raise RuntimeError(f"Failed to bind buffer memory: {e}")

    def _find_memory_type(
        self, type_filter: int, properties: int
    ) -> int:
        """Find suitable memory type index."""
        mem_properties = self.memory_properties
        for i in range(mem_properties.memoryTypeCount):
            if ((type_filter & (1 << i)) and 
                (mem_properties.memoryTypes[i].propertyFlags & properties) == properties):
                return i
        return -1

    def _find_free_block(
        self, size: int, alignment: int, memory_properties: int
    ) -> Optional[MemoryBlock]:
        """Find a suitable free block."""
        if memory_properties not in self.pools:
            return None

        blocks = self.pools[memory_properties]

        # Best-fit strategy
        best_fit = None
        min_waste = float("inf")

        for block in blocks:
            if not block.is_free:
                continue

            # Check alignment
            aligned_offset = (block.offset + alignment - 1) & ~(alignment - 1)
            waste = aligned_offset - block.offset

            if block.size - waste >= size and waste < min_waste:
                best_fit = block
                min_waste = waste

        return best_fit

    def _allocate_memory(self, size: int, memory_type: int) -> VkDeviceMemory:
        """Allocate new Vulkan memory."""
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=size,
            memoryTypeIndex=memory_type,
        )

        try:
            return vk.vkAllocateMemory(self.device, alloc_info, None)
        except Exception as e:
            raise RuntimeError(f"Memory allocation failed: {e}")

    def _map_memory(self, block: MemoryBlock) -> c_void_p:
        """Map host-visible memory."""
        try:
            return vk.vkMapMemory(
                self.device,
                block.memory,
                offset=block.offset,
                size=block.size,
                flags=0,
            )
        except Exception as e:
            raise RuntimeError(f"Memory mapping failed: {e}")

    def _maybe_merge_blocks(self, blocks: List[MemoryBlock]):
        """Merge adjacent free blocks."""
        i = 0
        while i < len(blocks) - 1:
            curr = blocks[i]
            next_block = blocks[i + 1]

            if (
                curr.is_free
                and next_block.is_free
                and curr.memory == next_block.memory
                and curr.offset + curr.size == next_block.offset
                and curr.memory_type == next_block.memory_type
            ):
                # Merge blocks
                curr.size += next_block.size
                if next_block.mapped_ptr is not None:
                    vk.vkUnmapMemory(self.device, next_block.memory)
                    self.allocation_stats["mapped_blocks"] -= 1
                blocks.pop(i + 1)
                continue

            i += 1

    def _update_fragmentation(self):
        """Update fragmentation statistics."""
        total_size = 0
        free_size = 0

        for blocks in self.pools.values():
            for block in blocks:
                total_size += block.size
                if block.is_free:
                    free_size += block.size

        if total_size > 0:
            self.allocation_stats["fragmentation"] = 1.0 - (free_size / total_size)

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            "hits": self.allocation_stats["hits"],
            "misses": self.allocation_stats["misses"],
            "fragmentation": self.allocation_stats["fragmentation"],
            "total_pools": sum(len(blocks) for blocks in self.pools.values()),
            "mapped_blocks": self.allocation_stats["mapped_blocks"],
        }

    def cleanup(self):
        """Free all allocated memory."""
        for blocks in self.pools.values():
            for block in blocks:
                if block.mapped_ptr is not None:
                    vk.vkUnmapMemory(self.device, block.memory)
                vk.vkFreeMemory(self.device, block.memory, None)
        self.pools.clear()
        self.allocation_stats = {
            "hits": 0,
            "misses": 0,
            "fragmentation": 0.0,
            "mapped_blocks": 0,
        }
