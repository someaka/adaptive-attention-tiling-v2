"""Vulkan Memory Pool Management.

This module provides a memory pool system for efficient allocation and reuse
of Vulkan memory resources, reducing fragmentation and allocation overhead.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ctypes import c_void_p, c_uint32, c_int, POINTER, Structure, cast, byref

import vulkan as vk


@dataclass
class MemoryBlock:
    """Represents a block of allocated memory."""

    memory: c_void_p  # VkDeviceMemory
    size: int
    offset: int
    is_free: bool = True


@dataclass
class MemoryPool:
    """Pool of memory blocks with the same properties."""

    total_size: int
    used_size: int
    blocks: List[MemoryBlock]
    memory_type_index: int


class MemoryPoolManager:
    """Manages memory pools for different memory types."""

    def __init__(self, device: c_void_p, physical_device: c_void_p):
        self.device = device
        self.physical_device = physical_device
        self.pools: Dict[int, MemoryPool] = {}  # memory_type_index -> pool
        self.block_size = 64 * 1024 * 1024  # 64MB default block size
        self.min_block_size = 1024 * 1024  # 1MB minimum block size

        # Get memory properties
        self.memory_properties = vk.VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(physical_device, byref(self.memory_properties))

    def _find_memory_type(
        self, type_filter: int, properties: int
    ) -> int:
        """Find suitable memory type index."""
        for i in range(self.memory_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                self.memory_properties.memoryTypes[i].propertyFlags & properties
            ) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")

    def _create_pool(self, memory_type_index: int, initial_size: int) -> MemoryPool:
        """Create a new memory pool."""
        # Allocate initial memory block
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=initial_size,
            memoryTypeIndex=memory_type_index
        )

        memory = c_void_p()
        result = vk.vkAllocateMemory(self.device, byref(alloc_info), None, byref(memory))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate memory: {result}")

        block = MemoryBlock(memory=memory, size=initial_size, offset=0)
        return MemoryPool(
            total_size=initial_size,
            used_size=0,
            blocks=[block],
            memory_type_index=memory_type_index,
        )

    def _find_free_block(self, pool: MemoryPool, size: int) -> Optional[MemoryBlock]:
        """Find a free block that can accommodate the requested size."""
        for block in pool.blocks:
            if block.is_free and block.size >= size:
                return block
        return None

    def _split_block(
        self, pool: MemoryPool, block: MemoryBlock, size: int
    ) -> Tuple[MemoryBlock, Optional[MemoryBlock]]:
        """Split a block if it's significantly larger than requested size."""
        min_split_size = max(self.min_block_size, size * 2)

        if block.size >= min_split_size:
            # Create new block for remaining space
            remaining_size = block.size - size
            remaining_offset = block.offset + size

            new_block = MemoryBlock(
                memory=block.memory, size=remaining_size, offset=remaining_offset
            )

            # Update original block
            block.size = size
            block.is_free = False

            pool.blocks.append(new_block)
            return block, new_block

        # No split needed
        block.is_free = False
        return block, None

    def allocate(self, size: int, memory_type_index: int) -> MemoryBlock:
        """Allocate memory from pool."""
        # Round up size to alignment
        aligned_size = ((size + 255) // 256) * 256

        # Get or create pool
        pool = self.pools.get(memory_type_index)
        if not pool:
            initial_size = max(self.block_size, aligned_size)
            pool = self._create_pool(memory_type_index, initial_size)
            self.pools[memory_type_index] = pool

        # Find free block
        block = self._find_free_block(pool, aligned_size)
        if block:
            used_block, _ = self._split_block(pool, block, aligned_size)
            pool.used_size += aligned_size
            return used_block

        # Need to allocate new block
        new_size = max(self.block_size, aligned_size)
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=new_size,
            memoryTypeIndex=memory_type_index
        )

        memory = c_void_p()
        try:
            result = vk.vkAllocateMemory(self.device, byref(alloc_info), None, byref(memory))
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to allocate memory: {result}")
        except Exception:
            # Try garbage collection
            self.garbage_collect()
            # Retry allocation
            result = vk.vkAllocateMemory(self.device, byref(alloc_info), None, byref(memory))
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to allocate memory after garbage collection: {result}")

        block = MemoryBlock(memory=memory, size=new_size, offset=0, is_free=False)

        pool.blocks.append(block)
        pool.total_size += new_size
        pool.used_size += aligned_size

        return block

    def free(self, block: MemoryBlock) -> None:
        """Mark a block as free for reuse."""
        block.is_free = True

        # Update pool statistics
        for pool in self.pools.values():
            if block in pool.blocks:
                pool.used_size -= block.size
                break

    def garbage_collect(self) -> None:
        """Perform garbage collection on memory pools."""
        for memory_type_index, pool in self.pools.items():
            # Collect consecutive free blocks
            i = 0
            while i < len(pool.blocks) - 1:
                current = pool.blocks[i]
                next_block = pool.blocks[i + 1]

                if (
                    current.is_free
                    and next_block.is_free
                    and current.memory == next_block.memory
                ):
                    # Merge blocks
                    current.size += next_block.size
                    pool.blocks.pop(i + 1)
                else:
                    i += 1

            # Remove completely free memory allocations
            memory_blocks: Dict[c_void_p, List[MemoryBlock]] = defaultdict(list)
            for block in pool.blocks:
                memory_blocks[block.memory].append(block)

            for memory, blocks in memory_blocks.items():
                if all(block.is_free for block in blocks):
                    # All blocks in this memory allocation are free
                    vk.vkFreeMemory(self.device, memory, None)
                    pool.blocks = [b for b in pool.blocks if b.memory != memory]
                    pool.total_size -= sum(block.size for block in blocks)

    def cleanup(self) -> None:
        """Clean up all memory pools."""
        for pool in self.pools.values():
            # Get unique memory handles
            memory_handles: Set[c_void_p] = {
                block.memory for block in pool.blocks
            }

            # Free all memory
            for memory in memory_handles:
                vk.vkFreeMemory(self.device, memory, None)

            pool.blocks.clear()
            pool.total_size = 0
            pool.used_size = 0

        self.pools.clear()
