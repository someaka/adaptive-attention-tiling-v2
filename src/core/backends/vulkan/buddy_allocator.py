"""Vulkan Memory Buddy Allocator.

This module implements a buddy memory allocation system for Vulkan memory management,
providing efficient memory allocation and deallocation with minimal fragmentation.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from ctypes import c_void_p, c_uint32, POINTER, Structure, c_int, c_uint64

import vulkan as vk


class VkMemoryType(Structure):
    """Vulkan memory type structure."""
    _fields_ = [
        ("propertyFlags", c_int),
        ("heapIndex", c_uint32),
    ]


class VkMemoryHeap(Structure):
    """Vulkan memory heap structure."""
    _fields_ = [
        ("size", c_uint64),
        ("flags", c_int),
    ]


class VkPhysicalDeviceMemoryProperties(Structure):
    """Vulkan physical device memory properties structure."""
    _fields_ = [
        ("memoryTypeCount", c_uint32),
        ("memoryTypes", VkMemoryType * vk.VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", c_uint32),
        ("memoryHeaps", VkMemoryHeap * vk.VK_MAX_MEMORY_HEAPS),
    ]


@dataclass
class BuddyBlock:
    """Represents a block in the buddy memory system."""

    memory: c_void_p  # VkDeviceMemory handle
    offset: int
    size: int
    order: int  # Power of 2 size index
    is_free: bool
    buddy_idx: int  # Index of buddy block


class BuddyAllocator:
    """Memory allocator using buddy system."""

    def __init__(
        self, 
        device: c_void_p,  # VkDevice handle
        physical_device: c_void_p,  # VkPhysicalDevice handle
        min_block_size: int, 
        max_order: int
    ):
        """Initialize buddy allocator.
        
        Args:
            device: Vulkan device handle
            physical_device: Physical device handle
            min_block_size: Minimum block size in bytes
            max_order: Maximum order (power of 2) for block sizes
        """
        self.device = device
        self.physical_device = physical_device
        self.min_block_size = min_block_size
        self.max_order = max_order

        # Get memory properties
        self.memory_properties = VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(physical_device, self.memory_properties)

        # Initialize free lists for each order
        self.free_lists: List[List[BuddyBlock]] = [[] for _ in range(max_order + 1)]

        # Track allocated blocks
        self.allocated_blocks: Dict[Tuple[c_void_p, int], BuddyBlock] = {}

        # Memory pools for different usage patterns
        self.memory_pools: Dict[int, List[c_void_p]] = {}

        # Statistics
        self.stats = {
            "allocations": 0,
            "frees": 0,
            "splits": 0,
            "merges": 0,
            "fragmentation": 0.0,
        }

    def allocate(
        self, size: int, alignment: int, memory_properties: int, memory_type_bits: int
    ) -> Tuple[c_void_p, int]:
        """Allocate memory using buddy system.
        
        Args:
            size: Size in bytes
            alignment: Required alignment
            memory_properties: Memory property flags
            memory_type_bits: Memory type bits from requirements
            
        Returns:
            Tuple of (memory handle, offset)
        """
        # Find suitable memory type
        memory_type = self._find_memory_type(memory_type_bits, memory_properties)
        if memory_type == -1:
            raise RuntimeError("Failed to find suitable memory type")

        # Calculate required order
        aligned_size = max(size, self.min_block_size)
        aligned_size = (aligned_size + alignment - 1) & ~(alignment - 1)
        required_order = max(
            0, math.ceil(math.log2(aligned_size / self.min_block_size))
        )

        # Find smallest available block that fits
        order = required_order
        while order <= self.max_order:
            if self.free_lists[order]:
                break
            order += 1

        # Need to allocate new memory if no block found
        if order > self.max_order:
            self._allocate_new_pool(memory_properties, memory_type, self.max_order)
            order = self.max_order

        # Get block and split if necessary
        block = self.free_lists[order].pop()

        while order > required_order:
            self.stats["splits"] += 1
            order -= 1
            size = self.min_block_size << order

            # Create buddy block
            buddy = BuddyBlock(
                memory=block.memory,
                offset=block.offset + size,
                size=size,
                order=order,
                is_free=True,
                buddy_idx=block.buddy_idx ^ 1,
            )

            # Update original block
            block.size = size
            block.order = order

            # Add buddy to free list
            self.free_lists[order].append(buddy)

        block.is_free = False
        self.allocated_blocks[(block.memory, block.offset)] = block
        self.stats["allocations"] += 1

        return block.memory, block.offset

    def free(self, memory: c_void_p, offset: int):
        """Free a memory block and merge with buddy if possible."""
        key = (memory, offset)
        if key not in self.allocated_blocks:
            return

        block = self.allocated_blocks.pop(key)
        block.is_free = True

        self.stats["frees"] += 1

        # Try to merge with buddy
        while block.order < self.max_order:
            buddy_offset = block.offset ^ (block.size)
            buddy_key = (block.memory, buddy_offset)

            # Find buddy in free lists
            buddy = None
            for b in self.free_lists[block.order]:
                if b.memory == block.memory and b.offset == buddy_offset:
                    buddy = b
                    break

            if not buddy or not buddy.is_free:
                break

            # Remove buddy from free list
            self.free_lists[block.order].remove(buddy)

            # Merge blocks
            self.stats["merges"] += 1
            if block.offset > buddy.offset:
                block, buddy = buddy, block

            block.size *= 2
            block.order += 1
            block.buddy_idx = block.buddy_idx & ~1

        # Add merged/unmerged block to free list
        self.free_lists[block.order].append(block)
        self._update_fragmentation()

    def _find_memory_type(
        self, type_filter: int, properties: int
    ) -> int:
        """Find suitable memory type index."""
        for i in range(self.memory_properties.memoryTypeCount):
            memory_type = self.memory_properties.memoryTypes[i]
            if ((type_filter & (1 << i)) and 
                (memory_type.propertyFlags & properties) == properties):
                return i
        return -1

    def _allocate_new_pool(
        self, memory_properties: int, memory_type: int, order: int
    ):
        """Allocate a new memory pool."""
        size = self.min_block_size << order

        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=size,
            memoryTypeIndex=memory_type,
        )

        try:
            memory = c_void_p()
            result = vk.vkAllocateMemory(self.device, alloc_info, None, memory)
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to allocate memory: {result}")
        except Exception as e:
            raise RuntimeError(f"Failed to allocate memory: {e}")

        if memory_properties not in self.memory_pools:
            self.memory_pools[memory_properties] = []
        self.memory_pools[memory_properties].append(memory)

        # Create initial free block
        block = BuddyBlock(
            memory=memory, offset=0, size=size, order=order, is_free=True, buddy_idx=0
        )

        self.free_lists[order].append(block)

    def _update_fragmentation(self):
        """Calculate current memory fragmentation."""
        total_size = 0
        free_size = 0

        for order, blocks in enumerate(self.free_lists):
            block_size = self.min_block_size << order
            for block in blocks:
                if block.is_free:
                    free_size += block_size
                total_size += block_size

        if total_size > 0:
            self.stats["fragmentation"] = 1.0 - (free_size / total_size)

    def get_stats(self) -> Dict:
        """Get allocator statistics."""
        return {
            "allocations": self.stats["allocations"],
            "frees": self.stats["frees"],
            "splits": self.stats["splits"],
            "merges": self.stats["merges"],
            "fragmentation": self.stats["fragmentation"],
            "total_pools": sum(len(pools) for pools in self.memory_pools.values()),
        }

    def cleanup(self):
        """Free all allocated memory."""
        for pools in self.memory_pools.values():
            for memory in pools:
                vk.vkFreeMemory(self.device, memory, None)

        self.free_lists = [[] for _ in range(self.max_order + 1)]
        self.allocated_blocks.clear()
        self.memory_pools.clear()
        self.stats = {
            "allocations": 0,
            "frees": 0,
            "splits": 0,
            "merges": 0,
            "fragmentation": 0.0,
        }
