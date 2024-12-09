"""Vulkan memory management system for adaptive attention tiling."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple

import vulkan as vk

from src.core.common.constants import (
    DEFAULT_BUFFER_BLOCK_SIZE,
)


class BufferUsage(Enum):
    """Buffer usage flags for memory allocation."""

    TILE_STATE = auto()
    CROSS_TILE = auto()
    METRICS = auto()
    UNIFORM = auto()


@dataclass
class BufferBlock:
    """Represents a block of Vulkan memory."""

    buffer: int  # VkBuffer
    memory: int  # VkDeviceMemory
    size: int
    offset: int
    is_free: bool = True


class MemoryPool:
    """Manages a pool of similar-purpose buffers."""

    def __init__(
        self,
        device: int,  # VkDevice
        usage: BufferUsage,
        block_size: int = DEFAULT_BUFFER_BLOCK_SIZE,
    ):
        self.device = device
        self.usage = usage
        self.block_size = block_size
        self.blocks: List[BufferBlock] = []
        self.allocations: Dict[int, BufferBlock] = {}  # id -> block mapping

    def allocate(self, size: int) -> Tuple[int, int, int]:  # Returns (VkBuffer, VkDeviceMemory, offset)
        """Allocate a buffer of specified size."""
        # Find existing free block that fits
        for block in self.blocks:
            if block.is_free and block.size >= size:
                block.is_free = False
                self.allocations[id(block.buffer)] = block
                return block.buffer, block.memory, block.offset

        # Create new block if none found
        block_size = max(size, self.block_size)
        buffer_info = vk.VkBufferCreateInfo(
            size=block_size,
            usage=self._get_usage_flags(),
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )

        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)

        # Get memory requirements
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)

        # Allocate memory
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_reqs.size,
            memoryTypeIndex=self._find_memory_type(mem_reqs.memoryTypeBits),
        )

        memory = vk.vkAllocateMemory(self.device, alloc_info, None)

        # Bind memory to buffer
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)

        block = BufferBlock(buffer=buffer, memory=memory, size=block_size, offset=0, is_free=False)
        self.blocks.append(block)
        self.allocations[id(buffer)] = block

        return buffer, memory, 0

    def free(self, buffer: int) -> None:  # VkBuffer
        """Free an allocated buffer."""
        if buffer_id := id(buffer) in self.allocations:
            block = self.allocations[buffer_id]
            block.is_free = True
            del self.allocations[buffer_id]

    def cleanup(self) -> None:
        """Clean up all allocations."""
        for block in self.blocks:
            vk.vkDestroyBuffer(self.device, block.buffer, None)
            vk.vkFreeMemory(self.device, block.memory, None)
        self.blocks.clear()
        self.allocations.clear()

    def _get_usage_flags(self) -> int:
        """Get buffer usage flags based on usage type."""
        if self.usage == BufferUsage.UNIFORM:
            return vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        return (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )

    def _find_memory_type(self, type_filter: int) -> int:
        """Find suitable memory type index."""
        props = vk.vkGetPhysicalDeviceMemoryProperties(self.device)
        for i in range(props.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                props.memoryTypes[i].propertyFlags
                & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            ):
                return i
        raise RuntimeError("Failed to find suitable memory type")


class VulkanMemoryManager:
    """Manages Vulkan memory allocation and buffers."""

    def __init__(self, device: int):  # VkDevice
        self.device = device
        self._pools: Dict[BufferUsage, MemoryPool] = {
            usage: MemoryPool(device, usage) for usage in BufferUsage
        }

    def allocate_tile_state(self, size: int) -> Tuple[int, int, int]:  # Returns (VkBuffer, VkDeviceMemory, offset)
        """Allocate memory for tile state."""
        return self._pools[BufferUsage.TILE_STATE].allocate(size)

    def allocate_cross_tile(self, size: int) -> Tuple[int, int, int]:  # Returns (VkBuffer, VkDeviceMemory, offset)
        """Allocate memory for cross-tile communication."""
        return self._pools[BufferUsage.CROSS_TILE].allocate(size)

    def allocate_metrics(self, size: int) -> Tuple[int, int, int]:  # Returns (VkBuffer, VkDeviceMemory, offset)
        """Allocate memory for metrics collection."""
        return self._pools[BufferUsage.METRICS].allocate(size)

    def allocate_uniform(self, size: int) -> Tuple[int, int, int]:  # Returns (VkBuffer, VkDeviceMemory, offset)
        """Allocate memory for uniform buffers."""
        return self._pools[BufferUsage.UNIFORM].allocate(size)

    def free_buffer(self, buffer: int, usage: BufferUsage) -> None:  # VkBuffer
        """Free an allocated buffer."""
        self._pools[usage].free(buffer)

    def cleanup(self) -> None:
        """Clean up all allocations."""
        for pool in self._pools.values():
            pool.cleanup()
