"""Enhanced command pool management with advanced features."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Optional
from ctypes import c_void_p, c_uint32, byref, POINTER, Structure

import vulkan as vk

# Constants
MAX_COMMAND_BUFFERS_PER_POOL = 32  # Moved constant here since it was not found in common.constants

# Vulkan type definitions
VkDevice = c_void_p
VkCommandPool = c_void_p
VkCommandBuffer = c_void_p

# Vulkan enums and flags
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_LEVEL_SECONDARY = 1

VK_COMMAND_POOL_CREATE_TRANSIENT_BIT = 0x00000001
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
VK_COMMAND_POOL_CREATE_PROTECTED_BIT = 0x00000004

VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT = 0x00000001


class CommandPoolType(Enum):
    """Types of command pools for different usage patterns."""

    TRANSIENT = auto()  # Short-lived commands
    PERSISTENT = auto()  # Long-lived commands
    PROTECTED = auto()  # Protected memory operations


@dataclass
class CommandPoolConfig:
    """Configuration for command pool creation."""

    type: CommandPoolType
    max_buffers: int = MAX_COMMAND_BUFFERS_PER_POOL
    allow_reset: bool = True
    protected: bool = False


class CommandPoolManager:
    """Advanced command pool management system."""

    def __init__(self, device: VkDevice, queue_family_index: int):
        self.device = device
        self.queue_family_index = queue_family_index
        self._pools: Dict[CommandPoolType, List[VkCommandPool]] = {
            pool_type: [] for pool_type in CommandPoolType
        }
        self._buffer_pools: Dict[VkCommandBuffer, VkCommandPool] = {}
        self._available_pools: Dict[CommandPoolType, Set[VkCommandPool]] = {
            pool_type: set() for pool_type in CommandPoolType
        }

    def create_pool(self, config: CommandPoolConfig) -> VkCommandPool:
        """Create a new command pool with specified configuration."""
        flags = self._get_pool_flags(config)

        create_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext=None,
            flags=flags,
            queueFamilyIndex=self.queue_family_index
        )

        pool = c_void_p()
        result = vk.vkCreateCommandPool(self.device, byref(create_info), None, byref(pool))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create command pool: {result}")

        self._pools[config.type].append(pool)
        self._available_pools[config.type].add(pool)

        return pool

    def allocate_buffers(
        self,
        pool_type: CommandPoolType,
        count: int,
        level: int = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    ) -> List[VkCommandBuffer]:
        """Allocate command buffers from appropriate pool."""
        pool = self._get_or_create_pool(pool_type)

        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=pool,
            level=level,
            commandBufferCount=count
        )

        command_buffers = (c_void_p * count)()
        result = vk.vkAllocateCommandBuffers(self.device, byref(alloc_info), command_buffers)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate command buffers: {result}")

        buffers = [command_buffers[i] for i in range(count)]
        
        # Track buffer-pool associations
        for buffer in buffers:
            self._buffer_pools[buffer] = pool

        return buffers

    def reset_pool(self, pool: VkCommandPool, release_resources: bool = False):
        """Reset a command pool and optionally release resources."""
        flags = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT if release_resources else 0
        result = vk.vkResetCommandPool(self.device, pool, flags)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to reset command pool: {result}")

    def trim_pools(self):
        """Trim unused memory from all pools."""
        for pools in self._pools.values():
            for pool in pools:
                vk.vkTrimCommandPool(self.device, pool, 0)

    def free_buffers(self, buffers: List[VkCommandBuffer]):
        """Free command buffers back to their pools."""
        pool_buffers: Dict[VkCommandPool, List[VkCommandBuffer]] = {}

        # Group buffers by their source pools
        for buffer in buffers:
            pool = self._buffer_pools.get(buffer)
            if pool:
                if pool not in pool_buffers:
                    pool_buffers[pool] = []
                pool_buffers[pool].append(buffer)
                del self._buffer_pools[buffer]

        # Free buffers pool by pool
        for pool, pool_buffers_list in pool_buffers.items():
            buffer_array = (c_void_p * len(pool_buffers_list))(*pool_buffers_list)
            vk.vkFreeCommandBuffers(self.device, pool, len(pool_buffers_list), buffer_array)

    def _get_or_create_pool(self, type: CommandPoolType) -> VkCommandPool:
        """Get an available pool or create new one if needed."""
        if not self._available_pools[type]:
            config = CommandPoolConfig(type=type)
            return self.create_pool(config)

        return next(iter(self._available_pools[type]))

    def _get_pool_flags(self, config: CommandPoolConfig) -> int:
        """Get command pool creation flags based on configuration."""
        flags = 0

        if config.type == CommandPoolType.TRANSIENT:
            flags |= VK_COMMAND_POOL_CREATE_TRANSIENT_BIT

        if config.allow_reset:
            flags |= VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT

        if config.protected:
            flags |= VK_COMMAND_POOL_CREATE_PROTECTED_BIT

        return flags

    def cleanup(self):
        """Clean up all command pools."""
        for pools in self._pools.values():
            for pool in pools:
                vk.vkDestroyCommandPool(self.device, pool, None)

        self._pools.clear()
        self._buffer_pools.clear()
        self._available_pools.clear()
