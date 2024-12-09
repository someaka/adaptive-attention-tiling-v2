"""Enhanced command pool management with advanced features."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set

import vulkan as vk

from src.core.common.constants import MAX_COMMAND_BUFFERS_PER_POOL


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

    def __init__(self, device: vk.Device, queue_family_index: int):
        self.device = device
        self.queue_family_index = queue_family_index
        self._pools: Dict[CommandPoolType, List[vk.CommandPool]] = {
            pool_type: [] for pool_type in CommandPoolType
        }
        self._buffer_pools: Dict[vk.CommandBuffer, vk.CommandPool] = {}
        self._available_pools: Dict[CommandPoolType, Set[vk.CommandPool]] = {
            pool_type: set() for pool_type in CommandPoolType
        }

    def create_pool(self, config: CommandPoolConfig) -> vk.CommandPool:
        """Create a new command pool with specified configuration."""
        flags = self._get_pool_flags(config)

        pool_info = vk.CommandPoolCreateInfo(
            flags=flags, queue_family_index=self.queue_family_index
        )

        pool = vk.create_command_pool(self.device, pool_info, None)
        self._pools[config.type].append(pool)
        self._available_pools[config.type].add(pool)

        return pool

    def allocate_buffers(
        self,
        pool_type: CommandPoolType,
        count: int,
        level: vk.CommandBufferLevel = vk.CommandBufferLevel.PRIMARY,
    ) -> List[vk.CommandBuffer]:
        """Allocate command buffers from appropriate pool."""
        pool = self._get_or_create_pool(pool_type)

        alloc_info = vk.CommandBufferAllocateInfo(
            command_pool=pool, level=level, command_buffer_count=count
        )

        buffers = vk.allocate_command_buffers(self.device, alloc_info)

        # Track buffer-pool associations
        for buffer in buffers:
            self._buffer_pools[buffer] = pool

        return buffers

    def reset_pool(self, pool: vk.CommandPool, release_resources: bool = False):
        """Reset a command pool and optionally release resources."""
        flags = vk.CommandPoolResetFlags.RELEASE_RESOURCES if release_resources else 0
        vk.reset_command_pool(self.device, pool, flags)

    def trim_pools(self):
        """Trim unused memory from all pools."""
        for pools in self._pools.values():
            for pool in pools:
                vk.trim_command_pool(self.device, pool, 0)

    def free_buffers(self, buffers: List[vk.CommandBuffer]):
        """Free command buffers back to their pools."""
        pool_buffers: Dict[vk.CommandPool, List[vk.CommandBuffer]] = {}

        # Group buffers by their source pools
        for buffer in buffers:
            pool = self._buffer_pools.get(buffer)
            if pool:
                pool_buffers.setdefault(pool, []).append(buffer)
                del self._buffer_pools[buffer]

        # Free buffers pool by pool
        for pool, pool_buffers in pool_buffers.items():
            vk.free_command_buffers(self.device, pool, pool_buffers)

    def _get_or_create_pool(self, type: CommandPoolType) -> vk.CommandPool:
        """Get an available pool or create new one if needed."""
        if not self._available_pools[type]:
            config = CommandPoolConfig(type=type)
            return self.create_pool(config)

        return next(iter(self._available_pools[type]))

    def _get_pool_flags(self, config: CommandPoolConfig) -> vk.CommandPoolCreateFlags:
        """Get command pool creation flags based on configuration."""
        flags = vk.CommandPoolCreateFlags(0)

        if config.type == CommandPoolType.TRANSIENT:
            flags |= vk.CommandPoolCreateFlags.TRANSIENT

        if config.allow_reset:
            flags |= vk.CommandPoolCreateFlags.RESET_COMMAND_BUFFER

        if config.protected:
            flags |= vk.CommandPoolCreateFlags.PROTECTED

        return flags

    def cleanup(self):
        """Clean up all command pools."""
        for pools in self._pools.values():
            for pool in pools:
                vk.destroy_command_pool(self.device, pool, None)

        self._pools.clear()
        self._buffer_pools.clear()
        self._available_pools.clear()
