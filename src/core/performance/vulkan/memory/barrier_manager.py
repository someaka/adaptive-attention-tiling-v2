"""Vulkan Memory Barrier Management.

This module provides tools for managing memory barriers and synchronization
in Vulkan operations, ensuring proper memory access patterns and dependencies.
"""

from dataclasses import dataclass
from enum import Enum, auto

import vulkan as vk


class AccessPattern(Enum):
    """Common access patterns for memory operations."""

    READ_ONLY = auto()
    WRITE_ONLY = auto()
    READ_WRITE = auto()
    TRANSFER_SRC = auto()
    TRANSFER_DST = auto()
    COMPUTE_SHADER = auto()


@dataclass
class BarrierInfo:
    """Information about a memory barrier."""

    src_stage: vk.PipelineStageFlags
    dst_stage: vk.PipelineStageFlags
    src_access: vk.AccessFlags
    dst_access: vk.AccessFlags
    queue_family_src: int = vk.QUEUE_FAMILY_IGNORED
    queue_family_dst: int = vk.QUEUE_FAMILY_IGNORED


class BarrierManager:
    """Manages memory barriers and synchronization."""

    def __init__(self):
        self._access_patterns = {
            AccessPattern.READ_ONLY: BarrierInfo(
                src_stage=vk.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                dst_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                src_access=0,
                dst_access=vk.ACCESS_SHADER_READ_BIT,
            ),
            AccessPattern.WRITE_ONLY: BarrierInfo(
                src_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dst_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                src_access=vk.ACCESS_SHADER_WRITE_BIT,
                dst_access=vk.ACCESS_SHADER_WRITE_BIT,
            ),
            AccessPattern.READ_WRITE: BarrierInfo(
                src_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dst_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                src_access=vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT,
                dst_access=vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT,
            ),
            AccessPattern.TRANSFER_SRC: BarrierInfo(
                src_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dst_stage=vk.PIPELINE_STAGE_TRANSFER_BIT,
                src_access=vk.ACCESS_SHADER_WRITE_BIT,
                dst_access=vk.ACCESS_TRANSFER_READ_BIT,
            ),
            AccessPattern.TRANSFER_DST: BarrierInfo(
                src_stage=vk.PIPELINE_STAGE_TRANSFER_BIT,
                dst_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                src_access=vk.ACCESS_TRANSFER_WRITE_BIT,
                dst_access=vk.ACCESS_SHADER_READ_BIT,
            ),
            AccessPattern.COMPUTE_SHADER: BarrierInfo(
                src_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dst_stage=vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                src_access=vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT,
                dst_access=vk.ACCESS_SHADER_READ_BIT | vk.ACCESS_SHADER_WRITE_BIT,
            ),
        }

    def get_barrier_info(
        self,
        pattern: AccessPattern,
        queue_family_src: int = vk.QUEUE_FAMILY_IGNORED,
        queue_family_dst: int = vk.QUEUE_FAMILY_IGNORED,
    ) -> BarrierInfo:
        """Get barrier info for a specific access pattern."""
        info = self._access_patterns[pattern]
        return BarrierInfo(
            src_stage=info.src_stage,
            dst_stage=info.dst_stage,
            src_access=info.src_access,
            dst_access=info.dst_access,
            queue_family_src=queue_family_src,
            queue_family_dst=queue_family_dst,
        )

    def buffer_barrier(
        self,
        command_buffer: vk.CommandBuffer,
        buffer: vk.Buffer,
        pattern: AccessPattern,
        offset: int = 0,
        size: int = vk.WHOLE_SIZE,
        queue_family_src: int = vk.QUEUE_FAMILY_IGNORED,
        queue_family_dst: int = vk.QUEUE_FAMILY_IGNORED,
    ) -> None:
        """Insert a buffer memory barrier."""
        info = self.get_barrier_info(pattern, queue_family_src, queue_family_dst)

        barrier = vk.BufferMemoryBarrier(
            srcAccessMask=info.src_access,
            dstAccessMask=info.dst_access,
            srcQueueFamilyIndex=info.queue_family_src,
            dstQueueFamilyIndex=info.queue_family_dst,
            buffer=buffer,
            offset=offset,
            size=size,
        )

        vk.CmdPipelineBarrier(
            command_buffer,
            info.src_stage,
            info.dst_stage,
            0,
            0,
            None,
            1,
            [barrier],
            0,
            None,
        )

    def global_barrier(
        self, command_buffer: vk.CommandBuffer, pattern: AccessPattern
    ) -> None:
        """Insert a global memory barrier."""
        info = self.get_barrier_info(pattern)

        barrier = vk.MemoryBarrier(
            srcAccessMask=info.src_access, dstAccessMask=info.dst_access
        )

        vk.CmdPipelineBarrier(
            command_buffer,
            info.src_stage,
            info.dst_stage,
            0,
            1,
            [barrier],
            0,
            None,
            0,
            None,
        )

    def execution_barrier(
        self,
        command_buffer: vk.CommandBuffer,
        src_stage: vk.PipelineStageFlags,
        dst_stage: vk.PipelineStageFlags,
    ) -> None:
        """Insert an execution barrier between pipeline stages."""
        vk.CmdPipelineBarrier(
            command_buffer, src_stage, dst_stage, 0, 0, None, 0, None, 0, None
        )
