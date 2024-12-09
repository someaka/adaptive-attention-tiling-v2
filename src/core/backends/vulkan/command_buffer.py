"""Command buffer management for compute operations."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import vulkan as vk

from .pipeline import PipelineType


@dataclass
class CommandConfig:
    """Configuration for command buffer recording."""

    pipeline_type: PipelineType
    descriptor_sets: List[int]  # List of VkDescriptorSet handles
    push_constants: bytes
    dispatch_x: int
    dispatch_y: int = 1
    dispatch_z: int = 1


class CommandBufferManager:
    """Manages command buffer allocation and recording with advanced batching."""

    def __init__(self, device: int, queue_family_index: int):  # device: VkDevice
        self.device = device
        self.queue_family_index = queue_family_index
        self.command_pool = self._create_command_pool()
        self.command_buffers: Dict[int, int] = {}  # Dict[id, VkCommandBuffer]
        self._batch_size = 64  # Optimal batch size for command submissions
        self._pending_submissions: List[Tuple[int, Optional[int]]] = []  # List[Tuple[VkCommandBuffer, Optional[VkFence]]]
        self._reusable_buffers: List[int] = []  # List[VkCommandBuffer]

    def allocate_command_buffer(self, reuse: bool = True) -> int:  # returns VkCommandBuffer
        """Allocate a new command buffer with optional reuse."""
        if reuse and self._reusable_buffers:
            buffer = self._reusable_buffers.pop()
            vk.vkResetCommandBuffer(buffer, 0)  # VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT = 0
            return buffer

        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )

        command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]
        self.command_buffers[id(command_buffer)] = command_buffer
        return command_buffer

    def record_compute_commands(
        self,
        command_buffer: int,
        pipeline: int,
        pipeline_layout: int,
        config: CommandConfig,
        memory_barriers: Optional[List[vk.VkMemoryBarrier]] = None,
    ) -> None:
        """Record compute commands with memory barriers."""
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.vkBeginCommandBuffer(command_buffer, begin_info)

        # Insert pre-execution memory barriers if needed
        if memory_barriers:
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                memory_barriers,
            )

        # Bind pipeline and resources
        vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)

        if config.descriptor_sets:
            vk.vkCmdBindDescriptorSets(
                command_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout,
                0,
                config.descriptor_sets,
                [],
            )

        if config.push_constants:
            vk.vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(config.push_constants),
                config.push_constants,
            )

        # Dispatch with optimal workgroup size
        local_size = 256  # Configurable based on device limits
        dispatch_x = (config.dispatch_x + local_size - 1) // local_size
        vk.vkCmdDispatch(command_buffer, dispatch_x, config.dispatch_y, config.dispatch_z)

        # Insert post-execution memory barriers if needed
        if memory_barriers:
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                memory_barriers,
            )

        vk.vkEndCommandBuffer(command_buffer)

    def submit_commands(
        self,
        command_buffer: int,
        queue: int,
        wait_semaphores: Optional[List[int]] = None,
        signal_semaphores: Optional[List[int]] = None,
        fence: Optional[int] = None,
        batch: bool = True,
    ) -> None:
        """Submit command buffer with optional batching."""
        if batch:
            self._pending_submissions.append((command_buffer, fence))
            if len(self._pending_submissions) >= self._batch_size:
                self._submit_batch(queue, wait_semaphores, signal_semaphores)
        else:
            submit_info = vk.VkSubmitInfo(
                waitSemaphores=wait_semaphores or [],
                waitDstStageMask=[vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT]
                * len(wait_semaphores or []),
                commandBuffers=[command_buffer],
                signalSemaphores=signal_semaphores or [],
            )
            vk.vkQueueSubmit(queue, [submit_info], fence)

    def _submit_batch(
        self,
        queue: int,
        wait_semaphores: Optional[List[int]] = None,
        signal_semaphores: Optional[List[int]] = None,
    ) -> None:
        """Submit a batch of command buffers."""
        if not self._pending_submissions:
            return

        # Create a fence for the entire batch
        batch_fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(), None)

        # Prepare submission
        command_buffers = [cb for cb, _ in self._pending_submissions]
        individual_fences = [f for _, f in self._pending_submissions if f is not None]

        submit_info = vk.VkSubmitInfo(
            waitSemaphores=wait_semaphores or [],
            waitDstStageMask=[vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT]
            * len(wait_semaphores or []),
            commandBuffers=command_buffers,
            signalSemaphores=signal_semaphores or [],
        )

        # Submit batch
        vk.vkQueueSubmit(queue, [submit_info], batch_fence)

        # Wait for completion and signal individual fences
        vk.vkWaitForFences(self.device, [batch_fence], True, vk.VK_MAX_UINT64)
        for fence in individual_fences:
            vk.vkSetFenceStatus(self.device, fence)

        # Cleanup
        vk.vkDestroyFence(self.device, batch_fence, None)
        self._pending_submissions.clear()

        # Return buffers to reusable pool
        self._reusable_buffers.extend(command_buffers)

    def flush_pending_submissions(
        self,
        queue: int,
        wait_semaphores: Optional[List[int]] = None,
        signal_semaphores: Optional[List[int]] = None,
    ) -> None:
        """Flush any pending command buffer submissions."""
        self._submit_batch(queue, wait_semaphores, signal_semaphores)

    def _create_command_pool(self) -> int:
        """Create command pool for compute queue family."""
        pool_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.queue_family_index,
        )
        return vk.vkCreateCommandPool(self.device, pool_info, None)

    def cleanup(self):
        """Clean up command pool and buffers."""
        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            self.command_buffers.clear()
