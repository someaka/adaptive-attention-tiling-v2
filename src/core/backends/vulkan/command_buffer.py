"""Command buffer management for compute operations."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from ctypes import c_void_p, c_uint32, c_uint64, byref, POINTER, cast, Array, Structure

import vulkan as vk

from .pipeline import PipelineType


@dataclass
class CommandConfig:
    """Configuration for command buffer recording."""

    pipeline_type: PipelineType
    descriptor_sets: List[c_void_p]  # List of VkDescriptorSet handles
    push_constants: bytes
    dispatch_x: int
    dispatch_y: int = 1
    dispatch_z: int = 1


def create_memory_barrier(
    src_access_mask: int,
    dst_access_mask: int
) -> dict:
    """Create a memory barrier for synchronization.
    
    Returns a dictionary that can be used to create a VkMemoryBarrier.
    The actual VkMemoryBarrier will be created when needed.
    """
    return {
        "sType": vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        "pNext": None,
        "srcAccessMask": src_access_mask,
        "dstAccessMask": dst_access_mask
    }


def create_vulkan_memory_barrier(barrier_dict: dict) -> Any:
    """Create a VkMemoryBarrier from a dictionary."""
    return vk.VkMemoryBarrier(**barrier_dict)


class VulkanSync:
    """Synchronization primitives for Vulkan operations."""
    
    def __init__(self, device: c_void_p):  # device: VkDevice
        """Initialize sync primitives."""
        self.device = device
        self.fences: Dict[int, c_void_p] = {}  # Dict[id, VkFence]
        self.semaphores: Dict[int, c_void_p] = {}  # Dict[id, VkSemaphore]
        
    def create_fence(self, signaled: bool = False) -> c_void_p:
        """Create a fence."""
        create_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT if signaled else 0
        )
        fence = c_void_p()
        result = vk.vkCreateFence(self.device, create_info, None, byref(fence))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create fence: {result}")
        self.fences[id(fence)] = fence
        return fence
        
    def create_semaphore(self) -> c_void_p:
        """Create a semaphore."""
        create_info = vk.VkSemaphoreCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        )
        semaphore = c_void_p()
        result = vk.vkCreateSemaphore(self.device, create_info, None, byref(semaphore))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create semaphore: {result}")
        self.semaphores[id(semaphore)] = semaphore
        return semaphore
        
    def wait_for_fence(self, fence: c_void_p, timeout: Optional[int] = None) -> bool:
        """Wait for a fence to be signaled."""
        result = vk.vkWaitForFences(
            self.device,
            1,
            [fence],
            vk.VK_TRUE,
            timeout if timeout is not None else c_uint64(-1).value
        )
        return result == vk.VK_SUCCESS
        
    def reset_fence(self, fence: c_void_p) -> None:
        """Reset a fence to unsignaled state."""
        vk.vkResetFences(self.device, 1, [fence])
        
    def destroy_fence(self, fence: c_void_p) -> None:
        """Destroy a fence."""
        vk.vkDestroyFence(self.device, fence, None)
        del self.fences[id(fence)]
        
    def destroy_semaphore(self, semaphore: c_void_p) -> None:
        """Destroy a semaphore."""
        vk.vkDestroySemaphore(self.device, semaphore, None)
        del self.semaphores[id(semaphore)]
        
    def cleanup(self) -> None:
        """Cleanup all sync primitives."""
        for fence in list(self.fences.values()):
            self.destroy_fence(fence)
        for semaphore in list(self.semaphores.values()):
            self.destroy_semaphore(semaphore)


class CommandBufferManager:
    """Manages command buffer allocation and recording with advanced batching."""

    def __init__(self, device: c_void_p, queue_family_index: int):
        self.device = device
        self.queue_family_index = queue_family_index
        self.command_pool = self._create_command_pool()
        self.command_buffers: Dict[int, c_void_p] = {}  # Dict[id, VkCommandBuffer]
        self._batch_size = 64  # Optimal batch size for command submissions
        self._pending_submissions: List[Tuple[c_void_p, Optional[c_void_p]]] = []  # List[Tuple[VkCommandBuffer, Optional[VkFence]]]
        self._reusable_buffers: List[c_void_p] = []  # List[VkCommandBuffer]

    def _create_command_pool(self) -> c_void_p:
        """Create command pool."""
        create_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        pool = c_void_p()
        result = vk.vkCreateCommandPool(self.device, create_info, None, byref(pool))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create command pool: {result}")
        return pool

    def allocate_command_buffer(self, reuse: bool = True) -> c_void_p:
        """Allocate a new command buffer with optional reuse."""
        if reuse and self._reusable_buffers:
            buffer = self._reusable_buffers.pop()
            vk.vkResetCommandBuffer(buffer, 0)
            return buffer

        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        command_buffers = (c_void_p * 1)()
        result = vk.vkAllocateCommandBuffers(self.device, alloc_info, command_buffers)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to allocate command buffer: {result}")
            
        command_buffer = command_buffers[0]
        self.command_buffers[id(command_buffer)] = command_buffer
        return command_buffer

    def record_compute_commands(
        self,
        command_buffer: c_void_p,
        pipeline: c_void_p,
        pipeline_layout: c_void_p,
        config: CommandConfig,
        memory_barriers: Optional[List[dict]] = None,
    ) -> None:
        """Record compute commands with memory barriers."""
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        result = vk.vkBeginCommandBuffer(command_buffer, begin_info)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to begin command buffer: {result}")

        # Insert pre-execution memory barriers if needed
        if memory_barriers:
            # Convert dictionary barriers to VkMemoryBarrier objects
            vk_barriers = [create_vulkan_memory_barrier(b) for b in memory_barriers]
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                len(vk_barriers),
                vk_barriers,
                0,
                None,
                0,
                None
            )

        # Bind pipeline and resources
        vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)

        if config.descriptor_sets:
            vk.vkCmdBindDescriptorSets(
                command_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout,
                0,
                len(config.descriptor_sets),
                config.descriptor_sets,
                0,
                None
            )

        if config.push_constants:
            vk.vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(config.push_constants),
                config.push_constants
            )

        # Dispatch with optimal workgroup size
        local_size = 256  # Configurable based on device limits
        dispatch_x = (config.dispatch_x + local_size - 1) // local_size
        vk.vkCmdDispatch(command_buffer, dispatch_x, config.dispatch_y, config.dispatch_z)

        # Insert post-execution memory barriers if needed
        if memory_barriers:
            # Convert dictionary barriers to VkMemoryBarrier objects
            vk_barriers = [create_vulkan_memory_barrier(b) for b in memory_barriers]
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                len(vk_barriers),
                vk_barriers,
                0,
                None,
                0,
                None
            )

        result = vk.vkEndCommandBuffer(command_buffer)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to end command buffer: {result}")

    def submit_commands(
        self,
        command_buffer: c_void_p,
        queue: c_void_p,
        wait_semaphores: Optional[List[c_void_p]] = None,
        signal_semaphores: Optional[List[c_void_p]] = None,
        fence: Optional[c_void_p] = None,
        batch: bool = True,
    ) -> None:
        """Submit command buffer with optional batching."""
        if batch:
            self._pending_submissions.append((command_buffer, fence))
            if len(self._pending_submissions) >= self._batch_size:
                self._submit_batch(queue, wait_semaphores, signal_semaphores)
        else:
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                waitSemaphoreCount=len(wait_semaphores) if wait_semaphores else 0,
                pWaitSemaphores=wait_semaphores if wait_semaphores else None,
                pWaitDstStageMask=(c_uint32 * len(wait_semaphores))(*([vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT] * len(wait_semaphores))) if wait_semaphores else None,
                commandBufferCount=1,
                pCommandBuffers=[command_buffer],
                signalSemaphoreCount=len(signal_semaphores) if signal_semaphores else 0,
                pSignalSemaphores=signal_semaphores if signal_semaphores else None
            )
            result = vk.vkQueueSubmit(queue, 1, [submit_info], fence)
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Failed to submit command buffer: {result}")

    def _submit_batch(
        self,
        queue: c_void_p,
        wait_semaphores: Optional[List[c_void_p]] = None,
        signal_semaphores: Optional[List[c_void_p]] = None,
    ) -> None:
        """Submit a batch of command buffers."""
        if not self._pending_submissions:
            return

        # Create submit info for each command buffer
        submit_infos = []
        for i, (cmd_buffer, fence) in enumerate(self._pending_submissions):
            # Only use semaphores for first and last submissions
            wait_count = len(wait_semaphores) if wait_semaphores is not None else 0
            signal_count = len(signal_semaphores) if signal_semaphores is not None else 0
            
            use_wait = i == 0 and wait_count > 0
            use_signal = i == len(self._pending_submissions) - 1 and signal_count > 0
            
            # Create stage mask array if needed
            stage_mask_arr = None
            if use_wait and wait_semaphores is not None:
                stage_mask_arr = (c_uint32 * wait_count)(
                    *([vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT] * wait_count)
                )
            
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                waitSemaphoreCount=wait_count if use_wait else 0,
                pWaitSemaphores=wait_semaphores if use_wait else None,
                pWaitDstStageMask=stage_mask_arr,
                commandBufferCount=1,
                pCommandBuffers=[cmd_buffer],
                signalSemaphoreCount=signal_count if use_signal else 0,
                pSignalSemaphores=signal_semaphores if use_signal else None
            )
            submit_infos.append(submit_info)

        # Submit batch
        result = vk.vkQueueSubmit(queue, len(submit_infos), submit_infos, self._pending_submissions[-1][1])
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to submit command buffer batch: {result}")

        # Clear pending submissions
        self._pending_submissions.clear()

    def cleanup(self) -> None:
        """Cleanup command pool and buffers."""
        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            self.command_pool = None
        self.command_buffers.clear()
        self._reusable_buffers.clear()
        self._pending_submissions.clear()
