"""Command buffer management for compute operations."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import vulkan as vk

from .pipeline import PipelineType


@dataclass
class CommandConfig:
    """Configuration for command buffer recording."""
    pipeline_type: PipelineType
    descriptor_sets: List[vk.DescriptorSet]
    push_constants: bytes
    dispatch_x: int
    dispatch_y: int = 1
    dispatch_z: int = 1


class CommandBufferManager:
    """Manages command buffer allocation and recording with advanced batching."""
    
    def __init__(self, device: vk.Device, queue_family_index: int):
        self.device = device
        self.queue_family_index = queue_family_index
        self.command_pool = self._create_command_pool()
        self.command_buffers: Dict[int, vk.CommandBuffer] = {}
        self._batch_size = 64  # Optimal batch size for command submissions
        self._pending_submissions: List[Tuple[vk.CommandBuffer, Optional[vk.Fence]]] = []
        self._reusable_buffers: List[vk.CommandBuffer] = []
        
    def allocate_command_buffer(self, reuse: bool = True) -> vk.CommandBuffer:
        """Allocate a new command buffer with optional reuse."""
        if reuse and self._reusable_buffers:
            buffer = self._reusable_buffers.pop()
            vk.reset_command_buffer(buffer, vk.CommandBufferResetFlags(0))
            return buffer
            
        alloc_info = vk.CommandBufferAllocateInfo(
            command_pool=self.command_pool,
            level=vk.CommandBufferLevel.PRIMARY,
            command_buffer_count=1
        )
        
        command_buffer = vk.allocate_command_buffers(self.device, alloc_info)[0]
        self.command_buffers[id(command_buffer)] = command_buffer
        return command_buffer
        
    def record_compute_commands(self,
                              command_buffer: vk.CommandBuffer,
                              pipeline: vk.Pipeline,
                              pipeline_layout: vk.PipelineLayout,
                              config: CommandConfig,
                              memory_barriers: Optional[List[vk.MemoryBarrier]] = None) -> None:
        """Record compute commands with memory barriers."""
        begin_info = vk.CommandBufferBeginInfo(
            flags=vk.CommandBufferUsageFlags.ONE_TIME_SUBMIT
        )
        vk.begin_command_buffer(command_buffer, begin_info)
        
        # Insert pre-execution memory barriers if needed
        if memory_barriers:
            vk.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask=vk.PipelineStageFlags.COMPUTE_SHADER,
                dst_stage_mask=vk.PipelineStageFlags.COMPUTE_SHADER,
                memory_barriers=memory_barriers
            )
        
        # Bind pipeline and resources
        vk.cmd_bind_pipeline(
            command_buffer,
            vk.PipelineBindPoint.COMPUTE,
            pipeline
        )
        
        if config.descriptor_sets:
            vk.cmd_bind_descriptor_sets(
                command_buffer,
                vk.PipelineBindPoint.COMPUTE,
                pipeline_layout,
                0,
                config.descriptor_sets,
                []
            )
        
        if config.push_constants:
            vk.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk.ShaderStageFlags.COMPUTE,
                0,
                len(config.push_constants),
                config.push_constants
            )
        
        # Dispatch with optimal workgroup size
        local_size = 256  # Configurable based on device limits
        dispatch_x = (config.dispatch_x + local_size - 1) // local_size
        vk.cmd_dispatch(
            command_buffer,
            dispatch_x,
            config.dispatch_y,
            config.dispatch_z
        )
        
        # Insert post-execution memory barriers if needed
        if memory_barriers:
            vk.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask=vk.PipelineStageFlags.COMPUTE_SHADER,
                dst_stage_mask=vk.PipelineStageFlags.COMPUTE_SHADER,
                memory_barriers=memory_barriers
            )
        
        vk.end_command_buffer(command_buffer)
    
    def submit_commands(self,
                       command_buffer: vk.CommandBuffer,
                       queue: vk.Queue,
                       wait_semaphores: Optional[List[vk.Semaphore]] = None,
                       signal_semaphores: Optional[List[vk.Semaphore]] = None,
                       fence: Optional[vk.Fence] = None,
                       batch: bool = True) -> None:
        """Submit command buffer with optional batching."""
        if batch:
            self._pending_submissions.append((command_buffer, fence))
            if len(self._pending_submissions) >= self._batch_size:
                self._submit_batch(queue, wait_semaphores, signal_semaphores)
        else:
            submit_info = vk.SubmitInfo(
                wait_semaphores=wait_semaphores or [],
                wait_dst_stage_mask=[vk.PipelineStageFlags.COMPUTE_SHADER] * len(wait_semaphores or []),
                command_buffers=[command_buffer],
                signal_semaphores=signal_semaphores or []
            )
            vk.queue_submit(queue, [submit_info], fence)
    
    def _submit_batch(self,
                     queue: vk.Queue,
                     wait_semaphores: Optional[List[vk.Semaphore]] = None,
                     signal_semaphores: Optional[List[vk.Semaphore]] = None) -> None:
        """Submit a batch of command buffers."""
        if not self._pending_submissions:
            return
            
        # Create a fence for the entire batch
        batch_fence = vk.create_fence(self.device, vk.FenceCreateInfo(), None)
        
        # Prepare submission
        command_buffers = [cb for cb, _ in self._pending_submissions]
        individual_fences = [f for _, f in self._pending_submissions if f is not None]
        
        submit_info = vk.SubmitInfo(
            wait_semaphores=wait_semaphores or [],
            wait_dst_stage_mask=[vk.PipelineStageFlags.COMPUTE_SHADER] * len(wait_semaphores or []),
            command_buffers=command_buffers,
            signal_semaphores=signal_semaphores or []
        )
        
        # Submit batch
        vk.queue_submit(queue, [submit_info], batch_fence)
        
        # Wait for completion and signal individual fences
        vk.wait_for_fences(self.device, [batch_fence], True, uint64_max)
        for fence in individual_fences:
            vk.set_fence_status(self.device, fence)
        
        # Cleanup
        vk.destroy_fence(self.device, batch_fence, None)
        self._pending_submissions.clear()
        
        # Return buffers to reusable pool
        self._reusable_buffers.extend(command_buffers)
    
    def flush_pending_submissions(self,
                                queue: vk.Queue,
                                wait_semaphores: Optional[List[vk.Semaphore]] = None,
                                signal_semaphores: Optional[List[vk.Semaphore]] = None) -> None:
        """Flush any pending command buffer submissions."""
        self._submit_batch(queue, wait_semaphores, signal_semaphores)
    
    def _create_command_pool(self) -> vk.CommandPool:
        """Create command pool for compute queue family."""
        pool_info = vk.CommandPoolCreateInfo(
            flags=vk.CommandPoolCreateFlags.RESET_COMMAND_BUFFER,
            queue_family_index=self.queue_family_index
        )
        return vk.create_command_pool(self.device, pool_info, None)
    
    def cleanup(self):
        """Clean up command pool and buffers."""
        if self.command_pool:
            vk.destroy_command_pool(self.device, self.command_pool, None)
            self.command_buffers.clear()
