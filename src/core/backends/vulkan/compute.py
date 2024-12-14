"""Vulkan compute interface for tensor operations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, cast
from ctypes import c_void_p, c_uint32, POINTER, cast, byref, Structure
import os

import torch
import numpy as np
import vulkan as vk

from .tensor_ops import VulkanTensorOps, TensorDescriptor as BaseTensorDescriptor
from .device import VulkanDevice
from .command_pool import CommandPoolType
from .command_buffer import CommandConfig
from .shader_manager import ShaderManager, ShaderType, ShaderConfig
from .pipeline import VulkanPipeline, PipelineType
from .memory_manager import BufferUsage
from .barrier_manager import BarrierManager, AccessPattern
from src.core.common.constants import SHADER_DIR


@dataclass
class TensorDescriptor(BaseTensorDescriptor):
    """Extended descriptor for tensor layout and memory."""
    descriptor_set: Optional[c_void_p] = None  # VkDescriptorSet handle
    tensor: Optional[torch.Tensor] = None


@dataclass
class ComputeMetrics:
    """Metrics for compute operations."""
    shader_size: int = 0
    compilation_success: bool = False
    active_warps: int = 0
    max_warps: int = 32
    memory_transfers: int = 0
    active_buffers: int = 0
    memory_leaks: int = 0
    descriptor_pool_fragmentation: float = 0.0
    active_descriptor_sets: int = 0


@dataclass
class ShaderInfo:
    """Information about a compiled shader."""
    specialization_constants: Dict[str, int]
    module: int  # VkShaderModule handle
    pipeline: int  # VkPipeline handle
    pipeline_layout: int  # VkPipelineLayout handle


def handle_to_int(handle: Optional[c_void_p]) -> int:
    """Convert a Vulkan handle to integer."""
    if handle is None:
        return 0
    # Use ctypes cast to safely convert to integer
    ptr = cast(handle, POINTER(c_uint32))
    if ptr is None or not ptr:  # Check for null pointer
        return 0
    return ptr.contents.value


class VulkanCompute:
    """High-level interface for Vulkan compute operations."""

    def __init__(self, enable_profiling: bool = False):
        # Initialize device
        self.device = VulkanDevice()
        if not self.device.initialize():
            raise RuntimeError("Failed to initialize Vulkan device")

        # Convert device handle to int first
        raw_device = self.device.device
        raw_queue = self.device.compute_queue
        
        device_int = handle_to_int(raw_device)
        queue_int = handle_to_int(raw_queue)
        
        # Ensure queue family index is not None
        queue_family_index = self.device.queue_family_indices.compute
        if queue_family_index is None:
            raise RuntimeError("Compute queue family index is None")

        # Initialize managers
        self.tensor_ops = VulkanTensorOps(
            device=device_int,
            queue=queue_int,
            queue_family_index=queue_family_index
        )
        self.shader_manager = ShaderManager(device_int)
        self.pipeline_manager = VulkanPipeline(device_int)
        self.barrier_manager = BarrierManager()
        
        # State
        self.enable_profiling = enable_profiling
        self.metrics = ComputeMetrics()
        self.workgroup_size = (16, 16)  # Default size
        self.active_buffers: Dict[int, Any] = {}
        self.active_descriptor_sets: Dict[int, Any] = {}

    def create_buffer(self, size: int) -> Dict[str, Any]:
        """Create a Vulkan buffer."""
        buffer, memory, offset = self.tensor_ops.memory_manager.allocate_tile_state(size)
        buffer_info = {
            "buffer": handle_to_int(buffer),  # Convert to int
            "memory": handle_to_int(memory),  # Convert to int
            "offset": offset
        }
        self.active_buffers[buffer_info["buffer"]] = buffer_info
        
        if self.enable_profiling:
            self.metrics.active_buffers += 1
            
        return buffer_info

    def delete_buffer(self, buffer_info: Dict[str, Any]) -> None:
        """Delete a Vulkan buffer."""
        buffer = c_void_p(buffer_info["buffer"])  # Convert back to c_void_p
        self.tensor_ops.memory_manager.free_buffer(buffer, BufferUsage.TILE_STATE)
        del self.active_buffers[buffer_info["buffer"]]
        
        if self.enable_profiling:
            self.metrics.active_buffers -= 1

    def create_descriptor_set(self, data: torch.Tensor) -> TensorDescriptor:
        """Create a descriptor set for the tensor."""
        desc = self.register_tensor(data)
        
        # Convert device handle to c_void_p for Vulkan API
        device_handle = c_void_p(self.device.device)
        
        # Allocate descriptor set
        desc.descriptor_set = self.pipeline_manager.allocate_descriptor_set(
            PipelineType.TILE_PROCESSOR
        )
        
        # Create buffer info for binding
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=c_void_p(desc.buffer),  # Convert int to c_void_p
            offset=desc.offset,
            range=desc.size
        )
        
        # Create write descriptor set
        write_set = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=desc.descriptor_set,
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=buffer_info
        )
        
        # Update descriptor set
        vk.vkUpdateDescriptorSets(device_handle, 1, [write_set], 0, None)
        
        # Store in active sets
        self.active_descriptor_sets[handle_to_int(desc.descriptor_set)] = desc
        
        if self.enable_profiling:
            self.metrics.active_descriptor_sets += 1
            
        return desc

    def delete_descriptor_set(self, desc: TensorDescriptor) -> None:
        """Delete a descriptor set."""
        if desc.descriptor_set is not None:
            # Free the descriptor set
            self.pipeline_manager.free_descriptor_set(desc.descriptor_set)
            
            # Remove from active sets
            set_id = handle_to_int(desc.descriptor_set)
            if set_id in self.active_descriptor_sets:
                del self.active_descriptor_sets[set_id]
            
            # Clear the descriptor set handle
            desc.descriptor_set = None
            
            if self.enable_profiling:
                self.metrics.active_descriptor_sets -= 1

    def get_metrics(self) -> ComputeMetrics:
        """Get performance metrics."""
        if not self.enable_profiling:
            return ComputeMetrics()
            
        # Update fragmentation metrics
        total_sets = len(self.active_descriptor_sets)
        if total_sets > 0:
            # Calculate fragmentation based on descriptor set allocation patterns
            max_sets = self.pipeline_manager.descriptor_pool_size
            used_sets = len(self.active_descriptor_sets)
            
            # Track gaps between allocated sets
            set_ids = sorted(self.active_descriptor_sets.keys())
            gaps = 0
            for i in range(len(set_ids) - 1):
                if set_ids[i+1] - set_ids[i] > 1:
                    gaps += 1
                    
            # Fragmentation is a combination of:
            # - Ratio of used vs allocated sets
            # - Number of gaps between allocations
            usage_ratio = used_sets / max_sets
            gap_ratio = gaps / max(1, used_sets - 1)
            
            self.metrics.descriptor_pool_fragmentation = (
                0.4 * (1 - usage_ratio) +  # Lower usage = higher fragmentation
                0.6 * gap_ratio           # More gaps = higher fragmentation
            )
            
        return self.metrics

    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        # Clean up active resources
        for buffer_info in list(self.active_buffers.values()):
            self.delete_buffer(buffer_info)
        for desc in list(self.active_descriptor_sets.values()):
            self.delete_descriptor_set(desc)
            
        # Clean up managers
        self.shader_manager.cleanup()
        self.pipeline_manager.cleanup()
        self.tensor_ops.cleanup()
        self.device.cleanup()

    def register_tensor(self, tensor: torch.Tensor) -> TensorDescriptor:
        """Register a tensor for Vulkan operations."""
        base_desc = self.tensor_ops.register_tensor(tensor)
        # Create extended descriptor with additional fields
        desc = TensorDescriptor(
            shape=base_desc.shape,
            dtype=base_desc.dtype,
            strides=base_desc.strides,
            buffer=base_desc.buffer,
            memory=base_desc.memory,
            offset=base_desc.offset,
            size=base_desc.size,
            is_contiguous=base_desc.is_contiguous,
            descriptor_set=None,  # Will be set when needed
            tensor=tensor  # Store reference to original tensor
        )
        return desc

    def compile_shader(self, shader_type: str, matrix_size: Tuple[int, int]) -> ShaderInfo:
        """Compile a shader for the given type and size."""
        shader_type_map = {
            "pattern": ShaderType.TILE_PROCESSOR,
            "flow": ShaderType.CROSS_TILE_ROUTER,
            "attention": ShaderType.TILE_PROCESSOR,  # Using tile processor for attention
        }
        
        pipeline_type_map = {
            "pattern": PipelineType.TILE_PROCESSOR,
            "flow": PipelineType.CROSS_TILE_ROUTER,
            "attention": PipelineType.ADAPTIVE_ATTENTION,
        }
        
        shader_enum = shader_type_map.get(shader_type)
        pipeline_enum = pipeline_type_map.get(shader_type)
        if shader_enum is None or pipeline_enum is None:
            raise ValueError(f"Unknown shader type: {shader_type}")
            
        try:
            # Create shader module with configuration
            config = ShaderConfig(
                local_size_x=self.workgroup_size[0],
                use_fp16=True,
                enable_validation=self.enable_profiling
            )
            shader_module = self.shader_manager.load_shader(shader_enum, config)
            
            # Read shader code
            shader_path = os.path.join(SHADER_DIR, f"{shader_type}.spv")
            with open(shader_path, "rb") as f:
                shader_code = f.read()
            
            # Create pipeline with push constant size
            push_constant_size = 16  # Size of PushConstants struct in shader
            pipeline = self.pipeline_manager.create_pipeline(
                pipeline_enum,
                shader_code,
                push_constant_size
            )
            
            # Get pipeline layout
            pipeline_layout = self.pipeline_manager.pipeline_layouts[pipeline_enum]
            
            constants = {
                "LOCAL_SIZE_X": self.workgroup_size[0],
                "LOCAL_SIZE_Y": self.workgroup_size[1]
            }
            
            if self.enable_profiling:
                shader_path = os.path.join(SHADER_DIR, f"{shader_type}.comp")
                if os.path.exists(shader_path):
                    self.metrics.shader_size = os.path.getsize(shader_path)
                self.metrics.compilation_success = True
                
            return ShaderInfo(
                specialization_constants=constants,
                module=shader_module,
                pipeline=pipeline,
                pipeline_layout=pipeline_layout
            )
            
        except Exception as e:
            if self.enable_profiling:
                self.metrics.compilation_success = False
            raise RuntimeError(f"Shader compilation failed: {e}")

    def set_workgroup_size(self, x: int, y: int) -> None:
        """Set the workgroup size for compute operations."""
        self.workgroup_size = (x, y)

    def execute_compute(self, data: torch.Tensor, shader_type: str = "compute") -> torch.Tensor:
        """Execute a compute operation."""
        # Get last two dimensions as tuple for shader compilation
        width, height = int(data.shape[-2]), int(data.shape[-1])
        matrix_size = (width, height)
        
        # Register input tensor
        input_desc = self.register_tensor(data)
        
        # Create output tensor
        output = torch.empty_like(data)
        output_desc = self.register_tensor(output)
        
        # Compile shader with proper matrix size tuple
        shader_info = self.compile_shader(shader_type, matrix_size)
        
        # Record compute commands
        command_buffer = self.tensor_ops.command_manager.allocate_command_buffer()
        
        # Memory barriers for input/output
        self.barrier_manager.add_compute_read_barrier(input_desc.buffer)
        self.barrier_manager.record_barriers(command_buffer)
        
        # Calculate dispatch dimensions
        dispatch_x = (width + self.workgroup_size[0] - 1) // self.workgroup_size[0]
        dispatch_y = (height + self.workgroup_size[1] - 1) // self.workgroup_size[1]
        
        # Create command config
        config = CommandConfig(
            pipeline_type=PipelineType.TILE_PROCESSOR,  # Convert string to enum
            descriptor_sets=[input_desc.descriptor_set, output_desc.descriptor_set],
            push_constants=b'',  # Empty for now
            dispatch_x=dispatch_x,
            dispatch_y=dispatch_y,
            dispatch_z=1
        )
        
        # Record compute commands
        self.tensor_ops.command_manager.record_compute_commands(
            command_buffer,
            c_void_p(shader_info.pipeline),  # Convert int to c_void_p
            c_void_p(shader_info.pipeline_layout),  # Convert int to c_void_p
            config
        )
        
        # Memory barrier for output
        self.barrier_manager.add_compute_write_barrier(output_desc.buffer)
        self.barrier_manager.record_barriers(command_buffer)
        
        # Submit and wait
        self.tensor_ops.command_manager.submit_commands(
            command_buffer,
            self.device.compute_queue,
            batch=False
        )
        
        if self.enable_profiling:
            self.metrics.active_warps = dispatch_x * dispatch_y * self.workgroup_size[0] * self.workgroup_size[1]
            
        # Copy result back to output tensor
        self.tensor_ops._copy_from_device(output_desc, output)
        return output

    def transfer_to_device(self, data: torch.Tensor) -> TensorDescriptor:
        """Transfer data to device memory."""
        desc = self.register_tensor(data)
        self.tensor_ops._copy_to_device(data, desc)
        
        if self.enable_profiling:
            self.metrics.memory_transfers += 1
            
        return desc

    def transfer_to_host(self, device_data: TensorDescriptor) -> torch.Tensor:
        """Transfer data from device to host memory."""
        output = torch.empty_like(device_data.tensor)
        self.tensor_ops._copy_from_device(device_data, output)
        
        if self.enable_profiling:
            self.metrics.memory_transfers += 1
            
        return output

    def compute_tile(
        self, input_tensor: torch.Tensor, resolution: float, d_state: int
    ) -> torch.Tensor:
        """Process a tile using Vulkan compute shader."""
        # Register input tensor
        input_descriptor = self.register_tensor(input_tensor)
            
        # Create output tensor
        output_tensor = torch.empty_like(input_tensor)
        output_descriptor = self.register_tensor(output_tensor)
        
        # Copy input data to device
        self.tensor_ops._copy_to_device(input_tensor, input_descriptor)
        
        # Compile tile processor shader
        shader_info = self.compile_shader("pattern", input_tensor.shape[-2:])  # Use last two dimensions
        
        # Record compute commands
        command_buffer = self.tensor_ops.command_manager.allocate_command_buffer(
            reuse=True
        )
        
        # Memory barrier for input
        self.barrier_manager.add_compute_read_barrier(input_descriptor.buffer)
        self.barrier_manager.record_barriers(command_buffer)
        
        # Calculate dispatch dimensions
        width, height = input_tensor.shape[-2:]
        dispatch_x = (width + self.workgroup_size[0] - 1) // self.workgroup_size[0]
        dispatch_y = (height + self.workgroup_size[1] - 1) // self.workgroup_size[1]
        
        # Create push constants struct
        push_constants = {
            "sequence_length": width,
            "d_model": height,
            "d_state": d_state,
            "min_resolution": resolution,
            "density_threshold": 0.1  # Configurable threshold
        }
        
        # Create command config
        config = CommandConfig(
            pipeline_type=PipelineType.TILE_PROCESSOR,  # Use proper enum
            descriptor_sets=[input_descriptor.descriptor_set, output_descriptor.descriptor_set],
            push_constants=bytes(str(push_constants), 'utf-8'),
            dispatch_x=dispatch_x,
            dispatch_y=dispatch_y,
            dispatch_z=1
        )
        
        # Record compute commands
        self.tensor_ops.command_manager.record_compute_commands(
            command_buffer,
            c_void_p(shader_info.pipeline),  # Convert int to c_void_p
            c_void_p(shader_info.pipeline_layout),  # Convert int to c_void_p
            config
        )
        
        # Memory barrier for output
        self.barrier_manager.add_compute_write_barrier(output_descriptor.buffer)
        self.barrier_manager.record_barriers(command_buffer)
        
        # Submit and wait
        self.tensor_ops.command_manager.submit_commands(
            command_buffer,
            self.device.compute_queue,
            batch=False
        )
        
        # Copy result back to output tensor
        self.tensor_ops._copy_from_device(output_descriptor, output_tensor)
        
        # Free command buffer for reuse
        self.tensor_ops.command_manager._reusable_buffers.append(command_buffer)
        
        return output_tensor
  