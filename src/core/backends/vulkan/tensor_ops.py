"""PyTorch tensor operations for Vulkan backend."""

from dataclasses import dataclass
import ctypes
import struct
from typing import Dict, List, Tuple

import torch
import vulkan as vk

from .command_buffer import CommandBufferManager, CommandConfig
from .memory_manager import BufferUsage, VulkanMemoryManager
from .pipeline import PipelineType, VulkanPipeline
from .shader_manager import ShaderManager


@dataclass
class TensorDescriptor:
    """Descriptor for tensor layout and memory."""

    shape: torch.Size
    dtype: torch.dtype
    strides: Tuple[int, ...]
    buffer: int  # VkBuffer handle
    offset: int
    size: int
    is_contiguous: bool


class VulkanCompute:
    """High-performance tensor operations using Vulkan compute."""

    def __init__(self, device: int, queue: int, queue_family_index: int):  # device: VkDevice, queue: VkQueue
        self.device = device
        self.queue = queue
        self.memory_manager = VulkanMemoryManager(device)
        self.command_manager = CommandBufferManager(device, queue_family_index)
        self.pipeline_manager = VulkanPipeline(device)
        self.shader_manager = ShaderManager(device)

        # Cache for tensor descriptors
        self._tensor_cache: Dict[torch.Tensor, TensorDescriptor] = {}

        # Initialize pipelines
        self._init_pipelines()

    def register_tensor(self, tensor: torch.Tensor) -> TensorDescriptor:
        """Register a PyTorch tensor for Vulkan operations."""
        if tensor in self._tensor_cache:
            return self._tensor_cache[tensor]

        # Get tensor info
        size_bytes = tensor.nelement() * tensor.element_size()
        is_contiguous = tensor.is_contiguous()

        # Allocate Vulkan buffer
        buffer, memory, offset = self.memory_manager.allocate_buffer(
            size_bytes,
            BufferUsage.TENSOR_STORAGE,
            vk.MemoryPropertyFlags.HOST_VISIBLE | vk.MemoryPropertyFlags.HOST_COHERENT,
        )

        # Create descriptor
        descriptor = TensorDescriptor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            strides=tensor.stride(),
            buffer=buffer,  # VkBuffer handle
            offset=offset,
            size=size_bytes,
            is_contiguous=is_contiguous,
        )

        # Copy data to device
        self._copy_to_device(tensor, descriptor)
        self._tensor_cache[tensor] = descriptor

        return descriptor

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication using Vulkan compute."""
        # Register tensors
        a_desc = self.register_tensor(a)
        b_desc = self.register_tensor(b)

        # Create output tensor
        out_shape = torch.Size((a.shape[0], b.shape[1]))
        out = torch.empty(out_shape, dtype=a.dtype, device="cpu")
        out_desc = self.register_tensor(out)

        # Configure compute dispatch
        M, K = a.shape
        N = b.shape[1]

        config = CommandConfig(
            pipeline_type=PipelineType.MATMUL,
            descriptor_sets=[self._create_matmul_descriptors(a_desc, b_desc, out_desc)],
            push_constants=self._create_matmul_push_constants(M, N, K),
            dispatch_x=(M + 15) // 16,
            dispatch_y=(N + 15) // 16,
        )

        # Record and submit commands
        cmd = self.command_manager.allocate_command_buffer()
        self.command_manager.record_compute_commands(
            cmd,
            self.pipeline_manager.get_pipeline(PipelineType.MATMUL),
            self.pipeline_manager.get_layout(PipelineType.MATMUL),
            config,
        )

        fence = vk.create_fence(self.device, vk.FenceCreateInfo(), None)
        self.command_manager.submit_commands(cmd, self.queue, fence=fence)

        # Wait for completion
        vk.wait_for_fences(self.device, [fence], True, uint64_max)

        # Copy result back
        self._copy_from_device(out_desc, out)

        # Cleanup
        vk.destroy_fence(self.device, fence, None)
        self.command_manager.free_buffers([cmd])

        return out

    def adaptive_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        resolution: float,
        density_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive attention with dynamic resolution.

        Args:
            queries: Query tensor of shape [batch, num_heads, seq_len, head_dim]
            keys: Key tensor of shape [batch, num_heads, seq_len, head_dim]
            values: Value tensor of shape [batch, num_heads, seq_len, head_dim]
            resolution: Target computational resolution (0-1)
            density_threshold: Threshold for information density

        Returns:
            Tuple of:
            - Output tensor of shape [batch, num_heads, seq_len, head_dim]
            - Metrics tensor of shape [batch, num_heads, num_tiles] containing
              information density per tile
        """
        # Register input tensors
        q_desc = self.register_tensor(queries)
        k_desc = self.register_tensor(keys)
        v_desc = self.register_tensor(values)

        # Create output tensors
        batch, num_heads, seq_len, head_dim = queries.shape
        tile_size = 32  # Base tile size
        num_tiles = seq_len // tile_size

        output = torch.zeros_like(queries)
        metrics = torch.zeros((batch, num_heads, num_tiles), dtype=queries.dtype)

        out_desc = self.register_tensor(output)
        metrics_desc = self.register_tensor(metrics)

        # Configure compute dispatch
        config = CommandConfig(
            pipeline_type=PipelineType.ADAPTIVE_ATTENTION,
            descriptor_sets=[
                self._create_attention_descriptors(
                    q_desc, k_desc, v_desc, out_desc, metrics_desc
                )
            ],
            push_constants=self._create_attention_push_constants(
                seq_len, num_heads, head_dim, tile_size, resolution, density_threshold
            ),
            dispatch_x=(batch * num_heads * num_tiles + 255) // 256,
        )

        # Record and submit commands
        cmd = self.command_manager.allocate_command_buffer()
        self.command_manager.record_compute_commands(
            cmd,
            self.pipeline_manager.get_pipeline(PipelineType.ADAPTIVE_ATTENTION),
            self.pipeline_manager.get_layout(PipelineType.ADAPTIVE_ATTENTION),
            config,
        )

        fence = vk.create_fence(self.device, vk.FenceCreateInfo(), None)
        self.command_manager.submit_commands(cmd, self.queue, fence=fence)

        # Wait for completion
        vk.wait_for_fences(self.device, [fence], True, uint64_max)

        # Copy results back
        self._copy_from_device(out_desc, output)
        self._copy_from_device(metrics_desc, metrics)

        # Cleanup
        vk.destroy_fence(self.device, fence, None)
        self.command_manager.free_buffers([cmd])

        return output, metrics

    def adaptive_attention_backward(
        self,
        grad_output: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        resolution: float,
        density_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradients for adaptive attention.

        Args:
            grad_output: Gradient tensor [batch, num_heads, seq_len, head_dim]
            queries: Query tensor [batch, num_heads, seq_len, head_dim]
            keys: Key tensor [batch, num_heads, seq_len, head_dim]
            values: Value tensor [batch, num_heads, seq_len, head_dim]
            resolution: Target computational resolution
            density_threshold: Information density threshold

        Returns:
            Tuple of gradients for (queries, keys, values)
        """
        # Register input tensors
        grad_desc = self.register_tensor(grad_output)
        q_desc = self.register_tensor(queries)
        k_desc = self.register_tensor(keys)
        v_desc = self.register_tensor(values)

        # Create gradient tensors
        grad_q = torch.zeros_like(queries)
        grad_k = torch.zeros_like(keys)
        grad_v = torch.zeros_like(values)

        grad_q_desc = self.register_tensor(grad_q)
        grad_k_desc = self.register_tensor(grad_k)
        grad_v_desc = self.register_tensor(grad_v)

        # Configure compute dispatch
        batch, num_heads, seq_len, head_dim = queries.shape
        tile_size = 32  # Base tile size
        num_tiles = seq_len // tile_size

        config = CommandConfig(
            pipeline_type=PipelineType.ADAPTIVE_ATTENTION_BACKWARD,
            descriptor_sets=[
                self._create_attention_backward_descriptors(
                    grad_desc,
                    q_desc,
                    k_desc,
                    v_desc,
                    grad_q_desc,
                    grad_k_desc,
                    grad_v_desc,
                )
            ],
            push_constants=self._create_attention_push_constants(
                seq_len, num_heads, head_dim, tile_size, resolution, density_threshold
            ),
            dispatch_x=(batch * num_heads * num_tiles + 255) // 256,
        )

        # Record and submit commands
        cmd = self.command_manager.allocate_command_buffer()
        self.command_manager.record_compute_commands(
            cmd,
            self.pipeline_manager.get_pipeline(
                PipelineType.ADAPTIVE_ATTENTION_BACKWARD
            ),
            self.pipeline_manager.get_layout(PipelineType.ADAPTIVE_ATTENTION_BACKWARD),
            config,
        )

        fence = vk.create_fence(self.device, vk.FenceCreateInfo(), None)
        self.command_manager.submit_commands(cmd, self.queue, fence=fence)

        # Wait for completion
        vk.wait_for_fences(self.device, [fence], True, uint64_max)

        # Copy results back
        self._copy_from_device(grad_q_desc, grad_q)
        self._copy_from_device(grad_k_desc, grad_k)
        self._copy_from_device(grad_v_desc, grad_v)

        # Cleanup
        vk.destroy_fence(self.device, fence, None)
        self.command_manager.free_buffers([cmd])

        return grad_q, grad_k, grad_v

    def _create_attention_descriptors(
        self,
        q_desc: TensorDescriptor,
        k_desc: TensorDescriptor,
        v_desc: TensorDescriptor,
        out_desc: TensorDescriptor,
        metrics_desc: TensorDescriptor,
    ) -> int:  # VkDescriptorSet handle
        """Create descriptor set for adaptive attention."""
        return self.shader_manager.create_compute_descriptors(
            PipelineType.ADAPTIVE_ATTENTION,
            [
                (0, q_desc.buffer),  # Queries
                (1, k_desc.buffer),  # Keys
                (2, v_desc.buffer),  # Values
                (3, out_desc.buffer),  # Output
                (4, metrics_desc.buffer),  # Tile metrics
            ],
        )

    def _create_attention_push_constants(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        tile_size: int,
        resolution: float,
        density_threshold: float,
    ) -> bytes:
        """Create push constants for adaptive attention."""
        return struct.pack(
            "IIIIff",
            seq_len,
            num_heads,
            head_dim,
            tile_size,
            resolution,
            density_threshold,
        )

    def _create_attention_backward_descriptors(
        self,
        grad_desc: TensorDescriptor,
        q_desc: TensorDescriptor,
        k_desc: TensorDescriptor,
        v_desc: TensorDescriptor,
        grad_q_desc: TensorDescriptor,
        grad_k_desc: TensorDescriptor,
        grad_v_desc: TensorDescriptor,
    ) -> int:  # VkDescriptorSet handle
        """Create descriptor set for backward attention pass."""
        return self.shader_manager.create_compute_descriptors(
            PipelineType.ADAPTIVE_ATTENTION_BACKWARD,
            [
                (0, grad_desc.buffer),  # Gradient output
                (1, q_desc.buffer),  # Queries
                (2, k_desc.buffer),  # Keys
                (3, v_desc.buffer),  # Values
                (4, grad_q_desc.buffer),  # Query gradients
                (5, grad_k_desc.buffer),  # Key gradients
                (6, grad_v_desc.buffer),  # Value gradients
            ],
        )

    def _copy_to_device(self, tensor: torch.Tensor, desc: TensorDescriptor):
        """Copy tensor data to Vulkan buffer."""
        memory = self.memory_manager.get_memory(desc.buffer)  # Get VkDeviceMemory handle
        data_ptr = vk.map_memory(
            self.device,
            memory,
            desc.offset,
            desc.size,
            0,
        )

        if desc.is_contiguous:
            # Fast path for contiguous tensors
            tensor_np = tensor.detach().cpu().numpy()
            src_ptr = tensor_np.ctypes.data
            ctypes.memmove(data_ptr, src_ptr, desc.size)
        else:
            # Slower path for non-contiguous tensors
            tensor_contig = tensor.contiguous()
            tensor_np = tensor_contig.detach().cpu().numpy()
            src_ptr = tensor_np.ctypes.data
            ctypes.memmove(data_ptr, src_ptr, desc.size)

        vk.unmap_memory(self.device, memory)

    def _copy_from_device(self, desc: TensorDescriptor, tensor: torch.Tensor):
        """Copy Vulkan buffer data back to tensor."""
        memory = self.memory_manager.get_memory(desc.buffer)  # Get VkDeviceMemory handle
        data_ptr = vk.map_memory(
            self.device,
            memory,
            desc.offset,
            desc.size,
            0,
        )

        tensor_np = tensor.detach().cpu().numpy()
        dst_ptr = tensor_np.ctypes.data
        ctypes.memmove(dst_ptr, data_ptr, desc.size)

        vk.unmap_memory(self.device, memory)

    def _create_matmul_descriptors(
        self,
        a_desc: TensorDescriptor,
        b_desc: TensorDescriptor,
        out_desc: TensorDescriptor,
    ) -> int:  # VkDescriptorSet handle
        """Create descriptor set for matrix multiplication."""
        return self.shader_manager.create_compute_descriptors(
            PipelineType.MATMUL,
            [
                (0, a_desc.buffer),  # Input matrix A
                (1, b_desc.buffer),  # Input matrix B
                (2, out_desc.buffer),  # Output matrix
            ],
        )

    def _create_matmul_push_constants(self, M: int, N: int, K: int) -> bytes:
        """Create push constants for matrix multiplication."""
        return struct.pack("III", M, N, K)

    def _init_pipelines(self):
        """Initialize compute pipelines."""
        # Load shader code
        with open("shaders/matmul.comp.spv", "rb") as f:
            matmul_code = f.read()
        with open("shaders/adaptive_attention.comp.spv", "rb") as f:
            attention_code = f.read()
        with open("shaders/adaptive_attention_backward.comp.spv", "rb") as f:
            attention_backward_code = f.read()

        # Create pipelines
        self.pipeline_manager.create_pipeline(
            PipelineType.MATMUL,
            matmul_code,
            12  # Push constant size (3 * uint32)
        )
        self.pipeline_manager.create_pipeline(
            PipelineType.ADAPTIVE_ATTENTION,
            attention_code,
            24  # Push constant size (6 * uint32)
        )
        self.pipeline_manager.create_pipeline(
            PipelineType.ADAPTIVE_ATTENTION_BACKWARD,
            attention_backward_code,
            24  # Push constant size (6 * uint32)
        )

    def cleanup(self):
        """Clean up Vulkan resources."""
        # Free tensor buffers
        for desc in self._tensor_cache.values():
            self.memory_manager.free_buffer(desc.buffer, BufferUsage.TENSOR_STORAGE)

        self._tensor_cache.clear()

        # Cleanup managers
        self.command_manager.cleanup()
        self.pipeline_manager.cleanup()
        self.shader_manager.cleanup()
        self.memory_manager.cleanup()
