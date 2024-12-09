"""GPU memory management system for efficient tensor operations."""

import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class GPUMemoryMetrics:
    """Metrics for GPU memory usage tracking."""

    allocated_memory: int  # Current allocated memory in bytes
    peak_memory: int  # Peak memory usage in bytes
    fragmentation_ratio: float  # Memory fragmentation ratio
    cache_memory: int  # Memory in GPU cache
    operation_type: str  # Type of operation (allocate, free, transfer)
    device_id: int  # GPU device ID


class GPUMemoryManager:
    """Manages GPU memory allocation and tracking for tensors."""

    def __init__(self, device_id: int = 0):
        """Initialize GPU memory manager.

        Args:
            device_id: GPU device ID to manage
        """
        self._device_id = device_id
        self._device = torch.device(f"cuda:{device_id}")
        self._allocated_memory = 0
        self._peak_memory = 0
        self._tensor_allocations: Dict[int, int] = {}  # tensor_id -> size
        self._metrics: List[GPUMemoryMetrics] = []
        self._cache_size = 1024 * 1024 * 128  # 128MB default cache

        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. GPU operations cannot be performed."
            )

        # Initialize CUDA device
        torch.cuda.set_device(device_id)

        # Empty cache to start fresh
        torch.cuda.empty_cache()

    def allocate_tensor(
        self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Allocate a new tensor on GPU with given size.

        Args:
            size: Tensor dimensions
            dtype: Tensor data type

        Returns:
            Allocated GPU tensor
        """
        # Calculate memory size
        element_size = torch.tensor([], dtype=dtype).element_size()
        memory_size = element_size * torch.prod(torch.tensor(size)).item()

        # Check if we need to clear cache
        if self._allocated_memory + memory_size > self._peak_memory + self._cache_size:
            torch.cuda.empty_cache()
            gc.collect()

        # Allocate tensor
        tensor = torch.zeros(size, dtype=dtype, device=self._device)
        tensor_id = id(tensor)

        # Update tracking
        self._allocated_memory += memory_size
        self._peak_memory = max(self._peak_memory, self._allocated_memory)
        self._tensor_allocations[tensor_id] = memory_size

        # Record metrics
        self._metrics.append(
            GPUMemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                cache_memory=self.get_cache_memory(),
                operation_type="allocate",
                device_id=self._device_id,
            )
        )

        return tensor

    def transfer_to_gpu(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Transfer a CPU tensor to GPU with memory tracking.

        Args:
            cpu_tensor: Tensor on CPU

        Returns:
            Tensor on GPU
        """
        memory_size = cpu_tensor.element_size() * cpu_tensor.nelement()

        # Check cache
        if self._allocated_memory + memory_size > self._peak_memory + self._cache_size:
            torch.cuda.empty_cache()
            gc.collect()

        # Transfer tensor
        gpu_tensor = cpu_tensor.to(self._device)
        tensor_id = id(gpu_tensor)

        # Update tracking
        self._allocated_memory += memory_size
        self._peak_memory = max(self._peak_memory, self._allocated_memory)
        self._tensor_allocations[tensor_id] = memory_size

        # Record metrics
        self._metrics.append(
            GPUMemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                cache_memory=self.get_cache_memory(),
                operation_type="transfer",
                device_id=self._device_id,
            )
        )

        return gpu_tensor

    def get_allocated_memory(self) -> int:
        """Get current allocated memory in bytes."""
        return self._allocated_memory

    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self._peak_memory

    def get_cache_memory(self) -> int:
        """Get current GPU cache memory in bytes."""
        return torch.cuda.memory_reserved(
            self._device_id
        ) - torch.cuda.memory_allocated(self._device_id)

    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self._allocated_memory:
            return 0.0

        # Calculate fragmentation based on tensor allocation patterns
        total_gaps = 0
        sorted_allocations = sorted(self._tensor_allocations.values())

        for i in range(len(sorted_allocations) - 1):
            gap = sorted_allocations[i + 1] - sorted_allocations[i]
            if gap > 0:
                total_gaps += gap

        return total_gaps / self._allocated_memory if self._allocated_memory else 0.0

    def optimize_memory_layout(self) -> None:
        """Optimize GPU memory layout to reduce fragmentation."""
        if not self._tensor_allocations:
            return

        # Get all tensors and their sizes
        tensors = []
        for tensor_id, size in self._tensor_allocations.items():
            tensor = None
            try:
                # Try to get tensor from id (may fail if tensor was deleted)
                tensor = ctypes.cast(tensor_id, ctypes.py_object).value
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                    tensors.append((tensor, size))
            except:
                pass

        if not tensors:
            return

        # Sort tensors by size
        tensors.sort(key=lambda x: x[1], reverse=True)

        # Reallocate tensors in size order to reduce fragmentation
        torch.cuda.empty_cache()
        for tensor, _ in tensors:
            if tensor.is_contiguous():
                tensor.storage().resize_(tensor.numel())

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.clear_cache()
        except:
            pass
