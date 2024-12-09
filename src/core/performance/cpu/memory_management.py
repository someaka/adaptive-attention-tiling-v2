import torch
import gc
import weakref
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

@dataclass
class MemoryMetrics:
    """Metrics for memory usage tracking."""
    allocated_memory: int
    peak_memory: int
    fragmentation_ratio: float
    operation_type: str

class MemoryManager:
    """Manages memory allocation and operations for tensors."""
    
    def __init__(self):
        self._allocated_memory = 0
        self._peak_memory = 0
        self._metrics = []
        self._tensor_allocations = {}  # Track tensor allocations
        
    def allocate_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Allocate a new tensor with given size."""
        tensor = torch.zeros(size)
        memory_size = tensor.element_size() * tensor.nelement()
        
        self._allocated_memory += memory_size
        self._peak_memory = max(self._peak_memory, self._allocated_memory)
        tensor_id = id(tensor)
        self._tensor_allocations[tensor_id] = memory_size
        
        # Use weakref to track tensor deletion
        def cleanup(_: weakref.ref) -> None:
            if tensor_id in self._tensor_allocations:
                self._allocated_memory -= self._tensor_allocations[tensor_id]
                del self._tensor_allocations[tensor_id]
                self._metrics.append(MemoryMetrics(
                    allocated_memory=self._allocated_memory,
                    peak_memory=self._peak_memory,
                    fragmentation_ratio=self.get_fragmentation_ratio(),
                    operation_type="deallocate"
                ))
        
        weakref.finalize(tensor, cleanup, weakref.ref(tensor))
        
        self._metrics.append(MemoryMetrics(
            allocated_memory=self._allocated_memory,
            peak_memory=self._peak_memory,
            fragmentation_ratio=self.get_fragmentation_ratio(),
            operation_type="allocate"
        ))
        
        return tensor
    
    def get_allocated_memory(self) -> int:
        """Get current allocated memory in bytes."""
        return self._allocated_memory
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self._peak_memory
    
    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self._allocated_memory:
            return 0.0
            
        # Calculate fragmentation based on tensor allocation patterns
        total_gaps = 0
        sorted_allocations = sorted(self._tensor_allocations.values())
        
        for i in range(len(sorted_allocations) - 1):
            gap = sorted_allocations[i+1] - sorted_allocations[i]
            if gap > 0:
                total_gaps += gap
                
        return total_gaps / self._allocated_memory if self._allocated_memory else 0.0
    
    def inplace_operation(self, tensor: torch.Tensor, operation: Callable[[torch.Tensor], None]) -> None:
        """Perform an in-place operation on a tensor."""
        operation(tensor)
        
        self._metrics.append(MemoryMetrics(
            allocated_memory=self._allocated_memory,
            peak_memory=self._peak_memory,
            fragmentation_ratio=self.get_fragmentation_ratio(),
            operation_type="inplace"
        ))
    
    def optimized_matmul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform memory-optimized matrix multiplication."""
        # Use a pre-allocated output tensor when possible
        out_shape = (x.size(0), y.size(1))
        result = torch.zeros(out_shape)
        
        # Perform multiplication in chunks to reduce memory usage
        chunk_size = min(128, x.size(1))
        for i in range(0, x.size(1), chunk_size):
            end = min(i + chunk_size, x.size(1))
            result += torch.matmul(x[:, i:end], y[i:end, :])
            
        self._metrics.append(MemoryMetrics(
            allocated_memory=self._allocated_memory,
            peak_memory=self._peak_memory,
            fragmentation_ratio=self.get_fragmentation_ratio(),
            operation_type="matmul"
        ))
        
        return result
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        # Clear all tracked allocations
        self._tensor_allocations.clear()
        self._allocated_memory = 0
        gc.collect()
