"""CPU memory management implementation."""

import gc
import weakref
from typing import Dict, Optional, Tuple, Union, Any

import torch
import numpy as np

from .memory_base import MemoryManagerBase, MemoryError


class CPUMemoryManager(MemoryManagerBase):
    """Memory manager for CPU operations."""

    def __init__(self):
        super().__init__()
        self._tensor_allocations: Dict[int, int] = {}  # tensor_id -> size
        self._tensor_refs: Dict[int, weakref.ReferenceType] = {}  # tensor_id -> weakref

    def allocate_tensor(self, size: Union[Tuple[int, ...], torch.Size], dtype: Any = torch.float32) -> torch.Tensor:
        """Allocate a CPU tensor.
        
        Args:
            size: Tensor dimensions
            dtype: Tensor data type
            
        Returns:
            Allocated tensor
        """
        try:
            # Create tensor
            tensor = torch.zeros(size, dtype=dtype)
            tensor_id = id(tensor)
            
            # Calculate memory size
            memory_size = tensor.element_size() * tensor.nelement()
            
            # Update tracking
            self._allocated_memory += memory_size
            self._peak_memory = max(self._peak_memory, self._allocated_memory)
            self._tensor_allocations[tensor_id] = memory_size
            
            # Create weakref for cleanup
            def cleanup(ref: weakref.ReferenceType) -> None:
                if tensor_id in self._tensor_allocations:
                    self._allocated_memory -= self._tensor_allocations[tensor_id]
                    del self._tensor_allocations[tensor_id]
                    if tensor_id in self._tensor_refs:
                        del self._tensor_refs[tensor_id]
                    self.record_metric("free")
                    
            # Store weakref
            self._tensor_refs[tensor_id] = weakref.ref(tensor, cleanup)
            
            self.record_metric("allocate")
            return tensor
            
        except Exception as e:
            raise MemoryError(f"Failed to allocate tensor: {e}")

    def free_tensor(self, tensor: torch.Tensor) -> None:
        """Free a CPU tensor.
        
        Args:
            tensor: Tensor to free
        """
        try:
            tensor_id = id(tensor)
            if tensor_id in self._tensor_allocations:
                self._allocated_memory -= self._tensor_allocations[tensor_id]
                del self._tensor_allocations[tensor_id]
                if tensor_id in self._tensor_refs:
                    del self._tensor_refs[tensor_id]
                self.record_metric("free")
                
        except Exception as e:
            raise MemoryError(f"Failed to free tensor: {e}")

    def copy_to_device(self, src: Union[torch.Tensor, np.ndarray], dst: torch.Tensor) -> None:
        """Copy data to CPU tensor.
        
        Args:
            src: Source data
            dst: Destination tensor
        """
        try:
            if isinstance(src, np.ndarray):
                dst.copy_(torch.from_numpy(src))
            else:
                dst.copy_(src)
            self.record_metric("copy")
            
        except Exception as e:
            raise MemoryError(f"Failed to copy data: {e}")

    def copy_from_device(self, src: torch.Tensor, dst: Union[torch.Tensor, np.ndarray]) -> None:
        """Copy data from CPU tensor.
        
        Args:
            src: Source tensor
            dst: Destination data
        """
        try:
            if isinstance(dst, np.ndarray):
                dst[...] = src.numpy()
            else:
                dst.copy_(src)
            self.record_metric("copy")
            
        except Exception as e:
            raise MemoryError(f"Failed to copy data: {e}")

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

    def cleanup(self) -> None:
        """Clean up memory resources."""
        try:
            # Clear all tracked allocations
            self._tensor_allocations.clear()
            self._tensor_refs.clear()
            self._allocated_memory = 0
            gc.collect()
            self.record_metric("cleanup")
            
        except Exception as e:
            raise MemoryError(f"Failed to cleanup: {e}")

    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.cleanup()
        except:
            pass 