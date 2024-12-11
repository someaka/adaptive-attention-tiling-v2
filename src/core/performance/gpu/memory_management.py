"""GPU memory management utilities."""

import torch
from typing import Optional, Dict, List
import gc


class GPUMemoryManager:
    """Manages GPU memory allocation and deallocation."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory manager.
        
        Args:
            device: GPU device to manage
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cache: Dict[str, List[torch.Tensor]] = {}
        
    def allocate(self, size: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Allocate GPU memory.
        
        Args:
            size: Tensor size
            dtype: Tensor dtype
            
        Returns:
            Allocated tensor
        """
        key = f"{size}_{dtype}"
        
        # Check cache
        if key in self._cache and self._cache[key]:
            return self._cache[key].pop()
            
        # Allocate new tensor
        return torch.empty(size, dtype=dtype, device=self.device)
        
    def free(self, tensor: torch.Tensor):
        """Free GPU memory.
        
        Args:
            tensor: Tensor to free
        """
        key = f"{tuple(tensor.size())}_{tensor.dtype}"
        
        # Add to cache
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append(tensor)
        
    def clear_cache(self):
        """Clear memory cache."""
        self._cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,
            'cached': torch.cuda.memory_reserved(self.device) / 1024**2,
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_cache()