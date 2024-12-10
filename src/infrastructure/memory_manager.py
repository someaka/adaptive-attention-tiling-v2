"""Memory management module for efficient resource utilization."""

from typing import Dict, Optional, Any
import numpy as np


class MemoryManager:
    """Manages memory allocation and pooling."""

    def __init__(self, pool_size: int = 1024):
        """Initialize memory manager.
        
        Args:
            pool_size: Size of the memory pool in MB
        """
        self.pool_size = pool_size * 1024 * 1024  # Convert to bytes
        self.pools: Dict[str, Any] = {}
        self.allocated = 0

    def allocate(self, size: int, dtype: np.dtype) -> Optional[np.ndarray]:
        """Allocate memory from pool.
        
        Args:
            size: Size in bytes
            dtype: NumPy dtype
            
        Returns:
            Allocated array or None if allocation fails
        """
        if self.allocated + size > self.pool_size:
            return None
            
        key = f"{size}_{dtype}"
        if key not in self.pools:
            self.pools[key] = []
            
        if self.pools[key]:
            return self.pools[key].pop()
            
        self.allocated += size
        return np.empty(size // dtype.itemsize, dtype=dtype)

    def deallocate(self, array: np.ndarray) -> None:
        """Return memory to pool.
        
        Args:
            array: Array to deallocate
        """
        key = f"{array.nbytes}_{array.dtype}"
        if key not in self.pools:
            self.pools[key] = []
        self.pools[key].append(array)

    def clear_pools(self) -> None:
        """Clear all memory pools."""
        self.pools.clear()
        self.allocated = 0
