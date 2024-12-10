"""CPU optimization module for performance enhancement."""

import os
import numpy as np


class CPUOptimizer:
    """Handles CPU-specific optimizations."""

    def __init__(self):
        """Initialize CPU optimizer."""
        self.vectorization_enabled = True
        self.cache_optimization_enabled = True
        self.thread_pool_size = self._get_optimal_thread_pool_size()

    def _get_optimal_thread_pool_size(self) -> int:
        """Determine optimal thread pool size based on system."""
        return max(1, len(os.sched_getaffinity(0)))

    def optimize_memory_layout(self, data: np.ndarray) -> np.ndarray:
        """Optimize memory layout for CPU operations."""
        return np.ascontiguousarray(data)

    def enable_vectorization(self, enabled: bool = True) -> None:
        """Enable or disable vectorization."""
        self.vectorization_enabled = enabled

    def enable_cache_optimization(self, enabled: bool = True) -> None:
        """Enable or disable cache optimization."""
        self.cache_optimization_enabled = enabled

    def set_thread_pool_size(self, size: int) -> None:
        """Set thread pool size."""
        self.thread_pool_size = max(1, size)
