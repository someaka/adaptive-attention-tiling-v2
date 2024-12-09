"""Memory Management Optimization for CPU Operations.

This module provides tools for optimizing memory usage, including pooling,
cache optimization, and memory access pattern improvements.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import gc
import psutil

@dataclass
class MemoryStats:
    """Statistics for memory operations."""
    allocation_size: int
    pool_hits: int
    cache_hits: int
    fragmentation: float
    access_pattern: str

class MemoryPool:
    """Memory pool for tensor reuse."""
    
    def __init__(self, max_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_size = max_size
        self.current_size = 0
        self.pools: Dict[Tuple[int, ...], List[torch.Tensor]] = defaultdict(list)
        self.stats = defaultdict(int)
    
    def acquire(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Acquire a tensor from the pool or create new if none available."""
        if shape in self.pools and self.pools[shape]:
            self.stats['pool_hits'] += 1
            return self.pools[shape].pop()
        
        self.stats['pool_misses'] += 1
        tensor = torch.empty(shape, dtype=dtype)
        self.current_size += tensor.numel() * tensor.element_size()
        return tensor
    
    def release(self, tensor: torch.Tensor) -> None:
        """Release a tensor back to the pool."""
        if self.current_size + tensor.numel() * tensor.element_size() > self.max_size:
            self._cleanup()
        
        shape = tuple(tensor.shape)
        self.pools[shape].append(tensor)
        self.current_size += tensor.numel() * tensor.element_size()
    
    def _cleanup(self) -> None:
        """Clean up least recently used tensors."""
        freed_size = 0
        target_size = self.max_size * 0.8  # Aim to free 20%
        
        for shape in sorted(self.pools, key=lambda x: len(self.pools[x])):
            while self.pools[shape] and self.current_size - freed_size > target_size:
                tensor = self.pools[shape].pop()
                freed_size += tensor.numel() * tensor.element_size()
        
        self.current_size -= freed_size
        gc.collect()

class CacheOptimizer:
    """Optimizes memory access patterns for better cache utilization."""
    
    def __init__(self, cache_line_size: int = 64):
        self.cache_line_size = cache_line_size
        self.stats = {'hits': 0, 'misses': 0}
    
    def optimize_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for cache efficiency."""
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Align to cache line size if possible
        if tensor.element_size() * tensor.size(-1) % self.cache_line_size != 0:
            pad_size = (self.cache_line_size - 
                       (tensor.element_size() * tensor.size(-1)) % self.cache_line_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        
        return tensor
    
    def prefetch(self, tensor: torch.Tensor, indices: torch.Tensor) -> None:
        """Prefetch data into cache."""
        # Simple prefetching strategy
        torch.index_select(tensor, 0, indices[0:1])
        self.stats['hits'] += 1

class MemoryManager:
    """Manages memory optimization strategies."""
    
    def __init__(self,
                 pool_size: int = 1024 * 1024 * 1024,
                 enable_monitoring: bool = True):
        self.pool = MemoryPool(max_size=pool_size)
        self.cache_optimizer = CacheOptimizer()
        self.enable_monitoring = enable_monitoring
        self.stats: List[MemoryStats] = []

    def create_pool(self, size: int) -> MemoryPool:
        """Create a new memory pool with specified size."""
        return MemoryPool(max_size=size)

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.stats.clear()

    def optimize_tensor(self, 
                       tensor: torch.Tensor,
                       access_pattern: str = "sequential") -> torch.Tensor:
        """Optimize tensor for memory efficiency."""
        # Get tensor from pool
        optimized = self.pool.acquire(tensor.shape, tensor.dtype)
        optimized.copy_(tensor)
        
        # Optimize layout
        optimized = self.cache_optimizer.optimize_layout(optimized)
        
        # Record stats
        self.stats.append(MemoryStats(
            allocation_size=tensor.numel() * tensor.element_size(),
            pool_hits=self.pool.stats['pool_hits'],
            cache_hits=self.cache_optimizer.stats['hits'],
            fragmentation=self._calculate_fragmentation(),
            access_pattern=access_pattern
        ))
        
        return optimized
    
    def release_tensor(self, tensor: torch.Tensor) -> None:
        """Release tensor back to pool."""
        self.pool.release(tensor)
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio."""
        total_allocated = sum(t.numel() * t.element_size() 
                            for pools in self.pool.pools.values()
                            for t in pools)
        return 1.0 - (total_allocated / self.pool.current_size 
                     if self.pool.current_size > 0 else 1.0)
    
    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """Get memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'pool_size': self.pool.current_size,
            'pool_hits': self.pool.stats['pool_hits'],
            'pool_misses': self.pool.stats['pool_misses'],
            'cache_hits': self.cache_optimizer.stats['hits'],
            'fragmentation': self._calculate_fragmentation()
        }
    
    def clear_stats(self) -> None:
        """Clear collected statistics."""
        self.stats.clear()
        self.pool.stats.clear()
        self.cache_optimizer.stats.clear()
