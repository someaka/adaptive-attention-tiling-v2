import numpy as np
from collections import OrderedDict
import bisect
import gc
import time
from functools import reduce
import weakref
from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, List, Optional, Set

import torch


@dataclass
class MemoryMetrics:
    """Metrics for memory usage tracking."""
    allocated_memory: int
    peak_memory: int
    fragmentation_ratio: float
    operation_type: str
    timestamp: float = field(default_factory=time.perf_counter)


class LRUCache:
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[weakref.ref]:
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: str, value: weakref.ref) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # Remove least recently used
        self.cache[key] = value


class SizeClassAllocator:
    """Size class allocator with buddy system for efficient memory management."""
    
    def __init__(self):
        # Use power-of-2 size classes for better alignment
        self.min_size = 32
        self.max_size = 2**21  # ~2MB
        self.size_classes = [2**i for i in range(int(np.log2(self.min_size)), int(np.log2(self.max_size)) + 1)]
        self.free_blocks: Dict[int, List[torch.Tensor]] = {size: [] for size in self.size_classes}
        self.allocation_counts: Dict[int, int] = {size: 0 for size in self.size_classes}
        
    def get_size_class(self, size: int) -> int:
        """Get the appropriate size class using buddy system."""
        if size > self.max_size:
            return size
        # Round up to next power of 2
        return 1 << (size - 1).bit_length()
        
    def split_block(self, block: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split a block into two buddies if possible."""
        current_size = block.nelement() * block.element_size()
        if current_size // 2 >= target_size:
            half_size = current_size // 2
            buddy = block.narrow(0, half_size, half_size)
            return block.narrow(0, 0, half_size), buddy
        return block, None
        
    def allocate(self, size: int) -> Optional[torch.Tensor]:
        """Allocate a block using buddy system."""
        size_class = self.get_size_class(size)
        
        # Try to find the smallest suitable block
        for current_size in self.size_classes:
            if current_size >= size_class and self.free_blocks[current_size]:
                block = self.free_blocks[current_size].pop()
                if current_size > size_class:
                    # Split into buddies if block is too large
                    block, buddy = self.split_block(block, size_class)
                    if buddy is not None:
                        self.free_blocks[current_size // 2].append(buddy)
                
                self.allocation_counts[size_class] += 1
                return block
                
        return None
        
    def coalesce_blocks(self) -> None:
        """Coalesce buddy blocks to reduce fragmentation."""
        for size in self.size_classes[:-1]:
            while len(self.free_blocks[size]) >= 2:
                buddy1 = self.free_blocks[size].pop()
                buddy2 = self.free_blocks[size].pop()
                coalesced = torch.cat([buddy1, buddy2])
                self.free_blocks[size * 2].append(coalesced)

    def deallocate(self, tensor: torch.Tensor, size: int) -> None:
        """Return a block to the pool."""
        size_class = self.get_size_class(size)
        if size_class not in self.free_blocks:
            self.free_blocks[size_class] = []
        self.free_blocks[size_class].append(tensor)
        if size_class in self.allocation_counts:
            self.allocation_counts[size_class] -= 1
    
    def clear(self) -> None:
        """Clear all blocks."""
        self.free_blocks.clear()
        # Reinitialize size classes
        self.free_blocks = {size: [] for size in self.size_classes}
        self.allocation_counts = {size: 0 for size in self.size_classes}


class MemoryManager:
    """Manages memory allocation and operations for tensors."""

    def __init__(self, cache_size: int = 32, cleanup_threshold: float = 0.7):
        self._allocated_memory = 0
        self._peak_memory = 0
        self._metrics: List[MemoryMetrics] = []
        self._tensor_allocations: Dict[int, int] = {}
        self._tensor_refs: List[weakref.ref] = []
        self._cache = LRUCache(cache_size)
        self._cache_hits = 0
        self._cache_misses = 0
        self._sorted_allocations: List[int] = []
        self._chunk_size = self._calculate_chunk_size()
        self._allocator = SizeClassAllocator()
        self._pool_hits = 0
        self._pool_misses = 0
        self._fixed_tensors: Set[int] = set()
        self._cleanup_threshold = cleanup_threshold
        self._last_cleanup = time.perf_counter()
        self._cleanup_interval = 1.0
        self._dead_tensors: Set[int] = set()  # Pre-allocated set for dead tensors
        self._memory_pressure = 0.0  # Track memory pressure
        self._total_gaps = 0  # Track total gaps for O(1) fragmentation calculation
        self._pressure_ema_alpha = 0.2  # EMA factor for memory pressure

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on CPU cache size."""
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index0/size') as f:
                l1_size = int(f.read().strip()[:-1]) * 1024  # Convert KB to bytes
                return min(256, l1_size // 4)  # Use quarter of L1 cache or 256
        except:
            return 256  # Default if can't read CPU cache size

    def _update_memory_pressure(self) -> None:
        """Update memory pressure metric using exponential moving average."""
        current_pressure = (
            self._allocated_memory / max(1, self._peak_memory) +
            self.get_fragmentation_ratio()
        ) / 2
        self._memory_pressure = (
            self._pressure_ema_alpha * current_pressure +
            (1 - self._pressure_ema_alpha) * self._memory_pressure
        )

    def _should_cleanup(self) -> bool:
        """Determine if cleanup should be performed based on memory pressure."""
        current_time = time.perf_counter()
        self._update_memory_pressure()
        
        return (
            current_time - self._last_cleanup > self._cleanup_interval and
            (self._memory_pressure > self._cleanup_threshold or
             self.get_fragmentation_ratio() > self._cleanup_threshold)
        )

    def fix_tensor_in_memory(self, tensor: torch.Tensor) -> None:
        """Fix a tensor in memory to prevent relocation."""
        tensor_id = id(tensor)
        if tensor_id not in self._fixed_tensors:
            self._fixed_tensors.add(tensor_id)

    def unfix_tensor(self, tensor: torch.Tensor) -> None:
        """Allow a tensor to be relocated in memory."""
        tensor_id = id(tensor)
        if tensor_id in self._fixed_tensors:
            self._fixed_tensors.remove(tensor_id)

    def allocate_tensor(self, size: Tuple[int, ...], fix: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate a new tensor with improved memory management."""
        if self._should_cleanup():
            self._cleanup_dead_refs()
            self._allocator.coalesce_blocks()
            self._last_cleanup = time.perf_counter()
        
        # Calculate memory size once and reuse
        memory_size = reduce(lambda x, y: x * y, size) * torch.finfo(dtype).bits // 8
        
        # Try to get from allocator first
        pooled_tensor = self._allocator.allocate(memory_size)
        if pooled_tensor is not None:
            self._pool_hits += 1
            pooled_tensor.resize_(size)
            if fix:
                self.fix_tensor_in_memory(pooled_tensor)
            return pooled_tensor
            
        self._pool_misses += 1
        
        # Rest of allocation logic...
        tensor = torch.zeros(size, dtype=dtype)
        
        # Update sorted allocations and gaps
        insert_idx = bisect.bisect_left(self._sorted_allocations, memory_size)
        if insert_idx > 0:
            self._total_gaps += memory_size - self._sorted_allocations[insert_idx - 1]
        if insert_idx < len(self._sorted_allocations):
            self._total_gaps += self._sorted_allocations[insert_idx] - memory_size
        self._sorted_allocations.insert(insert_idx, memory_size)
        
        memory_size = tensor.element_size() * tensor.nelement()
        self._allocated_memory += memory_size
        self._peak_memory = max(self._peak_memory, self._allocated_memory)
        tensor_id = id(tensor)
        self._tensor_allocations[tensor_id] = memory_size
        
        if fix:
            self.fix_tensor_in_memory(tensor)
        
        # Use weakref to track tensor deletion
        def cleanup(ref: weakref.ref) -> None:
            nonlocal tensor_id
            if tensor_id in self._tensor_allocations:
                size = self._tensor_allocations[tensor_id]
                self._allocated_memory -= size
                del self._tensor_allocations[tensor_id]
                if ref in self._tensor_refs:
                    self._tensor_refs.remove(ref)
                # Return tensor to pool if not fixed
                if tensor_id not in self._fixed_tensors:
                    tensor = ref()
                    if tensor is not None:
                        self._allocator.deallocate(tensor, size)
                else:
                    self._fixed_tensors.remove(tensor_id)
                # Remove from sorted allocations
                idx = bisect.bisect_left(self._sorted_allocations, size)
                if idx < len(self._sorted_allocations) and self._sorted_allocations[idx] == size:
                    self._sorted_allocations.pop(idx)
                self._metrics.append(
                    MemoryMetrics(
                        allocated_memory=self._allocated_memory,
                        peak_memory=self._peak_memory,
                        fragmentation_ratio=self.get_fragmentation_ratio(),
                        operation_type="deallocate",
                    )
                )
                tensor_id = None

        # Create and store weak reference
        ref = weakref.ref(tensor, cleanup)
        self._tensor_refs.append(ref)
        
        # Update cache if not fixed
        if not fix:
            self._cache.put(str(size), ref)

        self._metrics.append(
            MemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                operation_type="allocate",
            )
        )

        return tensor

    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio in O(1) time."""
        if not self._allocated_memory or len(self._sorted_allocations) <= 1:
            return 0.0
        return self._total_gaps / self._allocated_memory

    def get_allocated_memory(self) -> int:
        """Get current allocated memory in bytes."""
        # Force cleanup of dead references
        self._cleanup_dead_refs()
        return self._allocated_memory

    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self._peak_memory

    def get_cache_stats(self) -> Tuple[int, int]:
        """Get cache hit and miss statistics."""
        return self._cache_hits, self._cache_misses

    def get_pool_stats(self) -> Tuple[int, int]:
        """Get pool hit and miss statistics."""
        return self._pool_hits, self._pool_misses

    def _cleanup_dead_refs(self) -> None:
        """Clean up dead tensor references with optimized batch processing."""
        # Pre-allocate lists for better memory efficiency
        live_refs = []
        dead_sizes = []
        dead_indices = set()
        
        # Batch process references
        for ref in self._tensor_refs:
            tensor = ref()
            if tensor is not None:
                try:
                    _ = tensor.shape
                    live_refs.append(ref)
                except:
                    tensor_id = id(tensor)
                    if tensor_id in self._tensor_allocations:
                        size = self._tensor_allocations[tensor_id]
                        dead_sizes.append(size)
                        idx = bisect.bisect_left(self._sorted_allocations, size)
                        if idx < len(self._sorted_allocations) and self._sorted_allocations[idx] == size:
                            dead_indices.add(idx)
                            # Update gaps
                            if idx > 0:
                                self._total_gaps -= size - self._sorted_allocations[idx - 1]
                            if idx < len(self._sorted_allocations) - 1:
                                self._total_gaps -= self._sorted_allocations[idx + 1] - size
        
        # Batch cleanup
        if dead_sizes:
            # Update allocations
            self._allocated_memory -= sum(dead_sizes)
            
            # Update sorted allocations efficiently
            new_sorted = [
                size for i, size in enumerate(self._sorted_allocations)
                if i not in dead_indices
            ]
            self._sorted_allocations = new_sorted
            
            # Single garbage collection
            gc.collect()
        
        self._tensor_refs = live_refs
        self._update_memory_pressure()

    def optimized_matmul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform memory-optimized matrix multiplication with adaptive chunking."""
        out_shape = (x.size(0), y.size(1))
        result = torch.zeros(out_shape)

        # Use adaptive chunk size
        chunk_size = min(self._chunk_size, x.size(1))
        
        # Use strided operations for better cache utilization
        for i in range(0, x.size(1), chunk_size):
            end = min(i + chunk_size, x.size(1))
            # Use strided operation if possible
            if x.is_contiguous() and y.is_contiguous():
                result += torch.matmul(x[:, i:end], y[i:end, :])
            else:
                # Handle non-contiguous tensors
                x_chunk = x[:, i:end].contiguous()
                y_chunk = y[i:end, :].contiguous()
                result += torch.matmul(x_chunk, y_chunk)

        self._metrics.append(
            MemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                operation_type="matmul",
            )
        )

        return result

    def defragment_memory(self) -> None:
        """Defragment memory by consolidating blocks and reducing fragmentation."""
        # Only defragment if fragmentation is above threshold
        if self.get_fragmentation_ratio() < 0.3:
            return

        # Get all live tensors
        live_tensors = []
        for ref in self._tensor_refs:
            tensor = ref()
            if tensor is not None:
                try:
                    _ = tensor.shape  # Verify tensor is alive
                    live_tensors.append((tensor, tensor.element_size() * tensor.nelement()))
                except:
                    continue

        # Sort by size to minimize gaps
        live_tensors.sort(key=lambda x: x[1], reverse=True)

        # Clear current allocations
        self._sorted_allocations.clear()
        self._tensor_allocations.clear()
        self._allocated_memory = 0

        # Reallocate tensors in sorted order
        for tensor, size in live_tensors:
            # Try to get contiguous block from pool
            new_tensor = self._allocator.allocate(size)
            if new_tensor is None:
                new_tensor = torch.zeros_like(tensor)

            # Copy data
            new_tensor.copy_(tensor)
            tensor_id = id(new_tensor)
            self._tensor_allocations[tensor_id] = size
            self._allocated_memory += size
            bisect.insort(self._sorted_allocations, size)

        # Force garbage collection
        gc.collect()

    def optimize_memory_layout(self) -> None:
        """Optimize memory layout with improved strategies."""
        current_fragmentation = self.get_fragmentation_ratio()
        
        # Use different strategies based on fragmentation level
        if current_fragmentation > 0.7:
            # Severe fragmentation - full defrag
            self.defragment_memory()
            self._allocator = SizeClassAllocator()  # Reset allocator
        elif current_fragmentation > 0.4:
            # Moderate fragmentation - partial defrag
            self.defragment_memory()
        elif current_fragmentation > 0.2:
            # Light fragmentation - just compact
            total_blocks = sum(len(blocks) for blocks in self._allocator.free_blocks.values())
            if total_blocks > 10:
                self._allocator.clear()

        self._metrics.append(
            MemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                operation_type="optimize"
            )
        )

    def inplace_operation(
        self, tensor: torch.Tensor, operation: Callable[[torch.Tensor], None]
    ) -> None:
        """Perform an in-place operation on a tensor."""
        # Check if tensor is fixed in memory
        tensor_id = id(tensor)
        is_fixed = tensor_id in self._fixed_tensors
        
        # Temporarily unfix if needed
        if is_fixed:
            self.unfix_tensor(tensor)
            
        # Perform operation
        operation(tensor)
        
        # Re-fix if it was fixed
        if is_fixed:
            self.fix_tensor_in_memory(tensor)

        self._metrics.append(
            MemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                operation_type="inplace",
            )
        )

    def __del__(self):
        """Cleanup with improved efficiency."""
        self._fixed_tensors.clear()
        self._cleanup_dead_refs()
        self._tensor_allocations.clear()
        self._tensor_refs.clear()
        self._sorted_allocations.clear()
        self._allocator.clear()
        self._allocated_memory = 0
        gc.collect()
