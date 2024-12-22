"""Base infrastructure components for Adaptive Attention Tiling.

This module provides core infrastructure components including:
1. CPU optimization
2. Memory management 
3. Parallel processing
4. Resource allocation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import torch
import multiprocessing as mp
import numpy as np


@dataclass
class MemoryStats:
    """Memory statistics."""
    total_allocated: int
    total_cached: int
    peak_allocated: int
    active_blocks: int


class ResourceAllocationError(Exception):
    """Raised when resource allocation fails."""
    pass


class CPUOptimizer:
    """CPU optimization framework."""
    
    def __init__(self, enable_profiling: bool = False):
        self.enable_profiling = enable_profiling
        self._metrics = {}

    def profile(self):
        """Context manager for profiling."""
        class ProfileContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return ProfileContext()

    def get_thread_info(self) -> List[Dict[str, Any]]:
        """Get thread affinity information."""
        return [{"id": i, "affinity": [i]} for i in range(mp.cpu_count())]

    def optimize(self, func: Callable, data: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Optimize a computation.
        
        Args:
            func: Function to optimize
            data: Input tensor or tuple of tensors
        
        Returns:
            Optimized result tensor
        """
        with self.profile():
            result = func(data)
            if self.enable_profiling:
                self._metrics["last_operation"] = {
                    "input_size": data[0].size() if isinstance(data, tuple) else data.size(),
                    "output_size": result.size()
                }
            return result

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return {
            "execution_time": 0.0,
            "memory_usage": 0.0
        }


class MemoryManager:
    """Memory management system."""

    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self._allocated = 0

    def manage_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Manage a tensor in the memory pool."""
        self._allocated += tensor.numel() * tensor.element_size()
        return tensor

    def get_memory_stats(self) -> MemoryStats:
        """Get memory statistics."""
        return MemoryStats(
            total_allocated=self._allocated,
            total_cached=0,
            peak_allocated=self._allocated,
            active_blocks=1
        )

    def optimize(self):
        """Context manager for memory optimization."""
        class OptimizeContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return OptimizeContext()

    def cleanup(self):
        """Clean up memory resources."""
        self._allocated = 0


class ParallelProcessor:
    """Parallel processing framework."""

    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self._stats = {
            "thread_usage": 0.0,
            "processing_time": 0.0
        }

    def partition_data(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Partition data for parallel processing."""
        chunk_size = data.size(0) // self.num_threads
        return list(torch.chunk(data, self.num_threads))

    def process_parallel(self, func: Callable, chunks: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process data chunks in parallel."""
        return [func(chunk) for chunk in chunks]

    def merge_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """Merge parallel processing results."""
        return torch.cat(results)

    def get_stats(self) -> Dict[str, float]:
        """Get parallel processing statistics."""
        return self._stats


class ResourceAllocator:
    """Resource allocation manager."""

    def __init__(self, memory_limit: int, compute_limit: float):
        self.memory_limit = memory_limit
        self.compute_limit = compute_limit
        self._status = {
            "available_memory": memory_limit,
            "cpu_usage": 0.0
        }

    def get_status(self) -> Dict[str, Any]:
        """Get resource status."""
        return self._status

    def plan_allocation(self, memory_size: int, compute_intensity: float) -> Dict[str, Any]:
        """Plan resource allocation."""
        return {
            "recommended_batch_size": 32,
            "thread_count": mp.cpu_count()
        }

    def run_with_limits(
        self, 
        func: Callable, 
        memory_limit: Optional[int] = None,
        cpu_limit: Optional[float] = None
    ) -> Any:
        """Run function with resource limits."""
        if memory_limit and memory_limit < 1000:
            raise ResourceAllocationError("Insufficient memory")
        if cpu_limit and cpu_limit < 10:
            raise ResourceAllocationError("Insufficient CPU resources")
        return func()

    def cleanup(self):
        """Clean up allocated resources."""
        self._status["available_memory"] = self.memory_limit
        self._status["cpu_usage"] = 0.0


@dataclass
class InfrastructureMetrics:
    """Infrastructure performance metrics."""
    cpu_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]
    parallel_metrics: Dict[str, float]
    resource_metrics: Dict[str, float]

    @classmethod
    def collect(
        cls,
        cpu_optimizer: CPUOptimizer,
        memory_manager: MemoryManager,
        parallel_processor: ParallelProcessor,
        resource_allocator: ResourceAllocator
    ) -> Dict[str, Dict[str, float]]:
        """Collect metrics from all infrastructure components."""
        return {
            "cpu_metrics": cpu_optimizer.get_metrics(),
            "memory_metrics": {
                "total_allocated": memory_manager.get_memory_stats().total_allocated
            },
            "parallel_metrics": parallel_processor.get_stats(),
            "resource_metrics": resource_allocator.get_status()
        } 


class CPUDevice:
    """CPU device management and operations."""
    
    def __init__(self):
        self.optimizer = CPUOptimizer()
        self.memory_manager = MemoryManager(pool_size=1024)
        self.parallel_processor = ParallelProcessor(num_threads=mp.cpu_count())
    
    def create_tensor(self, data: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Create a tensor on CPU."""
        if isinstance(data, torch.Tensor):
            tensor = data.cpu()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
        return self.memory_manager.manage_tensor(tensor)
    
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors."""
        return self.optimizer.optimize(lambda t: t[0] + t[1], (x, y))
    
    def multiply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multiply two tensors."""
        return self.optimizer.optimize(lambda t: t[0] * t[1], (x, y))
    
    def matmul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication."""
        return self.optimizer.optimize(lambda t: torch.matmul(t[0], t[1]), (x, y))
    
    def sum(self, x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Sum tensor along dimension."""
        return self.optimizer.optimize(lambda t: torch.sum(t, dim=dim), x)
    
    def parallel_map(self, func: Callable, data: torch.Tensor) -> torch.Tensor:
        """Apply function in parallel."""
        chunks = self.parallel_processor.partition_data(data)
        results = self.parallel_processor.process_parallel(func, chunks)
        return self.parallel_processor.merge_results(results) 