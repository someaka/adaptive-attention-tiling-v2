"""CPU Optimization Module for Adaptive Attention Tiling.

This module provides CPU-specific optimizations including:
1. Vectorized operations for core algorithms
2. Memory access pattern optimization
3. Memory allocation reduction
4. Critical path profiling
"""

import cProfile
import time
import psutil
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, List, Optional, Callable

import line_profiler
import memory_profiler
import torch
from torch.utils.benchmark import Timer


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    execution_time: float
    memory_usage: float
    cpu_utilization: float
    cache_hits: float
    vectorization_efficiency: float

    def __post_init__(self):
        """Validate metric ranges."""
        if not 0 <= self.cpu_utilization <= 100:
            raise ValueError("CPU utilization must be between 0 and 100")
        if not 0 <= self.cache_hits <= 1:
            raise ValueError("Cache hits must be between 0 and 1")
        if not 0 <= self.vectorization_efficiency <= 1:
            raise ValueError("Vectorization efficiency must be between 0 and 1")


class CPUOptimizer:
    """CPU optimization framework for performance enhancement."""

    def __init__(
        self, enable_profiling: bool = True, enable_memory_tracking: bool = True
    ):
        """Initialize the CPU optimizer.

        Args:
            enable_profiling: Whether to enable CPU profiling
            enable_memory_tracking: Whether to track memory usage
        """
        self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking
        self._setup_profilers()

    def _setup_profilers(self):
        """Setup profiling tools."""
        self.cpu_profiler = cProfile.Profile()
        self.line_profiler = line_profiler.LineProfiler()
        self.memory_profiler = memory_profiler.profile

    def profile_execution(self, func: Callable) -> Callable:
        """Decorator for profiling function execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_profiling:
                return func(*args, **kwargs)

            # Start profiling
            start_time = time.perf_counter()
            self.cpu_profiler.enable()

            # Execute function
            result = func(*args, **kwargs)

            # Stop profiling
            self.cpu_profiler.disable()
            end_time = time.perf_counter()

            # Collect metrics
            stats = self.cpu_profiler.getstats()
            execution_time = end_time - start_time

            # Log results
            self._log_profile_results(func.__name__, execution_time, stats)

            return result

        return wrapper

    def optimize_memory_access(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize memory access patterns for tensor operations.

        Args:
            tensor: Input tensor to optimize

        Returns:
            Optimized tensor with improved memory layout
        """
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Align tensor for vectorization
        if tensor.stride()[-1] != 1:
            tensor = tensor.transpose(-2, -1)

        return tensor

    def vectorize_operation(
        self,
        func: Callable[..., torch.Tensor],
        inputs: List[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Vectorize operations using torch.vmap when possible.

        Args:
            func: Function to vectorize
            inputs: List of input tensors
            chunk_size: Optional chunk size for batched processing

        Returns:
            Result of vectorized operation
        """
        # Optimize memory layout
        inputs = [self.optimize_memory_access(x) for x in inputs]

        # Use vmap for vectorization
        vectorized_func = torch.vmap(func)

        # Process in chunks if specified
        if chunk_size is not None:
            results = []
            for i in range(0, inputs[0].size(0), chunk_size):
                chunk_inputs = [x[i : i + chunk_size] for x in inputs]
                chunk_result = vectorized_func(*chunk_inputs)
                results.append(chunk_result)
            return torch.cat(results, dim=0)

        return vectorized_func(*inputs)

    def optimize_computation(
        self, computation_graph: torch.nn.Module, sample_input: torch.Tensor
    ) -> torch.nn.Module:
        """Optimize a computation graph for CPU execution.

        Args:
            computation_graph: Module to optimize
            sample_input: Sample input for optimization

        Returns:
            Optimized computation graph
        """
        # Ensure graph is in eval mode
        computation_graph.eval()

        # Optimize memory layout
        sample_input = self.optimize_memory_access(sample_input)

        # Script the module for optimization
        try:
            optimized_graph = torch.jit.script(computation_graph)
            # Warmup run
            optimized_graph(sample_input)
        except Exception as e:
            print(f"Warning: JIT optimization failed: {e}")
            optimized_graph = computation_graph

        return optimized_graph

    def _log_profile_results(self, func_name: str, execution_time: float, stats: Any):
        """Log profiling results.

        Args:
            func_name: Name of profiled function
            execution_time: Total execution time
            stats: Profiling statistics
        """
        print(f"\nProfile results for {func_name}:")
        print(f"Total execution time: {execution_time:.4f} seconds")

        # Log detailed stats if available
        if stats:
            total_calls = sum(stat.callcount for stat in stats)
            print(f"Total function calls: {total_calls}")

            # Sort by internal time
            sorted_stats = sorted(stats, key=lambda x: x.inlinetime, reverse=True)[:5]

            print("\nTop 5 time-consuming functions:")
            for stat in sorted_stats:
                print(f"{stat.code.co_name}: {stat.inlinetime:.4f} seconds")

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.

        Returns:
            PerformanceMetrics object with current metrics
        """
        # Get CPU utilization using psutil
        cpu_percent = psutil.Process(os.getpid()).cpu_percent(interval=0.1)
        
        # Estimate cache performance using a simple benchmark
        cache_hits = self._measure_cache_performance()
        
        # Measure vectorization efficiency
        vec_efficiency = self._measure_vectorization_efficiency()
        
        # Collect current metrics
        metrics = PerformanceMetrics(
            execution_time=time.perf_counter(),
            memory_usage=memory_profiler.memory_usage()[0],
            cpu_utilization=cpu_percent,
            cache_hits=cache_hits,
            vectorization_efficiency=vec_efficiency
        )

        return metrics
        
    def _measure_cache_performance(self) -> float:
        """Measure cache performance using memory access patterns.
        
        Returns:
            float: Ratio of cache hits (0-1)
        """
        # Create two test tensors
        size = 1000000
        x = torch.randn(size)
        y = torch.randn(size)
        
        # Measure sequential access time
        t0 = time.perf_counter()
        torch.dot(x, y)
        sequential_time = time.perf_counter() - t0
        
        # Measure random access time (worst case)
        indices = torch.randperm(size)
        t0 = time.perf_counter()
        torch.dot(x[indices], y[indices])
        random_time = time.perf_counter() - t0
        
        # Compute cache hit ratio estimate
        # If random_time is much larger than sequential_time, it indicates poor cache utilization
        cache_ratio = sequential_time / max(random_time, sequential_time)
        return float(cache_ratio)  # Closer to 1 means better cache utilization
        
    def _measure_vectorization_efficiency(self) -> float:
        """Measure vectorization efficiency using torch.vmap.
        
        Returns:
            float: Vectorization efficiency ratio (0-1)
        """
        # Create test data
        batch_size = 1000
        feature_size = 100
        x = torch.randn(batch_size, feature_size)
        
        # Define test function
        def scalar_fn(x):
            return torch.sum(torch.sin(x))
        
        # Measure scalar version time
        t0 = time.perf_counter()
        for i in range(len(x)):
            scalar_fn(x[i])
        scalar_time = time.perf_counter() - t0
        
        # Measure vectorized version time
        vectorized_fn = torch.vmap(scalar_fn)
        t0 = time.perf_counter()
        vectorized_fn(x)
        vector_time = time.perf_counter() - t0
        
        # Compute efficiency ratio
        # If vector_time is much smaller than scalar_time, vectorization is efficient
        efficiency = scalar_time / max(vector_time * batch_size, scalar_time)
        return float(efficiency)  # Closer to 1 means better vectorization
