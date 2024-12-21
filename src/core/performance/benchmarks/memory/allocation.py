"""Memory Allocation Benchmarks.

This module provides benchmarks for memory operations including:
- Allocation patterns
- Transfer speeds
- Cache efficiency
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from ...cpu.memory import CacheOptimizer, MemoryManager


@dataclass
class MemoryBenchmarkResult:
    """Results from a memory benchmark run."""

    operation: str
    data_size: int
    allocation_time: Optional[float]
    transfer_time: Optional[float]
    bandwidth: Optional[float]
    cache_hits: Optional[int]
    cache_misses: Optional[int]
    fragmentation: float


class MemoryBenchmarks:
    """Benchmarks for memory operations."""

    def __init__(
        self, device: str = "cpu", enable_pooling: bool = True, runs_per_test: int = 10
    ):
        self.device = device
        self.runs_per_test = runs_per_test
        self.memory_manager = MemoryManager() if enable_pooling else None
        self.results: List[MemoryBenchmarkResult] = []

    def benchmark_allocation_patterns(
        self,
        sizes: List[int] = [1024, 4096, 16384, 65536],
        dtype: torch.dtype = torch.float32,
    ) -> List[MemoryBenchmarkResult]:
        """Benchmark different allocation patterns."""
        results = []

        for size in sizes:
            allocation_times = []
            fragmentation_values = []

            for _ in range(self.runs_per_test):
                # Measure allocation time
                start = time.perf_counter()
                if self.memory_manager:
                    tensor = self.memory_manager.pool.acquire((size,), dtype)
                else:
                    tensor = torch.empty(size, dtype=dtype, device=self.device)
                end = time.perf_counter()

                allocation_times.append(end - start)

                # Measure fragmentation
                if self.memory_manager:
                    fragmentation = self.memory_manager._calculate_fragmentation()
                else:
                    fragmentation = 0.0
                fragmentation_values.append(fragmentation)

                # Cleanup
                if self.memory_manager:
                    self.memory_manager.release_tensor(tensor)
                del tensor

            result = MemoryBenchmarkResult(
                operation="allocation",
                data_size=size * dtype.itemsize,
                allocation_time=sum(allocation_times) / len(allocation_times),
                transfer_time=None,
                bandwidth=None,
                cache_hits=None,
                cache_misses=None,
                fragmentation=sum(fragmentation_values) / len(fragmentation_values),
            )

            results.append(result)
            self.results.append(result)

        return results

    def benchmark_transfer_speeds(
        self,
        sizes: List[int] = [1024, 4096, 16384, 65536],
        dtype: torch.dtype = torch.float32,
    ) -> List[MemoryBenchmarkResult]:
        """Benchmark memory transfer speeds."""
        results = []

        for size in sizes:
            transfer_times = []
            bandwidths = []

            for _ in range(self.runs_per_test):
                # Create source and destination tensors
                src = torch.randn(size, dtype=dtype, device="cpu")
                start = time.perf_counter()
                dst = src.to(self.device)
                end = time.perf_counter()

                transfer_time = end - start
                transfer_times.append(transfer_time)

                # Calculate bandwidth (bytes/second)
                bytes_transferred = size * dtype.itemsize
                bandwidths.append(bytes_transferred / transfer_time)

                del src, dst

            result = MemoryBenchmarkResult(
                operation="transfer",
                data_size=size * dtype.itemsize,
                allocation_time=None,
                transfer_time=sum(transfer_times) / len(transfer_times),
                bandwidth=sum(bandwidths) / len(bandwidths),
                cache_hits=None,
                cache_misses=None,
                fragmentation=0.0,
            )

            results.append(result)
            self.results.append(result)

        return results

    def benchmark_cache_efficiency(
        self,
        sizes: List[int] = [1024, 4096, 16384, 65536],
        dtype: torch.dtype = torch.float32,
    ) -> List[MemoryBenchmarkResult]:
        """Benchmark cache efficiency."""
        results = []
        cache_optimizer = CacheOptimizer()

        for size in sizes:
            cache_hits = []
            cache_misses = []
            access_times = []

            for _ in range(self.runs_per_test):
                # Create test tensor
                tensor = torch.randn(size, dtype=dtype, device=self.device)

                # Optimize layout
                start = time.perf_counter()
                optimized = cache_optimizer.optimize_layout(tensor)

                # Perform random accesses
                indices = torch.randint(0, size, (1000,))
                for idx in indices:
                    cache_optimizer.prefetch(optimized, idx.unsqueeze(0))
                    _ = optimized[idx]

                end = time.perf_counter()
                access_times.append(end - start)

                cache_hits.append(cache_optimizer.stats["hits"])
                cache_misses.append(
                    0
                )  # Simplified - would need HW counters for real values

                del tensor, optimized

            result = MemoryBenchmarkResult(
                operation="cache",
                data_size=size * dtype.itemsize,
                allocation_time=None,
                transfer_time=sum(access_times) / len(access_times),
                bandwidth=None,
                cache_hits=int(sum(cache_hits) / len(cache_hits)),
                cache_misses=int(sum(cache_misses) / len(cache_misses)),
                fragmentation=0.0,
            )

            results.append(result)
            self.results.append(result)

        return results

    def run_all_benchmarks(self) -> Dict[str, List[MemoryBenchmarkResult]]:
        """Run all memory benchmarks."""
        return {
            "allocation": self.benchmark_allocation_patterns(),
            "transfer": self.benchmark_transfer_speeds(),
            "cache": self.benchmark_cache_efficiency(),
        }

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all benchmarks."""
        summary = {}
        for result in self.results:
            if result.operation not in summary:
                summary[result.operation] = {
                    "avg_data_size": result.data_size,
                    "avg_allocation_time": result.allocation_time or 0.0,
                    "avg_transfer_time": result.transfer_time or 0.0,
                    "avg_bandwidth": result.bandwidth or 0.0,
                    "avg_cache_hits": float(result.cache_hits or 0),
                    "avg_fragmentation": result.fragmentation,
                    "count": 1,
                }
            else:
                stats = summary[result.operation]
                stats["avg_data_size"] = (
                    stats["avg_data_size"] * stats["count"] + result.data_size
                ) / (stats["count"] + 1)
                if result.allocation_time:
                    stats["avg_allocation_time"] = (
                        stats["avg_allocation_time"] * stats["count"]
                        + result.allocation_time
                    ) / (stats["count"] + 1)
                if result.transfer_time:
                    stats["avg_transfer_time"] = (
                        stats["avg_transfer_time"] * stats["count"]
                        + result.transfer_time
                    ) / (stats["count"] + 1)
                if result.bandwidth:
                    stats["avg_bandwidth"] = (
                        stats["avg_bandwidth"] * stats["count"] + result.bandwidth
                    ) / (stats["count"] + 1)
                if result.cache_hits:
                    stats["avg_cache_hits"] = (
                        stats["avg_cache_hits"] * stats["count"] + float(result.cache_hits)
                    ) / (stats["count"] + 1)
                stats["avg_fragmentation"] = (
                    stats["avg_fragmentation"] * stats["count"] + result.fragmentation
                ) / (stats["count"] + 1)
                stats["count"] += 1
        return summary
