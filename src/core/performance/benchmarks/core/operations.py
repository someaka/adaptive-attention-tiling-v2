"""Core Operation Benchmarks.

This module provides benchmarks for core operations including:
- Attention computation
- Pattern formation
- Flow evolution
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from ...cpu.algorithms import AlgorithmOptimizer
from ...cpu.memory import MemoryManager
from ...cpu.vectorization import VectorizationOptimizer


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    input_size: Tuple[int, ...]
    execution_time: float
    memory_usage: float
    throughput: float
    device: str
    optimization_level: str


class CoreBenchmarks:
    """Benchmarks for core operations."""

    def __init__(
        self,
        device: str = "cpu",
        enable_optimizations: bool = True,
        runs_per_test: int = 10,
    ):
        self.device = device
        self.runs_per_test = runs_per_test

        # Initialize optimizers if enabled
        if enable_optimizations:
            self.vec_opt = VectorizationOptimizer()
            self.mem_opt = MemoryManager()
            self.algo_opt = AlgorithmOptimizer()
        else:
            self.vec_opt = None
            self.mem_opt = None
            self.algo_opt = None

        self.results: List[BenchmarkResult] = []

    def benchmark_attention(
        self, batch_size: int = 32, seq_length: int = 512, hidden_size: int = 768
    ) -> BenchmarkResult:
        """Benchmark attention computation."""
        # Create test data
        query = torch.randn(batch_size, seq_length, hidden_size, device=self.device)
        key = torch.randn(batch_size, seq_length, hidden_size, device=self.device)
        value = torch.randn(batch_size, seq_length, hidden_size, device=self.device)

        def run_attention():
            if self.vec_opt:
                return self.vec_opt.vectorize_attention(query, key, value)
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores / np.sqrt(hidden_size)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, value)

        # Run benchmark
        times = []
        memory_usage = []
        for _ in range(self.runs_per_test):
            start = time.perf_counter()
            _ = run_attention()
            end = time.perf_counter()
            times.append(end - start)
            memory_usage.append(
                torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
            )

        # Calculate metrics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        throughput = (batch_size * seq_length * hidden_size) / avg_time

        result = BenchmarkResult(
            operation="attention",
            input_size=(batch_size, seq_length, hidden_size),
            execution_time=avg_time,
            memory_usage=avg_memory,
            throughput=throughput,
            device=self.device,
            optimization_level="optimized" if self.vec_opt else "baseline",
        )

        self.results.append(result)
        return result

    def benchmark_pattern_formation(
        self, batch_size: int = 32, pattern_size: int = 64
    ) -> BenchmarkResult:
        """Benchmark pattern formation."""
        pattern = torch.randn(
            batch_size, pattern_size, pattern_size, device=self.device
        )

        def run_pattern_formation():
            if self.vec_opt:
                flow = torch.randn_like(pattern)
                return self.vec_opt.vectorize_pattern_dynamics(pattern, flow)
            return torch.tanh(pattern)  # Simple baseline

        # Run benchmark
        times = []
        memory_usage = []
        for _ in range(self.runs_per_test):
            start = time.perf_counter()
            _ = run_pattern_formation()
            end = time.perf_counter()
            times.append(end - start)
            memory_usage.append(
                torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
            )

        # Calculate metrics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        throughput = (batch_size * pattern_size * pattern_size) / avg_time

        result = BenchmarkResult(
            operation="pattern_formation",
            input_size=(batch_size, pattern_size, pattern_size),
            execution_time=avg_time,
            memory_usage=avg_memory,
            throughput=throughput,
            device=self.device,
            optimization_level="optimized" if self.vec_opt else "baseline",
        )

        self.results.append(result)
        return result

    def benchmark_flow_evolution(
        self, batch_size: int = 32, manifold_dim: int = 16
    ) -> BenchmarkResult:
        """Benchmark geometric flow evolution."""
        metric = torch.randn(batch_size, manifold_dim, manifold_dim, device=self.device)
        connection = torch.randn(
            batch_size, manifold_dim, manifold_dim, device=self.device
        )

        def run_flow_evolution():
            if self.vec_opt:
                return self.vec_opt.vectorize_geometric_flow(metric, connection)
            return -torch.matmul(metric, connection)  # Simple baseline

        # Run benchmark
        times = []
        memory_usage = []
        for _ in range(self.runs_per_test):
            start = time.perf_counter()
            _ = run_flow_evolution()
            end = time.perf_counter()
            times.append(end - start)
            memory_usage.append(
                torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
            )

        # Calculate metrics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        throughput = (batch_size * manifold_dim * manifold_dim) / avg_time

        result = BenchmarkResult(
            operation="flow_evolution",
            input_size=(batch_size, manifold_dim, manifold_dim),
            execution_time=avg_time,
            memory_usage=avg_memory,
            throughput=throughput,
            device=self.device,
            optimization_level="optimized" if self.vec_opt else "baseline",
        )

        self.results.append(result)
        return result

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks with default parameters."""
        results = []
        results.append(self.benchmark_attention())
        results.append(self.benchmark_pattern_formation())
        results.append(self.benchmark_flow_evolution())
        return results

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all benchmarks."""
        summary = {}
        for result in self.results:
            if result.operation not in summary:
                summary[result.operation] = {
                    "avg_time": result.execution_time,
                    "avg_memory": result.memory_usage,
                    "avg_throughput": result.throughput,
                    "count": 1,
                }
            else:
                stats = summary[result.operation]
                stats["avg_time"] = (
                    stats["avg_time"] * stats["count"] + result.execution_time
                ) / (stats["count"] + 1)
                stats["avg_memory"] = (
                    stats["avg_memory"] * stats["count"] + result.memory_usage
                ) / (stats["count"] + 1)
                stats["avg_throughput"] = (
                    stats["avg_throughput"] * stats["count"] + result.throughput
                ) / (stats["count"] + 1)
                stats["count"] += 1
        return summary
