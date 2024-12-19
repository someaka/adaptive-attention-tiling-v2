"""Core operation benchmarks for the Adaptive Attention Tiling system.

This module provides comprehensive benchmarks for core operations including:
1. Attention computation performance
2. Pattern formation and evolution
3. Memory allocation and transfer patterns
4. Scaling characteristics
"""

import pytest
import torch
import time
from typing import Tuple, cast

from src.core.attention import AttentionCompute
from src.core.benchmarks import BenchmarkMetrics
from src.core.flow import PatternFormationFlow
from src.core.patterns import PatternEvolution
from src.core.patterns.riemannian import PatternRiemannianStructure
from src.core.patterns import (
    BaseRiemannianStructure,
    RiemannianFramework,
    PatternRiemannianStructure,
    MetricTensor,
    ChristoffelSymbols,
    CurvatureTensor,
)


class TestCoreOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.sizes = [512, 1024, 2048, 4096]
        self.batch_sizes = [1, 8, 16, 32]
        self.iterations = 10
        self.metrics = BenchmarkMetrics()
        self.riemannian_framework = PatternRiemannianStructure(
            manifold_dim=64,
            pattern_dim=64  # Setting pattern_dim equal to manifold_dim for testing
        )

    def test_attention_computation(self):
        """Benchmark attention computation performance."""
        attention = AttentionCompute()

        for size in self.sizes:
            for batch_size in self.batch_sizes:
                # Generate test data
                query = torch.randn(batch_size, size, 64)
                key = torch.randn(batch_size, size, 64)
                value = torch.randn(batch_size, size, 64)

                # Warm-up run
                _ = attention(query, key, value)

                # Benchmark runs
                times = []
                memory_usage = []
                for _ in range(self.iterations):
                    start_mem = torch.zeros(1).element_size() * torch.zeros(1).numel()
                    start_time = time.perf_counter()
                    attention(query, key, value)
                    end_time = time.perf_counter()

                    times.append((end_time - start_time) * 1000)  # Convert to ms
                    memory_usage.append(
                        torch.zeros(1).element_size() * torch.zeros(1).numel() - start_mem
                    )

                # Record metrics
                self.metrics.record_operation(
                    name="attention_computation",
                    size=size,
                    batch_size=batch_size,
                    avg_time=sum(times) / len(times),
                    avg_memory=sum(memory_usage) / len(memory_usage),
                    throughput=size * batch_size / (sum(times) / len(times))
                )

    def test_pattern_formation(self):
        """Benchmark pattern formation and evolution efficiency."""
        pattern = PatternEvolution(framework=self.riemannian_framework)

        for size in self.sizes:
            # Initialize pattern state
            state = torch.randn(size, size)

            # Warm-up
            result = pattern.step(state, torch.randn_like(state), return_metrics=False)
            assert isinstance(result, tuple) and len(result) == 2

            # Benchmark evolution steps
            times = []
            stability = []
            for _step in range(self.iterations):
                start_time = time.perf_counter()
                # Use type assertion to ensure we get a 2-tuple
                step_result = cast(Tuple[torch.Tensor, torch.Tensor], 
                    pattern.step(state, torch.randn_like(state), return_metrics=False)
                )
                new_state, velocity = step_result
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)  # Convert to ms
                stability.append(torch.norm(new_state - state).item())
                state = new_state

            self.metrics.record_operation(
                name="pattern_formation",
                size=size,
                avg_time=sum(times) / len(times),
                stability=sum(stability) / len(stability),
                convergence_rate=stability[-1] / stability[0]
            )

    def test_flow_evolution(self):
        """Benchmark flow computation and evolution performance."""
        flow = PatternFormationFlow(
            manifold_dim=2,  # 2D flow
            hidden_dim=64,
            diffusion_strength=0.1,
            reaction_strength=1.0
        )

        for size in self.sizes:
            # Initialize flow field
            points = torch.randn(size, 2)  # 2D points
            metric = flow.compute_metric(points)

            # Warm-up
            _ = flow.flow_step(metric, timestep=0.1)

            # Benchmark flow computation
            times = []
            accuracy = []
            for _ in range(self.iterations):
                start_time = time.perf_counter()
                new_metric, metrics = flow.flow_step(metric, timestep=0.1)
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)  # Convert to ms
                # Compute accuracy as convergence of flow
                accuracy.append(metrics.flow_magnitude)
                metric = new_metric

            self.metrics.record_operation(
                name="flow_evolution",
                size=size,
                avg_time=sum(times) / len(times),
                accuracy=sum(accuracy) / len(accuracy)
            )

    def test_memory_patterns(self):
        """Analyze memory allocation patterns and efficiency."""
        for size in self.sizes:
            # Test different allocation patterns
            allocation_sizes = [size // 4, size // 2, size]

            for alloc_size in allocation_sizes:
                # Sequential allocation
                start_time = time.perf_counter()
                tensors = []
                for _ in range(4):
                    tensors.append(torch.randn(alloc_size, alloc_size))
                end_time = time.perf_counter()
                seq_time = (end_time - start_time) * 1000  # Convert to ms

                # Batch allocation
                start_time = time.perf_counter()
                torch.randn(4, alloc_size, alloc_size)
                end_time = time.perf_counter()
                batch_time = (end_time - start_time) * 1000  # Convert to ms

                self.metrics.record_operation(
                    name="memory_allocation",
                    size=alloc_size,
                    sequential_time=seq_time,
                    batch_time=batch_time,
                    efficiency=batch_time / seq_time
                )

    def test_scaling_characteristics(self):
        """Test scaling behavior with different problem sizes."""
        attention = AttentionCompute()

        # Strong scaling (fixed total size, varying batch)
        total_size = 8192
        for batch_size in self.batch_sizes:
            size = total_size // batch_size
            query = torch.randn(batch_size, size, 64)
            key = torch.randn(batch_size, size, 64)
            value = torch.randn(batch_size, size, 64)

            # Warm-up
            _ = attention(query, key, value)

            # Benchmark
            start_time = time.perf_counter()
            _ = attention(query, key, value)
            end_time = time.perf_counter()
            strong_time = (end_time - start_time) * 1000  # Convert to ms

            self.metrics.record_operation(
                name="strong_scaling",
                batch_size=batch_size,
                size=size,
                time=strong_time,
                efficiency=strong_time / batch_size
            )

        # Weak scaling (fixed size per batch, varying total)
        size_per_batch = 1024
        for batch_size in self.batch_sizes:
            query = torch.randn(batch_size, size_per_batch, 64)
            key = torch.randn(batch_size, size_per_batch, 64)
            value = torch.randn(batch_size, size_per_batch, 64)

            start_time = time.perf_counter()
            _ = attention(query, key, value)
            end_time = time.perf_counter()
            weak_time = (end_time - start_time) * 1000  # Convert to ms

            self.metrics.record_operation(
                name="weak_scaling",
                batch_size=batch_size,
                size=size_per_batch,
                time=weak_time,
                efficiency=weak_time / (batch_size * size_per_batch)
            )
