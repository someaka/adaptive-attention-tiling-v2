"""Core operation benchmarks for the Adaptive Attention Tiling system.

This module provides comprehensive benchmarks for core operations including:
1. Attention computation performance
2. Pattern formation and evolution
3. Memory allocation and transfer patterns
4. Scaling characteristics
"""

import pytest
import torch

from src.core.attention import AttentionCompute
from src.core.benchmarks import BenchmarkMetrics
from src.core.flow import FlowComputation
from src.core.patterns import PatternEvolution


class TestCoreOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.sizes = [512, 1024, 2048, 4096]
        self.batch_sizes = [1, 8, 16, 32]
        self.iterations = 10
        self.metrics = BenchmarkMetrics()

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
                    start_mem = torch.cuda.memory_allocated()
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    attention(query, key, value)
                    end_time.record()

                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
                    memory_usage.append(torch.cuda.memory_allocated() - start_mem)

                # Record metrics
                self.metrics.record_operation(
                    name="attention_computation",
                    size=size,
                    batch_size=batch_size,
                    avg_time=sum(times) / len(times),
                    avg_memory=sum(memory_usage) / len(memory_usage),
                    throughput=size * batch_size / (sum(times) / len(times)),
                )

    def test_pattern_formation(self):
        """Benchmark pattern formation and evolution efficiency."""
        pattern = PatternEvolution()

        for size in self.sizes:
            # Initialize pattern state
            state = torch.randn(size, size)

            # Warm-up
            _ = pattern.evolve(state)

            # Benchmark evolution steps
            times = []
            stability = []
            for _step in range(self.iterations):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                new_state = pattern.evolve(state)
                end_time.record()

                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
                stability.append(torch.norm(new_state - state).item())
                state = new_state

            self.metrics.record_operation(
                name="pattern_formation",
                size=size,
                avg_time=sum(times) / len(times),
                stability=sum(stability) / len(stability),
                convergence_rate=stability[-1] / stability[0],
            )

    def test_flow_evolution(self):
        """Benchmark flow computation and evolution performance."""
        flow = FlowComputation()

        for size in self.sizes:
            # Initialize flow field
            velocity = torch.randn(2, size, size)  # 2D velocity field
            density = torch.randn(size, size)

            # Warm-up
            _ = flow.compute(velocity, density)

            # Benchmark flow computation
            times = []
            accuracy = []
            for _ in range(self.iterations):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                result = flow.compute(velocity, density)
                end_time.record()

                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
                accuracy.append(flow.validate_solution(result))

            self.metrics.record_operation(
                name="flow_evolution",
                size=size,
                avg_time=sum(times) / len(times),
                accuracy=sum(accuracy) / len(accuracy),
            )

    def test_memory_patterns(self):
        """Analyze memory allocation patterns and efficiency."""
        for size in self.sizes:
            # Test different allocation patterns
            allocation_sizes = [size // 4, size // 2, size]

            for alloc_size in allocation_sizes:
                # Sequential allocation
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                tensors = []
                for _ in range(4):
                    tensors.append(torch.randn(alloc_size, alloc_size))
                end_time.record()

                torch.cuda.synchronize()
                seq_time = start_time.elapsed_time(end_time)

                # Batch allocation
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                torch.randn(4, alloc_size, alloc_size)
                end_time.record()

                torch.cuda.synchronize()
                batch_time = start_time.elapsed_time(end_time)

                self.metrics.record_operation(
                    name="memory_allocation",
                    size=alloc_size,
                    sequential_time=seq_time,
                    batch_time=batch_time,
                    efficiency=batch_time / seq_time,
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
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            _ = attention(query, key, value)
            end_time.record()

            torch.cuda.synchronize()
            strong_time = start_time.elapsed_time(end_time)

            self.metrics.record_operation(
                name="strong_scaling",
                batch_size=batch_size,
                size=size,
                time=strong_time,
                efficiency=strong_time / batch_size,
            )

        # Weak scaling (fixed size per batch, varying total)
        size_per_batch = 1024
        for batch_size in self.batch_sizes:
            query = torch.randn(batch_size, size_per_batch, 64)
            key = torch.randn(batch_size, size_per_batch, 64)
            value = torch.randn(batch_size, size_per_batch, 64)

            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            _ = attention(query, key, value)
            end_time.record()

            torch.cuda.synchronize()
            weak_time = start_time.elapsed_time(end_time)

            self.metrics.record_operation(
                name="weak_scaling",
                batch_size=batch_size,
                size=size_per_batch,
                time=weak_time,
                efficiency=weak_time / (batch_size * size_per_batch),
            )
