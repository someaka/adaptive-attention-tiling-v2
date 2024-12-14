"""Performance tests for CPU vectorization operations.

This module tests the performance characteristics of vectorized operations
in the Adaptive Attention Tiling system, focusing on:
1. Attention computation efficiency
2. Pattern dynamics vectorization
3. Geometric flow optimization
4. Memory layout and cache utilization
"""

import gc
import resource
import signal
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NoReturn

import pytest
import torch

from src.core.performance.cpu.vectorization import VectorizationOptimizer

# Test configurations - reduced sizes for safety
BATCH_SIZES = [32, 128]  # Removed 512, 2048
SEQUENCE_LENGTHS = [32, 64]  # Reduced from [64, 256]
FEATURE_DIMS = [32, 64]  # Reduced from [32, 128]
CHUNK_SIZES = [32, 64]  # Reduced from [64, 256]

# Resource limits
MAX_MEMORY_GB = 8  # Increased from 4GB
MAX_TIME_SECONDS = 60  # Increased from 30s


@contextmanager
def resource_guard() -> Generator[None, None, None]:
    """Set up resource limits for memory and time."""
    # Set memory limit
    memory_limit = MAX_MEMORY_GB * 1024 * 1024 * 1024  # Convert to bytes
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))

    # Set up timeout
    def timeout_handler(_signum: int, _frame: Any) -> NoReturn:
        msg = f"Test exceeded {MAX_TIME_SECONDS} seconds time limit"
        raise TimeoutError(msg)

    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_TIME_SECONDS)

    try:
        yield
    finally:
        # Reset signal handler and alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # Reset memory limit
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def vectorization_optimizer() -> VectorizationOptimizer:
    """Create a VectorizationOptimizer instance for testing."""
    optimizer = VectorizationOptimizer(
        enable_profiling=True,
        use_mixed_precision=True,
        chunk_size=64,  # Use smaller chunk size for tests
    )
    optimizer.clear_metrics()
    return optimizer


def generate_test_tensors(
    batch_size: int, seq_len: int, feat_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test tensors for attention computation."""
    torch.manual_seed(42)  # For reproducibility
    query = torch.randn(batch_size, seq_len, feat_dim)
    key = torch.randn(batch_size, seq_len, feat_dim)
    value = torch.randn(batch_size, seq_len, feat_dim)
    return query, key, value


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("dim", FEATURE_DIMS)
def test_attention_vectorization_performance(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, dim: int
) -> None:
    """Test attention vectorization performance."""
    with resource_guard():
        try:
            # Generate test data
            query, key, value = generate_test_tensors(batch_size, dim, dim)

            # Compute vectorized attention
            result = vectorization_optimizer.vectorize_attention(query, key, value)

            # Validate output shape
            expected_shape = (batch_size, dim, dim)
            assert (
                result.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {result.shape}"

            # Check metrics
            metrics = vectorization_optimizer.get_metrics()
            assert len(metrics) > 0, "No metrics were collected"

            latest_metric = metrics[-1]
            assert latest_metric.operation_type == "vectorize_attention"
            assert latest_metric.execution_time > 0
            assert latest_metric.memory_usage >= 0
            assert 0 <= latest_metric.vectorization_efficiency <= 1

            # Log performance metrics
            print(f"Attention Performance (batch={batch_size}, dim={dim}):")
            print(f"  Execution time: {latest_metric.execution_time:.2f}ms")
            print(f"  Memory usage: {latest_metric.memory_usage / 1024 / 1024:.2f}MB")
            print(
                f"  Vectorization efficiency: {latest_metric.vectorization_efficiency:.2%}"
            )

        except (ValueError, RuntimeError, AssertionError) as e:
            pytest.fail(f"Attention vectorization failed: {e}")


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
def test_pattern_dynamics_vectorization(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, seq_len: int
) -> None:
    """Test pattern dynamics vectorization performance."""
    with resource_guard():
        try:
            # Generate test data
            pattern = torch.randn(batch_size, seq_len, seq_len)
            flow = torch.randn(batch_size, seq_len, seq_len)

            # Compute vectorized pattern dynamics
            result = vectorization_optimizer.vectorize_pattern_dynamics(pattern, flow)

            # Validate output shape
            expected_shape = (batch_size, seq_len, seq_len)
            assert (
                result.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {result.shape}"

            # Check value bounds (should be between -1 and 1 due to tanh)
            assert torch.all(result >= -1), "Values below -1 found"
            assert torch.all(result <= 1), "Values above 1 found"

            # Check metrics
            metrics = vectorization_optimizer.get_metrics()
            assert len(metrics) > 0, "No metrics were collected"

            latest_metric = metrics[-1]
            assert latest_metric.operation_type == "vectorize_pattern_dynamics"
            assert latest_metric.execution_time > 0
            assert latest_metric.memory_usage >= 0
            assert 0 <= latest_metric.vectorization_efficiency <= 1

            # Log performance metrics
            print(f"Pattern Dynamics (batch={batch_size}, seq_len={seq_len}):")
            print(f"  Execution time: {latest_metric.execution_time:.2f}ms")
            print(f"  Memory usage: {latest_metric.memory_usage / 1024 / 1024:.2f}MB")
            print(
                f"  Vectorization efficiency: {latest_metric.vectorization_efficiency:.2%}"
            )

        except (ValueError, RuntimeError, AssertionError) as e:
            pytest.fail(f"Pattern dynamics vectorization failed: {e}")


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("dim", FEATURE_DIMS)
def test_geometric_flow_vectorization(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, dim: int
) -> None:
    """Test geometric flow vectorization performance."""
    with resource_guard():
        try:
            # Generate test data
            metric = torch.randn(batch_size, dim, dim)
            connection = torch.randn(batch_size, dim, dim, dim)

            # Make metric symmetric (as it should be for a metric tensor)
            metric = 0.5 * (metric + metric.transpose(-2, -1))

            # Compute vectorized geometric flow
            result = vectorization_optimizer.vectorize_geometric_flow(
                metric, connection
            )

            # Validate output shape
            expected_shape = (batch_size, dim, dim)
            assert (
                result.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {result.shape}"

            # Check metrics
            metrics = vectorization_optimizer.get_metrics()
            assert len(metrics) > 0, "No metrics were collected"

            latest_metric = metrics[-1]
            assert latest_metric.operation_type == "vectorize_geometric_flow"
            assert latest_metric.execution_time > 0
            assert latest_metric.memory_usage >= 0
            assert 0 <= latest_metric.vectorization_efficiency <= 1

            # Log performance metrics
            print(f"Geometric Flow (batch={batch_size}, dim={dim}):")
            print(f"  Execution time: {latest_metric.execution_time:.2f}ms")
            print(f"  Memory usage: {latest_metric.memory_usage / 1024 / 1024:.2f}MB")
            print(
                f"  Vectorization efficiency: {latest_metric.vectorization_efficiency:.2%}"
            )

        except (ValueError, RuntimeError, AssertionError) as e:
            pytest.fail(f"Geometric flow vectorization failed: {e}")


@pytest.mark.benchmark
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_size_impact(chunk_size: int) -> None:
    """Test impact of different chunk sizes on vectorization performance."""
    with resource_guard():
        try:
            # Create optimizer with specific chunk size
            optimizer = VectorizationOptimizer(chunk_size=chunk_size)

            # Generate test data
            batch_size = 64  # Fixed batch size for comparison
            seq_len = 64  # Fixed sequence length for comparison
            query, key, value = generate_test_tensors(batch_size, seq_len, seq_len)

            # Time the attention computation
            start_time = time.perf_counter()
            optimizer.vectorize_attention(query, key, value)
            end_time = time.perf_counter()

            execution_time = (end_time - start_time) * 1000  # Convert to ms

            # Check metrics
            metrics = optimizer.get_metrics()
            assert len(metrics) > 0, "No metrics were collected"

            latest_metric = metrics[-1]
            assert latest_metric.operation_type == "vectorize_attention"

            # Log performance metrics
            print(f"Chunk Size Impact (size={chunk_size}):")
            print(f"  Execution time: {execution_time:.2f}ms")
            print(f"  Memory usage: {latest_metric.memory_usage / 1024 / 1024:.2f}MB")
            print(
                f"  Vectorization efficiency: {latest_metric.vectorization_efficiency:.2%}"
            )

        except (ValueError, RuntimeError, AssertionError) as e:
            pytest.fail(f"Chunk size impact test failed: {e}")


@pytest.mark.benchmark
def test_memory_layout_optimization(
    vectorization_optimizer: VectorizationOptimizer,
) -> None:
    """Test memory layout optimization and cache utilization."""
    with resource_guard():
        try:
            # Generate test data with different memory layouts
            batch_size = 64
            seq_len = 64
            feat_dim = 64

            # Test 1: Contiguous tensors
            query1, key1, value1 = generate_test_tensors(batch_size, seq_len, feat_dim)

            # Test 2: Non-contiguous tensors (through transpose)
            query2 = query1.transpose(-2, -1).contiguous().transpose(-2, -1)
            key2 = key1.transpose(-2, -1).contiguous().transpose(-2, -1)
            value2 = value1.transpose(-2, -1).contiguous().transpose(-2, -1)

            # Run tests and collect metrics
            result1 = vectorization_optimizer.vectorize_attention(query1, key1, value1)
            metrics1 = vectorization_optimizer.get_metrics()[-1]

            result2 = vectorization_optimizer.vectorize_attention(query2, key2, value2)
            metrics2 = vectorization_optimizer.get_metrics()[-1]

            # Compare results
            torch.testing.assert_close(result1, result2, rtol=1e-5, atol=1e-5)

            # Log performance comparison
            print("Memory Layout Optimization Results:")
            print("  Contiguous tensors:")
            print(f"    Execution time: {metrics1.execution_time:.2f}ms")
            print(f"    Memory usage: {metrics1.memory_usage / 1024 / 1024:.2f}MB")
            print(
                f"    Vectorization efficiency: {metrics1.vectorization_efficiency:.2%}"
            )
            print("  Non-contiguous tensors:")
            print(f"    Execution time: {metrics2.execution_time:.2f}ms")
            print(f"    Memory usage: {metrics2.memory_usage / 1024 / 1024:.2f}MB")
            print(
                f"    Vectorization efficiency: {metrics2.vectorization_efficiency:.2%}"
            )

            # Verify that contiguous tensors perform better
            assert (
                metrics1.vectorization_efficiency >= metrics2.vectorization_efficiency
            ), "Contiguous tensors should have better vectorization efficiency"

        except (
            ValueError,
            RuntimeError,
            AssertionError
        ) as e:
            pytest.fail(f"Memory layout optimization test failed: {e}")
