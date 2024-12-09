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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NoReturn

import pytest
import torch

from src.core.performance.cpu.vectorization import VectorizationOptimizer

# Test configurations - reduced sizes for safety
BATCH_SIZES = [32, 128]  # Removed 512, 2048
SEQUENCE_LENGTHS = [64, 256]  # Removed 1024
FEATURE_DIMS = [32, 128]  # Removed 512
CHUNK_SIZES = [64, 256]  # Removed 1024

# Resource limits
MAX_MEMORY_GB = 4  # Maximum memory limit in GB
MAX_TIME_SECONDS = 30  # Maximum time limit per test in seconds


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
def vectorization_optimizer():
    """Create a VectorizationOptimizer instance for testing."""
    return VectorizationOptimizer(enable_profiling=True)


def generate_test_tensors(batch_size: int, seq_len: int, feat_dim: int) -> tuple[torch.Tensor, ...]:
    """Generate test tensors for attention computation."""
    with resource_guard():
        query = torch.randn(batch_size, seq_len, feat_dim)
        key = torch.randn(batch_size, seq_len, feat_dim)
        value = torch.randn(batch_size, seq_len, feat_dim)
        return query, key, value


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
@pytest.mark.parametrize("feat_dim", FEATURE_DIMS)
def test_attention_vectorization_performance(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, seq_len: int, feat_dim: int
):
    """Test attention computation vectorization performance."""
    with resource_guard():
        query, key, value = generate_test_tensors(batch_size, seq_len, feat_dim)

        # Warm-up run
        _ = vectorization_optimizer.vectorize_attention(query, key, value)
        vectorization_optimizer.clear_metrics()

        # Test run
        result = vectorization_optimizer.vectorize_attention(query, key, value)
        metrics = vectorization_optimizer.get_metrics()[0]

        # Performance assertions
        assert metrics.execution_time > 0
        assert metrics.memory_usage > 0
        assert 0 <= metrics.vectorization_efficiency <= 1
        assert metrics.operation_type == "vectorize_attention"

        # Shape and value assertions
        assert result.shape == (batch_size, seq_len, feat_dim)
        assert not torch.isnan(result).any()


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
def test_pattern_dynamics_vectorization(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, seq_len: int
):
    """Test pattern dynamics vectorization performance."""
    with resource_guard():
        pattern = torch.randn(batch_size, seq_len, seq_len)
        flow = torch.randn(batch_size, seq_len, seq_len)

        # Warm-up run
        _ = vectorization_optimizer.vectorize_pattern_dynamics(pattern, flow)
        vectorization_optimizer.clear_metrics()

        # Test run
        result = vectorization_optimizer.vectorize_pattern_dynamics(pattern, flow)
        metrics = vectorization_optimizer.get_metrics()[0]

        # Performance assertions
        assert metrics.execution_time > 0
        assert metrics.memory_usage > 0
        assert 0 <= metrics.vectorization_efficiency <= 1
        assert metrics.operation_type == "vectorize_pattern_dynamics"

        # Numerical stability assertions
        assert torch.all(result >= -1)
        assert torch.all(result <= 1)
        assert not torch.isnan(result).any()


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("dim", FEATURE_DIMS)
def test_geometric_flow_vectorization(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, dim: int
):
    """Test geometric flow vectorization performance."""
    with resource_guard():
        metric = torch.randn(batch_size, dim, dim)
        connection = torch.randn(batch_size, dim, dim)

        # Warm-up run
        _ = vectorization_optimizer.vectorize_geometric_flow(metric, connection)
        vectorization_optimizer.clear_metrics()

        # Test run
        result = vectorization_optimizer.vectorize_geometric_flow(metric, connection)
        metrics = vectorization_optimizer.get_metrics()[0]

        # Performance assertions
        assert metrics.execution_time > 0
        assert metrics.memory_usage > 0
        assert 0 <= metrics.vectorization_efficiency <= 1
        assert metrics.operation_type == "vectorize_geometric_flow"

        # Shape and stability assertions
        assert result.shape == (batch_size, dim, dim)
        assert not torch.isnan(result).any()


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_size_impact(chunk_size: int):
    """Test impact of different chunk sizes on vectorization performance."""
    with resource_guard():
        optimizer = VectorizationOptimizer(chunk_size=chunk_size)
        batch_size = 128  # Reduced from 1024
        seq_len = 64  # Reduced from 256
        feat_dim = 32  # Reduced from 64

        query, key, value = generate_test_tensors(batch_size, seq_len, feat_dim)

        # Warm-up run
        _ = optimizer.vectorize_attention(query, key, value)
        optimizer.clear_metrics()

        # Test run
        _ = optimizer.vectorize_attention(query, key, value)
        metrics = optimizer.get_metrics()[0]

        # Store metrics for analysis
        return {
            "chunk_size": chunk_size,
            "execution_time": metrics.execution_time,
            "memory_usage": metrics.memory_usage,
            "vectorization_efficiency": metrics.vectorization_efficiency,
        }


@pytest.mark.benchmark(min_rounds=5)
def test_memory_layout_optimization(vectorization_optimizer: VectorizationOptimizer):
    """Test memory layout optimization and cache utilization."""
    with resource_guard():
        # Test with non-contiguous tensor
        tensor = torch.randn(64, 32, 16)  # Reduced from 128, 64, 32
        permuted = tensor.permute(2, 0, 1)  # Create non-contiguous tensor

        efficiency = vectorization_optimizer._estimate_vectorization_efficiency(permuted)
        assert efficiency < 1.0  # Should be lower due to non-contiguous layout

        # Test with contiguous tensor
        contiguous = permuted.contiguous()
        efficiency = vectorization_optimizer._estimate_vectorization_efficiency(contiguous)
        assert efficiency > 0.9  # Should be high due to optimized layout
