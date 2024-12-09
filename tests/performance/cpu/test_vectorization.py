"""Performance tests for CPU vectorization operations.

This module tests the performance characteristics of vectorized operations
in the Adaptive Attention Tiling system, focusing on:
1. Attention computation efficiency
2. Pattern dynamics vectorization
3. Geometric flow optimization
4. Memory layout and cache utilization
"""

import pytest
import torch

from src.core.performance.cpu.vectorization import VectorizationOptimizer

# Test configurations
BATCH_SIZES = [32, 128, 512, 2048]
SEQUENCE_LENGTHS = [64, 256, 1024]
FEATURE_DIMS = [32, 128, 512]
CHUNK_SIZES = [64, 256, 1024]


@pytest.fixture
def vectorization_optimizer():
    """Create a VectorizationOptimizer instance for testing."""
    return VectorizationOptimizer(enable_profiling=True)


def generate_test_tensors(batch_size: int, seq_len: int, feat_dim: int) -> tuple[torch.Tensor, ...]:
    """Generate test tensors for attention computation."""
    query = torch.randn(batch_size, seq_len, feat_dim)
    key = torch.randn(batch_size, seq_len, feat_dim)
    value = torch.randn(batch_size, seq_len, feat_dim)
    return query, key, value


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
@pytest.mark.parametrize("feat_dim", FEATURE_DIMS)
def test_attention_vectorization_performance(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, seq_len: int, feat_dim: int
):
    """Test attention computation vectorization performance."""
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


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
def test_pattern_dynamics_vectorization(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, seq_len: int
):
    """Test pattern dynamics vectorization performance."""
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


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("dim", FEATURE_DIMS)
def test_geometric_flow_vectorization(
    vectorization_optimizer: VectorizationOptimizer, batch_size: int, dim: int
):
    """Test geometric flow vectorization performance."""
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


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_size_impact(chunk_size: int):
    """Test impact of different chunk sizes on vectorization performance."""
    optimizer = VectorizationOptimizer(chunk_size=chunk_size)
    batch_size = 1024
    seq_len = 256
    feat_dim = 64

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


def test_memory_layout_optimization(vectorization_optimizer: VectorizationOptimizer):
    """Test memory layout optimization and cache utilization."""
    # Test with non-contiguous tensor
    tensor = torch.randn(128, 64, 32)
    permuted = tensor.permute(2, 0, 1)  # Create non-contiguous tensor

    efficiency = vectorization_optimizer._estimate_vectorization_efficiency(permuted)
    assert efficiency < 1.0  # Should be lower due to non-contiguous layout

    # Test with contiguous tensor
    contiguous = permuted.contiguous()
    efficiency = vectorization_optimizer._estimate_vectorization_efficiency(contiguous)
    assert efficiency > 0.9  # Should be high due to optimized layout
