"""Performance tests for CPU algorithm efficiency.

This module tests the algorithmic efficiency of the
Adaptive Attention Tiling system, focusing on:
1. Fast path optimizations
2. Branching efficiency
3. Loop optimization
4. Numerical stability
"""

import pytest
import torch

from src.core.performance.cpu.algorithms import AlgorithmOptimizer

# Test configurations
MATRIX_SIZES = [(64, 64), (256, 256), (1024, 1024)]
BATCH_SIZES = [1, 16, 64]
SPARSITY_LEVELS = [0.1, 0.5, 0.9]
OPTIMIZATION_LEVELS = ["O0", "O1", "O2", "O3"]


@pytest.fixture
def algorithm_optimizer():
    """Create an AlgorithmOptimizer instance for testing."""
    return AlgorithmOptimizer(enable_profiling=True)


def generate_sparse_matrix(size: tuple[int, int], sparsity: float) -> torch.Tensor:
    """Generate a sparse matrix with given sparsity level."""
    matrix = torch.randn(size)
    mask = torch.rand(size) > sparsity
    return matrix * mask


@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
@pytest.mark.parametrize("sparsity", SPARSITY_LEVELS)
def test_fast_path_optimization(
    algorithm_optimizer: AlgorithmOptimizer, matrix_size: tuple[int, int], sparsity: float
):
    """Test fast path optimizations for sparse operations."""
    # Generate sparse matrices
    matrix_a = generate_sparse_matrix(matrix_size, sparsity)
    matrix_b = generate_sparse_matrix(matrix_size, sparsity)

    # Baseline computation (dense)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    baseline_result = torch.matmul(matrix_a, matrix_b)
    end_time.record()

    torch.cuda.synchronize()
    baseline_time = start_time.elapsed_time(end_time)

    # Optimized computation (sparse)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    optimized_result = algorithm_optimizer.fast_path_matmul(matrix_a, matrix_b)
    end_time.record()

    torch.cuda.synchronize()
    optimized_time = start_time.elapsed_time(end_time)

    # Performance assertions
    speedup = baseline_time / optimized_time
    expected_speedup = 1 / (1 - sparsity)  # Theoretical speedup
    assert speedup > expected_speedup * 0.5  # At least 50% of theoretical

    # Accuracy verification
    assert torch.allclose(baseline_result, optimized_result, rtol=1e-4)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_branch_prediction(
    algorithm_optimizer: AlgorithmOptimizer, batch_size: int, matrix_size: tuple[int, int]
):
    """Test branch prediction efficiency."""
    # Generate test data
    matrices = [torch.randn(matrix_size) for _ in range(batch_size)]
    thresholds = torch.linspace(-1, 1, batch_size)

    def baseline_branch(matrices, thresholds) -> list[torch.Tensor]:
        results = []
        for matrix, threshold in zip(matrices, thresholds):
            if torch.mean(matrix) > threshold:
                results.append(matrix * 2)
            else:
                results.append(matrix * 0.5)
        return results

    # Baseline branching
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    baseline_results = baseline_branch(matrices, thresholds)
    end_time.record()

    torch.cuda.synchronize()
    baseline_time = start_time.elapsed_time(end_time)

    # Optimized branching
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    optimized_results = algorithm_optimizer.optimized_branch(matrices, thresholds)
    end_time.record()

    torch.cuda.synchronize()
    optimized_time = start_time.elapsed_time(end_time)

    # Performance assertions
    assert optimized_time < baseline_time * 0.8  # At least 20% faster

    # Verify results
    for baseline, optimized in zip(baseline_results, optimized_results):
        assert torch.allclose(baseline, optimized, rtol=1e-4)


@pytest.mark.parametrize("optimization_level", OPTIMIZATION_LEVELS)
def test_loop_optimization(algorithm_optimizer: AlgorithmOptimizer, optimization_level: str):
    """Test loop optimization strategies."""
    size = 1024
    iterations = 100
    data = torch.randn(size, size)

    # Configure optimization level
    algorithm_optimizer.set_optimization_level(optimization_level)

    # Run optimized computation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    algorithm_optimizer.optimized_loop(data, iterations)
    end_time.record()

    torch.cuda.synchronize()
    start_time.elapsed_time(end_time)

    metrics = algorithm_optimizer.get_metrics()

    # Performance assertions based on optimization level
    if optimization_level == "O0":
        assert metrics.loop_unroll_factor == 1
    elif optimization_level == "O1":
        assert metrics.loop_unroll_factor >= 2
    elif optimization_level == "O2":
        assert metrics.loop_unroll_factor >= 4
        assert metrics.vectorization_enabled
    else:  # O3
        assert metrics.loop_unroll_factor >= 8
        assert metrics.vectorization_enabled
        assert metrics.cache_optimization_enabled


@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_numerical_stability(algorithm_optimizer: AlgorithmOptimizer, matrix_size: tuple[int, int]):
    """Test numerical stability of optimized computations."""
    # Generate test matrices
    matrix_a = torch.randn(matrix_size) * 1e6  # Large values
    matrix_b = torch.randn(matrix_size) * 1e-6  # Small values

    # Standard computation
    standard_result = torch.matmul(matrix_a, matrix_b)

    # Optimized computation with stability checks
    optimized_result = algorithm_optimizer.stable_matmul(matrix_a, matrix_b)

    # Get stability metrics
    metrics = algorithm_optimizer.get_metrics()

    # Stability assertions
    assert metrics.max_relative_error < 1e-5
    assert metrics.mean_relative_error < 1e-6
    assert not torch.isnan(optimized_result).any()
    assert not torch.isinf(optimized_result).any()

    # Compare results
    assert torch.allclose(standard_result, optimized_result, rtol=1e-4)


def test_optimization_overhead(algorithm_optimizer: AlgorithmOptimizer):
    """Test overhead of optimization techniques."""
    size = 256
    data = torch.randn(size, size)

    # Baseline computation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    baseline_result = torch.matmul(data, data)
    end_time.record()

    torch.cuda.synchronize()
    baseline_time = start_time.elapsed_time(end_time)

    # Optimized computation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    optimized_result = algorithm_optimizer.optimized_matmul(data, data)
    end_time.record()

    torch.cuda.synchronize()
    optimization_time = start_time.elapsed_time(end_time)

    # Get overhead metrics
    metrics = algorithm_optimizer.get_metrics()

    # Overhead assertions
    assert metrics.optimization_overhead < baseline_time * 0.1  # Less than 10%
    assert optimization_time < baseline_time * 1.1  # Total time within 10%
    assert torch.allclose(baseline_result, optimized_result, rtol=1e-4)
