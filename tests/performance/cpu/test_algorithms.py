"""Performance tests for CPU algorithm efficiency.

This module tests the algorithmic efficiency of the
Adaptive Attention Tiling system, focusing on:
1. Fast path optimizations
2. Branching efficiency
3. Loop optimization
4. Numerical stability
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

from src.core.performance.cpu.algorithms import AlgorithmOptimizer

# Test configurations - reduced sizes for safety
MATRIX_SIZES = [(64, 64), (256, 256)]  # Removed (1024, 1024)
BATCH_SIZES = [1, 16]  # Removed 64
SPARSITY_LEVELS = [0.1, 0.5, 0.9]
OPTIMIZATION_LEVELS = ["O0", "O1", "O2", "O3"]

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
def algorithm_optimizer():
    """Create an AlgorithmOptimizer instance for testing."""
    return AlgorithmOptimizer(enable_profiling=True)


def generate_sparse_matrix(size: tuple[int, int], sparsity: float) -> torch.Tensor:
    """Generate a sparse matrix with given sparsity level."""
    with resource_guard():
        matrix = torch.randn(size)
        mask = torch.rand(size) > sparsity
        return matrix * mask


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
@pytest.mark.parametrize("sparsity", SPARSITY_LEVELS)
def test_fast_path_optimization(
    algorithm_optimizer: AlgorithmOptimizer,
    matrix_size: tuple[int, int],
    sparsity: float,
):
    """Test fast path optimizations for sparse operations."""
    with resource_guard():
        # Generate sparse matrices
        matrix_a = generate_sparse_matrix(matrix_size, sparsity)
        matrix_b = generate_sparse_matrix(matrix_size, sparsity)

        # Register fast path for sparse matrix multiplication
        def is_sparse(x: torch.Tensor, threshold: float = 0.5) -> bool:
            return torch.count_nonzero(x) / x.numel() < threshold

        # Register optimized path
        algorithm_optimizer.register_fast_path(
            "sparse_matmul",
            lambda x, y: torch.sparse.mm(x.to_sparse(), y.to_sparse()).to_dense(),
            condition=lambda x, y: is_sparse(x) and is_sparse(y),
        )

        # Warm-up run
        _ = algorithm_optimizer.optimize_operation("sparse_matmul", matrix_a, matrix_b)
        algorithm_optimizer.clear_metrics()

        # Test run
        algorithm_optimizer.optimize_operation("sparse_matmul", matrix_a, matrix_b)
        metrics = algorithm_optimizer.get_metrics()[0]

        # Performance assertions
        assert metrics.execution_time > 0
        assert metrics.memory_usage > 0
        assert metrics.fast_path_hits >= (1 if sparsity > 0.5 else 0)
        assert metrics.instruction_count >= 0


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_branch_prediction(
    algorithm_optimizer: AlgorithmOptimizer,
    batch_size: int,
    matrix_size: tuple[int, int],
):
    """Test branch prediction efficiency."""
    with resource_guard():
        # Create input tensors
        inputs = [torch.randn(matrix_size) for _ in range(batch_size)]

        # Define test operations
        def operation_a(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        def operation_b(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

        # Register operations
        algorithm_optimizer.register_operation("op_a", operation_a)
        algorithm_optimizer.register_operation("op_b", operation_b)

        # Warm-up run
        for x in inputs:
            _ = algorithm_optimizer.optimize_operation(
                "op_a" if torch.mean(x) > 0 else "op_b", x
            )
        algorithm_optimizer.clear_metrics()

        # Test run
        results = []
        for x in inputs:
            result = algorithm_optimizer.optimize_operation(
                "op_a" if torch.mean(x) > 0 else "op_b", x
            )
            results.append(result)

        # Get metrics
        algorithm_optimizer.get_metrics()[0]

        # Verify results
        for x, y in zip(inputs, results):
            baseline = operation_a(x) if torch.mean(x) > 0 else operation_b(x)
            assert torch.allclose(baseline, y, rtol=1e-4)


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("optimization_level", OPTIMIZATION_LEVELS)
def test_loop_optimization(
    algorithm_optimizer: AlgorithmOptimizer, optimization_level: str
):
    """Test loop optimization strategies."""
    with resource_guard():
        size = 256  # Reduced from 1024
        matrix = torch.randn(size, size)

        # Configure optimization level
        algorithm_optimizer.set_optimization_level(optimization_level)

        # Define test operation with loops
        def loop_operation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    result[i, j] = torch.tanh(x[i, j])
            return result

        # Register operation
        algorithm_optimizer.register_operation("loop_op", loop_operation)

        # Warm-up run
        _ = algorithm_optimizer.optimize_operation("loop_op", matrix)
        algorithm_optimizer.clear_metrics()

        # Test run
        algorithm_optimizer.optimize_operation("loop_op", matrix)
        metrics = algorithm_optimizer.get_metrics()[0]

        # Optimization level specific assertions
        if optimization_level in ["O2", "O3"]:
            assert metrics.loop_fusion_applied
            assert metrics.cache_optimization_enabled


@pytest.mark.benchmark(min_rounds=5)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_numerical_stability(
    algorithm_optimizer: AlgorithmOptimizer, matrix_size: tuple[int, int]
):
    """Test numerical stability of optimized computations."""
    with resource_guard():
        # Generate test matrix
        matrix = torch.randn(matrix_size)

        # Define numerically sensitive operation
        def sensitive_operation(x: torch.Tensor) -> torch.Tensor:
            return torch.log1p(torch.exp(x))  # LogSumExp

        # Register operation
        algorithm_optimizer.register_operation("sensitive_op", sensitive_operation)

        # Warm-up run
        _ = algorithm_optimizer.optimize_operation("sensitive_op", matrix)
        algorithm_optimizer.clear_metrics()

        # Test run
        result = algorithm_optimizer.optimize_operation("sensitive_op", matrix)
        algorithm_optimizer.get_metrics()[0]

        # Compare with standard implementation
        standard_result = sensitive_operation(matrix)

        # Numerical stability assertions
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert torch.allclose(standard_result, result, rtol=1e-4)


@pytest.mark.benchmark(min_rounds=5)
def test_optimization_overhead(algorithm_optimizer: AlgorithmOptimizer):
    """Test overhead of optimization techniques."""
    with resource_guard():
        size = 256  # Reduced from 512
        matrix = torch.randn(size, size)

        def simple_operation(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        # Register operation
        algorithm_optimizer.register_operation("simple_op", simple_operation)

        # Measure baseline time
        start_time = time.perf_counter()
        baseline_result = simple_operation(matrix)
        baseline_time = time.perf_counter() - start_time

        # Warm-up run
        _ = algorithm_optimizer.optimize_operation("simple_op", matrix)
        algorithm_optimizer.clear_metrics()

        # Test run
        start_time = time.perf_counter()
        optimized_result = algorithm_optimizer.optimize_operation("simple_op", matrix)
        optimized_time = time.perf_counter() - start_time

        # Overhead assertions
        assert (
            optimized_time < baseline_time * 1.5
        )  # Optimization overhead should be reasonable
        assert torch.allclose(baseline_result, optimized_result, rtol=1e-4)
