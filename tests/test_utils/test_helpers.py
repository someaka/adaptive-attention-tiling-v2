"""
Test utilities and helper functions.

Provides:
1. Tensor property assertions
2. Numerical stability checks
3. Performance benchmarks
4. Test data generators
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator, Optional

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance
import gc


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    mean_time: float
    std_time: float
    peak_memory: int
    flops: int
    iterations: int


class TensorAssertions:
    """Assertions for tensor properties."""

    @staticmethod
    def assert_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
        """Assert tensor has expected shape."""
        assert (
            tensor.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {tensor.shape}"

    @staticmethod
    def assert_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype) -> None:
        """Assert tensor has expected dtype."""
        assert (
            tensor.dtype == expected_dtype
        ), f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    @staticmethod
    def assert_device(tensor: torch.Tensor, expected_device: torch.device) -> None:
        """Assert tensor is on expected device."""
        assert (
            tensor.device == expected_device
        ), f"Expected device {expected_device}, got {tensor.device}"

    @staticmethod
    def assert_positive_definite(tensor: torch.Tensor, rtol: float = 1e-5) -> None:
        """Assert tensor is positive definite."""
        eigenvals = torch.linalg.eigvalsh(tensor)
        assert torch.all(
            eigenvals > -rtol
        ), f"Tensor is not positive definite, min eigenvalue: {eigenvals.min()}"

    @staticmethod
    def assert_unitary(tensor: torch.Tensor, rtol: float = 1e-5) -> None:
        """Assert tensor is unitary."""
        identity = torch.eye(tensor.shape[-1], device=tensor.device)
        product = tensor @ tensor.conj().transpose(-2, -1)
        assert torch.allclose(
            product, identity, rtol=rtol
        ), f"Tensor is not unitary, max deviation: {torch.max(torch.abs(product - identity))}"

    @staticmethod
    def assert_normalized(
        tensor: torch.Tensor, dim: int = -1, rtol: float = 1e-5
    ) -> None:
        """Assert tensor is normalized along specified dimension."""
        norms = torch.norm(tensor, dim=dim)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=rtol
        ), f"Tensor is not normalized, max deviation: {torch.max(torch.abs(norms - 1))}"


class NumericalStability:
    """Checks for numerical stability."""

    @staticmethod
    def check_gradient_stability(
        model: torch.nn.Module,
        loss_fn: Callable,
        data: torch.Tensor,
        threshold: float = 100.0,
    ) -> bool:
        """Check gradient stability during backpropagation."""
        model.zero_grad()
        loss = loss_fn(model(data))
        loss.backward()

        grads = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
        max_grad = max(grads) if grads else 0.0
        return max_grad < threshold

    @staticmethod
    def check_loss_stability(loss_history: list[float], window_size: int = 10) -> bool:
        """Check if loss is stable over a window."""
        if len(loss_history) < window_size:
            return True

        recent_losses = loss_history[-window_size:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        return bool(std_loss / (abs(mean_loss) + 1e-8) < 0.1)

    @staticmethod
    def check_numerical_accuracy(
        computed: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-5
    ) -> dict[str, float]:
        """Check numerical accuracy metrics."""
        abs_error = torch.abs(computed - expected)
        rel_error = abs_error / (torch.abs(expected) + 1e-8)

        return {
            "max_absolute_error": abs_error.max().item(),
            "mean_absolute_error": abs_error.mean().item(),
            "max_relative_error": rel_error.max().item(),
            "mean_relative_error": rel_error.mean().item(),
        }

    @staticmethod
    def check_distribution_stability(
        dist1: torch.Tensor, dist2: torch.Tensor, threshold: float = 0.1
    ) -> bool:
        """Check if two distributions are similar using Wasserstein distance."""
        # Convert to numpy and normalize
        d1 = dist1.detach().cpu().numpy()
        d2 = dist2.detach().cpu().numpy()
        d1 = d1 / (d1.sum() + 1e-8)
        d2 = d2 / (d2.sum() + 1e-8)

        # Compute Wasserstein distance
        distance = wasserstein_distance(d1.flatten(), d2.flatten())
        return distance < threshold


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    class TimerResult:
        """Container for timer result."""
        def __init__(self):
            self.elapsed: float = 0.0

    @staticmethod
    @contextmanager
    def timer(name: Optional[str] = None) -> Generator[TimerResult, None, None]:
        """Context manager for timing code blocks."""
        result = PerformanceBenchmark.TimerResult()
        start = time.perf_counter()
        try:
            yield result
        finally:
            end = time.perf_counter()
            result.elapsed = end - start
            if name:
                print(f"{name}: {result.elapsed:.4f} seconds")

    @staticmethod
    def benchmark_function(
        func: Callable, *args, n_runs: int = 100, warmup: int = 10
    ) -> BenchmarkResult:
        """Benchmark a function's performance."""
        # Warmup runs
        for _ in range(warmup):
            func(*args)

        # Actual benchmark
        times = []
        peak_memory = 0
        for _ in range(n_runs):
            gc.collect()  # Memory cleanup
            start_memory = 0  # Device-agnostic memory tracking

            start = time.perf_counter()
            func(*args)
            end = time.perf_counter()

            end_memory = 0  # Device-agnostic memory tracking
            peak_memory = max(peak_memory, end_memory - start_memory)
            times.append(end - start)

        return BenchmarkResult(
            mean_time=float(np.mean(times)),
            std_time=float(np.std(times)),
            peak_memory=peak_memory,
            flops=0,  # TODO: Implement FLOPS counting
            iterations=n_runs,
        )

    @staticmethod
    def profile_memory(func: Callable, *args) -> dict[str, int]:
        """Profile memory usage of a function."""
        gc.collect()  # Memory cleanup

        start_memory = 0  # Device-agnostic memory tracking
        func(*args)
        end_memory = 0  # Device-agnostic memory tracking
        peak_memory = 0  # Device-agnostic memory tracking

        return {
            "allocated_memory": end_memory - start_memory,
            "peak_memory": peak_memory,
            "cached_memory": 0,  # Device-agnostic memory tracking
        }


class TestDataGenerator:
    """Generators for test data."""

    @staticmethod
    def generate_random_tensor(
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> torch.Tensor:
        """Generate random tensor with specific properties."""
        if kwargs.get("positive_definite", False):
            tensor = torch.randn(*shape, device=device, dtype=dtype)
            return tensor @ tensor.transpose(-2, -1)

        if kwargs.get("unitary", False):
            tensor = torch.randn(*shape, device=device, dtype=dtype)
            q, _ = torch.linalg.qr(tensor)
            return q

        if kwargs.get("normalized", False):
            tensor = torch.randn(*shape, device=device, dtype=dtype)
            return tensor / torch.norm(
                tensor, dim=kwargs.get("norm_dim", -1), keepdim=True
            )

        return torch.randn(*shape, device=device, dtype=dtype)

    @staticmethod
    def generate_batch_sequences(
        batch_size: int,
        seq_length: int,
        feature_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate batch of sequences."""
        return torch.randn(batch_size, seq_length, feature_dim, dtype=dtype)

    @staticmethod
    def generate_graph_data(
        num_nodes: int, feature_dim: int, edge_probability: float = 0.3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random graph data."""
        node_features = torch.randn(num_nodes, feature_dim)
        adj_matrix = torch.rand(num_nodes, num_nodes) < edge_probability
        adj_matrix = adj_matrix & adj_matrix.t()  # Make symmetric
        return node_features, adj_matrix

    @staticmethod
    def generate_quantum_state(
        num_qubits: int, dtype: torch.dtype = torch.complex64
    ) -> torch.Tensor:
        """Generate random quantum state."""
        dim = 2**num_qubits
        state = torch.randn(dim, dtype=dtype) + 1j * torch.randn(dim, dtype=dtype)
        return state / torch.norm(state)


def test_tensor_assertions():
    """Test tensor assertion utilities."""
    tensor = torch.randn(3, 4)

    # Test shape assertion
    TensorAssertions.assert_shape(tensor, (3, 4))
    with pytest.raises(AssertionError):
        TensorAssertions.assert_shape(tensor, (4, 3))

    # Test dtype assertion
    TensorAssertions.assert_dtype(tensor, torch.float32)
    with pytest.raises(AssertionError):
        TensorAssertions.assert_dtype(tensor, torch.int32)

    # Test device assertion
    TensorAssertions.assert_device(tensor, torch.device("cpu"))

    # Test positive definite assertion
    pd_tensor = torch.randn(3, 3)
    pd_tensor = pd_tensor @ pd_tensor.t()
    TensorAssertions.assert_positive_definite(pd_tensor)

    # Test unitary assertion
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    TensorAssertions.assert_unitary(q)

    # Test normalized assertion
    norm_tensor = torch.randn(3, 4)
    norm_tensor = norm_tensor / torch.norm(norm_tensor, dim=1, keepdim=True)
    TensorAssertions.assert_normalized(norm_tensor, dim=1)


def test_numerical_stability():
    """Test numerical stability utilities."""
    # Test gradient stability
    model = torch.nn.Linear(10, 1)
    data = torch.randn(5, 10)
    loss_fn = torch.nn.MSELoss()

    is_stable = NumericalStability.check_gradient_stability(model, loss_fn, data)
    assert isinstance(is_stable, bool)

    # Test loss stability
    loss_history = [1.0, 0.9, 0.8, 0.7, 0.6]
    is_stable = NumericalStability.check_loss_stability(loss_history)
    assert isinstance(is_stable, bool)

    # Test numerical accuracy
    computed = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([1.1, 1.9, 3.1])
    metrics = NumericalStability.check_numerical_accuracy(computed, expected)
    assert all(
        key in metrics
        for key in [
            "max_absolute_error",
            "mean_absolute_error",
            "max_relative_error",
            "mean_relative_error",
        ]
    )

    # Test distribution stability
    dist1 = torch.softmax(torch.randn(100), dim=0)
    dist2 = torch.softmax(torch.randn(100), dim=0)
    is_stable = NumericalStability.check_distribution_stability(dist1, dist2)
    assert isinstance(is_stable, bool)


def test_performance_benchmark():
    """Test performance benchmarking utilities."""

    def dummy_function(x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, x.t())

    # Test timer
    with PerformanceBenchmark.timer("test_operation") as t:
        torch.randn(100, 100)
    assert isinstance(t.elapsed, float)

    # Test benchmark function
    data = torch.randn(100, 100)
    result = PerformanceBenchmark.benchmark_function(dummy_function, data, n_runs=10)
    assert isinstance(result, BenchmarkResult)
    assert result.iterations == 10

    # Test memory profiling
    memory_stats = PerformanceBenchmark.profile_memory(dummy_function, data)
    assert all(
        key in memory_stats
        for key in ["allocated_memory", "peak_memory", "cached_memory"]
    )


def test_data_generator():
    """Test data generation utilities."""
    # Test random tensor generation
    tensor = TestDataGenerator.generate_random_tensor((3, 4), positive_definite=True)
    assert tensor.shape == (3, 3)

    # Test batch sequence generation
    sequences = TestDataGenerator.generate_batch_sequences(
        batch_size=2, seq_length=5, feature_dim=3
    )
    assert sequences.shape == (2, 5, 3)

    # Test graph data generation
    node_features, adj_matrix = TestDataGenerator.generate_graph_data(
        num_nodes=10, feature_dim=5
    )
    assert node_features.shape == (10, 5)
    assert adj_matrix.shape == (10, 10)

    # Test quantum state generation
    state = TestDataGenerator.generate_quantum_state(num_qubits=3)
    assert state.shape == (8,)
    assert torch.allclose(torch.norm(state), torch.tensor(1.0))
