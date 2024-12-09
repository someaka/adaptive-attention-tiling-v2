"""Test suite for verifying the testing framework functionality."""

from typing import cast

import pytest
import torch
from torch import nn

from tests.utils.test_helpers import (
    assert_tensor_equal,
    benchmark_forward_backward,
    generate_random_tensor,
    measure_memory_usage,
)


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self: "SimpleModel", hidden_size: int = 64) -> None:
        """Initialize the simple model.

        Args:
        ----
            hidden_size: Size of the hidden layer.

        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self: "SimpleModel", x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.linear(x))


def test_random_seed_fixture() -> None:
    """Test that random seeds are set correctly."""
    # Generate two random tensors with same seed
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(10, 10)
    assert not torch.allclose(tensor1, tensor2)

    # Reset seed and generate again
    torch.manual_seed(42)
    tensor3 = torch.randn(10, 10)
    torch.manual_seed(42)
    tensor4 = torch.randn(10, 10)
    assert torch.allclose(tensor3, tensor4)


def test_device_fixture(device: str) -> None:
    """Test that device fixture works correctly."""
    tensor = generate_random_tensor((10, 10), device)
    assert str(tensor.device) == device


def test_dtype_fixture(dtype: torch.dtype) -> None:
    """Test that dtype fixture works correctly."""
    tensor = generate_random_tensor((10, 10), "cpu", dtype)
    assert tensor.dtype == dtype


def test_benchmark_config_fixture(benchmark_config: dict[str, int]) -> None:
    """Test that benchmark configuration is correct."""
    assert isinstance(benchmark_config, dict)
    assert "warmup_iterations" in benchmark_config
    assert "num_iterations" in benchmark_config
    assert isinstance(benchmark_config["warmup_iterations"], int)
    assert isinstance(benchmark_config["num_iterations"], int)


def test_generate_random_tensor(device: str, dtype: torch.dtype) -> None:
    """Test random tensor generation."""
    shape = (2, 3, 4)
    tensor = generate_random_tensor(shape, device, dtype)

    assert tensor.shape == shape
    assert str(tensor.device) == device
    assert tensor.dtype == dtype

    # Test requires_grad
    tensor = generate_random_tensor(shape, device, dtype, requires_grad=True)
    assert tensor.requires_grad


def test_assert_tensor_equal() -> None:
    """Test tensor equality assertion."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    assert_tensor_equal(x, y)

    # Test with tolerance
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.001, 2.001, 3.001])
    assert_tensor_equal(x, y, rtol=1e-2)

    # Test failure
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    with pytest.raises(AssertionError):
        assert_tensor_equal(x, y)


def test_benchmark_forward_backward(device: str) -> None:
    """Test model benchmarking."""
    model = SimpleModel().to(device)
    input_shape = (32, 64)

    forward_time, backward_time = benchmark_forward_backward(
        model,
        input_shape,
        device=device,
    )

    assert isinstance(forward_time, float)
    assert isinstance(backward_time, float)
    assert forward_time > 0
    assert backward_time > 0


def test_memory_usage() -> None:
    """Test memory usage tracking.

    Verifies memory tracking works correctly for both Vulkan and CPU backends.
    """
    model = SimpleModel()
    input_shape = (32, 64)

    def _run_model() -> torch.Tensor:
        """Run the model and return output tensor."""
        x = generate_random_tensor(input_shape, requires_grad=True)
        out = cast(torch.Tensor, model(x))
        if out.requires_grad:
            out.backward(torch.ones_like(out))
        return out

    # Run memory tracking
    allocated, reserved = measure_memory_usage(_run_model)

    # Memory tracking behavior depends on backend
    if hasattr(torch, "vulkan") and torch.vulkan.is_available():
        # On Vulkan, we expect non-zero memory usage
        assert allocated > 0, "Expected non-zero memory allocation with Vulkan"
        assert reserved >= allocated, "Reserved memory should be >= allocated memory"
    else:
        # On CPU, we still track memory but values may be 0
        assert isinstance(allocated, int)
        assert isinstance(reserved, int)
        assert reserved >= allocated, "Reserved memory should be >= allocated memory"


@pytest.mark.slow
def test_slow_marker() -> None:
    """Test that slow marker works."""
    # Simulate a slow test
    import time

    time.sleep(0.1)


@pytest.mark.gpu
@pytest.mark.skip(reason="GPU acceleration not yet implemented (Issue #123)")
def test_gpu_marker() -> None:
    """Test GPU marker."""
    pytest.skip("GPU acceleration not available")


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "todo: mark test as incomplete/TODO")
