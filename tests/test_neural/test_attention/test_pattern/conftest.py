"""Common test fixtures for pattern dynamics tests."""

import pytest
import torch

from  src.neural.attention.pattern.pattern_dynamics import PatternDynamics


@pytest.fixture
def space_dim() -> int:
    """Spatial dimensions."""
    return 2


@pytest.fixture
def grid_size() -> int:
    """Grid size per dimension."""
    return 32


@pytest.fixture
def batch_size() -> int:
    """Batch size for testing."""
    return 8


@pytest.fixture
def pattern_system(space_dim, grid_size) -> PatternDynamics:
    """Create a test pattern dynamics system."""
    return PatternDynamics(
        grid_size=grid_size,
        space_dim=space_dim,
        dt=0.01,
        boundary="periodic"
    )


def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, rtol=1e-4, atol=1e-4, msg=""):
    """Custom tensor comparison with cleaner output."""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        print("\nExpected:", b.detach().cpu().numpy())
        print("Got:", a.detach().cpu().numpy())
        raise AssertionError(msg)


def assert_mass_conserved(initial: torch.Tensor, final: torch.Tensor, rtol=1e-4):
    """Assert that mass is conserved between two states."""
    initial_mass = initial.sum(dim=(-2, -1))
    final_mass = final.sum(dim=(-2, -1))
    try:
        assert_tensor_equal(initial_mass, final_mass, rtol=rtol, atol=rtol, msg="Mass should be conserved.")
    except AssertionError as e:
        print("\nInitial mass:", initial_mass.detach().cpu().numpy())
        print("Final mass:", final_mass.detach().cpu().numpy())
        raise e
