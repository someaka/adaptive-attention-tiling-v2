"""Unit tests for tiling components."""

import numpy as np
import numpy.random as npr
import pytest
import torch

from src.core.tiling.components import (
    LoadBalancer,
    LoadProfile,
    ResolutionAdapter,
    StateManager,
)

# Initialize random number generator
rng = npr.default_rng()


@pytest.fixture
def state_manager() -> StateManager:
    """Create a state manager for testing."""
    return StateManager(size=32)


@pytest.fixture
def resolution_adapter() -> ResolutionAdapter:
    """Create a resolution adapter for testing."""
    return ResolutionAdapter(
        min_resolution=0.1,
        max_resolution=1.0,
        momentum=0.9,
        hysteresis=0.1,
    )


@pytest.fixture
def load_balancer() -> LoadBalancer:
    """Create a load balancer for testing."""
    return LoadBalancer(balance_threshold=0.1, momentum=0.9, max_adjustment=0.2)


def test_state_compression(state_manager: StateManager) -> None:
    """Test state compression functionality."""
    # Create test state
    state = torch.randn(1, 32, 64)  # batch, seq_len, dim

    # Test compression
    compressed = state_manager.compress_state(state, target_resolution=0.5)
    assert compressed.shape[1] == 16  # Half resolution

    # Test invalid input
    with pytest.raises(ValueError, match="State must be a torch.Tensor"):
        state_manager.compress_state(None, target_resolution=0.5)
    with pytest.raises(ValueError, match="State must be a torch.Tensor"):
        state_manager.compress_state([1, 2, 3], target_resolution=0.5)


def test_state_expansion(state_manager: StateManager) -> None:
    """Test state expansion functionality."""
    # Create test state
    state = torch.randn(1, 16, 64)  # batch, seq_len, dim

    # Test expansion
    expanded = state_manager.expand_state(state, target_size=32)
    assert expanded.shape[1] == 32  # Full size

    # Test invalid input
    with pytest.raises(ValueError, match="State must be a torch.Tensor"):
        state_manager.expand_state(None, target_size=32)
    with pytest.raises(ValueError, match="State must be a torch.Tensor"):
        state_manager.expand_state([1, 2, 3], target_size=32)


def test_state_transfer(state_manager: StateManager) -> None:
    """Test state transfer functionality."""
    # Test case 1: None source state
    target = torch.randn(1, 32, 32)
    with pytest.raises(ValueError, match="Source state cannot be None"):
        state_manager.transfer_state(None, target)

    # Test case 2: None target state - should return clone of source
    source = torch.randn(1, 32, 32)
    result = state_manager.transfer_state(source, None)
    assert torch.equal(result, source)  # Should be equal but not the same object
    assert result is not source  # Should be a new tensor (clone)

    # Test case 3: Valid transfer
    source = torch.ones(1, 32, 32)
    target = torch.zeros(1, 32, 32)
    state_manager.transfer_state(source, target)
    # After blending with factor 0.5, target should be 0.5
    assert torch.allclose(target, torch.full_like(target, 0.5))


def test_resolution_adaptation(resolution_adapter: ResolutionAdapter) -> None:
    """Test resolution adaptation functionality."""
    current_res = 0.5

    # Test basic adaptation
    for _ in range(10):
        current_res = resolution_adapter.adapt(
            0.5 + rng.normal(0, 0.05),
            current_resolution=current_res,
        )
    assert current_res > 0.4
    assert current_res < 0.6


def test_hysteresis(resolution_adapter: ResolutionAdapter) -> None:
    """Test hysteresis functionality."""
    # Test that small changes don't cause oscillation
    current_res = 0.5
    # Small fluctuations around 0.5
    resolutions = []
    for _ in range(10):
        current_res = resolution_adapter.adapt(
            0.5 + rng.normal(0, 0.05),
            current_resolution=current_res,
        )
        resolutions.append(current_res)

    # Check that variations are small
    variations = np.diff(resolutions)
    assert np.mean(np.abs(variations)) < resolution_adapter.hysteresis


def test_momentum(resolution_adapter: ResolutionAdapter) -> None:
    """Test momentum functionality."""
    # Test that changes have momentum
    # Start from a lower resolution
    initial_res = 0.5

    # Sudden change - use a high density to force increase
    new_res = resolution_adapter.adapt(
        0.95,  # Try to go to max
        current_resolution=initial_res,
    )
    assert new_res > initial_res
    # Should not reach max due to momentum
    assert new_res < resolution_adapter.max_resolution

    # Should maintain direction due to momentum
    next_res = resolution_adapter.adapt(0.95, current_resolution=new_res)
    assert next_res > new_res


def test_load_balancing(load_balancer: LoadBalancer) -> None:
    """Test load balancing functionality."""
    # Test 1: Normal imbalance case
    current = LoadProfile(
        compute_cost=0.6,
        memory_usage=1000,
        resolution=1.0,
        density_metric=0.5,
    )
    neighbors = [
        LoadProfile(
            compute_cost=0.4,
            memory_usage=500,
            resolution=0.5,
            density_metric=0.5,
        ),
        LoadProfile(
            compute_cost=0.5,
            memory_usage=600,
            resolution=0.6,
            density_metric=0.5,
        ),
    ]

    # Test load balancing for normal case
    adjustment, metrics = load_balancer.balance_load(current, neighbors)

    # Verify adjustment direction and magnitude for normal case
    assert adjustment < 0  # Should decrease load since current is higher
    assert abs(adjustment) <= load_balancer.max_adjustment  # Should respect max adjustment
    assert metrics["status"] == "decreasing"

    # Test 2: Extreme case
    current_extreme = LoadProfile(
        compute_cost=0.9,
        memory_usage=2000,
        resolution=1.0,
        density_metric=0.5,
    )
    neighbors_extreme = [
        LoadProfile(
            compute_cost=0.1,
            memory_usage=500,
            resolution=0.5,
            density_metric=0.5,
        ),
    ]

    # Test load balancing for extreme case
    adjustment_extreme, metrics_extreme = load_balancer.balance_load(
        current_extreme,
        neighbors_extreme,
    )

    # For extreme imbalance (>0.5), the adjustment can exceed max_adjustment
    assert adjustment_extreme < 0  # Should decrease load
    assert metrics_extreme["load_imbalance"] > 0.5  # Verify it's an extreme case
    # The actual adjustment can be larger than max_adjustment for extreme cases

    # Test 3: Balanced case - all loads exactly equal
    balanced = LoadProfile(
        compute_cost=0.5,
        memory_usage=1000,
        resolution=1.0,
        density_metric=0.5,
    )
    balanced_neighbors = [
        LoadProfile(
            compute_cost=0.5,
            memory_usage=1000,
            resolution=1.0,
            density_metric=0.5,
        ),
        LoadProfile(
            compute_cost=0.5,
            memory_usage=1000,
            resolution=1.0,
            density_metric=0.5,
        ),
    ]
    (
        adjustment_balanced,
        metrics_balanced,
    ) = load_balancer.balance_load(balanced, balanced_neighbors)

    # For perfectly balanced case
    assert abs(adjustment_balanced) < 1e-6  # Should be very close to zero
    assert abs(metrics_balanced["load_imbalance"]) < 1e-6  # Should be very close to zero
    assert metrics_balanced["avg_load"] == balanced.compute_cost  # Average should match
    assert metrics_balanced["local_load"] == balanced.compute_cost  # Local should match


def test_load_balancer_extreme_cases(load_balancer: LoadBalancer) -> None:
    """Test load balancer behavior for extreme cases."""
    # Test 1: Normal imbalance case
    current = LoadProfile(
        compute_cost=0.7,
        memory_usage=1000,
        resolution=1.0,
        density_metric=0.5,
    )
    neighbors = [
        LoadProfile(
            compute_cost=0.3,
            memory_usage=500,
            resolution=0.5,
            density_metric=0.5,
        ),
    ]

    # Test load balancing for normal case
    adjustment, metrics = load_balancer.balance_load(current, neighbors)

    # For normal imbalance
    assert adjustment < 0  # Should decrease load since current is higher
    assert abs(adjustment) <= 0.5  # Allow larger adjustments for imbalanced cases
    assert metrics["status"] == "decreasing"

    # Test 2: Extreme imbalance case
    current_extreme = LoadProfile(
        compute_cost=0.9,
        memory_usage=1000,
        resolution=1.0,
        density_metric=0.5,
    )
    neighbors_extreme = [
        LoadProfile(
            compute_cost=0.1,
            memory_usage=500,
            resolution=0.5,
            density_metric=0.5,
        ),
    ]

    # Test load balancing for extreme case
    adjustment_extreme, metrics_extreme = load_balancer.balance_load(
        current_extreme,
        neighbors_extreme,
    )

    # For extreme imbalance (>0.5), the adjustment can exceed max_adjustment
    assert adjustment_extreme < 0  # Should decrease load
    assert metrics_extreme["load_imbalance"] > 0.5  # Verify it's an extreme case

    # Test 3: Perfectly balanced case
    balanced = LoadProfile(
        compute_cost=0.5,
        memory_usage=1000,
        resolution=1.0,
        density_metric=0.5,
    )
    balanced_neighbors = [
        LoadProfile(
            compute_cost=0.5,
            memory_usage=1000,
            resolution=1.0,
            density_metric=0.5,
        ),
        LoadProfile(
            compute_cost=0.5,
            memory_usage=1000,
            resolution=1.0,
            density_metric=0.5,
        ),
    ]
    (
        adjustment_balanced,
        metrics_balanced,
    ) = load_balancer.balance_load(balanced, balanced_neighbors)

    # For perfectly balanced case
    assert abs(adjustment_balanced) < 1e-6  # Should be very close to zero
    assert abs(metrics_balanced["load_imbalance"]) < 1e-6  # Should be very close to zero
    assert metrics_balanced["avg_load"] == balanced.compute_cost  # Average should match
    assert metrics_balanced["local_load"] == balanced.compute_cost  # Local should match


def test_load_metrics(load_balancer: LoadBalancer) -> None:
    """Test load metrics calculation."""
    # Create test profiles with imbalanced loads
    current = LoadProfile(
        compute_cost=0.8,
        memory_usage=1000,
        resolution=1.0,
        density_metric=0.5,
    )
    neighbors = [
        LoadProfile(
            compute_cost=0.2,
            memory_usage=500,
            resolution=0.5,
            density_metric=0.5,
        ),
    ]

    # Get balancing metrics
    _, metrics = load_balancer.balance_load(current, neighbors)

    # Verify metrics
    assert "local_load" in metrics
    assert "avg_load" in metrics
    assert "load_imbalance" in metrics
    assert "resolution_delta" in metrics
    assert "status" in metrics

    # Verify metric values
    assert metrics["local_load"] == 0.8
    assert abs(metrics["avg_load"] - 0.5) < 1e-6  # Average of 0.8 and 0.2
    assert metrics["load_imbalance"] > 0.0  # Should indicate imbalance
    assert metrics["resolution_delta"] < 0  # Should decrease resolution
