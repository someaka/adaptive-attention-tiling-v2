"""Tests for pattern control mechanisms."""

import torch
import pytest

from src.neural.attention.pattern.models import ControlSignal
from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal


def test_pattern_control(pattern_system, grid_size):
    """Test pattern control mechanisms."""
    # Create current and target patterns with controlled initialization
    current = torch.randn(1, 2, grid_size, grid_size)
    current = current / current.norm()  # Normalize current state
    
    target = torch.randn(1, 2, grid_size, grid_size)
    target = target / target.norm()  # Normalize target state

    # Define constraints with more reasonable bounds
    constraints = [
        lambda x: torch.relu(torch.abs(torch.mean(x)) - 0.5),  # Bound mean magnitude
        lambda x: torch.relu(torch.var(x) - 1.0),  # Upper bound on variance
    ]

    # Initial constraint check
    initial_violations = [constraint(current) for constraint in constraints]

    # Compute control signal
    control = pattern_system.pattern_control(current, target, constraints)

    # Test control signal properties
    assert isinstance(control, ControlSignal), "Should return control signal"
    assert isinstance(control.signal, torch.Tensor), "Should have tensor signal"
    assert control.signal.shape == current.shape, "Control should match state shape"
    assert torch.isfinite(control.signal).all(), "Control should be finite"

    # Apply control and test constraint improvement
    controlled_state = current + control.signal
    
    # Test that constraints are either satisfied or improved
    for i, constraint in enumerate(constraints):
        new_violation = constraint(controlled_state)
        assert new_violation <= initial_violations[i], "Control should not worsen constraints"

    # Test control moves toward target while respecting constraints
    distance_before = torch.norm(target - current)
    distance_after = torch.norm(target - controlled_state)
    
    # The control should either improve the distance to target or maintain constraints
    assert (distance_after < distance_before) or all(constraint(controlled_state) < 1e-2 for constraint in constraints), \
        "Control should either reduce distance to target or maintain constraints"

    # Test numerical properties
    assert torch.isfinite(controlled_state).all(), "Controlled state should be finite"
    assert not torch.isnan(controlled_state).any(), "Controlled state should not have NaN values"
    assert not torch.isinf(controlled_state).any(), "Controlled state should not have Inf values"


def test_spatiotemporal_evolution(pattern_system, grid_size):
    """Test spatiotemporal pattern evolution."""
    # Create initial pattern
    initial = torch.randn(1, 2, grid_size, grid_size)
    initial = initial / initial.norm()  # Normalize initial state

    # Define space-time coupling term
    def coupling(x, t):
        """Space-time coupling term."""
        t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)
        return 0.1 * torch.sin(2 * torch.pi * t_tensor) * x

    # Evolve pattern
    evolution = pattern_system.evolve_spatiotemporal(
        initial, coupling, steps=100
    )

    # Test evolution properties
    assert len(evolution) == 101, "Should return all time steps"
    assert all(state.shape == initial.shape for state in evolution), "Should preserve shape"

    # Test temporal coherence with more realistic bounds
    time_diffs = []
    for t in range(len(evolution)-1):
        diff = torch.norm(evolution[t+1] - evolution[t])
        time_diffs.append(diff.item())
    
    max_diff = max(time_diffs)
    avg_diff = sum(time_diffs) / len(time_diffs)
    assert max_diff < 2.0, f"Evolution should be temporally coherent (max diff: {max_diff:.3f})"
    assert avg_diff < 0.5, f"Average temporal difference should be small (avg: {avg_diff:.3f})"

    # Test spatial coherence
    spatial_diffs = []
    for state in evolution:
        spatial_diff = torch.norm(state[:, :, 1:, :] - state[:, :, :-1, :])
        spatial_diffs.append(spatial_diff.item())
        
        # Test state properties
        assert torch.isfinite(state).all(), "States should be finite"
        assert not torch.isnan(state).any(), "States should not have NaN values"
        assert not torch.isinf(state).any(), "States should not have Inf values"
    
    max_spatial = max(spatial_diffs)
    avg_spatial = sum(spatial_diffs) / len(spatial_diffs)
    assert max_spatial < 2.0, f"Evolution should be spatially coherent (max diff: {max_spatial:.3f})"
    assert avg_spatial < 0.5, f"Average spatial difference should be small (avg: {avg_spatial:.3f})"
