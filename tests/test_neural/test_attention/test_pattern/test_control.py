"""Tests for pattern control mechanisms."""

import torch
import pytest

from src.neural.attention.pattern.models import ControlSignal
from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal


def test_pattern_control(pattern_system, grid_size):
    """Test pattern control mechanisms."""
    # Create current and target patterns
    current = torch.randn(1, 2, grid_size, grid_size)
    target = torch.randn(1, 2, grid_size, grid_size)

    # Define constraints
    constraints = [
        lambda x: torch.mean(x) - 1.0,  # Mean constraint
        lambda x: torch.var(x) - 0.1,  # Variance constraint
    ]

    # Compute control signal
    control = pattern_system.pattern_control(current, target, constraints)

    # Test control signal properties
    assert isinstance(control, ControlSignal), "Should return control signal"
    assert control.magnitude is not None, "Should compute control magnitude"
    assert control.direction is not None, "Should compute control direction"

    # Test constraint satisfaction
    controlled_state = current + control.magnitude * control.direction
    for constraint in constraints:
        assert abs(constraint(controlled_state)) < 1e-2, "Should satisfy constraints"

    # Test control effectiveness
    distance_before = torch.norm(target - current)
    distance_after = torch.norm(target - controlled_state)
    assert distance_after < distance_before, "Control should reduce distance to target"


def test_spatiotemporal_evolution(pattern_system, grid_size):
    """Test spatiotemporal pattern evolution."""
    # Create initial pattern
    initial = torch.randn(1, 2, grid_size, grid_size)

    # Define space-time coupling term
    def coupling(x, t):
        """Space-time coupling term."""
        return 0.1 * torch.sin(2 * torch.pi * t) * x

    # Evolve pattern
    evolution = pattern_system.evolve_spatiotemporal(
        initial, coupling, steps=100
    )

    # Test evolution properties
    assert len(evolution) == 101, "Should return all time steps"
    assert all(state.shape == initial.shape for state in evolution), "Should preserve shape"

    # Test temporal coherence
    time_diffs = []
    for t in range(len(evolution)-1):
        diff = torch.norm(evolution[t+1] - evolution[t])
        time_diffs.append(diff.item())
    assert max(time_diffs) < 1.0, "Evolution should be temporally coherent"

    # Test spatial coherence
    for state in evolution:
        spatial_diff = torch.norm(state[:, :, 1:, :] - state[:, :, :-1, :])
        assert spatial_diff < 1.0, "Evolution should be spatially coherent"
