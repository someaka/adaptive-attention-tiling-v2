"""Tests for bifurcation analysis."""

import torch
import pytest

from src.neural.attention.pattern.models import BifurcationDiagram
from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal


def test_bifurcation_analysis(pattern_system, grid_size):
    """Test bifurcation analysis."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        du = param * u**2 * v - u
        dv = u**2 - v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Test diagram properties
    assert isinstance(diagram, BifurcationDiagram), "Should return bifurcation diagram"
    assert diagram.bifurcation_points.numel() > 0, "Should detect bifurcations"
    assert diagram.solution_states.shape[0] > 0, "Should have solution states"
    assert diagram.solution_params.shape[0] > 0, "Should have solution parameters"

    # Test that solution parameters are within expected range
    assert torch.all(diagram.solution_params >= 0), "Parameters should be non-negative"
    assert torch.all(diagram.solution_params <= 2), "Parameters should be <= 2"

    # Test that solution states have expected shape
    assert len(diagram.solution_states.shape) == 4, "Solution states should be 4D tensor"
    assert diagram.solution_states.shape[1:] == (2, grid_size, grid_size), \
        "Solution states should have correct spatial dimensions"

    # Test solution states are non-zero
    for state in diagram.solution_states:
        assert not torch.allclose(state, torch.zeros_like(state)), \
            "Solution states should not be zero"

    # Check that solution magnitude increases with parameter
    magnitudes = torch.norm(diagram.solution_states.reshape(diagram.solution_states.shape[0], -1), dim=1)
    diffs = magnitudes[1:] - magnitudes[:-1]
    assert torch.all(diffs >= -1e-6), \
        "Solution magnitude should not decrease significantly"


def test_bifurcation_detection_threshold(pattern_system, grid_size):
    """Test that bifurcation detection threshold is appropriate."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term with known bifurcation
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Pitchfork bifurcation at param = 1
        du = param * u - u**3
        dv = -v  # Simple linear decay
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Should detect the pitchfork bifurcation near param = 1
    bifurcation_params = diagram.bifurcation_points
    assert any(abs(p - 1.0) < 0.1 for p in bifurcation_params), \
        "Should detect bifurcation near param = 1"

    # Check stability changes near bifurcation points
    for param in diagram.bifurcation_points:
        param_idx = torch.argmin(torch.abs(parameter_range - param))
        if param_idx > 0 and param_idx < len(parameter_range) - 1:
            state_before = diagram.solution_states[param_idx - 1]
            state_after = diagram.solution_states[param_idx + 1]
            assert not torch.allclose(state_before, state_after, atol=1e-3), \
                "State should change significantly at bifurcation"


def test_stability_regions(pattern_system, grid_size):
    """Test that stability regions are correctly identified."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term with known stability change
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # System becomes unstable at param = 1
        du = (param - 1) * u
        dv = -v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Check stability changes near bifurcation points
    for param in diagram.bifurcation_points:
        param_idx = torch.argmin(torch.abs(parameter_range - param))
        if param_idx > 0 and param_idx < len(parameter_range) - 1:
            state_before = diagram.solution_states[param_idx - 1]
            state_after = diagram.solution_states[param_idx + 1]
            assert not torch.allclose(state_before, state_after, atol=1e-3), \
                "State should change significantly at bifurcation"


def test_solution_branches(pattern_system, grid_size):
    """Test that solution branches are correctly tracked."""
    # Create test pattern
    pattern = torch.ones(1, 2, grid_size, grid_size)  # Start from uniform state

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term with known solution structure
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Simple linear system with known solution u = param * u_initial
        du = (param - 1) * u
        dv = -v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Test solution states are non-zero
    for state in diagram.solution_states:
        assert not torch.allclose(state, torch.zeros_like(state)), \
            "Solution states should not be zero"

    # Check that solution magnitude increases with parameter
    magnitudes = torch.norm(diagram.solution_states.reshape(diagram.solution_states.shape[0], -1), dim=1)
    diffs = magnitudes[1:] - magnitudes[:-1]
    assert torch.all(diffs >= -1e-6), \
        "Solution magnitude should not decrease significantly"
