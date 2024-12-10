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
    assert isinstance(
        diagram, BifurcationDiagram
    ), "Should return bifurcation diagram"
    assert len(diagram.bifurcation_points) > 0, "Should detect bifurcations"

    # Test bifurcation classification
    for point in diagram.bifurcation_points:
        assert point.type in [
            "saddle-node",
            "hopf",
            "pitchfork",
        ], "Should classify bifurcation type"

    # Test normal form computation
    normal_form = pattern_system.compute_normal_form(diagram.bifurcation_points[0])
    assert normal_form is not None, "Should compute normal form"
