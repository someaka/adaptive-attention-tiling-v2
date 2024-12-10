"""Tests for pattern stability analysis."""

import torch
import pytest

from src.neural.attention.pattern.models import StabilityMetrics
from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal, assert_mass_conserved


def test_stability_analysis(pattern_system, grid_size):
    """Test pattern stability analysis."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Create perturbation
    perturbation = 0.1 * torch.randn_like(pattern)

    # Analyze stability
    metrics = pattern_system.stability_analysis(pattern, perturbation)

    # Test metric properties
    assert isinstance(metrics, StabilityMetrics), "Should return stability metrics"
    assert metrics.linear_stability is not None, "Should compute linear stability"
    assert metrics.nonlinear_stability is not None, "Should compute nonlinear stability"

    # Test Lyapunov spectrum
    lyapunov_spectrum = pattern_system.compute_lyapunov_spectrum(pattern)
    assert len(lyapunov_spectrum) > 0, "Should compute Lyapunov exponents"
    assert lyapunov_spectrum.dtype == torch.float64, "Lyapunov exponents should be real"

    # Test structural stability
    def perturbed_reaction(state):
        """Slightly perturbed reaction term."""
        u, v = state[:, 0], state[:, 1]
        du = u**2 * v - u + 0.1 * u**3
        dv = u**2 - v + 0.1 * v**3
        return torch.stack([du, dv], dim=1)

    structural_stability = pattern_system.test_structural_stability(
        pattern, perturbed_reaction
    )
    assert structural_stability > 0, "Should be structurally stable"
