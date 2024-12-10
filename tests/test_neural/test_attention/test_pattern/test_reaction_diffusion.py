"""Tests for reaction-diffusion dynamics."""

import torch
import pytest

from tests.test_neural.test_attention.test_pattern.conftest import assert_mass_conserved


def test_reaction_diffusion(pattern_system, grid_size, batch_size):
    """Test reaction-diffusion dynamics."""
    # Create initial state (Turing-like pattern)
    state = torch.randn(batch_size, 2, grid_size, grid_size)  # 2 species

    # Define diffusion tensor (different rates for species)
    diffusion_tensor = torch.tensor([[0.1, 0.0], [0.0, 0.05]])

    # Define reaction term (activator-inhibitor)
    def reaction_term(state):
        u, v = state[:, 0:1], state[:, 1:2]  # Keep dimensions
        # Mass-conserving reaction terms
        du = u * v / (1 + u**2) - u  # Saturating reaction
        dv = u**2 / (1 + u**2) - v   # Balancing term
        return torch.cat([du, dv], dim=1)  # Preserve dimensions

    # Evolve system
    evolved = pattern_system.reaction_diffusion(
        state, diffusion_tensor, reaction_term
    )

    # Test mass conservation
    assert_mass_conserved(state, evolved)

    # Test positivity preservation (if applicable)
    positive_state = torch.abs(state)
    evolved_positive = pattern_system.reaction_diffusion(
        positive_state, diffusion_tensor, reaction_term
    )
    assert torch.all(evolved_positive >= 0), "Positivity should be preserved"

    # Test pattern formation
    time_evolution = pattern_system.evolve_pattern(
        state, diffusion_tensor, reaction_term, steps=100
    )
    assert pattern_system.detect_pattern_formation(
        time_evolution
    ), "Should form stable patterns"
