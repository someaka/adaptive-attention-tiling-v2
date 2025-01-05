"""Tests for reaction-diffusion dynamics."""

import torch
import pytest

from tests.test_neural.test_attention.test_pattern.conftest import assert_mass_conserved


def test_reaction_diffusion(pattern_system, grid_size, batch_size):
    """Test reaction-diffusion dynamics."""
    # Create initial state (Turing-like pattern)
    state = torch.randn(batch_size, 2, grid_size, grid_size)  # 2 species

    # Define diffusion tensor (different rates for species)
    diffusion_tensor = torch.tensor([[0.1, 0.0], [0.0, 0.1]])  # Same diffusion rate for both species

    # Define reaction term (activator-inhibitor)
    def reaction_term(state, param=None):
        u, v = state[:, 0:1], state[:, 1:2]  # Keep dimensions
        # Mass-conserving reaction terms
        du = -0.5 * (u - v)  # Half of the difference
        dv = 0.5 * (u - v)   # Half of the difference with opposite sign
        return torch.cat([du, dv], dim=1)  # Preserve dimensions

    # Evolve system
    evolved = pattern_system.reaction_diffusion(
        state, 
        reaction_term,
        diffusion_tensor,  # Now passed as param
        diffusion_coefficient=0.1
    )

    # Test mass conservation
    initial_total_mass = state.sum()
    evolved_total_mass = evolved.sum()
    assert torch.allclose(initial_total_mass, evolved_total_mass, rtol=1e-4), "Total mass should be conserved"

    # Test positivity preservation (if applicable)
    positive_state = torch.abs(state)
    evolved_positive = pattern_system.reaction_diffusion(
        positive_state, 
        reaction_term,
        diffusion_tensor,  # Now passed as param
        diffusion_coefficient=0.1
    )
    assert torch.all(evolved_positive >= 0), "Positivity should be preserved"

    # Test pattern formation
    time_evolution = pattern_system.evolve_pattern(
        state,
        diffusion_coefficient=0.1,
        reaction_term=reaction_term,
        steps=100
    )
    assert pattern_system.detect_pattern_formation(
        time_evolution
    ), "Should form stable patterns"
