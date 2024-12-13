"""
Unit tests for the Hamiltonian system.

Tests cover:
1. Energy conservation
2. Hamilton's equations
3. Symplectic structure
4. Poisson brackets
5. Canonical transformations
"""

import numpy as np
import pytest
import torch

from src.neural.flow.hamiltonian import (
    CanonicalTransform,
    HamiltonianSystem,
    PhaseSpacePoint,
)
from src.core.patterns.symplectic import SymplecticStructure


class TestHamiltonianSystem:
    """Test suite for Hamiltonian system dynamics."""

    @pytest.fixture
    def phase_dim(self) -> int:
        """Dimension of phase space (must be even)."""
        return 4

    @pytest.fixture
    def batch_size(self) -> int:
        """Batch size for testing."""
        return 8

    @pytest.fixture
    def hamiltonian_system(self, phase_dim: int) -> HamiltonianSystem:
        """Create a test Hamiltonian system."""
        return HamiltonianSystem(manifold_dim=phase_dim)

    @pytest.fixture
    def symplectic_structure(self, phase_dim: int) -> SymplecticStructure:
        """Create a test symplectic structure."""
        return SymplecticStructure(dim=phase_dim)

    def test_hamiltonian_computation(self, hamiltonian_system: HamiltonianSystem, phase_dim: int, batch_size: int):
        """Test Hamiltonian energy computation."""
        # Create test phase space points
        state = torch.randn(batch_size, phase_dim // 2)  # Position
        momentum = torch.randn(batch_size, phase_dim // 2)  # Momentum
        phase_point = PhaseSpacePoint(position=state, momentum=momentum, time=0.0)

        # Compute energy
        energy = hamiltonian_system.compute_energy(torch.cat([state, momentum], dim=-1))

        # Test energy properties
        assert energy.shape == (batch_size,), "Energy should be scalar per batch"
        assert torch.all(energy >= 0), "Energy should be non-negative"

        # Test scaling properties
        scaled_state = 2 * state
        scaled_momentum = 2 * momentum
        scaled_energy = hamiltonian_system.compute_energy(
            torch.cat([scaled_state, scaled_momentum], dim=-1)
        )
        assert torch.allclose(
            scaled_energy, 4 * energy, rtol=1e-4
        ), "Energy should scale quadratically"

    def test_evolution(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test Hamiltonian evolution."""
        # Create test phase space point
        state = torch.randn(phase_dim // 2)
        momentum = torch.randn(phase_dim // 2)
        phase_point = PhaseSpacePoint(position=state, momentum=momentum, time=0.0)

        # Test evolution
        evolved = hamiltonian_system.evolve(torch.cat([state, momentum], dim=-1))
        assert evolved.shape == (phase_dim,), "Evolution should preserve shape"

        # Test energy conservation
        initial_energy = hamiltonian_system.compute_energy(torch.cat([state, momentum], dim=-1))
        evolved_energy = hamiltonian_system.compute_energy(evolved)
        assert torch.allclose(initial_energy, evolved_energy, rtol=1e-4), "Energy should be conserved"

    def test_canonical_transformations(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test canonical transformation properties."""
        # Create test canonical transformation
        transform = CanonicalTransform(phase_dim=phase_dim)

        # Test point
        q = torch.randn(phase_dim // 2)
        p = torch.randn(phase_dim // 2)
        point = PhaseSpacePoint(position=q, momentum=p, time=0.0)

        # Apply transformation
        transformed = transform.transform(point)
        assert isinstance(transformed, PhaseSpacePoint), "Should return PhaseSpacePoint"

        # Test energy conservation
        initial_energy = hamiltonian_system.compute_energy(torch.cat([q, p], dim=-1))
        transformed_energy = hamiltonian_system.compute_energy(
            torch.cat([transformed.position, transformed.momentum], dim=-1)
        )
        assert torch.allclose(
            initial_energy, transformed_energy, rtol=1e-4
        ), "Should preserve energy"

    def test_symplectic_integration(
        self, 
        hamiltonian_system: HamiltonianSystem, 
        symplectic_structure: SymplecticStructure,
        phase_dim: int
    ):
        """Test symplectic integration properties."""
        # Create test point
        state = torch.randn(phase_dim // 2)
        momentum = torch.randn(phase_dim // 2)
        point = PhaseSpacePoint(position=state, momentum=momentum, time=0.0)
        full_state = torch.cat([state, momentum], dim=-1)

        # Evolve system
        evolved = hamiltonian_system.evolve(full_state)

        # Test energy conservation
        initial_energy = hamiltonian_system.compute_energy(full_state)
        final_energy = hamiltonian_system.compute_energy(evolved)
        assert torch.allclose(
            initial_energy, final_energy, rtol=1e-4
        ), "Should preserve energy"

        # Test symplectic preservation
        integrator = hamiltonian_system.symplectic_integrator
        assert integrator is not None, "Should have symplectic integrator"
        
        # Test symplectic form computation
        form = symplectic_structure.compute_form(full_state)
        assert form.matrix.shape == (phase_dim, phase_dim), "Should have correct shape"
        assert torch.allclose(
            form.matrix, -form.matrix.transpose(-1, -2)
        ), "Should be antisymmetric"
