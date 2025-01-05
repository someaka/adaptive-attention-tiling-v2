"""
Unit tests for the Hamiltonian system.

Tests cover:
1. Energy conservation
2. Hamilton's equations
3. Symplectic structure
4. Canonical transformations
"""

import numpy as np
import pytest
import torch

from src.neural.flow.hamiltonian import (
    HamiltonianSystem,
    PhaseSpacePoint,
    CanonicalTransform
)


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

    def test_hamiltonian_computation(self, hamiltonian_system: HamiltonianSystem, phase_dim: int, batch_size: int):
        """Test Hamiltonian energy computation and conservation."""
        print("\n=== Testing Hamiltonian Energy Computation ===")
        
        # Create test phase space points
        points = torch.randn(batch_size, phase_dim)
        points.requires_grad_(True)
        
        print("\nTest points details:")
        print(f"Shape: {points.shape}")
        print(f"Mean: {points.mean():.4f}, std: {points.std():.4f}")
        print(f"Min: {points.min():.4f}, max: {points.max():.4f}")
        print(f"Requires grad: {points.requires_grad}")
        
        # Compute energy
        print("\nComputing initial energy...")
        energy = hamiltonian_system.compute_energy(points)
        
        # Test energy properties
        print("\nEnergy properties:")
        print(f"Shape: {energy.shape}")
        print(f"Mean: {energy.mean():.4f}, std: {energy.std():.4f}")
        print(f"Min: {energy.min():.4f}, max: {energy.max():.4f}")
        
        assert energy.shape == (batch_size,), "Energy should be scalar per batch"
        assert torch.all(energy >= 0), "Energy should be non-negative"
        
        # Test gradient computation
        print("\nComputing energy gradients...")
        grad = torch.autograd.grad(energy.sum(), points, create_graph=True)[0]
        
        print("\nGradient properties:")
        print(f"Shape: {grad.shape}")
        print(f"Mean: {grad.mean():.4f}, std: {grad.std():.4f}")
        print(f"Min: {grad.min():.4f}, max: {grad.max():.4f}")
        
        assert grad.shape == points.shape, "Gradient should match input shape"
        
        # Test scaling properties - should scale quadratically
        print("\nTesting quadratic scaling...")
        scaled_points = 2.0 * points
        scaled_energy = hamiltonian_system.compute_energy(scaled_points)
        ratio = scaled_energy / energy
        
        print("\nScaling test results:")
        print(f"Original energy mean: {energy.mean():.4f}")
        print(f"Scaled energy mean: {scaled_energy.mean():.4f}")
        print(f"Ratio mean: {ratio.mean():.4f}, std: {ratio.std():.4f}")
        print(f"Expected ratio: 4.0")
        print(f"Ratio error: {(ratio - 4.0).abs().mean():.4e}")
        
        assert torch.allclose(ratio, torch.tensor(4.0), rtol=1e-2), f"Energy should scale quadratically, got ratio {ratio}"

    def test_hamiltons_equations(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test Hamilton's equations of motion."""
        # Create test phase space point
        point = torch.randn(phase_dim)
        point.requires_grad_(True)

        # Compute Hamiltonian vector field
        vector_field = hamiltonian_system.compute_vector_field(point)
        
        # Split into position and momentum components
        dq_dt = vector_field[:phase_dim//2]  # Position derivatives
        dp_dt = vector_field[phase_dim//2:]  # Momentum derivatives
        
        # Compute energy gradients
        energy = hamiltonian_system.compute_energy(point)
        grad = torch.autograd.grad(energy, point, create_graph=True)[0]
        grad_q = grad[:phase_dim//2]  # ∂H/∂q
        grad_p = grad[phase_dim//2:]  # ∂H/∂p
        
        # Verify Hamilton's equations
        # dq/dt = ∂H/∂p
        assert torch.allclose(dq_dt, grad_p, rtol=1e-2), "Position evolution violates Hamilton's equations"
        # dp/dt = -∂H/∂q
        assert torch.allclose(dp_dt, -grad_q, rtol=1e-2), "Momentum evolution violates Hamilton's equations"

    def test_symplectic_properties(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test preservation of symplectic structure."""
        # Create test phase space points
        points = torch.randn(phase_dim)
        points.requires_grad_(True)

        # Evolve system
        evolved = hamiltonian_system.evolve(points)
        
        # Compute Jacobian of the evolution
        def compute_jacobian(x: torch.Tensor) -> torch.Tensor:
            """Helper to compute Jacobian as tensor."""
            jac = torch.autograd.functional.jacobian(
                lambda x: hamiltonian_system.evolve(x),
                x
            )
            # Convert from tuple if needed and ensure tensor
            if isinstance(jac, tuple):
                jac = torch.stack(jac)
            return jac
        
        jacobian = compute_jacobian(points)
        
        # Construct standard symplectic matrix J
        J = torch.zeros(phase_dim, phase_dim)
        n = phase_dim // 2
        J[:n, n:] = torch.eye(n)
        J[n:, :n] = -torch.eye(n)
        
        # Check symplectic condition: J^T M J = J
        M = torch.matmul(torch.matmul(jacobian.transpose(-2, -1), J), jacobian)
        assert torch.allclose(M, J, rtol=1e-2), "Evolution does not preserve symplectic structure"
        
        # Check volume preservation (Liouville's theorem)
        det = torch.det(jacobian)
        assert torch.allclose(det, torch.tensor(1.0), rtol=1e-2), f"Phase space volume not preserved, det = {det}"

    def test_canonical_transformations(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test canonical transformation properties."""
        print("\n=== Testing Canonical Transformations ===")
        
        # Create test canonical transformation
        transform = CanonicalTransform(
            phase_dim=phase_dim,
            hamiltonian_system=hamiltonian_system
        )
        
        # Create test phase space point
        point = torch.randn(phase_dim)
        point.requires_grad_(True)
        
        print("\nTest point details:")
        print(f"Shape: {point.shape}")
        print(f"Mean: {point.mean():.4f}, std: {point.std():.4f}")
        print(f"Min: {point.min():.4f}, max: {point.max():.4f}")
        print(f"Requires grad: {point.requires_grad}")
        
        # Convert to phase space point for transformation
        phase_point = PhaseSpacePoint(
            position=point[:phase_dim//2],
            momentum=point[phase_dim//2:],
            time=0.0
        )
        
        print("\nPhase space point details:")
        print(f"Position shape: {phase_point.position.shape}")
        print(f"Momentum shape: {phase_point.momentum.shape}")
        print(f"Position norm: {torch.norm(phase_point.position):.4f}")
        print(f"Momentum norm: {torch.norm(phase_point.momentum):.4f}")
        
        # Apply transformation
        print("\nApplying canonical transformation...")
        transformed = transform.transform(phase_point)
        
        print("\nTransformed point details:")
        print(f"Position shape: {transformed.position.shape}")
        print(f"Momentum shape: {transformed.momentum.shape}")
        print(f"Position norm: {torch.norm(transformed.position):.4f}")
        print(f"Momentum norm: {torch.norm(transformed.momentum):.4f}")
        
        # Compute initial and final energies
        initial_state = torch.cat([phase_point.position, phase_point.momentum])
        final_state = torch.cat([transformed.position, transformed.momentum])
        
        print("\nComputing energies...")
        initial_energy = hamiltonian_system.compute_energy(initial_state)
        final_energy = hamiltonian_system.compute_energy(final_state)
        
        print("\nEnergy conservation:")
        print(f"Initial energy: {initial_energy.item():.4f}")
        print(f"Final energy: {final_energy.item():.4f}")
        print(f"Energy difference: {abs(final_energy - initial_energy).item():.4e}")
        
        assert torch.allclose(initial_energy, final_energy, rtol=1e-2), "Energy should be conserved"
