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
        return HamiltonianSystem(dim=phase_dim, integrator="symplectic", order=4)

    def test_hamiltonian_computation(self, hamiltonian_system: HamiltonianSystem, phase_dim: int, batch_size: int):
        """Test Hamiltonian energy computation."""
        # Create test phase space points
        state = torch.randn(batch_size, phase_dim // 2)  # Position
        momentum = torch.randn(batch_size, phase_dim // 2)  # Momentum

        # Compute Hamiltonian (e.g., harmonic oscillator)
        energy = hamiltonian_system.compute_hamiltonian(state, momentum)

        # Test energy properties
        assert energy.shape == (batch_size,), "Energy should be scalar per batch"
        assert torch.all(energy >= 0), "Energy should be non-negative"

        # Test scaling properties
        scaled_state = 2 * state
        scaled_momentum = 2 * momentum
        scaled_energy = hamiltonian_system.compute_hamiltonian(
            scaled_state, scaled_momentum
        )
        assert torch.allclose(
            scaled_energy, 4 * energy, rtol=1e-4
        ), "Energy should scale quadratically"

        # Test separable Hamiltonian structure
        kinetic = hamiltonian_system.compute_kinetic(momentum)
        potential = hamiltonian_system.compute_potential(state)
        total = hamiltonian_system.compute_hamiltonian(state, momentum)
        assert torch.allclose(
            total, kinetic + potential, rtol=1e-5
        ), "Hamiltonian should be separable"

    def test_hamilton_equations(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test Hamilton's equations of motion."""
        # Create test phase space point
        state = torch.randn(phase_dim // 2)
        momentum = torch.randn(phase_dim // 2)

        # Compute equations of motion
        dstate_dt, dmomentum_dt = hamiltonian_system.hamilton_equations(state, momentum)

        # Test equation properties
        assert dstate_dt.shape == state.shape, "State evolution should preserve shape"
        assert (
            dmomentum_dt.shape == momentum.shape
        ), "Momentum evolution should preserve shape"

        # Test canonical relations
        def test_canonical_relations(q: torch.Tensor, p: torch.Tensor) -> bool:
            """Test canonical Hamilton equations."""
            dq_dt, dp_dt = hamiltonian_system.hamilton_equations(q, p)
            hamiltonian = hamiltonian_system.compute_hamiltonian(q, p)

            # ∂H/∂p = dq/dt, -∂H/∂q = dp/dt
            dham_dp = torch.autograd.grad(hamiltonian.sum(), p, create_graph=True)[0]
            dham_dq = torch.autograd.grad(hamiltonian.sum(), q, create_graph=True)[0]

            return torch.allclose(dq_dt, dham_dp, rtol=1e-4) and torch.allclose(
                dp_dt, -dham_dq, rtol=1e-4
            )

        assert test_canonical_relations(
            state, momentum
        ), "Should satisfy canonical relations"

        # Test time evolution
        trajectory = hamiltonian_system.evolve_trajectory(
            state, momentum, time_span=[0, 1], steps=100
        )
        assert len(trajectory) > 0, "Should compute trajectory"

        # Test energy conservation
        energies = [hamiltonian_system.compute_hamiltonian(s, p) for s, p in trajectory]
        assert torch.allclose(
            torch.stack(energies), energies[0].expand(len(energies)), rtol=1e-4
        ), "Energy should be conserved"

    def test_symplectic_form(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test symplectic structure properties."""
        # Create test vectors
        v1 = torch.randn(phase_dim)
        v2 = torch.randn(phase_dim)

        # Compute symplectic form
        omega = hamiltonian_system.symplectic_form(v1, v2)

        # Test antisymmetry
        omega_reverse = hamiltonian_system.symplectic_form(v2, v1)
        assert torch.allclose(
            omega, -omega_reverse, rtol=1e-5
        ), "Symplectic form should be antisymmetric"

        # Test non-degeneracy
        def test_non_degeneracy(v: torch.Tensor) -> bool:
            """Test if vector pairs with zero symplectic product."""
            return not torch.all(
                torch.abs(hamiltonian_system.symplectic_form(v, v2)) < 1e-5
                for v2 in torch.randn(10, phase_dim)
            )

        assert test_non_degeneracy(v1), "Symplectic form should be non-degenerate"

        # Test Jacobi identity
        def test_jacobi_identity(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> bool:
            """Test Jacobi identity for Poisson bracket."""
            bracket12 = hamiltonian_system.poisson_bracket(
                lambda x: x @ v1, lambda x: x @ v2
            )
            bracket23 = hamiltonian_system.poisson_bracket(
                lambda x: x @ v2, lambda x: x @ v3
            )
            bracket31 = hamiltonian_system.poisson_bracket(
                lambda x: x @ v3, lambda x: x @ v1
            )

            x = torch.randn(phase_dim)
            sum_cyclic = bracket12(x) @ v3 + bracket23(x) @ v1 + bracket31(x) @ v2
            return torch.abs(sum_cyclic) < 1e-4

        v3 = torch.randn(phase_dim)
        assert test_jacobi_identity(v1, v2, v3), "Should satisfy Jacobi identity"

    def test_poisson_bracket(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test Poisson bracket properties."""

        # Create test observables
        def f(x: torch.Tensor) -> torch.Tensor:
            """Test observable function."""
            return torch.sum(x**2)

        def g(x: torch.Tensor) -> torch.Tensor:
            """Test observable function."""
            return torch.prod(torch.cos(x))

        def h(x: torch.Tensor) -> torch.Tensor:
            """Test observable function."""
            return torch.sum(torch.exp(-(x**2)))

        # Test point
        x = torch.randn(phase_dim)

        # Test antisymmetry
        bracket_fg = hamiltonian_system.poisson_bracket(f, g)(x)
        bracket_gf = hamiltonian_system.poisson_bracket(g, f)(x)
        assert torch.allclose(
            bracket_fg, -bracket_gf, rtol=1e-4
        ), "Poisson bracket should be antisymmetric"

        # Test Leibniz rule
        def test_leibniz(f: callable, g: callable, h: callable, x: torch.Tensor) -> bool:
            """Test Leibniz rule for Poisson bracket."""
            bracket_fgh = hamiltonian_system.poisson_bracket(f, lambda x: g(x) * h(x))(x)
            bracket_fg_h = g(x) * hamiltonian_system.poisson_bracket(f, h)(x)
            bracket_fh_g = h(x) * hamiltonian_system.poisson_bracket(f, g)(x)
            return torch.allclose(bracket_fgh, bracket_fg_h + bracket_fh_g, rtol=1e-4)

        assert test_leibniz(f, g, h, x), "Should satisfy Leibniz rule"

        # Test Jacobi identity
        def test_poisson_jacobi(f: callable, g: callable, h: callable, x: torch.Tensor) -> bool:
            """Test Jacobi identity for Poisson bracket."""

            def bracket_fg(x):
                return hamiltonian_system.poisson_bracket(f, g)(x)

            def bracket_gh(x):
                return hamiltonian_system.poisson_bracket(g, h)(x)

            def bracket_hf(x):
                return hamiltonian_system.poisson_bracket(h, f)(x)

            sum_cyclic = (
                hamiltonian_system.poisson_bracket(f, bracket_gh)(x)
                + hamiltonian_system.poisson_bracket(g, bracket_hf)(x)
                + hamiltonian_system.poisson_bracket(h, bracket_fg)(x)
            )
            return torch.abs(sum_cyclic) < 1e-4

        assert test_poisson_jacobi(f, g, h, x), "Should satisfy Jacobi identity"

    def test_canonical_transformations(self, hamiltonian_system: HamiltonianSystem, phase_dim: int):
        """Test canonical transformation properties."""

        # Create test canonical transformation
        def canonical_map(q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Simple canonical transformation (rotation in phase space)."""
            theta = torch.tensor(np.pi / 4)
            Q = q * torch.cos(theta) - p * torch.sin(theta)
            P = q * torch.sin(theta) + p * torch.cos(theta)
            return Q, P

        transform = CanonicalTransform(canonical_map)

        # Test symplectic property
        def test_symplectic_preservation(q: torch.Tensor, p: torch.Tensor) -> bool:
            """Test if transformation preserves symplectic form."""
            Q, P = transform(q, p)
            omega_old = hamiltonian_system.symplectic_form(
                torch.cat([q, p]), torch.cat([q, p])
            )
            omega_new = hamiltonian_system.symplectic_form(
                torch.cat([Q, P]), torch.cat([Q, P])
            )
            return torch.allclose(omega_old, omega_new, rtol=1e-4)

        q = torch.randn(phase_dim // 2)
        p = torch.randn(phase_dim // 2)
        assert test_symplectic_preservation(q, p), "Should preserve symplectic form"

        # Test energy conservation
        H_before = hamiltonian_system.compute_hamiltonian(q, p)
        Q, P = transform(q, p)
        H_after = hamiltonian_system.compute_hamiltonian(Q, P)
        assert torch.allclose(H_before, H_after, rtol=1e-4), "Should preserve energy"

        # Test canonical equations in new coordinates
        dQ_dt, dP_dt = hamiltonian_system.hamilton_equations(Q, P)
        assert test_canonical_relations(Q, P), "Should preserve canonical relations"

    def test_energy_conservation(self, hamiltonian_system: HamiltonianSystem, batch_size: int, phase_dim: int):
        """Test energy conservation in Hamiltonian dynamics."""
        # Initialize phase space point
        state = hamiltonian_system.initialize_state(batch_size, phase_dim)

        # Test Hamiltonian computation
        def test_hamiltonian() -> None:
            """Test Hamiltonian function properties."""
            H = hamiltonian_system.compute_hamiltonian(state)
            assert H.shape == (batch_size,), "Should be scalar"
            # Test positivity for mechanical systems
            if hamiltonian_system.is_mechanical:
                assert torch.all(H >= 0), "Mechanical energy should be positive"
            return H

        test_hamiltonian()

        # Test Hamilton's equations
        def test_equations_of_motion() -> None:
            """Test Hamilton's equations."""
            dH = hamiltonian_system.compute_hamilton_equations(state)
            assert dH.shape == state.shape, "Should match state shape"
            # Test symplectic structure
            omega = hamiltonian_system.get_symplectic_form()
            assert torch.allclose(
                dH, omega @ hamiltonian_system.compute_gradient(state)
            ), "Should satisfy Hamilton's equations"
            return dH

        test_equations_of_motion()

        # Test energy conservation
        def test_conservation(state: torch.Tensor, time: float = 10.0) -> None:
            """Test energy conservation along flow."""
            trajectory = hamiltonian_system.evolve_state(state, final_time=time)
            energies = [hamiltonian_system.compute_hamiltonian(s) for s in trajectory]
            assert all(
                torch.allclose(e, energies[0], rtol=1e-3) for e in energies
            ), "Energy should be conserved"
            return trajectory

        test_conservation(state)

    def test_symplectic_structure(self, hamiltonian_system: HamiltonianSystem, batch_size: int, phase_dim: int):
        """Test symplectic geometry properties."""

        # Test symplectic form
        def test_symplectic_form() -> None:
            """Test properties of symplectic form."""
            omega = hamiltonian_system.get_symplectic_form()
            # Test antisymmetry
            assert torch.allclose(
                omega, -omega.transpose(-1, -2)
            ), "Should be antisymmetric"
            # Test non-degeneracy
            assert (
                torch.matrix_rank(omega) == omega.shape[-1]
            ), "Should be non-degenerate"
            return omega

        test_symplectic_form()

        # Test Poisson bracket
        def test_poisson_bracket() -> None:
            """Test Poisson bracket properties."""

            def f(x: torch.Tensor) -> torch.Tensor:
                return hamiltonian_system.compute_observable(x, "f")

            def g(x: torch.Tensor) -> torch.Tensor:
                return hamiltonian_system.compute_observable(x, "g")

            def h(x: torch.Tensor) -> torch.Tensor:
                return hamiltonian_system.compute_observable(x, "h")

            state = hamiltonian_system.initialize_state(batch_size, phase_dim)

            # Test antisymmetry
            assert torch.allclose(
                hamiltonian_system.poisson_bracket(f, g)(state),
                -hamiltonian_system.poisson_bracket(g, f)(state),
            ), "Poisson bracket should be antisymmetric"

            # Test Jacobi identity
            jacobi = (
                hamiltonian_system.poisson_bracket(
                    f, hamiltonian_system.poisson_bracket(g, h)
                )(state)
                + hamiltonian_system.poisson_bracket(
                    g, hamiltonian_system.poisson_bracket(h, f)
                )(state)
                + hamiltonian_system.poisson_bracket(
                    h, hamiltonian_system.poisson_bracket(f, g)
                )(state)
            )
            assert torch.allclose(
                jacobi, torch.zeros_like(jacobi)
            ), "Should satisfy Jacobi identity"

        test_poisson_bracket()

        # Test symplectic integration
        def test_symplectic_integration() -> None:
            """Test symplectic integrator properties."""
            state = hamiltonian_system.initialize_state(batch_size, phase_dim)
            trajectory = hamiltonian_system.symplectic_integrate(state, steps=100)
            # Test symplectic preservation
            for s1, s2 in zip(trajectory[:-1], trajectory[1:]):
                flow_map = hamiltonian_system.compute_flow_map(s1, s2)
                assert hamiltonian_system.is_symplectic(
                    flow_map
                ), "Flow should preserve symplectic form"
            return trajectory

        test_symplectic_integration()

    def test_canonical_transformations(self, hamiltonian_system: HamiltonianSystem, batch_size: int, phase_dim: int):
        """Test canonical transformation properties."""

        # Test generating functions
        def test_generating_functions() -> None:
            """Test different types of generating functions."""
            # Type 1 (q, Q)
            F1 = hamiltonian_system.get_generating_function(1)
            # Type 2 (q, P)
            F2 = hamiltonian_system.get_generating_function(2)
            # Type 3 (p, Q)
            F3 = hamiltonian_system.get_generating_function(3)
            # Type 4 (p, P)
            F4 = hamiltonian_system.get_generating_function(4)

            # Test canonical relations
            state = hamiltonian_system.initialize_state(batch_size, phase_dim)
            for F in [F1, F2, F3, F4]:
                transform = hamiltonian_system.compute_canonical_transform(F)
                transform(state)
                assert hamiltonian_system.is_canonical(
                    transform
                ), "Transform should be canonical"
            return F1, F2, F3, F4

        F1, F2, F3, F4 = test_generating_functions()

        # Test canonical invariants
        def test_invariants() -> None:
            """Test invariance of canonical quantities."""
            state = hamiltonian_system.initialize_state(batch_size, phase_dim)
            transform = hamiltonian_system.get_canonical_transform()
            new_state = transform(state)

            # Test Poincaré integral invariants
            for k in range(1, phase_dim // 2 + 1):
                old_invariant = hamiltonian_system.compute_poincare_invariant(state, k)
                new_invariant = hamiltonian_system.compute_poincare_invariant(
                    new_state, k
                )
                assert torch.allclose(
                    old_invariant, new_invariant, rtol=1e-3
                ), f"Should preserve {k}-form"

            # Test action variables
            old_actions = hamiltonian_system.compute_action_variables(state)
            new_actions = hamiltonian_system.compute_action_variables(new_state)
            assert torch.allclose(
                old_actions, new_actions, rtol=1e-3
            ), "Should preserve actions"

        test_invariants()

        # Test canonical perturbation theory
        def test_perturbation() -> None:
            """Test canonical perturbation methods."""
            H0 = hamiltonian_system.get_unperturbed_hamiltonian()
            V = hamiltonian_system.get_perturbation()

            # Test averaging principle
            avg_H = hamiltonian_system.compute_averaged_hamiltonian(H0, V)
            assert avg_H is not None, "Should compute average"

            # Test normal form
            normal_form = hamiltonian_system.compute_normal_form(H0, V)
            assert normal_form is not None, "Should compute normal form"

            # Test KAM tori
            if hamiltonian_system.is_integrable(H0):
                tori = hamiltonian_system.find_kam_tori(H0, V)
                assert len(tori) > 0, "Should find KAM tori"

        test_perturbation()
