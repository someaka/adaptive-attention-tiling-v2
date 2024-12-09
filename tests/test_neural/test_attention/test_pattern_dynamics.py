"""
Unit tests for the pattern dynamics system.

Tests cover:
1. Reaction-diffusion dynamics
2. Pattern stability analysis
3. Bifurcation detection
4. Pattern control mechanisms
5. Spatiotemporal evolution
"""

import numpy as np
import pytest
import torch

from src.neural.attention.pattern_dynamics import (
    BifurcationDiagram,
    ControlSignal,
    PatternDynamics,
    StabilityMetrics,
)


class TestPatternDynamics:
    """Test suite for pattern dynamics."""

    @pytest.fixture
    def space_dim(self) -> int:
        """Spatial dimensions."""
        return 2

    @pytest.fixture
    def grid_size(self) -> int:
        """Grid size per dimension."""
        return 32

    @pytest.fixture
    def batch_size(self) -> int:
        """Batch size for testing."""
        return 8

    @pytest.fixture
    def pattern_system(self, space_dim, grid_size) -> PatternDynamics:
        """Create a test pattern dynamics system."""
        return PatternDynamics(
            dim=space_dim, size=grid_size, dt=0.01, boundary="periodic"
        )

    def test_reaction_diffusion(
        self, pattern_system, grid_size, batch_size
    ) -> None:
        """Test reaction-diffusion dynamics."""
        # Create initial state (Turing-like pattern)
        state = torch.randn(batch_size, 2, grid_size, grid_size)  # 2 species

        # Define diffusion tensor (different rates for species)
        diffusion_tensor = torch.tensor([[0.1, 0.0], [0.0, 0.05]])

        # Define reaction term (activator-inhibitor)
        def reaction_term(state):
            u, v = state[:, 0], state[:, 1]
            du = u**2 * v - u
            dv = u**2 - v
            return torch.stack([du, dv], dim=1)

        # Evolve system
        evolved = pattern_system.reaction_diffusion(
            state, diffusion_tensor, reaction_term
        )

        # Test conservation (if applicable)
        total_mass_initial = state.sum(dim=[2, 3])  # Sum over spatial dimensions
        total_mass_final = evolved.sum(dim=[2, 3])
        assert torch.allclose(
            total_mass_initial, total_mass_final, rtol=1e-4
        ), "Mass should be conserved"

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

    def test_stability_analysis(
        self, pattern_system, grid_size
    ) -> None:
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
        assert (
            metrics.nonlinear_stability is not None
        ), "Should compute nonlinear stability"

        # Test Lyapunov spectrum
        lyapunov_spectrum = pattern_system.compute_lyapunov_spectrum(pattern)
        assert len(lyapunov_spectrum) > 0, "Should compute Lyapunov exponents"
        assert torch.all(
            torch.imag(lyapunov_spectrum) == 0
        ), "Lyapunov exponents should be real"

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

    def test_bifurcation_analysis(
        self, pattern_system, grid_size
    ) -> None:
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

    def test_pattern_control(
        self, pattern_system, grid_size
    ) -> None:
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

        # Test control properties
        assert isinstance(control, ControlSignal), "Should return control signal"
        assert control.shape == current.shape, "Control should match pattern shape"

        # Apply control
        controlled = pattern_system.apply_control(current, control)

        # Test constraint satisfaction
        for constraint in constraints:
            assert (
                torch.abs(constraint(controlled)) < 0.1
            ), "Control should satisfy constraints"

        # Test control optimality
        energy = pattern_system.control_energy(control)
        assert energy < pattern_system.control_energy(
            2 * control
        ), "Control should be energy-optimal"

        # Test reachability
        reachable = pattern_system.test_reachability(current, target)
        assert reachable, "Target should be reachable"

    def test_spatiotemporal_evolution(
        self, pattern_system, grid_size
    ) -> None:
        """Test spatiotemporal pattern evolution."""
        # Create initial condition
        initial = torch.randn(1, 2, grid_size, grid_size)

        # Define space-time coupling
        def coupling(x, t):
            """Space-time coupling term."""
            return 0.1 * torch.sin(2 * np.pi * t) * x

        # Evolve pattern
        evolution = pattern_system.evolve_spatiotemporal(
            initial, coupling, t_span=[0, 10], steps=100
        )

        # Test evolution properties
        assert len(evolution) > 0, "Should compute evolution"
        assert evolution[0].shape == initial.shape, "Should preserve shape"

        # Test space-time symmetries
        symmetries = pattern_system.find_spatiotemporal_symmetries(evolution)
        assert len(symmetries) > 0, "Should find symmetries"

        # Test pattern classification
        pattern_type = pattern_system.classify_pattern(evolution)
        assert pattern_type in [
            "stationary",
            "periodic",
            "quasi-periodic",
            "chaotic",
        ], "Should classify pattern type"

        # Test dimensionality estimation
        embedding_dim = pattern_system.estimate_embedding_dimension(evolution)
        assert embedding_dim > 0, "Should estimate embedding dimension"

    def test_reaction_diffusion_dynamics(
        self, pattern_system, batch_size, grid_size
    ) -> None:
        """Test reaction-diffusion dynamics in pattern formation."""
        # Initialize pattern state
        state = torch.randn(batch_size, 2, grid_size, grid_size)

        # Test diffusion term
        def test_diffusion(state, dt=0.01):
            """Test diffusion component."""
            diffused = pattern_system.apply_diffusion(state, dt)
            # Check mass conservation
            assert torch.allclose(
                diffused.sum(), state.sum(), rtol=1e-4
            ), "Diffusion should conserve mass"
            # Check smoothing effect
            assert torch.norm(torch.diff(diffused, dim=1)) < torch.norm(
                torch.diff(state, dim=1)
            ), "Diffusion should smooth gradients"
            return diffused

        test_diffusion(state)

        # Test reaction term
        def test_reaction(state):
            """Test reaction component."""
            reacted = pattern_system.apply_reaction(state)
            # Check boundedness
            assert torch.all(
                reacted <= pattern_system.max_concentration
            ), "Reaction should respect bounds"
            # Check fixed points
            fixed_points = pattern_system.find_reaction_fixed_points(state)
            assert len(fixed_points) > 0, "Should have fixed points"
            return reacted

        test_reaction(state)

        # Test full dynamics
        def test_full_dynamics(state, steps=100):
            """Test complete reaction-diffusion evolution."""
            trajectory = pattern_system.evolve_pattern(state, steps)
            assert trajectory.shape == (
                steps,
                batch_size,
                grid_size,
                grid_size,
                2,
            ), "Should have correct trajectory shape"
            # Test pattern formation
            final_pattern = trajectory[-1]
            wavelength = pattern_system.compute_pattern_wavelength(final_pattern)
            assert wavelength > 0, "Should form spatial patterns"
            return trajectory

        test_full_dynamics(state)

        # Test Turing instability
        def test_turing_instability() -> None:
            """Test conditions for Turing pattern formation."""
            jacobian = pattern_system.compute_reaction_jacobian(state)
            dispersion = pattern_system.compute_dispersion_relation(jacobian)
            # Check Turing conditions
            assert torch.any(dispersion > 0), "Should have positive growth rates"
            max_mode = torch.argmax(dispersion)
            assert max_mode > 0, "Most unstable mode should be non-zero"

        test_turing_instability()

    def test_stability_analysis(
        self, pattern_system, batch_size, grid_size
    ) -> None:
        """Test stability analysis of pattern dynamics."""
        # Initialize near fixed point
        fixed_point = pattern_system.find_homogeneous_state()
        perturbation = 0.01 * torch.randn(batch_size, grid_size, grid_size, 2)
        state = fixed_point + perturbation

        # Test linear stability
        def test_linear_stability():
            """Test linear stability analysis."""
            jacobian = pattern_system.compute_stability_matrix(fixed_point)
            eigenvals = torch.linalg.eigvals(jacobian)
            # Check spectral properties
            assert torch.all(
                eigenvals.real <= 0
            ), "Stable fixed point should have negative eigenvalues"
            return eigenvals

        test_linear_stability()

        # Test nonlinear stability
        def test_nonlinear_stability(state, time=10.0):
            """Test nonlinear stability properties."""
            trajectory = pattern_system.evolve_pattern(
                state, int(time / pattern_system.dt)
            )
            # Check Lyapunov function
            lyapunov = pattern_system.compute_lyapunov_function(trajectory, fixed_point)
            assert torch.all(
                torch.diff(lyapunov) <= 0
            ), "Lyapunov function should decrease"
            return lyapunov

        test_nonlinear_stability(state)

        # Test basin of attraction
        def test_basin_of_attraction() -> None:
            """Test basin of attraction analysis."""
            basin_boundary = pattern_system.estimate_basin_boundary(
                fixed_point, n_samples=100
            )
            assert basin_boundary > 0, "Should have finite basin"
            # Test convergence within basin
            state_in_basin = fixed_point + 0.5 * basin_boundary * perturbation
            final_state = pattern_system.evolve_pattern(state_in_basin, 1000)[-1]
            assert torch.allclose(
                final_state, fixed_point, rtol=1e-2
            ), "Should converge within basin"

        test_basin_of_attraction()

    def test_bifurcation_analysis(
        self, pattern_system, batch_size, grid_size
    ) -> None:
        """Test bifurcation analysis of pattern dynamics."""

        # Test parameter continuation
        def test_continuation():
            """Test numerical continuation of solutions."""
            branch = pattern_system.continue_solution(
                param_range=torch.linspace(0, 1, 100)
            )
            assert len(branch) > 0, "Should find solution branch"
            # Test branch stability
            stability = pattern_system.analyze_branch_stability(branch)
            assert len(stability) == len(branch), "Should analyze all points"
            return branch, stability

        branch, stability = test_continuation()

        # Test bifurcation detection
        def test_bifurcations():
            """Test bifurcation detection and classification."""
            bifurcations = pattern_system.detect_bifurcations(branch)
            assert len(bifurcations) > 0, "Should find bifurcations"
            # Test normal form computation
            for bif in bifurcations:
                coeffs = pattern_system.compute_normal_form(bif)
                assert len(coeffs) > 0, "Should compute normal form"
            return bifurcations

        bifurcations = test_bifurcations()

        # Test symmetry breaking
        def test_symmetry_breaking():
            """Test symmetry-breaking bifurcations."""
            symm = pattern_system.compute_symmetry_group()
            broken_symm = pattern_system.analyze_symmetry_breaking(bifurcations)
            assert len(broken_symm) > 0, "Should break symmetries"
            return symm, broken_symm

        symm, broken_symm = test_symmetry_breaking()

    def test_control_systems(
        self, pattern_system, batch_size, grid_size
    ) -> None:
        """Test control systems for pattern dynamics."""
        # Initialize target pattern
        target = pattern_system.generate_target_pattern(batch_size, grid_size, 2)

        # Test controllability
        def test_controllability():
            """Test system controllability."""
            control_matrix = pattern_system.compute_control_matrix()
            rank = torch.matrix_rank(control_matrix)
            assert rank == control_matrix.shape[0], "Should be controllable"
            return control_matrix

        test_controllability()

        # Test optimal control
        def test_optimal_control(target, time=1.0):
            """Test optimal control computation."""
            control = pattern_system.compute_optimal_control(target, final_time=time)
            assert (
                control.shape[1] == pattern_system.n_controls
            ), "Should have correct control dimensions"
            # Test control performance
            final_state = pattern_system.apply_control(control)
            assert torch.allclose(
                final_state, target, rtol=1e-2
            ), "Should reach target state"
            return control

        test_optimal_control(target)

        # Test feedback control
        def test_feedback_control():
            """Test feedback control law."""
            feedback_law = pattern_system.design_feedback_law()
            assert feedback_law.shape == (
                pattern_system.n_controls,
                pattern_system.state_dim,
            ), "Should have correct feedback dimensions"
            # Test closed-loop stability
            closed_loop_matrix = pattern_system.compute_closed_loop_matrix(feedback_law)
            eigenvals = torch.linalg.eigvals(closed_loop_matrix)
            assert torch.all(eigenvals.real < 0), "Closed-loop should be stable"
            return feedback_law

        test_feedback_control()

        # Test robustness
        def test_robustness():
            """Test control robustness."""
            # Parameter uncertainty
            uncertain_params = pattern_system.sample_uncertain_parameters()
            robust_control = pattern_system.compute_robust_control(
                target, uncertain_params
            )
            assert (
                robust_control.shape[1] == pattern_system.n_controls
            ), "Should have correct control dimensions"
            # Test performance bounds
            bounds = pattern_system.compute_performance_bounds(
                robust_control, uncertain_params
            )
            assert len(bounds) > 0, "Should compute bounds"
            return robust_control, bounds

        robust_control, bounds = test_robustness()

    def test_pattern_detection(
        self,
        hidden_dim: int,
        num_patterns: int,
        batch_size: int,
        seq_length: int,
    ) -> None:
        """Test pattern detection functionality."""
        # Create input states
        states = torch.randn(batch_size, seq_length, hidden_dim)

        # Initialize pattern dynamics
        pattern_dynamics = PatternDynamics(hidden_dim, num_patterns)

        # Detect patterns
        patterns, similarities = pattern_dynamics.detect_patterns(states)

        # Check shapes
        assert patterns.shape == (batch_size, seq_length, num_patterns)
        assert similarities.shape == (batch_size, seq_length, num_patterns)

        # Check values are normalized
        assert torch.all(patterns >= 0)
        assert torch.all(patterns <= 1)

    def test_forward_pass(
        self,
        hidden_dim: int,
        num_patterns: int,
        batch_size: int,
        seq_length: int,
    ) -> None:
        """Test forward pass of pattern dynamics."""
        # Create input states
        states = torch.randn(batch_size, seq_length, hidden_dim)

        # Initialize pattern dynamics
        pattern_dynamics = PatternDynamics(hidden_dim, num_patterns)

        # Run forward pass
        output = pattern_dynamics(states, return_patterns=True)

        # Check output dictionary contains expected keys
        assert "patterns" in output
        assert "similarities" in output
        assert "scores" in output

        # Check shapes
        assert output["patterns"].shape == (batch_size, seq_length, num_patterns)
        assert output["similarities"].shape == (batch_size, seq_length, num_patterns)
        assert output["scores"].shape == (batch_size, seq_length)
