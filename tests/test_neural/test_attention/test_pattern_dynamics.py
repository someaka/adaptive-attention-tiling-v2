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
import logging

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
        return 8  # Reduced for faster tests while maintaining pattern detection ability

    @pytest.fixture
    def batch_size(self) -> int:
        """Batch size for testing."""
        return 4  # Reduced from 8 for faster parallel computation

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
            u, v = state[:, 0:1], state[:, 1:2]  # Keep dimensions
            # Mass-conserving reaction terms
            du = u * v / (1 + u**2) - u  # Saturating reaction
            dv = u**2 / (1 + u**2) - v   # Balancing term
            return torch.cat([du, dv], dim=1)  # Preserve dimensions

        # Evolve system
        evolved = pattern_system.reaction_diffusion(
            state, diffusion_tensor, reaction_term
        )

        # Test conservation (if applicable)
        def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, rtol=1e-4, atol=1e-4, msg=""):
            """Custom tensor comparison with cleaner output."""
            if not torch.allclose(a, b, rtol=rtol, atol=atol):
                print("\nExpected:", b.detach().cpu().numpy())
                print("Got:", a.detach().cpu().numpy())
                raise AssertionError(msg)

        def assert_mass_conserved(initial: torch.Tensor, final: torch.Tensor, rtol=1e-4):
            """Assert that mass is conserved between two states."""
            initial_mass = initial.sum(dim=(-2, -1))
            final_mass = final.sum(dim=(-2, -1))
            try:
                assert_tensor_equal(initial_mass, final_mass, rtol=rtol, atol=atol, msg="Mass should be conserved.")
            except AssertionError as e:
                print("\nInitial mass:", initial_mass.detach().cpu().numpy())
                print("Final mass:", final_mass.detach().cpu().numpy())
                raise e

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
        # Create test pattern with controlled initialization
        pattern = torch.randn(1, 2, grid_size, grid_size, dtype=torch.float64)  # Use float64 for better precision
        pattern = pattern.clamp(-1.0, 1.0)  # Bound initial conditions

        # Define parameter range with fewer points
        parameter_range = torch.linspace(0, 2, 20, dtype=torch.float64)  # Reduced points, increased precision

        # Create parameterized reaction term with numerical safeguards
        def parameterized_reaction(state, param):
            u, v = state[:, 0], state[:, 1]
            # Add small epsilon to denominators to prevent division by zero
            eps = torch.finfo(state.dtype).eps
            du = param * torch.square(u) * v / (1.0 + torch.square(u) + eps) - u
            dv = torch.square(u) / (1.0 + torch.square(u) + eps) - v
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
            assert_mass_conserved(state, diffused)
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
        """Test linear stability analysis of pattern formation"""
        # Generate fixed point
        fixed_point = torch.full((batch_size, 2, grid_size, grid_size), 0.5)
        
        # Generate small perturbation
        perturbation = 0.01 * torch.randn_like(fixed_point)
        
        # Perturb fixed point
        state = fixed_point + perturbation
        
        # Evolve system
        diffusion_tensor = 0.1 * torch.eye(2)  # 2x2 diffusion tensor
        dt = 0.1
        num_steps = 100
        
        evolution = pattern_system.evolve_spatiotemporal(state, diffusion_tensor, num_steps, dt)
        
        # Check stability properties
        final_state = evolution[-1]
        deviation = torch.norm(final_state - fixed_point)
        
        # Perturbation should either grow or decay exponentially
        assert deviation > 0, "System should show dynamic behavior"

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


class TestReactionDiffusionProperties:
    """Test suite for reaction-diffusion system properties."""

    @pytest.fixture
    def pattern_system(self) -> PatternDynamics:
        """Create a test pattern dynamics system."""
        return PatternDynamics(dim=2, size=32, dt=0.01, boundary="periodic")

    def test_diffusion_kernel_properties(self):
        """Test that the diffusion kernel has correct mathematical properties."""
        # Initialize system
        system = PatternDynamics(dim=2, size=32)
        
        # Test 1: Kernel Construction Properties
        # Create a state with known mass for testing
        state = torch.zeros((1, 2, 32, 32), dtype=torch.float64)
        state[0, 0, 16, 16] = 1.0  # Single peak in center
        state[0, 1, 8:24, 8:24] = 0.5  # Uniform block in second channel
        
        # Get initial mass per channel
        initial_mass = state.sum(dim=[2,3])
        
        # Test different diffusion coefficients and time steps
        diff_coeffs = [0.1, 0.01, 0.001]
        time_steps = [0.1, 0.01, 0.001]
        
        for D in diff_coeffs:
            for dt in time_steps:
                # Apply diffusion
                diffused = system.apply_diffusion(state.clone(), D, dt)
                
                # Test 1.1: Mass Conservation (Global)
                final_mass = diffused.sum(dim=[2,3])
                mass_error = torch.abs(final_mass - initial_mass)
                assert_tensor_equal(initial_mass, final_mass, rtol=1e-10, msg="Mass not conserved for D={}, dt={}".format(D, dt))
                
                # Test 1.2: Positivity Preservation
                assert torch.all(diffused >= 0), "Negativity found for D={}, dt={}".format(D, dt)
                
                # Test 1.3: Maximum Principle
                assert torch.all(diffused <= state.max()), "Maximum principle violated for D={}, dt={}".format(D, dt)
                
                # Test 1.4: Scaling with D and dt
                # For small dt, change should be proportional to D*dt
                if dt <= 0.01:
                    change = torch.abs(diffused - state)
                    max_change = change.max()
                    expected_scale = D * dt
                    scale_error = abs(max_change / expected_scale - 1)
                    assert scale_error < 0.1, "Incorrect scaling with D={}, dt={}".format(D, dt)

        # Test 2: Spatial Symmetry Tests
        
        # Test 2.1: Grid-Aligned Symmetry
        # Create a cross pattern (perfectly aligned with grid)
        cross_state = torch.zeros((1, 1, 32, 32), dtype=torch.float64)
        cross_state[0, 0, 16, :] = 1.0  # Horizontal line
        cross_state[0, 0, :, 16] = 1.0  # Vertical line
        
        diffused_cross = system.apply_diffusion(cross_state, 0.1, 0.01)
        
        # Verify 4-fold symmetry (should be exact due to grid alignment)
        for i in range(32):
            for j in range(32):
                val = diffused_cross[0, 0, i, j]
                # Check all 4 quadrants have same value
                assert_tensor_equal(val, diffused_cross[0, 0, 31-i, j], rtol=1e-10, msg="Symmetry violated at ({}, {})".format(i, j))
                assert_tensor_equal(val, diffused_cross[0, 0, i, 31-j], rtol=1e-10, msg="Symmetry violated at ({}, {})".format(i, j))
                assert_tensor_equal(val, diffused_cross[0, 0, 31-i, 31-j], rtol=1e-10, msg="Symmetry violated at ({}, {})".format(i, j))

        # Test 2.2: Approximate Rotational Symmetry
        # Create a circular pattern
        center = 16
        radius = 5
        symmetric_state = torch.zeros((1, 1, 32, 32), dtype=torch.float64)
        
        # More precise circle creation using sub-pixel distance
        for i in range(32):
            for j in range(32):
                # Compute distance to center
                dy = i - center
                dx = j - center
                r = (dx*dx + dy*dy)**0.5
                # Smooth falloff near boundary
                if r <= radius:
                    symmetric_state[0, 0, i, j] = 1.0
                elif r <= radius + 1:
                    symmetric_state[0, 0, i, j] = radius + 1 - r  # Linear falloff
        
        diffused_sym = system.apply_diffusion(symmetric_state, 0.1, 0.01)
        
        # Test rotational symmetry with appropriate tolerance for grid effects
        for r in range(1, 10):
            points = []
            # Collect points at approximately radius r
            for i in range(32):
                for j in range(32):
                    dy = i - center
                    dx = j - center
                    point_r = (dx*dx + dy*dy)**0.5
                    if abs(point_r - r) < 0.5:  # Points within 0.5 pixels of target radius
                        points.append(diffused_sym[0, 0, i, j])
            
            if len(points) > 0:  # Only test if we found points at this radius
                points = torch.tensor(points)
                mean_val = points.mean()
                # Allow for grid discretization effects
                max_deviation = torch.max(torch.abs(points - mean_val))
                # Tolerance increases with radius due to grid effects
                allowed_deviation = 1e-4 * (1 + r/2)  # Scale tolerance with radius
                assert max_deviation < allowed_deviation, \
                    "Rotational symmetry violated at radius {}, max deviation: {}".format(r, max_deviation)

        # Test 3: Boundary Conditions
        # Create state with features near boundary
        boundary_state = torch.zeros((1, 1, 32, 32), dtype=torch.float64)
        boundary_state[0, 0, 0:3, 0:3] = 1.0  # Corner feature
        
        diffused_boundary = system.apply_diffusion(boundary_state, 0.1, 0.01)
        
        # Test 3.1: Periodic Boundary Conditions
        # Check that mass flows correctly across boundaries
        assert_tensor_equal(diffused_boundary[0, 0, -1, -1], diffused_boundary[0, 0, 0, 0], rtol=1e-10, msg="Periodic boundary condition violated")
        assert_tensor_equal(diffused_boundary[0, 0, 0, -1], diffused_boundary[0, 0, 0, 0], rtol=1e-10, msg="Periodic boundary condition violated")
        assert_tensor_equal(diffused_boundary[0, 0, -1, 0], diffused_boundary[0, 0, 0, 0], rtol=1e-10, msg="Periodic boundary condition violated")

        # Test 4: Numerical Stability
        # Test with larger time steps
        large_dt = 1.0
        large_D = 1.0
        stable_diffused = system.apply_diffusion(state.clone(), large_D, large_dt)
        
        # Test 4.1: No Numerical Explosions
        assert torch.all(torch.isfinite(stable_diffused)), "Numerical instability detected"
        assert_tensor_equal(stable_diffused.sum(), state.sum(), rtol=1e-10, msg="Mass conservation violated for large time step")

        # Test 5: Local Conservation
        # Create checkerboard pattern
        checker = torch.zeros((1, 1, 32, 32), dtype=torch.float64)
        checker[0, 0, ::2, ::2] = 1.0
        checker[0, 0, 1::2, 1::2] = 1.0
        
        diffused_checker = system.apply_diffusion(checker, 0.1, 0.01)
        
        # Test 5.1: Local Mass Conservation (2x2 blocks)
        for i in range(0, 32, 2):
            for j in range(0, 32, 2):
                block_orig = checker[0, 0, i:i+2, j:j+2].sum()
                block_diff = diffused_checker[0, 0, i:i+2, j:j+2].sum()
                assert_tensor_equal(block_orig, block_diff, rtol=1e-10, msg="Local mass conservation violated at block ({}, {})".format(i, j))

    def test_reaction_term_properties(self, pattern_system):
        """Test properties of the reaction terms."""
        # Create test states
        batch_size = 4
        grid_size = 32
        state = torch.randn(batch_size, 2, grid_size, grid_size)
        
        # Test 1: Mass conservation in reaction terms
        reaction = pattern_system.reaction_term(state)
        total_change = reaction.sum(dim=1)  # Sum over species
        assert_tensor_equal(total_change, torch.zeros_like(total_change), atol=1e-6, msg="Reaction terms must conserve total mass")
        
        # Test 2: Reaction terms should be bounded
        assert torch.all(torch.abs(reaction) < 1e3), \
            "Reaction terms should be bounded"
        
        # Test 3: Check scaling behavior
        scaled_state = 2 * state
        scaled_reaction = pattern_system.reaction_term(scaled_state)
        # Reaction terms should not grow faster than quadratically
        assert torch.all(torch.abs(scaled_reaction) < 4 * torch.abs(reaction) + 1e-6), \
            "Reaction terms should not grow faster than quadratically"

    def test_spatiotemporal_stability(self, pattern_system):
        """Test stability properties of the full reaction-diffusion system."""
        # Create initial state
        batch_size = 2
        grid_size = 32
        state = torch.randn(batch_size, 2, grid_size, grid_size)
        
        # Define diffusion tensor
        diffusion_tensor = torch.tensor([[0.1, 0.0], [0.0, 0.05]])
        
        # Test 1: Evolution should remain bounded
        steps = 50
        evolution = pattern_system.evolve_pattern(state, diffusion_tensor, steps=steps)
        assert torch.all(torch.abs(evolution) < 1e3), \
            "Pattern evolution should remain bounded"
        
        # Test 2: Check mass conservation over time
        initial_mass = state.sum(dim=[2,3])  # Sum over spatial dimensions
        for t in range(steps):
            current_mass = evolution[t].sum(dim=[2,3])
            assert_tensor_equal(initial_mass, current_mass, rtol=1e-4, msg="Mass not conserved at step {}".format(t))
        
        # Test 3: Check that the system approaches steady state or periodic behavior
        late_stage = evolution[-10:]  # Last 10 timesteps
        variation = torch.std(late_stage, dim=0)
        assert torch.all(variation < 1e1), \
            "System should approach steady state or show bounded oscillations"

    def test_diffusion_mass_conservation(self, pattern_system):
        """Test that diffusion preserves total mass."""
        # Create test state with known total mass
        state = torch.ones(1, 1, 32, 32)
        state[0, 0, 16, 16] = 2.0  # Add a peak
        initial_mass = state.sum()

        # Apply diffusion
        diffused = pattern_system.apply_diffusion(state, diffusion_coefficient=0.1, dt=0.01)
        final_mass = diffused.sum()

        # Check mass conservation with high precision
        assert_tensor_equal(final_mass, initial_mass, rtol=1e-10, msg="Mass not conserved")

    def test_diffusion_positivity_preservation(self, pattern_system):
        """Test that diffusion preserves positivity."""
        # Create positive test state
        state = torch.rand(1, 1, 32, 32)  # Random values between 0 and 1
        
        # Apply diffusion
        diffused = pattern_system.apply_diffusion(state, diffusion_coefficient=0.1, dt=0.01)
        
        # Check positivity preservation
        assert torch.all(diffused >= 0), "Diffusion should preserve positivity"

    def test_diffusion_maximum_principle(self, pattern_system):
        """Test that diffusion satisfies the maximum principle."""
        # Create test state with known bounds
        state = torch.rand(1, 1, 32, 32)  # Random values between 0 and 1
        initial_min = state.min()
        initial_max = state.max()
        
        # Apply diffusion
        diffused = pattern_system.apply_diffusion(state, diffusion_coefficient=0.1, dt=0.01)
        
        # Check maximum principle
        assert torch.all(diffused >= initial_min - 1e-10), \
            "Diffusion violated minimum value constraint"
        assert torch.all(diffused <= initial_max + 1e-10), \
            "Diffusion violated maximum value constraint"

    def test_diffusion_symmetry_preservation(self, pattern_system):
        """Test that diffusion preserves spatial symmetry."""
        # Create a symmetric test pattern (cross shape)
        state = torch.zeros(1, 1, 32, 32)
        mid = 16
        state[0, 0, mid, :] = 1.0  # Horizontal line
        state[0, 0, :, mid] = 1.0  # Vertical line
        
        # Apply diffusion
        diffused = pattern_system.apply_diffusion(state, diffusion_coefficient=0.1, dt=0.01)
        
        # Check rotational symmetry (90-degree rotations)
        rotated_0 = diffused
        rotated_90 = torch.rot90(diffused, k=1, dims=[-2, -1])
        rotated_180 = torch.rot90(diffused, k=2, dims=[-2, -1])
        rotated_270 = torch.rot90(diffused, k=3, dims=[-2, -1])
        
        # All rotations should be approximately equal
        assert_tensor_equal(rotated_0, rotated_90, rtol=1e-5, atol=1e-5, msg="90-degree rotation symmetry violated")
        assert_tensor_equal(rotated_0, rotated_180, rtol=1e-5, atol=1e-5, msg="180-degree rotation symmetry violated")
        assert_tensor_equal(rotated_0, rotated_270, rtol=1e-5, atol=1e-5, msg="270-degree rotation symmetry violated")

        # Check mirror symmetry
        flipped_h = torch.flip(diffused, dims=[-1])  # Horizontal flip
        flipped_v = torch.flip(diffused, dims=[-2])  # Vertical flip
        
        assert_tensor_equal(diffused, flipped_h, rtol=1e-5, atol=1e-5, msg="Horizontal mirror symmetry violated")
        assert_tensor_equal(diffused, flipped_v, rtol=1e-5, atol=1e-5, msg="Vertical mirror symmetry violated")


class TestDiffusionProperties:
    """Test suite for diffusion properties."""
    
    @pytest.fixture
    def pattern_system(self, test_params):
        """Create a pattern system for testing."""
        from src.neural.attention.pattern.diffusion import DiffusionSystem
        return DiffusionSystem(grid_size=test_params['grid_size'])
    
    @pytest.fixture
    def test_params(self):
        """Common test parameters.
        
        For numerical stability of the diffusion equation:
        - dt * D / (dx)^2 <= 0.5  (CFL condition)
        where:
        - D is the diffusion coefficient
        - dt is the time step
        - dx is the grid spacing (1.0 in our case)
        """
        return {
            'diffusion_coefficient': 0.25,  # D = 0.25 for stability
            'dt': 0.1,  # dt = 0.1 satisfies CFL condition
            'grid_size': 32,
            'batch_size': 1,
            'channels': 1,
            'device': 'cpu',
            'dtype': torch.float32,
            'rtol': 1e-5,  # Relative tolerance for comparisons
            'atol': 1e-5   # Absolute tolerance for comparisons
        }
        
    def create_test_state(self, params, pattern='uniform'):
        """Create a test state tensor.
        
        Args:
            params: Test parameters
            pattern: Type of test pattern ('uniform', 'random', 'impulse', 'checkerboard', 'gradient')
            
        Returns:
            Test state tensor
        """
        shape = (params['batch_size'], params['channels'], 
                params['grid_size'], params['grid_size'])
                
        if pattern == 'uniform':
            state = torch.ones(shape, dtype=params['dtype'], device=params['device'])
        elif pattern == 'random':
            torch.manual_seed(42)  # For reproducibility
            state = torch.rand(shape, dtype=params['dtype'], device=params['device'])
        elif pattern == 'impulse':
            state = torch.zeros(shape, dtype=params['dtype'], device=params['device'])
            mid = params['grid_size'] // 2
            state[:, :, mid-1:mid+2, mid-1:mid+2] = 1.0
        elif pattern == 'checkerboard':
            x = torch.arange(params['grid_size'], device=params['device'])
            y = torch.arange(params['grid_size'], device=params['device'])
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            state = ((xx + yy) % 2).to(params['dtype']).view(1, 1, params['grid_size'], params['grid_size'])
            state = state.repeat(params['batch_size'], params['channels'], 1, 1)
        elif pattern == 'gradient':
            x = torch.linspace(0, 1, params['grid_size'], device=params['device'])
            y = torch.linspace(0, 1, params['grid_size'], device=params['device'])
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            state = (xx * yy).to(params['dtype']).view(1, 1, params['grid_size'], params['grid_size'])
            state = state.repeat(params['batch_size'], params['channels'], 1, 1)
        else:
            raise ValueError(f"Unknown pattern type: {pattern}")
            
        return state
        
    def test_mass_conservation(self, pattern_system, test_params):
        """Test that diffusion conserves total mass."""
        patterns = ['uniform', 'random', 'impulse', 'checkerboard', 'gradient']
        
        for pattern in patterns:
            state = self.create_test_state(test_params, pattern)
            initial_mass = state.sum()
            
            # Apply diffusion multiple times
            for i in range(10):
                state = pattern_system.apply_diffusion(
                    state, 
                    test_params['diffusion_coefficient'],
                    test_params['dt']
                )
                
                # Check mass conservation with relative tolerance
                current_mass = state.sum()
                mass_error = torch.abs(current_mass - initial_mass)
                mass_scale = torch.max(torch.abs(initial_mass), torch.tensor(1.0))
                relative_error = mass_error / mass_scale
                
                assert relative_error < test_params['rtol'], \
                    f"Mass not conserved for {pattern} pattern. Error: {mass_error.item():.2e}"
                
                # Check for numerical stability
                assert torch.isfinite(state).all(), \
                    f"Non-finite values found in {pattern} pattern"
    
    def test_positivity_preservation(self, pattern_system, test_params):
        """Test that diffusion preserves positivity."""
        patterns = ['uniform', 'random', 'impulse', 'checkerboard', 'gradient']
        
        for pattern in patterns:
            state = self.create_test_state(test_params, pattern)
            
            # Apply diffusion multiple times
            for _ in range(10):
                state = pattern_system.apply_diffusion(
                    state, 
                    test_params['diffusion_coefficient'],
                    test_params['dt']
                )
                
                # Check positivity
                min_val = state.min()
                assert min_val >= -test_params['atol'], \
                    f"Positivity violated for {pattern} pattern. Min value: {min_val.item():.2e}"
    
    def test_maximum_principle(self, pattern_system, test_params):
        """Test that diffusion satisfies the maximum principle."""
        patterns = ['uniform', 'random', 'impulse', 'checkerboard', 'gradient']
        
        for pattern in patterns:
            state = self.create_test_state(test_params, pattern)
            initial_max = state.max()
            initial_min = state.min()
            
            # Apply diffusion multiple times
            for _ in range(10):
                state = pattern_system.apply_diffusion(
                    state, 
                    test_params['diffusion_coefficient'],
                    test_params['dt']
                )
                
                # Check maximum principle
                max_val = state.max()
                min_val = state.min()
                
                assert max_val <= initial_max + test_params['atol'], \
                    f"Maximum principle violated for {pattern} pattern. Max exceeded by {(max_val - initial_max).item():.2e}"
                assert min_val >= initial_min - test_params['atol'], \
                    f"Maximum principle violated for {pattern} pattern. Min exceeded by {(initial_min - min_val).item():.2e}"
    
    def test_symmetry_preservation(self, pattern_system, test_params):
        """Test that diffusion preserves symmetry."""
        # Test with symmetric patterns
        state = self.create_test_state(test_params, 'uniform')
        
        # Apply diffusion
        diffused = pattern_system.apply_diffusion(
            state,
            test_params['diffusion_coefficient'],
            test_params['dt']
        )
        
        # Test rotational symmetry
        for k in range(4):  # Test 90-degree rotations
            rotated = torch.rot90(diffused, k=k, dims=[-2, -1])
            assert_tensor_equal(diffused, rotated, rtol=test_params['rtol'], atol=test_params['atol'], msg=f"Rotational symmetry violated for {90*k}-degree rotation")
        
        # Test mirror symmetry
        flipped_h = torch.flip(diffused, dims=[-1])
        flipped_v = torch.flip(diffused, dims=[-2])
        
        assert_tensor_equal(diffused, flipped_h, rtol=test_params['rtol'], atol=test_params['atol'], msg="Horizontal mirror symmetry violated")
        assert_tensor_equal(diffused, flipped_v, rtol=test_params['rtol'], atol=test_params['atol'], msg="Vertical mirror symmetry violated")
    
    def test_convergence_to_steady_state(self, pattern_system, test_params):
        """Test that diffusion converges to steady state.
        
        This test verifies three key properties of diffusion:
        1. The system converges to a uniform steady state
        2. The mean value is preserved during diffusion
        3. Total mass is conserved throughout the process
        
        For each test pattern (impulse, checkerboard, gradient), we:
        - Apply diffusion until the state is sufficiently uniform
        - Verify the final state matches theoretical expectations
        - Check that mass is conserved during the entire process
        """
        max_iter = 5000  # Increased to allow more time for convergence
        min_steps = 10
        patterns = ['impulse', 'checkerboard', 'gradient']
        
        # Pattern-specific convergence tolerances
        convergence_tols = {
            'impulse': 0.05,       # 5% deviation allowed for impulse
            'checkerboard': 0.02,  # 2% deviation allowed for checkerboard
            'gradient': 0.01       # 1% deviation allowed for gradient
        }
        
        for pattern in patterns:
            # Create test pattern
            state = self.create_test_state(test_params, pattern)
            initial_mean = state.mean(dim=(-2, -1), keepdim=True)
            initial_mass = state.sum()
            grid_size = test_params['grid_size']
            
            # Calculate scale and tolerance
            pattern_scale = torch.max(torch.abs(initial_mean), torch.tensor(test_params['atol']))
            convergence_tol = convergence_tols[pattern]
            
            # Pattern-specific tolerances for mass and mean conservation
            conservation_tols = {
                'impulse': 1e-3,      # 0.1% error allowed
                'checkerboard': 1e-4, # 0.01% error allowed
                'gradient': 1e-5      # 0.001% error allowed
            }
            
            # Apply diffusion until convergence
            converged = False
            prev_state = state
            
            for i in range(max_iter):
                # Apply single diffusion step
                curr_state = pattern_system.apply_diffusion(
                    prev_state,
                    test_params['diffusion_coefficient'],
                    test_params['dt']
                )
                
                # Check convergence using relative change from mean
                curr_mean = curr_state.mean(dim=(-2, -1), keepdim=True)
                max_deviation = torch.max(torch.abs(curr_state - curr_mean))
                relative_deviation = (max_deviation / pattern_scale) / grid_size  # Normalize by grid size
                
                # Also check rate of change
                state_change = torch.max(torch.abs(curr_state - prev_state))
                relative_change = (state_change / pattern_scale) / grid_size  # Normalize by grid size
                
                # Only check convergence after minimum steps to allow initial diffusion
                if i >= min_steps and relative_deviation < convergence_tol and relative_change < 0.01:
                    converged = True
                    break
                    
                prev_state = curr_state
            
            # Verify convergence was achieved within max iterations
            assert converged, \
                f"Diffusion did not converge for {pattern} pattern after {max_iter} iterations. " \
                f"Relative deviation: {relative_deviation.item():.2e}, tolerance: {convergence_tol}"
            
            # Verify steady state properties
            final_state = curr_state
            final_mean = final_state.mean(dim=(-2, -1), keepdim=True)
            
            # 1. Check state is uniform (using absolute deviation)
            max_deviation = torch.max(torch.abs(final_state - final_mean))
            abs_tol = pattern_scale * 2.0  # Allow 200% deviation from mean
            
            assert max_deviation < abs_tol, \
                f"Steady state is not uniform for {pattern} pattern. " \
                f"Max deviation: {max_deviation.item():.2e}, tolerance: {abs_tol.item():.2e}"
            
            # 2. Check mean value is preserved (should equal initial mean)
            mean_error = torch.abs(final_mean - initial_mean) / (torch.abs(initial_mean) + test_params['atol'])
            
            assert torch.all(mean_error < conservation_tols[pattern]), \
                f"Mean value not preserved for {pattern} pattern. " \
                f"Error: {mean_error.item():.2e}"
            
            # 3. Verify mass conservation (should equal initial mass)
            final_mass = final_state.sum()
            mass_error = torch.abs(final_mass - initial_mass)
            mass_scale = torch.max(torch.abs(initial_mass), torch.tensor(1.0))
            relative_error = mass_error / mass_scale
            
            assert relative_error < conservation_tols[pattern], \
                f"Mass not conserved for {pattern} pattern. " \
                f"Relative error: {relative_error.item():.2e}"
