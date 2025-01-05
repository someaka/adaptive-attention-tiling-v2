"""
Unit tests for the pattern dynamics system.

Tests cover:
1. Reaction-diffusion dynamics
2. Pattern stability analysis
3. Bifurcation detection
4. Pattern control mechanisms
5. Spatiotemporal evolution
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pytest
import torch
import torch.linalg
import logging

from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.attention.pattern.models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationPoint,
    BifurcationDiagram
)

# Test helper functions
def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4, msg: str = "") -> None:
    """Custom tensor comparison with cleaner output."""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        print("\nExpected:", b.detach().cpu().numpy())
        print("Got:", a.detach().cpu().numpy())
        raise AssertionError(msg)

def assert_mass_conserved(initial: torch.Tensor, final: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    """Assert that mass is conserved between two states."""
    initial_mass = initial.sum(dim=(-2, -1))
    final_mass = final.sum(dim=(-2, -1))
    try:
        assert_tensor_equal(initial_mass, final_mass, rtol=rtol, atol=atol, msg="Mass should be conserved.")
    except AssertionError as e:
        print("\nInitial mass:", initial_mass.detach().cpu().numpy())
        print("Final mass:", final_mass.detach().cpu().numpy())
        raise e

class TestPatternDynamics:
    """Test suite for pattern dynamics."""

    @pytest.fixture
    def pattern_system(self) -> PatternDynamics:
        """Create a test pattern dynamics system."""
        return PatternDynamics(
            grid_size=8,  # Small grid for testing
            space_dim=2,  # 2D patterns
            boundary="periodic",
            dt=0.01,
            hidden_dim=64,
            num_modes=8
        )

    def test_stability_analysis_basic(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test basic pattern stability analysis."""
        # Create test pattern
        pattern = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)

        # Create perturbation
        perturbation = 0.1 * torch.randn_like(pattern)

        # Analyze stability using stability analyzer
        metrics = pattern_system.stability_analysis(pattern, perturbation)

        # Test metric properties
        assert isinstance(metrics, StabilityMetrics), "Should return stability metrics"
        assert metrics.linear_stability is not None, "Should compute linear stability"
        assert metrics.nonlinear_stability is not None, "Should compute nonlinear stability"
        assert metrics.lyapunov_spectrum is not None, "Should compute Lyapunov spectrum"
        assert isinstance(metrics.structural_stability, float), "Should compute structural stability"

    def test_stability_analysis_advanced(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test advanced stability analysis features."""
        # Create test pattern
        pattern = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)

        # Test Lyapunov spectrum
        lyapunov_spectrum = pattern_system.compute_lyapunov_spectrum(pattern)
        assert len(lyapunov_spectrum) > 0, "Should compute Lyapunov exponents"
        # Convert to complex tensor before checking imaginary parts
        lyapunov_complex = lyapunov_spectrum.to(torch.complex64)
        assert torch.all(
            torch.abs(lyapunov_complex.imag) < 1e-6
        ), "Lyapunov exponents should be real"

        # Test structural stability
        def perturbed_reaction(state: torch.Tensor) -> torch.Tensor:
            """Slightly perturbed reaction term."""
            u, v = state[:, 0], state[:, 1]
            du = u**2 * v - u + 0.1 * u**3
            dv = u**2 - v + 0.1 * v**3
            return torch.stack([du, dv], dim=1)

        structural_stability = pattern_system.test_structural_stability(
            pattern, perturbed_reaction
        )
        assert structural_stability > 0, "Should be structurally stable"

    def test_pattern_formation(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test pattern formation dynamics."""
        logging.info("Starting pattern formation test")

        # Create initial state with physically meaningful perturbations
        # Use activator-inhibitor initial conditions: nearly uniform with small random perturbations
        base_activator = 1.0
        base_inhibitor = 1.0
        state = torch.zeros(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        state[:, 0] = base_activator + 0.01 * torch.randn(pattern_system.size, pattern_system.size)  # Activator
        state[:, 1] = base_inhibitor + 0.01 * torch.randn(pattern_system.size, pattern_system.size)  # Inhibitor
        
        # Ensure positive concentrations
        state = torch.clamp(state, min=0.0)
        
        logging.info(f"Initial state shape: {state.shape}")
        logging.info(f"Initial activator - mean: {state[:,0].mean():.6f}, std: {state[:,0].std():.6f}")
        logging.info(f"Initial inhibitor - mean: {state[:,1].mean():.6f}, std: {state[:,1].std():.6f}")

        # Define reaction term with proper scaling
        def reaction_term(state: torch.Tensor) -> torch.Tensor:
            """Pattern-forming reaction terms with physical constraints."""
            # Ensure positive concentrations
            state = torch.clamp(state, min=0.0)
            u, v = state[:, 0:1], state[:, 1:2]  # Activator and inhibitor
            
            # Standard activator-inhibitor dynamics
            # du/dt = u²v - u (autocatalysis and decay)
            # dv/dt = u² - v (production and decay)
            du = u * u * v - u
            dv = u * u - v
            
            # Combine terms without artificial normalization
            reaction = torch.cat([du, dv], dim=1)
            
            # Apply reasonable bounds to reaction rates
            reaction = torch.clamp(reaction, min=-10.0, max=10.0)
            
            logging.debug(f"Reaction components - du: {du.mean():.6f}, dv: {dv.mean():.6f}")
            return reaction

        # Initialize evolution parameters with physical timescales
        base_dt = pattern_system.dt
        min_dt = base_dt * 0.1
        max_dt = base_dt * 2.0
        target_change_rate = 0.05  # Allow more significant changes for pattern formation
        adaptation_rate = 0.1  # Gentler adaptation
        stability_window = 20  # Longer window for better stability assessment
        max_steps = 1000  # More steps to allow pattern development
        min_steps = 100
        convergence_threshold = 0.001  # More reasonable for pattern dynamics

        # Storage for monitoring evolution
        time_evolution = []
        change_rates = []
        step_sizes = []
        current_state = state.clone()
        time_evolution.append(current_state.clone())
        
        # Track pattern metrics
        pattern_metrics = {
            'spatial_variance': [],
            'activator_inhibitor_ratio': [],
            'mass_conservation': []
        }

        logging.info("Starting pattern evolution with adaptive step size")
        step = 0
        while step < max_steps:
            # Store current dt
            step_sizes.append(pattern_system.dt)
            
            # Evolve one step
            next_state = pattern_system.step(
                current_state,
                diffusion_coefficient=0.1,
                reaction_term=reaction_term
            )
            
            # Ensure physical constraints
            next_state = torch.clamp(next_state, min=0.0)
            
            # Calculate change metrics
            change_rate = torch.norm(next_state - current_state) / (torch.norm(current_state) + 1e-6)
            change_rates.append(change_rate.item())
            
            # Calculate pattern metrics
            spatial_var = torch.var(next_state, dim=(-2, -1)).mean()
            act_inhib_ratio = (next_state[:,0].mean() / (next_state[:,1].mean() + 1e-6)).item()
            mass_conservation = torch.sum(next_state) / (torch.sum(current_state) + 1e-6)
            
            pattern_metrics['spatial_variance'].append(spatial_var.item())
            pattern_metrics['activator_inhibitor_ratio'].append(act_inhib_ratio)
            pattern_metrics['mass_conservation'].append(mass_conservation.item())
            
            # Log every 10 steps
            if step % 10 == 0:
                logging.info(f"Step {step}:")
                logging.info(f"  Change rate: {change_rate.item():.6f}")
                logging.info(f"  Current dt: {pattern_system.dt:.6f}")
                logging.info(f"  Spatial variance: {spatial_var.item():.6f}")
                logging.info(f"  Activator/Inhibitor ratio: {act_inhib_ratio:.6f}")
                logging.info(f"  Mass conservation: {mass_conservation.item():.6f}")

            # Store evolved state
            time_evolution.append(next_state.clone())
            current_state = next_state

            # After stability window steps, adjust dt based on change rate
            if step >= stability_window:
                recent_rates = torch.tensor(change_rates[-stability_window:])
                avg_rate = recent_rates.mean().item()
                rate_std = recent_rates.std().item()
                
                # Adjust dt based on average change rate
                if avg_rate > target_change_rate * 1.2:  # Too fast
                    new_dt = max(pattern_system.dt * (1 - adaptation_rate), min_dt)
                    pattern_system.dt = new_dt
                elif avg_rate < target_change_rate * 0.8 and rate_std < target_change_rate * 0.1:  # Too slow and stable
                    new_dt = min(pattern_system.dt * (1 + adaptation_rate), max_dt)
                    pattern_system.dt = new_dt

                # Check for convergence after minimum steps
                # Consider both change rate and pattern formation metrics
                if step >= min_steps:
                    recent_spatial_var = pattern_metrics['spatial_variance'][-stability_window:]
                    spatial_var_stable = (torch.tensor(recent_spatial_var).std() < convergence_threshold)
                    
                    if avg_rate < convergence_threshold and spatial_var_stable:
                        logging.info(f"Converged after {step} steps:")
                        logging.info(f"  Change rate: {avg_rate:.6f}")
                        logging.info(f"  Spatial variance stability: {torch.tensor(recent_spatial_var).std():.6f}")
                        break

            step += 1

        logging.info(f"Evolution complete - produced {len(time_evolution)} timesteps")
        logging.info(f"Final dt: {pattern_system.dt:.6f}")
        logging.info(f"Step size history - min: {min(step_sizes):.6f}, max: {max(step_sizes):.6f}, mean: {sum(step_sizes)/len(step_sizes):.6f}")
        logging.info(f"Change rate history - min: {min(change_rates):.6f}, max: {max(change_rates):.6f}, mean: {sum(change_rates)/len(change_rates):.6f}")

        # Analyze final pattern
        final_pattern = time_evolution[-1]
        
        # Check for pattern formation using multiple criteria
        spatial_structure = torch.var(final_pattern, dim=(-2, -1)).mean() > 0.01
        
        # More robust temporal stability check
        stability_window = 20
        recent_rates = torch.tensor(change_rates[-stability_window:])
        avg_rate = recent_rates.mean().item()
        rate_std = recent_rates.std().item()
        temporal_stability = (avg_rate < convergence_threshold * 2.0) and (rate_std < convergence_threshold)
        
        mass_conserved = abs(pattern_metrics['mass_conservation'][-1] - 1.0) < 0.1
        
        pattern_formed = spatial_structure and temporal_stability and mass_conserved
        
        logging.info(f"Pattern analysis:")
        logging.info(f"  Spatial structure: {spatial_structure}")
        logging.info(f"  Temporal stability: {temporal_stability}")
        logging.info(f"  Mass conservation: {mass_conserved}")
        logging.info(f"  Pattern formed: {pattern_formed}")

        # Analyze stability
        stability_result = pattern_system.stability.is_stable(final_pattern, threshold=0.1)
        stability_value = pattern_system.stability.compute_stability(final_pattern)
        logging.info(f"Stability analysis - value: {stability_value:.6f}, threshold: 0.1, stable: {stability_result}")

        # Compute eigenvalues for detailed analysis
        eigenvalues = pattern_system.stability.compute_eigenvalues(final_pattern)[0]
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        logging.info(f"Eigenvalue analysis:")
        logging.info(f"  Max real part: {real_parts.max():.6f}")
        logging.info(f"  Min real part: {real_parts.min():.6f}")
        logging.info(f"  Max imag part magnitude: {torch.abs(imag_parts).max():.6f}")

        # Run assertions with detailed error messages
        assert pattern_formed, "Pattern formation criteria not met"
        
        try:
            assert stability_result, f"Final pattern should be stable (stability value: {stability_value:.6f})"
        except AssertionError as e:
            logging.error("Stability assertion failed:")
            logging.error(f"  Stability value: {stability_value:.6f}")
            logging.error(f"  Max eigenvalue real part: {real_parts.max():.6f}")
            logging.error(f"  Pattern statistics - mean: {final_pattern.mean():.6f}, std: {final_pattern.std():.6f}")
            logging.error(f"  Final change rate: {change_rates[-1]:.6f}")
            logging.error(f"  Final dt: {pattern_system.dt:.6f}")
            raise e

    def test_forward_pass(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test forward pass of pattern dynamics."""
        # Create input states with correct dimensions
        batch_size = 1
        size = pattern_system.size
        dim = pattern_system.dim
        states = torch.randn(batch_size, dim, size, size)  # [batch, channels, height, width]

        # Run forward pass
        output_dict = pattern_system(states, return_patterns=True)

        # Verify output contains patterns
        assert 'patterns' in output_dict, "Output should contain patterns"
        patterns = output_dict['patterns']

        # Verify patterns shape
        assert patterns.shape[0] > 0, "Should have at least one pattern"
        assert patterns.shape[1] == batch_size, "Batch size should be preserved"
        assert patterns.shape[2] == dim, "Pattern dimensions should be preserved"
        assert patterns.shape[3] == size and patterns.shape[4] == size, "Spatial dimensions should be preserved"

        # Check output dictionary contains expected keys
        assert "routing_scores" in output_dict
        assert "pattern_scores" in output_dict

        # Check shapes
        assert output_dict["routing_scores"].shape[0] == batch_size, "Should have correct routing score shape"
        assert output_dict["pattern_scores"].shape[0] == patterns.shape[0], "Should have scores for each pattern"

        # Test evolution
        trajectory = pattern_system.evolve_pattern(states, steps=10)
        assert len(trajectory) == 10, "Should have correct number of steps"
        assert all(s.shape == states.shape for s in trajectory), "Should preserve shape throughout evolution"

    def test_pattern_control(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test pattern control."""
        # Create current and target patterns
        current = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        target = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        
        # Define constraints
        def constraint(state: torch.Tensor) -> torch.Tensor:
            """Example constraint function."""
            return torch.norm(state)

        # Compute control signal
        control = pattern_system.pattern_control(current, target, constraints=[constraint])
        
        # Check control signal properties
        assert isinstance(control, ControlSignal), "Should return control signal"
        assert control.magnitude is not None, "Should compute control magnitude"
        assert control.signal is not None, "Should compute control signal"

    def test_lyapunov_spectrum(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test Lyapunov spectrum computation."""
        # Create test pattern
        pattern = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        
        # Compute Lyapunov spectrum
        spectrum = pattern_system.compute_lyapunov_spectrum(pattern)
        
        # Check spectrum properties
        assert isinstance(spectrum, torch.Tensor), "Should return tensor"
        assert spectrum.ndim == 1, "Should be 1D array"
        assert spectrum.shape[0] > 0, "Should have at least one exponent"
        # Check if sorted in descending order
        spectrum_sorted, _ = torch.sort(spectrum, descending=True)
        assert torch.allclose(spectrum, spectrum_sorted), "Should be sorted descending"

    def test_detect_pattern_formation(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test pattern formation detection."""
        # Create evolving patterns
        patterns = []
        pattern = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        
        # Evolve pattern for several steps
        for _ in range(10):
            pattern = pattern_system.compute_next_state(pattern)
            patterns.append(pattern)
        
        # Check pattern formation
        has_pattern = pattern_system.detect_pattern_formation(patterns)
        assert isinstance(has_pattern, bool), "Should return boolean"

class TestReactionDiffusion:
    """Test suite for reaction-diffusion dynamics."""

    @pytest.fixture
    def pattern_system(self) -> PatternDynamics:
        """Create a test pattern dynamics system."""
        return PatternDynamics(
            grid_size=8,  # Small grid for testing
            space_dim=2,  # 2D patterns
            boundary="periodic",
            dt=0.01,
            hidden_dim=64,
            num_modes=8
        )

    def test_reaction_term(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test reaction term computation."""
        # Create test state
        state = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        
        # Compute reaction term
        reaction = pattern_system.reaction.reaction_term(state)
        
        # Check output
        assert reaction.shape == state.shape, "Should preserve shape"
        assert not torch.allclose(reaction, state), "Should modify state"

    def test_diffusion_term(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test diffusion term computation."""
        # Create test state
        state = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        
        # Apply diffusion
        diffused = pattern_system.diffusion.apply_diffusion(
            state, 
            diffusion_coefficient=0.1,
            dt=pattern_system.dt
        )
        
        # Check output
        assert diffused.shape == state.shape, "Should preserve shape"
        assert not torch.allclose(diffused, state), "Should modify state"

    def test_combined_dynamics(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test combined reaction-diffusion dynamics."""
        # Create test state
        state = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        
        # Evolve state
        evolved = pattern_system.compute_next_state(state)
        
        # Check output
        assert evolved.shape == state.shape, "Should preserve shape"
        assert not torch.allclose(evolved, state), "Should modify state"

class TestDiffusionProperties:
    """Test suite for diffusion properties."""

    @pytest.fixture
    def pattern_system(self) -> PatternDynamics:
        """Create a test pattern dynamics system."""
        return PatternDynamics(
            grid_size=8,  # Small grid for testing
            space_dim=2,  # 2D patterns
            boundary="periodic",
            dt=0.01,
            hidden_dim=64,
            num_modes=8
        )

    def test_diffusion_conservation(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test mass conservation in diffusion."""
        # Create test state
        state = torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        initial_mass = state.sum()
        
        # Apply diffusion
        diffused = pattern_system.diffusion.apply_diffusion(
            state,
            diffusion_coefficient=0.1,
            dt=pattern_system.dt
        )
        final_mass = diffused.sum()
        
        # Check mass conservation
        assert torch.allclose(initial_mass, final_mass, rtol=1e-4), "Mass should be conserved"

    def test_diffusion_smoothing(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test smoothing effect of diffusion."""
        # Create test state with high frequency components
        x = torch.linspace(-4, 4, pattern_system.size)
        y = torch.linspace(-4, 4, pattern_system.size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        state = torch.sin(2 * torch.pi * X) * torch.sin(2 * torch.pi * Y)
        state = state.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Apply diffusion
        diffused = pattern_system.diffusion.apply_diffusion(
            state,
            diffusion_coefficient=0.1,
            dt=pattern_system.dt
        )
        
        # Check smoothing
        assert torch.std(diffused) < torch.std(state), "Diffusion should reduce variance"

    def test_boundary_conditions(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test periodic boundary conditions."""
        # Create test state with strong gradient at boundary
        state = torch.zeros(1, pattern_system.dim, pattern_system.size, pattern_system.size)
        state[..., 0] = 1.0  # Set first row to 1
        
        # Apply diffusion
        diffused = pattern_system.diffusion.apply_diffusion(
            state,
            diffusion_coefficient=0.1,
            dt=pattern_system.dt
        )
        
        # Check periodic boundary conditions
        if pattern_system.boundary == "periodic":
            # Check that diffusion occurs across boundary
            assert diffused[..., -1].mean() > 0, "Should diffuse across periodic boundary"
            assert diffused[..., 0].mean() < 1, "Should diffuse across periodic boundary"

