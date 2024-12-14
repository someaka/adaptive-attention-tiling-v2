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
        assert torch.all(
            torch.imag(lyapunov_spectrum) == 0
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
        # Create initial state with small random perturbations
        state = torch.ones(1, pattern_system.dim, pattern_system.size, pattern_system.size) + 0.01 * torch.randn(1, pattern_system.dim, pattern_system.size, pattern_system.size)

        # Define reaction term
        def reaction_term(state: torch.Tensor) -> torch.Tensor:
            """Pattern-forming reaction terms."""
            u, v = state[:, 0:1], state[:, 1:2]
            du = u**2 * v - u
            dv = u**2 - v
            return torch.cat([du, dv], dim=1)

        # Evolve system
        time_evolution = pattern_system.evolve_pattern(
            state, diffusion_coefficient=0.1, reaction_term=reaction_term, steps=100
        )

        # Test pattern formation
        final_pattern = time_evolution[-1]
        assert pattern_system.detect_pattern_formation(time_evolution), "Should detect pattern formation"

        # Test pattern stability
        assert pattern_system.stability.is_stable(final_pattern), "Final pattern should be stable"

    def test_forward_pass(
        self, pattern_system: PatternDynamics
    ) -> None:
        """Test forward pass of pattern dynamics."""
        # Create input states
        seq_length = 16
        states = torch.randn(1, seq_length, pattern_system.hidden_dim)

        # Run forward pass
        output = pattern_system(states, return_patterns=True)

        # Check output dictionary contains expected keys
        assert "routing_scores" in output
        assert "patterns" in output
        assert "pattern_scores" in output

        # Check shapes
        assert output["routing_scores"].shape == (1, seq_length), "Should have correct routing score shape"
        assert output["patterns"].shape[0] > 0, "Should have pattern evolution"
        assert output["pattern_scores"].shape[0] == output["patterns"].shape[0], "Should have scores for each pattern"

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

