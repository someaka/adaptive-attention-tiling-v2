"""
Unit tests for pattern stability validation.

Tests cover:
1. Linear stability analysis
2. Nonlinear stability analysis
3. Perturbation response
4. Lyapunov analysis
5. Mode decomposition
6. Dynamical systems
7. Bifurcation theory
8. Stability analysis
"""

from typing import Dict

import numpy as np
import pytest
import torch

from src.validation.flow.stability import (
    LinearStabilityValidator,
    NonlinearStabilityValidator,
    LinearStabilityValidation,
    NonlinearStabilityValidation
)
from src.validation.patterns.perturbation import (
    PerturbationAnalyzer,
    PerturbationMetrics
)
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.tiling.geometric_flow import GeometricFlow


class TestPatternStability:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def spatial_dim(self) -> int:
        return 32

    @pytest.fixture
    def time_steps(self) -> int:
        return 100

    @pytest.fixture
    def hidden_dim(self) -> int:
        return 64

    @pytest.fixture
    def manifold_dim(self) -> int:
        return 16

    @pytest.fixture
    def flow(self, hidden_dim: int, manifold_dim: int) -> GeometricFlow:
        return GeometricFlow(hidden_dim=hidden_dim, manifold_dim=manifold_dim)

    def test_linear_stability(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test linear stability analysis."""
        # Create linear stability analyzer
        analyzer = LinearStabilityValidator()

        # Generate test patterns with known stability
        def generate_stable_pattern():
            """Generate linearly stable pattern."""
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            return torch.sin(x) + 0.1 * torch.randn_like(x)

        def generate_unstable_pattern():
            """Generate linearly unstable pattern."""
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            return torch.exp(0.1 * x) * torch.sin(x)

        # Test stable patterns
        stable_patterns = torch.stack(
            [generate_stable_pattern() for _ in range(batch_size)]
        )
        stable_result = analyzer.validate_stability(flow, stable_patterns)
        assert isinstance(stable_result, LinearStabilityValidation)
        assert stable_result.stable
        assert torch.all(stable_result.growth_rates < 1.0)

        # Test unstable patterns
        unstable_patterns = torch.stack(
            [generate_unstable_pattern() for _ in range(batch_size)]
        )
        unstable_result = analyzer.validate_stability(flow, unstable_patterns)
        assert isinstance(unstable_result, LinearStabilityValidation)
        assert not unstable_result.stable
        assert torch.any(unstable_result.growth_rates > 1.0)

    def test_nonlinear_stability(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test nonlinear stability analysis."""
        # Create nonlinear stability analyzer
        analyzer = NonlinearStabilityValidator()

        # Generate test patterns with nonlinear dynamics
        def generate_pattern(stability: str) -> torch.Tensor:
            """Generate pattern with specified stability."""
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            if stability == "stable":
                return torch.tanh(torch.sin(x))
            return torch.sin(x) + 0.1 * torch.sin(2 * x)

        # Test stable patterns
        stable_patterns = torch.stack(
            [generate_pattern("stable") for _ in range(batch_size)]
        )
        stable_result = analyzer.validate_stability(flow, stable_patterns)
        assert isinstance(stable_result, NonlinearStabilityValidation)
        assert stable_result.stable
        assert stable_result.basin_size > 0

        # Test marginally stable patterns
        unstable_patterns = torch.stack(
            [generate_pattern("unstable") for _ in range(batch_size)]
        )
        unstable_result = analyzer.validate_stability(flow, unstable_patterns)
        assert isinstance(unstable_result, NonlinearStabilityValidation)
        assert not unstable_result.stable

    def test_perturbation_response(
        self,
        batch_size: int,
        spatial_dim: int,
        time_steps: int,
    ):
        """Test perturbation response analysis."""
        # Create perturbation analyzer
        analyzer = PerturbationAnalyzer()

        # Create pattern dynamics
        dynamics = PatternDynamics()

        # Generate test patterns and perturbations
        patterns = torch.stack(
            [
                torch.sin(torch.linspace(0, 2 * np.pi, spatial_dim))
                for _ in range(batch_size)
            ]
        )
        perturbation = 0.1 * torch.randn_like(patterns[0])

        # Test perturbation analysis
        result = analyzer.analyze_perturbation(dynamics, patterns[0], perturbation)
        assert isinstance(result, PerturbationMetrics)
        assert hasattr(result, "linear_response")
        assert hasattr(result, "recovery_time")
        assert hasattr(result, "stability_margin")
        assert hasattr(result, "max_amplitude")

        # Verify response properties
        assert result.recovery_time >= 0
        assert result.stability_margin >= 0
        assert result.max_amplitude >= 0

        # Test evolution response
        evolved = dynamics.evolve_pattern(patterns[0] + perturbation, steps=time_steps)
        assert isinstance(evolved, torch.Tensor)
        assert evolved.shape[0] == time_steps
        assert evolved.shape[-1] == spatial_dim

    def test_lyapunov_analysis(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test Lyapunov stability analysis."""
        # Create nonlinear stability validator for Lyapunov analysis
        analyzer = NonlinearStabilityValidator()

        # Generate test trajectory
        def generate_trajectory() -> torch.Tensor:
            """Generate test trajectory."""
            t = torch.linspace(0, 10, 100)
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            X, T = torch.meshgrid(x, t, indexing="ij")
            return torch.sin(X - 0.3 * T) + 0.1 * torch.randn_like(X)

        trajectories = torch.stack([generate_trajectory() for _ in range(batch_size)])

        # Test nonlinear stability validation
        result = analyzer.validate_stability(flow, trajectories)
        assert isinstance(result, NonlinearStabilityValidation)
        assert hasattr(result, "lyapunov_function")
        assert result.lyapunov_function >= 0

    def test_mode_stability(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test stability analysis of pattern modes."""
        # Create linear stability validator for mode analysis
        analyzer = LinearStabilityValidator()

        # Generate test patterns with multiple modes
        x = torch.linspace(0, 2 * np.pi, spatial_dim)
        patterns = []
        for _ in range(batch_size):
            # Combine multiple modes with different stabilities
            pattern = (
                torch.sin(x)  # Stable mode
                + 0.5 * torch.sin(2 * x)  # Less stable mode
                + 0.1 * torch.sin(3 * x)
            )  # Least stable mode
            patterns.append(pattern)
        patterns = torch.stack(patterns)

        # Test stability validation with mode decomposition
        result = analyzer.validate_stability(flow, patterns)
        assert isinstance(result, LinearStabilityValidation)
        assert hasattr(result, "eigenvectors")
        assert hasattr(result, "eigenvalues")
        assert result.eigenvectors.shape[-1] == patterns.shape[-1]

    def test_stability_metrics(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test stability metrics computation and aggregation."""
        # Create both linear and nonlinear analyzers
        linear_analyzer = LinearStabilityValidator()
        nonlinear_analyzer = NonlinearStabilityValidator()

        # Generate test patterns
        patterns = torch.stack(
            [
                torch.sin(torch.linspace(0, 2 * np.pi, spatial_dim))
                for _ in range(batch_size)
            ]
        )

        # Test both types of stability analysis
        linear_result = linear_analyzer.validate_stability(flow, patterns)
        nonlinear_result = nonlinear_analyzer.validate_stability(flow, patterns)

        # Verify metric properties
        assert isinstance(linear_result, LinearStabilityValidation)
        assert isinstance(nonlinear_result, NonlinearStabilityValidation)
        assert hasattr(linear_result, "stable")
        assert hasattr(nonlinear_result, "stable")

    def test_validation_integration(
        self,
        flow: GeometricFlow,
        batch_size: int,
        spatial_dim: int,
        time_steps: int,
    ):
        """Test integrated stability validation."""
        # Create analyzers
        linear_analyzer = LinearStabilityValidator()
        nonlinear_analyzer = NonlinearStabilityValidator()
        perturbation_analyzer = PerturbationAnalyzer()
        dynamics = PatternDynamics()

        # Generate test pattern evolution
        time_series = []
        for _ in range(batch_size):
            t = torch.linspace(0, 10, time_steps)
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            X, T = torch.meshgrid(x, t, indexing="ij")
            pattern = torch.sin(X - 0.3 * T) + 0.1 * torch.randn_like(X)
            time_series.append(pattern)
        time_series = torch.stack(time_series)

        # Run stability validations
        linear_result = linear_analyzer.validate_stability(flow, time_series[:, 0])
        nonlinear_result = nonlinear_analyzer.validate_stability(flow, time_series[:, 0])
        perturbation_result = perturbation_analyzer.analyze_perturbation(
            dynamics, time_series[0, 0]
        )

        # Verify results
        assert isinstance(linear_result, LinearStabilityValidation)
        assert isinstance(nonlinear_result, NonlinearStabilityValidation)
        assert isinstance(perturbation_result, PerturbationMetrics)

    def test_dynamical_system(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test dynamical system properties."""
        # Create pattern dynamics
        dynamics = PatternDynamics()

        # Generate test pattern
        pattern = torch.sin(torch.linspace(0, 2 * np.pi, spatial_dim))

        # Test evolution
        evolved = dynamics.evolve_pattern(pattern, steps=10)
        assert isinstance(evolved, torch.Tensor)
        assert evolved.shape[0] == 10  # Number of time steps
        assert evolved.shape[-1] == spatial_dim

        # Test Jacobian computation
        jacobian = dynamics.compute_jacobian(pattern)
        assert isinstance(jacobian, torch.Tensor)
        assert jacobian.shape == (pattern.numel(), pattern.numel())

    def test_bifurcation_theory(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test bifurcation theory properties."""
        # Create nonlinear stability validator
        analyzer = NonlinearStabilityValidator()

        # Generate test patterns at different parameter values
        patterns = []
        for param in torch.linspace(0, 1, batch_size):
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            pattern = torch.sin(x) + param * torch.sin(2 * x)
            patterns.append(pattern)
        patterns = torch.stack(patterns)

        # Test stability across parameter range
        result = analyzer.validate_stability(flow, patterns)
        assert isinstance(result, NonlinearStabilityValidation)
        assert hasattr(result, "stable")
        assert hasattr(result, "basin_size")

    def test_stability_analysis(
        self, flow: GeometricFlow, batch_size: int, spatial_dim: int
    ):
        """Test stability analysis methods."""
        # Create analyzers
        linear_analyzer = LinearStabilityValidator()
        nonlinear_analyzer = NonlinearStabilityValidator()
        dynamics = PatternDynamics()

        # Generate test pattern
        pattern = torch.sin(torch.linspace(0, 2 * np.pi, spatial_dim))

        # Test linear stability
        linear_result = linear_analyzer.validate_stability(flow, pattern)
        assert isinstance(linear_result, LinearStabilityValidation)
        assert hasattr(linear_result, "eigenvalues")
        assert hasattr(linear_result, "eigenvectors")

        # Test nonlinear stability
        nonlinear_result = nonlinear_analyzer.validate_stability(flow, pattern)
        assert isinstance(nonlinear_result, NonlinearStabilityValidation)
        assert hasattr(nonlinear_result, "lyapunov_function")
        assert hasattr(nonlinear_result, "basin_size")

        # Test evolution stability
        evolved = dynamics.evolve_pattern(pattern, steps=10)
        assert isinstance(evolved, torch.Tensor)
        assert evolved.shape[0] == 10
        assert evolved.shape[-1] == spatial_dim
