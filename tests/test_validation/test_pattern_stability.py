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
import os
import numpy as np
import pytest
import torch
import yaml

from validation.flow.flow_stability import (
    LinearStabilityValidator,
    NonlinearStabilityValidator,
    LinearStabilityValidation,
    NonlinearStabilityValidation,
    StabilityValidationResult
)
from src.validation.patterns.perturbation import (
    PerturbationAnalyzer,
    PerturbationMetrics
)
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.tiling.geometric_flow import GeometricFlow


class TestPatternStability:
    @pytest.fixture
    def test_config(self):
        """Load test configuration based on environment."""
        config_name = os.environ.get("TEST_REGIME", "debug")
        config_path = f"configs/test_regimens/{config_name}.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    @pytest.fixture
    def setup_test_parameters(self, test_config):
        """Setup test parameters from configuration."""
        return {
            'batch_size': int(test_config['fiber_bundle']['batch_size']),
            'spatial_dim': int(test_config['geometric_tests']['dimensions']),
            'time_steps': int(test_config['parallel_transport']['path_points']),
            'hidden_dim': int(test_config['geometric_tests']['hidden_dim']),
            'manifold_dim': int(test_config['quantum_geometric']['manifold_dim']),
            'dt': float(test_config['quantum_geometric']['dt']),
            'tolerance': float(test_config['fiber_bundle']['tolerance'])
        }

    @pytest.fixture
    def flow(self, setup_test_parameters) -> GeometricFlow:
        return GeometricFlow(
            hidden_dim=setup_test_parameters['hidden_dim'],
            manifold_dim=setup_test_parameters['manifold_dim']
        )

    def test_linear_stability(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test linear stability analysis."""
        # Create linear stability analyzer
        analyzer = LinearStabilityValidator()

        # Generate test patterns with known stability
        def generate_stable_pattern():
            """Generate linearly stable pattern."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = torch.sin(X) + torch.sin(Y) + 0.1 * torch.randn_like(X)
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        def generate_unstable_pattern():
            """Generate linearly unstable pattern."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = torch.exp(0.1 * X) * torch.sin(Y)
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        # Test stable patterns
        stable_patterns = torch.cat(
            [generate_stable_pattern() for _ in range(setup_test_parameters['batch_size'])], dim=0
        )
        stable_result = analyzer.validate_stability(flow, stable_patterns)
        assert isinstance(stable_result, (LinearStabilityValidation, StabilityValidationResult))
        assert stable_result.is_valid
        assert stable_result.data is not None
        assert 'eigenvalues' in stable_result.data
        assert 'growth_rates' in stable_result.data
        assert torch.all(stable_result.data['growth_rates'] < 1.0)

        # Test unstable patterns
        unstable_patterns = torch.cat(
            [generate_unstable_pattern() for _ in range(setup_test_parameters['batch_size'])], dim=0
        )
        unstable_result = analyzer.validate_stability(flow, unstable_patterns)
        assert isinstance(unstable_result, (LinearStabilityValidation, StabilityValidationResult))
        assert not unstable_result.is_valid
        assert unstable_result.data is not None
        assert 'eigenvalues' in unstable_result.data
        assert 'growth_rates' in unstable_result.data
        assert torch.any(unstable_result.data['growth_rates'] > 1.0)

    def test_nonlinear_stability(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test nonlinear stability analysis."""
        # Create nonlinear stability analyzer
        analyzer = NonlinearStabilityValidator()

        # Generate test patterns with nonlinear dynamics
        def generate_pattern(stability: str) -> torch.Tensor:
            """Generate pattern with specified stability."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            if stability == "stable":
                pattern = torch.tanh(torch.sin(X) + torch.sin(Y))
            else:
                pattern = torch.sin(X) + torch.sin(Y) + 0.1 * torch.sin(2 * X + Y)
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        # Test stable patterns
        stable_patterns = torch.cat(
            [generate_pattern("stable") for _ in range(setup_test_parameters['batch_size'])], dim=0
        )
        stable_result = analyzer.validate_stability(flow, stable_patterns)
        assert isinstance(stable_result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert stable_result.is_valid
        assert stable_result.data is not None
        assert 'basin_size' in stable_result.data
        assert stable_result.data['basin_size'] > 0

        # Test marginally stable patterns
        unstable_patterns = torch.cat(
            [generate_pattern("unstable") for _ in range(setup_test_parameters['batch_size'])], dim=0
        )
        unstable_result = analyzer.validate_stability(flow, unstable_patterns)
        assert isinstance(unstable_result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert not unstable_result.is_valid

    def test_perturbation_response(
        self,
        setup_test_parameters: Dict,
    ):
        """Test perturbation response analysis."""
        # Create perturbation analyzer
        analyzer = PerturbationAnalyzer()

        # Create pattern dynamics
        dynamics = PatternDynamics()

        # Generate test patterns and perturbations
        def generate_pattern():
            """Generate test pattern."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = torch.sin(X) + torch.sin(Y)
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        patterns = torch.cat(
            [generate_pattern() for _ in range(setup_test_parameters['batch_size'])], dim=0
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
        evolved = dynamics.evolve_pattern(patterns[0] + perturbation, steps=setup_test_parameters['time_steps'])
        assert isinstance(evolved, torch.Tensor)
        assert evolved.shape[0] == setup_test_parameters['time_steps']
        assert evolved.shape[-1] == setup_test_parameters['spatial_dim']

    def test_lyapunov_analysis(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test Lyapunov stability analysis."""
        # Create nonlinear stability validator for Lyapunov analysis
        analyzer = NonlinearStabilityValidator()

        # Generate test trajectory
        def generate_trajectory() -> torch.Tensor:
            """Generate test trajectory."""
            # Use fewer time points and avoid meshgrid
            t = torch.linspace(0, 5, 25)  # Reduced time range and points
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            trajectory = torch.zeros(1, 1, setup_test_parameters['spatial_dim'], setup_test_parameters['spatial_dim'], len(t))
            for i, ti in enumerate(t):
                pattern = torch.sin(X - 0.3 * ti) + torch.sin(Y - 0.2 * ti) + 0.1 * torch.randn_like(X)
                trajectory[..., i] = pattern
            return trajectory.squeeze(-1)

        trajectories = torch.cat([generate_trajectory() for _ in range(setup_test_parameters['batch_size'])], dim=0)

        # Test nonlinear stability validation
        result = analyzer.validate_stability(flow, trajectories)
        assert isinstance(result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert result.data is not None
        assert 'lyapunov_function' in result.data
        assert result.data['lyapunov_function'] >= 0

    def test_mode_stability(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test stability analysis of pattern modes."""
        # Create linear stability validator for mode analysis
        analyzer = LinearStabilityValidator()

        # Generate test patterns with multiple modes
        def generate_pattern():
            """Generate test pattern with multiple modes."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = (
                torch.sin(X) + torch.sin(Y)  # Stable mode
                + 0.5 * torch.sin(2 * X + Y)  # Less stable mode
                + 0.1 * torch.sin(3 * X + 2 * Y)  # Least stable mode
            )
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        patterns = torch.cat([generate_pattern() for _ in range(setup_test_parameters['batch_size'])], dim=0)

        # Test stability validation with mode decomposition
        result = analyzer.validate_stability(flow, patterns)
        assert isinstance(result, (LinearStabilityValidation, StabilityValidationResult))
        assert result.data is not None
        assert 'eigenvalues' in result.data
        assert 'eigenvectors' in result.data
        assert result.data['eigenvectors'].shape[-1] == patterns.shape[-1]

    def test_stability_metrics(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test stability metrics computation and aggregation."""
        # Create both linear and nonlinear analyzers
        linear_analyzer = LinearStabilityValidator()
        nonlinear_analyzer = NonlinearStabilityValidator()

        # Generate test patterns
        def generate_pattern():
            """Generate test pattern."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = torch.sin(X) + torch.sin(Y)
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        patterns = torch.cat([generate_pattern() for _ in range(setup_test_parameters['batch_size'])], dim=0)

        # Test both types of stability analysis
        linear_result = linear_analyzer.validate_stability(flow, patterns)
        nonlinear_result = nonlinear_analyzer.validate_stability(flow, patterns)

        # Verify metric properties
        assert isinstance(linear_result, (LinearStabilityValidation, StabilityValidationResult))
        assert isinstance(nonlinear_result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert linear_result.data is not None
        assert nonlinear_result.data is not None
        assert 'eigenvalues' in linear_result.data
        assert 'lyapunov_function' in nonlinear_result.data

    def test_validation_integration(
        self,
        flow: GeometricFlow,
        setup_test_parameters: Dict,
    ):
        """Test integrated stability validation."""
        # Create analyzers
        linear_analyzer = LinearStabilityValidator()
        nonlinear_analyzer = NonlinearStabilityValidator()
        perturbation_analyzer = PerturbationAnalyzer()
        dynamics = PatternDynamics()

        # Generate test pattern evolution
        def generate_pattern():
            """Generate test pattern evolution."""
            t = torch.linspace(0, 10, setup_test_parameters['time_steps'])
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = torch.zeros(1, 1, setup_test_parameters['spatial_dim'], setup_test_parameters['spatial_dim'], setup_test_parameters['time_steps'])
            for i, ti in enumerate(t):
                pattern[..., i] = torch.sin(X - 0.3 * ti) + torch.sin(Y - 0.2 * ti) + 0.1 * torch.randn_like(X)
            return pattern.squeeze(-1)

        time_series = torch.cat([generate_pattern() for _ in range(setup_test_parameters['batch_size'])], dim=0)

        # Run stability validations
        linear_result = linear_analyzer.validate_stability(flow, time_series[..., 0])
        nonlinear_result = nonlinear_analyzer.validate_stability(flow, time_series[..., 0])
        perturbation_result = perturbation_analyzer.analyze_perturbation(
            dynamics, time_series[0, ..., 0], 0.1 * torch.randn_like(time_series[0, ..., 0])
        )

        # Verify results
        assert isinstance(linear_result, (LinearStabilityValidation, StabilityValidationResult))
        assert isinstance(nonlinear_result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert isinstance(perturbation_result, PerturbationMetrics)

    def test_dynamical_system(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test dynamical system properties."""
        # Create pattern dynamics
        dynamics = PatternDynamics()

        # Generate test pattern
        x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
        y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
        X, Y = torch.meshgrid(x, y, indexing="ij")
        pattern = torch.sin(X) + torch.sin(Y)
        # Reshape to [1, 1, spatial_dim, spatial_dim]
        pattern = pattern.unsqueeze(0).unsqueeze(0)

        # Test evolution
        evolved = dynamics.evolve_pattern(pattern, steps=setup_test_parameters['time_steps'])
        assert isinstance(evolved, torch.Tensor)
        assert evolved.shape[0] == setup_test_parameters['time_steps']  # Number of time steps
        assert evolved.shape[-1] == setup_test_parameters['spatial_dim']

        # Test Jacobian computation
        jacobian = dynamics.compute_jacobian(pattern)
        assert isinstance(jacobian, torch.Tensor)
        assert jacobian.shape == (pattern.numel(), pattern.numel())

    def test_bifurcation_theory(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test bifurcation theory properties."""
        # Create nonlinear stability validator
        analyzer = NonlinearStabilityValidator()

        # Generate test patterns at different parameter values
        def generate_pattern(param: float) -> torch.Tensor:
            """Generate test pattern with parameter."""
            x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            pattern = torch.sin(X) + torch.sin(Y) + param * torch.sin(2 * X + Y)
            # Reshape to [1, 1, spatial_dim, spatial_dim]
            return pattern.unsqueeze(0).unsqueeze(0)

        patterns = torch.cat(
            [generate_pattern(float(param)) for param in torch.linspace(0, 1, setup_test_parameters['batch_size'])],
            dim=0
        )

        # Test stability across parameter range
        result = analyzer.validate_stability(flow, patterns)
        assert isinstance(result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert result.data is not None
        assert 'basin_size' in result.data

    def test_stability_analysis(
        self, flow: GeometricFlow, setup_test_parameters: Dict
    ):
        """Test stability analysis methods."""
        # Create analyzers
        linear_analyzer = LinearStabilityValidator()
        nonlinear_analyzer = NonlinearStabilityValidator()
        dynamics = PatternDynamics()

        # Generate test pattern
        x = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
        y = torch.linspace(0, 2 * np.pi, setup_test_parameters['spatial_dim'])
        X, Y = torch.meshgrid(x, y, indexing="ij")
        pattern = torch.sin(X) + torch.sin(Y)
        # Reshape to [1, 1, spatial_dim, spatial_dim]
        pattern = pattern.unsqueeze(0).unsqueeze(0)

        # Test linear stability
        linear_result = linear_analyzer.validate_stability(flow, pattern)
        assert isinstance(linear_result, (LinearStabilityValidation, StabilityValidationResult))
        assert linear_result.data is not None
        assert 'eigenvalues' in linear_result.data
        assert 'eigenvectors' in linear_result.data

        # Test nonlinear stability
        nonlinear_result = nonlinear_analyzer.validate_stability(flow, pattern)
        assert isinstance(nonlinear_result, (NonlinearStabilityValidation, StabilityValidationResult))
        assert nonlinear_result.data is not None
        assert 'lyapunov_function' in nonlinear_result.data
        assert 'basin_size' in nonlinear_result.data

        # Test evolution stability
        evolved = dynamics.evolve_pattern(pattern, steps=setup_test_parameters['time_steps'])
        assert isinstance(evolved, torch.Tensor)
        assert evolved.shape[0] == setup_test_parameters['time_steps']
        assert evolved.shape[-1] == setup_test_parameters['spatial_dim']
