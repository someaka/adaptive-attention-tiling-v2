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

from src.validation.patterns.stability import (
    LinearStabilityAnalyzer,
    LyapunovAnalyzer,
    NonlinearStabilityAnalyzer,
    PatternStabilityValidator,
    StabilityMetrics,
)
from src.validation.patterns.perturbation import PerturbationAnalyzer


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
    def validator(self) -> PatternStabilityValidator:
        return PatternStabilityValidator(
            linear_threshold=0.1, nonlinear_threshold=0.2, lyapunov_threshold=0.01
        )

    def test_linear_stability(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test linear stability analysis."""
        # Create linear stability analyzer
        analyzer = LinearStabilityAnalyzer()

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
        stable_result = analyzer.analyze_stability(stable_patterns)
        assert stable_result.is_stable
        assert stable_result.growth_rate < validator.linear_threshold

        # Test unstable patterns
        unstable_patterns = torch.stack(
            [generate_unstable_pattern() for _ in range(batch_size)]
        )
        unstable_result = analyzer.analyze_stability(unstable_patterns)
        assert not unstable_result.is_stable
        assert unstable_result.growth_rate > validator.linear_threshold

        # Test eigenvalue computation
        eigenvalues = analyzer.compute_eigenvalues(stable_patterns)
        assert torch.all(eigenvalues.real <= validator.linear_threshold)

    def test_nonlinear_stability(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test nonlinear stability analysis."""
        # Create nonlinear stability analyzer
        analyzer = NonlinearStabilityAnalyzer()

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
        stable_result = analyzer.analyze_stability(stable_patterns)
        assert stable_result.is_stable
        assert stable_result.basin_radius > 0

        # Test marginally stable patterns
        unstable_patterns = torch.stack(
            [generate_pattern("unstable") for _ in range(batch_size)]
        )
        unstable_result = analyzer.analyze_stability(unstable_patterns)
        assert not unstable_result.is_stable

        # Test basin of attraction
        basin = analyzer.compute_basin_of_attraction(stable_patterns)
        assert basin.radius > 0
        assert hasattr(basin, "boundary")

    def test_perturbation_response(
        self,
        validator: PatternStabilityValidator,
        batch_size: int,
        spatial_dim: int,
        time_steps: int,
    ):
        """Test perturbation response analysis."""
        # Create perturbation analyzer
        analyzer = PerturbationAnalyzer()

        # Generate test patterns and perturbations
        patterns = torch.stack(
            [
                torch.sin(torch.linspace(0, 2 * np.pi, spatial_dim))
                for _ in range(batch_size)
            ]
        )
        perturbations = [
            0.01 * torch.randn_like(patterns),  # Small perturbation
            0.1 * torch.randn_like(patterns),  # Medium perturbation
            1.0 * torch.randn_like(patterns),  # Large perturbation
        ]

        # Test response to different perturbation sizes
        for perturbation in perturbations:
            response = analyzer.analyze_response(patterns, perturbation)
            assert hasattr(response, "amplitude")
            assert hasattr(response, "decay_rate")
            assert hasattr(response, "recovery_time")

            # Verify response properties
            assert response.amplitude >= 0
            assert response.decay_rate >= 0
            assert response.recovery_time >= 0

        # Test recovery prediction
        recovery = analyzer.predict_recovery(patterns, perturbations[0])
        assert recovery.shape == (batch_size, time_steps, spatial_dim)

        # Test stability margin
        margin = analyzer.compute_stability_margin(patterns)
        assert margin > 0

    def test_lyapunov_analysis(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test Lyapunov stability analysis."""
        # Create Lyapunov analyzer
        analyzer = LyapunovAnalyzer()

        # Generate test trajectory
        def generate_trajectory() -> torch.Tensor:
            """Generate test trajectory."""
            t = torch.linspace(0, 10, 100)
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            X, T = torch.meshgrid(x, t, indexing="ij")
            return torch.sin(X - 0.3 * T) + 0.1 * torch.randn_like(X)

        trajectories = torch.stack([generate_trajectory() for _ in range(batch_size)])

        # Test Lyapunov exponent computation
        exponents = analyzer.compute_lyapunov_exponents(trajectories)
        assert exponents.shape[0] == batch_size
        assert torch.all(exponents < validator.lyapunov_threshold)

        # Test stability classification
        stability = analyzer.classify_stability(trajectories)
        assert isinstance(stability, dict)
        assert "type" in stability
        assert "confidence" in stability

        # Test attractor dimension
        dimension = analyzer.estimate_attractor_dimension(trajectories)
        assert dimension > 0

        # Test predictability time
        pred_time = analyzer.estimate_predictability_time(trajectories)
        assert pred_time > 0

    def test_mode_stability(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test stability analysis of pattern modes."""
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

        # Test mode decomposition
        modes = validator.decompose_modes(patterns)
        assert len(modes) >= 3

        # Test stability of individual modes
        mode_stability = validator.analyze_mode_stability(modes)
        assert len(mode_stability) == len(modes)
        assert all(0 <= stability <= 1 for stability in mode_stability)

        # Test mode interaction analysis
        interactions = validator.analyze_mode_interactions(modes)
        assert isinstance(interactions, torch.Tensor)
        assert interactions.shape == (len(modes), len(modes))

    def test_stability_metrics(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test stability metrics computation and aggregation."""
        # Generate test patterns
        patterns = torch.stack(
            [
                torch.sin(torch.linspace(0, 2 * np.pi, spatial_dim))
                for _ in range(batch_size)
            ]
        )

        # Compute stability metrics
        metrics = validator.compute_stability_metrics(patterns)
        assert isinstance(metrics, StabilityMetrics)

        # Test metric properties
        assert hasattr(metrics, "linear_stability")
        assert hasattr(metrics, "nonlinear_stability")
        assert hasattr(metrics, "lyapunov_stability")

        # Test metric bounds
        assert 0 <= metrics.linear_stability <= 1
        assert 0 <= metrics.nonlinear_stability <= 1
        assert 0 <= metrics.lyapunov_stability <= 1

        # Test metric aggregation
        overall = metrics.compute_overall_stability()
        assert 0 <= overall <= 1

        # Test stability classification
        classification = metrics.classify_stability()
        assert "category" in classification
        assert "confidence" in classification

    def test_validation_integration(
        self,
        validator: PatternStabilityValidator,
        batch_size: int,
        spatial_dim: int,
        time_steps: int,
    ):
        """Test integrated stability validation."""
        # Generate test pattern evolution
        time_series = []
        for _ in range(batch_size):
            t = torch.linspace(0, 10, time_steps)
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            X, T = torch.meshgrid(x, t, indexing="ij")
            pattern = torch.sin(X - 0.3 * T) + 0.1 * torch.randn_like(X)
            time_series.append(pattern)
        time_series = torch.stack(time_series)

        # Run full validation
        result = validator.validate_stability(time_series)
        assert isinstance(result, Dict)
        assert "linear" in result
        assert "nonlinear" in result
        assert "perturbation" in result
        assert "lyapunov" in result
        assert "modes" in result

        # Check validation scores
        assert all(0 <= score <= 1 for score in result.values())
        assert "overall_score" in result

        # Test validation with parameters
        params = torch.linspace(0, 2, time_steps)
        param_result = validator.validate_stability(time_series, parameters=params)
        assert "parameter_dependence" in param_result

    def test_dynamical_system(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test dynamical system properties."""

        # Test vector field
        def test_vector_field():
            """Test vector field properties."""
            # Get vector field
            F = validator.get_vector_field()
            assert validator.validate_vector_field(F)

            # Test smoothness
            assert validator.validate_smoothness(F)

            # Test boundedness
            if validator.is_bounded():
                assert validator.validate_boundedness(F)

            return F

        F = test_vector_field()

        # Test flow properties
        def test_flow():
            """Test flow properties."""
            # Get flow
            phi = validator.get_flow(F)
            assert validator.validate_flow(phi)

            # Test group property
            assert validator.validate_group_property(phi)

            # Test continuity
            assert validator.validate_continuity(phi)

            return phi

        phi = test_flow()

        # Test invariant sets
        def test_invariant_sets():
            """Test invariant set properties."""
            # Get invariant sets
            inv_sets = validator.get_invariant_sets()
            assert validator.validate_invariance(inv_sets)

            # Test stability
            for inv_set in inv_sets:
                assert validator.validate_set_stability(inv_set)

            # Test chain recurrence
            if validator.has_chain_recurrence():
                assert validator.validate_chain_recurrence(inv_sets)

            return inv_sets

        inv_sets = test_invariant_sets()

    def test_bifurcation_theory(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test bifurcation theory properties."""

        # Test parameter dependence
        def test_parameters():
            """Test parameter dependence."""
            # Get parameter space
            params = validator.get_parameter_space()
            assert validator.validate_parameter_space(params)

            # Test smoothness
            assert validator.validate_parameter_smoothness(params)

            # Test compactness
            if validator.is_compact():
                assert validator.validate_compactness(params)

            return params

        params = test_parameters()

        # Test bifurcation types
        def test_bifurcations():
            """Test bifurcation types."""
            # Get bifurcations
            bifs = validator.get_bifurcations()
            assert validator.validate_bifurcations(bifs)

            # Test normal forms
            for bif in bifs:
                assert validator.validate_normal_form(bif)

            # Test codimension
            if validator.has_codimension():
                for bif in bifs:
                    assert validator.validate_codimension(bif)

            return bifs

        bifs = test_bifurcations()

        # Test unfolding theory
        def test_unfolding():
            """Test unfolding theory."""
            # Get unfolding
            unf = validator.get_unfolding()
            assert validator.validate_unfolding(unf)

            # Test versality
            assert validator.validate_versality(unf)

            # Test recognition
            if validator.has_recognition():
                assert validator.validate_recognition(unf)

            return unf

        unf = test_unfolding()

    def test_stability_analysis(
        self, validator: PatternStabilityValidator, batch_size: int, spatial_dim: int
    ):
        """Test stability analysis methods."""

        # Test equilibrium stability
        def test_equilibrium():
            """Test equilibrium stability."""
            # Get equilibria
            eq = validator.get_equilibria()
            assert validator.validate_equilibria(eq)

            # Test hyperbolicity
            for e in eq:
                assert validator.validate_hyperbolicity(e)

            # Test index
            if validator.has_index():
                for e in eq:
                    assert validator.validate_index(e)

            return eq

        eq = test_equilibrium()

        # Test periodic orbit stability
        def test_periodic():
            """Test periodic orbit stability."""
            # Get periodic orbits
            po = validator.get_periodic_orbits()
            assert validator.validate_periodic_orbits(po)

            # Test Floquet theory
            for p in po:
                assert validator.validate_floquet(p)

            # Test averaging
            if validator.has_averaging():
                for p in po:
                    assert validator.validate_averaging(p)

            return po

        po = test_periodic()

        # Test heteroclinic stability
        def test_heteroclinic():
            """Test heteroclinic stability."""
            # Get heteroclinic orbits
            het = validator.get_heteroclinic_orbits()
            assert validator.validate_heteroclinic(het)

            # Test transversality
            for h in het:
                assert validator.validate_transversality(h)

            # Test persistence
            if validator.has_persistence():
                for h in het:
                    assert validator.validate_persistence(h)

            return het

        het = test_heteroclinic()
