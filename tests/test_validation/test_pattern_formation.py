"""
Unit tests for pattern formation validation.

Tests cover:
1. Pattern emergence
2. Spatial organization
3. Temporal evolution
4. Bifurcation analysis
5. Mode decomposition
6. Reaction-diffusion pattern formation
7. Symmetry breaking in pattern formation
8. Pattern stability analysis
"""

import pytest
import torch
import numpy as np

from src.validation.patterns.formation import (
    PatternFormationValidator,
    EmergenceValidator,
    SpatialValidator,
    TemporalValidator,
    BifurcationAnalyzer,
    PatternModeAnalyzer,
    EmergenceValidation,
    SpatialValidation,
    TemporalValidation,
    BifurcationPoint,
    EmergenceMetrics,
    SpatialMetrics,
    TemporalMetrics,
    FormationValidationResult,
)


class TestPatternFormation:
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
    def validator(self) -> PatternFormationValidator:
        return PatternFormationValidator(
            tolerance=1e-6,
            coherence_threshold=0.8,
            symmetry_threshold=0.9,
            defect_threshold=0.1,
            frequency_threshold=0.1,
            phase_threshold=0.1,
        )

    def test_pattern_emergence(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test pattern emergence validation."""
        # Create test pattern
        pattern = torch.zeros(batch_size, spatial_dim)
        for i in range(batch_size):
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            pattern[i] = torch.sin(x) + 0.1 * torch.randn_like(x)

        # Test emergence validation
        result = validator.emergence_validator.validate_emergence(pattern)
        assert isinstance(result, EmergenceValidation)
        assert hasattr(result, "emerged")
        assert hasattr(result, "formation_time")
        assert hasattr(result, "coherence")
        assert hasattr(result, "stability")

    def test_spatial_organization(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test spatial organization validation."""
        # Create test pattern
        pattern = torch.zeros(batch_size, spatial_dim)
        for i in range(batch_size):
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            pattern[i] = torch.sin(x) + 0.1 * torch.randn_like(x)

        # Test spatial validation
        result = validator.spatial_validator.validate_spatial(pattern)
        assert isinstance(result, SpatialValidation)
        assert hasattr(result, "wavelength")
        assert hasattr(result, "symmetry")
        assert hasattr(result, "defects")
        assert hasattr(result, "correlation")

    def test_temporal_evolution(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test temporal evolution validation."""
        # Create test pattern sequence
        time_steps = 100
        trajectory = torch.zeros(time_steps, batch_size, spatial_dim)
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            for i in range(batch_size):
                x = torch.linspace(0, 2 * np.pi, spatial_dim)
                trajectory[t, i] = torch.sin(x + phase) + 0.1 * torch.randn_like(x)

        # Test temporal validation
        result = validator.temporal_validator.validate_temporal(trajectory)
        assert isinstance(result, TemporalValidation)
        assert hasattr(result, "frequency")
        assert hasattr(result, "phase_locked")
        assert hasattr(result, "drift_rate")
        assert hasattr(result, "persistence")

    def test_bifurcation_analysis(
        self, validator: PatternFormationValidator, spatial_dim: int
    ):
        """Test bifurcation analysis."""
        # Create test pattern sequence with varying parameter
        parameter_range = (-1.0, 1.0)
        num_points = 100
        parameter_values = torch.linspace(parameter_range[0], parameter_range[1], num_points)
        patterns = []
        for param in parameter_values:
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            pattern = torch.sin(x) + param * torch.cos(2 * x) + 0.1 * torch.randn_like(x)
            patterns.append(pattern)
        patterns = torch.stack(patterns)

        # Test bifurcation analysis
        bifurcation_analyzer = BifurcationAnalyzer(parameter_range, num_points)
        points = bifurcation_analyzer.analyze_bifurcations(None, patterns[0], "param")
        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, BifurcationPoint)
            assert hasattr(point, "parameter_value")
            assert hasattr(point, "pattern_type")
            assert hasattr(point, "eigenvalues")
            assert hasattr(point, "eigenvectors")

    def test_mode_decomposition(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test mode decomposition."""
        # Create test pattern with multiple modes
        pattern = torch.zeros(batch_size, spatial_dim)
        for i in range(batch_size):
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            pattern[i] = torch.sin(x) + 0.5 * torch.sin(2 * x) + 0.1 * torch.randn_like(x)

        # Test mode decomposition
        decomposer = PatternModeAnalyzer()
        modes = decomposer.analyze_modes(pattern)
        assert isinstance(modes, dict)
        assert "modes" in modes
        assert "amplitudes" in modes
        assert "correlations" in modes
        assert isinstance(modes["modes"], torch.Tensor)
        assert isinstance(modes["amplitudes"], torch.Tensor)
        assert isinstance(modes["correlations"], torch.Tensor)

    def test_validation_integration(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test integration of all validation components."""
        # Create test pattern sequence
        time_steps = 100
        trajectory = torch.zeros(time_steps, batch_size, spatial_dim)
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            for i in range(batch_size):
                x = torch.linspace(0, 2 * np.pi, spatial_dim)
                trajectory[t, i] = torch.sin(x + phase) + 0.1 * torch.randn_like(x)

        # Test emergence validation
        emergence = validator.emergence_validator.validate_emergence(trajectory[-1])
        assert isinstance(emergence, EmergenceValidation)

        # Test spatial validation
        spatial = validator.spatial_validator.validate_spatial(trajectory[-1])
        assert isinstance(spatial, SpatialValidation)

        # Test temporal validation
        temporal = validator.temporal_validator.validate_temporal(trajectory)
        assert isinstance(temporal, TemporalValidation)

        # Test complete validation
        result = validator.validate(None, trajectory[0], time_steps)
        assert isinstance(result, FormationValidationResult)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'message')
        assert hasattr(result, 'data')
        assert result.data is not None, "Validation result data should not be None"
        assert 'emergence' in result.data
        assert 'spatial' in result.data
        assert 'temporal' in result.data

    def test_reaction_diffusion(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test reaction-diffusion pattern validation."""
        # Create test pattern sequence
        time_steps = 100
        trajectory = torch.zeros(time_steps, batch_size, spatial_dim)
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            for i in range(batch_size):
                x = torch.linspace(0, 2 * np.pi, spatial_dim)
                # Simulate Turing pattern formation
                u = torch.sin(x + phase) + 0.1 * torch.randn_like(x)
                v = torch.cos(2 * x + phase) + 0.1 * torch.randn_like(x)
                trajectory[t, i] = u - v

        # Test emergence validation
        emergence = validator.emergence_validator.validate_emergence(trajectory[-1])
        assert isinstance(emergence, EmergenceValidation)
        assert hasattr(emergence, "emerged")
        assert hasattr(emergence, "formation_time")

        # Test spatial validation
        spatial = validator.spatial_validator.validate_spatial(trajectory[-1])
        assert isinstance(spatial, SpatialValidation)
        assert hasattr(spatial, "wavelength")
        assert hasattr(spatial, "symmetry")

        # Test temporal validation
        temporal = validator.temporal_validator.validate_temporal(trajectory)
        assert isinstance(temporal, TemporalValidation)
        assert hasattr(temporal, "frequency")
        assert hasattr(temporal, "phase_locked")

    def test_symmetry_breaking(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test symmetry breaking detection."""
        # Create test pattern
        pattern = torch.zeros(batch_size, spatial_dim)
        for i in range(batch_size):
            x = torch.linspace(0, 2 * np.pi, spatial_dim)
            pattern[i] = torch.sin(x) + 0.1 * torch.randn_like(x)

        # Test spatial validation
        result = validator.spatial_validator.validate_spatial(pattern)
        assert isinstance(result, SpatialValidation)
        assert hasattr(result, "wavelength")
        assert hasattr(result, "symmetry")
        assert hasattr(result, "defects")
        assert hasattr(result, "correlation")

        # Test symmetry properties
        assert result.symmetry in ["translation", "rotation", "reflection", "none"]
        assert result.defects >= 0
        assert 0 <= result.correlation <= 1

    def test_pattern_stability(
        self, validator: PatternFormationValidator, batch_size: int, spatial_dim: int
    ):
        """Test pattern stability validation."""
        # Create test pattern sequence
        time_steps = 100
        trajectory = torch.zeros(time_steps, batch_size, spatial_dim)
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            for i in range(batch_size):
                x = torch.linspace(0, 2 * np.pi, spatial_dim)
                trajectory[t, i] = torch.sin(x + phase) + 0.1 * torch.randn_like(x)

        # Test emergence validation
        emergence = validator.emergence_validator.validate_emergence(trajectory[-1])
        assert isinstance(emergence, EmergenceValidation)
        assert hasattr(emergence, "stability")

        # Test temporal validation
        temporal = validator.temporal_validator.validate_temporal(trajectory)
        assert isinstance(temporal, TemporalValidation)
        assert hasattr(temporal, "persistence")
