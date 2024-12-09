"""
Unit tests for flow validation.

Tests cover:
1. Energy conservation
2. Flow monotonicity
3. Long-time existence
4. Singularity detection
"""

import pytest
import torch

from src.validation.geometric.flow import (
    EnergyMetrics,
    FlowProperties,
    FlowValidator,
    SingularityDetector,
    ValidationResult,
)


class TestFlowValidation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def dim(self) -> int:
        return 8

    @pytest.fixture
    def time_steps(self) -> int:
        return 100

    @pytest.fixture
    def validator(self) -> FlowValidator:
        return FlowValidator(
            energy_threshold=1e-6,
            monotonicity_threshold=1e-4,
            singularity_threshold=1.0,
        )

    def test_energy_conservation(
        self, validator: FlowValidator, batch_size: int, dim: int
    ):
        """Test energy conservation in flows."""

        # Generate test flow
        def generate_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate conservative flow."""
            omega = torch.randn(batch_size)
            return torch.stack(
                [
                    torch.cos(omega.unsqueeze(-1) * t),
                    torch.sin(omega.unsqueeze(-1) * t),
                ],
                dim=-1,
            )

        # Generate time points
        t = torch.linspace(0, 10, 100)
        flow = generate_flow(t)

        # Test energy conservation
        result = validator.validate_energy_conservation(flow)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "energy_variation" in result.metrics

        # Test energy metrics
        metrics = validator.compute_energy_metrics(flow)
        assert isinstance(metrics, EnergyMetrics)
        assert hasattr(metrics, "total_energy")
        assert hasattr(metrics, "energy_derivative")

        # Test non-conservative flow
        def non_conservative_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate non-conservative flow."""
            return torch.exp(-0.1 * t) * generate_flow(t)

        flow_nc = non_conservative_flow(t)
        result = validator.validate_energy_conservation(flow_nc)
        assert not result.is_valid

    def test_flow_monotonicity(
        self, validator: FlowValidator, batch_size: int, dim: int
    ):
        """Test flow monotonicity properties."""

        # Generate monotonic flow
        def generate_monotonic_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate monotonically decreasing flow."""
            return torch.exp(-0.1 * t.unsqueeze(-1)) * torch.ones(batch_size, dim)

        t = torch.linspace(0, 10, 100)
        flow = generate_monotonic_flow(t)

        # Test monotonicity validation
        result = validator.validate_monotonicity(flow)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "monotonicity_measure" in result.metrics

        # Test flow properties
        properties = validator.compute_flow_properties(flow)
        assert isinstance(properties, FlowProperties)
        assert hasattr(properties, "derivative")
        assert hasattr(properties, "second_derivative")

        # Test non-monotonic flow
        def non_monotonic_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate oscillating flow."""
            return torch.sin(t.unsqueeze(-1)) * torch.ones(batch_size, dim)

        flow_nm = non_monotonic_flow(t)
        result = validator.validate_monotonicity(flow_nm)
        assert not result.is_valid

    def test_long_time_existence(
        self, validator: FlowValidator, batch_size: int, dim: int, time_steps: int
    ):
        """Test long-time existence properties."""

        # Generate long-time flow
        def generate_stable_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate stable long-time flow."""
            return torch.tanh(0.1 * t.unsqueeze(-1)) * torch.ones(batch_size, dim)

        t = torch.linspace(0, 100, time_steps)
        flow = generate_stable_flow(t)

        # Test long-time existence
        result = validator.validate_long_time_existence(flow)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "existence_time" in result.metrics

        # Test stability metrics
        stability = validator.compute_stability_metrics(flow)
        assert "lyapunov_exponent" in stability
        assert "stability_radius" in stability

        # Test finite-time blowup
        def blowup_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate flow with finite-time blowup."""
            return 1 / (1 - 0.1 * t.unsqueeze(-1)) * torch.ones(batch_size, dim)

        t_short = torch.linspace(0, 5, 50)
        flow_bu = blowup_flow(t_short)
        result = validator.validate_long_time_existence(flow_bu)
        assert not result.is_valid

    def test_singularity_detection(
        self, validator: FlowValidator, batch_size: int, dim: int
    ):
        """Test singularity detection in flows."""
        # Create singularity detector
        detector = SingularityDetector(threshold=validator.singularity_threshold)

        # Generate flow with singularity
        def generate_singular_flow(t: torch.Tensor) -> torch.Tensor:
            """Generate flow with removable singularity."""
            x = t.unsqueeze(-1) * torch.ones(batch_size, dim)
            return torch.sin(x) / x

        t = torch.linspace(-5, 5, 1000)
        flow = generate_singular_flow(t)

        # Test singularity detection
        singularities = detector.detect_singularities(flow, t)
        assert len(singularities) > 0
        for sing in singularities:
            assert hasattr(sing, "location")
            assert hasattr(sing, "type")

        # Test singularity classification
        classification = detector.classify_singularities(flow, t)
        assert isinstance(classification, dict)
        assert "removable" in classification
        assert "essential" in classification

        # Test singularity validation
        result = validator.validate_singularities(flow, t)
        assert isinstance(result, ValidationResult)
        assert "singularity_count" in result.metrics
        assert "singularity_strength" in result.metrics

    def test_flow_properties(self, validator: FlowValidator, batch_size: int, dim: int):
        """Test flow property computations."""
        # Generate test flow
        t = torch.linspace(0, 10, 100)
        flow = torch.exp(-0.1 * t.unsqueeze(-1)) * torch.randn(batch_size, dim)

        # Compute flow properties
        properties = validator.compute_flow_properties(flow)
        assert isinstance(properties, FlowProperties)

        # Test basic properties
        assert hasattr(properties, "velocity")
        assert hasattr(properties, "acceleration")
        assert hasattr(properties, "curvature")

        # Test derived properties
        assert hasattr(properties, "arc_length")
        assert hasattr(properties, "torsion")

        # Test property bounds
        assert torch.all(properties.arc_length >= 0)
        assert properties.velocity.shape == flow.shape
        assert properties.acceleration.shape == flow.shape

    def test_flow_decomposition(
        self, validator: FlowValidator, batch_size: int, dim: int
    ):
        """Test flow decomposition methods."""
        # Generate test flow
        t = torch.linspace(0, 10, 100)
        flow = torch.exp(-0.1 * t.unsqueeze(-1)) * torch.randn(batch_size, dim)

        # Test Hodge decomposition
        components = validator.decompose_flow(flow)
        assert "harmonic" in components
        assert "exact" in components
        assert "coexact" in components

        # Verify decomposition
        reconstructed = sum(components.values())
        assert torch.allclose(reconstructed, flow, rtol=1e-4)

        # Test component properties
        for component in components.values():
            assert component.shape == flow.shape
            properties = validator.compute_flow_properties(component)
            assert isinstance(properties, FlowProperties)

    def test_validation_integration(
        self, validator: FlowValidator, batch_size: int, dim: int, time_steps: int
    ):
        """Test integrated flow validation."""
        # Generate test flow
        t = torch.linspace(0, 10, time_steps)
        flow = torch.exp(-0.1 * t.unsqueeze(-1)) * torch.randn(batch_size, dim)

        # Run full validation
        result = validator.validate_flow(flow, t)
        assert isinstance(result, dict)
        assert "energy" in result
        assert "monotonicity" in result
        assert "existence" in result
        assert "singularities" in result

        # Check validation scores
        assert all(0 <= score <= 1 for score in result.values())
        assert "overall_score" in result

        # Test validation with parameters
        params = torch.linspace(0, 1, time_steps)
        param_result = validator.validate_flow_family(flow, t, params)
        assert "parameter_dependence" in param_result

        # Test validation summary
        summary = validator.get_validation_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0
