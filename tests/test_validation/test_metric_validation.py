"""
Unit tests for metric validation.

Tests cover:
1. Positive definite properties
2. Compatibility conditions
3. Smoothness properties
4. Curvature bounds
"""

from typing import Dict

import numpy as np
import pytest
import torch

from src.validation.geometric.metric import (
    CurvatureBounds,
    MetricProperties,
    MetricValidator,
    SmoothnessMetrics,
)
from src.validation.framework import ValidationResult


class TestMetricValidation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def dim(self) -> int:
        return 8

    @pytest.fixture
    def validator(self) -> MetricValidator:
        return MetricValidator(
            positive_definite_threshold=1e-6,
            smoothness_threshold=0.1,
            curvature_bound=1.0,
        )

    def test_positive_definite(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test positive definite validation."""
        # Generate positive definite metric
        matrix = torch.randn(batch_size, dim, dim)
        pd_metric = matrix @ matrix.transpose(-1, -2)
        pd_metric = (
            pd_metric + torch.eye(dim) * 1e-3
        )  # Ensure strict positive definiteness

        # Test positive definite validation
        result = validator.validate_positive_definite(pd_metric)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.metrics["min_eigenvalue"] > 0

        # Test non-positive definite metric
        non_pd_metric = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        non_pd_metric[:, 0, 0] = -1.0
        result = validator.validate_positive_definite(non_pd_metric)
        assert not result.is_valid
        assert result.metrics["min_eigenvalue"] < 0

        # Test borderline case
        borderline_metric = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        borderline_metric[:, 0, 0] = validator.positive_definite_threshold / 2
        result = validator.validate_positive_definite(borderline_metric)
        assert not result.is_valid

    def test_compatibility(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test metric compatibility conditions."""
        # Generate compatible metrics
        base_metric = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        compatible_metric = base_metric + 0.1 * torch.randn(batch_size, dim, dim)
        compatible_metric = (
            compatible_metric + compatible_metric.transpose(-1, -2)
        ) / 2

        # Test compatibility validation
        result = validator.validate_compatibility(base_metric, compatible_metric)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "compatibility_score" in result.metrics

        # Test incompatible metrics
        incompatible_metric = torch.randn(batch_size, dim, dim)
        incompatible_metric = incompatible_metric @ incompatible_metric.transpose(
            -1, -2
        )
        result = validator.validate_compatibility(base_metric, incompatible_metric)
        assert not result.is_valid

        # Test transition functions
        def transition_fn(x: torch.Tensor) -> torch.Tensor:
            return x + 0.1 * torch.sin(x)

        result = validator.validate_transition(
            base_metric, compatible_metric, transition_fn
        )
        assert isinstance(result, ValidationResult)
        assert "transition_smoothness" in result.metrics

    def test_smoothness(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test metric smoothness properties."""
        # Generate smooth metric field
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        def smooth_metric_field(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Generate smooth metric field."""
            metric = torch.eye(dim).unsqueeze(0).repeat(x.shape[0], 1, 1)
            metric = metric + 0.1 * torch.sin(2 * np.pi * x).unsqueeze(-1).unsqueeze(-1)
            return metric

        metrics = smooth_metric_field(X.flatten(), Y.flatten())

        # Test smoothness validation
        result = validator.validate_smoothness(metrics, (X.flatten(), Y.flatten()))
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "smoothness_measure" in result.metrics
        assert "derivative_bounds" in result.metrics

        # Test non-smooth metric field
        def non_smooth_field(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Generate non-smooth metric field."""
            metric = torch.eye(dim).unsqueeze(0).repeat(x.shape[0], 1, 1)
            metric = metric + torch.sign(x).unsqueeze(-1).unsqueeze(-1)
            return metric

        non_smooth_metrics = non_smooth_field(X.flatten(), Y.flatten())
        result = validator.validate_smoothness(
            non_smooth_metrics, (X.flatten(), Y.flatten())
        )
        assert not result.is_valid

    def test_curvature_bounds(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test metric curvature bounds."""
        # Generate metric with bounded curvature
        matrix = torch.randn(batch_size, dim, dim)
        metric = matrix @ matrix.transpose(-1, -2)
        metric = metric / torch.norm(metric, dim=(-2, -1), keepdim=True)

        # Test curvature validation
        result = validator.validate_curvature(metric)
        assert isinstance(result, ValidationResult)
        assert "sectional_curvature" in result.metrics
        assert "ricci_curvature" in result.metrics
        assert "scalar_curvature" in result.metrics

        # Test curvature bounds
        bounds = validator.compute_curvature_bounds(metric)
        assert isinstance(bounds, CurvatureBounds)
        assert hasattr(bounds, "sectional_bounds")
        assert hasattr(bounds, "ricci_bounds")
        assert hasattr(bounds, "scalar_bounds")

        # Test specific curvature types
        sectional = validator.compute_sectional_curvature(metric)
        assert sectional.shape[:-2] == metric.shape[:-2]

        ricci = validator.compute_ricci_curvature(metric)
        assert ricci.shape[:-1] == metric.shape[:-1]

        scalar = validator.compute_scalar_curvature(metric)
        assert scalar.shape == metric.shape[:-2]

    def test_metric_properties(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test metric property computations."""
        # Generate test metric
        matrix = torch.randn(batch_size, dim, dim)
        metric = matrix @ matrix.transpose(-1, -2)

        # Compute metric properties
        properties = validator.compute_metric_properties(metric)
        assert isinstance(properties, MetricProperties)

        # Test property computations
        assert hasattr(properties, "determinant")
        assert hasattr(properties, "trace")
        assert hasattr(properties, "eigenvalues")
        assert hasattr(properties, "condition_number")

        # Test property bounds
        assert torch.all(properties.determinant > 0)
        assert torch.all(properties.trace > 0)
        assert torch.all(properties.condition_number >= 1)

        # Test derived properties
        volume_form = validator.compute_volume_form(metric)
        assert volume_form.shape == metric.shape[:-2]

        christoffel = validator.compute_christoffel_symbols(metric)
        assert christoffel.shape[-3:] == (dim, dim, dim)

    def test_smoothness_metrics(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test smoothness metrics computation."""
        # Generate smooth metric field
        t = torch.linspace(0, 1, 10)
        metrics = []
        for ti in t:
            matrix = torch.randn(batch_size, dim, dim)
            metric = matrix @ matrix.transpose(-1, -2)
            metric = metric + ti.item() * torch.eye(dim)
            metrics.append(metric)
        metrics = torch.stack(metrics)

        # Compute smoothness metrics
        smoothness = validator.compute_smoothness_metrics(metrics, t)
        assert isinstance(smoothness, SmoothnessMetrics)

        # Test metric properties
        assert hasattr(smoothness, "variation")
        assert hasattr(smoothness, "derivative_norm")
        assert hasattr(smoothness, "continuity")

        # Test metric bounds
        assert torch.all(smoothness.variation >= 0)
        assert torch.all(smoothness.derivative_norm >= 0)
        assert torch.all(smoothness.continuity >= 0)

        # Test smoothness measures
        lipschitz = validator.compute_lipschitz_constant(metrics, t)
        assert lipschitz >= 0

        holder = validator.compute_holder_constant(metrics, t)
        assert holder >= 0

    def test_validation_integration(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test integrated metric validation."""
        # Generate test metric
        matrix = torch.randn(batch_size, dim, dim)
        metric = matrix @ matrix.transpose(-1, -2)
        metric = metric + torch.eye(dim)  # Ensure positive definiteness

        # Run full validation
        result = validator.validate_metric(metric)
        assert isinstance(result, Dict)
        assert "positive_definite" in result
        assert "smoothness" in result
        assert "curvature" in result
        assert "properties" in result

        # Check validation scores
        assert all(0 <= score <= 1 for score in result.values())
        assert "overall_score" in result

        # Test validation with parameters
        params = torch.linspace(0, 1, 10)
        param_result = validator.validate_metric_family(metric, params)
        assert "parameter_dependence" in param_result

        # Test validation summary
        summary = validator.get_validation_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_riemannian_metric(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test Riemannian metric validation."""

        # Test positive definiteness
        def test_positive_definite():
            """Test positive definiteness of metric."""
            metric = validator.get_test_metric()
            assert validator.validate_positive_definite(metric)

            # Test signature
            signature = validator.get_metric_signature(metric)
            assert validator.validate_signature(metric, signature)

            return metric

        metric = test_positive_definite()

        # Test compatibility
        def test_compatibility():
            """Test metric compatibility with connection."""
            connection = validator.get_test_connection()
            assert validator.validate_metric_compatibility(metric, connection)

            # Test parallel transport
            transport = validator.get_parallel_transport(metric, connection)
            assert validator.validate_parallel_transport(transport)

            return connection, transport

        connection, transport = test_compatibility()

        # Test curvature properties
        def test_curvature():
            """Test curvature properties of metric."""
            curvature = validator.compute_curvature(metric, connection)

            # Test symmetries
            assert validator.validate_curvature_symmetries(curvature)

            # Test sectional curvature
            K = validator.compute_sectional_curvature(metric, curvature)
            assert validator.validate_sectional_bounds(K)

            # Test Ricci curvature
            Ric = validator.compute_ricci_curvature(metric, curvature)
            assert validator.validate_ricci_bounds(Ric)

            return curvature, K, Ric

        curvature, K, Ric = test_curvature()

    def test_kahler_metric(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test Kähler metric validation."""

        # Test complex structure
        def test_complex_structure():
            """Test complex structure compatibility."""
            J = validator.get_complex_structure()
            assert validator.validate_complex_structure(J)

            # Test integrability
            assert validator.validate_integrability(J)

            return J

        J = test_complex_structure()

        # Test Kähler form
        def test_kahler_form():
            """Test Kähler form properties."""
            omega = validator.get_kahler_form()

            # Test closure
            assert validator.validate_closure(omega)

            # Test non-degeneracy
            assert validator.validate_non_degeneracy(omega)

            return omega

        omega = test_kahler_form()

        # Test Kähler identities
        def test_kahler_identities():
            """Test Kähler identities."""
            # Get operators
            d = validator.get_exterior_derivative()
            dc = validator.get_conjugate_derivative()

            # Test commutation relations
            assert validator.validate_kahler_identities(d, dc)

            # Test Lefschetz decomposition
            L = validator.get_lefschetz_operator(omega)
            assert validator.validate_lefschetz_decomposition(L)

            return d, dc, L

        d, dc, L = test_kahler_identities()

    def test_metric_compatibility(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test metric compatibility conditions."""

        # Test connection compatibility
        def test_connection_compatibility():
            """Test connection compatibility with metric."""
            # Get connection
            connection = validator.get_test_connection()
            assert validator.validate_connection_compatibility(connection)

            # Test torsion
            torsion = validator.compute_torsion(connection)
            assert validator.validate_torsion_free(torsion)

            return connection, torsion

        connection, torsion = test_connection_compatibility()

        # Test holonomy
        def test_holonomy():
            """Test holonomy properties."""
            # Get holonomy group
            hol = validator.get_holonomy_group()
            assert validator.validate_holonomy_reduction(hol)

            # Test holonomy algebra
            hol_alg = validator.get_holonomy_algebra(hol)
            assert validator.validate_holonomy_algebra(hol_alg)

            return hol, hol_alg

        hol, hol_alg = test_holonomy()

        # Test characteristic classes
        def test_characteristic_classes():
            """Test characteristic classes."""
            # Get Chern classes
            chern = validator.compute_chern_classes()
            assert validator.validate_chern_classes(chern)

            # Get Pontryagin classes
            pont = validator.compute_pontryagin_classes()
            assert validator.validate_pontryagin_classes(pont)

            return chern, pont

        chern, pont = test_characteristic_classes()

    def test_geodesic_completeness(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test geodesic completeness of the metric."""
        
        # Test local completeness
        def test_local_completeness():
            """Test local geodesic completeness."""
            # Get random point and direction
            p = torch.randn(batch_size, dim)
            v = torch.randn(batch_size, dim)
            
            # Verify that geodesics can be extended locally
            assert validator.check_local_completeness(p, v), "Metric should be locally complete"
            
            # Check existence of normal neighborhood
            assert validator.check_normal_neighborhood(p), "Should have normal neighborhood"
            
            return p, v
            
        p, v = test_local_completeness()
        
        # Test global completeness
        def test_global_completeness():
            """Test global geodesic completeness."""
            # Verify Hopf-Rinow conditions
            assert validator.check_hopf_rinow_conditions(), "Should satisfy Hopf-Rinow conditions"
            
            # Check metric bounds
            assert validator.check_metric_bounds(), "Metric should be bounded"
            
            # Verify completeness of geodesic flow
            assert validator.check_geodesic_completeness(), "Metric should be geodesically complete"
            
        test_global_completeness()

    def test_score_function(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test score function computation."""
        points = torch.randn(batch_size, dim)
        score = validator.compute_score_function(points)
        
        # Check shape
        assert score.shape == (batch_size, dim)
        
        # Check gradient relationship
        expected_score = -points  # For Gaussian distribution
        assert torch.allclose(score, expected_score, atol=1e-5)

    def test_metric_gradient(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test metric gradient computation."""
        points = torch.randn(batch_size, dim, requires_grad=True)
        grad = validator.compute_metric_gradient(points)
        
        # Check shape
        assert grad.shape == (batch_size, dim, dim, dim)
        
        # Check symmetry in i,j indices
        assert torch.allclose(
            grad.transpose(1, 2), grad, atol=1e-5
        )

    def test_pattern_energy(self, validator: MetricValidator, dim: int):
        """Test pattern energy computation."""
        energy = validator.compute_pattern_energy()
        
        # Check shape
        assert energy.shape == (dim,)
        
        # Check non-negativity
        assert torch.all(energy >= 0)

    def test_height_function(self, validator: MetricValidator):
        """Test height function computation."""
        height = validator.compute_height_function()
        
        # Check shape
        assert height.shape == torch.Size([])
        
        # Validate bounds
        assert validator.validate_global_bounds(height.item())

    def test_global_bounds(self, validator: MetricValidator):
        """Test global bound validation."""
        # Test valid height
        valid_height = 0.5 * validator.energy_threshold
        assert validator.validate_global_bounds(valid_height)
        
        # Test invalid upper bound
        invalid_upper = 2.0 * validator.energy_threshold
        assert not validator.validate_global_bounds(invalid_upper)
        
        # Test invalid lower bound
        invalid_lower = -100.0 * validator.manifold_dim
        assert not validator.validate_global_bounds(invalid_lower)
