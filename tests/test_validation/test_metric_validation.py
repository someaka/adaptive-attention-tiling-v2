"""
Unit tests for metric validation.

Tests cover:
1. Positive definite properties
2. Compatibility conditions
3. Smoothness properties
4. Curvature bounds
5. Fisher-Rao metric properties
6. Metric family validation
7. Error handling
"""

from typing import Dict, Optional

import numpy as np
import pytest
import torch

from src.validation.geometric.metric import (
    MetricValidator,
    MetricValidation,
    CurvatureBounds,
    MetricProperties
)

class TestMetricValidation:
    """Test metric validation functionality."""

    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def dim(self) -> int:
        return 8

    @pytest.fixture
    def validator(self, dim: int) -> MetricValidator:
        return MetricValidator(
            manifold_dim=dim,
            tolerance=1e-6
        )

    def test_positive_definite(
        self, validator: MetricValidator, batch_size: int, dim: int
    ):
        """Test positive definite validation."""
        # Generate positive definite metric
        matrix = torch.randn(batch_size, dim, dim)
        pd_metric = matrix @ matrix.transpose(-1, -2)
        pd_metric = pd_metric + torch.eye(dim) * 1e-3  # Ensure strict positive definiteness

        # Test positive definite validation
        result = validator.validate_metric(pd_metric)
        assert isinstance(result, MetricValidation)
        assert result.positive_definite
        assert result.eigenvalues is not None and torch.all(result.eigenvalues > 0)

        # Test non-positive definite metric
        non_pd_metric = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        non_pd_metric[:, 0, 0] = -1.0
        result = validator.validate_metric(non_pd_metric)
        assert not result.positive_definite
        assert result.eigenvalues is not None and torch.any(result.eigenvalues < 0)

        # Test borderline case
        borderline_metric = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        borderline_metric[:, 0, 0] = validator.eigenvalue_threshold / 2
        result = validator.validate_metric(borderline_metric)
        assert not result.positive_definite

    def test_compatibility(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test metric compatibility conditions."""
        # Generate compatible metric
        base_metric = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        compatible_metric = base_metric + 0.1 * torch.randn(batch_size, dim, dim)
        compatible_metric = (compatible_metric + compatible_metric.transpose(-1, -2)) / 2

        # Test compatibility validation
        result = validator.validate_metric(compatible_metric)
        assert isinstance(result, MetricValidation)
        assert result.positive_definite

        # Test connection compatibility
        connection = validator.get_test_connection()
        assert validator.validate_connection_compatibility(connection)

        # Test torsion
        torsion = validator.compute_torsion(connection)
        assert validator.validate_torsion_free(torsion)

    def test_fisher_rao_metric(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test Fisher-Rao metric validation."""
        # Generate test points
        points = torch.randn(batch_size, dim)
        
        # Compute Fisher-Rao metric
        score = validator.compute_score_function(points)
        fisher_metric = torch.einsum('bi,bj->bij', score, score)
        
        # Test validation
        assert validator.validate_fisher_rao(fisher_metric)
        
        # Test non-Fisher-Rao metric
        non_fisher = torch.randn(batch_size, dim, dim)
        non_fisher = non_fisher @ non_fisher.transpose(-1, -2)
        assert not validator.validate_fisher_rao(non_fisher)

    def test_curvature_validation(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test curvature validation."""
        # Generate test metric
        matrix = torch.randn(batch_size, dim, dim)
        metric = matrix @ matrix.transpose(-1, -2)
        metric = metric / torch.norm(metric, dim=(-2, -1), keepdim=True)
        
        # Test curvature bounds
        bounds = validator.validate_curvature_bounds(metric)
        assert isinstance(bounds, CurvatureBounds)
        assert bounds.sectional_bounds is not None
        assert bounds.ricci_bounds is not None
        assert bounds.scalar_bounds is not None
        
        # Test curvature symmetries
        sectional = validator.compute_sectional_curvature(metric)
        assert validator.validate_sectional_bounds(sectional)
        
        ricci = validator.compute_ricci_curvature(metric)
        assert validator.validate_ricci_bounds(ricci)
        
        # Test curvature tensor symmetries
        riemann = torch.zeros(batch_size, dim, dim, dim, dim)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        riemann[:,i,j,k,l] = sectional[:,i,j] * (
                            metric[:,i,k] * metric[:,j,l] -
                            metric[:,i,l] * metric[:,j,k]
                        )
        assert validator.validate_curvature_symmetries(riemann)

    def test_metric_family_validation(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test metric family validation."""
        # Generate metric family
        parameters = torch.linspace(0, 1, 10)
        metrics = []
        for t in parameters:
            matrix = torch.randn(batch_size, dim, dim)
            metric = matrix @ matrix.transpose(-1, -2)
            metric = metric + t.item() * torch.eye(dim)
            metrics.append(metric)
        metrics = torch.stack(metrics)
        
        # Test validation
        result = validator.validate_metric_family(metrics, parameters)
        assert isinstance(result, dict)
        assert "positive_definite" in result
        assert "smooth_variation" in result
        assert "parameter_dependence" in result
        
        # Test validation summary
        summary = validator.get_validation_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_metric_properties(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test metric property computations."""
        # Generate test metric
        matrix = torch.randn(batch_size, dim, dim)
        metric = matrix @ matrix.transpose(-1, -2)
        metric = metric + torch.eye(dim)  # Ensure positive definiteness
        
        # Test property validation
        properties = validator.validate_metric_properties(metric)
        assert isinstance(properties, MetricProperties)
        
        # Test basic properties
        assert properties.determinant is not None
        assert properties.trace is not None
        assert properties.eigenvalues is not None
        assert properties.condition_number is not None
        
        # Test derived properties
        assert properties.volume_form is not None
        assert properties.christoffel_symbols is not None
        assert properties.sectional_curvature is not None
        assert properties.ricci_curvature is not None
        assert properties.scalar_curvature is not None
        
        # Test property bounds
        assert torch.all(properties.determinant > 0)
        assert torch.all(properties.trace > 0)
        assert properties.condition_number >= 1
        
        # Test derived property shapes
        assert properties.volume_form is not None and properties.volume_form.shape == (batch_size,)
        assert properties.christoffel_symbols is not None and properties.christoffel_symbols.shape == (batch_size, dim, dim, dim)
        assert properties.sectional_curvature is not None and properties.sectional_curvature.shape == (batch_size, dim, dim)
        assert properties.ricci_curvature is not None and properties.ricci_curvature.shape == (batch_size, dim, dim)
        assert properties.scalar_curvature is not None and properties.scalar_curvature.shape == (batch_size,)

    def test_error_handling(self, validator: MetricValidator, batch_size: int, dim: int):
        """Test error handling in validation."""
        # Test invalid metric shape
        invalid_shape = torch.randn(batch_size, dim)
        with pytest.raises(ValueError, match="Invalid metric shape"):
            validator.validate_metric(invalid_shape)
            
        # Test non-symmetric metric
        non_symmetric = torch.randn(batch_size, dim, dim)
        with pytest.raises(ValueError, match="Non-symmetric metric"):
            validator.validate_metric(non_symmetric)
            
        # Test NaN/Inf values
        invalid_values = torch.full((batch_size, dim, dim), float('nan'))
        with pytest.raises(ValueError, match="Contains NaN or Inf values"):
            validator.validate_metric(invalid_values)
            
        # Test incompatible dimensions
        incompatible = torch.randn(batch_size, dim+1, dim)
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            validator.validate_metric(incompatible)
