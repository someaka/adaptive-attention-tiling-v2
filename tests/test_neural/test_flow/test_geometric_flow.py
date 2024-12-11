"""
Unit tests for the geometric flow system.

Tests cover:
1. Ricci tensor computation
2. Flow evolution steps
3. Singularity detection
4. Flow normalization
5. Geometric invariants
"""

import numpy as np
import pytest
import torch

from src.neural.flow.geometric_flow import (
    FlowMetrics,
    GeometricFlow,
    RicciTensorNetwork as RicciTensor,
    SingularityInfo as Singularity,
)
from src.validation.geometric.flow import (
    FlowStabilityValidation,
    EnergyValidation,
    ConvergenceValidation,
)


class TestGeometricFlow:
    """Test geometric flow implementation."""
    
    @pytest.fixture
    def manifold_dim(self):
        """Return manifold dimension for tests."""
        return 4

    @pytest.fixture
    def batch_size(self):
        """Batch size for testing."""
        return 8

    @pytest.fixture
    def flow_system(self, manifold_dim):
        """Create flow system fixture."""
        return GeometricFlow(manifold_dim=manifold_dim)
        
    @pytest.fixture
    def points(self):
        """Create test points."""
        return torch.randn(4, 4, requires_grad=True)
        
    @pytest.fixture
    def metric(self, points, flow_system):
        """Create test metric."""
        return flow_system.compute_metric(points)
        
    def test_ricci_tensor(self, points, metric, flow_system):
        """Test Ricci tensor computation."""
        ricci = flow_system.compute_ricci_tensor(metric)
        assert isinstance(ricci, RicciTensor)
        assert ricci.tensor.shape == metric.shape
        
    def test_flow_step(self, points, metric, flow_system):
        """Test flow step computation."""
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        assert flow.shape == points.shape
        
    def test_singularity_detection(self, points, metric, flow_system):
        """Test singularity detection."""
        singularities = flow_system.detect_singularities(metric)
        assert isinstance(singularities, list)
        if len(singularities) > 0:
            assert isinstance(singularities[0], Singularity)
            
    def test_flow_normalization(self, points, metric, flow_system):
        """Test flow normalization."""
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Check volume preservation
        vol1 = torch.sqrt(torch.det(metric))
        new_points = points + flow_system.dt * flow
        new_metric = flow_system.compute_metric(new_points)
        vol2 = torch.sqrt(torch.det(new_metric))
        
        assert torch.allclose(vol1, vol2, rtol=1e-4)
        
    def test_geometric_invariants(self, points, metric, flow_system):
        """Test geometric invariants."""
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Evolve metric
        new_points = points + flow_system.dt * flow
        new_metric = flow_system.compute_metric(new_points)
        
        # Check invariants
        validator = FlowValidator()
        assert validator.validate_invariants(new_metric, flow)
        
    def test_ricci_flow(self, points, metric, flow_system):
        """Test Ricci flow evolution."""
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Check Ricci flow equation
        new_points = points + flow_system.dt * flow
        new_metric = flow_system.compute_metric(new_points)
        
        flow_deriv = (new_metric - metric) / flow_system.dt
        assert torch.allclose(flow_deriv, -2 * ricci.tensor, rtol=1e-2)
        
    def test_mean_curvature_flow(self, points, metric, flow_system):
        """Test mean curvature flow."""
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Check mean curvature
        mean_curv = flow_system.compute_mean_curvature(metric)
        assert mean_curv.shape == (metric.shape[0],)
        
    def test_singularity_analysis(self, points, metric, flow_system):
        """Test singularity analysis."""
        # Add artificial singularity
        singular_metric = metric.clone()
        singular_metric[:,0,0] = 0.1  # Near-singular
        
        # Detect singularities
        singularities = flow_system.detect_singularities(singular_metric)
        assert len(singularities) > 0
        
        # Check resolution
        for sing in singularities:
            assert sing.is_removable()


class FlowValidator:
    """Validator for geometric flow properties."""
    
    def validate_invariants(self, metric: torch.Tensor, flow: torch.Tensor) -> bool:
        """Validate geometric invariants of the flow.
        
        Args:
            metric: Metric tensor
            flow: Flow vector field
            
        Returns:
            True if invariants are preserved
        """
        batch_size = metric.shape[0]
        n = metric.shape[1]
        
        # Check volume preservation
        vol_form = torch.sqrt(torch.abs(torch.det(metric)))
        flow_div = torch.zeros(batch_size, device=metric.device)
        for i in range(n):
            flow_div += torch.diagonal(metric, dim1=1, dim2=2)[:, i] * flow[:, i]
            
        return torch.allclose(flow_div, torch.zeros_like(flow_div), rtol=1e-3)
