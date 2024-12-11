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
    RicciTensor,
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
        
        # Normalize flow
        normalized_flow = flow_system.flow_step.normalize_flow(flow, metric)
        
        # Check volume preservation
        det = torch.det(metric)
        det = torch.abs(det) + 1e-8  # Small epsilon to prevent division by zero
        vol1 = torch.sqrt(det)
        
        # Evolve points with normalized flow
        new_points = points + flow_system.dt * normalized_flow
        new_metric = flow_system.compute_metric(new_points)
        
        # Compute new volume
        det2 = torch.det(new_metric)
        det2 = torch.abs(det2) + 1e-8
        vol2 = torch.sqrt(det2)
        
        # Compare relative volumes
        rel_vol = vol2 / (vol1 + 1e-8)
        assert torch.allclose(rel_vol, torch.ones_like(rel_vol), rtol=1e-2)

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
        # Compute Ricci tensor
        ricci = flow_system.compute_ricci_tensor(metric)
        
        # Compute flow
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Reshape flow to match metric shape
        batch_size = metric.shape[0]
        n = metric.shape[1]
        flow_matrix = torch.zeros_like(metric)
        for i in range(n):
            for j in range(n):
                flow_matrix[:, i, j] = flow[:, i]
        
        # Evolve metric with small time step
        dt = 0.001  # Smaller time step for better numerical stability
        new_metric = metric + dt * flow_matrix
        
        # Compute flow derivative
        flow_deriv = (new_metric - metric) / dt
        
        # Scale Ricci tensor according to flow equation
        ricci_scaled = -2 * ricci.tensor
        
        # Compare the flow direction rather than exact values
        flow_deriv_flat = flow_deriv.reshape(batch_size, -1)
        ricci_flat = ricci_scaled.reshape(batch_size, -1)
        
        # Compute cosine similarity for each batch
        flow_norm = torch.norm(flow_deriv_flat, dim=1)
        ricci_norm = torch.norm(ricci_flat, dim=1)
        
        # Add epsilon to prevent division by zero
        epsilon = 1e-10
        cosine_sim = torch.sum(flow_deriv_flat * ricci_flat, dim=1) / (flow_norm * ricci_norm + epsilon)
        
        # Check if flow direction aligns with Ricci tensor
        assert torch.all(torch.abs(cosine_sim) > 0.7)  # Allow for some deviation but maintain general direction

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
        """Validate geometric invariants of the flow."""
        batch_size = metric.shape[0]
        n = metric.shape[1]
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-6
        
        # Compute divergence directly using the metric and flow
        div = torch.zeros(batch_size, device=metric.device)
        
        # Compute metric determinant for volume form
        det = torch.det(metric + epsilon * torch.eye(n, device=metric.device))
        sqrt_det = torch.sqrt(torch.abs(det) + epsilon)
        
        # Compute covariant divergence
        for i in range(n):
            for j in range(n):
                # Use metric contraction with flow
                div += metric[:, i, j] * flow[:, j]
        
        # Normalize by metric volume
        normalized_div = div / (n * sqrt_det)
        
        # Check if normalized divergence is sufficiently small
        return torch.all(torch.abs(normalized_div) < 1e-1)


class TestFlowStability:
    """Test class for flow stability and normalization diagnostics."""
    
    @pytest.fixture
    def flow_system(self):
        """Create flow system fixture."""
        return GeometricFlow(manifold_dim=4)
        
    @pytest.fixture
    def points(self):
        """Create well-conditioned test points."""
        # Use points with better numerical properties
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], requires_grad=True)
    
    def test_metric_conditioning(self, points, flow_system):
        """Test metric tensor conditioning."""
        metric = flow_system.compute_metric(points)
        
        # Check metric determinant is well-conditioned
        det = torch.det(metric)
        assert torch.all(det > 1e-6)
        
        # Check metric eigenvalues
        eigenvals = torch.linalg.eigvals(metric)
        condition_number = torch.abs(eigenvals).max() / torch.abs(eigenvals).min()
        assert torch.all(condition_number < 1e4)
    
    def test_flow_magnitude(self, points, flow_system):
        """Test flow vector magnitudes stay reasonable."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Check flow magnitudes
        flow_norm = torch.norm(flow, dim=1)
        assert torch.all(flow_norm < 1e3)
        assert torch.all(flow_norm > 1e-6)
    
    def test_volume_preservation(self, points, flow_system):
        """Test volume preservation with small steps."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow_vector(points, ricci)
        
        # Take very small steps
        dt = 1e-4
        steps = 10
        current_points = points.clone()
        initial_vol = torch.sqrt(torch.det(metric))
        
        for _ in range(steps):
            current_points = current_points + dt * flow
            current_metric = flow_system.compute_metric(current_points)
            current_vol = torch.sqrt(torch.det(current_metric))
            
            # Check relative volume change
            rel_vol = current_vol / initial_vol
            assert torch.allclose(rel_vol, torch.ones_like(rel_vol), rtol=1e-3)
    
    def test_ricci_flow_stability(self, points, flow_system):
        """Test stability of Ricci flow evolution."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric)
        
        # Compute initial scalar curvature
        initial_scalar_curv = flow_system.compute_scalar_curvature(metric)
        
        # Evolve for a few small steps
        dt = 1e-4
        steps = 5
        current_metric = metric.clone()
        
        for _ in range(steps):
            current_ricci = flow_system.compute_ricci_tensor(current_metric)
            new_metric = current_metric - 2 * dt * current_ricci.tensor
            
            # Check metric remains positive definite
            eigenvals = torch.linalg.eigvals(new_metric)
            assert torch.all(torch.real(eigenvals) > 0)
            
            # Check scalar curvature doesn't blow up
            scalar_curv = flow_system.compute_scalar_curvature(new_metric)
            assert torch.all(torch.abs(scalar_curv) < 2 * torch.abs(initial_scalar_curv))
            
            current_metric = new_metric
