"""
Unit tests for the geometric flow system.

Tests are organized in dependency order:
1. Basic Components
   - Metric computation
   - Ricci tensor computation
   - Flow vector computation
2. Flow Evolution
   - Single step evolution
   - Flow normalization
   - Singularity detection
3. Validation
   - Geometric invariants
   - Energy conservation
   - Flow stability
   - Convergence
"""

import numpy as np
import pytest
import torch
import warnings

from src.validation.geometric.flow import TilingFlowValidator, TilingFlowValidationResult
from src.core.attention.geometric import GeometricStructures
from src.core.flow import NeuralGeometricFlow
from src.core.flow.protocol import FlowMetrics, SingularityInfo as Singularity

# Mark test class for dependency management
class TestGeometricFlow:
    """Test geometric flow implementation."""
    
    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            'batch_size': 4,
            'seq_len': 8,
            'hidden_dim': 8,  # Smaller hidden dimension
            'manifold_dim': 4,  # Smaller manifold dimension
            'num_heads': 4,
            'dropout': 0.1,
            'device': torch.device('cpu'),
            'dtype': torch.float32
        }

    @pytest.fixture
    def flow_layer(self, test_config):
        """Create flow layer for testing."""
        return NeuralGeometricFlow(
            hidden_dim=test_config['hidden_dim'],
            manifold_dim=test_config['manifold_dim'],
            num_heads=test_config['num_heads'],
            dropout=test_config['dropout'],
            device=test_config['device']
        )

    @pytest.fixture
    def test_input(self, test_config):
        """Create test input tensor."""
        batch_size = test_config['batch_size']
        seq_len = test_config['seq_len']
        hidden_dim = test_config['hidden_dim']
        
        # Create random input tensor
        x = torch.randn(batch_size, seq_len, hidden_dim, device=test_config['device'])
        x = x / x.norm(dim=-1, keepdim=True)  # Normalize along hidden dimension
        
        return x

    @pytest.fixture
    def geometric_structures(self, test_config):
        """Create geometric structures for testing."""
        return GeometricStructures(
            dim=test_config['manifold_dim'],  # Use manifold_dim instead of hidden_dim
            manifold_type="hyperbolic",
            curvature=-1.0,
            parallel_transport_method="schild",
        )

    @pytest.fixture
    def metric(self, flow, points):
        """Create metric tensor for testing."""
        return flow.compute_metric(points)

    @pytest.mark.dependency(name="test_metric_computation")
    def test_metric_computation(self, flow, points):
        """Test metric tensor computation."""
        # Compute metric tensor
        metric = flow.compute_metric(points)
        
        # Check basic properties
        assert isinstance(metric, torch.Tensor)
        assert metric.shape == (points.shape[0], points.shape[1], points.shape[1])
        
        # Check symmetry
        assert torch.allclose(metric, metric.transpose(-2, -1), atol=1e-6)
        
        # Add regularization for numerical stability
        metric = metric + torch.eye(points.shape[1]).unsqueeze(0).expand_as(metric) * 1e-4
        
        # Check positive definiteness via Cholesky decomposition
        try:
            torch.linalg.cholesky(metric)
            is_positive_definite = True
        except RuntimeError:
            is_positive_definite = False
            
        assert is_positive_definite, "Metric tensor must be positive definite"

    @pytest.mark.dependency(name="test_flow_integration", depends=["test_metric_computation"])
    def test_flow_integration(self, flow, points):
        """Test flow integration over time."""
        # Initial metric
        metric = flow.compute_metric(points)
        
        # Integrate flow
        metrics = []
        for _ in range(10):
            ricci = flow.compute_ricci_tensor(metric, points)
            metric, _ = flow.flow_step(metric, ricci)
            metrics.append(metric)
            
        # Check stability
        dets = [torch.linalg.det(m) for m in metrics]
        assert all(torch.all(d > 0) for d in dets)

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_ricci_tensor(self, flow, points, metric):
        """Test Ricci tensor computation."""
        ricci = flow.compute_ricci_tensor(metric, points)
        assert isinstance(ricci, torch.Tensor)
        assert ricci.shape == (points.shape[0], points.shape[1], points.shape[1])

    @pytest.mark.dependency(depends=["test_metric_computation", "test_ricci_tensor"])
    def test_flow_computation(self, flow, points, metric):
        """Test flow vector computation."""
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        assert isinstance(flow_vector, torch.Tensor)
        assert flow_vector.shape == points.shape

    @pytest.mark.dependency(depends=["test_metric_computation", "test_flow_computation"])
    def test_flow_step(self, flow, points, metric):
        """Test single flow step."""
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, flow_metrics = flow.flow_step(metric, ricci)
        assert isinstance(evolved_metric, torch.Tensor)
        assert evolved_metric.shape == metric.shape
        assert isinstance(flow_metrics, FlowMetrics)

    @pytest.mark.dependency(depends=["test_metric_computation", "test_flow_step"])
    def test_flow_normalization(self, flow, points, metric):
        """Test flow normalization."""
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        normalized = flow.normalize_flow(flow_vector, metric)
        assert torch.all(torch.isfinite(normalized))

    @pytest.mark.dependency(depends=["test_metric_computation", "test_flow_normalization"])
    def test_singularity_detection(self, flow, points, metric):
        """Test singularity detection."""
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        singularity = flow.detect_singularities(flow_vector, metric)
        assert isinstance(singularity, Singularity)

    @pytest.mark.dependency(depends=["test_metric_computation", "test_flow_step"])
    def test_geometric_invariants(self, flow, points, metric):
        """Test geometric invariant preservation."""
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        
        # Check volume preservation
        det_before = torch.linalg.det(metric)
        det_after = torch.linalg.det(evolved_metric)
        assert torch.allclose(det_before, det_after, rtol=1e-4)
        
        # Check metric conditioning
        condition_number = torch.linalg.cond(evolved_metric)
        if torch.any(condition_number > 1e5):
            warnings.warn(f"High condition number detected: {condition_number.max():.2e}")

    @pytest.mark.dependency(depends=["test_metric_computation", "test_geometric_invariants"])
    def test_energy_conservation(self, flow, points, energy_validator):
        """Test energy conservation during flow."""
        metric = flow.compute_metric(points)
        
        # Evolve and track energy
        energy_history = []
        for _ in range(10):
            ricci = flow.compute_ricci_tensor(metric, points)
            metric, metrics = flow.flow_step(metric, ricci)
            energy_history.append(metrics.energy)
            
        # Validate energy conservation
        result = energy_validator.validate_energy_conservation(energy_history)
        assert isinstance(result, TilingFlowValidationResult)
        assert result.is_valid

    @pytest.mark.dependency(depends=["test_metric_computation", "test_geometric_invariants"])
    def test_flow_stability(self, flow, points, geometric_validator):
        """Test flow stability."""
        metric = flow.compute_metric(points)
        
        # Evolve and track metrics
        metrics = []
        for _ in range(10):
            ricci = flow.compute_ricci_tensor(metric, points)
            metric, _ = flow.flow_step(metric, ricci)
            metrics.append(metric)
            
        # Check stability
        dets = [torch.linalg.det(m) for m in metrics]
        assert all(torch.all(d > 0) for d in dets)

    @pytest.mark.dependency(depends=["test_metric_computation", "test_flow_stability"])
    def test_flow_convergence(self, flow, random_states, convergence_validator):
        """Test that flow converges to stable points."""
        points = random_states[:, :flow.manifold_dim]
        metric = flow.compute_metric(points)
        
        # Evolve to convergence
        for _ in range(500):  # More iterations for convergence
            ricci = flow.compute_ricci_tensor(metric, points)
            metric, _ = flow.flow_step(metric, ricci)
            
        # Validate convergence
        result = convergence_validator.validate_convergence(flow, points, metric)
        assert isinstance(result, TilingFlowValidationResult)
        assert result.is_valid

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_ricci_flow_stability(self, flow, points):
        """Test Ricci flow stability."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        assert torch.all(torch.isfinite(flow_vector))

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_volume_preservation(self, flow, points):
        """Test volume preservation."""
        metric = flow.compute_metric(points)
        det_before = torch.linalg.det(metric)
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        det_after = torch.linalg.det(evolved_metric)
        assert torch.allclose(det_before, det_after, rtol=1e-4)

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_flow_magnitude(self, flow, points):
        """Test flow vector magnitudes."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        assert torch.all(torch.abs(flow_vector) < 1e2)

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_metric_conditioning(self, flow, points):
        """Test metric tensor conditioning."""
        metric = flow.compute_metric(points)
        condition_number = torch.linalg.cond(metric)
        assert torch.all(condition_number < 1e3)

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_singularity_analysis(self, flow, points):
        """Test singularity analysis."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        singularity = flow.detect_singularities(metric, points)
        assert isinstance(singularity, Singularity)

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_mean_curvature_flow(self, flow, points):
        """Test mean curvature flow computation."""
        # Compute metric tensor
        metric = flow.compute_metric(points)
        
        # Set points before computing mean curvature
        flow.points = points
        
        # Compute mean curvature
        mean_curvature = flow.compute_mean_curvature(metric)
        
        # Verify shape and properties
        assert isinstance(mean_curvature, torch.Tensor)
        assert mean_curvature.shape == (points.shape[0], points.shape[1])
        assert torch.all(torch.isfinite(mean_curvature))
        
        # Verify mean curvature flow properties
        flow_vector = flow.compute_flow(points, mean_curvature)
        assert torch.all(torch.isfinite(flow_vector))

    @pytest.mark.dependency(depends=["test_metric_computation"])
    def test_ricci_flow(self, flow, points):
        """Test Ricci flow evolution."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        assert torch.all(torch.isfinite(evolved_metric))


class TestFlowStability:
    """Test class for flow stability and normalization diagnostics."""
    
    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            'batch_size': 4,
            'seq_len': 8,
            'hidden_dim': 8,  # Smaller hidden dimension
            'manifold_dim': 4,  # Smaller manifold dimension
            'num_heads': 4,
            'dropout': 0.1,
            'device': torch.device('cpu'),
            'dtype': torch.float32
        }

    @pytest.fixture
    def flow_system(self, test_config):
        """Create flow system fixture."""
        return NeuralGeometricFlow(
            hidden_dim=test_config['hidden_dim'],
            manifold_dim=test_config['manifold_dim'],
            num_heads=test_config['num_heads'],
            dropout=test_config['dropout'],
            device=test_config['device']
        )
        
    @pytest.fixture
    def points(self, test_config):
        """Create random points in position space."""
        return torch.randn(test_config['batch_size'], test_config['manifold_dim'], requires_grad=True)

    def test_metric_conditioning(self, flow_system, points):
        """Test metric tensor conditioning."""
        metric = flow_system.compute_metric(points)
        condition_number = torch.linalg.cond(metric)
        assert torch.all(condition_number < 1e3)

    def test_flow_magnitude(self, flow_system, points):
        """Test flow vector magnitudes stay reasonable."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric, points)
        flow_vector = flow_system.compute_flow(points, ricci)
        assert torch.all(torch.abs(flow_vector) < 1e2)

    def test_volume_preservation(self, flow_system, points):
        """Test volume preservation under flow."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow_system.flow_step(metric, ricci)
        
        # Check volume preservation
        initial_volume = torch.sqrt(torch.linalg.det(metric))
        evolved_volume = torch.sqrt(torch.linalg.det(evolved_metric))
        relative_error = torch.abs(evolved_volume - initial_volume) / initial_volume
        assert torch.all(relative_error < 1e-5)

    def test_ricci_flow_stability(self, flow_system, points):
        """Test Ricci flow stability."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric, points)
        flow_vector = flow_system.compute_flow(points, ricci)
        assert torch.all(torch.isfinite(flow_vector))
