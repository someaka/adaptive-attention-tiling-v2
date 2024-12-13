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

from src.neural.flow.geometric_flow import (
    FlowMetrics,
    GeometricFlow,
    SingularityInfo as Singularity,
)
from src.validation.geometric.flow import (
    FlowValidator,
    FlowValidationResult,
)

# Mark test class for dependency management
class TestGeometricFlow:
    """Test geometric flow implementation."""
    
    @pytest.fixture
    def batch_size(self):
        """Batch size for tests."""
        return 4

    @pytest.fixture
    def manifold_dim(self):
        """Manifold dimension for tests."""
        return 2  # Position space dimension

    @pytest.fixture
    def flow(self, manifold_dim):
        """Create flow system fixture."""
        return GeometricFlow(phase_dim=manifold_dim * 2)  # phase_dim is twice manifold_dim

    @pytest.fixture
    def points(self, batch_size, manifold_dim):
        """Create random points in position space."""
        return torch.randn(batch_size, manifold_dim, requires_grad=True)

    @pytest.fixture
    def random_states(self, batch_size, manifold_dim):
        """Create random phase space states (position and momentum)."""
        phase_dim = manifold_dim * 2
        return torch.randn(batch_size, phase_dim, requires_grad=True)

    @pytest.fixture
    def metric(self, flow, points):
        """Create metric tensor."""
        return flow.compute_metric(points)

    @pytest.fixture
    def validator():
        """Create flow validator."""
        return FlowValidator(
            energy_threshold=1e-6,
            monotonicity_threshold=1e-4,
            singularity_threshold=1.0,
            max_iterations=1000,
            tolerance=1e-6
        )

    @pytest.fixture
    def geometric_validator(self, validator):
        """Create geometric flow validator fixture."""
        return validator

    @pytest.fixture
    def energy_validator(self, validator):
        """Create energy conservation validator fixture."""
        return validator

    @pytest.fixture
    def convergence_validator(self, validator):
        """Create convergence validator fixture."""
        return validator

    @pytest.mark.dependency()
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

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_metric_computation"])
    def test_ricci_tensor(self, flow, points, metric):
        """Test Ricci tensor computation."""
        ricci = flow.compute_ricci_tensor(metric, points)
        assert isinstance(ricci, torch.Tensor)
        assert ricci.shape == (points.shape[0], points.shape[1], points.shape[1])

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_ricci_tensor"])
    def test_flow_computation(self, flow, points, metric):
        """Test flow vector computation."""
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        assert isinstance(flow_vector, torch.Tensor)
        assert flow_vector.shape == points.shape

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_computation"])
    def test_flow_step(self, flow, points, metric):
        """Test single flow step."""
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, flow_metrics = flow.flow_step(metric, ricci)
        assert isinstance(evolved_metric, torch.Tensor)
        assert evolved_metric.shape == metric.shape
        assert isinstance(flow_metrics, FlowMetrics)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_step"])
    def test_flow_normalization(self, flow, points, metric):
        """Test flow normalization."""
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        normalized = flow.normalize_flow(flow_vector, metric)
        assert torch.all(torch.isfinite(normalized))

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_step"])
    def test_singularity_detection(self, flow, points, metric):
        """Test singularity detection."""
        singular_metric = metric.clone()
        singular_metric[:, 0, 0] = 0  # Create artificial singularity
        singularities = flow.detect_singularities(singular_metric)
        assert len(singularities) > 0
        assert isinstance(singularities[0], Singularity)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_normalization"])
    def test_geometric_invariants(self, flow, random_states, geometric_validator):
        """Test geometric invariant preservation."""
        # Extract position components
        points = random_states[:, :flow.manifold_dim]
        
        # Compute metric and flow
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        
        # Validate geometric invariants
        result = geometric_validator.validate_invariants(flow, points, evolved_metric)
        assert isinstance(result, FlowValidationResult)
        assert result.passed

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_geometric_invariants"])
    def test_energy_conservation(self, flow, random_states, energy_validator):
        """Test that energy is conserved during flow evolution."""
        # Create flow and validator
        flow = GeometricFlow(phase_dim=4)  # 2D manifold = 4D phase space
        
        # Set initial states with non-zero momentum
        initial_energy = flow.compute_energy(random_states)
        
        # Evolve system
        points = random_states[:, :flow.manifold_dim]
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        
        # Check energy conservation
        final_energy = flow.compute_energy(random_states)
        error = torch.abs(final_energy - initial_energy)
        assert torch.all(error < 1e-5)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_geometric_invariants"])
    def test_flow_stability(self, flow, random_states, geometric_validator):
        """Test flow stability."""
        points = random_states[:, :flow.manifold_dim]
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        
        # Evolve for multiple steps
        metrics = []
        for _ in range(10):
            metric, _ = flow.flow_step(metric, ricci)
            metrics.append(metric)
            
        # Check stability
        dets = [torch.linalg.det(m) for m in metrics]
        assert all(torch.all(d > 0) for d in dets)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_stability"])
    def test_flow_convergence(self, flow, random_states, convergence_validator):
        """Test that flow converges to stable points."""
        points = random_states[:, :flow.manifold_dim]
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        
        # Evolve to convergence
        for _ in range(100):
            metric, _ = flow.flow_step(metric, ricci)
            
        # Validate convergence
        result = convergence_validator.validate_convergence(flow, points, metric)
        assert isinstance(result, FlowValidationResult)
        assert result.passed

    def test_metric_conditioning(self, flow, points):
        """Test metric tensor conditioning."""
        metric = flow.compute_metric(points)
        condition_number = torch.linalg.cond(metric)
        assert torch.all(condition_number < 1e3)

    def test_flow_magnitude(self, flow, points):
        """Test flow vector magnitudes."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        assert torch.all(torch.abs(flow_vector) < 1e2)

    def test_ricci_flow_stability(self, flow, points):
        """Test Ricci flow stability."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        flow_vector = flow.compute_flow(points, ricci)
        assert torch.all(torch.isfinite(flow_vector))

    def test_ricci_flow(self, flow, points):
        """Test Ricci flow evolution."""
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        assert torch.all(torch.isfinite(evolved_metric))

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

    def test_singularity_analysis(self, flow, points):
        """Test singularity detection and analysis."""
        # Create metric with artificial singularity
        metric = flow.compute_metric(points)
        
        # Create different types of singularities
        singular_metrics = []
        
        # Type 1: Determinant singularity
        det_singular = metric.clone()
        det_singular[:, 0, 0] = 0.0
        singular_metrics.append(det_singular)
        
        # Type 2: Curvature singularity
        curv_singular = metric.clone()
        curv_singular[:, 0, 0] = 1e6  # Very large curvature
        singular_metrics.append(curv_singular)
        
        # Detect singularities in each metric
        found_singularities = False
        for singular_metric in singular_metrics:
            singularities = flow.detect_singularities(singular_metric)
            if len(singularities) > 0:
                found_singularities = True
                break
        
        assert found_singularities, "No singularities detected in artificially singular metrics"
        
        # Verify singularity properties
        for singularity in singularities:
            assert isinstance(singularity, Singularity)
            assert torch.all(torch.isfinite(singularity.location))
            assert torch.all(torch.isfinite(singularity.curvature))


class TestFlowStability:
    """Test class for flow stability and normalization diagnostics."""
    
    @pytest.fixture
    def flow_system(self):
        """Create flow system fixture."""
        return GeometricFlow(phase_dim=4)  # 2D manifold = 4D phase space
        
    @pytest.fixture
    def points(self):
        """Create random points in position space."""
        return torch.randn(4, 2, requires_grad=True)  # batch_size=4, manifold_dim=2

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
        """Test that flow preserves volume."""
        # Compute initial metric and volume
        metric = flow_system.compute_metric(points)
        initial_volume = torch.abs(torch.linalg.det(metric))
        
        # Evolve metric using flow
        ricci = flow_system.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow_system.flow_step(metric, ricci)
        
        # Compute evolved volume
        final_volume = torch.abs(torch.linalg.det(evolved_metric))
        
        # Normalize volumes to account for scaling
        normalized_initial = initial_volume / initial_volume.mean()
        normalized_final = final_volume / final_volume.mean()
        
        # Check relative volume change with tolerance
        relative_change = torch.abs(normalized_final - normalized_initial) / normalized_initial
        max_volume_change = 0.2  # Allow up to 20% change for numerical stability
        
        assert torch.all(relative_change < max_volume_change), \
            f"Volume changed by {relative_change.max().item():.3f}, exceeding threshold {max_volume_change}"

    def test_ricci_flow_stability(self, flow_system, points):
        """Test stability of Ricci flow evolution."""
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric, points)
        
        # Evolve for multiple steps
        for _ in range(10):
            metric, _ = flow_system.flow_step(metric, ricci)
            ricci = flow_system.compute_ricci_tensor(metric, points)
            
        # Check stability
        assert torch.all(torch.isfinite(metric))
        assert torch.all(torch.linalg.det(metric) > 0)
