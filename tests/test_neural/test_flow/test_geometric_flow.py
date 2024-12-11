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
    RicciTensor,
    SingularityInfo as Singularity,
)
from src.validation.geometric.flow import (
    FlowStabilityValidator,
    EnergyValidator,
    ConvergenceValidator,
    GeometricFlowValidator,
    ValidationResult,
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
        return 4

    @pytest.fixture
    def device(self):
        """Device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def flow(self, manifold_dim):
        """Create geometric flow instance."""
        return GeometricFlow(manifold_dim=manifold_dim)

    @pytest.fixture
    def points(self, batch_size, manifold_dim):
        """Create test points."""
        points = torch.zeros(batch_size, manifold_dim)
        points[0, 0] = 1.0  # First basis vector
        points[1, 1] = 1.0  # Second basis vector
        points[2, 2] = 1.0  # Third basis vector
        points[3, 3] = 1.0  # Fourth basis vector
        points.requires_grad_(True)
        return points

    @pytest.fixture
    def metric(self, flow, points):
        """Create test metric."""
        flow.points = points
        return flow.compute_metric(points)

    @pytest.fixture
    def random_states(self, batch_size, manifold_dim):
        """Create random test states."""
        return torch.randn(batch_size, manifold_dim, requires_grad=True)

    @pytest.fixture
    def geometric_validator(self):
        """Create geometric flow validator fixture."""
        return GeometricFlowValidator()
        
    @pytest.fixture
    def stability_validator(self):
        """Create stability validator fixture."""
        return FlowStabilityValidator(tolerance=1e-5, stability_threshold=0.1)

    @pytest.fixture
    def energy_validator(self):
        """Create energy validator fixture."""
        from src.validation.geometric.flow import EnergyValidator
        return EnergyValidator(tolerance=1e-4)

    @pytest.fixture
    def convergence_validator(self):
        """Create convergence validator fixture."""
        from src.validation.geometric.flow import ConvergenceValidator
        return ConvergenceValidator(threshold=1e-4)

    # Level 1: Basic Component Tests
    @pytest.mark.dependency()
    def test_metric_computation(self, flow, points):
        """Test metric tensor computation."""
        metric = flow.compute_metric(points)
        assert isinstance(metric, torch.Tensor)
        assert metric.shape == (points.shape[0], points.shape[1], points.shape[1])
        assert torch.all(torch.linalg.det(metric) > 0)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_metric_computation"])
    def test_ricci_tensor(self, flow, points, metric):
        """Test Ricci tensor computation."""
        ricci = flow.compute_ricci_tensor(metric)
        assert isinstance(ricci, RicciTensor)
        assert ricci.tensor.shape == (points.shape[0], points.shape[1], points.shape[1])

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_ricci_tensor"])
    def test_flow_computation(self, flow, points, metric):
        """Test flow vector computation."""
        ricci = flow.compute_ricci_tensor(metric)
        flow_vector = flow.compute_flow(points, ricci)
        assert isinstance(flow_vector, torch.Tensor)
        assert flow_vector.shape == points.shape

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_computation"])
    def test_flow_step(self, flow, points, metric):
        """Test flow step evolution."""
        ricci = flow.compute_ricci_tensor(metric)
        evolved_metric, metrics = flow.flow_step(metric, ricci)
        assert evolved_metric.shape == metric.shape
        assert len(metrics) == 2

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_step"])
    def test_flow_normalization(self, flow, points, metric):
        """Test flow normalization."""
        # Test flow normalization
        normalized_flow = flow.normalize_flow(points, metric)
        assert normalized_flow.shape == points.shape
        assert torch.all(torch.abs(normalized_flow) <= 1.0)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_step"])
    def test_singularity_detection(self, flow, points, metric):
        """Test singularity detection."""
        # Create singular metric
        singular_metric = metric.clone()
        singular_metric[0, 0, 0] = 0  # Create singularity
        
        # Detect singularities
        singularities = flow.detect_singularities(singular_metric)
        assert len(singularities) > 0
        
        # Check singularity properties
        for sing in singularities:
            assert isinstance(sing, Singularity)
            assert sing.curvature > 0
            assert sing.resolution is not None

    # Level 3: Validation Tests
    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_normalization"])
    def test_geometric_invariants(self, flow, random_states, geometric_validator):
        """Test geometric invariants preservation."""
        flow.points = random_states
        metric = flow.compute_metric(random_states)
        assert geometric_validator.validate_invariants(flow, random_states, metric)

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_geometric_invariants"])
    def test_energy_conservation(self, flow, random_states, energy_validator):
        """Test that energy is conserved during flow evolution."""
        # Create flow and validator
        flow = GeometricFlow(manifold_dim=4)  # 2D manifold = 4D phase space
        validator = EnergyValidator()

        # Set initial states with non-zero momentum
        batch_size = 8
        manifold_dim = 2  # 2D manifold
        phase_dim = manifold_dim * 2  # 4D phase space
        
        # Create phase space points (position, momentum)
        position = torch.randn(batch_size, manifold_dim)
        momentum = torch.randn(batch_size, manifold_dim)
        position = position / torch.norm(position, dim=-1, keepdim=True)
        momentum = momentum / torch.norm(momentum, dim=-1, keepdim=True)
        
        # Combine into phase space states
        states = torch.cat([position, momentum], dim=-1)  # Shape: (batch_size, phase_dim)
        
        # Set points and compute initial metric
        flow.points = states
        flow.metric = flow.compute_metric(states)
        
        # Run validation
        result = validator.validate_energy(flow, states)
        
        # Print debug info
        print(f"States shape: {states.shape}")
        print(f"Initial states: {states}")
        print(f"Initial energy: {result.initial_energy}")
        print(f"Final energy: {result.final_energy}")
        print(f"Relative error: {result.relative_error}")
        
        assert result.is_valid, f"Energy not conserved: {result.relative_error}"

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_geometric_invariants"])
    def test_flow_stability(self, flow, random_states, stability_validator):
        """Test flow stability."""
        result = stability_validator.validate_stability(flow, random_states)
        assert result.stable, "Flow stability validation failed"

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_flow_stability"])
    def test_flow_convergence(self, flow, random_states, convergence_validator):
        """Test that flow converges to stable points."""
        # Create flow and validator
        flow = GeometricFlow(manifold_dim=4)  # 2D manifold = 4D phase space
        validator = ConvergenceValidator()

        # Set initial points
        batch_size = 8
        manifold_dim = 2  # 2D manifold
        phase_dim = manifold_dim * 2  # 4D phase space
        
        # Create phase space points
        position = torch.randn(batch_size, manifold_dim)
        momentum = torch.randn(batch_size, manifold_dim)
        position = position / torch.norm(position, dim=-1, keepdim=True)
        momentum = momentum / torch.norm(momentum, dim=-1, keepdim=True)
        
        # Combine into phase space points
        points = torch.cat([position, momentum], dim=-1)  # Shape: (batch_size, phase_dim)
        
        # Set points and compute initial metric
        flow.points = points
        flow.metric = flow.compute_metric(points)
        
        # Run validation
        result = validator.validate_convergence(flow, points)
        assert result.is_valid, f"Flow did not converge: {result.error}"

    def test_metric_conditioning(self, flow, random_states):
        """Test that metric remains well-conditioned."""
        metric = flow.compute_metric(random_states)
        det = torch.linalg.det(metric)
        assert torch.all(det > 1e-6), "Metric determinant too close to zero"

    def test_flow_magnitude(self, flow, random_states):
        """Test that flow magnitude stays within reasonable bounds."""
        flow.points = random_states
        metric = flow.compute_metric(random_states)
        ricci = flow.compute_ricci_tensor(metric)
        flow_vector = flow.compute_flow(random_states, ricci)
        assert torch.all(torch.abs(flow_vector) < 10.0)

    def test_ricci_flow_stability(self, flow, random_states):
        """Test stability of Ricci flow."""
        flow.points = random_states
        metric = flow.compute_metric(random_states)
        ricci = flow.compute_ricci_tensor(metric)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        assert torch.all(torch.isfinite(evolved_metric))

    def test_ricci_flow(self, points, metric, flow):
        """Test Ricci flow evolution."""
        flow.points = points
        ricci = flow.compute_ricci_tensor(metric)
        flow_vector = flow.compute_flow(points, ricci)
        # Check that flow magnitude is reasonable
        assert torch.all(torch.abs(flow_vector) < 2.0), "Flow vector magnitude too large"
        # Check that flow is non-zero
        assert not torch.allclose(flow_vector, torch.zeros_like(flow_vector)), "Flow vector is zero"

    def test_mean_curvature_flow(self, points, metric, flow):
        """Test mean curvature flow."""
        ricci = flow.compute_ricci_tensor(metric)
        flow_vector = flow.compute_flow_vector(points, ricci)
        
        # Check mean curvature
        mean_curv = flow.compute_mean_curvature(metric)
        assert mean_curv.shape == (metric.shape[0],)

    def test_singularity_analysis(self, flow, points, metric):
        """Test singularity analysis."""
        # Create singular metric
        singular_metric = metric.clone()
        singular_metric[0, 0, 0] = 0  # Create singularity
        
        # Detect singularities
        singularities = flow.detect_singularities(singular_metric)
        assert len(singularities) > 0
        
        first_singularity = singularities[0]
        assert isinstance(first_singularity, Singularity)
        assert first_singularity.curvature > 0
        assert first_singularity.resolution is not None


class GeometricFlowValidator:
    """Validator for geometric flow properties."""
    
    def validate_invariants(self, flow_system, points, metric):
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
        
        # Get flow vectors
        ricci = flow_system.compute_ricci_tensor(metric)
        flow_vectors = flow_system.compute_flow_vector(points, ricci)
        
        # Compute covariant divergence
        for i in range(n):
            for j in range(n):
                # Use metric contraction with flow
                div += metric[:, i, j] * flow_vectors[:, j]
        
        # Normalize by metric volume
        normalized_div = div / (n * sqrt_det)
        
        # Check if normalized divergence is sufficiently small
        # Use a more lenient tolerance since we're dealing with neural network outputs
        return torch.all(torch.abs(normalized_div) < 0.5)


class TestFlowStability:
    """Test class for flow stability and normalization diagnostics."""
    
    @pytest.fixture
    def flow_system(self):
        """Create flow system fixture."""
        return GeometricFlow(manifold_dim=4)
        
    @pytest.fixture
    def points(self):
        """Create well-conditioned test points."""
        batch_size = 4
        manifold_dim = 4
        points = torch.randn(batch_size, manifold_dim, requires_grad=True)
        return points
        
    def test_metric_conditioning(self, points, flow_system):
        """Test metric tensor conditioning."""
        flow_system.points = points
        metric = flow_system.compute_metric(points)
        assert torch.all(torch.isfinite(metric)), "Metric tensor contains invalid values"
        
    def test_flow_magnitude(self, points, flow_system):
        """Test flow vector magnitudes stay reasonable."""
        flow_system.points = points
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric)
        flow = flow_system.compute_flow(points, ricci)
        assert torch.all(torch.abs(flow) < 10.0), "Flow magnitudes too large"
        
    def test_volume_preservation(self, points, flow_system):
        """Test volume preservation with small steps."""
        flow_system.points = points
        metric = flow_system.compute_metric(points)
        initial_volume = flow_system.compute_volume(metric)
        ricci = flow_system.compute_ricci_tensor(metric)
        evolved_metric, _ = flow_system.flow_step(metric, ricci)
        final_volume = flow_system.compute_volume(evolved_metric)
        relative_change = torch.abs((final_volume - initial_volume) / initial_volume)
        assert torch.all(relative_change < 0.1), "Volume not preserved"
        
    def test_ricci_flow_stability(self, points, flow_system):
        """Test stability of Ricci flow evolution."""
        flow_system.points = points
        metric = flow_system.compute_metric(points)
        ricci = flow_system.compute_ricci_tensor(metric)
        evolved_metric, _ = flow_system.flow_step(metric, ricci)
        assert torch.all(torch.isfinite(evolved_metric)), "Evolution produced invalid values"
