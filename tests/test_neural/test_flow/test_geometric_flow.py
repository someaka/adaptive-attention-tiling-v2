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
    FlowStabilityValidator,
    EnergyValidator,
    ConvergenceValidator,
    GeometricFlowValidator,
    ValidationResult,
)


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
        return EnergyValidator(tolerance=1e-5)

    @pytest.fixture
    def convergence_validator(self):
        """Create convergence validator fixture."""
        return ConvergenceValidator(threshold=1e-4, max_iterations=1000)

    def test_ricci_tensor(self, flow, points, metric):
        """Test Ricci tensor computation."""
        ricci = flow.compute_ricci_tensor(metric)
        assert isinstance(ricci, RicciTensor)
        assert ricci.tensor.shape == (4, 4, 4)

    def test_flow_step(self, flow, points, metric):
        """Test flow step evolution."""
        ricci = flow.compute_ricci_tensor(metric)
        evolved_metric, metrics = flow.flow_step(metric, ricci)
        assert evolved_metric.shape == metric.shape
        assert len(metrics) == 2

    def test_singularity_detection(self, points, metric, flow):
        """Test singularity detection."""
        singularities = flow.detect_singularities(metric)
        assert isinstance(singularities, list)
        if len(singularities) > 0:
            assert isinstance(singularities[0], Singularity)
            
    def test_flow_normalization(self, points, metric, flow):
        """Test flow normalization."""
        ricci = flow.compute_ricci_tensor(metric)
        flow_vector = flow.compute_flow(points, ricci)
        
        # Normalize flow
        normalized_flow = flow.normalize_flow(flow_vector, metric)
        
        # Check volume preservation
        det = torch.det(metric)
        vol1 = torch.sqrt(det)
        
        # Evolve points with normalized flow
        dt = 0.01
        new_points = points + dt * normalized_flow
        new_metric = flow.compute_metric(new_points)
        
        # Compute new volume
        det2 = torch.det(new_metric)
        vol2 = torch.sqrt(det2)
        
        # Check relative volume change
        rel_vol = vol2 / (vol1 + 1e-8)
        assert torch.allclose(rel_vol, torch.ones_like(rel_vol), rtol=1e-2)

    def test_geometric_invariants(self, flow, random_states, geometric_validator):
        """Test that geometric flow preserves geometric invariants."""
        flow.points = random_states
        metric = flow.compute_metric(random_states)
        assert geometric_validator.validate_invariants(flow, random_states, metric)

    def test_flow_stability(self, flow, random_states, stability_validator):
        """Test that geometric flow is numerically stable."""
        flow.points = random_states
        metric = flow.compute_metric(random_states)
        result = stability_validator.validate_stability(flow, random_states)
        assert result.stable

    def test_energy_conservation(self, flow, random_states, energy_validator):
        """Test that geometric flow conserves energy."""
        flow.points = random_states
        dt = 0.001  # Use smaller time step for better energy conservation
        trajectories, metrics, _ = flow.evolve(random_states, num_steps=100, dt=dt)
        validation = energy_validator.validate_energy(flow.hamiltonian, trajectories, time_steps=50)
        assert validation.conserved, f"Energy not conserved: relative error={validation.relative_error}, drift={validation.drift_rate}"

    def test_flow_convergence(self, flow, random_states, convergence_validator):
        """Test that geometric flow converges."""
        flow.points = random_states
        trajectories, metrics, _ = flow.evolve(random_states, num_steps=50, dt=0.001)
        result = convergence_validator.validate_convergence(flow, trajectories[-1])
        assert result.converged, f"Flow did not converge after {len(trajectories)} steps"

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
        
    def test_singularity_analysis(self, points, metric, flow):
        """Test singularity analysis."""
        # Create a near-singular metric by scaling one component close to zero
        singular_metric = metric.clone()
        singular_metric[:, 0, 0] = 1e-8  # Near-singular component
        
        # Run singularity analysis
        singularities = flow.detect_singularities(points, singular_metric)
        
        # Check that at least one singularity was detected
        assert len(singularities) > 0, "No singularities detected in near-singular metric"
        
        # Check that the first singularity has the expected properties
        first_singularity = singularities[0]
        assert isinstance(first_singularity, Singularity)
        assert first_singularity.severity > 0.5, "Singularity severity too low"

    def test_flow_stability_old(self, points, metric, flow, stability_validator):
        """Test flow stability."""
        result = stability_validator.validate_stability(flow, points)
        assert result.stable, "Flow stability validation failed"
        
    def test_energy_conservation_old(self, points, metric, flow, energy_validator):
        """Test energy conservation."""
        trajectories, metrics, _ = flow.evolve(points, num_steps=50, dt=0.001)
        validation = energy_validator.validate_energy(flow.hamiltonian, trajectories, time_steps=50)
        assert validation.conserved, f"Energy conservation validation failed: {validation.relative_error}"

    def test_convergence_old(self, points, metric, flow, convergence_validator):
        """Test convergence properties."""
        trajectories, metrics, _ = flow.evolve(points, num_steps=50, dt=0.001)
        result = convergence_validator.validate_convergence(flow, trajectories[-1])
        assert result.converged, "Flow did not converge"


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
        
        # Compute covariant divergence
        for i in range(n):
            for j in range(n):
                # Use metric contraction with flow
                div += metric[:, i, j] * flow_system.compute_flow_vector(points, flow_system.compute_ricci_tensor(metric))[:, j]
        
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
