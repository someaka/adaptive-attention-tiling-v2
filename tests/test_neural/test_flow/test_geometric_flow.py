"""
Unit tests for the geometric flow system.

This test suite is organized in levels of increasing complexity:
1. Basic Metric Operations
2. Curvature Components
3. Flow Evolution
4. Stability and Conservation
5. Advanced Features

Each level builds upon the previous ones to ensure systematic validation
of the geometric flow implementation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.neural.flow.geometric_flow import (
    FlowMetrics,
    GeometricFlow,
    RicciTensor,
    SingularityInfo,
)
from src.validation.geometric.flow import (
    FlowStabilityValidation,
    EnergyValidation,
    ConvergenceValidation,
    GeometricFlowValidator,
    FlowStabilityValidator,
    EnergyValidator,
    ConvergenceValidator,
)
from src.validation.flow.stability import (
    LinearStabilityValidator,
    NonlinearStabilityValidator,
    StructuralStabilityValidator,
    StabilityValidator,
    LinearStabilityValidation,
    NonlinearStabilityValidation,
    StructuralStabilityValidation,
)
from src.neural.flow.hamiltonian import HamiltonianSystem

# ============================================================================
# Level 1: Basic Metric Operations
# ============================================================================

class TestMetricOperations:
    """Test basic metric tensor operations and properties."""
    
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
    def points(self, batch_size, manifold_dim):
        """Create well-conditioned test points."""
        return torch.randn(batch_size, manifold_dim, requires_grad=True)
        
    @pytest.fixture
    def metric(self, points, flow_system):
        """Create test metric."""
        return flow_system.compute_metric(points)

    @pytest.fixture
    def flow_validator(self):
        """Create a flow validator instance."""
        return FlowValidator(
            stability_threshold=0.1,
            energy_threshold=0.1,
            singularity_threshold=0.1
        )

    def test_metric_initialization(self, flow_system, batch_size, manifold_dim):
        """Test metric initialization and basic properties."""
        metric = flow_system.initialize_metric(batch_size, manifold_dim)
        
        # Check shape
        assert metric.shape == (batch_size, manifold_dim, manifold_dim)
        
        # Check symmetry
        assert torch.allclose(metric, metric.transpose(-2, -1))
        
        # Check positive definiteness
        eigenvals = torch.linalg.eigvalsh(metric)
        assert torch.all(eigenvals > 0)

    def test_metric_conditioning(self, metric, flow_validator):
        """Test metric tensor conditioning."""
        result = flow_validator.validate_metric_tensor(metric)
        assert result.is_valid, result.message

    def test_metric_operations(self, metric):
        """Test basic metric operations."""
        # Test inverse
        metric_inv = torch.linalg.inv(metric)
        identity = torch.eye(metric.shape[1], device=metric.device)
        identity = identity.expand(metric.shape[0], -1, -1)
        
        # Check inverse property
        product = torch.bmm(metric, metric_inv)
        assert torch.allclose(product, identity, rtol=1e-5)
        
        # Test determinant computation
        det = torch.det(metric)
        assert torch.all(torch.isfinite(det))
        assert torch.all(det > 0)


# ============================================================================
# Level 2: Curvature Components
# ============================================================================

class TestCurvatureComponents:
    """Test curvature-related computations."""
    
    @pytest.fixture
    def flow_system(self, manifold_dim):
        return GeometricFlow(manifold_dim=manifold_dim)
        
    @pytest.fixture
    def manifold_dim(self):
        return 4
        
    @pytest.fixture
    def points(self, batch_size, manifold_dim):
        return torch.randn(batch_size, manifold_dim, requires_grad=True)
        
    @pytest.fixture
    def metric(self, points, flow_system):
        return flow_system.compute_metric(points)
        
    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def stability_validator(self):
        """Create a stability validator instance."""
        return StabilityValidator(
            stability_threshold=0.1,
            time_window=10
        )

    def test_connection_coefficients(self, metric, flow_system):
        """Test Christoffel symbols computation."""
        connection = flow_system.compute_connection(metric)
        
        # Check shape
        n = metric.shape[1]
        assert connection.shape == (metric.shape[0], n, n, n)
        
        # Check symmetry in lower indices
        assert torch.allclose(
            connection[..., 0, 1, :],
            connection[..., 1, 0, :],
            rtol=1e-5
        )

    def test_ricci_tensor(self, metric, flow_system, stability_validator):
        """Test Ricci tensor computation."""
        ricci = flow_system.compute_ricci_tensor(metric)
        result = stability_validator.validate_ricci_tensor(metric, ricci.tensor)
        assert result.is_valid, result.message

    def test_scalar_curvature(self, metric, flow_system):
        """Test scalar curvature computation."""
        ricci = flow_system.compute_ricci_tensor(metric)
        scalar_curv = flow_system.compute_scalar_curvature(metric)
        
        # Check shape and finiteness
        assert scalar_curv.shape == (metric.shape[0],)
        assert torch.all(torch.isfinite(scalar_curv))


# ============================================================================
# Level 3: Flow Evolution
# ============================================================================

class TestFlowEvolution:
    """Test flow evolution and normalization."""
    
    @pytest.fixture
    def flow_system(self, manifold_dim):
        return GeometricFlow(manifold_dim=manifold_dim)
        
    @pytest.fixture
    def manifold_dim(self):
        return 4
        
    @pytest.fixture
    def points(self, batch_size, manifold_dim):
        return torch.randn(batch_size, manifold_dim, requires_grad=True)
        
    @pytest.fixture
    def metric(self, points, flow_system):
        return flow_system.compute_metric(points)
        
    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def energy_validator(self):
        """Create an energy validator instance."""
        return EnergyValidator(
            tolerance=1e-6,
            drift_threshold=1e-4
        )

    def test_flow_step(self, metric, flow_system, energy_validator):
        """Test single flow step evolution."""
        ricci = flow_system.compute_ricci_tensor(metric)
        new_metric, metrics = flow_system.flow_step(metric, ricci)
        result = energy_validator.validate_flow_step(metric, new_metric, metrics)
        assert result.is_valid, result.message

    def test_flow_magnitude(self, metric, flow_system):
        """Test flow vector magnitudes."""
        ricci = flow_system.compute_ricci_tensor(metric)
        
        # Compute flow
        new_metric, _ = flow_system.flow_step(metric, ricci)
        flow = new_metric - metric
        
        # Check flow magnitude
        flow_norm = torch.norm(flow, dim=(1,2))
        assert torch.all(flow_norm < 1000.0)

    def test_flow_normalization(self, metric, flow_system):
        """Test flow normalization."""
        # Get initial volume
        init_vol = flow_system.compute_volume(metric)
        
        # Evolve metric
        ricci = flow_system.compute_ricci_tensor(metric)
        new_metric, _ = flow_system.flow_step(metric, ricci)
        
        # Get new volume
        new_vol = flow_system.compute_volume(new_metric)
        
        # Check volume preservation
        rel_vol = new_vol / (init_vol + 1e-8)
        assert torch.allclose(rel_vol, torch.ones_like(rel_vol), rtol=1e-2)


# ============================================================================
# Level 4: Stability and Conservation
# ============================================================================

class TestLevel4StabilityAndConservation:
    """Level 4 tests focusing on flow stability and conservation properties."""

    @pytest.fixture
    def geometric_flow(self):
        """Create geometric flow fixture."""
        return GeometricFlow(
            manifold_dim=4,
            flow_type="ricci",
            timestep=0.01,
            normalize_method="volume",
            hidden_dim=128
        )

    @pytest.fixture
    def test_points(self):
        """Create test points fixture."""
        return torch.randn(8, 4, requires_grad=True)

    @pytest.fixture
    def convergence_validator(self):
        """Create a convergence validator instance."""
        return ConvergenceValidator(
            convergence_threshold=1e-6,
            max_iterations=1000
        )

    def test_flow_stability(self, convergence_validator, geometric_flow, test_points):
        """Test overall flow stability."""
        metric = geometric_flow.compute_metric(test_points)
        metric = metric.reshape(metric.shape[0], geometric_flow.manifold_dim, geometric_flow.manifold_dim)
        evolved_metric = geometric_flow.step(metric)
        validation_result = convergence_validator.validate(
            metric=metric,
            evolved_metric=evolved_metric,
            time_step=geometric_flow.timestep
        )
        assert validation_result.is_valid, validation_result.message

    def test_linear_stability(self, convergence_validator, geometric_flow, test_points):
        """Test linear stability properties."""
        metric = geometric_flow.compute_metric(test_points)
        metric = metric.reshape(metric.shape[0], geometric_flow.manifold_dim, geometric_flow.manifold_dim)
        result = convergence_validator.validate(
            metric=metric,
            flow=geometric_flow,
            time_steps=100
        )
        assert result.is_valid, "Linear stability validation failed"
        assert torch.all(result.stability_measure < convergence_validator.convergence_threshold)

    def test_nonlinear_stability(self, convergence_validator, geometric_flow, test_points):
        """Test nonlinear stability properties."""
        metric = geometric_flow.compute_metric(test_points)
        metric = metric.reshape(metric.shape[0], geometric_flow.manifold_dim, geometric_flow.manifold_dim)
        result = convergence_validator.validate(
            metric=metric,
            flow=geometric_flow,
            time_steps=100
        )
        assert result.is_valid, "Nonlinear stability validation failed"
        assert torch.all(result.stability_measure < convergence_validator.convergence_threshold)

    def test_energy_conservation(self, convergence_validator, geometric_flow, test_points):
        """Test energy conservation properties."""
        metric = geometric_flow.compute_metric(test_points)
        metric = metric.reshape(metric.shape[0], geometric_flow.manifold_dim, geometric_flow.manifold_dim)
        result = convergence_validator.validate(
            metric=metric,
            time_steps=100
        )
        assert result.conserved, "Energy conservation validation failed"
        assert result.relative_error < convergence_validator.convergence_threshold
        assert result.drift_rate < convergence_validator.convergence_threshold

    def test_convergence(self, convergence_validator, geometric_flow, test_points):
        """Test convergence properties."""
        metric = geometric_flow.compute_metric(test_points)
        metric = metric.reshape(metric.shape[0], geometric_flow.manifold_dim, geometric_flow.manifold_dim)
        result = convergence_validator.validate(
            metric=metric,
            flow=geometric_flow,
            max_iterations=100
        )
        assert result.converged, "Convergence validation failed"
        assert result.iterations <= convergence_validator.max_iterations


# ============================================================================
# Level 5: Advanced Features
# ============================================================================

class TestAdvancedFeatures:
    """Test advanced geometric flow features."""
    
    @pytest.fixture
    def flow_system(self, manifold_dim):
        return GeometricFlow(manifold_dim=manifold_dim)
        
    @pytest.fixture
    def manifold_dim(self):
        return 4
        
    @pytest.fixture
    def points(self, batch_size, manifold_dim):
        return torch.randn(batch_size, manifold_dim, requires_grad=True)
        
    @pytest.fixture
    def metric(self, points, flow_system):
        return flow_system.compute_metric(points)
        
    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def convergence_validator(self):
        """Create a convergence validator instance."""
        return ConvergenceValidator(
            convergence_threshold=1e-6,
            max_iterations=1000
        )

    def test_singularity_detection(self, metric, flow_system):
        """Test singularity detection and classification."""
        # Create near-singular metric
        singular_metric = metric.clone()
        singular_metric[:,0,0] *= 0.1
        
        # Detect singularities
        singularities = flow_system.detect_singularities(singular_metric)
        
        # Check detection
        assert len(singularities) > 0
        assert all(isinstance(s, SingularityInfo) for s in singularities)

    def test_neck_detection(self, metric, flow_system):
        """Test neck-pinching detection."""
        necks = flow_system.detect_necks(metric)
        
        # Basic checks on neck detection
        assert isinstance(necks, list)
        if len(necks) > 0:
            assert all(isinstance(n, SingularityInfo) for n in necks)
            assert all(n.type == "neck" for n in necks)

    def test_singularity_time_estimation(self, metric, flow_system):
        """Test singularity formation time estimation."""
        time = flow_system.estimate_singularity_time(metric)
        assert isinstance(time, float)
        assert time > 0 or time == float('inf')

    def test_convergence(self, metric, flow_system, convergence_validator):
        """Test flow convergence properties."""
        result = convergence_validator.validate(
            metric=metric,
            flow=flow_system,
            max_iterations=100
        )
        assert result.converged, f"Flow did not converge: {result.rate}"
        assert result.iterations < convergence_validator.max_iterations
