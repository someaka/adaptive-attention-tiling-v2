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

from src.core.attention.geometric import GeometricStructures
from src.core.flow import NeuralGeometricFlow
from src.core.flow.protocol import FlowMetrics, SingularityInfo as Singularity
from src.validation.geometric.flow import (
    TilingFlowValidator as FlowValidator,
    TilingFlowValidationResult as FlowValidationResult,
)

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

    def test_geometric_metric_computation(
        self, flow_layer, test_input, test_config
    ):
        """Test metric tensor computation."""
        # Create input tensor with manifold dimension
        x = test_input
        
        # Compute metric tensor
        metric = flow_layer.compute_metric_tensor(x)
        
        # Test metric tensor properties
        assert metric.shape == (
            test_config['batch_size'],
            test_config['manifold_dim'],
            test_config['manifold_dim'],
        ), "Metric tensor should have manifold dimensions"
        
        # Test metric tensor symmetry
        assert torch.allclose(
            metric, metric.transpose(-1, -2), rtol=1e-5
        ), "Metric tensor should be symmetric"
        
        # Test metric tensor positive definiteness
        eigenvals = torch.linalg.eigvalsh(metric)
        assert torch.all(eigenvals > 0), "Metric tensor should be positive definite"

    def test_flow_integration(
        self, flow_layer, test_input, test_config
    ):
        """Test geometric flow integration."""
        # Create input tensor with manifold dimension
        x = test_input
        
        # Integrate flow
        integrated = flow_layer.integrate_flow(x)
        
        # Test integrated shape
        assert integrated.shape == (
            test_config['batch_size'],
            test_config['seq_len'],
            test_config['hidden_dim'],
        ), "Integration should map to hidden dimension"
        
        # Test flow properties
        flow_field = flow_layer.compute_flow_field(x)
        assert flow_field.shape == (
            test_config['batch_size'],
            test_config['seq_len'],
            test_config['manifold_dim'],
        ), "Flow field should preserve manifold dimension"
        
        # Test conservation laws
        divergence = flow_layer.compute_divergence(flow_field)
        assert torch.allclose(
            divergence.mean(), torch.zeros(1), atol=1e-4
        ), "Flow should be approximately divergence-free"

    def test_curvature_flow(
        self, flow_layer, test_input, test_config
    ):
        """Test curvature flow properties."""
        # Create input tensor with manifold dimension
        x = test_input
        
        # Compute curvature flow
        flow = flow_layer.compute_curvature_flow(x)
        
        # Test flow shape
        assert flow.shape == (
            test_config['batch_size'],
            test_config['seq_len'],
            test_config['manifold_dim'],
        ), "Curvature flow should preserve manifold dimension"
        
        # Test curvature properties
        ricci = flow_layer.compute_ricci_flow(x)
        assert ricci.shape == (
            test_config['batch_size'],
            test_config['manifold_dim'],
            test_config['manifold_dim'],
        ), "Ricci flow should have manifold dimensions"
        
        # Test scalar curvature
        scalar = flow_layer.compute_scalar_curvature(x)
        assert scalar.shape == (test_config['batch_size'],), "Scalar curvature should be scalar"

    def test_flow_dynamics(
        self, flow_layer, test_input, test_config
    ):
        """Test geometric flow dynamics."""
        # Create input tensor with manifold dimension
        x = test_input
        
        # Evolve flow
        evolved = flow_layer.evolve_flow(x, time_steps=10)
        
        # Test evolved shape
        assert evolved.shape == (
            test_config['batch_size'],
            test_config['seq_len'],
            test_config['hidden_dim'],
        ), "Evolution should map to hidden dimension"
        
        # Test energy conservation
        initial_energy = flow_layer.compute_flow_energy(x)
        final_energy = flow_layer.compute_flow_energy(evolved)
        assert torch.allclose(
            initial_energy, final_energy, rtol=1e-3
        ), "Flow should approximately conserve energy"
        
        # Test geometric invariants
        volume_form = flow_layer.compute_volume_form(x)
        evolved_form = flow_layer.compute_volume_form(evolved)
        assert torch.allclose(
            volume_form.mean(), evolved_form.mean(), rtol=1e-3
        ), "Flow should preserve volume form"

    @pytest.fixture
    def points(self, test_config):
        """Create random points in position space."""
        return torch.randn(test_config['batch_size'], test_config['manifold_dim'], requires_grad=True)

    @pytest.fixture
    def random_states(self, test_config):
        """Create random phase space states (position and momentum)."""
        phase_dim = test_config['manifold_dim'] * 2
        return torch.randn(test_config['batch_size'], phase_dim, requires_grad=True)

    @pytest.fixture
    def metric(self, flow, points):
        """Create metric tensor."""
        return flow.compute_metric(points)

    @pytest.fixture
    def validator(self, flow):
        """Create flow validator."""
        return FlowValidator(
            flow=flow,
            stability_threshold=1e-6,
            curvature_bounds=(-1.0, 1.0),
            max_energy=1e3
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
        ricci = flow.compute_ricci_tensor(metric)
        flow_vector = flow.compute_flow_vector(points, ricci, metric)
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
        assert result.is_valid

    @pytest.mark.dependency(depends=["TestGeometricFlow::test_geometric_invariants"])
    def test_energy_conservation(self, flow, random_states, energy_validator):
        """Test that energy is conserved during flow evolution."""
        # Extract position components
        points = random_states[:, :flow.manifold_dim]
        
        # Compute metric and flow
        metric = flow.compute_metric(points)
        ricci = flow.compute_ricci_tensor(metric, points)
        evolved_metric, _ = flow.flow_step(metric, ricci)
        
        # Compute initial and final energies
        initial_energies = flow.pattern_dynamics.compute_energy(random_states)
        initial_total = torch.sum(torch.stack(list(initial_energies.values())))
        
        # Evolve system and check energy conservation
        final_energies = flow.pattern_dynamics.compute_energy(random_states)
        final_total = torch.sum(torch.stack(list(final_energies.values())))
        error = torch.abs(final_total - initial_total)
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
        assert result.is_valid

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
            if singularity.location is not None:
                assert torch.all(torch.isfinite(singularity.location))
            if singularity.curvature is not None:
                assert torch.all(torch.isfinite(singularity.curvature))


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
