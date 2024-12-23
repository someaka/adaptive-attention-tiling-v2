"""Tests for neural geometric flow implementation."""

import pytest
import torch
from torch import Tensor
import yaml
from pathlib import Path

from src.core.common.dimensions import QuantumTensor, GeometricTensor, DimensionConfig
from src.core.flow.neural import NeuralGeometricFlow
from src.core.quantum.types import QuantumState


def load_test_config():
    config_path = Path("configs/test_regimens/tiny.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def test_config():
    return load_test_config()


@pytest.fixture
def manifold_dim():
    """Dimension réduite pour les tests."""
    return 8


@pytest.fixture
def hidden_dim():
    """Dimension cachée réduite pour les tests."""
    return 16


@pytest.fixture
def batch_size():
    """Taille de batch réduite pour les tests."""
    return 4


@pytest.fixture
def dimension_config(manifold_dim, hidden_dim):
    """Create test dimension configuration."""
    return DimensionConfig(
        attention_depth=4,
        quantum_dim=hidden_dim,
        geometric_dim=manifold_dim,
        flow_dim=hidden_dim,
        emergence_dim=hidden_dim
    )


@pytest.fixture
def neural_flow(manifold_dim, hidden_dim):
    """Create test neural geometric flow."""
    return NeuralGeometricFlow(
        manifold_dim=manifold_dim,
        hidden_dim=hidden_dim,
        dtype=torch.float32,
        device=torch.device('cpu')
    )


@pytest.fixture
def device():
    """Device for tests."""
    return torch.device('cpu')


class MockQuantumState(QuantumState):
    def __init__(self, hidden_dim: int, batch_size: int = 2, device: torch.device = torch.device('cpu')):
        self.amplitudes = torch.randn(batch_size, hidden_dim, device=device)
        self.basis_labels = [f"basis_{i}" for i in range(hidden_dim)]
        self.phase = torch.zeros(1, device=device)
        self._hidden_dim = hidden_dim
        
    def density_matrix(self) -> Tensor:
        return torch.randn(self.amplitudes.shape[0], self._hidden_dim, self._hidden_dim, device=self.amplitudes.device)

    def ricci_tensor(self) -> Tensor:
        batch_size = self.amplitudes.shape[0]
        # Create a positive definite Ricci tensor
        base = torch.randn(batch_size, self._hidden_dim, self._hidden_dim, device=self.amplitudes.device)
        return torch.matmul(base, base.transpose(-1, -2))  # Ensure positive definiteness


class TestNeuralGeometricFlow:
    """Test neural geometric flow implementation."""

    def test_tensor_types(self, neural_flow, batch_size, device):
        """Test tensor type conversions."""
        # Create test tensors with consistent dimensions
        shape = (batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        quantum_tensor = torch.randn(*shape, device=device)
        geometric_tensor = torch.randn(*shape, device=device)
        
        # Convert to specific tensor types
        quantum_tensor = neural_flow.dim_manager.to_quantum_tensor(quantum_tensor)
        geometric_tensor = neural_flow.dim_manager.to_geometric_tensor(geometric_tensor)
        
        # Verify types and operations
        assert isinstance(quantum_tensor, QuantumTensor)
        assert isinstance(geometric_tensor, GeometricTensor)
        assert torch.allclose(quantum_tensor + quantum_tensor, 2 * quantum_tensor)
        mean_value = geometric_tensor.float().mean()
        expected_mean = mean_value.clone().detach()
        assert torch.allclose(geometric_tensor.mean(), expected_mean)

    def test_dimension_validation(self, neural_flow, batch_size, device):
        """Test dimension validation."""
        # Test with valid dimensions
        valid_tensor = torch.randn(batch_size, neural_flow.hidden_dim, device=device)
        assert neural_flow.dim_manager.verify_dimension(valid_tensor)
        
        # Test with invalid dimensions
        invalid_tensor = torch.randn(batch_size, neural_flow.manifold_dim // 2, device=device)
        with pytest.raises(ValueError, match="Tensor dimension .* below minimum"):
            neural_flow.dim_manager.verify_dimension(invalid_tensor)
            
        # Test projection with invalid dimensions
        with pytest.raises(ValueError):
            neural_flow.dim_manager.validate_and_project(
                invalid_tensor,
                target_dim=neural_flow.hidden_dim,
                dtype=torch.float32,
                device=device
            )

    def test_quantum_corrections_types(self, neural_flow, batch_size, device):
        """Test quantum corrections with tensor types."""
        # Create test instances with proper dimensions
        state = MockQuantumState(neural_flow.hidden_dim, batch_size, device)
        metric = torch.randn(batch_size, neural_flow.hidden_dim, neural_flow.hidden_dim, device=device)
        
        # Compute corrections and verify result
        result = neural_flow.compute_quantum_corrections(state, metric)
        
        # Verify result type and shape
        assert isinstance(result, Tensor)
        assert result.shape == (batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        
        # Verify intermediate conversions
        with torch.no_grad():
            state_matrix = state.density_matrix()
            quantum_tensor = neural_flow.dim_manager.to_quantum_tensor(state_matrix)
            assert isinstance(quantum_tensor, QuantumTensor)

    def test_metric_computation(self, neural_flow, batch_size, device):
        """Test metric tensor computation."""
        points = torch.randn(batch_size, neural_flow.manifold_dim, device=device)
        metric = neural_flow.compute_metric(points)
        
        # Verify shape and symmetry
        assert metric.shape == (batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        assert torch.allclose(metric, metric.transpose(-1, -2), rtol=1e-4, atol=1e-4)
        
        # Verify positive definiteness with a small tolerance
        eigenvalues = torch.linalg.eigvalsh(metric)  # Use eigvalsh for symmetric matrices
        assert torch.all(eigenvalues > -1e-6)  # Allow for small numerical errors

    def test_flow_step(self, neural_flow, batch_size, device):
        """Test geometric flow step."""
        # Utiliser un batch_size réduit mais les dimensions complètes du modèle
        test_batch_size = min(batch_size, 4)  # Limiter à 4 échantillons
        
        metric = torch.eye(neural_flow.manifold_dim, device=device)
        metric = metric.unsqueeze(0).repeat(test_batch_size, 1, 1)
        
        # Créer un tenseur de Ricci valide avec les dimensions complètes
        ricci = torch.randn(test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim, device=device)
        # Rendre le tenseur de Ricci symétrique
        ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
        
        # Vérifier les dimensions avant le flow_step
        assert metric.shape == (test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        assert ricci.shape == (test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        
        new_metric, flow_metrics = neural_flow.flow_step(metric, ricci)
        
        # Vérifier les dimensions et propriétés après le flow_step
        assert new_metric.shape == metric.shape
        assert torch.allclose(new_metric, new_metric.transpose(-1, -2))
        
        # Vérifier que les métriques de flux sont valides
        assert isinstance(flow_metrics.flow_magnitude, torch.Tensor)
        assert isinstance(flow_metrics.metric_determinant, torch.Tensor)
        assert isinstance(flow_metrics.ricci_scalar, torch.Tensor)
        
        # Vérifier les dimensions des métriques
        assert flow_metrics.flow_magnitude.shape == (test_batch_size,)
        assert flow_metrics.metric_determinant.shape == (test_batch_size,)
        assert flow_metrics.ricci_scalar.shape == (test_batch_size,)

    def test_geometric_operations(self, neural_flow, batch_size, device):
        """Test geometric operations."""
        # Create test points with correct dimensions
        points = torch.randn(batch_size, neural_flow.manifold_dim, device=device)
        
        # Compute connection coefficients
        connection = neural_flow.compute_connection(points)
        assert connection.shape == (batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim, neural_flow.manifold_dim)
        
        # Compute Ricci tensor
        ricci = neural_flow.compute_ricci(points)
        assert ricci.shape == (batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        assert torch.allclose(ricci, ricci.transpose(-1, -2))  # Verify symmetry
        
        # Compute scalar curvature
        scalar = neural_flow.compute_scalar_curvature(points)
        assert scalar.shape == (batch_size,)
        assert torch.all(torch.isfinite(scalar))  # Verify finite values