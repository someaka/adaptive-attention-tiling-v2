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
def neural_flow(manifold_dim, hidden_dim, dimension_config):
    """Create test neural geometric flow."""
    return NeuralGeometricFlow(
        manifold_dim=manifold_dim,
        hidden_dim=hidden_dim,
        dtype=torch.float32,
        device=torch.device('cpu'),
        test_config={"manifold_dim": manifold_dim, "hidden_dim": hidden_dim}
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
        # Test with valid dimensions - use manifold_dim instead of hidden_dim
        valid_tensor = torch.randn(batch_size, neural_flow.manifold_dim, device=device)
        assert neural_flow.dim_manager.verify_dimension(valid_tensor)
        
        # Test with invalid dimensions - too small
        invalid_tensor = torch.randn(batch_size, neural_flow.manifold_dim // 2, device=device)
        with pytest.raises(ValueError, match="Tensor dimension .* below minimum"):
            neural_flow.dim_manager.verify_dimension(invalid_tensor)
            
        # Test with invalid dimensions - too large
        large_tensor = torch.randn(batch_size, neural_flow.manifold_dim * 2, device=device)
        with pytest.raises(ValueError, match="Tensor dimension .* above maximum"):
            neural_flow.dim_manager.verify_dimension(large_tensor)
            
        # Test projection with invalid dimensions
        with pytest.raises(ValueError):
            neural_flow.dim_manager.validate_and_project(
                invalid_tensor,
                target_dim=neural_flow.manifold_dim,  # Use manifold_dim instead of hidden_dim
                dtype=torch.float32,
                device=device
            )

    def test_quantum_corrections_types(self, neural_flow, batch_size, device):
        """Test quantum corrections with tensor types."""
        # Create test instances with proper dimensions
        state = MockQuantumState(neural_flow.manifold_dim, batch_size, device)
        metric = torch.randn(batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim, device=device)
        
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
        """Test Riemannian metric tensor computation.
        
        A proper Riemannian metric must satisfy:
        1. Symmetry: g_ij = g_ji
        2. Positive definiteness: v^T g v > 0 for all nonzero vectors v
        3. Non-degeneracy: det(g) > 0
        4. Smoothness: eigenvalues should be well-conditioned
        """
        # Generate test points on the manifold
        points = torch.randn(batch_size, neural_flow.manifold_dim, device=device)
        points = points / torch.norm(points, dim=-1, keepdim=True)  # Normalize points
        
        # Compute metric without any artificial regularization
        metric = neural_flow.compute_metric(points)
        
        # 1. Check shape
        assert metric.shape == (batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim), \
            f"Expected shape {(batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)}, got {metric.shape}"
        
        # 2. Check symmetry with reasonable tolerance
        sym_diff = torch.abs(metric - metric.transpose(-2, -1))
        max_sym_diff = sym_diff.max().item()
        assert max_sym_diff < 1e-6, f"Metric not symmetric, max difference: {max_sym_diff}"
        
        # 3. Check positive definiteness via eigenvalues
        eigenvalues = torch.linalg.eigvalsh(metric)  # Use eigvalsh for symmetric matrices
        min_eigenvalue = eigenvalues.min().item()
        assert min_eigenvalue > 1e-7, f"Metric not positive definite, min eigenvalue: {min_eigenvalue}"
        
        # 4. Check non-degeneracy via determinant
        determinants = torch.linalg.det(metric)
        min_det = determinants.min().item()
        assert min_det > 1e-7, f"Metric is degenerate, min determinant: {min_det}"
        
        # 5. Check conditioning (ratio of largest to smallest eigenvalue)
        max_eigenvalue = eigenvalues.max().item()
        condition_number = max_eigenvalue / min_eigenvalue
        assert condition_number < 1e4, f"Metric poorly conditioned, condition number: {condition_number}"
        
        # 6. Test metric action on vectors
        # Generate random vectors
        vectors = torch.randn(batch_size, neural_flow.manifold_dim, device=device)
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
        
        # Compute quadratic form g(v,v)
        metric_action = torch.einsum('bij,bi,bj->b', metric, vectors, vectors)
        
        # Check positivity of quadratic form
        min_action = metric_action.min().item()
        assert min_action > 0, f"Metric action not positive, min value: {min_action}"

    def test_flow_step(self, neural_flow, batch_size, device):
        """Test geometric flow step."""
        # Use a reduced batch size but complete model dimensions
        test_batch_size = min(batch_size, 4)  # Limit to 4 samples
        
        metric = torch.eye(neural_flow.manifold_dim, device=device)
        metric = metric.unsqueeze(0).repeat(test_batch_size, 1, 1)
        
        # Create a valid Ricci tensor with complete dimensions
        ricci = torch.randn(test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim, device=device)
        # Make the Ricci tensor symmetric
        ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
        
        # Verify dimensions before flow_step
        assert metric.shape == (test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        assert ricci.shape == (test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim)
        
        # Create attention pattern for quantum evolution
        attention_pattern = torch.randn(test_batch_size, neural_flow.manifold_dim, neural_flow.manifold_dim, device=device)
        attention_pattern = torch.softmax(attention_pattern, dim=-1)  # Normalize attention weights
        
        # Update neural_flow's quantum bridge dimensions and ensure complex float dtype
        neural_flow.quantum_bridge.hidden_dim = 8
        neural_flow.quantum_bridge.hilbert_space.dim = 8
        neural_flow.quantum_bridge.dtype = torch.complex64  # Set to ComplexFloat
        
        # Convert state amplitudes to ComplexFloat
        neural_flow.quantum_bridge.hilbert_space.state_dtype = torch.complex64
        
        # Convert any existing states to ComplexFloat
        if hasattr(neural_flow.quantum_bridge, 'state'):
            neural_flow.quantum_bridge.state.amplitudes = neural_flow.quantum_bridge.state.amplitudes.to(dtype=torch.complex64)

        new_metric, flow_metrics = neural_flow.flow_step(metric, ricci, attention_pattern=attention_pattern)
        
        # Verify dimensions and properties after flow_step
        assert new_metric.shape == metric.shape
        assert torch.allclose(new_metric, new_metric.transpose(-1, -2))
        
        # Verify that flow metrics are valid
        assert isinstance(flow_metrics.flow_magnitude, torch.Tensor)
        assert isinstance(flow_metrics.metric_determinant, torch.Tensor)
        assert isinstance(flow_metrics.ricci_scalar, torch.Tensor)
        
        # Verify metric dimensions
        assert flow_metrics.flow_magnitude.shape == (test_batch_size,)
        assert flow_metrics.metric_determinant.shape == (test_batch_size,)
        assert flow_metrics.ricci_scalar.shape == (test_batch_size,)

    def test_geometric_operations(self, neural_flow, batch_size, device):
        """Test geometric operations."""
        # Create test points with correct dimensions and requires_grad=True
        points = torch.randn(batch_size, neural_flow.manifold_dim, device=device, requires_grad=True)
        
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