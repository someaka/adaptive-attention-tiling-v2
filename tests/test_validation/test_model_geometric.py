"""Tests for model-specific geometric validation."""

import torch
import torch.nn as nn
import pytest
import logging
from typing import Dict, List, Optional

from src.validation.geometric.model import ModelGeometricValidator, ValidationResult
from src.core.patterns.riemannian import PatternRiemannianStructure


def tensor_repr(tensor: Optional[torch.Tensor], max_elements: int = 8) -> str:
    """Create a shortened string representation of tensors."""
    if tensor is None:
        return "None"
    shape = list(tensor.shape)
    if len(shape) == 0:
        return f"tensor({tensor.item():.4f})"
    if sum(shape) <= max_elements:
        return str(tensor)
    return f"tensor(shape={shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f})"


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result for readable output."""
    data_repr = {}
    for k, v in result.data.items():
        if isinstance(v, dict):
            data_repr[k] = {sk: tensor_repr(sv) for sk, sv in v.items()}
        elif isinstance(v, list):
            data_repr[k] = [tensor_repr(item) for item in v]
        else:
            data_repr[k] = tensor_repr(v)
    return f"ValidationResult(is_valid={result.is_valid}, data={data_repr}, message='{result.message}')"


class MockLayer(nn.Module):
    """Mock layer for testing."""
    
    def __init__(self, manifold_dim: int = 4):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.riemannian_framework = PatternRiemannianStructure(manifold_dim)
        
        # Initialize with small metric factors to ensure bounded energy
        with torch.no_grad():
            self.riemannian_framework.metric_factors.data = (
                self.riemannian_framework.metric_factors.data * 0.01
            )
        
    def metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor."""
        points = points.clone()
        if not points.requires_grad:
            points.requires_grad_(True)
        return self.riemannian_framework.compute_metric(points)
        
    def get_riemannian_framework(self, points: torch.Tensor) -> PatternRiemannianStructure:
        """Get Riemannian framework."""
        points = points.clone()
        if not points.requires_grad:
            points.requires_grad_(True)
        return self.riemannian_framework


class MockAttentionHead(nn.Module):
    """Mock attention head for testing."""
    
    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
    
    def query_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Get query space metric."""
        return torch.eye(self.query_dim).expand(points.shape[0], -1, -1)
    
    def key_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Get key space metric."""
        return torch.eye(self.key_dim).expand(points.shape[0], -1, -1)
        
    def compute_attention(self, query_points: torch.Tensor, key_points: torch.Tensor) -> torch.Tensor:
        """Compute mock attention scores."""
        # Ensure points require gradients
        if not query_points.requires_grad:
            query_points = query_points.detach().requires_grad_(True)
        if not key_points.requires_grad:
            key_points = key_points.detach().requires_grad_(True)
            
        # Compute attention scores [batch_size x num_keys]
        scores = torch.einsum('bd,kd->bk', query_points, key_points)
        scores = torch.softmax(scores, dim=-1)
        return scores


class MockModelGeometry:
    """Mock model geometry for testing."""
    
    def __init__(self):
        self.manifold_dim = 16
        self.query_dim = 16
        self.key_dim = 16
        self.layers = {
            'input': MockLayer(16),
            'hidden': MockLayer(16),
            'output': MockLayer(16),
            'query_0': MockLayer(16),
            'key_0': MockLayer(16)
        }
        self.attention_heads = [
            MockAttentionHead(16, 16)
        ]
        
    def metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor for the model."""
        batch_size = points.shape[0]
        # Return identity metric for testing
        return torch.eye(self.manifold_dim).expand(batch_size, -1, -1)
        
    def sectional_curvature(self, points: torch.Tensor) -> torch.Tensor:
        """Compute sectional curvature."""
        batch_size = points.shape[0]
        # Return zero curvature for testing
        return torch.zeros(batch_size)


class TestModelGeometricValidator:
    @pytest.fixture
    def mock_layer(self) -> MockLayer:
        return MockLayer()
    
    @pytest.fixture
    def validator(self, mock_layer: MockLayer) -> ModelGeometricValidator:
        return ModelGeometricValidator(
            model_geometry=MockModelGeometry(),
            tolerance=1e-6,
            curvature_bounds=(-1.0, 1.0)
        )

    @pytest.fixture
    def batch_size(self) -> int:
        return 16
        
    def test_validate_layer_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test layer geometry validation."""
        points = torch.randn(batch_size, 16)
        
        # Test input layer
        result = validator.validate_layer_geometry('input', points)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        
        # Test hidden layer
        result = validator.validate_layer_geometry('hidden', points)
        assert result.is_valid
        
        # Test invalid layer
        with pytest.raises(ValueError):
            validator.validate_layer_geometry('nonexistent', points)

    def test_validate_attention_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test attention geometry validation."""
        query_points = torch.randn(batch_size, 16)
        key_points = torch.randn(batch_size, 16)
        
        result = validator.validate_attention_geometry(0, query_points, key_points)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        
        # Check results structure
        assert 'query_validation' in result.data
        assert 'key_validation' in result.data
        assert 'compatibility' in result.data

    def test_validate_cross_layer_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test cross-layer geometry validation."""
        points = torch.randn(batch_size, 16)
        
        result = validator.validate_cross_layer_geometry(
            'input', 'hidden', points
        )
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        
        # Check metric compatibility
        assert result.data['metric_compatibility']
        
        # Check connection compatibility
        assert result.data['connection_compatibility']

    def test_validate_model_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test complete model geometry validation."""
        result = validator.validate_model_geometry(batch_size=batch_size)
        
        # For debugging, print shortened result
        logging.info("Validation result: %s", format_validation_result(result))
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        
        # Check validation structure
        assert 'layer_validations' in result.data
        assert 'attention_validations' in result.data
        assert 'global_properties' in result.data
        
        # Check layer results
        layer_results = result.data['layer_validations']
        assert all(r.is_valid for r in layer_results.values())
        
        # Check attention results
        attention_results = result.data['attention_validations']
        assert all(r.is_valid for r in attention_results.values())
        
        # Check global properties
        global_props = result.data['global_properties']
        assert global_props.is_valid
        assert global_props.data['complete']
        assert global_props.data['curvature_valid']
        assert global_props.data['energy_valid']

    def test_geometric_preservation(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test geometric preservation."""
        # Create mock attention head
        head = MockAttentionHead(16, 16)
        
        # Generate test points with gradients
        query_points = torch.randn(batch_size, 16, requires_grad=True)
        key_points = torch.randn(batch_size, 16, requires_grad=True)
        
        # Compute attention scores
        scores = head.compute_attention(query_points, key_points)
        
        # Create mock metrics
        query_metric = torch.eye(16).unsqueeze(0).expand(batch_size, -1, -1).requires_grad_(True)
        key_metric = torch.eye(16).unsqueeze(0).unsqueeze(0).expand(batch_size, scores.shape[1], -1, -1).requires_grad_(True)
        
        # Check geometric preservation
        assert validator._check_geometric_preservation(query_metric, key_metric, scores)

    def test_global_energy(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test global energy bounds."""
        points = torch.randn(batch_size, 16, requires_grad=True)
        
        # Check energy bounds
        energy_valid = validator._check_global_energy(points)
        assert energy_valid

    def test_metric_gradients(self):
        """Test that metric gradients are properly computed."""
        layer = MockLayer(manifold_dim=4)
        points = torch.randn(2, 4, requires_grad=True)
        
        # Compute metric
        metric = layer.metric(points)
        
        # Should be able to compute gradients
        loss = metric.sum()
        grad = torch.autograd.grad(loss, points, allow_unused=True)[0]
        assert grad is not None, "Gradient should not be None"
        
        # Test that metric depends on points
        metric2 = layer.metric(points + 0.1)
        assert not torch.allclose(metric, metric2), "Metric should depend on points"

    def test_validate_model_geometry_resource_guard(self, validator):
        """Test resource guards in model validation."""
        # Test with small batch size and manifold dim
        result = validator.validate_model_geometry(
            batch_size=4, manifold_dim=4, max_memory_gb=1.0
        )
        
        # For debugging, print shortened result
        logging.info("Resource guard test result: %s", format_validation_result(result))
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        
        # Test memory limit exceeded
        with pytest.raises(ValueError, match="Validation would require.*memory"):
            validator.validate_model_geometry(
                batch_size=1000, manifold_dim=1000, max_memory_gb=0.1
            )
