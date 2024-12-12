"""Tests for model-specific geometric validation."""

import pytest
import torch
import torch.nn as nn

from src.validation.geometric.model import ModelGeometricValidator
from src.core.models.base import ModelGeometry, LayerGeometry
from src.validation.base import ValidationResult


class MockLayer(LayerGeometry):
    def __init__(self, manifold_dim: int):
        super().__init__(manifold_dim)
        
    def metric(self, points: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.manifold_dim).expand(points.shape[0], -1, -1)
        
    def connection(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros(points.shape[0], self.manifold_dim, self.manifold_dim, self.manifold_dim)


class MockAttentionHead(nn.Module):
    """Mock attention head for testing."""
    
    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        
    def query_metric(self, points: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.query_dim).expand(points.shape[0], -1, -1)
        
    def key_metric(self, points: torch.Tensor) -> torch.Tensor:
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


class MockModelGeometry(ModelGeometry):
    def __init__(self):
        super().__init__(
            manifold_dim=16,
            query_dim=16,
            key_dim=16,
            layers={
                'input': MockLayer(16),
                'hidden': MockLayer(16),
                'output': MockLayer(16),
                'query_0': MockLayer(16),
                'key_0': MockLayer(16)
            },
            attention_heads=[
                MockAttentionHead(16, 16)
            ]
        )


class TestModelGeometricValidator:
    @pytest.fixture
    def model_geometry(self) -> ModelGeometry:
        return MockModelGeometry()
        
    @pytest.fixture
    def validator(self, model_geometry: ModelGeometry) -> ModelGeometricValidator:
        return ModelGeometricValidator(
            model_geometry=model_geometry,
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
        self, validator: ModelGeometricValidator
    ):
        """Test complete model geometry validation."""
        result = validator.validate_model_geometry()
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
