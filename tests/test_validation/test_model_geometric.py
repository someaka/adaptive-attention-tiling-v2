"""Tests for model-specific geometric validation."""

import torch
import torch.nn as nn
import pytest
import logging
from typing import Dict, Protocol, Optional, Tuple, Union, List, Iterator, Any
from dataclasses import dataclass

from src.validation.geometric.model import ValidationResult, ModelGeometricValidator
from src.core.patterns.riemannian import PatternRiemannianStructure


def tensor_repr(value: Any, max_elements: int = 8) -> str:
    """Create a shortened string representation of tensors and other values."""
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        shape = list(value.shape)
        if len(shape) == 0:
            return f"tensor({value.item():.4f})"
        if sum(shape) <= max_elements:
            return str(value)
        return f"tensor(shape={shape}, mean={value.mean():.4f}, std={value.std():.4f})"
    return str(value)


def format_validation_result(result: Union[ValidationResult, Dict[str, ValidationResult]]) -> str:
    """Format validation result for readable output."""
    if isinstance(result, dict):
        # Handle dictionary of validation results
        formatted = {}
        for k, v in result.items():
            formatted[k] = format_validation_result(v)
        return str(formatted)
    
    # Handle single ValidationResult
    data_repr = {}
    for k, v in result.data.items():
        if isinstance(v, dict):
            data_repr[k] = {sk: tensor_repr(sv) for sk, sv in v.items()}
        elif isinstance(v, list):
            data_repr[k] = [tensor_repr(item) for item in v]
        else:
            data_repr[k] = tensor_repr(v)
    return f"ValidationResult(is_valid={result.is_valid}, data={data_repr}, message='{result.message}')"


class ModelGeometry(Protocol):
    """Protocol for model geometry."""
    manifold_dim: int
    query_dim: int
    key_dim: int
    
    def get_layer(self, layer_name: str) -> 'MockLayer':
        """Get layer by name."""
        ...
        
    def metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor."""
        ...
        
    def sectional_curvature(self, points: torch.Tensor) -> torch.Tensor:
        """Compute sectional curvature."""
        ...
        
    def connection(self, points: torch.Tensor) -> torch.Tensor:
        """Compute connection."""
        ...
        
    def parameters(self) -> Iterator[nn.Parameter]:
        """Get model parameters."""
        ...


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
        batch_size = points.shape[0]
        metric = torch.eye(self.query_dim).expand(batch_size, -1, -1)
        # Add a small position-dependent perturbation to make it more interesting
        for i in range(batch_size):
            metric[i] = metric[i] + 0.01 * torch.outer(points[i], points[i])
        return metric
    
    def key_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Get key space metric."""
        batch_size = points.shape[0]
        metric = torch.eye(self.key_dim).expand(batch_size, -1, -1)
        # Add a small position-dependent perturbation to make it more interesting
        for i in range(batch_size):
            metric[i] = metric[i] + 0.01 * torch.outer(points[i], points[i])
        return metric
        
    def compute_attention(self, query_points: torch.Tensor, key_points: torch.Tensor) -> torch.Tensor:
        """Compute mock attention scores."""
        # Ensure points require gradients
        if not query_points.requires_grad:
            query_points = query_points.detach().requires_grad_(True)
        if not key_points.requires_grad:
            key_points = key_points.detach().requires_grad_(True)
            
        # Compute attention scores using the metric
        query_metric = self.query_metric(query_points)
        key_metric = self.key_metric(key_points)
        
        # Compute distances using both metrics
        scores = torch.zeros(query_points.shape[0], key_points.shape[0], device=query_points.device)
        for i in range(query_points.shape[0]):
            for j in range(key_points.shape[0]):
                query_diff = query_points[i] - key_points[j]
                key_diff = key_points[j] - query_points[i]
                query_dist = torch.sqrt((query_diff @ query_metric[i] @ query_diff).sum())
                key_dist = torch.sqrt((key_diff @ key_metric[j] @ key_diff).sum())
                # Use average distance
                scores[i,j] = -0.5 * (query_dist + key_dist)
        
        # Apply softmax
        scores = torch.softmax(scores, dim=-1)
        return scores


class MockModelGeometry(ModelGeometry):
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
        return self.layers['input'].metric(points)
        
    def sectional_curvature(self, points: torch.Tensor) -> torch.Tensor:
        """Compute sectional curvature."""
        return torch.zeros_like(points)
        
    def get_layer(self, layer_name: str) -> MockLayer:
        """Get layer by name."""
        return self.layers[layer_name]
        
    def connection(self, points: torch.Tensor) -> torch.Tensor:
        """Compute connection."""
        return self.layers['input'].riemannian_framework.compute_christoffel(points)
        
    def parameters(self) -> Iterator[nn.Parameter]:
        """Get model parameters."""
        for layer in self.layers.values():
            yield from layer.riemannian_framework.parameters()


class TestModelGeometricValidator:
    @pytest.fixture
    def mock_layer(self) -> MockLayer:
        return MockLayer()
    
    @pytest.fixture
    def validator(self, mock_layer: MockLayer) -> ModelGeometricValidator:
        """Create validator fixture."""
        model_geometry = MockModelGeometry()
        return ModelGeometricValidator(model_geometry=model_geometry)  # type: ignore
    
    @pytest.fixture
    def batch_size(self) -> int:
        return 16
        
    def test_validate_layer_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test layer geometry validation."""
        points = torch.randn(batch_size, 16, requires_grad=True)
        
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
        # Create mock attention head
        head = MockAttentionHead(16, 16)
        
        # Generate test points with gradients
        query_points = torch.randn(batch_size, 16, requires_grad=True)
        key_points = torch.randn(batch_size, 16, requires_grad=True)
        
        # Get validation result
        result = validator.validate_attention_geometry(0, query_points, key_points)
        
        # Debug logging
        print("\nAttention Geometry Debug:")
        for key, value in result.data.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, value={value}")
            else:
                print(f"{key}: {value}")
        
        # Check eigenvalues
        query_eigenvals = torch.linalg.eigvalsh(result.data['query_metric'])
        key_eigenvals = torch.linalg.eigvalsh(result.data['key_metric'])
        print(f"\nQuery Eigenvalues: min={query_eigenvals.min().item():.6f}, max={query_eigenvals.max().item():.6f}")
        print(f"Key Eigenvalues: min={key_eigenvals.min().item():.6f}, max={key_eigenvals.max().item():.6f}")
        
        assert result.is_valid

    def test_validate_cross_layer_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ):
        """Test cross-layer geometry validation."""
        points = torch.randn(batch_size, 16, requires_grad=True)
        
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
        
        # Check result type
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert isinstance(result.data, dict)
        
        # Check layer validations
        assert 'layers' in result.data
        layer_results = result.data.get('layers', {})
        assert isinstance(layer_results, dict)
        
        # Check required layers exist
        required_layers = {'input', 'hidden', 'output'}
        assert all(layer in layer_results for layer in required_layers)
        
        # Check validation results
        for layer_name, validation in layer_results.items():
            assert isinstance(validation, ValidationResult)
            assert validation.is_valid
            assert isinstance(validation.data, dict)
            assert 'complete' in validation.data
            assert 'curvature_valid' in validation.data
            assert 'energy_valid' in validation.data

    def test_geometric_preservation(self, validator: ModelGeometricValidator, batch_size: int):
        """Test geometric preservation check."""
        head = MockAttentionHead(query_dim=batch_size, key_dim=batch_size)
        
        # Create base metric that's symmetric positive definite with controlled eigenvalues
        base_metric = torch.eye(batch_size)
        # Add small random perturbation to make it interesting but still well-conditioned
        perturbation = torch.randn(batch_size, batch_size) * 0.1
        base_metric = base_metric + perturbation @ perturbation.T
        
        # Create query and key metrics with similar structure
        query_metric = base_metric.unsqueeze(0).expand(batch_size, -1, -1).clone()
        key_metric = base_metric.unsqueeze(0).unsqueeze(0).expand(batch_size, batch_size, -1, -1).clone()
        
        # Create attention scores that preserve the geometric structure
        # Use cosine similarity between random vectors to ensure scores are related to geometry
        points = torch.randn(batch_size, batch_size)
        points = points / points.norm(dim=1, keepdim=True)
        scores = torch.mm(points, points.T)
        
        # Debug logging
        print("\nGeometric Preservation Debug:")
        print(f"Query Metric Shape: {query_metric.shape}")
        print(f"Query Metric Eigenvalues: {torch.linalg.eigvalsh(query_metric[0])}")
        print(f"Key Metric Shape: {key_metric.shape}")
        print(f"Key Metric Eigenvalues: {torch.linalg.eigvalsh(key_metric[0,0])}")
        print(f"Scores Shape: {scores.shape}")
        
        # Call the validator
        result = validator._check_geometric_preservation(query_metric, key_metric, scores)
        
        # Print intermediate results
        print("\nIntermediate Values:")
        print(f"Query Distances: {validator.query_distances}")
        print(f"Key Distances: {validator.key_distances}")
        print(f"Score Distances: {validator.score_distances}")
        print(f"\nDifferences:")
        print(f"Query-Key Diff: {validator.query_key_diff}")
        print(f"Query-Score Diff: {validator.query_score_diff}")
        print(f"Key-Score Diff: {validator.key_score_diff}")
        
        assert result

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
