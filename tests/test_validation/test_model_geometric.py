"""Tests for geometric model validation."""

import pytest
import torch
from torch import Tensor
from torch import nn
from typing import Optional, Protocol, runtime_checkable, cast

from src.core.patterns import (
    BaseRiemannianStructure,
    RiemannianFramework,
    PatternRiemannianStructure,
    MetricTensor,
    ChristoffelSymbols,
    CurvatureTensor,
)

from src.core.models.base import LayerGeometry, ModelGeometry
from src.validation.geometric.model import ModelGeometricValidator, ValidationResult


# Type aliases
Points = Tensor  # Shape: (batch_size, dim)
MetricTensor = Tensor  # Shape: (..., dim, dim)
Scores = Tensor  # Shape: (batch_size, batch_size)


class MockLayer(LayerGeometry):
    """Mock layer with Riemannian structure."""
    
    def __init__(self, manifold_dim: int = 16, pattern_dim: Optional[int] = None) -> None:
        super().__init__(manifold_dim, pattern_dim)
        self.riemannian_framework = PatternRiemannianStructure(
            manifold_dim=manifold_dim,
            pattern_dim=pattern_dim if pattern_dim is not None else manifold_dim
        )
        
        # Initialize with small metric factors for bounded energy
        with torch.no_grad():
            self.metric_tensor.data *= 0.01
            
    def metric(self, points: Points) -> MetricTensor:
        """Compute metric tensor."""
        batch_size = points.shape[0]
        return self.metric_tensor.expand(batch_size, -1, -1)
        
    def get_riemannian_framework(self, points: Points) -> RiemannianFramework:
        """Get Riemannian framework."""
        return self.riemannian_framework


@runtime_checkable
class AttentionHead(Protocol):
    """Protocol for attention heads."""
    
    query_dim: int
    key_dim: int
    
    def compute_attention(self, query_points: Points, key_points: Points) -> Scores:
        """Compute attention scores.
        
        Args:
            query_points: Query points tensor
            key_points: Key points tensor
            
        Returns:
            Attention scores tensor
        """
        ...


class MockAttentionHead(nn.Module):
    """Mock attention head."""
    
    def __init__(self, query_dim: int = 16, key_dim: int = 16) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        
        # Initialize query/key projections
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        
        # Initialize with small weights
        with torch.no_grad():
            self.query_proj.weight.data *= 0.1
            self.key_proj.weight.data *= 0.1
            
    def compute_attention(self, query_points: Points, key_points: Points) -> Scores:
        """Compute attention scores."""
        # Project points
        query = self.query_proj(query_points)
        key = self.key_proj(key_points)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))
        return torch.softmax(scores / scores.size(-1)**0.5, dim=-1)


class MockModelGeometry(ModelGeometry):
    """Mock model geometry."""
    
    def __init__(self) -> None:
        super().__init__(
            manifold_dim=16,
            query_dim=16,
            key_dim=16,
            layers={
                'input': MockLayer(16),
                'hidden': MockLayer(16),
                'output': MockLayer(16)
            },
            attention_heads=[MockAttentionHead(16, 16)]
        )


@pytest.fixture
def batch_size() -> int:
    """Batch size for testing."""
    return 16


@pytest.fixture
def mock_model() -> ModelGeometry:
    """Create mock model geometry."""
    return MockModelGeometry()


@pytest.fixture
def validator(mock_model: ModelGeometry) -> ModelGeometricValidator:
    """Create geometric validator."""
    return ModelGeometricValidator(
        model_geometry=mock_model,
        tolerance=1e-6,
        curvature_bounds=(-1.0, 1.0)
    )


class TestModelGeometricValidator:
    """Tests for model geometric validation."""
    
    def test_validate_layer_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test layer geometry validation."""
        # Generate random points
        points = torch.randn(batch_size, validator.model_geometry.manifold_dim)
        
        # Validate input layer
        result = validator.validate_layer_geometry('input', points)
        assert result.is_valid, f"Layer validation failed: {result.message}"
        
        # Check metric properties
        assert result.data is not None, "Validation data is None"
        assert 'metric_tensor' in result.data, "Metric tensor not found in validation data"
        metric = cast(Tensor, result.data['metric_tensor'])
        assert isinstance(metric, torch.Tensor)
        assert metric.shape == (batch_size, points.size(1), points.size(1))
        assert torch.allclose(metric, metric.transpose(-2, -1))
        
        # Check eigenvalues
        eigenvals = torch.linalg.eigvalsh(metric)
        assert torch.all(eigenvals > 0), "Metric has non-positive eigenvalues"
        
    def test_validate_attention_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test attention geometry validation."""
        # Generate query and key points
        query_points = torch.randn(batch_size, validator.model_geometry.query_dim)
        key_points = torch.randn(batch_size, validator.model_geometry.key_dim)
        
        # Validate attention geometry
        result = validator.validate_attention_geometry(0, query_points, key_points)
        assert result.data is not None, "Validation data is None"
        
        # Print debug info
        print("\nDistance Statistics:")
        print("Query distances:")
        assert 'query_metric' in result.data, "Query metric not found in validation data"
        query_metric = cast(Tensor, result.data['query_metric'])
        query_dists = validator._compute_pairwise_distances(
            query_points, 
            query_metric
        )
        print(f"  Range: [{query_dists.min():.6f}, {query_dists.max():.6f}]")
        print(f"  Mean: {query_dists.mean():.6f}")
        print(f"  Std: {query_dists.std():.6f}\n")
        
        print("Key distances:")
        assert 'key_metric' in result.data, "Key metric not found in validation data"
        key_metric = cast(Tensor, result.data['key_metric'])
        key_dists = validator._compute_pairwise_distances(
            key_points, 
            key_metric
        )
        print(f"  Range: [{key_dists.min():.6f}, {key_dists.max():.6f}]")
        print(f"  Mean: {key_dists.mean():.6f}")
        print(f"  Std: {key_dists.std():.6f}\n")
        
        print("Score distances:")
        assert 'attention_scores' in result.data, "Attention scores not found in validation data"
        scores = cast(Tensor, result.data['attention_scores'])
        score_dists = validator._compute_pairwise_distances(
            scores, 
            torch.eye(scores.size(-1), device=scores.device)
        )
        print(f"  Range: [{score_dists.min():.6f}, {score_dists.max():.6f}]")
        print(f"  Mean: {score_dists.mean():.6f}")
        print(f"  Std: {score_dists.std():.6f}\n")
        
        # Check basic properties
        assert isinstance(result.data['query_metric'], torch.Tensor)
        assert isinstance(result.data['key_metric'], torch.Tensor)
        assert isinstance(result.data['attention_scores'], torch.Tensor)
        
        # Check attention score properties
        scores = cast(Tensor, result.data['attention_scores'])
        assert torch.allclose(scores.sum(dim=-1), torch.ones_like(scores[:, 0]))
        assert torch.all(scores >= 0)
        
        # Check geometric preservation
        assert 'preserves_geometry' in result.data, "Geometric preservation result not found"
        assert result.data['preserves_geometry'], (
            "Attention does not preserve geometric structure"
        )
        
    def test_geometric_preservation(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test geometric preservation check."""
        # Generate random points and perturbation
        points = torch.randn(batch_size, validator.model_geometry.manifold_dim)
        perturbation = torch.randn_like(points) * 0.1
        
        # Get base metric
        layer = validator.model_geometry.layers['hidden']
        base_metric = cast(Tensor, layer.metric_tensor.data)
        
        # Print metric properties
        print("\nBase Metric Properties:")
        print(f"Shape: {base_metric.shape}")
        print(f"Symmetric: {torch.allclose(cast(Tensor, base_metric), cast(Tensor, base_metric.transpose(-2, -1)))}")
        eigenvals = torch.linalg.eigvalsh(base_metric)
        print(f"Eigenvalue range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]\n")
        
        # Compute attention scores
        head = cast(AttentionHead, validator.model_geometry.attention_heads[0])
        scores = head.compute_attention(points, points + perturbation)
        
        # Print score properties
        print("Attention Scores Properties:")
        print(f"Shape: {scores.shape}")
        print(f"Range: [{scores.min():.6f}, {scores.max():.6f}]")
        print(f"Row sums: {scores.sum(dim=-1)}\n")
        
        # Check geometric preservation
        preserves = validator._check_geometric_preservation(points, base_metric, scores)
        assert preserves, "Attention does not preserve geometric structure"

    def test_geometric_preservation_distances(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test distance properties in geometric preservation."""
        # Generate points
        points = torch.randn(batch_size, validator.model_geometry.manifold_dim)
        
        # Compute metrics and scores
        layer = validator.model_geometry.layers['hidden']
        base_metric = cast(Tensor, layer.metric_tensor.data)
        head = cast(AttentionHead, validator.model_geometry.attention_heads[0])
        scores = head.compute_attention(points, points)
        
        # Compute distances
        query_distances = validator._compute_pairwise_distances(points, base_metric)
        key_distances = validator._compute_pairwise_distances(points, base_metric)
        score_distances = validator._compute_pairwise_distances(
            scores,
            torch.eye(scores.size(-1), device=scores.device)
        )

        # Check distance properties
        for name, distances in [
            ('Query', query_distances),
            ('Key', key_distances),
            ('Score', score_distances)
        ]:
            # Non-negativity
            assert torch.all(distances >= 0), f"{name} distances contain negative values"
            
            # Self-distances are zero
            assert torch.allclose(
                torch.diagonal(distances),
                torch.zeros(batch_size),
                atol=1e-6
            ), f"{name} self-distances are non-zero"
            
            # Symmetry
            assert torch.allclose(
                distances,
                distances.transpose(-2, -1),
                atol=1e-6
            ), f"{name} distances are not symmetric"
            
            # Normalization
            assert torch.all(distances <= 1.0), f"{name} distances exceed 1.0"
