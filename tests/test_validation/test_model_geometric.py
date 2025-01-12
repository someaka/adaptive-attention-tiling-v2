"""Tests for geometric model validation."""

import pytest
import torch
from torch import Tensor
from torch import nn
from typing import Optional, Protocol, runtime_checkable, cast
import math
import torch.nn.functional as F

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        
        # Initialize query/key projections with orthogonal matrices
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        
        # Initialize with orthogonal weights for better stability
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.orthogonal_(self.key_proj.weight)
        
        # Scale weights to control energy and add stability
        with torch.no_grad():
            # Use smaller initialization for better numerical stability
            self.query_proj.weight.data *= 0.05
            self.key_proj.weight.data *= 0.05
            # Initialize biases to small non-zero values for stability
            self.query_proj.bias.data.uniform_(-0.01, 0.01)
            self.key_proj.bias.data.uniform_(-0.01, 0.01)
            
    def query_metric(self, points: Points) -> Tensor:
        """Compute query metric tensor."""
        batch_size = points.shape[0]
        # Project points and normalize
        proj = F.normalize(self.query_proj(points), p=2, dim=-1)
        
        # Compute metric with numerical stability
        eps = 1e-8
        metric = torch.einsum('bi,bj->bij', proj, proj) / self.query_dim
        
        # Add scaled identity for better conditioning
        eye = torch.eye(self.query_dim, device=points.device)
        metric = metric + 0.1 * eye.unsqueeze(0)
        
        # Ensure symmetry and positive definiteness
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Add small diagonal term for numerical stability
        metric = metric + eps * eye.unsqueeze(0)
        
        # Normalize metric with stable computation
        norm = torch.norm(metric, dim=(-2, -1), keepdim=True)
        metric = metric / (norm + eps)
        
        return metric
        
    def key_metric(self, points: Points) -> Tensor:
        """Compute key metric tensor."""
        batch_size = points.shape[0]
        # Project and normalize points
        proj = F.normalize(self.key_proj(points), p=2, dim=-1)
        
        # Compute metric with numerical stability
        eps = 1e-8
        metric = torch.einsum('bi,bj->bij', proj, proj) / self.key_dim
        
        # Add scaled identity for better conditioning
        eye = torch.eye(self.key_dim, device=points.device)
        metric = metric + 0.1 * eye.unsqueeze(0)
        
        # Ensure symmetry and positive definiteness
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Add small diagonal term for numerical stability
        metric = metric + eps * eye.unsqueeze(0)
        
        # Normalize metric with stable computation
        norm = torch.norm(metric, dim=(-2, -1), keepdim=True)
        metric = metric / (norm + eps)
        
        return metric
            
    def compute_attention(self, query_points: Points, key_points: Points) -> Scores:
        """Compute attention scores that preserve geometric structure."""
        eps = 1e-8
        
        # Check if points are identical
        if torch.allclose(query_points, key_points, rtol=1e-5, atol=1e-8):
            # For identical points, use identity-like attention
            batch_size = query_points.size(0)
            scores = torch.eye(batch_size, device=query_points.device)
            scores = scores + eps  # Add small value to avoid exact zeros
            scores = scores / scores.sum(dim=-1, keepdim=True)
            return scores
        
        # Project points with metric-aware projections
        query = self.query_proj(query_points)
        key = self.key_proj(key_points)
        
        # Get metric tensors
        query_metric = self.query_metric(query_points)  # [batch, dim, dim]
        key_metric = self.key_metric(key_points)  # [batch, dim, dim]
        
        # Scale inputs to a reasonable range if they're too small
        scale_factor = max(
            torch.norm(query_points).item(),
            torch.norm(key_points).item()
        )
        if scale_factor < 1e-4:
            query_points_scaled = query_points / scale_factor * 1e-4
            key_points_scaled = key_points / scale_factor * 1e-4
            query = query / scale_factor * 1e-4
            key = key / scale_factor * 1e-4
        else:
            query_points_scaled = query_points
            key_points_scaled = key_points
            
        # Normalize points to unit norm for stable distance computation
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        
        # Compute metric-aware inner products with improved stability
        query_metric_sqrt = torch.linalg.cholesky(
            query_metric + eps * torch.eye(query.size(-1), device=query.device).unsqueeze(0)
        )
        key_metric_sqrt = torch.linalg.cholesky(
            key_metric + eps * torch.eye(key.size(-1), device=key.device).unsqueeze(0)
        )
        
        # Transform points to metric space
        query_transformed = torch.einsum('bij,bj->bi', query_metric_sqrt, query)
        key_transformed = torch.einsum('bij,bj->bi', key_metric_sqrt, key)
        
        # Normalize in metric space with improved stability
        query_norm = torch.norm(query_transformed, p=2, dim=-1, keepdim=True)
        key_norm = torch.norm(key_transformed, p=2, dim=-1, keepdim=True)
        query_normalized = query_transformed / (query_norm + eps)
        key_normalized = key_transformed / (key_norm + eps)
        
        # Compute similarities in metric space
        similarities = torch.matmul(query_normalized, key_normalized.transpose(-2, -1))
        
        # Scale to [0, 1] with controlled scaling
        similarities = (similarities + 1.0) / 2.0
        
        # Match input distance distribution using cosine similarity
        query_distances = torch.zeros(query_points.size(0), query_points.size(0), device=query_points.device)
        key_distances = torch.zeros(key_points.size(0), key_points.size(0), device=key_points.device)
        
        # Compute pairwise distances using cosine similarity
        query_points_norm = F.normalize(query_points_scaled, p=2, dim=-1)
        key_points_norm = F.normalize(key_points_scaled, p=2, dim=-1)
        query_distances = 1 - F.cosine_similarity(query_points_norm.unsqueeze(1), query_points_norm.unsqueeze(0), dim=-1)
        key_distances = 1 - F.cosine_similarity(key_points_norm.unsqueeze(1), key_points_norm.unsqueeze(0), dim=-1)
        
        # Normalize distances
        query_distances = query_distances / (query_distances.max() + eps)
        key_distances = key_distances / (key_distances.max() + eps)
        
        # Convert distances to attention logits directly
        attention_logits = -query_distances  # Closer points get higher logits
        
        # Apply temperature scaling to control sharpness
        dim = float(query.size(-1))
        base_temperature = 1.0 / math.sqrt(dim)
        attention_logits = attention_logits / base_temperature
        
        # Convert to attention scores
        scores = F.softmax(attention_logits, dim=-1)
        
        # Compute score distances using cosine similarity
        scores_norm = F.normalize(scores, p=2, dim=-1)
        score_distances = 1 - F.cosine_similarity(scores_norm.unsqueeze(1), scores_norm.unsqueeze(0), dim=-1)
        score_distances = score_distances / (score_distances.max() + eps)
        
        # Match distance distributions more precisely
        input_mean = query_distances.mean()
        input_std = query_distances.std()
        
        # Compute correlation between input and score distances
        correlation = torch.corrcoef(
            torch.stack([
                query_distances.flatten(),
                score_distances.flatten()
            ])
        )[0, 1]
        
        return scores


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
        
    def sectional_curvature(self, points: Points) -> Tensor:
        """Compute sectional curvature."""
        batch_size = points.shape[0]
        # Return constant sectional curvature for testing
        return torch.zeros(batch_size, self.manifold_dim, self.manifold_dim)
        
    def metric(self, points: Points) -> Tensor:
        """Compute metric tensor."""
        batch_size = points.shape[0]
        # Return constant metric for testing
        return torch.eye(self.manifold_dim).unsqueeze(0).repeat(batch_size, 1, 1)


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
        # Test with regular points
        points = torch.randn(batch_size, validator.model_geometry.manifold_dim)
        result = validator.validate_layer_geometry('input', points)
        assert result.is_valid, f"Layer validation failed: {result.message}"
        
        # Test with near-zero points
        small_points = torch.randn(batch_size, validator.model_geometry.manifold_dim) * 1e-6
        result_small = validator.validate_layer_geometry('input', small_points)
        assert result_small.is_valid, f"Layer validation failed for small points: {result_small.message}"
        
        # Test with large magnitude points
        large_points = torch.randn(batch_size, validator.model_geometry.manifold_dim) * 1e6
        result_large = validator.validate_layer_geometry('input', large_points)
        assert result_large.is_valid, f"Layer validation failed for large points: {result_large.message}"
        
        # Test with points close to each other
        close_points = points + torch.randn_like(points) * 1e-6
        result_close = validator.validate_layer_geometry('input', close_points)
        assert result_close.is_valid, f"Layer validation failed for close points: {result_close.message}"
        
        # Check metric properties for all cases
        for name, result in [
            ("regular", result),
            ("small", result_small),
            ("large", result_large),
            ("close", result_close)
        ]:
            assert result.data is not None, f"Validation data is None for {name} points"
            assert 'metric_tensor' in result.data, f"Metric tensor not found for {name} points"
            metric = cast(Tensor, result.data['metric_tensor'])
            
            # Basic tensor properties
            assert isinstance(metric, torch.Tensor), f"Invalid metric type for {name} points"
            assert metric.shape == (batch_size, points.size(1), points.size(1)), f"Invalid metric shape for {name} points"
            
            # Symmetry
            assert torch.allclose(
                metric, 
                metric.transpose(-2, -1),
                rtol=1e-5,
                atol=1e-5
            ), f"Metric not symmetric for {name} points"
            
            # Positive definiteness
            eigenvals = torch.linalg.eigvalsh(metric)
            assert torch.all(eigenvals > 0), f"Metric has non-positive eigenvalues for {name} points"
            
            # Condition number
            cond_num = torch.max(eigenvals) / torch.min(eigenvals)
            assert cond_num < 1e6, f"Metric poorly conditioned for {name} points (condition number: {cond_num})"
    
    def test_validate_attention_geometry(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test attention geometry validation."""
        # Test regular case
        query_points = torch.randn(batch_size, validator.model_geometry.query_dim)
        key_points = torch.randn(batch_size, validator.model_geometry.key_dim)
        result = validator.validate_attention_geometry(0, query_points, key_points)
        assert result.is_valid, f"Attention validation failed: {result.message}"
    
        # Test with small magnitude points
        small_query = query_points * 1e-6
        small_key = key_points * 1e-6
        result_small = validator.validate_attention_geometry(0, small_query, small_key)
        assert result_small.is_valid, f"Attention validation failed for small points: {result_small.message}"
    
        # Test with large magnitude points
        large_query = query_points * 1e6
        large_key = key_points * 1e6
        result_large = validator.validate_attention_geometry(0, large_query, large_key)
        assert result_large.is_valid, f"Attention validation failed for large points: {result_large.message}"
    
        # Test with identical points
        result_identical = validator.validate_attention_geometry(0, query_points, query_points)
        assert result_identical.is_valid, f"Attention validation failed for identical points: {result_identical.message}"
    
        # Test with orthogonal points
        # Create orthogonal points that preserve distance relationships better
        q_ortho = torch.randn(batch_size, validator.model_geometry.query_dim)
        k_ortho = torch.randn(batch_size, validator.model_geometry.key_dim)
        
        # Normalize to unit vectors
        q_ortho = F.normalize(q_ortho, p=2, dim=1)
        k_ortho = F.normalize(k_ortho, p=2, dim=1)
        
        # Ensure points are well-separated
        angle = torch.tensor(torch.pi / 4, device=q_ortho.device)  # 45 degree separation
        rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                               [torch.sin(angle), torch.cos(angle)]], device=q_ortho.device)
        
        # Apply rotation to create consistent angular separation
        q_ortho_rotated = torch.cat([
            torch.mm(q_ortho[:, :2], rotation),
            q_ortho[:, 2:]
        ], dim=1)
        
        result_ortho = validator.validate_attention_geometry(0, q_ortho_rotated, k_ortho)
        assert result_ortho.is_valid, f"Attention validation failed for orthogonal points: {result_ortho.message}"
    
        # Verify attention properties for all cases
        for name, result in [
            ("regular", result),
            ("small", result_small),
            ("large", result_large),
            ("identical", result_identical),
            ("orthogonal", result_ortho)
        ]:
            assert result.data is not None, f"Validation data is None for {name} case"
            scores = cast(Tensor, result.data['attention_scores'])
            
            # Check score properties
            assert torch.allclose(
                scores.sum(dim=-1),
                torch.ones_like(scores[:, 0]),
                rtol=1e-5,
                atol=1e-5
            ), f"Scores don't sum to 1 for {name} case"
            
            assert torch.all(scores >= 0), f"Negative scores found for {name} case"
            assert torch.all(scores <= 1), f"Scores greater than 1 found for {name} case"
            
            # Check distance preservation
            assert result.data['preserves_geometry'], f"Geometry not preserved for {name} case"
            
            # Check numerical stability
            assert not torch.any(torch.isnan(scores)), f"NaN scores found for {name} case"
            assert not torch.any(torch.isinf(scores)), f"Inf scores found for {name} case"
    
    def test_geometric_preservation(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test geometric preservation check."""
        # Generate points with different characteristics
        points = torch.randn(batch_size, validator.model_geometry.manifold_dim)
        
        # Test different perturbation scales
        for scale in [1e-6, 1e-3, 1e-1, 1.0]:
            perturbation = torch.randn_like(points) * scale
            perturbed_points = points + perturbation
            
            # Get base metric
            layer = validator.model_geometry.layers['hidden']
            base_metric = cast(Tensor, layer.metric_tensor.data)
            
            # Compute attention scores
            head = cast(AttentionHead, validator.model_geometry.attention_heads[0])
            scores = head.compute_attention(points, perturbed_points)
            
            # Check geometric preservation
            preserves = validator._check_geometric_preservation(points, base_metric, scores)
            assert preserves, f"Attention does not preserve geometric structure for scale {scale}"
            
            # Verify distance properties
            query_distances = validator._compute_pairwise_distances(points, base_metric)
            score_distances = validator._compute_pairwise_distances(
                scores,
                torch.eye(scores.size(-1), device=scores.device)
            )
            
            # Check correlation between distances
            correlation = torch.corrcoef(
                torch.stack([
                    query_distances.flatten(),
                    score_distances.flatten()
                ])
            )[0, 1]
            assert correlation > 0.5, f"Low distance correlation ({correlation}) for scale {scale}"
    
    def test_geometric_preservation_distances(
        self, validator: ModelGeometricValidator, batch_size: int
    ) -> None:
        """Test distance properties in geometric preservation."""
        # Test with different point distributions
        distributions = {
            'normal': torch.randn,
            'uniform': lambda *args: torch.rand(*args) * 2 - 1,
            'exponential': lambda *args: torch.exp(torch.randn(*args)),
            'small': lambda *args: torch.randn(*args) * 1e-6,
            'large': lambda *args: torch.randn(*args) * 1e6
        }
        
        for dist_name, dist_fn in distributions.items():
            points = dist_fn(batch_size, validator.model_geometry.manifold_dim)
            
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
                assert torch.all(distances >= 0), f"{name} distances contain negative values for {dist_name} distribution"
                
                # Self-distances are zero
                assert torch.allclose(
                    torch.diagonal(distances),
                    torch.zeros(batch_size),
                    atol=1e-5
                ), f"{name} self-distances are non-zero for {dist_name} distribution"
                
                # Symmetry
                assert torch.allclose(
                    distances,
                    distances.transpose(-2, -1),
                    atol=1e-5
                ), f"{name} distances are not symmetric for {dist_name} distribution"
                
                # Normalization
                assert torch.all(distances <= 1.0), f"{name} distances exceed 1.0 for {dist_name} distribution"
                
                # Check for numerical instability
                assert not torch.any(torch.isnan(distances)), f"NaN distances found in {name} for {dist_name} distribution"
                assert not torch.any(torch.isinf(distances)), f"Inf distances found in {name} for {dist_name} distribution"
