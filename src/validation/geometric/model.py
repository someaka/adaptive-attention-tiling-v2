"""Model-specific geometric validation implementation.

This module provides validation methods specific to neural model geometries:
- Layer-wise metric validation
- Attention head geometry validation
- Cross-layer geometric compatibility
- Model-specific curvature bounds
"""

from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .metric import GeometricMetricValidator
from ...core.models.base import ModelGeometry
from ...core.patterns.riemannian import RiemannianFramework


def tensor_repr(tensor: Optional[Tensor], max_elements: int = 8) -> str:
    """Create a shortened string representation of tensors."""
    if tensor is None:
        return "None"
    shape = list(tensor.shape)
    if len(shape) == 0:
        return f"tensor({tensor.item():.4f})"
    if sum(shape) <= max_elements:
        return str(tensor)
    return f"tensor(shape={shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f})"


@dataclass
class ValidationResult:
    """Result of a geometric validation."""
    is_valid: bool
    data: Dict[str, Union[torch.Tensor, bool, str, 'ValidationResult', Dict[str, Union[torch.Tensor, bool, str, 'ValidationResult']]]]
    message: str = ""
    
    def __post_init__(self):
        """Validate the data structure."""
        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")
            
    def __repr__(self) -> str:
        """Create a shortened string representation."""
        data_repr = {k: tensor_repr(v) if isinstance(v, torch.Tensor) else str(v) for k, v in self.data.items()}
        return f"ValidationResult(valid={self.is_valid}, data={data_repr}, message='{self.message}')"


class ModelGeometricValidator:
    """Model-specific geometric validation."""

    def __init__(
        self,
        model_geometry: ModelGeometry,
        tolerance: float = 1e-6,
        curvature_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        """Initialize model geometric validator.
        
        Args:
            model_geometry: Geometry of the neural model
            tolerance: Tolerance for validation checks
            curvature_bounds: (lower, upper) bounds for curvature
        """
        self.model_geometry = model_geometry
        self.tolerance = tolerance
        self.curvature_bounds = curvature_bounds
        
        # Initialize layer-wise validators
        self.layer_validators = {}
        for name, layer in model_geometry.layers.items():
            manifold_dim = getattr(layer, 'manifold_dim', None)
            if manifold_dim is None or not isinstance(manifold_dim, int):
                continue
            self.layer_validators[name] = GeometricMetricValidator(
                manifold_dim=manifold_dim,
                tolerance=tolerance,
                curvature_bounds=curvature_bounds
            )

    def validate_layer_geometry(
        self, layer_name: str, points: torch.Tensor
    ) -> ValidationResult:
        """Validate geometry of a specific layer.
        
        Args:
            layer_name: Name of the layer to validate
            points: Points tensor for validation
            
        Returns:
            Validation results for the layer
        """
        # Get geometric tensors from model geometry
        metric_tensor = getattr(self.model_geometry, 'metric')(points)
        sectional_curvature = getattr(self.model_geometry, 'sectional_curvature')(points)
        
        # Check metric properties
        is_valid = True
        message = []
        
        # Check positive definiteness
        eigenvals = torch.linalg.eigvalsh(metric_tensor)
        if not (eigenvals > 0).all():
            is_valid = False
            message.append(f"Metric tensor is not positive definite")
            
        # Check symmetry
        if not torch.allclose(metric_tensor, metric_tensor.transpose(-2, -1), atol=self.tolerance):
            is_valid = False
            message.append(f"Metric tensor is not symmetric")
            
        # Check curvature bounds
        lower_bound, upper_bound = self.curvature_bounds
        if not ((sectional_curvature >= lower_bound - self.tolerance) & 
                (sectional_curvature <= upper_bound + self.tolerance)).all():
            is_valid = False
            message.append(
                f"Sectional curvature exceeds bounds [{lower_bound}, {upper_bound}]"
            )
        
        # Prepare validation data
        data = {
            "metric_tensor": metric_tensor,
            "sectional_curvature": sectional_curvature,
            "eigenvalues": eigenvals
        }
        
        return ValidationResult(
            is_valid=is_valid,
            data=data,
            message="; ".join(message) if message else "Layer geometry is valid"
        )

    def validate_attention_geometry(
        self, head_idx: int, query_points: torch.Tensor, key_points: torch.Tensor
    ) -> ValidationResult:
        """Validate attention head geometry.
        
        Args:
            head_idx: Index of attention head
            query_points: Query space points
            key_points: Key space points
            
        Returns:
            Validation results for attention geometry
        """
        head = self.model_geometry.attention_heads[head_idx]
        
        # Get metrics for query and key spaces
        query_metric = getattr(head, 'query_metric')(query_points)
        key_metric = getattr(head, 'key_metric')(key_points)
        
        # Compute attention scores
        scores = getattr(head, 'compute_attention')(query_points, key_points)
        
        # Check geometric preservation
        preserves_geometry = self._check_geometric_preservation(
            query_metric.detach(), key_metric.detach(), scores.detach()
        )
        
        # Check compatibility
        compatible = self._validate_attention_compatibility(
            head, query_points, key_points
        )
        
        return ValidationResult(
            is_valid=preserves_geometry and compatible,
            data={
                'query_metric': query_metric.detach(),
                'key_metric': key_metric.detach(),
                'attention_scores': scores.detach(),
                'preserves_geometry': preserves_geometry,
                'compatible': compatible
            },
            message="Attention geometry validation"
        )

    def validate_cross_layer_geometry(
        self, layer1: str, layer2: str, points: torch.Tensor
    ) -> ValidationResult:
        """Validate geometric compatibility between layers.
        
        Args:
            layer1: First layer name
            layer2: Second layer name
            points: Points tensor
            
        Returns:
            Validation results for cross-layer geometry
        """
        # Validate individual layers
        result1 = self.validate_layer_geometry(layer1, points)
        result2 = self.validate_layer_geometry(layer2, points)
        
        # Check metric compatibility
        metric_compatible = self._check_metric_compatibility(
            layer1, layer2, points
        )
        
        # Check connection compatibility
        connection_compatible = self._check_connection_compatibility(
            layer1, layer2, points
        )
        
        return ValidationResult(
            is_valid=result1.is_valid and result2.is_valid and 
                    metric_compatible and connection_compatible,
            data={
                f"{layer1}_validation": result1,
                f"{layer2}_validation": result2,
                "metric_compatibility": metric_compatible,
                "connection_compatibility": connection_compatible
            }
        )

    def validate_model_geometry(
        self,
        batch_size: int = 16,
        manifold_dim: Optional[int] = None,
        max_memory_gb: float = 1.0,
    ) -> Dict[str, ValidationResult]:
        """Validate the geometric properties of the model.
        
        Args:
            batch_size: Number of points to validate
            manifold_dim: Dimension of the manifold (defaults to model's manifold_dim)
            max_memory_gb: Maximum memory usage in GB
            
        Returns:
            Dictionary mapping layer names to validation results
        """
        if manifold_dim is None:
            manifold_dim = self.model_geometry.manifold_dim
            if not isinstance(manifold_dim, int):
                raise ValueError("model_geometry.manifold_dim must be an integer")

        # Generate random points for validation
        points = torch.randn(batch_size, manifold_dim, device=next(self.model_geometry.parameters()).device)
        
        # Validate each layer
        layer_validations = {}
        for layer_name in self.layer_validators:
            layer_validations[layer_name] = self.validate_layer_geometry(layer_name, points)
            
        # Validate attention heads
        attention_validations = {}
        for head_idx in range(len(self.model_geometry.attention_heads)):
            query_points = torch.randn(batch_size, self.model_geometry.query_dim)
            key_points = torch.randn(batch_size, self.model_geometry.key_dim)
            attention_validations[f'head_{head_idx}'] = self.validate_attention_geometry(
                head_idx, query_points, key_points
            )
            
        # Validate global properties
        global_props = self._validate_global_properties(points)
        
        # Combine all validations
        validations = {}
        validations.update(layer_validations)
        validations.update(attention_validations)
        validations['global'] = global_props
        
        return validations

    def _validate_attention_compatibility(
        self, head: nn.Module, query_points: torch.Tensor, key_points: torch.Tensor
    ) -> bool:
        """Validate compatibility between query and key spaces.
        
        Args:
            head: Attention head module
            query_points: Query space points
            key_points: Key space points
            
        Returns:
            True if spaces are compatible
        """
        # Get metrics
        query_metric = getattr(head, 'query_metric')(query_points)
        key_metric = getattr(head, 'key_metric')(key_points)
        
        # Check dimensions match
        if query_metric.shape[-1] != key_metric.shape[-1]:
            return False
            
        # Check metrics are positive definite
        query_eigenvals = torch.linalg.eigvalsh(query_metric.detach())
        key_eigenvals = torch.linalg.eigvalsh(key_metric.detach())
        
        if not (query_eigenvals.gt(0).all() and key_eigenvals.gt(0).all()):
            return False
            
        # Check condition numbers are similar
        query_cond = query_eigenvals.max() / query_eigenvals.min()
        key_cond = key_eigenvals.max() / key_eigenvals.min()
        
        return float(abs(query_cond - key_cond).item()) < self.tolerance

    def _check_metric_compatibility(
        self, layer1: str, layer2: str, points: torch.Tensor
    ) -> bool:
        """Check metric compatibility between layers.
        
        Args:
            layer1: First layer name
            layer2: Second layer name
            points: Points tensor
            
        Returns:
            True if metrics are compatible
        """
        l1 = self.model_geometry.layers[layer1]
        l2 = self.model_geometry.layers[layer2]
        
        # Get metrics
        g1 = getattr(self.model_geometry, 'metric')(points)
        g2 = getattr(self.model_geometry, 'metric')(points)
        
        # Check if metrics are close enough
        return torch.allclose(g1, g2, rtol=self.tolerance)

    def _check_connection_compatibility(
        self, layer1: str, layer2: str, points: torch.Tensor
    ) -> bool:
        """Check connection compatibility between layers.
        
        Args:
            layer1: First layer name
            layer2: Second layer name
            points: Points tensor
            
        Returns:
            True if connections are compatible
        """
        l1 = self.model_geometry.layers[layer1]
        l2 = self.model_geometry.layers[layer2]
        
        # Get connections (Christoffel symbols)
        c1 = getattr(self.model_geometry, 'connection')(points)
        c2 = getattr(self.model_geometry, 'connection')(points)
        
        # Check if connections are close enough
        return torch.allclose(c1, c2, rtol=self.tolerance)

    def _check_geometric_preservation(
        self,
        query_metric: torch.Tensor,
        key_metric: torch.Tensor,
        attention_scores: torch.Tensor
    ) -> bool:
        """Check if attention preserves geometric structure.
        
        Args:
            query_metric: Query space metric (batch_size x query_dim x query_dim)
            key_metric: Key space metric (batch_size x key_dim x key_dim)
            attention_scores: Attention scores (batch_size x num_keys)
            
        Returns:
            True if geometric structure is preserved
        """
        # Check if attention scores preserve distances
        query_distances = torch.cdist(query_metric, query_metric)
        key_distances = torch.cdist(key_metric, key_metric)
        score_distances = torch.cdist(attention_scores, attention_scores)
        
        # Compare distance matrices
        query_key_diff = torch.abs(query_distances - key_distances).mean()
        query_score_diff = torch.abs(query_distances - score_distances).mean()
        
        # Check if differences are within tolerance
        return float(query_key_diff.item()) < self.tolerance and float(query_score_diff.item()) < self.tolerance

    def _validate_global_properties(
        self, points: torch.Tensor
    ) -> ValidationResult:
        """Validate global geometric properties of model.
        
        Args:
            points: Points tensor
            
        Returns:
            Validation results for global properties
        """
        # Check completeness
        complete = True
        for layer_name in self.layer_validators:
            if not self._check_layer_curvature(layer_name, points):
                complete = False
                break
                
        # Check curvature bounds
        curvature_valid = all(
            self._check_layer_curvature(name, points)
            for name in self.layer_validators
        )
        
        # Check energy bounds
        energy_valid = self._check_global_energy(points)
        
        return ValidationResult(
            is_valid=complete and curvature_valid and energy_valid,
            data={
                'complete': complete,
                'curvature_valid': curvature_valid,
                'energy_valid': energy_valid
            },
            message="Global geometric properties validation"
        )

    def _check_layer_curvature(
        self, layer_name: str, points: torch.Tensor
    ) -> bool:
        """Check if layer satisfies curvature bounds.
        
        Args:
            layer_name: Name of layer to check
            points: Points tensor
            
        Returns:
            True if curvature bounds are satisfied
        """
        # Compute sectional curvature
        sectional_curvature = getattr(self.model_geometry, 'sectional_curvature')(points)
        
        lower_bound, upper_bound = self.curvature_bounds
        return bool(sectional_curvature.ge(lower_bound).all() and sectional_curvature.le(upper_bound).all())

    def _check_global_energy(self, points: torch.Tensor) -> bool:
        """Check global energy bounds of model."""
        total_energy = 0.0
        for layer_name in self.layer_validators:
            layer = self.model_geometry.layers[layer_name]
            metric_tensor = getattr(self.model_geometry, 'metric')(points)
            total_energy += float(metric_tensor.abs().mean().item())
            
        return total_energy < 1e3  # Arbitrary threshold for demonstration

    def _compute_sectional_curvature(
        self, riemann: torch.Tensor, metric: torch.Tensor
    ) -> torch.Tensor:
        """Compute sectional curvature from Riemann tensor.
        
        Args:
            riemann: Riemann tensor (batch_size x dim x dim x dim x dim)
            metric: Metric tensor (batch_size x dim x dim)
            
        Returns:
            Sectional curvature tensor (batch_size x dim x dim)
        """
        # Contract Riemann tensor with inverse metric to get sectional curvature
        # First compute inverse metric
        metric_inv = torch.linalg.inv(metric)
        
        # Contract Riemann tensor with inverse metric for both indices
        sectional = torch.einsum('...ijkl,...km,...ln->...ijmn', riemann, metric_inv, metric_inv)
        
        # Take trace to get scalar curvature at each point and direction
        sectional_curvature = torch.einsum('...ijij->...', sectional) / (
            metric.size(-1) * (metric.size(-1) - 1)
        )
        
        return sectional_curvature
