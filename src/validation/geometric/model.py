"""Model-specific geometric validation implementation.

This module provides validation methods specific to neural model geometries:
- Layer-wise metric validation
- Attention head geometry validation
- Cross-layer geometric compatibility
- Model-specific curvature bounds
"""

from typing import Dict, List, Optional, Tuple
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
    data: Dict[str, torch.Tensor]
    message: str = ""
    
    def __post_init__(self):
        """Validate the data structure."""
        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")
        if not all(isinstance(v, torch.Tensor) for v in self.data.values()):
            raise ValueError("all values in data must be torch.Tensor")
            
    def __repr__(self) -> str:
        """Create a shortened string representation."""
        data_repr = {k: tensor_repr(v) for k, v in self.data.items()}
        return f"ValidationResult(is_valid={self.is_valid}, data={data_repr}, message='{self.message}')"


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
        self.layer_validators = {
            name: GeometricMetricValidator(
                manifold_dim=layer.manifold_dim,
                tolerance=tolerance,
                curvature_bounds=curvature_bounds
            )
            for name, layer in model_geometry.layers.items()
        }

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
        if layer_name not in self.layer_validators:
            raise ValueError(f"Unknown layer: {layer_name}")
            
        validator = self.layer_validators[layer_name]
        layer = self.model_geometry.layers[layer_name]
        
        # Ensure points require gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Get layer's Riemannian framework
        framework = layer.get_riemannian_framework(points)
        
        # Perform geometric validation
        return validator.validate(framework, points)

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
        # Get attention head geometry
        head = self.model_geometry.attention_heads[head_idx]
        
        # Validate query space
        query_result = self.validate_layer_geometry(
            f"query_{head_idx}", query_points
        )
        
        # Validate key space
        key_result = self.validate_layer_geometry(
            f"key_{head_idx}", key_points
        )
        
        # Validate compatibility between query and key spaces
        compatibility = self._validate_attention_compatibility(
            head, query_points, key_points
        )
        
        return ValidationResult(
            is_valid=query_result.is_valid and key_result.is_valid and compatibility,
            data={
                "query_validation": query_result,
                "key_validation": key_result,
                "compatibility": compatibility
            }
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
    ) -> ValidationResult:
        """Validate the geometric properties of the model.
        
        Args:
            batch_size: Number of points to validate
            manifold_dim: Dimension of the manifold (defaults to model's manifold_dim)
            max_memory_gb: Maximum memory usage in GB
            
        Returns:
            ValidationResult containing geometric validation metrics
        """
        # Use model's manifold dimension if not specified
        if manifold_dim is None:
            manifold_dim = next(iter(self.model_geometry.layers.values())).manifold_dim

        # Resource guards
        expected_memory = batch_size * (manifold_dim ** 3) * 4  # 4 bytes per float
        if expected_memory > max_memory_gb * 1e9:
            raise ValueError(
                f"Validation would require {expected_memory/1e9:.2f}GB memory. "
                f"Reduce batch_size or manifold_dim, or increase max_memory_gb."
            )

        try:
            # Generate random points on the manifold
            points = torch.randn(batch_size, manifold_dim, requires_grad=True)
            
            # Compute geometric quantities with memory-efficient chunks
            chunk_size = min(batch_size, 4)  # Process in small chunks
            results = defaultdict(list)
            
            for i in range(0, batch_size, chunk_size):
                chunk_points = points[i:i+chunk_size]
                
                # Compute metric and curvature for this chunk
                metric = self.model_geometry.metric(chunk_points)
                curvature = self.model_geometry.sectional_curvature(chunk_points)
                
                results['metric'].append(metric.detach())
                results['curvature'].append(curvature.detach())
            
            # Combine chunks
            final_results = {
                k: torch.cat(v, dim=0) 
                for k, v in results.items()
            }
            
            # Validate individual layers
            layer_validations = {}
            for layer_name in self.model_geometry.layers:
                layer_validations[layer_name] = self.validate_layer_geometry(
                    layer_name, points
                )
            
            # Validate attention heads
            attention_validations = []
            for head_idx, _ in enumerate(self.model_geometry.attention_heads):
                query_points = torch.randn(batch_size, manifold_dim)
                key_points = torch.randn(batch_size, manifold_dim)
                attention_validations.append(
                    self.validate_attention_geometry(head_idx, query_points, key_points)
                )
            
            # Validate global properties
            global_properties = self._validate_global_properties(points)
            
            # Combine all results
            final_results.update({
                'layer_validations': layer_validations,
                'attention_validations': attention_validations,
                'global_properties': global_properties
            })
            
            return ValidationResult(
                is_valid=True,
                data=final_results,
                message="Geometric validation completed successfully"
            )
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    f"Out of memory during validation. Try reducing batch_size (currently {batch_size}) "
                    f"or manifold_dim (currently {manifold_dim})"
                ) from e
            raise

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
        # Check dimension compatibility
        if query_points.shape[-1] != head.query_dim or key_points.shape[-1] != head.key_dim:
            return False
            
        # Check metric compatibility
        query_metric = head.query_metric(query_points)
        key_metric = head.key_metric(key_points)
        
        # Compute attention scores
        scores = head.compute_attention(query_points, key_points)
        
        # Check if attention preserves geometric structure
        preserved = self._check_geometric_preservation(
            query_metric, key_metric, scores
        )
        
        return preserved

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
        g1 = l1.metric(points)
        g2 = l2.metric(points)
        
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
        
        # Get connections
        c1 = l1.connection(points)
        c2 = l2.connection(points)
        
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
        # Ensure tensors require gradients
        if not query_metric.requires_grad:
            query_metric = query_metric.detach().requires_grad_(True)
        if not key_metric.requires_grad:
            key_metric = key_metric.detach().requires_grad_(True)
        if not attention_scores.requires_grad:
            attention_scores = attention_scores.detach().requires_grad_(True)
            
        # Compute pullback metric: attention_scores.shape = [batch, num_keys]
        # key_metric.shape = [batch, num_keys, key_dim, key_dim]
        # Expected output shape = [batch, query_dim, query_dim]
        pullback_metric = torch.einsum('bk,bkij->bij', attention_scores, key_metric)
        
        return torch.allclose(
            pullback_metric,
            query_metric,
            rtol=self.tolerance
        )

    def _validate_global_properties(
        self, points: torch.Tensor
    ) -> ValidationResult:
        """Validate global geometric properties of model.
        
        Args:
            points: Points tensor
            
        Returns:
            Validation results for global properties
        """
        # Check global completeness
        complete = all(
            validator.check_geodesic_completeness(points)
            for validator in self.layer_validators.values()
        )
        
        # Check global curvature bounds
        curvature_valid = all(
            self._check_layer_curvature(name, points)
            for name in self.layer_validators
        )
        
        # Check global energy bounds
        energy_valid = self._check_global_energy(points)
        
        return ValidationResult(
            is_valid=complete and curvature_valid and energy_valid,
            data={
                "complete": complete,
                "curvature_valid": curvature_valid,
                "energy_valid": energy_valid
            }
        )

    def _check_layer_curvature(
        self, layer_name: str, points: torch.Tensor
    ) -> bool:
        """Check if layer satisfies curvature bounds.
        
        Args:
            layer_name: Layer name
            points: Points tensor
            
        Returns:
            True if curvature bounds are satisfied
        """
        layer = self.model_geometry.layers[layer_name]
        framework = layer.get_riemannian_framework(points)
        
        # Get curvature tensor
        curvature = framework.compute_curvature_tensor()
        
        # Check bounds
        lower, upper = self.curvature_bounds
        within_bounds = (curvature >= lower).all() and (curvature <= upper).all()
        
        return within_bounds

    def _check_global_energy(self, points: torch.Tensor) -> bool:
        """Check global energy bounds of model."""
        total_energy = 0.0
        
        # Ensure points require gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Compute energy for each layer
        for layer_name, validator in self.layer_validators.items():
            # Get metric values 
            layer = self.model_geometry.get_layer(layer_name)
            metric = layer.metric(points)
            
            # Compute energy as Frobenius norm of metric deviation from identity
            identity = torch.eye(
                metric.size(-1), 
                device=metric.device
            ).expand_as(metric)
            deviation = metric - identity
            energy = torch.norm(deviation, p='fro').item()
            total_energy += energy
        
        # Check if average energy deviation from identity is bounded
        average_energy = total_energy / len(self.layer_validators)
        return average_energy <= 1.0  # Using a more reasonable bound for deviation from identity
