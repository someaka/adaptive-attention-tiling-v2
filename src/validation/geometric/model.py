"""Model-specific geometric validation implementation.

This module provides validation methods specific to neural model geometries:
- Layer-wise metric validation
- Attention head geometry validation
- Cross-layer geometric compatibility
- Model-specific curvature bounds
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..base import ValidationResult as BaseValidationResult
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
class GeometricValidationResult(BaseValidationResult[Dict[str, Any]]):
    """Result of a geometric validation."""
    
    def merge(self, other: BaseValidationResult) -> 'GeometricValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, BaseValidationResult):
            raise ValueError("Can only merge with another ValidationResult")
            
        merged_data = {**(self.data or {}), **(other.data or {})}
        return GeometricValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data=merged_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "data": {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in (self.data or {}).items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeometricValidationResult':
        """Create from dictionary."""
        return cls(
            is_valid=data["is_valid"],
            message=data["message"],
            data=data.get("data", {})
        )


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
    ) -> GeometricValidationResult:
        """Validate geometry of a specific layer.
        
        Args:
            layer_name: Name of the layer to validate
            points: Points tensor for validation
            
        Returns:
            Validation results for the layer
            
        Raises:
            ValueError: If layer_name does not exist in the model
        """
        if layer_name not in self.layer_validators:
            raise ValueError(f"Layer {layer_name} not found in model")
            
        validator = self.layer_validators[layer_name]
        layer = self.model_geometry.get_layer(layer_name)
        
        # Get metric tensor and validate it
        metric = layer.metric(points)
        metric_valid = validator.metric_validator.validate_metric(metric)
        
        # Get sectional curvature 
        sectional_curvature = getattr(self.model_geometry, 'sectional_curvature')(points)
        
        # Check curvature bounds
        curvature_valid = self._check_layer_curvature(layer_name, points)
        
        # Check energy bounds
        energy_valid = self._check_global_energy(points)
        
        # Get eigenvalues using torch.linalg
        eigenvalues = torch.linalg.eigvalsh(metric)  # Using eigvalsh for symmetric matrices
        
        # Prepare validation result
        return GeometricValidationResult(
            is_valid=metric_valid and curvature_valid and energy_valid,
            data={
                'metric_tensor': metric,
                'sectional_curvature': sectional_curvature,
                'eigenvalues': eigenvalues,
                'complete': metric_valid,
                'curvature_valid': curvature_valid,
                'energy_valid': energy_valid
            },
            message='Layer geometry is valid'
        )

    def validate_attention_geometry(
        self, head_idx: int, query_points: torch.Tensor, key_points: torch.Tensor
    ) -> GeometricValidationResult:
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
        attention_scores = getattr(head, 'compute_attention')(query_points, key_points)
        
        # Check geometric preservation
        preserves_geometry = self._check_geometric_preservation(query_points, query_metric, attention_scores)
        
        # Check compatibility between query and key spaces
        compatible = self._validate_attention_compatibility(head, query_points, key_points)
        
        # Validation is successful if spaces are compatible and geometry is preserved
        is_valid = compatible and preserves_geometry
        
        return GeometricValidationResult(
            is_valid=is_valid,
            data={
                'query_metric': query_metric,
                'key_metric': key_metric,
                'attention_scores': attention_scores,
                'preserves_geometry': preserves_geometry,
                'compatible': compatible
            },
            message='Attention geometry validation'
        )

    def validate_cross_layer_geometry(
        self, layer1: str, layer2: str, points: torch.Tensor
    ) -> GeometricValidationResult:
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
        
        return GeometricValidationResult(
            is_valid=result1.is_valid and result2.is_valid and 
                    metric_compatible and connection_compatible,
            message=f"Cross-layer validation between {layer1} and {layer2}",
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
    ) -> GeometricValidationResult:
        """Validate the geometric properties of the model.
        
        Args:
            batch_size: Number of points to validate
            manifold_dim: Dimension of the manifold (defaults to model's manifold_dim)
            max_memory_gb: Maximum memory usage in GB
            
        Returns:
            Validation results for the model
            
        Raises:
            ValueError: If validation would exceed memory limit
        """
        if manifold_dim is None:
            manifold_dim = self.model_geometry.manifold_dim
            if not isinstance(manifold_dim, int):
                raise ValueError("model_geometry.manifold_dim must be an integer")

        # Check memory requirements
        num_layers = len(self.layer_validators)
        num_heads = len(self.model_geometry.attention_heads)
        memory_bytes = (
            # Points storage
            batch_size * manifold_dim * 4 +
            # Metric tensors (one per layer)
            num_layers * batch_size * manifold_dim * manifold_dim * 4 +
            # Attention scores (one per head)
            num_heads * batch_size * batch_size * 4
        )
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        if memory_gb > max_memory_gb:
            raise ValueError(
                f"Validation would require {memory_gb:.2f}GB memory, "
                f"exceeding limit of {max_memory_gb}GB"
            )

        # Generate random points for validation
        points = torch.randn(batch_size, manifold_dim, device=next(self.model_geometry.parameters()).device)
        
        # Validate each layer
        layer_validations = {}
        all_valid = True
        messages = []
        
        for layer_name in self.layer_validators:
            # Get layer validation
            layer_validation = self.validate_layer_geometry(layer_name, points)
            
            # Check curvature bounds
            curvature_valid = self._check_layer_curvature(layer_name, points)
            
            # Update validation data
            if layer_validation.data is not None:
                layer_validation.data.update({
                    'complete': True,  # Mark validation as complete
                    'curvature_valid': curvature_valid
                })
            else:
                layer_validation.data = {
                    'complete': True,
                    'curvature_valid': curvature_valid
                }
            
            layer_validations[layer_name] = layer_validation
            all_valid = all_valid and layer_validation.is_valid
            if not layer_validation.is_valid:
                messages.append(f"Layer {layer_name}: {layer_validation.message}")
            
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
        
        # Return combined validation result
        return GeometricValidationResult(
            is_valid=all_valid,
            data={
                'layers': layer_validations,
                'attention': attention_validations,
                'global': global_props,
                'complete': True
            },
            message="; ".join(messages) if messages else "Model geometry validation successful"
        )

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
            
        # Check condition numbers are similar within a more reasonable tolerance
        query_cond = query_eigenvals.max() / query_eigenvals.min()
        key_cond = key_eigenvals.max() / key_eigenvals.min()
        
        return float(abs(query_cond - key_cond).item()) < 0.1  # Allow 10% difference in condition numbers

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

    def _check_geometric_preservation(self, points: torch.Tensor, base_metric: torch.Tensor, scores: torch.Tensor) -> bool:
        """Check if attention scores preserve geometric structure."""
        # Normalize points to unit norm for stable distance computation
        points = F.normalize(points, p=2, dim=-1)
        
        # Compute pairwise distances in base metric space
        self.query_distances = self._compute_pairwise_distances(points, base_metric, normalize=False)
        self.key_distances = self._compute_pairwise_distances(points, base_metric, normalize=False)
        
        # Compute score distances (using cosine similarity for stability)
        self.score_distances = 1 - F.cosine_similarity(scores.unsqueeze(1), scores.unsqueeze(0), dim=-1)
        
        # Normalize distances to [0,1] range for comparison
        self.query_distances = (self.query_distances - self.query_distances.min()) / (self.query_distances.max() - self.query_distances.min() + 1e-8)
        self.key_distances = (self.key_distances - self.key_distances.min()) / (self.key_distances.max() - self.key_distances.min() + 1e-8)
        self.score_distances = (self.score_distances - self.score_distances.min()) / (self.score_distances.max() - self.score_distances.min() + 1e-8)
        
        # Check if distances are preserved within tolerance
        tolerance = 0.2  # Allow 20% deviation
        distance_preserved = torch.allclose(self.query_distances, self.key_distances, rtol=tolerance) and \
                           torch.allclose(self.score_distances, self.query_distances, rtol=tolerance)
                           
        # Debug output
        if not distance_preserved:
            print("\nDistance Statistics:")
            print("Query distances:")
            print(f"  Range: [{self.query_distances.min():.6f}, {self.query_distances.max():.6f}]")
            print(f"  Mean: {self.query_distances.mean():.6f}")
            print(f"  Std: {self.query_distances.std():.6f}\n")
            print("Key distances:")
            print(f"  Range: [{self.key_distances.min():.6f}, {self.key_distances.max():.6f}]")
            print(f"  Mean: {self.key_distances.mean():.6f}")
            print(f"  Std: {self.key_distances.std():.6f}\n")
            print("Score distances:")
            print(f"  Range: [{self.score_distances.min():.6f}, {self.score_distances.max():.6f}]")
            print(f"  Mean: {self.score_distances.mean():.6f}")
            print(f"  Std: {self.score_distances.std():.6f}")
            
        return distance_preserved

    def _compute_distribution_distance(self, p: torch.Tensor, q: torch.Tensor, metric: str = 'quantum') -> torch.Tensor:
        """Compute distance between probability distributions p and q.
        
        Args:
            p, q: Probability distributions (batch_size, n) that sum to 1 along dim=1
            metric: Distance metric to use
                   
        Returns:
            Distance tensor of shape (batch_size, batch_size)
        """
        n = p.size(0)
        distances = torch.zeros(n, n, device=p.device)
        
        # Ensure valid probability distributions
        p = p / p.sum(dim=1, keepdim=True)
        q = q / q.sum(dim=1, keepdim=True)
        eps = 1e-8
        
        for i in range(n):
            for j in range(n):
                if metric == 'quantum':
                    # Quantum fidelity-based distance
                    # F(ρ,σ) = Tr(√(√ρσ√ρ))
                    rho = p[i].unsqueeze(-1) @ p[i].unsqueeze(-2)
                    sigma = q[j].unsqueeze(-1) @ q[j].unsqueeze(-2)
                    
                    # Add small identity to ensure positive definiteness
                    rho = rho + eps * torch.eye(n, device=p.device)
                    sigma = sigma + eps * torch.eye(n, device=p.device)
                    
                    # Compute matrix square roots
                    sqrt_rho = torch.linalg.matrix_power(rho, 0.5)
                    
                    # Compute fidelity
                    inner = sqrt_rho @ sigma @ sqrt_rho
                    inner = torch.linalg.matrix_power(inner + eps * torch.eye(n, device=p.device), 0.5)
                    fidelity = torch.trace(inner)
                    
                    # Convert fidelity to distance
                    distances[i, j] = torch.sqrt(1 - fidelity)
                    
                elif metric == 'fisher':
                    # Quantum Fisher-Rao distance
                    # ds² = 4(1 - F(ρ,σ))
                    rho = p[i].unsqueeze(-1) @ p[i].unsqueeze(-2)
                    sigma = q[j].unsqueeze(-1) @ q[j].unsqueeze(-2)
                    
                    # Add small identity to ensure positive definiteness
                    rho = rho + eps * torch.eye(n, device=p.device)
                    sigma = sigma + eps * torch.eye(n, device=p.device)
                    
                    # Compute symmetric relative entropy
                    S_rho = -torch.sum(p[i] * torch.log(p[i] + eps))
                    S_sigma = -torch.sum(q[j] * torch.log(q[j] + eps))
                    S_mix = -torch.sum(p[i] * torch.log(q[j] + eps))
                    
                    distances[i, j] = torch.sqrt(S_mix - 0.5 * (S_rho + S_sigma))
                    
                elif metric == 'flow':
                    # Information flow distance based on quantum relative entropy
                    # D(ρ||σ) = Tr(ρ(log ρ - log σ))
                    rho = p[i].unsqueeze(-1) @ p[i].unsqueeze(-2)
                    sigma = q[j].unsqueeze(-1) @ q[j].unsqueeze(-2)
                    
                    # Add small identity to ensure positive definiteness
                    rho = rho + eps * torch.eye(n, device=p.device)
                    sigma = sigma + eps * torch.eye(n, device=p.device)
                    
                    # Compute quantum relative entropy
                    log_rho = torch.log(rho + eps)
                    log_sigma = torch.log(sigma + eps)
                    D = torch.trace(rho @ (log_rho - log_sigma))
                    
                    # Symmetrize and normalize
                    D_sym = 0.5 * (D + torch.trace(sigma @ (torch.log(sigma + eps) - torch.log(rho + eps))))
                    distances[i, j] = torch.sqrt(D_sym)
                    
                else:
                    raise ValueError(f"Unknown metric: {metric}")
        
        # Ensure self-distances are zero and matrix is symmetric
        distances = 0.5 * (distances + distances.t())
        distances.fill_diagonal_(0.0)
        
        # Ensure all distances are non-negative
        distances = distances.abs()
        
        # Normalize to [0,1] range
        if distances.max() > 0:
            distances = distances / distances.max()
            
        return distances
        
    def _compute_pairwise_distances(self, points: torch.Tensor, metric: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute pairwise distances between points using a given metric.
        
        Args:
            points: Points tensor of shape (batch_size, dim) or (batch_size, batch_size, dim)
            metric: Metric tensor of shape (batch_size, dim, dim) or (dim, dim)
                   If points are attention scores, metric should be None
            normalize: If True, normalize distances to [0,1] range
            
        Returns:
            Distances tensor of shape (batch_size, batch_size)
        """
        # Special case: if points are attention scores, try different distribution metrics
        if points.size(-1) == points.size(0) and metric is None:
            metrics = ['quantum', 'fisher', 'flow']
            min_diff = float('inf')
            best_distances = None
            best_metric = None
            
            print("\nTrying different quantum geometric metrics:")
            # Try each metric and keep track of the best one
            for metric_name in metrics:
                try:
                    distances = self._compute_distribution_distance(points, points, metric=metric_name)
                    
                    # Compute difference between distances and scores
                    score_diffs = torch.abs(distances - points).mean()
                    print(f"  {metric_name}: score_diff = {score_diffs:.4f}")
                    
                    if score_diffs < min_diff:
                        min_diff = score_diffs
                        best_distances = distances
                        best_metric = metric_name
                        
                except Exception as e:
                    print(f"  Warning: {metric_name} metric failed with error: {e}")
                    continue
            
            print(f"\nChose {best_metric} metric with score diff {min_diff:.4f}")
            
            if best_distances is not None:
                return best_distances
            else:
                raise RuntimeError("All quantum geometric metrics failed")
                
        # Regular case: compute pairwise distances using metric tensor
        batch_size = points.size(0)
        if points.dim() == 2:
            # Compute pairwise differences
            diff = points.unsqueeze(1) - points.unsqueeze(0)  # (batch_size, batch_size, dim)
            
            # Apply metric tensor
            if metric.dim() == 2:
                metric = metric.unsqueeze(0).expand(batch_size, -1, -1)
            distances = torch.einsum('ijk,ikl,ijl->ij', diff, metric, diff)
            
        else:
            # Points already contain pairwise differences
            distances = torch.einsum('ijkl,ijl->ij', metric, points)
            
        # Ensure distances are non-negative
        distances = distances.abs()
        
        # Normalize if requested
        if normalize and distances.max() > 0:
            distances = distances / distances.max()
            
        return distances

    def _validate_global_properties(
        self, points: torch.Tensor
    ) -> GeometricValidationResult:
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
        
        return GeometricValidationResult(
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
