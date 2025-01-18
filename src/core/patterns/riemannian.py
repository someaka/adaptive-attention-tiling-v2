"""
Base Implementation of Riemannian Geometry.

This module provides the foundational implementation of Riemannian geometric structures,
serving as the base class for more specialized implementations. It focuses on exact
computations using autograd for derivatives.

Key features:
1. Exact derivative computation using autograd
2. Numerically stable geometric operations
3. Comprehensive validation and error checking
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, cast, Callable
from typing_extensions import Protocol, runtime_checkable
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if none exists
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

from .riemannian_base import (
    RiemannianStructure,
    RiemannianValidator,
    ValidationMixin,
    MetricTensor,
    ChristoffelSymbols,
    CurvatureTensor,
)
from ...utils.device import get_device

class BaseRiemannianStructure(nn.Module, RiemannianStructure[Tensor], ValidationMixin):
    """Base implementation of Riemannian geometric structure.
    
    This class provides the core implementation of Riemannian geometry using
    autograd for exact derivative computation. It serves as the foundation
    for more specialized implementations.
    
    Attributes:
        manifold_dim: Dimension of the manifold
        hidden_dim: Hidden dimension for computations
        device: Computation device
        dtype: Data type for computations
        cache: Cache for intermediate computations
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize Riemannian structure.
        
        Args:
            manifold_dim: Dimension of the manifold
            hidden_dim: Hidden dimension for computations
            device: Computation device (defaults to CPU)
            dtype: Data type for computations
        """
        super().__init__()
        
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        
        # Initialize metric as identity plus low-rank perturbation for stability
        self.metric_factors = nn.Parameter(
            torch.randn(manifold_dim, manifold_dim, device=self.device, dtype=self.dtype) * 0.01
        )
        
        # Initialize connection coefficients
        self.connection_coeffs = nn.Parameter(
            torch.zeros(manifold_dim, manifold_dim, manifold_dim, device=self.device, dtype=self.dtype)
        )
        
        # Initialize metric parameter as None (will be set by subclasses if needed)
        self.metric_param = None
        
        # Cache for intermediate computations
        self.cache: Dict[str, Any] = {}
        
    def _clear_cache(self) -> None:
        """Clear computation cache."""
        self.cache.clear()
        
    def _ensure_gradients(self, tensor: Tensor) -> Tensor:
        """Ensure tensor has gradients enabled.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor with gradients enabled
        """
        if not tensor.requires_grad:
            tensor = tensor.detach().requires_grad_(True)
        return tensor
        
    def validate_metric_properties(self, metric: MetricTensor[Tensor]) -> bool:
        """Validate that metric tensor satisfies required properties.
        
        Checks:
        1. Symmetry
        2. Positive definiteness
        3. Smoothness
        4. Compatibility conditions
        
        Args:
            metric: Metric tensor to validate
            
        Returns:
            Whether metric satisfies all properties
        """
        # Check symmetry
        is_symmetric = bool(torch.allclose(
            metric.values,
            metric.values.transpose(-2, -1),
            rtol=1e-10
        ))
        
        # Check positive definiteness
        eigenvals = torch.linalg.eigvalsh(metric.values)
        is_positive_definite = bool(torch.all(eigenvals > 0))
        
        # Check dimension compatibility
        has_correct_dim = (
            metric.dimension == self.manifold_dim and
            metric.values.shape[-2:] == (self.manifold_dim, self.manifold_dim)
        )
        
        # Check smoothness via gradient existence
        try:
            grad = torch.autograd.grad(
                metric.values.sum(),
                self.metric_factors,
                create_graph=True
            )[0]
            is_smooth = grad is not None
        except Exception:
            is_smooth = False
            
        return bool(is_symmetric and is_positive_definite and has_correct_dim and is_smooth)
        
    def validate_connection_properties(self, connection: ChristoffelSymbols[Tensor]) -> bool:
        """Validate that connection satisfies required properties.
        
        Checks:
        1. Symmetry in lower indices
        2. Metric compatibility
        3. Torsion-free condition
        
        Args:
            connection: Christoffel symbols to validate
            
        Returns:
            Whether connection satisfies all properties
        """
        # Check symmetry in lower indices
        is_symmetric = torch.allclose(
            connection.values.transpose(-2, -1),
            connection.values,
            rtol=1e-10
        )
        
        # Check metric compatibility
        metric = connection.metric
        points = self._ensure_gradients(torch.zeros(
            1, self.manifold_dim,
            device=self.device,
            dtype=self.dtype
        ))
        
        metric_deriv = torch.autograd.grad(
            metric.values.sum(),
            points,
            create_graph=True
        )[0]
        
        is_compatible = torch.allclose(
            metric_deriv,
            torch.zeros_like(metric_deriv),
            rtol=1e-8
        )
        
        # Check torsion-free condition
        torsion = (
            connection.values
            - connection.values.transpose(-2, -1)
        )
        is_torsion_free = torch.allclose(
            torsion,
            torch.zeros_like(torsion),
            rtol=1e-10
        )
        
        return is_symmetric and is_compatible and is_torsion_free
        
    def compute_metric(self, points: Tensor) -> MetricTensor[Tensor]:
        """Compute the metric tensor.
        
        Implements a stable metric computation using:
        g = I + V^T V
        where V is a learnable factor matrix.
        
        Args:
            points: Points at which to compute metric
            
        Returns:
            Metric tensor at points
        """
        batch_size = points.shape[0]
        
        # Compute base metric with correct dtype
        identity = torch.eye(
            self.manifold_dim,
            dtype=self.dtype,
            device=self.device
        ).expand(batch_size, -1, -1)
        
        # Add learned perturbation
        perturbation = torch.matmul(
            self.metric_factors.T,
            self.metric_factors
        )
        
        # Ensure perturbation has proper batch dimension
        perturbation = perturbation.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine identity and perturbation without in-place operations
        values = identity + perturbation
        
        # Convert to proper dtype while maintaining gradient chain
        if values.dtype != self.dtype:
            values = values.to(dtype=self.dtype)
        
        # Add hook to track metric gradients
        if values.requires_grad:
            def metric_hook(grad):
                logger.debug(f"\n=== Metric Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            values.register_hook(metric_hook)
        
        # Add hook to track metric parameter gradients
        if self.metric_factors.requires_grad:
            def metric_param_hook(grad):
                logger.debug(f"\n=== Metric Parameter Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            self.metric_factors.register_hook(metric_param_hook)
        
        # Create metric tensor with gradient tracking
        metric = MetricTensor(values=values, dimension=self.manifold_dim)
        
        # Add hook to track metric tensor gradients
        if metric.values.requires_grad:
            def metric_tensor_hook(grad):
                logger.debug(f"\n=== Metric Tensor Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            metric.values.register_hook(metric_tensor_hook)
        
        return metric
        
    def compute_christoffel(self, points: Tensor) -> ChristoffelSymbols[Tensor]:
        """Compute Christoffel symbols using autograd with metric parameter gradients."""
        # Get metric tensor with gradient tracking
        metric = self.compute_metric(points)
        metric_values = metric.values  # Shape: [batch_size, dim, dim]
        
        # Compute metric derivatives using autograd
        batch_size = points.shape[0]
        dim = self.manifold_dim
        
        # Initialize metric derivatives tensor
        metric_derivatives = torch.zeros(
            batch_size, dim, dim, dim,
            device=points.device,
            dtype=points.dtype
        )
        
        # For each coordinate, compute derivatives using autograd
        for k in range(dim):
            # Create graph for metric derivatives
            metric_k = metric_values[..., k]
            
            # Compute gradients with respect to points
            inputs = [points]
            if hasattr(self, 'metric_param') and self.metric_param is not None:
                inputs.append(self.metric_param)
            
            grads = torch.autograd.grad(
                metric_k.sum(),
                inputs,
                create_graph=True,
                allow_unused=True,
                retain_graph=True
            )
            
            # Handle point gradients
            point_grad = grads[0]
            if point_grad is not None:
                point_grad = point_grad.reshape(batch_size, dim, dim)
                metric_derivatives[..., k] = point_grad
            
            # Handle metric parameter gradients if present
            if len(grads) > 1 and grads[1] is not None:
                metric_grad = grads[1]
                # Reshape and accumulate metric gradients
                metric_grad = metric_grad.reshape(batch_size, dim, dim)
                metric_derivatives[..., k] = metric_derivatives[..., k] + metric_grad
        
        # Compute inverse metric with gradient tracking
        metric_inverse = torch.inverse(metric_values)
        
        # Pre-allocate Christoffel symbols tensor
        christoffel_values = torch.zeros(
            batch_size, dim, dim, dim,
            device=points.device,
            dtype=points.dtype
        )
        
        # Compute Christoffel symbols using the formula:
        # Γᵏᵢⱼ = 1/2 gᵏˡ (∂ᵢgⱼˡ + ∂ⱼgᵢˡ - ∂ˡgᵢⱼ)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # First term: gᵏˡ ∂ᵢgⱼˡ
                    term1 = torch.einsum('...l,...jl->...', metric_inverse[:, k, :], metric_derivatives[:, i, j, :])
                    
                    # Second term: gᵏˡ ∂ⱼgᵢˡ
                    term2 = torch.einsum('...l,...il->...', metric_inverse[:, k, :], metric_derivatives[:, j, i, :])
                    
                    # Third term: -gᵏˡ ∂ˡgᵢⱼ
                    term3 = -torch.einsum('...l,...ij->...', metric_inverse[:, k, :], metric_derivatives[:, :, i, j])
                    
                    # Combine terms without in-place operations
                    christoffel_values[:, k, i, j] = 0.5 * (term1 + term2 - term3)
        
        # Add hook to track Christoffel gradients
        if christoffel_values.requires_grad:
            def christoffel_hook(grad):
                logger.debug(f"\n=== Christoffel Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            christoffel_values.register_hook(christoffel_hook)
        
        # Ensure metric parameter gradients are connected
        if hasattr(self, 'metric_param') and self.metric_param is not None and self.metric_param.requires_grad:
            def metric_param_hook(grad):
                logger.debug(f"\n=== Metric Parameter Gradient Hook (from Christoffel) ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            self.metric_param.register_hook(metric_param_hook)
        
        return ChristoffelSymbols(
            values=christoffel_values,
            metric=metric,
            is_symmetric=True  # Guaranteed by construction
        )
        
    def parallel_transport(
        self,
        vector: Tensor,
        path: Tensor,
        connection: Optional[ChristoffelSymbols[Tensor]] = None
    ) -> Tensor:
        """Parallel transport using numerical integration."""
        # Ensure inputs require gradients without detaching
        if not vector.requires_grad:
            vector = vector.clone().requires_grad_(True)
        if not path.requires_grad:
            path = path.clone().requires_grad_(True)
            
        logger.debug(f"\n=== Starting Parallel Transport ===")
        logger.debug(f"Vector shape: {vector.shape}, requires_grad: {vector.requires_grad}")
        logger.debug(f"Path shape: {path.shape}, requires_grad: {path.requires_grad}")
        
        # Initialize result list to avoid in-place operations
        result = []
        result.append(vector)  # Initial condition
        
        # Get connection if not provided
        if connection is None:
            connection = self.compute_christoffel(path)
            logger.debug(f"Connection values shape: {connection.values.shape}, requires_grad: {connection.values.requires_grad}")
            
        # Compute path tangent vectors
        tangents = path[1:] - path[:-1]
        logger.debug(f"Tangents shape: {tangents.shape}, requires_grad: {tangents.requires_grad}")
        
        # Parallel transport equation
        for t in range(len(path) - 1):
            current_vector = result[-1]  # Use directly without detach
            logger.debug(f"\nStep {t}:")
            logger.debug(f"Current vector requires_grad: {current_vector.requires_grad}")
            
            # Transport step with gradient tracking
            step = -torch.einsum(
                '...ijk,...j,...k->...i',
                connection.values[t],
                current_vector,
                tangents[t]
            )
            
            # Add gradient hook for step
            if step.requires_grad:
                def step_hook(grad):
                    logger.debug(f"\n=== Transport Step {t} Gradient Hook ===")
                    logger.debug(f"Shape: {grad.shape}")
                    logger.debug(f"Gradient norm: {torch.norm(grad)}")
                    logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                    logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                    logger.debug(f"Mean: {grad.mean().item()}")
                    logger.debug(f"Std: {grad.std().item()}")
                    return grad
                step.register_hook(step_hook)
            
            # Update with gradient tracking
            next_vector = current_vector + step
            logger.debug(f"Next vector requires_grad: {next_vector.requires_grad}")
            
            result.append(next_vector)
            
        # Stack results into a tensor
        return torch.stack(result)
        
    def compute_curvature(
        self,
        points: Tensor,
        christoffel: Optional[ChristoffelSymbols[Tensor]] = None
    ) -> CurvatureTensor[Tensor]:
        """Compute curvature tensors.
        
        Implements the formula:
        R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
        
        Args:
            points: Points at which to compute curvature
            christoffel: Optional pre-computed Christoffel symbols
            
        Returns:
            Full curvature information
        """
        if christoffel is None:
            christoffel = self.compute_christoffel(points)
            
        batch_size = points.shape[0]
        
        # Initialize curvature tensors
        riemann = torch.zeros(
            batch_size,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            device=self.device,
            dtype=self.dtype
        )
        
        # Compute Christoffel derivatives
        christoffel_grad = torch.autograd.grad(
            christoffel.values.sum(),
            points,
            create_graph=True,
            allow_unused=True
        )[0]
        
        # Handle None gradients by providing a zero tensor
        if christoffel_grad is None:
            christoffel_grad = torch.zeros(
                batch_size,
                self.manifold_dim,
                self.manifold_dim,
                self.manifold_dim,
                self.manifold_dim,
                device=self.device,
                dtype=self.dtype
            )
        
        # Construct Riemann tensor
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # Derivative terms
                        term1 = christoffel_grad[..., i, j, l, k]
                        term2 = -christoffel_grad[..., i, j, k, l]
                        
                        # Connection terms
                        term3 = torch.sum(
                            christoffel.values[..., i, :, k] *
                            christoffel.values[..., :, j, l]
                        )
                        term4 = -torch.sum(
                            christoffel.values[..., i, :, l] *
                            christoffel.values[..., :, j, k]
                        )
                        
                        riemann[..., i, j, k, l] = term1 + term2 + term3 + term4
                        
        # Compute Ricci tensor
        ricci = torch.einsum('...ijkj->...ik', riemann)
        
        # Compute scalar curvature
        scalar_curvatures = torch.einsum('...ii', ricci)
        
        return CurvatureTensor(
            riemann=riemann,
            ricci=ricci,
            scalar_curvatures=scalar_curvatures
        )
        
    def geodesic_flow(
        self,
        initial_point: Tensor,
        initial_velocity: Tensor,
        steps: int = 100,
        step_size: float = 0.01
    ) -> Tuple[Tensor, Tensor]:
        """Compute geodesic flow using numerical integration."""
        # Ensure inputs require gradients
        initial_point = initial_point.detach().requires_grad_(True)
        initial_velocity = initial_velocity.detach().requires_grad_(True)
        
        # Initialize lists for points and velocities to avoid in-place operations
        points = [initial_point]  # Initial point
        velocities = [initial_velocity]  # Initial velocity
        
        # Integrate geodesic equation
        for t in range(steps - 1):
            # Get Christoffel symbols at current point
            christoffel = self.compute_christoffel(points[-1].unsqueeze(0))
            
            # Compute acceleration
            acceleration = -torch.einsum(
                '...ijk,...j,...k->...i',
                christoffel.values[0],
                velocities[-1].detach(),  # Detach to prevent in-place op issues
                velocities[-1]
            )
            
            # Update velocity and position
            next_velocity = velocities[-1] + step_size * acceleration
            next_point = points[-1] + step_size * velocities[-1]
            
            points.append(next_point)
            velocities.append(next_velocity)
            
        # Stack results into tensors
        points_tensor = torch.stack(points)
        velocities_tensor = torch.stack(velocities)
        
        return points_tensor, velocities_tensor
        
    def lie_derivative_metric(
        self,
        point: Tensor,
        vector_field: Callable[[Tensor], Tensor]
    ) -> MetricTensor[Tensor]:
        """Compute Lie derivative of metric.
        
        Implements the formula:
        (L_X g)_ij = X^k ∂_k g_ij + g_kj ∂_i X^k + g_ik ∂_j X^k
        
        Args:
            point: Point at which to compute derivative
            vector_field: Function computing vector field
            
        Returns:
            Lie derivative of metric tensor
        """
        point = self._ensure_gradients(point)
        
        # Get metric at point
        metric = self.compute_metric(point.unsqueeze(0))
        
        # Compute vector field and its derivatives
        X = vector_field(point)
        X.requires_grad_(True)  # Ensure X requires gradients
        
        X_grad_result = torch.autograd.grad(
            X.sum(), point, create_graph=True, allow_unused=True
        )[0]
        
        # Handle None gradients
        if X_grad_result is None:
            X_grad_result = torch.zeros(
                self.manifold_dim,
                self.manifold_dim,
                device=self.device,
                dtype=self.dtype
            )
        X_grad = X_grad_result
        
        # Compute metric derivatives
        metric_grad_result = torch.autograd.grad(
            metric.values.sum(),
            point,
            create_graph=True,
            allow_unused=True
        )[0]
        
        # Handle None gradients
        if metric_grad_result is None:
            metric_grad_result = torch.zeros(
                self.manifold_dim,
                self.manifold_dim,
                self.manifold_dim,
                device=self.device,
                dtype=self.dtype
            )
        metric_grad = metric_grad_result
        
        # Compute Lie derivative components
        # Reshape tensors for einsum
        X = X.reshape(-1)  # Shape: [dim]
        metric_grad = metric_grad.reshape(self.manifold_dim, self.manifold_dim, self.manifold_dim)  # Shape: [dim, dim, dim]
        X_grad = X_grad.reshape(self.manifold_dim, self.manifold_dim)  # Shape: [dim, dim]
        metric_vals = metric.values[0]  # Shape: [dim, dim]
        
        # Compute terms with proper dimensions
        term1 = torch.einsum('k,kij->ij', X, metric_grad)
        term2 = torch.einsum('ki,kj->ij', X_grad, metric_vals)
        term3 = torch.einsum('kj,ki->ij', metric_vals, X_grad)
        
        # Combine terms
        lie_deriv = term1 + term2 + term3
        
        return MetricTensor(
            values=lie_deriv.unsqueeze(0),
            dimension=self.manifold_dim
        )
        
    def sectional_curvature(
        self,
        point: Tensor,
        v1: Tensor,
        v2: Tensor
    ) -> Tensor:
        """Compute sectional curvature.
        
        Implements the formula:
        K(v1,v2) = <R(v1,v2)v2,v1> / (|v1∧v2|²)
        
        Args:
            point: Point at which to compute curvature
            v1: First vector spanning plane
            v2: Second vector spanning plane
            
        Returns:
            Sectional curvature value
        """
        # Get metric and curvature
        metric = self.compute_metric(point.unsqueeze(0))
        curvature = self.compute_curvature(point.unsqueeze(0))
        
        # Compute numerator <R(v1,v2)v2,v1>
        numerator = torch.einsum(
            'ijkl,i,j,k,l->',
            curvature.riemann[0],
            v1, v2, v2, v1
        )
        
        # Compute denominator |v1∧v2|²
        g = metric.values[0]
        g11 = torch.einsum('i,ij,j->', v1, g, v1)
        g12 = torch.einsum('i,ij,j->', v1, g, v2)
        g22 = torch.einsum('i,ij,j->', v2, g, v2)
        denominator = g11 * g22 - g12 * g12
        
        return numerator / denominator
        
    def exp_map(self, point: Tensor, vector: Tensor) -> Tensor:
        """Compute exponential map at a point in a given direction.
        
        Implements the exponential map using parallel transport along geodesics.
        
        Args:
            point: Base point on manifold
            vector: Tangent vector at base point
            
        Returns:
            Point reached by following geodesic in direction of vector
        """
        # Ensure points have gradients
        point = self._ensure_gradients(point)
        vector = self._ensure_gradients(vector)
        
        # Get metric at point
        metric = self.compute_metric(point).values
        
        # Compute vector norm using metric (with batch dimensions)
        # Original: vector_norm = torch.sqrt(torch.einsum('i,ij,j->', vector, metric, vector))
        # New: Handle batch dimensions correctly
        vector_norm = torch.sqrt(torch.einsum('...i,...ij,...j->...', vector, metric, vector))
        
        # Normalize vector
        vector = vector / (vector_norm.unsqueeze(-1) + 1e-8)
        
        # Get Christoffel symbols at point
        christoffel = self.compute_christoffel(point)
        
        # Compute geodesic using parallel transport
        t = torch.linspace(0, 1, steps=10, device=self.device, dtype=self.dtype)
        current_point = point
        
        for step in t[1:]:  # Skip t=0 since we start at point
            # Transport vector using connection
            transported = self.parallel_transport(vector, current_point, christoffel)
            
            # Update position
            current_point = current_point + step * transported
            
            # Project back to manifold if needed
            if hasattr(self, 'project_to_manifold'):
                current_point = self.project_to_manifold(current_point)
                
        return current_point

class PatternRiemannianStructure(BaseRiemannianStructure):
    """Pattern-specific Riemannian structure implementation."""
    
    def __init__(
        self,
        manifold_dim: int,
        pattern_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize pattern Riemannian structure."""
        super().__init__(manifold_dim, pattern_dim, device=device, dtype=dtype)
        self.pattern_dim = pattern_dim
        self.metric_param = None
        
    def set_metric_param(self, metric_param: nn.Parameter) -> None:
        """Set metric parameter."""
        self.metric_param = metric_param
        
    def compute_metric(self, points: Tensor) -> MetricTensor[Tensor]:
        """Compute metric tensor with pattern-specific structure."""
        batch_size = points.shape[0]
        
        # Use metric parameter directly if available
        if self.metric_param is not None:
            # Create a view and expand (not repeat) to maintain gradient chain
            values = self.metric_param.view(1, self.manifold_dim, self.manifold_dim)
            values = values.expand(batch_size, -1, -1)  # Use expand instead of repeat
            
            # Add hook to track metric gradients
            if values.requires_grad:
                def metric_hook(grad):
                    logger.debug(f"\n=== Metric Gradient Hook ===")
                    logger.debug(f"Shape: {grad.shape}")
                    logger.debug(f"Gradient norm: {torch.norm(grad)}")
                    logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                    logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                    logger.debug(f"Mean: {grad.mean().item()}")
                    logger.debug(f"Std: {grad.std().item()}")
                    return grad
                values.register_hook(metric_hook)
        else:
            # Fall back to base implementation
            return super().compute_metric(points)
        
        # Create metric tensor with gradient tracking
        metric = MetricTensor(values=values, dimension=self.manifold_dim)
        
        # Add hook to track metric tensor gradients
        if metric.values.requires_grad:
            def metric_tensor_hook(grad):
                logger.debug(f"\n=== Metric Tensor Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            metric.values.register_hook(metric_tensor_hook)
        
        return metric
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the Riemannian framework to input data."""
        return self.forward(*args, **kwargs)
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass implementing the geometric computation."""
        return self.compute_metric(args[0])
        
    @property
    def structure(self) -> RiemannianStructure[Tensor]:
        """Get the underlying Riemannian structure."""
        return self
        
    def compute_riemann(self, points: Tensor) -> CurvatureTensor[Tensor]:
        """Compute Riemann curvature tensor at given points."""
        return self.compute_curvature(points)
        
    def get_metric_tensor(self, points: Tensor) -> Tensor:
        """Get raw metric tensor values at given points."""
        metric = self.compute_metric(points)
        return metric.values
        
    def get_christoffel_values(self, points: Tensor) -> Tensor:
        """Get raw Christoffel symbol values at given points."""
        christoffel = self.compute_christoffel(points)
        return christoffel.values
        
    def get_riemann_tensor(self, points: Tensor) -> Tensor:
        """Get raw Riemann tensor values at given points."""
        riemann = self.compute_curvature(points)
        return riemann.riemann
        
    def project_to_base(self, total_space: torch.Tensor) -> torch.Tensor:
        """Project from total space to base manifold.
        
        Args:
            total_space: Point in total space
            
        Returns:
            Projected point on base manifold
        """
        # Simple projection that preserves the first manifold_dim components
        return total_space[..., :self.manifold_dim]
        
    def compute_christoffel(self, points: Tensor) -> ChristoffelSymbols[Tensor]:
        """Compute Christoffel symbols using autograd with metric parameter gradients.
        
        Args:
            points: Points at which to compute Christoffel symbols
            
        Returns:
            ChristoffelSymbols object containing the computed symbols
        """
        # Get metric tensor with gradient tracking
        metric = self.compute_metric(points)
        metric_values = metric.values  # Shape: [batch_size, dim, dim]
        
        # Compute metric derivatives using autograd
        batch_size = points.shape[0]
        dim = self.manifold_dim
        
        # Initialize metric derivatives tensor
        metric_derivatives = torch.zeros(
            batch_size, dim, dim, dim,
            device=points.device,
            dtype=points.dtype
        )
        
        # For each coordinate, compute derivatives using autograd
        for k in range(dim):
            # Create graph for metric derivatives
            metric_k = metric_values[..., k]
            
            # Compute gradients with respect to points
            inputs = [points]
            if hasattr(self, 'metric_param') and self.metric_param is not None:
                inputs.append(self.metric_param)
            
            grads = torch.autograd.grad(
                metric_k.sum(),
                inputs,
                create_graph=True,
                allow_unused=True,
                retain_graph=True
            )
            
            # Handle point gradients
            point_grad = grads[0]
            if point_grad is not None:
                point_grad = point_grad.reshape(batch_size, dim, dim)
                metric_derivatives[..., k] = point_grad
            
            # Handle metric parameter gradients if present
            if len(grads) > 1 and grads[1] is not None:
                metric_grad = grads[1]
                # Reshape and accumulate metric gradients
                metric_grad = metric_grad.reshape(batch_size, dim, dim)
                metric_derivatives[..., k] = metric_derivatives[..., k] + metric_grad
        
        # Compute inverse metric with gradient tracking
        metric_inverse = torch.inverse(metric_values)
        
        # Pre-allocate Christoffel symbols tensor
        christoffel_values = torch.zeros(
            batch_size, dim, dim, dim,
            device=points.device,
            dtype=points.dtype
        )
        
        # Compute Christoffel symbols using the formula:
        # Γᵏᵢⱼ = 1/2 gᵏˡ (∂ᵢgⱼˡ + ∂ⱼgᵢˡ - ∂ˡgᵢⱼ)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # First term: gᵏˡ ∂ᵢgⱼˡ
                    term1 = torch.einsum('...l,...jl->...', metric_inverse[:, k, :], metric_derivatives[:, i, j, :])
                    
                    # Second term: gᵏˡ ∂ⱼgᵢˡ
                    term2 = torch.einsum('...l,...il->...', metric_inverse[:, k, :], metric_derivatives[:, j, i, :])
                    
                    # Third term: -gᵏˡ ∂ˡgᵢⱼ
                    term3 = -torch.einsum('...l,...ij->...', metric_inverse[:, k, :], metric_derivatives[:, :, i, j])
                    
                    # Combine terms without in-place operations
                    christoffel_values[:, k, i, j] = 0.5 * (term1 + term2 - term3)
        
        # Add hook to track Christoffel gradients
        if christoffel_values.requires_grad:
            def christoffel_hook(grad):
                logger.debug(f"\n=== Christoffel Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            christoffel_values.register_hook(christoffel_hook)
        
        # Ensure metric parameter gradients are connected
        if hasattr(self, 'metric_param') and self.metric_param is not None and self.metric_param.requires_grad:
            def metric_param_hook(grad):
                logger.debug(f"\n=== Metric Parameter Gradient Hook (from Christoffel) ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            self.metric_param.register_hook(metric_param_hook)
        
        return ChristoffelSymbols(
            values=christoffel_values,
            metric=metric,
            is_symmetric=True  # Guaranteed by construction
        )

@runtime_checkable
class RiemannianFramework(Protocol):
    """Protocol defining the framework for Riemannian geometric computations.
    
    This protocol extends the base RiemannianStructure with additional
    pattern-specific operations and validations.
    """
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the Riemannian framework to input data."""
        ...
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass implementing the geometric computation."""
        ...
        
    @property
    def structure(self) -> RiemannianStructure[Tensor]:
        """Get the underlying Riemannian structure."""
        ...

    def exp_map(self, point: Tensor, vector: Tensor) -> Tensor:
        """Compute exponential map at a point in a given direction."""
        ...
        
    def compute_metric(self, points: Tensor) -> MetricTensor[Tensor]:
        """Compute the metric tensor at given points."""
        ...
        
    def compute_christoffel(self, points: Tensor) -> ChristoffelSymbols[Tensor]:
        """Compute Christoffel symbols at given points."""
        ...
        
    def compute_riemann(self, points: Tensor) -> CurvatureTensor[Tensor]:
        """Compute Riemann curvature tensor at given points."""
        ...
        
    def get_metric_tensor(self, points: Tensor) -> Tensor:
        """Get raw metric tensor values at given points."""
        metric = self.compute_metric(points)
        return metric.values
        
    def get_christoffel_values(self, points: Tensor) -> Tensor:
        """Get raw Christoffel symbol values at given points."""
        christoffel = self.compute_christoffel(points)
        return christoffel.values
        
    def get_riemann_tensor(self, points: Tensor) -> Tensor:
        """Get raw Riemann tensor values at given points."""
        riemann = self.compute_riemann(points)
        return riemann.riemann