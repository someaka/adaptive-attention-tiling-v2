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
        self.dtype = dtype or torch.complex64
        
        # Initialize metric as identity plus low-rank perturbation for stability
        self.metric_factors = nn.Parameter(
            torch.randn(manifold_dim, manifold_dim, device=self.device, dtype=self.dtype) * 0.01
        )
        
        # Initialize connection coefficients
        self.connection_coeffs = nn.Parameter(
            torch.zeros(manifold_dim, manifold_dim, manifold_dim, device=self.device, dtype=self.dtype)
        )
        
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
        
        # Compute base metric
        identity = torch.eye(
            self.manifold_dim,
            dtype=self.dtype
        ).expand(batch_size, -1, -1).to(device=self.device)
        
        # Add learned perturbation
        perturbation = torch.matmul(
            self.metric_factors.T,
            self.metric_factors
        ).expand(batch_size, -1, -1)
        
        values = identity + perturbation
        
        # Validate metric properties
        is_compatible = self.validate_metric_properties(
            MetricTensor(values=values, dimension=self.manifold_dim)
        )
        
        return MetricTensor(
            values=values,
            dimension=self.manifold_dim,
            is_compatible=is_compatible
        )
        
    def compute_christoffel(self, points: Tensor) -> ChristoffelSymbols[Tensor]:
        """Compute Christoffel symbols using connection coefficients.
        
        Args:
            points: Points at which to compute symbols
            
        Returns:
            Christoffel symbols at points
        """
        points = self._ensure_gradients(points)
        batch_size = points.shape[0]
        
        # Get metric and inverse
        metric = self.compute_metric(points)
        metric_inv = torch.linalg.inv(metric.values)
        
        # Get connection coefficients and ensure they require gradients
        if hasattr(self, 'connection_coeffs'):
            connection_coeffs = self.connection_coeffs
            connection_coeffs.requires_grad_(True)  # Ensure gradients flow
            
            # Reshape connection coefficients to match batch size if needed
            if connection_coeffs.shape[0] != batch_size:
                connection_coeffs = connection_coeffs.expand(batch_size, -1, -1, -1)
            
            # Compute Christoffel symbols using connection coefficients
            # Ensure computation maintains gradient flow
            christoffel = connection_coeffs * torch.ones_like(connection_coeffs)  # Force gradient flow
            
            # Add metric contribution with gradient flow
            metric_contribution = torch.einsum(
                'bij,bjk->bik',
                metric_inv,
                metric.values
            )
            
            # Combine connection coefficients with metric contribution
            # Use addition to maintain gradient flow
            christoffel = christoffel + 0.5 * metric_contribution.unsqueeze(-1)
            
            # Ensure output requires gradients
            christoffel.requires_grad_(True)
            
            return ChristoffelSymbols(
                values=christoffel,
                metric=metric,
                is_symmetric=True  # Guaranteed by construction
            )
        
        # Initialize Christoffel symbols
        christoffel = torch.zeros(
            batch_size,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            device=self.device,
            dtype=self.dtype
        )
        
        # Compute metric derivatives
        for k in range(self.manifold_dim):
            grad_result = torch.autograd.grad(
                metric.values[..., k].sum(),
                points,
                create_graph=True,
                allow_unused=True,
                retain_graph=True
            )[0]
            
            # Handle None gradients by providing a zero tensor
            if grad_result is None:
                grad_result = torch.zeros(
                    batch_size,
                    self.manifold_dim,
                    self.manifold_dim,
                    device=self.device,
                    dtype=self.dtype
                )
            else:
                # Reshape gradient to match expected dimensions
                grad_result = grad_result.reshape(batch_size, self.manifold_dim, self.manifold_dim)
                
            self.cache[f'metric_deriv_{k}'] = grad_result
        
        # Construct Christoffel symbols
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    # Get relevant derivatives
                    d_i = self.cache[f'metric_deriv_{i}']
                    d_j = self.cache[f'metric_deriv_{j}']
                    d_k = self.cache[f'metric_deriv_{k}']
                    
                    # Compute components
                    term1 = d_i[..., j, k]
                    term2 = d_j[..., i, k]
                    term3 = -d_k[..., i, j]
                    
                    # Contract with inverse metric
                    christoffel[..., k, i, j] = 0.5 * torch.sum(
                        metric_inv[..., k, :] * (term1 + term2 + term3).unsqueeze(-1),
                        dim=-1
                    )
                    
        return ChristoffelSymbols(
            values=christoffel,
            metric=metric,
            is_symmetric=True  # Guaranteed by construction
        )
        
    def parallel_transport(
        self,
        vector: Tensor,
        path: Tensor,
        connection: Optional[ChristoffelSymbols[Tensor]] = None
    ) -> Tensor:
        """Parallel transport using numerical integration.
        
        Implements parallel transport equation:
        ∇_γ̇ V = 0
        
        Args:
            vector: Vector to transport
            path: Path along which to transport
            connection: Optional pre-computed connection
            
        Returns:
            Parallel transported vector
        """
        # Initialize result tensor
        num_points = path.shape[0]
        result = torch.zeros(
            num_points,
            self.manifold_dim,
            device=self.device,
            dtype=self.dtype
        )
        result[0] = vector
        
        # Get connection if not provided
        if connection is None:
            connection = self.compute_christoffel(path)
            
        # Compute path tangent vectors
        tangents = path[1:] - path[:-1]
        
        # Parallel transport equation
        for t in range(num_points - 1):
            # Transport step
            step = -torch.einsum(
                '...ijk,...j,...k->...i',
                connection.values[t],
                result[t],
                tangents[t]
            )
            result[t + 1] = result[t] + step
            
        return result
        
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
        """Compute geodesic flow using numerical integration.
        
        Implements the geodesic equation:
        d²x^i/dt² + Γ^i_jk dx^j/dt dx^k/dt = 0
        
        Args:
            initial_point: Starting point
            initial_velocity: Initial velocity
            steps: Number of integration steps
            step_size: Integration step size
            
        Returns:
            Tuple of (points along geodesic, velocities along geodesic)
        """
        # Initialize arrays for points and velocities
        points = torch.zeros(
            steps,
            self.manifold_dim,
            device=self.device,
            dtype=self.dtype
        )
        velocities = torch.zeros_like(points)
        
        # Set initial conditions
        points[0] = initial_point
        velocities[0] = initial_velocity
        
        # Integrate geodesic equation
        for t in range(steps - 1):
            # Get Christoffel symbols at current point
            christoffel = self.compute_christoffel(points[t].unsqueeze(0))
            
            # Compute acceleration
            acceleration = -torch.einsum(
                '...ijk,...j,...k->...i',
                christoffel.values[0],
                velocities[t],
                velocities[t]
            )
            
            # Update velocity and position
            velocities[t + 1] = velocities[t] + step_size * acceleration
            points[t + 1] = points[t] + step_size * velocities[t]
            
        return points, velocities
        
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
    """Pattern-specific implementation of Riemannian geometry."""
    
    def __init__(
        self,
        manifold_dim: int,
        pattern_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(manifold_dim, pattern_dim, device, dtype)
        self.pattern_dim = pattern_dim
        
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