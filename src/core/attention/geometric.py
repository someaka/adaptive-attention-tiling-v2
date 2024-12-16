"""Advanced geometric structures for information manifolds and quantum geometric attention.

This module implements:
- Geometric structures for manifold operations
- Hyperbolic and Euclidean manifold operations
- Parallel transport methods
- Quantum geometric integration

Theoretical Foundations:
-----------------------
The implementation is based on the following mathematical framework:

1. Hyperbolic Space:
   The hyperboloid model represents hyperbolic space as the upper sheet of a 
   two-sheeted hyperboloid in Minkowski space:
   H³ = {x ∈ ℝ^{n,1} | ⟨x,x⟩ = -1, x₀ > 0}
   where ⟨·,·⟩ is the Minkowski inner product.

2. Core Operations:
   a) Minkowski Inner Product:
      ⟨x,y⟩ = -x₀y₀ + ∑ᵢxᵢyᵢ
      where x₀,y₀ are time components and xᵢ,yᵢ are space components

   b) Exponential Map:
      exp_x(v) = cosh(|v|)x + sinh(|v|)v/|v|
      where |v| is the Minkowski norm of v

   c) Logarithm Map:
      log_x(y) = d * (y + ⟨x,y⟩x) / √(⟨y + ⟨x,y⟩x, y + ⟨x,y⟩x⟩)
      where d = arccosh(-⟨x,y⟩)

3. Stability Considerations:
   - Proper handling of small norms (≤ 1e-7)
   - Clamping of values to prevent numerical instability
   - Separate handling of time and space components
   - Proper projection to hyperboloid and tangent space

4. Curvature Scaling:
   The curvature K affects the metric through scaling:
   g_K = K * g_{-1}
   where g_{-1} is the metric with K = -1

References:
----------
1. Information Geometry and Pattern Theory
2. Quantum Geometric Framework
3. Hyperbolic Neural Architectures
4. Geometric Deep Learning
"""

from typing import Optional, Dict, Tuple, Union
from typing_extensions import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.memory_management import register_tensor, optimize_memory, clear_memory


class ParallelTransport(nn.Module):
    """Parallel transport methods on Riemannian manifolds."""

    def __init__(self, dim: int, method: Literal["schild", "pole"] = "schild"):
        super().__init__()
        self.dim = dim
        self.method = method

    def forward(
        self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
        connection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transport vector v from x to y."""
        # Handle same point case
        if torch.allclose(x, y):
            return v

        if self.method == "schild":
            return self.schild_ladder(v, x, y, connection)
        return self.pole_ladder(v, x, y)

    def schild_ladder(
        self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
        connection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Schild's ladder parallel transport with improved stability."""
        # Handle zero vector case
        if torch.allclose(v, torch.zeros_like(v)):
            return torch.zeros_like(v)

        # Handle same point case
        if torch.allclose(x, y):
            return v

        # Compute midpoint with improved averaging
        m = 0.5 * (x + y)

        if connection is not None:
            # Transport v to m using connection with stability
            v_m = v - 0.5 * torch.einsum("ijk,j,k->i", connection, y - x, v)
            
            # Transport from m to y with stability
            return v_m - 0.5 * torch.einsum("ijk,j,k->i", connection, y - m, v_m)
        
        # Default transport without connection
        return v

    def pole_ladder(
        self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Pole ladder parallel transport with improved stability."""
        # Handle zero vector case
        if torch.allclose(v, torch.zeros_like(v)):
            return torch.zeros_like(v)

        # Handle same point case
        if torch.allclose(x, y):
            return v

        # Compute geodesic midpoint with improved stability
        m = 0.5 * (x + y)

        # Compute pole point with proper normalization
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_normalized = v / (v_norm + 1e-8)
        p = m + v_normalized

        # Double reflection with improved numerics and stability
        v1 = 2 * (p - x)
        v2 = 2 * (y - p)

        # Preserve norm of original vector
        result = v2 - v1
        result = result * (torch.norm(v) / torch.norm(result).clamp(min=1e-8))

        return result


class HyperbolicExponential(nn.Module):
    """Exponential map for hyperbolic space.
    
    This class implements the exponential map in the hyperboloid model of hyperbolic space.
    The exponential map takes a point x and a tangent vector v at x, and returns the point
    y reached by following the geodesic starting at x with initial velocity v.
    
    Key Operations:
    1. Minkowski Inner Product: ⟨x,y⟩ = -x₀y₀ + ∑ᵢxᵢyᵢ
    2. Exponential Map: exp_x(v) = cosh(|v|)x + sinh(|v|)v/|v|
    3. Projection to Hyperboloid: H³ = {x | ⟨x,x⟩ = -1, x₀ > 0}
    4. Projection to Tangent Space: T_xH³ = {v | ⟨x,v⟩ = 0}
    
    The implementation includes careful handling of numerical stability and edge cases.
    """
    
    def __init__(self, dim: int, curvature: float = -1.0):
        """Initialize exponential map.
        
        Args:
            dim: Dimension of the hyperbolic space
            curvature: Sectional curvature (default: -1.0)
        """
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature), requires_grad=False)
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product with improved numerical stability."""
        with optimize_memory("minkowski_inner"):
            time_inner = register_tensor(x[..., 0] * y[..., 0], "minkowski_inner")
            space_inner = register_tensor(torch.sum(x[..., 1:] * y[..., 1:], dim=-1), "minkowski_inner")
            return register_tensor(-time_inner + space_inner, "minkowski_inner")
    
    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski norm with improved numerical stability."""
        with optimize_memory("minkowski_norm"):
            inner = register_tensor(self.minkowski_inner(x, x), "minkowski_norm")
            return register_tensor(torch.sqrt(torch.clamp(inner, min=1e-12)), "minkowski_norm")
    
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project a point onto the hyperboloid model of hyperbolic space."""
        with optimize_memory("project_hyperboloid"):
            # Compute space-like norm
            space_norm = register_tensor(torch.norm(x[..., 1:], p=2, dim=-1), "project_hyperboloid")
            
            # Normalize spatial components
            x_spatial = register_tensor(x[..., 1:] / space_norm.unsqueeze(-1).clamp(min=1e-7), "project_hyperboloid")
            
            # Scale spatial components
            scale = register_tensor(
                torch.sqrt(space_norm.pow(2) / (1 + space_norm.pow(2))), 
                "project_hyperboloid"
            )
            x_spatial = register_tensor(x_spatial * scale.unsqueeze(-1), "project_hyperboloid")
            
            # Compute time component
            t = register_tensor(
                torch.sqrt(1 + torch.sum(x_spatial * x_spatial, dim=-1)), 
                "project_hyperboloid"
            )
            
            return register_tensor(torch.cat([t.unsqueeze(-1), x_spatial], dim=-1), "project_hyperboloid")
    
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the tangent space at x."""
        with optimize_memory("project_tangent"):
            # Project x to hyperboloid
            x = register_tensor(self.project_to_hyperboloid(x), "project_tangent")
            
            # Project v to be orthogonal to x
            inner = register_tensor(self.minkowski_inner(x, v), "project_tangent")
            v_proj = register_tensor(v + inner.unsqueeze(-1) * x, "project_tangent")
            
            # Normalize spatial components
            space_norm = register_tensor(
                torch.norm(v_proj[..., 1:], p=2, dim=-1, keepdim=True).clamp(min=1e-7),
                "project_tangent"
            )
            v_proj_normalized = v_proj.clone()
            v_proj_normalized[..., 1:] = register_tensor(v_proj[..., 1:] / space_norm, "project_tangent")
            
            # Compute time component for exact orthogonality
            space_inner = register_tensor(
                torch.sum(x[..., 1:] * v_proj_normalized[..., 1:], dim=-1),
                "project_tangent"
            )
            v_proj_normalized[..., 0] = register_tensor(space_inner / x[..., 0], "project_tangent")
            
            return register_tensor(v_proj_normalized, "project_tangent")
    
    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply exponential map with improved numerical stability."""
        with optimize_memory("exp_map"):
            # Convert to double precision for better accuracy
            x_double = x.to(torch.float64)
            v_double = v.to(torch.float64)
            
            # Project points to hyperbolic space
            x_double = register_tensor(self.project_to_hyperboloid(x_double), "exp_map")
            v_double = register_tensor(self.project_to_tangent(x_double, v_double), "exp_map")
            
            # Handle numerical stability
            eps = torch.finfo(torch.float64).eps
            
            # Compute vector norm with stability
            v_norm = register_tensor(self.minkowski_norm(v_double).clamp(min=eps), "exp_map")
            
            # For zero vectors, return x
            zero_mask = v_norm <= eps
            if torch.all(zero_mask):
                return register_tensor(x_double.to(x.dtype), "exp_map")
            
            # Compute hyperbolic functions with stability
            v_norm_safe = torch.where(zero_mask, torch.ones_like(v_norm), v_norm)
            v_norm_safe = v_norm_safe.clamp(min=eps, max=20.0)  # Prevent overflow in hyperbolic functions
            
            cosh_term = register_tensor(torch.cosh(v_norm_safe), "exp_map")
            sinh_term = register_tensor(torch.sinh(v_norm_safe), "exp_map")
            
            # Compute result using the exponential map formula with stability
            result = register_tensor(
                cosh_term.unsqueeze(-1) * x_double + 
                sinh_term.unsqueeze(-1) * v_double / v_norm_safe.unsqueeze(-1).clamp(min=eps),
                "exp_map"
            )
            
            # Handle zero vectors
            if torch.any(zero_mask):
                result = register_tensor(
                    torch.where(zero_mask.unsqueeze(-1), x_double, result),
                    "exp_map"
                )
            
            # Project back to hyperboloid for numerical stability
            result = register_tensor(self.project_to_hyperboloid(result), "exp_map")
            
            # Final stability check
            if torch.any(torch.isnan(result)):
                result = torch.where(
                    torch.isnan(result),
                    x_double,  # Use base point for any NaN values
                    result
                )
            
            # Convert back to original precision
            return register_tensor(result.to(x.dtype), "exp_map")


class HyperbolicLogarithm(nn.Module):
    """Logarithm map for hyperbolic space.
    
    This class implements the logarithm map in the hyperboloid model of hyperbolic space.
    The logarithm map is the inverse of the exponential map: given two points x,y on the
    hyperboloid, it returns the initial velocity v of the geodesic from x to y.
    
    Key Operations:
    1. Minkowski Inner Product: ⟨x,y⟩ = -x₀y₀ + ∑ᵢxᵢyᵢ
    2. Logarithm Map: log_x(y) = d * (y + ⟨x,y⟩x) / √(⟨y + ⟨x,y⟩x, y + ⟨x,y⟩x⟩)
       where d = arccosh(-⟨x,y⟩)
    3. Projection to Hyperboloid: H³ = {x | ⟨x,x⟩ = -1, x₀ > 0}
    4. Projection to Tangent Space: T_xH³ = {v | ⟨x,v⟩ = 0}
    
    The implementation includes careful handling of numerical stability and edge cases.
    """
    
    def __init__(self, dim: int, curvature: float = -1.0):
        """Initialize logarithm map.
        
        Args:
            dim: Dimension of the hyperbolic space
            curvature: Sectional curvature (default: -1.0)
        """
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature), requires_grad=False)
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product <x,y> = -x₀y₀ + ∑ᵢxᵢyᵢ."""
        time_inner = x[..., 0] * y[..., 0]
        space_inner = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        return -time_inner + space_inner
    
    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski norm √(<x,x>)."""
        inner = self.minkowski_inner(x, x)
        return torch.sqrt(torch.clamp(inner, min=1e-12))
    
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project a point onto the unit hyperboloid H³ = {x | <x,x> = -1, x₀ > 0}."""
        # Compute space-like norm
        space_norm = torch.norm(x[..., 1:], p=2, dim=-1)
        
        # Normalize spatial components
        x_spatial = x[..., 1:] / space_norm.unsqueeze(-1).clamp(min=1e-7)
        
        # Scale spatial components to ensure Minkowski norm is -1
        scale = torch.sqrt(space_norm.pow(2) / (1 + space_norm.pow(2)))
        x_spatial = x_spatial * scale.unsqueeze(-1)
        
        # Compute time component
        t = torch.sqrt(1 + torch.sum(x_spatial * x_spatial, dim=-1))
        
        return torch.cat([t.unsqueeze(-1), x_spatial], dim=-1)
    
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the tangent space at x."""
        # Project x to hyperboloid
        x = self.project_to_hyperboloid(x)
        
        # Project v to be orthogonal to x
        inner = self.minkowski_inner(x, v)
        v_proj = v + inner.unsqueeze(-1) * x
        
        # Normalize spatial components
        space_norm = torch.norm(v_proj[..., 1:], p=2, dim=-1, keepdim=True).clamp(min=1e-7)
        v_proj_normalized = v_proj.clone()
        v_proj_normalized[..., 1:] = v_proj[..., 1:] / space_norm
        
        # Compute time component for exact orthogonality
        space_inner = torch.sum(x[..., 1:] * v_proj_normalized[..., 1:], dim=-1)
        v_proj_normalized[..., 0] = space_inner / x[..., 0]
        
        return v_proj_normalized
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply logarithm map with improved numerical stability."""
        with optimize_memory("log_map"):
            # Project points to hyperbolic space
            x = register_tensor(self.project_to_hyperboloid(x), "log_map")
            y = register_tensor(self.project_to_hyperboloid(y), "log_map")
            
            # Handle numerical stability
            eps = torch.finfo(x.dtype).eps
            
            # Compute Minkowski inner product with stability
            inner = register_tensor(-self.minkowski_inner(x, y), "log_map")
            inner = register_tensor(torch.clamp(inner, min=1.0 + eps, max=20.0), "log_map")  # Prevent overflow
            
            # For identical points, return zero vector
            same_points = inner <= 1.0 + eps
            if torch.all(same_points):
                return register_tensor(torch.zeros_like(x), "log_map")
            
            # Compute hyperbolic distance with stability
            dist = register_tensor(torch.acosh(inner), "log_map")
            dist = register_tensor(dist.clamp(min=eps, max=20.0), "log_map")  # Prevent overflow
            
            # Compute direction vector y + ⟨x,y⟩x with stability
            direction = register_tensor(y + inner.unsqueeze(-1) * x, "log_map")
            
            # Compute norm of direction vector with stability
            direction_norm = register_tensor(
                torch.sqrt(self.minkowski_inner(direction, direction).clamp(min=eps)),
                "log_map"
            )
            
            # Compute result using the logarithm map formula with stability
            result = register_tensor(
                dist.unsqueeze(-1) * direction / direction_norm.unsqueeze(-1).clamp(min=eps),
                "log_map"
            )
            
            # Handle identical points
            if torch.any(same_points):
                result = register_tensor(
                    torch.where(same_points.unsqueeze(-1), torch.zeros_like(x), result),
                    "log_map"
                )
            
            # Project to tangent space for numerical stability
            result = register_tensor(self.project_to_tangent(x, result), "log_map")
            
            # Final stability check
            if torch.any(torch.isnan(result)):
                result = torch.where(
                    torch.isnan(result),
                    torch.zeros_like(result),  # Use zero vector for any NaN values
                    result
                )
            
            return result


class EuclideanExponential(nn.Module):
    """Exponential map for Euclidean space."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply exponential map."""
        return x + v


class EuclideanLogarithm(nn.Module):
    """Logarithmic map for Euclidean space."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply logarithm map."""
        return y - x


class GeometricStructures(nn.Module):
    """Advanced geometric structures for information manifolds."""

    exp_map: Union[HyperbolicExponential, EuclideanExponential]
    log_map: Union[HyperbolicLogarithm, EuclideanLogarithm]
    transport: ParallelTransport
    metric: nn.Parameter
    connection: nn.Parameter
    curvature_tensor: nn.Parameter

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        manifold_type: Literal["hyperbolic", "euclidean"] = "hyperbolic",
        curvature: float = -1.0,
        parallel_transport_method: Literal["schild", "pole"] = "schild",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.parallel_transport_method = parallel_transport_method

        # Geometric tensors with improved initialization
        self.register_parameter('metric', nn.Parameter(torch.eye(dim)))
        self.register_parameter('connection', nn.Parameter(torch.zeros(dim, dim, dim)))
        self.register_parameter('curvature_tensor', nn.Parameter(torch.zeros(dim, dim, dim, dim)))

        # Manifold structures
        if manifold_type == "hyperbolic":
            self.register_module('exp_map', HyperbolicExponential(dim, curvature))
            self.register_module('log_map', HyperbolicLogarithm(dim, curvature))
        else:  # Euclidean default
            self.register_module('exp_map', EuclideanExponential(dim))
            self.register_module('log_map', EuclideanLogarithm(dim))

        # Parallel transport with improved method
        self.register_module('transport', ParallelTransport(dim, method=parallel_transport_method))

    def compute_sectional_curvature(
        self, x: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
    ) -> torch.Tensor:
        """Compute sectional curvature in plane spanned by v1, v2."""
        # Compute Riemann tensor components with improved numerical stability
        riemann = torch.einsum("ijkl,i,j,k,l->", self.curvature_tensor, v1, v2, v1, v2)
        
        # Compute area of parallelogram with metric tensor
        area = torch.sqrt(
            torch.abs(
                torch.einsum("ij,i,j->", self.metric, v1, v1)
                * torch.einsum("ij,i,j->", self.metric, v2, v2)
                - (torch.einsum("ij,i,j->", self.metric, v1, v2)) ** 2
            )
        )
        
        return riemann / (area.clamp(min=1e-8) ** 2)

    def compute_geodesic(
        self, x: torch.Tensor, v: torch.Tensor, steps: int = 100
    ) -> torch.Tensor:
        """Compute geodesic curve starting at x with initial velocity v."""
        # Initialize geodesic
        curve = torch.zeros(steps + 1, *x.shape, device=x.device)
        curve[0] = x

        # Initial velocity
        velocity = v

        # Integrate geodesic equation with improved stability
        dt = 1.0 / steps
        for t in range(steps):
            # Update position using exponential map
            curve[t + 1] = self.exp_map.forward(curve[t], velocity * dt)

            # Parallel transport velocity
            velocity = self.transport.forward(velocity, curve[t], curve[t + 1])

        return curve

    def compute_geodesic_distance(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute geodesic distance between points."""
        # Project points to hyperbolic space
        if isinstance(self.exp_map, HyperbolicExponential):
            x_proj = self.exp_map.project_to_hyperboloid(x)
            y_proj = self.exp_map.project_to_hyperboloid(y)
            
            # For same points, return zero
            if torch.allclose(x_proj, y_proj, atol=1e-7):
                return torch.zeros(x.shape[:-1], device=x.device)
            
            # Compute hyperbolic distance directly using Minkowski inner product
            inner = -self.exp_map.minkowski_inner(x_proj, y_proj)  # Should be ≥ 1
            inner = torch.clamp(inner, min=1.0 + 1e-7)  # Ensure we stay in valid range
            
            # Return arccosh of inner product (true hyperbolic distance)
            return torch.acosh(inner)
        else:
            # For Euclidean case, use standard Euclidean distance
            return torch.norm(y - x, p=2, dim=-1)

    def parallel_transport_batch(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport vectors v from x to y."""
        return self.transport.forward(v, x, y)

    def compute_exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Map point from tangent space to manifold."""
        return self.exp_map.forward(x, v)

    def compute_logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map point from manifold to tangent space."""
        return self.log_map.forward(x, y)

    def process_points(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process points through geometric operations with quantum integration."""
        results = {}

        # Compute geodesic distance with improved metric
        distance = self.compute_geodesic_distance(x, y)
        results["distance"] = distance

        if return_diagnostics and v is not None:
            # Get orthogonal vector to v with Gram-Schmidt
            v_perp = torch.zeros_like(v)
            v_perp[..., 0] = -v[..., 1]
            v_perp[..., 1] = v[..., 0]
            v_perp = v_perp / torch.norm(v_perp, dim=-1, keepdim=True).clamp(min=1e-8)

            # Compute curvature with improved stability
            curvature = self.compute_sectional_curvature(x, v, v_perp)
            results["sectional_curvature"] = curvature

            # Add quantum geometric metrics
            if hasattr(self, "quantum_metric"):
                results["quantum_metric"] = torch.einsum(
                    "...i,...j,ij->...", v, v, self.quantum_metric
                )

        return results

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process points through geometric operations."""
        return self.process_points(x, y, v, return_diagnostics)
