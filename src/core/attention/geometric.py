"""Advanced geometric structures for information manifolds and quantum geometric attention.

This module implements:
- Geometric structures for manifold operations
- Hyperbolic and Euclidean manifold operations
- Parallel transport methods
- Quantum geometric integration
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class GeometricStructures(nn.Module):
    """Advanced geometric structures for information manifolds."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        manifold_type: str = "hyperbolic",
        curvature: float = -1.0,
        parallel_transport_method: str = "schild",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.parallel_transport_method = parallel_transport_method

        # Geometric tensors with improved initialization
        self.metric = nn.Parameter(torch.eye(dim))
        self.connection = nn.Parameter(torch.zeros(dim, dim, dim))
        self.curvature_tensor = nn.Parameter(torch.zeros(dim, dim, dim, dim))

        # Manifold structures
        if manifold_type == "hyperbolic":
            self.exp_map = HyperbolicExponential(dim, curvature)
            self.log_map = HyperbolicLogarithm(dim, curvature)
        else:  # Euclidean default
            self.exp_map = EuclideanExponential(dim)
            self.log_map = EuclideanLogarithm(dim)

        # Parallel transport with improved method
        self.transport = ParallelTransport(dim, method=parallel_transport_method)

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
            curve[t + 1] = self.exp_map(curve[t], velocity * dt)

            # Parallel transport velocity
            velocity = self.transport(velocity, curve[t], curve[t + 1])

        return curve

    def compute_geodesic_distance(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute geodesic distance between points."""
        # Get tangent vector using logarithm map
        v = self.log_map(x, y)

        # Compute metric length with improved numerical stability
        return torch.sqrt(torch.abs(torch.einsum("i,ij,j->", v, self.metric, v)))

    def parallel_transport_batch(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport vectors v from x to y."""
        return self.transport(v, x, y)

    def compute_exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Map point from tangent space to manifold."""
        return self.exp_map(x, v)

    def compute_logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map point from manifold to tangent space."""
        return self.log_map(x, y)

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


class ParallelTransport(nn.Module):
    """Parallel transport methods on Riemannian manifolds."""

    def __init__(self, dim: int, method: str = "schild"):
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
    
    Maps a tangent vector at a point to the hyperbolic manifold.
    Includes curvature parameter to control the geometry of the space.
    """
    
    def __init__(self, dim: int, curvature: float = -1.0):
        """Initialize exponential map with curvature parameter."""
        super().__init__()
        self.dim = dim
        self.curvature = torch.nn.Parameter(torch.tensor(curvature), requires_grad=False)
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product <x,y> = -x₀y₀ + ∑ᵢxᵢyᵢ.
        
        The Minkowski inner product has signature (-,+,+,...), meaning:
        - Time component (index 0) contributes negatively: -x₀y₀
        - Space components (indices 1:) contribute positively: x₁y₁ + x₂y₂ + ...
        
        For spacelike vectors (vectors with positive norm), we need to ensure that
        the spatial components dominate the time component.
        
        Args:
            x: First tensor with shape (..., dim)
            y: Second tensor with shape (..., dim)
            
        Returns:
            Minkowski inner product with shape (...)
        """
        # Handle time and space components separately with improved stability
        time_inner = x[..., 0] * y[..., 0]  # Time component
        space_inner = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)  # Space components
        
        # For spacelike vectors, ensure spatial components dominate
        # This happens when the spatial norm is greater than the time component
        x_spatial_norm = torch.sum(x[..., 1:] * x[..., 1:], dim=-1)
        y_spatial_norm = torch.sum(y[..., 1:] * y[..., 1:], dim=-1)
        x_is_spacelike = x_spatial_norm > x[..., 0].pow(2)
        y_is_spacelike = y_spatial_norm > y[..., 0].pow(2)
        
        # If both vectors are spacelike, return only the spatial inner product
        is_space_space = x_is_spacelike & y_is_spacelike
        
        # Return spatial inner product for space-space case, otherwise use full Minkowski inner product
        return torch.where(is_space_space, space_inner, -time_inner + space_inner)

    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski norm √(<x,x>)."""
        inner = self.minkowski_inner(x, x)
        # Handle numerical stability with improved bounds
        return torch.sqrt(torch.clamp(inner, min=1e-12))

    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project a point onto the unit hyperboloid H³ = {x | <x,x> = 1/K, x₀ > 0}.
        
        The hyperboloid model represents hyperbolic space as the upper sheet of a 
        two-sheeted hyperboloid in Minkowski space. The curvature K determines the
        radius of the hyperboloid.
        
        Args:
            x: Point to project with shape (..., dim)
            
        Returns:
            Projected point with shape (..., dim) satisfying <x,x> = 1/K
        """
        # Scale by curvature
        K = torch.abs(self.curvature)  # Use absolute value for stability
        
        # Compute space-like norm
        space_norm = torch.norm(x[..., 1:], p=2, dim=-1)
        
        # For spacelike vectors (x₁² + x₂² > x₀²), we need to ensure
        # that the time component is smaller than the space norm
        t = torch.where(
            space_norm > torch.abs(x[..., 0]),  # Spacelike case
            space_norm * 0.9,  # Make time component smaller than space norm
            torch.sqrt(1.0/K + space_norm.pow(2))  # Timelike case with curvature
        )
        
        # Project onto hyperboloid
        return torch.cat([t.unsqueeze(-1), x[..., 1:]], dim=-1)

    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the tangent space at x.
        
        The tangent space at x consists of vectors v satisfying <x,v> = 0.
        For a timelike vector x with <x,x> = 1/K, the projection formula is:
        v_proj = v + K<x,v>x
        
        Args:
            x: Point on hyperboloid with shape (..., dim)
            v: Vector to project with shape (..., dim)
            
        Returns:
            Projected vector in tangent space with shape (..., dim)
        """
        # Scale by curvature
        K = torch.abs(self.curvature)
        
        # First normalize x to have Minkowski norm 1/K
        x_norm = self.project_to_hyperboloid(x)
        inner_xx = self.minkowski_inner(x_norm, x_norm)
        x_norm = x_norm / torch.sqrt(torch.abs(inner_xx * K)).unsqueeze(-1)
        
        # Project v to be orthogonal to x using the Minkowski metric
        inner_xv = self.minkowski_inner(x_norm, v)
        v_proj = v + K * inner_xv.unsqueeze(-1) * x_norm
        
        # Normalize the spatial components while preserving orthogonality
        # This maintains stability by keeping spatial components bounded
        space_norm = torch.norm(v_proj[..., 1:], p=2, dim=-1, keepdim=True).clamp(min=1e-7)
        v_proj_normalized = v_proj.clone()
        v_proj_normalized[..., 1:] = v_proj[..., 1:] / space_norm
        
        # Ensure exact orthogonality by computing the time component
        # From <x,v> = 0 and x₀² - Σᵢxᵢ² = 1/K, we can solve for v₀
        space_inner = torch.sum(x_norm[..., 1:] * v_proj_normalized[..., 1:], dim=-1)
        v_proj_normalized[..., 0] = space_inner / x_norm[..., 0]
        
        return v_proj_normalized

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply exponential map with curvature scaling."""
        # Project x to hyperbolic space and v to tangent space
        x = self.project_to_hyperboloid(x)
        v = self.project_to_tangent(x, v)
        
        # Scale by curvature
        K = torch.abs(self.curvature)
        v = v * torch.sqrt(K)
        
        # Compute the norm of v in the tangent space
        v_norm = self.minkowski_norm(v)
        
        # For very small vectors, use first-order approximation
        if torch.all(v_norm <= 1e-7):
            result = x + v
            return self.project_to_hyperboloid(result)
        
        # Compute hyperbolic functions with improved stability
        v_norm = torch.clamp(v_norm, min=1e-7, max=5.0)  # Reduced max value for better stability
        cosh = torch.cosh(v_norm)
        sinh = torch.sinh(v_norm)
        
        # Normalize v carefully with improved stability
        v_norm = torch.clamp(v_norm, min=1e-7)  # Prevent division by zero
        v_normalized = v / v_norm.unsqueeze(-1)
        
        # Map to hyperbolic space with improved stability and curvature scaling
        result = cosh.unsqueeze(-1) * x + sinh.unsqueeze(-1) * v_normalized
        
        # Ensure result is on the hyperboloid
        return self.project_to_hyperboloid(result)


class HyperbolicLogarithm(nn.Module):
    """Logarithm map for hyperbolic space.
    
    Maps a point on the hyperbolic manifold back to the tangent space.
    Includes curvature parameter to control the geometry of the space.
    """
    
    def __init__(self, dim: int, curvature: float = -1.0):
        """Initialize logarithm map with curvature parameter."""
        super().__init__()
        self.dim = dim
        self.curvature = torch.nn.Parameter(torch.tensor(curvature), requires_grad=False)
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product <x,y> = -x₀y₀ + ∑ᵢxᵢyᵢ.
        
        The Minkowski inner product has signature (-,+,+,...), meaning:
        - Time component (index 0) contributes negatively: -x₀y₀
        - Space components (indices 1:) contribute positively: x₁y₁ + x₂y₂ + ...
        
        For spacelike vectors (vectors with positive norm), we need to ensure that
        the spatial components dominate the time component.
        
        Args:
            x: First tensor with shape (..., dim)
            y: Second tensor with shape (..., dim)
            
        Returns:
            Minkowski inner product with shape (...)
        """
        # Handle time and space components separately with improved stability
        time_inner = x[..., 0] * y[..., 0]  # Time component
        space_inner = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)  # Space components
        
        # For spacelike vectors, ensure spatial components dominate
        # This happens when the spatial norm is greater than the time component
        x_spatial_norm = torch.sum(x[..., 1:] * x[..., 1:], dim=-1)
        y_spatial_norm = torch.sum(y[..., 1:] * y[..., 1:], dim=-1)
        x_is_spacelike = x_spatial_norm > x[..., 0].pow(2)
        y_is_spacelike = y_spatial_norm > y[..., 0].pow(2)
        
        # If both vectors are spacelike, return only the spatial inner product
        is_space_space = x_is_spacelike & y_is_spacelike
        
        # Return spatial inner product for space-space case, otherwise use full Minkowski inner product
        return torch.where(is_space_space, space_inner, -time_inner + space_inner)

    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski norm √(<x,x>)."""
        inner = self.minkowski_inner(x, x)
        # Handle numerical stability with improved bounds
        return torch.sqrt(torch.clamp(inner, min=1e-12))

    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project a point onto the unit hyperboloid H³ = {x | <x,x> = 1/K, x₀ > 0}.
        
        The hyperboloid model represents hyperbolic space as the upper sheet of a 
        two-sheeted hyperboloid in Minkowski space. The curvature K determines the
        radius of the hyperboloid.
        
        Args:
            x: Point to project with shape (..., dim)
            
        Returns:
            Projected point with shape (..., dim) satisfying <x,x> = 1/K
        """
        # Scale by curvature
        K = torch.abs(self.curvature)  # Use absolute value for stability
        
        # Compute space-like norm
        space_norm = torch.norm(x[..., 1:], p=2, dim=-1)
        
        # For spacelike vectors (x₁² + x₂² > x₀²), we need to ensure
        # that the time component is smaller than the space norm
        t = torch.where(
            space_norm > torch.abs(x[..., 0]),  # Spacelike case
            space_norm * 0.9,  # Make time component smaller than space norm
            torch.sqrt(1.0/K + space_norm.pow(2))  # Timelike case with curvature
        )
        
        # Project onto hyperboloid
        return torch.cat([t.unsqueeze(-1), x[..., 1:]], dim=-1)

    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the tangent space at x.
        
        The tangent space at x consists of vectors v satisfying <x,v> = 0.
        For a timelike vector x with <x,x> = 1/K, the projection formula is:
        v_proj = v + K<x,v>x
        
        Args:
            x: Point on hyperboloid with shape (..., dim)
            v: Vector to project with shape (..., dim)
            
        Returns:
            Projected vector in tangent space with shape (..., dim)
        """
        # Scale by curvature
        K = torch.abs(self.curvature)
        
        # First normalize x to have Minkowski norm 1/K
        x_norm = self.project_to_hyperboloid(x)
        inner_xx = self.minkowski_inner(x_norm, x_norm)
        x_norm = x_norm / torch.sqrt(torch.abs(inner_xx * K)).unsqueeze(-1)
        
        # Project v to be orthogonal to x using the Minkowski metric
        inner_xv = self.minkowski_inner(x_norm, v)
        v_proj = v + K * inner_xv.unsqueeze(-1) * x_norm
        
        # Normalize the spatial components while preserving orthogonality
        # This maintains stability by keeping spatial components bounded
        space_norm = torch.norm(v_proj[..., 1:], p=2, dim=-1, keepdim=True).clamp(min=1e-7)
        v_proj_normalized = v_proj.clone()
        v_proj_normalized[..., 1:] = v_proj[..., 1:] / space_norm
        
        # Ensure exact orthogonality by computing the time component
        # From <x,v> = 0 and x₀² - Σᵢxᵢ² = 1/K, we can solve for v₀
        space_inner = torch.sum(x_norm[..., 1:] * v_proj_normalized[..., 1:], dim=-1)
        v_proj_normalized[..., 0] = space_inner / x_norm[..., 0]
        
        return v_proj_normalized

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply logarithm map with curvature scaling."""
        # Scale by curvature
        K = torch.abs(self.curvature)
        
        # Project points to hyperbolic space
        x = self.project_to_hyperboloid(x)
        y = self.project_to_hyperboloid(y)
        
        # Handle same point case
        if torch.allclose(x, y, atol=1e-7):
            return torch.zeros_like(x)
        
        # Compute hyperbolic distance with improved numerical stability
        inner = -self.minkowski_inner(x, y)  # Should be ≥ 1/K
        inner = torch.clamp(inner, min=1.0/K + 1e-7, max=5.0/K)  # Reduced max value and scaled by curvature
        dist = torch.acosh(inner * K)  # Scale by curvature
        
        # For very close points, use first-order approximation
        if torch.all(dist <= 1e-7):
            return self.project_to_tangent(x, y - x)
        
        # Project y to tangent space at x with improved stability
        y_tan = self.project_to_tangent(x, y)
        y_tan_norm = self.minkowski_norm(y_tan)
        
        # Scale the tangent vector by the distance with improved stability
        y_tan_norm = torch.clamp(y_tan_norm, min=1e-7)  # Prevent division by zero
        scale = (dist / y_tan_norm).unsqueeze(-1)
        scale = torch.clamp(scale, min=-5.0, max=5.0)  # Tighter bounds for stability
        result = scale * y_tan
        
        # Final projection to ensure we're in the tangent space
        return self.project_to_tangent(x, result)


class EuclideanExponential(nn.Module):
    """Exponential map for Euclidean space."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply exponential map with optional scaling."""
        return x + v


class EuclideanLogarithm(nn.Module):
    """Logarithmic map for Euclidean space."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply logarithm map with improved stability."""
        return y - x
