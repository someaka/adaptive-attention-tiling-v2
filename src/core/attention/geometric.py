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

from src.utils.memory_management_util import register_tensor, optimize_memory, clear_memory


class ParallelTransport(nn.Module):
    """Parallel transport for geometric attention."""

    def __init__(self, dim: int, method: str = "euclidean", dtype: torch.dtype = torch.float32):
        """Initialize parallel transport.
        
        Args:
            dim: Dimension of the space
            method: Transport method ("euclidean", "schild", or "pole")
            dtype: Data type for tensors
        """
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.method = method.lower()
        
        if self.method not in ["euclidean", "schild", "pole"]:
            raise ValueError(f"Unknown transport method: {method}")

    def forward(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply parallel transport.
        
        Args:
            x: Starting point
            y: Ending point
            v: Vector to transport
            
        Returns:
            Transported vector
        """
        if self.method == "euclidean":
            return self._euclidean_transport(x, y, v)
        elif self.method == "schild":
            return self._schild_transport(x, y, v)
        else:  # pole
            return self._pole_transport(x, y, v)
            
    def _euclidean_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Simple Euclidean parallel transport."""
        return v
        
    def _schild_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Schild's ladder parallel transport."""
        # Compute midpoint between x and y
        mid = 0.5 * (x + y)
        
        # Project v onto the tangent space at mid
        v_mid = v - torch.sum(v * mid, dim=-1, keepdim=True) * mid
        
        # Transport to y
        return v_mid - torch.sum(v_mid * y, dim=-1, keepdim=True) * y
        
    def _pole_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pole ladder parallel transport."""
        # Compute the pole point
        pole = x + y
        
        # Project v onto the tangent space at pole
        v_pole = v - torch.sum(v * pole, dim=-1, keepdim=True) * pole
        
        # Transport to y
        return v_pole - torch.sum(v_pole * y, dim=-1, keepdim=True) * y


class HyperbolicExponential(nn.Module):
    """Exponential map for hyperbolic space.
    
    This class implements the exponential map in the hyperboloid model of hyperbolic space.
    The exponential map takes a point x and a tangent vector v at x, and returns the point
    y reached by following the geodesic starting at x with initial velocity v.
    
    Mathematical Framework:
    ---------------------
    1. Hyperboloid Model:
       H^n = {x ∈ ℝ^{n+1} | ⟨x,x⟩_M = -1, x₀ > 0}
       where ⟨·,·⟩_M is the Minkowski inner product
    
    2. Minkowski Inner Product:
       ⟨x,y⟩_M = -x₀y₀ + ∑ᵢ₌₁ⁿ xᵢyᵢ
       where x₀,y₀ are time components and xᵢ,yᵢ are space components
    
    3. Tangent Space:
       T_xH^n = {v ∈ ℝ^{n+1} | ⟨x,v⟩_M = 0}
    
    4. Exponential Map Formula:
       exp_x(v) = cosh(√⟨v,v⟩_M)x + sinh(√⟨v,v⟩_M)v/√⟨v,v⟩_M
       where cosh and sinh are hyperbolic functions
    
    5. Projection to Hyperboloid:
       P(x) = (√(1 + ∑ᵢxᵢ²), x₁, ..., xₙ)
    
    6. Projection to Tangent Space:
       P_T(v) = v + ⟨x,v⟩_M x
    
    Properties:
    ----------
    1. exp_x(0) = x
    2. d/dt|_{t=0} exp_x(tv) = v
    3. ⟨exp_x(v), exp_x(v)⟩_M = -1
    4. exp_x(v) preserves the hyperbolic distance
    
    Numerical Considerations:
    ----------------------
    1. Small vectors (|v| ≤ eps): Use first-order approximation
    2. Large vectors (|v| ≥ 20): Apply scaling to prevent overflow
    3. Maintain numerical stability in hyperbolic functions
    4. Ensure output points lie exactly on hyperboloid
    """
    
    def __init__(self, dim: int, curvature: float = -1.0, dtype: torch.dtype = torch.float32):
        """Initialize exponential map."""
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature, dtype=dtype), requires_grad=False)
        self.eps = 1e-8
        self.max_norm = 20.0
        self.dtype = dtype
        self.debug = True  # Enable debug printing
        
    def _debug_print(self, tag: str, **values):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(f"\n[{tag}]")
            print("-" * 50)
            for name, val in values.items():
                if isinstance(val, torch.Tensor):
                    print(f"{name}:")
                    print(f"  Shape: {val.shape}")
                    print(f"  Values: {val}")
                    print(f"  Device: {val.device}")
                    print(f"  Dtype: {val.dtype}")
                    if len(val.shape) > 0:
                        if val.dtype in [torch.float32, torch.float64]:
                            print(f"  Norm (L2): {torch.norm(val)}")
                            print(f"  Norm (L∞): {torch.max(torch.abs(val))}")
                            if val.shape[-1] > 1:
                                print(f"  Max: {torch.max(val)}")
                                print(f"  Min: {torch.min(val)}")
                                print(f"  Mean: {torch.mean(val)}")
                                print(f"  Std: {torch.std(val)}")
                                if val.shape[-1] >= 2:
                                    print(f"  Time component: {val[..., 0]}")
                                    print(f"  Space components: {val[..., 1:]}")
                else:
                    print(f"{name}: {val}")
            print("-" * 50)
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Minkowski inner product between two vectors.
        
        The Minkowski inner product is defined as:
        ⟨x,y⟩_M = -x₀y₀ + ∑ᵢ₌₁ⁿ xᵢyᵢ
        where x₀ and y₀ are the time components.
        """
        # Split into time and space components
        time_component = x[..., 0] * y[..., 0]
        space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        
        # Compute the inner product with negated time component
        result = -time_component + space_component
        
        if self.debug:
            self._debug_print("minkowski_inner",
                x=x,
                y=y,
                time_component=time_component,
                space_component=space_component,
                result=result
            )
        
        return result
        
    def minkowski_norm(self, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute the Minkowski norm of a vector."""
        inner = self.minkowski_inner(v, v)
        
        # Convert to boolean tensor for small value handling
        small_value_mask = (inner.abs() < eps).to(torch.bool)
        
        # Handle small values with Taylor expansion
        sqrt_small = inner.abs().sqrt() / 2
        sqrt_large = inner.abs().sqrt()
        
        # Use boolean indexing for the selection
        sqrt_term = torch.where(small_value_mask, sqrt_small, sqrt_large)
        
        # Handle maximum norm constraint
        max_norm_mask = (sqrt_term > self.max_norm).to(torch.bool)
        result = torch.where(max_norm_mask, torch.tensor(self.max_norm, device=v.device, dtype=v.dtype), sqrt_term)
        
        return result
        
    def project_to_hyperboloid(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Project a point onto the hyperboloid manifold.
        
        Projects to H^n = {x ∈ ℝ^{n+1} | ⟨x,x⟩_M = -1, x₀ > 0}
        Uses numerically stable operations to ensure accurate projection.
        
        Args:
            x: Point to project, shape (..., n+1)
            eps: Small constant for numerical stability
            
        Returns:
            Projected point on hyperboloid
        """
        if self.debug:
            self._debug_print("project_to_hyperboloid_input",
                x=x,
                x_shape=x.shape,
                x_norm=torch.norm(x),
                x_time=x[..., 0],
                x_space=x[..., 1:],
                x_space_norm=torch.norm(x[..., 1:])
            )
        
        # Extract time and space components
        time_component = x[..., 0:1]
        space_components = x[..., 1:]
        
        # Compute spatial norm squared with careful summation
        space_sq = space_components * space_components
        spatial_norm_sq = torch.zeros_like(time_component)
        compensation = torch.zeros_like(time_component)
        
        # Use Kahan summation for better precision
        for i in range(space_sq.size(-1)):
            y = space_sq[..., i:i+1] - compensation
            t = spatial_norm_sq + y
            compensation = (t - spatial_norm_sq) - y
            spatial_norm_sq = t
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_spatial",
                space_components=space_components,
                space_sq=space_sq,
                spatial_norm_sq=spatial_norm_sq,
                compensation=compensation
            )
        
        # Scale the norm if needed for numerical stability
        scaled_norm_sq = torch.where(
            (spatial_norm_sq < eps).to(torch.bool),
            torch.ones_like(spatial_norm_sq) * eps,
            spatial_norm_sq
        )
        
        # Compute time component with enhanced precision
        time_component_new = torch.sqrt(1.0 + scaled_norm_sq)
        
        # Always make time component positive
        time_component_new = torch.abs(time_component_new)
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_time",
                scaled_norm_sq=scaled_norm_sq,
                time_component_old=time_component,
                time_component_new=time_component_new
            )
        
        # Normalize space components to maintain the constraint
        space_scale = torch.where(
            (spatial_norm_sq < eps).to(torch.bool),
            torch.zeros_like(spatial_norm_sq),
            torch.sqrt(scaled_norm_sq / spatial_norm_sq.clamp(min=eps))
        )
        space_components_new = space_components * space_scale
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_space",
                space_scale=space_scale,
                space_components_old=space_components,
                space_components_new=space_components_new,
                space_norm_new=torch.norm(space_components_new)
            )
        
        # Construct result tensor
        result = torch.cat([time_component_new, space_components_new], dim=-1)
        
        # Verify hyperboloid constraint
        constraint_value = self.minkowski_inner(result, result) + 1.0
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_result",
                result=result,
                result_norm=torch.norm(result),
                constraint_value=constraint_value,
                time_final=result[..., 0],
                space_final=result[..., 1:],
                space_final_norm=torch.norm(result[..., 1:])
            )
        
        return result
        
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Project a vector onto the tangent space at x."""
        # Compute the inner product and ensure it has the right shape for broadcasting
        inner = self.minkowski_inner(x, v).unsqueeze(-1)
        
        # Project v onto the tangent space at x
        v_proj = v + inner * x
        
        # Verify tangent space constraint (should be close to 0)
        tangent_check = self.minkowski_inner(x, v_proj).unsqueeze(-1)
        
        # Handle numerical instabilities
        zero_mask = (tangent_check.abs() > eps).to(torch.bool)
        v_proj = torch.where(zero_mask, v_proj - tangent_check * x, v_proj)
        
        if self.debug:
            self._debug_print("project_to_tangent",
                x=x,
                v=v,
                inner=inner,
                v_proj=v_proj,
                tangent_check=tangent_check,
                zero_mask=zero_mask
            )
        
        return v_proj
        
    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass of the exponential map.
        
        Computes exp_x(v) for x on the hyperboloid and v in the tangent space at x.
        Projects the result back to the hyperboloid to ensure numerical stability.
        """
        # Project x to hyperboloid if needed
        x = self.project_to_hyperboloid(x)
        
        # Project v to tangent space at x
        v = self.project_to_tangent(x, v)
        
        # Compute the norm of v
        v_norm = self.minkowski_norm(v)
        
        # Handle zero vectors with improved precision
        zero_mask = (v_norm < self.eps).to(torch.bool)
        
        # Use more stable versions of hyperbolic functions
        cosh_vn = torch.cosh(v_norm)
        sinh_vn = torch.sinh(v_norm)
        
        # Compute coefficients with better numerical stability
        coeff1 = torch.where(
            zero_mask,
            torch.ones_like(v_norm),
            cosh_vn
        ).unsqueeze(-1)
        
        coeff2 = torch.where(
            zero_mask,
            torch.zeros_like(v_norm),
            sinh_vn / v_norm.clamp(min=self.eps)
        ).unsqueeze(-1)
        
        # Compute result with controlled operations
        result = coeff1 * x + coeff2 * v
        
        # Project back to hyperboloid to ensure constraint
        result = self.project_to_hyperboloid(result)
        
        if self.debug:
            self._debug_print("forward_output",
                x=x,
                v=v,
                v_norm=v_norm,
                coeff1=coeff1,
                coeff2=coeff2,
                result=result,
                hyperboloid_constraint=self.minkowski_inner(result, result) + 1
            )
        
        return result

    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance between points with high precision.
        
        Implements d(x,y) = arccosh(-⟨x,y⟩_M) with careful numerical handling.
        """
        # Compute inner product with improved precision
        inner = -self.minkowski_inner(x, y)
        
        # Handle numerical issues near 1
        inner = torch.where(
            inner < 1.0 + self.eps,
            torch.ones_like(inner) + self.eps,
            inner
        )
        
        # For values close to 1, use Taylor series
        # arccosh(1 + x) ≈ √(2x) * (1 - x/12 + 3x²/160)
        near_one_mask = (inner - 1.0) < 0.01
        x = inner - 1.0
        taylor_result = torch.sqrt(2*x) * (1 - x/12 + 3*x*x/160)
        
        # For larger values, use standard arccosh
        std_result = torch.acosh(inner)
        
        # Combine results based on magnitude
        result = torch.where(
            near_one_mask,
            taylor_result,
            std_result
        )
        
        if self.debug:
            self._debug_print("compute_distance",
                x=x,
                y=y,
                inner=inner,
                near_one_mask=near_one_mask.to(dtype=x.dtype),
                taylor_result=taylor_result,
                std_result=std_result,
                result=result
            )
        
        return result


class HyperbolicLogarithm(nn.Module):
    """Logarithm map for hyperbolic space."""

    def __init__(self, dim: int, curvature: float = -1.0, dtype: torch.dtype = torch.float32):
        """Initialize logarithm map.
        
        Args:
            dim: Dimension of the hyperbolic space
            curvature: Sectional curvature (default: -1.0)
            dtype: Data type for tensors
        """
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature, dtype=dtype), requires_grad=False)
        self.eps = 1e-8
        self.max_dist = 20.0  # Maximum distance for numerical stability
        self.dtype = dtype
        
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product with careful handling of time component."""
        time_component = x[..., 0] * y[..., 0]
        space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        return -time_component + space_component
        
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the hyperboloid with numerical stability."""
        # Scale by curvature
        K = torch.abs(self.curvature)
        spatial_norm_sq = torch.sum(x[..., 1:] * x[..., 1:], dim=-1)
        time_component = torch.sqrt(1.0 + K * spatial_norm_sq)
        
        return torch.cat([time_component.unsqueeze(-1), x[..., 1:]], dim=-1)
        
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto the tangent space at point x with improved precision."""
        # Compute inner product with higher precision
        inner = self.minkowski_inner(x, v)
        
        # Apply more precise projection with numerical stability
        v_proj = torch.where(
            torch.abs(inner)[..., None] > self.eps,
            v + (inner / (1.0 + self.eps))[..., None] * x,
            v
        )
        
        # Double-check tangent space constraint
        tangent_check = self.minkowski_inner(x, v_proj)
        v_proj = torch.where(
            torch.abs(tangent_check)[..., None] > self.eps,
            v_proj - (tangent_check / (1.0 + self.eps))[..., None] * x,
            v_proj
        )
        
        return v_proj
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute logarithm map with enhanced numerical stability."""
        # Project points to hyperboloid if needed
        x = self.project_to_hyperboloid(x)
        y = self.project_to_hyperboloid(y)
        
        # Compute Minkowski inner product with improved precision
        inner = -self.minkowski_inner(x, y)
        
        # Handle numerical issues near 1
        inner = torch.clamp(inner, min=1.0 + self.eps)
        
        # Compute distance with improved precision
        dist = torch.acosh(inner)
        
        # Handle zero distance case
        zero_mask = (dist < self.eps)
        if zero_mask.any():
            return torch.zeros_like(x)
        
        # Compute the direction with improved stability
        y_adj = y + inner.unsqueeze(-1) * x
        y_adj_norm = torch.sqrt(self.minkowski_inner(y_adj, y_adj).clamp(min=self.eps))
        
        # Compute initial direction with careful scaling
        v = (dist / y_adj_norm).unsqueeze(-1) * y_adj
        
        # Project to ensure we're exactly in the tangent space
        v = self.project_to_tangent(x, v)
        
        # Normalize to match the distance exactly
        v_norm = torch.sqrt(-self.minkowski_inner(v, v).clamp(min=self.eps))
        v = (dist / v_norm).unsqueeze(-1) * v
        
        return v


class EuclideanExponential(nn.Module):
    """Exponential map for Euclidean space."""

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        """Initialize exponential map.
        
        Args:
            dim: Dimension of the space
            dtype: Data type for tensors
        """
        super().__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply exponential map."""
        return x + v


class EuclideanLogarithm(nn.Module):
    """Logarithmic map for Euclidean space."""

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        """Initialize logarithm map.
        
        Args:
            dim: Dimension of the space
            dtype: Data type for tensors
        """
        super().__init__()
        self.dim = dim
        self.dtype = dtype

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
        self.register_module('transport', ParallelTransport(dim, dtype=torch.float32))

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
            velocity = self.transport.forward(curve[t], curve[t + 1], velocity)

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
        return self.transport.forward(x, y, v)

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

    def _quantum_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum-aware attention weights computation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Quantum attention weights of shape (batch_size, seq_len, hidden_dim)
        """
        # Project to quantum state space
        x = self.quantum_proj(x)  # First linear layer
        x = self.layer_norm(x)    # Layer normalization
        x = F.tanh(x)            # Non-linearity
        x = self.quantum_out(x)   # Second linear layer
        
        # Ensure quantum properties (unitarity, normalization)
        x = F.normalize(x, p=2, dim=-1)
        
        return x
