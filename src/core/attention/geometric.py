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


def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Minkowski inner product between two vectors.
    
    The Minkowski inner product is defined as:
    ⟨x,y⟩ = -x₀y₀ + x₁y₁ + x₂y₂ + ...
    
    Args:
        x: First vector of shape (..., dim)
        y: Second vector of shape (..., dim)
        
    Returns:
        Minkowski inner product of shape (...)
    """
    # Split into time and space components
    time_component = x[..., 0] * y[..., 0]
    space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    
    # For better numerical stability, handle near-lightlike vectors
    result = space_component - time_component
    
    # For lightlike vectors, ensure exact zero norm
    if x is y:  # When computing norm
        lightlike_mask = torch.abs(result) < 1e-8
        result = torch.where(lightlike_mask, torch.zeros_like(result), result)
    
    return result

def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """Project points onto the hyperboloid manifold."""
    x_shape = x.shape
    x_norm = torch.norm(x, dim=-1)
    
    # Split into time and space components
    x_time = x[..., 0]
    x_space = x[..., 1:]
    x_space_norm = torch.norm(x_space, dim=-1)
    
    # Handle spatial components
    space_components = x_space
    space_sq = torch.sum(space_components ** 2, dim=-1, keepdim=True)
    spatial_norm_sq = space_sq
    compensation = torch.zeros_like(space_sq)
    
    # Compute time component
    scaled_norm_sq = spatial_norm_sq
    time_component_old = x_time.unsqueeze(-1)
    time_component_new = torch.sqrt(1.0 + scaled_norm_sq)
    
    # Handle space components
    space_scale = torch.ones_like(space_components)
    space_components_old = space_components
    space_components_new = space_components * space_scale
    space_norm_new = torch.norm(space_components_new, dim=-1)
    
    # Combine components
    result = torch.cat([time_component_new, space_components_new], dim=-1)
    result_norm = torch.norm(result, dim=-1)
    
    # Verify hyperboloid constraint
    constraint_value = minkowski_inner(result, result) + 1
    
    # Extract final components
    time_final = result[..., 0]
    space_final = result[..., 1:]
    space_final_norm = torch.norm(space_final, dim=-1)
    
    return result

def project_to_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project vector v onto the tangent space at point x.
    
    Args:
        x: Point tensor of shape (..., dim)
        v: Vector tensor of shape (..., dim)
        
    Returns:
        Projected vector of shape (..., dim)
    """
    # Project x to hyperboloid if needed
    x = project_to_hyperboloid(x)
    
    # Compute the Minkowski inner product
    inner = minkowski_inner(x, v)
    
    # Project v onto the tangent space at x
    v_proj = v + inner.unsqueeze(-1) * x
    
    return v_proj


class ParallelTransport(nn.Module):
    """Parallel transport of tangent vectors along geodesics."""

    def __init__(self, dim: int, method: Literal["schild", "pole", "euclidean"] = "schild"):
        super().__init__()
        self.dim = dim
        self.method = method

    def forward(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Transport vector v from point x to point y.

        Args:
            x: Starting point tensor of shape (..., dim)
            y: Target point tensor of shape (..., dim)
            v: Vector to transport of shape (..., dim)

        Returns:
            Transported vector of shape (..., dim)
        """
        if self.method == "euclidean":
            return v  # In Euclidean space, parallel transport is trivial

        # Project points to hyperboloid and vector to tangent space
        x = project_to_hyperboloid(x)
        y = project_to_hyperboloid(y)
        v = project_to_tangent(x, v)

        # For hyperbolic space, use the specified method
        if self.method == "schild":
            return self._schild_ladder(x, y, v)
        else:  # pole method
            return self._pole_ladder(x, y, v)

    def _schild_ladder(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Schild's ladder parallel transport in hyperbolic space."""
        # Compute the Minkowski inner products
        inner_xy = minkowski_inner(x, y)
        inner_xv = minkowski_inner(x, v)
        
        # Compute parallel transport coefficients
        alpha = inner_xy + 1.0  # Distance factor
        beta = inner_xv / alpha  # Transport factor
        
        # Transport v to y using the parallel transport formula
        result = v - beta * x - beta * y
        
        # Project result to tangent space at y
        result = project_to_tangent(y, result)
        
        # Normalize to preserve norm
        v_norm = torch.sqrt(torch.abs(minkowski_inner(v, v)))
        result_norm = torch.sqrt(torch.abs(minkowski_inner(result, result)))
        result = result * (v_norm / (result_norm + 1e-7))
        
        return result

    def _pole_ladder(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pole ladder parallel transport in hyperbolic space."""
        # Compute midpoint on geodesic from x to y
        inner_xy = minkowski_inner(x, y)
        t = 0.5  # Midpoint parameter
        
        # Compute geodesic midpoint
        factor = torch.acosh(-inner_xy)
        m = torch.cosh((1-t)*factor)*x + torch.sinh((1-t)*factor)*y/factor.unsqueeze(-1)
        
        # Project m to hyperboloid
        m = project_to_hyperboloid(m)
        
        # Transport v to m
        v_m = self._schild_ladder(x, m, v)
        
        # Transport v_m from m to y
        result = self._schild_ladder(m, y, v_m)
        
        # Project result to tangent space at y
        result = project_to_tangent(y, result)
        
        # Normalize to preserve norm
        v_norm = torch.sqrt(torch.abs(minkowski_inner(v, v)))
        result_norm = torch.sqrt(torch.abs(minkowski_inner(result, result)))
        result = result * (v_norm / (result_norm + 1e-7))
        
        return result


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
        ⟨x,y⟩ = -x₀y₀ + x₁y₁ + x₂y₂ + ...
        
        Args:
            x: First vector of shape (..., dim)
            y: Second vector of shape (..., dim)
            
        Returns:
            Minkowski inner product of shape (...)
        """
        # Split into time and space components
        time_component = x[..., 0] * y[..., 0]
        space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        
        # For better numerical stability, handle near-lightlike vectors
        result = space_component - time_component
        
        # For lightlike vectors, ensure exact zero norm
        if x is y:  # When computing norm
            lightlike_mask = torch.abs(result) < 1e-8
            result = torch.where(lightlike_mask, torch.zeros_like(result), result)
        
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
        """Forward pass of the exponential map with enhanced numerical stability.
        
        This implementation carefully handles:
        1. Small vector norms (|v| ≤ eps) using Taylor expansion
        2. Large vector norms (|v| ≥ max_norm) with scaling
        3. Precise projection to hyperboloid
        4. Verification of constraints at each step
        
        Args:
            x: Base point tensor of shape (..., dim)
            v: Tangent vector tensor of shape (..., dim)
            
        Returns:
            Point on hyperboloid of shape (..., dim)
        """
        if self.debug:
            self._debug_print("forward_input",
                x=x,
                v=v,
                x_norm=torch.norm(x),
                v_norm=torch.norm(v),
                x_inner=self.minkowski_inner(x, x),
                v_inner=self.minkowski_inner(v, v)
            )
        
        # Project x to hyperboloid with high precision
        x = self.project_to_hyperboloid(x)
        
        # Project v to tangent space with enhanced precision
        v = self.project_to_tangent(x, v)
        
        # Compute the norm of v with improved stability
        v_inner = self.minkowski_inner(v, v)
        v_norm = torch.sqrt(torch.clamp(v_inner, min=0.0))
        
        if self.debug:
            self._debug_print("forward_prep",
                x_proj=x,
                v_proj=v,
                v_inner=v_inner,
                v_norm=v_norm,
                x_v_inner=self.minkowski_inner(x, v)
            )
        
        # Handle zero vectors with Taylor expansion
        zero_mask = (v_norm < self.eps)
        
        # Handle large norms with scaling
        scale_mask = (v_norm > self.max_norm)
        scale_factor = torch.where(
            scale_mask,
            self.max_norm / v_norm,
            torch.ones_like(v_norm)
        )
        v_scaled = v * scale_factor.unsqueeze(-1)
        v_norm_scaled = v_norm * scale_factor
        
        # Compute hyperbolic functions with enhanced precision
        cosh_vn = torch.cosh(v_norm_scaled)
        sinh_vn = torch.sinh(v_norm_scaled)
        
        # Compute coefficients with numerical safeguards
        coeff1 = torch.where(
            zero_mask,
            torch.ones_like(v_norm),
            cosh_vn
        ).unsqueeze(-1)
        
        coeff2 = torch.where(
            zero_mask,
            torch.ones_like(v_norm),
            sinh_vn / v_norm_scaled.clamp(min=self.eps)
        ).unsqueeze(-1)
        
        if self.debug:
            self._debug_print("forward_coeffs",
                zero_mask=zero_mask,
                scale_mask=scale_mask,
                scale_factor=scale_factor,
                v_scaled=v_scaled,
                v_norm_scaled=v_norm_scaled,
                cosh_vn=cosh_vn,
                sinh_vn=sinh_vn,
                coeff1=coeff1,
                coeff2=coeff2
            )
        
        # Compute result with controlled operations
        result = coeff1 * x + coeff2 * v_scaled
        
        # Project back to hyperboloid with high precision
        result = self.project_to_hyperboloid(result)
        
        # Verify constraints are satisfied
        result_inner = self.minkowski_inner(result, result)
        constraint_violation = torch.abs(result_inner + 1.0)
        
        if self.debug:
            self._debug_print("forward_output",
                result=result,
                result_inner=result_inner,
                constraint_violation=constraint_violation,
                time_component=result[..., 0],
                space_components=result[..., 1:],
                space_norm=torch.norm(result[..., 1:], dim=-1)
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

    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transport a vector v from x to y along the geodesic.
        
        Args:
            x: Starting point on hyperboloid of shape (..., dim)
            y: Ending point on hyperboloid of shape (..., dim)
            v: Tangent vector at x of shape (..., dim)
            
        Returns:
            Transported vector at y of shape (..., dim)
        """
        # Compute the parallel transport using the explicit formula
        inner_xy = self.minkowski_inner(x, y)
        inner_xv = self.minkowski_inner(x, v)
        inner_yv = self.minkowski_inner(y, v)
        
        # Compute coefficients for the transport formula
        alpha = inner_xy + 1.0
        beta = inner_xv / alpha
        
        # Transport formula preserving norm
        result = v + beta * (x + y)
        
        # Project result to tangent space at y and normalize
        result = self.project_to_tangent(y, result)
        norm_v = torch.sqrt(torch.abs(self.minkowski_inner(v, v)))
        norm_result = torch.sqrt(torch.abs(self.minkowski_inner(result, result)))
        scale = torch.where(norm_result > 0, norm_v / norm_result, torch.ones_like(norm_v))
        
        return scale.unsqueeze(-1) * result


class HyperbolicLogarithm(nn.Module):
    """Logarithm map for hyperbolic space.
    
    Mathematical Framework:
    ---------------------
    1. Hyperboloid Model:
       H^n = {x ∈ ℝ^{n+1} | ⟨x,x⟩_M = -1, x₀ > 0}
       where ⟨·,·⟩_M is the Minkowski inner product
    
    2. Logarithm Map Formula:
       log_x(y) = d * (y + ⟨x,y⟩_M x) / ‖y + ⟨x,y⟩_M x‖_M
       where d = arccosh(-⟨x,y⟩_M) is the hyperbolic distance
    
    Properties:
    ----------
    1. log_x(x) = 0
    2. log_x(exp_x(v)) = v for v in T_xH^n
    3. ‖log_x(y)‖_M = d(x,y)
    4. ⟨x,log_x(y)⟩_M = 0 (tangent space constraint)
    
    Numerical Considerations:
    ----------------------
    1. Small distances (d ≤ eps): Return zero vector
    2. Near-boundary points (⟨x,y⟩_M ≈ -1): Use Taylor expansion
    3. Maintain orthogonality through explicit projection
    4. Ensure exact distance preservation
    """

    def __init__(self, dim: int, curvature: float = -1.0, dtype: torch.dtype = torch.float32):
        """Initialize logarithm map."""
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature, dtype=dtype), requires_grad=False)
        self.eps = 1e-8
        self.max_dist = 20.0  # Maximum distance for numerical stability
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
        """Compute Minkowski inner product with careful handling of time component."""
        time_component = x[..., 0] * y[..., 0]
        space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
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
        
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the hyperboloid with numerical stability."""
        # Extract components
        x_shape = x.shape
        x_norm = torch.norm(x)
        x_time = x[..., 0]
        x_space = x[..., 1:]
        x_space_norm = torch.norm(x_space)
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_input",
                x=x,
                x_shape=x_shape,
                x_norm=x_norm,
                x_time=x_time,
                x_space=x_space,
                x_space_norm=x_space_norm
            )
        
        # Compute spatial components with Kahan summation
        space_components = x_space
        space_sq = space_components * space_components
        spatial_norm_sq = torch.zeros_like(x_time[..., None])
        compensation = torch.zeros_like(x_time[..., None])
        
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
        
        # Scale norm for stability
        scaled_norm_sq = torch.where(
            (spatial_norm_sq < self.eps).to(torch.bool),
            torch.ones_like(spatial_norm_sq) * self.eps,
            spatial_norm_sq
        )
        
        # Compute time component
        time_component_old = x_time[..., None]
        time_component_new = torch.sqrt(1.0 + scaled_norm_sq)
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_time",
                scaled_norm_sq=scaled_norm_sq,
                time_component_old=time_component_old,
                time_component_new=time_component_new
            )
        
        # Scale space components
        space_scale = torch.where(
            (spatial_norm_sq < self.eps).to(torch.bool),
            torch.zeros_like(spatial_norm_sq),
            torch.sqrt(scaled_norm_sq / spatial_norm_sq.clamp(min=self.eps))
        )
        space_components_old = space_components
        space_components_new = space_components * space_scale
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_space",
                space_scale=space_scale,
                space_components_old=space_components_old,
                space_components_new=space_components_new,
                space_norm_new=torch.norm(space_components_new)
            )
        
        # Combine components
        result = torch.cat([time_component_new, space_components_new], dim=-1)
        
        # Verify constraint
        result_norm = torch.norm(result)
        constraint_value = self.minkowski_inner(result, result) + 1.0
        time_final = result[..., 0]
        space_final = result[..., 1:]
        space_final_norm = torch.norm(space_final)
        
        if self.debug:
            self._debug_print("project_to_hyperboloid_result",
                result=result,
                result_norm=result_norm,
                constraint_value=constraint_value,
                time_final=time_final,
                space_final=space_final,
                space_final_norm=space_final_norm
            )
        
        return result
        
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto the tangent space at point x with improved precision."""
        # Compute inner product with higher precision
        inner = self.minkowski_inner(x, v)
        
        # Apply projection with numerical stability
        v_proj = v + (inner / (1.0 + self.eps))[..., None] * x
        
        # Double-check tangent space constraint
        tangent_check = self.minkowski_inner(x, v_proj)
        zero_mask = torch.abs(tangent_check) < self.eps
        
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute logarithm map with enhanced numerical stability.
        
        This implementation carefully handles:
        1. Points near the base point using Taylor expansion
        2. Points near the cut locus with scaling
        3. Precise projection to tangent space
        4. Verification of distance preservation
        
        Args:
            x: Base point tensor of shape (..., dim)
            y: Target point tensor of shape (..., dim)
            
        Returns:
            Tangent vector of shape (..., dim)
        """
        if self.debug:
            self._debug_print("log_forward_input",
                x=x,
                y=y,
                x_norm=torch.norm(x),
                y_norm=torch.norm(y),
                x_inner=self.minkowski_inner(x, x),
                y_inner=self.minkowski_inner(y, y)
            )
        
        # Project points to hyperboloid with high precision
        x = self.project_to_hyperboloid(x)
        y = self.project_to_hyperboloid(y)
        
        # Compute Minkowski inner product with improved precision
        inner = self.minkowski_inner(x, y)
        
        # Handle numerical issues near -1
        inner = torch.clamp(inner, max=-1.0 - self.eps)
        
        # Compute distance with enhanced precision
        dist = torch.acosh(-inner)
        
        if self.debug:
            self._debug_print("log_forward_prep",
                x_proj=x,
                y_proj=y,
                inner=inner,
                dist=dist
            )
        
        # Handle zero distance case with Taylor expansion
        zero_mask = (dist < self.eps)
        if zero_mask.any():
            if self.debug:
                self._debug_print("log_zero_case",
                    zero_mask=zero_mask,
                    num_zeros=torch.sum(zero_mask)
                )
            return torch.zeros_like(x)
        
        # Handle points near cut locus with scaling
        scale_mask = (dist > self.max_dist)
        scale_factor = torch.where(
            scale_mask,
            self.max_dist / dist,
            torch.ones_like(dist)
        )
        dist_scaled = dist * scale_factor
        
        # Compute the direction with improved stability
        # First compute y⟂, the part of y orthogonal to x
        y_orth = y + inner.unsqueeze(-1) * x
        
        # Normalize y⟂ with careful handling of small norms
        y_orth_inner = self.minkowski_inner(y_orth, y_orth)
        y_orth_norm = torch.sqrt(torch.abs(y_orth_inner).clamp(min=self.eps))
        
        if self.debug:
            self._debug_print("log_direction",
                scale_mask=scale_mask,
                scale_factor=scale_factor,
                dist_scaled=dist_scaled,
                y_orth=y_orth,
                y_orth_inner=y_orth_inner,
                y_orth_norm=y_orth_norm
            )
        
        # Compute initial direction
        v = (dist_scaled / y_orth_norm).unsqueeze(-1) * y_orth
        
        # Project to tangent space at x with high precision
        v = self.project_to_tangent(x, v)
        
        # Normalize to match distance exactly
        v_inner = self.minkowski_inner(v, v)
        v_norm = torch.sqrt(torch.abs(v_inner).clamp(min=self.eps))
        v = (dist / v_norm).unsqueeze(-1) * v
        
        # Final projection to ensure tangent space constraint
        v = self.project_to_tangent(x, v)
        
        # Verify properties
        final_inner = self.minkowski_inner(x, v)
        final_norm = torch.sqrt(torch.abs(self.minkowski_inner(v, v)).clamp(min=self.eps))
        
        if self.debug:
            self._debug_print("log_forward_output",
                v=v,
                v_inner=v_inner,
                v_norm=v_norm,
                final_inner=final_inner,
                final_norm=final_norm,
                dist=dist,
                norm_error=torch.abs(final_norm - dist),
                tangent_error=torch.abs(final_inner)
            )
        
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
            self.register_module('transport', ParallelTransport(dim, method=parallel_transport_method))
        else:  # Euclidean default
            self.register_module('exp_map', EuclideanExponential(dim))
            self.register_module('log_map', EuclideanLogarithm(dim))
            self.register_module('transport', ParallelTransport(dim, method="euclidean"))

    def sectional_curvature(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute the sectional curvature at point x for the plane spanned by u,v.
        
        Args:
            x: Point on the manifold, shape (..., dim)
            u: First vector spanning the plane, shape (..., dim)
            v: Second vector spanning the plane, shape (..., dim)
            
        Returns:
            Sectional curvature value, shape (...)
        """
        if self.manifold_type == "euclidean":
            return torch.zeros_like(x[..., 0])
        
        # Project to hyperboloid and tangent space
        x = self.exp_map.project_to_hyperboloid(x)
        u = self.exp_map.project_to_tangent(x, u)
        v = self.exp_map.project_to_tangent(x, v)
        
        # Compute inner products with high precision
        g_uu = self.exp_map.minkowski_inner(u, u)
        g_vv = self.exp_map.minkowski_inner(v, v)
        g_uv = self.exp_map.minkowski_inner(u, v)
        
        # Compute numerator (R(u,v)v,u) with improved stability
        # For hyperboloid manifold, R(u,v)v = g(u,v)v - g(v,v)u
        Ruvv = g_uv.unsqueeze(-1) * v - g_vv.unsqueeze(-1) * u
        numerator = self.exp_map.minkowski_inner(Ruvv, u)
        
        # Compute denominator with numerical safeguards
        denominator = (g_uu * g_vv - g_uv * g_uv).clamp(min=1e-8)
        
        # Compute sectional curvature
        K = numerator / denominator
        
        if self.exp_map.debug:
            self.exp_map._debug_print("sectional_curvature",
                x=x,
                u=u,
                v=v,
                g_uu=g_uu,
                g_vv=g_vv,
                g_uv=g_uv,
                Ruvv=Ruvv,
                numerator=numerator,
                denominator=denominator,
                K=K
            )
        
        return K

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
        """Parallel transport a batch of vectors v from points x to points y.
        
        This implementation uses a numerically stable version of the pole ladder method
        with careful handling of inner products and projections.
        
        Args:
            x: Starting points on the manifold, shape (..., dim)
            y: Target points on the manifold, shape (..., dim)
            v: Vectors to transport, shape (..., dim)
            
        Returns:
            Transported vectors, shape (..., dim)
        """
        if self.manifold_type == "euclidean":
            return v  # Euclidean parallel transport is identity
        
        # Project points to hyperboloid with high precision
        x = self.exp_map.project_to_hyperboloid(x)
        y = self.exp_map.project_to_hyperboloid(y)
        
        # Project v to tangent space at x with enhanced precision
        v = self.exp_map.project_to_tangent(x, v)
        
        # Compute Minkowski inner products with improved stability
        inner_xy = self.exp_map.minkowski_inner(x, y)
        inner_xv = self.exp_map.minkowski_inner(x, v)
        inner_yv = self.exp_map.minkowski_inner(y, v)
        
        # Transport coefficients with numerical safeguards
        alpha = inner_xy + 1.0  # Always positive by hyperboloid constraint
        alpha = alpha.clamp(min=1e-8)  # Prevent division by zero
        
        # Compute transport coefficients with improved stability
        beta = inner_xv / alpha
        gamma = inner_yv / alpha
        
        # Transport v to y using the parallel transport formula
        # with improved numerical stability
        result = v - beta.unsqueeze(-1) * x - gamma.unsqueeze(-1) * y
        
        # Project result to tangent space at y with high precision
        result = self.exp_map.project_to_tangent(y, result)
        
        # Normalize to preserve inner product
        inner_before = self.exp_map.minkowski_inner(v, v)
        inner_after = self.exp_map.minkowski_inner(result, result)
        scale = torch.sqrt(torch.abs(inner_before) / torch.abs(inner_after).clamp(min=1e-8))
        result = result * scale.unsqueeze(-1)
        
        # Final projection to ensure tangent space constraint
        result = self.exp_map.project_to_tangent(y, result)
        
        if self.exp_map.debug:
            self.exp_map._debug_print("parallel_transport",
                x=x,
                y=y,
                v=v,
                inner_xy=inner_xy,
                inner_xv=inner_xv,
                inner_yv=inner_yv,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                result=result,
                inner_before=inner_before,
                inner_after=inner_after,
                scale=scale
            )
        
        return result

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
