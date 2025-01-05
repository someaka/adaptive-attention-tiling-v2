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

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Tuple, Union
from typing_extensions import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.memory_management_util import register_tensor, optimize_memory, clear_memory

# Configure logging
def setup_geometric_logging():
    """Set up logging configuration for geometric operations."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a unique log file for geometric operations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/geometric_operations_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger('geometric_operations')

# Initialize logger
logger = setup_geometric_logging()

def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Minkowski inner product between two vectors."""
    logger.debug(f"Computing Minkowski inner product")
    logger.debug(f"Input shapes: x: {x.shape}, y: {y.shape}")
    
    # Check if either vector is lightlike
    x_lightlike = torch.abs(x[..., 0]**2 - torch.sum(x[..., 1:]**2, dim=-1)) < 1e-10
    y_lightlike = torch.abs(y[..., 0]**2 - torch.sum(y[..., 1:]**2, dim=-1)) < 1e-10
    
    logger.debug(f"Lightlike vectors detected - x: {torch.sum(x_lightlike)}, y: {torch.sum(y_lightlike)}")
    
    # Split into time and space components
    time_component = x[..., 0] * y[..., 0]
    space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    
    logger.debug(f"Components - Time: {time_component}, Space: {space_component}")
    
    # For better numerical stability, handle near-lightlike vectors
    result = space_component - time_component
    
    # If either vector is lightlike, ensure exact zero norm
    lightlike_mask = x_lightlike | y_lightlike
    result = torch.where(lightlike_mask, torch.zeros_like(result), result)
    
    logger.debug(f"Final result: {result}")
    return result

def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """Project points onto the hyperboloid manifold."""
    logger.debug(f"Projecting to hyperboloid")
    logger.debug(f"Input shape: {x.shape}")
    
    x_shape = x.shape
    x_norm = torch.norm(x)
    
    logger.debug(f"Input norm: {x_norm}")
    
    # Split into time and space components
    x_time = x[..., 0]
    x_space = x[..., 1:]
    x_space_norm = torch.norm(x_space, dim=-1)
    
    logger.debug(f"Initial components - Time: {x_time}, Space norm: {x_space_norm}")
    
    # Handle spatial components
    space_components = x_space
    space_sq = torch.sum(space_components ** 2, dim=-1, keepdim=True)
    spatial_norm_sq = space_sq
    compensation = torch.zeros_like(space_sq)
    
    # Compute time component
    scaled_norm_sq = spatial_norm_sq
    time_component_old = x_time.unsqueeze(-1)
    time_component_new = torch.sqrt(1.0 + scaled_norm_sq)
    
    logger.debug(f"Time component - Old: {time_component_old}, New: {time_component_new}")
    
    # Handle space components
    space_scale = torch.ones_like(space_components)
    space_components_old = space_components
    space_components_new = space_components * space_scale
    space_norm_new = torch.norm(space_components_new, dim=-1)
    
    logger.debug(f"Space components - Old norm: {torch.norm(space_components_old)}, New norm: {space_norm_new}")
    
    # Combine components
    result = torch.cat([time_component_new, space_components_new], dim=-1)
    result_norm = torch.norm(result, dim=-1)
    
    # Verify hyperboloid constraint
    constraint_value = minkowski_inner(result, result) + 1
    
    logger.debug(f"Constraint verification - Value: {constraint_value}, Max violation: {torch.max(torch.abs(constraint_value))}")
    
    # Extract final components
    time_final = result[..., 0]
    space_final = result[..., 1:]
    space_final_norm = torch.norm(space_final, dim=-1)
    
    logger.debug(f"Final components - Time: {time_final}, Space norm: {space_final_norm}")
    logger.debug(f"Final result shape: {result.shape}, norm: {torch.norm(result)}")
    
    return result

def project_to_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project vector onto tangent space with enhanced precision."""
    logger.debug(f"Projecting to tangent space - shapes: x={x.shape}, v={v.shape}")
    
    # Handle zero vector case
    if torch.all(torch.abs(v) < 1e-7):
        return torch.zeros_like(v)
    
    # Project x to hyperboloid if needed
    x = project_to_hyperboloid(x)
    
    # Normalize input vector for better numerical stability
    v_norm = torch.norm(v)
    if v_norm > 1e-7:
        v = v / v_norm
    
    # Split into time and space components
    x_time = x[..., 0]
    x_space = x[..., 1:]
    v_time = v[..., 0]
    v_space = v[..., 1:]
    
    # Compute inner product of space components with high precision
    space_inner = torch.sum(x_space * v_space, dim=-1)
    
    # Compute time component to satisfy tangent space constraint
    # ⟨x,v⟩ = 0 ⟨=⟩ -x₀v₀ + Σᵢxᵢvᵢ = 0
    # ⟨=⟩ v₀ = Σᵢxᵢvᵢ/x₀
    v_time_new = space_inner / x_time
    
    # Combine components
    result = torch.cat([v_time_new.unsqueeze(-1), v_space], dim=-1)
    
    # First projection to ensure tangent space constraint
    inner = minkowski_inner(x, result)
    result = result - minkowski_inner(result, x).unsqueeze(-1) * x
    
    # Verify tangent space constraint
    inner_after = minkowski_inner(x, result)
    if not torch.allclose(inner_after, torch.zeros_like(inner_after), atol=1e-10):
        logger.warning(f"Tangent space constraint violation after first projection: {inner_after}")
        # Second projection with higher precision
        result = result - minkowski_inner(result, x).unsqueeze(-1) * x
        
        # Normalize result for stability
        result_norm = torch.norm(result)
        if result_norm > 1e-7:
            result = result / result_norm
    
    # Restore original scale if input was normalized
    if v_norm > 1e-7:
        result = result * v_norm
    
    logger.debug(f"Projection result - shape: {result.shape}, inner: {inner_after}")
    
    return result


class ParallelTransport(nn.Module):
    """Parallel transport implementation with improved stability."""

    def __init__(
        self,
        dim: int,
        method: Literal["schild", "pole", "euclidean"] = "schild",
        exp_map: Optional["HyperbolicExponential"] = None,
    ):
        super().__init__()
        self.dim = dim
        self.method = method
        self.exp_map = exp_map
        self.logger = logging.getLogger(f"{__name__}.ParallelTransport")
        self.logger.info(f"Initialized ParallelTransport - dim: {dim}, method: {method}")

    def _schild_ladder(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Schild's ladder parallel transport with improved stability."""
        assert self.exp_map is not None, "exp_map must be provided for hyperbolic transport"
        self.logger.debug(f"Schild's ladder transport - shapes: x={x.shape}, y={y.shape}, v={v.shape}")

        # Project points to hyperboloid and vector to tangent space with high precision
        x = self.exp_map.project_to_hyperboloid(x)
        y = self.exp_map.project_to_hyperboloid(y)
        v = self.exp_map.project_to_tangent(x, v)

        # Handle zero vector case
        if torch.all(torch.abs(v) < 1e-7):
            return torch.zeros_like(v)

        # Handle same point case
        if torch.allclose(x, y, rtol=1e-7, atol=1e-7):
            return v

        # Split into time and space components
        x_time = x[..., 0]
        x_space = x[..., 1:]
        y_time = y[..., 0]
        y_space = y[..., 1:]
        v_time = v[..., 0]
        v_space = v[..., 1:]

        # Compute geodesic midpoint with improved stability
        xy = y - x  # Difference vector
        xy_inner = -self.exp_map.minkowski_inner(xy, xy)  # Squared distance
        t = 0.5  # Midpoint parameter
        
        # Compute parallel transport coefficients
        alpha = torch.cosh(torch.sqrt(xy_inner.clamp(min=1e-8)) * t)
        beta = torch.sinh(torch.sqrt(xy_inner.clamp(min=1e-8)) * t) / torch.sqrt(xy_inner.clamp(min=1e-8))
        
        # Compute midpoint using stable formula
        mid = alpha.unsqueeze(-1) * x + beta.unsqueeze(-1) * xy
        mid = self.exp_map.project_to_hyperboloid(mid)
        
        # Split midpoint into components
        mid_time = mid[..., 0]
        mid_space = mid[..., 1:]

        # Transport v to midpoint using stable formula
        # First compute space components
        v_mid_space = v_space - (torch.sum(v_space * mid_space, dim=-1) / mid_time).unsqueeze(-1) * mid_space
        
        # Then compute time component to satisfy tangent space constraint
        v_mid_time = torch.sum(v_mid_space * mid_space, dim=-1) / mid_time
        
        # Combine components
        v_mid = torch.cat([v_mid_time.unsqueeze(-1), v_mid_space], dim=-1)
        v_mid = self.exp_map.project_to_tangent(mid, v_mid)
        
        # Transport from midpoint to y using the same process
        # First compute space components
        result_space = v_mid[..., 1:] - (torch.sum(v_mid[..., 1:] * y_space, dim=-1) / y_time).unsqueeze(-1) * y_space
        
        # Then compute time component to satisfy tangent space constraint
        result_time = torch.sum(result_space * y_space, dim=-1) / y_time
        
        # Combine components
        result = torch.cat([result_time.unsqueeze(-1), result_space], dim=-1)
        result = self.exp_map.project_to_tangent(y, result)

        # Normalize to preserve inner product
        v_inner = self.exp_map.minkowski_inner(v, v)
        result_inner = self.exp_map.minkowski_inner(result, result)
        scale = torch.sqrt(torch.abs(v_inner) / torch.abs(result_inner).clamp(min=1e-8))
        result = result * scale.unsqueeze(-1)

        # Final projection to ensure tangent space constraint
        result = self.exp_map.project_to_tangent(y, result)

        return result

    def _pole_ladder(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Pole ladder parallel transport with improved stability."""
        assert self.exp_map is not None, "exp_map must be provided for hyperbolic transport"
        self.logger.debug(f"Pole ladder transport - shapes: x={x.shape}, y={y.shape}, v={v.shape}")

        # Project points to hyperboloid and vector to tangent space
        x = self.exp_map.project_to_hyperboloid(x)
        y = self.exp_map.project_to_hyperboloid(y)
        v = self.exp_map.project_to_tangent(x, v)

        # Handle zero vector case
        if torch.all(torch.abs(v) < 1e-7):
            return torch.zeros_like(v)

        # Handle same point case
        if torch.allclose(x, y, rtol=1e-7, atol=1e-7):
            return v

        # Split into time and space components
        x_time = x[..., 0]
        x_space = x[..., 1:]
        y_time = y[..., 0]
        y_space = y[..., 1:]
        v_time = v[..., 0]
        v_space = v[..., 1:]

        # Compute pole point with improved stability
        pole = x + y
        pole_inner = -self.exp_map.minkowski_inner(pole, pole)
        pole = pole / torch.sqrt(pole_inner.clamp(min=1e-8)).unsqueeze(-1)
        pole = self.exp_map.project_to_hyperboloid(pole)
        
        # Split pole into components
        pole_time = pole[..., 0]
        pole_space = pole[..., 1:]

        # Transport v to pole
        # First compute space components
        v_pole_space = v_space - (torch.sum(v_space * pole_space, dim=-1) / pole_time).unsqueeze(-1) * pole_space
        
        # Then compute time component to satisfy tangent space constraint
        v_pole_time = torch.sum(v_pole_space * pole_space, dim=-1) / pole_time
        
        # Combine components
        v_pole = torch.cat([v_pole_time.unsqueeze(-1), v_pole_space], dim=-1)
        v_pole = self.exp_map.project_to_tangent(pole, v_pole)

        # Transport from pole to y
        # First compute space components
        result_space = v_pole[..., 1:] - (torch.sum(v_pole[..., 1:] * y_space, dim=-1) / y_time).unsqueeze(-1) * y_space
        
        # Then compute time component to satisfy tangent space constraint
        result_time = torch.sum(result_space * y_space, dim=-1) / y_time
        
        # Combine components
        result = torch.cat([result_time.unsqueeze(-1), result_space], dim=-1)
        result = self.exp_map.project_to_tangent(y, result)

        # Normalize to preserve inner product
        v_inner = self.exp_map.minkowski_inner(v, v)
        result_inner = self.exp_map.minkowski_inner(result, result)
        scale = torch.sqrt(torch.abs(v_inner) / torch.abs(result_inner).clamp(min=1e-8))
        result = result * scale.unsqueeze(-1)

        # Final projection to ensure tangent space constraint
        result = self.exp_map.project_to_tangent(y, result)

        return result

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for parallel transport."""
        self.logger.debug(f"Forward pass - shapes: x={x.shape}, y={y.shape}, v={v.shape}")

        if self.method == "euclidean":
            self.logger.debug("Using Euclidean transport (identity)")
            return v

        if self.method == "schild":
            return self._schild_ladder(x, y, v)
        else:  # pole ladder
            return self._pole_ladder(x, y, v)


class HyperbolicExponential(nn.Module):
    """Exponential map for hyperbolic space."""
    
    def __init__(self, dim: int, curvature: float = -1.0, dtype: torch.dtype = torch.float32):
        """Initialize exponential map."""
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature, dtype=dtype), requires_grad=False)
        self.eps = 1e-8
        self.max_norm = 20.0
        self.dtype = dtype
        self.logger = logging.getLogger(f"{__name__}.HyperbolicExponential")
        self.logger.info(f"Initialized HyperbolicExponential - dim: {dim}, curvature: {curvature}")
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product with careful handling of time component."""
        self.logger.debug(f"Computing Minkowski inner product - shapes: x={x.shape}, y={y.shape}")
        
        time_component = x[..., 0] * y[..., 0]
        space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        result = -time_component + space_component
        
        self.logger.debug(f"Inner product components - time: {time_component}, space: {space_component}")
        self.logger.debug(f"Inner product result: {result}")
        
        return result
        
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto hyperboloid with numerical stability."""
        self.logger.debug(f"Projecting to hyperboloid - input shape: {x.shape}")
        
        # Extract components
        x_time = x[..., 0]
        x_space = x[..., 1:]
        x_space_norm = torch.norm(x_space, dim=-1)
        
        self.logger.debug(f"Input components - time: {x_time}, space norm: {x_space_norm}")
        
        # Compute spatial norm squared with Kahan summation
        space_components = x_space
        space_sq = space_components * space_components
        spatial_norm_sq = torch.zeros_like(x_time[..., None])
        compensation = torch.zeros_like(x_time[..., None])
        
        for i in range(space_sq.size(-1)):
            y = space_sq[..., i:i+1] - compensation
            t = spatial_norm_sq + y
            compensation = (t - spatial_norm_sq) - y
            spatial_norm_sq = t
        
        self.logger.debug(f"Spatial norm computation - norm squared: {spatial_norm_sq}")
        
        # Scale norm for stability
        scaled_norm_sq = torch.where(
            (spatial_norm_sq < self.eps).to(torch.bool),
            torch.ones_like(spatial_norm_sq) * self.eps,
            spatial_norm_sq
        )
        
        # Compute time component
        time_component_new = torch.sqrt(1.0 + scaled_norm_sq)
        
        self.logger.debug(f"New time component: {time_component_new}")
        
        # Scale space components
        space_scale = torch.where(
            (spatial_norm_sq < self.eps).to(torch.bool),
            torch.zeros_like(spatial_norm_sq),
            torch.sqrt(scaled_norm_sq / spatial_norm_sq.clamp(min=self.eps))
        )
        space_components_new = space_components * space_scale
        
        self.logger.debug(f"Space components - scale: {space_scale}, new norm: {torch.norm(space_components_new)}")
        
        # Combine components
        result = torch.cat([time_component_new, space_components_new], dim=-1)
        
        # Verify constraint
        constraint_value = self.minkowski_inner(result, result) + 1.0
        
        self.logger.debug(f"Projection result - shape: {result.shape}, constraint violation: {torch.max(torch.abs(constraint_value))}")
        
        return result
        
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector onto tangent space with enhanced precision."""
        self.logger.debug(f"Projecting to tangent space - shapes: x={x.shape}, v={v.shape}")
        
        # Handle zero vector case
        if torch.all(torch.abs(v) < 1e-7):
            return torch.zeros_like(v)
        
        # Project x to hyperboloid if needed
        x = self.project_to_hyperboloid(x)
        
        # Split into time and space components
        x_time = x[..., 0]
        x_space = x[..., 1:]
        v_time = v[..., 0]
        v_space = v[..., 1:]
        
        # Compute inner product of space components
        space_inner = torch.sum(x_space * v_space, dim=-1)
        
        # Compute time component to satisfy tangent space constraint
        # ⟨x,v⟩ = 0 ⟨=⟩ -x₀v₀ + Σᵢxᵢvᵢ = 0
        # ⟨=⟩ v₀ = Σᵢxᵢvᵢ/x₀
        v_time_new = space_inner / x_time
        
        # Combine components
        result = torch.cat([v_time_new.unsqueeze(-1), v_space], dim=-1)
        
        # Verify tangent space constraint
        inner = self.minkowski_inner(x, result)
        if not torch.allclose(inner, torch.zeros_like(inner), atol=1e-6):
            self.logger.warning(f"Tangent space constraint violation: {inner}")
            # Project again if needed
            result = result - self.minkowski_inner(result, x).unsqueeze(-1) * x
        
        self.logger.debug(f"Projection result - shape: {result.shape}, inner: {inner}")
        
        return result
        
    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass of exponential map with enhanced stability."""
        self.logger.debug(f"Exponential map forward pass - shapes: x={x.shape}, v={v.shape}")
        
        # Project to hyperboloid and tangent space
        x = self.project_to_hyperboloid(x)
        v = self.project_to_tangent(x, v)
        
        # Compute vector norm
        v_inner = self.minkowski_inner(v, v)
        v_norm = torch.sqrt(torch.clamp(v_inner, min=0.0))
        
        self.logger.debug(f"Vector properties - inner: {v_inner}, norm: {v_norm}")
        
        # Handle zero vectors
        zero_mask = (v_norm < self.eps)
        
        # Handle large norms
        scale_mask = (v_norm > self.max_norm)
        scale_factor = torch.where(
            scale_mask,
            self.max_norm / v_norm,
            torch.ones_like(v_norm)
        )
        v_scaled = v * scale_factor.unsqueeze(-1)
        v_norm_scaled = v_norm * scale_factor
        
        self.logger.debug(f"Vector scaling - scale factor: {scale_factor}, scaled norm: {v_norm_scaled}")
        
        # Compute hyperbolic functions
        cosh_vn = torch.cosh(v_norm_scaled)
        sinh_vn = torch.sinh(v_norm_scaled)
        
        # Compute coefficients
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
        
        self.logger.debug(f"Map coefficients - cosh: {coeff1}, sinh/norm: {coeff2}")
        
        # Compute result
        result = coeff1 * x + coeff2 * v_scaled
        
        # Project back to hyperboloid
        result = self.project_to_hyperboloid(result)
        
        # Verify constraints
        result_inner = self.minkowski_inner(result, result)
        constraint_violation = torch.abs(result_inner + 1.0)
        
        self.logger.debug(f"Final result - shape: {result.shape}, constraint violation: {torch.max(constraint_violation)}")
        
        return result


class HyperbolicLogarithm(nn.Module):
    """Logarithm map for hyperbolic space."""

    def __init__(self, dim: int, curvature: float = -1.0, dtype: torch.dtype = torch.float32):
        """Initialize logarithm map."""
        super().__init__()
        self.dim = dim
        self.curvature = nn.Parameter(torch.tensor(curvature, dtype=dtype), requires_grad=False)
        self.eps = 1e-8
        self.max_dist = 20.0
        self.dtype = dtype
        self.logger = logging.getLogger(f"{__name__}.HyperbolicLogarithm")
        self.logger.info(f"Initialized HyperbolicLogarithm - dim: {dim}, curvature: {curvature}")
        
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product with careful handling of time component."""
        self.logger.debug(f"Computing Minkowski inner product - shapes: x={x.shape}, y={y.shape}")
        
        time_component = x[..., 0] * y[..., 0]
        space_component = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        result = -time_component + space_component
        
        self.logger.debug(f"Inner product components - time: {time_component}, space: {space_component}")
        self.logger.debug(f"Inner product result: {result}")
        
        return result
        
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto hyperboloid with numerical stability."""
        self.logger.debug(f"Projecting to hyperboloid - input shape: {x.shape}")
        
        # Extract components
        x_time = x[..., 0]
        x_space = x[..., 1:]
        x_space_norm = torch.norm(x_space, dim=-1)
        
        self.logger.debug(f"Input components - time: {x_time}, space norm: {x_space_norm}")
        
        # Compute spatial norm squared with Kahan summation
        space_components = x_space
        space_sq = space_components * space_components
        spatial_norm_sq = torch.zeros_like(x_time[..., None])
        compensation = torch.zeros_like(x_time[..., None])
        
        for i in range(space_sq.size(-1)):
            y = space_sq[..., i:i+1] - compensation
            t = spatial_norm_sq + y
            compensation = (t - spatial_norm_sq) - y
            spatial_norm_sq = t
        
        self.logger.debug(f"Spatial norm computation - norm squared: {spatial_norm_sq}")
        
        # Scale norm for stability
        scaled_norm_sq = torch.where(
            (spatial_norm_sq < self.eps).to(torch.bool),
            torch.ones_like(spatial_norm_sq) * self.eps,
            spatial_norm_sq
        )
        
        # Compute time component
        time_component_new = torch.sqrt(1.0 + scaled_norm_sq)
        
        self.logger.debug(f"New time component: {time_component_new}")
        
        # Scale space components
        space_scale = torch.where(
            (spatial_norm_sq < self.eps).to(torch.bool),
            torch.zeros_like(spatial_norm_sq),
            torch.sqrt(scaled_norm_sq / spatial_norm_sq.clamp(min=self.eps))
        )
        space_components_new = space_components * space_scale
        
        self.logger.debug(f"Space components - scale: {space_scale}, new norm: {torch.norm(space_components_new)}")
        
        # Combine components
        result = torch.cat([time_component_new, space_components_new], dim=-1)
        
        # Verify constraint
        constraint_value = self.minkowski_inner(result, result) + 1.0
        
        self.logger.debug(f"Projection result - shape: {result.shape}, constraint violation: {torch.max(torch.abs(constraint_value))}")
        
        return result
        
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector onto tangent space with enhanced precision."""
        self.logger.debug(f"Projecting to tangent space - shapes: x={x.shape}, v={v.shape}")
        
        # Compute inner product with higher precision
        inner = self.minkowski_inner(x, v)
        
        # Apply projection with numerical stability
        v_proj = v + (inner / (1.0 + self.eps))[..., None] * x
        
        # Verify tangent space constraint
        tangent_check = self.minkowski_inner(x, v_proj)
        
        self.logger.debug(f"Tangent projection - inner: {inner}, constraint check: {tangent_check}")
        
        return v_proj
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of logarithm map with enhanced stability."""
        self.logger.debug(f"Logarithm map forward pass - shapes: x={x.shape}, y={y.shape}")
        
        # Project points to hyperboloid
        x = self.project_to_hyperboloid(x)
        y = self.project_to_hyperboloid(y)
        
        # Compute inner product
        inner = self.minkowski_inner(x, y)
        inner = torch.clamp(inner, max=-1.0 - self.eps)
        
        # Compute distance
        dist = torch.acosh(-inner)
        
        self.logger.debug(f"Distance computation - inner: {inner}, distance: {dist}")
        
        # Handle zero distance case
        zero_mask = (dist < self.eps)
        if zero_mask.any():
            self.logger.debug(f"Zero distance cases detected: {torch.sum(zero_mask)}")
            return torch.zeros_like(x)
            
        # Handle points near cut locus
        scale_mask = (dist > self.max_dist)
        scale_factor = torch.where(
            scale_mask,
            self.max_dist / dist,
            torch.ones_like(dist)
        )
        dist_scaled = dist * scale_factor
        
        self.logger.debug(f"Distance scaling - scale factor: {scale_factor}, scaled distance: {dist_scaled}")
        
        # Compute direction
        y_orth = y + inner.unsqueeze(-1) * x
        y_orth_inner = self.minkowski_inner(y_orth, y_orth)
        y_orth_norm = torch.sqrt(torch.abs(y_orth_inner).clamp(min=self.eps))
        
        self.logger.debug(f"Direction computation - orthogonal norm: {y_orth_norm}")
        
        # Compute initial direction
        v = (dist_scaled / y_orth_norm).unsqueeze(-1) * y_orth
        
        # Project to tangent space
        v = self.project_to_tangent(x, v)
        
        # Normalize to match distance
        v_inner = self.minkowski_inner(v, v)
        v_norm = torch.sqrt(torch.abs(v_inner).clamp(min=self.eps))
        v = (dist / v_norm).unsqueeze(-1) * v
        
        # Final projection
        v = self.project_to_tangent(x, v)
        
        # Verify properties
        final_inner = self.minkowski_inner(x, v)
        final_norm = torch.sqrt(torch.abs(self.minkowski_inner(v, v)).clamp(min=self.eps))
        
        self.logger.debug(f"Final result - shape: {v.shape}, norm error: {torch.abs(final_norm - dist)}")
        self.logger.debug(f"Tangent space constraint: {torch.abs(final_inner)}")
        
        return v


class EuclideanExponential(nn.Module):
    """Exponential map for Euclidean space."""

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.logger = logging.getLogger(f"{__name__}.EuclideanExponential")
        self.logger.info(f"Initialized EuclideanExponential - dim: {dim}")

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply exponential map."""
        self.logger.debug(f"Euclidean exponential map - shapes: x={x.shape}, v={v.shape}")
        return x + v


class EuclideanLogarithm(nn.Module):
    """Logarithmic map for Euclidean space."""

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.logger = logging.getLogger(f"{__name__}.EuclideanLogarithm")
        self.logger.info(f"Initialized EuclideanLogarithm - dim: {dim}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply logarithm map."""
        self.logger.debug(f"Euclidean logarithm map - shapes: x={x.shape}, y={y.shape}")
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
        self.logger = logging.getLogger(f"{__name__}.GeometricStructures")
        
        self.logger.info(f"Initializing GeometricStructures - dim: {dim}, manifold_type: {manifold_type}")

        # Geometric tensors with improved initialization
        self.register_parameter('metric', nn.Parameter(torch.eye(dim)))
        self.register_parameter('connection', nn.Parameter(torch.zeros(dim, dim, dim)))
        self.register_parameter('curvature_tensor', nn.Parameter(torch.zeros(dim, dim, dim, dim)))
        
        self.logger.debug("Initialized geometric tensors")

        # Manifold structures
        if manifold_type == "hyperbolic":
            exp_map = HyperbolicExponential(dim, curvature)
            self.register_module('exp_map', exp_map)
            self.register_module('log_map', HyperbolicLogarithm(dim, curvature))
            self.register_module('transport', ParallelTransport(dim, method=parallel_transport_method, exp_map=exp_map))
            self.logger.info("Initialized hyperbolic manifold structures")
        else:  # Euclidean default
            self.register_module('exp_map', EuclideanExponential(dim))
            self.register_module('log_map', EuclideanLogarithm(dim))
            self.register_module('transport', ParallelTransport(dim, method="euclidean"))
            self.logger.info("Initialized Euclidean manifold structures")

    def sectional_curvature(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute the sectional curvature at point x for the plane spanned by u,v."""
        self.logger.debug(f"Computing sectional curvature - shapes: x={x.shape}, u={u.shape}, v={v.shape}")
        
        if self.manifold_type == "euclidean":
            self.logger.debug("Using Euclidean sectional curvature (zero)")
            return torch.zeros_like(x[..., 0])
        
        # Project to hyperboloid and tangent space
        x = self.exp_map.project_to_hyperboloid(x)
        u = self.exp_map.project_to_tangent(x, u)
        v = self.exp_map.project_to_tangent(x, v)
        
        # Compute inner products with high precision
        g_uu = self.exp_map.minkowski_inner(u, u)
        g_vv = self.exp_map.minkowski_inner(v, v)
        g_uv = self.exp_map.minkowski_inner(u, v)
        
        self.logger.debug(f"Inner products - g_uu: {g_uu}, g_vv: {g_vv}, g_uv: {g_uv}")
        
        # For hyperbolic space, the sectional curvature is -1
        return -torch.ones_like(g_uu)

    def compute_geodesic(
        self, x: torch.Tensor, v: torch.Tensor, steps: int = 100
    ) -> torch.Tensor:
        """Compute geodesic curve starting at x with initial velocity v."""
        self.logger.debug(f"Computing geodesic - shapes: x={x.shape}, v={v.shape}, steps={steps}")
        
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
            
            if t % 10 == 0:  # Log every 10 steps
                self.logger.debug(f"Geodesic step {t} - position norm: {torch.norm(curve[t+1])}")

        return curve

    def compute_geodesic_distance(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute geodesic distance between points."""
        self.logger.debug(f"Computing geodesic distance - shapes: x={x.shape}, y={y.shape}")
        
        # Project points to hyperbolic space
        if isinstance(self.exp_map, HyperbolicExponential):
            x_proj = self.exp_map.project_to_hyperboloid(x)
            y_proj = self.exp_map.project_to_hyperboloid(y)
            
            # For same points, return zero
            if torch.allclose(x_proj, y_proj, atol=1e-7):
                self.logger.debug("Points are identical, returning zero distance")
                return torch.zeros(x.shape[:-1], device=x.device)
            
            # Compute hyperbolic distance directly using Minkowski inner product
            inner = -self.exp_map.minkowski_inner(x_proj, y_proj)  # Should be ≥ 1
            inner = torch.clamp(inner, min=1.0 + 1e-7)  # Ensure we stay in valid range
            
            # Return arccosh of inner product (true hyperbolic distance)
            distance = torch.acosh(inner)
            self.logger.debug(f"Hyperbolic distance: {distance}")
            return distance
        else:
            # For Euclidean case, use standard Euclidean distance
            distance = torch.norm(y - x, p=2, dim=-1)
            self.logger.debug(f"Euclidean distance: {distance}")
            return distance

    def parallel_transport_batch(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport a batch of vectors v from points x to points y."""
        self.logger.debug(f"Batch parallel transport - shapes: x={x.shape}, y={y.shape}, v={v.shape}")
        
        if self.manifold_type == "euclidean":
            self.logger.debug("Using Euclidean parallel transport (identity)")
            return v
        
        # Project points to hyperboloid with high precision
        x = self.exp_map.project_to_hyperboloid(x)
        y = self.exp_map.project_to_hyperboloid(y)
        
        # Project v to tangent space at x with enhanced precision
        v = self.exp_map.project_to_tangent(x, v)
        
        # Store original inner product for preservation
        v_inner = self.exp_map.minkowski_inner(v, v)
        
        self.logger.debug(f"Original inner product: {v_inner}")
        
        # Compute Minkowski inner products with improved stability
        inner_xy = self.exp_map.minkowski_inner(x, y)
        inner_xv = self.exp_map.minkowski_inner(x, v)
        inner_yv = self.exp_map.minkowski_inner(y, v)
        
        self.logger.debug(f"Inner products - xy: {inner_xy}, xv: {inner_xv}, yv: {inner_yv}")
        
        # Transport coefficients with numerical safeguards
        alpha = inner_xy + 1.0  # Always positive by hyperboloid constraint
        alpha = alpha.clamp(min=1e-8)  # Prevent division by zero
        
        # Compute transport coefficients with improved stability
        beta = inner_xv / alpha
        gamma = inner_yv / alpha
        
        self.logger.debug(f"Transport coefficients - alpha: {alpha}, beta: {beta}, gamma: {gamma}")
        
        # Transport v to y using the parallel transport formula
        result = v - beta.unsqueeze(-1) * x - gamma.unsqueeze(-1) * y
        
        # Project result to tangent space at y with high precision
        result = self.exp_map.project_to_tangent(y, result)
        
        # Normalize to preserve inner product with improved stability
        result_inner = self.exp_map.minkowski_inner(result, result)
        scale = torch.sqrt(torch.abs(v_inner) / torch.abs(result_inner).clamp(min=1e-8))
        result = result * scale.unsqueeze(-1)
        
        self.logger.debug(f"Result inner product: {result_inner}, Scale: {scale}")
        
        # Final projection to ensure tangent space constraint
        result = self.exp_map.project_to_tangent(y, result)
        
        self.logger.debug(f"Final result norm: {torch.norm(result)}")
        
        return result

    def compute_exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Map point from tangent space to manifold."""
        self.logger.debug(f"Computing exponential map - shapes: x={x.shape}, v={v.shape}")
        return self.exp_map.forward(x, v)

    def compute_logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map point from manifold to tangent space."""
        self.logger.debug(f"Computing logarithmic map - shapes: x={x.shape}, y={y.shape}")
        return self.log_map.forward(x, y)

    def process_points(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process points through geometric operations with quantum integration."""
        self.logger.debug(f"Processing points - shapes: x={x.shape}, y={y.shape}")
        if v is not None:
            self.logger.debug(f"Vector shape: {v.shape}")
        
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
            
            self.logger.debug(f"Orthogonal vector norm: {torch.norm(v_perp)}")

            # Compute curvature with improved stability
            curvature = self.compute_sectional_curvature(x, v, v_perp)
            results["sectional_curvature"] = curvature
            
            self.logger.debug(f"Sectional curvature: {curvature}")

            # Add quantum geometric metrics
            if hasattr(self, "quantum_metric"):
                quantum_metric = torch.einsum(
                    "...i,...j,ij->...", v, v, self.quantum_metric
                )
                results["quantum_metric"] = quantum_metric
                self.logger.debug(f"Quantum metric: {quantum_metric}")

        return results

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process points through geometric operations."""
        self.logger.debug("Forward pass")
        return self.process_points(x, y, v, return_diagnostics)
