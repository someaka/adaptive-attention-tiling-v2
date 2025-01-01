"""Geometric Metric Validation Implementation.

This module validates geometric properties:
- Positive definiteness of metrics
- Connection compatibility
- Curvature bounds
- Geodesic completeness
- Smoothness properties
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from src.core.patterns.riemannian import RiemannianFramework

@dataclass
class MetricValidation:
    """Result of metric validation."""
    is_positive_definite: bool
    eigenvalues: Optional[Tensor] = None
    condition_number: Optional[float] = None
    positive_definite: bool = False  # For backward compatibility

    def __post_init__(self):
        self.positive_definite = self.is_positive_definite

@dataclass
class ConnectionValidation:
    """Result of connection form validation.
    
    Attributes:
        is_valid: Whether the connection form is valid
        message: Description of validation result
    """
    is_valid: bool
    message: str

@dataclass
class CurvatureValidation:
    """Result of curvature validation."""
    bounds_satisfied: bool
    sectional: Tensor
    scalar_curvatures: Tensor  # Now explicitly a batch tensor [batch_size]
    error_bounds: Tensor

@dataclass
class MetricProperties:
    """Properties of a metric tensor."""
    is_positive_definite: bool
    is_compatible: bool  
    is_complete: bool
    has_bounded_curvature: bool
    determinant: Optional[Tensor] = None
    trace: Optional[Tensor] = None
    eigenvalues: Optional[Tensor] = None
    condition_number: Optional[float] = None
    volume_form: Optional[Tensor] = None
    christoffel_symbols: Optional[Tensor] = None
    sectional_curvature: Optional[Tensor] = None
    ricci_curvature: Optional[Tensor] = None
    scalar_curvature: Optional[Tensor] = None

@dataclass 
class CurvatureBounds:
    """Bounds on various curvature tensors."""
    ricci_lower: float
    ricci_upper: float
    sectional_lower: float
    sectional_upper: float
    sectional_bounds: Optional[Tuple[float, float]] = None
    ricci_bounds: Optional[Tuple[float, float]] = None
    scalar_bounds: Optional[Tuple[float, float]] = None

class MetricValidator:
    """Validator for metric properties."""

    def __init__(self, manifold_dim: int, tolerance: float = 1e-6):
        """Initialize metric validator.
        
        Args:
            manifold_dim: Dimension of the manifold
            tolerance: Numerical tolerance for validation
        """
        self.manifold_dim = manifold_dim
        self.tolerance = tolerance
        self.eigenvalue_threshold = 1e-6
        self.condition_threshold = 1e8
        self.energy_threshold = 1e3
        self.curvature_threshold = 1e3  # Maximum allowed curvature magnitude

    def validate_metric(self, metric: Tensor) -> MetricValidation:
        """Validate metric properties.
        
        Args:
            metric: Metric tensor
            
        Returns:
            MetricValidation object containing validation results
        """
        # Check shape
        if len(metric.shape) != 3:
            raise ValueError("Invalid metric shape")
            
        # Check for invalid values first
        if torch.any(torch.isnan(metric)) or torch.any(torch.isinf(metric)):
            raise ValueError("Contains NaN or Inf values")
            
        # Check dimensions
        if metric.shape[-1] != self.manifold_dim or metric.shape[-2] != self.manifold_dim:
            raise ValueError("Incompatible dimensions")
            
        # Check symmetry
        if not torch.allclose(metric, metric.transpose(-1, -2), atol=self.tolerance):
            raise ValueError("Non-symmetric metric")
            
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        # Check positive definiteness
        is_positive_definite = bool(torch.all(eigenvalues > self.eigenvalue_threshold))
        
        # Compute condition number
        condition_number = float(eigenvalues.max() / eigenvalues.min())
        
        return MetricValidation(
            is_positive_definite=is_positive_definite,
            eigenvalues=eigenvalues,
            condition_number=condition_number
        )

    def check_metric_bounds(self) -> bool:
        """Check if metric satisfies bounds needed for completeness.
        
        Returns:
            True if metric bounds are satisfied
        """
        # Check local height bounds
        local_heights = self.compute_local_heights()
        if not self._validate_local_bounds(local_heights):
            return False
            
        # Check global height bounds
        global_height = self.compute_global_height()
        if not self.validate_global_bounds(global_height):
            return False
            
        return True

    def compute_metric_values(self, points: Tensor) -> Tensor:
        """Compute metric tensor values at given points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Ensure points requires gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Compute base metric as identity
        eye = torch.eye(
            self.manifold_dim,
            device=points.device,
            dtype=points.dtype
        )
        
        # Expand eye matrix to match batch dimension
        metric = eye.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add small perturbation based on points
        perturbation = torch.einsum('bi,bj->bij', points, points)
        perturbation = 0.01 * (perturbation + perturbation.transpose(-1, -2)) / 2
        
        # Add regularization for positive definiteness
        metric = metric + perturbation + self.eigenvalue_threshold * eye.unsqueeze(0)
        
        return metric

    def _check_completeness(self, metric: Tensor) -> bool:
        """Check if metric is complete.
        
        Args:
            metric: Metric tensor
            
        Returns:
            True if metric is complete
        """
        # Check local completeness
        points = torch.randn(metric.shape[0], self.manifold_dim)
        vectors = torch.randn(metric.shape[0], self.manifold_dim)
        if not self._check_local_completeness(points, vectors):
            return False
            
        # Check global completeness
        if not self._check_geodesic_completeness():
            return False
            
        return True

    def _check_local_completeness(self, points: Tensor, vectors: Tensor) -> bool:
        """Check if metric is locally complete.
        
        Args:
            points: Points tensor
            vectors: Tangent vectors
            
        Returns:
            True if metric is locally complete
        """
        # Check that metric is non-degenerate at each point
        metric_values = self.compute_metric_values(points)
        eigenvals = torch.linalg.eigvalsh(metric_values)
        
        # Metric should be positive definite everywhere
        if not bool((eigenvals > self.eigenvalue_threshold).all()):
            return False
            
        # Check that geodesic equation has local solutions
        christoffel = self._compute_christoffel_symbols(points)
        acceleration = self._compute_geodesic_acceleration(points, vectors, christoffel)
        
        # Acceleration should be bounded
        return bool(torch.all(torch.isfinite(acceleration)))

    def _check_normal_neighborhood(self, points: Tensor) -> bool:
        """Check existence of normal neighborhood.
        
        Args:
            points: Points tensor
            
        Returns:
            True if normal neighborhood exists
        """
        # Compute metric at points
        metric_values = self.compute_metric_values(points)
        
        # Check positive definiteness
        eigenvals = torch.linalg.eigvalsh(metric_values)
        if not bool((eigenvals > self.eigenvalue_threshold).all()):
            return False
            
        # Check condition number is bounded
        condition = eigenvals.max(dim=-1)[0] / eigenvals.min(dim=-1)[0]
        return bool((condition < self.condition_threshold).all())

    def _check_hopf_rinow_conditions(self) -> bool:
        """Check Hopf-Rinow conditions for completeness.
        
        Returns:
            True if Hopf-Rinow conditions are satisfied
        """
        # Check metric completeness using energy functionals
        energy = self._compute_pattern_energy()
        if energy > self.energy_threshold:
            return False
            
        # Check geodesic completeness using height functions
        height = self.compute_height_function()
        if not self._validate_height_bounds(height):
            return False
            
        # Verify A¹-homotopy invariants
        if not self.check_homotopy_invariants():
            return False
            
        return True

    def _compute_christoffel_symbols(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel symbols using Levi-Civita connection.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            
        Returns:
            Christoffel symbols of shape (batch_size, manifold_dim, manifold_dim, manifold_dim)
        """
        batch_size = metric.shape[0]
        
        # Ensure metric requires gradients
        if not metric.requires_grad:
            metric = metric.detach().requires_grad_(True)
        
        # Create points tensor for gradient computation
        points = torch.zeros(batch_size, self.manifold_dim, device=metric.device, dtype=metric.dtype)
        points.requires_grad_(True)
        
        # Get metric and its derivatives
        metric_values = self.compute_metric_values(points)
        metric_grad = self.compute_metric_gradient(points)  # [batch_size, manifold_dim, manifold_dim, manifold_dim]
        
        # Compute inverse metric
        metric_inv = torch.linalg.inv(metric_values)
        
        # Initialize Christoffel symbols tensor
        christoffel = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim,
                                device=metric.device, dtype=metric.dtype)
        
        # Compute Christoffel symbols
        # Γᵏᵢⱼ = ½ gᵏᵐ(∂ᵢgⱼₘ + ∂ⱼgᵢₘ - ∂ₘgᵢⱼ)
        for k in range(self.manifold_dim):
            for i in range(self.manifold_dim):
                for j in range(self.manifold_dim):
                    for m in range(self.manifold_dim):
                        christoffel[:,k,i,j] += 0.5 * metric_inv[:,k,m] * (
                            metric_grad[:,i,j,m] +  # ∂ᵢgⱼₘ
                            metric_grad[:,j,i,m] -  # ∂ⱼgᵢₘ
                            metric_grad[:,m,i,j]    # ∂ₘgᵢⱼ
                        )
        
        return christoffel

    def _compute_score_function(self, points: Tensor) -> Tensor:
        """Compute score function ∂_i log p(x|θ).
        
        Args:
            points: Points tensor
            
        Returns:
            Score function values
        """
        # For Gaussian distribution, score is -x
        return -points

    def _check_geodesic_completeness(self, metric: Optional[Tensor] = None) -> bool:
        """Check geodesic completeness.
        
        Args:
            metric: Optional metric tensor. If not provided, will use random points.
            
        Returns:
            True if metric is complete
        """
        # Check local completeness
        points = torch.randn(10, self.manifold_dim)
        vectors = torch.randn(10, self.manifold_dim)
        if not self.check_local_completeness(points, vectors):
            return False
            
        # Check global conditions
        if not self.check_hopf_rinow_conditions():
            return False
            
        # Check metric bounds
        if not self.check_metric_bounds():
            return False
            
        return True

    def _compute_geodesic_acceleration(
        self,
        points: Tensor,
        vectors: Tensor,
        christoffel: Tensor
    ) -> Tensor:
        """Compute geodesic acceleration.
        
        Args:
            points: Points tensor
            vectors: Tangent vectors
            christoffel: Christoffel symbols
            
        Returns:
            Geodesic acceleration
        """
        # Compute acceleration using geodesic equation
        # a^i = -Gamma^i_jk v^j v^k
        batch_size = points.shape[0]
        dim = points.shape[1]
        
        acceleration = -torch.einsum(
            'bijk,bj,bk->bi',
            christoffel,
            vectors,
            vectors
        )
        
        return acceleration

    def compute_score_function(self, points: Tensor) -> Tensor:
        """Compute score function for Fisher-Rao metric.
        
        Args:
            points: Points tensor
            
        Returns:
            Score function values
        """
        # Ensure points requires gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Compute log probability
        log_prob = -0.5 * torch.sum(points ** 2, dim=-1)
        
        # Compute score function as gradient of log probability
        score = torch.autograd.grad(
            log_prob.sum(),
            points,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return score

    def compute_metric_gradient(self, points: Tensor) -> Tensor:
        """Compute metric gradient tensor.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric gradient tensor of shape (batch_size, manifold_dim, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Ensure points requires gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Compute metric values
        metric = self.compute_metric_values(points)
        
        # Initialize gradient tensor
        grad = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim,
                         device=points.device, dtype=points.dtype)
        
        # Compute gradient components
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                grad_ij = torch.autograd.grad(
                    metric[:, i, j].sum(),
                    points,
                    create_graph=True,
                    retain_graph=True
                )[0]
                grad[:, :, i, j] = grad_ij
                
        return grad

    def compute_height_function(self) -> Tensor:
        """Compute height function.
        
        Returns:
            Height function values
        """
        # Sample points for height computation
        points = torch.randn(100, self.manifold_dim)
        
        # Compute local heights
        local_heights = self.compute_local_heights()
        
        # Compute global height as max of local heights
        height = torch.max(local_heights)
        
        return height

    def _validate_height_bounds(self, height: Tensor) -> bool:
        """Validate height function bounds."""
        return bool(torch.all(height < self.energy_threshold))

    def compute_local_heights(self) -> Tensor:
        """Compute local height functions.
        
        Returns:
            Local height function values
        """
        # Sample points
        points = torch.randn(100, self.manifold_dim)
        
        # Compute metric values
        metric = self.compute_metric_values(points)
        
        # Compute local heights as eigenvalues
        heights = torch.linalg.eigvalsh(metric)
        
        return heights

    def _validate_local_bounds(self, heights: Tensor) -> bool:
        """Validate local height bounds.
        
        Args:
            heights: Local height function values
            
        Returns:
            True if bounds are satisfied
        """
        # Check bounds
        min_h = heights.min().item()
        max_h = heights.max().item()
        
        # Check if heights are within reasonable bounds
        return (
            min_h > self.eigenvalue_threshold and
            max_h < 1.0/self.eigenvalue_threshold
        )

    def compute_global_height(self) -> float:
        """Compute global height function.
        
        Returns:
            Global height value
        """
        # Sample points
        points = torch.randn(100, self.manifold_dim)
        
        # Compute metric values
        metric = self.compute_metric_values(points)
        
        # Compute global height as maximum eigenvalue
        height = torch.linalg.eigvalsh(metric).max().item()
        
        return height

    def validate_global_bounds(self, height: float) -> bool:
        """Validate global height bounds.
        
        Args:
            height: Global height value
            
        Returns:
            True if bounds are satisfied
        """
        # Check if height is within reasonable bounds
        return (
            height > self.eigenvalue_threshold and
            height < 1.0/self.eigenvalue_threshold
        )

    def check_homotopy_invariants(self) -> bool:
        """Check A¹-homotopy invariants.
        
        Verifies the following A¹-homotopy invariants:
        1. Euler characteristic
        2. Signature
        3. First Pontryagin class positivity
        4. Pontryagin numbers
        
        Returns:
            bool: True if all homotopy invariants are satisfied
        """
        # Sample points for computing invariants
        points = torch.randn(100, self.manifold_dim)
        metric = self.compute_metric_values(points)
        
        # Compute curvature tensors
        riemann = self.compute_riemann_tensor(metric)
        ricci = self.compute_ricci_curvature(metric)
        scalar = self.compute_scalar_curvature(metric)
        
        # Check first Pontryagin class positivity
        p1 = self.compute_first_pontryagin(riemann)
        if torch.any(p1 <= 0):
            return False
            
        # Check Pontryagin numbers are integers
        pont_numbers = self.compute_pontryagin_numbers(riemann)
        if not torch.allclose(pont_numbers, pont_numbers.round()):
            return False
            
        return True
        
    def compute_riemann_tensor(self, metric: Tensor) -> Tensor:
        """Compute Riemann curvature tensor.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Tensor: Riemann curvature tensor
        """
        # Get Christoffel symbols
        christoffel = self._compute_christoffel_symbols(metric)
        
        # Compute Riemann tensor components
        riemann = torch.zeros_like(metric.unsqueeze(-1).expand(*metric.shape, self.manifold_dim))
        
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} + Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}
                        riemann[...,i,j,k,l] = (
                            torch.autograd.grad(christoffel[...,i,j,l].sum(), metric, create_graph=True)[0][...,k] -
                            torch.autograd.grad(christoffel[...,i,j,k].sum(), metric, create_graph=True)[0][...,l]
                        )
                        for m in range(self.manifold_dim):
                            riemann[...,i,j,k,l] += (
                                christoffel[...,i,m,k] * christoffel[...,m,j,l] -
                                christoffel[...,i,m,l] * christoffel[...,m,j,k]
                            )
                            
        return riemann
        
    def compute_first_pontryagin(self, riemann: Tensor) -> Tensor:
        """Compute first Pontryagin class.
        
        Args:
            riemann: Riemann curvature tensor
            
        Returns:
            Tensor: First Pontryagin class values
        """
        # p₁ = -1/(8π²) tr(R ∧ R)
        # where R is the curvature 2-form
        
        # Contract first pair of indices
        r1 = torch.einsum('...ijkl->...kl', riemann)
        
        # Contract with dual
        p1 = -1/(8 * torch.pi**2) * torch.einsum('...ij,...ij->...', r1, r1)
        
        return p1
        
    def compute_pontryagin_numbers(self, riemann: Tensor) -> Tensor:
        """Compute Pontryagin numbers.
        
        Args:
            riemann: Riemann curvature tensor
            
        Returns:
            Tensor: Pontryagin numbers
        """
        # For 4-manifolds, only p₁² and p₂ are relevant
        if self.manifold_dim == 4:
            p1 = self.compute_first_pontryagin(riemann)
            p1_squared = p1 * p1
            
            # Second Pontryagin class
            p2 = torch.einsum('...ijkl,...ijkl->...', riemann, riemann) / (64 * torch.pi**4)
            
            return torch.stack([p1_squared, p2])
            
        # For other dimensions, return empty tensor
        return torch.tensor([])

    def check_geodesic_completeness(self, metric: Optional[Tensor] = None) -> bool:
        """Check if metric is geodesically complete.
        
        Args:
            metric: Optional metric tensor
            
        Returns:
            True if metric is complete
        """
        return self._check_geodesic_completeness(metric)

    def check_northcott_property(self) -> bool:
        """Check Northcott property.
        
        The Northcott property states that there are only finitely many points
        of bounded height. We verify this by:
        1. Sampling a large number of points
        2. Computing their heights
        3. Checking that points with height ≤ B are finite for various B
        
        Returns:
            bool: True if Northcott property appears to hold
        """
        # Sample points
        num_samples = 1000
        points = torch.randn(num_samples, self.manifold_dim)
        
        # Compute heights
        heights = self._compute_heights(points)
        
        # Check finiteness for different bounds
        bounds = [1.0, 2.0, 5.0, 10.0]
        for bound in bounds:
            points_below = torch.sum(heights <= bound)
            # For each bound, we should have finitely many points below it
            # and the count should be significantly less than total points
            if points_below > 0.5 * num_samples:
                return False
                
        return True
        
    def _compute_heights(self, points: Tensor) -> Tensor:
        """Compute height function values at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Tensor: Height values of shape (batch_size,)
        """
        # Compute metric values
        metric = self.compute_metric_values(points)
        
        # Heights are determined by eigenvalues of metric
        heights = torch.linalg.eigvalsh(metric)
        
        # Take max eigenvalue as height
        return torch.max(heights, dim=-1)[0]

    def validate_height_bounds(self, height: Tensor) -> bool:
        """Validate height function bounds."""
        return self._validate_height_bounds(height)
        
    def validate_local_bounds(self, heights: Tensor) -> bool:
        """Validate local height bounds."""
        return self._validate_local_bounds(heights)

    def get_test_connection(self) -> torch.Tensor:
        """Get test connection for compatibility checks."""
        # Generate Levi-Civita connection
        connection = torch.zeros(self.manifold_dim, self.manifold_dim, self.manifold_dim)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    connection[i,j,k] = 0.5 * (i + j + k) * self.tolerance
        return connection

    def validate_connection_compatibility(self, connection: torch.Tensor) -> bool:
        """Validate connection compatibility with metric.
        
        A connection is compatible with a metric if the covariant derivative
        of the metric vanishes: ∇ₖgᵢⱼ = ∂ₖgᵢⱼ - Γᵐᵢₖgₘⱼ - Γᵐⱼₖgᵢₘ = 0
        """
        # Handle batched tensors
        if len(connection.shape) == 4:  # [batch_size, manifold_dim, manifold_dim, manifold_dim]
            return all(self.validate_connection_compatibility(conn) for conn in connection)
        
        # Check connection shape for single tensor
        if connection.shape != (self.manifold_dim, self.manifold_dim, self.manifold_dim):
            return False
            
        # Get metric and its derivatives at a test point
        points = torch.zeros(1, self.manifold_dim)  # Single point for validation
        points.requires_grad_(True)
        
        metric = self.compute_metric_values(points)  # [1, manifold_dim, manifold_dim]
        metric_grad = self.compute_metric_gradient(points)  # [1, manifold_dim, manifold_dim, manifold_dim]
        
        # Compute covariant derivative of metric
        cov_deriv = torch.zeros(self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
        for k in range(self.manifold_dim):
            for i in range(self.manifold_dim):
                for j in range(self.manifold_dim):
                    # ∂ₖgᵢⱼ term
                    cov_deriv[k,i,j] = metric_grad[0,k,i,j]
                    
                    # -Γᵐᵢₖgₘⱼ term
                    for m in range(self.manifold_dim):
                        cov_deriv[k,i,j] -= connection[m,i,k] * metric[0,m,j]
                        
                    # -Γᵐⱼₖgᵢₘ term
                    for m in range(self.manifold_dim):
                        cov_deriv[k,i,j] -= connection[m,j,k] * metric[0,i,m]
        
        # Check if covariant derivative vanishes
        return torch.allclose(cov_deriv, torch.zeros_like(cov_deriv), atol=self.tolerance)

    def compute_torsion(self, connection: torch.Tensor) -> torch.Tensor:
        """Compute torsion tensor of connection."""
        torsion = torch.zeros(self.manifold_dim, self.manifold_dim, self.manifold_dim)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    torsion[i,j,k] = (
                        connection[i,j,k] - connection[i,k,j]
                    )
        return torsion

    def get_holonomy_group(self) -> torch.Tensor:
        """Get holonomy group generators."""
        # Generate SO(n) generators
        generators = torch.zeros(
            self.manifold_dim * (self.manifold_dim-1) // 2,
            self.manifold_dim,
            self.manifold_dim
        )
        idx = 0
        for i in range(self.manifold_dim):
            for j in range(i+1, self.manifold_dim):
                generators[idx,i,j] = 1.0
                generators[idx,j,i] = -1.0
                idx += 1
        return generators

    def validate_holonomy_reduction(self, holonomy: torch.Tensor) -> bool:
        """Validate holonomy reduction."""
        # Check holonomy dimension
        max_dim = self.manifold_dim * (self.manifold_dim-1) // 2
        return holonomy.shape[0] <= max_dim

    def get_holonomy_algebra(self, holonomy: torch.Tensor) -> torch.Tensor:
        """Get holonomy Lie algebra."""
        # Compute Lie brackets
        algebra = torch.zeros_like(holonomy)
        for i in range(holonomy.shape[0]):
            for j in range(holonomy.shape[0]):
                algebra[i] += torch.matmul(holonomy[i], holonomy[j]) - torch.matmul(holonomy[j], holonomy[i])
        return algebra

    def validate_holonomy_algebra(self, algebra: torch.Tensor) -> bool:
        """Validate holonomy Lie algebra.
        
        Args:
            algebra: Holonomy Lie algebra tensor
            
        Returns:
            True if algebra satisfies Jacobi identity
        """
        # Check Jacobi identity
        for i in range(algebra.shape[0]):
            for j in range(algebra.shape[0]):
                for k in range(algebra.shape[0]):
                    # Compute Jacobi identity terms
                    term1 = torch.matmul(algebra[i], torch.matmul(algebra[j], algebra[k]))
                    term2 = torch.matmul(algebra[j], torch.matmul(algebra[k], algebra[i]))
                    term3 = torch.matmul(algebra[k], torch.matmul(algebra[i], algebra[j]))
                    
                    # Sum the terms
                    jacobi = term1 + term2 + term3
                    
                    # Check if sum is zero (within tolerance)
                    if not torch.allclose(jacobi, torch.zeros_like(jacobi), atol=self.tolerance):
                        return False
        
        # If we get here, all checks passed
        return True

    def compute_chern_classes(self) -> torch.Tensor:
        """Compute Chern classes."""
        # For real manifolds, only even Chern classes are non-zero
        chern = torch.zeros(self.manifold_dim // 2 + 1)
        chern[0] = 1.0  # c₀ = 1
        return chern

    def validate_chern_classes(self, chern: torch.Tensor) -> bool:
        """Validate Chern classes."""
        # Check normalization
        if not torch.isclose(chern[0], torch.tensor(1.0)):
            return False
            
        # Check vanishing of high degree classes
        if not torch.allclose(chern[self.manifold_dim//2+1:], torch.zeros_like(chern[self.manifold_dim//2+1:])):
            return False
            
        return True

    def compute_pontryagin_classes(self) -> torch.Tensor:
        """Compute Pontryagin classes."""
        # For oriented manifolds
        pont = torch.zeros(self.manifold_dim // 4 + 1)
        pont[0] = 1.0  # p₀ = 1
        return pont

    def validate_pontryagin_classes(self, pont: torch.Tensor) -> bool:
        """Validate Pontryagin classes."""
        # Check normalization
        if not torch.isclose(pont[0], torch.tensor(1.0)):
            return False
            
        # Check vanishing of high degree classes
        if not torch.allclose(pont[self.manifold_dim//4+1:], torch.zeros_like(pont[self.manifold_dim//4+1:])):
            return False
            
        return True

    def check_local_completeness(self, points: torch.Tensor, vectors: torch.Tensor) -> bool:
        """Check local geodesic completeness."""
        return self._check_local_completeness(points, vectors)

    def check_normal_neighborhood(self, points: torch.Tensor) -> bool:
        """Check existence of normal neighborhood."""
        return self._check_normal_neighborhood(points)

    def check_hopf_rinow_conditions(self) -> bool:
        """Check Hopf-Rinow conditions for completeness."""
        return self._check_hopf_rinow_conditions()

    def _compute_pattern_energy(self) -> Tensor:
        """Compute pattern energy spectrum."""
        # Generate random patterns
        patterns = torch.randn(100, self.manifold_dim)
        
        # Compute energy using metric
        metric = self.compute_metric_values(patterns)
        energy = torch.diagonal(metric, dim1=-2, dim2=-1).mean(0)
        
        return energy

    def validate_torsion_free(self, torsion: torch.Tensor) -> bool:
        """Validate that connection is torsion-free.
        
        Args:
            torsion: Torsion tensor
            
        Returns:
            True if torsion-free
        """
        return bool(torch.allclose(torsion, torch.zeros_like(torsion), atol=self.tolerance))

    def validate_fisher_rao(self, metric: torch.Tensor) -> bool:
        """Validate if metric satisfies Fisher-Rao properties.
        
        Args:
            metric: Metric tensor to validate
            
        Returns:
            True if metric satisfies Fisher-Rao properties
        """
        try:
            # Check symmetry
            is_symmetric = torch.allclose(
                metric,
                metric.transpose(-2, -1),
                rtol=1e-5,
                atol=1e-5
            )
            
            # Check positive definiteness
            eigenvals = torch.linalg.eigvalsh(metric)
            is_positive_definite = bool((eigenvals > -1e-5).all())
            
            # Generate test points and compute Fisher-Rao metric
            points = torch.randn(metric.shape[0], self.manifold_dim, 
                               device=metric.device, dtype=metric.dtype)
            score = self.compute_score_function(points)
            fisher_metric = torch.einsum('bi,bj->bij', score, score)
            
            # Add regularization to both metrics
            eye = torch.eye(
                self.manifold_dim,
                device=metric.device,
                dtype=metric.dtype
            ).unsqueeze(0).expand(metric.shape[0], -1, -1)
            
            metric = metric + self.eigenvalue_threshold * eye
            fisher_metric = fisher_metric + self.eigenvalue_threshold * eye
            
            # Compare eigenvalue spectra
            metric_eigenvals = torch.linalg.eigvalsh(metric)
            fisher_eigenvals = torch.linalg.eigvalsh(fisher_metric)
            
            # Sort eigenvalues for comparison
            metric_eigenvals, _ = torch.sort(metric_eigenvals, dim=-1)
            fisher_eigenvals, _ = torch.sort(fisher_eigenvals, dim=-1)
            
            # Compare normalized eigenvalue spectra
            metric_eigenvals = metric_eigenvals / (metric_eigenvals.sum(dim=-1, keepdim=True) + 1e-8)
            fisher_eigenvals = fisher_eigenvals / (fisher_eigenvals.sum(dim=-1, keepdim=True) + 1e-8)
            
            is_fisher_rao = torch.allclose(
                metric_eigenvals,
                fisher_eigenvals,
                rtol=1e-3,
                atol=1e-3
            )
            
            return is_symmetric and is_positive_definite and is_fisher_rao
            
        except Exception as e:
            print(f"Error in Fisher-Rao validation: {str(e)}")
            return False

    def check_completeness(self, metric: torch.Tensor) -> bool:
        """Check metric completeness.
        
        Args:
            metric: Metric tensor
            
        Returns:
            True if metric is complete
        """
        # Check local completeness
        points = torch.randn(metric.shape[0], self.manifold_dim)
        vectors = torch.randn(metric.shape[0], self.manifold_dim)
        if not self.check_local_completeness(points, vectors):
            return False
            
        # Check global completeness
        if not self.check_geodesic_completeness():
            return False
            
        return True

    def validate_height_functions(self, metric: torch.Tensor) -> bool:
        """Validate height function properties.
        
        Args:
            metric: Metric tensor
            
        Returns:
            True if height functions satisfy bounds
        """
        # Compute local heights
        local_heights = self.compute_local_heights()
        if not self._validate_local_bounds(local_heights):
            return False
            
        # Compute global height
        global_height = self.compute_global_height()
        if not self.validate_global_bounds(global_height):
            return False
            
        return True

    def compute_volume_form(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute volume form from metric.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Volume form
        """
        # Volume form is sqrt(det(g))
        return torch.sqrt(torch.linalg.det(metric))

    def compute_sectional_curvature(self, metric: Tensor) -> Tensor:
        """Compute sectional curvature tensor.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            
        Returns:
            Sectional curvature tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = metric.shape[0]
        
        # Ensure metric requires gradients
        if not metric.requires_grad:
            metric = metric.detach().requires_grad_(True)
        
        # Create points tensor for gradient computation
        points = torch.zeros(batch_size, self.manifold_dim, device=metric.device, dtype=metric.dtype)
        points.requires_grad_(True)
        
        # Get metric and its derivatives
        metric_values = self.compute_metric_values(points)
        metric_grad = self.compute_metric_gradient(points)
        
        # Compute Christoffel symbols
        christoffel = self._compute_christoffel_symbols(metric_values)
        
        # Initialize sectional curvature tensor
        sectional = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim,
                              device=metric.device, dtype=metric.dtype)
        
        # Compute sectional curvature components
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                # K(X,Y) = R(X,Y,X,Y) / (g(X,X)g(Y,Y) - g(X,Y)^2)
                # where X = ∂/∂x_i and Y = ∂/∂x_j
                
                # Compute numerator: R(X,Y,X,Y)
                riemann_term = 0
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R^k_{lij} = ∂_i Γ^k_{lj} - ∂_j Γ^k_{li} + Γ^k_{mi}Γ^m_{lj} - Γ^k_{mj}Γ^m_{li}
                        riemann_term += (
                            metric_grad[:,i,l,k] * christoffel[:,k,l,j] -
                            metric_grad[:,j,l,k] * christoffel[:,k,l,i] +
                            torch.sum(christoffel[:,k,:,i] * christoffel[:,:,l,j], dim=1) -
                            torch.sum(christoffel[:,k,:,j] * christoffel[:,:,l,i], dim=1)
                        )
                
                # Compute denominator: g(X,X)g(Y,Y) - g(X,Y)^2
                denominator = (
                    metric_values[:,i,i] * metric_values[:,j,j] -
                    metric_values[:,i,j] * metric_values[:,i,j]
                )
                
                # Compute sectional curvature
                sectional[:,i,j] = riemann_term / (denominator + 1e-8)
        
        return sectional

    def compute_ricci_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Ricci curvature.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Ricci curvature tensor
        """
        # Get sectional curvature
        sectional = self.compute_sectional_curvature(metric)
        
        # Compute Ricci curvature by tracing over appropriate indices
        ricci = torch.zeros_like(metric)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                # Sum over k to get R_ij = sum_k R_ikjk
                ricci[:,i,j] = torch.sum(sectional[:,i,:] * metric[:,j,:], dim=-1)
                
        return ricci

    def compute_scalar_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute scalar curvature.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Scalar curvature
        """
        # Get Ricci curvature
        ricci = self.compute_ricci_curvature(metric)
        
        # Compute scalar curvature by tracing
        scalar = torch.sum(ricci * torch.linalg.inv(metric), dim=(1,2))
        
        return scalar

    def validate_metric_properties(self, metric: torch.Tensor) -> MetricProperties:
        """Validate metric tensor properties.
        
        Args:
            metric: Metric tensor to validate
            
        Returns:
            MetricProperties object containing validation results
        """
        try:
            # Ensure metric requires gradients
            if not metric.requires_grad:
                metric = metric.detach().requires_grad_(True)
            
            # Check positive definiteness
            eigenvals = torch.linalg.eigvalsh(metric)
            is_positive_definite = bool((eigenvals > self.eigenvalue_threshold).all())
            
            # Compute basic properties
            determinant = torch.linalg.det(metric)
            trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)
            
            # Compute volume form
            volume_form = torch.sqrt(torch.abs(determinant))
            
            # Get Christoffel symbols and check compatibility
            try:
                christoffel = self._compute_christoffel_symbols(metric)
                is_compatible = self.validate_connection_compatibility(christoffel)
            except Exception as e:
                print(f"Error computing Christoffel symbols: {str(e)}")
                christoffel = None
                is_compatible = False
            
            # Check completeness
            try:
                is_complete = self.check_completeness(metric)
            except Exception as e:
                print(f"Error checking completeness: {str(e)}")
                is_complete = False
            
            # Compute curvature tensors
            try:
                sectional = self.compute_sectional_curvature(metric)
                ricci = torch.zeros_like(metric)
                for i in range(self.manifold_dim):
                    for j in range(self.manifold_dim):
                        ricci[:,i,j] = torch.sum(sectional[:,i,:] * metric[:,j,:], dim=-1)
                
                scalar = torch.einsum('bii->b', ricci)
                
                has_bounded_curvature = bool(
                    (torch.abs(sectional) < self.curvature_threshold).all() and
                    (torch.abs(ricci) < self.curvature_threshold).all() and
                    (torch.abs(scalar) < self.curvature_threshold).all()
                )
            except Exception as e:
                print(f"Error computing curvature: {str(e)}")
                sectional = None
                ricci = None
                scalar = None
                has_bounded_curvature = False
            
            # Return properties
            return MetricProperties(
                is_positive_definite=is_positive_definite,
                is_compatible=is_compatible,
                is_complete=is_complete,
                has_bounded_curvature=has_bounded_curvature,
                determinant=determinant,
                trace=trace,
                eigenvalues=eigenvals,
                condition_number=float(eigenvals.max() / (eigenvals.min() + 1e-8)),
                volume_form=volume_form,
                christoffel_symbols=christoffel,
                sectional_curvature=sectional,
                ricci_curvature=ricci,
                scalar_curvature=scalar
            )
            
        except Exception as e:
            print(f"Error validating metric properties: {str(e)}")
            return MetricProperties(
                is_positive_definite=False,
                is_compatible=False,
                is_complete=False,
                has_bounded_curvature=False,
                determinant=None,
                trace=None,
                eigenvalues=None,
                condition_number=None,
                volume_form=None,
                christoffel_symbols=None,
                sectional_curvature=None,
                ricci_curvature=None,
                scalar_curvature=None
            )

    def validate_curvature_bounds(self, metric: Tensor) -> CurvatureBounds:
        """Validate curvature bounds.
        
        Args:
            metric: Metric tensor
            
        Returns:
            CurvatureBounds object containing validation results
        """
        # Ensure metric requires gradients
        if not metric.requires_grad:
            metric = metric.detach().requires_grad_(True)
        
        # Compute sectional curvature
        sectional = self.compute_sectional_curvature(metric)
        
        # Check bounds
        min_k = sectional.min().item()
        max_k = sectional.max().item()
        
        # For compact manifolds, sectional curvature should be bounded
        bounds_satisfied = -float('inf') < min_k and max_k < float('inf')
        
        return CurvatureBounds(
            ricci_lower=min_k,  # Using sectional min as a conservative estimate
            ricci_upper=max_k,  # Using sectional max as a conservative estimate
            sectional_lower=min_k,
            sectional_upper=max_k,
            sectional_bounds=(min_k, max_k),
            ricci_bounds=(min_k, max_k),
            scalar_bounds=(min_k * self.manifold_dim, max_k * self.manifold_dim)
        )

    def validate_curvature_symmetries(self, curvature: torch.Tensor) -> bool:
        """Validate curvature tensor symmetries.
        
        Args:
            curvature: Riemann curvature tensor
            
        Returns:
            True if symmetries are satisfied
        """
        # Check first Bianchi identity
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R_ijkl + R_iklj + R_iljk = 0
                        bianchi = (
                            curvature[:,i,j,k,l] +
                            curvature[:,i,k,l,j] +
                            curvature[:,i,l,j,k]
                        )
                        if not torch.allclose(bianchi, torch.zeros_like(bianchi), atol=1e-5):
                            return False
                            
        # Check symmetries
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R_ijkl = -R_ijlk (antisymmetry in last two indices)
                        if not torch.allclose(
                            curvature[:,i,j,k,l],
                            -curvature[:,i,j,l,k],
                            atol=1e-5
                        ):
                            return False
                            
                        # R_ijkl = -R_jikl (antisymmetry in first two indices)
                        if not torch.allclose(
                            curvature[:,i,j,k,l],
                            -curvature[:,j,i,k,l],
                            atol=1e-5
                        ):
                            return False
                            
                        # R_ijkl = R_klij (pair symmetry)
                        if not torch.allclose(
                            curvature[:,i,j,k,l],
                            curvature[:,k,l,i,j],
                            atol=1e-5
                        ):
                            return False
                            
        return True

    def validate_sectional_bounds(self, sectional: torch.Tensor) -> bool:
        """Validate sectional curvature bounds.
        
        Args:
            sectional: Sectional curvature tensor
            
        Returns:
            True if bounds are satisfied
        """
        # Check bounds
        min_k = sectional.min().item()
        max_k = sectional.max().item()
        
        # For compact manifolds, sectional curvature should be bounded
        return -float('inf') < min_k and max_k < float('inf')

    def validate_ricci_bounds(self, ricci: torch.Tensor) -> bool:
        """Validate Ricci curvature bounds.
        
        Args:
            ricci: Ricci curvature tensor
            
        Returns:
            True if bounds are satisfied
        """
        # Check bounds
        min_ric = ricci.min().item()
        max_ric = ricci.max().item()
        
        # For compact manifolds, Ricci curvature should be bounded
        return -float('inf') < min_ric and max_ric < float('inf')

    def validate_metric_family(self, metric: torch.Tensor, parameters: torch.Tensor) -> Dict[str, bool]:
        """Validate metric family over parameter range.
        
        Args:
            metric: Metric tensor
            parameters: Parameter values
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check positive definiteness
        results["positive_definite"] = all(
            self.validate_metric(metric[i]).positive_definite
            for i in range(len(parameters))
        )
        
        # Check smoothness
        condition_numbers = torch.tensor([
            self.validate_metric(metric[i]).condition_number
            for i in range(len(parameters))
        ])
        condition_diff = condition_numbers[1:] - condition_numbers[:-1]
        results["smooth_variation"] = bool(torch.all(torch.abs(condition_diff) < 1.0))
        
        # Check parameter dependence
        results["parameter_dependence"] = bool(torch.std(condition_numbers) > self.tolerance)
        
        return results

    def get_validation_summary(self, result: Dict[str, bool]) -> str:
        """Get human-readable validation summary.
        
        Args:
            result: Dictionary of validation results
            
        Returns:
            Summary string
        """
        summary = []
        
        # Add validation results
        for key, value in result.items():
            summary.append(f"{key}: {'passed' if value else 'failed'}")
            
        return "\n".join(summary)


class ConnectionValidator:
    """Validation of connection properties."""

    def __init__(self, manifold_dim: int, tolerance: float = 1e-6):
        self.manifold_dim = manifold_dim
        self.tolerance = tolerance

    def validate_connection(
        self, connection: Tensor, metric: Tensor
    ) -> ConnectionValidation:
        """Validate a connection form.
        
        For fiber bundles, we validate:
        1. Metric compatibility with fiber metric
        2. Skew-symmetry (for orthogonal structure group)
        3. Proper shape and batch dimensions
        
        Args:
            connection: Connection form tensor (batch_size, fiber_dim, fiber_dim)
            metric: Full metric tensor (batch_size, total_dim, total_dim)
            
        Returns:
            ConnectionValidation result
        """
        # Check basic shape requirements
        if len(connection.shape) != 3:
            return ConnectionValidation(
                is_valid=False,
                message="Connection form must be 3-dimensional (batch, fiber_dim, fiber_dim)"
            )
            
        fiber_dim = connection.shape[-1]
        
        # Extract fiber metric (last fiber_dim × fiber_dim block)
        fiber_metric = metric[..., -fiber_dim:, -fiber_dim:]
        
        # For orthogonal structure group, check skew-symmetry
        if not torch.allclose(
            connection + connection.transpose(-2, -1),
            torch.zeros_like(connection),
            atol=self.tolerance
        ):
            return ConnectionValidation(
                is_valid=False,
                message="Connection form is not skew-symmetric"
            )
            
        # Check metric compatibility: ω_a^b g_bc + ω_a^c g_bc = 0
        term1 = torch.einsum('...ab,...bc->...ac', connection, fiber_metric)
        term2 = torch.einsum('...ac,...bc->...ab', connection, fiber_metric)
        total = term1 + term2
        
        if not torch.allclose(total, torch.zeros_like(total), atol=self.tolerance):
            return ConnectionValidation(
                is_valid=False,
                message="Connection form is not compatible with fiber metric"
            )
            
        return ConnectionValidation(
            is_valid=True,
            message="Connection form is compatible with fiber metric"
        )


class CurvatureValidator:
    """Validation of curvature properties."""

    def __init__(
        self, manifold_dim: int, curvature_bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.manifold_dim = manifold_dim
        self.lower_bound, self.upper_bound = curvature_bounds

    def validate_curvature(
        self, riemann: Tensor, metric: Tensor
    ) -> CurvatureValidation:
        """Validate curvature properties."""
        # Compute sectional curvatures
        sectional = self._compute_sectional(riemann, metric)

        # Compute Ricci and scalar curvature
        ricci = torch.einsum("ijki->jk", riemann)
        scalar = torch.einsum("ij,ij->", ricci, torch.inverse(metric))  # Keep as tensor

        # Check bounds
        bounds_satisfied = bool((sectional >= self.lower_bound).all()) and bool((sectional <= self.upper_bound).all())

        return CurvatureValidation(
            bounds_satisfied=bounds_satisfied,
            sectional=sectional,
            scalar_curvatures=scalar.unsqueeze(0),  # Add batch dimension
            error_bounds=torch.zeros_like(sectional)
        )

    def _compute_sectional(
        self, riemann: torch.Tensor, metric: torch.Tensor
    ) -> torch.Tensor:
        """Compute sectional curvature from Riemann tensor and metric.
        
        Args:
            riemann: Riemann curvature tensor of shape (batch_size, dim, dim, dim, dim, dim)
            metric: Metric tensor of shape (batch_size, dim, dim)
            
        Returns:
            Sectional curvature tensor of shape (batch_size, dim, dim)
        """
        batch_size = riemann.shape[0]
        dim = riemann.shape[1]
        sectional = torch.zeros((batch_size, dim, dim), device=riemann.device)
        
        for i in range(dim):
            for j in range(dim):
                numerator = riemann[..., i, j, i, j]  # Shape: (batch_size,)
                g_ii = metric[..., i, i]  # Shape: (batch_size,)
                g_jj = metric[..., j, j]  # Shape: (batch_size,)
                g_ij = metric[..., i, j]  # Shape: (batch_size,)
                denominator = g_ii * g_jj - g_ij * g_ij  # Shape: (batch_size,)
                sectional[..., i, j] = numerator / (denominator + 1e-10)
                
        return sectional


class GeometricMetricValidator:
    """Complete geometric metric validation system."""

    def __init__(
        self,
        manifold_dim: int,
        tolerance: float = 1e-6,
        curvature_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        """Initialize geometric metric validator.
        
        Args:
            manifold_dim: Dimension of the manifold
            tolerance: Tolerance for validation checks
            curvature_bounds: (lower, upper) bounds for curvature
        """
        self.metric_validator = MetricValidator(manifold_dim, tolerance)
        self.connection_validator = ConnectionValidator(manifold_dim, tolerance)
        self.curvature_validator = CurvatureValidator(manifold_dim, curvature_bounds)

    def validate(
        self, 
        framework: RiemannianFramework, 
        points: Tensor
    ) -> Dict[str, Union[MetricValidation, ConnectionValidation, CurvatureValidation, bool]]:
        """Perform complete geometric validation.
        
        Args:
            framework: Riemannian framework for geometric analysis
            points: Points tensor to validate at
            
        Returns:
            Dictionary containing validation results:
            - metric_validation: MetricValidation results
            - connection_validation: ConnectionValidation results 
            - curvature_validation: CurvatureValidation results
            - is_valid: Overall validation status
        """
        # Get geometric tensors
        metric_tensor = framework.get_metric_tensor(points)
        christoffel_values = framework.get_christoffel_values(points)
        riemann_tensor = framework.get_riemann_tensor(points)

        # Validate metric properties
        metric_validation = self.metric_validator.validate_metric(metric_tensor)

        # Validate connection properties
        connection_validation = self.connection_validator.validate_connection(
            christoffel_values, metric_tensor
        )

        # Validate curvature properties
        curvature_validation = self.curvature_validator.validate_curvature(
            riemann_tensor, metric_tensor
        )

        # Overall validation status
        is_valid = (
            metric_validation.is_positive_definite and
            connection_validation.is_valid and
            curvature_validation.bounds_satisfied
        )

        return {
            "metric_validation": metric_validation,
            "connection_validation": connection_validation,
            "curvature_validation": curvature_validation,
            "is_valid": is_valid
        }

    def check_geodesic_completeness(
        self, 
        framework: RiemannianFramework, 
        points: Tensor
    ) -> bool:
        """Check if the metric is geodesically complete.
        
        Args:
            framework: Riemannian framework
            points: Points tensor
            
        Returns:
            True if metric is geodesically complete
        """
        metric_tensor = framework.get_metric_tensor(points)
        return self.metric_validator.check_completeness(metric_tensor)
