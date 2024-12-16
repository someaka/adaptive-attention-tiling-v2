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
    scalar: float
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
        self.manifold_dim = manifold_dim
        self.tolerance = tolerance
        self.eigenvalue_threshold = 1e-6
        self.energy_threshold = 1e3
        self.condition_threshold = 1e4

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
            
        # Check symmetry
        if not torch.allclose(metric, metric.transpose(-1, -2), atol=self.tolerance):
            raise ValueError("Non-symmetric metric")
            
        # Check for invalid values
        if torch.any(torch.isnan(metric)) or torch.any(torch.isinf(metric)):
            raise ValueError("Contains NaN or Inf values")
            
        # Check dimensions
        if metric.shape[-1] != self.manifold_dim:
            raise ValueError("Incompatible dimensions")
            
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
        """Compute metric tensor values at points using Fisher-Rao structure.
        
        Args:
            points: Points tensor (batch_size x dim)
            
        Returns:
            Metric tensor values (batch_size x dim x dim)
        """
        batch_size = points.shape[0]
        
        # Compute Fisher-Rao metric components
        # g_ij = E[∂_i log p(x|θ) ∂_j log p(x|θ)]
        score_fn = self._compute_score_function(points)
        metric = torch.einsum('bi,bj->bij', score_fn, score_fn)
        
        # Add regularization for numerical stability
        metric = metric + self.eigenvalue_threshold * torch.eye(self.manifold_dim).expand(batch_size, -1, -1)
        
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

    def _compute_christoffel_symbols(self, points: Tensor) -> Tensor:
        """Compute Christoffel symbols using Levi-Civita connection.
        
        Args:
            points: Points tensor
            
        Returns:
            Christoffel symbols
        """
        batch_size = points.shape[0]
        
        # Get metric and its derivatives
        metric = self.compute_metric_values(points)
        metric_grad = self.compute_metric_gradient(points)
        
        # Compute inverse metric
        metric_inv = torch.linalg.inv(metric)
        
        # Compute Christoffel symbols
        christoffel = torch.einsum(
            'bkl,bijl->bijk',
            metric_inv,
            0.5 * (
                metric_grad 
                + torch.transpose(metric_grad, 2, 3)
                - torch.transpose(metric_grad, 1, 3)
            )
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
        return self._compute_score_function(points)

    def compute_metric_gradient(self, metric: Tensor) -> Tensor:
        """Compute metric gradient tensor.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Metric gradient tensor
        """
        batch_size = metric.shape[0]
        
        # Compute gradient
        grad = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                grad_ij = torch.autograd.grad(
                    metric[:, i, j].sum(),
                    metric,
                    create_graph=True
                )[0]
                grad[:, i, j] = grad_ij
                
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
        """Check A¹-homotopy invariants."""
        # TODO: Implement proper invariant checks
        return True

    def check_geodesic_completeness(self, metric: Optional[Tensor] = None) -> bool:
        """Check if metric is geodesically complete.
        
        Args:
            metric: Optional metric tensor
            
        Returns:
            True if metric is complete
        """
        return self._check_geodesic_completeness(metric)

    def check_northcott_property(self) -> bool:
        """Check Northcott property."""
        # TODO: Implement proper Northcott check
        return True
        
    def validate_height_bounds(self, height: Tensor) -> bool:
        """Validate height function bounds."""
        # TODO: Implement proper bound validation
        return True
        
    def validate_local_bounds(self, heights: Tensor) -> bool:
        """Validate local height bounds."""
        # TODO: Implement proper local bound validation
        return True

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
        """Validate connection compatibility with metric."""
        # Check connection shape
        if connection.shape != (self.manifold_dim, self.manifold_dim, self.manifold_dim):
            return False
            
        # Check symmetry in lower indices
        for i in range(self.manifold_dim):
            if not torch.allclose(
                connection[i,:,:], 
                connection[i,:,:].transpose(0,1), 
                atol=self.tolerance
            ):
                return False
                
        return True

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
        """Validate Fisher-Rao metric properties.
        
        Args:
            metric: Metric tensor
            
        Returns:
            True if metric satisfies Fisher-Rao properties
        """
        # Check positive definiteness
        result = self.validate_metric(metric)
        if not result.positive_definite:
            return False
            
        # Check compatibility with score function
        points = torch.randn(metric.shape[0], self.manifold_dim)
        score = self.compute_score_function(points)
        
        # Compute Fisher-Rao metric
        fisher_metric = torch.einsum('bi,bj->bij', score, score)
        
        # Check if close to input metric
        return bool(torch.allclose(metric, fisher_metric, atol=self.tolerance))

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

    def compute_christoffel_symbols(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel symbols from metric.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Christoffel symbols
        """
        batch_size = metric.shape[0]
        
        # Compute metric inverse
        metric_inv = torch.linalg.inv(metric)
        
        # Compute metric derivatives
        grad = self.compute_metric_gradient(metric)
        
        # Compute Christoffel symbols
        christoffel = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    christoffel[:,i,j,k] = 0.5 * torch.sum(
                        metric_inv[:,i,:] * (
                            grad[:,j,k,:] + grad[:,k,j,:] - grad[:,:,j,k]
                        ),
                        dim=1
                    )
                    
        return christoffel

    def compute_sectional_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute sectional curvature.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Sectional curvature tensor
        """
        # Get Christoffel symbols
        christoffel = self.compute_christoffel_symbols(metric)
        
        # Compute Riemann curvature tensor
        riemann = torch.zeros_like(christoffel)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} + Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}
                        riemann[:,i,j,k,l] = (
                            torch.autograd.grad(christoffel[:,i,j,l].sum(), metric, create_graph=True)[0][:,k] -
                            torch.autograd.grad(christoffel[:,i,j,k].sum(), metric, create_graph=True)[0][:,l]
                        )
                        for m in range(self.manifold_dim):
                            riemann[:,i,j,k,l] += (
                                christoffel[:,i,m,k] * christoffel[:,m,j,l] -
                                christoffel[:,i,m,l] * christoffel[:,m,j,k]
                            )
                            
        # Compute sectional curvature
        sectional = torch.zeros(metric.shape[0], self.manifold_dim, self.manifold_dim)
        for i in range(self.manifold_dim):
            for j in range(i+1, self.manifold_dim):
                sectional[:,i,j] = riemann[:,i,j,i,j] / (
                    metric[:,i,i] * metric[:,j,j] - metric[:,i,j]**2
                )
                sectional[:,j,i] = sectional[:,i,j]
                
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
        
        # Compute Ricci curvature by tracing
        ricci = torch.zeros_like(metric)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                ricci[:,i,j] = torch.sum(sectional[:,i,:] * metric[:,j,:], dim=1)
                
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
        """Validate metric properties."""
        # Compute basic properties
        determinant = torch.linalg.det(metric)
        trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)
        eigenvalues = torch.linalg.eigvalsh(metric)
        condition_number = eigenvalues.max() / eigenvalues.min()
        
        # Compute derived properties
        volume_form = self.compute_volume_form(metric)
        christoffel = self._compute_christoffel_symbols(metric)
        
        # Compute curvature properties
        sectional = self.compute_sectional_curvature(metric)
        ricci = self.compute_ricci_curvature(metric)
        scalar = self.compute_scalar_curvature(metric)
        
        return MetricProperties(
            is_positive_definite=bool(torch.all(eigenvalues > self.eigenvalue_threshold)),
            is_compatible=self.validate_connection_compatibility(self.get_test_connection()),
            is_complete=self._check_completeness(metric),
            has_bounded_curvature=self.check_metric_bounds(),
            determinant=determinant,
            trace=trace,
            eigenvalues=eigenvalues,
            condition_number=condition_number,
            volume_form=volume_form,
            christoffel_symbols=christoffel,
            sectional_curvature=sectional,
            ricci_curvature=ricci,
            scalar_curvature=scalar
        )

    def validate_curvature_bounds(self, metric: torch.Tensor) -> CurvatureBounds:
        """Validate curvature bounds.
        
        Args:
            metric: Metric tensor
            
        Returns:
            CurvatureBounds object containing validation results
        """
        # Compute curvature tensors
        sectional = self.compute_sectional_curvature(metric)
        ricci = self.compute_ricci_curvature(metric)
        scalar = self.compute_scalar_curvature(metric)
        
        # Compute bounds
        sectional_bounds = (sectional.min().item(), sectional.max().item())
        ricci_bounds = (ricci.min().item(), ricci.max().item())
        scalar_bounds = (scalar.min().item(), scalar.max().item())
        
        return CurvatureBounds(
            ricci_lower=ricci_bounds[0],
            ricci_upper=ricci_bounds[1],
            sectional_lower=sectional_bounds[0],
            sectional_upper=sectional_bounds[1],
            sectional_bounds=sectional_bounds,
            ricci_bounds=ricci_bounds,
            scalar_bounds=scalar_bounds
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
                        bianchi = (
                            curvature[:,i,j,k,l] +
                            curvature[:,i,k,l,j] +
                            curvature[:,i,l,j,k]
                        )
                        if not torch.allclose(bianchi, torch.zeros_like(bianchi), atol=self.tolerance):
                            return False
                            
        # Check symmetries
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R_ijkl = -R_ijlk
                        if not torch.allclose(
                            curvature[:,i,j,k,l],
                            -curvature[:,i,j,l,k],
                            atol=self.tolerance
                        ):
                            return False
                            
                        # R_ijkl = -R_jikl
                        if not torch.allclose(
                            curvature[:,i,j,k,l],
                            -curvature[:,j,i,k,l],
                            atol=self.tolerance
                        ):
                            return False
                            
                        # R_ijkl = R_klij
                        if not torch.allclose(
                            curvature[:,i,j,k,l],
                            curvature[:,k,l,i,j],
                            atol=self.tolerance
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
        scalar = float(torch.einsum("ij,ij->", ricci, torch.inverse(metric)).item())

        # Check bounds
        bounds_satisfied = bool((sectional >= self.lower_bound).all()) and bool((sectional <= self.upper_bound).all())

        return CurvatureValidation(
            bounds_satisfied=bounds_satisfied,
            sectional=sectional,
            scalar=scalar,
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
        # Compute geometric quantities
        metric = framework.compute_metric(points)
        connection = framework.compute_christoffel(points)
        riemann = framework.compute_riemann(points)

        # Validate metric properties
        metric_validation = self.metric_validator.validate_metric(metric)

        # Validate connection properties
        connection_validation = self.connection_validator.validate_connection(
            connection, metric
        )

        # Validate curvature properties
        curvature_validation = self.curvature_validator.validate_curvature(
            riemann, metric
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
        metric = framework.compute_metric(points)
        return self.metric_validator.check_completeness(metric)
