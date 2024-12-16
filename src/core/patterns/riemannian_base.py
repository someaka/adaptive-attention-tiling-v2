"""
Core Protocol for Riemannian Geometry.

This module defines the foundational protocol and data structures for Riemannian geometry
implementations. It provides the core interface that all Riemannian manifold implementations
must satisfy, ensuring geometric consistency and mathematical correctness.

Key components:
1. RiemannianStructure Protocol - Core interface for Riemannian geometry
2. Geometric Data Structures - Standard representations for geometric objects
3. Validation Protocols - Interfaces for geometric validation
"""

from dataclasses import dataclass
from typing import (
    Generic, Optional, Protocol, Tuple, TypeVar, Union,
    runtime_checkable, cast, Type, Callable
)
import torch
from torch import Tensor

# Type variables for generic implementations
T = TypeVar('T')  # Usually torch.Tensor
Scalar = Union[float, Tensor]  # For scalar quantities
VectorField = Callable[[T], T]  # Type for vector fields

@dataclass
class MetricTensor(Generic[T]):
    """Metric tensor on a Riemannian manifold.
    
    Attributes:
        values: Components of the metric tensor
        dimension: Dimension of the manifold
        is_compatible: Whether metric satisfies compatibility conditions
    """
    values: T
    dimension: int
    is_compatible: bool = True

@dataclass
class ChristoffelSymbols(Generic[T]):
    """Christoffel symbols of the Levi-Civita connection.
    
    Attributes:
        values: Components of the Christoffel symbols
        metric: Associated metric tensor
        is_symmetric: Whether symbols are symmetric in lower indices
    """
    values: T
    metric: MetricTensor[T]
    is_symmetric: bool = True

@dataclass
class CurvatureTensor(Generic[T]):
    """Riemann curvature tensor components.
    
    Attributes:
        riemann: Full Riemann curvature tensor
        ricci: Ricci curvature tensor
        scalar: Scalar curvature
    """
    riemann: T
    ricci: T
    scalar: Scalar

@runtime_checkable
class RiemannianStructure(Protocol[T]):
    """Protocol defining required operations for Riemannian geometry.
    
    This protocol specifies the minimal interface that any Riemannian manifold
    implementation must provide. It captures the essential geometric operations
    needed for differential geometric computations.
    
    Type Parameters:
        T: The type of the underlying space (usually torch.Tensor)
    """
    
    def compute_metric(self, points: T) -> MetricTensor[T]:
        """Compute the metric tensor at given points.
        
        The metric tensor g_ij must satisfy:
        1. Symmetry: g_ij = g_ji
        2. Positive definiteness: v^i g_ij v^j > 0 for all v ≠ 0
        3. Smoothness: Components are smooth functions
        
        Args:
            points: Points at which to compute metric
            
        Returns:
            Metric tensor at the points
        """
        ...
        
    def compute_christoffel(self, points: T) -> ChristoffelSymbols[T]:
        """Compute Christoffel symbols of the Levi-Civita connection.
        
        The Christoffel symbols Γ^i_jk must satisfy:
        1. Symmetry in lower indices: Γ^i_jk = Γ^i_kj
        2. Metric compatibility: ∇_k g_ij = 0
        3. Torsion-free: Γ^i_jk - Γ^i_kj = 0
        
        Args:
            points: Points at which to compute symbols
            
        Returns:
            Christoffel symbols at the points
        """
        ...
        
    def parallel_transport(
        self, vector: T, path: T, 
        connection: Optional[ChristoffelSymbols[T]] = None
    ) -> T:
        """Parallel transport a vector along a path.
        
        Parallel transport preserves:
        1. Vector norm (metric compatibility)
        2. Angles between vectors
        3. Geodesic properties
        
        Args:
            vector: Vector to transport
            path: Path along which to transport
            connection: Optional pre-computed connection
            
        Returns:
            The parallel transported vector
        """
        ...
        
    def compute_curvature(
        self, points: T,
        christoffel: Optional[ChristoffelSymbols[T]] = None
    ) -> CurvatureTensor[T]:
        """Compute the Riemann curvature tensor.
        
        The curvature tensor R^i_jkl must satisfy:
        1. Antisymmetry: R^i_jkl = -R^i_jlk
        2. First Bianchi identity
        3. Second Bianchi identity
        
        Args:
            points: Points at which to compute curvature
            christoffel: Optional pre-computed Christoffel symbols
            
        Returns:
            Full curvature information
        """
        ...
        
    def geodesic_flow(
        self, initial_point: T, initial_velocity: T,
        steps: int = 100, step_size: float = 0.01
    ) -> Tuple[T, T]:
        """Compute geodesic flow from initial conditions.
        
        Geodesics satisfy:
        1. Local length minimization
        2. Parallel transport of tangent vector
        3. Constant speed parameterization
        
        Args:
            initial_point: Starting point
            initial_velocity: Initial velocity
            steps: Number of integration steps
            step_size: Size of each integration step
            
        Returns:
            Tuple of (points along geodesic, velocities along geodesic)
        """
        ...
        
    def lie_derivative_metric(
        self, point: T, vector_field: VectorField
    ) -> MetricTensor[T]:
        """Compute Lie derivative of metric along vector field.
        
        The Lie derivative measures:
        1. Change of metric under flow
        2. Killing field properties
        3. Isometric deformations
        
        Args:
            point: Point at which to compute derivative
            vector_field: Function computing vector field
            
        Returns:
            Lie derivative of metric tensor
        """
        ...
        
    def sectional_curvature(
        self, point: T, v1: T, v2: T
    ) -> Scalar:
        """Compute sectional curvature in plane spanned by vectors.
        
        Sectional curvature K satisfies:
        1. Symmetry: K(v1,v2) = K(v2,v1)
        2. Scaling invariance
        3. Relates to Gaussian curvature
        
        Args:
            point: Point at which to compute curvature
            v1: First vector spanning plane
            v2: Second vector spanning plane
            
        Returns:
            Sectional curvature value
        """
        ...
        
    def validate_metric_properties(
        self, metric: MetricTensor[T]
    ) -> bool:
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
        ...
        
    def validate_connection_properties(
        self, connection: ChristoffelSymbols[T]
    ) -> bool:
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
        ... 

@runtime_checkable
class RiemannianValidator(Protocol[T]):
    """Protocol for validating Riemannian geometric properties.
    
    This protocol defines the interface for validation operations that ensure
    geometric consistency and mathematical correctness of Riemannian structures.
    
    Type Parameters:
        T: The type of the underlying space (usually torch.Tensor)
    """
    
    def validate_metric_properties(
        self, metric: MetricTensor[T]
    ) -> bool:
        """Validate metric tensor properties."""
        ...
        
    def validate_connection_properties(
        self, connection: ChristoffelSymbols[T]
    ) -> bool:
        """Validate connection properties."""
        ...

class ValidationMixin:
    """Mixin class providing validation functionality.
    
    This mixin provides default implementations of validation methods
    that can be used by Riemannian structure implementations.
    """
    
    def validate_metric_properties(
        self, metric: MetricTensor[Tensor]
    ) -> bool:
        """Default metric validation."""
        return True
        
    def validate_connection_properties(
        self, connection: ChristoffelSymbols[Tensor]
    ) -> bool:
        """Default connection validation."""
        return True
  