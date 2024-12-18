"""
Shared type definitions for fiber bundles.

This module contains the shared type definitions used across different
fiber bundle implementations to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Tuple, TypeVar, Dict

T = TypeVar("T")
S = TypeVar("S")
StructureGroup = TypeVar("StructureGroup", bound=str)

__all__ = [
    'FiberBundle',
    'LocalChart',
    'FiberChart',
    'T',
    'S',
    'StructureGroup',
]

@dataclass
class LocalChart(Generic[T]):
    """Local coordinate chart on the base manifold.
    
    Attributes:
        coordinates: Local coordinates in the chart
        dimension: Dimension of the chart
        transition_maps: Dictionary of transitions to other charts
    """
    coordinates: T
    dimension: int
    transition_maps: dict


@dataclass
class FiberChart(Generic[T, StructureGroup]):
    """Local trivialization of the fiber.
    
    Attributes:
        fiber_coordinates: Coordinates in the fiber
        structure_group: Group acting on the fiber
        transition_functions: Dictionary of fiber transitions
    """
    fiber_coordinates: T
    structure_group: StructureGroup
    transition_functions: dict


class FiberBundle(Protocol[T]):
    """Protocol defining required operations for fiber bundles.
    
    This protocol specifies the minimal interface that any fiber bundle
    implementation must provide. It captures the essential geometric
    operations needed for pattern analysis.
    
    Type Parameters:
        T: The type of the underlying space (usually torch.Tensor)
    """

    def bundle_projection(self, total_space: T) -> T:
        """Projects from total space to base space.
        
        Args:
            total_space: Point in the total space E
            
        Returns:
            The projection π(p) in the base space M
        """
        ...

    def local_trivialization(self, point: T) -> Tuple[LocalChart[T], FiberChart[T, str]]:
        """Provides local product structure.
        
        Args:
            point: Point p in the total space E
            
        Returns:
            Tuple (φ₁(p), φ₂(p)) giving local coordinates in U×F
        """
        ...

    def transition_functions(self, chart1: T, chart2: T) -> T:
        """Computes transition between charts.
        
        Args:
            chart1: First local chart
            chart2: Second local chart
            
        Returns:
            The transition function g₁₂ between charts
        """
        ...

    def connection_form(self, tangent_vector: T) -> T:
        """Computes the connection form for parallel transport.
        
        Args:
            tangent_vector: Tangent vector X at a point
            
        Returns:
            The connection form ω(X) valued in the Lie algebra
        """
        ...

    def parallel_transport(self, section: T, path: T) -> T:
        """Parallel transports a section along a path.
        
        Args:
            section: Section to transport
            path: Path along which to transport
            
        Returns:
            The parallel transported section
        """
        ... 