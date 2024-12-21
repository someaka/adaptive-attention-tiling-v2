"""
Shared type definitions for fiber bundles.

This module contains the shared type definitions used across different
fiber bundle implementations to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Tuple, TypeVar, Dict, Optional, List, Union, Callable, Any
import torch
from torch import Tensor

T = TypeVar("T")
S = TypeVar("S")
StructureGroup = TypeVar("StructureGroup", bound=str)

__all__ = [
    'FiberBundle',
    'LocalChart',
    'FiberChart',
    'FiberTypeManager',
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


@dataclass
class FiberType:
    """Represents a fiber type with its properties.
    
    Attributes:
        name: Name of the fiber type
        dimension: Dimension of the fiber
        structure_group: Associated structure group
        is_complex: Whether the fiber is complex
        is_oriented: Whether the fiber is oriented
        metric_type: Type of metric on the fiber
    """
    name: str
    dimension: int
    structure_group: str
    is_complex: bool = False
    is_oriented: bool = True
    metric_type: str = "Euclidean"


class FiberTypeManager:
    """Manages fiber types and their transformations.
    
    This class provides centralized management of fiber types, including:
    1. Type registration and validation
    2. Type conversion between different fiber types
    3. Structure group compatibility checking
    4. Metric preservation validation
    
    Attributes:
        _registered_types: Dictionary of registered fiber types
        _conversion_map: Dictionary mapping valid type conversions
        _structure_groups: Dictionary of structure group properties
    """
    
    def __init__(self):
        """Initialize fiber type manager."""
        self._registered_types: Dict[str, FiberType] = {}
        self._conversion_map: Dict[Tuple[str, str], Callable[[Tensor], Tensor]] = {}
        self._structure_groups: Dict[str, Dict[str, Any]] = {
            'SO3': {
                'dimension': 3,
                'is_compact': True,
                'is_connected': True,
                'compatible_types': ['Vector', 'Principal']
            },
            'U1': {
                'dimension': 1,
                'is_compact': True,
                'is_connected': True,
                'compatible_types': ['Complex', 'Principal']
            }
        }
        
        # Register standard fiber types
        self.register_fiber_type(
            FiberType("Vector", 3, "SO3", is_complex=False)
        )
        self.register_fiber_type(
            FiberType("Principal", 3, "SO3", is_complex=False)
        )
        self.register_fiber_type(
            FiberType("Complex", 1, "U1", is_complex=True)
        )

    def register_fiber_type(self, fiber_type: FiberType) -> None:
        """Register a new fiber type.
        
        Args:
            fiber_type: FiberType instance to register
            
        Raises:
            ValueError: If type is already registered or incompatible
        """
        if fiber_type.name in self._registered_types:
            raise ValueError(f"Fiber type {fiber_type.name} already registered")
            
        if fiber_type.structure_group not in self._structure_groups:
            raise ValueError(f"Unknown structure group {fiber_type.structure_group}")
            
        group_info = self._structure_groups[fiber_type.structure_group]
        if 'compatible_types' not in group_info:
            group_info['compatible_types'] = []
        if fiber_type.name not in group_info['compatible_types']:
            group_info['compatible_types'].append(fiber_type.name)
            
        self._registered_types[fiber_type.name] = fiber_type

    def register_conversion(
        self,
        source_type: str,
        target_type: str,
        conversion_fn: Callable[[Tensor], Tensor]
    ) -> None:
        """Register a conversion function between fiber types.
        
        Args:
            source_type: Source fiber type name
            target_type: Target fiber type name
            conversion_fn: Function implementing the conversion
            
        Raises:
            ValueError: If types not registered or conversion exists
        """
        if source_type not in self._registered_types:
            raise ValueError(f"Source type {source_type} not registered")
        if target_type not in self._registered_types:
            raise ValueError(f"Target type {target_type} not registered")
            
        key = (source_type, target_type)
        if key in self._conversion_map:
            raise ValueError(f"Conversion {source_type} -> {target_type} already exists")
            
        self._conversion_map[key] = conversion_fn

    def validate_fiber_type(
        self,
        section: Tensor,
        fiber_type: str,
        fiber_dim: int
    ) -> bool:
        """Validate that section has correct fiber type.
        
        Args:
            section: Section to validate
            fiber_type: Name of fiber type
            fiber_dim: Dimension of fiber
            
        Returns:
            bool: Whether section has valid type
            
        Raises:
            ValueError: If fiber type not registered
        """
        if fiber_type not in self._registered_types:
            raise ValueError(f"Unknown fiber type {fiber_type}")
            
        type_info = self._registered_types[fiber_type]
        
        if fiber_type == 'Vector':
            return section.shape[-1] == fiber_dim
        elif fiber_type == 'Principal':
            return (
                section.shape[-2:] == (fiber_dim, fiber_dim) and
                self._validate_group_element(section, type_info.structure_group)
            )
        elif fiber_type == 'Complex':
            return (
                section.shape[-1] == 1 and
                torch.is_complex(section)
            )
        return False

    def convert_fiber_type(
        self,
        section: Tensor,
        source_type: str,
        target_type: str,
        fiber_dim: int
    ) -> Tensor:
        """Convert section between fiber types.
        
        Args:
            section: Section to convert
            source_type: Source fiber type name
            target_type: Target fiber type name
            fiber_dim: Dimension of fiber
            
        Returns:
            Converted section
            
        Raises:
            ValueError: If conversion not possible
        """
        if source_type == target_type:
            return section
            
        key = (source_type, target_type)
        if key not in self._conversion_map:
            # Try standard conversions
            if target_type == 'Vector' and source_type == 'Principal':
                return self._principal_to_vector(section, fiber_dim)
            elif target_type == 'Principal' and source_type == 'Vector':
                return self._vector_to_principal(section, fiber_dim)
            else:
                raise ValueError(f"No conversion {source_type} -> {target_type}")
                
        return self._conversion_map[key](section)

    def _validate_group_element(self, element: Tensor, group: str) -> bool:
        """Validate that tensor represents valid group element."""
        if group == 'SO3':
            # Check orthogonality and determinant
            is_orthogonal = torch.allclose(
                torch.matmul(element, element.transpose(-2, -1)),
                torch.eye(element.shape[-1], device=element.device),
                rtol=1e-5
            )
            has_unit_det = torch.allclose(
                torch.det(element),
                torch.ones(1, device=element.device),
                rtol=1e-5
            )
            return is_orthogonal and has_unit_det
        elif group == 'U1':
            # Check unit modulus
            return torch.allclose(
                torch.abs(element),
                torch.ones_like(element),
                rtol=1e-5
            )
        return False

    def _principal_to_vector(self, section: Tensor, fiber_dim: int) -> Tensor:
        """Convert from principal to vector bundle section."""
        return torch.matmul(
            section,
            torch.ones(
                *section.shape[:-2], fiber_dim, 1,
                device=section.device
            )
        ).squeeze(-1)

    def _vector_to_principal(self, section: Tensor, fiber_dim: int) -> Tensor:
        """Convert from vector to principal bundle section."""
        return torch.eye(
            fiber_dim,
            device=section.device
        ).expand(*section.shape[:-1], -1, -1)

    def get_fiber_type(self, name: str) -> Optional[FiberType]:
        """Get fiber type information by name."""
        return self._registered_types.get(name)

    def get_structure_group(self, name: str) -> Optional[Dict]:
        """Get structure group information by name."""
        return self._structure_groups.get(name)

    def list_fiber_types(self) -> List[str]:
        """Get list of registered fiber types."""
        return list(self._registered_types.keys())

    def list_structure_groups(self) -> List[str]:
        """Get list of supported structure groups."""
        return list(self._structure_groups.keys())

    def check_compatibility(
        self,
        fiber_type: str,
        structure_group: str
    ) -> bool:
        """Check if fiber type is compatible with structure group."""
        if structure_group not in self._structure_groups:
            return False
        return fiber_type in self._structure_groups[structure_group]['compatible_types']