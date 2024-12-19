"""Shared type definitions and protocols.

This module contains shared type definitions and protocols used across
different components of the system to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, Dict, Any, Optional, List, Tuple
import torch
from torch import nn

# Type variables
T = TypeVar('T')
StructureGroup = TypeVar('StructureGroup', bound=str)

@dataclass
class RiemannianMetric:
    """Riemannian metric tensor and associated structures."""
    
    tensor: torch.Tensor  # Shape: (..., dim, dim)
    christoffel_symbols: Optional[torch.Tensor] = None  # Shape: (..., dim, dim, dim)
    curvature: Optional[torch.Tensor] = None  # Shape: (..., dim, dim, dim, dim)
    
    def __post_init__(self):
        """Validate metric tensor properties."""
        if not torch.allclose(self.tensor, self.tensor.transpose(-1, -2)):
            raise ValueError("Metric tensor must be symmetric")
        
        # Check positive definiteness (optional, can be expensive)
        if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigvalsh'):
            eigenvals = torch.linalg.eigvalsh(self.tensor)
            if not (eigenvals > 0).all():
                raise ValueError("Metric tensor must be positive definite")

class FiberBundleProtocol(Protocol):
    """Protocol defining the interface for fiber bundles."""
    
    base_dim: int
    fiber_dim: int
    total_dim: int
    
    def bundle_projection(self, total_space: torch.Tensor) -> torch.Tensor:
        """Project from total space to base space."""
        ...
        
    def get_fiber(self, point: torch.Tensor) -> torch.Tensor:
        """Get fiber at a point."""
        ...
        
    def get_connection(self, point: torch.Tensor) -> torch.Tensor:
        """Get connection at a point."""
        ...

class GeometricFlowProtocol(Protocol):
    """Protocol defining the interface for geometric flows.
    
    This protocol establishes the core interface that all geometric flow
    implementations must satisfy, including Ricci flow and pattern-specific flows.
    """
    
    def compute_ricci_tensor(
        self,
        metric: torch.Tensor,
        connection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the Ricci curvature tensor.
        
        Args:
            metric: The metric tensor at current point
            connection: Optional connection form
            
        Returns:
            The Ricci curvature tensor
        """
        ...
        
    def flow_step(
        self,
        metric: torch.Tensor,
        ricci: torch.Tensor,
        timestep: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform one step of geometric flow.
        
        Args:
            metric: Current metric tensor
            ricci: Ricci curvature tensor
            timestep: Integration time step
            
        Returns:
            Tuple of (new_metric, flow_metrics)
        """
        ...
        
    def detect_singularities(
        self,
        flow: torch.Tensor,
        threshold: float = 1e-6
    ) -> List[Dict[str, Any]]:
        """Detect singularities in the flow.
        
        Args:
            flow: The geometric flow tensor
            threshold: Detection threshold
            
        Returns:
            List of detected singularities with metadata
        """
        ...
        
    def normalize_flow(
        self,
        flow: torch.Tensor,
        normalization: str = "ricci"
    ) -> torch.Tensor:
        """Normalize the geometric flow.
        
        Args:
            flow: Flow tensor to normalize
            normalization: Type of normalization
            
        Returns:
            Normalized flow tensor
        """
        ...
        
    def forward(
        self,
        x: torch.Tensor,
        return_path: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply geometric flow."""
        ...
        
    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor."""
        ...

@dataclass
class PatternState:
    """State of a pattern in the fiber bundle."""
    
    coordinates: torch.Tensor
    metric: torch.Tensor
    connection: Optional[torch.Tensor] = None
    curvature: Optional[torch.Tensor] = None