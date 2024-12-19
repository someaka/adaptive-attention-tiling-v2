"""Protocol definitions for geometric flows.

This module defines the protocols and data structures used by geometric flow implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, Union

import torch
from torch import Tensor

T = TypeVar('T')  # Generic type for tensor-like objects

@dataclass
class FlowMetrics:
    """Metrics computed during flow evolution."""
    
    flow_magnitude: float
    """Magnitude of the flow vector field."""
    
    metric_determinant: float
    """Determinant of the metric tensor."""
    
    ricci_scalar: float
    """Scalar curvature."""
    
    energy: float
    """Total energy of the system."""
    
    singularity: float
    """Measure of singularity formation."""
    
    normalized_flow: float
    """Normalized flow magnitude."""

@dataclass
class SingularityInfo(Generic[T]):
    """Information about detected singularities."""
    
    index: int
    """Index of the singularity in the batch."""
    
    determinant: float
    """Determinant at singularity."""
    
    condition_number: float
    """Condition number of metric at singularity."""
    
    min_eigenvalue: float
    """Minimum eigenvalue at singularity."""
    
    location: Optional[T]
    """Location of singularity if available."""
    
    curvature: Optional[T]
    """Curvature at singularity if available."""

class GeometricFlowProtocol(Protocol[T]):
    """Protocol defining the interface for geometric flow implementations."""
    
    @abstractmethod
    def compute_metric(
        self,
        points: T,
        connection: Optional[T] = None
    ) -> T:
        """Compute metric tensor at points."""
        pass
    
    @abstractmethod
    def compute_connection(
        self,
        metric: T,
        points: Optional[T] = None
    ) -> T:
        """Compute connection coefficients."""
        pass
    
    @abstractmethod
    def compute_curvature(
        self,
        metric: T,
        connection: Optional[T] = None
    ) -> T:
        """Compute curvature tensor."""
        pass
    
    @abstractmethod
    def compute_ricci_tensor(
        self,
        metric: T,
        points: Optional[T] = None,
        connection: Optional[T] = None
    ) -> T:
        """Compute Ricci tensor."""
        pass
    
    @abstractmethod
    def flow_step(
        self,
        metric: T,
        ricci: Optional[T] = None,
        timestep: float = 0.1
    ) -> Tuple[T, FlowMetrics]:
        """Perform flow step with metrics."""
        pass
    
    @abstractmethod
    def detect_singularities(
        self,
        metric: T,
        points: Optional[T] = None,
        threshold: float = 1e-6
    ) -> List[SingularityInfo[T]]:
        """Detect flow singularities."""
        pass
    
    @abstractmethod
    def normalize_flow(
        self,
        flow: T,
        metric: Optional[T] = None,
        method: str = "ricci"
    ) -> T:
        """Normalize flow vector field."""
        pass
    
    @abstractmethod
    def parallel_transport(
        self,
        vector: T,
        start_point: T,
        end_point: T,
        connection: Optional[T] = None
    ) -> T:
        """Parallel transport vector along geodesic."""
        pass
    
    @abstractmethod
    def compute_geodesic(
        self,
        start_point: T,
        end_point: T,
        num_steps: int = 10
    ) -> T:
        """Compute geodesic between points."""
        pass 