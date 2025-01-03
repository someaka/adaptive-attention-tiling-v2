"""Protocol definitions for geometric flows.

This module defines the protocols and data structures used by geometric flow implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn

T = TypeVar('T')  # Generic type for tensor-like objects

@dataclass
class FlowMetrics:
    """Metrics computed during flow evolution."""
    
    flow_magnitude: Union[float, Tensor]
    """Magnitude of the flow vector field."""
    
    metric_determinant: Union[float, Tensor]
    """Determinant of the metric tensor."""
    
    ricci_scalar: Union[float, Tensor]
    """Scalar curvature."""
    
    energy: Union[float, Tensor]
    """Total energy of the system."""
    
    singularity: Union[float, Tensor]
    """Measure of singularity formation."""
    
    normalized_flow: Union[float, Tensor]
    """Normalized flow magnitude."""

@dataclass
class QuantumFlowMetrics(FlowMetrics):
    """Extended metrics for quantum geometric flow."""
    
    quantum_entropy: Tensor
    """Von Neumann entropy of the quantum state."""
    
    berry_phase: Optional[Tensor] = None
    """Berry phase along the flow path."""
    
    mean_curvature: Optional[Tensor] = None
    """Mean curvature of the quantum manifold."""
    
    quantum_corrections: Optional[Tensor] = None
    """Quantum corrections to the classical flow."""
    
    def __init__(
        self,
        flow_magnitude: Union[float, Tensor],
        metric_determinant: Union[float, Tensor],
        ricci_scalar: Union[float, Tensor],
        energy: Union[float, Tensor],
        singularity: Union[float, Tensor],
        normalized_flow: Union[float, Tensor],
        quantum_entropy: Union[float, Tensor, None],
        berry_phase: Optional[Union[float, Tensor]] = None,
        mean_curvature: Optional[Union[float, Tensor]] = None,
        quantum_corrections: Optional[Union[float, Tensor]] = None,
        device: Optional[torch.device] = None
    ):
        # Get device from input tensors or default to CPU
        device = device or (
            flow_magnitude.device if isinstance(flow_magnitude, Tensor)
            else metric_determinant.device if isinstance(metric_determinant, Tensor)
            else ricci_scalar.device if isinstance(ricci_scalar, Tensor)
            else energy.device if isinstance(energy, Tensor)
            else singularity.device if isinstance(singularity, Tensor)
            else normalized_flow.device if isinstance(normalized_flow, Tensor)
            else quantum_entropy.device if isinstance(quantum_entropy, Tensor)
            else berry_phase.device if isinstance(berry_phase, Tensor)
            else mean_curvature.device if isinstance(mean_curvature, Tensor)
            else quantum_corrections.device if isinstance(quantum_corrections, Tensor)
            else torch.device('cpu')
        )
        
        # Convert base metrics to tensors
        self.flow_magnitude = (
            flow_magnitude if isinstance(flow_magnitude, Tensor)
            else torch.tensor(float(flow_magnitude), device=device)
        )
        
        self.metric_determinant = (
            metric_determinant if isinstance(metric_determinant, Tensor)
            else torch.tensor(float(metric_determinant), device=device)
        )
        
        self.ricci_scalar = (
            ricci_scalar if isinstance(ricci_scalar, Tensor)
            else torch.tensor(float(ricci_scalar), device=device)
        )
        
        self.energy = (
            energy if isinstance(energy, Tensor)
            else torch.tensor(float(energy), device=device)
        )
        
        self.singularity = (
            singularity if isinstance(singularity, Tensor)
            else torch.tensor(float(singularity), device=device)
        )
        
        self.normalized_flow = (
            normalized_flow if isinstance(normalized_flow, Tensor)
            else torch.tensor(float(normalized_flow), device=device)
        )
        
        # Convert quantum metrics to tensors
        self.quantum_entropy = (
            quantum_entropy if isinstance(quantum_entropy, Tensor)
            else torch.tensor(float(quantum_entropy), device=device) if quantum_entropy is not None
            else torch.tensor(0.0, device=device)
        )
        
        # Convert optional fields to tensors if they're not None
        self.berry_phase = (
            berry_phase if isinstance(berry_phase, Tensor)
            else torch.tensor(float(berry_phase), device=device) if berry_phase is not None
            else None
        )
        
        self.mean_curvature = (
            mean_curvature if isinstance(mean_curvature, Tensor)
            else torch.tensor(float(mean_curvature), device=device) if mean_curvature is not None
            else None
        )
        
        self.quantum_corrections = (
            quantum_corrections if isinstance(quantum_corrections, Tensor)
            else torch.tensor(float(quantum_corrections), device=device) if quantum_corrections is not None
            else None
        )
    
    def to_device(self, device: torch.device) -> 'QuantumFlowMetrics':
        """Move all tensor fields to the specified device."""
        return QuantumFlowMetrics(
            flow_magnitude=self.flow_magnitude,
            metric_determinant=self.metric_determinant,
            ricci_scalar=self.ricci_scalar,
            energy=self.energy,
            singularity=self.singularity,
            normalized_flow=self.normalized_flow,
            quantum_entropy=self.quantum_entropy.to(device),
            berry_phase=self.berry_phase.to(device) if self.berry_phase is not None else None,
            mean_curvature=self.mean_curvature.to(device) if self.mean_curvature is not None else None,
            quantum_corrections=self.quantum_corrections.to(device) if self.quantum_corrections is not None else None,
            device=device
        )

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

class RicciTensorNetwork(nn.Module):
    """Neural network for computing Ricci tensor components."""
    
    def __init__(self, manifold_dim: int, hidden_dim: Optional[int] = None):
        """Initialize Ricci tensor network.
        
        Args:
            manifold_dim: Dimension of the manifold
            hidden_dim: Hidden dimension for computations (defaults to 2*manifold_dim)
        """
        super().__init__()
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim or (2 * manifold_dim)
        
        # Network layers
        self.projection = nn.Linear(manifold_dim, self.hidden_dim)
        self.hidden = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.output = nn.Linear(self.hidden_dim, manifold_dim * manifold_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """Compute Ricci tensor components.
        
        Args:
            x: Input tensor of shape (..., manifold_dim)
            
        Returns:
            Ricci tensor components of shape (..., manifold_dim, manifold_dim)
        """
        # Project to hidden space
        h = self.projection(x)
        
        # Apply hidden layers
        h = self.hidden(h)
        
        # Project to Ricci tensor space
        ricci = self.output(h)
        
        # Reshape to matrix form
        return ricci.view(*ricci.shape[:-1], self.manifold_dim, self.manifold_dim)
  