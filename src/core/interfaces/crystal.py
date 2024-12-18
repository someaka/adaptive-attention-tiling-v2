"""Crystal Interface Definition.

This module defines the interface for crystal structures that integrates:
1. Basic crystal operations (lattice, symmetry, band structure)
2. Geometric components (Riemann geometry, curvature)
3. Hamiltonian mechanics (phase space, evolution)
4. Scale operations (RG flow, anomalies)
"""

from typing import List, Tuple, Protocol, TypeVar, Optional, Dict, Callable
from typing_extensions import runtime_checkable
from dataclasses import dataclass

import torch

from .quantum import IQuantumState
from ..quantum.crystal import BravaisLattice, BlochFunction
from ..quantum.geometric_flow import GeometricFlowMetrics

T_co = TypeVar('T_co', bound=torch.Tensor, covariant=True)

@dataclass
class ScaleConnection:
    """Connection between different scales in the crystal structure."""
    source_scale: float
    target_scale: float
    connection_map: torch.Tensor  # Linear map between scales
    holonomy: torch.Tensor  # Parallel transport around scale loop
    metric: torch.Tensor  # Metric tensor at each scale
    curvature: torch.Tensor  # Curvature of the scale connection

@dataclass
class CrystalState:
    """State representation in crystal structure."""
    lattice_coords: torch.Tensor  # Crystal lattice coordinates
    momentum: torch.Tensor  # Crystal momentum
    band_index: int  # Band structure index
    phase: float  # Geometric phase
    time: float  # Evolution time

@dataclass 
class CrystalMetrics:
    """Metrics for crystal analysis."""
    geometric: GeometricFlowMetrics  # Geometric flow metrics
    band_energy: torch.Tensor  # Band structure energy
    symmetry_order: int  # Symmetry group order
    scale_anomaly: Optional[torch.Tensor]  # Scale anomaly if present

@dataclass
class RGFlow:
    """Renormalization group flow data."""
    beta_function: Callable[[torch.Tensor], torch.Tensor]
    fixed_points: List[torch.Tensor]
    stability: List[torch.Tensor]  # Stability matrices at fixed points
    flow_lines: List[torch.Tensor]  # Trajectories in coupling space

@dataclass
class AnomalyPolynomial:
    """Anomaly polynomial data."""
    coefficients: torch.Tensor  # Polynomial coefficients
    variables: List[str]  # Variable names
    degree: int  # Polynomial degree
    type: str  # Type of anomaly

@dataclass
class SymmetryOperation:
    """Crystal symmetry operation."""
    matrix: torch.Tensor  # Transformation matrix
    translation: torch.Tensor  # Translation vector
    order: int  # Order of the operation
    type: str  # Type of symmetry

@dataclass
class IBandStructure:
    """Band structure data."""
    energies: torch.Tensor  # Energy eigenvalues
    states: List[BlochFunction]  # Corresponding eigenstates
    k_points: torch.Tensor  # k-points along path
    labels: List[str]  # Labels for high-symmetry points

@runtime_checkable
class ICrystal(Protocol[T_co]):
    """Unified interface for crystal structures."""
    
    def initialize_lattice(self, dim: int, lattice_type: str = "cubic") -> BravaisLattice:
        """Initialize crystal lattice structure."""
        ...
        
    def compute_band_structure(
        self,
        state: CrystalState,
        num_bands: int = 4
    ) -> Tuple[torch.Tensor, List[BlochFunction]]:
        """Compute band structure and Bloch functions."""
        ...
        
    def detect_symmetries(
        self,
        state: CrystalState,
        tolerance: float = 1e-6
    ) -> List[SymmetryOperation]:
        """Detect crystal symmetries."""
        ...
        
    # Geometric Components
    
    def compute_metric(self, state: CrystalState) -> torch.Tensor:
        """Compute metric tensor on crystal manifold."""
        ...
        
    def compute_curvature(
        self,
        state: CrystalState,
        metric: torch.Tensor
    ) -> torch.Tensor:
        """Compute Riemann curvature tensor."""
        ...
        
    def parallel_transport(
        self,
        state: CrystalState,
        vector: torch.Tensor,
        path: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport vector along path."""
        ...
        
    # Hamiltonian Components
    
    def compute_hamiltonian(self, state: CrystalState) -> torch.Tensor:
        """Compute system Hamiltonian."""
        ...
        
    def evolve_state(
        self,
        state: CrystalState,
        time: float,
        dt: float = 0.01
    ) -> CrystalState:
        """Evolve state using Hamiltonian dynamics."""
        ...
        
    def compute_conserved_quantities(
        self,
        state: CrystalState
    ) -> Dict[str, torch.Tensor]:
        """Compute conserved quantities."""
        ...
        
    # Scale Components
    
    def connect_scales(
        self,
        state1: CrystalState,
        state2: CrystalState,
        scale_factor: float
    ) -> ScaleConnection:
        """Connect states at different scales."""
        ...
        
    def compute_rg_flow(
        self,
        state: CrystalState,
        couplings: torch.Tensor
    ) -> RGFlow:
        """Compute renormalization group flow."""
        ...
        
    def detect_anomalies(
        self,
        state: CrystalState
    ) -> List[AnomalyPolynomial]:
        """Detect scale anomalies."""
        ...
        
    def check_scale_invariance(
        self,
        state: CrystalState,
        scale_factor: float
    ) -> bool:
        """Check if state is scale invariant."""
        ...
        
    # Cohomology Components
    
    def compute_cocycle(
        self,
        state1: CrystalState,
        state2: CrystalState,
        scale: float
    ) -> torch.Tensor:
        """Compute cocycle between states."""
        ...
        
    def compute_coboundary(
        self,
        state: CrystalState,
        scale1: float,
        scale2: float
    ) -> torch.Tensor:
        """Compute coboundary of state."""
        ...
        
    def detect_obstructions(
        self,
        states: List[CrystalState],
        scales: List[float]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Detect cohomological obstructions."""
        ...
        
    # Quantum Integration
    
    def to_quantum_state(self, state: CrystalState) -> IQuantumState:
        """Convert to quantum state representation."""
        ...
        
    def from_quantum_state(self, quantum_state: IQuantumState) -> CrystalState:
        """Convert from quantum state representation."""
        ...
        
    # Analysis Methods
    
    def compute_metrics(self, state: CrystalState) -> CrystalMetrics:
        """Compute comprehensive crystal metrics."""
        ...
        
    def check_compatibility(
        self,
        state1: CrystalState,
        state2: CrystalState
    ) -> bool:
        """Check if states are compatible for operations."""
        ...
        
    def validate_state(self, state: CrystalState) -> bool:
        """Validate crystal state."""
        ... 