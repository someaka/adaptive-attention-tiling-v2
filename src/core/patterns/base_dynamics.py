"""Base pattern dynamics implementation.

This module provides the base implementation of pattern dynamics that:
1. Implements the core IPatternDynamics interface
2. Provides extension points for specializations
3. Handles common infrastructure
4. Supports pluggable subsystems
5. Manages state lifecycle
6. Crystal structure integration
"""

from typing import Dict, List, Optional, Tuple, Any, Protocol, TypeVar, Sequence, Callable, cast, Union, Generic
from typing_extensions import Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import numpy as np

from .dynamics import PatternDynamics as IPatternDynamics
from ..interfaces.quantum import (
    IQuantumState,
    PatternState,
    GeometricFlow,
    EntanglementMetrics,
    EvolutionType
)
from ..interfaces.crystal import (
    ICrystal,
    RGFlow,
    ScaleConnection,
    AnomalyPolynomial,
    IBandStructure
)
from ..crystal.scale import (
    ScaleSystem,
    ScaleCohomology
)
from ..crystal.refraction import (
    RefractionSystem,
    SymmetryDetector,
    BandStructureAnalyzer
)
from ...neural.attention.pattern.stability import StabilityAnalyzer
from ...neural.attention.pattern.models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationPoint,
    BifurcationDiagram
)
from ...neural.attention.pattern.diffusion import DiffusionSystem
from ...neural.attention.pattern.reaction import ReactionSystem
from ..quantum.crystal import BravaisLattice, BrillouinZone

# Type for the protocol - bound to torch.Tensor
T = TypeVar('T', bound=torch.Tensor)

@runtime_checkable
@dataclass
class GeometricFlowImpl(GeometricFlow[torch.Tensor]):
    """Implementation of GeometricFlow protocol."""
    vector_field: torch.Tensor
    metric: torch.Tensor
    
    def evolve_state(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve state using flow field."""
        return state + dt * self.vector_field
        
    def metric_tensor(self) -> torch.Tensor:
        """Get metric tensor."""
        return self.metric
        
    def compute_flow(self, state: torch.Tensor) -> torch.Tensor:
        """Compute flow field at state."""
        return self.vector_field
        
    def compute_stability(self, state: torch.Tensor) -> torch.Tensor:
        """Compute stability at state."""
        if not state.requires_grad:
            state = state.detach().requires_grad_(True)
            
        div = torch.zeros_like(state)
        for i in range(state.shape[1]):
            div[:,i] = torch.autograd.grad(
                self.vector_field[:,i].sum(),
                state,
                create_graph=True
            )[0][:,i]
        
        return -torch.mean(div, dim=1)

class BasePatternDynamics(IPatternDynamics[T], ABC, Generic[T]):
    """Base implementation of pattern dynamics."""
    
    def __init__(
        self,
        dt: float = 0.01,
        space_dim: int = 2,
        device: Optional[torch.device] = None,
        manifold_type: Literal["hyperbolic", "euclidean"] = "euclidean",
        max_concentration: float = 1.0,
        min_concentration: float = 0.0,
        grid_size: int = 32,
        enable_crystal: bool = True
    ):
        """Initialize base pattern dynamics."""
        self.dt = dt
        self.space_dim = space_dim
        self.device = device or torch.device('cpu')
        self.manifold_type = manifold_type
        self.max_concentration = max_concentration
        self.min_concentration = min_concentration
        self.grid_size = grid_size
        self.enable_crystal = enable_crystal
        
        # Initialize systems
        self.stability_analyzer = StabilityAnalyzer(self)
        self.diffusion_system = DiffusionSystem(grid_size=grid_size)
        self.reaction_system = ReactionSystem(grid_size=grid_size)
        
        # Initialize crystal systems if enabled
        if enable_crystal:
            self.scale_system = ScaleSystem(dim=space_dim)
            self.refraction_system = RefractionSystem(dim=space_dim)
            lattice = BravaisLattice(dim=space_dim, lattice_type="cubic")
            self.symmetry_detector = SymmetryDetector(lattice=lattice)
            
        # Initialize state validation
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.space_dim < 1:
            raise ValueError("Space dimension must be at least 1")
        if self.max_concentration <= self.min_concentration:
            raise ValueError("Max concentration must be greater than min concentration")
            
    def _validate_state(
        self,
        state: Union[T, ReactionDiffusionState]
    ) -> None:
        """Validate state tensor shape and values."""
        if isinstance(state, ReactionDiffusionState):
            if state.activator.dim() != 4 or state.inhibitor.dim() != 4:
                raise ValueError(f"Expected 4D tensors, got shapes {state.activator.shape}, {state.inhibitor.shape}")
            if state.activator.size(1) != 1 or state.inhibitor.size(1) != 1:
                raise ValueError("Channel dimension must be 1")
        elif isinstance(state, torch.Tensor):
            if state.dim() != 4:
                raise ValueError(f"Expected 4D tensor, got shape {state.shape}")
            if state.size(1) != 2:
                raise ValueError(f"Expected 2 channels, got {state.size(1)}")
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def _to_reaction_diffusion_state(
        self,
        state: Union[T, ReactionDiffusionState]
    ) -> ReactionDiffusionState:
        """Convert state to ReactionDiffusionState if needed."""
        if isinstance(state, ReactionDiffusionState):
            return state
        elif isinstance(state, torch.Tensor):
            if state.shape[1] != 2:
                raise ValueError("Tensor state must have shape [batch, 2, height, width]")
            return ReactionDiffusionState(
                activator=state[:,0:1],
                inhibitor=state[:,1:2],
                time=0.0
            )
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def _from_reaction_diffusion_state(
        self,
        state: ReactionDiffusionState
    ) -> T:
        """Convert ReactionDiffusionState back to T."""
        combined = torch.cat([state.activator, state.inhibitor], dim=1)
        return cast(T, combined)

    def evolve(
        self,
        state: T,
        time: float,
        evolution_type: EvolutionType = EvolutionType.GEOMETRIC,
        control: Optional[ControlSignal] = None
    ) -> T:
        """Evolve pattern state forward in time."""
        if time < 0:
            raise ValueError("Evolution time must be non-negative")
        self._validate_state(state)
        
        # Convert to ReactionDiffusionState for processing
        rd_state = self._to_reaction_diffusion_state(state)
        
        try:
            steps = int(time / self.dt)
            if steps == 0:
                return state
                
            current = rd_state
            current_time = 0.0
            
            for _ in range(steps):
                # Apply control if provided
                if control is not None:
                    current = ReactionDiffusionState(
                        activator=control.apply(current.activator),
                        inhibitor=control.apply(current.inhibitor),
                        time=current_time
                    )
                
                # Compute flow
                flow = self.compute_flow(current)
                
                # Update state based on evolution type
                if evolution_type == EvolutionType.GEOMETRIC:
                    current = self._geometric_evolution_step(current, flow)
                else:
                    current = self._quantum_evolution_step(current, flow)
                    
                # Update time
                current_time += self.dt
                current.time = current_time
                
                # Enforce bounds
                current = self._enforce_bounds(current)
            
            # Convert back to original type
            return self._from_reaction_diffusion_state(current)
            
        except Exception as e:
            raise RuntimeError(f"Evolution failed: {str(e)}") from e
            
    def _geometric_evolution_step(
        self,
        state: ReactionDiffusionState,
        flow: GeometricFlow[torch.Tensor]
    ) -> ReactionDiffusionState:
        """Perform one step of geometric evolution."""
        # Combine state for evolution
        combined = torch.cat([state.activator, state.inhibitor], dim=1)
        
        # Evolve combined state
        evolved = flow.evolve_state(combined, self.dt)
        
        # Split back into components
        return ReactionDiffusionState(
            activator=evolved[:,0:1],
            inhibitor=evolved[:,1:2],
            time=state.time + self.dt
        )
    
    @abstractmethod
    def _quantum_evolution_step(
        self,
        state: ReactionDiffusionState,
        flow: GeometricFlow[torch.Tensor]
    ) -> ReactionDiffusionState:
        """Perform one step of quantum evolution."""
        pass
        
    def _enforce_bounds(
        self,
        state: ReactionDiffusionState
    ) -> ReactionDiffusionState:
        """Enforce concentration bounds on state."""
        state.activator.clamp_(self.min_concentration, self.max_concentration)
        state.inhibitor.clamp_(self.min_concentration, self.max_concentration)
        return state
    
    def compute_flow(
        self,
        state: Union[T, ReactionDiffusionState]
    ) -> GeometricFlow[torch.Tensor]:
        """Compute geometric flow field."""
        # Convert to ReactionDiffusionState if needed
        rd_state = self._to_reaction_diffusion_state(state)
        
        # Compute flow components
        diffusion = self._compute_diffusion(rd_state)
        reaction = self._compute_reaction(rd_state)
        
        # Combine components
        flow_field = diffusion + reaction
        
        # Get metric
        metric = self._compute_metric_tensor(rd_state)
        
        # Create proper GeometricFlow implementation
        return GeometricFlowImpl(
            vector_field=flow_field,
            metric=metric
        )
        
    def _compute_diffusion(
        self,
        state: ReactionDiffusionState
    ) -> torch.Tensor:
        """Compute diffusion term using DiffusionSystem."""
        activator_diff = self.diffusion_system(
            state.activator,
            diffusion_coefficient=0.1,
            dt=self.dt
        )
        inhibitor_diff = self.diffusion_system(
            state.inhibitor,
            diffusion_coefficient=0.1,
            dt=self.dt
        )
        
        return torch.cat([activator_diff, inhibitor_diff], dim=1)
    
    def _compute_reaction(
        self,
        state: ReactionDiffusionState
    ) -> torch.Tensor:
        """Compute reaction term using ReactionSystem."""
        combined = torch.cat([state.activator, state.inhibitor], dim=1)
        return self.reaction_system.reaction_term(combined)
    
    def _compute_metric_tensor(
        self,
        state: ReactionDiffusionState
    ) -> torch.Tensor:
        """Compute metric tensor based on manifold type."""
        combined = torch.cat([state.activator, state.inhibitor], dim=1)
        
        if self.manifold_type == "hyperbolic":
            return torch.eye(
                self.space_dim,
                device=self.device
            ) * (1.0 / (1.0 - torch.sum(combined**2, dim=-1, keepdim=True)).clamp(min=1e-6))
        else:
            return torch.eye(self.space_dim, device=self.device)
    
    def compute_energy(
        self,
        state: T
    ) -> Dict[str, torch.Tensor]:
        """Compute energy components."""
        rd_state = self._to_reaction_diffusion_state(state)
        
        # Compute gradients for kinetic energy
        activator_grad = torch.gradient(rd_state.activator)
        inhibitor_grad = torch.gradient(rd_state.inhibitor)
        kinetic = torch.tensor(0.5 * (
            sum(torch.sum(g**2) for g in activator_grad) +
            sum(torch.sum(g**2) for g in inhibitor_grad)
        ), device=self.device)
        
        # Get reaction term for potential
        reaction = self._compute_reaction(rd_state)
        potential = torch.tensor(0.5 * torch.sum(reaction**2).item(), device=self.device)
        
        # Compute total energy
        total = kinetic + potential
        
        return {
            "kinetic": kinetic,
            "potential": potential,
            "total": total
        }
        
    def compute_stability(
        self,
        state: T,
        perturbation: Optional[torch.Tensor] = None
    ) -> StabilityMetrics:
        """Compute stability metrics for the pattern state."""
        rd_state = self._to_reaction_diffusion_state(state)
        
        # Use default perturbation if none provided
        if perturbation is None:
            perturbation = torch.randn_like(
                torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
            ) * 0.01
            
        # Get stability metrics
        metrics = self.stability_analyzer.analyze_stability(
            torch.cat([rd_state.activator, rd_state.inhibitor], dim=1),
            perturbation
        )
        
        return metrics

    def find_bifurcation(
        self,
        state: T,
        parameter_range: torch.Tensor,
        parameter_name: str = 'dt'
    ) -> Optional[BifurcationPoint]:
        """Find bifurcation point."""
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        # Find bifurcation
        param_value = self.stability_analyzer.find_bifurcation(
            combined,
            parameter_range,
            parameter_name=parameter_name
        )
        
        if param_value is None:
            return None
        
        # Get state and eigenvalues at bifurcation
        state_at_bif = self.evolve(state, float(param_value))
        rd_state_at_bif = self._to_reaction_diffusion_state(state_at_bif)
        combined_state = torch.cat([rd_state_at_bif.activator, rd_state_at_bif.inhibitor], dim=1)
        eigenvalues = self.stability_analyzer.compute_eigenvalues(combined_state)[0]
        
        return BifurcationPoint(
            parameter=float(param_value),
            state=combined_state,
            eigenvalues=eigenvalues,
            type="unknown"
        )

    def compute_lyapunov_spectrum(
        self,
        state: T,
        steps: int = 100
    ) -> torch.Tensor:
        """Compute Lyapunov spectrum."""
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        return self.stability_analyzer.compute_lyapunov_spectrum(
            combined,
            steps=steps,
            dt=self.dt
        )

    def is_stable(
        self,
        state: T,
        threshold: float = 0.1
    ) -> bool:
        """Check stability."""
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        return self.stability_analyzer.is_stable(
            combined,
            threshold=threshold
        )

    @abstractmethod
    def compute_bifurcation(
        self,
        state: T,
        parameter: torch.Tensor,
        parameter_range: Tuple[float, float]
    ) -> BifurcationDiagram:
        """Analyze bifurcation behavior of the system."""
        pass
        
    @abstractmethod
    def to_quantum_state(self, state: T) -> IQuantumState[torch.Tensor]:
        """Convert pattern state to quantum state representation.
        
        This method should convert the pattern state to a pure quantum state
        representation that can be used with quantum operations.
        
        Args:
            state: Pattern state to convert
            
        Returns:
            Quantum state representation
            
        Raises:
            ValueError: If state is invalid
        """
        pass
        
    @abstractmethod
    def from_quantum_state(self, quantum_state: IQuantumState[torch.Tensor]) -> T:
        """Convert quantum state back to pattern state.
        
        This method should convert a quantum state back to the pattern
        representation, preserving the relevant physical properties.
        
        Args:
            quantum_state: Quantum state to convert
            
        Returns:
            Pattern state representation
            
        Raises:
            ValueError: If quantum state is invalid
        """
        pass

    @abstractmethod
    def compute_bifurcation_diagram(
        self,
        state: T,
        parameter_range: Tuple[float, float],
        num_points: int = 100
    ) -> BifurcationDiagram:
        """Compute full bifurcation diagram."""
        pass

    def analyze_crystal_structure(self, state: T) -> Dict[str, Any]:
        """Analyze crystal structure of the pattern state."""
        if not self.enable_crystal:
            raise RuntimeError("Crystal analysis is disabled")
            
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        # Analyze symmetries
        symmetries = self.symmetry_detector.detect_point_symmetries(combined)
        
        # Analyze band structure
        brillouin_zone = BrillouinZone(self.symmetry_detector.lattice)
        band_analyzer = BandStructureAnalyzer(brillouin_zone)
        bands = band_analyzer.compute_band_structure(self._generate_k_path(brillouin_zone))
        
        # Analyze scale hierarchy
        scale_analysis = self.scale_system.analyze_scales(
            [combined],
            torch.randn(1, self.scale_system.rg_flow.coupling_dim)
        )
        hierarchy = scale_analysis[3]  # Get cohomology results
        
        return {
            "symmetry_groups": symmetries,
            "band_structure": bands,
            "scale_hierarchy": hierarchy
        }
    
    def compute_scale_flow(self, state: T) -> RGFlow:
        """Compute renormalization flow of the pattern state."""
        if not self.enable_crystal:
            raise RuntimeError("Crystal analysis is disabled")
            
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        # Generate initial couplings for RG analysis
        initial_couplings = torch.randn(1, self.scale_system.rg_flow.coupling_dim)
        
        # Analyze scales and return RG flow
        scale_analysis = self.scale_system.analyze_scales(
            [combined],
            initial_couplings
        )
        
        # Convert to interface RGFlow type
        rg_flow = scale_analysis[0]  # Get RG flow from analysis
        return RGFlow(
            beta_function=rg_flow.beta_function,
            fixed_points=rg_flow.fixed_points,
            stability=rg_flow.stability,
            flow_lines=rg_flow.flow_lines
        )
    
    def analyze_band_structure(self, state: T) -> IBandStructure:
        """Analyze band structure of the pattern state."""
        if not self.enable_crystal:
            raise RuntimeError("Crystal analysis is disabled")
            
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        # Initialize band analyzer with proper Brillouin zone
        brillouin_zone = BrillouinZone(self.symmetry_detector.lattice)
        band_analyzer = BandStructureAnalyzer(brillouin_zone)
        
        # Generate k-path and compute band structure
        k_path = self._generate_k_path(brillouin_zone)
        return band_analyzer.compute_band_structure(k_path)
    
    def compute_scale_connection(self, state: T) -> ScaleConnection:
        """Compute scale connection of the pattern state."""
        if not self.enable_crystal:
            raise RuntimeError("Crystal analysis is disabled")
            
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        # Get implementation scale connection
        impl_connection = self.scale_system.connection
        
        # Get the first connection module
        connection = impl_connection.connections[0]
        
        # Ensure we have a proper tensor for the connection map
        if not hasattr(connection, 'weight'):
            # Fallback to identity if no weight attribute
            connection_map = torch.eye(self.space_dim, device=self.device)
        else:
            # Get the weight tensor data
            weight = connection.weight
            if isinstance(weight, torch.nn.Parameter):
                connection_map = weight.data
            elif isinstance(weight, torch.Tensor):
                connection_map = weight
            else:
                # Fallback if weight is neither Parameter nor Tensor
                connection_map = torch.eye(self.space_dim, device=self.device)
        
        # Create interface ScaleConnection
        return ScaleConnection(
            source_scale=1.0,  # Default source scale
            target_scale=2.0,  # Default target scale (one level up)
            connection_map=connection_map,  # Now guaranteed to be a Tensor
            holonomy=impl_connection.compute_holonomy([combined]),
            metric=torch.eye(self.space_dim, device=self.device),
            curvature=torch.zeros((self.space_dim, self.space_dim, self.space_dim, self.space_dim), device=self.device)
        )
    
    def _generate_k_path(self, brillouin_zone: BrillouinZone) -> torch.Tensor:
        """Generate k-point path for band structure calculation."""
        # Get high symmetry points
        points = brillouin_zone.high_symmetry_points
        
        # Create path through high symmetry points
        k_points = []
        num_points = 20  # Points between each high symmetry point
        
        # Generate path through main high symmetry points
        main_points = ['Γ', 'X', 'M', 'Γ']  # Example path for cubic lattice
        for i in range(len(main_points) - 1):
            start = points[main_points[i]]
            end = points[main_points[i + 1]]
            
            # Generate points along line
            for t in torch.linspace(0, 1, num_points):
                k_points.append(start + t * (end - start))
                
        return torch.stack(k_points)

    def compute_rg_flow(
        self,
        state: T,
        couplings: torch.Tensor
    ) -> RGFlow:
        """Compute renormalization group flow."""
        if not self.enable_crystal:
            raise RuntimeError("Crystal analysis is disabled")
        
        rd_state = self._to_reaction_diffusion_state(state)
        combined = torch.cat([rd_state.activator, rd_state.inhibitor], dim=1)
        
        # Analyze scales and return RG flow
        scale_analysis = self.scale_system.analyze_scales(
            [combined],
            couplings
        )
        
        # Convert to interface RGFlow type
        rg_flow = scale_analysis[0]  # Get RG flow from analysis
        return RGFlow(
            beta_function=rg_flow.beta_function,
            fixed_points=rg_flow.fixed_points,
            stability=rg_flow.stability,
            flow_lines=rg_flow.flow_lines
        )

    def _compute_band_structure(self, pattern: torch.Tensor) -> IBandStructure:
        """Compute band structure for the current pattern."""
        return self.refraction_system.analyze_pattern(pattern)[2]