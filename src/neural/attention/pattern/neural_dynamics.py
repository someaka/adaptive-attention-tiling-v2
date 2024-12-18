"""Neural pattern dynamics implementation.

This module implements pattern dynamics specialized for neural attention:
1. Neural-specific quantum evolution
2. Attention-based pattern control
3. Neural stability analysis
4. Hidden state projections
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, TypeVar, Generic, Protocol, runtime_checkable, cast, Union
import torch
import torch.nn as nn

from ....core.patterns.base_dynamics import BasePatternDynamics
from ....core.interfaces.quantum import (
    IQuantumState,
    EvolutionType,
    GeometricFlow,
    MeasurementResult,
    EntanglementMetrics,
    PatternState
)
from ....core.interfaces.geometric import HilbertSpace
from .quantum import QuantumState, QuantumGeometricTensor
from .models import (
    StabilityMetrics,
    BifurcationDiagram,
    ReactionDiffusionState
)

T = TypeVar('T', bound=torch.Tensor)

class NeuralPatternDynamics(BasePatternDynamics[T]):
    """Neural-specific pattern dynamics implementation."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_modes: int = 8,
        num_heads: int = 4,
        temperature: float = 0.1,
        **kwargs
    ):
        """Initialize neural pattern dynamics.
        
        Args:
            hidden_dim: Hidden state dimension
            num_modes: Number of pattern modes
            num_heads: Number of attention heads
            temperature: Temperature for attention
            **kwargs: Base class arguments
        """
        super().__init__(**kwargs)
        
        # Validate neural parameters
        if hidden_dim < 1:
            raise ValueError("Hidden dimension must be positive")
        if num_modes < 1:
            raise ValueError("Number of modes must be positive")
        if num_heads < 1:
            raise ValueError("Number of attention heads must be positive")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Initialize neural components
        self.mode_proj = nn.Linear(self.space_dim, hidden_dim)
        self.pattern_proj = nn.Linear(hidden_dim, self.space_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # Move to device
        self.to(self.device)
        
    def to(self, device: torch.device) -> 'NeuralPatternDynamics[T]':
        """Move neural components to device."""
        self.mode_proj = self.mode_proj.to(device)
        self.pattern_proj = self.pattern_proj.to(device)
        self.attention = self.attention.to(device)
        return self
        
    def compute_energy(
        self,
        state: T
    ) -> Dict[str, torch.Tensor]:
        """Compute energy components including neural contributions.
        
        Args:
            state: Current pattern state
            
        Returns:
            Dictionary containing energy components
        """
        # Get base energy components
        base_energy = super().compute_energy(cast(T, state))
        
        # Add neural energy components
        hidden = self.mode_proj(state)
        attention_energy = torch.sum(hidden**2) / (2.0 * self.temperature)
        
        # Update energy dictionary
        base_energy.update({
            "attention": attention_energy,
            "total": base_energy["total"] + attention_energy
        })
        
        return base_energy
        
    def _validate_state(self, state: T) -> None:
        """Validate state tensor shape and values."""
        if state.dim() != 2:
            raise ValueError(f"Expected 2D state tensor, got shape {state.shape}")
        if state.size(1) != self.space_dim:
            raise ValueError(
                f"State spatial dimension {state.size(1)} "
                f"does not match space_dim {self.space_dim}"
            )
            
    def to_quantum_state(self, state: T) -> IQuantumState[T]:
        """Convert pattern state to quantum state.
        
        This implementation:
        1. Projects to hidden space
        2. Applies attention mechanism
        3. Creates quantum state
        
        Args:
            state: Pattern state to convert
            
        Returns:
            Quantum state representation
        """
        self._validate_state(state)
        
        # Project to hidden space
        hidden = self.mode_proj(state)
        
        # Apply attention mechanism
        attention_output = self.attention(
            hidden.unsqueeze(0),
            hidden.unsqueeze(0),
            hidden.unsqueeze(0)
        )
        attention_weights = attention_output[1]
        phase = torch.angle(attention_weights).squeeze(0)
        
        # Create quantum state that implements IQuantumState[T]
        quantum_state = cast(IQuantumState[T], QuantumState(
            amplitude=hidden,
            phase=phase
        ))
        return quantum_state
    
    def from_quantum_state(self, quantum_state: IQuantumState[T]) -> T:
        """Convert quantum state back to pattern state.
        
        This implementation:
        1. Combines amplitude and phase
        2. Projects back to pattern space
        
        Args:
            quantum_state: Quantum state to convert
            
        Returns:
            Pattern state representation
        """
        if not isinstance(quantum_state, QuantumState):
            raise ValueError("Expected QuantumState instance")
            
        # Combine amplitude and phase
        complex_state = quantum_state.amplitude * torch.exp(1j * quantum_state.phase)
        
        # Project back to pattern space
        pattern_state = cast(T, self.pattern_proj(complex_state.real))
        
        return pattern_state
    
    def _quantum_evolution_step(
        self,
        state: T,
        flow: GeometricFlow[T]
    ) -> T:
        """Neural-specific quantum evolution step."""
        self._validate_state(state)
        
        # Project to hidden space
        hidden = self.mode_proj(state)
        
        # Apply attention mechanism
        attended = self.attention(
            hidden.unsqueeze(0),
            hidden.unsqueeze(0),
            hidden.unsqueeze(0)
        )[0].squeeze(0)
        
        # Apply flow in hidden space
        evolved = flow.evolve_state(attended, self.dt)
        
        # Project back to state space
        return cast(T, self.pattern_proj(evolved))
    
    def _compute_diffusion(self, state: T) -> T:
        """Neural-specific diffusion using attention."""
        # Project to attention space
        query = self.mode_proj(state)
        key = query
        value = query
        
        # Apply attention diffusion
        diffused, _ = self.attention(query, key, value)
        
        # Project back and compute difference
        result = self.pattern_proj(diffused)
        return cast(T, (result - state) / self.dt)
    
    def _compute_reaction(self, state: T) -> T:
        """Neural-specific nonlinear reaction term."""
        # Compute activation gradients
        hidden = self.mode_proj(state)
        activated = torch.tanh(hidden / self.temperature)
        
        # Project back to get reaction term
        reaction = self.pattern_proj(activated - hidden)
        return cast(T, reaction)
    
    def compute_stability(
        self,
        state: T,
        perturbation: Optional[T] = None
    ) -> Dict[str, float]:
        """Compute neural stability metrics."""
        self._validate_state(state)
        
        # Get stability in hidden space
        hidden = self.mode_proj(state)
        
        if perturbation is not None:
            self._validate_state(perturbation)
            # Project perturbation
            perturbed = self.mode_proj(state + perturbation)
            # Compute stability metrics
            sensitivity = torch.norm(perturbed - hidden) / torch.norm(perturbation)
        else:
            sensitivity = torch.tensor(0.0, device=self.device)
            
        # Compute attention stability
        attention_output = self.attention(
            hidden.unsqueeze(0),
            hidden.unsqueeze(0),
            hidden.unsqueeze(0)
        )
        attention_weights = attention_output[1]
        attention_stability = -torch.norm(attention_weights)
        
        return {
            "sensitivity": float(sensitivity),
            "attention_stability": float(attention_stability)
        }
    
    def compute_bifurcation_diagram(
        self,
        state: T,
        parameter_range: Tuple[float, float],
        num_points: int = 100
    ) -> BifurcationDiagram:
        """Compute full bifurcation diagram."""
        self._validate_state(state)
        
        # Initialize storage
        solution_states = []
        solution_params = []
        bifurcation_points = []
        
        # Scan parameter range
        param_values = torch.linspace(
            parameter_range[0],
            parameter_range[1],
            num_points,
            device=self.device
        )
        
        prev_stability = None
        for param in param_values:
            # Update temperature parameter
            old_temp = self.temperature
            self.temperature = float(param)
            
            try:
                # Evolve state
                evolved = self.evolve(state, self.dt)
                solution_states.append(evolved)
                solution_params.append(float(param))
                
                # Compute stability
                stability = self.compute_stability(evolved)
                
                # Check for bifurcations
                if prev_stability is not None:
                    if (stability["attention_stability"] * 
                        prev_stability["attention_stability"]) < 0:
                        # Found a bifurcation point
                        bifurcation_points.append(float(param))
                
                prev_stability = stability
                
            finally:
                # Restore temperature
                self.temperature = old_temp
        
        return BifurcationDiagram(
            solution_states=torch.stack(solution_states),
            solution_params=torch.tensor(solution_params),
            bifurcation_points=torch.tensor(bifurcation_points)
        )
    
    def _compute_metric_tensor(self, state: Union[T, ReactionDiffusionState]) -> torch.Tensor:
        """Compute metric tensor for neural state."""
        # Convert tensor to ReactionDiffusionState if needed
        if isinstance(state, torch.Tensor):
            rd_state = self._to_reaction_diffusion_state(state)
        else:
            rd_state = state
        
        # Call base class implementation with properly cast state
        return super()._compute_metric_tensor(rd_state)
    
    def _to_reaction_diffusion_state(self, state: T) -> ReactionDiffusionState:
        """Convert tensor state to ReactionDiffusionState."""
        # Split tensor into activator and inhibitor components
        if state.shape[1] != 2:
            raise ValueError(f"Expected state with 2 components, got shape {state.shape}")
            
        return ReactionDiffusionState(
            activator=state[:,0:1],
            inhibitor=state[:,1:2],
            time=0.0
        )