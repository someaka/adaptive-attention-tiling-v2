"""Data models for pattern dynamics."""

from dataclasses import dataclass
from typing import List, Optional, Callable
import torch


@dataclass
class ReactionDiffusionState:
    """State of the reaction-diffusion system."""
    activator: torch.Tensor  # Activator concentration
    inhibitor: torch.Tensor  # Inhibitor concentration
    gradients: Optional[torch.Tensor] = None  # Spatial gradients (optional)
    time: float = 0.0  # Current time (default 0.0)

    def sum(self, dim=None) -> torch.Tensor:
        """Compute sum of activator and inhibitor concentrations."""
        if dim is None:
            return self.activator.sum() + self.inhibitor.sum()
        else:
            return self.activator.sum(dim=dim) + self.inhibitor.sum(dim=dim)


@dataclass
class StabilityInfo:
    """Information about pattern stability."""
    eigenvalues: torch.Tensor  # Stability eigenvalues
    eigenvectors: torch.Tensor  # Corresponding modes
    growth_rates: torch.Tensor  # Mode growth rates
    stable: bool  # Overall stability flag


@dataclass
class StabilityMetrics:
    """Metrics for pattern stability analysis."""
    linear_stability: torch.Tensor
    nonlinear_stability: torch.Tensor
    lyapunov_spectrum: torch.Tensor
    structural_stability: float


class ControlSignal:
    """Control signal for pattern dynamics."""

    signal: torch.Tensor
    direction: torch.Tensor
    constraints: List[Callable]

    def __init__(
        self,
        signal: torch.Tensor,
        direction: Optional[torch.Tensor] = None,
        constraints: Optional[List[Callable]] = None
    ):
        """Initialize control signal.
        
        Args:
            signal: Control signal tensor
            direction: Optional preferred direction
            constraints: Optional list of constraint functions
        """
        self.signal = signal
        self.direction = direction if direction is not None else torch.zeros_like(signal)
        self.constraints = constraints if constraints is not None else []

    @property
    def magnitude(self) -> torch.Tensor:
        """Get control signal magnitude.
        
        Returns:
            Magnitude of the control signal
        """
        return torch.norm(self.signal)

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply control signal to state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Controlled state tensor
        """
        # Apply control signal
        controlled = state + self.signal
        
        # Project along preferred direction if specified
        if torch.any(self.direction != 0):
            direction_norm = self.direction / torch.norm(self.direction)
            projection = torch.sum(controlled * direction_norm, dim=(-2, -1), keepdim=True)
            controlled = projection * direction_norm
        
        # Apply constraints
        for constraint in self.constraints:
            controlled = constraint(controlled)
            
        return controlled


@dataclass
class BifurcationPoint:
    """Bifurcation point information."""
    parameter: float
    state: torch.Tensor
    eigenvalues: torch.Tensor
    type: str


@dataclass
class BifurcationDiagram:
    """Bifurcation diagram for pattern dynamics."""
    solution_states: torch.Tensor  # States at each parameter value
    solution_params: torch.Tensor  # Parameter values where solutions were found
    bifurcation_points: torch.Tensor  # Parameter values where bifurcations occur
    metrics: Optional[dict] = None  # Optional dictionary of analysis metrics
