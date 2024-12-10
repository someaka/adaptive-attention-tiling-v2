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


@dataclass
class ControlSignal:
    """Control signal for pattern formation."""
    magnitude: torch.Tensor
    direction: torch.Tensor
    constraints: List[Callable]


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
    parameter_range: torch.Tensor
    bifurcation_points: List[BifurcationPoint]
    solution_branches: torch.Tensor
    stability_regions: torch.Tensor
