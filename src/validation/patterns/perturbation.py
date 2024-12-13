"""Pattern perturbation analysis implementation."""

from typing import Optional, Dict, List, Tuple, Union
import torch
import numpy as np

from ...neural.attention.pattern.dynamics import PatternDynamics
from ...neural.attention.pattern.models import StabilityMetrics, StabilityInfo
from ..base import BaseValidator


class PerturbationAnalyzer(BaseValidator):
    """Analyzer for pattern perturbation response."""

    def __init__(
        self,
        dynamics: PatternDynamics,
        threshold: float = 0.1,
        window_size: int = 100,
        num_modes: int = 8
    ):
        """Initialize perturbation analyzer.
        
        Args:
            dynamics: Pattern dynamics system
            threshold: Recovery threshold
            window_size: Analysis window size
            num_modes: Number of modes to analyze
        """
        super().__init__()
        self.dynamics = dynamics
        self.threshold = threshold
        self.window = window_size
        self.num_modes = num_modes

    def analyze_perturbation(
        self,
        state: torch.Tensor,
        perturbation: torch.Tensor,
        time_steps: int = 1000
    ) -> StabilityInfo:
        """Analyze perturbation response.
        
        Args:
            state: Base state
            perturbation: Perturbation to apply
            time_steps: Number of time steps
            
        Returns:
            Stability analysis results
        """
        # Get base and perturbed trajectories
        base_traj = self.dynamics.evolve(state, time_steps)
        perturbed_traj = self.dynamics.evolve(state + perturbation, time_steps)
        
        # Compute stability metrics
        metrics = self._compute_stability_metrics(base_traj, perturbed_traj)
        
        # Get recovery time
        recovery_time = self._compute_recovery_time(base_traj, perturbed_traj)
        
        return StabilityInfo(
            metrics=metrics,
            is_stable=metrics.max_lyapunov < self.threshold,
            recovery_time=recovery_time
        )

    def _compute_stability_metrics(
        self,
        base_traj: torch.Tensor,
        perturbed_traj: torch.Tensor
    ) -> StabilityMetrics:
        """Compute stability metrics from trajectories.
        
        Args:
            base_traj: Base trajectory
            perturbed_traj: Perturbed trajectory
            
        Returns:
            Stability metrics
        """
        # Compute error trajectory
        error = torch.norm(perturbed_traj - base_traj, dim=(-2, -1))
        
        # Get max Lyapunov exponent from error growth
        growth_rates = torch.log(error[1:] / error[:-1])
        max_lyap = torch.mean(growth_rates).item()
        
        # Compute stability margin
        margin = self.threshold - max_lyap
        
        # Get recovery rate
        recovery_rate = torch.mean(error[1:] / error[:-1]).item()
        
        return StabilityMetrics(
            max_lyapunov=max_lyap,
            stability_margin=margin,
            recovery_rate=recovery_rate
        )

    def _compute_recovery_time(
        self,
        base_traj: torch.Tensor,
        perturbed_traj: torch.Tensor
    ) -> Optional[int]:
        """Compute recovery time between trajectories.
        
        Args:
            base_traj: Base trajectory
            perturbed_traj: Perturbed trajectory
            
        Returns:
            Recovery time steps or None if no recovery
        """
        # Compute error over time
        error = torch.norm(perturbed_traj - base_traj, dim=(-2, -1))
        
        # Find first time error drops below threshold
        recovered = torch.where(error < self.threshold)[0]
        
        if len(recovered) > 0:
            return recovered[0].item()
        else:
            return None

    def generate_perturbations(
        self,
        state: torch.Tensor,
        num_perturbations: int = 10,
        magnitude: float = 0.1
    ) -> torch.Tensor:
        """Generate random perturbations for analysis.
        
        Args:
            state: Base state to perturb
            num_perturbations: Number of perturbations to generate
            magnitude: Perturbation magnitude
            
        Returns:
            Tensor of perturbations
        """
        # Generate random perturbations
        shape = (num_perturbations,) + state.shape
        perturbations = torch.randn(shape, device=state.device)
        
        # Normalize and scale
        norms = torch.norm(perturbations, dim=(-2, -1), keepdim=True)
        perturbations = perturbations / norms * magnitude
        
        return perturbations
