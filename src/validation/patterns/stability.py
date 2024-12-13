"""Pattern stability validation implementation."""

from typing import Optional, Dict, List, Tuple, Union
import torch
import numpy as np

from ...neural.attention.pattern.dynamics import PatternDynamics
from ...neural.attention.pattern.models import StabilityMetrics, StabilityInfo
from ..base import BaseValidator


class PatternStabilityValidator(BaseValidator):
    """Validator for pattern stability properties."""

    def __init__(
        self,
        dynamics: PatternDynamics,
        threshold: float = 0.1,
        window_size: int = 100,
        num_modes: int = 8
    ):
        """Initialize pattern stability validator.
        
        Args:
            dynamics: Pattern dynamics system
            threshold: Stability threshold
            window_size: Analysis window size
            num_modes: Number of stability modes to analyze
        """
        super().__init__()
        self.dynamics = dynamics
        self.threshold = threshold
        self.window = window_size
        self.num_modes = num_modes

    def validate(
        self,
        state: torch.Tensor,
        time_steps: int = 1000,
        perturbation: Optional[torch.Tensor] = None
    ) -> StabilityInfo:
        """Validate stability of pattern dynamics.
        
        Args:
            state: Initial state
            time_steps: Number of time steps
            perturbation: Optional perturbation to apply
            
        Returns:
            Stability validation results
        """
        # Compute base trajectory
        base_traj = self.dynamics.evolve(state, time_steps)
        
        # Apply perturbation if provided
        if perturbation is not None:
            perturbed_state = state + perturbation
            perturbed_traj = self.dynamics.evolve(perturbed_state, time_steps)
        else:
            perturbed_traj = None
            
        # Analyze stability
        metrics = self._compute_stability_metrics(base_traj, perturbed_traj)
        
        # Package results
        info = StabilityInfo(
            metrics=metrics,
            is_stable=metrics.max_lyapunov < self.threshold,
            recovery_time=self._compute_recovery_time(base_traj, perturbed_traj)
            if perturbed_traj is not None else None
        )
        
        return info

    def _compute_stability_metrics(
        self,
        base_traj: torch.Tensor,
        perturbed_traj: Optional[torch.Tensor] = None
    ) -> StabilityMetrics:
        """Compute stability metrics from trajectories.
        
        Args:
            base_traj: Base trajectory
            perturbed_traj: Optional perturbed trajectory
            
        Returns:
            Stability metrics
        """
        # Get Jacobian at final state
        final_state = base_traj[-1]
        J = self.dynamics.compute_jacobian(final_state)
        
        # Compute eigenvalues
        eigvals = torch.linalg.eigvals(J)
        max_lyap = torch.max(eigvals.real).item()
        
        # Compute stability margin
        margin = self.threshold - max_lyap
        
        # Compute recovery metrics if perturbed trajectory available
        if perturbed_traj is not None:
            recovery_error = torch.norm(base_traj - perturbed_traj, dim=(-2, -1))
            recovery_rate = torch.mean(recovery_error[1:] / recovery_error[:-1])
        else:
            recovery_rate = None
            
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
        if perturbed_traj is None:
            return None
            
        # Compute error over time
        error = torch.norm(base_traj - perturbed_traj, dim=(-2, -1))
        
        # Find first time error drops below threshold
        recovered = torch.where(error < self.threshold)[0]
        
        if len(recovered) > 0:
            return recovered[0].item()
        else:
            return None
