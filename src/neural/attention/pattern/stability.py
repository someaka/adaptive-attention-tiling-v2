"""Implementation of stability analysis."""

from typing import Union, Callable
import torch
from torch import nn

from .models import ReactionDiffusionState, StabilityInfo, StabilityMetrics


class StabilityAnalyzer:
    """Analyzer for pattern stability and bifurcations."""
    
    def __init__(self, pattern_system):
        """Initialize stability analyzer.
        
        Args:
            pattern_system: Pattern dynamics system to analyze
        """
        self.pattern_system = pattern_system
        
    def is_stable(
        self,
        state: torch.Tensor,
        reaction_term: Callable[[torch.Tensor], torch.Tensor],
        threshold: float = 1.0
    ) -> bool:
        """Check if a state is stable under the given dynamics.
        
        Args:
            state (torch.Tensor): State to check stability for
            reaction_term (Callable): Reaction function
            threshold (float): Stability threshold
            
        Returns:
            bool: True if state is stable, False otherwise
        """
        # Check if state is numerically valid
        if torch.isnan(state).any() or torch.isinf(state).any():
            return False
            
        # Compute stability value
        stability = torch.tensor(self.compute_stability(state), device=state.device)
            
        # Check if stability value is valid and below threshold
        if torch.isnan(stability) or torch.isinf(stability):
            return False
            
        return stability < threshold

    def compute_stability(
        self,
        state: torch.Tensor,
        max_iter: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """Compute stability value for a state using power iteration.
        
        Args:
            state (torch.Tensor): State to compute stability for
            max_iter (int): Maximum power iteration steps
            tolerance (float): Convergence tolerance
            
        Returns:
            float: Stability value
        """
        # Initialize random perturbation
        perturbation = torch.randn_like(state)
        perturbation = perturbation / torch.norm(perturbation)
        
        # Power iteration to find largest eigenvalue
        prev_lambda = 0.0
        for _ in range(max_iter):
            # Apply Jacobian-vector product
            Av = self._apply_jacobian(state, perturbation)
            
            # Normalize
            lambda_i = torch.norm(Av)
            if lambda_i == 0:
                return 0.0
                
            perturbation = Av / lambda_i
            
            # Check convergence
            if abs(lambda_i - prev_lambda) < tolerance:
                break
                
            prev_lambda = lambda_i
            
        return prev_lambda.item()

    def _apply_jacobian(
        self,
        state: torch.Tensor,
        vector: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """Apply Jacobian matrix to vector using finite differences.
        
        Args:
            state (torch.Tensor): State point to evaluate Jacobian at
            vector (torch.Tensor): Vector to multiply with Jacobian
            eps (float): Finite difference step size
            
        Returns:
            torch.Tensor: Result of Jacobian-vector product
        """
        # Forward evaluation
        state_plus = state + eps * vector
        state_minus = state - eps * vector
        
        # Compute finite difference approximation
        diff = (state_plus - state_minus) / (2 * eps)
        
        return diff
