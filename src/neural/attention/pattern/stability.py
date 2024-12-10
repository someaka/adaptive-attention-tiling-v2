"""Implementation of stability analysis."""

from typing import Union, Callable, Optional
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
        threshold: float = 0.1
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
            
        # Compute stability value using reaction term
        stability = self.compute_stability(state, reaction_term=reaction_term)
            
        # Check if stability value is valid and below threshold
        if torch.isnan(stability) or torch.isinf(stability):
            return False
            
        return stability < threshold

    def compute_stability(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None,
        max_iter: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """Compute stability value for a state using power iteration.
        
        Args:
            state (torch.Tensor): State to compute stability for
            reaction_term (Optional[Callable]): Optional reaction term function
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
            Av = self._apply_jacobian(state, perturbation, reaction_term)
            
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
        reaction_term: Optional[Callable] = None,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """Apply Jacobian matrix to vector using finite differences.
        
        Args:
            state (torch.Tensor): State point to evaluate Jacobian at
            vector (torch.Tensor): Vector to multiply with Jacobian
            reaction_term (Optional[Callable]): Optional reaction term function
            eps (float): Finite difference step size
            
        Returns:
            torch.Tensor: Result of Jacobian-vector product
        """
        # Forward evaluation with reaction term only
        state_plus = state + eps * vector
        state_minus = state - eps * vector
        
        if reaction_term is not None:
            state_plus = reaction_term(state_plus)
            state_minus = reaction_term(state_minus)
        else:
            state_plus = self.pattern_system.reaction.reaction_term(state_plus)
            state_minus = self.pattern_system.reaction.reaction_term(state_minus)
        
        # Compute finite difference approximation
        diff = (state_plus - state_minus) / (2 * eps)
        
        return diff

    def compute_lyapunov_spectrum(
        self,
        state: torch.Tensor,
        steps: int = 100,
        dt: float = None
    ) -> torch.Tensor:
        """Compute Lyapunov spectrum for a state.
        
        Args:
            state: State to compute spectrum for
            steps: Number of integration steps
            dt: Time step (optional)
            
        Returns:
            torch.Tensor: Lyapunov spectrum
        """
        # Get system size
        batch, channels, height, width = state.shape
        n = channels * height * width
        
        # Initialize orthonormal perturbations
        Q = torch.eye(n, dtype=torch.float64, device=state.device)
        Q = Q.reshape(n, channels, height, width)
        
        # Initialize Lyapunov exponents
        lyap = torch.zeros(n, dtype=torch.float64, device=state.device)
        
        # Time integration
        dt = dt or self.pattern_system.dt
        for _ in range(steps):
            # Evolve perturbations
            for i in range(n):
                # Apply Jacobian
                Q[i] = self._apply_jacobian(state, Q[i])
                
                # Normalize
                norm = torch.norm(Q[i])
                if norm > 0:
                    Q[i] = Q[i] / norm
                    lyap[i] += torch.log(norm)
        
        # Average and sort
        lyap = lyap / (steps * dt)
        lyap = lyap.sort(descending=True)[0]
        
        return lyap.to(torch.float64)

    def analyze_stability(
        self,
        state: torch.Tensor,
        perturbation: torch.Tensor
    ) -> StabilityMetrics:
        """Analyze stability of a state under perturbation.
        
        Args:
            state: State to analyze stability for
            perturbation: Perturbation to apply
            
        Returns:
            StabilityMetrics: Stability analysis results
        """
        # Compute linear stability using eigenvalue analysis
        stability_matrix = self.pattern_system.compute_stability_matrix(state)
        eigenvals = torch.linalg.eigvals(stability_matrix)
        linear_stability = torch.max(eigenvals.real).item()
        
        # Compute nonlinear stability using perturbation
        perturbed_state = state + perturbation
        nonlinear_stability = self.compute_stability(perturbed_state)
        
        # Return metrics
        return StabilityMetrics(
            linear_stability=linear_stability,
            nonlinear_stability=nonlinear_stability,
            lyapunov_spectrum=eigenvals,
            structural_stability=float(nonlinear_stability / linear_stability)
        )
