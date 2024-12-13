"""Pattern perturbation analysis implementation.

This module provides tools for analyzing pattern response to perturbations:
- Linear response analysis
- Nonlinear effects
- Stability under perturbations
- Recovery time estimation
"""

from typing import Dict, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass

from src.neural.attention.pattern.dynamics import PatternDynamics

@dataclass
class PerturbationMetrics:
    """Metrics for perturbation analysis."""
    
    linear_response: torch.Tensor
    """Linear response to perturbation."""
    
    recovery_time: float
    """Time to recover from perturbation."""
    
    stability_margin: float
    """Margin of stability."""
    
    max_amplitude: float
    """Maximum perturbation amplitude."""


class PerturbationAnalyzer:
    """Analysis of pattern response to perturbations."""
    
    def __init__(self, tolerance: float = 1e-6, max_time: int = 1000):
        """Initialize analyzer.
        
        Args:
            tolerance: Numerical tolerance
            max_time: Maximum simulation time
        """
        self.tolerance = tolerance
        self.max_time = max_time
        
    def analyze_perturbation(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> PerturbationMetrics:
        """Analyze pattern response to perturbation.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Base pattern to perturb
            perturbation: Optional specific perturbation to analyze
            
        Returns:
            Perturbation analysis metrics
        """
        if perturbation is None:
            perturbation = self._generate_perturbation(pattern)
            
        # Compute linear response
        linear_response = self._compute_linear_response(
            dynamics, pattern, perturbation)
            
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(
            dynamics, pattern, perturbation)
            
        # Compute stability margin
        stability_margin = self._compute_stability_margin(
            dynamics, pattern, perturbation)
            
        # Find maximum stable amplitude
        max_amplitude = self._find_max_amplitude(
            dynamics, pattern)
            
        return PerturbationMetrics(
            linear_response=linear_response,
            recovery_time=recovery_time,
            stability_margin=stability_margin,
            max_amplitude=max_amplitude
        )
        
    def _generate_perturbation(
        self,
        pattern: torch.Tensor,
        amplitude: float = 0.1
    ) -> torch.Tensor:
        """Generate random perturbation of pattern.
        
        Args:
            pattern: Pattern to perturb
            amplitude: Perturbation amplitude
            
        Returns:
            Perturbation tensor
        """
        return torch.randn_like(pattern) * amplitude
        
    def _compute_linear_response(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Compute linearized response to perturbation.
        
        Args:
            dynamics: Pattern dynamics
            pattern: Base pattern
            perturbation: Perturbation to analyze
            
        Returns:
            Linear response tensor
        """
        # Get Jacobian
        jacobian = dynamics.compute_jacobian(pattern)
        
        # Compute linear response
        response = torch.matmul(jacobian, perturbation.flatten())
        return response.reshape(pattern.shape)
        
    def _estimate_recovery_time(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        perturbation: torch.Tensor
    ) -> float:
        """Estimate time to recover from perturbation.
        
        Args:
            dynamics: Pattern dynamics
            pattern: Base pattern
            perturbation: Perturbation to analyze
            
        Returns:
            Recovery time estimate
        """
        perturbed = pattern + perturbation
        
        # Simulate until recovery
        current = perturbed
        for t in range(self.max_time):
            current = dynamics.evolve(current, 1)
            
            # Check if recovered
            if torch.norm(current - pattern) < self.tolerance:
                return float(t)
                
        return float(self.max_time)
        
    def _compute_stability_margin(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        perturbation: torch.Tensor
    ) -> float:
        """Compute stability margin for perturbation.
        
        Args:
            dynamics: Pattern dynamics
            pattern: Base pattern
            perturbation: Perturbation to analyze
            
        Returns:
            Stability margin
        """
        # Get eigenvalues of Jacobian
        jacobian = dynamics.compute_jacobian(pattern)
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        # Stability margin is negative of largest real part
        return float(-torch.max(eigenvalues.real))
        
    def _find_max_amplitude(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
    ) -> float:
        """Find maximum stable perturbation amplitude.
        
        Args:
            dynamics: Pattern dynamics
            pattern: Base pattern
            
        Returns:
            Maximum stable amplitude
        """
        # Binary search for maximum amplitude
        low = 0.0
        high = 1.0
        
        for _ in range(20):  # 20 iterations for precision
            mid = (low + high) / 2
            perturbation = self._generate_perturbation(pattern, mid)
            
            # Check if stable
            perturbed = pattern + perturbation
            final = dynamics.evolve(perturbed, self.max_time)
            
            if torch.norm(final - pattern) < self.tolerance:
                low = mid  # Stable, try larger
            else:
                high = mid  # Unstable, try smaller
                
        return low  # Return largest stable amplitude
