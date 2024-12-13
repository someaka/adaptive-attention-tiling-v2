"""Pattern formation module.

This module implements pattern formation dynamics and analysis tools.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

class BifurcationAnalyzer:
    """Analyzer for bifurcation points in pattern dynamics."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 10
    ):
        """Initialize bifurcation analyzer.
        
        Args:
            threshold: Threshold for detecting bifurcations
            window_size: Window size for temporal analysis
        """
        self.threshold = threshold
        self.window_size = window_size
        
    def detect_bifurcations(
        self,
        pattern: torch.Tensor,
        parameter: torch.Tensor
    ) -> List[float]:
        """Detect bifurcation points in pattern evolution.
        
        Args:
            pattern: Pattern evolution tensor [time, ...]
            parameter: Control parameter values
            
        Returns:
            List of bifurcation points
        """
        # Compute stability metrics along parameter range
        stability_metrics = []
        for i in range(len(parameter)):
            metrics = self._compute_stability_metrics(pattern[i])
            stability_metrics.append(metrics)
            
        # Detect significant changes in stability
        bifurcations = []
        for i in range(1, len(stability_metrics)):
            if self._is_bifurcation(
                stability_metrics[i-1],
                stability_metrics[i]
            ):
                bifurcations.append(float(parameter[i].item()))
                
        return bifurcations
        
    def _compute_stability_metrics(
        self,
        pattern: torch.Tensor
    ) -> Dict[str, float]:
        """Compute stability metrics for pattern state."""
        # Compute temporal derivatives
        if pattern.dim() > 1:
            grad = torch.gradient(pattern)[0]
            mean_rate = torch.mean(torch.abs(grad)).item()
            max_rate = torch.max(torch.abs(grad)).item()
        else:
            mean_rate = 0.0
            max_rate = 0.0
            
        # Compute amplitude metrics
        mean_amp = torch.mean(torch.abs(pattern)).item()
        max_amp = torch.max(torch.abs(pattern)).item()
        
        return {
            "mean_rate": mean_rate,
            "max_rate": max_rate,
            "mean_amplitude": mean_amp,
            "max_amplitude": max_amp
        }
        
    def _is_bifurcation(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float]
    ) -> bool:
        """Check if transition between states is a bifurcation."""
        # Check for significant changes in metrics
        for key in metrics1:
            if abs(metrics2[key] - metrics1[key]) > self.threshold:
                return True
        return False
        
    def analyze_stability(
        self,
        pattern: torch.Tensor,
        parameter_range: Tuple[float, float],
        num_points: int = 100
    ) -> Dict[str, Any]:
        """Analyze pattern stability across parameter range.
        
        Args:
            pattern: Initial pattern state
            parameter_range: Range of parameter values
            num_points: Number of points to sample
            
        Returns:
            Dictionary with stability analysis results
        """
        # Generate parameter values
        parameters = torch.linspace(
            parameter_range[0],
            parameter_range[1],
            num_points
        )
        
        # Evolve pattern across parameter range
        evolution = []
        for param in parameters:
            state = self._evolve_pattern(pattern, param)
            evolution.append(state)
            
        evolution = torch.stack(evolution)
        
        # Find bifurcation points
        bifurcations = self.detect_bifurcations(evolution, parameters)
        
        # Compute stability metrics
        stability = []
        for state in evolution:
            metrics = self._compute_stability_metrics(state)
            stability.append(metrics)
            
        return {
            "bifurcation_points": bifurcations,
            "stability_metrics": stability,
            "parameter_values": parameters,
            "evolution": evolution
        }
        
    def _evolve_pattern(
        self,
        pattern: torch.Tensor,
        parameter: Union[float, torch.Tensor],
        time_steps: int = 100
    ) -> torch.Tensor:
        """Evolve pattern for given parameter value.
        
        Args:
            pattern: Initial pattern state
            parameter: Evolution parameter (float or tensor)
            time_steps: Number of time steps
            
        Returns:
            Evolved pattern state
        """
        if isinstance(parameter, torch.Tensor):
            parameter = parameter.item()
        evolved = pattern + parameter * torch.randn_like(pattern)
        return evolved

class PatternFormation:
    """Class for pattern formation dynamics."""
    
    def __init__(self, 
                 dim: int = 3,
                 dt: float = 0.1,
                 diffusion_coeff: float = 0.1,
                 reaction_coeff: float = 1.0):
        """Initialize pattern formation.
        
        Args:
            dim: Dimension of pattern space
            dt: Time step for integration
            diffusion_coeff: Diffusion coefficient
            reaction_coeff: Reaction coefficient
        """
        self.dim = dim
        self.dt = dt
        self.diffusion_coeff = diffusion_coeff
        self.reaction_coeff = reaction_coeff
        
        # Initialize diffusion kernel
        self.diffusion_kernel = torch.tensor([[[0.2, 0.6, 0.2]]])
        
    def evolve(self, 
               pattern: torch.Tensor,
               time_steps: int) -> torch.Tensor:
        """Evolve pattern according to reaction-diffusion dynamics.
        
        Args:
            pattern: Initial pattern tensor of shape (batch_size, dim)
            time_steps: Number of time steps to evolve
            
        Returns:
            torch.Tensor: Evolved pattern trajectory of shape (batch_size, time_steps, dim)
        """
        batch_size = pattern.size(0)
        
        # Initialize trajectory tensor
        trajectory = torch.zeros(batch_size, time_steps, self.dim)
        trajectory[:, 0] = pattern
        
        # Evolve pattern
        for t in range(1, time_steps):
            # Diffusion term
            diffusion = torch.nn.functional.conv1d(
                trajectory[:, t-1:t].unsqueeze(1),
                self.diffusion_kernel,
                padding=1
            ).squeeze(1)
            
            # Reaction term (cubic nonlinearity)
            reaction = trajectory[:, t-1] * (1 - trajectory[:, t-1]**2)
            
            # Update pattern
            trajectory[:, t] = trajectory[:, t-1] + self.dt * (
                self.diffusion_coeff * diffusion + 
                self.reaction_coeff * reaction
            )
            
        return trajectory
        
    def compute_energy(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute energy of pattern.
        
        Args:
            pattern: Pattern tensor of shape (batch_size, dim)
            
        Returns:
            torch.Tensor: Energy of pattern
        """
        # Compute gradient term
        grad = torch.diff(pattern, dim=-1)
        grad_energy = torch.sum(grad**2, dim=-1)
        
        # Compute potential term (double-well potential)
        potential = 0.25 * pattern**4 - 0.5 * pattern**2
        potential_energy = torch.sum(potential, dim=-1)
        
        return grad_energy + potential_energy
        
    def compute_stability(self, pattern: torch.Tensor) -> Dict[str, Any]:
        """Compute stability metrics for pattern.
        
        Args:
            pattern: Pattern tensor of shape (batch_size, dim)
            
        Returns:
            Dict containing stability metrics
        """
        # Compute Jacobian
        x = pattern.requires_grad_(True)
        y = self.evolve(x, time_steps=2)[:, 1]
        jac = torch.autograd.functional.jacobian(
            lambda x: self.evolve(x, time_steps=2)[:, 1],
            pattern
        )
        
        # Convert jacobian to proper tensor shape
        if isinstance(jac, tuple):
            jac = torch.stack(list(jac))
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(jac)
        
        # Compute stability metrics
        max_eigenval = torch.max(eigenvals.real)
        stability_margin = -max_eigenval.item()
        
        return {
            'stability_margin': stability_margin,
            'max_eigenvalue': max_eigenval.item(),
            'eigenvalues': eigenvals.detach()
        }
