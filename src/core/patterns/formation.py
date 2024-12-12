"""Pattern formation module.

This module implements pattern formation dynamics and analysis tools.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

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
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(jac.squeeze())
        
        # Compute stability metrics
        max_eigenval = torch.max(eigenvals.real)
        stability_margin = -max_eigenval.item()
        
        return {
            'stability_margin': stability_margin,
            'max_eigenvalue': max_eigenval.item(),
            'eigenvalues': eigenvals.detach()
        }
