"""Pattern Dynamics Implementation for Neural Attention.

This module has been refactored into the pattern/ directory for better organization.
This file now serves as a compatibility layer for existing code.

The implementation is split into:
- models.py: Data models and state classes
- diffusion.py: Diffusion system implementation
- reaction.py: Reaction system implementation
- stability.py: Stability analysis
- dynamics.py: Main pattern dynamics implementation
"""

from typing import List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pattern.models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationPoint,
    BifurcationDiagram
)

from .pattern.dynamics import PatternDynamics as _PatternDynamics
from .pattern.diffusion import DiffusionSystem

class PatternDynamics(_PatternDynamics):
    """Pattern dynamics system with attention-specific features."""
    
    def __init__(
        self,
        grid_size: int = 32,
        space_dim: int = 2,
        boundary: str = "periodic",
        dt: float = 0.01,
        hidden_dim: int = 64,
        num_modes: int = 8,
    ):
        """Initialize pattern dynamics system.
        
        Args:
            grid_size (int): Grid size
            space_dim (int): Spatial dimensions
            boundary (str): Boundary conditions
            dt (float): Time step
            hidden_dim (int): Hidden dimension
            num_modes (int): Number of modes
        """
        super().__init__(
            grid_size=grid_size,
            space_dim=space_dim,
            boundary=boundary,
            dt=dt,
            hidden_dim=hidden_dim,
            num_modes=num_modes
        )
        self.diffusion = DiffusionSystem(grid_size=grid_size)
        self.max_concentration = 1.0  # Maximum allowed concentration
        self.min_concentration = 0.0  # Minimum allowed concentration

    def reaction_term(self, state):
        """Compute reaction term for pattern dynamics
        
        Args:
            state (torch.Tensor): Current state tensor
            
        Returns:
            torch.Tensor: Reaction term contribution
        """
        # First clamp state to valid range to avoid instabilities
        state = torch.clamp(state, self.min_concentration, self.max_concentration)
        
        # Compute mass-conserving reaction term
        # Use a bistable reaction term that respects bounds
        mid = (self.max_concentration + self.min_concentration) / 2
        scale = (self.max_concentration - self.min_concentration) / 2
        
        # Normalize state to [-1, 1] range
        x = (state - mid) / scale
        
        # Compute reaction using cubic term
        reaction = x * (1 - x**2)
        
        # Scale back to original range
        reaction = reaction * scale
        
        # Calculate mean to ensure mass conservation
        mean_reaction = reaction.mean(dim=(-2, -1), keepdim=True)
        
        # Subtract mean to make reaction term mass-conserving
        reaction = reaction - mean_reaction
        
        # Ensure bounds are respected
        max_reaction = self.max_concentration - state
        min_reaction = self.min_concentration - state
        reaction = torch.minimum(reaction, max_reaction)
        reaction = torch.maximum(reaction, min_reaction)
        
        return reaction

    def generate_target_pattern(self, batch_size, grid_size, num_channels):
        """Generate target pattern for control
        
        Args:
            batch_size (int): Batch size
            grid_size (int): Grid size
            num_channels (int): Number of channels
            
        Returns:
            torch.Tensor: Target pattern
        """
        # Generate simple target pattern (can be customized)
        target = torch.zeros((batch_size, num_channels, grid_size, grid_size))
        center = grid_size // 2
        radius = grid_size // 4
        for i in range(grid_size):
            for j in range(grid_size):
                dist = ((i - center)**2 + (j - center)**2)**0.5
                if dist < radius:
                    target[:, :, i, j] = 1.0
        return target

    def evolve_spatiotemporal(self, state, diffusion_tensor, num_steps=None, dt=0.1, t_span=None, steps=None):
        """Evolve pattern in space and time
        
        Args:
            state (torch.Tensor): Initial state
            diffusion_tensor (torch.Tensor): Diffusion tensor
            num_steps (int, optional): Number of time steps
            dt (float): Time step size
            t_span (tuple, optional): Time span (t_start, t_end). If provided, overrides num_steps
            steps (int, optional): Alternative way to specify num_steps
            
        Returns:
            torch.Tensor: Time evolution of state
        """
        # Handle different ways of specifying steps
        if steps is not None:
            num_steps = steps
        elif t_span is not None:
            t_start, t_end = t_span
            num_steps = int((t_end - t_start) / dt)
        
        if num_steps is None:
            raise ValueError("Must provide one of: num_steps, t_span, or steps")
            
        evolution = []
        current_state = state
        
        for _ in range(num_steps):
            # Apply diffusion
            diffused = self.diffusion.apply_diffusion(current_state, diffusion_tensor, dt)
            
            # Apply reaction
            reacted = diffused + dt * self.reaction_term(diffused)
            
            # Clip to concentration bounds
            current_state = torch.clamp(reacted, self.min_concentration, self.max_concentration)
            evolution.append(current_state)
            
        return torch.stack(evolution, dim=0)


# Re-export all the public classes and functions
__all__ = [
    'ReactionDiffusionState',
    'StabilityInfo',
    'StabilityMetrics',
    'ControlSignal',
    'BifurcationPoint',
    'BifurcationDiagram',
    'PatternDynamics'
]
