"""Implementation of reaction dynamics."""

from typing import Optional, Callable
import torch
from torch import nn


class ReactionSystem:
    """Handles reaction terms and dynamics."""

    def __init__(self, grid_size: int):
        """Initialize reaction system.
        
        Args:
            grid_size: Size of square grid
        """
        self.grid_size = grid_size
        
        # Initialize neural networks for reaction terms
        input_size = grid_size * grid_size * 2  # Flattened size for both species
        hidden_size = 64
        
        self.activator_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, grid_size * grid_size)
        )
        
        self.inhibitor_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, grid_size * grid_size)
        )
    
    def reaction_term(self, state: torch.Tensor) -> torch.Tensor:
        """Default reaction term for pattern formation.
        
        This implements a simple activator-inhibitor system with:
            - Autocatalytic production of activator
            - Linear degradation of both species
            - Nonlinear inhibition
            
        Args:
            state: Input tensor [batch, channels, height, width]
            
        Returns:
            Reaction term tensor [batch, channels, height, width]
        """
        # Extract activator and inhibitor
        activator = state[:,0:1]  # [batch, 1, height, width]
        inhibitor = state[:,1:2]  # [batch, 1, height, width]
        
        # Compute reaction terms
        activator_term = activator * activator / (1.0 + inhibitor) - activator
        inhibitor_term = activator * activator - inhibitor
        
        # Combine terms
        return torch.cat([activator_term, inhibitor_term], dim=1)
    
    def apply_reaction(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None
    ) -> torch.Tensor:
        """Apply reaction term to state.
        
        Args:
            state: Input tensor [batch, channels, height, width]
            reaction_term: Optional reaction term function
            
        Returns:
            Reacted tensor [batch, channels, height, width]
        """
        if reaction_term is None:
            reaction_term = self.reaction_term
        
        # Convert to double for numerical stability
        state = state.to(torch.float64)
        
        # Compute reaction term
        reaction = reaction_term(state)
        
        # Add reaction to state
        reacted = state + reaction
        
        # Ensure non-negativity
        reacted = torch.clamp(reacted, min=0.0)
        
        return reacted.to(state.dtype)
    
    def find_reaction_fixed_points(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Find fixed points of the reaction term.
        
        Args:
            state: Input tensor [batch, channels, height, width]
            
        Returns:
            Fixed point tensor [batch, channels, height, width]
        """
        # Convert to double
        state = state.to(torch.float64)
        
        # Initialize fixed point search
        fixed_point = state.clone()
        
        # Iterate until convergence
        max_iter = 100
        tol = 1e-6
        
        for _ in range(max_iter):
            # Compute reaction term
            reaction = self.reaction_term(fixed_point)
            
            # Update fixed point estimate
            new_fixed_point = fixed_point + reaction
            
            # Check convergence
            if (new_fixed_point - fixed_point).abs().max() < tol:
                return new_fixed_point.to(state.dtype)
            
            fixed_point = new_fixed_point
        
        return fixed_point.to(state.dtype)
