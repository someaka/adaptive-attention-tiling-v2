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
            nn.Linear(input_size, hidden_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size, grid_size * grid_size, dtype=torch.float32)
        )
        
        self.inhibitor_network = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size, grid_size * grid_size, dtype=torch.float32)
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
        
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        
        # Compute reaction terms with improved numerical stability
        activator_term = (activator * activator) / (1.0 + inhibitor + eps) - activator
        inhibitor_term = torch.clamp(activator * activator, min=0.0, max=10.0) - inhibitor
        
        # Scale terms for stability
        scale = 0.1
        activator_term = scale * activator_term
        inhibitor_term = scale * inhibitor_term
        
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
        orig_dtype = state.dtype
        state = state.to(torch.float64)
        
        # Compute reaction term with gradient clipping
        with torch.no_grad():
            try:
                # Try calling with just state (for default reaction term)
                reaction = reaction_term(state)
            except TypeError:
                # If that fails, assume it's a parameterized reaction term
                # that takes both state and parameter
                param = getattr(reaction_term, 'param', None)
                if param is not None:
                    reaction = reaction_term(state, param)
                else:
                    raise ValueError("Reaction term must take either state or (state, param)")
                    
            reaction = torch.clamp(reaction, min=-10.0, max=10.0)
        
        # Add reaction to state
        reacted = state + reaction
        
        # Ensure non-negativity and upper bound
        reacted = torch.clamp(reacted, min=0.0, max=100.0)
        
        return reacted.to(orig_dtype)
    
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
