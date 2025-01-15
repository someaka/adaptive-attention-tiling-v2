"""Implementation of reaction dynamics."""

from typing import Optional, Callable, Protocol
import torch
from torch import nn
from functools import partial
import logging


class ReactionTerm(Protocol):
    """Protocol for reaction terms."""
    def __call__(self, state: torch.Tensor) -> torch.Tensor: ...


class ReactionSystem(nn.Module):
    """Handles reaction terms and dynamics."""

    def __init__(self, grid_size: int):
        """Initialize reaction system.
        
        Args:
            grid_size: Size of square grid
        """
        super().__init__()
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
    
    def reaction_term(self, state: torch.Tensor, should_log: bool = False) -> torch.Tensor:
        """Default reaction term for pattern formation."""
        # Extract activator and inhibitor
        activator = state[:,0:1]  # [batch, 1, height, width]
        inhibitor = state[:,1:2]  # [batch, 1, height, width]
        
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        
        # Compute reaction terms with improved numerical stability
        activator_squared = activator * activator
        denominator = 1.0 + inhibitor + eps
        
        activator_term = (activator_squared) / denominator - activator
        inhibitor_term = torch.clamp(activator_squared, min=0.0, max=10.0) - inhibitor
        
        # Scale terms for stability
        scale = 0.1
        activator_term = scale * activator_term
        inhibitor_term = scale * inhibitor_term
        
        # Combine terms
        result = torch.cat([activator_term, inhibitor_term], dim=1)
        
        # Log key metrics if requested
        if should_log:
            logging.info(f"Reaction calculation:")
            logging.info(f"  - Activator term - mean: {activator_term.mean():.6f}, std: {activator_term.std():.6f}")
            logging.info(f"  - Inhibitor term - mean: {inhibitor_term.mean():.6f}, std: {inhibitor_term.std():.6f}")
            logging.info(f"  - Combined result - mean: {result.mean():.6f}, std: {result.std():.6f}, norm: {torch.norm(result):.6f}")
        
        return result

    def _wrap_reaction_term(self, reaction_term: Callable, state: torch.Tensor) -> torch.Tensor:
        """Wrap reaction term to handle parameterized functions.
        
        Args:
            reaction_term: Reaction term function
            state: Input state tensor
            
        Returns:
            Reaction term output
        """
        try:
            # Try calling with just state
            return reaction_term(state)
        except TypeError:
            # If that fails, check for param attribute
            param = getattr(reaction_term, 'param', None)
            if param is not None:
                # Create a partial function with the param
                wrapped_term = partial(reaction_term, param=param)
                return wrapped_term(state)
            else:
                raise ValueError("Reaction term must take either state or have param attribute")
    
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
            reaction = self._wrap_reaction_term(reaction_term, state)
            reaction = torch.clamp(reaction, min=-10.0, max=10.0)
        
        # Add reaction to state
        reacted = state + reaction
        
        # Ensure non-negativity and upper bound
        reacted = torch.clamp(reacted, min=0.0, max=100.0)
        
        return reacted.to(orig_dtype)
    
    def find_reaction_fixed_points(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None
    ) -> torch.Tensor:
        """Find fixed points of the reaction term.
        
        Args:
            state: Input tensor [batch, channels, height, width]
            reaction_term: Optional custom reaction term function
            
        Returns:
            Fixed point tensor [batch, channels, height, width]
        """
        if reaction_term is None:
            reaction_term = self.reaction_term

        # Convert to double
        state = state.to(torch.float64)
        
        # Initialize fixed point search
        fixed_point = state.clone()
        
        # Iterate until convergence
        max_iter = 100
        tol = 1e-6
        
        for _ in range(max_iter):
            # Compute reaction term with proper parameter handling
            reaction = self._wrap_reaction_term(reaction_term, fixed_point)
            
            # Update fixed point estimate
            new_fixed_point = fixed_point + reaction
            
            # Check convergence
            if (new_fixed_point - fixed_point).abs().max() < tol:
                return new_fixed_point.to(state.dtype)
            
            fixed_point = new_fixed_point
        
        return fixed_point.to(state.dtype)

    def compute_reaction(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None
    ) -> torch.Tensor:
        """Compute reaction term for a given state.
        
        This method computes the reaction term without applying it to the state.
        For applying the reaction term, use apply_reaction instead.
        
        Args:
            state: Input tensor [batch, channels, height, width]
            reaction_term: Optional custom reaction term function
            
        Returns:
            Reaction term tensor [batch, channels, height, width]
        """
        if reaction_term is None:
            reaction_term = self.reaction_term
        
        # Convert to double for numerical stability
        orig_dtype = state.dtype
        state = state.to(torch.float64)
        
        # Compute reaction term with gradient clipping
        with torch.no_grad():
            reaction = self._wrap_reaction_term(reaction_term, state)
            reaction = torch.clamp(reaction, min=-10.0, max=10.0)
        
        return reaction.to(orig_dtype)
