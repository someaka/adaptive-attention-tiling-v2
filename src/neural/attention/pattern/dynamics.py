"""Main pattern dynamics implementation."""

from typing import Optional, Union, Tuple, List, Callable
import torch

from .models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal
)
from .diffusion import DiffusionSystem
from .reaction import ReactionSystem
from .stability import StabilityAnalyzer


class PatternDynamics:
    """Complete pattern dynamics system."""

    def __init__(
        self,
        dim: int,
        size: int,
        dt: float = 0.01,
        boundary: str = "periodic",
        hidden_dim: int = 64,
        num_modes: int = 8,
    ):
        """Initialize pattern dynamics system.
        
        Args:
            dim: Number of channels/species
            size: Grid size
            dt: Time step
            boundary: Boundary condition type
            hidden_dim: Hidden layer dimension
            num_modes: Number of stability modes
        """
        self.dim = dim
        self.size = size
        self.dt = dt
        self.boundary = boundary
        
        # Initialize subsystems
        self.diffusion = DiffusionSystem(size)
        self.reaction = ReactionSystem(size)
        self.stability = StabilityAnalyzer(
            input_dim=dim * size * size,
            num_modes=num_modes,
            hidden_dim=hidden_dim
        )
    
    def apply_diffusion(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Apply diffusion to state.
        
        Args:
            state: Input state tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient
            dt: Optional time step override
            
        Returns:
            Diffused state tensor
        """
        dt = dt if dt is not None else self.dt
        return self.diffusion.apply_diffusion(state, diffusion_coefficient, dt)
        
    def apply_reaction(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Apply reaction to state.
        
        Args:
            state: Input state tensor [batch, channels, height, width]
            reaction_term: Optional reaction term function
            dt: Optional time step override
            
        Returns:
            Reacted state tensor
        """
        dt = dt if dt is not None else self.dt
        return self.reaction.apply_reaction(state, reaction_term, dt)
        
    def evolve_pattern(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float,
        steps: int = 100,
        reaction_term: Optional[Callable] = None,
        dt: Optional[float] = None
    ) -> List[torch.Tensor]:
        """Evolve pattern with reaction-diffusion dynamics.
        
        Args:
            state: Initial state tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient
            steps: Number of evolution steps
            reaction_term: Optional reaction term function
            dt: Optional time step override
            
        Returns:
            List of evolved states
        """
        dt = dt if dt is not None else self.dt
        evolution = [state]
        
        for _ in range(steps):
            # Apply diffusion
            diffused = self.apply_diffusion(state, diffusion_coefficient, dt)
            
            # Apply reaction if provided
            if reaction_term is not None:
                diffused = self.apply_reaction(diffused, reaction_term, dt)
                
            evolution.append(diffused)
            state = diffused
            
        return evolution
    
    def test_convergence(
        self,
        state: torch.Tensor,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> bool:
        """Test if pattern has converged to steady state.
        
        Args:
            state: Input state tensor
            max_iter: Maximum iterations to test
            tol: Convergence tolerance
            
        Returns:
            True if converged, False otherwise
        """
        # Convert to double for stability
        state = state.to(torch.float64)
        prev_state = state
        
        for _ in range(max_iter):
            # Apply diffusion with small time step
            curr_state = self.apply_diffusion(prev_state, 0.1, 0.01)
            
            # Check convergence
            diff = torch.abs(curr_state - prev_state).max()
            if diff < tol:
                return True
                
            prev_state = curr_state
            
        return False
    
    def reaction_diffusion(
        self,
        state: Optional[Union[ReactionDiffusionState, torch.Tensor]] = None,
        diffusion_tensor: Optional[torch.Tensor] = None,
        reaction_term: Optional[Callable] = None,
        *,
        batch_size: Optional[Union[int, torch.Tensor]] = None,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evolve reaction-diffusion system."""
        # Handle default arguments
        if grid_size is None:
            grid_size = self.size
        
        if batch_size is None:
            batch_size = 1
        
        if diffusion_tensor is None:
            diffusion_tensor = torch.eye(self.dim)
        
        # Initialize state if needed
        if state is None:
            state = torch.rand(batch_size, self.dim, grid_size, grid_size)
        elif isinstance(state, ReactionDiffusionState):
            state = torch.cat([state.activator, state.inhibitor], dim=1)
        
        # Apply reaction-diffusion step
        diffused = self.apply_diffusion(state, diffusion_tensor[0,0].item(), self.dt)
        reacted = self.apply_reaction(diffused, reaction_term)
        
        return reacted
    
    def stability_analysis(
        self,
        fixed_point: Union[ReactionDiffusionState, torch.Tensor],
        perturbation: torch.Tensor,
    ) -> StabilityMetrics:
        """Analyze stability around fixed point."""
        return self.stability.analyze_stability(fixed_point, perturbation)
    
    def find_reaction_fixed_points(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Find fixed points of the reaction term."""
        return self.reaction.find_reaction_fixed_points(state)
