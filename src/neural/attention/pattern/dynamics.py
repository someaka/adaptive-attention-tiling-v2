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
        
        # Convert to float64 for numerical stability
        orig_dtype = state.dtype
        state = state.to(torch.float64)
        
        # Apply diffusion with numerical stability
        diffused = self.diffusion.apply_diffusion(state, diffusion_coefficient, dt)
        
        # Ensure non-negative values (if applicable)
        if torch.all(state >= 0):
            diffused = torch.clamp(diffused, min=0.0)
            
        # Convert back to original dtype
        return diffused.to(orig_dtype)
        
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
            dt: Optional time step override (not used in base implementation)
            
        Returns:
            Reacted state tensor
        """
        # Convert to float64 for numerical stability
        orig_dtype = state.dtype
        state = state.to(torch.float64)
        
        # Apply reaction with numerical stability
        reacted = self.reaction.apply_reaction(state, reaction_term)
        
        # Ensure non-negative values (if applicable)
        if torch.all(state >= 0):
            reacted = torch.clamp(reacted, min=0.0)
            
        # Convert back to original dtype
        return reacted.to(orig_dtype)
        
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
                diffused = self.apply_reaction(diffused, reaction_term)
                
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
        """Evolve reaction-diffusion system.
        
        Args:
            state: Initial state (ReactionDiffusionState or tensor)
            diffusion_tensor: Diffusion coefficients matrix
            reaction_term: Optional reaction term function
            batch_size: Optional batch size for initialization
            grid_size: Optional grid size override
            
        Returns:
            Evolved state tensor
        """
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
        
        # Extract species and apply diffusion
        evolved = []
        for i in range(self.dim):
            species = state[:,i:i+1]  # Keep channel dimension
            diffused = self.apply_diffusion(
                species,
                diffusion_coefficient=diffusion_tensor[i,i].item(),
                dt=self.dt
            )
            evolved.append(diffused)
        
        # Combine species and apply reaction
        evolved = torch.cat(evolved, dim=1)
        if reaction_term is not None:
            evolved = self.apply_reaction(evolved, reaction_term)
        
        return evolved
    
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

    def find_homogeneous_state(self) -> torch.Tensor:
        """Find homogeneous steady state.
        
        Returns:
            Homogeneous state tensor
        """
        # Initialize with uniform random state
        state = torch.rand(1, self.dim, self.size, self.size)
        
        # Apply diffusion until convergence
        max_iter = 1000
        tol = 1e-6
        
        for _ in range(max_iter):
            # Apply strong diffusion to homogenize
            state = self.apply_diffusion(state, diffusion_coefficient=1.0, dt=0.1)
            
            # Check if state is uniform
            mean = state.mean(dim=(-2, -1), keepdim=True)
            if torch.max(torch.abs(state - mean)) < tol:
                return state
        
        return state

    def pattern_control(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        constraints: List[Callable]
    ) -> ControlSignal:
        """Compute control signal to drive system towards target state.
        
        Args:
            current: Current state tensor
            target: Target state tensor
            constraints: List of constraint functions
            
        Returns:
            Control signal
        """
        # Initialize control signal
        control = torch.zeros_like(current)
        
        # Compute error
        error = target - current
        
        # Scale control by error and constraints
        for constraint in constraints:
            # Evaluate constraint
            violation = constraint(current)
            
            # Add gradient-based correction
            if violation > 0:
                # Use autograd to get constraint gradient
                current.requires_grad_(True)
                c = constraint(current)
                c.backward()
                grad = current.grad
                current.requires_grad_(False)
                
                # Update control to reduce violation
                control = control - 0.1 * violation * grad
        
        # Add error correction
        control = control + 0.1 * error
        
        return ControlSignal(control)

    def evolve_spatiotemporal(
        self,
        initial: torch.Tensor,
        coupling: Callable,
        steps: int = 100
    ) -> List[torch.Tensor]:
        """Evolve spatiotemporal pattern with coupling.
        
        Args:
            initial: Initial state tensor
            coupling: Coupling function between spatial points
            steps: Number of evolution steps
            
        Returns:
            List of evolved states
        """
        evolution = [initial]
        state = initial
        
        for _ in range(steps):
            # Apply spatial coupling
            coupled = coupling(state)
            
            # Apply diffusion to coupled state
            diffused = self.apply_diffusion(coupled, diffusion_coefficient=0.1)
            
            evolution.append(diffused)
            state = diffused
        
        return evolution

    def detect_pattern_formation(
        self,
        evolution: List[torch.Tensor]
    ) -> bool:
        """Detect if pattern formation occurred.
        
        Args:
            evolution: List of states from time evolution
            
        Returns:
            True if stable pattern formed
        """
        if len(evolution) < 2:
            return False
            
        # Check if final states are similar (stable pattern)
        final_states = evolution[-10:]
        if len(final_states) < 2:
            return False
            
        # Compute changes between consecutive states
        changes = []
        for i in range(len(final_states)-1):
            diff = torch.abs(final_states[i+1] - final_states[i]).mean()
            changes.append(diff.item())
            
        # Pattern formed if changes are small and consistent
        mean_change = sum(changes) / len(changes)
        return mean_change < 1e-3
