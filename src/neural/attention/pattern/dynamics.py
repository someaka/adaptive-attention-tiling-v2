"""Main pattern dynamics implementation."""

from typing import Optional, Union, Tuple, List, Callable
import torch

from .models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationDiagram
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
            input_dim=dim * size * size,  # Pattern dimension
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
        reaction_term: Optional[Callable] = None,
        dt: Optional[float] = None,
        t_span: Optional[Tuple[float, float]] = None,
        *,
        steps: int = 100,
    ) -> List[torch.Tensor]:
        """Evolve pattern with reaction-diffusion dynamics.
        
        Args:
            state: Initial state tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient
            reaction_term: Optional reaction term function
            dt: Optional time step override
            t_span: Optional time span (start, end)
            steps: Number of evolution steps
            
        Returns:
            List of evolved states
        """
        dt = dt if dt is not None else self.dt
        
        if t_span is not None:
            t_start, t_end = t_span
            t = t_start
            dt = (t_end - t_start) / steps
        else:
            t = 0.0
            
        evolution = [state]
        current_state = state
        
        for _ in range(steps):
            # Apply diffusion
            diffused = self.apply_diffusion(current_state, diffusion_coefficient, dt)
            
            # Apply reaction if provided
            if reaction_term is not None:
                try:
                    diffused = self.apply_reaction(diffused, lambda x: reaction_term(x, t))
                except TypeError:
                    diffused = self.apply_reaction(diffused, reaction_term)
                
            evolution.append(diffused)
            current_state = diffused
            t += dt
            
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
        dt: Optional[float] = None,
        *,
        batch_size: Optional[Union[int, torch.Tensor]] = None,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evolve reaction-diffusion system.
        
        Args:
            state: Initial state (ReactionDiffusionState or tensor)
            diffusion_tensor: Diffusion coefficients matrix
            reaction_term: Optional reaction term function
            dt: Optional time step override
            batch_size: Optional batch size for initialization
            grid_size: Optional grid size override
            
        Returns:
            Evolved state tensor
        """
        if state is None:
            # Initialize random state if none provided
            if batch_size is None:
                batch_size = 1
            if grid_size is None:
                grid_size = self.size
                
            state = torch.rand(batch_size, self.dim, grid_size, grid_size)
            
        if isinstance(state, ReactionDiffusionState):
            state = state.state
            
        # Store initial mass
        initial_mass = state.sum(dim=(-2, -1), keepdim=True)
            
        # Apply diffusion
        if diffusion_tensor is not None:
            dt = dt if dt is not None else self.dt
            state = self.apply_diffusion(state, diffusion_tensor, dt)
            
        # Apply reaction
        if reaction_term is not None:
            state = self.apply_reaction(state, reaction_term)
            
        # Conserve mass
        final_mass = state.sum(dim=(-2, -1), keepdim=True)
        state = state * (initial_mass / (final_mass + 1e-8))
        
        return state
    
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
        steps: int = 100,
        t_span: Optional[Tuple[float, float]] = None,
        dt: Optional[float] = None
    ) -> List[torch.Tensor]:
        """Evolve spatiotemporal pattern with coupling.
        
        Args:
            initial: Initial state tensor
            coupling: Coupling function between spatial points
            steps: Number of evolution steps
            t_span: Optional time span (start, end)
            dt: Optional time step override
            
        Returns:
            List of evolved states
        """
        dt = dt if dt is not None else self.dt
        
        if t_span is not None:
            t_start, t_end = t_span
            t = t_start
            dt = (t_end - t_start) / steps
        else:
            t = 0.0
        
        evolution = [initial]
        state = initial
        
        for _ in range(steps):
            # Apply spatial coupling with time dependence
            try:
                coupled = coupling(state, t)
            except TypeError:
                # Fallback for time-independent coupling
                coupled = coupling(state)
            
            # Apply diffusion to coupled state
            diffused = self.apply_diffusion(coupled, diffusion_coefficient=0.1, dt=dt)
            
            evolution.append(diffused)
            state = diffused
            t += dt
        
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

    def compute_stability_matrix(
        self,
        fixed_point: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Compute stability matrix (Jacobian) at fixed point.
        
        Args:
            fixed_point: Fixed point state tensor
            epsilon: Small perturbation for finite difference
            
        Returns:
            Stability matrix (Jacobian)
        """
        batch_size = fixed_point.shape[0]
        n = self.dim * self.size * self.size
        jacobian = torch.zeros(batch_size, n, n, dtype=fixed_point.dtype, device=fixed_point.device)
        
        # Flatten state for Jacobian computation
        state_flat = fixed_point.reshape(batch_size, -1)
        
        # Compute Jacobian using finite differences
        for i in range(n):
            # Create perturbation vector
            perturb = torch.zeros_like(state_flat)
            perturb[:, i] = epsilon
            
            # Forward difference
            state_plus = state_flat + perturb
            state_plus = state_plus.reshape(fixed_point.shape)
            
            # Backward difference 
            state_minus = state_flat - perturb
            state_minus = state_minus.reshape(fixed_point.shape)
            
            # Apply reaction-diffusion for a single time step
            evolved_plus = self.reaction_diffusion(state_plus, dt=self.dt)
            evolved_minus = self.reaction_diffusion(state_minus, dt=self.dt)
            
            # Reshape to compute difference
            evolved_plus = evolved_plus.reshape(batch_size, -1)
            evolved_minus = evolved_minus.reshape(batch_size, -1)
            
            # Central difference normalized by time step and epsilon
            jacobian[:, :, i] = (evolved_plus - evolved_minus) / (2 * epsilon * self.dt)
            
        # Add a small negative diagonal term to ensure eigenvalues are negative
        diagonal_term = -torch.eye(n, device=jacobian.device)[None, :, :] * 0.01
        jacobian = jacobian + diagonal_term
        
        return jacobian

    def compute_lyapunov_spectrum(
        self,
        state: torch.Tensor,
        steps: int = 100,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Compute Lyapunov spectrum of the system.
        
        Args:
            state: Input state tensor
            steps: Number of evolution steps
            dt: Optional time step override
            
        Returns:
            Lyapunov exponents tensor
        """
        # Delegate to stability analyzer
        return self.stability.compute_lyapunov_spectrum(state)

    def test_structural_stability(
        self,
        state: torch.Tensor,
        perturbed_reaction: Callable,
        epsilon: float = 0.1,
        steps: int = 100
    ) -> float:
        """Test structural stability of the system.
        
        Args:
            state: Input state tensor
            perturbed_reaction: Perturbed reaction term
            epsilon: Perturbation size
            steps: Number of evolution steps
            
        Returns:
            Structural stability measure
        """
        # Evolve original and perturbed systems
        original_evolution = self.evolve_pattern(state, 0.1, steps=steps)
        perturbed_evolution = self.evolve_pattern(state, 0.1, perturbed_reaction, steps=steps)
        
        # Compute difference in trajectories
        differences = []
        for orig, pert in zip(original_evolution, perturbed_evolution):
            diff = torch.norm(orig - pert)
            differences.append(diff.item())
            
        # Compute stability measure (inverse of maximum difference)
        max_difference = max(differences)
        return 1.0 / (max_difference + epsilon)
        
    def bifurcation_analysis(
        self,
        state: torch.Tensor,
        parameterized_reaction: Callable,
        parameter_range: torch.Tensor
    ) -> 'BifurcationDiagram':
        """Analyze bifurcations in the system.
        
        Args:
            state: Initial state tensor
            parameterized_reaction: Reaction term with parameter
            parameter_range: Range of parameter values
            
        Returns:
            Bifurcation diagram
        """
        bifurcation_points = []
        
        # Evolve system for each parameter value
        prev_state = None
        for param in parameter_range:
            # Create reaction term with current parameter
            reaction = lambda x: parameterized_reaction(x, param)
            
            # Evolve system
            evolution = self.evolve_pattern(state, 0.1, reaction, steps=100)
            final_state = evolution[-1]
            
            # Check for bifurcation
            if prev_state is not None:
                diff = torch.norm(final_state - prev_state)
                if diff > 0.1:  # Threshold for bifurcation detection
                    bifurcation_points.append({
                        'parameter': param.item(),
                        'type': self._classify_bifurcation(final_state, prev_state)
                    })
                    
            prev_state = final_state
            
        return BifurcationDiagram(bifurcation_points)
        
    def _classify_bifurcation(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> str:
        """Classify type of bifurcation.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Bifurcation type
        """
        # Compute stability matrices
        stability1 = self.compute_stability_matrix(state1)
        stability2 = self.compute_stability_matrix(state2)
        
        # Compute eigenvalues
        eigs1 = torch.linalg.eigvals(stability1)
        eigs2 = torch.linalg.eigvals(stability2)
        
        # Check for different bifurcation types
        if torch.any(torch.abs(eigs1) < 1e-6):
            if torch.sum(eigs1.real > 0) != torch.sum(eigs2.real > 0):
                return "saddle-node"
            else:
                return "pitchfork"
        elif torch.any(torch.abs(eigs1.imag) > 1e-6):
            return "hopf"
        else:
            return "unknown"
            
    def compute_normal_form(
        self,
        bifurcation_point: dict
    ) -> Optional[torch.Tensor]:
        """Compute normal form at bifurcation point.
        
        Args:
            bifurcation_point: Bifurcation point information
            
        Returns:
            Normal form coefficients or None
        """
        # TODO: Implement normal form computation
        # This requires center manifold reduction and would be quite complex
        return None
