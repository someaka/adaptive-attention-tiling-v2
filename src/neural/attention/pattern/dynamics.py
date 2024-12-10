"""Main pattern dynamics implementation."""

from typing import Optional, Union, Tuple, List, Callable
import torch

from .models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationDiagram,
    BifurcationPoint
)
from .diffusion import DiffusionSystem
from .reaction import ReactionSystem
from .stability import StabilityAnalyzer


class PatternDynamics:
    """Complete pattern dynamics system."""

    def __init__(
        self,
        grid_size: int = 32,
        space_dim: int = 2,
        boundary: str = 'periodic',
        dt: float = 0.01,
        num_modes: int = 8,
        hidden_dim: int = 64
    ):
        """Initialize pattern dynamics system.
        
        Args:
            grid_size (int, optional): Size of grid. Defaults to 32.
            space_dim (int, optional): Spatial dimensions. Defaults to 2.
            boundary (str, optional): Boundary conditions. Defaults to 'periodic'.
            dt (float, optional): Time step. Defaults to 0.01.
            num_modes (int, optional): Number of stability modes. Defaults to 8.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 64.
        """
        self.size = grid_size
        self.dim = space_dim
        self.dt = dt
        self.boundary = boundary
        
        # Initialize subsystems
        self.diffusion = DiffusionSystem(self.size)
        self.reaction = ReactionSystem(self.size)
        self.stability = StabilityAnalyzer(self)
        self.stability_analyzer = StabilityAnalyzer(self)
    
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
            state: Initial state tensor
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
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None,
        max_iterations: int = 100,
        dt: float = 0.1
    ) -> torch.Tensor:
        """Compute one step of reaction-diffusion dynamics.
        
        Args:
            state: Current state tensor
            reaction_term: Optional custom reaction term
            max_iterations: Maximum iterations for numerical integration
            dt: Time step for numerical integration
            
        Returns:
            Updated state tensor
        """
        # Default to standard reaction term if none provided
        if reaction_term is None:
            reaction_term = self.reaction.apply_reaction
            
        # Numerical integration with resource bounds
        new_state = state
        with torch.no_grad():
            for _ in range(max_iterations):
                # Compute reaction and diffusion terms
                reaction = reaction_term(new_state)
                diffusion = self.diffusion.apply_diffusion(new_state, 0.1, dt)
                
                # Update state
                delta = reaction + diffusion
                new_state = new_state + dt * delta
                
                # Clamp values to prevent overflow
                new_state = torch.clamp(new_state, min=-1e6, max=1e6)
                
                # Check for numerical stability
                if torch.isnan(new_state).any():
                    return state  # Return original state if unstable
                    
                # Check for convergence
                if torch.max(torch.abs(delta)) < 1e-6:
                    break
                    
        return new_state
    
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
        # Initialize with small random values for better convergence
        state = torch.rand(1, self.dim, self.size, self.size, dtype=torch.float64) * 0.1
        
        max_iter = 100  # Reduced iterations
        tol = 1e-4  # Relaxed tolerance
        min_change = torch.finfo(torch.float64).eps * 100  # Minimum meaningful change
        
        prev_state = state.clone()
        for _ in range(max_iter):
            # Apply strong diffusion with numerical bounds
            state = self.apply_diffusion(state, diffusion_coefficient=1.0, dt=0.1)
            state = torch.clamp(state, min=0.0, max=100.0)
            
            # Check convergence with robust metric
            mean = state.mean(dim=(-2, -1), keepdim=True)
            max_diff = torch.max(torch.abs(state - mean))
            
            # Also check if state has stopped changing
            state_change = torch.max(torch.abs(state - prev_state))
            if max_diff < tol or state_change < min_change:
                break
                
            prev_state = state.clone()
        
        return state.to(torch.float32)  # Return in standard precision

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
        state: torch.Tensor,
        epsilon: float = 1e-6,
        chunk_size: int = 500
    ) -> torch.Tensor:
        """Compute stability matrix for current state.
        
        Args:
            state: Current state tensor
            epsilon: Small perturbation for numerical derivatives
            chunk_size: Size of chunks to process at once
            
        Returns:
            Stability matrix
        """
        # Ensure state is on CPU for stability
        state = state.cpu()
        
        # Get flattened state size
        state_size = state.numel()
        
        # Initialize stability matrix
        stability_matrix = torch.zeros((state_size, state_size), dtype=state.dtype)
        
        # Compute base reaction-diffusion once
        f_minus = self.reaction_diffusion(state).reshape(-1)
        
        # Process in chunks to manage memory
        for chunk_start in range(0, state_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, state_size)
            chunk_size_actual = chunk_end - chunk_start
            
            # Create perturbed states for this chunk efficiently
            perturbed_states = state.repeat(chunk_size_actual, 1, 1, 1)
            indices = torch.arange(chunk_start, chunk_end)
            flat_perturbations = torch.zeros(chunk_size_actual, state_size, dtype=state.dtype)
            flat_perturbations[torch.arange(chunk_size_actual), indices] = epsilon
            perturbations = flat_perturbations.view(-1, *state.shape[1:])
            perturbed_states = perturbed_states + perturbations
            
            # Compute derivatives in batch
            with torch.no_grad():
                f_plus = torch.stack([
                    self.reaction_diffusion(p_state) 
                    for p_state in perturbed_states
                ]).reshape(chunk_size_actual, -1)
                
                # Numerical derivatives for this chunk
                derivatives = (f_plus - f_minus) / epsilon
                
                # Store in stability matrix
                stability_matrix[:, chunk_start:chunk_end] = derivatives.T
                
                # Check for numerical instability
                if torch.isnan(derivatives).any():
                    raise RuntimeError("Numerical instability in stability matrix computation")
        
        return stability_matrix

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
        pattern: torch.Tensor,
        parameterized_reaction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        parameter_range: torch.Tensor,
        max_iter: int = 100,
        max_states: int = 1000,
    ) -> BifurcationDiagram:
        """Analyze bifurcations by varying parameter.
        
        Args:
            pattern (torch.Tensor): Initial pattern state
            parameterized_reaction (Callable): Reaction function with parameter
            parameter_range (torch.Tensor): Range of parameter values to test
            max_iter (int): Maximum iterations per parameter value
            max_states (int): Maximum number of states to track
            
        Returns:
            BifurcationDiagram: Bifurcation analysis results
        """
        solution_states = []
        solution_params = []
        bifurcation_points = []
        
        prev_state = None
        prev_param = None
        prev_stability = None
        
        # Analyze each parameter value
        for param in parameter_range:
            # Define reaction function for this parameter value
            reaction = lambda x: parameterized_reaction(x, param)
            
            # Check stability before simulation
            if not self.stability_analyzer.is_stable(pattern, reaction):
                continue
                
            # Simulate dynamics
            state = pattern.clone()
            for _ in range(max_iter):
                next_state = self.reaction_diffusion(state, reaction)
                
                # Check stability during simulation
                if not self.stability_analyzer.is_stable(next_state, reaction):
                    break
                    
                delta = torch.abs(next_state - state)
                if torch.max(delta) < 1e-6:
                    break
                state = next_state
            
            # Compute stability
            stability = self.stability_analyzer.compute_stability(state)
            
            # Store state if unique
            if len(solution_states) < max_states:
                solution_states.append(state)
                solution_params.append(param)
            
            # Check for bifurcation
            if prev_state is not None:
                state_diff = torch.max(torch.abs(state - prev_state))
                stability_diff = abs(stability - prev_stability)
                
                if state_diff > 0.1 or stability_diff > 0.1:
                    bifurcation_points.append(param)
            
            prev_state = state
            prev_param = param
            prev_stability = stability
        
        return BifurcationDiagram(
            solution_states=torch.stack(solution_states),
            solution_params=torch.stack(solution_params),
            bifurcation_points=torch.tensor(bifurcation_points),
        )

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

    def _classify_bifurcation(
        self,
        point1: BifurcationPoint,
        point2: BifurcationPoint,
        eigenvals: torch.Tensor,
        prev_eigenvals: torch.Tensor,
        stability_change: float,
        state_change: float
    ) -> str:
        """Classify the type of bifurcation between two points."""
        # Check for Hopf bifurcation - complex conjugate pair crossing imaginary axis
        complex_pairs = (eigenvals.imag != 0).sum() - (prev_eigenvals.imag != 0).sum()
        if complex_pairs > 0 and abs(stability_change) > 1e-6:
            # Verify pure imaginary eigenvalues at critical point
            if torch.any(torch.abs(eigenvals.real) < 1e-6):
                # Check for conjugate pair
                if torch.any(torch.abs(eigenvals.imag) > 1e-6):
                    return "hopf"
        
        # Check for pitchfork bifurcation - odd symmetry and eigenvalue crossing
        if abs(stability_change) > 1e-6 and state_change > 1e-6:
            # Check for single eigenvalue crossing zero
            real_crossings = torch.sum(
                (eigenvals.real * prev_eigenvals.real < 0) & 
                (eigenvals.imag == 0)
            )
            if real_crossings == 1:
                # Check for cubic-like nonlinearity
                state_diff = point2.state - point1.state
                param_diff = point2.parameter - point1.parameter
                if param_diff > 0:
                    # Look for characteristic shape of pitchfork
                    if torch.all(state_diff * torch.sign(point1.state) < 0):
                        return "pitchfork"
        
        # Check for saddle-node bifurcation - real eigenvalue crossing zero
        if abs(stability_change) > 1e-6:
            # Verify single real eigenvalue crossing
            real_crossings = torch.sum(
                (eigenvals.real * prev_eigenvals.real < 0) & 
                (eigenvals.imag == 0)
            )
            if real_crossings == 1:
                # Verify no symmetry breaking
                if not torch.allclose(point1.state, -point2.state, rtol=1e-5):
                    return "saddle-node"
        
        return "unknown"
