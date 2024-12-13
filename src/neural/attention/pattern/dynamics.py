"""Main pattern dynamics implementation."""

from typing import Optional, Union, Tuple, List, Callable
import torch
import numpy as np

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

    def compute_next_state(self, state: torch.Tensor) -> torch.Tensor:
        """Perform one step of pattern dynamics.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            torch.Tensor: Next state
        """
        # Apply reaction and diffusion
        reaction = self.reaction.reaction_term(state)
        diffusion = self.diffusion.apply_diffusion(state, diffusion_coefficient=0.1, dt=self.dt)
        
        # Combine using timestep
        next_state = state + self.dt * (reaction + diffusion)
        
        return next_state

    def evolve_pattern(
        self,
        pattern: torch.Tensor,
        diffusion_coefficient: float = 0.1,
        reaction_term: Optional[Callable] = None,
        steps: int = 100
    ) -> List[torch.Tensor]:
        """Evolve pattern forward in time.
        
        Args:
            pattern: Initial pattern
            diffusion_coefficient: Diffusion coefficient
            reaction_term: Optional custom reaction term function
            steps: Number of timesteps
            
        Returns:
            List of evolved patterns
        """
        trajectory = []
        current = pattern
        
        for _ in range(steps):
            trajectory.append(current)
            # Apply reaction and diffusion with optional custom reaction term
            if reaction_term is not None:
                reaction = reaction_term(current)
            else:
                reaction = self.reaction.reaction_term(current)
            diffusion = self.diffusion.apply_diffusion(current, diffusion_coefficient=diffusion_coefficient, dt=self.dt)
            current = current + self.dt * (reaction + diffusion)
            
        return trajectory
        
    def compute_jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrix of dynamics.
        
        Args:
            state: Current state
            
        Returns:
            Jacobian matrix
        """
        return self.compute_stability_matrix(state, epsilon=1e-6, chunk_size=10)
        
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
        # Get state shape and ensure proper dimensions
        if len(state.shape) == 4:  # [batch, channels, height, width]
            batch_size = state.shape[0]
            n = state.shape[1] * state.shape[2] * state.shape[3]
            state = state.view(batch_size, n)
        else:  # [channels, height, width]
            batch_size = 1
            n = state.numel()
            state = state.view(1, n)
        
        # Initialize Jacobian efficiently
        J = torch.zeros((n, n), dtype=state.dtype, device=state.device)
        
        # Use vectorized operations for efficiency
        for i in range(0, n, chunk_size):  # Process in chunks for memory efficiency
            end_idx = min(i + chunk_size, n)
            curr_chunk_size = end_idx - i
            
            # Create perturbations for this chunk
            perturb = torch.zeros((curr_chunk_size, n), dtype=state.dtype, device=state.device)
            perturb[range(curr_chunk_size), range(i, end_idx)] = epsilon
            
            # Forward differences (vectorized)
            states_plus = (state + perturb).view(-1, self.dim, self.size, self.size)
            forward = self.compute_next_state(states_plus).view(curr_chunk_size, n)
            
            # Backward differences (vectorized)
            states_minus = (state - perturb).view(-1, self.dim, self.size, self.size)
            backward = self.compute_next_state(states_minus).view(curr_chunk_size, n)
            
            # Central differences
            J[i:end_idx] = (forward - backward) / (2 * epsilon)
        
        return J

    def compute_eigenvalues(self, state: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalues of linearized dynamics.
        
        Args:
            state (torch.Tensor): State to compute eigenvalues at
            
        Returns:
            torch.Tensor: Eigenvalues
        """
        # Compute Jacobian efficiently
        jacobian = self.compute_jacobian(state)
        
        # Use a more efficient eigenvalue computation method
        try:
            # Try using eigvals first with GPU if available
            if torch.cuda.is_available():
                jacobian = jacobian.cuda()
                eigenvalues = torch.linalg.eigvals(jacobian)
                eigenvalues = eigenvalues.cpu()
            else:
                # Use numpy for CPU computation which is generally faster
                eigenvalues = torch.from_numpy(
                    np.linalg.eigvals(jacobian.numpy())
                ).to(torch.complex64)
        except Exception:
            # Fallback to a more stable but slower method
            eigenvalues, _ = torch.linalg.eig(jacobian)
            
        return eigenvalues

    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute energy of pattern state.
        
        Args:
            state (torch.Tensor): Pattern state
            
        Returns:
            torch.Tensor: Energy value
        """
        # Ensure proper dimensions
        if len(state.shape) == 4:  # [batch, channels, height, width]
            state = state.view(state.shape[0], -1)
        else:  # [channels, height, width]
            state = state.view(1, -1)
        
        # Compute kinetic energy efficiently
        kinetic = 0.5 * torch.sum(state * state, dim=1)
        
        # Compute field terms
        reaction = self.reaction.reaction_term(state.view(-1, self.dim, self.size, self.size))
        diffusion = self.diffusion.apply_diffusion(
            state.view(-1, self.dim, self.size, self.size),
            diffusion_coefficient=0.1,
            dt=self.dt
        )
        
        # Reshape fields
        reaction = reaction.view(state.shape)
        diffusion = diffusion.view(state.shape)
        
        # Compute potential energy efficiently
        potential = -0.5 * torch.sum(state * (reaction + diffusion), dim=1)
        
        # Return total energy (averaged over batch if needed)
        total_energy = kinetic + potential
        if total_energy.shape[0] > 1:
            return total_energy.mean()
        return total_energy[0]

    def is_stable(self, state: torch.Tensor, threshold: float = 0.1) -> bool:
        """Check if a state is stable.
        
        Args:
            state (torch.Tensor): State to check stability for
            threshold (float): Stability threshold
            
        Returns:
            bool: True if state is stable
        """
        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues(state)
        
        # Check if all real parts are negative
        max_real = torch.max(eigenvalues.real)
        
        return bool(max_real < threshold)

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
        return bool(mean_change < 1e-3)

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
        # Delegate to stability analyzer with the provided parameters
        return self.stability.compute_lyapunov_spectrum(state, steps=steps)

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
        original_evolution = self.evolve_pattern(state, diffusion_coefficient=0.1, steps=steps)
        perturbed_evolution = self.evolve_pattern(
            state,
            diffusion_coefficient=0.1,
            reaction_term=perturbed_reaction,
            steps=steps
        )
        
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
        prev_eigenvals = None
        
        # Track stability changes
        stability_history = []
        
        # Analyze each parameter value
        for param in parameter_range:
            # Define reaction function for this parameter value
            reaction = lambda x: parameterized_reaction(x, param)
            
            # Simulate dynamics with smaller time step near zero
            state = pattern.clone()
            converged = False
            dt = 0.1 if isinstance(param, float) else min(0.1, float(torch.abs(param).item()) + 0.01)  # Smaller dt near bifurcation
            
            for _ in range(max_iter):
                next_state = self.reaction_diffusion(state, reaction, dt=dt)
                
                # Check convergence with relaxed threshold
                delta = torch.abs(next_state - state).mean()
                if delta < 1e-4:  # Relaxed convergence threshold
                    converged = True
                    break
                    
                state = next_state
            
            # Only analyze converged states
            if converged:
                # Compute stability matrix and eigenvalues
                stability_matrix = self.compute_stability_matrix(state, epsilon=1e-4)
                eigenvals = torch.linalg.eigvals(stability_matrix)
                
                # Store state
                if len(solution_states) < max_states:
                    solution_states.append(state)
                    solution_params.append(param)
                
                # Check for bifurcation using eigenvalue analysis
                if prev_eigenvals is not None:
                    # 1. Check for eigenvalue crossing zero (stability change)
                    curr_stable = torch.all(eigenvals.real < 0)
                    prev_stable = torch.all(prev_eigenvals.real < 0)
                    stability_change = curr_stable != prev_stable
                    
                    # 2. Check for new complex eigenvalues (Hopf bifurcation)
                    curr_complex = torch.any(torch.abs(eigenvals.imag) > 1e-4)
                    prev_complex = torch.any(torch.abs(prev_eigenvals.imag) > 1e-4)
                    hopf_change = curr_complex != prev_complex
                    
                    # 3. Check for state changes
                    if prev_state is not None:
                        state_diff = torch.max(torch.abs(state - prev_state))
                        state_change = state_diff > 0.01  # More sensitive threshold
                    else:
                        state_change = False
                    
                    # 4. Check for eigenvalue magnitude changes
                    eigenval_diff = torch.max(torch.abs(eigenvals - prev_eigenvals))
                    eigenval_change = eigenval_diff > 0.01
                    
                    # Detect bifurcation if any criteria met
                    if stability_change or hopf_change or state_change or eigenval_change:
                        bifurcation_points.append(param)
                
                prev_state = state.clone()
                prev_param = param
                prev_eigenvals = eigenvals
        
        # Convert lists to tensors
        solution_states = torch.stack(solution_states) if solution_states else torch.tensor([])
        solution_params = torch.tensor(solution_params) if solution_params else torch.tensor([])
        bifurcation_points = torch.tensor(bifurcation_points) if bifurcation_points else torch.tensor([])
        
        return BifurcationDiagram(
            solution_states=solution_states,
            solution_params=solution_params,
            bifurcation_points=bifurcation_points
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

    def step(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float = 0.1,
        reaction_term: Optional[Callable] = None
    ) -> torch.Tensor:
        """Take a single timestep in pattern evolution.
        
        Args:
            state: Current state tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient
            reaction_term: Optional reaction term function
            
        Returns:
            Next state tensor
        """
        # Apply diffusion
        next_state = self.apply_diffusion(state, diffusion_coefficient, self.dt)
        
        # Apply reaction if provided
        if reaction_term is not None:
            next_state = self.apply_reaction(next_state, reaction_term)
            
        return next_state

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
        # Ensure state has batch dimension
        if len(state.shape) == 3:
            state = state.unsqueeze(0)  # Add batch dimension
            
        # Apply reaction term
        if reaction_term is not None:
            reaction = reaction_term(state)
        else:
            reaction = self.reaction.reaction_term(state)
            
        # Scale up reaction term for stronger dynamics
        reaction = reaction * 200.0  # Increased scaling
            
        # Apply diffusion with proper shape
        diffused = self.diffusion.apply_diffusion(state, 0.5, dt)  # Increased diffusion coefficient
        
        # Combine reaction and diffusion with proper time step
        updated = state + dt * (reaction + diffused)
        
        # Clamp with wider bounds
        updated = torch.clamp(updated, min=-1000.0, max=1000.0)
        
        # Remove batch dimension if it was added
        if len(state.shape) == 4 and state.shape[0] == 1:
            updated = updated.squeeze(0)
            
        return updated
    
    def stability_analysis(
        self,
        fixed_point: Union[ReactionDiffusionState, torch.Tensor],
        perturbation: torch.Tensor,
    ) -> StabilityMetrics:
        """Analyze stability around fixed point.
        
        Args:
            fixed_point: Fixed point state (either ReactionDiffusionState or Tensor)
            perturbation: Perturbation tensor
            
        Returns:
            StabilityMetrics: Stability analysis results
        """
        # Convert ReactionDiffusionState to tensor if needed
        if isinstance(fixed_point, ReactionDiffusionState):
            state_tensor = torch.cat([fixed_point.activator, fixed_point.inhibitor], dim=1)
        else:
            state_tensor = fixed_point
            
        return self.stability.analyze_stability(state_tensor, perturbation)
    
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

    def compute_linearization(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute linearized dynamics matrix.
        
        Args:
            pattern: Input pattern tensor
            
        Returns:
            Linearized dynamics matrix
        """
        # Get current state
        state = pattern.detach().requires_grad_(True)
        
        # Compute dynamics
        dynamics = self.compute_next_state(state)
        
        # Compute Jacobian matrix
        jacobian = torch.autograd.functional.jacobian(self.compute_next_state, state)
        
        # Convert to tensor and reshape to matrix form
        if isinstance(jacobian, tuple):
            jacobian = torch.stack(list(jacobian))
        batch_size = pattern.size(0)
        state_size = pattern.numel() // batch_size
        return jacobian.view(batch_size, state_size, state_size)

    def apply_symmetry(self, pattern: torch.Tensor, symmetry: torch.Tensor) -> torch.Tensor:
        """Apply symmetry transformation to pattern.
        
        Args:
            pattern: Input pattern tensor
            symmetry: Symmetry transformation matrix
            
        Returns:
            Transformed pattern
        """
        # Reshape pattern for matrix multiplication
        batch_size = pattern.size(0)
        state_size = pattern.numel() // batch_size
        pattern_flat = pattern.reshape(batch_size, state_size)
        
        # Apply symmetry transformation
        transformed = torch.matmul(pattern_flat, symmetry)
        
        # Reshape back to original shape
        return transformed.reshape_as(pattern)

    def apply_scale_transform(self, pattern: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply scale transformation to pattern.
        
        Args:
            pattern: Input pattern tensor
            scale: Scale transformation tensor
            
        Returns:
            Transformed pattern
        """
        # Apply scaling
        transformed = pattern * scale.unsqueeze(-1).unsqueeze(-1)
        
        # Normalize to preserve total intensity
        return transformed / transformed.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
