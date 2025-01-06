"""Implementation of pattern dynamics."""

from typing import List, Optional, Callable, Dict, Any, Union, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import logging
import numpy as np

from .models import ReactionDiffusionState, StabilityInfo, StabilityMetrics, ControlSignal
from .stability import StabilityAnalyzer
from .reaction import ReactionSystem
from .diffusion import DiffusionSystem

from ...flow.hamiltonian import HamiltonianSystem
from .stability import StabilityAnalyzer
from .quantum import QuantumState, QuantumGeometricTensor

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

# Configure logging
logger = logging.getLogger(__name__)

class PatternDynamics(nn.Module):
    """Complete pattern dynamics system with quantum integration."""

    def __init__(
        self,
        grid_size: int = 32,
        space_dim: int = 2,
        boundary: str = 'periodic',
        dt: float = 0.01,
        num_modes: int = 8,
        hidden_dim: int = 64,
        quantum_enabled: bool = False
    ):
        """Initialize pattern dynamics system.
        
        Args:
            grid_size (int, optional): Size of grid. Defaults to 32.
            space_dim (int, optional): Spatial dimensions. Defaults to 2.
            boundary (str, optional): Boundary conditions. Defaults to 'periodic'.
            dt (float, optional): Time step. Defaults to 0.01.
            num_modes (int, optional): Number of stability modes. Defaults to 8.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 64.
            quantum_enabled (bool, optional): Enable quantum features. Defaults to False.
        """
        super().__init__()
        self.size = grid_size
        self.dim = space_dim
        self.dt = dt
        self.boundary = boundary
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.quantum_enabled = quantum_enabled
        
        # Initialize subsystems
        self.diffusion = DiffusionSystem(self.size)
        self.reaction = ReactionSystem(self.size)
        self.stability = StabilityAnalyzer(self)
        
        if quantum_enabled:
            # Initialize quantum components
            from ....core.flow.quantum import QuantumGeometricFlow
            from ....core.quantum.types import QuantumState
            self.quantum_flow = QuantumGeometricFlow(manifold_dim=space_dim, hidden_dim=hidden_dim)
            self.quantum_tensor = QuantumGeometricTensor(dim=space_dim)
            
    def _log_tensor_stats(self, tensor: torch.Tensor, name: str, channel: Optional[int] = None):
        """Helper to log tensor statistics."""
        if channel is not None:
            tensor = tensor[:, channel]
            prefix = f"Channel {channel} - "
        else:
            prefix = ""
            
        logger.info(f"\n=== {prefix}{name} Statistics ===")
        logger.info(f"Shape: {tensor.shape}")
        logger.info(f"Dtype: {tensor.dtype}")
        
        if torch.is_complex(tensor):
            abs_tensor = torch.abs(tensor)
            logger.info(f"Abs Min: {torch.min(abs_tensor)}")
            logger.info(f"Abs Max: {torch.max(abs_tensor)}")
            logger.info(f"Abs Mean: {torch.mean(abs_tensor)}")
            logger.info(f"Abs Std: {torch.std(abs_tensor)}")
            logger.info(f"Phase min: {torch.min(torch.angle(tensor))}")
            logger.info(f"Phase max: {torch.max(torch.angle(tensor))}")
        else:
            logger.info(f"Min: {torch.min(tensor)}")
            logger.info(f"Max: {torch.max(tensor)}")
            logger.info(f"Mean: {torch.mean(tensor)}")
            logger.info(f"Std: {torch.std(tensor)}")
            
        logger.info(f"Norm: {torch.norm(tensor)}")
        if len(tensor.shape) > 1:
            logger.info(f"Channel norms: {torch.norm(tensor, dim=1)}")

    def _to_quantum_state(self, state: torch.Tensor) -> QuantumState:
        """Convert classical state to quantum state.
        
        Args:
            state: Classical state tensor
            
        Returns:
            Quantum state
        """
        logger.info("\n=== _to_quantum_state conversion steps ===")
        self._log_tensor_stats(state, "Input state")
        
        # Log channel-wise statistics
        for c in range(state.shape[1]):
            self._log_tensor_stats(state, f"Input state channel {c}", channel=c)
        
        # Ensure state is normalized globally
        state = state / torch.norm(state)
        logger.info("\n=== After global normalization ===")
        self._log_tensor_stats(state, "Normalized state")
        for c in range(state.shape[1]):
            self._log_tensor_stats(state, f"Normalized state channel {c}", channel=c)
        
        # Convert to complex and ensure float64
        amplitudes = state.to(torch.complex128)
        logger.info("\n=== After complex conversion ===")
        self._log_tensor_stats(amplitudes, "Complex amplitudes")
        for c in range(amplitudes.shape[1]):
            self._log_tensor_stats(amplitudes, f"Complex amplitudes channel {c}", channel=c)
        
        # Initialize phase to zero
        phase = torch.zeros_like(amplitudes, dtype=torch.complex128)
        logger.info(f"\n=== Phase initialization ===")
        logger.info(f"Phase shape: {phase.shape}")
        logger.info(f"Phase dtype: {phase.dtype}")
        
        # Create basis labels based on state shape
        basis_size = state.shape[-1]
        basis_labels = [f"basis_{i}" for i in range(basis_size)]
        
        # Create quantum state
        quantum_state = QuantumState(
            amplitudes=amplitudes,
            basis_labels=basis_labels,
            phase=phase
        )
        
        # Log quantum state properties
        logger.info("\n=== Final quantum state ===")
        self._log_tensor_stats(quantum_state.amplitudes, "Quantum state amplitudes")
        for c in range(quantum_state.amplitudes.shape[1]):
            self._log_tensor_stats(quantum_state.amplitudes, f"Quantum state channel {c}", channel=c)
        
        return quantum_state
        
    def _from_quantum_state(self, quantum_state: QuantumState) -> torch.Tensor:
        """Convert quantum state to classical state.
        
        Args:
            quantum_state: Quantum state
            
        Returns:
            Classical state tensor
        """
        logger.info("\n=== _from_quantum_state conversion steps ===")
        self._log_tensor_stats(quantum_state.amplitudes, "Input quantum state")
        for c in range(quantum_state.amplitudes.shape[1]):
            self._log_tensor_stats(quantum_state.amplitudes, f"Input quantum state channel {c}", channel=c)
        
        # Get amplitudes and phase
        amplitudes = quantum_state.amplitudes
        phase = quantum_state.phase
        
        logger.info("\n=== Phase information ===")
        self._log_tensor_stats(phase, "Phase tensor")
        
        # Combine amplitude and phase
        state = amplitudes * torch.exp(1j * phase)
        logger.info("\n=== After phase combination ===")
        self._log_tensor_stats(state, "Combined state")
        for c in range(state.shape[1]):
            self._log_tensor_stats(state, f"Combined state channel {c}", channel=c)
        
        # Convert to real and normalize channel-wise
        state = state.real.to(torch.float64)
        logger.info("\n=== After real conversion ===")
        self._log_tensor_stats(state, "Real state")
        for c in range(state.shape[1]):
            self._log_tensor_stats(state, f"Real state channel {c}", channel=c)
            
        # Normalize each channel independently to preserve relative magnitudes
        for c in range(state.shape[1]):
            channel = state[:, c]
            channel_norm = torch.norm(channel)
            if channel_norm > 1e-8:  # Avoid division by zero
                state[:, c] = channel / channel_norm
                
        logger.info("\n=== After channel-wise normalization ===")
        self._log_tensor_stats(state, "Channel normalized state")
        for c in range(state.shape[1]):
            self._log_tensor_stats(state, f"Channel normalized state {c}", channel=c)
        
        # Final global normalization
        state = state / torch.norm(state)
        logger.info("\n=== After final global normalization ===")
        self._log_tensor_stats(state, "Final state")
        for c in range(state.shape[1]):
            self._log_tensor_stats(state, f"Final state channel {c}", channel=c)
        
        return state

    def compute_next_state(self, state: torch.Tensor) -> torch.Tensor:
        """Perform one step of pattern dynamics.
    
        Args:
            state (torch.Tensor): Current state
    
        Returns:
            torch.Tensor: Next state
        """
        if self.quantum_enabled:
            # Convert to quantum state
            quantum_state = self._to_quantum_state(state)
    
            # Compute quantum geometric tensor
            Q = self.quantum_tensor.compute_tensor(quantum_state)
    
            # Decompose into metric and Berry curvature
            g, B = self.quantum_tensor.decompose(Q)
    
            # Apply quantum evolution using proper Hamiltonian
            if not hasattr(self, 'hamiltonian_system'):
                # Each point in phase space has 2 components (real and imaginary)
                # For numerical stability, we use a smaller manifold dimension
                manifold_dim = 16  # Fixed dimension for stability
                self.hamiltonian_system = HamiltonianSystem(manifold_dim=manifold_dim)
    
            # Convert quantum state to phase space representation
            # Reshape the amplitudes to combine all spatial dimensions
            batch_size = quantum_state.amplitudes.shape[0]
            spatial_dims = quantum_state.amplitudes.shape[1:]  # Save spatial dimensions for later
            total_dim = int(np.prod(spatial_dims))  # Total number of elements in spatial dimensions
            amplitudes_flat = quantum_state.amplitudes.reshape(batch_size, total_dim)  # Flatten all spatial dims
    
            # Project to lower-dimensional manifold
            real_part = amplitudes_flat.real
            imag_part = amplitudes_flat.imag
            phase_space = torch.cat([real_part, imag_part], dim=-1)  # Shape: (batch_size, total_dim * 2)
            
            # Project to manifold dimension
            phase_space = phase_space[..., :self.hamiltonian_system.manifold_dim]
    
            # Evolve using Hamiltonian dynamics
            evolved_phase = self.hamiltonian_system.evolve(phase_space, dt=self.dt)
    
            # Pad back to original dimension
            evolved_phase = torch.nn.functional.pad(
                evolved_phase,
                (0, total_dim * 2 - self.hamiltonian_system.manifold_dim),
                mode='constant',
                value=0
            )
    
            # Reshape back to original dimensions
            half_dim = evolved_phase.shape[-1] // 2
            real_evolved = evolved_phase[..., :half_dim]
            imag_evolved = evolved_phase[..., half_dim:]
            evolved_complex = torch.complex(real_evolved, imag_evolved)
            evolved_reshaped = evolved_complex.reshape(batch_size, *spatial_dims)
    
            # Convert back to quantum state
            evolved = QuantumState(
                amplitudes=evolved_reshaped,
                basis_labels=quantum_state.basis_labels,
                phase=quantum_state.phase
            )
            
            # Convert back to classical state
            next_state = self._from_quantum_state(evolved)
            
            # Ensure normalization
            next_state = next_state / torch.norm(next_state)
            
            return next_state
        else:
            # Classical evolution
            next_state = self.diffusion(
                state,
                diffusion_coefficient=0.1,  # Default diffusion coefficient
                dt=self.dt
            )
            next_state = next_state / torch.norm(next_state)
            return next_state

    def evolve_pattern(
        self,
        pattern: torch.Tensor,
        diffusion_coefficient: float = 0.1,
        reaction_term: Optional[Callable] = None,
        steps: int = 100
    ) -> List[torch.Tensor]:
        """Evolve pattern forward in time."""
        logging.info(f"Starting pattern evolution - diffusion_coeff={diffusion_coefficient}, steps={steps}")
        logging.info(f"Initial state - shape: {pattern.shape}, mean: {pattern.mean():.6f}, std: {pattern.std():.6f}, norm: {torch.norm(pattern):.6f}")
        
        trajectory = []
        current = pattern
        
        # Track key metrics
        norms = []
        means = []
        stds = []
        mass_changes = []
        
        for step in range(steps):
            # Enable logging only at key points
            should_log = (step == 0 or step == steps-1 or step % 10 == 0)
            
            # Normalize current pattern
            norm = torch.norm(current.to(torch.float32), dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            current = current / norm
            
            trajectory.append(current)
            
            # Apply reaction and diffusion
            if reaction_term is not None:
                reaction = reaction_term(current)
            else:
                reaction = self.reaction.reaction_term(current, should_log=should_log)
                
            diffusion = self.diffusion.apply_diffusion(
                current, 
                diffusion_coefficient=diffusion_coefficient, 
                dt=self.dt,
                should_log=should_log
            )
            
            # Update state with stability check
            update = self.dt * (reaction + diffusion)
            update_norm = torch.norm(update)
            if update_norm > 1.0:
                update = update * (0.9 / update_norm)
                
            current = current + update
            
            # Track metrics
            norms.append(torch.norm(current).item())
            means.append(current.mean().item())
            stds.append(current.std().item())
            mass_changes.append(torch.abs(current.sum() - pattern.sum()) / pattern.sum())
            
            # Log every 10 steps
            if should_log:
                logging.info(f"Step {step}:")
                logging.info(f"  - Norm: {norms[-1]:.6f}")
                logging.info(f"  - Mean: {means[-1]:.6f}")
                logging.info(f"  - Std: {stds[-1]:.6f}")
                logging.info(f"  - Mass change: {mass_changes[-1]:.6f}")
        
        # Log final statistics
        logging.info("\nEvolution complete:")
        logging.info(f"  - Initial/final norm: {norms[0]:.6f}/{norms[-1]:.6f}")
        logging.info(f"  - Initial/final mean: {means[0]:.6f}/{means[-1]:.6f}")
        logging.info(f"  - Initial/final std: {stds[0]:.6f}/{stds[-1]:.6f}")
        logging.info(f"  - Final mass change: {mass_changes[-1]:.6f}")
        logging.info(f"  - Norm stability: {torch.std(torch.tensor(norms[-10:])):.6f}")
        
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
            # Try using eigvals with device
            eigenvalues = torch.linalg.eigvals(jacobian)
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
        if reaction_term is None:
            reaction_term = self.reaction.reaction_term
            
        # Compute reaction term
        with torch.no_grad():
            reaction = reaction_term(state)
            reaction = torch.clamp(reaction, min=-10.0, max=10.0)
            
        # Apply reaction term with time step
        if dt is None:
            dt = self.dt
            
        next_state = state + dt * reaction
        
        # Convert back to original dtype
        return next_state.to(orig_dtype)
        
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
        logging.info(f"Analyzing pattern formation over {len(evolution)} timesteps")
        
        if len(evolution) < 2:
            logging.warning("Insufficient timesteps for pattern detection")
            return False
            
        # Check if final states are similar (stable pattern)
        final_states = evolution[-10:]
        if len(final_states) < 2:
            logging.warning("Insufficient final states for stability analysis")
            return False
            
        # Compute changes between consecutive states
        changes = []
        for i in range(len(final_states)-1):
            diff = torch.abs(final_states[i+1] - final_states[i]).mean()
            changes.append(diff.item())
            logging.debug(f"State change at step {i}: {diff.item():.6f}")
            
        # Pattern formed if changes are small and consistent
        mean_change = sum(changes) / len(changes)
        std_change = torch.tensor(changes).std().item()
        logging.info(f"Pattern formation analysis - mean change: {mean_change:.6f}, std: {std_change:.6f}")
        
        is_stable = mean_change < 1e-3
        logging.info(f"Pattern formation {'detected' if is_stable else 'not detected'}")
        return bool(is_stable)

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
        initial_state: torch.Tensor,
        parameter_range: torch.Tensor,
        reaction_term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_iter: int = 50,
        convergence_threshold: float = 1e-4,
        stability_threshold: float = 0.1,
        eps: float = 1e-6,
    ) -> BifurcationDiagram:
        """Analyze bifurcations by varying parameters.
        
        Args:
            initial_state: Initial pattern state
            parameter_range: Range of parameter values to analyze
            reaction_term: Function that takes (state, param) and returns reaction term
            max_iter: Maximum iterations per parameter value
            convergence_threshold: Threshold for convergence
            stability_threshold: Threshold for stability changes
            eps: Finite difference epsilon
            
        Returns:
            BifurcationDiagram: Bifurcation analysis results
        """
        # Initialize stability analyzer
        stability = StabilityAnalyzer(self)
        
        # Initialize storage
        solution_states = []
        solution_params = []
        bifurcation_points = []
        
        # Initialize metrics
        metrics = {
            'iterations': [],
            'final_deltas': [],
            'eigenvalues': [],
            'stability_values': [],
            'state_changes': [],
            'convergence_rates': []
        }
        
        print(f"Starting bifurcation analysis:")
        print(f"Parameter range: [{parameter_range[0]:.3f}, {parameter_range[-1]:.3f}]")
        print(f"Number of points: {len(parameter_range)}")
        print(f"Max iterations: {max_iter}")
        
        # Previous state and stability for tracking changes
        prev_state = None
        prev_eigenvals = None
        
        # Analyze each parameter value
        for param in parameter_range:
            # Set parameter value
            self.parameter = param.item()
            
            # Initialize state
            state = initial_state.clone()
            
            # Track convergence
            converged = False
            final_delta = float('inf')
            
            # Evolve state
            for i in range(max_iter):
                # Step forward using the reaction term
                new_state = self.step(state, reaction_term=lambda x: reaction_term(x, param))
                
                # Check convergence
                delta = torch.norm(new_state - state).item()
                if delta < convergence_threshold:
                    converged = True
                    final_delta = delta
                    break
                    
                state = new_state
            
            # Store convergence metrics
            metrics['iterations'].append(i + 1)
            metrics['final_deltas'].append(final_delta)
            
            # Compute stability metrics
            stability_info = stability.analyze_stability(state, eps * torch.randn_like(state))
            eigenvals = stability_info.lyapunov_spectrum
            stability_value = stability_info.linear_stability
            
            metrics['eigenvalues'].append(eigenvals)
            metrics['stability_values'].append(stability_value)
            
            # Store solution
            solution_states.append(state)
            solution_params.append(param)
            
            # Check for bifurcation
            if prev_state is not None and prev_eigenvals is not None:
                # Check stability change
                stability_change = (stability_value * prev_eigenvals[0].real < 0)
                
                # Check state change
                state_diff = torch.norm(state - prev_state)
                state_change = state_diff > stability_threshold
                
                metrics['state_changes'].append(state_diff.item())
                
                # Detect bifurcation
                if stability_change or state_change:
                    bifurcation_points.append(param)
                    print(f"Bifurcation detected at parameter = {param:.3f}")
                    print(f"Stability change: {stability_change}")
                    print(f"State change: {state_change}")
                    print(f"State diff: {state_diff:.3e}")
            
            # Update previous values
            prev_state = state
            prev_eigenvals = eigenvals
            
            # Print progress
            if (len(parameter_range) >= 10) and (len(solution_params) % (len(parameter_range) // 10) == 0):
                print(f"Progress: {100 * len(solution_params) / len(parameter_range):.0f}%")
        
        # Convert to tensors
        solution_states = torch.stack(solution_states)
        solution_params = torch.tensor(solution_params)
        bifurcation_points = torch.tensor(bifurcation_points) if bifurcation_points else torch.tensor([])
        
        # Print summary
        print("\nAnalysis complete:")
        print(f"Average iterations: {sum(metrics['iterations']) / len(metrics['iterations']):.1f}")
        print(f"Final deltas: min={min(metrics['final_deltas']):.3e}, max={max(metrics['final_deltas']):.3e}")
        print(f"Bifurcation points detected: {len(bifurcation_points)}")
        
        return BifurcationDiagram(
            solution_states=solution_states,
            solution_params=solution_params,
            bifurcation_points=bifurcation_points
        )

    def compute_normal_form(
        self,
        bifurcation_point: dict
    ) -> torch.Tensor:
        """Compute normal form at bifurcation point.
        
        This method computes the normal form coefficients for the bifurcation
        using center manifold reduction. The process involves:
        1. Computing the center eigenspace
        2. Projecting dynamics onto center manifold
        3. Computing nonlinear terms up to cubic order
        
        Args:
            bifurcation_point: Dictionary containing:
                - state: State at bifurcation point
                - parameter: Critical parameter value
                - type: Type of bifurcation ("hopf", "pitchfork", "saddle-node")
            
        Returns:
            torch.Tensor: Normal form coefficients
        """
        # Extract bifurcation information
        state = bifurcation_point["state"]
        param = bifurcation_point["parameter"]
        bif_type = bifurcation_point["type"]
        
        # Compute Jacobian at bifurcation point
        J = self.compute_jacobian(state)
        
        # Get eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eig(J)
        
        # Find center eigenspace (eigenvalues close to imaginary axis)
        center_mask = torch.abs(eigenvals.real) < 1e-6
        center_eigenvals = eigenvals[center_mask]
        center_eigenvecs = eigenvecs[:, center_mask]
        
        # Project state onto center eigenspace
        center_coords = torch.matmul(center_eigenvecs.T.conj(), state.reshape(-1))
        
        # Compute nonlinear terms based on bifurcation type
        if bif_type == "hopf":
            # For Hopf, compute first Lyapunov coefficient
            omega = center_eigenvals[0].imag  # Frequency at bifurcation
            
            # Compute cubic terms in normal form
            cubic_term = self._compute_hopf_coefficient(state, center_eigenvecs, omega)
            
            # Return [frequency, cubic coefficient]
            return torch.stack([omega, cubic_term])
            
        elif bif_type == "pitchfork":
            # For pitchfork, compute cubic coefficient
            cubic_term = self._compute_pitchfork_coefficient(state, center_eigenvecs)
            
            # Return [0, cubic coefficient]
            return torch.stack([torch.zeros(1, device=state.device), cubic_term])
            
        else:  # saddle-node
            # For saddle-node, compute quadratic coefficient
            quad_term = self._compute_saddle_node_coefficient(state, center_eigenvecs)
            
            # Return [quadratic coefficient, 0]
            return torch.stack([quad_term, torch.zeros(1, device=state.device)])
            
    def _compute_hopf_coefficient(
        self,
        state: torch.Tensor,
        eigenvecs: torch.Tensor,
        omega: torch.Tensor
    ) -> torch.Tensor:
        """Compute first Lyapunov coefficient for Hopf bifurcation."""
        # Get critical eigenvector
        q = eigenvecs[:, 0]  # Complex eigenvector
        
        # Compute adjoint eigenvector
        p = torch.linalg.solve(self.compute_jacobian(state).T, q)
        p = p / torch.dot(p, q)
        
        # Compute second and third order derivatives
        eps = 1e-6
        state_flat = state.reshape(-1)
        n = len(state_flat)
        
        # Second derivatives (using finite differences)
        H = torch.zeros((n, n, n), dtype=torch.complex64, device=state.device)
        for i in range(n):
            for j in range(n):
                ei = torch.zeros(n, device=state.device)
                ej = torch.zeros(n, device=state.device)
                ei[i] = eps
                ej[j] = eps
                
                f_ij = self.compute_next_state((state_flat + ei + ej).reshape_as(state))
                f_i = self.compute_next_state((state_flat + ei).reshape_as(state))
                f_j = self.compute_next_state((state_flat + ej).reshape_as(state))
                f_0 = self.compute_next_state(state)
                
                H[:, i, j] = (f_ij - f_i - f_j + f_0).reshape(-1) / (eps * eps)
        
        # Compute g21 term (coefficient of z|z|^2 term)
        g21 = torch.zeros(1, dtype=torch.complex64, device=state.device)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    g21 += p[i] * H[i, j, k] * q[j] * q[k].conj()
                    
        # First Lyapunov coefficient
        l1 = (1 / (2 * omega)) * g21.real
        
        return l1
        
    def _compute_pitchfork_coefficient(
        self,
        state: torch.Tensor,
        eigenvecs: torch.Tensor
    ) -> torch.Tensor:
        """Compute cubic coefficient for pitchfork bifurcation."""
        # Get critical eigenvector
        v = eigenvecs[:, 0].real
        
        # Compute cubic term using finite differences
        eps = 1e-6
        state_flat = state.reshape(-1)
        
        # Evaluate field in positive and negative directions
        f_pos = self.compute_next_state((state_flat + eps * v).reshape_as(state))
        f_neg = self.compute_next_state((state_flat - eps * v).reshape_as(state))
        f_0 = self.compute_next_state(state)
        
        # Approximate cubic coefficient
        cubic = (f_pos + f_neg - 2 * f_0).reshape(-1) / (eps * eps)
        coef = torch.dot(v, cubic) / (6 * torch.norm(v)**4)
        
        return coef
        
    def _compute_saddle_node_coefficient(
        self,
        state: torch.Tensor,
        eigenvecs: torch.Tensor
    ) -> torch.Tensor:
        """Compute quadratic coefficient for saddle-node bifurcation."""
        # Get critical eigenvector
        v = eigenvecs[:, 0].real
        
        # Compute quadratic term using finite differences
        eps = 1e-6
        state_flat = state.reshape(-1)
        
        # Evaluate field in critical direction
        f_pos = self.compute_next_state((state_flat + eps * v).reshape_as(state))
        f_neg = self.compute_next_state((state_flat - eps * v).reshape_as(state))
        f_0 = self.compute_next_state(state)
        
        # Approximate quadratic coefficient
        quad = (f_pos - 2 * f_0 + f_neg).reshape(-1) / (eps * eps)
        coef = torch.dot(v, quad) / (2 * torch.norm(v)**3)
        
        return coef

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
            next_state = self.apply_reaction(next_state, reaction_term, self.dt)
            
        return next_state

    def reaction_diffusion(
        self,
        state: torch.Tensor,
        reaction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        param: torch.Tensor,
        dt: float = 1.0,
        diffusion_coefficient: float = 0.1,
    ) -> torch.Tensor:
        """Compute one step of reaction-diffusion dynamics."""
        # Add batch dimension if needed
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            print(f"Added batch dimension, new shape: {state.shape}")

        # Add noise to zero state to break symmetry
        if torch.all(state == 0):
            noise = torch.randn_like(state) * 0.1
            state = state + noise
            print(f"Added noise to zero state - new mean: {state.mean():.6f}, std: {state.std():.6f}")

        # Compute reaction term
        reaction_term = reaction(state, param)
        print(f"Custom reaction term - mean: {reaction_term.mean():.6f}, std: {reaction_term.std():.6f}")

        # Apply diffusion term (Laplacian)
        padded = F.pad(state, (1, 1, 1, 1), mode="circular")
        diffusion_term = (
            padded[:, :, :-2, 1:-1]  # up
            + padded[:, :, 2:, 1:-1]  # down
            + padded[:, :, 1:-1, :-2]  # left
            + padded[:, :, 1:-1, 2:]  # right
            - 4 * state
        )
        print(f"Diffusion term - mean: {diffusion_term.mean():.6f}, std: {diffusion_term.std():.6f}")

        # Combine updates with larger step size
        update = dt * reaction_term + diffusion_coefficient * diffusion_term
        print(f"Combined update (before adding to state) - mean: {update.mean():.6f}, std: {update.std():.6f}")

        # Update state
        state = state + update
        print(f"After adding update - mean: {state.mean():.6f}, std: {state.std():.6f}")

        # Normalize if growth is extreme (increased threshold) while preserving mass
        current_norm = torch.norm(state)
        print(f"Current norm: {current_norm:.6f}")
        if current_norm > 10.0:
            # Preserve mass by normalizing each component separately
            state_mean = state.mean(dim=(-2, -1), keepdim=True)
            state = state_mean + (state - state_mean) * (10.0 / current_norm)

        # Remove batch dimension if it was added
        if len(state.shape) == 4 and state.shape[0] == 1:
            state = state.squeeze(0)
            print(f"Removed batch dimension, final shape: {state.shape}")

        print(f"Final state - mean: {state.mean():.6f}, std: {state.std():.6f}\n")
        return state
    
    def compute_stability(
        self,
        state: torch.Tensor,
        reaction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        param: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> float:
        """Compute stability of the current state by analyzing the Jacobian."""
        print("\nComputing Jacobian:")
        print(f"Input state - shape: {state.shape}, mean: {state.mean():.6f}, std: {state.std():.6f}")
        print(f"Epsilon: {epsilon}")

        # Add batch dimension if needed
        if len(state.shape) == 3:  # [channels, height, width]
            state = state.unsqueeze(0)  # [batch, channels, height, width]

        # Get dimensions
        batch_size, num_channels, height, width = state.shape
        spatial_size = height * width
        total_size = num_channels * spatial_size
        print(f"Dimensions - batch: {num_channels}, channels: {spatial_size}, spatial_size: {width}")
        print(f"Total Jacobian size: {total_size}x{total_size}")

        # Reshape state for Jacobian computation
        state_flat = state.reshape(batch_size, -1)  # [batch, channels * height * width]
        print(f"Reshaped state - shape: {state.shape}")

        # Initialize Jacobian
        jacobian = torch.zeros(total_size, total_size)

        # Compute Jacobian column by column using finite differences
        for i in range(total_size):
            print(f"\nColumn {i}/{total_size}:")
            # Create perturbation
            perturb = torch.zeros_like(state_flat)
            perturb[0, i] = epsilon

            # Forward difference
            state_plus = state_flat + perturb
            state_plus = state_plus.reshape(batch_size, num_channels, height, width)
            f_plus = reaction(state_plus, param)
            f_plus_flat = f_plus.reshape(batch_size, -1)
            print(f"f_plus - mean: {f_plus.mean():.6f}, std: {f_plus.std():.6f}")

            # Backward difference
            state_minus = state_flat - perturb
            state_minus = state_minus.reshape(batch_size, num_channels, height, width)
            f_minus = reaction(state_minus, param)
            f_minus_flat = f_minus.reshape(batch_size, -1)
            print(f"f_minus - mean: {f_minus.mean():.6f}, std: {f_minus.std():.6f}")

            # Compute column of Jacobian
            diff = (f_plus_flat - f_minus_flat) / (2 * epsilon)
            jacobian[:, i] = diff[0]

            if i % 10 == 0:  # Print stats periodically
                print(f"Column {i} stats - mean: {diff.mean():.6f}, std: {diff.std():.6f}, norm: {torch.norm(diff):.6f}")

        # Print Jacobian statistics
        print("\nJacobian computation complete:")
        print(f"Max difference: {jacobian.max():.6f}")
        print(f"Min difference: {jacobian.min():.6f}")
        print(f"Average difference norm: {torch.norm(jacobian, dim=1).mean():.6f}")
        print(f"Jacobian stats - mean: {jacobian.mean():.6f}, std: {jacobian.std():.6f}, norm: {torch.norm(jacobian):.6f}")

        # Return maximum eigenvalue magnitude as stability measure
        eigenvalues = torch.linalg.eigvals(jacobian)
        stability = torch.max(torch.abs(eigenvalues)).item()
        return stability

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

    def forward(self, states: torch.Tensor, return_patterns: bool = False) -> dict[str, torch.Tensor]:
        """Forward pass through pattern dynamics.
        
        Args:
            states: Input states [batch, heads, seq_len, dim]
            return_patterns: Whether to return pattern information
            
        Returns:
            Dictionary containing:
                - routing_scores: Attention routing scores
                - patterns: Pattern states if return_patterns=True
                - pattern_scores: Pattern importance scores if return_patterns=True
        """
        # Compute pattern evolution
        patterns = self.evolve_pattern(states)
        
        # Compute routing scores from final pattern state
        routing_scores = torch.softmax(patterns[-1].mean(dim=-1), dim=-1)
        
        # Prepare results
        results = {
            "routing_scores": routing_scores
        }
        
        if return_patterns:
            results.update({
                "patterns": torch.stack(patterns),
                "pattern_scores": torch.softmax(
                    torch.stack([self.compute_energy(p) for p in patterns]), 
                    dim=0
                )
            })
            
        return results

    def __call__(self, states: torch.Tensor, return_patterns: bool = False) -> dict[str, torch.Tensor]:
        """Make the class callable."""
        return self.forward(states, return_patterns)

    def compute_quantum_potential(self, state: torch.Tensor) -> torch.Tensor:
        """Compute quantum potential for pattern state.
        
        Uses the quantum geometric flow infrastructure to compute proper
        quantum corrections to the classical dynamics.
        
        Args:
            state: Pattern state tensor
            
        Returns:
            Quantum potential tensor with same shape as input
        """
        if not self.quantum_enabled:
            raise RuntimeError("Quantum features not enabled")
            
        # Convert to quantum state
        quantum_state = self._to_quantum_state(state)
        
        # Get quantum metrics including corrections
        metrics = self.quantum_flow.compute_quantum_metrics(quantum_state)
        
        # Get quantum geometric tensor
        Q = self.quantum_flow.compute_quantum_metric_tensor(quantum_state)
        
        # Compute geometric contribution
        V_geometric = -0.5 * torch.einsum('...ij,...ij->...', Q, Q)
        
        # Add corrections if available
        corrections = metrics.get("quantum_corrections")
        if corrections is not None:
            V_corrections = -0.5 * torch.sum(corrections * corrections, dim=-1)
            V_quantum = V_geometric + V_corrections
        else:
            V_quantum = V_geometric
        
        # Ensure numerical stability
        V_quantum = torch.clamp(V_quantum, min=-10.0, max=10.0)
        
        return V_quantum.real

    def compute_berry_phase(self, state: torch.Tensor, path: torch.Tensor) -> float:
        """Compute Berry phase for a closed path in parameter space.
        
        Args:
            state: Initial state tensor
            path: Parameter space path tensor [num_points, dim]
            
        Returns:
            Berry phase (in radians)
        """
        if not self.quantum_enabled:
            raise RuntimeError("Quantum features not enabled")
            
        # Convert to quantum state
        quantum_state = self._to_quantum_state(state)
        
        # Initialize phase
        berry_phase = 0.0
        
        # Parallel transport around path
        for i in range(path.shape[0] - 1):
            # Get current and next points
            p1 = path[i]
            p2 = path[i + 1]
            
            # Compute quantum geometric tensor at current point
            Q = self.quantum_tensor.compute_tensor(quantum_state)
            
            # Extract Berry curvature
            _, B = self.quantum_tensor.decompose(Q)
            
            # Compute contribution to Berry phase
            dp = p2 - p1
            berry_phase += torch.sum(B @ dp)
            
            # Evolve state to next point using proper parallel transport
            quantum_state = self._parallel_transport(quantum_state, p1, p2)
            
        return float(berry_phase)

    def _parallel_transport(
        self,
        state: QuantumState,
        p1: torch.Tensor,
        p2: torch.Tensor
    ) -> QuantumState:
        """Parallel transport quantum state between points.
        
        Uses the quantum geometric flow infrastructure for proper parallel transport.
        
        Args:
            state: Quantum state
            p1: Starting point
            p2: End point
            
        Returns:
            Transported quantum state
        """
        # Compute transport vector
        vector = p2 - p1
        
        # Use quantum flow parallel transport
        return self.quantum_flow.parallel_transport_state(
            state=state,
            vector=vector,
            connection=None  # Let the quantum flow compute the connection
        )

    def evolve(
        self,
        state: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        """Evolve pattern state forward in time.
        
        Args:
            state: Current state tensor
            time: Evolution time
            
        Returns:
            Evolved state
        """
        # Convert to float64 for numerical stability
        state = state.to(torch.float64)
        
        # Apply diffusion with time step
        diffused = self.apply_diffusion(state, diffusion_coefficient=0.1, dt=time)
        
        # Apply reaction with time step
        reacted = self.apply_reaction(diffused, dt=time)
        
        # Ensure non-negative values and normalize
        evolved = torch.clamp(reacted, min=0.0)
        evolved = evolved / (evolved.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Convert back to original dtype
        return evolved.to(state.dtype)
