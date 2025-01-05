"""Implementation of stability analysis."""

from typing import Union, Callable, Optional, Tuple
import torch
from torch import nn

from .models import ReactionDiffusionState, StabilityInfo, StabilityMetrics


class StabilityAnalyzer:
    """Analyzer for pattern stability and bifurcations."""
    
    def __init__(self, pattern_system):
        """Initialize stability analyzer.
        
        Args:
            pattern_system: Pattern dynamics system to analyze
        """
        self.pattern_system = pattern_system
        
    def is_stable(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        threshold: float = 0.1
    ) -> bool:
        """Check if a state is stable under the given dynamics.
        
        Args:
            state (torch.Tensor): State to check stability for
            reaction_term (Callable, optional): Reaction function. If None, uses system reaction.
            threshold (float): Stability threshold
            
        Returns:
            bool: True if state is stable, False otherwise
        """
        # Check if state is numerically valid
        if torch.isnan(state).any() or torch.isinf(state).any():
            return False
            
        # Use system reaction if none provided
        if reaction_term is None:
            reaction_term = self.pattern_system.reaction.compute_reaction
            
        # Compute stability value using reaction term
        stability = self.compute_stability(state, reaction_term=reaction_term)
            
        # Check if stability value is valid and below threshold
        if torch.isnan(stability) or torch.isinf(stability):
            return False
            
        return bool(stability < threshold)

    def compute_stability(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute stability value for a state.
        
        Args:
            state (torch.Tensor): State to compute stability for
            reaction_term (Callable, optional): Reaction function. If None, uses system reaction.
            
        Returns:
            torch.Tensor: Stability value
        """
        # Use system reaction if none provided
        if reaction_term is None:
            reaction_term = self.pattern_system.reaction.compute_reaction
            
        # Get Jacobian
        jacobian = self.compute_jacobian(state, reaction_term=reaction_term)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        # Get maximum real part
        max_real = torch.max(eigenvalues.real)
        
        return max_real
        
    def compute_jacobian(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute Jacobian matrix for pattern dynamics.
        
        Args:
            state (torch.Tensor): State to compute Jacobian at
            reaction_term (Callable, optional): Reaction function. If None, uses system reaction.
            eps (float): Finite difference epsilon
            
        Returns:
            torch.Tensor: Jacobian matrix
        """
        print("\nComputing Jacobian:")
        print(f"Input state - shape: {state.shape}, mean: {state.mean():.6f}, std: {state.std():.6f}")
        print(f"Epsilon: {eps}")
        
        # Get pattern shape
        batch_size = state.shape[0]
        channels = state.shape[1]
        spatial_size = int(torch.prod(torch.tensor(state.shape[2:])).item())
        n = channels * spatial_size
        
        print(f"Dimensions - batch: {batch_size}, channels: {channels}, spatial_size: {spatial_size}")
        print(f"Total Jacobian size: {n}x{n}")
        
        # Initialize Jacobian
        J = torch.zeros((n, n), dtype=state.dtype, device=state.device)
        
        # Reshape state to preserve batch dimension while flattening spatial dimensions
        state_flat = state.reshape(batch_size, channels, -1)
        print(f"Reshaped state - shape: {state_flat.shape}")
        
        # Track numerical properties
        max_diff = -float('inf')
        min_diff = float('inf')
        total_diff_norm = 0
        
        for i in range(n):
            # Create perturbation
            perturb = torch.zeros_like(state_flat)
            channel_idx = i // spatial_size
            spatial_idx = i % spatial_size
            perturb[:, channel_idx, spatial_idx] = eps
            
            # Compute forward difference
            if reaction_term is not None:
                # Reshape perturbation to match original state shape for reaction term
                perturb_shaped = perturb.reshape(state.shape)
                state_shaped = state_flat.reshape(state.shape)
                
                f_plus = reaction_term(state_shaped + perturb_shaped)
                f_minus = reaction_term(state_shaped - perturb_shaped)
                
                print(f"\nColumn {i}/{n}:")
                print(f"f_plus - mean: {f_plus.mean():.6f}, std: {f_plus.std():.6f}")
                print(f"f_minus - mean: {f_minus.mean():.6f}, std: {f_minus.std():.6f}")
            else:
                f_plus = self.pattern_system.reaction.compute_reaction(state_flat + perturb)
                f_minus = self.pattern_system.reaction.compute_reaction(state_flat - perturb)
            
            # Central difference
            diff = (f_plus - f_minus) / (2 * eps)
            
            # Handle case where reaction returns more components than input
            if diff.shape[1] > channels:
                diff = diff[:, :channels]  # Only use first channels components
            
            diff_mean = diff.mean().item()
            diff_std = diff.std().item()
            diff_norm = torch.norm(diff).item()
            
            max_diff = max(max_diff, diff_mean)
            min_diff = min(min_diff, diff_mean)
            total_diff_norm += diff_norm
            
            # Take mean over batch dimension and reshape to match Jacobian column
            diff_flat = diff.mean(dim=0).reshape(-1)  # Flatten all dimensions after batch
            J[:, i] = diff_flat  # Assign to column
            
            if i % 10 == 0:  # Log every 10th column
                print(f"Column {i} stats - mean: {diff_mean:.6f}, std: {diff_std:.6f}, norm: {diff_norm:.6f}")
        
        print("\nJacobian computation complete:")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Min difference: {min_diff:.6f}")
        print(f"Average difference norm: {total_diff_norm/n:.6f}")
        print(f"Jacobian stats - mean: {J.mean():.6f}, std: {J.std():.6f}, norm: {torch.norm(J):.6f}")
        
        return J
        
    def compute_eigenvalues(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of linearized dynamics.
        
        Args:
            state (torch.Tensor): State to compute eigenvalues at
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Eigenvalues and eigenvectors
        """
        # Get Jacobian
        jacobian = self.compute_jacobian(state)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(jacobian)
        
        return eigenvalues, eigenvectors
        
    def find_bifurcation(
        self,
        state: torch.Tensor,
        parameter_range: torch.Tensor,
        parameter_name: str = 'dt',
        threshold: float = 0.1
    ) -> Optional[float]:
        """Find bifurcation point by varying parameter.
        
        Args:
            state (torch.Tensor): Initial state
            parameter_range (torch.Tensor): Range of parameter values to check
            parameter_name (str): Name of parameter to vary
            threshold (float): Stability threshold
            
        Returns:
            Optional[float]: Bifurcation point if found, None otherwise
        """
        # Store original parameter value
        original_value = getattr(self.pattern_system, parameter_name)
        
        # Check stability across parameter range
        for value in parameter_range:
            # Set parameter value
            setattr(self.pattern_system, parameter_name, value.item())
            
            # Check stability
            if not self.is_stable(state, threshold=threshold):
                # Reset parameter
                setattr(self.pattern_system, parameter_name, original_value)
                return value.item()
                
        # Reset parameter
        setattr(self.pattern_system, parameter_name, original_value)
        return None

    def analyze_stability(
        self,
        state: torch.Tensor,
        perturbation: torch.Tensor
    ) -> StabilityMetrics:
        """Analyze stability of a state under perturbation.
        
        Args:
            state: State to analyze stability for
            perturbation: Perturbation to apply
            
        Returns:
            StabilityMetrics: Stability analysis results
        """
        # Compute linear stability using eigenvalue analysis
        stability_matrix = self.pattern_system.compute_stability_matrix(state)
        eigenvals = torch.linalg.eigvals(stability_matrix)
        linear_stability = torch.max(eigenvals.real)  # Keep as tensor
        
        # Compute nonlinear stability using perturbation
        perturbed_state = state + perturbation
        nonlinear_stability = self.compute_stability(perturbed_state)
        
        # Return metrics with proper types
        return StabilityMetrics(
            linear_stability=linear_stability,
            nonlinear_stability=nonlinear_stability,
            lyapunov_spectrum=eigenvals,
            structural_stability=float((nonlinear_stability / linear_stability).item())  # Convert to float
        )

    def compute_lyapunov_spectrum(
        self,
        state: torch.Tensor,
        steps: int = 100,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """Compute Lyapunov spectrum for a state.
        
        Args:
            state: State to compute spectrum for
            steps: Number of integration steps
            dt: Time step (optional)
            
        Returns:
            torch.Tensor: Lyapunov spectrum
        """
        # Get system size
        batch, channels, height, width = state.shape
        n = channels * height * width
        
        # Initialize orthonormal perturbations
        Q = torch.eye(n, dtype=torch.float64, device=state.device)
        Q = Q.reshape(n, channels, height, width)
        
        # Initialize Lyapunov exponents
        lyap = torch.zeros(n, dtype=torch.float64, device=state.device)
        
        # Time integration
        dt = dt if dt is not None else self.pattern_system.dt  # Use system dt if not provided
        if dt is None:
            dt = 0.01  # Default value if no dt is available
            
        for _ in range(steps):
            # Evolve perturbations
            for i in range(n):
                # Apply Jacobian
                Q[i] = self._apply_jacobian(state, Q[i])
                
                # Normalize
                norm = torch.norm(Q[i])
                if norm > 0:
                    Q[i] = Q[i] / norm
                    lyap[i] += torch.log(norm)
        
        # Average and sort
        lyap = lyap / (steps * dt)
        lyap = lyap.sort(descending=True)[0]
        
        return lyap.to(torch.float64)

    def _apply_jacobian(
        self,
        state: torch.Tensor,
        vector: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """Apply Jacobian matrix to vector using finite differences.
        
        Args:
            state (torch.Tensor): State point to evaluate Jacobian at
            vector (torch.Tensor): Vector to multiply with Jacobian
            eps (float): Finite difference step size
            
        Returns:
            torch.Tensor: Result of Jacobian-vector product
        """
        # Forward evaluation with reaction term only
        state_plus = state + eps * vector
        state_minus = state - eps * vector
        
        state_plus = self.pattern_system.step(state_plus)
        state_minus = self.pattern_system.step(state_minus)
        
        # Compute finite difference approximation
        diff = (state_plus - state_minus) / (2 * eps)
        
        return diff
