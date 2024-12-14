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
        jacobian = self.compute_jacobian(state)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        # Get maximum real part
        max_real = torch.max(eigenvalues.real)
        
        return max_real
        
    def compute_jacobian(
        self,
        state: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute Jacobian matrix for pattern dynamics.
        
        Args:
            state (torch.Tensor): State to compute Jacobian at
            eps (float): Finite difference epsilon
            
        Returns:
            torch.Tensor: Jacobian matrix
        """
        # Get pattern shape
        batch_size = state.shape[0]
        n = int(torch.prod(torch.tensor(state.shape[1:])).item())
        
        # Reshape pattern for Jacobian computation
        state_flat = state.reshape(batch_size, -1)
        
        # Initialize Jacobian
        J = torch.zeros((batch_size, n, n), dtype=torch.float64)
        
        # Compute Jacobian using finite differences
        for i in range(n):
            perturb = torch.zeros_like(state_flat)
            perturb[:, i] = eps
            
            # Forward difference
            state_plus = state_flat + perturb
            state_plus = state_plus.reshape(state.shape)
            forward = self.pattern_system.step(state_plus)
            forward = forward.reshape(batch_size, -1)
            
            # Backward difference  
            state_minus = state_flat - perturb
            state_minus = state_minus.reshape(state.shape)
            backward = self.pattern_system.step(state_minus)
            backward = backward.reshape(batch_size, -1)
            
            # Central difference
            J[:, :, i] = (forward - backward) / (2 * eps)
            
        # Average over batch
        J = torch.mean(J, dim=0)
        
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
