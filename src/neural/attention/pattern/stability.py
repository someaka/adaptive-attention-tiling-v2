"""Implementation of stability analysis."""

from typing import Union
import torch
from torch import nn

from .models import ReactionDiffusionState, StabilityInfo, StabilityMetrics


class StabilityAnalyzer:
    """Analysis of pattern stability."""

    def __init__(self, input_dim: int, num_modes: int = 8, hidden_dim: int = 64):
        """Initialize stability analyzer.
        
        Args:
            input_dim: Dimension of input state
            num_modes: Number of stability modes to analyze
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        
        # Stability analysis networks
        self.stability_network = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim * 2, dtype=torch.float64),  # 2x input for pattern + perturbation
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_modes * 2, dtype=torch.float64),
        )
        
        # Lyapunov spectrum network
        self.lyapunov_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes * 2, dtype=torch.float64),
        )
        
        # Mode decomposition
        self.mode_analyzer = nn.Sequential(
            nn.Linear(input_dim, num_modes, dtype=torch.float64),
            nn.Softmax(dim=-1)
        )
    
    def analyze_stability(
        self,
        fixed_point: Union[ReactionDiffusionState, torch.Tensor],
        perturbation: torch.Tensor,
    ) -> StabilityMetrics:
        """Analyze stability around fixed point.
        
        Args:
            fixed_point: Fixed point state
            perturbation: Small perturbation tensor
            
        Returns:
            StabilityMetrics object
        """
        # Convert to tensor if needed
        if not isinstance(fixed_point, torch.Tensor):
            fixed_point = fixed_point.to_tensor()
            
        # Ensure same dtype
        fixed_point = fixed_point.to(dtype=torch.float64)
        perturbation = perturbation.to(dtype=torch.float64)
        
        # Add batch dimension if needed
        if fixed_point.dim() == 3:
            fixed_point = fixed_point.unsqueeze(0)
        if perturbation.dim() == 3:
            perturbation = perturbation.unsqueeze(0)
            
        # Compute stability metrics
        linear_stability = self._compute_linear_stability(fixed_point, perturbation)
        nonlinear_stability = self._compute_nonlinear_stability(fixed_point, perturbation)
        
        # Compute Lyapunov spectrum
        lyapunov_spectrum = self.compute_lyapunov_spectrum(fixed_point)
        
        # Compute structural stability (minimum eigenvalue)
        structural_stability = float(linear_stability.min())
        
        return StabilityMetrics(
            linear_stability=linear_stability,
            nonlinear_stability=nonlinear_stability,
            lyapunov_spectrum=lyapunov_spectrum,
            structural_stability=structural_stability
        )
    
    def compute_lyapunov_spectrum(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute Lyapunov spectrum."""
        # Ensure correct dtype
        pattern = pattern.to(dtype=torch.float64)
        
        # Flatten input
        flat_pattern = pattern.reshape(pattern.shape[0], -1)
        
        # Get stability modes
        modes = self.mode_analyzer(flat_pattern)
        
        # Compute spectrum
        spectrum = self.lyapunov_network(flat_pattern)
        spectrum = spectrum.reshape(-1, self.num_modes, 2)  # [batch, modes, (re/im)]
        
        # Convert to complex tensor and take real part
        spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
        spectrum = torch.real(spectrum)
        
        return spectrum
    
    def _compute_linear_stability(self, pattern: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor:
        """Compute linear stability metric."""
        # Flatten inputs
        flat_pattern = pattern.reshape(pattern.shape[0], -1)
        flat_perturb = perturbation.reshape(perturbation.shape[0], -1)
        
        # Combine pattern and perturbation
        combined = torch.cat([flat_pattern, flat_perturb], dim=-1)
        
        # Get stability eigenvalues
        eigenvalues = self.stability_network(combined)
        eigenvalues = eigenvalues.reshape(-1, self.num_modes, 2)  # [batch, modes, (re/im)]
        
        # Return maximum real part
        return eigenvalues[..., 0].max(dim=-1)[0]
    
    def _compute_nonlinear_stability(
        self,
        pattern: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonlinear stability metric."""
        # Flatten inputs
        flat_pattern = pattern.reshape(pattern.shape[0], -1)
        flat_perturb = perturbation.reshape(perturbation.shape[0], -1)
        
        # Combine pattern and perturbation
        combined = torch.cat([flat_pattern, flat_perturb], dim=-1)
        
        # Get stability metric
        metric = self.stability_network(combined)
        
        return metric[..., 0]  # Return real part
