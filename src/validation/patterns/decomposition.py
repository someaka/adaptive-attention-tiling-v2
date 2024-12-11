"""Pattern mode decomposition implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from scipy.fft import fft2, ifft2


@dataclass
class ModeDecomposer:
    """Decompose patterns into spatial modes."""
    
    def __init__(
        self,
        n_modes: int = 10,
        tolerance: float = 1e-6
    ):
        """Initialize decomposer.
        
        Args:
            n_modes: Number of modes to extract
            tolerance: Tolerance for mode extraction
        """
        self.n_modes = n_modes
        self.tolerance = tolerance
        
    def analyze_modes(
        self,
        pattern: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze pattern modes.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Dictionary with mode analysis results
        """
        # Compute Fourier modes
        fourier_modes = self._compute_fourier_modes(pattern)
        
        # Extract dominant modes
        dominant_modes = self._extract_dominant_modes(fourier_modes)
        
        # Compute mode interactions
        interactions = self._compute_mode_interactions(dominant_modes)
        
        return {
            "fourier_modes": fourier_modes,
            "dominant_modes": dominant_modes,
            "mode_interactions": interactions
        }
        
    def _compute_fourier_modes(
        self,
        pattern: torch.Tensor
    ) -> torch.Tensor:
        """Compute Fourier modes of pattern.
        
        Args:
            pattern: Input pattern tensor
            
        Returns:
            Fourier mode tensor
        """
        # Convert to numpy for FFT
        pattern_np = pattern.detach().cpu().numpy()
        
        # Compute 2D FFT
        modes = fft2(pattern_np)
        
        # Convert back to torch
        return torch.tensor(
            np.abs(modes),
            device=pattern.device
        )
        
    def _extract_dominant_modes(
        self,
        modes: torch.Tensor
    ) -> torch.Tensor:
        """Extract dominant Fourier modes.
        
        Args:
            modes: Fourier mode tensor
            
        Returns:
            Dominant mode tensor
        """
        # Flatten modes
        flat_modes = modes.reshape(-1)
        
        # Find dominant modes
        values, indices = torch.topk(
            flat_modes,
            min(self.n_modes, len(flat_modes))
        )
        
        # Filter by tolerance
        mask = values > self.tolerance
        values = values[mask]
        indices = indices[mask]
        
        # Get mode coordinates
        height, width = modes.shape
        rows = indices // width
        cols = indices % width
        
        # Stack coordinates and values
        return torch.stack([
            rows.float(),
            cols.float(),
            values
        ], dim=1)
        
    def _compute_mode_interactions(
        self,
        modes: torch.Tensor
    ) -> torch.Tensor:
        """Compute interactions between modes.
        
        Args:
            modes: Mode tensor [n_modes, 3]
            
        Returns:
            Interaction tensor [n_modes, n_modes]
        """
        n_modes = len(modes)
        interactions = torch.zeros(n_modes, n_modes)
        
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                # Compute mode coupling
                coupling = torch.sum(
                    modes[i, :2] * modes[j, :2]
                ) / (
                    torch.norm(modes[i, :2]) *
                    torch.norm(modes[j, :2])
                )
                
                # Weight by mode amplitudes
                coupling *= modes[i, 2] * modes[j, 2]
                
                interactions[i, j] = coupling
                interactions[j, i] = coupling
                
        return interactions
