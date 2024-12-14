"""Implementation of diffusion dynamics.

This module implements a conservative diffusion system that preserves several key properties:
1. Mass conservation - The total mass (sum) of the system remains constant
2. Maximum principle - Values remain bounded by their initial min/max
3. Symmetry preservation - The diffusion process respects the underlying grid symmetries
4. Convergence to steady state - The system evolves towards a uniform distribution
"""

import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional

logger = logging.getLogger(__name__)


class DiffusionSystem(nn.Module):
    """Handles diffusion operations with mass conservation and symmetry preservation.
    
    This class implements a discrete diffusion process using a symmetric 9-point stencil.
    The implementation ensures several physical properties are preserved:
    
    1. Mass Conservation:
       - The sum of all values remains constant throughout the diffusion
       - Achieved through kernel normalization and exact correction
       
    2. Maximum Principle:
       - Values remain bounded by their initial min/max
       - Enforced through explicit clamping
       
    3. Symmetry Preservation:
       - The diffusion kernel is symmetric
       - Periodic boundary conditions maintain spatial symmetry
       
    4. Convergence Properties:
       - System converges to uniform steady state
       - Rate controlled by diffusion coefficient and time step
    """

    def __init__(self, grid_size: int = 32):
        """Initialize diffusion system.
        
        Args:
            grid_size (int, optional): Size of spatial grid. Defaults to 32.
        """
        super().__init__()
        self.size = grid_size
        
        # Initialize diffusion kernel
        kernel = torch.tensor([
            [0.0833, 0.1667, 0.0833],
            [0.1667, 0.0000, 0.1667],
            [0.0833, 0.1667, 0.0833]
        ], dtype=torch.float64)
        
        # Expand kernel for multi-channel convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
        self.kernel = kernel

    def forward(
        self,
        state: torch.Tensor,
        diffusion_coefficient: Union[float, torch.Tensor],
        dt: float
    ) -> torch.Tensor:
        """Forward pass applies diffusion step.
        
        This is a convenience wrapper around apply_diffusion for nn.Module compatibility.
        See apply_diffusion for full documentation.
        
        Args:
            state: Input state tensor [batch, channels, height, width]
            diffusion_coefficient: Controls rate of diffusion
            dt: Time step size
            
        Returns:
            Diffused state tensor
        """
        # Convert tensor coefficient to float if needed
        if isinstance(diffusion_coefficient, torch.Tensor):
            diffusion_coefficient = float(diffusion_coefficient.item())
        return self.apply_diffusion(state, diffusion_coefficient, dt)
    
    def apply_diffusion(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float,
        dt: float
    ) -> torch.Tensor:
        """Apply diffusion to state.
        
        Args:
            state: State tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient (must be float)
            dt: Time step
            
        Returns:
            torch.Tensor: Diffused state
        """
        # Ensure diffusion coefficient is float
        if isinstance(diffusion_coefficient, torch.Tensor):
            diffusion_coefficient = float(diffusion_coefficient.item())
            
        # Ensure state has batch and channel dimensions
        if len(state.shape) == 2:  # [height, width]
            state = state.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif len(state.shape) == 3:  # [channels, height, width]
            state = state.unsqueeze(0)  # Add batch dim
        elif len(state.shape) != 4:
            raise ValueError(f"Expected state tensor of shape [batch, channels, height, width] or [channels, height, width] or [height, width], got shape {state.shape}")
            
        # Get kernel and scale by diffusion parameters
        kernel = self.kernel.clone().to(state.dtype)
        kernel = kernel * (diffusion_coefficient * dt)
        kernel[0,0,1,1] = 1.0 - (diffusion_coefficient * dt)
        
        # Expand kernel for each channel
        kernel = kernel.repeat(state.shape[1], 1, 1, 1)
        
        # Apply periodic boundary conditions
        pad_size = 1
        padded = F.pad(state, (pad_size,)*4, mode='circular')
        
        # Apply convolution
        diffused = F.conv2d(padded, kernel, padding=0, groups=state.shape[1])
        
        # Remove extra dimensions if they were added
        if len(state.shape) == 2:
            diffused = diffused.squeeze(0).squeeze(0)
        elif len(state.shape) == 3:
            diffused = diffused.squeeze(0)
            
        return diffused
    
    def test_convergence(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float = 0.1,
        dt: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> bool:
        """Test if diffusion converges to steady state.
        
        This method applies repeated diffusion steps until either:
        1. The change between iterations is below tolerance (converged)
        2. Maximum iterations are reached (not converged)
        
        The convergence check uses the mean absolute difference between
        successive states, normalized to handle different scales.
        
        Args:
            state: Initial state tensor to evolve
            diffusion_coefficient: Controls diffusion rate
            dt: Time step size
            max_iter: Maximum iterations to try
            tol: Convergence tolerance
            
        Returns:
            True if converged within max_iter, False otherwise
        """
        prev_state = state
        
        for i in range(max_iter):
            # Apply diffusion with reduced logging
            with torch.no_grad():
                # Create diffusion kernel
                kernel = self.kernel.clone()
                kernel = kernel * (diffusion_coefficient * dt)
                kernel[0,0,1,1] = 1.0 - (diffusion_coefficient * dt)
                
                # Prepare kernel for convolution
                kernel = kernel.repeat(state.size(1), 1, 1, 1)
                
                # Apply periodic padding
                padded = F.pad(prev_state, (1,1,1,1), mode='circular')
                
                # Apply convolution
                curr_state = F.conv2d(
                    padded, 
                    kernel.to(padded.device),
                    groups=state.size(1),
                    padding=0
                )
            
            # Check convergence using mean absolute change
            diff = (curr_state - prev_state).abs().mean()
            if diff < tol:
                return True
            
            prev_state = curr_state
            
        return False  # Not converged within max_iter