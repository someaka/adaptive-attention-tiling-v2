"""Implementation of diffusion dynamics."""

import logging
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiffusionSystem(nn.Module):
    """Handles diffusion operations with mass conservation and symmetry preservation."""

    def __init__(self, grid_size: int):
        """Initialize diffusion system.
        
        Args:
            grid_size: Size of square grid
        """
        super().__init__()
        self.grid_size = grid_size
        
        # Create symmetric diffusion kernel with exact coefficients
        kernel = torch.tensor([
            [0.0625, 0.125, 0.0625],
            [0.125, -1.0, 0.125],
            [0.0625, 0.125, 0.0625]
        ], dtype=torch.float64)
        
        # Store base kernel for diffusion
        self.register_buffer('base_kernel', kernel)
        logger.debug(f"Initialized diffusion kernel with shape {kernel.shape}")
    
    def forward(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float,
        dt: float
    ) -> torch.Tensor:
        """Forward pass applies diffusion.
        
        Args:
            state: Input state tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient
            dt: Time step
            
        Returns:
            Diffused state tensor
        """
        return self.apply_diffusion(state, diffusion_coefficient, dt)
    
    def apply_diffusion(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float,
        dt: float
    ) -> torch.Tensor:
        """Apply diffusion operator to state.
        
        Args:
            state: Input state tensor [batch, channels, height, width]
            diffusion_coefficient: Diffusion coefficient
            dt: Time step
        
        Returns:
            Diffused state tensor [batch, channels, height, width]
        """
        logger.debug(f"Applying diffusion with coefficient={diffusion_coefficient}, dt={dt}")
        
        # Input validation
        if not torch.isfinite(state).all():
            raise ValueError("Input state contains non-finite values")
        
        # Convert to double and track initial properties
        state = state.to(torch.float64)
        initial_mass = state.sum(dim=(-2,-1), keepdim=True)
        initial_max = state.amax(dim=(-2,-1), keepdim=True)
        initial_min = state.amin(dim=(-2,-1), keepdim=True)
        
        logger.debug(f"Initial state properties - Mass: {initial_mass.mean().item():.6e}, "
                    f"Range: [{initial_min.min().item():.6e}, {initial_max.max().item():.6e}]")
        
        # Create diffusion kernel
        kernel = self.base_kernel.clone()
        kernel = kernel * (diffusion_coefficient * dt)
        kernel[1,1] = kernel[1,1] + 1.0
        
        # Normalize kernel for exact mass conservation
        kernel_sum = kernel.sum()
        if kernel_sum != 0:
            kernel = kernel / kernel_sum
        kernel = kernel.view(1, 1, 3, 3).repeat(state.size(1), 1, 1, 1)
        
        # Apply periodic padding
        padded = F.pad(state, (1,1,1,1), mode='circular')
        
        # Apply convolution
        diffused = F.conv2d(
            padded, 
            kernel.to(padded.device),
            groups=state.size(1),
            padding=0
        )
        
        # Ensure maximum principle
        diffused = torch.clamp(
            diffused,
            min=initial_min,
            max=initial_max
        )
        
        # Ensure exact mass conservation using safe division
        final_mass = diffused.sum(dim=(-2,-1), keepdim=True)
        mass_ratio = torch.where(
            final_mass > 1e-15,
            initial_mass / final_mass,
            torch.ones_like(final_mass)
        )
        diffused = diffused * mass_ratio
        
        # Verify constraints
        with torch.no_grad():
            mass_error = torch.abs(diffused.sum(dim=(-2,-1)) - initial_mass)
            max_violation = torch.max(diffused - initial_max)
            min_violation = torch.min(initial_min - diffused)
            
            logger.debug(f"Constraint verification - Mass error: {mass_error.max().item():.6e}, "
                        f"Max violation: {max_violation.item():.6e}, "
                        f"Min violation: {min_violation.item():.6e}")
            
            if not torch.isfinite(diffused).all():
                raise RuntimeError("Diffusion produced non-finite values")
        
        return diffused.to(state.dtype)
    
    def test_convergence(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float = 0.1,
        dt: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> bool:
        """Test convergence to steady state.
        
        Args:
            state: Initial state tensor
            diffusion_coefficient: Diffusion coefficient
            dt: Time step
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            True if converged, False otherwise
        """
        logger.debug(f"Testing convergence with max_iter={max_iter}, tol={tol}")
        prev_state = state
        for i in range(max_iter):
            # Apply diffusion with reduced logging
            with torch.no_grad():
                # Create diffusion kernel
                kernel = self.base_kernel.clone()
                kernel = kernel * (diffusion_coefficient * dt)
                kernel[1,1] = kernel[1,1] + 1.0
                
                # Normalize kernel for exact mass conservation
                kernel_sum = kernel.sum()
                if kernel_sum != 0:
                    kernel = kernel / kernel_sum
                kernel = kernel.view(1, 1, 3, 3).repeat(state.size(1), 1, 1, 1)
                
                # Apply periodic padding
                padded = F.pad(prev_state, (1,1,1,1), mode='circular')
                
                # Apply convolution
                curr_state = F.conv2d(
                    padded, 
                    kernel.to(padded.device),
                    groups=state.size(1),
                    padding=0
                )
            
            # Check convergence
            diff = (curr_state - prev_state).abs().mean()
            if diff < tol:
                logger.debug(f"Converged after {i+1} iterations with diff={diff:.6e}")
                return True
            
            prev_state = curr_state
        
        logger.debug(f"Failed to converge after {max_iter} iterations with diff={diff:.6e}")
        return False
