"""Pattern dynamics implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np


@dataclass
class PatternDynamics:
    """Base class for pattern dynamics."""
    
    def __init__(
        self,
        dt: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize dynamics.
        
        Args:
            dt: Time step size
            device: Computation device
        """
        self.dt = dt
        self.device = device
        
    def evolve_pattern_field(
        self,
        pattern: torch.Tensor,
        field_operator: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evolve pattern field with field-theoretic dynamics.
        
        Implements pattern field evolution using:
        1. Local field dynamics
        2. Non-local interactions
        3. Conservation laws
        
        Args:
            pattern: Pattern field tensor [batch_size, *spatial_dims]
            field_operator: Optional field evolution operator [*spatial_dims, *spatial_dims]
            
        Returns:
            Tuple of (evolved_pattern, evolution_metrics)
        """
        # Initialize metrics
        metrics: Dict[str, Any] = {}
        
        # Get dimensions
        batch_size = pattern.shape[0]
        spatial_dims = pattern.shape[1:]
        pattern_size = int(torch.prod(torch.tensor(spatial_dims)).item())
        
        # Compute local dynamics
        if field_operator is None:
            # Use default Laplacian evolution
            field_operator = self._compute_laplacian(pattern)
            
        # Reshape pattern for matrix multiplication
        pattern_flat = pattern.reshape(batch_size, pattern_size)
        
        # Ensure field operator has correct shape for matrix multiplication
        if len(field_operator.shape) == 2:
            # Single operator for all batches
            field_op_flat = field_operator.reshape(pattern_size, pattern_size)
            field_op_flat = field_op_flat.unsqueeze(0).expand(batch_size, pattern_size, pattern_size)
        else:
            # Batch of operators
            field_op_flat = field_operator.reshape(batch_size, pattern_size, pattern_size)
            
        # Apply field operator
        evolved_flat = torch.bmm(field_op_flat, pattern_flat.unsqueeze(-1)).squeeze(-1)
        
        # Reshape back to original dimensions
        evolved_pattern = evolved_flat.reshape(batch_size, *spatial_dims)
        
        # Compute evolution metrics
        metrics['field_operator_norm'] = torch.norm(field_operator)
        metrics['pattern_norm'] = torch.norm(pattern)
        metrics['evolved_norm'] = torch.norm(evolved_pattern)
        
        return evolved_pattern, metrics
        
    def _compute_laplacian(
        self,
        pattern: torch.Tensor
    ) -> torch.Tensor:
        """Compute discrete Laplacian operator for pattern field."""
        # Get pattern dimensions
        *batch_dims, height, width = pattern.shape
        
        # Compute 2D Laplacian stencil with same dtype as input
        laplacian = torch.zeros((*batch_dims, height, width), device=self.device, dtype=pattern.dtype)
        laplacian[..., 1:, :] += pattern[..., :-1, :]    # Up
        laplacian[..., :-1, :] += pattern[..., 1:, :]    # Down
        laplacian[..., :, 1:] += pattern[..., :, :-1]    # Left
        laplacian[..., :, :-1] += pattern[..., :, 1:]    # Right
        laplacian = laplacian - 4 * pattern               # Center
        
        return laplacian

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
        # Calculate number of steps needed
        num_steps = int(time / self.dt)
        current_state = state
        
        # Get initial energy
        initial_energy = self.compute_energy(state)['total']
        
        # Evolve for the required number of steps
        for _ in range(num_steps):
            # Use evolve_pattern_field which is already implemented
            evolved_state, _ = self.evolve_pattern_field(current_state)
            
            # Compute current energy
            current_energy = self.compute_energy(evolved_state)['total']
            
            # Rescale evolved state to conserve energy
            scale_factor = torch.sqrt(initial_energy / (current_energy + 1e-9))
            evolved_state = evolved_state * scale_factor
            
            current_state = evolved_state
            
        return current_state
        
    def compute_flow(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow field at current state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Flow field tensor
        """
        # Compute Laplacian which gives us the diffusive flow
        flow = self._compute_laplacian(state)
        
        # Scale by time step
        flow = self.dt * flow
        
        return flow
        
    def compute_energy(
        self,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute energy components.
        
        Args:
            state: Current state tensor
            
        Returns:
            Dictionary of energy components
        """
        # Get pattern dimensions
        *batch_dims, height, width = state.shape
        
        # Compute kinetic energy (using squared gradients)
        gradients = torch.gradient(state)
        kinetic = 0.5 * sum(torch.sum(torch.abs(grad) ** 2, dim=tuple(range(len(batch_dims)))) 
                           for grad in gradients)
        
        # Compute potential energy (using pattern amplitude)
        potential = 0.5 * torch.sum(torch.abs(state) ** 2, dim=tuple(range(len(batch_dims))))
        
        # Compute total energy
        total = kinetic + potential
        
        # Convert to tensors if needed
        kinetic = torch.as_tensor(kinetic, device=state.device)
        potential = torch.as_tensor(potential, device=state.device)
        total = torch.as_tensor(total, device=state.device)
        
        # Return energy components
        return {
            'kinetic': kinetic,
            'potential': potential,
            'total': total
        }
        
    def compute_conserved_quantities(
        self,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute conserved quantities.
        
        Args:
            state: Current state tensor
            
        Returns:
            Dictionary of conserved quantities
        """
        # Get pattern dimensions
        *batch_dims, height, width = state.shape
        
        # Compute total mass (sum over spatial dimensions)
        mass = torch.sum(state, dim=tuple(range(len(batch_dims), len(state.shape))))
        
        # Compute energy components
        energy = self.compute_energy(state)
        
        # Return all conserved quantities
        return {
            'mass': mass,
            'kinetic_energy': energy['kinetic'],
            'potential_energy': energy['potential'],
            'total_energy': energy['total']
        }
