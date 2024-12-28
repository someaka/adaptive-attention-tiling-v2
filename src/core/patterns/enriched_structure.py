"""Enriched categorical structure for handling transitions in pattern spaces.

This module implements the enriched categorical framework that provides a natural way
to handle transitions between different dimensional spaces using enriched morphisms
and categorical composition laws.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .operadic_structure import OperadicOperation

@dataclass
class EnrichedMorphism:
    """Represents an enriched morphism between pattern spaces."""
    source_space: Tensor
    target_space: Tensor
    structure_map: Tensor  # The enriched morphism structure

class EnrichedTransition(ABC):
    """Abstract base class for enriched transitions."""
    
    @abstractmethod
    def create_morphism(self, source: Tensor, target: Tensor) -> EnrichedMorphism:
        """Create an enriched morphism between spaces."""
        pass
    
    @abstractmethod
    def compose(self, first: EnrichedMorphism, second: EnrichedMorphism) -> EnrichedMorphism:
        """Compose enriched morphisms."""
        pass

class WaveEmergence:
    """Implements wave equation based emergence of structures."""
    
    def __init__(self, dt: float = 0.1, num_steps: int = 10, dtype: torch.dtype = torch.float32):
        """Initialize wave emergence structure.
        
        Args:
            dt: Time step for wave evolution
            num_steps: Number of evolution steps
            dtype: Data type for tensors
        """
        self.dt = dt
        self.num_steps = num_steps
        self.dtype = dtype
    
    def evolve_structure(self, pattern: Tensor, direction: Tensor) -> Tensor:
        """Evolve pattern structure using wave equation.
        
        Args:
            pattern: Input pattern tensor
            direction: Evolution direction tensor
            
        Returns:
            Evolved pattern tensor
        """
        # Initialize wave components with gradients
        u = pattern.to(dtype=self.dtype).detach().requires_grad_(True)
        v = torch.zeros_like(pattern, dtype=self.dtype).requires_grad_(True)  # Velocity field
        
        # Ensure compatible shapes for direction
        if len(direction.shape) == 2 and len(u.shape) == 2:
            # Both are matrices
            if direction.shape[1] != u.shape[1]:
                # Different dimensions - use padding
                min_dim = min(direction.shape[1], u.shape[1])
                direction = direction[..., :min_dim]
                u = u[..., :min_dim]
                v = v[..., :min_dim]
        elif len(direction.shape) == 2 and len(u.shape) == 1:
            # Direction is matrix, u is vector
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)
        elif len(direction.shape) == 1 and len(u.shape) == 2:
            # Direction is vector, u is matrix
            direction = direction.unsqueeze(0)
        
        # Evolve using wave equation
        for _ in range(self.num_steps):
            # Update velocity
            v = v + self.dt * self._laplacian(u)
            
            # Update position
            u = u + self.dt * v
            
            # Apply direction constraint
            if len(direction.shape) == len(u.shape):
                # Same shape - use element-wise multiplication
                u = u + self.dt * direction * u
            else:
                # Different shapes - use matrix multiplication
                u = u + self.dt * torch.matmul(direction, u.transpose(-2, -1)).transpose(-2, -1)
            
            # Normalize
            u = u / torch.norm(u, dim=-1, keepdim=True).clamp(min=1e-6)
            
        # Restore original shape
        if len(pattern.shape) == 1 and len(u.shape) == 2:
            u = u.squeeze(0)
            
        return u.detach()
    
    def _laplacian(self, tensor: Tensor) -> Tensor:
        """Compute discrete Laplacian of tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Laplacian of input tensor
        """
        # Handle complex tensors by splitting real and imaginary parts
        if tensor.is_complex():
            real_laplacian = self._laplacian(tensor.real)
            imag_laplacian = self._laplacian(tensor.imag)
            return real_laplacian + 1j * imag_laplacian
            
        # Use constant padding for 1D case
        if len(tensor.shape) == 1:
            padded = torch.nn.functional.pad(
                tensor.unsqueeze(0),
                (1, 1),
                mode='constant',
                value=tensor[0].item()
            ).squeeze(0)
            return padded[:-2] + padded[2:] - 2 * tensor
        else:
            # For 2D case, use padding along last dimension
            padded = torch.nn.functional.pad(
                tensor,
                (1, 1),
                mode='constant',
                value=tensor[..., 0].mean().item()
            )
            return padded[..., :-2] + padded[..., 2:] - 2 * tensor

class PatternTransition(EnrichedTransition):
    """Implements enriched transitions for pattern spaces."""
    
    def __init__(self, wave_emergence: Optional[WaveEmergence] = None, dtype: torch.dtype = torch.float32):
        """Initialize pattern transition structure.
        
        Args:
            wave_emergence: Wave emergence structure for natural transitions
            dtype: Data type for tensors
        """
        self.wave = wave_emergence or WaveEmergence(dtype=dtype)
        
    def create_morphism(self, source: Tensor, target: Tensor) -> EnrichedMorphism:
        """Create an enriched morphism between pattern spaces.
        
        Args:
            source: Source pattern tensor
            target: Target pattern tensor
            
        Returns:
            Enriched morphism between spaces
        """
        # Create structure map using wave evolution
        direction = self._compute_direction(source, target)
        
        # Ensure proper shape for structure map
        if len(source.shape) == 2:
            # For matrix inputs, create transformation matrix
            structure = torch.eye(source.shape[0], device=source.device, dtype=source.dtype)
            evolved = self.wave.evolve_structure(source, direction)
            
            # Handle evolved tensor shape
            n = source.shape[0]  # Number of rows
            evolved_flat = evolved.reshape(-1)  # Flatten completely
            
            # Pad or trim evolved tensor to match source size
            target_size = source.shape[0] * source.shape[1]  # Total size needed
            if evolved_flat.numel() < target_size:
                # Pad with zeros
                padded = torch.zeros(target_size, device=evolved.device, dtype=evolved.dtype)
                padded[:evolved_flat.numel()] = evolved_flat
                evolved_flat = padded
            else:
                # Trim to size
                evolved_flat = evolved_flat[:target_size]
            
            # Reshape to match source dimensions
            evolved = evolved_flat.reshape(source.shape)
            
            # Compute transformation using evolved state
            evolved_flat = evolved.reshape(evolved.shape[0], -1)  # (n, m)
            source_flat = source.reshape(source.shape[0], -1)     # (n, m)
            
            # Compute transformation matrix using pseudoinverse
            pinv = torch.pinverse(source_flat)                    # (m, n)
            structure = torch.matmul(evolved_flat, pinv)          # (n, n)
            
            # Ensure structure is square
            if structure.shape[0] != structure.shape[1]:
                n = min(structure.shape[0], structure.shape[1])
                structure = structure[:n, :n]

        else:
            # For vector inputs, use direct evolution
            structure = self.wave.evolve_structure(source, direction)

        return EnrichedMorphism(
            source_space=source,
            target_space=target,
            structure_map=structure
        )
    
    def compose(self, first: EnrichedMorphism, second: EnrichedMorphism) -> EnrichedMorphism:
        """Compose enriched morphisms.
        
        Args:
            first: First morphism to compose
            second: Second morphism to compose
            
        Returns:
            Composed enriched morphism
        """
        # Ensure tensors have same shape for comparison
        first_target = first.target_space.reshape(-1)
        second_source = second.source_space.reshape(-1)
        
        # Pad shorter tensor if needed
        if first_target.numel() < second_source.numel():
            padded = torch.zeros_like(second_source)
            padded[:first_target.numel()] = first_target
            first_target = padded
        elif second_source.numel() < first_target.numel():
            padded = torch.zeros_like(first_target)
            padded[:second_source.numel()] = second_source
            second_source = padded
            
        # Check if tensors match after normalization
        if not torch.allclose(first_target, second_source):
            raise ValueError("Morphisms cannot be composed: target space of first morphism does not match source space of second morphism")

        # Compose structure maps
        composed_structure = torch.matmul(second.structure_map, first.structure_map)

        return EnrichedMorphism(
            source_space=first.source_space,
            target_space=second.target_space,
            structure_map=composed_structure
        )
    
    def _compute_direction(self, source: Tensor, target: Tensor) -> Tensor:
        """Compute optimal direction for wave evolution.
        
        Args:
            source: Source pattern tensor
            target: Target pattern tensor
            
        Returns:
            Direction tensor for wave evolution
        """
        # Handle different input shapes
        if len(source.shape) == 2 and len(target.shape) == 2:
            # Matrix inputs
            if source.shape[1] != target.shape[1]:
                # Different dimensions - use padding
                min_dim = min(source.shape[1], target.shape[1])
                padded_source = source[:, :min_dim]
                padded_target = target[:, :min_dim]
                projection = torch.matmul(padded_target, padded_source.transpose(-2, -1))
            else:
                # Same dimensions
                projection = torch.matmul(target, source.transpose(-2, -1))
        else:
            # Vector inputs
            if len(source.shape) == 1:
                source = source.unsqueeze(0)
            if len(target.shape) == 1:
                target = target.unsqueeze(0)
                
            # Ensure compatible dimensions
            min_dim = min(source.shape[-1], target.shape[-1])
            source = source[..., :min_dim]
            target = target[..., :min_dim]
            
            # Compute projection
            projection = torch.sum(target * source, dim=-1, keepdim=True)
        
        # Normalize direction
        direction = projection / torch.norm(projection, dim=-1, keepdim=True).clamp(min=1e-6)
        
        return direction 