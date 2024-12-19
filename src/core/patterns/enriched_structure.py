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
    
    def __init__(self, dt: float = 0.1, num_steps: int = 10):
        """Initialize wave emergence structure.
        
        Args:
            dt: Time step for wave evolution
            num_steps: Number of evolution steps
        """
        self.dt = dt
        self.num_steps = num_steps
        
    def evolve_structure(self, pattern: Tensor, direction: Tensor) -> Tensor:
        """Evolve pattern structure using wave equation.
        
        Args:
            pattern: Input pattern tensor
            direction: Evolution direction tensor
            
        Returns:
            Evolved pattern tensor
        """
        # Initialize wave components
        u = pattern
        v = torch.zeros_like(pattern)  # Velocity field
        
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
            
        return u
    
    def _laplacian(self, tensor: Tensor) -> Tensor:
        """Compute discrete Laplacian of tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Laplacian of input tensor
        """
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
    
    def __init__(self, wave_emergence: Optional[WaveEmergence] = None):
        """Initialize pattern transition structure.
        
        Args:
            wave_emergence: Wave emergence structure for natural transitions
        """
        self.wave = wave_emergence or WaveEmergence()
        
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
            structure = torch.eye(source.shape[0], device=source.device)
            evolved = self.wave.evolve_structure(source, direction)
            structure = torch.matmul(evolved, torch.pinverse(source))
        else:
            # For vector inputs, use wave evolution directly
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
        if not torch.allclose(first.target_space, second.source_space):
            raise ValueError("Morphisms not composable")
            
        # Ensure proper matrix shapes for composition
        if len(first.structure_map.shape) == 2:
            # For matrix structure maps
            composed_structure = torch.matmul(second.structure_map, first.structure_map)
        else:
            # For vector structure maps, use element-wise composition
            composed_structure = second.structure_map * first.structure_map
        
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