"""Pattern Dynamics Implementation for Neural Attention.

This module has been refactored into the pattern/ directory for better organization.
This file now serves as a compatibility layer for existing code.

The implementation is split into:
- models.py: Data models and state classes
- diffusion.py: Diffusion system implementation
- reaction.py: Reaction system implementation
- stability.py: Stability analysis
- dynamics.py: Main pattern dynamics implementation
"""

from typing import List, Optional, Tuple, Callable, Union
import torch

from .pattern.models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationPoint,
    BifurcationDiagram
)

from .pattern.dynamics import PatternDynamics as _PatternDynamics


class PatternDynamics(_PatternDynamics):
    """Pattern dynamics system with attention-specific features."""
    
    def __init__(
        self,
        dim: int = 2,  # Default: activator-inhibitor system
        size: int = 32,  # Default grid size
        dt: float = 0.01,
        boundary: str = "periodic",
        hidden_dim: int = 64,
        num_modes: int = 8,
    ):
        """Initialize pattern dynamics system.
        
        Args:
            dim: Number of channels/species (default: 2 for activator-inhibitor)
            size: Grid size for spatial patterns
            dt: Time step for evolution
            boundary: Boundary condition type
            hidden_dim: Hidden layer dimension for neural networks
            num_modes: Number of stability modes to analyze
        """
        super().__init__(
            dim=dim,
            size=size,
            dt=dt,
            boundary=boundary,
            hidden_dim=hidden_dim,
            num_modes=num_modes
        )


# Re-export all the public classes and functions
__all__ = [
    'ReactionDiffusionState',
    'StabilityInfo',
    'StabilityMetrics',
    'ControlSignal',
    'BifurcationPoint',
    'BifurcationDiagram',
    'PatternDynamics'
]
