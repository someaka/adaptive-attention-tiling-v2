from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from src.core.crystal.scale_classes.complextanh import ComplexTanh



class ScaleInvariance:
    """Implementation of scale invariance detection."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale invariance detector.
        
        Args:
            dim: Dimension of the space
            num_scales: Number of scale levels
            dtype: Data type for tensors
        """
        self.dim = dim
        self.num_scales = num_scales
        self.dtype = dtype

        # Initialize scale transformation networks
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()
        self.scale_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2, dtype=dtype),
                activation,
                nn.Linear(dim * 2, dim, dtype=dtype)
            ) for _ in range(num_scales - 1)
        ])

    def check_invariance(self, state: torch.Tensor, scale: float) -> bool:
        """Check if state is invariant under scale transformation."""
        # Ensure state has correct shape and dtype
        if state.dim() > 2:
            state = state.reshape(-1, self.dim)
        elif state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Convert to correct dtype
        state = state.to(dtype=self.dtype)
        
        # Get scale index
        scale_idx = int(np.log2(scale))
        if scale_idx >= len(self.scale_transform):
            return False
            
        # Apply transformation
        transformed = self.scale_transform[scale_idx](state)
        
        # Check invariance with tolerance
        diff = torch.norm(transformed - state)
        tolerance = 1e-4 * torch.norm(state)
        
        # Convert tensor comparison to boolean
        return bool((diff < tolerance).item())

    def find_invariant_structures(self, states: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Find scale invariant structures and their scale factors."""
        # Ensure states have correct shape
        if states.dim() > 2:
            states = states.reshape(-1, self.dim)
        elif states.dim() == 1:
            states = states.unsqueeze(0)
            
        invariants = []
        for state in states:
            for scale in [2**i for i in range(self.num_scales)]:
                if self.check_invariance(state, scale):
                    invariants.append((state, scale))

        return invariants
