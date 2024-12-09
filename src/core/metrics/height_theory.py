"""Height Theory for Pattern Analysis.

This module implements arithmetic height theory for analyzing patterns:
- Local height computation
- Prime base structure
- Canonical heights
- Growth analysis
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

class HeightStructure:
    """Implementation of arithmetic height theory."""
    
    def __init__(
        self,
        num_primes: int = 8,
        base_field: str = 'real'  # 'real' or 'p-adic'
    ):
        self.num_primes = num_primes
        self.base_field = base_field
        
        # Initialize prime bases
        self.prime_bases = torch.tensor(
            [2, 3, 5, 7, 11, 13, 17, 19][:num_primes],
            dtype=torch.float32
        )
        
        # Initialize local height functions
        self.local_heights = {
            p: self._init_local_height(p) 
            for p in self.prime_bases
        }
    
    def _init_local_height(self, prime: int) -> nn.Module:
        """Initialize local height computation for given prime."""
        if self.base_field == 'real':
            return nn.Sequential(
                nn.Linear(1, prime),
                nn.ReLU(),
                nn.Linear(prime, 1)
            )
        else:  # p-adic
            return lambda x: torch.log(torch.abs(x)) / math.log(prime)
    
    def compute_local_height(
        self,
        point: torch.Tensor,
        prime: int
    ) -> torch.Tensor:
        """Compute local height at given prime."""
        if self.base_field == 'real':
            return self.local_heights[prime](point.unsqueeze(-1)).squeeze(-1)
        else:
            return self.local_heights[prime](point)
    
    def compute_canonical_height(
        self,
        point: torch.Tensor
    ) -> torch.Tensor:
        """Compute canonical height combining all local heights."""
        local_contributions = []
        for p in self.prime_bases:
            local_height = self.compute_local_height(point, p)
            local_contributions.append(local_height)
        
        return torch.sum(torch.stack(local_contributions, dim=-1), dim=-1)
    
    def analyze_growth(
        self,
        points: List[torch.Tensor],
        window_size: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Analyze height growth over sequence of points."""
        heights = torch.stack([
            self.compute_canonical_height(p) for p in points
        ])
        
        # Compute growth statistics
        growth_rate = (heights[1:] - heights[:-1]).mean()
        volatility = heights.std()
        
        # Compute moving averages
        if len(heights) >= window_size:
            ma = torch.conv1d(
                heights.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, window_size) / window_size,
                padding='valid'
            ).squeeze()
        else:
            ma = heights.mean().unsqueeze(0)
        
        return {
            'heights': heights,
            'growth_rate': growth_rate,
            'volatility': volatility,
            'moving_avg': ma
        }

class AdaptiveHeightTheory:
    """Adaptive height theory with learning capabilities."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_primes: int = 8,
        learning_rate: float = 0.001
    ):
        self.hidden_dim = hidden_dim
        self.height_structure = HeightStructure(num_primes)
        
        # Learnable components
        self.height_projection = nn.Linear(hidden_dim, num_primes)
        self.optimizer = torch.optim.Adam(
            self.height_projection.parameters(),
            lr=learning_rate
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute adaptive heights with statistics."""
        # Project to prime basis
        prime_coords = self.height_projection(x)
        
        # Compute heights
        heights = self.height_structure.compute_canonical_height(prime_coords)
        
        # Analyze growth if we have history
        if hasattr(self, 'history'):
            growth_stats = self.height_structure.analyze_growth(
                self.history + [prime_coords]
            )
        else:
            growth_stats = {
                'heights': heights,
                'growth_rate': torch.tensor(0.),
                'volatility': torch.tensor(0.),
                'moving_avg': heights.mean().unsqueeze(0)
            }
        
        # Update history
        self.history = self.history[-9:] + [prime_coords] if hasattr(self, 'history') else [prime_coords]
        
        return heights, growth_stats
    
    def adapt(
        self,
        loss: torch.Tensor
    ) -> None:
        """Adapt height computation based on loss."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
