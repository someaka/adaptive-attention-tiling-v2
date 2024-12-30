import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention

class QuantumGeometricTransformer(nn.Module):
    """Transformer using quantum geometric attention."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int = 8,
        tile_size: int = 64,
        manifold_type: str = "hyperbolic",
        curvature: float = -1.0,
    ):
        super().__init__()
        
        # Attention layers
        self.layers = nn.ModuleList(
            [
                QuantumGeometricAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    tile_size=tile_size,
                    manifold_type=manifold_type,
                    curvature=curvature,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, return_patterns: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        """Apply quantum geometric transformer.

        Args:
            x: Input tensor
            return_patterns: Whether to return pattern metrics

        Returns:
            Tuple containing:
            - Processed tensor
            - Optional list of pattern metrics from each layer
        """
        metrics_list: List[Dict[str, Any]] = []

        for layer in self.layers:
            # Apply attention with residual
            attended, metrics = layer(x, return_patterns=return_patterns)
            x = x + attended

            # Apply normalization
            x = self.norm(x)

            if return_patterns:
                metrics_list.append(metrics)

        if return_patterns:
            return x, metrics_list

        return x, None
