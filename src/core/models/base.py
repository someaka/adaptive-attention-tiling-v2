"""Base classes for model geometry.

This module provides base classes for representing geometric properties of neural models:
- ModelGeometry: Base class for model geometric structure
- LayerGeometry: Base class for layer-specific geometry
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..patterns.riemannian import PatternRiemannianStructure


class LayerGeometry(nn.Module):
    """Base class for layer geometry."""

    def __init__(self, manifold_dim: int):
        """Initialize layer geometry.
        
        Args:
            manifold_dim: Dimension of the layer manifold
        """
        super().__init__()
        self.manifold_dim = manifold_dim
        self.metric_tensor = nn.Parameter(torch.eye(manifold_dim))
        self.connection_coeffs = nn.Parameter(
            torch.zeros(manifold_dim, manifold_dim, manifold_dim)
        )

    def metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor (batch_size x dim)
            
        Returns:
            Metric tensor values (batch_size x dim x dim)
        """
        batch_size = points.shape[0]
        return self.metric_tensor.expand(batch_size, -1, -1)

    def connection(self, points: torch.Tensor) -> torch.Tensor:
        """Compute connection coefficients at points.
        
        Args:
            points: Points tensor (batch_size x dim)
            
        Returns:
            Connection coefficients (batch_size x dim x dim x dim)
        """
        batch_size = points.shape[0]
        return self.connection_coeffs.expand(batch_size, -1, -1, -1)

    def get_riemannian_framework(self, points: torch.Tensor) -> PatternRiemannianStructure:
        """Get Riemannian framework for this layer.
        
        Args:
            points: Points tensor
            
        Returns:
            PatternRiemannianStructure instance
        """
        framework = PatternRiemannianStructure(
            manifold_dim=self.manifold_dim
        )
        
        # Set metric and connection from layer parameters
        framework.metric_factors = nn.Parameter(self.metric_tensor)
        framework.connection_coeffs = nn.Parameter(self.connection_coeffs)
        
        return framework


class ModelGeometry(nn.Module):
    """Base class for model geometry."""

    def __init__(
        self,
        manifold_dim: int,
        query_dim: int,
        key_dim: int,
        layers: Optional[Dict[str, LayerGeometry]] = None,
        attention_heads: Optional[List[nn.Module]] = None
    ):
        """Initialize model geometry.
        
        Args:
            manifold_dim: Dimension of model manifold
            query_dim: Dimension of query space
            key_dim: Dimension of key space
            layers: Dictionary of layer geometries
            attention_heads: List of attention head modules
        """
        super().__init__()
        self.manifold_dim = manifold_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        
        self.layers = nn.ModuleDict(layers or {})
        self.attention_heads = nn.ModuleList(attention_heads or [])

    def add_layer(self, name: str, layer: LayerGeometry):
        """Add a layer geometry.
        
        Args:
            name: Layer name
            layer: Layer geometry
        """
        self.layers[name] = layer

    def add_attention_head(self, head: nn.Module):
        """Add an attention head.
        
        Args:
            head: Attention head module
        """
        self.attention_heads.append(head)

    def get_layer(self, name: str) -> LayerGeometry:
        """Get layer geometry by name.
        
        Args:
            name: Layer name
            
        Returns:
            Layer geometry
        """
        if name not in self.layers:
            raise ValueError(f"Unknown layer: {name}")
        return self.layers[name]

    def get_attention_head(self, idx: int) -> nn.Module:
        """Get attention head by index.
        
        Args:
            idx: Head index
            
        Returns:
            Attention head module
        """
        if idx >= len(self.attention_heads):
            raise ValueError(f"Invalid head index: {idx}")
        return self.attention_heads[idx]
