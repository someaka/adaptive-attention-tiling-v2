"""Attention computation module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AttentionCompute(nn.Module):
    """Compute attention scores and outputs."""
    
    def __init__(self, dropout: float = 0.0):
        """Initialize attention compute.
        
        Args:
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = dropout
    
    def compute_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention scores.
        
        Args:
            query: Query tensor [batch, heads, seq_len, dim]
            key: Key tensor [batch, heads, seq_len, dim] 
            mask: Optional attention mask
            scale: Optional scaling factor
            
        Returns:
            Attention scores [batch, heads, seq_len, seq_len]
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Apply scaling
        if scale is not None:
            scores = scores * scale
            
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax
        scores = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout > 0:
            scores = F.dropout(scores, p=self.dropout, training=self.training)
            
        return scores
        
    def compute_output(
        self,
        scores: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention output.
        
        Args:
            scores: Attention scores [batch, heads, seq_len, seq_len]
            value: Value tensor [batch, heads, seq_len, dim]
            
        Returns:
            Attention output [batch, heads, seq_len, dim]
        """
        return torch.matmul(scores, value)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward pass for attention computation.
        
        Args:
            query: Query tensor [batch, seq_len, dim]
            key: Key tensor [batch, seq_len, dim]
            value: Value tensor [batch, seq_len, dim]
            mask: Optional attention mask
            scale: Optional scaling factor. If None, uses 1/sqrt(dim)
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        # Add head dimension if not present
        if query.dim() == 3:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        # Compute default scale if not provided
        if scale is None:
            scale = float(1.0 / torch.sqrt(torch.tensor(query.size(-1), dtype=query.dtype)))

        # Compute attention scores
        scores = self.compute_scores(query, key, mask, scale)

        # Compute attention output
        output = self.compute_output(scores, value)

        # Remove head dimension if it was added
        if output.size(1) == 1:
            output = output.squeeze(1)

        return output
