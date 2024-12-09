"""Backend interface definitions."""
from typing import Protocol, Dict, Any, Optional
import torch

class AttentionBackend(Protocol):
    """Protocol defining the interface for attention backends."""
    
    def prepare_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tile_size: int
    ) -> Dict[str, Any]: ...
    
    def compute_attention(
        self,
        prepared_inputs: Dict[str, Any],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
    
    def optimize_tiling(
        self,
        sequence_length: int,
        batch_size: int,
        num_heads: int
    ) -> Dict[str, int]: ...
