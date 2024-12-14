"""Backend interface definitions."""

from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol

import torch


class ResolutionStrategy(Enum):
    """Strategy for tile resolution adaptation."""

    FIXED = auto()
    ADAPTIVE = auto()
    DYNAMIC = auto()


class AttentionTile(Protocol):
    """Protocol for attention tiles."""

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...

    def adapt_resolution(
        self, strategy: ResolutionStrategy, scale_factor: float
    ) -> None: ...

    def get_memory_stats(self) -> Dict[str, float]: ...

    @property
    def neighbors(self) -> List["AttentionTile"]: ...


class ResourceProfile:
    """Profile of computational resources."""

    def __init__(
        self, compute_limit: float, memory_limit: float, min_resolution: float
    ):
        self.compute_limit = compute_limit
        self.memory_limit = memory_limit
        self.min_resolution = min_resolution


class AttentionBackend(Protocol):
    """Protocol defining the interface for attention backends."""

    def prepare_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tile_size: int,
    ) -> dict[str, Any]: ...

    def compute_attention(
        self, prepared_inputs: dict[str, Any], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...

    def optimize_tiling(
        self, sequence_length: int, batch_size: int, num_heads: int
    ) -> dict[str, int]: ...

    def cleanup(self) -> None: ...
