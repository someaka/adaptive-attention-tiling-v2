"""Base attention mechanism definitions."""

from typing import Optional, Protocol

import torch

from ..backends.base import AttentionBackend


class BaseAttention(Protocol):
    """Base protocol for attention mechanisms."""

    backend: AttentionBackend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...
