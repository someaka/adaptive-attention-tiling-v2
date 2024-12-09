"""Compatibility module for old imports."""

from .quantum_attention_tile import LoadBalancer, LoadProfile
from .quantum_geometric_attention import ResolutionAdapter
from .state_manager import StateManager

__all__ = [
    "LoadBalancer",
    "LoadProfile",
    "ResolutionAdapter",
    "StateManager",
]
