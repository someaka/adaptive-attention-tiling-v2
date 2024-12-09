"""Compatibility module for old imports."""

from .state_manager import StateManager
from .quantum_geometric_attention import ResolutionAdapter
from .quantum_attention_tile import LoadBalancer, LoadProfile

__all__ = [
    'LoadBalancer',
    'LoadProfile',
    'ResolutionAdapter',
    'StateManager',
]
