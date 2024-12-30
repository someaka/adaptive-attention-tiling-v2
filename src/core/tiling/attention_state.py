"""Advanced state management for quantum geometric attention.

This module provides the enhanced state management system for the quantum geometric attention framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.core.tiling.state_manager import StateManager, StateConfig, StateType
from src.core.quantum.state_space import QuantumState


@dataclass
class AttentionState:
    """Enhanced state management for quantum geometric attention."""
    
    # Core quantum state management
    state_manager: StateManager
    
    # Geometric components
    geometric_state: torch.Tensor
    manifold_state: torch.Tensor
    
    # Attention components
    attention_scores: Optional[torch.Tensor] = None
    attention_patterns: Optional[Dict[str, torch.Tensor]] = None
    
    # Entanglement tracking
    entanglement_history: Optional[Dict[str, List[float]]] = None
    
    # Metrics and validation
    metrics: Optional[Dict[str, torch.Tensor]] = None
    
    @classmethod
    def initialize(
        cls,
        hidden_dim: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> 'AttentionState':
        """Initialize a new attention state with proper configuration."""
        config = StateConfig(
            dim=hidden_dim,
            type=StateType.PURE,
            dtype=dtype
        )
        
        state_manager = StateManager(config, device)
        
        geometric_state = torch.zeros(
            (num_heads, hidden_dim),
            dtype=dtype,
            device=device
        )
        
        manifold_state = torch.zeros(
            (num_heads, hidden_dim),
            dtype=dtype,
            device=device
        )
        
        return cls(
            state_manager=state_manager,
            geometric_state=geometric_state,
            manifold_state=manifold_state,
            attention_scores=None,
            attention_patterns={},
            entanglement_history={},
            metrics={}
        )
    
    def update_quantum_state(
        self,
        key: str,
        state: torch.Tensor,
        learning_rate: float = 0.01
    ) -> torch.Tensor:
        """Update quantum state through state manager."""
        return self.state_manager.update_state(key, state, learning_rate)
    
    def track_entanglement(
        self,
        source_scale: float,
        target_scale: float,
        entropy: torch.Tensor
    ) -> None:
        """Track entanglement between scales."""
        self.state_manager.update_entanglement(
            source_scale,
            target_scale,
            entropy
        )
        
    def get_entanglement_metrics(
        self,
        source_scale: Optional[float] = None,
        target_scale: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """Get entanglement history and metrics."""
        return self.state_manager.get_entanglement_history(
            source_scale,
            target_scale
        )
    
    def validate_state(self, state: torch.Tensor) -> bool:
        """Validate quantum state through state manager."""
        return self.state_manager.validate_state(state)
    
    def update_attention_pattern(
        self,
        key: str,
        pattern: torch.Tensor
    ) -> None:
        """Track attention pattern history."""
        if self.attention_patterns is None:
            self.attention_patterns = {}
        self.attention_patterns[key] = pattern
        
    def update_metrics(
        self,
        metric_name: str,
        value: torch.Tensor
    ) -> None:
        """Update state metrics."""
        if self.metrics is None:
            self.metrics = {}
        self.metrics[metric_name] = value 