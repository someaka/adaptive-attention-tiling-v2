"""Attention state management for quantum geometric attention."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch

from src.core.quantum.state_space import QuantumState
from src.core.tiling.state_manager import StateConfig, StateManager, StateType


@dataclass
class AttentionState:
    """Attention state with quantum geometric properties."""
    state_manager: StateManager
    geometric_state: torch.Tensor
    attention_scores: Optional[torch.Tensor] = None
    attention_patterns: Dict[str, torch.Tensor] = field(default_factory=dict)
    entanglement_history: Dict[str, list] = field(default_factory=dict)
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize state manager with geometric state."""
        # Initialize states in state manager if not already initialized
        if "input" not in self.state_manager.states:
            self.state_manager.initialize_state("input")
        if "manifold" not in self.state_manager.states:
            self.state_manager.initialize_state("manifold")
            
        # Update states with geometric state
        self.state_manager.states["manifold"].copy_(self.geometric_state)
        
    def validate_state(self, state: torch.Tensor) -> bool:
        """Validate tensor properties and normalization."""
        if not isinstance(state, torch.Tensor):
            return False
        if state.dim() != self.geometric_state.dim():
            return False
        if torch.isnan(state).any() or torch.isinf(state).any():
            return False
        if state.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            return False
        
        # Check dimensions match geometric state
        if state.shape != self.geometric_state.shape:
            return False
        
        return True

    def update_quantum_state(self, key: str, state: torch.Tensor) -> QuantumState:
        """Update quantum state through state manager.
        
        The quantum state is normalized globally across all dimensions except batch
        to maintain proper quantum mechanical properties.
        """
        # Validate state tensor
        if not self.validate_state(state):
            raise ValueError("Invalid state tensor")

        # Initialize state if it doesn't exist
        if key not in self.state_manager.states:
            self.state_manager.initialize_state(key)

        # Update state in manager
        self.state_manager.update_state(key, state)

        # Store original norms before normalization
        # Global normalization across all dimensions except batch
        original_norms = torch.sqrt(torch.sum(torch.abs(state) ** 2, 
                                            dim=tuple(range(1, len(state.shape))), 
                                            keepdim=True))  # (batch, 1, 1, 1)
        
        # Global normalization across all dimensions except batch
        normalized_state = state / original_norms.clamp(min=1e-8)

        # Convert input state to quantum state
        quantum_state = QuantumState(
            amplitudes=normalized_state.to(torch.complex128),
            basis_labels=[str(i) for i in range(state.shape[-1])],
            phase=torch.zeros(1, dtype=torch.complex128, device=state.device),
            original_norm=original_norms.to(torch.float64)
        )

        return quantum_state

    def track_entanglement(self, source_scale: float, target_scale: float, entropy: torch.Tensor):
        """Track entanglement between scales."""
        if not isinstance(entropy, torch.Tensor) or entropy.numel() != 1:
            raise ValueError("Entropy must be a scalar tensor")
            
        key = f"{source_scale:.1f}->{target_scale:.1f}"
        if key not in self.entanglement_history:
            self.entanglement_history[key] = []
        self.entanglement_history[key].append(float(entropy.item()))

    def update_metrics(self, key: str, value: torch.Tensor):
        """Update metrics dictionary."""
        self.metrics[key] = value

    def update_attention_pattern(self, key: str, pattern: torch.Tensor):
        """Update attention pattern dictionary."""
        self.attention_patterns[key] = pattern

    @classmethod
    def initialize(
        cls,
        hidden_dim: int,
        num_heads: int = 8,
        batch_size: int = 1,
        seq_length: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')
    ) -> 'AttentionState':
        """Initialize attention state with given dimensions."""
        if hidden_dim <= 0 or num_heads <= 0 or batch_size <= 0 or seq_length <= 0:
            raise ValueError("All dimensions must be positive")
            
        # Create state config
        config = StateConfig(
            dim=hidden_dim,
            type=StateType.PURE,
            epsilon=1e-5,
            max_entanglement=1.0,
            dtype=dtype
        )
        
        # Initialize state manager
        state_manager = StateManager(config=config)
        
        # Initialize geometric state
        geometric_state = torch.randn(
            batch_size, num_heads, seq_length, hidden_dim,
            dtype=dtype, device=device
        )
        geometric_state = geometric_state / torch.norm(geometric_state, dim=-1, keepdim=True)
        
        return cls(state_manager=state_manager, geometric_state=geometric_state) 