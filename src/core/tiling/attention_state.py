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
    key_padding_mask: Optional[torch.Tensor] = None  # Shape: [batch_size, seq_length]
    attention_mask: Optional[torch.Tensor] = None  # Shape: [seq_length, seq_length] or [batch_size, num_heads, seq_length, seq_length]

    def __post_init__(self):
        """Initialize state manager with geometric state."""
        # Initialize states in state manager if not already initialized
        if "input" not in self.state_manager.states:
            self.state_manager.initialize_state("input", shape=self.geometric_state.shape)
        if "manifold" not in self.state_manager.states:
            self.state_manager.initialize_state("manifold", shape=self.geometric_state.shape)
            
        # Update states with geometric state
        self.state_manager.states["manifold"].copy_(self.geometric_state)

        # Get dimensions from debug info if available
        debug_info = self.state_manager.states.get("debug_info", {})
        num_heads = debug_info.get("num_heads", 1)
        
        # Get shape info
        if self.geometric_state.dim() == 4:
            batch_size, num_heads, seq_length, _ = self.geometric_state.shape
        else:  # 3D tensor with combined batch and heads
            combined_batch, seq_length, _ = self.geometric_state.shape
            batch_size = combined_batch // num_heads

        # Initialize masks if not provided
        if self.key_padding_mask is None:
            # [batch_size, seq_length] - all True by default
            self.key_padding_mask = torch.ones(
                batch_size, seq_length,
                dtype=torch.bool,
                device=self.geometric_state.device
            )
        if self.attention_mask is None:
            # [batch_size, num_heads, seq_length, seq_length] - all True by default
            self.attention_mask = torch.ones(
                batch_size, num_heads, seq_length, seq_length,
                dtype=torch.bool,
                device=self.geometric_state.device
            )

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

        # Create QuantumState to handle normalization
        quantum_state = QuantumState(
            amplitudes=state.to(torch.complex128),
            basis_labels=[str(i) for i in range(state.shape[-1])],
            phase=torch.zeros(1, dtype=torch.complex128, device=state.device),
            original_norm=None  # Let QuantumState compute and store this
        )

        # Update state in manager with normalized amplitudes
        self.state_manager.update_state(key, quantum_state.amplitudes)

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

    def apply_masks(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Apply key padding and attention masks to attention scores.
        
        Args:
            attention_scores: Attention scores tensor [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Masked attention scores
        """
        # Get batch size and num_heads from attention scores shape
        batch_size, num_heads = attention_scores.shape[:2]
        
        # Apply key padding mask if it exists
        if self.key_padding_mask is not None:
            # Expand key padding mask to match attention scores shape
            key_padding_mask = self.key_padding_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            key_padding_mask = key_padding_mask.expand(-1, num_heads, attention_scores.size(2), -1)  # [batch_size, num_heads, seq_len, seq_len]
            
            # Create real -inf value since we only care about the real part for masking
            neg_inf = float('-inf')
            attention_scores = torch.where(key_padding_mask, attention_scores, neg_inf)

        # Apply attention mask if it exists
        if self.attention_mask is not None:
            # If attention mask is 2D [seq_len, seq_len], expand it
            if self.attention_mask.dim() == 2:
                mask = self.attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                mask = mask.expand(batch_size, num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
            else:
                # If 4D [batch_size, num_heads, seq_len, seq_len], use as is
                mask = self.attention_mask
            
            # Create real -inf value since we only care about the real part for masking
            neg_inf = float('-inf')
            attention_scores = torch.where(mask, attention_scores, neg_inf)

        return attention_scores

    def set_key_padding_mask(self, mask: torch.Tensor):
        """Set the key padding mask.
        
        Args:
            mask: Boolean tensor of shape [batch_size, seq_length] where True indicates valid tokens
        """
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("Key padding mask must be a boolean tensor")
        if mask.dim() != 2:
            raise ValueError("Key padding mask must be 2-dimensional [batch_size, seq_length]")
        
        # Get dimensions from debug info if available
        debug_info = self.state_manager.states.get("debug_info", {})
        input_shape = debug_info.get("input_shape")
        if input_shape:
            batch_size = input_shape[0]
            seq_length = input_shape[2] if len(input_shape) == 4 else input_shape[1]
        else:
            # Fallback to geometric state shape
            if self.geometric_state.dim() == 4:
                batch_size, _, seq_length, _ = self.geometric_state.shape
            else:
                combined_batch, seq_length, _ = self.geometric_state.shape
                num_heads = debug_info.get("num_heads", 1)
                batch_size = combined_batch // num_heads
            
        if mask.shape != (batch_size, seq_length):
            raise ValueError("Key padding mask sequence length must match geometric state")
            
        self.key_padding_mask = mask

    def set_attention_mask(self, mask: torch.Tensor):
        """Set the attention mask.
        
        Args:
            mask: Boolean tensor of shape [seq_length, seq_length] or 
                 [batch_size, num_heads, seq_length, seq_length] where True indicates allowed attention
        """
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("Attention mask must be a boolean tensor")
        if mask.dim() not in [2, 4]:
            raise ValueError("Attention mask must be 2D or 4D")
            
        # Get dimensions from debug info if available
        debug_info = self.state_manager.states.get("debug_info", {})
        input_shape = debug_info.get("input_shape")
        if input_shape:
            batch_size = input_shape[0]
            num_heads = input_shape[1] if len(input_shape) == 4 else 1
            seq_length = input_shape[2] if len(input_shape) == 4 else input_shape[1]
        else:
            # Fallback to geometric state shape
            if self.geometric_state.dim() == 4:
                batch_size, num_heads, seq_length, _ = self.geometric_state.shape
            else:
                combined_batch, seq_length, _ = self.geometric_state.shape
                num_heads = debug_info.get("num_heads", 1)
                batch_size = combined_batch // num_heads
            
        if mask.dim() == 2:
            if mask.shape != (seq_length, seq_length):
                raise ValueError(f"2D attention mask must have shape [{seq_length}, {seq_length}], got {mask.shape}")
        else:  # 4D
            if mask.shape != (batch_size, num_heads, seq_length, seq_length):
                raise ValueError(f"4D attention mask must have shape [{batch_size}, {num_heads}, {seq_length}, {seq_length}], got {mask.shape}")
                
        self.attention_mask = mask

    @classmethod
    def initialize(
        cls,
        hidden_dim: int,
        num_heads: int = 8,
        batch_size: int = 1,
        seq_length: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
        causal: bool = False
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
        
        # Create causal mask if requested
        attention_mask = None
        if causal:
            # Create base causal mask [seq_length, seq_length]
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
                diagonal=1
            ).logical_not()
            # Expand to [batch_size, num_heads, seq_length, seq_length]
            attention_mask = causal_mask.expand(batch_size, num_heads, seq_length, seq_length)
        
        return cls(
            state_manager=state_manager,
            geometric_state=geometric_state,
            attention_mask=attention_mask
        ) 