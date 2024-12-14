"""Configuration for Quantum Geometric Attention.

This module defines the configuration parameters for the quantum geometric attention framework.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GeometricFlowConfig:
    """Configuration for geometric flow."""

    dim: int = 64
    num_heads: int = 8
    dropout: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    max_position_embeddings_geometric: int = 512

    # Geometric flow parameters
    flow_epsilon: float = 1e-6
    flow_steps: int = 10
    flow_lr: float = 0.01
    curvature_threshold: float = 0.1
    stability_threshold: float = 0.05
    min_delta: float = 1e-8
    max_delta: float = 1.0
    MIN_RESOLUTION: float = 1e-4  # Minimum resolution for geometric computations
    MIN_DENSITY: float = 0.1  # Minimum density threshold for attention patterns
    MAX_DENSITY: float = 0.9  # Maximum density threshold for attention patterns

    # Quantum parameters
    quantum_dim: int = 32
    entanglement_threshold: float = 0.8
    measurement_samples: int = 100

    # Optimization parameters
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Architecture parameters
    num_hidden_layers: int = 12
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps_geometric: float = 1e-12

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GeometricFlowConfig":
        """Create config from dictionary."""
        return cls(
            **{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        )


# Global configuration instance
CONFIG = GeometricFlowConfig()
