"""Configuration for holographic network testing."""

from dataclasses import dataclass

@dataclass
class HolographicTestConfig:
    """Configuration for holographic network testing."""
    dim: int = 4
    hidden_dim: int = 16
    n_layers: int = 3
    dtype: str = "complex64"
    z_uv: float = 0.1
    z_ir: float = 10.0
    eps: float = 1e-5 