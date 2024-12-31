"""Configuration for holographic neural networks and testing."""

from dataclasses import dataclass
import torch


@dataclass
class HolographicTestConfig:
    """Configuration for holographic testing and training."""
    
    # Model parameters
    dim: int = 4
    hidden_dim: int = 64
    n_layers: int = 3
    dtype: torch.dtype = torch.complex64
    z_uv: float = 0.1
    z_ir: float = 10.0
    
    # Training parameters
    batch_size: int = 128
    n_warmup: int = 100
    n_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 50
    min_delta: float = 1e-6
    
    # Loss weights
    basic_loss_weight: float = 1.0
    quantum_loss_weight: float = 0.1
    phase_loss_weight: float = 0.01
    
    # Test thresholds
    norm_preservation_threshold: float = 1e-4
    convergence_threshold: float = 1e-3
    quantum_correction_threshold: float = 1e-4
    scaling_error_threshold: float = 1e-4
    
    # Holographic test thresholds
    uv_boundary_threshold: float = 1e-4
    radial_scaling_threshold: float = 1e-4
    reconstruction_threshold: float = 1e-4
    c_theorem_threshold: float = 1e-6
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    pretrained_model_path: str = "pretrained/holographic_net.pt" 