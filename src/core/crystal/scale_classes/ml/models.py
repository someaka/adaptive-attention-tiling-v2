import torch
import torch.nn as nn
from typing import Optional, Tuple

class ComplexNorm(nn.Module):
    """Custom normalization layer for complex numbers."""
    def __init__(self, num_features: int, dtype: torch.dtype = torch.complex64, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.dtype = dtype
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance over batch dimension
        mean = x.mean(dim=0, keepdim=True)
        var = torch.mean(torch.abs(x - mean)**2, dim=0, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[None, :] * x_norm + self.bias[None, :]

class ResidualBlock(nn.Module):
    """Residual block with normalization and skip connection."""
    def __init__(self, hidden_dim: int, dtype: torch.dtype = torch.complex64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            ComplexNorm(hidden_dim, dtype=dtype),
            nn.Tanh(),
        )
        self.norm = ComplexNorm(hidden_dim, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))

class HolographicNet(nn.Module):
    """Neural network for learning holographic mappings between UV and IR data.
    
    This network learns the mapping between boundary (UV) and bulk (IR) data while
    preserving essential physical properties like scaling behavior and quantum corrections.
    
    Architecture:
    - Multi-layer complex-valued network
    - Residual connections to preserve geometric structure
    - Scale-equivariant layers for proper UV/IR scaling
    - Quantum correction layers for OPE effects
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 32,
        n_layers: int = 4,
        dtype: torch.dtype = torch.complex64,
        z_uv: float = 0.1,
        z_ir: float = 10.0
    ):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.z_uv = z_uv
        self.z_ir = z_ir
        self.z_ratio = z_ir / z_uv
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexNorm(hidden_dim, dtype=dtype),
            nn.Tanh(),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dtype=dtype)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, dim, dtype=dtype)
        
        # Learnable scaling factor
        self.log_scale = nn.Parameter(
            torch.log(torch.tensor(self.z_ratio**(-dim), dtype=torch.float32))
        )
        
        # Learnable quantum corrections
        self.correction_weights = nn.Parameter(
            torch.randn(3, dtype=dtype) * 0.1
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with physics-informed values."""
        # Initialize network to approximate identity + small perturbation
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    if param.dim() == 2:
                        # For weight matrices, use Glorot initialization
                        fan_in, fan_out = param.size(1), param.size(0)
                        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
                        param.data.copy_(std * torch.randn_like(param))
                elif 'bias' in name:
                    param.data.zero_()
            
            # Initialize correction weights close to analytical solution
            self.correction_weights.data.copy_(torch.tensor(
                [0.1/n for n in range(1, 4)],
                dtype=self.dtype
            ))
            
            # Initialize log_scale to match classical scaling
            self.log_scale.data.copy_(torch.log(torch.tensor(
                self.z_ratio**(-self.dim),
                dtype=torch.float32
            )))
    
    def compute_quantum_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum corrections using OPE-inspired terms."""
        corrections = torch.zeros_like(x)
        z_ratio = self.z_ratio  # Use actual z_ratio instead of exp(log_scale)
        
        for n, weight in enumerate(self.correction_weights, 1):
            power = -self.dim + 2*n
            correction = weight * x * z_ratio**power / torch.tensor(float(n), dtype=x.dtype)
            corrections = corrections + correction
        
        return corrections
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections and quantum corrections."""
        # Compute quantum corrections first
        corrections = self.compute_quantum_corrections(x)
        
        # Process input through network
        out = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            out = block(out)
        
        # Output projection
        out = self.output_proj(out)
        
        # Apply scaling and add corrections
        scale = torch.exp(self.log_scale.real)
        out = out * scale + corrections
        
        return out 