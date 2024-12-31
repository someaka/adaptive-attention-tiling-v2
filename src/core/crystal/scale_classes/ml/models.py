"""Neural network models for holographic lifting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ComplexNorm(nn.Module):
    """Custom normalization layer for complex numbers."""
    def __init__(self, num_features: int, dtype: torch.dtype = torch.complex64, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.dtype = dtype
        self.eps = eps
        # Separate weights for real and imaginary parts
        self.weight_real = nn.Parameter(torch.ones(num_features))
        self.weight_imag = nn.Parameter(torch.zeros(num_features))
        self.bias_real = nn.Parameter(torch.zeros(num_features))
        self.bias_imag = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into real and imaginary parts
        x_real = x.real
        x_imag = x.imag
        
        # Compute mean and variance for real and imaginary parts separately
        mean_real = x_real.mean(dim=0, keepdim=True)
        mean_imag = x_imag.mean(dim=0, keepdim=True)
        
        # Center the data
        x_real_centered = x_real - mean_real
        x_imag_centered = x_imag - mean_imag
        
        # Compute variance using both real and imaginary parts
        var = (x_real_centered**2 + x_imag_centered**2).mean(dim=0, keepdim=True)
        std = torch.sqrt(var + self.eps)
        
        # Normalize both parts
        x_real_norm = x_real_centered / std
        x_imag_norm = x_imag_centered / std
        
        # Apply affine transformation
        out_real = x_real_norm * self.weight_real[None, :] - x_imag_norm * self.weight_imag[None, :] + self.bias_real[None, :]
        out_imag = x_real_norm * self.weight_imag[None, :] + x_imag_norm * self.weight_real[None, :] + self.bias_imag[None, :]
        
        # Combine real and imaginary parts
        return torch.complex(out_real, out_imag)


class ResidualBlock(nn.Module):
    """Residual block with normalization and skip connection."""
    def __init__(self, hidden_dim: int, dtype: torch.dtype = torch.complex64, eps: float = 1e-5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            ComplexNorm(hidden_dim, dtype=dtype, eps=eps),
            nn.Tanh(),
        )
        self.norm = ComplexNorm(hidden_dim, dtype=dtype, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))


class HolographicNet(nn.Module):
    """Neural network for learning holographic mappings between UV and IR data."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 16,
        n_layers: int = 4,
        dtype: torch.dtype = torch.complex64,
        z_uv: float = 0.1,
        z_ir: float = 10.0,
        eps: float = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.z_uv = z_uv
        self.z_ir = z_ir
        self.z_ratio = z_ir / z_uv
        self.eps = eps
        
        # Learnable scaling factor initialized to analytical solution
        self.log_scale = nn.Parameter(
            torch.log(torch.tensor(self.z_ratio**(-dim), dtype=torch.float32))
        )
        
        # Learnable quantum corrections initialized to analytical values
        self.correction_weights = nn.Parameter(
            torch.tensor([(-1)**n * 0.1 / (n * (1 + self.z_ratio**2)) 
                         for n in range(1, 4)], dtype=dtype)
        )
        
        # Simple feed-forward network for learning corrections
        self.layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim, dtype=dtype),
            *[nn.Linear(hidden_dim, hidden_dim, dtype=dtype) for _ in range(n_layers-2)],
            nn.Linear(hidden_dim, dim, dtype=dtype)
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with physics-informed values."""
        with torch.no_grad():
            # Initialize network weights to small values
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    fan_in = layer.weight.size(1)
                    std = torch.sqrt(torch.tensor(2.0 / fan_in)) * 0.01  # Very small scale
                    if layer.weight.is_complex():
                        layer.weight.data.copy_(std * (torch.randn_like(layer.weight.real) + 1j * torch.randn_like(layer.weight.imag)))
                    else:
                        layer.weight.data.copy_(std * torch.randn_like(layer.weight))
                    if layer.bias is not None:
                        layer.bias.data.zero_()
            
            # Initialize correction weights to match analytical solution
            self.correction_weights.data.copy_(torch.tensor(
                [(-1)**n * 0.1 / (n * (1 + self.z_ratio**2)) for n in range(1, 4)],
                dtype=self.dtype
            ))
            
            # Initialize log_scale to match classical scaling
            self.log_scale.data.copy_(torch.log(torch.tensor(
                self.z_ratio**(-self.dim),
                dtype=torch.float32
            )))
    
    def compute_quantum_corrections(self, x: torch.Tensor, input_norm: torch.Tensor) -> torch.Tensor:
        """Compute quantum corrections using OPE-inspired terms."""
        corrections = torch.zeros_like(x)
        z_ratio = self.z_ratio
        base_scale = torch.sigmoid(self.log_scale.real) * 0.01  # Small scale
        
        for n in range(1, 4):
            power = -self.dim + 2*n
            weight = self.correction_weights[n-1]
            correction = weight * x * z_ratio**power
            corrections = corrections + correction * base_scale
        
        return corrections * input_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing holographic mapping."""
        # Store original input and compute norm
        x_orig = x
        input_norm = torch.norm(x, dim=1, keepdim=True)
        x = x / (input_norm + self.eps)
        
        # Apply classical scaling with stable phase
        scale = torch.exp(self.log_scale.real)
        if x.is_complex():
            phase = torch.exp(1j * torch.angle(torch.tensor(self.z_ratio**(-self.dim), dtype=x.dtype)))
            x = x * scale * phase
        else:
            x = x * scale
        
        # Learn corrections through network
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
        
        # Combine classical scaling and learned corrections
        x = x + h * 0.1  # Small contribution from learned part
        
        # Add analytical quantum corrections
        corrections = self.compute_quantum_corrections(x_orig / (input_norm + self.eps), input_norm)
        x = x + corrections
        
        # Normalize output to match input norm
        output_norm = torch.norm(x, dim=1, keepdim=True)
        x = x * (input_norm / (output_norm + self.eps))
        
        return x 