"""Neural network models for holographic lifting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ComplexLayerNorm(nn.Module):
    """Complex-valued layer normalization with improved stability."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stable complex normalization."""
        # Ensure input is complex
        if not x.is_complex():
            x = x.to(dtype=torch.complex64)
        
        # Compute mean and variance over last dimension only
        mean = x.mean(dim=-1, keepdim=True)
        var_real = (x.real - mean.real).pow(2).mean(dim=-1, keepdim=True)
        var_imag = (x.imag - mean.imag).pow(2).mean(dim=-1, keepdim=True)
        var = var_real + var_imag
        
        # Normalize with stable denominator
        denom = (var + self.eps).sqrt()
        x_centered = x - mean
        x_norm = x_centered / denom
        
        return x_norm


class ComplexNorm(nn.Module):
    """Complex normalization layer."""
    
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize complex tensor without in-place operations."""
        # Compute norm along specified dimension
        norm = torch.norm(x, dim=self.dim, keepdim=True)
        
        # Create scale factor that preserves norm
        scale = torch.where(norm > self.eps, norm, torch.ones_like(norm))
        
        # Normalize without in-place operations
        x_normalized = x / (scale + self.eps)  # Add eps to denominator for stability
        
        # Scale back to original norm with stability check
        return x_normalized * torch.where(scale > self.eps, scale, torch.ones_like(scale))


class ResidualBlock(nn.Module):
    """Residual block with complex support and improved stability."""
    def __init__(self, dim: int, hidden_dim: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexLayerNorm(),
            ComplexTanh(),
            nn.Linear(hidden_dim, dim, dtype=dtype),
            ComplexLayerNorm()
        )
        
        # Initialize with small values for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if dtype is not None and dtype.is_complex:
                    m.weight.data = m.weight.data.to(dtype=dtype)
                    if m.bias is not None:
                        m.bias.data = m.bias.data.to(dtype=dtype)
                
                # Initialize weights
                if m.weight.is_complex():
                    nn.init.uniform_(m.weight.real, -0.01, 0.01)
                    nn.init.uniform_(m.weight.imag, -0.01, 0.01)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias.real, -0.001, 0.001)
                        nn.init.uniform_(m.bias.imag, -0.001, 0.001)
                else:
                    # Convert to complex if needed
                    m.weight.data = m.weight.data.to(dtype=dtype)
                    if m.bias is not None:
                        m.bias.data = m.bias.data.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Scale residual for stability
        return x + 0.1 * self.net(x)


class ComplexTanh(nn.Module):
    """Complex-valued tanh activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex tanh while preserving shape."""
        if not x.is_complex():
            x = x.to(dtype=torch.complex64)
            
        # Complex tanh(z) = (e^z - e^-z)/(e^z + e^-z)
        # Use more stable implementation to avoid overflow
        x_real = x.real
        x_imag = x.imag
        
        # Compute real and imaginary parts separately
        real_part = torch.tanh(x_real) * (1 - torch.tanh(x_imag).pow(2))
        imag_part = torch.tanh(x_imag) * (1 - torch.tanh(x_real).pow(2))
        
        return torch.complex(real_part, imag_part)


class HolographicNet(nn.Module):
    """Neural network for holographic transformations with improved stability."""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        dtype: torch.dtype = torch.complex64,
        z_uv: float = 0.1,
        z_ir: float = 10.0
    ):
        super().__init__()
        self._dim = dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dtype = dtype
        self.z_uv = z_uv
        self.z_ir = z_ir
        
        # Input projection with careful initialization
        self.input_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexLayerNorm(),
            ComplexTanh()
        )
        
        # Residual blocks with improved stability
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dtype)
            for _ in range(n_layers)
        ])
        
        # Output projection with normalization
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, dim, dtype=dtype),
            ComplexLayerNorm()
        )
        
        # Initialize all weights with improved stability
        self._init_weights()
        
        # Correction weights for quantum effects
        self.correction_weights = nn.Parameter(
            0.01 * torch.randn(n_layers, dtype=dtype)
        )
    
    def _init_weights(self):
        """Initialize weights with improved stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.is_complex():
                    nn.init.uniform_(m.weight.real, -0.01, 0.01)
                    nn.init.uniform_(m.weight.imag, -0.01, 0.01)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias.real, -0.001, 0.001)
                        nn.init.uniform_(m.bias.imag, -0.001, 0.001)
                else:
                    # Convert to complex if needed
                    m.weight.data = m.weight.data.to(dtype=self._dtype)
                    if m.bias is not None:
                        m.bias.data = m.bias.data.to(dtype=self._dtype)
    
    @property
    def dim(self) -> int:
        """Get the dimension of the model."""
        return self._dim
    
    @property
    def hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        return self._hidden_dim
    
    @property
    def n_layers(self) -> int:
        """Get the number of layers in the model."""
        return self._n_layers
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the model."""
        return self._dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved stability and norm preservation."""
        # Ensure input is complex
        if not x.is_complex():
            x = x.to(dtype=self._dtype)
        
        # Store input norms for later
        input_norms = torch.norm(x, dim=-1, keepdim=True)
        
        # Input projection
        x = self.input_proj(x)  # Shape: [batch_size, hidden_dim]
        
        # Apply residual layers with accumulated normalization
        for layer in self.layers:
            x = layer(x)
            x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)  # Normalize between layers
            x = x * input_norms  # Rescale to preserve input norm
        
        # Output projection with final normalization
        x = self.output_proj(x)  # Shape: [batch_size, dim]
        
        # Final normalization to match input norms
        output_norms = torch.norm(x, dim=-1, keepdim=True)
        x = x * (input_norms / (output_norms + 1e-8))
            
        return x
    
    def compute_quantum_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum corrections that scale linearly with input."""
        # Ensure input is complex
        if not x.is_complex():
            x = x.to(dtype=self._dtype)
            
        # Get input norm for scaling
        input_norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Normalize input to unit norm for consistent processing
        x_normalized = x / (input_norm + 1e-8)
        
        # Project to hidden space
        h = self.input_proj(x_normalized)
        
        # Compute corrections using residual connections
        corrections = torch.zeros_like(x_normalized)
        for i, layer in enumerate(self.layers):
            # Apply layer and get residual
            h_next = layer(h)
            residual = h_next - h
            
            # Project residual back to input space and scale
            layer_correction = self.output_proj(residual)
            corrections = corrections + self.correction_weights[i] * layer_correction
            
            # Update hidden state
            h = h_next
            
        # Ensure corrections are much smaller than normalized input
        corr_norm = torch.norm(corrections, dim=-1, keepdim=True)
        scale_factor = torch.minimum(
            torch.ones_like(corr_norm),
            0.01 / (corr_norm + 1e-8)
        )
        corrections = corrections * scale_factor
        
        # Scale back by input norm to ensure linear scaling
        corrections = corrections * input_norm
        
        return corrections
    
    def holographic_lift(self, boundary: torch.Tensor, radial_points: torch.Tensor) -> torch.Tensor:
        """Lift boundary data to bulk data using holographic mapping."""
        # Ensure inputs are complex
        if not boundary.is_complex():
            boundary = boundary.to(dtype=self._dtype)
        if not radial_points.is_complex():
            radial_points = radial_points.to(dtype=self._dtype)
            
        # Store original shape and flatten if needed
        orig_shape = boundary.shape
        if boundary.dim() > 2:
            boundary = boundary.reshape(-1, self.dim)
            
        # Initialize bulk data tensor
        bulk_shape = boundary.shape[:-1] + (len(radial_points),) + boundary.shape[-1:]
        bulk_data = torch.zeros(bulk_shape, dtype=self._dtype, device=boundary.device)
        
        # Compute bulk data at each radial point
        for i, z in enumerate(radial_points):
            # Scale input based on radial position
            scale = torch.abs(z / self.z_uv)
            scaled_boundary = boundary * scale
            
            # Apply network transformation
            bulk_slice = self(scaled_boundary)
            
            # Add quantum corrections scaled by radial position
            corrections = self.compute_quantum_corrections(scaled_boundary)
            bulk_slice = bulk_slice + corrections * (scale ** 2)
            
            # Store result
            bulk_data[..., i, :] = bulk_slice
            
        # Restore original shape if needed
        if len(orig_shape) > 2:
            bulk_data = bulk_data.reshape(orig_shape[:-1] + (len(radial_points),) + orig_shape[-1:])
            
        return bulk_data
