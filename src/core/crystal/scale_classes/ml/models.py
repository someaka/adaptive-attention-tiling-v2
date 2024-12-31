"""Neural network models for holographic lifting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex tensors."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, dtype: torch.dtype = torch.complex64):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.dtype = dtype
        
        # Initialize learnable parameters with small non-zero values
        if dtype.is_complex:
            self.weight = nn.Parameter(torch.complex(
                torch.ones(normalized_shape),
                torch.zeros(normalized_shape)
            ).to(dtype=dtype))
            self.bias = nn.Parameter(torch.complex(
                0.01 * torch.randn(normalized_shape),
                0.01 * torch.randn(normalized_shape)
            ).to(dtype=dtype))
        else:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
            self.bias = nn.Parameter(0.01 * torch.randn(normalized_shape, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with complex normalization avoiding in-place operations."""
        # Create a copy of input to prevent in-place modification
        x = x.clone()
        
        # Compute mean and variance separately for real and imaginary parts
        mean_real = x.real.mean(dim=-1, keepdim=True)
        mean_imag = x.imag.mean(dim=-1, keepdim=True) if x.is_complex() else torch.zeros_like(mean_real)
        
        var_real = ((x.real - mean_real) ** 2).mean(dim=-1, keepdim=True)
        var_imag = ((x.imag - mean_imag) ** 2).mean(dim=-1, keepdim=True) if x.is_complex() else torch.zeros_like(var_real)
        
        # Normalize real and imaginary parts without in-place operations
        denom_real = torch.sqrt(var_real + self.eps)
        denom_imag = torch.sqrt(var_imag + self.eps) if x.is_complex() else torch.ones_like(denom_real)
        
        x_real = (x.real - mean_real) / denom_real
        x_imag = (x.imag - mean_imag) / denom_imag if x.is_complex() else torch.zeros_like(x_real)
        
        # Apply learnable parameters without in-place operations
        out_real = x_real * self.weight.real + self.bias.real
        out_imag = x_imag * self.weight.imag + self.bias.imag if x.is_complex() else torch.zeros_like(out_real)
        
        # Combine real and imaginary parts
        return torch.complex(out_real, out_imag) if x.is_complex() else out_real


class ComplexNorm(nn.Module):
    """Complex normalization layer."""
    
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize complex tensor without in-place operations."""
        # Create a copy of input to prevent in-place modification
        x = x.clone()
        
        # Compute norm along specified dimension
        norm = torch.norm(x, dim=self.dim, keepdim=True)
        
        # Create scale factor that preserves norm
        scale = torch.where(norm > self.eps, norm, torch.ones_like(norm))
        
        # Normalize without in-place operations
        x_normalized = x / (scale + self.eps)  # Add eps to denominator for stability
        
        return x_normalized * scale  # Scale back to original norm


class ResidualBlock(nn.Module):
    """Residual block with complex support."""
    
    def __init__(self, dim: int, hidden_dim: int, dtype: torch.dtype):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, dtype=dtype)
        self.linear2 = nn.Linear(hidden_dim, dim, dtype=dtype)
        self.norm = ComplexNorm(dim=1)
        
        # Initialize weights with physics-informed values
        with torch.no_grad():
            # Glorot initialization for complex weights
            std = math.sqrt(2.0 / (dim + hidden_dim))
            if dtype.is_complex:
                self.linear1.weight.data = std * (torch.randn_like(self.linear1.weight.real) + 1j * torch.randn_like(self.linear1.weight.imag))
                self.linear2.weight.data = std * (torch.randn_like(self.linear2.weight.real) + 1j * torch.randn_like(self.linear2.weight.imag))
                # Initialize biases with small complex values
                self.linear1.bias.data = (std * 0.1) * (torch.randn_like(self.linear1.bias.real) + 1j * torch.randn_like(self.linear1.bias.imag))
                self.linear2.bias.data = (std * 0.1) * (torch.randn_like(self.linear2.bias.real) + 1j * torch.randn_like(self.linear2.bias.imag))
            else:
                self.linear1.weight.data = std * torch.randn_like(self.linear1.weight)
                self.linear2.weight.data = std * torch.randn_like(self.linear2.weight)
                # Initialize biases with small values
                self.linear1.bias.data = (std * 0.1) * torch.randn_like(self.linear1.bias)
                self.linear2.bias.data = (std * 0.1) * torch.randn_like(self.linear2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Store input for residual connection
        identity = x.clone()
        
        # First linear layer with activation
        out = self.linear1(x)
        out = torch.tanh(out)
        
        # Second linear layer
        out = self.linear2(out)
        
        # Add residual connection without in-place operation
        out = torch.add(out, identity)
        
        # Normalize output
        return self.norm(out)


class HolographicNet(nn.Module):
    """Neural network for holographic lifting."""
    
    def __init__(self, dim: int, hidden_dim: int, n_layers: int, dtype: torch.dtype,
                 z_uv: float = 0.1, z_ir: float = 10.0):
        super().__init__()
        self._dim = dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dtype = dtype
        self._z_uv = z_uv
        self._z_ir = z_ir
        
        # Input projection with complex layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexLayerNorm(hidden_dim, dtype=dtype),
            ComplexNorm(dim=1)
        )
        
        # Residual blocks with complex layer norm
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(hidden_dim, hidden_dim * 2, dtype),
                ComplexLayerNorm(hidden_dim, dtype=dtype)
            )
            for _ in range(n_layers)
        ])
        
        # Output projection with complex layer norm
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, dim, dtype=dtype),
            ComplexLayerNorm(dim, dtype=dtype)
        )
        
        # Initialize learnable parameters with physics-informed values
        with torch.no_grad():
            # Initialize log_scale based on classical scaling
            z_ratio = z_ir / z_uv
            if dtype.is_complex:
                self.log_scale = nn.Parameter(torch.complex(
                    torch.tensor(math.log(z_ratio**(-dim))),
                    torch.zeros(1)
                ).to(dtype=dtype))
            else:
                self.log_scale = nn.Parameter(torch.tensor(
                    math.log(z_ratio**(-dim)),
                    dtype=dtype
                ))
            
            # Initialize correction weights based on OPE coefficients
            self.correction_weights = nn.Parameter(torch.tensor([
                (-1)**n / math.factorial(n) for n in range(1, 4)
            ], dtype=dtype))
            
            # Initialize input projection
            std = math.sqrt(2.0 / (dim + hidden_dim))
            if dtype.is_complex:
                self.input_proj[0].weight.data = std * (torch.randn_like(self.input_proj[0].weight.real) + 1j * torch.randn_like(self.input_proj[0].weight.imag))
                self.input_proj[0].bias.data = (std * 0.1) * (torch.randn_like(self.input_proj[0].bias.real) + 1j * torch.randn_like(self.input_proj[0].bias.imag))
            else:
                self.input_proj[0].weight.data = std * torch.randn_like(self.input_proj[0].weight)
                self.input_proj[0].bias.data = (std * 0.1) * torch.randn_like(self.input_proj[0].bias)
            
            # Initialize output projection
            std = math.sqrt(2.0 / (hidden_dim + dim))
            if dtype.is_complex:
                self.output_proj[0].weight.data = std * (torch.randn_like(self.output_proj[0].weight.real) + 1j * torch.randn_like(self.output_proj[0].weight.imag))
                self.output_proj[0].bias.data = (std * 0.1) * (torch.randn_like(self.output_proj[0].bias.real) + 1j * torch.randn_like(self.output_proj[0].bias.imag))
            else:
                self.output_proj[0].weight.data = std * torch.randn_like(self.output_proj[0].weight)
                self.output_proj[0].bias.data = (std * 0.1) * torch.randn_like(self.output_proj[0].bias)
            
            # Initialize layer norm parameters
            for m in self.modules():
                if isinstance(m, ComplexLayerNorm):
                    if dtype.is_complex:
                        m.weight.data = torch.complex(
                            torch.ones_like(m.weight.real),
                            torch.zeros_like(m.weight.imag)
                        )
                        m.bias.data = torch.complex(
                            torch.zeros_like(m.bias.real),
                            torch.zeros_like(m.bias.imag)
                        )
                    else:
                        m.weight.data = torch.ones_like(m.weight)
                        m.bias.data = torch.zeros_like(m.bias)
    
    @property
    def dim(self) -> int:
        """Get dimension."""
        return self._dim
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension."""
        return self._hidden_dim
    
    @property
    def n_layers(self) -> int:
        """Get number of layers."""
        return self._n_layers
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data type."""
        return self._dtype
    
    @property
    def z_uv(self) -> float:
        """Get UV cutoff."""
        return self._z_uv
    
    @property
    def z_ir(self) -> float:
        """Get IR cutoff."""
        return self._z_ir
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Store input norm for later
        input_norm = torch.norm(x, dim=1, keepdim=True)
        
        # Create a copy of input to prevent in-place modification
        x = x.clone()
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection with safe operations
        x = self.output_proj(x)
        
        # Normalize output to match input norm
        norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.where(norm > 1e-8, x / norm, x)
        x = x * input_norm
        
        return x
    
    def compute_quantum_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum corrections using OPE with proper scaling."""
        corrections = []
        input_norm = torch.norm(x)
        x = x.clone()  # Create a copy to prevent in-place modification
        
        # Base scaling factor - make corrections much smaller than input
        base_scale = 0.001 * input_norm
        
        for n, weight in enumerate(self.correction_weights, 1):
            # Compute power and scale for this order
            power = -self.dim + 2*n
            z_ratio = self.z_ir/self.z_uv
            
            # Scale correction by factorial and power
            scale = base_scale / (math.factorial(n) * n)
            correction = x * scale * z_ratio**power
            
            # Ensure correction norm is much smaller than input
            corr_norm = torch.norm(correction)
            if corr_norm > 0:
                correction = correction * (0.1 / n) * (input_norm / corr_norm)
            
            corrections.append(correction * weight)
        
        # Sum corrections with exponential damping
        total_correction = torch.zeros_like(x)
        for n, corr in enumerate(corrections):
            damping = torch.tensor(-float(n), device=x.device, dtype=x.dtype).exp()
            total_correction = total_correction + corr * damping
            
        return total_correction
    
    def holographic_lift(self, boundary: torch.Tensor, radial_points: torch.Tensor) -> torch.Tensor:
        """Lift boundary data to bulk using holographic principle."""
        # Store boundary norm for later
        boundary_norm = torch.norm(boundary)
        
        # Create a copy of boundary to prevent in-place modification
        boundary = boundary.clone()
        
        # Compute classical scaling with improved accuracy
        z_ratios = torch.abs(radial_points / self.z_uv).reshape(-1, 1, 1)
        bulk = boundary.unsqueeze(0) * z_ratios**(-self.dim)
        
        # Add quantum corrections with proper scaling
        corrections = self.compute_quantum_corrections(boundary)
        correction_scale = 0.001 / (1 + z_ratios**2)  # Increase correction scale
        bulk = bulk + corrections.unsqueeze(0) * correction_scale
        
        # Apply network transformation with improved norm preservation
        bulk = torch.stack([
            self.forward(slice_.detach().clone())  # Create a copy to prevent in-place modification
            for slice_ in bulk
        ])
        
        # Normalize bulk to preserve boundary norm with improved accuracy
        norm = torch.norm(bulk, dim=1, keepdim=True)
        bulk = torch.where(norm > 1e-8, bulk / norm, bulk)
        bulk = bulk * boundary_norm
        
        return bulk 