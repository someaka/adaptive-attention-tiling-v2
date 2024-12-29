from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
from contextlib import contextmanager
import gc
import logging
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn, Tensor

# Import memory optimization utilities
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from src.utils.memory_management import optimize_memory, register_tensor
from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention
from src.core.crystal.scale_classes.rgflow import RGFlow
from src.core.crystal.scale_classes.complextanh import ComplexTanh
from src.core.crystal.scale_classes.scaleinvariance import ScaleInvariance
from src.core.crystal.scale_classes.anomalydetector import (
    AnomalyDetector, 
    AnomalyPolynomial, 
    ScaleConnection, 
    ScaleConnectionData
)
from src.core.crystal.scale_classes.renormalizationflow import RenormalizationFlow





class HolographicLifter:
    """Handles holographic lifting and UV/IR transformations."""
    
    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.dtype = dtype
        # Radial coordinates for UV/IR connection
        self.Z_UV = 0.1
        self.Z_IR = 10.0
        self.Z_RATIO = self.Z_UV / self.Z_IR  # = 0.01
        # Use ComplexTanh for all networks if dtype is complex
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()

        # For complex inputs, we need to double the input size to handle real and imaginary parts
        input_dim = dim * 2 * 2  # 2 operators concatenated * 2 for real/imag
        hidden_dim = input_dim * 2  # 2x hidden layer size for better reconstruction
        output_dim = dim * 2  # Output has same dimension as input operators for complex case

        # Always use float32 for the network, we'll handle complex conversion in operator_product_expansion
        network_dtype = torch.float32

        self.ope_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=network_dtype),
            activation,
            nn.Linear(hidden_dim, hidden_dim, dtype=network_dtype),  # Additional hidden layer
            activation,
            nn.Linear(hidden_dim, output_dim, dtype=network_dtype)
        )
        
    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype) if tensor.dtype != self.dtype else tensor
        
    
    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion using neural network."""
        # Store original shape for later reshaping
        original_shape = op1.shape
        is_1d = len(original_shape) == 1

        # Normalize inputs
        op1_norm = op1.flatten()[:self.dim] / (torch.norm(op1) + 1e-8)
        op2_norm = op2.flatten()[:self.dim] / (torch.norm(op2) + 1e-8)

        # Concatenate inputs
        input_tensor = torch.cat([op1_norm, op2_norm])

        # Convert to float32 for network processing
        if input_tensor.is_complex():
            # Split complex into real and imaginary parts
            input_real = input_tensor.real
            input_imag = input_tensor.imag
            input_tensor = torch.cat([input_real, input_imag])

        # Ensure input tensor has correct shape for network
        input_tensor = input_tensor[:self.dim * 2 * 2]  # Truncate to expected input size
        if len(input_tensor) < self.dim * 2 * 2:
            # Pad if needed
            input_tensor = torch.nn.functional.pad(input_tensor, (0, self.dim * 2 * 2 - len(input_tensor)))

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # Process through network
        result = self.ope_net(input_tensor)

        # Remove batch dimension
        result = result.squeeze(0)

        # Convert back to complex if needed
        if self.dtype == torch.complex64:
            # Split result into real and imaginary parts
            half_size = result.size(0) // 2
            result_real = result[:half_size]
            result_imag = result[half_size:]
            result = torch.complex(result_real, result_imag)

        # For 1D case, we only need the first self.dim elements
        if is_1d:
            result = result[:self.dim]
        else:
            # Pad or truncate result to match original size
            result_flat = result.flatten()
            orig_size = int(np.prod(original_shape))
            if len(result_flat) < orig_size:
                result_flat = torch.nn.functional.pad(result_flat, (0, orig_size - len(result_flat)))
            else:
                result_flat = result_flat[:orig_size]
            result = result_flat

        # Reshape to match input dimensions
        result = result.reshape(original_shape)

        # Enforce symmetry by averaging with reversed input result
        input_reversed = torch.cat([op2_norm, op1_norm])
        if input_reversed.is_complex():
            input_reversed_real = input_reversed.real
            input_reversed_imag = input_reversed.imag
            input_reversed = torch.cat([input_reversed_real, input_reversed_imag])
        input_reversed = input_reversed[:self.dim * 2 * 2]  # Truncate to expected input size
        if len(input_reversed) < self.dim * 2 * 2:
            # Pad if needed
            input_reversed = torch.nn.functional.pad(input_reversed, (0, self.dim * 2 * 2 - len(input_reversed)))
        input_reversed = input_reversed.unsqueeze(0)  # Add batch dimension
        result_reversed = self.ope_net(input_reversed)
        result_reversed = result_reversed.squeeze(0)  # Remove batch dimension
        if self.dtype == torch.complex64:
            half_size = result_reversed.size(0) // 2
            result_reversed_real = result_reversed[:half_size]
            result_reversed_imag = result_reversed[half_size:]
            result_reversed = torch.complex(result_reversed_real, result_reversed_imag)

        # For 1D case, we only need the first self.dim elements
        if is_1d:
            result_reversed = result_reversed[:self.dim]
        else:
            # Pad or truncate result to match original size
            result_reversed_flat = result_reversed.flatten()
            if len(result_reversed_flat) < orig_size:
                result_reversed_flat = torch.nn.functional.pad(result_reversed_flat, (0, orig_size - len(result_reversed_flat)))
            else:
                result_reversed_flat = result_reversed_flat[:orig_size]
            result_reversed = result_reversed_flat

        result_reversed = result_reversed.reshape(original_shape)
        result = (result + result_reversed) / 2

        # Normalize result
        result = result / (torch.norm(result) + 1e-8)

        return result

    def holographic_lift(self, boundary_field: torch.Tensor, radial_points: torch.Tensor) -> torch.Tensor:
        """Lift boundary field to bulk using holographic correspondence.
        
        Args:
            boundary_field: Field values at UV boundary
            radial_points: Radial coordinates for bulk field
            
        Returns:
            Bulk field values at each radial point
        """
        boundary_field = self._ensure_dtype(boundary_field)
        radial_points = self._ensure_dtype(radial_points)
        
        # Initialize bulk field
        bulk_shape = (len(radial_points),) + boundary_field.shape
        bulk_field = torch.zeros(bulk_shape, dtype=self.dtype)
        
        # Set UV boundary condition
        bulk_field[0] = boundary_field
        
        # Store original norm for normalization
        boundary_norm = torch.norm(boundary_field)
        
        # Store z_ratio for reconstruction
        self.Z_UV = radial_points[0]
        self.Z_IR = radial_points[-1]
        self.Z_RATIO = self.Z_UV / self.Z_IR
        
        # Store phase evolution for reconstruction
        self.phase_evolution = []
        
        # Store boundary norm for reconstruction
        self.boundary_norm = boundary_norm
        
        # Compute bulk field at each radial point
        for i, z in enumerate(radial_points):
            # Apply radial scaling with proper normalization
            # The scaling should be relative to the UV cutoff z_uv = radial_points[0]
            z_ratio = torch.abs(z / self.Z_UV)  # This is dimensionless
            scaled_field = boundary_field * z_ratio**(-self.dim)
            bulk_field[i] = scaled_field
            
            # Add quantum corrections from OPE
            if i > 0:  # No corrections at UV boundary
                # Normalize input for OPE
                prev_field = bulk_field[i-1] / torch.norm(bulk_field[i-1])
                correction = self.compute_ope(prev_field)
                
                # Store phase evolution for reconstruction
                phase = torch.angle(correction.flatten()[0]) if correction.is_complex() else torch.tensor(0.0)
                self.phase_evolution.append(phase.item())
                
                # Scale correction by boundary norm and proper radial dependence
                # The quantum corrections scale as z^(2-dim) relative to classical scaling
                correction = correction * boundary_norm * z_ratio**(2-self.dim)
                
                # Add correction with proper relative strength
                # Scale down corrections more at large z to maintain stability
                correction_scale = 0.05 / (1 + z_ratio**2)  # Decreases faster with z
                bulk_field[i] = bulk_field[i] + correction * correction_scale
        
        return bulk_field
    

    def extract_uv_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract UV (boundary) data from bulk field."""
        # UV data is at the boundary (first slice)
        return field[0]

    def extract_ir_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract IR (deep bulk) data from bulk field."""
        # IR data is at the deepest bulk point (last slice)
        return field[-1]

    def reconstruct_from_ir(self, ir_data: torch.Tensor) -> torch.Tensor:
        """Reconstruct UV data from IR data using holographic principle.
        
        Args:
            ir_data: Field values at IR boundary
            
        Returns:
            Reconstructed field values at UV boundary
        """
        # Use dimensionless ratio for scaling
        z_ratio = self.Z_RATIO  # Use the ratio from initialization
        
        # Store original norm
        ir_norm = torch.norm(ir_data)
        
        # Initialize UV data with leading term
        # The scaling should be inverted compared to holographic_lift
        # We need to multiply by z_ratio^dim to compensate for the IR scaling
        uv_data = ir_data * z_ratio**(-self.dim)  # Note: negative power to invert the IR scaling
        
        # Add subleading corrections from conformal dimension
        # These come from expanding the bulk-boundary propagator
        correction_sum = torch.zeros_like(ir_data)
        for n in range(1, 6):  # Increase order of corrections
            # Each correction is suppressed by additional powers of z_ratio
            # Use alternating signs to maintain proper phase relationships
            sign = (-1)**n
            correction = sign * ir_data * z_ratio**(-self.dim + 2*n) / math.factorial(n)
            correction_sum = correction_sum + correction
        
        # Scale down corrections to maintain stability
        correction_scale = 0.1 / (1 + z_ratio**2)
        uv_data = uv_data + correction_sum * correction_scale
        
        # Add quantum corrections using OPE
        # Use compute_ope to match the phase handling in holographic_lift
        if ir_norm > 0 and hasattr(self, 'phase_evolution') and self.phase_evolution:
            # Normalize input for OPE
            ir_norm_field = ir_data / ir_norm
            correction = self.compute_ope(ir_norm_field)
            
            # Scale correction by IR norm and proper radial dependence
            # The quantum corrections scale as z^(2-dim) relative to classical scaling
            correction = correction * ir_norm * z_ratio**(-self.dim + 2)
            
            # Add correction with proper relative strength
            # Scale down corrections more at large z to maintain stability
            correction_scale = 0.05 / (1 + z_ratio**2)  # Match the scale used in holographic_lift
            
            # Apply stored phase evolution
            phase_sum = sum(self.phase_evolution)
            phase_evolution = torch.exp(1j * torch.tensor(phase_sum, dtype=self.dtype))
            correction = correction * phase_evolution
            
            uv_data = uv_data + correction * correction_scale
        
        # Normalize the result to match the original UV norm
        if hasattr(self, 'boundary_norm'):
            target_norm = self.boundary_norm
        else:
            # If boundary_norm is not available, estimate it from IR data
            target_norm = ir_norm * z_ratio**(-self.dim)
        
        uv_data = uv_data / (torch.norm(uv_data) + 1e-8) * target_norm
        
        return uv_data

    def compute_c_function(self, bulk_field: torch.Tensor, radial_points: torch.Tensor) -> torch.Tensor:
        """Compute c-function (central charge) along RG flow.
        
        Args:
            bulk_field: Bulk field values
            radial_points: Radial coordinates
            
        Returns:
            c-function values at each radial point
        """
        bulk_field = self._ensure_dtype(bulk_field)
        radial_points = self._ensure_dtype(radial_points)
        
        # Initialize c-function
        c_function = torch.zeros(len(radial_points), dtype=torch.float64)
        
        # Compute area law contribution
        area_law = torch.abs(radial_points) ** (-self.dim)
        
        # Compute quantum corrections
        for i, z in enumerate(radial_points):
            # Area law contribution
            c_function[i] = torch.norm(bulk_field[i]) * area_law[i]
            
            # Add quantum corrections
            if i > 0:  # No corrections at UV boundary
                # Compute derivative contribution
                if i < len(radial_points) - 1:
                    dz = radial_points[i+1] - radial_points[i]
                    df = bulk_field[i+1] - bulk_field[i]
                    deriv = torch.norm(df/dz)
                    c_function[i] += deriv * torch.abs(z)**(2-self.dim)
                
                # Ensure monotonicity
                if i > 0:
                    c_function[i] = torch.minimum(c_function[i], c_function[i-1])
                    
        return c_function

    def compute_ope(self, field: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion coefficients.
        
        Args:
            field: Input field
            
        Returns:
            OPE coefficients with same shape as input
        """
        field = self._ensure_dtype(field)
        
        # Normalize field
        field_norm = field / torch.norm(field)
        
        # Compute OPE using neural network
        correction = self.operator_product_expansion(field_norm, field_norm)
        
        # Scale correction by field norm
        correction = correction * torch.norm(field)
        
        return correction
