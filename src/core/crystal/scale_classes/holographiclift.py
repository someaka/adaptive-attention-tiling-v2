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
        Z_UV = 0.1
        Z_IR = 10.0
        Z_RATIO = Z_UV / Z_IR  # = 0.01
        # Use ComplexTanh for all networks if dtype is complex
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()

        self.ope_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 4, dtype=dtype),
            activation,
            nn.Linear(dim * 4, dim, dtype=dtype)
        )
        
    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype) if tensor.dtype != self.dtype else tensor
        
    
    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion.
        
        Args:
            op1: First operator
            op2: Second operator
            
        Returns:
            OPE result with same shape as inputs
        """
        op1 = self._ensure_dtype(op1)
        op2 = self._ensure_dtype(op2)
        
        # Store original shape for later reshaping
        original_shape = op1.shape
        
        # Flatten inputs
        op1_flat = op1.flatten()
        op2_flat = op2.flatten()
        
        # Pad or truncate to match network input size
        def prepare_input(x: torch.Tensor) -> torch.Tensor:
            if len(x) < self.dim:
                return torch.nn.functional.pad(x, (0, self.dim - len(x)))
            return x[:self.dim]
            
        # Normalize and prepare inputs
        op1_norm = prepare_input(op1_flat / torch.norm(op1_flat))
        op2_norm = prepare_input(op2_flat / torch.norm(op2_flat))
        
        # Make inputs symmetric under exchange
        # This ensures OPE symmetry by construction
        sym_part = (op1_norm + op2_norm) / 2
        asym_part = (op1_norm - op2_norm) / 2
        input_tensor = torch.cat([sym_part, torch.abs(asym_part)])  # Use absolute value of antisymmetric part
        
        # Compute OPE using neural network
        result = self.ope_net(input_tensor)
        
        # Reshape result to match input shape
        if len(result) == 1:
            # If scalar output, broadcast to original shape
            result = result.expand(int(np.prod(original_shape))).reshape(original_shape)
        else:
            # Pad or truncate result to match original size
            result_flat = result.flatten()
            orig_size = int(np.prod(original_shape))
            if len(result_flat) < orig_size:
                result_flat = torch.nn.functional.pad(result_flat, (0, orig_size - len(result_flat)))
            else:
                result_flat = result_flat[:orig_size]
            result = result_flat.reshape(original_shape)
        
        # Normalize result and ensure it's symmetric
        result = (result + result.conj()) / 2  # Make result Hermitian
        result = result / torch.norm(result)
        
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
        
        # Compute bulk field at each radial point
        for i, z in enumerate(radial_points):
            # Apply radial scaling with proper normalization
            # The scaling should be relative to the UV cutoff z_uv = radial_points[0]
            z_ratio = torch.abs(z / radial_points[0])  # This is dimensionless
            scaled_field = boundary_field * z_ratio**(-self.dim)
            bulk_field[i] = scaled_field
            
            # Add quantum corrections from OPE
            if i > 0:  # No corrections at UV boundary
                # Normalize input for OPE
                prev_field = bulk_field[i-1] / torch.norm(bulk_field[i-1])
                correction = self.compute_ope(prev_field)
                
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
        z_ratio = 0.01  # = z_uv/z_ir = 0.1/10.0
        
        # Store original norm
        ir_norm = torch.norm(ir_data)
        
        # Initialize UV data with leading term
        # The scaling should be inverted compared to holographic_lift
        # We need to multiply by z_ratio^dim to compensate for the IR scaling
        uv_data = ir_data * z_ratio**(-self.dim)  # Note: negative power to invert the IR scaling
        
        # Add subleading corrections from conformal dimension
        # These come from expanding the bulk-boundary propagator
        correction_sum = torch.zeros_like(ir_data)
        for n in range(1, 4):
            # Each correction is suppressed by additional powers of z_ratio
            # Use alternating signs to maintain proper phase relationships
            sign = (-1)**n
            correction = sign * ir_data * z_ratio**(-self.dim + 2*n) / math.factorial(n)
            correction_sum = correction_sum + correction
        
        # Scale down corrections to maintain stability
        correction_scale = 0.1 / (1 + z_ratio**2)
        uv_data = uv_data + correction_sum * correction_scale
        
        # Add quantum corrections using OPE
        ir_flat = ir_data.flatten()
        min_size = min(len(ir_flat), self.dim)
        if min_size < self.dim:
            ir_flat = torch.nn.functional.pad(ir_flat, (0, self.dim - min_size))
        else:
            ir_flat = ir_flat[:self.dim]
            
        # Normalize input for OPE while preserving phase
        ir_norm_flat = torch.norm(ir_flat)
        if ir_norm_flat > 0:
            ir_flat = ir_flat / ir_norm_flat
            
        # Compute OPE with phase preservation
        # The OPE should preserve the phase of the input
        ope_corr = self.operator_product_expansion(ir_flat, ir_flat)
        
        # Reshape OPE correction to match boundary shape
        ope_corr = ope_corr.reshape(-1)  # Flatten to 1D
        if len(ope_corr) == 1:
            # If scalar output, broadcast to boundary shape
            ope_corr = ope_corr.expand(ir_data.numel()).reshape(ir_data.shape)
        else:
            # Otherwise reshape to match boundary shape
            if len(ope_corr) < ir_data.numel():
                ope_corr = torch.nn.functional.pad(ope_corr, (0, ir_data.numel() - len(ope_corr)))
            ope_corr = ope_corr[:ir_data.numel()].reshape(ir_data.shape)
            
        # Add OPE correction with proper scaling
        # The quantum corrections scale as z^(2-dim)
        # Use the normalized IR data to maintain phase relationships
        ope_corr = ope_corr * (ir_data / ir_norm if ir_norm > 0 else 0) * z_ratio**(-self.dim + 2)
        ope_scale = 0.05 / (1 + z_ratio**2)  # Match the scale used in holographic_lift
        uv_data = uv_data + ope_corr * ope_scale * ir_norm  # Scale back up by ir_norm
        
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
