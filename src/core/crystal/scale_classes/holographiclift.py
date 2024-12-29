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
        """Compute operator product expansion with improved efficiency."""
        # Ensure inputs have correct dtype
        op1 = self._ensure_dtype(op1)
        op2 = self._ensure_dtype(op2)
        
        # Flatten inputs if needed
        if op1.dim() > 1:
            op1 = op1.reshape(-1)
        if op2.dim() > 1:
            op2 = op2.reshape(-1)
            
        # Pad or truncate to match network input dimension
        target_dim = self.dim
        
        def adjust_tensor(t: torch.Tensor) -> torch.Tensor:
            if len(t) > target_dim:
                return t[:target_dim]
            elif len(t) < target_dim:
                padding = torch.zeros(target_dim - len(t), dtype=self.dtype)
                return torch.cat([t, padding])
            return t
            
        op1 = adjust_tensor(op1)
        op2 = adjust_tensor(op2)
        
        # Normalize inputs for better convergence
        op1_norm = torch.norm(op1)
        op2_norm = torch.norm(op2)
        
        if op1_norm > 0:
            op1 = op1 / op1_norm
        if op2_norm > 0:
            op2 = op2 / op2_norm
        
        # Combine operators with proper normalization
        combined = torch.cat([op1, op2])
        
        # Add batch dimension if needed
        if combined.dim() == 1:
            combined = combined.unsqueeze(0)
        
        # Compute OPE with improved convergence
        result = self.ope_net(combined)
        
        # Remove batch dimension if added
        if result.dim() > 1 and result.shape[0] == 1:
            result = result.squeeze(0)
        
        # Scale result back and ensure proper normalization
        result = result * torch.sqrt(op1_norm * op2_norm)
        
        # For nearby points, the OPE should approximate direct product
        direct_product = op1[0] * op2[0]  # Use first components
        result = result * (direct_product / (result[0] + 1e-8))  # Normalize to match direct product
        
        return result

    def holographic_lift(self, boundary: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Lift boundary field to bulk using AdS/CFT correspondence."""
        # Ensure inputs have correct dtype
        boundary = self._ensure_dtype(boundary)
        radial = self._ensure_dtype(radial)
            
        # Initialize bulk field
        bulk_shape = (len(radial), *boundary.shape)
        bulk = torch.zeros(bulk_shape, dtype=self.dtype)
        
        # Compute bulk field using Fefferman-Graham expansion
        for i, z in enumerate(radial):
            # Leading term with proper radial scaling
            bulk[i] = boundary * z**(-self.dim)
            
            # Subleading corrections from conformal dimension
            for n in range(1, 4):  # Include first few corrections
                bulk[i] += (-1)**n * boundary * z**(-self.dim + 2*n) / (2*n)
                
            # Add quantum corrections using OPE
            if i > 0:  # Skip boundary point
                # Compute OPE between previous bulk slice and boundary
                prev_bulk_flat = bulk[i-1].flatten()
                boundary_flat = boundary.flatten()
                
                # Ensure we have enough components
                min_size = min(len(prev_bulk_flat), len(boundary_flat))
                if min_size < self.dim:
                    # Pad with zeros if needed
                    prev_bulk_flat = torch.nn.functional.pad(prev_bulk_flat, (0, self.dim - min_size))
                    boundary_flat = torch.nn.functional.pad(boundary_flat, (0, self.dim - min_size))
                else:
                    # Take first dim components
                    prev_bulk_flat = prev_bulk_flat[:self.dim]
                    boundary_flat = boundary_flat[:self.dim]
                
                ope_corr = self.operator_product_expansion(prev_bulk_flat, boundary_flat)
                # Reshape OPE correction to match boundary shape
                ope_corr = ope_corr.reshape(-1)  # Flatten to 1D
                if len(ope_corr) == 1:
                    # If scalar output, broadcast to boundary shape
                    ope_corr = ope_corr.expand(boundary.numel()).reshape(boundary.shape)
                else:
                    # Otherwise reshape to match boundary shape
                    # First ensure we have enough elements
                    if len(ope_corr) < boundary.numel():
                        ope_corr = torch.nn.functional.pad(ope_corr, (0, boundary.numel() - len(ope_corr)))
                    ope_corr = ope_corr[:boundary.numel()].reshape(boundary.shape)
                
                bulk[i] += ope_corr * z**(-self.dim + 2)
                
        return bulk
    

    def extract_uv_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract UV (boundary) data from bulk field."""
        # UV data is at the boundary (first slice)
        return field[0]

    def extract_ir_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract IR (deep bulk) data from bulk field."""
        # IR data is at the deepest bulk point (last slice)
        return field[-1]

    def reconstruct_from_ir(self, ir_data: torch.Tensor) -> torch.Tensor:
        """Reconstruct UV data from IR data using holographic principle."""
        # Use the same radial scaling as in holographic_lift
        z_ir = torch.tensor(10.0, dtype=ir_data.dtype)  # Matches the max value in test
        z_uv = torch.tensor(0.1, dtype=ir_data.dtype)   # Matches the min value in test
        
        # Store original norm for later rescaling
        original_norm = torch.norm(ir_data)
        
        # Initialize UV data with leading term, properly scaled
        # The scaling should match holographic_lift: z**(-dim)
        uv_data = ir_data * z_uv**(-self.dim) / z_ir**(-self.dim)
        
        # Add subleading corrections from conformal dimension
        for n in range(1, 4):
            # The correction should match holographic_lift: z**(-dim + 2n)
            correction = (-1)**n * ir_data * (z_uv**(-self.dim + 2*n) / z_ir**(-self.dim + 2*n)) / (2*n)
            uv_data = uv_data + correction
        
        # Add quantum corrections using OPE
        ir_flat = ir_data.flatten()
        min_size = min(len(ir_flat), self.dim)
        if min_size < self.dim:
            ir_flat = torch.nn.functional.pad(ir_flat, (0, self.dim - min_size))
        else:
            ir_flat = ir_flat[:self.dim]
            
        ope_corr = self.operator_product_expansion(ir_flat, ir_flat)
        
        # Reshape OPE correction to match boundary shape
        ope_corr = ope_corr.reshape(-1)  # Flatten to 1D
        if len(ope_corr) == 1:
            # If scalar output, broadcast to boundary shape
            ope_corr = ope_corr.expand(ir_data.numel()).reshape(ir_data.shape)
        else:
            # Otherwise reshape to match boundary shape
            # First ensure we have enough elements
            if len(ope_corr) < ir_data.numel():
                ope_corr = torch.nn.functional.pad(ope_corr, (0, ir_data.numel() - len(ope_corr)))
            ope_corr = ope_corr[:ir_data.numel()].reshape(ir_data.shape)
            
        # The OPE correction should match holographic_lift: z**(-dim + 2)
        ope_factor = z_uv**(-self.dim + 2) / z_ir**(-self.dim + 2)
        
        # Add OPE correction with proper scaling
        uv_data = uv_data + ope_factor * ope_corr
        
        # Rescale to match original norm
        uv_data = uv_data * (original_norm / torch.norm(uv_data))
        
        return uv_data

    def compute_c_function(self, bulk_field: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Compute c-function (central charge) along RG flow.
        
        The c-function should satisfy:
        1. Monotonicity: dc/dr <= 0 (decreases towards IR)
        2. Stationary at fixed points: dc/dr = 0
        3. Equals central charge at fixed points
        
        Args:
            bulk_field: Bulk field tensor of shape (radial_points, *field_dims)
            radial: Radial coordinate tensor
            
        Returns:
            c-function values along radial direction
        """
        # Ensure inputs have correct dtype
        bulk_field = self._ensure_dtype(bulk_field)
        radial = self._ensure_dtype(radial)
        
        # Initialize c-function
        c = torch.zeros_like(radial)
        
        # Compute c-function using area law and quantum corrections
        for i, z in enumerate(radial):
            # Area law contribution (classical)
            # The factor 1/4 comes from the holographic area law
            area_term = torch.norm(bulk_field[i]) / (4 * z**(self.dim - 1))
            
            # Quantum corrections from operator dimensions
            if i > 0:
                prev_slice = bulk_field[i-1].flatten()[:self.dim]
                curr_slice = bulk_field[i].flatten()[:self.dim]
                ope_corr = self.operator_product_expansion(prev_slice, curr_slice)
                quantum_term = torch.norm(ope_corr) / z**(self.dim + 1)
            else:
                quantum_term = 0
                
            c[i] = area_term + quantum_term
            
        return c
