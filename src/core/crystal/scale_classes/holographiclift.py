from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any, Union, Optional, Generator
from contextlib import contextmanager, nullcontext
import gc
import logging
from functools import lru_cache
import math
import time
from functools import reduce
import weakref

import numpy as np
import torch
from torch import nn, Tensor

# Import memory optimization utilities
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from src.utils.memory_management import optimize_memory, register_tensor
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
from src.core.patterns.operadic_handler import OperadicStructureHandler
from src.validation.geometric.model import GeometricValidationResult
from src.validation.base import ValidationResult
from src.validation.geometric.symplectic import OperadicValidator
from src.validation.geometric.operadic_structure import OperadicStructureValidator
from src.core.crystal.scale_classes.ml.models import HolographicNet

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict


class ValidationError(Exception):
    """Raised when tensor validation fails."""
    pass


class MemoryError(Exception):
    """Raised when memory management fails."""
    pass


class ScalingError(Exception):
    """Raised when scaling computation fails."""
    pass


class OperatorError(Exception):
    """Raised when operator computation fails."""
    pass


class ComputationError(Exception):
    """Raised when general computation fails."""
    pass


class PhaseError(Exception):
    """Raised when phase tracking fails."""
    pass


class HolographicError(Exception):
    """Base class for holographic lifting errors."""
    pass


class ReconstructionError(Exception):
    """Error raised when UV reconstruction fails."""
    pass


class HolographicLifter(nn.Module):
    """Handles holographic lifting and UV/IR transformations.
    
    This class implements the holographic principle, mapping boundary (UV) data
    to bulk (IR) data through a radial dimension. The process incorporates both
    classical scaling and quantum corrections.
    
    Physics Background:
    - The holographic principle relates (d+1)-dimensional bulk physics to 
      d-dimensional boundary physics
    - The radial coordinate acts as an energy scale, with UV (high energy) at 
      small radius and IR (low energy) at large radius
    - Quantum corrections are computed through operator product expansion (OPE)
    - Phase evolution tracks the quantum interference effects
    
    Key Features:
    - Classical scaling based on conformal dimension
    - Quantum corrections through OPE
    - Phase evolution tracking
    - Memory-efficient tensor operations
    - Validation of geometric structures
    """
    
    # Physics constants
    Z_UV_DEFAULT = None    # Will be set based on dimension
    Z_IR_DEFAULT = None    # Will be set based on dimension
    
    # Numerical stability
    EPSILON = 1e-8                # Small number for numerical stability
    CORRECTION_SCALE_UV = 0.05    # UV correction strength (quantum effects at small distance)
    CORRECTION_SCALE_IR = 0.1     # IR correction strength (quantum effects at large distance)
    
    # Performance tuning
    CLEANUP_INTERVAL = 10         # Steps between memory cleanup
    MAX_CORRECTION_ORDER = 6      # Maximum order for IR corrections (convergence control)
    MAX_PHASES = 1000            # Maximum number of phases to track (quantum coherence)
    CACHE_SIZE = 64              # Memory manager cache size (performance vs memory tradeoff)
    CLEANUP_THRESHOLD = 0.7      # Memory cleanup threshold (fraction of peak memory)
    
    # Quantum corrections
    QUANTUM_SCALE = 0.1  # Scale of quantum corrections
    
    def __init__(
        self, 
        dim: int, 
        radial_points: int = 100,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize HolographicLifter.
        
        Args:
            dim: Dimension of the boundary theory
            radial_points: Number of points in radial direction (default: 100)
            dtype: Data type for tensor operations (supports complex)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        super().__init__()
        
        # Validate input parameters
        if dim < 1:
            raise ValidationError("Dimension must be positive")
        if radial_points < 2:
            raise ValidationError("Must have at least 2 radial points")
            
        # Basic parameters
        self.dim = dim
        self.dtype = dtype
        self.radial_points = radial_points
        
        # Physics-based scale initialization
        self.Z_UV = 1.0 / self.dim  # UV scale decreases with dimension
        self.Z_IR = float(self.dim)  # IR scale increases with dimension
        self.Z_RATIO = self.Z_UV / self.Z_IR  # Natural scaling ratio
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize memory manager with optimal cache size
        self.memory_manager = MemoryManager(
            cache_size=self.CACHE_SIZE,
            cleanup_threshold=self.CLEANUP_THRESHOLD
        )
        
        # Pre-allocate phase tracking with memory management
        self.max_phases = self.MAX_PHASES
        self.phase_evolution = self._allocate_tensor(
            (self.max_phases,), 
            dtype=torch.float32,
            name="phase_evolution"
        )
        self.current_phase = 0
        
        # Initialize neural network for holographic lifting
        self.holographic_net = HolographicNet(
            dim=self.dim,
            hidden_dim=self.dim * 4,  # Larger hidden dimension for better representation
            n_layers=4,  # More layers for deeper network
            dtype=self.dtype,
            z_uv=self.Z_UV,
            z_ir=self.Z_IR
        ).to(device=self.device)

        # Initialize operadic handler with quantum attention
        self.operad_handler = OperadicStructureHandler(
            base_dim=self.dim,
            hidden_dim=self.dim * 2,  # Double size for better representation
            preserve_symplectic=True,  # For quantum structure
            preserve_metric=True,      # For geometric structure
            dtype=self.dtype
        )
        
        # Initialize composition law with proper scaling
        self._initialize_composition_law()
        
        # Validate initial setup
        validation = self._validate_operadic_structure('quantum')
        if not validation.is_valid:
            raise ValidationError(f"Failed to initialize operadic structure: {validation.message}")

    def _update_memory_pressure(self) -> None:
        """Update memory pressure and trigger cleanup if needed.
        
        This method:
        1. Computes current memory pressure
        2. Updates exponential moving average
        3. Triggers cleanup if pressure exceeds threshold
        
        Memory pressure is defined as the ratio of currently allocated memory
        to peak memory usage. When this ratio exceeds the cleanup threshold,
        garbage collection is triggered.
        """
        try:
            # Get current memory usage
            allocated = self.memory_manager.get_allocated_memory()
            peak = self.memory_manager.get_peak_memory()
            
            # Update pressure using EMA
            current_pressure = allocated / (peak + self.EPSILON)
            if current_pressure > self.memory_manager._cleanup_threshold:
                self.memory_manager._cleanup_dead_refs()
                
        except Exception as e:
            logging.warning(f"Memory pressure update failed: {str(e)}")
            
    def _allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
        """Allocate a new tensor with memory tracking.
        
        Args:
            shape: Shape of tensor to allocate
            dtype: Data type of tensor
            name: Name for tracking purposes
            
        Returns:
            Newly allocated tensor
            
        Raises:
            MemoryError: If allocation fails
        """
        try:
            tensor = self.memory_manager.allocate_tensor(shape, dtype=dtype)
            return tensor
        except Exception as e:
            raise MemoryError(f"Failed to allocate tensor {name}: {str(e)}")
            
    def _free_tensor(self, tensor: torch.Tensor) -> None:
        """Free a tracked tensor.
        
        Args:
            tensor: Tensor to free
        """
        try:
            # Let memory manager handle cleanup through weak references
            tensor_id = id(tensor)
            if tensor_id in self.memory_manager._tensor_allocations:
                del self.memory_manager._tensor_allocations[tensor_id]
        except Exception as e:
            logging.warning(f"Failed to free tensor: {str(e)}")
            
    @contextmanager
    def _track_tensor(self, tensor: torch.Tensor, name: str) -> Generator[torch.Tensor, None, None]:
        """Context manager for tensor memory tracking.
        
        This manager ensures proper cleanup even if exceptions occur.
        
        Args:
            tensor: Tensor to track
            name: Name for tracking purposes
            
        Yields:
            The tracked tensor
            
        Raises:
            MemoryError: If tracking fails
        """
        try:
            # Register tensor with memory manager
            size = reduce(lambda x, y: x * y, tensor.shape) * tensor.element_size()
            self.memory_manager._tensor_allocations[id(tensor)] = size
            
            # Create weak reference for cleanup
            ref = weakref.ref(tensor, self.memory_manager._create_cleanup_callback(id(tensor)))
            self.memory_manager._tensor_refs.append(ref)
            
            yield tensor
            
        finally:
            try:
                # Cleanup if needed
                tensor_id = id(tensor)
                if tensor_id in self.memory_manager._tensor_allocations:
                    del self.memory_manager._tensor_allocations[tensor_id]
            except Exception as e:
                logging.warning(f"Failed to unregister tensor {name}: {str(e)}")
                
    def _ensure_memory_efficiency(self, min_free_memory: float = 0.2) -> None:
        """Ensure sufficient memory is available.
        
        This method:
        1. Checks current memory usage against peak
        2. Triggers cleanup if needed
        3. Raises error if cleanup insufficient
        
        Args:
            min_free_memory: Minimum fraction of free memory required
            
        Raises:
            MemoryError: If insufficient memory is available after cleanup
            
        Notes:
            The min_free_memory parameter should be set based on expected
            memory requirements for upcoming operations. Default of 0.2
            means at least 20% of peak memory should be free.
        """
        allocated = self.memory_manager.get_allocated_memory()
        peak = self.memory_manager.get_peak_memory()
        free_fraction = 1.0 - (allocated / peak if peak > 0 else 0.0)
        
        if free_fraction < min_free_memory:
            self.memory_manager._cleanup_dead_refs()
            
            # Check if cleanup helped
            allocated = self.memory_manager.get_allocated_memory()
            peak = self.memory_manager.get_peak_memory()
            free_fraction = 1.0 - (allocated / peak if peak > 0 else 0.0)
            
            if free_fraction < min_free_memory:
                raise MemoryError(
                    f"Insufficient memory: {free_fraction:.1%} free, need {min_free_memory:.1%}"
                )
        
    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype) if tensor.dtype != self.dtype else tensor
        
    
    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion with improved symmetry."""
        try:
            # Validate inputs
            op1 = self._ensure_dtype(op1)
            op2 = self._ensure_dtype(op2)

            # Normalize operators
            op1_norm = torch.norm(op1)
            op2_norm = torch.norm(op2)
            if op1_norm > self.EPSILON:
                op1 = op1 / op1_norm
            if op2_norm > self.EPSILON:
                op2 = op2 / op2_norm

            # Compute symmetric combination
            result = 0.5 * (op1 * op2 + op2 * op1)

            # Add quantum corrections
            overlap = torch.sum(torch.conj(op1) * op2).real
            correction = overlap * (op1 + op2) / 2

            # Ensure symmetry in final result
            result = result + correction * self.CORRECTION_SCALE_UV

            # Normalize result
            result_norm = torch.norm(result)
            if result_norm > self.EPSILON:
                result = result / result_norm

            return result

        except Exception as e:
            raise OperatorError(f"OPE computation failed: {str(e)}")

    def _track_phase(self, correction: torch.Tensor) -> None:
        """Track phase evolution from correction tensor.
        
        This method extracts and stores the phase information from quantum
        corrections, which is essential for tracking quantum coherence effects
        in the holographic lifting process.
        
        Physics Background:
        - Phase tracks quantum interference
        - Evolution captures coherence effects
        - Important for quantum corrections
        - Stored in phase evolution array
        
        Args:
            correction: Tensor to extract phase from
            
        Raises:
            ValidationError: If input validation fails
            MemoryError: If memory management fails
        """
        try:
            # Validate input
            validation = self._validate_tensor(correction, "correction")
            if not validation.is_valid:
                raise ValidationError(f"Invalid correction tensor: {validation.message}")
                
            correction = self._ensure_dtype(correction)
            
            # Track input tensor
            with self._track_tensor(correction, "correction"):
                if correction.is_complex() and self.current_phase < self.max_phases:
                    # Extract phase from first element
                    phase = torch.angle(correction.flatten()[0])
                    
                    # Update phase evolution array
                    self.phase_evolution[self.current_phase] = phase
                    self.current_phase += 1
                    
                    # Update memory pressure periodically
                    if self.current_phase % self.CLEANUP_INTERVAL == 0:
                        self._update_memory_pressure()
                        
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Memory management failed in phase tracking: {str(e)}")

    def _compute_radial_scaling(self, radial_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute radial scaling factors for bulk field reconstruction.
        
        This method computes the scaling factors needed for:
        1. Classical scaling: field ~ r^(-dim)
        2. Quantum corrections: ~ r^(2-dim)
        
        Physics Background:
        - Classical scaling follows from conformal invariance
        - Quantum corrections represent relevant deformations
        - The scaling dimensions determine the RG flow
        
        Args:
            radial_points: Radial coordinates for bulk reconstruction
            
        Returns:
            Tuple of (classical_scaling, quantum_scaling) tensors
            
        Raises:
            ScalingError: If scaling computation fails
        """
        try:
            # Validate input
            validation = self._validate_tensor(radial_points, "radial_points")
            if not validation.is_valid:
                raise ValidationError(f"Invalid radial points: {validation.message}")
                
            # Ensure proper dtype
            radial_points = self._ensure_dtype(radial_points)
            
            # Compute dimensionless ratios
            z_ratios = torch.abs(radial_points / radial_points[0])
            
            # Classical scaling
            classical_scaling = z_ratios**(-self.dim)
            
            # Quantum corrections
            quantum_scaling = z_ratios**(2 - self.dim)  # Relevant deformation
            
            # Track memory usage
            self._update_memory_pressure()
            
            return classical_scaling, quantum_scaling
            
        except Exception as e:
            raise ScalingError(f"Failed to compute radial scaling: {str(e)}")
            
    def _apply_radial_scaling(self, tensor: torch.Tensor, z_ratio: torch.Tensor) -> torch.Tensor:
        """Apply radial scaling to tensor with proper complex handling.
        
        Args:
            tensor: Input tensor to scale
            z_ratio: Ratio of z coordinates to scale by
            
        Returns:
            Scaled tensor with proper phase
        """
        try:
            # Handle complex tensors
            if tensor.is_complex():
                # Extract real and imaginary parts
                real_part = tensor.real
                imag_part = tensor.imag
                
                # Compute classical scaling
                scale = torch.abs(z_ratio)**(-self.dim)
                phase = torch.exp(-1j * torch.angle(z_ratio) * self.dim)
                
                # Apply scaling to real and imaginary parts separately
                scaled_real = real_part * scale
                scaled_imag = imag_part * scale
                
                # Recombine with proper phase
                scaled_complex = torch.complex(scaled_real, scaled_imag) * phase
                
                # Add quantum corrections
                quantum_power = 2 - self.dim  # Relevant deformation
                correction_scale = self.CORRECTION_SCALE_UV / (1 + torch.abs(z_ratio)**2)
                quantum_phase = torch.exp(-1j * torch.angle(z_ratio) * quantum_power)
                quantum_scale = torch.abs(z_ratio)**quantum_power
                
                # Apply quantum correction with proper phase
                correction = tensor * quantum_scale * quantum_phase * correction_scale
                
                return scaled_complex + correction
                
            else:
                # For real tensors, simpler scaling
                classical_scale = z_ratio**(-self.dim)
                scaled_tensor = tensor * classical_scale
                
                # Add quantum corrections for real case
                quantum_power = 2 - self.dim
                correction_scale = self.CORRECTION_SCALE_UV / (1 + z_ratio**2)
                correction = tensor * z_ratio**quantum_power * correction_scale
                
                return scaled_tensor + correction
                
        except Exception as e:
            raise ScalingError(f"Failed to apply radial scaling: {str(e)}")

    def holographic_lift(self, boundary_field: torch.Tensor, radial_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lift boundary field to bulk using holographic principle.
        
        This method implements the holographic lifting by:
        1. Computing classical scaling
        2. Adding quantum corrections
        3. Tracking phase evolution
        
        Args:
            boundary_field: Field values at UV boundary
            radial_points: Optional radial coordinates (default: None, uses preset)
            
        Returns:
            Bulk field values at specified radial points
            
        Raises:
            ValidationError: If input validation fails
            ScalingError: If scaling computation fails
        """
        try:
            # Validate and prepare inputs
            boundary_field = self._ensure_dtype(boundary_field)
            if radial_points is None:
                radial_points = torch.linspace(self.Z_UV, self.Z_IR, self.radial_points, dtype=self.dtype)
            radial_points = self._ensure_dtype(radial_points)

            # Initialize bulk field
            bulk_shape = (len(radial_points),) + boundary_field.shape
            bulk_field = torch.zeros(bulk_shape, dtype=self.dtype)

            # Compute UV data (first slice)
            bulk_field[0] = boundary_field.clone()

            # Compute bulk field at each radial point
            for i, z in enumerate(radial_points[1:], 1):
                # Compute dimensionless ratio
                z_ratio = z / radial_points[0]

                # Apply classical scaling with quantum corrections
                bulk_field[i] = self._apply_radial_scaling(boundary_field, z_ratio)

                # Track phase evolution
                self._track_phase(bulk_field[i])

            return bulk_field

        except Exception as e:
            raise ScalingError(f"Failed to compute holographic lift: {str(e)}")

    def extract_uv_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract UV (boundary) data from bulk field.
        
        The UV data represents the high-energy behavior of the field at the 
        boundary of the bulk space. This corresponds to the microscopic degrees
        of freedom in the dual field theory.
        
        Physics Background:
        - UV = high energy/small distance physics
        - Located at minimal radial coordinate
        - Represents microscopic degrees of freedom
        - Dual to local operators in boundary theory
        
        Args:
            field: Bulk field tensor with shape (radial_points, *boundary_dims)
            
        Returns:
            Tensor of boundary field values
            
        Raises:
            ValidationError: If field validation fails
            MemoryError: If memory management fails
        """
        try:
            # Validate input
            validation = self._validate_tensor(field, "bulk_field")
            if not validation.is_valid:
                raise ValidationError(f"Invalid bulk field: {validation.message}")
                
            if len(field.shape) < 1:
                raise ValidationError("Bulk field must have at least one dimension (radial)")
                
            if field.shape[0] != self.radial_points:
                raise ValidationError(
                    f"Bulk field must have {self.radial_points} radial points, got {field.shape[0]}"
                )
                
            field = self._ensure_dtype(field)
            
            # Track tensor in memory manager
            with self._track_tensor(field, "bulk_field"):
                # Extract UV data (boundary slice)
                uv_shape = field.shape[1:]  # Remove radial dimension
                uv_data = self._allocate_tensor(uv_shape, self.dtype, "uv_data")
                uv_data.copy_(field[0])  # Copy boundary slice
                
                # Track UV data
                with self._track_tensor(uv_data, "uv_data"):
                    return uv_data
                    
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Memory management failed in UV extraction: {str(e)}")

    def extract_ir_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract IR (deep bulk) data from bulk field.
        
        The IR data represents the low-energy behavior of the field deep in
        the bulk. This corresponds to the macroscopic, collective degrees
        of freedom in the dual field theory.
        
        Physics Background:
        - IR = low energy/large distance physics
        - Located at maximal radial coordinate
        - Represents macroscopic/collective behavior
        - Dual to non-local operators in boundary theory
        
        Args:
            field: Bulk field tensor with shape (radial_points, *boundary_dims)
            
        Returns:
            Tensor of deep bulk field values
            
        Raises:
            ValidationError: If field validation fails
            MemoryError: If memory management fails
        """
        try:
            # Validate input
            validation = self._validate_tensor(field, "bulk_field")
            if not validation.is_valid:
                raise ValidationError(f"Invalid bulk field: {validation.message}")
                
            if len(field.shape) < 1:
                raise ValidationError("Bulk field must have at least one dimension (radial)")
                
            if field.shape[0] != self.radial_points:
                raise ValidationError(
                    f"Bulk field must have {self.radial_points} radial points, got {field.shape[0]}"
                )
                
            field = self._ensure_dtype(field)
            
            # Track tensor in memory manager
            with self._track_tensor(field, "bulk_field"):
                # Extract IR data (deepest bulk slice)
                ir_shape = field.shape[1:]  # Remove radial dimension
                ir_data = self._allocate_tensor(ir_shape, self.dtype, "ir_data")
                ir_data.copy_(field[-1])  # Copy deepest bulk slice
                
                # Track IR data
                with self._track_tensor(ir_data, "ir_data"):
                    return ir_data
                    
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Memory management failed in IR extraction: {str(e)}")

    def reconstruct_from_ir(self, ir_field: torch.Tensor) -> torch.Tensor:
        """Reconstruct UV field from IR data.
        
        This method implements the inverse holographic mapping by:
        1. Computing inverse classical scaling
        2. Subtracting quantum corrections
        3. Preserving phase information
        
        Args:
            ir_field: Field values at IR boundary
            
        Returns:
            Reconstructed UV field values
            
        Raises:
            ValidationError: If input validation fails
            ScalingError: If scaling computation fails
        """
        try:
            # Validate input
            ir_field = self._ensure_dtype(ir_field)
            
            # Compute dimensionless ratio
            z_ratio = torch.tensor(self.Z_IR / self.Z_UV, dtype=ir_field.dtype)
            
            # Compute inverse classical scaling
            if ir_field.is_complex():
                phase = torch.exp(1j * torch.angle(z_ratio) * self.dim)
                classical_scale = torch.abs(z_ratio)**self.dim
            else:
                phase = torch.tensor(1.0, dtype=ir_field.dtype)
                classical_scale = z_ratio**self.dim
            
            # Apply inverse classical scaling with proper phase
            uv_field = ir_field * classical_scale * phase
            
            # Subtract quantum corrections
            quantum_power = 2 - self.dim  # Relevant deformation
            correction_scale = -self.CORRECTION_SCALE_UV / (1 + torch.abs(z_ratio)**2)  # Note the negative sign
            
            # Compute inverse quantum correction with proper phase
            if ir_field.is_complex():
                phase = torch.exp(1j * torch.angle(z_ratio) * quantum_power)
                quantum_scale = torch.abs(z_ratio)**quantum_power
            else:
                phase = torch.tensor(1.0, dtype=ir_field.dtype)
                quantum_scale = z_ratio**quantum_power
            
            # Apply inverse quantum correction
            correction = uv_field * quantum_scale * phase * correction_scale
            uv_field = uv_field + correction
            
            return uv_field
            
        except Exception as e:
            raise ScalingError(f"Failed to reconstruct UV field: {str(e)}")

    def reconstruct_from_uv(self, uv_data: torch.Tensor) -> torch.Tensor:
        """Reconstruct IR data from UV data using holographic principle.
        
        This method reconstructs the IR (deep bulk) data from UV (boundary) data
        through the holographic lifting process. It captures how collective
        behavior emerges from microscopic degrees of freedom.
        
        Physics Background:
        - Reconstruction follows RG flow
        - Captures emergence of collective behavior
        - Includes quantum corrections
        - Preserves holographic correspondence
        
        Args:
            uv_data: Field values at UV boundary
            
        Returns:
            Reconstructed field values at IR boundary
            
        Raises:
            ValidationError: If input validation fails or tensor properties are invalid
            MemoryError: If memory management fails during tensor operations
            HolographicError: If reconstruction process fails due to physics constraints
            
        Notes:
            The reconstruction process follows the RG flow from UV to IR,
            incorporating quantum corrections at each scale. The method ensures
            that the holographic correspondence is preserved throughout the
            reconstruction process.
        """
        try:
            # Validate input
            validation = self._validate_tensor(uv_data, "uv_data")
            if not validation.is_valid:
                raise ValidationError(f"Invalid UV data: {validation.message}")
                
            uv_data = self._ensure_dtype(uv_data)
            
            # Track input tensor
            with self._track_tensor(uv_data, "uv_data"):
                # Store original norm
                uv_norm = torch.norm(uv_data)
                
                # Initialize IR data with leading term
                ir_data = self._allocate_tensor(uv_data.shape, self.dtype, "ir_data")
                ir_data.copy_(uv_data)
                ir_data.mul_(self.Z_RATIO ** self.dim)  # Apply IR scaling
                
                # Track IR data
                with self._track_tensor(ir_data, "ir_data"):
                    # Add quantum corrections
                    correction_sum = self._allocate_tensor(uv_data.shape, self.dtype, "corrections")
                    correction_sum.zero_()
                    
                    # Track corrections
                    with self._track_tensor(correction_sum, "corrections"):
                        for n in range(1, 6):  # Increase order of corrections
                            # Each correction includes quantum effects
                            sign = (-1)**n
                            power = self.dim + 2*n
                            scale = self.Z_RATIO ** power / math.factorial(n)
                            
                            correction = uv_data * scale * sign
                            correction_sum.add_(correction)
                            
                            # Periodic memory optimization in reconstruction
                            if n % 2 == 0:
                                self._update_memory_pressure()
                                self._ensure_memory_efficiency(min_free_memory=0.2)  # Standard threshold for corrections
                        
                        # Scale corrections for stability
                        correction_scale = 0.1 / (1 + self.Z_RATIO**2)
                        ir_data.add_(correction_sum.mul(correction_scale))
                        
                        return ir_data
                        
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Memory management failed in UV reconstruction: {str(e)}")

    def compute_c_function(self, bulk_field: torch.Tensor, radial_points: torch.Tensor) -> torch.Tensor:
        """Compute c-function along RG flow.
        
        The c-function measures the number of degrees of freedom
        at each energy scale. It must decrease monotonically from
        UV to IR according to the c-theorem.
        
        Physics Background:
        - Measures effective degrees of freedom
        - Monotonically decreasing along RG flow
        - Related to entanglement entropy
        - Captures quantum phase transitions
        
        Args:
            bulk_field: Field values in bulk
            radial_points: Radial coordinates
            
        Returns:
            c-function values at each radial point
            
        Raises:
            ValidationError: If input validation fails
            ComputationError: If c-function computation fails
        """
        try:
            # Validate inputs
            for name, tensor in [("bulk_field", bulk_field), ("radial_points", radial_points)]:
                validation = self._validate_tensor(tensor, name)
                if not validation.is_valid:
                    raise ValidationError(f"Invalid {name}: {validation.message}")
                    
            bulk_field = self._ensure_dtype(bulk_field)
            radial_points = self._ensure_dtype(radial_points)
            
            # Track input tensors
            with self._track_tensor(bulk_field, "bulk_field"), \
                 self._track_tensor(radial_points, "radial_points"):
                
                # Compute field gradients
                gradients = self._allocate_tensor(bulk_field.shape, self.dtype, "gradients")
                gradients[:-1] = bulk_field[1:] - bulk_field[:-1]
                gradients[-1] = gradients[-2]  # Copy last gradient
                
                # Track gradients
                with self._track_tensor(gradients, "gradients"):
                    # Compute c-function components
                    c_func = self._allocate_tensor(radial_points.shape, self.dtype, "c_function")
                    
                    # Track c-function
                    with self._track_tensor(c_func, "c_function"):
                        # Classical contribution (UV dominant)
                        classical = torch.abs(bulk_field) ** 2
                        # Reduce over non-radial dimensions
                        classical = classical.mean(dim=tuple(range(1, len(classical.shape))))
                        
                        # Quantum corrections (IR relevant)
                        quantum = torch.abs(gradients) ** 2
                        # Reduce over non-radial dimensions
                        quantum = quantum.mean(dim=tuple(range(1, len(quantum.shape))))
                        # Scale by radial coordinate
                        quantum = quantum * torch.abs(radial_points) ** 2
                        
                        # Combine with proper scaling
                        c_func = classical + self.QUANTUM_SCALE * quantum
                        
                        # Ensure monotonicity by taking cumulative minimum
                        if c_func.is_complex():
                            # For complex tensors, apply to real and imaginary parts separately
                            real_cummin = torch.cummin(c_func.real, dim=0)[0]
                            imag_cummin = torch.cummin(c_func.imag, dim=0)[0]
                            c_func = torch.complex(real_cummin, imag_cummin)
                        else:
                            # For real tensors, apply directly
                            c_func = torch.cummin(c_func, dim=0)[0]
                        
                        # Add UV enhancement
                        uv_factor = torch.exp(-torch.abs(radial_points) / self.Z_UV)
                        c_func = c_func * (1 + uv_factor)
                        
                        return c_func
                        
        except ValidationError:
            raise
        except Exception as e:
            raise ComputationError(f"Failed to compute c-function: {str(e)}")

    def compute_ope(self, field: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion coefficients.
        
        The OPE captures quantum corrections to classical field behavior:
        - For real fields: uses spatial gradients to capture local corrections
        - For complex fields: separates into real and imaginary parts
        
        Physics Background:
        - OPE represents short-distance product of quantum operators
        - The expansion captures all possible local operators
        - Coefficients determine the strength of quantum corrections
        - Phase information tracks quantum coherence effects
        - Real/imaginary parts handle different quantum sectors
        - Spatial gradients capture local quantum fluctuations
        
        Implementation Details:
        - Pre-allocates tensors for real/imaginary parts
        - Handles both real and complex fields
        - Tracks memory usage for large fields
        - Preserves quantum phase information
        """
        try:
            # Validate input
            validation = self._validate_tensor(field, "input_field")
            if not validation.is_valid:
                raise ValidationError(f"Invalid input field: {validation.message}")
                
            field = self._ensure_dtype(field)
            
            # Track input field
            with self._track_tensor(field, "input_field"):
                # For complex fields, compute OPE in a phase-preserving way
                if field.is_complex():
                    # Pre-allocate real and imaginary parts
                    op1 = self._allocate_tensor(field.shape, torch.float32, "op1_real")
                    op2 = self._allocate_tensor(field.shape, torch.float32, "op2_imag")
                    
                    # Extract real and imaginary parts
                    op1.copy_(field.real)
                    op2.copy_(field.imag)
                    
                    # Track operators and compute OPE
                    with self._track_tensor(op1, "op1_real"), \
                         self._track_tensor(op2, "op2_imag"):
                        # Compute OPE between real and imaginary parts
                        real_ope = 0.5 * (op1 * op2 + op2 * op1)  # Symmetrized cross term
                        imag_ope = 0.5 * (op2 * op1 - op1 * op2)  # Anti-symmetrized cross term
                        
                        # Combine with proper phase
                        correction = torch.complex(real_ope, imag_ope)
                        
                        # Normalize to preserve OPE symmetry
                        norm = torch.norm(correction)
                        if norm > self.EPSILON:
                            correction = correction / norm
                        else:
                            correction = torch.zeros_like(correction)
                        
                        return correction
                else:
                    # For real fields, use spatial shift
                    op1 = field
                    
                    # Pre-allocate shifted operator
                    op2 = self._allocate_tensor(field.shape, field.dtype, "op2_shifted")
                    op2.copy_(torch.roll(field, shifts=1, dims=-1))
                    
                    # Track shifted operator and compute OPE
                    with self._track_tensor(op2, "op2_shifted"):
                        # Compute OPE between different operators
                        correction = 0.5 * (op1 * op2 + op2 * op1)  # Symmetrized product
                        
                        # Normalize to preserve OPE symmetry
                        norm = torch.norm(correction)
                        if norm > self.EPSILON:
                            correction = correction / norm
                        else:
                            correction = torch.zeros_like(correction)
                        
                        return correction
                    
        except ValidationError:
            raise
        except Exception as e:
            raise OperatorError(f"Failed to compute OPE: {str(e)}")

    def _validate_operadic_structure(self, structure_type: str) -> GeometricValidationResult:
        """Validate operadic structure type and properties.
        
        This method validates both the structure type and its geometric properties:
        - Structure type must be one of the valid types
        - Symplectic form must be preserved if quantum structure
        - Metric must be preserved if geometric structure
        - Base dimension must match holographic dimension
        
        Args:
            structure_type: Type of structure to preserve
            
        Returns:
            GeometricValidationResult with validation status and details
        """
        try:
            # Use OperadicStructureValidator for validation
            validator = OperadicStructureValidator(tolerance=self.EPSILON)
            result = validator.validate_all(
                handler=self.operad_handler,
                structure_type=structure_type,
                expected_dim=self.dim,
                dtype=self.dtype
            )
            
            # Convert OperadicValidationResult to GeometricValidationResult
            return GeometricValidationResult(
                is_valid=result.is_valid,
                message=result.message,
                data=result.data
            )
            
        except Exception as e:
            return GeometricValidationResult(
                is_valid=False,
                message=f"Error during operadic validation: {str(e)}",
                data={
                    'structure_type': structure_type,
                    'error': str(e)
                }
            )

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> GeometricValidationResult:
        """Validate tensor properties.
        
        This method performs comprehensive tensor validation:
        - Type checking
        - Value range validation (NaN, Inf)
        - Device compatibility
        - Dtype compatibility
        - Shape constraints based on tensor name
        
        Args:
            tensor: Tensor to validate
            name: Name of tensor for error messages
            
        Returns:
            GeometricValidationResult with validation status and details
        """
        if not isinstance(tensor, torch.Tensor):
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} must be a torch.Tensor, got {type(tensor)}",
                data={
                    "expected_type": "torch.Tensor",
                    "actual_type": str(type(tensor)),
                    "tensor_name": name
                }
            )
            
        if torch.isnan(tensor).any():
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} contains NaN values",
                data={
                    "has_nan": True,
                    "tensor_name": name,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype)
                }
            )
            
        if torch.isinf(tensor).any():
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} contains infinite values",
                data={
                    "has_inf": True,
                    "tensor_name": name,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype)
                }
            )
            
        # Validate device compatibility
        if tensor.device != self.device:
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} must be on device {self.device}, got {tensor.device}",
                data={
                    "expected_device": str(self.device),
                    "actual_device": str(tensor.device),
                    "tensor_name": name,
                    "shape": tensor.shape
                }
            )
            
        # Validate dtype compatibility
        if tensor.dtype != self.dtype:
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} must have dtype {self.dtype}, got {tensor.dtype}",
                data={
                    "expected_dtype": str(self.dtype),
                    "actual_dtype": str(tensor.dtype),
                    "tensor_name": name,
                    "shape": tensor.shape
                }
            )
            
        # Validate tensor is contiguous for performance
        if not tensor.is_contiguous():
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} must be contiguous for optimal performance",
                data={
                    "is_contiguous": False,
                    "tensor_name": name,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device)
                }
            )
            
        # Validate tensor shape based on name
        if name == "radial_points" and len(tensor.shape) != 1:
            return GeometricValidationResult(
                is_valid=False,
                message=f"{name} must be 1-dimensional, got shape {tensor.shape}",
                data={
                    "expected_dims": 1,
                    "actual_dims": len(tensor.shape),
                    "shape": tensor.shape,
                    "tensor_name": name
                }
            )
            
        # All validations passed
        return GeometricValidationResult(
            is_valid=True, 
            message=f"{name} validation passed",
            data={
                "tensor_name": name,
                "shape": tensor.shape,
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "is_contiguous": True,
                "has_nan": False,
                "has_inf": False,
                "memory_size": tensor.numel() * tensor.element_size()
            }
        )

    def validate_holographic_data(
        self,
        boundary_field: torch.Tensor,
        radial_points: torch.Tensor
    ) -> GeometricValidationResult:
        """Validate data for holographic lifting.
    
        This method validates both the boundary field and radial coordinates:
        - Ensures proper tensor properties (no NaN/Inf)
        - Validates radial coordinate structure
        - Checks device and dtype compatibility
    
        Physics Background:
        - Boundary field represents UV (high-energy) data
        - Radial coordinate acts as energy scale
        - Must be strictly increasing for RG flow
        - UV/IR ratio determines scaling behavior
    
        Args:
            boundary_field: Field values at UV boundary
            radial_points: Radial coordinates
    
        Returns:
            GeometricValidationResult with validation status and metrics
    
        Notes:
            The validation ensures that the holographic lifting process
            will maintain both numerical stability and physical meaning.
            The radial points must increase monotonically to represent
            proper RG flow from UV to IR.
        """
        # Validate tensors
        field_validation = self._validate_tensor(boundary_field, "boundary_field")
        radial_validation = self._validate_tensor(radial_points, "radial_points")
    
        if not (field_validation.is_valid and radial_validation.is_valid):
            return GeometricValidationResult(
                is_valid=False,
                message=f"{field_validation.message}; {radial_validation.message}"
            )
    
        # Validate device compatibility
        if boundary_field.device != radial_points.device:
            return GeometricValidationResult(
                is_valid=False,
                message="boundary_field and radial_points must be on same device",
                data={
                    "boundary_device": str(boundary_field.device),
                    "radial_device": str(radial_points.device)
                }
            )
    
        # Validate dtype compatibility
        if not (boundary_field.dtype == self.dtype and radial_points.dtype == self.dtype):
            return GeometricValidationResult(
                is_valid=False,
                message=f"Tensors must have dtype {self.dtype}",
                data={
                    "boundary_dtype": str(boundary_field.dtype),
                    "radial_dtype": str(radial_points.dtype)
                }
            )
    
        # Validate radial points structure
        if len(radial_points.shape) != 1:
            return GeometricValidationResult(
                is_valid=False,
                message=f"radial_points must be 1D, got shape {radial_points.shape}",
                data={"shape": radial_points.shape}
            )
    
        if len(radial_points) < 2:
            return GeometricValidationResult(
                is_valid=False,
                message="Need at least 2 radial points for lifting",
                data={"num_points": len(radial_points)}
            )
    
        # For complex radial points, compare real parts
        # Imaginary parts should be zero for physical radial coordinates
        if radial_points.is_complex():
            if not torch.allclose(radial_points.imag, torch.zeros_like(radial_points.imag), atol=self.EPSILON):
                return GeometricValidationResult(
                    is_valid=False,
                    message="Radial points must have zero imaginary part",
                    data={"max_imag": torch.max(torch.abs(radial_points.imag)).item()}
                )
            radial_diffs = radial_points[1:].real - radial_points[:-1].real
            if not torch.all(radial_diffs > 0):
                return GeometricValidationResult(
                    is_valid=False,
                    message="Radial points must be strictly increasing",
                    data={"min_diff": torch.min(radial_diffs).item()}
                )
        else:
            if not torch.all(radial_points[1:] > radial_points[:-1]):
                return GeometricValidationResult(
                    is_valid=False,
                    message="Radial points must be strictly increasing",
                    data={"min_diff": torch.min(radial_points[1:] - radial_points[:-1]).item()}
                )
        
        return GeometricValidationResult(
            is_valid=True,
            message="Holographic data validated successfully",
            data={
                "num_points": len(radial_points),
                "uv_ir_ratio": (radial_points[-1] / radial_points[0]).real.item()
                if radial_points.is_complex() else
                (radial_points[-1] / radial_points[0]).item()
            }
        )

    def _initialize_composition_law(self):
        """Initialize composition law for operadic structure.
        
        This method initializes the composition law that governs how operators
        compose in the operadic structure. The composition law preserves both
        quantum and geometric structures.
        """
        # Create initial composition law with proper dimensions
        composition_law = torch.eye(self.dim, dtype=self.dtype, device=self.device)
        
        # Add quantum structure through scaling - use float32 for arange then convert to complex
        quantum_scale = torch.exp(-torch.arange(self.dim, dtype=torch.float32, device=self.device))
        if self.dtype.is_complex:
            quantum_scale = torch.complex(quantum_scale, torch.zeros_like(quantum_scale))
            quantum_scale = quantum_scale.to(self.dtype)
        
        # Add geometric structure through parallel transport
        if hasattr(self.operad_handler, 'transport'):
            composition_law = self.operad_handler.transport(
                composition_law,
                torch.zeros_like(composition_law),
                composition_law
            )
            
        # Scale the composition law
        composition_law = composition_law * quantum_scale.unsqueeze(0)
        
        # Normalize the composition law
        composition_law = composition_law / torch.norm(composition_law)
        
        # Set the composition law in the operadic handler
        self.operad_handler.composition_law = composition_law
        
        # Validate the composition law
        validation = self._validate_operadic_structure('quantum')
        if not validation.is_valid:
            raise ValidationError(f"Failed to initialize composition law: {validation.message}")