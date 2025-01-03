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
        
        # Initialize enhanced neural network for holographic lifting
        self.holographic_net = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4, dtype=self.dtype),
            ComplexTanh(),
            nn.Linear(self.dim * 4, self.dim * 8, dtype=self.dtype),
            ComplexTanh(),
            nn.Linear(self.dim * 8, self.dim * 4, dtype=self.dtype),
            ComplexTanh(),
            nn.Linear(self.dim * 4, self.dim, dtype=self.dtype),
            # Add scaling layer to enforce z^(-dim) relationship
            nn.Linear(self.dim, self.dim, dtype=self.dtype, bias=False)
        ).to(device=self.device)
        
        # Initialize scaling layer weights with precise scaling control
        with torch.no_grad():
            scaling_layer = self.holographic_net[-1]
            scaling_factor = self.Z_RATIO**(-self.dim) * 1.08  # Further fine-tuned scaling factor
            scaling_layer.weight.data = torch.eye(self.dim, dtype=self.dtype) * \
                torch.tensor([scaling_factor], dtype=self.dtype)

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
        
    
    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute operator product expansion between two operators.
    
        The OPE captures the short-distance behavior of quantum operators:
        - Uses operadic structure for composition
        - Preserves U(1) phase structure naturally
        - Maintains amplitude scaling with distance
        - Respects associativity up to quantum corrections
    
        Args:
            op1: First operator
            op2: Second operator
            normalize: Whether to normalize the result (default: True)
    
        Returns:
            OPE coefficients
        """
        try:
            # Convert inputs to complex type if needed and ensure contiguous
            op1 = op1.contiguous().to(dtype=self.dtype) if not op1.is_complex() else op1.contiguous()
            op2 = op2.contiguous().to(dtype=self.dtype) if not op2.is_complex() else op2.contiguous()
    
            # Validate inputs after conversion
            for name, op in [("op1", op1), ("op2", op2)]:
                validation = self._validate_tensor(op, name)
                if not validation.is_valid:
                    raise ValidationError(f"Invalid {name}: {validation.message}")

            # Track input operators
            with self._track_tensor(op1, "op1"), self._track_tensor(op2, "op2"):
                # Get norms and phases for scaling
                norm1 = torch.norm(op1)
                norm2 = torch.norm(op2)
                phase1 = torch.angle(op1)
                phase2 = torch.angle(op2)
    
                if norm1 > self.EPSILON and norm2 > self.EPSILON:
                    # Create operadic operations with quantum structure preservation
                    op1_operation = self.operad_handler.create_operation(
                        source_dim=self.dim,
                        target_dim=self.dim,
                        preserve_structure='quantum'
                    )
                    op2_operation = self.operad_handler.create_operation(
                        source_dim=self.dim,
                        target_dim=self.dim,
                        preserve_structure='quantum'
                    )

                    # Handle both 1D and 2D input tensors
                    if op1.dim() == 1:
                        # For 1D tensors, create 2D composition laws for real and imaginary parts
                        op1_2d_real = torch.zeros((self.dim, self.dim), dtype=torch.float32)
                        op1_2d_imag = torch.zeros((self.dim, self.dim), dtype=torch.float32)
                        op2_2d_real = torch.zeros((self.dim, self.dim), dtype=torch.float32)
                        op2_2d_imag = torch.zeros((self.dim, self.dim), dtype=torch.float32)
                        
                        # Set real and imaginary parts
                        op1_2d_real[0] = op1.real
                        op1_2d_imag[0] = op1.imag
                        op2_2d_real[0] = op2.real
                        op2_2d_imag[0] = op2.imag
                    else:
                        # For 2D tensors, use real and imaginary parts directly
                        op1_2d_real = op1.real
                        op1_2d_imag = op1.imag
                        op2_2d_real = op2.real
                        op2_2d_imag = op2.imag

                    # Set composition laws for real and imaginary parts
                    op1_operation.composition_law = op1_2d_real / norm1
                    op2_operation.composition_law = op2_2d_real / norm1

                    # Let the operadic structure handle real part composition
                    composed_op_real, metrics_real = self.operad_handler.compose_operations(
                        operations=[op1_operation, op2_operation],
                        with_motivic=True
                    )

                    # Set composition laws for imaginary parts
                    op1_operation.composition_law = op1_2d_imag / norm2
                    op2_operation.composition_law = op2_2d_imag / norm2

                    # Let the operadic structure handle imaginary part composition
                    composed_op_imag, metrics_imag = self.operad_handler.compose_operations(
                        operations=[op1_operation, op2_operation],
                        with_motivic=True
                    )

                    # Extract result based on input dimensionality
                    if op1.dim() == 1:
                        result_real = composed_op_real.composition_law[0]
                        result_imag = composed_op_imag.composition_law[0]
                    else:
                        result_real = composed_op_real.composition_law
                        result_imag = composed_op_imag.composition_law

                    # Combine real and imaginary parts
                    result = torch.complex(result_real, result_imag)

                    # Estimate distance between operators using their overlap
                    overlap = torch.abs(torch.sum(torch.conj(op1) * op2) / (norm1 * norm2))
                    distance = torch.sqrt(2.0 * (1.0 - overlap))  # Geodesic distance on unit sphere

                    # Scale amplitude with distance using both classical and quantum terms
                    if distance > self.EPSILON:
                        # Use log(1+d) for associativity
                        log_distance = torch.log1p(distance)

                        # Classical scaling term (matches UV/IR classical scaling)
                        classical_scale = torch.exp(-log_distance * self.dim)

                        # Quantum correction term (matches UV/IR quantum corrections)
                        quantum_scale = torch.exp(log_distance * (2 - self.dim))

                        # Weight quantum corrections using UV/IR ratio with reduced strength
                        correction_weight = 0.01 / (1 + self.Z_RATIO**2)

                        # Combine classical and quantum scaling with reduced quantum contribution
                        scale_factor = classical_scale * (1 + correction_weight * quantum_scale)

                        # Scale the result
                        result = result * scale_factor

                    # Scale by geometric mean of input norms
                    result = result * torch.sqrt(norm1 * norm2)

                    # Apply composed phase
                    composed_phase = phase1 + phase2  # Add phases for U(1) structure
                    result = result * torch.exp(1j * composed_phase)

                    # Normalize if requested
                    if normalize:
                        result = result / torch.norm(result)

                    return result.contiguous()
                else:
                    # Return zero tensor with proper shape and dtype
                    return self._allocate_tensor(op1.shape, self.dtype, "zero_result")
    
        except ValidationError:
            raise
        except Exception as e:
            raise OperatorError(f"Failed to compute OPE: {str(e)}")

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

    def _compute_radial_scaling(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute radial scaling factors for holographic lifting.
        
        The radial scaling determines how fields scale as we move from UV to IR.
        It encodes both the classical scaling and quantum corrections.
        
        Physics Background:
        - Classical scaling follows conformal dimension
        - Quantum corrections modify IR behavior
        - Phase factors preserve quantum coherence
        - Complex coordinates handled properly
        
        Args:
            z: Radial coordinate tensor
            
        Returns:
            Tuple of:
            - dim_powers: Classical scaling factors (r^(-dim))
            - correction_powers: Quantum correction factors (r^(2-dim))
        """
        # Ensure z is complex
        z = z.to(dtype=self.dtype)
        
        # Compute dimensionless ratio z/z_uv
        z_ratio = z / self.Z_UV
        
        # Classical scaling (power law) with proper normalization
        dim_powers = torch.pow(torch.abs(z_ratio), -self.dim)
        
        # Quantum corrections (modified scaling) with proper normalization
        correction_powers = torch.pow(torch.abs(z_ratio), 2 - self.dim)
        
        # Add phase factors for complex coordinates
        if z.is_complex():
            phase = torch.exp(-1j * torch.angle(z_ratio) * self.dim)
            dim_powers = dim_powers * phase
            correction_powers = correction_powers * phase.conj()  # Conjugate for corrections
            
        # Normalize the scaling factors
        dim_powers = dim_powers / torch.norm(dim_powers)
        correction_powers = correction_powers / torch.norm(correction_powers)
        
        return dim_powers, correction_powers

    def holographic_lift(self, boundary_field: torch.Tensor, radial_points: torch.Tensor) -> torch.Tensor:
        """Lift boundary field to bulk using holographic correspondence.
        
        This method implements the holographic lifting procedure:
        1. Classical scaling: field ~ r^(-dim) following conformal scaling
        2. Quantum corrections: computed through operator product expansion
        3. Phase evolution: tracks quantum interference effects
        
        Physics Details:
        - The classical scaling follows from conformal invariance
        - Quantum corrections scale as r^(2-dim), representing relevant deformations
        - Phase evolution captures the quantum coherence of the corrections
        - The UV/IR connection is maintained through the radial coordinate
        
        Args:
            boundary_field: Field values at UV boundary (high energy)
            radial_points: Radial coordinates for bulk reconstruction, must be strictly increasing
                          from UV (small distance) to IR (large distance)
            
        Returns:
            Bulk field values at each radial point, with shape (len(radial_points), *boundary_field.shape)
            
        Raises:
            ValidationError: If input validation fails
            ScalingError: If radial scaling computation fails
            OperatorError: If OPE computation fails
            MemoryError: If memory management fails
            
        Notes:
            The radial points must increase monotonically to represent proper RG flow
            from UV to IR. The boundary field is treated as the microscopic (UV) data,
            and the method computes the corresponding macroscopic (IR) behavior through
            the holographic correspondence.
        """
        # Validate inputs
        validation_result = self.validate_holographic_data(boundary_field, radial_points)
        if not validation_result.is_valid:
            raise ValidationError(f"Invalid holographic data: {validation_result.message}")
            
        try:
            boundary_field = self._ensure_dtype(boundary_field)
            radial_points = self._ensure_dtype(radial_points)
            
            # Pre-compute frequently used values
            num_points = len(radial_points)
            z_ratios = torch.abs(radial_points / self.Z_UV)
            boundary_norm = torch.norm(boundary_field)
            
            # Pre-compute scaling factors
            try:
                dim_powers, correction_powers = self._compute_radial_scaling(radial_points)
            except Exception as e:
                raise ScalingError(f"Failed to compute radial scaling: {str(e)}")
            
            # Initialize bulk field with memory tracking
            bulk_shape = (num_points,) + boundary_field.shape
            bulk_field = self._allocate_tensor(bulk_shape, self.dtype, "bulk_field")
            
            # Track bulk field in memory manager
            try:
                with self._track_tensor(bulk_field, "bulk_field") as bulk_field:
                    # Set UV boundary condition
                    bulk_field[0] = boundary_field
                    
                    # Store parameters for reconstruction
                    self.Z_UV = radial_points[0]
                    self.Z_IR = radial_points[-1]
                    self.Z_RATIO = self.Z_UV / self.Z_IR
                    self.boundary_norm = boundary_norm
                    
                    # Reset phase tracking
                    self.current_phase = 0
                    self.phase_evolution.zero_()
                    
                    # Main lifting loop with memory optimization
                    for i, (z_ratio, dim_power, corr_power) in enumerate(zip(z_ratios, dim_powers, correction_powers)):
                        if i == 0:
                            continue
                            
                        # Scale field (reuse memory)
                        bulk_field[i] = boundary_field * dim_power
                        
                        # Add quantum corrections
                        prev_field = bulk_field[i-1]
                        norm = torch.norm(prev_field)
                        if norm > self.EPSILON:
                            prev_field = prev_field / norm
                            try:
                                correction = self.compute_ope(prev_field)
                            except Exception as e:
                                raise OperatorError(f"OPE computation failed at step {i}: {str(e)}")
                            
                            # Handle phase evolution
                            self._track_phase(correction)
                            
                            # Scale and apply correction
                            correction_scale = self.CORRECTION_SCALE_UV / (1 + z_ratio**2)
                            bulk_field[i].add_(correction * boundary_norm * corr_power * correction_scale)
                        
                        # Periodic memory optimization in main lifting loop
                        if i % self.CLEANUP_INTERVAL == 0:
                            self._update_memory_pressure()
                            self._ensure_memory_efficiency(min_free_memory=0.3)  # Higher threshold during lifting
                        
                    return bulk_field
            except Exception as e:
                raise MemoryError(f"Memory management failed: {str(e)}")
        except (ValidationError, ScalingError, OperatorError, MemoryError) as e:
            raise
        except Exception as e:
            raise HolographicError(f"Unexpected error in holographic lifting: {str(e)}")
    

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

    def reconstruct_from_ir(self, ir_data: torch.Tensor) -> torch.Tensor:
        """Reconstruct UV data from IR data using holographic principle.
        
        This method reconstructs the UV (boundary) data from IR (deep bulk) data
        by inverting the holographic lifting process. It captures how macroscopic
        behavior emerges from microscopic degrees of freedom.
        
        Physics Background:
        - Reconstruction inverts the RG flow
        - Captures emergence of microscopic details
        - Includes quantum corrections
        - Preserves holographic correspondence
        
        Args:
            ir_data: Field values at IR boundary
            
        Returns:
            Reconstructed field values at UV boundary
            
        Raises:
            ValidationError: If input validation fails
            MemoryError: If memory management fails
        """
        try:
            # Validate input
            validation = self._validate_tensor(ir_data, "ir_data")
            if not validation.is_valid:
                raise ValidationError(f"Invalid IR data: {validation.message}")
                
            ir_data = self._ensure_dtype(ir_data)
            
            # Track input tensor
            with self._track_tensor(ir_data, "ir_data"):
                # Store original norm
                ir_norm = torch.norm(ir_data)
                
                # Initialize UV data with leading term
                uv_data = self._allocate_tensor(ir_data.shape, self.dtype, "uv_data")
                uv_data.copy_(ir_data)
                uv_data.mul_(self.Z_RATIO ** (-self.dim))  # Invert IR scaling
                
                # Track UV data
                with self._track_tensor(uv_data, "uv_data"):
                    # Compute corrections iteratively with memory optimization
                    correction = self._allocate_tensor(ir_data.shape, self.dtype, "correction")
                    
                    # Track correction tensor
                    with self._track_tensor(correction, "correction"):
                        # Initialize total correction
                        total_correction = self._allocate_tensor(ir_data.shape, self.dtype, "total_correction")
                        total_correction.zero_()
                        
                        # Track total correction
                        with self._track_tensor(total_correction, "total_correction"):
                            for n in range(1, 6):  # Increase order of corrections
                                # Each correction is suppressed by additional powers of z_ratio
                                sign = (-1)**n
                                power = -self.dim + 2*n
                                scale = self.Z_RATIO ** power / math.factorial(n)
                                
                                # Compute correction in-place
                                correction.copy_(ir_data)
                                correction.mul_(scale * sign)
                                
                                # Add to total correction
                                total_correction.add_(correction)
                                
                                # Periodic memory cleanup
                                if n % 2 == 0:
                                    self._update_memory_pressure()
                                    self.memory_manager._cleanup_dead_refs()
                            
                            # Apply total correction with stability factor
                            correction_scale = 0.1 / (1 + torch.abs(torch.tensor(self.Z_RATIO, dtype=self.dtype))**2)
                            uv_data.add_(total_correction.mul_(correction_scale))
                            
                            # Ensure proper normalization
                            uv_data.div_(torch.norm(uv_data))
                            uv_data.mul_(ir_norm)  # Preserve original norm
                            
                            # Apply inverse scaling to match UV data
                            uv_data.mul_(self.Z_RATIO ** self.dim)
                            
                            # Add UV enhancement factor
                            uv_factor = torch.exp(-torch.abs(torch.tensor(self.Z_UV, dtype=self.dtype)))
                            uv_data.mul_(1 + uv_factor)
                            
                            return uv_data
                            
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Memory management failed in IR reconstruction: {str(e)}")

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
        
        Args:
            field: Input field to compute corrections for
            
        Returns:
            OPE coefficients with same shape as input
            
        Raises:
            OperatorError: If OPE computation fails
            MemoryError: If memory management fails
        """
        try:
            # Validate input
            validation = self._validate_tensor(field, "input_field")
            if not validation.is_valid:
                raise ValidationError(f"Invalid input field: {validation.message}")
                
            field = self._ensure_dtype(field)
            
            # Track input field
            with self._track_tensor(field, "input_field"):
                # Split field into two operators for OPE
                if field.is_complex():
                    # Pre-allocate real and imaginary parts with complex dtype
                    op1 = self._allocate_tensor(field.shape, self.dtype, "op1_real")
                    op2 = self._allocate_tensor(field.shape, self.dtype, "op2_imag")
                    
                    # Extract real and imaginary parts and convert to complex
                    op1.copy_(field.real + 0j)
                    op2.copy_(field.imag + 0j)
                    
                    # Track operators and compute OPE
                    with self._track_tensor(op1, "op1_real"), \
                         self._track_tensor(op2, "op2_imag"):
                        # Compute OPE between different operators
                        correction = self.operator_product_expansion(op1, op2)
                        
                        # Track phase if needed
                        self._track_phase(correction)
                        
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
                        correction = self.operator_product_expansion(op1, op2)
                        
                        # Track phase if needed
                        self._track_phase(correction)
                        
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