"""
Pattern-Specific Fiber Bundle Implementation.

This module extends the base fiber bundle with pattern-specific features
for analyzing feature spaces and pattern dynamics.

The implementation is organized into three main components:
1. Configuration management via BundleConfig
2. Tensor state management via TensorStateContext
3. Core fiber bundle operations
"""

from __future__ import annotations  # Enable forward references in type hints

from dataclasses import dataclass
from functools import cached_property
from types import TracebackType
from typing import (
    Optional, Dict, Any, List, Tuple, TypeVar, Type, 
    Final, Protocol, runtime_checkable
)
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if none exists
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

from ...patterns.fiber_bundle import BaseFiberBundle
from ...patterns.fiber_types import (
    FiberBundle,
    LocalChart,
    FiberChart,
    StructureGroup,
    FiberTypeManager,
)
from ...patterns.riemannian_base import (
    MetricTensor,
    RiemannianStructure,
    ChristoffelSymbols,
    CurvatureTensor
)
from ...patterns.motivic_riemannian import (
    MotivicRiemannianStructure,
    MotivicMetricTensor
)
from ...patterns.cohomology import (
    MotivicCohomology,
    ArithmeticForm,
    HeightStructure
)
from ...patterns.riemannian_flow import RiemannianFlow
from ...patterns.formation import PatternFormation
from ...patterns.dynamics import PatternDynamics
from ...patterns.evolution import PatternEvolution
from ...patterns.symplectic import SymplecticStructure
from ...patterns.riemannian import PatternRiemannianStructure
from ...patterns.operadic_structure import (
    AttentionOperad,
    OperadicOperation
)
from ...patterns.enriched_structure import PatternTransition, WaveEmergence
from ...tiling.geometric_flow import GeometricFlow

# Type variables for generic type hints
T_co = TypeVar('T_co', bound=Tensor, covariant=True)  # Covariant type for tensor operations

@dataclass(frozen=True)
class BundleConfig:
    """Configuration for pattern fiber bundle.
    
    This immutable configuration class defines all parameters needed for
    the pattern fiber bundle's operation, ensuring type safety and
    preventing accidental modification during runtime.
    
    Attributes:
        base_dim: Dimension of base manifold
        fiber_dim: Dimension of fiber (typically SO(3))
        num_primes: Number of primes for height structure
        motive_rank: Rank for motivic structure
        integration_steps: Steps for geometric flow
        dt: Time step for integration
        stability_threshold: Threshold for stability
        learning_rate: Learning rate for evolution
        momentum: Momentum for evolution
    """
    base_dim: int
    fiber_dim: int
    num_primes: int
    motive_rank: int
    integration_steps: int
    dt: float
    stability_threshold: float
    learning_rate: float
    momentum: float

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if any(v <= 0 for v in [
            self.base_dim, self.fiber_dim, self.num_primes,
            self.motive_rank, self.integration_steps, self.dt,
            self.stability_threshold, self.learning_rate
        ]):
            raise ValueError("All parameters must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

@runtime_checkable
class TensorStateManager(Protocol[T_co]):
    """Protocol for tensor state management."""
    def __enter__(self) -> T_co: ...
    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType]) -> bool: ...

class TensorStateContext:
    """Context manager for temporary tensor state changes.
    
    Provides a safe way to temporarily modify tensor device and dtype,
    ensuring proper restoration of original state even if exceptions occur.
    
    This class is used internally by PatternFiberBundle to manage tensor
    state transitions efficiently and safely.
    
    Attributes:
        bundle: Parent fiber bundle instance
        original_tensor: Original tensor before state change
        target: Optional tensor to match device and dtype
        original_device: Original tensor device
        original_dtype: Original tensor dtype
        original_requires_grad: Original tensor requires_grad
        modified_tensor: Modified tensor with temporary state
    """
    
    def __init__(self, bundle: 'PatternFiberBundle', tensor: Tensor, target: Optional[Tensor] = None):
        self.bundle = bundle
        self.original_tensor = tensor
        self.target = target
        self.original_device = tensor.device
        self.original_dtype = tensor.dtype
        self.original_requires_grad = tensor.requires_grad
        self.modified_tensor: Optional[Tensor] = None
        
    def __enter__(self) -> Tensor:
        """Transform tensor to target state and store modified version.
        
        Returns:
            Modified tensor with target device and dtype
            
        Raises:
            RuntimeError: If tensor state modification fails
        """
        try:
            # Ensure we preserve requires_grad state
            self.modified_tensor = self.bundle._ensure_tensor_format(
                self.original_tensor,
                self.target
            )
            if self.original_requires_grad:
                self.modified_tensor.requires_grad_(True)
            return self.modified_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to modify tensor state: {str(e)}")
        
    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                exc_val: Optional[BaseException], 
                exc_tb: Optional[TracebackType]) -> bool:
        """Restore original tensor state if needed.
        
        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Traceback of exception that occurred, if any
        
        Returns:
            bool: False to propagate exceptions
        """
        if self.modified_tensor is not None:
            if (self.modified_tensor.device != self.original_device or 
                self.modified_tensor.dtype != self.original_dtype or
                self.modified_tensor.requires_grad != self.original_requires_grad):
                try:
                    # Create a new tensor with original state while preserving gradients
                    restored = self.modified_tensor.to(
                        device=self.original_device,
                        dtype=self.original_dtype,
                        non_blocking=True
                    )
                    if self.original_requires_grad:
                        restored.requires_grad_(True)
                    # Copy data back to original tensor to maintain gradient chain
                    if self.original_tensor.requires_grad:
                        self.original_tensor.data.copy_(restored.data)
                    else:
                        self.original_tensor.copy_(restored)
                except RuntimeError as e:
                    warnings.warn(f"Failed to restore tensor state: {str(e)}")
        self.modified_tensor = None
        return False


class PatternFiberBundle(BaseFiberBundle):
    """Pattern-specific implementation of fiber bundle.
    
    This module implements a specialized fiber bundle for analyzing patterns in feature spaces.
    It extends the base fiber bundle with pattern-specific features including:
    1. Pattern dynamics and evolution
    2. Geometric flow analysis
    3. Stability metrics
    4. Cohomology with pattern structure
    
    The implementation follows these design principles:
    1. Performance optimization for batch operations
    2. Memory efficiency through in-place operations
    3. Clear separation of concerns between components
    4. Type safety and runtime validation
    
    Technical Details:
    - Uses PyTorch for tensor operations
    - Implements symplectic geometry for pattern analysis
    - Provides cohomology computations with pattern structure
    - Supports parallel transport with pattern evolution
    
    Constants:
        Metric Constants:
            _FIBER_PERT_SCALE: Scale for fiber metric perturbation (0.025)
            _REG_SCALE_BASE: Base regularization scale (1e-3)
            _REG_SCALE_FIBER: Fiber regularization (10x base, 1e-2)
            _SYMPLECTIC_WEIGHT: Weight for symplectic contribution (0.1)
        
        Evolution Constants:
            _EVOLUTION_TIME_STEPS: Steps for unstable point evolution (10)
            _DEFAULT_NUM_LAYERS: Default layers in geometric flow (2)
            _DEFAULT_REACTION_COEFF: Default reaction coefficient (1.0)
            _DEFAULT_DIFFUSION_COEFF: Default diffusion coefficient (0.1)
    """
    
    #--------------------------------------------------------------------------
    # Constants
    #--------------------------------------------------------------------------
    
    # Metric Constants
    _FIBER_PERT_SCALE: Final[float] = 0.025  # Scale for fiber metric perturbation
    _REG_SCALE_BASE: Final[float] = 1e-3     # Base regularization scale
    _REG_SCALE_FIBER: Final[float] = 1e-2    # Fiber regularization (10x base)
    _SYMPLECTIC_WEIGHT: Final[float] = 0.1    # Weight for symplectic contribution
    
    # Evolution Constants
    _EVOLUTION_TIME_STEPS: Final[int] = 10    # Number of steps for unstable point evolution
    _DEFAULT_NUM_LAYERS: Final[int] = 2       # Default number of layers in geometric flow
    _DEFAULT_REACTION_COEFF: Final[float] = 1.0  # Default reaction coefficient
    _DEFAULT_DIFFUSION_COEFF: Final[float] = 0.1  # Default diffusion coefficient
    
    #--------------------------------------------------------------------------
    # Initialization and Setup
    #--------------------------------------------------------------------------

    def __init__(
        self,
        base_dim: int = 2,
        fiber_dim: int = 3,  # SO(3) fiber dimension
        structure_group: str = "SO3",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_primes: int = 8,
        motive_rank: int = 4,
        integration_steps: int = 10,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ):
        """Initialize pattern fiber bundle."""
        # Initialize base bundle and device
        super().__init__(base_dim, fiber_dim, structure_group, device=device, dtype=dtype)
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        self._structure_group_str = structure_group
        
        # Initialize fiber type manager
        self.fiber_type_manager = FiberTypeManager()
        self._fiber_type = "Vector"  # Default to vector bundle
        
        # Store configuration in type-safe dataclass
        self._config = BundleConfig(
            base_dim=base_dim,
            fiber_dim=fiber_dim,
            num_primes=num_primes,
            motive_rank=motive_rank,
            integration_steps=integration_steps,
            dt=dt,
            stability_threshold=stability_threshold,
            learning_rate=learning_rate,
            momentum=momentum,
        )
        
        # Initialize all components
        self._initialize_components()
        
        # Move to device
        self.to(self.device)

    def _initialize_components(self) -> None:
        """Initialize bundle components efficiently."""
        # Initialize geometric components
        self.riemannian_framework = PatternRiemannianStructure(
            manifold_dim=self.total_dim,
            pattern_dim=self.fiber_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.geometric_flow = RiemannianFlow(
            manifold_dim=self._config.base_dim,
            hidden_dim=self._config.fiber_dim,
            num_layers=self._DEFAULT_NUM_LAYERS,
            dt=self._config.dt,
            stability_threshold=self._config.stability_threshold,
            use_parallel_transport=True
        )
        
        # Initialize pattern components
        self.pattern_formation = PatternFormation(
            dim=self._config.fiber_dim,
            dt=self._config.dt,
            diffusion_coeff=self._DEFAULT_DIFFUSION_COEFF,
            reaction_coeff=self._DEFAULT_REACTION_COEFF
        )
        self.pattern_dynamics = PatternDynamics(
            dt=self._config.dt,
            device=self.device
        )
        self.pattern_evolution = PatternEvolution(
            framework=self.riemannian_framework,
            learning_rate=self._config.learning_rate,
            momentum=self._config.momentum
        )
        
        # Initialize arithmetic components
        self.height_structure = HeightStructure(
            num_primes=self._config.num_primes
        )
        
        # Initialize connection parameter with requires_grad=True
        self.connection = nn.Parameter(
            torch.zeros(
                self.base_dim,
                self.fiber_dim,
                self.fiber_dim,
                device=self.device,
                dtype=self.dtype
            ),
            requires_grad=True
        )
        
        # Initialize metric parameter with requires_grad=True
        self.metric = nn.Parameter(
            torch.eye(
                self.total_dim,
                device=self.device,
                dtype=self.dtype
            ),
            requires_grad=True
        )
        
        # Initialize symplectic structure
        self.symplectic = SymplecticStructure(
            dim=self.fiber_dim,
            preserve_structure=True,
            wave_enabled=True
        )
        
        # Initialize operadic structure
        self.operadic = AttentionOperad(
            base_dim=self._config.base_dim,
            preserve_symplectic=True,
            preserve_metric=True,
            dtype=self.dtype
        )
        
        # Initialize wave components
        self.wave = WaveEmergence(
            dt=self._config.dt,
            num_steps=self._config.integration_steps
        )
        
        # Initialize transition components
        self.transition = PatternTransition(
            wave_emergence=self.wave
        )

        # Register components as submodules to ensure proper parameter management
        self.add_module('riemannian_framework', self.riemannian_framework)
        self.add_module('geometric_flow', self.geometric_flow)
        self.add_module('pattern_formation', self.pattern_formation)
        self.add_module('pattern_dynamics', self.pattern_dynamics)
        self.add_module('pattern_evolution', self.pattern_evolution)
        self.add_module('height_structure', self.height_structure)
        self.add_module('symplectic', self.symplectic)
        self.add_module('operadic', self.operadic)
        self.add_module('wave', self.wave)
        self.add_module('transition', self.transition)

        # Initialize metric factors with proper gradient tracking
        metric_factors = torch.randn(self.total_dim, self.total_dim, device=self.device, dtype=self.dtype) * 0.01
        self.register_parameter('metric_factors', nn.Parameter(metric_factors, requires_grad=True))
        
        # Set the metric parameter in the Riemannian framework and ensure gradient tracking
        self.riemannian_framework.set_metric_param(nn.Parameter(self.metric.data.clone(), requires_grad=True))
        self.riemannian_framework.metric_factors = self.metric_factors

    def _initialize_basis_matrices(self) -> None:
        """Initialize Lie algebra basis matrices for SO(3)."""
        self.basis_matrices = torch.zeros(
            self.fiber_dim * (self.fiber_dim - 1) // 2,  # Number of SO(3) generators
            self.fiber_dim,
            self.fiber_dim,
            device=self.device
        )
        
        for idx, (i, j) in enumerate(
            (i, j) 
            for i in range(self.fiber_dim) 
            for j in range(i + 1, self.fiber_dim)
        ):
            basis = torch.zeros(self.fiber_dim, self.fiber_dim, device=self.device)
            basis[i, j] = 1.0
            basis[j, i] = -1.0
            self.basis_matrices[idx] = basis

    #--------------------------------------------------------------------------
    # Cached Properties
    #--------------------------------------------------------------------------

    @cached_property
    def _triu_indices(self) -> Tuple[Tensor, Tensor]:
        """Cached upper triangular indices for fiber perturbation."""
        i_indices, j_indices = torch.triu_indices(self.fiber_dim, self.fiber_dim)
        return i_indices.to(self.device), j_indices.to(self.device)

    @cached_property
    def _eye_matrix(self) -> Tensor:
        """Cached identity matrix for positive definiteness."""
        return torch.eye(
            self.total_dim,
            device=self.device,
            dtype=torch.float32
        )

    @cached_property
    def _reg_scale(self) -> Tensor:
        """Cached regularization scale for positive definiteness."""
        reg_scale = torch.ones(
            self.total_dim, 
            self.total_dim,
            device=self.device,
            dtype=torch.float32
        ) * 1e-3
        reg_scale[self.base_dim:, self.base_dim:] *= 10.0
        return reg_scale

    #--------------------------------------------------------------------------
    # Tensor State Management
    #--------------------------------------------------------------------------

    def _ensure_tensor_format(self, tensor: Tensor, target_tensor: Optional[Tensor] = None) -> Tensor:
        """Ensure tensor has correct device and numeric type.
        
        This method efficiently handles tensor format conversion with minimal
        memory allocation and device transfers.
        
        Args:
            tensor: Input tensor to format
            target_tensor: Optional tensor to match device and dtype
            
        Returns:
            Properly formatted tensor with optimal memory layout
        """
        if not isinstance(tensor, Tensor):
            return torch.tensor(tensor, device=self.device)
        
        needs_device_change = (
            target_tensor is not None and tensor.device != target_tensor.device or
            target_tensor is None and tensor.device != self.device
        )
        needs_dtype_change = (
            target_tensor is not None and tensor.dtype != target_tensor.dtype
        )
        needs_contiguous = not tensor.is_contiguous()
        
        # Optimize device and dtype changes by combining them when possible
        if needs_device_change or needs_dtype_change:
            target_device = target_tensor.device if target_tensor is not None else self.device
            target_dtype = target_tensor.dtype if target_tensor is not None else tensor.dtype
            tensor = tensor.to(device=target_device, dtype=target_dtype, non_blocking=True)
        
        # Make contiguous only if needed
        if needs_contiguous:
            tensor = tensor.contiguous()
            
        return tensor

    def _ensure_broadcasting(self, tensor: Tensor, target_tensor: Tensor) -> Tensor:
        """Ensure proper broadcasting dimensions for tensor operations.
        
        This method efficiently handles tensor broadcasting with minimal
        memory allocation and optimal view/expand operations.
        
        Args:
            tensor: Tensor to broadcast
            target_tensor: Target tensor to match dimensions
            
        Returns:
            Properly broadcasted tensor with optimal memory layout
            
        Note:
            Uses view + expand instead of repeat for memory efficiency
        """
        tensor = self._ensure_tensor_format(tensor, target_tensor)
        
        if tensor.shape == target_tensor.shape:
            return tensor
            
        try:
            # Try direct expansion first (most memory efficient)
            return tensor.expand_as(target_tensor)
        except RuntimeError:
            # If direct expansion fails, try adding missing dimensions
            missing_dims = len(target_tensor.shape) - len(tensor.shape)
            if missing_dims > 0:
                # Use view instead of unsqueeze for better performance
                new_shape = (1,) * missing_dims + tensor.shape
                tensor = tensor.view(*new_shape)
                return tensor.expand_as(target_tensor)
            raise  # Re-raise if we can't handle the broadcasting

    def with_tensor_state(self, tensor: Tensor, target: Optional[Tensor] = None) -> TensorStateContext:
        """Create a context for temporary tensor state changes.
        
        This is the recommended way to handle tensor state changes in a safe
        and efficient manner.
        
        Args:
            tensor: Tensor to modify
            target: Optional tensor to match device and dtype
            
        Returns:
            Context manager for tensor state changes
            
        Example:
            >>> with self.with_tensor_state(tensor, target) as formatted_tensor:
            ...     # Do operations with formatted_tensor
            ...     # Original state is restored after the block
        """
        return TensorStateContext(self, tensor, target)

    def to(self, device: torch.device) -> 'PatternFiberBundle':
        """Move the bundle to the specified device."""
        self.device = device
        return super().to(device)
        
    #--------------------------------------------------------------------------
    # Core Bundle Operations
    #--------------------------------------------------------------------------

    def _handle_dimension_transition(self, tensor: Tensor) -> Tensor:
        """Handle dimension transitions for tensors using operadic structure.
        
        Args:
            tensor: Input tensor to handle dimension transition for
            
        Returns:
            Tensor with correct dimensions through operadic transitions
        """
        if tensor.dim() == 2:  # For 2D tensors (matrices)
            if tensor.shape[0] != self.base_dim or tensor.shape[1] != self.fiber_dim:
                # Create operadic operations for both dimensions
                base_operation = self.operadic.create_operation(
                    source_dim=tensor.shape[0],
                    target_dim=self.base_dim
                )
                fiber_operation = self.operadic.create_operation(
                    source_dim=tensor.shape[1],
                    target_dim=self.fiber_dim
                )
                
                # Apply operadic composition laws
                result = torch.einsum('ij,jk,kl->il',
                                    base_operation.composition_law,
                                    tensor,
                                    fiber_operation.composition_law.transpose(-2, -1))
                
                return result
        elif tensor.dim() == 1:  # For 1D tensors (vectors)
            if tensor.shape[-1] != self.fiber_dim:
                # Create operadic operation for fiber dimension
                operation = self.operadic.create_operation(
                    source_dim=tensor.shape[-1],
                    target_dim=self.fiber_dim
                )
                
                # Apply operadic composition law
                result = torch.matmul(
                    tensor.unsqueeze(0),
                    operation.composition_law.transpose(-2, -1)
                ).squeeze(0)
                
                return result
        return tensor

    def _handle_dimension(self, tensor: Tensor, target_dim: Optional[int] = None) -> Tensor:
        """Handle dimension transitions using operadic structure.
        
        Args:
            tensor: Input tensor to transform
            target_dim: Optional target dimension (defaults to fiber_dim)
            
        Returns:
            Transformed tensor with correct dimensions through operadic transitions
            
        Raises:
            ValueError: If tensor dimension is invalid or operation fails
            RuntimeError: If transformation fails
        """
        if target_dim is None:
            target_dim = self.fiber_dim
            
        if tensor.shape[-1] == target_dim:
            return tensor
            
        try:
            # Save original shape for proper reshaping
            original_shape = tensor.shape[:-1]
            
            # Create operadic operation for dimension transition
            operation = self.operadic.create_operation(
                source_dim=tensor.shape[-1],
                target_dim=target_dim
            )
            
            # Reshape tensor to match operation dimensions
            reshaped = tensor.reshape(-1, tensor.shape[-1])
            
            # Apply operadic composition law
            result = torch.matmul(reshaped, operation.composition_law.t())
            
            # Restore original shape
            result = result.reshape(*original_shape, target_dim)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to transform tensor dimensions: {str(e)}") from e

    def connection_form(self, tangent_vector: Tensor) -> Tensor:
        """Compute connection form using operadic structure.
        
        Args:
            tangent_vector: Tangent vector to compute connection for
            
        Returns:
            Connection form tensor representing the local connection on the bundle
            
        Raises:
            RuntimeError: If computation fails
        """
        logger.debug(f"Computing connection form for tangent_vector shape: {tangent_vector.shape}")
        
        try:
            tangent_vector = self._ensure_tensor_format(tangent_vector)
            
            # Split into base and fiber components efficiently
            base_components = tangent_vector[..., :self.base_dim]
            fiber_components = tangent_vector[..., self.base_dim:]
            
            # Handle batch dimension
            batch_size = tangent_vector.shape[0] if tangent_vector.dim() > 1 else 1
            if tangent_vector.dim() == 1:
                base_components = base_components.unsqueeze(0)
                fiber_components = fiber_components.unsqueeze(0)
            
            # For purely vertical vectors, return identity transformation
            if torch.allclose(base_components, torch.zeros_like(base_components)):
                result = torch.zeros(
                    batch_size,
                    self.fiber_dim,
                    self.fiber_dim,
                    device=tangent_vector.device,
                    dtype=tangent_vector.dtype
                )
                # Use scatter to preserve gradients
                diagonal_indices = torch.arange(self.fiber_dim, device=tangent_vector.device)
                result.scatter_(-1, diagonal_indices.view(1, -1, 1).expand(batch_size, -1, 1), 
                              fiber_components.unsqueeze(-1))
                return result if tangent_vector.dim() > 1 else result.squeeze(0)
            
            # Compute connection form with proper broadcasting
            # Reshape base components for proper broadcasting
            # [batch_size, base_dim] -> [batch_size, base_dim, 1]
            base_components = base_components.unsqueeze(-1)
            
            # Unsqueeze connection for proper broadcasting
            # connection shape: [base_dim, fiber_dim, fiber_dim] -> [1, base_dim, fiber_dim, fiber_dim]
            connection = self.connection.unsqueeze(0)
            
            # Expand to match batch size
            # connection shape: [1, base_dim, fiber_dim, fiber_dim] -> [batch_size, base_dim, fiber_dim, fiber_dim]
            connection = connection.expand(batch_size, -1, -1, -1)
            
            # Contract base components with connection using batched matrix multiplication
            # base_components: [batch_size, base_dim, 1]
            # connection: [batch_size, base_dim, fiber_dim, fiber_dim]
            # result: [batch_size, fiber_dim, fiber_dim]
            result = torch.sum(base_components.unsqueeze(-1) * connection, dim=1)
            
            # Add skew-symmetry while preserving gradients
            result = 0.5 * (result - result.transpose(-2, -1))
            
            # Add gradient hook to ensure proper gradient flow
            if self.connection.requires_grad:
                def connection_grad_hook(grad):
                    # Scale gradient by base components, taking mean over batch dimension
                    base_scale = base_components.mean(dim=0, keepdim=True)  # [1, base_dim, 1]
                    return grad * base_scale.squeeze(0)  # [base_dim, fiber_dim, fiber_dim]
                
                result.register_hook(connection_grad_hook)
            
            if isinstance(result, Tensor):
                logger.debug(f"Connection form result requires_grad: {result.requires_grad}")
                if result.grad_fn is not None:
                    logger.debug(f"Connection form result grad_fn: {result.grad_fn}")
            
            return result if tangent_vector.dim() > 1 else result.squeeze(0)
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute connection form: {str(e)}") from e

    def local_trivialization(self, point: Tensor) -> Tuple[LocalChart[Tensor], FiberChart[Tensor, str]]:
        """Compute local trivialization using enriched structure.
        
        Args:
            point: Point in total space
            
        Returns:
            Tuple of (local_chart, fiber_chart)
        """
        # Get base coordinates through projection
        base_coords = self.bundle_projection(point)
        
        # Get fiber coordinates
        fiber_coords = point[..., self.base_dim:self.base_dim + self.fiber_dim]
        
        # Use operadic structure for symplectic form computation
        symplectic_form = self.symplectic.compute_form(fiber_coords)
        
        # Create transition maps dictionary with geometric flow
        transition_maps = {
            'geometric_flow': self.geometric_flow,
            'symplectic_form': symplectic_form,
            'pattern_dynamics': self.pattern_dynamics
        }
        
        # Create local chart with enhanced structure
        local_chart = LocalChart(
            coordinates=base_coords,
            dimension=self.base_dim,
            transition_maps=transition_maps
        )
        
        # Create fiber chart with pattern-specific features
        fiber_chart = FiberChart(
            fiber_coordinates=fiber_coords,
            structure_group=self._structure_group_str,
            transition_functions={
                'evolution': self.pattern_evolution,
                'dynamics': self.pattern_dynamics,
                'symplectic': symplectic_form
            }
        )
        
        return local_chart, fiber_chart
        
    #--------------------------------------------------------------------------
    # Metric and Geometry Operations
    #--------------------------------------------------------------------------

    def _compute_metric_blocks(
        self,
        base_points: Tensor,
        fiber_points: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        base_dim: int,
        fiber_dim: int,
        metric: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute metric blocks efficiently."""
        # Pre-allocate all tensors at once for better memory locality
        base_metric = torch.empty(
            (batch_size, base_dim, base_dim),
            device=device, dtype=dtype
        )
        fiber_metric = torch.empty(
            (batch_size, fiber_dim, fiber_dim),
            device=device, dtype=dtype
        )
        cross_terms = torch.empty(
            (batch_size, base_dim, fiber_dim),
            device=device, dtype=dtype
        )
        
        # Compute metrics in parallel with pre-allocated outputs
        base_future = torch.jit.fork(
            lambda: base_metric.copy_(
                self.geometric_flow.compute_metric(base_points)
            )
        )
        fiber_future = torch.jit.fork(
            lambda: fiber_metric.copy_(
                self._compute_fiber_perturbation(fiber_points, fiber_dim)
            )
        )
        cross_future = torch.jit.fork(
            lambda: cross_terms.copy_(
                metric[:base_dim, base_dim:]
                if metric.shape[0] > base_dim
                else torch.zeros_like(cross_terms)
            )
        )
        
        # Wait for all computations to complete
        torch.jit.wait(base_future)
        torch.jit.wait(fiber_future)
        torch.jit.wait(cross_future)
        
        return base_metric, fiber_metric, cross_terms

    def _ensure_positive_definite(
        self,
        values: Tensor,
        batch_size: int,
        total_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        reg_scale_base: float = 1e-3,
        reg_scale_fiber: float = 1e-2
    ) -> Tensor:
        """Ensure metric tensor is positive definite efficiently."""
        # Clone tensor to avoid in-place operation issues
        values = values.clone()
        
        # Symmetrize
        values = 0.5 * (values + values.transpose(-2, -1))
        
        # Create regularization term efficiently
        eye = torch.eye(total_dim, device=device, dtype=dtype).expand(batch_size, -1, -1)
        reg = torch.ones(total_dim, total_dim, device=device, dtype=dtype) * reg_scale_base
        reg[self.base_dim:, self.base_dim:] *= (reg_scale_fiber / reg_scale_base)  # Fiber regularization
        reg = reg.expand(batch_size, -1, -1)
        
        # Add regularization
        values = values + eye * reg
        
        return values

    def compute_metric(self, points: torch.Tensor) -> MotivicMetricTensor:
        """Compute metric tensor with pattern-specific features using operadic structure."""
        logger.debug(f"Computing metric at points shape: {points.shape}")
        if isinstance(self.metric, nn.Parameter):
            logger.debug(f"Metric parameter requires_grad: {self.metric.requires_grad}")
            if self.metric.grad_fn is not None:
                logger.debug(f"Metric parameter grad_fn: {self.metric.grad_fn}")
        
        points = self._ensure_tensor_format(points)
        batch_size = points.shape[0]
        
        # Split points
        base_points = points[..., :self.base_dim]
        fiber_points = points[..., self.base_dim:self.base_dim + self.fiber_dim]
        
        # Initialize values with metric parameter - use clone to maintain gradient chain
        values = self.metric.clone().unsqueeze(0).expand(batch_size, -1, -1).clone()  # Clone after expand
        
        # Compute base metric from geometric flow
        base_metric = self.geometric_flow.compute_metric(base_points)
        base_part = values[:, :self.base_dim, :self.base_dim].clone()  # Clone before modifying
        base_part = base_part + base_metric
        values[:, :self.base_dim, :self.base_dim] = base_part
        
        # Compute fiber metric with perturbation
        fiber_metric = self._compute_fiber_perturbation(fiber_points, self.fiber_dim)
        fiber_part = values[:, self.base_dim:, self.base_dim:].clone()  # Clone before modifying
        fiber_part = fiber_part + fiber_metric
        values[:, self.base_dim:, self.base_dim:] = fiber_part
        
        # Create operadic operation for cross terms
        operation = self.operadic.create_operation(
            source_dim=self.base_dim,
            target_dim=self.fiber_dim
        )
        
        # Compute cross terms using base metric and composition law
        cross_terms = torch.matmul(
            base_metric,
            operation.composition_law
        )
        cross_terms = cross_terms.transpose(-2, -1)
        
        # Add cross terms to values tensor
        cross_part_upper = values[:, :self.base_dim, self.base_dim:].clone()  # Clone before modifying
        cross_part_lower = values[:, self.base_dim:, :self.base_dim].clone()  # Clone before modifying
        cross_part_upper = cross_part_upper + cross_terms
        cross_part_lower = cross_part_lower + cross_terms.transpose(-2, -1)
        values[:, :self.base_dim, self.base_dim:] = cross_part_upper
        values[:, self.base_dim:, :self.base_dim] = cross_part_lower
        
        # Make symmetric explicitly while preserving gradients
        values = 0.5 * (values + values.transpose(-2, -1))
        
        # Add regularization for numerical stability
        reg = torch.eye(
            self.total_dim,
            device=points.device,
            dtype=points.dtype
        ).unsqueeze(0) * self._REG_SCALE_BASE
        
        # Increase regularization for fiber part
        reg[:, self.base_dim:, self.base_dim:] *= (self._REG_SCALE_FIBER / self._REG_SCALE_BASE)
        values = values + reg
        
        # Add hook to track metric gradients
        if values.requires_grad:
            def metric_grad_hook(grad):
                logger.debug(f"\n=== Metric Gradient Hook ===")
                logger.debug(f"Gradient shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            values.register_hook(metric_grad_hook)
        
        result = MotivicMetricTensor(
            values=values,
            dimension=self.total_dim,
            height_structure=self.height_structure,
            is_compatible=True
        )

        if isinstance(result.values, Tensor):
            logger.debug(f"Metric result requires_grad: {result.values.requires_grad}")
            if result.values.grad_fn is not None:
                logger.debug(f"Metric result grad_fn: {result.values.grad_fn}")
        
        return result

    def _compute_fiber_perturbation(
        self,
        fiber_points: Tensor,
        fiber_dim: int,
        pert_scale: float = 0.025
    ) -> Tensor:
        """Compute fiber metric perturbation efficiently."""
        batch_size = fiber_points.size(0)
        
        # Pre-allocate output tensor with optimal memory layout
        perturbation = torch.zeros(
            batch_size, 
            fiber_dim, 
            fiber_dim,
            device=fiber_points.device,
            dtype=fiber_points.dtype
        ).contiguous()
        
        # Compute outer product efficiently using batched matmul
        # Reshape for broadcasting: [batch, dim, 1] x [batch, 1, dim]
        fiber_points_2d = fiber_points.view(batch_size, -1, 1)
        perturbation = torch.baddbmm(
            perturbation,
            fiber_points_2d,
            fiber_points_2d.transpose(-2, -1),
            alpha=1.0,
            beta=0.0
        )
        
        # Add perturbation scale to diagonal in-place
        perturbation.diagonal(dim1=1, dim2=2).add_(pert_scale)
        
        return perturbation

    def _project_metric_compatible(self, matrix: Tensor, metric: Tensor) -> Tensor:
        """Project matrix to metric-compatible subspace.
        
        Performs orthogonal projection of a matrix onto the space of metric-compatible
        transformations. A transformation A is metric-compatible if:
            g(Ax, y) + g(x, Ay) = 0
        where g is the metric tensor.
        
        Args:
            matrix: Matrix to project
            metric: Metric tensor to be compatible with
            
        Returns:
            Projected matrix that is metric-compatible
            
        Note:
            The projection is computed using the formula:
            P(A) = 1/2(A - g^(-1)A^T g)
            where g is the metric tensor and A is the input matrix.
        """
        g_inv = torch.inverse(metric)
        skew = 0.5 * (matrix - matrix.transpose(-2, -1))
        metric_compat = -torch.matmul(
            torch.matmul(metric, skew.transpose(-2, -1)),
            g_inv
        )
        return 0.5 * (skew + metric_compat)

    #--------------------------------------------------------------------------
    # Evolution and Transport
    #--------------------------------------------------------------------------

    def _evolve_batch_efficient(
        self,
        base_transport: Tensor,
        transport_gradients: Tensor,
        stability_threshold: float,
        time_steps: int,
        pattern_formation: PatternFormation,
        reaction_coeff: float = 1.0,
        diffusion_coeff: float = 0.1
    ) -> Tensor:
        """Efficiently evolve a batch of transport points."""
        batch_size = base_transport.size(0)
        
        # Create reaction and diffusion coefficients
        reaction = torch.full(
            (batch_size,),
            reaction_coeff,
            device=base_transport.device,
            dtype=base_transport.dtype
        )
        diffusion = torch.full(
            (batch_size,),
            1.0 + diffusion_coeff,
            device=base_transport.device,
            dtype=base_transport.dtype
        )
        
        # Initialize evolved tensor with gradient tracking
        evolved = torch.zeros_like(base_transport, requires_grad=True)  # Replace clone with zeros_like

        # First point is unchanged
        evolved[0] = base_transport[0]  # Direct assignment maintains gradient chain

        # Evolve remaining points with gradient tracking
        evolved_rest = base_transport[1:] + transport_gradients * reaction.unsqueeze(-1)
        evolved_rest = evolved_rest * diffusion.unsqueeze(-1)
        evolved[1:] = evolved_rest  # Direct assignment maintains gradient chain
        
        # Compute stability with gradient tracking
        stability = pattern_formation.compute_stability(evolved)
        unstable_mask = stability["stability_margin"] < stability_threshold
        
        if torch.any(unstable_mask):
            # Create unstable points tensor with gradient tracking
            unstable_points = evolved[unstable_mask].unsqueeze(1)
            
            # Evolve unstable points with gradient tracking
            evolved_unstable = pattern_formation.evolve(
                unstable_points,
                time_steps
            )
            
            # Update unstable points while maintaining gradients
            evolved_unstable_flat = evolved_unstable[:, -1].reshape(-1, evolved.size(-1))
            mask_expanded = unstable_mask.unsqueeze(-1).expand_as(evolved)
            
            # Use direct tensor operations to maintain gradient flow
            evolved = torch.where(mask_expanded, evolved_unstable_flat, evolved)
            
            # Add gradient hook for debugging
            if evolved.requires_grad:
                def hook(grad):
                    logger.debug("\n=== Evolution Gradient Hook ===")
                    logger.debug(f"Shape: {grad.shape}")
                    logger.debug(f"Norm: {torch.norm(grad)}")
                    logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                    logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                    logger.debug(f"Mean: {grad.mean().item()}")
                    logger.debug(f"Std: {grad.std().item()}")
                    return grad
                evolved.register_hook(hook)
        
        return evolved

    def parallel_transport(self, section: Tensor, path: Tensor) -> Tensor:
        """Parallel transport a section along a path with pattern evolution and stability control."""
        # Only log initial input states
        logger.debug(f"=== Parallel Transport Start ===")
        logger.debug(f"Section: shape={section.shape}, requires_grad={section.requires_grad}")
        logger.debug(f"Path: shape={path.shape}")
        
        # Ensure section is a tensor and on the correct device
        result: Tensor = self._ensure_tensor_format(section)
        
        # Ensure we have a properly formatted error fallback tensor
        error_section = self._ensure_tensor_format(section)
        if len(error_section.shape) == 1:
            error_section = error_section.unsqueeze(0)
        error_result = error_section.unsqueeze(0).expand(len(path), *error_section.shape)
        
        with self.with_tensor_state(result) as formatted_section, \
             self.with_tensor_state(path) as formatted_path:
            try:
                # Handle batch dimension if present
                original_batch_shape = None
                if len(formatted_section.shape) == 3:
                    original_batch_shape = formatted_section.shape[:-1]
                    formatted_section = formatted_section.reshape(-1, formatted_section.shape[-1])
                elif len(formatted_section.shape) == 1:
                    formatted_section = formatted_section.unsqueeze(0)

                # Extract metrics without clone to maintain gradient chain
                fiber_metric = self.metric[self.base_dim:, self.base_dim:].clone()  # Clone to maintain gradient chain
                base_metric = self.metric[:self.base_dim, :self.base_dim].clone()  # Clone to maintain gradient chain
                
                # Store original metric norm with gradient tracking
                metric_inner = torch.einsum('bi,ij,bj->b', formatted_section, fiber_metric, formatted_section)
                original_norm = torch.sqrt(metric_inner + 1e-7)
                
                # Initialize transport tensor - avoid in-place ops on leaves
                base_transport = torch.zeros(
                    len(formatted_path), formatted_section.shape[0], formatted_section.shape[1],
                    device=formatted_section.device, dtype=formatted_section.dtype,
                    requires_grad=True
                )
                
                # Set initial condition without in-place operation
                base_transport = torch.cat([
                    formatted_section.unsqueeze(0),
                    base_transport[1:]
                ])

                # Compute path tangents
                path_tangent = formatted_path[1:] - formatted_path[:-1]
                
                # Normalize path tangents
                metric_path = torch.einsum('bi,ij->bj', path_tangent, base_metric)
                path_inner = torch.einsum('bi,bi->b', path_tangent, metric_path)
                path_metric_norms = torch.sqrt(path_inner + 1e-7)
                path_tangent = path_tangent / path_metric_norms.unsqueeze(-1)

                # RK4 integration
                dt = 1.0 / (len(formatted_path) - 1)
                transported = [formatted_section]  # Store intermediate results in list
                
                for current_point in range(len(formatted_path) - 1):
                    current = transported[-1]  # Get last transported point
                    tangent = path_tangent[current_point]

                    # RK4 steps
                    k1 = self._transport_step(current, tangent, fiber_metric)
                    k2 = self._transport_step(current + 0.5*dt*k1, tangent, fiber_metric)
                    k3 = self._transport_step(current + 0.5*dt*k2, tangent, fiber_metric)
                    k4 = self._transport_step(current + dt*k3, tangent, fiber_metric)

                    # Compute next point
                    rk4_step = (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                    next_point = current + rk4_step

                    # Normalize
                    metric_next = torch.einsum('bi,ij,bj->b', next_point, fiber_metric, next_point)
                    current_norm = torch.sqrt(metric_next + 1e-7)
                    scale_factor = original_norm / (current_norm + 1e-7)
                    next_point = next_point * scale_factor.unsqueeze(-1)
                    
                    transported.append(next_point)

                # Stack results
                result = torch.stack(transported)

                # Restore original batch shape if needed
                if original_batch_shape is not None:
                    result = result.reshape(len(formatted_path), *original_batch_shape[1:], self.fiber_dim)
                else:
                    result = result.squeeze(1)

                logger.debug("=== Parallel Transport Complete ===")
                return result

            except Exception as e:
                logger.error(f"Error in parallel transport: {str(e)}")
                return error_result
        
        # This should never be reached, but ensures all code paths return a Tensor
        return error_result

    def _transport_step(self, section: Tensor, tangent: Tensor, fiber_metric: Tensor) -> Tensor:
        """Compute transport step."""
        # Compute connection form with gradient tracking
        tangent_base = tangent[:self.base_dim]
        connection = self.connection.clone()  # Clone to maintain gradient chain
        connection_form = torch.einsum('i,ijk->jk', tangent_base, connection)
        
        # Apply to section with gradient tracking
        return torch.einsum('jk,bj->bk', connection_form, section)

    #--------------------------------------------------------------------------
    # Stability and Analysis
    #--------------------------------------------------------------------------

    def _compute_stability_metrics(
        self,
        local_chart: LocalChart,
        fiber_chart: FiberChart,
        point: Tensor,
        base_dim: int,
        num_primes: int
    ) -> Dict[str, Tensor]:
        """Compute stability metrics efficiently."""
        # Pre-allocate output tensors for better memory locality
        batch_size = point.size(0)
        geometric_metric = torch.empty(
            (batch_size, base_dim, base_dim),
            device=point.device,
            dtype=point.dtype
        )
        pattern_stability = torch.empty(
            (batch_size,),
            device=point.device,
            dtype=point.dtype
        )
        height_data = torch.empty(
            (batch_size, num_primes),
            device=point.device,
            dtype=point.dtype
        )
        
        # Launch parallel computations with pre-allocated outputs
        geometric_future = torch.jit.fork(
            lambda: geometric_metric.copy_(
                self.geometric_flow.compute_metric(local_chart.coordinates)
            )
        )
        pattern_future = torch.jit.fork(
            lambda: pattern_stability.copy_(
                self.pattern_formation.compute_stability(fiber_chart.fiber_coordinates)["stability_margin"]
            )
        )
        height_future = torch.jit.fork(
            lambda: height_data.copy_(
                self.height_structure.compute_height(point)
            )
        )
        
        # Wait for all computations to complete
        torch.jit.wait(geometric_future)
        torch.jit.wait(pattern_future)
        torch.jit.wait(height_future)
        
        # Return results in dictionary
        return {
            "geometric_stability": geometric_metric,
            "pattern_stability": pattern_stability,
            "height_stability": height_data
        }

    def compute_stability(self, point: Tensor) -> Dict[str, Any]:
        """Compute pattern stability metrics efficiently."""
        point = self._ensure_tensor_format(point)
        
        # Get local coordinates through trivialization
        with torch.no_grad():  # Disable gradients for efficiency
            local_chart, fiber_chart = self.local_trivialization(point)
            return self._compute_stability_metrics(
                local_chart, fiber_chart, point,
                self.base_dim, self.height_structure.num_primes
            )

    def compute_holonomy_group(self, holonomies: List[Tensor]) -> Tensor:
        """Compute the holonomy group efficiently."""
        with torch.no_grad():
            # Pre-allocate output tensor
            holonomy_group = self._ensure_tensor_format(
                super().compute_holonomy_group(holonomies)
            )
            
            # Compute stability and structure in parallel
            stability_future = torch.jit.fork(
                self.pattern_formation.compute_stability,
                holonomy_group
            )
            
            # Get stability result and broadcast efficiently
            stability = torch.jit.wait(stability_future)
            pattern_structure = self._ensure_broadcasting(
                stability["pattern_stability"],
                holonomy_group
            )
            
            # Use in-place multiplication for efficiency
            holonomy_group.mul_(pattern_structure)
            return holonomy_group

    def compute_holonomy_algebra(self, holonomies: List[Tensor]) -> Tensor:
        """Compute the holonomy Lie algebra with pattern features."""
        with torch.no_grad():  # Disable gradients for efficiency
            base_algebra = self._ensure_tensor_format(
                super().compute_holonomy_algebra(holonomies)
            )
            
            # Compute symplectic structure efficiently
            symplectic_form = self.symplectic.compute_form(base_algebra)
            symplectic_matrix = self._ensure_broadcasting(
                symplectic_form.matrix,
                base_algebra
            )
            
            # Use in-place addition for efficiency
            base_algebra.add_(symplectic_matrix, alpha=self._SYMPLECTIC_WEIGHT)
            return base_algebra

    def compute_cohomology(self, point: Tensor) -> Tensor:
        """Compute cohomology class with pattern features.
        
        Computes the cohomology class by combining:
        1. Arithmetic form structure
        2. Height data from arithmetic geometry
        3. Pattern stability information
        
        Args:
            point: Point in total space to compute cohomology at
            
        Returns:
            Cohomology class tensor enriched with pattern structure
        """
        point = self._ensure_tensor_format(point)
        
        # Create arithmetic form with degree 2 for bundle cohomology
        form = ArithmeticForm(degree=2, coefficients=point)
        
        # Compute height and stability data in parallel
        height_data = self.height_structure.compute_height(point)
        stability_dict = self.pattern_formation.compute_stability(point)
        
        # Combine height and stability data efficiently
        pattern_stability = self._ensure_broadcasting(
            stability_dict["pattern_stability"],
            height_data
        )
        
        # Set enriched height data and return coefficients
        form.height_data = height_data * pattern_stability
        return form.coefficients

    def __del__(self) -> None:
        """Clean up cached tensors and components when the bundle is deleted."""
        # Clear cached properties safely
        for attr in ['_triu_indices', '_eye_matrix', '_reg_scale', 'basis_matrices']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Clear component references safely
        # Note: Components handle their own cleanup in their respective __del__ methods
        for attr in [
            'geometric_flow',
            'pattern_formation',
            'pattern_evolution',
            'height_structure',
            'symplectic',
            'operadic',
            'wave',
            'transition',
            'riemannian_framework',
            'pattern_dynamics'
        ]:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def structure_group_action(self, point: Tensor, group_element: Tensor) -> Tensor:
        """Apply structure group action with dimension handling.
        
        Args:
            point: Point to transform
            group_element: Structure group element
            
        Returns:
            Transformed point
        """
        # Ensure point has correct dimensions
        point = self._ensure_tensor_format(point)
        
        # Split point into base and fiber components
        base_coords = point[..., :self.base_dim]
        fiber_coords = point[..., self.base_dim:self.base_dim + self.fiber_dim]
        
        # Handle fiber dimension transition if needed
        if fiber_coords.shape[-1] != group_element.shape[-1]:
            operation = self.operadic.create_operation(
                source_dim=fiber_coords.shape[-1],
                target_dim=group_element.shape[-1]
            )
            fiber_coords = torch.einsum('bi,ij->bj', 
                                          fiber_coords.reshape(-1, fiber_coords.shape[-1]),
                                          operation.composition_law)
            fiber_coords = fiber_coords.reshape(-1, group_element.shape[-1])
        
        # Apply group action to fiber coordinates
        transformed_fiber = torch.matmul(
            group_element,
            fiber_coords.unsqueeze(-1)
        ).squeeze(-1)
        
        # Project back to original fiber dimension if needed
        if transformed_fiber.shape[-1] != self.fiber_dim:
            operation = self.operadic.create_operation(
                source_dim=transformed_fiber.shape[-1],
                target_dim=self.fiber_dim
            )
            transformed_fiber = torch.einsum('bi,ij->bj',
                                           transformed_fiber.reshape(-1, transformed_fiber.shape[-1]),
                                           operation.composition_law)
            transformed_fiber = transformed_fiber.reshape(-1, self.fiber_dim)
        
        # Recombine with base coordinates
        result = torch.cat([base_coords, transformed_fiber], dim=-1)
        
        return result

    #--------------------------------------------------------------------------
    # Enhanced Bundle Construction
    #--------------------------------------------------------------------------

    def construct_bundle(
        self,
        structure_group: str,
        fiber_type: str,
        base_manifold: Optional[str] = None
    ) -> None:
        """Construct fiber bundle with specified structure.
        
        Args:
            structure_group: Structure group ('SO3', 'U1', etc.)
            fiber_type: Type of fiber ('Vector', 'Principal', etc.)
            base_manifold: Optional base manifold type
        """
        self._structure_group_str = structure_group
        self._fiber_type = fiber_type
        self._base_manifold = base_manifold or 'Euclidean'
        
        # Initialize structure group
        if structure_group == 'SO3':
            self._initialize_so3_structure()
        elif structure_group == 'U1':
            self._initialize_u1_structure()
        else:
            raise ValueError(f"Unsupported structure group: {structure_group}")
            
        # Initialize fiber type
        if fiber_type == 'Vector':
            self._initialize_vector_fiber()
        elif fiber_type == 'Principal':
            self._initialize_principal_fiber()
        else:
            raise ValueError(f"Unsupported fiber type: {fiber_type}")

    def _initialize_so3_structure(self) -> None:
        """Initialize SO(3) structure group."""
        # Initialize Lie algebra basis
        self._initialize_basis_matrices()
        
        # Initialize group action
        def so3_action(g: Tensor, v: Tensor) -> Tensor:
            return torch.matmul(g, v)
        self._group_action = so3_action

    def _initialize_u1_structure(self) -> None:
        """Initialize U(1) structure group."""
        # Initialize phase action
        def u1_action(theta: Tensor, v: Tensor) -> Tensor:
            return v * torch.exp(1j * theta)
        self._group_action = u1_action

    def _initialize_vector_fiber(self) -> None:
        """Initialize vector bundle structure."""
        self.fiber_transition = lambda g, v: torch.matmul(g, v)
        self.fiber_metric = lambda v1, v2: torch.sum(v1 * v2, dim=-1)

    def _initialize_principal_fiber(self) -> None:
        """Initialize principal bundle structure."""
        self.fiber_transition = self._group_action
        self.fiber_metric = lambda g1, g2: torch.trace(
            torch.matmul(g1, g2.transpose(-2, -1))
        )

    #--------------------------------------------------------------------------
    # Type Management
    #--------------------------------------------------------------------------

    def validate_fiber_type(self, section: Tensor) -> bool:
        """Validate that section has correct fiber type.
        
        Args:
            section: Section to validate
            
        Returns:
            bool: Whether section has valid type
        """
        return self.fiber_type_manager.validate_fiber_type(
            section,
            self._fiber_type,
            self.fiber_dim
        )

    def convert_fiber_type(
        self,
        section: Tensor,
        target_type: str
    ) -> Tensor:
        """Convert section between fiber types.
        
        Args:
            section: Section to convert
            target_type: Target fiber type
            
        Returns:
            Converted section
        """
        return self.fiber_type_manager.convert_fiber_type(
            section,
            self._fiber_type,
            target_type,
            self.fiber_dim
        )

    def set_fiber_type(self, fiber_type: str) -> None:
        """Set the fiber type.
        
        Args:
            fiber_type: Name of fiber type to use
            
        Raises:
            ValueError: If fiber type not compatible
        """
        if not self.fiber_type_manager.check_compatibility(
            fiber_type,
            self._structure_group_str
        ):
            raise ValueError(
                f"Fiber type {fiber_type} not compatible with "
                f"structure group {self._structure_group_str}"
            )
        self._fiber_type = fiber_type

    #--------------------------------------------------------------------------
    # Enhanced Connection Forms
    #--------------------------------------------------------------------------

    def compute_connection(
        self,
        tangent_vector: Tensor,
        connection_type: str = 'Levi-Civita'
    ) -> Tensor:
        """Compute connection form with specified type."""
        logger.debug(f"\n=== Computing Connection Form ===")
        logger.debug(f"Tangent vector shape: {tangent_vector.shape}, requires_grad: {tangent_vector.requires_grad}")
        if tangent_vector.grad_fn:
            logger.debug(f"Tangent vector grad_fn: {tangent_vector.grad_fn}")
        
        if connection_type == 'Levi-Civita':
            # Get metric and compute Christoffel symbols using existing implementation
            metric = self.compute_metric(tangent_vector)
            # Use metric values directly to maintain gradient chain
            metric_values = metric.values
            logger.debug(f"\n=== Metric Computation ===")
            logger.debug(f"Metric shape: {metric_values.shape}, requires_grad: {metric_values.requires_grad}")
            if metric_values.grad_fn:
                logger.debug(f"Metric values grad_fn: {metric_values.grad_fn}")
            
            # Add hook to track metric gradients
            if metric_values.requires_grad:
                def metric_hook(grad):
                    logger.debug(f"\n=== Metric Gradient Hook (Connection) ===")
                    logger.debug(f"Shape: {grad.shape}")
                    logger.debug(f"Gradient norm: {torch.norm(grad)}")
                    logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                    logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                    logger.debug(f"Mean: {grad.mean().item()}")
                    logger.debug(f"Std: {grad.std().item()}")
                    return grad
                metric_values.register_hook(metric_hook)
            
            # Create new metric tensor with direct values
            metric = MetricTensor(values=metric_values, dimension=self.total_dim)
            
            # Compute Christoffel symbols with explicit gradient tracking
            christoffel = self.riemannian_framework.compute_christoffel(tangent_vector)
            # Use Christoffel values directly to maintain gradient chain
            christoffel_values = christoffel.values
            logger.debug(f"\n=== Christoffel Computation ===")
            logger.debug(f"Christoffel shape: {christoffel_values.shape}, requires_grad: {christoffel_values.requires_grad}")
            if christoffel_values.grad_fn:
                logger.debug(f"Christoffel values grad_fn: {christoffel_values.grad_fn}")
            
            # Add hook to track Christoffel gradients
            if christoffel_values.requires_grad:
                def christoffel_hook(grad):
                    logger.debug(f"\n=== Christoffel Gradient Hook ===")
                    logger.debug(f"Shape: {grad.shape}")
                    logger.debug(f"Gradient norm: {torch.norm(grad)}")
                    logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                    logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                    logger.debug(f"Mean: {grad.mean().item()}")
                    logger.debug(f"Std: {grad.std().item()}")
                    return grad
                christoffel_values.register_hook(christoffel_hook)
            
            # Use Christoffel values directly as connection form to maintain gradient chain
            connection_form = christoffel_values
            logger.debug(f"\n=== Connection Form Creation ===")
            logger.debug(f"Connection form shape: {connection_form.shape}, requires_grad: {connection_form.requires_grad}")
            if connection_form.grad_fn:
                logger.debug(f"Connection form grad_fn: {connection_form.grad_fn}")
            
            # Add hook to track connection form gradients
            if connection_form.requires_grad:
                def connection_hook(grad):
                    logger.debug(f"\n=== Connection Form Gradient Hook ===")
                    logger.debug(f"Shape: {grad.shape}")
                    logger.debug(f"Gradient norm: {torch.norm(grad)}")
                    logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                    logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                    logger.debug(f"Mean: {grad.mean().item()}")
                    logger.debug(f"Std: {grad.std().item()}")
                    return grad
                connection_form.register_hook(connection_hook)
            
            return connection_form
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")

    def compute_curvature(self) -> CurvatureTensor:
        """Compute curvature using existing RiemannianStructure implementation."""
        points = torch.ones(1, self.total_dim, device=self.device)
        return self.riemannian_framework.compute_curvature(points)

    def validate_connection(self, connection: ChristoffelSymbols) -> bool:
        """Validate connection using existing implementation."""
        return self.riemannian_framework.validate_connection_properties(connection)

    def validate_metric(self, metric: MetricTensor) -> bool:
        """Validate metric using existing implementation."""
        return self.riemannian_framework.validate_metric_properties(metric)

    def _ensure_compatible_dimensions(
        self,
        section: Tensor,
        path: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Ensure section and path have compatible dimensions.
        
        Args:
            section: Section tensor to make compatible
            path: Path tensor to make compatible
            
        Returns:
            Tuple of (compatible_section, compatible_path)
        """
        # Handle section dimension
        if section.shape[-1] != self.fiber_dim:
            section = self._handle_dimension_transition(section)
        
        # Handle path dimension
        if path.shape[-1] != self.base_dim:
            path = self._handle_dimension_transition(path)
        
        # Handle length compatibility
        if len(path) > len(section):
            path = path[:len(section)]
        elif len(path) < len(section):
            # Pad path by repeating last point
            pad_length = len(section) - len(path)
            path = torch.cat([
                path,
                path[-1:].expand(pad_length, -1)
            ])
        
        return section, path

    def _evolve(self, section: Tensor, path: Tensor, dt: float = 0.1) -> Tuple[Tensor, Tensor]:
        """Evolve a section along a path using RK4 integration."""
        # Get fiber metric using narrow to maintain gradient chain
        # Instead of using narrow directly, we'll use a view and clone to maintain gradient chain
        fiber_metric = self.metric[self.base_dim:, self.base_dim:].clone()
        logger.debug(f"\n=== Fiber Metric Extraction ===")
        logger.debug(f"Fiber metric shape: {fiber_metric.shape}, requires_grad: {fiber_metric.requires_grad}")
        if fiber_metric.grad_fn:
            logger.debug(f"Fiber metric grad_fn: {fiber_metric.grad_fn}")

        # Register gradient hook for fiber metric
        if fiber_metric.requires_grad:
            def fiber_metric_hook(grad):
                logger.debug(f"\n=== Fiber Metric Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            fiber_metric.register_hook(fiber_metric_hook)
        
        # Log shapes and gradient requirements
        logger.debug(f"Section shape: {section.shape}, requires_grad: {section.requires_grad}")
        logger.debug(f"Tangent shape: {path.shape}, requires_grad: {path.requires_grad}")
        logger.debug(f"Fiber metric shape: {fiber_metric.shape}, requires_grad: {fiber_metric.requires_grad}")
        
        # Compute transport step with explicit gradient tracking
        transport_step = self._transport_step(section, path, fiber_metric)
        
        # Add hook to track transport step gradients
        if transport_step.requires_grad:
            def transport_hook(grad):
                logger.debug(f"\n=== Transport Step Gradient Hook ===")
                logger.debug(f"Gradient shape: {grad.shape}")
                logger.debug(f"Gradient norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            transport_step.register_hook(transport_hook)
        
        return transport_step, fiber_metric

    def create_grad_hook(self, name):
        def hook(grad):
            logger.debug(f"\n=== {name} Gradient Hook ===")
            logger.debug(f"Shape: {grad.shape}")
            logger.debug(f"Gradient norm: {torch.norm(grad)}")
            logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
            logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
            logger.debug(f"Mean: {grad.mean().item()}")
            logger.debug(f"Std: {grad.std().item()}")
            return grad
        return hook

    def track_gradient_flow(self, tensor: Tensor, name: str) -> None:
        """Track gradient flow through a tensor by adding hooks and logging.
        
        Args:
            tensor: Tensor to track gradients for
            name: Name of the tensor for logging
        """
        if not tensor.requires_grad:
            logger.debug(f"\n=== {name} ===")
            logger.debug("No gradient tracking (requires_grad=False)")
            return
        
        logger.debug(f"\n=== {name} ===")
        logger.debug(f"Shape: {tensor.shape}")
        logger.debug(f"requires_grad: {tensor.requires_grad}")
        if tensor.grad_fn:
            logger.debug(f"grad_fn: {tensor.grad_fn}")
        
        def hook(grad):
            logger.debug(f"\n=== {name} Gradient ===")
            logger.debug(f"Shape: {grad.shape}")
            logger.debug(f"Norm: {torch.norm(grad)}")
            logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
            logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
            logger.debug(f"Mean: {grad.mean().item()}")
            logger.debug(f"Std: {grad.std().item()}")
            return grad
        
        tensor.register_hook(hook)

    def _add_gradient_hook(self, tensor: Tensor, name: str) -> None:
        if tensor.requires_grad:
            def hook(grad):
                logger.debug(f"\n=== {name} Gradient Hook ===")
                logger.debug(f"Shape: {grad.shape}")
                logger.debug(f"Norm: {torch.norm(grad)}")
                logger.debug(f"Has NaN: {torch.isnan(grad).any()}")
                logger.debug(f"Has Inf: {torch.isinf(grad).any()}")
                logger.debug(f"Mean: {grad.mean().item()}")
                logger.debug(f"Std: {grad.std().item()}")
                return grad
            tensor.register_hook(hook)
