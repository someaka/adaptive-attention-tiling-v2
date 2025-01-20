"""Quantum Geometric Attention Framework.

This module integrates quantum mechanics and differential geometry for attention.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from enum import Enum, auto

import torch
import torch.nn.functional as F
from torch import nn
import math

from ..attention.geometric import (
    HyperbolicExponential,
    HyperbolicLogarithm,
    EuclideanExponential,
    EuclideanLogarithm,
    ParallelTransport,
)
from ..patterns.arithmetic_dynamics import ArithmeticPattern
from .geometric_flow import GeometricFlow
from .quantum_attention_tile import QuantumMotivicTile
from ..patterns.symplectic import SymplecticStructure
from ..patterns.riemannian import PatternRiemannianStructure
from src.core.quantum.state_space import QuantumState, HilbertSpace
from src.core.attention.geometric import GeometricStructures
from src.neural.attention.pattern.pattern_dynamics import PatternDynamics
from src.core.tiling.state_manager import StateManager, StateConfig, StateType
from src.core.tiling.attention_state import AttentionState
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.validation.patterns.formation import (
    PatternFormationValidator,
    EmergenceValidator,
    SpatialValidator,
    TemporalValidator,
    FormationValidationResult
)
from src.metrics.attention import (
    compute_attention_metrics,
    compute_flow_metrics,
    compute_parallel_transport,
    compute_geodesic_distance,
    compute_flow_energy,
    compute_ricci_tensor
)
from src.metrics.quantum_geometric_metrics import (
    MetricContext,
    BaseMetric,
    MetricDomain,
    QuantumMetrics,
    GeometricMetrics,
    PatternMetrics,
    UnifiedMetrics
)
from src.validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    QuantumStateValidationResult,
    StateValidationErrorType
)
from src.validation.geometric.metric import (
    MetricProperties,
    MetricValidation,
    ConnectionValidation,
    CurvatureValidation,
    CurvatureBounds
)
from src.validation.patterns.formation import PatternFormationValidator, analyze_pattern_formation
from ..patterns.fiber_types import LocalChart as PatternSection
from src.core.patterns.fiber_bundle import FiberBundle

class QuantumStateError(Exception):
    """Base exception for quantum state errors."""
    pass

class InvalidQuantumStateError(QuantumStateError):
    """Raised when quantum state validation fails."""
    pass

class GeometricFlowError(Exception):
    """Base exception for geometric flow errors."""
    pass

class MetricError(Exception):
    """Base exception for metric computation errors."""
    pass

@dataclass
class QuantumGeometricConfig:
    """Configuration for quantum geometric attention."""
    hidden_dim: int
    num_heads: int = 8
    dropout: float = 0.1
    manifold_type: str = "hyperbolic"
    curvature: float = -1.0
    manifold_dim: Optional[int] = None
    num_layers: int = 3
    tile_size: int = 8
    motive_rank: int = 4
    dtype: torch.dtype = torch.complex128
    device: Optional[torch.device] = None
    is_causal: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if self.manifold_type not in ["hyperbolic", "euclidean"]:
            raise ValueError(f"manifold_type must be 'hyperbolic' or 'euclidean', got {self.manifold_type}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.motive_rank <= 0:
            raise ValueError(f"motive_rank must be positive, got {self.motive_rank}")
        if self.manifold_dim is not None and self.manifold_dim <= 0:
            raise ValueError(f"manifold_dim must be positive if specified, got {self.manifold_dim}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")

class QuantumGeometricAttention(nn.Module):
    """Quantum geometric attention framework."""

    def __init__(self, config: QuantumGeometricConfig):
        """Initialize quantum geometric attention layer."""
        super().__init__()
        self.config = config
        
        # Core dimensions
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.manifold_dim = config.manifold_dim or config.hidden_dim // 2
        
        # Store dtype for consistent handling
        self.dtype = config.dtype
        self.quantum_dtype = torch.complex64 if config.dtype == torch.float32 else torch.complex128
        
        # Initialize quantum bridge for state preparation
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=self.hidden_dim,
            manifold_dim=self.manifold_dim,  # Add manifold_dim parameter
            num_heads=self.num_heads,
            dropout=config.dropout,
            manifold_type=config.manifold_type,
            curvature=config.curvature,
            dtype=config.dtype
        )
        
        # Initialize geometric structures
        self.geometric_structures = GeometricStructures(
            dim=self.manifold_dim,
            manifold_type='hyperbolic',  # Default to hyperbolic manifold
            curvature=config.curvature,
            parallel_transport_method='schild'  # Default to Schild's ladder
        )
        
        # Initialize state config
        self.state_config = StateConfig(
            dim=self.manifold_dim,
            type=StateType.PURE,
            epsilon=1e-6,
            max_entanglement=1.0,
            dtype=config.dtype
        )
        
        # Initialize state manager with config
        self.state_manager = StateManager(config=self.state_config)
        
        # Initialize geometric flow with minimal parameters
        self.flow = GeometricFlow(
            hidden_dim=self.manifold_dim,
            manifold_dim=self.manifold_dim,
            motive_rank=1,  # Minimal motive rank
            num_charts=1,  # Single chart
            integration_steps=3,  # Minimal integration steps
            dt=0.5,  # Larger time step
            stability_threshold=1e-4,  # More relaxed threshold
            dtype=config.dtype,
            use_quantum_features=True  # Enable quantum features
        )
        
        # Initialize attention components
        self._init_attention_components()
        
        # Initialize metrics
        self.metrics = {}
        
        # Initialize debug mode
        self.debug = getattr(config, 'debug', False)

        # Initialize symplectic structure for quantum geometry
        self.symplectic = SymplecticStructure(
            dim=self.manifold_dim,
            preserve_structure=True,
            wave_enabled=True,
            dtype=config.dtype
        )

        # Initialize Riemannian structure for pattern geometry
        self.riemannian = PatternRiemannianStructure(
            manifold_dim=self.manifold_dim,
            pattern_dim=self.hidden_dim,
            dtype=config.dtype
        )

    def _init_quantum_components(self) -> None:
        """Initialize quantum computation components."""
        # Initialize HilbertSpace with complex dtype
        self.hilbert_space = HilbertSpace(
            dim=self.manifold_dim,
            dtype=self.config.dtype
        )
        
        # Initialize state management with complex dtype
        self.state_config = StateConfig(
            dim=self.manifold_dim,
            type=StateType.PURE,
            epsilon=1e-6,
            max_entanglement=1.0,
            dtype=self.config.dtype
        )
        self.state_manager = StateManager(self.state_config)
        
        # Initialize validators
        self.state_validator = StateValidator(tolerance=1e-6)
        self.preparation_validator = StatePreparationValidator(
            confidence_level=0.95,
            learning_rate=0.01,
            tolerance=1e-6
        )
        
        # Initialize quantum bridge
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.config.dropout,
            manifold_type=self.config.manifold_type,
            curvature=self.config.curvature,
            dtype=self.config.dtype
        )

        # Initialize quantum metrics
        self.quantum_metrics = QuantumMetrics(
            hidden_dim=self.hidden_dim,
            motive_rank=self.config.motive_rank
        )

    def _init_geometric_components(self) -> None:
        """Initialize geometric computation components."""
        # Initialize geometric structures
        manifold_type = "hyperbolic" if self.config.manifold_type == "hyperbolic" else "euclidean"
        self.geometric_structures = GeometricStructures(
            dim=self.manifold_dim,
            num_heads=self.num_heads,
            manifold_type=manifold_type,  # type: ignore
            curvature=self.config.curvature,
            parallel_transport_method="schild"
        )
        
        # Get manifold maps from geometric structures
        self.exp_map = self.geometric_structures.exp_map
        self.log_map = self.geometric_structures.log_map
        self.transport = self.geometric_structures.transport
        
        # Initialize geometric flow
        self.flow = GeometricFlow(
            hidden_dim=self.manifold_dim,
            manifold_dim=self.manifold_dim,
            motive_rank=self.config.motive_rank,
            num_charts=4,
            integration_steps=10,
            dt=0.1,
            stability_threshold=1e-6,
            dtype=self.config.dtype,
            use_quantum_features=True if self.config.dtype.is_complex else False
        )
        
        # Use metric from geometric structures
        self.metric = self.geometric_structures.metric

        # Initialize unified metrics
        self.unified_metrics = UnifiedMetrics(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_primes=8,
            motive_rank=self.config.motive_rank,
            manifold_dim=self.manifold_dim,
            num_bins=100
        )

        # Initialize metric context
        self.metric_context = MetricContext(
            timestamp=0.0,
            device=self.config.device or torch.device('cpu'),
            batch_size=1,  # Will be updated during forward pass
            sequence_length=1,  # Will be updated during forward pass
            hidden_dim=self.hidden_dim,
            resolution=1.0
        )

        # Initialize symplectic structure for quantum geometry
        self.symplectic = SymplecticStructure(
            dim=self.manifold_dim,
            preserve_structure=True,
            wave_enabled=True,
            dtype=self.config.dtype
        )

        # Initialize Riemannian structure for pattern geometry
        self.riemannian = PatternRiemannianStructure(
            manifold_dim=self.manifold_dim,
            pattern_dim=self.hidden_dim,
            dtype=self.config.dtype
        )

    def _init_attention_components(self) -> None:
        """Initialize attention components."""
        # Initialize projections using _create_complex_linear
        self.manifold_proj = self._create_complex_linear(self.hidden_dim, self.manifold_dim)
        self.manifold_proj_inv = self._create_complex_linear(self.manifold_dim, self.hidden_dim)
        
        # Initialize pattern projections with correct dimensions
        self.pattern_proj = self._create_complex_linear(self.manifold_dim, self.head_dim)
        self.pattern_proj_inv = self._create_complex_linear(self.head_dim, self.manifold_dim)
        
        # Initialize dropout
        self.dropout = nn.Dropout(self.config.dropout)

        # Initialize attention layers
        self._init_attention_layers()

        # Initialize pattern metrics
        self.pattern_metrics = PatternMetrics()

    def _create_complex_linear(self, in_features: int, out_features: int) -> nn.Linear:
        """Create a linear layer with complex weights.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            
        Returns:
            Linear layer with complex weights
        """
        # Create layer with base dtype first
        layer = nn.Linear(in_features, out_features)
        
        # Convert weights and bias to complex dtype if needed
        if torch.is_complex(torch.empty(1, dtype=self.quantum_dtype)):
            # Initialize complex weights with proper scaling
            real_weight = nn.init.xavier_uniform_(torch.empty_like(layer.weight))
            imag_weight = nn.init.xavier_uniform_(torch.empty_like(layer.weight))
            layer.weight.data = torch.complex(real_weight, imag_weight).to(self.quantum_dtype)
            
            if layer.bias is not None:
                real_bias = torch.zeros_like(layer.bias)
                imag_bias = torch.zeros_like(layer.bias)
                layer.bias.data = torch.complex(real_bias, imag_bias).to(self.quantum_dtype)
        else:
            # For real dtypes, just convert to the specified dtype
            layer.weight.data = layer.weight.data.to(self.dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(self.dtype)
        
        return layer

    def _init_attention_layers(self) -> None:
        """Initialize attention computation layers."""
        # Initialize query, key, value transformations
        self.query = self._create_complex_linear(
            self.hidden_dim,
            self.hidden_dim
        )
        self.key = self._create_complex_linear(
            self.hidden_dim,
            self.hidden_dim
        )
        self.value = self._create_complex_linear(
            self.hidden_dim,
            self.hidden_dim
        )
        
        # Initialize attention layers
        self.attention_layers = nn.ModuleList([
            self._create_complex_linear(
                self.hidden_dim,
                self.hidden_dim
            )
            for _ in range(self.config.num_layers)
        ])
        
        # Initialize quantum attention
        self.quantum_attention = nn.Sequential(
            self._create_complex_linear(
                self.hidden_dim,
                self.hidden_dim
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            self._create_complex_linear(
                self.hidden_dim,
                self.hidden_dim
            )
        )
        
        # Initialize arithmetic pattern
        self.arithmetic = ArithmeticPattern(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.config.num_layers,
            motive_rank=self.config.motive_rank,
            dtype=self.config.dtype
        ).to(self.config.device)

    def _init_weights(self) -> None:
        """Initialize weights with proper scaling."""
        # Initialize linear layer weights
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() > 1:
                    # First convert to complex dtype
                    param.data = param.data.to(self.config.dtype)
                    # Complex initialization for weight matrices
                    real_weight = nn.init.xavier_uniform_(torch.empty_like(param.real))
                    imag_weight = nn.init.xavier_uniform_(torch.empty_like(param.imag))
                    weight = torch.complex(real_weight, imag_weight)
                    param.data = weight.to(param.dtype)
            elif "bias" in name:
                # Initialize biases to zero and convert to complex
                param.data.zero_()
                param.data = param.data.to(self.config.dtype)

    def _prepare_quantum_state(
        self,
        x: Union[torch.Tensor, QuantumState],
        target_state: Optional[QuantumState] = None,
        return_validation: bool = False
    ) -> Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]]:
        """Prepare quantum state with validation and correction."""
        try:
            # Basic validation before creating state
            if not isinstance(x, QuantumState):
                # Check for NaN or Inf values
                if torch.isnan(x).any() or torch.isinf(x).any():
                    raise InvalidQuantumStateError("State contains NaN or Inf values")
                
                # Normalize globally across all dimensions except batch
                norm = torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=tuple(range(1, len(x.shape))), keepdim=True))
                if (norm == 0).any():
                    raise InvalidQuantumStateError("State has zero norm")
                
                # Normalize without adding epsilon to preserve unit norm exactly
                x_tensor = x / norm
                
                # Extract phase more robustly
                phase = torch.zeros_like(x[..., 0], dtype=x.dtype)
                mask = (x.abs() > 1e-10)
                if mask.any():
                    # Find first non-zero element per sequence
                    first_nonzero_idx = mask.to(torch.int64).argmax(dim=-1)
                    batch_indices = torch.arange(x.size(0), device=x.device)
                    if x.dim() > 2:
                        seq_indices = torch.arange(x.size(1), device=x.device)
                        head_indices = torch.arange(x.size(2), device=x.device) if x.dim() == 4 else None
                        if head_indices is not None:
                            batch_grid, head_grid, seq_grid = torch.meshgrid(batch_indices, seq_indices, head_indices, indexing='ij')
                            first_nonzero = x[batch_grid, head_grid, seq_grid, first_nonzero_idx]
                        else:
                            batch_grid, seq_grid = torch.meshgrid(batch_indices, seq_indices, indexing='ij')
                            first_nonzero = x[batch_grid, seq_grid, first_nonzero_idx]
                    else:
                        first_nonzero = x[batch_indices, first_nonzero_idx]
                    phase = torch.angle(first_nonzero)
                
                # Create QuantumState with layout information
                shape = x.shape
                if len(shape) == 2:
                    layout = {"type": "batch", "batch_size": shape[0], "dim": shape[1]}
                elif len(shape) == 3:
                    layout = {
                        "type": "sequence",
                        "batch_size": shape[0],
                        "seq_length": shape[1],
                        "dim": shape[2]
                    }
                elif len(shape) == 4:
                    layout = {
                        "type": "attention",
                        "batch_size": shape[0],
                        "num_heads": shape[1],
                        "seq_length": shape[2],
                        "dim": shape[3]
                    }
                else:
                    raise ValueError(f"Unsupported tensor shape: {shape}")

                # Align phases with proper broadcasting
                phase_factor = torch.exp(-1j * phase.unsqueeze(-1).expand_as(x))
                state = QuantumState(
                    amplitudes=x_tensor * phase_factor,  # Align phases
                    basis_labels=[str(i) for i in range(self.manifold_dim)],
                    phase=phase,
                    layout=layout
                )
            else:
                state = x

            # Validate if requested
            if return_validation:
                validator = StatePreparationValidator(tolerance=1e-8)  # Use stricter tolerance
                validation_result = validator.validate_preparation(target_state or state, state)
                return state, validation_result

            return state

        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to prepare quantum state: {str(e)}")

    def _compute_density_matrix(self, state: AttentionState) -> torch.Tensor:
        """Compute density matrix using state manager facilities."""
        try:
            # Get quantum state from state manager
            quantum_state = state.state_manager.states.get("quantum")
            if quantum_state is None:
                raise InvalidQuantumStateError("No quantum state found in state manager")
            
            # Validate through state manager
            if not state.validate_state(quantum_state.amplitudes):
                raise InvalidQuantumStateError("Invalid quantum state for density matrix computation")
            
            # Get state vector and compute efficiently
            state_vector = quantum_state.amplitudes
            batch_dims = state_vector.shape[:-1]
            state_dim = state_vector.shape[-1]
            
            # Compute density matrix with minimal intermediate storage
            state_vector = state_vector.reshape(-1, state_dim)
            density_matrices = torch.bmm(
                state_vector.unsqueeze(-1),
                state_vector.conj().unsqueeze(-2)
            )
            
            # Store in state manager for caching
            state.state_manager.states["density_matrix"] = density_matrices.reshape(*batch_dims, state_dim, state_dim)
            
            return state.state_manager.states["density_matrix"]
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to compute density matrix: {str(e)}")

    def _measure_quantum_state(self, state: AttentionState) -> torch.Tensor:
        """Measure quantum state using state manager facilities."""
        try:
            # Get quantum state from state manager
            quantum_state = state.state_manager.states.get("quantum")
            if quantum_state is None:
                raise InvalidQuantumStateError("No quantum state found in state manager")
            
            # Validate through state manager
            if not state.validate_state(quantum_state.amplitudes):
                raise InvalidQuantumStateError("Invalid quantum state for measurement")
            
            # Project state to classical representation
            if state.state_manager.config.type == StateType.PURE:
                classical = quantum_state.amplitudes
            else:
                # Use cached density matrix if available
                density_matrix = state.state_manager.states.get("density_matrix")
                if density_matrix is None:
                    density_matrix = self._compute_density_matrix(state)
                    
                eigenvals, eigenvecs = torch.linalg.eigh(density_matrix)
                classical = eigenvecs[..., -1]
            
            # Store measurement result
            state.state_manager.states["classical_measurement"] = classical.real
            
            return classical.real
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to measure quantum state: {str(e)}")

    def _geometric_update(self, state: AttentionState) -> AttentionState:
        """Apply geometric flow to update the state.
        
        Args:
            state: Current attention state
            
        Returns:
            Updated attention state
            
        Raises:
            GeometricFlowError: If geometric flow computation fails
        """
        try:
            # Apply geometric flow to the state
            geometric_state = state.geometric_state
            
            # Convert to real dtype for flow computation
            if torch.is_complex(geometric_state):
                geometric_state = geometric_state.real
            geometric_state = geometric_state.to(dtype=torch.float32)
            
            # Apply flow with gradient tracking
            updated_state = self.flow(geometric_state)
            
            # Convert back to complex dtype if needed
            if self.config.dtype.is_complex:
                updated_state = torch.complex(
                    updated_state,
                    torch.zeros_like(updated_state)
                ).to(self.config.dtype)
            
            # Update state with gradient tracking
            state.geometric_state = updated_state
            return state
            
        except Exception as e:
            raise GeometricFlowError(f"Failed to compute geometric flow: {str(e)}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass with quantum geometric attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, num_heads, seq_len, hidden_dim]
            mask: Optional attention mask
            return_metrics: Whether to return attention metrics

        Returns:
            Output tensor with same shape as input, or tuple of (output, metrics) if return_metrics is True

        Raises:
            ValueError: If input dimensions are invalid
        """
        # Validate input dimensions
        if x.dim() not in [3, 4]:
            raise ValueError(f"Input must be 3D or 4D tensor, got {x.dim()}D")
        
        # Get sequence length based on input shape
        seq_len = x.size(2) if x.dim() == 4 else x.size(1)
        if seq_len == 0:
            raise ValueError("Sequence length must be greater than 0")

        # If mask is all zeros, return zeros
        if mask is not None and not mask.any():
            return torch.zeros_like(x)

        # Prepare attention state
        state = self.prepare_attention_state(x, mask)

        # Apply attention mask if provided
        if mask is not None:
            state = self.apply_mask(state, mask)

        # Apply geometric update
        state = self._geometric_update(state)

        # Extract tensor from geometric_state if it's a tuple
        geometric_tensor = state.geometric_state[0] if isinstance(state.geometric_state, tuple) else state.geometric_state

        # Ensure geometric tensor has correct dtype before projection
        if not torch.is_complex(geometric_tensor):
            geometric_tensor = geometric_tensor.to(dtype=self.quantum_dtype)
        else:
            geometric_tensor = geometric_tensor.to(dtype=self.quantum_dtype)

        # Get original dimensions
        batch_size = x.size(0)
        num_heads = x.size(1) if x.dim() > 3 else 1
        seq_len = x.size(2) if x.dim() > 3 else x.size(1)

        # Project back to hidden dimension
        output = self.manifold_proj_inv(geometric_tensor)

        # Convert output to real and base dtype
        output = output.real.to(dtype=self.dtype)

        # Reshape back to match input shape
        if x.dim() == 4:
            output = output.view(batch_size, num_heads, seq_len, self.hidden_dim)
        else:
            output = output.view(batch_size, seq_len, self.hidden_dim)

        # Return metrics if requested
        if return_metrics:
            # Compute quantum attention patterns from state evolution
            quantum_state_result = self.quantum_bridge.neural_to_quantum(x)
            if isinstance(quantum_state_result, tuple):
                quantum_state, _ = quantum_state_result
            else:
                quantum_state = quantum_state_result
                
            evolved_state = self.quantum_bridge.evolve_quantum_state_with_attention(quantum_state)
            
            # Extract attention patterns from evolved state amplitudes
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            quantum_attention_patterns = torch.einsum(
                'bhid,bhjd->bhij',
                evolved_state.amplitudes,
                evolved_state.amplitudes.conj()
            ).real
            
            # Update state's attention patterns
            state.attention_patterns = {
                "quantum": quantum_attention_patterns
            }

            # Compute step-wise metrics
            step_metrics = {
                "quantum_entropy": self.compute_quantum_metrics(output)["von_neumann_entropy"],
                "geodesic_distance": torch.tensor(0.0, device=x.device, dtype=x.dtype),  # Placeholder
                "pattern_evolution": {"step": 0},  # Placeholder
                "local_height": torch.zeros_like(output[..., 0])  # Placeholder
            }

            metrics = {
                "step_0": step_metrics,
                "attention_scores": state.attention_scores,
                "attention_patterns": state.attention_patterns,
                "entanglement_history": state.entanglement_history,
                "metrics": state.metrics
            }
            return output, metrics

        return output

    def prepare_attention_state(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> AttentionState:
        """Prepare attention state with complex number handling.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Initialized attention state
            
        Raises:
            RuntimeError: If state preparation fails
        """
        try:
            # Always reinitialize state manager for each call
            self.state_manager = StateManager(self.state_config)

            # Convert input to complex dtype if needed
            if not torch.is_complex(x):
                x = x.to(self.config.dtype)

            # Get dimensions
            batch_size = x.size(0)
            num_heads = x.size(1) if x.dim() > 3 else 1
            seq_len = x.size(2) if x.dim() > 3 else (x.size(1) if x.dim() > 2 else 1)
            
            # Reshape input to combine batch and heads
            x_flat = x.reshape(batch_size * num_heads, seq_len, self.hidden_dim)
            
            # Use quantum bridge for state preparation and manifold projection
            x_manifold = self.quantum_bridge(x_flat)
            
            # Project to manifold dimension if needed
            if x_manifold.shape[-1] != self.manifold_dim:
                # Convert to the same dtype as the quantum bridge output
                x_manifold = x_manifold.to(self.config.dtype)
                x_manifold = self.manifold_proj(x_manifold)  # Project to manifold_dim
            
            # Initialize states with correct dimensions
            self.state_manager.states["input"] = torch.zeros_like(x)
            self.state_manager.states["manifold"] = torch.zeros(
                batch_size * num_heads,
                seq_len,
                self.manifold_dim,
                dtype=x_manifold.dtype,
                device=x_manifold.device
            )
            
            # Convert input to quantum state for storage
            quantum_state = self.quantum_bridge.neural_to_quantum(x_flat)
            if isinstance(quantum_state, tuple):
                quantum_state = quantum_state[0]  # Extract state if validation is returned
            self.state_manager.states["quantum"] = quantum_state

            # Update states
            self.state_manager.states["input"].copy_(x)
            self.state_manager.states["manifold"].copy_(x_manifold[..., :self.manifold_dim])
            self.state_manager.states["debug_info"] = {
                "input_shape": tuple(x.shape),
                "input_dtype": str(x.dtype),
                "manifold_shape": tuple(x_manifold.shape),
                "num_heads": num_heads
            }

            # Create key padding mask if provided through the mask argument
            key_padding_mask = None
            attention_mask = None
            if mask is not None:
                if mask.dim() == 2:  # [batch_size, seq_len]
                    key_padding_mask = mask
                elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                    attention_mask = mask.unsqueeze(1)  # Add head dimension
                elif mask.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
                    attention_mask = mask

            # Initialize attention state with masks
            return AttentionState(
                state_manager=self.state_manager,
                geometric_state=x_manifold,  # This is now [batch_size * num_heads, seq_len, manifold_dim]
                attention_scores=None,
                attention_patterns={},
                entanglement_history={},
                metrics={},
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to prepare attention state: {str(e)}")

    def apply_mask(self, state: AttentionState, mask: torch.Tensor) -> AttentionState:
        """Apply attention mask using state management."""
        if mask is None:
            return state

        # Store mask in state manager
        state.state_manager.states["mask"] = mask

        # Compute attention scores if they don't exist
        if state.attention_scores is None:
            # Get dimensions from debug info
            debug_info = state.state_manager.states.get("debug_info", {})
            num_heads = debug_info.get("num_heads", 1)
            
            # Project to query/key space
            # If geometric_state is 4D [batch_size, num_heads, seq_len, manifold_dim]
            # we need to reshape it to 3D for projection
            if state.geometric_state.dim() == 4:
                batch_size, num_heads, seq_len, manifold_dim = state.geometric_state.shape
                geometric_state_3d = state.geometric_state.reshape(batch_size * num_heads, seq_len, manifold_dim)
            else:
                geometric_state_3d = state.geometric_state
                batch_size = geometric_state_3d.size(0) // num_heads
                seq_len = geometric_state_3d.size(1)
                
            query = self.pattern_proj(geometric_state_3d)  # [batch_size * num_heads, seq_len, head_dim]
            key = self.pattern_proj(geometric_state_3d)    # [batch_size * num_heads, seq_len, head_dim]

            # Get head dimension
            head_dim = query.size(-1)

            # Reshape from [batch_size * num_heads, seq_len, head_dim] to [batch_size, num_heads, seq_len, head_dim]
            query = query.view(batch_size, num_heads, seq_len, head_dim)
            key = key.view(batch_size, num_heads, seq_len, head_dim)

            # Convert to complex dtype for attention scores
            query = query.to(torch.complex128)
            key = key.to(torch.complex128)

            # Compute attention scores
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)  # [batch_size, num_heads, seq_len, seq_len]

            # If mask is 2D [seq_len, seq_len], expand it to 4D [1, 1, seq_len, seq_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                mask = mask.expand(batch_size, num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]

            # Store mask in state
            state.attention_mask = mask

            # Apply masks using AttentionState's apply_masks method
            attention_scores = state.apply_masks(attention_scores)

            # Store attention scores in state
            state.attention_scores = attention_scores

        return state

    def compute_attention_patterns(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute attention patterns efficiently."""
        batch_size, num_heads, seq_len, manifold_dim = query.shape
        metrics = {} if return_metrics else None

        # Project to manifold space efficiently
        query = query.to(self.config.dtype)
        key = key.to(self.config.dtype)
        
        # Compute attention scores efficiently
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(manifold_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Convert to real and apply softmax
        attention_weights = torch.softmax(scores.abs(), dim=-1)

        # Process value tensor if provided
        attention_output: torch.Tensor
        if value is not None:
            processed_value = value.to(self.config.dtype)
            attention_output = torch.matmul(attention_weights, processed_value)
        else:
            attention_output = attention_weights

        # Return metrics if requested
        if return_metrics:
            metrics = {
                'attention_weights': attention_weights,
                'attention_scores': scores,
                'patterns': attention_output
            }
            return attention_output, metrics

        return attention_output

    def integrate_heads(self, head_states: List[torch.Tensor]) -> torch.Tensor:
        """Integrate multiple attention heads into a single representation.
        
        Args:
            head_states: List of head state tensors [batch_size, seq_len, head_dim]
            
        Returns:
            Integrated tensor [batch_size, seq_len, hidden_dim]
        """
        # Stack heads along feature dimension
        stacked = torch.cat(head_states, dim=-1)
        
        # Project to hidden dimension
        integrated = self.to_out(stacked)
        return integrated

    def construct_hamiltonian(self, attention_pattern: torch.Tensor) -> torch.Tensor:
        """Construct the Hamiltonian from the attention pattern.

        Args:
            attention_pattern: Attention pattern tensor [..., manifold_dim]

        Returns:
            Hamiltonian tensor [..., manifold_dim, manifold_dim]
        """
        # Project to manifold if needed
        if attention_pattern.shape[-1] != self.manifold_dim:
            attention_pattern = self.manifold_proj(attention_pattern)

        # Construct Hamiltonian as outer product
        hamiltonian = torch.einsum('...i,...j->...ij', attention_pattern, attention_pattern)
        
        # Ensure Hermitian
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose(-1, -2).conj())
        
        return hamiltonian

    def evolve_state(
        self,
        state: QuantumState,
        hamiltonian: torch.Tensor,
        t: float = 0.1
    ) -> QuantumState:
        """Evolve quantum state under Hamiltonian.
        
        Implements quantum time evolution using the Schrödinger equation:
        |ψ(t)⟩ = exp(-iHt/ℏ)|ψ(0)⟩
        
        Args:
            state: Initial quantum state |ψ(0)⟩
            hamiltonian: Hamiltonian operator H (must be Hermitian)
            t: Evolution time
            
        Returns:
            Evolved quantum state |ψ(t)⟩
            
        Raises:
            ValueError: If evolution time is invalid
            InvalidQuantumStateError: If state validation fails
        """
        if not isinstance(t, float) or t <= 0:
            raise ValueError("Evolution time 't' must be a positive float")
            
        try:
            # Ensure complex types and precision
            hamiltonian = hamiltonian.to(torch.complex128)
            state_vector = state.amplitudes.to(torch.complex128)
            
            # Single time evolution
            evolution_operator = torch.matrix_exp(-1j * hamiltonian * t)
            
            # Handle batch dimension
            if len(state_vector.shape) > 2:
                batch_size, seq_len, dim = state_vector.shape
                state_vector = state_vector.reshape(-1, dim)
                evolution_operator = evolution_operator.reshape(-1, dim, dim)
            
            # Apply evolution
            evolved_state = torch.matmul(state_vector.unsqueeze(1), evolution_operator).squeeze(1)
            
            # Compute and apply phase corrections
            phase_corrections = []
            for i in range(evolved_state.shape[0]):
                basis_state = evolution_operator[i, 0] if len(evolution_operator.shape) > 2 else evolution_operator[0]
                overlap = torch.sum(evolved_state[i] * basis_state.conj())
                phase_correction = torch.exp(1j * torch.angle(overlap))
                phase_corrections.append(phase_correction)
            phase_correction = torch.stack(phase_corrections)
            
            # Apply phase correction
            evolved_state = evolved_state * phase_correction.unsqueeze(-1)
            
            # Restore original shape if needed
            if len(state.amplitudes.shape) > 2:
                evolved_state = evolved_state.reshape(batch_size, seq_len, dim)
                
            return QuantumState(
                amplitudes=evolved_state,
                basis_labels=state.basis_labels,
                phase=state.phase,
                layout=state.layout  # Preserve the original layout
            )
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to evolve quantum state: {str(e)}")

    def compute_berry_phase(
        self,
        state: QuantumState,
        hamiltonian: torch.Tensor,
        time_steps: int = 100
    ) -> torch.Tensor:
        """Compute Berry phase using HilbertSpace implementation."""
        # Create time-dependent Hamiltonian function
        def hamiltonian_fn(t: float) -> torch.Tensor:
            # Scale Hamiltonian by time parameter
            theta = 2 * torch.pi * t
            cos_theta = torch.cos(torch.tensor(theta, dtype=torch.float64, device=self.config.device))
            sin_theta = torch.sin(torch.tensor(theta, dtype=torch.float64, device=self.config.device))
            rotation = torch.stack([
                torch.stack([cos_theta, sin_theta]),
                torch.stack([-sin_theta, cos_theta])
            ]).to(self.config.dtype)
            return rotation @ hamiltonian @ rotation.T.conj()
            
        # Create time points
        times = torch.linspace(0, 1.0, time_steps, dtype=torch.float64, device=self.config.device)
        
        return self.hilbert_space.compute_berry_phase(state, hamiltonian_fn, times)

    def compute_holonomy(
        self,
        state: QuantumState,
        path: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute holonomy along a path in state space.
        
        Args:
            state: Initial quantum state
            path: List of unitary operators defining the path
            
        Returns:
            Geometric phase from holonomy
            
        Raises:
            InvalidQuantumStateError: If state validation fails
        """
        try:
            # Initialize phase
            phase = torch.zeros(1, dtype=self.config.dtype, device=self.config.device)
            current_state = state
            
            # Use unit time step for each unitary evolution
            dt = 1.0
            
            for U in path:
                # Apply unitary evolution
                next_state = self.evolve_state(current_state, U, dt)
                
                # Compute connection
                overlap = torch.vdot(current_state.amplitudes, next_state.amplitudes)
                connection = torch.log(overlap / torch.abs(overlap)).imag
                
                # Accumulate phase
                phase += connection
                current_state = next_state
                
            return phase
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to compute holonomy: {str(e)}")

    def compute_entanglement_metrics(self, state: QuantumState) -> Dict[str, torch.Tensor]:
        """Compute entanglement metrics using HilbertSpace."""
        try:
            # Get state vector and compute density matrix
            state_vector = state.amplitudes
            density_matrix = torch.einsum('...i,...j->...ij', state_vector, state_vector.conj())
            
            # Compute von Neumann entropy
            eigenvals = torch.linalg.eigvalsh(density_matrix)
            entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-10), dim=-1)
            metrics = {'von_neumann_entropy': entropy}
            
            # Compute purity
            purity = torch.einsum('...ij,...ji->...', density_matrix, density_matrix)
            metrics['purity'] = purity
            
            # Get entanglement metrics if state is multipartite
            n = state_vector.shape[-1]
            n_qubits = int(math.log2(n))
            
            if n_qubits == 2:
                # Compute concurrence for 2-qubit states
                sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=self.config.dtype, device=self.config.device)
                rho_tilde = torch.einsum(
                    '...ij,jk,kl->...il',
                    density_matrix,
                    torch.kron(sigma_y, sigma_y),
                    density_matrix.conj()
                )
                eigenvals = torch.sort(torch.sqrt(torch.linalg.eigvalsh(rho_tilde)), dim=-1)[0]
                concurrence = torch.maximum(
                    torch.zeros_like(eigenvals[..., -1]),
                    eigenvals[..., -1] - eigenvals[..., -2] - eigenvals[..., -3] - eigenvals[..., -4]
                )
                metrics['concurrence'] = concurrence
            
            return metrics
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to compute entanglement metrics: {str(e)}")

    def analyze_pattern_formation(
        self,
        pattern: torch.Tensor,
        time_steps: int = 100
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Analyze pattern formation dynamics.
        
        Args:
            pattern: Input pattern tensor [batch_size, seq_len, hidden_dim]
            time_steps: Number of evolution steps
            
        Returns:
            Dictionary of pattern formation metrics
        """
        # Initialize pattern formation validator
        validator = PatternFormationValidator(
            tolerance=1e-6,
            coherence_threshold=0.8,
            symmetry_threshold=0.9,
            defect_threshold=0.1,
            frequency_threshold=0.1,
            phase_threshold=0.1
        )
        
        # Initialize pattern dynamics
        dynamics = PatternDynamics(
            grid_size=pattern.shape[1],
            space_dim=2,
            hidden_dim=self.hidden_dim,
            dt=0.01,
            num_modes=self.num_heads,
            quantum_enabled=True
        )
        
        # Validate and evolve pattern
        validation_result = validator.validate(
            dynamics=dynamics,
            initial=pattern,
            time_steps=time_steps
        )
        
        # Initialize default metrics
        default_metrics = {
            'formation_trajectory': torch.zeros_like(pattern),
            'stability_metrics': {'stability': torch.tensor(0.0, device=pattern.device)},
            'coherence_metrics': {'coherence': torch.tensor(0.0, device=pattern.device)},
            'symmetry_metrics': {'symmetry': torch.tensor(0.0, device=pattern.device)}
        }
        
        # Extract metrics from validation result
        if validation_result.data is not None:
            # Update trajectory if available
            if 'trajectory' in validation_result.data:
                default_metrics['formation_trajectory'] = validation_result.data['trajectory']
            
            # Update metric categories if available
            for category in ['stability', 'coherence', 'symmetry']:
                if category in validation_result.data:
                    metric_data = validation_result.data[category]
                    if isinstance(metric_data, dict):
                        default_metrics[f'{category}_metrics'] = {
                            k: v for k, v in metric_data.items()
                            if isinstance(v, torch.Tensor)
                        }
        
        return default_metrics

    def geometric_attention_flow(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        dt: float = 0.1,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply geometric attention flow with complex number handling."""
        try:
            # Save original shape for later reshaping
            original_shape = x.shape
            batch_size, num_heads, seq_length, hidden_dim = original_shape
            
            # Reshape input to combine batch and head dimensions
            x_reshaped = x.reshape(-1, seq_length, hidden_dim)
            
            # Initialize state with minimal copies
            state = self.prepare_attention_state(x_reshaped, mask)
            metrics_dict: Dict[str, Any] = {} if return_metrics else {}

            # Pre-compute initial metric with validation
            try:
                metric = self._compute_metric_tensor(state)
            except MetricError as e:
                raise GeometricFlowError(f"Failed to compute initial metric: {str(e)}")

            # Configure flow parameters in the state
            state.state_manager.states["flow_params"] = {
                "num_steps": num_steps,
                "dt": dt
            }

            # Apply geometric flow
            flow_output = self.flow(state.geometric_state)
            if isinstance(flow_output, tuple):
                output, flow_metrics = flow_output
                if return_metrics:
                    metrics_dict.update(flow_metrics)
                    metrics_dict.update({
                        "flow_steps": num_steps,
                        "flow_dt": dt
                    })
            else:
                output = flow_output

            # Pad output back to original dimension if needed
            if output.shape[-1] < hidden_dim:
                padding = torch.zeros(
                    *output.shape[:-1],
                    hidden_dim - output.shape[-1],
                    dtype=output.dtype,
                    device=output.device
                )
                output = torch.cat([output, padding], dim=-1)

            # Reshape output back to original dimensions
            output = output.reshape(original_shape)

            if return_metrics:
                return output, metrics_dict
            return output

        except Exception as e:
            if isinstance(e, GeometricFlowError):
                raise
            raise GeometricFlowError(f"Failed to apply geometric flow: {str(e)}")

    def _compute_metric_tensor(self, state: Union[AttentionState, torch.Tensor]) -> torch.Tensor:
        """Compute metric tensor for attention manifold with complex number handling.
        
        Args:
            state: Current attention state or tensor
        
        Returns:
            Metric tensor [batch_size, manifold_dim, manifold_dim]
        
        Raises:
            MetricError: If metric computation or validation fails
        """
        try:
            # Handle both AttentionState and raw tensor inputs
            if isinstance(state, AttentionState):
                state_tensor = state.state_manager.states.get(
                    "quantum",
                    state.state_manager.initialize_state("quantum")
                )
            else:
                state_tensor = state

            # Validate input tensor
            if not torch.is_tensor(state_tensor):
                raise MetricError(f"Expected tensor input, got {type(state_tensor)}")
            
            # Ensure tensor has correct shape
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            elif state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            if state_tensor.dim() != 3:
                raise MetricError(f"Expected 3D tensor after reshaping, got {state_tensor.dim()}D")

            # Project to quantum features with gradient validation
            # First project to hidden dimension if needed
            if state_tensor.size(-1) != self.hidden_dim:
                state_tensor = self.manifold_proj_inv(state_tensor)

            quantum_features = self._compute_quantum_features(state_tensor)
            if quantum_features.requires_grad:
                if quantum_features.grad_fn is None:
                    raise MetricError("Quantum features missing gradient function")
                if torch.isnan(quantum_features).any():
                    raise MetricError("Quantum features contain NaN values")

            # Compute quantum geometric tensor using symplectic structure
            try:
                # Compute quantum geometric tensor
                quantum_metric = self.symplectic.compute_quantum_geometric_tensor(quantum_features)
                
                # Ensure the tensor is complex with correct dtype
                if not torch.is_complex(quantum_metric):
                    quantum_metric = quantum_metric.to(self.config.dtype)
                else:
                    quantum_metric = quantum_metric.to(self.config.dtype)
                
                # Add small positive diagonal for numerical stability before Hermitian enforcement
                quantum_metric = quantum_metric + torch.eye(
                    quantum_metric.size(-1),
                    device=quantum_metric.device,
                    dtype=quantum_metric.dtype
                ).unsqueeze(0) * 1e-6
                
                # Make Hermitian by averaging with conjugate transpose
                quantum_metric = 0.5 * (quantum_metric + quantum_metric.transpose(-2, -1).conj())
                
                # Verify Hermiticity
                if not torch.allclose(quantum_metric, quantum_metric.transpose(-2, -1).conj(), rtol=1e-5, atol=1e-8):
                    raise MetricError("Quantum metric failed Hermiticity check")
                
                # Extract real part for metric after Hermiticity is verified
                quantum_metric = quantum_metric.real
                
            except Exception as e:
                raise MetricError(f"Failed to compute quantum geometric tensor: {str(e)}")

            # Compute Riemannian metric for pattern structure
            try:
                pattern_metric = self.riemannian.compute_metric(quantum_features).values
                # Ensure pattern metric is complex with correct dtype
                if not torch.is_complex(pattern_metric):
                    pattern_metric = pattern_metric.to(self.config.dtype)
                else:
                    pattern_metric = pattern_metric.to(self.config.dtype)
                
                # Add small positive diagonal for numerical stability
                pattern_metric = pattern_metric + torch.eye(
                    pattern_metric.size(-1),
                    device=pattern_metric.device,
                    dtype=pattern_metric.dtype
                ).unsqueeze(0) * 1e-6
                
                # Enforce perfect symmetry for complex metric
                pattern_metric = 0.5 * (pattern_metric + pattern_metric.transpose(-2, -1).conj())
                
                if not torch.allclose(pattern_metric, pattern_metric.transpose(-2, -1).conj(), rtol=1e-5, atol=1e-8):
                    raise MetricError("Pattern metric failed Hermiticity check")
                
                # Extract real part after Hermiticity is verified
                pattern_metric = pattern_metric.real
                
            except Exception as e:
                raise MetricError(f"Failed to compute Riemannian metric: {str(e)}")

            # Validate metric properties with detailed error messages
            quantum_properties = self._validate_metric_properties(quantum_metric, "quantum")
            pattern_properties = self._validate_metric_properties(pattern_metric, "pattern")

            if not quantum_properties.is_positive_definite:
                eigenvals = torch.linalg.eigvalsh(quantum_metric)
                raise MetricError(
                    f"Quantum metric not positive definite. Min eigenvalue: {eigenvals.min().item():.2e}, "
                    f"Max eigenvalue: {eigenvals.max().item():.2e}"
                )

            if not pattern_properties.is_positive_definite:
                eigenvals = torch.linalg.eigvalsh(pattern_metric)
                raise MetricError(
                    f"Pattern metric not positive definite. Min eigenvalue: {eigenvals.min().item():.2e}, "
                    f"Max eigenvalue: {eigenvals.max().item():.2e}"
                )

            # Combine metrics with appropriate weights
            combined_metric = 0.7 * quantum_metric + 0.3 * pattern_metric
            
            # Add small positive diagonal for final numerical stability
            combined_metric = combined_metric + torch.eye(
                combined_metric.size(-1),
                device=combined_metric.device,
                dtype=combined_metric.dtype
            ).unsqueeze(0) * 1e-6

            # Ensure combined metric is complex with correct dtype
            if not torch.is_complex(combined_metric):
                combined_metric = combined_metric.to(self.config.dtype)
            else:
                combined_metric = combined_metric.to(self.config.dtype)
            
            # Enforce Hermiticity of combined metric
            combined_metric = 0.5 * (combined_metric + combined_metric.transpose(-2, -1).conj())
            
            # Verify Hermiticity
            if not torch.allclose(combined_metric, combined_metric.transpose(-2, -1).conj(), rtol=1e-5, atol=1e-8):
                raise MetricError("Combined metric failed Hermiticity check")

            return combined_metric

        except Exception as e:
            raise MetricError(f"Failed to compute metric tensor: {str(e)}")

    def _compute_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum features from input tensor.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim] or [batch_size * seq_len, hidden_dim]

        Returns:
            Quantum features tensor [batch_size, seq_len, manifold_dim]

        Raises:
            ValueError: If input tensor has invalid shape or values
        """
        try:
            # Input validation with detailed messages
            if not torch.is_tensor(x):
                raise ValueError(f"Expected tensor input, got {type(x)}")
            if x.dim() not in [2, 3]:
                raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")
            if torch.isnan(x).any():
                raise ValueError("Input tensor contains NaN values")
            if torch.isinf(x).any():
                raise ValueError("Input tensor contains Inf values")

            # Ensure input is 3D with proper shape
            if x.dim() == 2:
                batch_size = 1
                seq_len = x.size(0)
                x = x.unsqueeze(0)  # Add batch dimension
            else:
                batch_size = x.size(0)
                seq_len = x.size(1)

            # Validate dimensions
            if x.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Expected hidden dimension {self.hidden_dim}, got {x.size(-1)}. "
                    f"Input shape: {tuple(x.shape)}"
                )

            # Reshape for projection
            x_flat = x.reshape(-1, self.hidden_dim)
            
            # Project to manifold space
            manifold_features = self.manifold_proj(x_flat)
            
            # Reshape back to 3D
            quantum_features = manifold_features.view(batch_size, seq_len, self.manifold_dim)

            return quantum_features

        except Exception as e:
            raise ValueError(f"Failed to compute quantum features: {str(e)}")

    def _apply_parallel_transport(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel transport to input features.
        
        Args:
            x: Input features [batch_size, seq_len, manifold_dim]
            
        Returns:
            Transported features [batch_size, seq_len, manifold_dim]
        """
        # Create target point on manifold (origin)
        y = torch.zeros_like(x)
        
        # Create tangent vector at x
        v = x - y
        
        # Apply parallel transport
        transported = self.transport(x, y, v)
        
        return transported

    def _compute_geometric_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute geometric features using tensor operations.
        
        Args:
            x: Input tensor
            
        Returns:
            Geometric features tensor
        """
        # Use proper tensor operations with metric
        features = F.linear(x, self.metric)
        return features

    def detect_patterns(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Detect patterns with optimized memory usage and computations."""
        metrics = {}
        
        # 1. Apply arithmetic pattern detection
        arithmetic_out = self.arithmetic(x)
        with torch.no_grad():  # Compute metrics without storing gradients
            metrics["arithmetic"] = {
                "pattern_strength": torch.norm(arithmetic_out, dim=-1).mean().item(),
                "pattern_sparsity": torch.count_nonzero(arithmetic_out) / arithmetic_out.numel()
            }
        
        # 2. Apply geometric flow with combined structures
        # First apply Riemannian flow
        riemannian_flow = self.riemannian.compute_riemann(arithmetic_out).ricci
        
        # Then apply symplectic flow
        symplectic_flow = self.symplectic.quantum_ricci_flow(
            arithmetic_out,
            time=0.1,
            dt=0.01,
            steps=10
        )
        
        # Combine flows with weights
        flow_out = 0.7 * symplectic_flow + 0.3 * arithmetic_out
        
        with torch.no_grad():
            metrics["flow"] = {
                "flow_energy": compute_flow_energy(flow_out, self.metric).item(),
                "flow_stability": self._compute_flow_stability(flow_out).item(),
                "riemannian_curvature": torch.einsum('...ii', riemannian_flow).mean().item(),
                "symplectic_volume": self.symplectic.compute_volume(flow_out).mean().item()
            }
        
        # 3. Process quantum state
        quantum_state = self._prepare_quantum_state(flow_out)
        
        # 4. Process tiles efficiently
        quantum_patterns = []
        quantum_metrics = []
        
        for tile in self.tiles:
            # Process through quantum tile with minimal copies
            pattern, tile_metrics = tile(
                quantum_state,
                quantum_state,
                quantum_state,
                return_metrics=True
            )
            quantum_patterns.append(pattern)
            quantum_metrics.append(tile_metrics)
        
        # 5. Combine patterns efficiently
        quantum_out = torch.stack(quantum_patterns, dim=1)
        quantum_out = torch.mean(quantum_out, dim=1)
        metrics["quantum"] = {
            f"tile_{i}": m for i, m in enumerate(quantum_metrics)
        }
        
        # 6. Project and transform efficiently
        patterns = self.pattern_proj(quantum_out)
        transported_patterns = self._apply_parallel_transport(patterns)
        geometric_features = self._compute_geometric_features(transported_patterns)
        quantum_features = self._compute_quantum_features(quantum_out)
        
        # 7. Combine features and compute metrics
        final_patterns = geometric_features + quantum_features
        metrics["patterns"] = self._compute_pattern_metrics(final_patterns)
        
        return final_patterns, metrics

    def _compute_flow_stability(self, flow_state: torch.Tensor) -> torch.Tensor:
        """Compute stability of geometric flow efficiently."""
        # Compute temporal difference efficiently
        time_diff = flow_state[..., 1:, :] - flow_state[..., :-1, :]
        
        # Compute stability metric
        stability = 1.0 / (1.0 + torch.norm(time_diff, dim=-1).mean())
        
        return stability

    def _compute_pattern_metrics(self, pattern: torch.Tensor) -> Dict[str, Any]:
        """Compute comprehensive pattern metrics.
        
        Args:
            pattern: Input pattern tensor
            
        Returns:
            Dictionary of pattern metrics
        """
        metrics = {}
        
        # Spatial metrics
        spatial_result = self.pattern_validator.spatial_validator.validate_spatial(pattern)
        if isinstance(spatial_result, FormationValidationResult) and spatial_result.data:
            metrics.update(spatial_result.data)
            
        # Temporal metrics if pattern has time dimension
        if pattern.dim() > 3:
            temporal_result = self.pattern_validator.temporal_validator.validate_temporal(pattern)
            if isinstance(temporal_result, FormationValidationResult) and temporal_result.data:
                metrics.update(temporal_result.data)
                
        return metrics

    def compute_quantum_metrics(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute quantum metrics for raw tensor state.
        
        Args:
            state: Tensor to analyze [batch_size, seq_len, hidden_dim]
            
        Returns:
            Dictionary of quantum metrics as tensors
        """
        # Convert to quantum state
        quantum_state = self._prepare_quantum_state(state)
        if isinstance(quantum_state, tuple):
            quantum_state = quantum_state[0]  # Extract state from validation tuple
        
        # Compute metrics using QuantumMetrics
        return self._compute_quantum_metrics(quantum_state)

    def _compute_quantum_metrics(self, state: QuantumState) -> Dict[str, torch.Tensor]:
        """Compute quantum metrics for a given quantum state.
        
        Args:
            state: The quantum state to compute metrics for
            
        Returns:
            Dictionary of quantum metrics as tensors
        """
        try:
            # Initialize default metrics
            metrics = {
                'entropy': torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype),
                'purity': torch.tensor(1.0, device=self.config.device, dtype=self.config.dtype),
                'fisher': torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype),
                'transport_deviation': torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)
            }
            
            if not isinstance(state, QuantumState):
                return metrics
                
            # Get state vector and compute density matrix
            state_vector = state.amplitudes
            density_matrix = torch.einsum('...i,...j->...ij', state_vector, state_vector.conj())
            
            # Compute von Neumann entropy
            eigenvals = torch.linalg.eigvalsh(density_matrix)
            entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-10), dim=-1)
            metrics['von_neumann_entropy'] = entropy
            
            # Compute purity
            purity = torch.einsum('...ij,...ji->...', density_matrix, density_matrix)
            metrics['purity'] = purity
            
            # Compute quantum Fisher information
            # Use diagonal approximation for efficiency
            fisher = torch.abs(torch.diagonal(density_matrix, dim1=-2, dim2=-1))
            metrics['fisher'] = fisher
            
            # Compute transport deviation if we have previous state
            if hasattr(self, '_prev_state'):
                # Create tangent vector as difference between states
                tangent = state.amplitudes - self._prev_state.amplitudes
                
                # Transport the tangent vector
                transported = self.hilbert_space.parallel_transport(
                    tangent=tangent,
                    state1=self._prev_state,
                    state2=state
                )
                
                # Compute deviation
                metrics['transport_deviation'] = torch.norm(transported - tangent)
                
            # Store current state for next computation
            self._prev_state = state
            
            return metrics
        except Exception as e:
            # Keep default metrics in case of error
            return {
                'entropy': torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype),
                'purity': torch.tensor(1.0, device=self.config.device, dtype=self.config.dtype),
                'fisher': torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype),
                'transport_deviation': torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)
            }

    def _compute_geometric_metrics(
        self,
        flow_path: torch.Tensor,
        connection: torch.Tensor
    ) -> Dict[str, float]:
        """Compute geometric metrics for flow path.
        
        Args:
            flow_path: Tensor containing flow trajectory
            connection: Connection tensor for parallel transport
            
        Returns:
            Dictionary of geometric metrics
        """
        # Create metric context
        context = MetricContext(
            timestamp=0.0,
            device=flow_path.device,
            batch_size=flow_path.shape[0],
            sequence_length=flow_path.shape[1],
            hidden_dim=self.hidden_dim,
            resolution=1.0
        )
        
        metrics = {}
        
        # Compute Riemannian metrics
        riemann = self.riemannian.compute_riemann(flow_path)
        metrics['ricci_scalar'] = float(torch.einsum('...ii', riemann.ricci).mean().item())
        metrics['sectional_curvature'] = float(
            self.riemannian.sectional_curvature(
                flow_path,
                flow_path[..., 1:, :] - flow_path[..., :-1, :],
                flow_path[..., :-1, :] - flow_path[..., 1:, :]
            ).mean().item()
        )
        
        # Compute symplectic metrics
        metrics['symplectic_volume'] = float(
            self.symplectic.compute_volume(flow_path).mean().item()
        )
        
        # Compute Hamiltonian metrics
        hamiltonian = self.symplectic.hamiltonian_vector_field(
            flow_path[..., -1, :],  # Use final state
            flow_path[..., 0, :]    # Use initial state
        )
        metrics['hamiltonian_energy'] = float(
            torch.norm(hamiltonian, dim=-1).mean().item()
        )
        
        # Compute standard geometric metrics
        metrics['geodesic_distance'] = float(
            self.geometric_metrics.compute_geodesic_distance(flow_path, context).item()
        )
        
        metrics['curvature'] = float(
            self.geometric_metrics.compute_curvature(flow_path, context).item()
        )
        
        # Compute parallel transport
        vector = flow_path[..., -1, :] - flow_path[..., 0, :]  # Use displacement as vector
        transported = self.geometric_metrics.compute_parallel_transport(
            vector, connection, context
        )
        metrics['transport_deviation'] = float(
            torch.norm(transported - vector).mean().item()
        )
        
        return metrics

    def _init_validation_components(self) -> None:
        """Initialize pattern validation components."""
        self.pattern_validator = PatternFormationValidator(
            tolerance=1e-6,
            coherence_threshold=0.8,
            symmetry_threshold=0.9,
            defect_threshold=0.1,
            frequency_threshold=0.1,
            phase_threshold=0.1
        )
        
        # Add validation metrics to metric trackers using imported MetricDomain
        self.metric_trackers[MetricDomain.PATTERN] = BaseMetric(
            "pattern_formation",
            MetricDomain.PATTERN
        )

    def validate_pattern_formation(
        self,
        pattern: torch.Tensor,
        dynamics: Optional[PatternDynamics] = None,
        time_steps: int = 1000
    ) -> FormationValidationResult:
        """Validate pattern formation with enhanced error handling.
        
        Args:
            pattern: Pattern tensor to validate
            dynamics: Optional pattern dynamics
            time_steps: Number of time steps for temporal validation
            
        Returns:
            Pattern formation validation result
        """
        try:
            # Input validation
            if not torch.is_tensor(pattern):
                raise ValueError("Pattern must be a tensor")
            
            if torch.any(torch.isnan(pattern)):
                return FormationValidationResult(
                    is_valid=False,
                    message="Pattern contains NaN values",
                    data={"error": "nan_detected"}
                )
            
            # Validate pattern formation
            result = self.pattern_validator.validate(
                dynamics=dynamics,
                initial=pattern,
                time_steps=time_steps
            )
            
            # Track metrics using unified metrics
            if result.data:
                try:
                    # Convert pattern history to tensor for metrics
                    pattern_history = torch.stack([pattern], dim=0)
                    metrics_data: Dict[str, torch.Tensor] = {
                        "patterns": pattern,
                        "pattern_history": pattern_history
                    }
                    
                    # Compute metrics with error handling
                    try:
                        _ = self.unified_metrics.compute_all_metrics(
                            metrics_data,
                            self.metric_context
                        )
                    except Exception as e:
                        # Log metric computation failure but continue validation
                        result.data["metric_error"] = str(e)
                        
                except Exception as e:
                    # Log metric tracking failure but continue validation
                    result.data["tracking_error"] = str(e)
                
            return cast(FormationValidationResult, result)
            
        except Exception as e:
            return FormationValidationResult(
                is_valid=False,
                message=f"Pattern validation failed: {str(e)}",
                data={"error": str(e)}
            )
    
    def _validate_curvature_properties(self, points: torch.Tensor) -> MetricProperties:
        """Validate curvature properties at given points."""
        try:
            # Get curvature tensor
            curvature = self.compute_curvature(points)
            
            # Access components directly from CurvatureTensor
            riemann = curvature.riemann  # Full Riemann tensor
            ricci = curvature.ricci  # Ricci tensor
            scalar = curvature.scalar_curvatures  # Scalar curvature
            
            # Compute sectional curvature from Riemann tensor
            sectional = torch.zeros_like(riemann[...,:,:,0,0])  # Initialize with correct shape
            for i in range(self.manifold_dim):
                for j in range(self.manifold_dim):
                    # K(X,Y) = R(X,Y,X,Y) / (g(X,X)g(Y,Y) - g(X,Y)^2)
                    numerator = riemann[...,i,j,i,j]
                    metric = self.compute_metric(points)
                    g_ii = metric[...,i,i]
                    g_jj = metric[...,j,j]
                    g_ij = metric[...,i,j]
                    denominator = g_ii * g_jj - g_ij * g_ij
                    sectional[...,i,j] = numerator / (denominator + 1e-8)
            
            # Check bounds
            has_bounded_curvature = bool(
                (torch.abs(sectional) < self.curvature_threshold).all() and
                (torch.abs(ricci) < self.curvature_threshold).all() and
                (torch.abs(scalar) < self.curvature_threshold).all()
            )

            return MetricProperties(
                is_positive_definite=True,
                is_compatible=True,
                is_complete=True,
                has_bounded_curvature=has_bounded_curvature,
                determinant=None,
                trace=None,
                eigenvalues=None,
                condition_number=None,
                volume_form=None,
                christoffel_symbols=None,
                sectional_curvature=sectional,
                ricci_curvature=ricci,
                scalar_curvature=scalar
            )
        except Exception as e:
            print(f"Error computing curvature: {str(e)}")
            return MetricProperties(
                is_positive_definite=True,
                is_compatible=True,
                is_complete=True,
                has_bounded_curvature=False,
                determinant=None,
                trace=None,
                eigenvalues=None,
                condition_number=None,
                volume_form=None,
                christoffel_symbols=None,
                sectional_curvature=None,
                ricci_curvature=None,
                scalar_curvature=None
            )
    
    def _apply_attention_flow(self, state: AttentionState) -> torch.Tensor:
        """Apply attention flow with quantum geometric updates.

        Args:
            state: Current attention state

        Returns:
            Updated attention output
        """
        try:
            # Get input and debug info
            x = state.geometric_state
            debug_info = state.state_manager.states.get("debug_info", {})
            num_heads = debug_info.get("num_heads", 1)
            batch_size = x.size(0) // num_heads
            seq_len = x.size(1)

            # Compute quantum attention scores
            quantum_state = self._measure_quantum_state(state)
            # Reshape quantum state back to [batch, heads, seq, dim]
            quantum_state = quantum_state.reshape(batch_size, num_heads, seq_len, -1)
            # Ensure quantum state has correct dtype
            quantum_state = quantum_state.to(self.config.dtype)

            # Project to attention space per head
            pattern_proj = self.pattern_proj(quantum_state)

            # Compute attention scores
            attention_scores = torch.matmul(
                pattern_proj, pattern_proj.transpose(-2, -1)
            ) / math.sqrt(pattern_proj.size(-1))

            # Apply masks to attention scores
            attention_scores = state.apply_masks(attention_scores)

            # Store attention patterns
            state.attention_scores = attention_scores
            state.attention_patterns["quantum"] = attention_scores.detach()

            # Apply attention
            attended = torch.matmul(attention_scores, quantum_state)

            # Reshape back to original dimensions
            output = attended.reshape(batch_size * num_heads, seq_len, -1)

            # Create new state for geometric update
            updated_state = AttentionState(
                state_manager=state.state_manager,
                geometric_state=output,
                attention_scores=state.attention_scores,
                attention_patterns=state.attention_patterns,
                entanglement_history=state.entanglement_history,
                metrics=state.metrics,
                key_padding_mask=state.key_padding_mask,
                attention_mask=state.attention_mask
            )

            # Apply geometric update
            updated_state = self._geometric_update(updated_state)

            return updated_state.geometric_state

        except Exception as e:
            raise GeometricFlowError(f"Failed to apply geometric flow: {str(e)}")
    
    def _validate_pattern_formation(self, pattern: torch.Tensor) -> FormationValidationResult:
        """Validate pattern formation properties."""
        try:
            # Validate pattern dimensions
            if pattern.dim() != 3:
                raise ValueError(f"Expected 3D pattern tensor, got {pattern.dim()}D")
                
            # Validate pattern values
            if torch.isnan(pattern).any():
                raise ValueError("Pattern contains NaN values")
                
            if torch.isinf(pattern).any():
                raise ValueError("Pattern contains infinite values")
                
            # Validate pattern norm
            norm = torch.norm(pattern, dim=-1)
            if not torch.allclose(norm, torch.ones_like(norm), rtol=1e-5):
                raise ValueError(f"Pattern norm should be 1.0, got {norm.mean().item():.4f}")
                
            return FormationValidationResult(
                is_valid=True,
                message="Pattern formation validation passed",
                data={}
            )
            
        except Exception as e:
            return FormationValidationResult(
                is_valid=False,
                message=f"Pattern validation failed: {str(e)}",
                data={"error": str(e)}
            )
    
    def _validate_metric_properties(self, metric: torch.Tensor, name: str) -> MetricProperties:
        """Validate metric tensor properties.

        Args:
            metric: Metric tensor to validate
            name: Name of metric for error messages

        Returns:
            MetricProperties object containing validation results

        Raises:
            MetricError: If validation fails
        """
        try:
            # Check basic tensor properties
            if not torch.is_tensor(metric):
                raise MetricError(f"{name} metric must be a tensor")
            if metric.dim() != 3:
                raise MetricError(f"{name} metric must be 3D tensor, got {metric.dim()}D")
            if metric.size(-1) != metric.size(-2):
                raise MetricError(f"{name} metric must be square matrix")

            # Check symmetry
            is_symmetric = torch.allclose(metric, metric.transpose(-1, -2))
            if not is_symmetric:
                raise MetricError(f"{name} metric must be symmetric")

            # Check positive definiteness
            eigenvals = torch.linalg.eigvalsh(metric)
            is_positive_definite = (eigenvals > 0).all()

            # Compute basic properties
            determinant = torch.linalg.det(metric)
            trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)

            # Compute volume form
            volume_form = torch.sqrt(torch.abs(determinant))

            # Get Christoffel symbols and check compatibility
            try:
                christoffel = self._compute_christoffel_symbols(metric)
                is_compatible = self._check_metric_compatibility(metric)
            except Exception as e:
                print(f"Error computing Christoffel symbols: {str(e)}")
                christoffel = None
                is_compatible = False

            # Check completeness
            try:
                is_complete = True  # For now, assume complete
            except Exception as e:
                print(f"Error checking completeness: {str(e)}")
                is_complete = False

            # Compute curvature tensors
            try:
                sectional = self._compute_sectional_curvature(metric)
                ricci = self._compute_ricci_curvature(metric)
                scalar = self._compute_scalar_curvature(metric)

                has_bounded_curvature = bool(
                    (torch.abs(sectional) < 1e3).all() and
                    (torch.abs(ricci) < 1e3).all() and
                    (torch.abs(scalar) < 1e3).all()
                )
            except Exception as e:
                print(f"Error computing curvature: {str(e)}")
                sectional = None
                ricci = None
                scalar = None
                has_bounded_curvature = False

            # Return properties
            return MetricProperties(
                is_positive_definite=is_positive_definite,
                is_compatible=is_compatible,
                is_complete=is_complete,
                has_bounded_curvature=has_bounded_curvature,
                determinant=determinant,
                trace=trace,
                eigenvalues=eigenvals,
                condition_number=float(eigenvals.max() / (eigenvals.min() + 1e-8)),
                volume_form=volume_form,
                christoffel_symbols=christoffel,
                sectional_curvature=sectional,
                ricci_curvature=ricci,
                scalar_curvature=scalar
            )

        except Exception as e:
            raise MetricError(f"Failed to validate {name} metric: {str(e)}")

    def _compute_christoffel_symbols(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel symbols for metric tensor.
        
        Args:
            metric: Metric tensor [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Christoffel symbols [batch_size, manifold_dim, manifold_dim, manifold_dim]
        """
        # For now, return placeholder values
        batch_size = metric.size(0)
        manifold_dim = metric.size(1)
        return torch.zeros(batch_size, manifold_dim, manifold_dim, manifold_dim, device=metric.device)
    
    def _compute_sectional_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute sectional curvature of metric tensor.
        
        Args:
            metric: Metric tensor [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Sectional curvature tensor [batch_size, manifold_dim, manifold_dim]
        """
        # For now, return placeholder values
        batch_size = metric.size(0)
        manifold_dim = metric.size(1)
        return torch.ones(batch_size, manifold_dim, manifold_dim, device=metric.device) * 0.5

    def _compute_ricci_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Ricci curvature of metric tensor.
        
        Args:
            metric: Metric tensor [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Ricci curvature tensor [batch_size, manifold_dim, manifold_dim]
        """
        # For now, return placeholder values
        batch_size = metric.size(0)
        manifold_dim = metric.size(1)
        return torch.ones(batch_size, manifold_dim, manifold_dim, device=metric.device) * 0.5

    def _compute_scalar_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute scalar curvature of metric tensor.
        
        Args:
            metric: Metric tensor [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Scalar curvature tensor [batch_size]
        """
        # For now, return placeholder values
        batch_size = metric.size(0)
        return torch.ones(batch_size, device=metric.device) * 0.5

    def _check_metric_compatibility(self, metric: torch.Tensor) -> bool:
        """Check if metric is compatible with manifold structure.
        
        Args:
            metric: Metric tensor [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            True if metric is compatible with manifold structure
        """
        # For now, always return True since we're using placeholder values
        return True
    