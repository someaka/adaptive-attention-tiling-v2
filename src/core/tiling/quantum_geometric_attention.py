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
    dtype: torch.dtype = torch.complex64
    device: Optional[torch.device] = None
    is_causal: bool = False

class QuantumGeometricAttention(nn.Module):
    """Quantum geometric attention framework."""

    def __init__(self, config: QuantumGeometricConfig):
        """Initialize quantum geometric attention.
        
        Args:
            config: Configuration object containing all parameters
        """
        super().__init__()
        self.config = config
        
        # Core dimensions
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.manifold_dim = config.manifold_dim or config.hidden_dim // 2

        # Initialize quantum infrastructure
        self._init_quantum_components()
        
        # Initialize geometric components
        self._init_geometric_components()
        
        # Initialize attention components
        self._init_attention_components()
        
        # Initialize weights
        self._init_weights()

    def _init_quantum_components(self) -> None:
        """Initialize quantum computation components."""
        # Initialize HilbertSpace
        self.hilbert_space = HilbertSpace(
            dim=self.manifold_dim,
            dtype=self.config.dtype
        )
        
        # Initialize state management
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
        self.preparation_validator = StatePreparationValidator(tolerance=1e-6)
        
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
            dtype=self.config.dtype
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
        """Initialize attention computation components."""
        # Initialize tiles
        self.tiles = nn.ModuleList([
            QuantumMotivicTile(
                size=self.config.tile_size,
                hidden_dim=self.hidden_dim,
                num_heads=1,
                dropout=self.config.dropout,
                resolution=1.0,
                cohomology_dim=self.manifold_dim,
                motive_rank=self.config.motive_rank,
                dtype=self.config.dtype
            )
            for _ in range(self.num_heads)
        ])
        
        # Initialize projections
        self._init_projections()
        
        # Initialize attention layers
        self._init_attention_layers()

        # Initialize pattern metrics
        self.pattern_metrics = PatternMetrics()

    def _init_projections(self) -> None:
        """Initialize all projection layers."""
        # Initialize manifold projections
        self.manifold_proj = self._create_complex_linear(
            self.head_dim,
            self.manifold_dim
        )
        self.manifold_proj_inv = self._create_complex_linear(
            self.manifold_dim,
            self.head_dim
        )
        
        # Initialize pattern projections
        self.pattern_proj = self._create_complex_linear(
            self.manifold_dim,
            self.head_dim
        )
        self.pattern_proj_inv = self._create_complex_linear(
            self.head_dim,
            self.manifold_dim
        )
        
        # Initialize QKV projections
        expanded_dim = self.num_heads * self.hidden_dim
        self.to_qkv = self._create_complex_linear(
            self.hidden_dim,
            3 * expanded_dim
        )
        self.to_out = self._create_complex_linear(
            expanded_dim,
            self.hidden_dim
        )

    def _create_complex_linear(
        self,
        in_features: int,
        out_features: int
    ) -> nn.Linear:
        """Create a complex-valued linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            
        Returns:
            Complex linear layer
        """
        layer = nn.Linear(
            in_features,
            out_features,
            dtype=self.config.dtype
        )
        
        # Convert weights to complex
        with torch.no_grad():
            layer.weight.data = layer.weight.data.to(self.config.dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(self.config.dtype)
                
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
        def _init_complex_weight(weight: torch.Tensor) -> None:
            real_weight = torch.empty_like(weight.real)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(weight.imag)
            nn.init.orthogonal_(imag_weight)
            weight.copy_(torch.complex(real_weight, imag_weight))

        def _init_complex_bias(bias: torch.Tensor) -> None:
            if bias is not None:
                bias.zero_()

        # Initialize all complex weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_complex_weight(module.weight.data)
                _init_complex_bias(module.bias)

    def _prepare_quantum_state(
        self,
        x: torch.Tensor,
        return_validation: bool = False
    ) -> Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]]:
        """Prepare quantum state efficiently with validation.
        
        Args:
            x: Input tensor to convert to quantum state
            return_validation: Whether to return validation result
            
        Returns:
            Prepared quantum state and optional validation result
            
        Raises:
            InvalidQuantumStateError: If state preparation fails
        """
        try:
            # Convert to quantum state with minimal copies
            if not isinstance(x, QuantumState):
                if not torch.is_complex(x):
                    x = x.to(self.config.dtype)
                
                # Normalize state vector efficiently
                norm = torch.norm(x, dim=-1, keepdim=True)
                if torch.any(norm == 0):
                    if return_validation:
                        return QuantumState(
                            amplitudes=torch.zeros_like(x),
                            basis_labels=[str(i) for i in range(self.manifold_dim)],
                            phase=torch.zeros(1, dtype=self.config.dtype, device=self.config.device)
                        ), QuantumStateValidationResult(
                            is_valid=False,
                            message="Zero norm state detected",
                            error_type=StateValidationErrorType.INVALID_NORM,
                            data={"norm": 0.0}
                        )
                    raise InvalidQuantumStateError("Zero norm state detected")
                x = x.div_(norm)  # In-place division
                
                state = QuantumState(
                    amplitudes=x,
                    basis_labels=[str(i) for i in range(self.manifold_dim)],
                    phase=torch.zeros(1, dtype=self.config.dtype, device=self.config.device)
                )
            else:
                state = x
            
            # Validate and correct state efficiently
            validation_result = self.preparation_validator.validate_preparation(
                target=state,
                prepared=state
            )
            
            if not validation_result.is_valid:
                error_type = validation_result.error_type or StateValidationErrorType.INVALID_NORM
                error_data = validation_result.data or {}
                
                # Apply corrections in-place where possible
                corrected_state = None
                if error_type == StateValidationErrorType.INVALID_NORM:
                    corrected_state = self._correct_norm(state)
                elif error_type == StateValidationErrorType.INVALID_PHASE:
                    corrected_state = self._correct_phase(state)
                elif error_type == StateValidationErrorType.INVALID_DIMENSIONS:
                    corrected_state = self._correct_basis(state)
                
                if corrected_state is None:
                    if return_validation:
                        return state, QuantumStateValidationResult(
                            is_valid=False,
                            message=f"Uncorrectable state error: {error_type}",
                            error_type=error_type,
                            data=error_data
                        )
                    raise InvalidQuantumStateError(f"Uncorrectable state error: {error_type}")
                
                # Final validation
                final_validation = self.preparation_validator.validate_preparation(
                    target=corrected_state,
                    prepared=corrected_state
                )
                
                if not final_validation.is_valid:
                    if return_validation:
                        return state, QuantumStateValidationResult(
                            is_valid=False,
                            message="State correction failed",
                            error_type=error_type,
                            data={
                                **error_data,
                                "final_validation": final_validation.to_dict()
                            }
                        )
                    raise InvalidQuantumStateError("State correction failed")
                
                state = corrected_state
                # Update validation result with correction info
                validation_result = QuantumStateValidationResult(
                    is_valid=True,
                    message="State corrected successfully",
                    data={
                        **error_data,
                        "corrections_applied": {
                            "error_type": error_type.value,
                            "original_validation": validation_result.to_dict(),
                            "final_validation": final_validation.to_dict()
                        }
                    }
                )
            
            if return_validation:
                return state, validation_result
                
            return state
            
        except Exception as e:
            if return_validation:
                return QuantumState(
                    amplitudes=torch.zeros_like(x),
                    basis_labels=[str(i) for i in range(self.manifold_dim)],
                    phase=torch.zeros(1, dtype=self.config.dtype, device=self.config.device)
                ), QuantumStateValidationResult(
                    is_valid=False,
                    message=str(e),
                    error_type=StateValidationErrorType.INVALID_NORM,
                    data={"error": str(e)}
                )
            raise InvalidQuantumStateError(f"Failed to prepare quantum state: {str(e)}")

    def _correct_norm(self, state: QuantumState) -> QuantumState:
        """Correct state normalization."""
        amplitudes = state.amplitudes
        norm = torch.norm(amplitudes, dim=-1, keepdim=True)
        return QuantumState(
            amplitudes=amplitudes / (norm + 1e-10),
            basis_labels=state.basis_labels,
            phase=state.phase
        )

    def _correct_phase(self, state: QuantumState) -> QuantumState:
        """Correct state phase."""
        amplitudes = state.amplitudes
        phase = torch.angle(amplitudes[..., 0:1])
        return QuantumState(
            amplitudes=amplitudes * torch.exp(-1j * phase),
            basis_labels=state.basis_labels,
            phase=torch.zeros_like(state.phase)
        )

    def _correct_basis(self, state: QuantumState) -> QuantumState:
        """Correct state basis."""
        return QuantumState(
            amplitudes=state.amplitudes,
            basis_labels=[str(i) for i in range(self.manifold_dim)],
            phase=state.phase
        )

    def _compute_density_matrix(self, state: QuantumState) -> torch.Tensor:
        """Compute density matrix with optimized memory usage."""
        try:
            # Validate state
            properties = self.state_validator.validate_state(state)
            if not properties.is_normalized:
                raise InvalidQuantumStateError("Invalid state for density matrix computation")
                
            # Get state vector and compute efficiently
            state_vector = state.amplitudes
            batch_dims = state_vector.shape[:-1]
            state_dim = state_vector.shape[-1]
            
            # Compute density matrix with minimal intermediate storage
            state_vector = state_vector.reshape(-1, state_dim)
            density_matrices = torch.bmm(
                state_vector.unsqueeze(-1),
                state_vector.conj().unsqueeze(-2)
            )
            
            return density_matrices.reshape(*batch_dims, state_dim, state_dim)
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to compute density matrix: {str(e)}")

    def _measure_quantum_state(self, state: QuantumState) -> torch.Tensor:
        """Measure quantum state and project to classical representation.
        
        Args:
            state: Quantum state to measure
            
        Returns:
            Classical tensor representation
            
        Raises:
            InvalidQuantumStateError: If state is invalid
        """
        try:
            # Validate state before measurement
            properties = self.state_validator.validate_state(state)
            
            # Project state to classical representation
            if properties.is_pure:
                # For pure states, use direct projection
                classical = state.amplitudes
            else:
                # For mixed states, use density matrix eigendecomposition
                density_matrix = self._compute_density_matrix(state)
                eigenvals, eigenvecs = torch.linalg.eigh(density_matrix)
                # Use dominant eigenvector
                classical = eigenvecs[..., -1]
                
            return classical.real
            
        except Exception as e:
            raise InvalidQuantumStateError(f"Failed to measure quantum state: {str(e)}")

    def _geometric_update(self, x: torch.Tensor) -> torch.Tensor:
        """Updates based on manifold structure using geometric flow.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Updated tensor with same shape as input
            
        Raises:
            GeometricFlowError: If flow computation fails
        """
        try:
            # Project to manifold space if needed
            if x.shape[-1] != self.flow.manifold_dim:
                x_manifold = x[..., :self.flow.manifold_dim]
            else:
                x_manifold = x
            
            # Apply geometric flow
            x_flowed, _ = self.flow(
                x_manifold,
                return_path=False
            )
            
            return x_flowed
            
        except Exception as e:
            raise GeometricFlowError(f"Failed to compute geometric flow: {str(e)}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply quantum geometric attention with pattern validation.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            return_metrics: Whether to return metrics
            
        Returns:
            Output tensor and optional metrics
        """
        try:
            # Initialize state
            state = self.prepare_attention_state(x, mask)
            metrics_dict = {} if return_metrics else {}
            
            # Apply attention flow
            output = self._apply_attention_flow(state)
            
            if return_metrics:
                # Validate pattern formation
                pattern_result = self.validate_pattern_formation(output)
                metrics_dict['pattern_formation'] = pattern_result.data or {}
                
                return output, metrics_dict
                
            return output
            
        except Exception as e:
            raise RuntimeError(f"Error in quantum geometric attention: {str(e)}")

    def prepare_attention_state(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> AttentionState:
        """Prepare attention state with optional mask."""
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=self.config.dtype)
        
        # Initialize state with proper dimensions from input
        batch_size, seq_len, head_dim = x.shape
        
        # Initialize AttentionState
        state = AttentionState.initialize(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            device=x.device,
            dtype=self.config.dtype
        )
        
        # Store input as quantum state, properly shaped for multi-head attention
        quantum_state = x.view(batch_size, seq_len, 1, head_dim).expand(-1, -1, self.num_heads, -1)
        quantum_state = quantum_state.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Store quantum state in state manager
        state.state_manager.states["quantum"] = quantum_state
        
        # Initialize geometric state to match quantum state dimensions
        state.geometric_state = quantum_state.clone()
        state.manifold_state = quantum_state.clone()
        
        return state

    def apply_mask(self, state: AttentionState, mask: torch.Tensor) -> AttentionState:
        """Apply attention mask to state.
        
        Args:
            state: Current attention state
            mask: Boolean mask tensor [batch_size, seq_len]
            
        Returns:
            Masked attention state
        """
        if mask is None:
            return state
            
        # Get attention scores from state
        attention_scores = state.attention_scores
        if attention_scores is None:
            # Initialize attention scores if they don't exist
            batch_size = mask.size(0)
            seq_len = mask.size(1)
            attention_scores = torch.zeros(
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
                device=mask.device,
                dtype=self.config.dtype
            )
            state.attention_scores = attention_scores
            
        # Create mask for attention scores
        # Expand mask to match attention scores shape [batch_size, num_heads, seq_len, seq_len]
        expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand(
            -1,
            self.num_heads,
            -1,
            mask.size(1)
        )
        
        # Apply mask by setting masked positions to -inf
        masked_scores = attention_scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Update state with masked scores
        state.attention_scores = masked_scores
        
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
        batch_size, num_heads, seq_len, head_dim = query.shape
        metrics = {} if return_metrics else None

        # Project to manifold space efficiently
        query = query.to(self.config.dtype)
        key = key.to(self.config.dtype)
        
        # Reshape and project in one operation
        query = self.manifold_proj(query.reshape(-1, head_dim)).view(batch_size, num_heads, seq_len, -1)
        key = self.manifold_proj(key.reshape(-1, head_dim)).view(batch_size, num_heads, seq_len, -1)

        # Compute attention scores efficiently
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Convert to real and apply softmax
        attention_weights = torch.softmax(scores.abs(), dim=-1)

        # Process value tensor if provided
        attention_output: torch.Tensor
        if value is not None:
            processed_value = value.to(self.config.dtype)
            processed_value = self.manifold_proj(processed_value.reshape(-1, head_dim)).view(batch_size, num_heads, seq_len, -1)
            attention_output = torch.matmul(attention_weights, processed_value)
        else:
            attention_output = attention_weights

        # Return metrics if requested
        if return_metrics:
            metrics = {
                'attention_weights': attention_weights,
                'attention_scores': scores,
                'query_proj': query,
                'key_proj': key,
                'value_proj': processed_value if 'processed_value' in locals() else None
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
                phase=state.phase
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
        """Apply geometric attention flow with enhanced error handling."""
        try:
            # Initialize state with minimal copies
            state = self.prepare_attention_state(x, mask)
            metrics_dict: Dict[str, Any] = {} if return_metrics else {}
            
            # Pre-compute initial metric with validation
            try:
                metric = self._compute_metric_tensor(state)
            except MetricError as e:
                raise GeometricFlowError(f"Failed to compute initial metric: {str(e)}")
            
            current = x
            
            # Store flow path for metrics if needed
            flow_path: List[torch.Tensor] = [current.clone()] if return_metrics else []
            
            # Perform flow steps efficiently
            for step in range(num_steps):
                try:
                    # Compute Ricci tensor using Riemannian structure
                    ricci = self.riemannian.compute_riemann(current).ricci
                    
                    # Compute symplectic flow
                    symplectic_flow = self.symplectic.quantum_ricci_flow(
                        current,
                        time=dt,
                        dt=dt/10,
                        steps=10
                    )
                    
                    # Validate flow before combining
                    if torch.any(torch.isnan(symplectic_flow)):
                        raise GeometricFlowError(f"NaN detected in symplectic flow at step {step}")
                    
                    # Combine Ricci flow and symplectic flow
                    metric.add_(dt * ricci)
                    current = 0.7 * symplectic_flow + 0.3 * current
                    
                    # Validate current state
                    if torch.any(torch.isnan(current)):
                        raise GeometricFlowError(f"NaN detected in flow state at step {step}")
                    
                    # Project and flow efficiently
                    current = self.manifold_proj(current.view(-1, self.head_dim))
                    current = self.manifold_proj_inv(current).view(*x.shape)
                    
                    # Store flow path and compute metrics if needed
                    if return_metrics:
                        flow_path.append(current.clone())
                        flow_tensor = torch.stack(flow_path, dim=1)
                        
                        # Use unified metrics for all computations
                        # Convert flow path to tensor for metrics
                        flow_path_tensor = torch.stack(flow_path, dim=1)
                        metrics_data: Dict[str, torch.Tensor] = {
                            "attention_patterns": current,
                            "patterns": current,
                            "flow_path": flow_tensor,
                            "pattern_history": flow_path_tensor
                        }
                        
                        try:
                            step_metrics = self.unified_metrics.compute_all_metrics(
                                metrics_data,
                                self.metric_context
                            )
                            metrics_dict[f'step_{step}'] = step_metrics
                        except Exception as e:
                            # Log metric computation failure but continue flow
                            metrics_dict[f'step_{step}_error'] = str(e)
                            
                except Exception as e:
                    if isinstance(e, (GeometricFlowError, InvalidQuantumStateError)):
                        raise
                    raise GeometricFlowError(f"Error at flow step {step}: {str(e)}")
            
            if return_metrics:
                return current, metrics_dict
            
            return current
            
        except Exception as e:
            if isinstance(e, (InvalidQuantumStateError, GeometricFlowError)):
                raise
            raise RuntimeError(f"Error in geometric attention flow: {str(e)}")

    def _compute_metric_tensor(self, state: Union[AttentionState, torch.Tensor]) -> torch.Tensor:
        """Compute metric tensor for attention manifold with enhanced validation.
        
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
                
            # Project to quantum features
            quantum_features = self._compute_quantum_features(state_tensor)
            
            # Compute quantum geometric tensor using symplectic structure
            quantum_metric = self.symplectic.compute_quantum_geometric_tensor(quantum_features)
            
            # Compute Riemannian metric for pattern structure
            pattern_metric = self.riemannian.compute_metric(quantum_features).values
            
            # Validate metric properties before combining
            quantum_properties = self._validate_metric_properties(quantum_metric, "quantum")
            pattern_properties = self._validate_metric_properties(pattern_metric, "pattern")
            
            if not (quantum_properties.is_positive_definite and pattern_properties.is_positive_definite):
                raise MetricError(
                    "Invalid metric properties detected:\n"
                    f"Quantum metric positive definite: {quantum_properties.is_positive_definite}\n"
                    f"Pattern metric positive definite: {pattern_properties.is_positive_definite}"
                )
            
            # Combine metrics with appropriate weights
            combined_metric = 0.7 * quantum_metric + 0.3 * pattern_metric
            
            # Validate combined metric
            combined_properties = self._validate_metric_properties(combined_metric, "combined")
            if not combined_properties.is_positive_definite:
                raise MetricError("Combined metric failed positive definiteness check")
            
            return combined_metric
            
        except Exception as e:
            if isinstance(e, MetricError):
                raise
            raise MetricError(f"Failed to compute metric tensor: {str(e)}")

    def _validate_metric_properties(
        self, 
        metric: torch.Tensor,
        metric_type: str
    ) -> MetricProperties:
        """Validate metric tensor properties with detailed diagnostics.
        
        Args:
            metric: Metric tensor to validate
            metric_type: Type of metric for error reporting
            
        Returns:
            MetricProperties object containing validation results
        """
        try:
            # Check basic tensor properties
            if not torch.is_tensor(metric):
                raise ValueError(f"{metric_type} metric must be a tensor")
            
            if metric.dim() < 2:
                raise ValueError(f"{metric_type} metric must be at least 2-dimensional")
            
            # Compute eigenvalues for positive definiteness check
            try:
                eigenvals = torch.linalg.eigvalsh(metric)
            except Exception as e:
                raise MetricError(f"Failed to compute eigenvalues for {metric_type} metric: {str(e)}")
            
            is_positive_definite = bool(torch.all(eigenvals > -1e-6).item())
            
            # Compute condition number if possible
            try:
                condition_number = float(torch.max(eigenvals) / torch.min(eigenvals.abs()))
            except Exception:
                condition_number = None
            
            # Compute determinant and trace
            try:
                determinant = torch.linalg.det(metric)
                trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)
            except Exception:
                determinant = None
                trace = None
            
            # Check compatibility with Riemannian structure
            try:
                is_compatible = bool(torch.allclose(
                    metric,
                    metric.transpose(-1, -2).conj(),
                    rtol=1e-5
                ))
            except Exception:
                is_compatible = False
            
            # Check completeness (simplified)
            try:
                is_complete = bool(torch.all(torch.abs(determinant) > 1e-6).item()) if determinant is not None else False
            except Exception:
                is_complete = False
            
            # Compute curvature properties if possible
            try:
                riemann = self.riemannian.compute_riemann(metric)
                # Extract curvature components from Riemann tensor
                sectional_curvature = riemann.riemann  # Access the Riemann tensor component
                # Compute Ricci tensor by contracting first and last indices
                ricci_curvature = torch.einsum('...ijkj->...ik', sectional_curvature)
                # Compute scalar curvature by taking trace of Ricci tensor
                scalar_curvature = torch.einsum('...ii', ricci_curvature)
            except Exception:
                sectional_curvature = None
                ricci_curvature = None
                scalar_curvature = None
            
            # Check if curvature is bounded
            has_bounded_curvature = False
            if sectional_curvature is not None:
                try:
                    has_bounded_curvature = bool(torch.all(torch.abs(sectional_curvature) < 1e6).item())
                except Exception:
                    pass
                
            return MetricProperties(
                is_positive_definite=is_positive_definite,
                is_compatible=is_compatible,
                is_complete=is_complete,
                has_bounded_curvature=has_bounded_curvature,
                determinant=determinant,
                trace=trace,
                eigenvalues=eigenvals,
                condition_number=condition_number,
                sectional_curvature=sectional_curvature,
                ricci_curvature=ricci_curvature,
                scalar_curvature=scalar_curvature
            )
            
        except Exception as e:
            if isinstance(e, MetricError):
                raise
            raise MetricError(f"Failed to validate {metric_type} metric properties: {str(e)}")

    def _compute_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum features efficiently with minimal memory usage."""
        if x.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D")
        
        # Reshape input with minimal copies
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        
        # Project to manifold space
        if x.shape[-1] == self.hidden_dim:
            # Reshape for head processing
            x = x.view(-1, self.num_heads, self.head_dim)
            x = x.reshape(-1, self.head_dim)
        
        # Project efficiently
        x = self.manifold_proj(x)
        
        # Process through tiles with minimal intermediate storage
        for tile in self.tiles:
            x = tile(x)
            
        # Restore original shape if needed
        if len(orig_shape) == 3:
            x = x.view(orig_shape[0], orig_shape[1], -1)
            
        return x

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

    def _compute_quantum_metrics(self, state: QuantumState) -> Dict[str, float]:
        """Compute quantum metrics for a given quantum state.
        
        Args:
            state: The quantum state to compute metrics for
            
        Returns:
            Dictionary of quantum metrics
        """
        try:
            # Initialize default metrics
            metrics = {
                'entropy': 0.0,
                'purity': 1.0,
                'fisher': 0.0,
                'transport_deviation': 0.0
            }
            
            if not isinstance(state, QuantumState):
                return metrics
                
            # Get state vector and compute density matrix
            state_vector = state.amplitudes
            density_matrix = torch.einsum('...i,...j->...ij', state_vector, state_vector.conj())
            
            # Compute von Neumann entropy
            eigenvals = torch.linalg.eigvalsh(density_matrix)
            eigenvals = torch.clamp(eigenvals, min=1e-10)
            eigenvals = eigenvals / torch.sum(eigenvals)
            metrics['entropy'] = float(-torch.sum(eigenvals * torch.log(eigenvals)).item())
            
            # Compute purity
            purity = torch.einsum('...ij,...ji->...', density_matrix, density_matrix)
            metrics['purity'] = float(purity.mean().item())
            
            # Compute quantum Fisher information
            # Use diagonal approximation for efficiency
            fisher = torch.abs(torch.diagonal(density_matrix, dim1=-2, dim2=-1))
            metrics['fisher'] = float(fisher.mean().item())
            
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
                metrics['transport_deviation'] = float(
                    torch.norm(transported - tangent).mean().item()
                )
                
            # Store current state for next computation
            self._prev_state = state
            
            return metrics
        except Exception as e:
            # Keep default metrics in case of error
            return {
                'entropy': 0.0,
                'purity': 1.0,
                'fisher': 0.0,
                'transport_deviation': 0.0
            }

    def _compute_quantum_state_metrics(self, state: torch.Tensor) -> Dict[str, float]:
        """Compute quantum metrics for raw tensor state.
        
        Args:
            state: Tensor to analyze [batch_size, seq_len, hidden_dim]
            
        Returns:
            Dictionary of quantum metrics
        """
        # Convert to quantum state
        quantum_state = self._prepare_quantum_state(state)
        if isinstance(quantum_state, tuple):
            quantum_state = quantum_state[0]  # Extract state from validation tuple
        
        # Compute metrics using QuantumMetrics
        return self._compute_quantum_metrics(quantum_state)

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
    