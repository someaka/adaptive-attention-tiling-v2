"""Neural Quantum Bridge Implementation.

This module implements the bridge between neural and quantum states,
providing clean state management and validation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, cast, TypeVar, assert_type
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from ..patterns.fiber_types import LocalChart
from ..patterns.fiber_bundle import BaseFiberBundle
from ..patterns.riemannian_base import RiemannianStructure
from ..patterns.symplectic import SymplecticStructure
from ..patterns.arithmetic_dynamics import ArithmeticPattern
from ..patterns.fiber_types import LocalChart as PatternSection
from .types import QuantumState
from .state_space import HilbertSpace
from src.core.crystal.scale import ScaleSystem
from ..tiling.state_manager import StateManager, StateConfig, StateType
from ..tiling.quantum_attention_tile import QuantumMotivicTile
from ..tiling.patterns.pattern_fiber_bundle import PatternFiberBundle
from ..patterns.cohomology import (
    MotivicCohomology,
    QuantumMotivicCohomology,
    ArithmeticForm,
    RiemannianFiberBundle
)
from ...validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    QuantumStateValidationResult,
    StateValidationErrorType
)


class NeuralQuantumBridge(nn.Module):
    """Bridge between neural and quantum representations."""

    def __init__(
        self,
        hidden_dim: int,
        manifold_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        manifold_type: str = "hyperbolic",
        curvature: float = -1.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
    ):
        """Initialize neural quantum bridge.
        
        Args:
            hidden_dim: Hidden dimension
            manifold_dim: Manifold dimension (defaults to hidden_dim // 2)
            num_heads: Number of attention heads
            dropout: Dropout probability
            manifold_type: Type of manifold geometry
            curvature: Manifold curvature
            dtype: Data type for computation
            device: Device to use (defaults to CPU)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim or hidden_dim // 2
        self.num_heads = num_heads
        self.dropout = dropout
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        
        # Initialize layer normalization for real and imaginary parts
        self.layer_norm_real = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64,  # Use float64 for real part
            device=self.device
        )
        self.layer_norm_imag = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64,  # Use float64 for imaginary part
            device=self.device
        )
        
        # Initialize manifold normalization for real and imaginary parts
        self.manifold_norm_real = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64,  # Use float64 for real part
            device=self.device
        )
        self.manifold_norm_imag = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64,  # Use float64 for imaginary part
            device=self.device
        )
        
        # Initialize state preparation and validation
        self.state_preparation = StatePreparationValidator()
        self.state_validator = StateValidator()
        
        # Initialize state manager with device
        self.state_manager = StateManager(
            config=StateConfig(
                dim=hidden_dim,
                type=StateType.PURE,
                epsilon=1e-6,
                max_entanglement=1.0,
                dtype=dtype
            )
        )
        self.state_manager.device = self.device
        
        # Initialize Hilbert space
        self.hilbert_space = HilbertSpace(dim=self.manifold_dim, dtype=dtype)
        
        # Initialize pattern bundle
        self.pattern_bundle = PatternFiberBundle(
            base_dim=hidden_dim,
            fiber_dim=hidden_dim,
            structure_group="O(n)",
            motive_rank=4,
            num_primes=8,
            device=self.device,
            dtype=dtype
        )
        
        # Initialize inverse projection with double input dimension for real and imaginary parts
        self.inverse_projection = nn.Linear(
            2 * hidden_dim,  # Input is 2 * hidden_dim for concatenated real and imaginary parts
            hidden_dim,  # Output is hidden_dim to match neural representation
            device=self.device,
            dtype=self.dtype
        )

        # Initialize state preparation validator
        self.state_validator = StatePreparationValidator(tolerance=1e-6)
        
        # Initialize state validator
        self.validator = StateValidator(tolerance=1e-6)

    def neural_to_quantum(
        self,
        x: torch.Tensor,
        return_validation: bool = False
    ) -> Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]]:
        """Convert neural representation to quantum state.
        
        Args:
            x: Neural representation tensor
            return_validation: Whether to return validation result
            
        Returns:
            Quantum state or tuple of (quantum state, validation result)
        """
        # Validate input dimensions
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"Input tensor must have hidden dimension {self.hidden_dim}, got {x.shape[-1]}")
        
        # Check for non-finite values
        if not torch.all(torch.isfinite(x)):
            raise ValueError("Input tensor contains non-finite values")
            
        # Check for zero-norm patterns
        if torch.all(x == 0):
            raise ValueError("Input tensor has zero norm")
        
        # Store original shape and norm
        original_shape = x.shape
        original_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        
        # Get dimensions
        if len(x.shape) == 2:  # [batch_size, hidden_dim]
            batch_size = x.shape[0]
            seq_len = 1
            num_heads = 1
            original_norm = original_norm.reshape(batch_size, 1)
        elif len(x.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            num_heads = 1
            original_norm = original_norm.reshape(batch_size, seq_len, 1)
        else:  # [batch_size, num_heads, seq_len, hidden_dim]
            batch_size = x.shape[0]
            num_heads = x.shape[1]
            seq_len = x.shape[2]
            original_norm = original_norm.reshape(batch_size, num_heads, seq_len, 1)
        
        # Flatten input for processing
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Project to manifold dimension while preserving gradients
        x_manifold = x_flat.clone()
        x_manifold.requires_grad_(True)  # Enable gradients
        
        # Get hidden dimension from class attribute
        hidden_dim = self.hidden_dim // 2  # Half dimension for real/imag split
            
        x_manifold.retain_grad()  # Retain gradients for intermediate values
        
        # Convert to complex while preserving sign information
        if torch.is_complex(x_manifold):
            # If input is already complex, use it directly
            x_complex = x_manifold
            x_phase = torch.angle(x_complex)
        else:
            # For real input, convert to complex while preserving sign
            x_abs = torch.abs(x_manifold)
            x_sign = torch.sign(x_manifold)
            x_phase = torch.where(x_sign < 0, torch.tensor(np.pi, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
            x_real = x_abs * torch.cos(x_phase)
            x_imag = x_abs * torch.sin(x_phase)
            x_complex = torch.complex(x_real, x_imag)
        
        # Reshape to [batch_size, num_heads, seq_len, hidden_dim] for per-head normalization
        x_complex = x_complex.reshape(batch_size, num_heads, seq_len, -1)
        
        # Normalize per head
        norms = torch.sqrt(torch.sum(torch.abs(x_complex) ** 2, dim=(-2, -1), keepdim=True))
        x_complex = x_complex / (norms + 1e-8)
        
        # Create quantum state with proper initialization and gradient tracking
        state = QuantumState(
            amplitudes=x_complex,
            basis_labels=[str(i) for i in range(hidden_dim)],
            phase=x_phase,
            original_norm=original_norm,
            layout={
                "type": "batch" if num_heads == 1 and seq_len == 1 else "sequence" if num_heads == 1 else "attention",
                "batch_size": batch_size,
                "num_heads": num_heads,
                "seq_length": seq_len,
                "dim": hidden_dim
            }
        )
        
        # Ensure state amplitudes have gradients
        state.amplitudes.requires_grad_(True)
        state.amplitudes.retain_grad()
        
        if return_validation:
            # Validate state preparation
            validation = self.state_preparation.validate_preparation(
                target=state,
                prepared=state
            )
            return (state, validation)
        
        return state

    def quantum_to_neural(
        self,
        state: QuantumState,
        original_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """Convert quantum state to neural representation.
        
        Args:
            state: Quantum state to convert
            original_shape: Optional shape of input tensor
            
        Returns:
            Neural representation tensor
        """
        # Get classical amplitudes and preserve both real and imaginary parts
        classical_flat = state.amplitudes
        
        # Reshape for projection if needed
        if len(classical_flat.shape) == 4:  # [batch_size, num_heads, seq_len, hidden_dim]
            classical_flat = classical_flat.reshape(-1, classical_flat.shape[-1])
        
        # Extract real and imaginary parts
        real_part = classical_flat.real
        imag_part = classical_flat.imag
        
        # Reconstruct complex output
        output = torch.complex(real_part, imag_part)
        output = output.to(torch.complex128)  # Ensure consistent dtype
        
        # Reshape to match original dimensions
        if original_shape is not None:
            # Ensure output has the same size as original_shape
            if output.numel() != np.prod(original_shape):
                # If sizes don't match, we need to project back to original dimension
                output = F.linear(output, self.inverse_projection.weight[:, :output.shape[-1]], self.inverse_projection.bias)
            output = output.reshape(original_shape)
        else:
            # Infer dimensions from layout if available
            if hasattr(state, 'layout') and state.layout is not None:
                layout = state.layout
                batch_size = layout.get('batch_size', 1)
                seq_len = layout.get('seq_length', 1)
                num_heads = layout.get('num_heads', 1)
                hidden_dim = self.hidden_dim
                
                # Project back to original dimension if needed
                if output.shape[-1] != hidden_dim:
                    output = F.linear(output, self.inverse_projection.weight[:, :output.shape[-1]], self.inverse_projection.bias)
                
                if layout.get('type') == 'batch':
                    output = output.reshape(batch_size, hidden_dim)
                elif layout.get('type') == 'sequence':
                    output = output.reshape(batch_size, seq_len, hidden_dim)
                else:  # attention
                    output = output.reshape(batch_size, num_heads, seq_len, hidden_dim)
            else:
                # Fallback to shape-based inference
                if len(classical_flat.shape) == 2:  # [batch_size, manifold_dim]
                    output = output.reshape(classical_flat.shape[0], -1)
                elif len(classical_flat.shape) == 3:  # [batch_size, seq_len, manifold_dim]
                    output = output.reshape(classical_flat.shape[0], classical_flat.shape[1], -1)
                else:  # [batch_size, num_heads, seq_len, manifold_dim]
                    output = output.reshape(classical_flat.shape[0], classical_flat.shape[1], classical_flat.shape[2], -1)
        
        # If original norm is stored in quantum state, restore it after projection
        if hasattr(state, 'original_norm') and state.original_norm is not None:
            current_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
            # Ensure original_norm has the same shape as current_norm
            original_norm = state.original_norm.to(self.dtype)
            
            # Expand original_norm to match current_norm shape if we have layout info
            if hasattr(state, 'layout') and state.layout is not None:
                if len(current_norm.shape) == 4:  # [batch_size, num_heads, seq_len, 1]
                    original_norm = original_norm.reshape(
                        state.layout.get('batch_size', 1),
                        state.layout.get('num_heads', 1),
                        state.layout.get('seq_length', 1),
                        1
                    )
                elif len(current_norm.shape) == 3:  # [batch_size, seq_len, 1]
                    original_norm = original_norm.reshape(
                        state.layout.get('batch_size', 1),
                        state.layout.get('seq_length', 1),
                        1
                    )
                elif len(current_norm.shape) == 2:  # [batch_size, 1]
                    original_norm = original_norm.reshape(
                        state.layout.get('batch_size', 1),
                        1
                    )
            else:
                # Fallback to direct reshape if no layout info
                original_norm = original_norm.reshape(current_norm.shape)
                
            output = output * (original_norm / (current_norm + 1e-8))
        
        return output

    def evolve_quantum_state_with_attention(
        self,
        state: QuantumState,
        attention_pattern: Optional[torch.Tensor] = None,
        time: float = 1.0
    ) -> QuantumState:
        """Evolve quantum state using attention pattern.
    
        Args:
            state: Input quantum state
            attention_pattern: Optional attention pattern tensor
            time: Evolution time
    
        Returns:
            Evolved quantum state
        """
        if state is None:
            raise ValueError("Quantum state cannot be None")
    
        # Get state dimensions
        batch_size = state.amplitudes.shape[0]
        num_heads = state.amplitudes.shape[1] if len(state.amplitudes.shape) > 3 else 1
        has_seq_dim = len(state.amplitudes.shape) > 2
        seq_len = state.amplitudes.shape[-2] if has_seq_dim else 1
        state_dim = state.amplitudes.shape[-1]
    
        # Ensure consistent dtype throughout evolution
        target_dtype = state.amplitudes.dtype
        working_dtype = torch.complex128  # Use complex128 for internal computations
    
        # Reshape amplitudes to [batch_size * num_heads, seq_len, state_dim]
        if has_seq_dim:
            amplitudes_reshaped = state.amplitudes.reshape(batch_size * num_heads, seq_len, state_dim)
        else:
            # Add sequence dimension if not present
            amplitudes_reshaped = state.amplitudes.reshape(batch_size * num_heads, 1, state_dim)
    
        # Create or use provided attention pattern
        if attention_pattern is None:
            # Create identity attention pattern efficiently
            attention_pattern = torch.eye(
                state_dim,
                device=state.amplitudes.device,
                dtype=working_dtype
            ).unsqueeze(0).expand(batch_size * num_heads, seq_len, state_dim, state_dim)
        else:
            # Convert attention pattern to working dtype
            attention_pattern = attention_pattern.to(working_dtype)
    
            # Reshape attention pattern to match state dimensions
            if len(attention_pattern.shape) == 4:  # [batch, heads, seq, seq]
                # Reshape to [batch * heads, seq, seq]
                attention_pattern = attention_pattern.reshape(batch_size * num_heads, seq_len, seq_len)
            elif len(attention_pattern.shape) == 3:  # [batch, seq, seq]
                # Check if this is a [batch, manifold_dim, manifold_dim] attention pattern
                if attention_pattern.shape[1] == attention_pattern.shape[2]:
                    # Project attention pattern to state dimension if needed
                    if attention_pattern.shape[1] != state_dim:
                        # Create a projection matrix from manifold_dim to state_dim
                        projection = torch.zeros(
                            attention_pattern.shape[1],
                            state_dim,
                            device=attention_pattern.device,
                            dtype=working_dtype
                        )
                        min_size = min(attention_pattern.shape[1], state_dim)
                        projection[:min_size, :min_size] = torch.eye(min_size, device=attention_pattern.device, dtype=working_dtype)
                        
                        # Project the attention pattern
                        attention_pattern = torch.matmul(
                            torch.matmul(projection.t(), attention_pattern),
                            projection
                        )
                    
                    # Add sequence dimension and expand for heads
                    attention_pattern = attention_pattern.unsqueeze(1)  # [batch, 1, state_dim, state_dim]
                    if num_heads > 1:
                        attention_pattern = attention_pattern.expand(-1, num_heads, -1, -1)  # [batch, heads, state_dim, state_dim]
                    attention_pattern = attention_pattern.reshape(batch_size * num_heads, 1, state_dim, state_dim)
                else:
                    # This is a [batch, seq, seq] attention pattern
                    if num_heads > 1:
                        # For multi-head state with single-head attention, expand to match batch_size * num_heads
                        attention_pattern = attention_pattern.unsqueeze(1)  # [batch, 1, seq, seq]
                        attention_pattern = attention_pattern.expand(-1, num_heads, -1, -1)  # [batch, heads, seq, seq]
                        attention_pattern = attention_pattern.reshape(batch_size * num_heads, seq_len, seq_len)
                    else:
                        # For single-head state with single-head attention, just reshape
                        attention_pattern = attention_pattern.reshape(batch_size * num_heads, seq_len, seq_len)
    
            # Pad or truncate attention pattern if needed
            if attention_pattern.shape[-1] != state_dim:
                padded_attention = torch.zeros(
                    batch_size * num_heads,
                    seq_len,
                    state_dim,
                    state_dim,
                    device=attention_pattern.device,
                    dtype=working_dtype
                )
                min_size = min(seq_len, state_dim)
                # For sequence-to-sequence attention, expand to state_dim x state_dim
                # First create a block diagonal matrix
                for i in range(seq_len):
                    for j in range(seq_len):
                        # Extract the attention weight for this position and broadcast
                        weight = attention_pattern[:, i, j].unsqueeze(-1).unsqueeze(-1)
                        # Create a diagonal matrix for this position
                        diag_matrix = torch.eye(min_size, device=attention_pattern.device, dtype=working_dtype)
                        # Multiply with broadcasted weight
                        diag_matrix = diag_matrix * weight
                        padded_attention[:, i, :min_size, :min_size] = diag_matrix
                attention_pattern = padded_attention
        
        # Add head-specific perturbations to metric and connection
        head_indices = torch.arange(num_heads, device=state.amplitudes.device).repeat_interleave(batch_size)
        head_scale = (1.0 + 0.1 * head_indices).reshape(-1, 1, 1)  # Reduced scale factor from 0.25 to 0.1
        
        # Scale attention pattern per head
        attention_pattern = attention_pattern * head_scale.unsqueeze(-1)  # Scale each head's attention pattern differently
        
        # Construct Hamiltonian from attention pattern and pattern bundle metric
        attention_regularized = attention_pattern + torch.eye(
            state_dim,
            device=attention_pattern.device,
            dtype=working_dtype
        ).unsqueeze(0).unsqueeze(0) * 1e-6
    
        # Use pattern bundle metric and connection to modify Hamiltonian
        metric = self.pattern_bundle.metric.to(working_dtype)
        connection = self.pattern_bundle.connection.to(working_dtype)
    
        # Extract base metric and connection parts
        base_metric = metric[:state_dim, :state_dim]
        base_connection = connection[:state_dim, :state_dim, :state_dim]

        # Add head-specific perturbations to metric and connection
        head_indices = torch.arange(num_heads, device=state.amplitudes.device).repeat_interleave(batch_size)
        head_scale = (1.0 + 0.1 * head_indices).reshape(-1, 1, 1)  # Reduced scale factor from 0.25 to 0.1
        
        # Expand base metric and connection for batch processing with head-specific scaling
        base_metric = base_metric.reshape(1, 1, state_dim, state_dim)  # [1, 1, state_dim, state_dim]
        base_metric_expanded = base_metric.expand(batch_size * num_heads, seq_len, state_dim, state_dim)  # [batch*heads, seq, state_dim, state_dim]
        base_metric_expanded = base_metric_expanded * head_scale.unsqueeze(-1)  # Scale each head's metric differently

        base_connection = base_connection.reshape(1, 1, state_dim, state_dim, state_dim)  # [1, 1, state_dim, state_dim, state_dim]
        base_connection_expanded = base_connection.expand(batch_size * num_heads, seq_len, state_dim, state_dim, state_dim)  # [batch*heads, seq, state_dim, state_dim, state_dim]
        base_connection_expanded = base_connection_expanded * head_scale.unsqueeze(-1).unsqueeze(-1)  # Scale each head's connection differently
    
        # Compute metric attention per head and sequence position
        metric_attention = torch.matmul(
            base_metric_expanded,  # [batch*heads, seq, state_dim, state_dim]
            torch.matmul(
                attention_regularized,  # [batch*heads, seq, state_dim, state_dim]
                base_metric_expanded.transpose(-2, -1)  # [batch*heads, seq, state_dim, state_dim]
            )
        )  # [batch*heads, seq, state_dim, state_dim]

        # Compute connection term with proper dimensions
        connection_term = torch.einsum('bhijk,bhk->bhij',
                                     base_connection_expanded,  # [batch*heads, seq, state_dim, state_dim, state_dim]
                                     amplitudes_reshaped)  # [batch*heads, seq, state_dim]

        # Ensure gradients flow through connection term
        connection_term = connection_term.clone()  # Clone to ensure gradient flow
        connection_term.requires_grad_(True)  # Enable gradients

        # Add connection term to metric attention
        metric_attention = metric_attention + connection_term
        
        # Compute matrix logarithm with metric-modified attention
        try:
            hamiltonian = torch.matrix_exp(metric_attention)  # [batch*heads, seq, state_dim, state_dim]
        except RuntimeError:
            # If matrix exponential fails, use simpler Hamiltonian
            hamiltonian = -1j * (metric_attention - torch.eye(
                state_dim,
                device=attention_pattern.device,
                dtype=working_dtype
            ).unsqueeze(0).unsqueeze(0))  # [batch*heads, seq, state_dim, state_dim]
        
        # Compute evolution operator U = exp(-iHt)
        U = torch.matrix_exp(-time * hamiltonian)  # [batch*heads, seq, state_dim, state_dim]
        
        # Convert amplitudes to working dtype for evolution
        amplitudes_float = amplitudes_reshaped.to(working_dtype)  # [batch*heads, seq, state_dim]
        
        # Evolve state
        evolved_amplitudes = torch.matmul(U, amplitudes_float.unsqueeze(-1)).squeeze(-1)  # [batch*heads, seq, state_dim]
        
        # Convert back to target dtype
        evolved_amplitudes = evolved_amplitudes.to(target_dtype)
        
        # Reshape back to original dimensions
        if len(state.amplitudes.shape) > 3:
            evolved_amplitudes = evolved_amplitudes.reshape(batch_size, num_heads, seq_len, state_dim)
        elif len(state.amplitudes.shape) > 2:
            evolved_amplitudes = evolved_amplitudes.reshape(batch_size, seq_len, state_dim)
        else:
            evolved_amplitudes = evolved_amplitudes.reshape(batch_size, state_dim)
            
        # Normalize evolved amplitudes per head
        if len(state.amplitudes.shape) > 3:
            # Compute norm across sequence and hidden dimensions only
            norm = torch.sqrt(torch.sum(torch.abs(evolved_amplitudes) ** 2, dim=(-2, -1), keepdim=True))
            evolved_amplitudes = evolved_amplitudes / norm.clamp(min=1e-8)
        
        # Create new quantum state
        evolved_state = QuantumState(
            amplitudes=evolved_amplitudes,
            basis_labels=state.basis_labels,
            phase=state.phase,
            layout=state.layout,  # Preserve the original layout
            original_norm=state.original_norm  # Preserve the original norm
        )
        
        return evolved_state

    def construct_pattern_bundle(
        self,
        pattern: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[PatternSection, Tuple[PatternSection, Dict[str, torch.Tensor]]]:
        """Construct pattern space fiber bundle from neural pattern.
        
        Args:
            pattern: Input pattern tensor
            return_metrics: Whether to return bundle metrics
            
        Returns:
            Pattern bundle section or tuple of (section, metrics)
        """
        # Get local trivialization
        local_chart, fiber_chart = self.pattern_bundle.local_trivialization(pattern)
        
        if not return_metrics:
            return local_chart
            
        # Compute bundle metrics
        metrics = {
            "connection": self.pattern_bundle.connection_form(pattern),
            "transition": self.pattern_bundle.transition_functions(local_chart, local_chart),
            "projection": self.pattern_bundle.bundle_projection(pattern)
        }
        
        return local_chart, metrics

    def evolve_pattern_bundle(
        self,
        section: PatternSection,
        time: float = 1.0,
        scale_factor: Optional[float] = None
    ) -> Tuple[PatternSection, Dict[str, Any]]:
        """Evolve pattern bundle section using quantum geometric flow.
        
        Args:
            section: Input pattern section
            time: Evolution time
            scale_factor: Optional scale factor for multi-scale evolution
            
        Returns:
            Tuple of (evolved section, evolution metrics)
            
        Raises:
            ValueError: If section is invalid or time is negative
            RuntimeError: If evolution fails
        """
        if section is None:
            raise ValueError("Pattern section cannot be None")
        if time < 0:
            raise ValueError("Evolution time must be non-negative")
            
        try:
            metrics: Dict[str, Any] = {}
            device = section.coordinates.device
            
            # Ensure all components are on the same device
            if self.device != device:
                self.device = device
                self.to(device)
            
            # 1. Create path for parallel transport with validation
            try:
                path = torch.linspace(0, time, steps=10, device=device)
                path = path.unsqueeze(-1).expand(-1, self.hidden_dim)
            except Exception as e:
                raise RuntimeError(f"Failed to create transport path: {str(e)}")
            
            # 2. Apply quantum evolution with validation
            try:
                quantum_result = self.neural_to_quantum(section.coordinates)
                if isinstance(quantum_result, tuple):
                    quantum_state = quantum_result[0]  # Extract just the quantum state
                else:
                    quantum_state = quantum_result
                    
                evolved_state = self.evolve_quantum_state_with_attention(quantum_state)
                    
                # Ensure we have valid quantum states for metrics
                if not isinstance(quantum_state, QuantumState) or not isinstance(evolved_state, QuantumState):
                    raise ValueError("Expected QuantumState for evolution metrics")
                    
                metrics["quantum_evolution"] = {
                    "initial_norm": float(quantum_state.norm().mean().item()),
                    "final_norm": float(evolved_state.norm().mean().item())
                }
            except Exception as e:
                raise RuntimeError(f"Quantum evolution failed: {str(e)}")
            
            # 3. Parallel transport with validation
            try:
                evolved_coordinates = self.pattern_bundle.parallel_transport(
                    section.coordinates,
                    path
                )
                path_diff = path[-1] - path[0]
                coord_diff = evolved_coordinates[-1] - section.coordinates
                path_norm = torch.linalg.vector_norm(path_diff)
                coord_norm = torch.linalg.vector_norm(coord_diff)
                metrics["transport"] = {
                    "path_length": float(path_norm.item()),
                    "coordinate_shift": float(coord_norm.item())
                }
            except Exception as e:
                raise RuntimeError(f"Parallel transport failed: {str(e)}")
            
            # 4. Apply scale transition if requested
            if scale_factor is not None:
                try:
                    # Get current scale from section properties
                    current_scale = getattr(section, 'scale', 1.0)
                    target_scale = current_scale * scale_factor
                    
                    # Create default couplings tensor
                    couplings = torch.zeros(1, self.hidden_dim, device=device)
                    
                    # Analyze scale transition using scale system
                    evolved_coords_batch = evolved_coordinates[-1].unsqueeze(0)
                    scale_results = self.scale_system.analyze_scales(
                        states=[evolved_coords_batch],
                        scale_factors=[current_scale, target_scale]
                    )
                    rg_flow, anomalies = scale_results["fixed_points"], scale_results["anomalies"]
                    
                    # Apply scale transformation using connection
                    evolved_coords_scaled = self.scale_system.connection.connect_scales(
                        source_state=evolved_coords_batch,
                        source_scale=current_scale,
                        target_scale=target_scale
                    )
                    evolved_coordinates = evolved_coords_scaled.squeeze(0)
                    
                    # Convert scale results to serializable format
                    metrics["scale"] = {
                        "initial_scale": float(current_scale),
                        "target_scale": float(target_scale),
                        "rg_flow": rg_flow.tolist() if isinstance(rg_flow, torch.Tensor) else rg_flow,
                        "anomalies": [a.tolist() if isinstance(a, torch.Tensor) else a for a in anomalies]
                    }
                except Exception as e:
                    raise RuntimeError(f"Scale transition failed: {str(e)}")
            else:
                evolved_coordinates = evolved_coordinates[-1]
                
            # 5. Convert evolved quantum state back to classical coordinates
            try:
                classical_coords = self.quantum_to_neural(evolved_state)
                evolved_coordinates = evolved_coordinates + classical_coords
            except Exception as e:
                raise RuntimeError(f"Quantum to neural conversion failed: {str(e)}")
            
            # 6. Create new section with updated transition maps
            try:
                local_chart, fiber_chart = self.pattern_bundle.local_trivialization(evolved_coordinates)
                evolved_section = PatternSection(
                    coordinates=evolved_coordinates,
                    dimension=self.hidden_dim,
                    transition_maps=local_chart.transition_maps
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create evolved section: {str(e)}")
            
            # 7. Validate evolution
            try:
                metric_tensor = self.pattern_bundle.riemannian_framework.compute_metric(evolved_coordinates)
                coord_norm = torch.linalg.vector_norm(evolved_coordinates)
                metrics["validation"] = {
                    "coordinate_norm": float(coord_norm.item()),
                    "transition_consistency": float(torch.trace(metric_tensor.values).item())
                }
            except Exception as e:
                raise RuntimeError(f"Evolution validation failed: {str(e)}")
            
            return evolved_section, metrics
            
        except Exception as e:
            raise RuntimeError(f"Pattern bundle evolution failed: {str(e)}")

    def compute_scale_cohomology(
        self,
        pattern: torch.Tensor,
        return_metrics: bool = False
    ) -> Dict[str, Any]:
        """Compute scale cohomology for pattern.
        
        Args:
            pattern: Input pattern tensor
            return_metrics: Whether to return cohomology metrics
            
        Returns:
            Dictionary of cohomology results
            
        Raises:
            ValueError: If pattern tensor is invalid or empty
            RuntimeError: If scale analysis fails
        """
        if pattern is None or pattern.numel() == 0:
            raise ValueError("Pattern tensor cannot be None or empty")
            
        try:
            # Convert single tensor to list for scale analysis
            states = [pattern]
            
            # Create default couplings tensor
            couplings = torch.zeros(1, self.hidden_dim, device=pattern.device)
            
            # Analyze scales with error handling
            try:
                rg_flow, anomalies, invariants, cohomology_results = self.scale_system.analyze_scales(
                    states=states,
                    scale_factors=[1.0]  # Default scale factor for single state
                )
            except Exception as e:
                raise RuntimeError(f"Scale analysis failed: {str(e)}")
            
            # Validate results
            if rg_flow is None or anomalies is None or invariants is None:
                raise RuntimeError("Scale analysis returned invalid results")
                
            return {
                "rg_flow": rg_flow,
                "anomalies": anomalies,
                "invariants": invariants,
                "cohomology": cohomology_results
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute scale cohomology: {str(e)}")

    def evolve_scale_cohomology(
        self,
        states: List[torch.Tensor],
        time: float = 1.0
    ) -> Dict[str, Any]:
        """Evolve states using scale flow.
        
        Args:
            states: List of input states
            time: Evolution time
            
        Returns:
            Dictionary of evolution results
        """
        # Create default couplings tensor
        couplings = torch.zeros(len(states), self.hidden_dim, device=states[0].device)
        
        # Analyze evolution and convert to dict
        rg_flow, anomalies, invariants, cohomology_results = self.scale_system.analyze_scales(
            states=states,
            scale_factors=[1.0] * len(states)  # One scale factor per state
        )
        
        return {
            "rg_flow": rg_flow,
            "anomalies": anomalies,
            "invariants": invariants,
            "cohomology": cohomology_results
        }

    def compute_motivic_structure(
        self,
        pattern: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute motivic structure for pattern.
        
        Args:
            pattern: Input pattern tensor
            return_metrics: Whether to return structure metrics
            
        Returns:
            Motivic structure tensor or tuple of (structure, metrics)
            
        Raises:
            ValueError: If pattern tensor is invalid
            RuntimeError: If motivic computation fails
        """
        if pattern is None or pattern.numel() == 0:
            raise ValueError("Pattern tensor cannot be None or empty")
            
        try:
            # Create arithmetic form from pattern (degree 1 for vector fields)
            form = ArithmeticForm(degree=1, coefficients=pattern)
            
            # Compute motive with validation
            try:
                motive = self.motivic_system.compute_motive(form)
                if motive is None:
                    raise RuntimeError("Motive computation returned None")
            except Exception as e:
                raise RuntimeError(f"Failed to compute motive: {str(e)}")
            
            if not return_metrics:
                return motive
                
            # Compute metrics with error handling
            try:
                metrics = {
                    "pattern_stability": torch.tensor(
                        float(self.motivic_system._compute_stability(form)), 
                        device=pattern.device
                    ),
                    "cross_tile_flow": torch.tensor(
                        float(self.motivic_system._compute_flow(form)), 
                        device=pattern.device
                    ),
                    "edge_utilization": torch.tensor(
                        float(self.motivic_system._compute_edge_util(form)), 
                        device=pattern.device
                    ),
                    "info_density": torch.tensor(
                        float(self.motivic_system._compute_density(form)), 
                        device=pattern.device
                    )
                }
            except Exception as e:
                raise RuntimeError(f"Failed to compute metrics: {str(e)}")
            
            return motive, metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute motivic structure: {str(e)}")

    def evolve_motivic_structure(
        self,
        form: ArithmeticForm,
        time: float = 1.0
    ) -> torch.Tensor:
        """Evolve motivic structure using arithmetic flow.
        
        Args:
            form: Input arithmetic form
            time: Evolution time
            
        Returns:
            Evolved motivic structure tensor
            
        Raises:
            ValueError: If form is invalid or time is negative
            RuntimeError: If evolution fails
        """
        if form is None:
            raise ValueError("Arithmetic form cannot be None")
        if time < 0:
            raise ValueError("Evolution time must be non-negative")
            
        try:
            # Compute initial motive with validation
            try:
                initial_motive = self.motivic_system.compute_motive(form)
                if initial_motive is None:
                    raise RuntimeError("Initial motive computation returned None")
            except Exception as e:
                raise RuntimeError(f"Failed to compute initial motive: {str(e)}")
            
            # Use dynamics for evolution with error handling
            try:
                evolved_state = self.motivic_system.dynamics.compute_dynamics(initial_motive)
                if evolved_state is None:
                    raise RuntimeError("Evolution returned None state")
            except Exception as e:
                raise RuntimeError(f"Failed to evolve state: {str(e)}")
            
            # Create new form with evolved state (keeping same degree)
            try:
                evolved_form = ArithmeticForm(
                    degree=form.degree,
                    coefficients=evolved_state
                )
                return self.motivic_system.compute_motive(evolved_form)
            except Exception as e:
                raise RuntimeError(f"Failed to create evolved form: {str(e)}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to evolve motivic structure: {str(e)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bridge.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, hidden_dim]
            
        Returns:
            Output tensor with same shape as input
        """
        # Store original shape for later
        original_shape = x.shape
        
        # Reshape to [batch_size * num_heads, seq_len, hidden_dim] for independent head processing
        x_reshaped = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        # Apply layer normalization to real and imaginary parts
        if torch.is_complex(x_reshaped):
            # Convert complex weights to real for layer norm
            real_weight = self.layer_norm_real.weight.real.to(dtype=torch.float64)
            real_bias = self.layer_norm_real.bias.real.to(dtype=torch.float64)
            imag_weight = self.layer_norm_imag.weight.real.to(dtype=torch.float64)
            imag_bias = self.layer_norm_imag.bias.real.to(dtype=torch.float64)
            
            # Apply layer norm with real weights
            x_real = F.layer_norm(
                x_reshaped.real.to(dtype=torch.float64),
                self.layer_norm_real.normalized_shape,
                weight=real_weight,
                bias=real_bias,
                eps=self.layer_norm_real.eps
            )
            x_imag = F.layer_norm(
                x_reshaped.imag.to(dtype=torch.float64),
                self.layer_norm_imag.normalized_shape,
                weight=imag_weight,
                bias=imag_bias,
                eps=self.layer_norm_imag.eps
            )
            x_normalized = torch.complex(x_real, x_imag)
        else:
            # For real inputs, use the real part to create imaginary part
            real_weight = self.layer_norm_real.weight.real.to(dtype=torch.float64)
            real_bias = self.layer_norm_real.bias.real.to(dtype=torch.float64)
            imag_weight = self.layer_norm_imag.weight.real.to(dtype=torch.float64)
            imag_bias = self.layer_norm_imag.bias.real.to(dtype=torch.float64)
            
            x_real = F.layer_norm(
                x_reshaped.to(dtype=torch.float64),
                self.layer_norm_real.normalized_shape,
                weight=real_weight,
                bias=real_bias,
                eps=self.layer_norm_real.eps
            )
            x_imag = F.layer_norm(
                torch.tanh(x_reshaped).to(dtype=torch.float64),
                self.layer_norm_imag.normalized_shape,
                weight=imag_weight,
                bias=imag_bias,
                eps=self.layer_norm_imag.eps
            )
            x_normalized = torch.complex(x_real, x_imag)
        
        # Convert to quantum state
        quantum_result = self.neural_to_quantum(x_normalized)
        if isinstance(quantum_result, tuple):
            quantum_state = quantum_result[0]
        else:
            quantum_state = quantum_result
        
        # Evolve quantum state
        evolved_state = self.evolve_quantum_state_with_attention(quantum_state)
        
        # Convert back to neural representation
        output = self.quantum_to_neural(evolved_state)
        
        # Reshape back to original shape
        output = output.reshape(original_shape)
        
        return output

    def bridge_scales(
        self,
        state: torch.Tensor,
        source_scale: float,
        target_scale: float
    ) -> torch.Tensor:
        """Bridge between different scales using quantum operations.

        Args:
            state: Input state tensor
            source_scale: Source scale factor
            target_scale: Target scale factor

        Returns:
            Transformed state tensor

        Raises:
            ValueError: If input parameters are invalid or state dimensions are incorrect
            RuntimeError: If scale bridging fails
        """
        # Validate inputs
        if state is None or state.numel() == 0:
            raise ValueError("State tensor cannot be None or empty")
        if source_scale <= 0 or target_scale <= 0:
            raise ValueError("Scale factors must be positive")
        if not torch.isfinite(state).all():
            raise ValueError("State tensor contains non-finite values")

        try:
            # Store original norm and sign for scale preservation
            original_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
            original_sign = torch.sgn(state) if torch.is_complex(state) else torch.sign(state)
            scale_factor = target_scale / source_scale

            # Convert to quantum state with validation
            quantum_result = self.neural_to_quantum(state, return_validation=True)
            if isinstance(quantum_result, tuple):
                quantum_state, validation = quantum_result

                if not validation.is_valid and validation.error_type is not None:
                    # Apply correction if state preparation failed
                    quantum_state = self.state_preparation.correct_state(
                        quantum_state,
                        validation.error_type
                    )
            else:
                quantum_state = quantum_result

            # Validate quantum state
            if not isinstance(quantum_state, QuantumState):
                raise ValueError("Failed to create valid quantum state")

            # Get scale ratio for time evolution with validation
            try:
                scale_ratio = np.log2(target_scale / source_scale)
                if not np.isfinite(scale_ratio):
                    raise ValueError("Invalid scale ratio computed")
                time = torch.sigmoid(torch.tensor(scale_ratio)).item() * 0.5
            except Exception as e:
                raise RuntimeError(f"Scale ratio computation failed: {str(e)}")

            # Evolve state using quantum geometric attention
            try:
                evolved_state = self.evolve_quantum_state_with_attention(
                    quantum_state,
                    time=time
                )
            except Exception as e:
                raise RuntimeError(f"Quantum evolution failed: {str(e)}")

            # Convert back to neural representation
            try:
                neural_state = self.quantum_to_neural(evolved_state, state.shape)

                # Validate output state
                if not torch.isfinite(neural_state).all():
                    raise ValueError("Neural state contains non-finite values")
            except Exception as e:
                raise RuntimeError(f"Neural state conversion failed: {str(e)}")

            # Interpolate between initial and evolved states while preserving phase
            try:
                alpha = 0.7  # Bias towards initial state to preserve structure
                
                # For complex inputs, preserve full phase information
                if torch.is_complex(state):
                    # Get phase from original state
                    original_phase = torch.angle(state)
                    # Get magnitude from interpolation
                    neural_state_abs = torch.abs(neural_state)
                    state_abs = torch.abs(state)
                    interpolated_abs = alpha * state_abs + (1 - alpha) * neural_state_abs
                    # Reconstruct with original phase
                    neural_state = interpolated_abs * torch.exp(1j * original_phase)
                else:
                    # For real inputs, preserve sign as before
                    neural_state_abs = torch.abs(neural_state)
                    state_abs = torch.abs(state)
                    interpolated_abs = alpha * state_abs + (1 - alpha) * neural_state_abs
                    neural_state = interpolated_abs * original_sign

                # Apply scale factor to preserve normalization
                current_norm = torch.linalg.vector_norm(neural_state, dim=-1, keepdim=True)
                neural_state = neural_state * (original_norm * scale_factor) / (current_norm + 1e-8)

                # Validate final state
                if not torch.isfinite(neural_state).all():
                    raise ValueError("Final state contains non-finite values")
            except Exception as e:
                raise RuntimeError(f"State interpolation failed: {str(e)}")

            # Track cross-scale entanglement
            try:
                self._update_entanglement_tracking(
                    source_scale=source_scale,
                    target_scale=target_scale,
                    evolved_state=evolved_state
                )
            except Exception as e:
                # Don't fail the whole operation if entanglement tracking fails
                print(f"Warning: Entanglement tracking failed: {str(e)}")

            return neural_state

        except ValueError as ve:
            # Re-raise ValueError directly
            raise ve
        except Exception as e:
            raise RuntimeError(f"Scale bridging failed: {str(e)}")
        
    def _update_entanglement_tracking(
        self,
        source_scale: float,
        target_scale: float,
        evolved_state: QuantumState
    ) -> None:
        """Update entanglement tracking between scales.
        
        Args:
            source_scale: Source scale factor
            target_scale: Target scale factor
            evolved_state: Evolved quantum state
        """
        try:
            # Compute entanglement entropy
            entropy = self.hilbert_space.compute_entanglement_entropy(evolved_state)
            
            # Take mean across batch and convert to tensor
            mean_entropy = entropy.mean().clone().detach()
            
            # Store in state manager
            self.state_manager.update_entanglement(
                source_scale=source_scale,
                target_scale=target_scale,
                entropy=mean_entropy
            )
        except Exception as e:
            print(f"Warning: Entanglement tracking failed: {str(e)}")

    def compute_coherence(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum coherence between two states.

        Args:
            state1: First state tensor of shape (batch_size, hidden_dim)
            state2: Second state tensor of shape (batch_size, hidden_dim)

        Returns:
            Coherence values of shape (batch_size,)
        """
        # Convert to quantum states
        quantum_result1 = self.neural_to_quantum(state1, return_validation=True)
        quantum_result2 = self.neural_to_quantum(state2, return_validation=True)

        # Extract quantum states
        quantum_state1 = quantum_result1[0] if isinstance(quantum_result1, tuple) else quantum_result1
        quantum_state2 = quantum_result2[0] if isinstance(quantum_result2, tuple) else quantum_result2

        # Get density matrices
        rho1 = quantum_state1.density_matrix()
        rho2 = quantum_state2.density_matrix()

        # Compute fidelity for each batch element
        batch_size = state1.shape[0]  # Use input tensor's batch size
        fidelity = torch.zeros(batch_size, device=rho1.device)
        
        for i in range(batch_size):
            # Compute sqrt(rho1) * rho2 * sqrt(rho1)
            sqrt_rho1 = torch.matrix_power(rho1[i], 1)  # Use integer power
            product = torch.matmul(sqrt_rho1, torch.matmul(rho2[i], sqrt_rho1))
            
            # Compute trace of the square root
            eigenvalues = torch.linalg.eigvals(product)
            fidelity[i] = torch.sqrt(torch.abs(eigenvalues.sum()))

        # Normalize to [0, 1] range
        fidelity = torch.clamp(fidelity, 0.0, 1.0)
        
        return fidelity

    def evolve_quantum_state(self, state: QuantumState, time: float = 1.0) -> QuantumState:
        """Evolve quantum state using Hamiltonian evolution."""
        return self.evolve_quantum_state_with_attention(state, time=time)

    def evolve_pattern_bundle_with_attention(self, pattern_bundle: torch.Tensor, time: float = 1.0) -> torch.Tensor:
        """Evolve pattern bundle through quantum geometric flow with attention.
        
        Args:
            pattern_bundle: Pattern bundle tensor
            time: Evolution time (default: 1.0)
            
        Returns:
            Evolved pattern bundle
        """
        # Convert to quantum state
        quantum_result = self.neural_to_quantum(pattern_bundle)
        if isinstance(quantum_result, tuple):
            quantum_state = quantum_result[0]
        else:
            quantum_state = quantum_result
            
        evolved_state = self.evolve_quantum_state_with_attention(quantum_state, time=time)
        
        # Convert back to neural representation
        evolved_pattern = self.quantum_to_neural(evolved_state)
        
        return evolved_pattern

    def evolve_scale_cohomology_with_attention(self, x: List[torch.Tensor], time: float = 1.0) -> List[torch.Tensor]:
        """Evolve scale cohomology through quantum geometric flow with attention.
        
        Args:
            x: List of scale tensors
            time: Evolution time (default: 1.0)
            
        Returns:
            Evolved scale tensors
        """
        # Convert to quantum state
        quantum_result = self.neural_to_quantum(x[0])
        if isinstance(quantum_result, tuple):
            quantum_state = quantum_result[0]
        else:
            quantum_state = quantum_result
            
        # Ensure we have valid quantum states for metrics
        if not isinstance(quantum_state, QuantumState):
            raise ValueError("Invalid quantum state type")
            
        # Evolve through quantum attention
        evolved_state = self.evolve_quantum_state_with_attention(quantum_state, time=time)
        
        # Convert back to neural representation
        evolved_scale = self.quantum_to_neural(evolved_state)
        
        return [evolved_scale]

    def evolve_geometric_flow_with_attention(
        self,
        x: torch.Tensor,
        time: float,
        **kwargs
    ) -> torch.Tensor:
        """Evolve tensor through quantum geometric flow with attention.
        
        Args:
            x: Input tensor
            time: Evolution time
            **kwargs: Additional arguments
            
        Returns:
            Evolved tensor
        """
        # Convert to quantum state
        quantum_result = self.neural_to_quantum(x)
        if isinstance(quantum_result, tuple):
            quantum_state = quantum_result[0]
        else:
            quantum_state = quantum_result
            
        # Evolve state using quantum geometric attention
        evolved_state = self.evolve_quantum_state_with_attention(
            quantum_state,
            time=time
        )
        
        # Convert back to neural representation
        evolved_tensor = self.quantum_to_neural(evolved_state)
        
        return evolved_tensor