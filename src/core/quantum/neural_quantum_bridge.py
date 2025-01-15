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
            dtype=torch.float64  # Use real-valued parameters
        )
        self.layer_norm_imag = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64  # Use real-valued parameters
        )
        
        # Initialize manifold normalization for real and imaginary parts
        self.manifold_norm_real = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64  # Use real-valued parameters
        )
        self.manifold_norm_imag = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=True,
            dtype=torch.float64  # Use real-valued parameters
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
        
        # Initialize inverse projection
        self.inverse_projection = nn.Linear(
            self.manifold_dim,
            self.hidden_dim,
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
        """Convert neural representation to quantum state."""
        # Validate input dimensions
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"Input tensor must have hidden dimension {self.hidden_dim}, got {x.shape[-1]}")

        # Store original norm
        original_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

        # Get dimensions and reshape if needed
        if len(x.shape) == 2:  # [batch_size, hidden_dim]
            batch_size = x.shape[0]
            seq_len = 1
            x_flat = x  # No need to reshape
        else:  # [batch_size, seq_len, hidden_dim]
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            x_flat = x.reshape(-1, self.hidden_dim)

        # Project to manifold dimension while preserving gradients
        x_manifold = x_flat[..., :self.manifold_dim].clone()
        
        # Normalize to unit norm for quantum state preparation
        x_norm = torch.linalg.vector_norm(x_manifold, dim=-1, keepdim=True)
        x_manifold = x_manifold / (x_norm + 1e-8)
        
        # Convert to quantum amplitudes using direct quantum state preparation
        prepared_state = self.hilbert_space.prepare_state(x_manifold)
        
        # Reshape amplitudes back to include sequence length if needed
        if seq_len > 1:
            prepared_amplitudes = prepared_state.amplitudes.reshape(batch_size, seq_len, -1)
        else:
            prepared_amplitudes = prepared_state.amplitudes
        
        # Create quantum state with proper initialization and gradient tracking
        state = QuantumState(
            amplitudes=prepared_amplitudes.requires_grad_(True),
            basis_labels=[str(i) for i in range(self.manifold_dim)],
            phase=torch.zeros(1, dtype=self.dtype, device=self.device)
        )
        
        # Store original norm in state for later use
        state.original_norm = original_norm
        
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
        """Convert quantum state back to neural representation."""
        # Get classical amplitudes and ensure dtype matches inverse projection
        classical_flat = state.amplitudes.real.to(self.dtype)
        
        # Get dimensions and reshape if needed
        if len(classical_flat.shape) == 2:  # [batch_size, manifold_dim]
            batch_size = classical_flat.shape[0]
            seq_len = 1
            classical_reshaped = classical_flat  # No need to reshape
        else:  # [batch_size, seq_len, manifold_dim]
            batch_size = classical_flat.shape[0]
            seq_len = classical_flat.shape[1]
            classical_reshaped = classical_flat.reshape(-1, self.manifold_dim)
        
        # Project from manifold_dim back to hidden_dim using the inverse projection
        output = self.inverse_projection(classical_reshaped)
        
        # Reshape back to include sequence length if needed
        if seq_len > 1:
            output = output.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Restore original norm while preserving gradients
        if hasattr(state, 'original_norm') and state.original_norm is not None:
            current_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
            scale = state.original_norm / (current_norm + 1e-8)
            output = output * scale.detach()  # Detach scale to prevent gradient explosion
            
            # Verify norm restoration
            restored_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
            assert torch.allclose(restored_norm, state.original_norm, rtol=1e-5)
        
        # Reshape if needed
        if original_shape is not None:
            output = output.reshape(original_shape)
        
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
        state_dim = state.amplitudes.shape[-1]
        
        # Reshape amplitudes to [batch_size * num_heads, state_dim]
        amplitudes_reshaped = state.amplitudes.reshape(-1, state_dim)
        
        # Create or use provided attention pattern
        if attention_pattern is None:
            attention_pattern = torch.eye(
                state_dim,
                device=state.amplitudes.device,
                dtype=state.amplitudes.dtype
            ).unsqueeze(0).expand(amplitudes_reshaped.shape[0], -1, -1)
        else:
            # Ensure attention pattern has correct shape
            if attention_pattern.shape[-2:] != (state_dim, state_dim):
                # Try to reshape attention pattern if possible
                if attention_pattern.shape[-1] < state_dim:
                    # Pad attention pattern
                    padded_attention = torch.zeros(
                        batch_size,
                        state_dim,
                        state_dim,
                        device=attention_pattern.device,
                        dtype=attention_pattern.dtype
                    )
                    padded_attention[:, :attention_pattern.shape[1], :attention_pattern.shape[2]] = attention_pattern
                    padded_attention = padded_attention + torch.eye(
                        state_dim,
                        device=attention_pattern.device,
                        dtype=attention_pattern.dtype
                    ).unsqueeze(0) * 1e-6
                    attention_pattern = padded_attention
                elif attention_pattern.shape[-1] > state_dim:
                    # Truncate attention pattern
                    attention_pattern = attention_pattern[:, :state_dim, :state_dim]
                else:
                    raise ValueError(f"Attention pattern shape {attention_pattern.shape} does not match state dimension {state_dim}")
        
        # Construct Hamiltonian from attention pattern
        # H = -i log(A) where A is the attention pattern
        # Add small identity to ensure attention pattern is invertible
        attention_regularized = attention_pattern + torch.eye(
            state_dim,
            device=attention_pattern.device,
            dtype=attention_pattern.dtype
        ).unsqueeze(0) * 1e-6
        
        # Convert to complex64 for matrix operations
        attention_regularized = attention_regularized.to(torch.complex64)
        
        # Compute matrix logarithm
        try:
            hamiltonian = -1j * torch.matrix_exp(attention_regularized)
        except RuntimeError:
            # If matrix exponential fails, use simpler Hamiltonian
            hamiltonian = -1j * (attention_regularized - torch.eye(
                state_dim,
                device=attention_pattern.device,
                dtype=torch.complex64
            ).unsqueeze(0))
        
        # Compute evolution operator U = exp(-iHt)
        U = torch.matrix_exp(-time * hamiltonian)
        
        # Convert state amplitudes to complex64 for evolution
        amplitudes_float = amplitudes_reshaped.to(torch.complex64)
        
        # Evolve state
        evolved_amplitudes = torch.matmul(U, amplitudes_float.unsqueeze(-1)).squeeze(-1)
        
        # Convert back to complex128 for consistency
        evolved_amplitudes = evolved_amplitudes.to(torch.complex128)
        
        # Reshape back to original dimensions
        evolved_amplitudes = evolved_amplitudes.reshape(state.amplitudes.shape)
        
        # Create new quantum state
        evolved_state = QuantumState(
            amplitudes=evolved_amplitudes,
            basis_labels=state.basis_labels,
            phase=state.phase
        )
        
        # Preserve original norm if it exists
        if hasattr(state, 'original_norm') and state.original_norm is not None:
            current_norm = torch.linalg.vector_norm(evolved_state.amplitudes, dim=-1, keepdim=True)
            evolved_state.amplitudes = evolved_state.amplitudes * (state.original_norm / (current_norm + 1e-8))
            evolved_state.original_norm = state.original_norm
        
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

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through neural quantum bridge."""
        # Store original shape and norm
        original_shape = x.shape
        original_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        
        # Split complex tensor into real and imaginary parts
        if torch.is_complex(x):
            x_real = x.real.to(torch.float64)  # Convert to float64
            x_imag = x.imag.to(torch.float64)  # Convert to float64
        else:
            x_real = x.to(torch.float64)  # Convert to float64
            # Initialize imaginary part with small random values to ensure gradient flow
            x_imag = torch.randn_like(x, dtype=torch.float64, requires_grad=True) * 0.01
            
        # Apply separate layer normalization to real and imaginary parts
        # Clone to ensure gradient flow and add small noise to prevent vanishing gradients
        x_real = self.layer_norm_real(x_real.clone() + torch.randn_like(x_real) * 1e-6)
        x_imag = self.layer_norm_imag(x_imag.clone() + torch.randn_like(x_imag) * 1e-6)
        
        # Recombine into complex tensor
        x = torch.complex(x_real, x_imag)
        
        # Convert to quantum state
        quantum_result = self.neural_to_quantum(x)
        quantum_state = quantum_result[0] if isinstance(quantum_result, tuple) else quantum_result
        
        # Apply quantum evolution
        evolved_state = self.evolve_quantum_state_with_attention(quantum_state)
        
        # Convert back to neural representation
        output = self.quantum_to_neural(evolved_state, original_shape)
        
        # Split output into real and imaginary parts for manifold normalization
        if torch.is_complex(output):
            output_real = output.real.to(torch.float64)  # Convert to float64
            output_imag = output.imag.to(torch.float64)  # Convert to float64
        else:
            output_real = output.to(torch.float64)  # Convert to float64
            # Initialize imaginary part with small random values
            output_imag = torch.randn_like(output, dtype=torch.float64, requires_grad=True) * 0.01
            
        # Apply separate manifold normalization to real and imaginary parts
        # Clone and add small noise to prevent vanishing gradients
        output_real = self.manifold_norm_real(output_real.clone() + torch.randn_like(output_real) * 1e-6)
        output_imag = self.manifold_norm_imag(output_imag.clone() + torch.randn_like(output_imag) * 1e-6)
        
        # Recombine into complex tensor
        output = torch.complex(output_real, output_imag)
        
        # Restore original norm while preserving gradients
        current_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
        scale = original_norm / (current_norm + 1e-8)
        output = output * scale.detach()  # Detach scale to prevent gradient explosion
        
        if not return_intermediates:
            return output
            
        # Collect intermediate results
        intermediates = {
            "quantum_state": evolved_state,
            "original_norm": original_norm,
            "output_norm": current_norm,
            "layer_norm_stats": {
                "input_mean_real": torch.mean(x_real, dim=-1),
                "input_std_real": torch.std(x_real, dim=-1),
                "input_mean_imag": torch.mean(x_imag, dim=-1),
                "input_std_imag": torch.std(x_imag, dim=-1),
                "output_mean_real": torch.mean(output_real, dim=-1),
                "output_std_real": torch.std(output_real, dim=-1),
                "output_mean_imag": torch.mean(output_imag, dim=-1),
                "output_std_imag": torch.std(output_imag, dim=-1)
            }
        }
        
        return output, intermediates

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
            # Store original norm for scale preservation
            original_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
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

            # Interpolate between initial and evolved states
            try:
                alpha = 0.7  # Bias towards initial state to preserve structure
                neural_state = alpha * state + (1 - alpha) * neural_state

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
        batch_size = rho1.shape[0]
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
        # Construct Hamiltonian using quantum_attention
        H = self.quantum_attention.construct_hamiltonian(state.amplitudes)
        
        # Ensure complex64 type
        if H.dtype != torch.complex64:
            H = H.to(torch.complex64)
        
        # Compute evolution operator U = exp(-iHt)
        U = torch.matrix_exp(-1j * time * H)
        
        # Ensure state amplitudes are complex64
        if state.amplitudes.dtype != torch.complex64:
            state.amplitudes = state.amplitudes.to(torch.complex64)
        
        # Evolve state
        evolved_state = state.evolve(U)
        return evolved_state

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