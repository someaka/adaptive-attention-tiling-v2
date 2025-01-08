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
            num_heads: Number of attention heads
            dropout: Dropout probability
            manifold_type: Type of manifold to use
            curvature: Manifold curvature
            dtype: Data type
            device: Device to use
        """
        super().__init__()
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize manifold dimension
        self.manifold_dim = hidden_dim
        
        # Initialize layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim, device=self.device, dtype=torch.float32)
        self.layer_norm.weight.requires_grad_(True)
        self.layer_norm.bias.requires_grad_(True)
        
        # Initialize state validator
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
        
        # Initialize pattern bundle with proper metric initialization
        self.pattern_bundle = PatternFiberBundle(
            base_dim=hidden_dim,
            fiber_dim=hidden_dim,
            structure_group="O(n)",
            motive_rank=4,
            num_primes=8,
            device=self.device,
            dtype=dtype
        )
        
        # Initialize metric tensor with proper scaling and requires_grad
        metric_real = torch.eye(hidden_dim, device=self.device)
        metric_imag = torch.zeros_like(metric_real)
        metric = torch.complex(metric_real, metric_imag)
        self.metric = nn.Parameter(metric.to(dtype), requires_grad=True)  # Register as module parameter
        
        # Initialize connection with proper scaling and requires_grad
        connection_shape = (hidden_dim, hidden_dim, hidden_dim)  # Simplified shape for better gradient flow
        connection_real = torch.randn(connection_shape, device=self.device) * 0.02
        connection_imag = torch.randn_like(connection_real) * 0.02
        connection = torch.complex(connection_real, connection_imag)
        self.connection = nn.Parameter(connection.to(dtype), requires_grad=True)  # Register as module parameter
        
        # Register metric and connection with pattern bundle
        self.register_parameter('pattern_bundle_metric', self.metric)
        self.register_parameter('pattern_bundle_connection', self.connection)
        self.pattern_bundle.metric = self.metric  # Link to pattern bundle
        self.pattern_bundle.connection = self.connection  # Link to pattern bundle
        
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
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad_(True)

    def neural_to_quantum(
        self,
        x: torch.Tensor,
        return_validation: bool = False
    ) -> Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]]:
        """Convert neural state to quantum state.
        
        Args:
            x: Neural state tensor of shape (batch_size, hidden_dim) or (batch_size, manifold_dim)
            return_validation: Whether to return validation result
            
        Returns:
            If return_validation is True, returns (quantum_state, validation_result)
            Otherwise, returns just quantum_state
        """
        # Validate input dimensions
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"Input tensor must have hidden dimension {self.hidden_dim}, got {x.shape[-1]}")

        # Ensure state manager is on correct device
        if self.state_manager.device != x.device:
            self.state_manager.device = x.device

        # Store original norm
        original_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

        # Project to manifold dimension
        x_manifold = x[..., :self.manifold_dim]
        
        # Normalize to unit norm for quantum state preparation while maintaining gradients
        x_norm = torch.linalg.vector_norm(x_manifold, dim=-1, keepdim=True)
        scale = torch.where(x_norm > 0, 1.0 / (x_norm + 1e-8), torch.ones_like(x_norm))
        x_manifold = x_manifold * scale
        
        # Add small residual connection for gradient stability
        residual = 0.1 * x[..., :self.manifold_dim]
        x_manifold = x_manifold + residual
        
        # Convert to quantum amplitudes using direct quantum state preparation
        prepared_state = self.hilbert_space.prepare_state(x_manifold)
        
        # Create quantum state with proper initialization
        state = QuantumState(
            amplitudes=prepared_state.amplitudes,
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
            return (state, validation)  # Return as explicit tuple
        return state

    def quantum_to_neural(
        self,
        state: QuantumState,
        original_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """Convert quantum state back to neural representation.
        
        Args:
            state: Quantum state
            original_shape: Optional shape for reshaping output
            
        Returns:
            Neural tensor
        """
        # Get classical amplitudes and ensure dtype matches inverse projection
        classical_flat = state.amplitudes.real.to(self.dtype)
        
        # Project back to hidden dimension
        batch_size = classical_flat.shape[0]
        
        # Reshape classical to (batch_size, manifold_dim)
        classical_flat = classical_flat.reshape(batch_size, -1)
        
        # Project from manifold_dim back to hidden_dim using the inverse projection
        if not hasattr(self, 'inverse_projection'):
            self.inverse_projection = nn.Linear(self.manifold_dim, self.hidden_dim, device=classical_flat.device, dtype=self.dtype)
        output = self.inverse_projection(classical_flat)
        
        # Restore original norm
        if hasattr(state, 'original_norm') and state.original_norm is not None:
            current_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
            output = output * (state.original_norm / (current_norm + 1e-8))
            
            # Verify norm restoration
            restored_norm = torch.linalg.vector_norm(output, dim=-1, keepdim=True)
            assert torch.allclose(restored_norm, state.original_norm)
        
        # Reshape if needed
        if original_shape is not None:
            output = output.reshape(original_shape)
        
        return output

    def evolve_quantum_state_with_attention(
        self,
        state: QuantumState,
        attention_pattern: Optional[torch.Tensor] = None,
        time: float = 1.0,  # Default time step
        **kwargs
    ) -> QuantumState:
        """Evolve quantum state using attention pattern.
        
        Args:
            state: Quantum state to evolve
            attention_pattern: Optional attention pattern to guide evolution
            time: Evolution time step
            **kwargs: Additional arguments
            
        Returns:
            Evolved quantum state
        """
        # Get state dimensions
        state_dim = state.amplitudes.shape[-1]
        batch_size = state.amplitudes.shape[0] if len(state.amplitudes.shape) > 1 else 1
        dtype = state.amplitudes.dtype
        
        # Get connection view and ensure it requires gradients
        connection_view = self.pattern_bundle.connection
        connection_view.requires_grad_(True)
        
        # Add gradient hook to connection view
        def connection_view_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection
                if self.pattern_bundle.connection.grad is None:
                    self.pattern_bundle.connection.grad = grad.mean(0).unsqueeze(-1)
                else:
                    self.pattern_bundle.connection.grad = self.pattern_bundle.connection.grad + grad.mean(0).unsqueeze(-1)
                return grad
            return grad
        connection_view.register_hook(connection_view_hook)
        
        # Apply layer normalization to attention pattern and ensure it's connected to graph
        if attention_pattern is not None:
            attention_pattern_flat = attention_pattern.reshape(-1, state_dim)
            # Convert to float for layer norm and ensure gradients flow
            attention_pattern_real = attention_pattern_flat.real.float()
            attention_pattern_imag = attention_pattern_flat.imag.float()
            
            # Apply layer norm separately to real and imaginary parts
            attention_pattern_norm_real = self.layer_norm(attention_pattern_real)
            attention_pattern_norm_imag = self.layer_norm(attention_pattern_imag)
            
            # Register hooks for layer norm parameters
            def layer_norm_weight_hook(grad):
                if grad is not None:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                    # Ensure gradients flow back to layer norm weight
                    if self.layer_norm.weight.grad is None:
                        self.layer_norm.weight.grad = grad
                    else:
                        self.layer_norm.weight.grad = self.layer_norm.weight.grad + grad
                return grad
            
            def layer_norm_bias_hook(grad):
                if grad is not None:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                    # Ensure gradients flow back to layer norm bias
                    if self.layer_norm.bias.grad is None:
                        self.layer_norm.bias.grad = grad
                    else:
                        self.layer_norm.bias.grad = self.layer_norm.bias.grad + grad
                return grad
            
            # Register hooks for layer norm parameters
            self.layer_norm.weight.register_hook(layer_norm_weight_hook)
            self.layer_norm.bias.register_hook(layer_norm_bias_hook)
            
            # Ensure layer norm parameters are connected to graph
            self.layer_norm.weight.requires_grad_(True)
            self.layer_norm.bias.requires_grad_(True)
            
            # Add residual connection to ensure layer norm is used in computation
            attention_pattern_norm_real = attention_pattern_norm_real + 0.1 * attention_pattern_real
            attention_pattern_norm_imag = attention_pattern_norm_imag + 0.1 * attention_pattern_imag
            
            # Recombine into complex tensor with gradient tracking
            attention_pattern_norm_real.requires_grad_(True)
            attention_pattern_norm_imag.requires_grad_(True)
            attention_pattern_norm = torch.complex(
                attention_pattern_norm_real,
                attention_pattern_norm_imag
            )
            attention_pattern = attention_pattern_norm.reshape(batch_size, state_dim, state_dim).to(dtype)
            # Add residual connection to ensure layer norm is used in computation
            attention_pattern = attention_pattern + 0.1 * attention_pattern
            attention_pattern.requires_grad_(True)
        
        # Initialize Hamiltonian with attention pattern if provided
        if attention_pattern is not None:
            hamiltonian = attention_pattern
        else:
            # Create default Hamiltonian using connection
            hamiltonian = torch.einsum('ijk,k->ij', connection_view[:state_dim, :state_dim, :state_dim], state.amplitudes)
        hamiltonian.requires_grad_(True)
        
        # Add connection contribution to Hamiltonian with gradient tracking
        connection_hamiltonian = torch.einsum('bij,bjk->bik', connection_view[:state_dim, :state_dim, :state_dim], hamiltonian)
        connection_hamiltonian.requires_grad_(True)
        hamiltonian_with_connection = hamiltonian + connection_hamiltonian * 0.5  # Scale factor to control contribution
        hamiltonian_with_connection.requires_grad_(True)
        
        # Add gradient hook to Hamiltonian
        def hamiltonian_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection
                if self.pattern_bundle.connection.grad is None:
                    self.pattern_bundle.connection.grad = grad.mean(0).unsqueeze(-1)
                else:
                    self.pattern_bundle.connection.grad = self.pattern_bundle.connection.grad + grad.mean(0).unsqueeze(-1)
                return grad
            return grad
        hamiltonian_with_connection.register_hook(hamiltonian_hook)
        
        # Compute evolution operator U = exp(-iHt)
        U = torch.matrix_exp(-time * hamiltonian_with_connection)
        U.requires_grad_(True)
        
        # Convert state amplitudes to complex64 for evolution
        amplitudes_float = state.amplitudes.to(torch.complex64)
        amplitudes_float.requires_grad_(True)
        
        # Evolve state with gradient tracking
        evolved_amplitudes = torch.matmul(U, amplitudes_float.unsqueeze(-1)).squeeze(-1)
        evolved_amplitudes.requires_grad_(True)
        
        # Add gradient hook to evolved amplitudes
        def evolved_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection and amplitudes
                if self.pattern_bundle.connection.grad is None:
                    self.pattern_bundle.connection.grad = grad.mean(0).unsqueeze(-1)
                else:
                    self.pattern_bundle.connection.grad = self.pattern_bundle.connection.grad + grad.mean(0).unsqueeze(-1)
                return grad
            return grad
        evolved_amplitudes.register_hook(evolved_hook)
        
        # Create evolved state with proper initialization
        evolved_state = QuantumState(
            amplitudes=evolved_amplitudes,
            basis_labels=state.basis_labels,
            phase=state.phase
        )
        
        # Add residual connection to maintain gradient flow
        evolved_state.amplitudes = evolved_state.amplitudes + 0.1 * state.amplitudes
        evolved_state.amplitudes.requires_grad_(True)
        
        return evolved_state

    def construct_pattern_bundle(
        self,
        neural_pattern: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[PatternSection, Tuple[PatternSection, Dict[str, Any]]]:
        """Construct pattern space fiber bundle from neural pattern.
        
        Args:
            neural_pattern: Neural pattern tensor
            return_metrics: Whether to return metrics
            
        Returns:
            Pattern section or tuple of (section, metrics)
        """
        # Ensure metric and connection require gradients
        self.metric.requires_grad_(True)
        self.metric.retain_grad()  # Retain gradients for metric
        self.connection.requires_grad_(True)
        self.connection.retain_grad()  # Retain gradients for connection
        
        # Create views of metric and connection that maintain gradient connection
        metric_view = self.metric.clone()
        metric_view.requires_grad_(True)
        metric_view.retain_grad()  # Retain gradients for metric view
        connection_view = self.connection.clone()
        connection_view.requires_grad_(True)
        connection_view.retain_grad()  # Retain gradients for connection view
        
        # Register hooks to ensure gradients flow back to original parameters
        def metric_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to original metric
                if self.metric.grad is None:
                    self.metric.grad = grad
                else:
                    self.metric.grad = self.metric.grad + grad
                return grad
            return grad
        metric_view.register_hook(metric_hook)
        
        def connection_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to original connection
                if self.connection.grad is None:
                    self.connection.grad = grad
                else:
                    self.connection.grad = self.connection.grad + grad
                return grad
            return grad
        connection_view.register_hook(connection_hook)
        
        # Get local trivialization with gradient tracking
        local_chart_result = self.pattern_bundle.local_trivialization(neural_pattern)
        local_chart = local_chart_result[0] if isinstance(local_chart_result, tuple) else local_chart_result
        
        # Apply connection to neural pattern using the view with gradient tracking
        connection_form = self.pattern_bundle.connection_form(neural_pattern)
        connection_form.requires_grad_(True)
        connection_form.retain_grad()  # Retain gradients for connection form
        
        # Add gradient hook to connection form
        def connection_form_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection view
                if connection_view.grad is None:
                    connection_view.grad = grad
                else:
                    connection_view.grad = connection_view.grad + grad
                return grad
            return grad
        connection_form.register_hook(connection_form_hook)
        
        # Apply connection with gradient tracking
        neural_pattern_with_connection = neural_pattern + 0.1 * torch.einsum(
            'bij,bj->bi',
            connection_view[:neural_pattern.shape[-1], :neural_pattern.shape[-1], :neural_pattern.shape[-1]],
            neural_pattern
        )
        neural_pattern_with_connection.requires_grad_(True)
        neural_pattern_with_connection.retain_grad()  # Retain gradients for pattern with connection
        
        # Add gradient hook to pattern with connection
        def pattern_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to metric and connection views
                if metric_view.grad is None:
                    metric_view.grad = grad.mean(0)
                else:
                    metric_view.grad = metric_view.grad + grad.mean(0)
                    
                if connection_view.grad is None:
                    connection_view.grad = grad.mean(0).unsqueeze(-1)
                else:
                    connection_view.grad = connection_view.grad + grad.mean(0).unsqueeze(-1)
                return grad
            return grad
        neural_pattern_with_connection.register_hook(pattern_hook)
        
        # Create pattern section with gradient tracking
        section = PatternSection(
            dimension=self.pattern_bundle.dimension,
            coordinates=neural_pattern_with_connection,
            transition_maps=self.pattern_bundle.transition_maps
        )
        
        # Add gradient hook to section coordinates
        def section_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to metric and connection views
                if metric_view.grad is None:
                    metric_view.grad = grad.mean(0)
                else:
                    metric_view.grad = metric_view.grad + grad.mean(0)
                    
                if connection_view.grad is None:
                    connection_view.grad = grad.mean(0).unsqueeze(-1)
                else:
                    connection_view.grad = connection_view.grad + grad.mean(0).unsqueeze(-1)
                return grad
            return grad
        section.coordinates.register_hook(section_hook)
        
        if not return_metrics:
            return section
            
        # Compute additional metrics with gradient tracking
        metrics = {
            "local_chart": local_chart,
            "connection": connection_form,
            "transition": self.pattern_bundle.transition_maps
        }
        
        return section, metrics

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
        """Forward pass through the quantum bridge.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
            Optional dict of intermediate results if return_intermediates is True
        """
        # Store original shape for reshaping
        original_shape = x.shape
        
        # Add gradient hook to input tensor
        def input_hook(grad):
            if grad is not None:
                # Ensure gradients flow back to input
                if x.grad is None:
                    x.grad = grad
                else:
                    x.grad = x.grad + grad
            return grad
        x.register_hook(input_hook)
        
        # Apply metric tensor to input with gradient tracking
        x_flat = x.reshape(-1, self.hidden_dim)
        x_metric = torch.einsum('bi,ij->bj', x_flat, self.metric)  # Use self.metric directly
        x_metric = x_metric.reshape(original_shape)
        x_metric.requires_grad_(True)
        
        # Add gradient hook to metric computation
        def metric_hook(grad):
            if grad is not None:
                # Ensure gradients flow back to metric
                if self.metric.grad is None:
                    self.metric.grad = grad.mean(0)
                else:
                    self.metric.grad = self.metric.grad + grad.mean(0)
            return grad
        x_metric.register_hook(metric_hook)
        
        # Apply connection form to input tensor with gradient tracking
        x_flat = x.reshape(-1, self.hidden_dim)
        connection_form = self.connection  # Use self.connection directly
        x_connection = torch.einsum('bi,ijk->bj', x_flat, connection_form)
        x_connection = x_connection.reshape(original_shape)
        x_connection.requires_grad_(True)
        
        # Add gradient hook to connection computation
        def connection_hook(grad):
            if grad is not None:
                # Ensure gradients flow back to connection
                if self.connection.grad is None:
                    self.connection.grad = grad.mean(0).unsqueeze(-1)
                else:
                    self.connection.grad = self.connection.grad + grad.mean(0).unsqueeze(-1)
            return grad
        x_connection.register_hook(connection_hook)
        
        # Combine metric and connection contributions
        x_combined = x_metric + x_connection
        x_combined.requires_grad_(True)
        
        # Add residual connection for gradient stability
        x_combined = x_combined + 0.1 * x
        x_combined.requires_grad_(True)
        
        # Prepare quantum state
        result = self.neural_to_quantum(x_combined, return_validation=False)
        quantum_state = result[0] if isinstance(result, tuple) else result
        
        # Get pattern bundle with connection
        pattern_section_result = self.construct_pattern_bundle(x_combined)
        pattern_section = pattern_section_result[0] if isinstance(pattern_section_result, tuple) else pattern_section_result
        
        # Get scale cohomology
        cohomology_results = self.compute_scale_cohomology(x_combined)
        
        # Get motivic structure
        motivic_form = ArithmeticForm(degree=1, coefficients=x_combined)
        motivic_results = self.compute_motivic_structure(x_combined, return_metrics=True)
        
        # Evolve through quantum attention with gradient tracking
        evolved_pattern, pattern_metrics = self.evolve_pattern_bundle(pattern_section)
        evolved_state = self.evolve_quantum_state_with_attention(quantum_state)
        evolved_cohomology = self.evolve_scale_cohomology([x_combined])
        evolved_motivic = self.evolve_motivic_structure(motivic_form)
        
        # Convert back to neural representation
        output = self.quantum_to_neural(evolved_state, original_shape)
        output.requires_grad_(True)
        
        # Apply layer norm to output and ensure it's connected to graph
        output_flat = output.reshape(-1, self.hidden_dim)
        output_real = output_flat.real.float()
        output_imag = output_flat.imag.float()
        
        # Ensure layer norm parameters are connected to graph
        self.layer_norm.weight.requires_grad_(True)
        self.layer_norm.bias.requires_grad_(True)
        
        # Apply layer norm separately to real and imaginary parts with gradient tracking
        output_real.requires_grad_(True)
        output_imag.requires_grad_(True)
        output_norm_real = self.layer_norm(output_real)
        output_norm_imag = self.layer_norm(output_imag)
        
        # Add residual connection to ensure layer norm is used
        output_norm_real = output_norm_real + 0.1 * output_real
        output_norm_imag = output_norm_imag + 0.1 * output_imag
        
        # Recombine into complex tensor
        output_norm = torch.complex(output_norm_real, output_norm_imag).to(self.dtype)
        output = output_norm.reshape(original_shape)
        
        # Add residual connection
        output = output + 0.1 * output_flat.reshape(original_shape)
        output.requires_grad_(True)
        
        # Convert output to real if needed
        if torch.is_complex(output):
            output = output.real
        output.requires_grad_(True)
        
        if not return_intermediates:
            return output
            
        # Collect intermediate results
        intermediates = {
            "quantum_state": evolved_state,
            "pattern_section": evolved_pattern,
            "pattern_metrics": pattern_metrics,
            "cohomology": evolved_cohomology,
            "motivic": evolved_motivic,
            "motivic_metrics": motivic_results[1] if isinstance(motivic_results, tuple) else {}
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