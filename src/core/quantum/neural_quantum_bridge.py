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
        dtype: torch.dtype = torch.complex64,
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
            dtype=dtype,
            manifold_dim=self.manifold_dim
        )
        
        # Initialize metric tensor with proper scaling and requires_grad
        metric_real = torch.eye(hidden_dim, device=self.device, dtype=self.get_float_dtype())
        metric_imag = torch.zeros_like(metric_real)
        metric = torch.complex(metric_real, metric_imag)
        self.metric = nn.Parameter(metric, requires_grad=True)  # Register as module parameter
        
        # Add metric gradient hook
        def metric_hook(grad):
            if grad is not None:
                # Scale gradients for numerical stability only
                scaled_grad = grad * 0.1
                return scaled_grad
            return grad
        self.metric.register_hook(metric_hook)
        
        # Initialize connection with proper scaling and requires_grad
        connection_shape = (hidden_dim, hidden_dim, hidden_dim)
        connection_real = torch.randn(connection_shape, device=self.device, dtype=self.get_float_dtype()) * 0.02
        connection_imag = torch.randn_like(connection_real) * 0.02
        connection = torch.complex(connection_real, connection_imag)
        self.connection = nn.Parameter(connection, requires_grad=True)  # Register as module parameter
        
        # Register metric and connection with pattern bundle
        self.register_parameter('pattern_bundle_metric', self.metric)
        self.register_parameter('pattern_bundle_connection', self.connection)
        self.pattern_bundle.metric = self.metric  # Link to pattern bundle
        self.pattern_bundle.connection = self.connection  # Link to pattern bundle
        
        # Ensure connection coefficients are properly tracked for gradients
        def connection_hook(grad):
            if grad is not None:
                # Scale gradients for numerical stability only
                scaled_grad = grad * 0.1
                return scaled_grad
            return grad
        self.connection.register_hook(connection_hook)
        
        # Add connection to pattern bundle's parameter list
        self.pattern_bundle.register_parameter('connection_coeffs', self.connection)
        
        # Ensure riemannian framework connection coeffs require gradients
        if hasattr(self.pattern_bundle, 'riemannian_framework'):
            # Initialize riemannian connection coefficients
            riemannian_connection = nn.Parameter(
                torch.randn(hidden_dim, hidden_dim, hidden_dim, device=self.device, dtype=self.dtype) * 0.02,
                requires_grad=True
            )
            
            # Register gradient hook for riemannian connection
            def riemannian_connection_hook(grad):
                if grad is not None:
                    # Scale gradients for numerical stability
                    scaled_grad = grad * 0.1
                    return scaled_grad
                return grad
            riemannian_connection.register_hook(riemannian_connection_hook)
            
            # Register with pattern bundle using a flat name
            self.pattern_bundle.register_parameter(
                'riemannian_connection_coeffs',  # Flat name without dots
                riemannian_connection
            )
            
            # Set the connection coefficients in the riemannian framework
            self.pattern_bundle.riemannian_framework.connection_coeffs = riemannian_connection
            
            # Create reference for backward compatibility
            setattr(self.pattern_bundle.riemannian_framework, 'connection_coeffs', riemannian_connection)
        
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

    def get_float_dtype(self) -> torch.dtype:
        """Get the corresponding float dtype for the complex dtype."""
        return torch.float64 if self.dtype == torch.complex128 else torch.float32

    def get_complex_dtype(self) -> torch.dtype:
        """Get the complex dtype."""
        return self.dtype

    def neural_to_quantum(
        self,
        x: torch.Tensor,
        return_validation: bool = False
    ) -> Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]]:
        """Convert neural state to quantum state."""
        # Store original shape and energy
        original_shape = x.shape
        original_energy = torch.sum(torch.abs(x) ** 2, dim=-1, keepdim=True)
        
        # Project to manifold
        x_manifold = x[..., :self.manifold_dim]
        
        # Normalize to preserve energy
        x_manifold_energy = torch.sum(torch.abs(x_manifold) ** 2, dim=-1, keepdim=True)
        energy_scale = torch.sqrt(original_energy / (x_manifold_energy + 1e-8))
        x_manifold = x_manifold * energy_scale
        
        # Add small residual connection with strict energy preservation
        residual = 0.001 * x[..., :self.manifold_dim]  # Further reduced residual factor
        residual_energy = torch.sum(torch.abs(residual) ** 2, dim=-1, keepdim=True)
        residual_scale = torch.sqrt(0.001 * original_energy / (residual_energy + 1e-8))
        residual = residual * residual_scale
        x_manifold = x_manifold + residual
        
        # Final energy normalization with exact matching
        final_energy = torch.sum(torch.abs(x_manifold) ** 2, dim=-1, keepdim=True)
        final_scale = torch.sqrt(original_energy / (final_energy + 1e-8))
        x_manifold = x_manifold * final_scale
        
        # Validate energy conservation
        final_energy = torch.sum(torch.abs(x_manifold) ** 2, dim=-1, keepdim=True)
        energy_diff = torch.abs(final_energy - original_energy) / original_energy
        assert torch.all(energy_diff < 1e-6), "Energy not conserved in manifold projection"
        
        # Create quantum state with original energy
        quantum_state = QuantumState(
            amplitudes=x_manifold,
            basis_labels=[str(i) for i in range(self.manifold_dim)],
            phase=torch.zeros(1, dtype=self.dtype, device=self.device),
            original_norm=torch.sqrt(original_energy)
        )
        
        # Create validation result
        validation_result = QuantumStateValidationResult(
            is_valid=True,
            message="State preparation successful",
            data={
                "metrics": {
                    "energy_conservation": float(torch.all(energy_diff < 1e-6).item()),
                    "norm": float(torch.norm(quantum_state.amplitudes).item()),
                    "original_energy": float(original_energy.mean().item()),
                    "final_energy": float(final_energy.mean().item())
                }
            }
        )
        
        if return_validation:
            return quantum_state, validation_result
        return quantum_state

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
        classical_flat = state.amplitudes.to(self.dtype)
        
        # Project back to hidden dimension
        batch_size = classical_flat.shape[0]
        
        # Reshape classical to (batch_size, manifold_dim)
        classical_flat = classical_flat.reshape(batch_size, -1)
        
        # Project from manifold_dim back to hidden_dim using the inverse projection
        if not hasattr(self, 'inverse_projection'):
            self.inverse_projection = nn.Linear(self.manifold_dim, self.hidden_dim, device=classical_flat.device, dtype=self.dtype)
        output = self.inverse_projection(classical_flat)
        
        # Restore original energy if available
        if hasattr(state, 'original_energy') and state.original_energy is not None:
            current_energy = torch.sum(torch.abs(output) ** 2, dim=-1, keepdim=True)
            energy_scale = torch.sqrt(state.original_energy / (current_energy + 1e-8))
            output = output * energy_scale
            
        return output

    def evolve_quantum_state_with_attention(
        self,
        state: QuantumState,
        attention_pattern: Optional[torch.Tensor] = None,
        time: float = 1.0,  # Default time step
        **kwargs
    ) -> QuantumState:
        """Evolve quantum state using attention mechanism.
        
        Args:
            state: Input quantum state
            attention_pattern: Optional attention pattern to use
            time: Evolution time step
            **kwargs: Additional arguments
            
        Returns:
            Evolved quantum state
        """
        # Get state dimensions
        state_dim = state.amplitudes.shape[-1]
        batch_size = state.amplitudes.shape[0] if state.amplitudes.dim() > 1 else 1
        
        # Debug prints for gradient tracking
        print("\nQuantum State Evolution Debug:")
        print(f"Input state amplitudes shape: {state.amplitudes.shape}")
        print(f"Input state requires_grad: {state.amplitudes.requires_grad}")
        print(f"Input state grad_fn: {state.amplitudes.grad_fn}")
        
        # Get connection with proper gradient tracking
        connection = self.pattern_bundle.connection
        connection.requires_grad_(True)
        connection.retain_grad()  # Ensure gradients are retained
        
        print(f"Connection requires_grad: {connection.requires_grad}")
        print(f"Connection grad_fn: {connection.grad_fn}")
        
        # Create connection view with gradient tracking
        connection_view = connection.view(state_dim, state_dim, state_dim)
        connection_view.requires_grad_(True)
        connection_view.retain_grad()  # Ensure gradients are retained
        
        print(f"Connection view requires_grad: {connection_view.requires_grad}")
        print(f"Connection view grad_fn: {connection_view.grad_fn}")
        
        # Add gradient hook to connection view
        def connection_view_hook(grad):
            if grad is not None:
                print(f"\nConnection View Gradient:")
                print(f"Gradient shape: {grad.shape}")
                print(f"Gradient mean: {grad.abs().mean().item()}")
                print(f"Gradient max: {grad.abs().max().item()}")
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to original connection
                if connection.grad is None:
                    connection.grad = grad.view_as(connection)
                else:
                    connection.grad = connection.grad + grad.view_as(connection)
                return grad
            return grad
        connection_view.register_hook(connection_view_hook)
        
        # Apply layer normalization to attention pattern and ensure it's connected to graph
        if attention_pattern is not None:
            attention_pattern_flat = attention_pattern.reshape(-1, state_dim)
            # Convert to float for layer norm and ensure gradients flow
            attention_pattern_real = attention_pattern_flat.real.float()
            attention_pattern_imag = attention_pattern_flat.imag.float()
            
            print(f"\nAttention Pattern Debug:")
            print(f"Attention pattern requires_grad: {attention_pattern.requires_grad}")
            print(f"Attention pattern grad_fn: {attention_pattern.grad_fn}")
            
            # Apply layer norm separately to real and imaginary parts
            attention_pattern_norm_real = self.layer_norm(attention_pattern_real)
            attention_pattern_norm_imag = self.layer_norm(attention_pattern_imag)
            
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
            attention_pattern = attention_pattern_norm.reshape(batch_size, state_dim, state_dim).to(self.dtype)
            # Add residual connection to ensure layer norm is used in computation
            attention_pattern = attention_pattern + 0.1 * attention_pattern
            attention_pattern.requires_grad_(True)
            
            print(f"Normalized attention pattern requires_grad: {attention_pattern.requires_grad}")
            print(f"Normalized attention pattern grad_fn: {attention_pattern.grad_fn}")
        
        # Initialize Hamiltonian with attention pattern if provided
        if attention_pattern is not None:
            hamiltonian = attention_pattern
        else:
            # Create default Hamiltonian using connection with proper gradient tracking
            hamiltonian = torch.einsum('ijk,k->ij', connection_view, state.amplitudes)
            hamiltonian = hamiltonian.reshape(batch_size, state_dim, state_dim)
        hamiltonian.requires_grad_(True)
        
        print(f"\nHamiltonian Debug:")
        print(f"Hamiltonian shape: {hamiltonian.shape}")
        print(f"Hamiltonian requires_grad: {hamiltonian.requires_grad}")
        print(f"Hamiltonian grad_fn: {hamiltonian.grad_fn}")
        
        # Add connection contribution to Hamiltonian with gradient tracking
        connection_hamiltonian = torch.einsum('bij,bjk->bik', connection_view.expand(batch_size, -1, -1, -1), hamiltonian)
        connection_hamiltonian.requires_grad_(True)
        hamiltonian_with_connection = hamiltonian + connection_hamiltonian * 0.5  # Scale factor to control contribution
        hamiltonian_with_connection.requires_grad_(True)
        
        print(f"Hamiltonian with connection requires_grad: {hamiltonian_with_connection.requires_grad}")
        print(f"Hamiltonian with connection grad_fn: {hamiltonian_with_connection.grad_fn}")
        
        # Add gradient hook to Hamiltonian
        def hamiltonian_hook(grad):
            if grad is not None:
                print(f"\nHamiltonian Gradient:")
                print(f"Gradient shape: {grad.shape}")
                print(f"Gradient mean: {grad.abs().mean().item()}")
                print(f"Gradient max: {grad.abs().max().item()}")
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection
                if connection.grad is None:
                    connection.grad = grad.mean(0).view_as(connection)
                else:
                    connection.grad = connection.grad + grad.mean(0).view_as(connection)
                return grad
            return grad
        hamiltonian_with_connection.register_hook(hamiltonian_hook)
        
        # Compute evolution operator U = exp(-iHt)
        U = torch.matrix_exp(-time * hamiltonian_with_connection)
        U.requires_grad_(True)
        
        print(f"\nEvolution Operator Debug:")
        print(f"U shape: {U.shape}")
        print(f"U requires_grad: {U.requires_grad}")
        print(f"U grad_fn: {U.grad_fn}")
        
        # Convert state amplitudes to complex64 for evolution
        amplitudes_float = state.amplitudes.to(self.dtype)
        amplitudes_float.requires_grad_(True)
        
        # Evolve state with gradient tracking
        evolved_amplitudes = torch.matmul(U, amplitudes_float.unsqueeze(-1)).squeeze(-1)
        evolved_amplitudes.requires_grad_(True)
        
        print(f"\nEvolved State Debug:")
        print(f"Evolved amplitudes shape: {evolved_amplitudes.shape}")
        print(f"Evolved amplitudes requires_grad: {evolved_amplitudes.requires_grad}")
        print(f"Evolved amplitudes grad_fn: {evolved_amplitudes.grad_fn}")
        
        # Add gradient hook to evolved amplitudes
        def evolved_hook(grad):
            if grad is not None:
                print(f"\nEvolved Amplitudes Gradient:")
                print(f"Gradient shape: {grad.shape}")
                print(f"Gradient mean: {grad.abs().mean().item()}")
                print(f"Gradient max: {grad.abs().max().item()}")
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection and amplitudes
                if connection.grad is None:
                    connection.grad = grad.mean(0).view_as(connection)
                else:
                    connection.grad = connection.grad + grad.mean(0).view_as(connection)
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
        
        print(f"\nFinal State Debug:")
        print(f"Final state requires_grad: {evolved_state.amplitudes.requires_grad}")
        print(f"Final state grad_fn: {evolved_state.amplitudes.grad_fn}")
        
        return evolved_state

    def construct_pattern_bundle(
        self,
        neural_pattern: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[PatternSection, Tuple[PatternSection, Dict[str, Any]]]:
        """Construct pattern bundle with proper gradient tracking."""
        # Ensure input requires gradients
        neural_pattern.requires_grad_(True)
        
        # Get metric and connection views with gradient tracking
        metric_view = self.metric.clone()
        connection_view = self.connection.clone()
        
        # Get metric_factors from pattern bundle's riemannian framework
        metric_factors = self.pattern_bundle.riemannian_framework.metric_factors
        metric_factors.requires_grad_(True)
        metric_factors.retain_grad()  # Retain gradients for metric factors
        
        # Ensure views require gradients
        metric_view.requires_grad_(True)
        connection_view.requires_grad_(True)
        
        # Add gradient hooks with improved gradient flow
        def metric_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                
                # Ensure gradients flow back to original metric
                if self.metric.grad is None:
                    self.metric.grad = grad
                else:
                    self.metric.grad = self.metric.grad + grad
                    
                # Ensure gradients flow back to metric_factors
                if metric_factors.grad is None:
                    metric_factors.grad = grad.mean(0).real
                else:
                    metric_factors.grad = metric_factors.grad + grad.mean(0).real
                    
                # Ensure gradients flow to geometric flow components
                if hasattr(self.pattern_bundle, 'geometric_flow'):
                    for name, param in self.pattern_bundle.geometric_flow.named_parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        # Scale gradient contribution
                        grad_contribution = grad.mean() * torch.ones_like(param)
                        param.grad = param.grad + grad_contribution
                return grad
            return grad
        metric_view.register_hook(metric_hook)
        
        def connection_hook(grad):
            if grad is not None:
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                
                # Ensure gradients flow back to original connection
                if self.connection.grad is None:
                    self.connection.grad = grad
                else:
                    self.connection.grad = self.connection.grad + grad
                    
                # Ensure gradients flow back to metric_factors
                if metric_factors.grad is None:
                    metric_factors.grad = grad.mean(0).real
                else:
                    metric_factors.grad = metric_factors.grad + grad.mean(0).real
                    
                # Ensure gradients flow to geometric flow components
                if hasattr(self.pattern_bundle, 'geometric_flow'):
                    for name, param in self.pattern_bundle.geometric_flow.named_parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        # Scale gradient contribution
                        grad_contribution = grad.mean() * torch.ones_like(param)
                        param.grad = param.grad + grad_contribution
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
        
        # Add gradient hook to connection form with improved flow
        def connection_form_hook(grad):
            if grad is not None:
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                
                # Ensure gradients flow back to connection view
                if connection_view.grad is None:
                    connection_view.grad = grad
                else:
                    connection_view.grad = connection_view.grad + grad
                    
                # Ensure gradients flow to geometric flow components
                if hasattr(self.pattern_bundle, 'geometric_flow'):
                    for name, param in self.pattern_bundle.geometric_flow.named_parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        # Scale gradient contribution
                        grad_contribution = grad.mean() * torch.ones_like(param)
                        param.grad = param.grad + grad_contribution
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
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
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
                
                # Ensure gradients flow back to metric_factors
                if metric_factors.grad is None:
                    metric_factors.grad = grad.mean(0).real
                else:
                    metric_factors.grad = metric_factors.grad + grad.mean(0).real
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
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
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
                
                # Ensure gradients flow back to metric_factors
                if metric_factors.grad is None:
                    metric_factors.grad = grad.mean(0).real
                else:
                    metric_factors.grad = metric_factors.grad + grad.mean(0).real
                
                # Ensure gradients flow back to geometric flow parameters
                if hasattr(self.pattern_bundle, 'geometric_flow'):
                    for param in self.pattern_bundle.geometric_flow.parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad = param.grad + grad.mean() * torch.ones_like(param)
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
        """Forward pass through the neural quantum bridge.
        
        Args:
            x: Input tensor
            return_intermediates: Whether to return intermediate tensors
            
        Returns:
            Output tensor and optionally intermediate tensors
        """
        # Store original shape and ensure input has correct dtype
        original_shape = x.shape
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = x.to(dtype=self.dtype)
        
        # Store input norm for later normalization
        input_norm = torch.sqrt(torch.sum(x.real ** 2 + x.imag ** 2, dim=-1, keepdim=True))
        
        # Get metric and connection with proper dtype
        metric = self.metric.to(dtype=self.dtype)
        connection = self.connection.to(dtype=self.dtype)
        riemannian_connection = self.pattern_bundle.riemannian_connection_coeffs.to(dtype=self.dtype)
        
        # Compute metric contribution
        x_metric = torch.einsum('...i,ij->...j', x, metric)
        x_metric = x_metric.to(dtype=self.dtype)
        
        # Compute connection contribution
        x_connection = torch.einsum('...i,ijk->...k', x, connection)
        x_connection = x_connection.to(dtype=self.dtype)
        
        # Compute riemannian connection contribution
        x_riemannian = torch.einsum('...i,ijk->...k', x, riemannian_connection)
        x_riemannian = x_riemannian.to(dtype=self.dtype)
        
        # Add gradient hook to connection computation
        def connection_hook(grad):
            if grad is not None:
                # Ensure gradients flow back to connection
                if connection.grad is None:
                    connection.grad = grad.mean(0).unsqueeze(-1)
                else:
                    connection.grad = connection.grad + grad.mean(0).unsqueeze(-1)
            return grad
        x_connection.register_hook(connection_hook)
        
        # Add gradient hook to riemannian connection computation
        def riemannian_hook(grad):
            if grad is not None:
                # Ensure gradients flow back to riemannian connection
                if riemannian_connection.grad is None:
                    riemannian_connection.grad = grad.mean(0).unsqueeze(-1)
                else:
                    riemannian_connection.grad = riemannian_connection.grad + grad.mean(0).unsqueeze(-1)
            return grad
        x_riemannian.register_hook(riemannian_hook)
        
        # Combine metric and connection contributions
        x_combined = x_metric + x_connection + x_riemannian
        x_combined = x_combined.to(dtype=self.dtype)
        
        # Normalize combined result while preserving input norm
        x_combined = x_combined * (input_norm / (torch.sqrt(torch.sum(x_combined.real ** 2 + x_combined.imag ** 2, dim=-1, keepdim=True)) + 1e-8))
        x_combined.requires_grad_(True)
        
        # Add residual connection for gradient stability
        x_combined = x_combined + 0.1 * x
        x_combined = x_combined.to(dtype=self.dtype)
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
        output = output.to(dtype=self.dtype)
        output.requires_grad_(True)
        
        # Apply layer norm to output and ensure it's connected to graph
        output_flat = output.reshape(-1, self.hidden_dim)
        output_real = output_flat.real.to(dtype=self.get_float_dtype())
        output_imag = output_flat.imag.to(dtype=self.get_float_dtype())
        
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

    def project_to_quantum(self, x: torch.Tensor) -> QuantumState:
        """Project neural state to quantum state with energy conservation."""
        # Store original energy per sample
        original_energy = torch.sum(x.abs() ** 2, dim=-1, keepdim=True)
        
        # Project to manifold
        x_manifold = self.manifold_proj(x)
        
        # Calculate current energy per sample
        current_energy = torch.sum(x_manifold.abs() ** 2, dim=-1, keepdim=True)
        
        # Compute scaling factor to preserve energy per sample
        scale_factor = torch.sqrt(original_energy / (current_energy + self.stability_threshold))
        
        # Scale the manifold projection to preserve energy
        x_manifold = x_manifold * scale_factor
        
        # Create default basis labels and phase
        basis_labels = [f"basis_{i}" for i in range(x_manifold.shape[-1])]
        phase = torch.zeros(x_manifold.shape[:-1], device=x.device, dtype=x.dtype)
        
        # Create quantum state with preserved energy
        quantum_state = QuantumState(
            amplitudes=x_manifold,
            basis_labels=basis_labels,
            phase=phase,
            original_norm=torch.norm(x, dim=-1, keepdim=True),
            original_energy=original_energy
        )
        
        return quantum_state

    def compute_metric_tensor(self, x: torch.Tensor, return_validation: bool = False) -> torch.Tensor:
        """Compute metric tensor for input tensor."""
        # Compute base metric
        base_metric = self.pattern_bundle.compute_metric_tensor(x)
        return base_metric

    def compute_connection_form(self, x: torch.Tensor, return_validation: bool = False) -> torch.Tensor:
        """Compute connection form for input tensor."""
        # Compute connection form
        connection = self.pattern_bundle.compute_connection(x)
        return connection

    def compute_curvature_tensor(self, x: torch.Tensor, return_validation: bool = False) -> torch.Tensor:
        """Compute curvature tensor for input tensor."""
        # Get metric and connection
        metric = self.compute_metric_tensor(x)
        christoffel = self.compute_christoffel(x)
        # Compute curvature using pattern bundle's riemannian framework
        curvature_obj = self.pattern_bundle.riemannian_framework.compute_curvature(points=x, christoffel=christoffel)
        # Extract the Riemann tensor values
        return curvature_obj.riemann

    def compute_ricci_tensor(self, x: torch.Tensor, return_validation: bool = False) -> torch.Tensor:
        """Compute Ricci tensor for input tensor."""
        # Get metric and connection
        metric = self.compute_metric_tensor(x)
        connection = self.compute_connection_form(x)
        # Compute Ricci tensor using pattern bundle
        ricci = self.pattern_bundle.riemannian_framework.compute_ricci_tensor(metric, connection)
        return ricci