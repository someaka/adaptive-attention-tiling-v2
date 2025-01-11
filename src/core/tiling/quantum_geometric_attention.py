"""Quantum Geometric Attention Framework.

This module integrates:
- Quantum Motivic Structure
- Arithmetic Dynamics
- Geometric Flow
- Pattern Recognition
- Advanced Geometric Structures
- Pattern Dynamics

Into a unified framework for understanding computational patterns
through the lens of quantum geometry and arithmetic dynamics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, cast

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

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
from  src.neural.attention.pattern.pattern_dynamics import PatternDynamics
from src.core.tiling.state_manager import StateManager, StateConfig, StateType
from src.core.tiling.attention_state import AttentionState
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.metrics.attention import (
    AttentionMetrics,
    FlowMetrics,
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
    PatternMetrics
)
from src.validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    QuantumStateValidationResult,
    StateValidationErrorType
)
from src.validation.patterns.formation import PatternFormationValidator
from ..patterns.fiber_types import LocalChart as PatternSection

# Configure logging
logger = logging.getLogger(__name__)

class EnergyConservingLinear(nn.Module):
    """Linear layer that conserves energy during forward pass."""
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.complex64, device: Optional[torch.device] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight and bias with correct dimensions [out_features, in_features]
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.weight = nn.Parameter(torch.complex(
            torch.randn(out_features, in_features, dtype=real_dtype, device=device) / np.sqrt(in_features),
            torch.randn(out_features, in_features, dtype=real_dtype, device=device) / np.sqrt(in_features)
        ))
        self.bias = nn.Parameter(torch.complex(
            torch.zeros(out_features, dtype=real_dtype, device=device),
            torch.zeros(out_features, dtype=real_dtype, device=device)
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with energy conservation."""
        # Compute initial energy
        initial_energy = torch.sum(input.abs() ** 2, dim=-1, keepdim=True)

        # Apply linear transformation
        output = F.linear(input, self.weight, self.bias)  # weight shape is [out_features, in_features]

        # Compute output energy
        output_energy = torch.sum(output.abs() ** 2, dim=-1, keepdim=True)

        # Scale output to conserve energy
        scale_factor = torch.sqrt(initial_energy / (output_energy + 1e-8))
        output = output * scale_factor

        return output

    def reset_parameters(self):
        """Reset the parameters to their initial values."""
        nn.init.kaiming_uniform_(self.weight.real, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weight.imag, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.real)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias.real, -bound, bound)
        nn.init.uniform_(self.bias.imag, -bound, bound)


class QuantumGeometricAttention(nn.Module):
    """Quantum geometric attention framework."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        manifold_type: str = "hyperbolic",
        curvature: float = -1.0,
        manifold_dim: Optional[int] = None,
        num_layers: int = 3,
        tile_size: int = 8,
        motive_rank: int = 4,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        is_causal: bool = False,
    ):
        """Initialize quantum geometric attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            manifold_type: Type of manifold to use
            curvature: Manifold curvature
            manifold_dim: Manifold dimension (defaults to hidden_dim)
            num_layers: Number of attention layers
            tile_size: Size of attention tiles
            motive_rank: Rank of motive
            dtype: Data type
            device: Device to use
            is_causal: Whether to use causal attention
        """
        super().__init__()
        
        # Add shape validation flag
        self.validate_shapes = True
        self.shape_errors = []
        
        # Store parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.manifold_dim = manifold_dim if manifold_dim is not None else hidden_dim
        self.num_layers = num_layers
        self.tile_size = tile_size
        self.motive_rank = motive_rank
        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')
        self.is_causal = is_causal
        
        # Validate dtype
        if dtype not in [torch.complex64, torch.complex128]:
            raise ValueError("dtype must be either torch.complex64 or torch.complex128")

        # Initialize manifold projections with complex dtype and correct dimensions
        self.manifold_proj = EnergyConservingLinear(self.hidden_dim, self.manifold_dim, dtype=self.dtype, device=self.device)
        self.manifold_proj_inv = EnergyConservingLinear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        # Initialize quantum bridge with complex dtype
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            manifold_type=manifold_type,
            curvature=curvature,
            dtype=self.dtype,
            device=device
        )
        
        # Add energy conservation hook to quantum bridge
        def quantum_bridge_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                return grad
            return grad
        
        for param in self.quantum_bridge.parameters():
            param.register_hook(quantum_bridge_hook)
        
        # Initialize tiles with complex dtype and proper manifold dimension
        self.tiles = nn.ModuleList([
            QuantumMotivicTile(
                size=tile_size,
                hidden_dim=self.manifold_dim,  # Use manifold_dim for tiles
                num_heads=1,
                dropout=dropout,
                resolution=1.0,
                cohomology_dim=self.manifold_dim,
                motive_rank=motive_rank,
                dtype=self.dtype
            )
            for _ in range(num_heads)
        ])
        
        # Initialize attention layers with complex dtype
        self.attention_layers = nn.ModuleList([
            nn.Linear(self.manifold_dim, self.manifold_dim, dtype=self.dtype, device=device)
            for _ in range(num_layers)
        ])
        
        # Initialize metric tensors with proper manifold dimension
        self.base_metric = nn.Parameter(
            torch.eye(self.manifold_dim, dtype=self.dtype, device=device),
            requires_grad=True
        )
        self.metric = nn.Parameter(
            torch.eye(self.manifold_dim, dtype=self.dtype, device=device),
            requires_grad=True
        )
        self.pattern_metric = nn.Parameter(
            torch.eye(self.manifold_dim, dtype=self.dtype, device=device),
            requires_grad=True
        )
        self.combined_metric = nn.Parameter(
            torch.eye(self.manifold_dim, dtype=self.dtype, device=device),
            requires_grad=True
        )
        
        # Initialize flow with proper manifold dimension
        self.flow = GeometricFlow(
            hidden_dim=hidden_dim,
            manifold_dim=self.manifold_dim,  # Use manifold_dim for flow
            motive_rank=motive_rank,
            num_charts=4,
            integration_steps=10,
            dt=0.1,
            stability_threshold=1e-6,
            dtype=self.dtype
        )
        
        # Initialize dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize pattern bundle
        self.pattern_bundle = PatternRiemannianStructure(
            manifold_dim=self.manifold_dim,
            pattern_dim=self.manifold_dim,  # Use manifold_dim for pattern dimension
            device=self.device,
            dtype=self.dtype
        )
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad_(True)
            
        # Initialize complex weights
        self._init_complex_weights()

    def _init_complex_weights(self):
        """Initialize weights with proper complex values."""
        def init_complex_linear(layer):
            if isinstance(layer, nn.Linear):
                # Initialize real and imaginary parts separately
                weight_shape = layer.weight.shape
                std = 1.0 / math.sqrt(weight_shape[1])
                
                # Initialize real and imaginary parts with proper float dtype
                float_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
                real_weight = torch.randn(weight_shape, device=self.device, dtype=float_dtype) * std
                imag_weight = torch.randn(weight_shape, device=self.device, dtype=float_dtype) * std
                
                # Create complex weight tensor with proper complex dtype
                complex_weight = torch.complex(real_weight, imag_weight).to(self.dtype)
                layer.weight = nn.Parameter(complex_weight)
                
                if layer.bias is not None:
                    real_bias = torch.randn(weight_shape[0], device=self.device, dtype=float_dtype) * std
                    imag_bias = torch.randn(weight_shape[0], device=self.device, dtype=float_dtype) * std
                    
                    # Create complex bias tensor with proper complex dtype
                    complex_bias = torch.complex(real_bias, imag_bias).to(self.dtype)
                    layer.bias = nn.Parameter(complex_bias)

        # Initialize all attention layers
        for layer in self.attention_layers:
            init_complex_linear(layer)
        
        # Initialize manifold projections
        init_complex_linear(self.manifold_proj)
        init_complex_linear(self.manifold_proj_inv)

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize manifold projections with real weights first
        with torch.no_grad():
            # Get corresponding float dtype
            float_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
            
            # Create real orthogonal matrices with proper float dtype
            real_weight = torch.empty_like(self.manifold_proj.weight.real, dtype=float_dtype)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(self.manifold_proj.weight.imag, dtype=float_dtype)
            nn.init.orthogonal_(imag_weight)
            
            # Combine into complex weight with proper dtype
            self.manifold_proj.weight.copy_(torch.complex(real_weight, imag_weight).to(self.dtype))
            self.manifold_proj.bias.zero_()
            
            # Same for inverse projection
            real_weight = torch.empty_like(self.manifold_proj_inv.weight.real, dtype=float_dtype)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(self.manifold_proj_inv.weight.imag, dtype=float_dtype)
            nn.init.orthogonal_(imag_weight)
            self.manifold_proj_inv.weight.copy_(torch.complex(real_weight, imag_weight).to(self.dtype))
            self.manifold_proj_inv.bias.zero_()

    def compute_fisher_information(self, states: torch.Tensor) -> torch.Tensor:
        """Compute Fisher information metric for states."""
        params = list(self.parameters())  # Convert iterator to list
        grad_log_p = torch.autograd.grad(
            states.sum(), params, create_graph=True, retain_graph=True
        )
        return torch.stack([g.pow(2).sum() for g in grad_log_p])

    def detect_patterns(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Detect patterns in input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            - Pattern tensor of shape (batch_size, seq_len, hidden_dim)
            - Combined metrics dictionary
        """
        batch_size, seq_len, _ = x.shape
        metrics = {}

        # 1. Apply arithmetic pattern detection
        arithmetic_out, arithmetic_layer_metrics = self.arithmetic(x)
        metrics["arithmetic"] = arithmetic_layer_metrics

        # 2. Apply geometric flow
        flow_out, flow_layer_metrics = self.flow(arithmetic_out)
        metrics["flow"] = flow_layer_metrics

        # 3. Convert to quantum state
        quantum_state = self.classical_to_quantum(flow_out)
        
        # 4. Apply quantum pattern detection
        quantum_patterns = []
        quantum_metrics = []
        
        for h, tile in enumerate(self.tiles):
            # Process through quantum tile
            tile_pattern, tile_metrics = tile(
                quantum_state,
                quantum_state,
                quantum_state,
                return_metrics=True
            )
            quantum_patterns.append(tile_pattern)
            quantum_metrics.append(tile_metrics)
        
        # Combine quantum patterns
        quantum_out = torch.stack(quantum_patterns, dim=1).mean(dim=1)
        metrics["quantum"] = {
            f"tile_{i}": m for i, m in enumerate(quantum_metrics)
        }

        # 5. Project to pattern space with geometric structure
        patterns = self.pattern_proj(quantum_out)
        
        # 6. Apply parallel transport for geometric consistency
        transported_patterns = self._apply_parallel_transport(patterns)
        
        # 7. Compute geometric features
        geometric_features = self._compute_geometric_features(transported_patterns)
        
        # 8. Combine with quantum features
        quantum_features = self._compute_quantum_features(quantum_out)
        final_patterns = geometric_features + quantum_features
        
        # 9. Compute pattern metrics
        pattern_metrics = {
            "entropy": self.compute_entropy(final_patterns).mean().item(),
            "complexity": self.compute_complexity(final_patterns).mean().item(),
            "stability": self.compute_stability(final_patterns).mean().item(),
            "sparsity": self.compute_sparsity(final_patterns).mean().item()
        }
        metrics["patterns"] = pattern_metrics

        return final_patterns, metrics

    def complex_softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply softmax to complex tensor by using the absolute values.
        
        Args:
            x: Complex tensor
            dim: Dimension along which to apply softmax
            
        Returns:
            Complex tensor with softmax applied to absolute values
        """
        abs_x = x.abs()
        max_val = torch.max(abs_x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(abs_x - max_val)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        softmax_abs = exp_x / (sum_exp_x + 1e-8)
        return x * (softmax_abs / (abs_x + 1e-8))

    def complex_dropout(self, x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Apply dropout to complex tensor by handling real and imaginary parts separately.
        
        Args:
            x: Complex input tensor
            p: Dropout probability
            
        Returns:
            Complex tensor with dropout applied
        """
        if not self.training or p == 0:
            return x
            
        mask = torch.ones_like(x.real, device=x.device)
        mask = F.dropout(mask, p=p, training=True, inplace=False)
        return x * mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with shape validation."""
        self._clear_shape_errors()
        
        try:
            # Validate input shape
            self._check_input_shapes(x)
            
            # Original forward pass code with added shape checks
            batch_size, seq_len, _ = x.shape
            
            # Store original input shape and energy for gradient flow
            self.x_flat = x.reshape(-1, x.shape[-1])
            self._validate_shape(
                self.x_flat,
                (batch_size * seq_len, self.hidden_dim),
                "x_flat"
            )
            
            # Store original input shape and energy for gradient flow
            self.x_flat = x.reshape(-1, x.shape[-1])  # Store x_flat as class attribute
            self.x_flat.requires_grad_(True)  # Ensure requires_grad is True
            self.x_flat.retain_grad()  # Retain gradients for x_flat
            initial_energy = torch.sum(self.x_flat.abs() ** 2)
            print(f"x_flat requires_grad: {self.x_flat.requires_grad}")
            
            def normalize_with_energy(tensor, target_energy):
                """Normalize tensor while preserving target energy and phase information."""
                # Handle both real and complex tensors
                if torch.is_complex(tensor):
                    current_energy = torch.sum(tensor.real ** 2 + tensor.imag ** 2, dim=-1, keepdim=True)
                else:
                    current_energy = torch.sum(tensor ** 2, dim=-1, keepdim=True)
                
                # Ensure scale is properly broadcast
                scale = torch.sqrt(target_energy / (current_energy + 1e-8))
                if scale.dim() < tensor.dim():
                    scale = scale.view(*scale.shape, *([1] * (tensor.dim() - scale.dim())))
                
                # Scale tensor while preserving phase
                return tensor * scale

            # Convert input to complex with proper dtype
            if not torch.is_complex(self.x_flat):
                # If input is real, convert to complex
                float_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
                x_real = self.x_flat.to(float_dtype)
                x_imag = torch.zeros_like(x_real, dtype=float_dtype)
                x_complex = torch.complex(x_real, x_imag).to(self.dtype)
            else:
                # If input is already complex, just ensure it has the right dtype
                x_complex = self.x_flat.to(self.dtype)

            # Project input to manifold space using x_flat
            x_manifold = self.manifold_proj(x_complex.reshape(batch_size, seq_len, -1))
            
            # Calculate initial energy per sequence position
            initial_energy = torch.sum(x_complex.reshape(batch_size, seq_len, -1).real ** 2 + 
                             x_complex.reshape(batch_size, seq_len, -1).imag ** 2, 
                             dim=-1, keepdim=True)
            
            # Ensure x_manifold is complex
            if not torch.is_complex(x_manifold):
                x_manifold = torch.complex(x_manifold, torch.zeros_like(x_manifold))
            
            # Normalize manifold projection to match initial energy
            x_manifold = normalize_with_energy(x_manifold, initial_energy)
            print(f"After manifold projection shape: {x_manifold.shape}")
            print(f"x_manifold requires_grad: {x_manifold.requires_grad}")
            
            # Initialize base metric tensor with proper dimensions
            base_metric = self.base_metric.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim)
            base_metric.requires_grad_(True)

            # Use pattern_metric parameter with proper dimensions
            pattern_metric = self.pattern_metric.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim)
            pattern_metric.requires_grad_(True)
            pattern_metric.retain_grad()  # Retain gradients for pattern_metric

            # Use metric parameter with proper dimensions
            metric = self.metric.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim)
            metric.requires_grad_(True)
            metric.retain_grad()  # Retain gradients for metric

            # Get metric_factors from riemannian_framework and ensure gradient tracking
            metric_factors = self.quantum_bridge.pattern_bundle.riemannian_framework.metric_factors
            metric_factors.requires_grad_(True)
            metric_factors.retain_grad()  # Ensure gradients are retained

            # Create metric combination with gradient tracking
            metric_combination = base_metric + 0.5 * (metric + pattern_metric)
            metric_combination.requires_grad_(True)

            # Add metric_factors contribution while preserving dimensions
            metric_factors_reshaped = metric_factors.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim)
            metric_factors_contribution = torch.matmul(metric_factors_reshaped, metric_factors_reshaped.t())
            metric_combination = metric_combination + 0.1 * metric_factors_contribution
            metric_combination.requires_grad_(True)

            # Add gradient hook to metric combination
            def metric_combination_hook(grad):
                if grad is not None:
                    # Create full-size gradient for pattern_metric
                    full_grad = torch.zeros_like(self.pattern_metric)
                    full_grad.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim).copy_(0.5 * grad)
                    # Ensure gradients flow back to pattern_metric
                    if self.pattern_metric.grad is None:
                        self.pattern_metric.grad = full_grad
                    else:
                        self.pattern_metric.grad = self.pattern_metric.grad + full_grad

                    # Create full-size gradient for metric_factors
                    metric_factors_grad = torch.zeros_like(metric_factors)
                    # Compute the contribution for the top-left block
                    top_left_grad = 0.1 * grad @ metric_factors_reshaped
                    metric_factors_grad.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim).copy_(top_left_grad)
                    # Ensure gradients flow back to metric_factors
                    if metric_factors.grad is None:
                        metric_factors.grad = metric_factors_grad
                    else:
                        metric_factors.grad = metric_factors.grad + metric_factors_grad
                    return grad
                return grad
            metric_combination.register_hook(metric_combination_hook)
            
            # Create new combined metric tensor
            combined_metric = metric_combination + self.combined_metric.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim)
            combined_metric.requires_grad_(True)
            combined_metric.retain_grad()  # Retain gradients for combined metric
            
            # Add gradient hook to combined metric
            def combined_metric_hook(grad):
                if grad is not None:
                    # Create full-size gradient
                    full_grad = torch.zeros_like(self.pattern_metric)
                    full_grad.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim).copy_(0.5 * grad)
                    # Ensure gradients flow back to pattern_metric
                    if self.pattern_metric.grad is None:
                        self.pattern_metric.grad = full_grad
                    else:
                        self.pattern_metric.grad = self.pattern_metric.grad + full_grad
                    return grad
                return grad
            combined_metric.register_hook(combined_metric_hook)

            # Apply metric with gradient stabilization
            x_metric = torch.matmul(x_manifold, combined_metric)
            metric_norm = torch.norm(combined_metric)
            x_metric = x_metric * (1.0 + 1e-6 * metric_norm)
            x_metric = torch.matmul(x_metric, combined_metric)
            x_metric.requires_grad_(True)
            
            # Add gradient hook to x_metric
            def x_metric_hook(grad):
                if grad is not None:
                    # Take mean over batch dimension to match combined_metric shape
                    batch_grad = torch.mean(grad.transpose(-2, -1) @ x_manifold, dim=0)
                    # Create full-size gradient with zeros
                    full_grad = torch.zeros_like(self.quantum_bridge.pattern_bundle_metric)
                    # Copy the batch_grad into the top-left quadrant of full_grad
                    full_grad[:self.manifold_dim, :self.manifold_dim] = batch_grad
                    # Ensure gradients flow back to pattern_metric through combined_metric
                    if combined_metric.grad is None:
                        combined_metric.grad = batch_grad
                    else:
                        combined_metric.grad = combined_metric.grad + batch_grad
                    # Also ensure gradients flow to pattern bundle metric
                    if self.quantum_bridge.pattern_bundle_metric.grad is None:
                        self.quantum_bridge.pattern_bundle_metric.grad = full_grad
                    else:
                        self.quantum_bridge.pattern_bundle_metric.grad = self.quantum_bridge.pattern_bundle_metric.grad + full_grad
                    return grad
                return grad
            x_metric.register_hook(x_metric_hook)
            
            print(f"metric requires_grad: {combined_metric.requires_grad}")
            print(f"pattern_metric requires_grad: {pattern_metric.requires_grad}")
            print(f"x_metric requires_grad: {x_metric.requires_grad}")
            
            # Get connection and ensure consistent types
            connection = self.quantum_bridge.pattern_bundle.connection
            if connection.shape[-1] != self.manifold_dim:
                # Project connection to manifold dimension using narrow
                connection = connection.narrow(-3, 0, self.manifold_dim).narrow(-2, 0, self.manifold_dim).narrow(-1, 0, self.manifold_dim)
            connection.requires_grad_(True)  # Ensure connection requires gradients
            connection.retain_grad()  # Retain gradients for connection
            print(f"connection requires_grad: {connection.requires_grad}")
            
            # Ensure pattern bundle metric requires gradients
            pattern_bundle_metric = self.quantum_bridge.pattern_bundle_metric.narrow(0, 0, self.manifold_dim).narrow(1, 0, self.manifold_dim)
            pattern_bundle_metric.requires_grad_(True)
            pattern_bundle_metric.retain_grad()  # Retain gradients for pattern metric
            
            # Convert x_metric to complex if connection is complex
            if torch.is_complex(connection):
                x_metric = x_metric.to(dtype=torch.complex64)
            
            # Apply connection to input with gradient tracking
            x_with_connection = x_metric + torch.einsum('bsi,ijk->bsj', x_metric, connection)

            # Create stronger gradient path for connection
            connection_scale = connection.abs().mean()
            x_with_connection = x_with_connection * (1.0 + connection_scale * 1e-3)

            # Add direct gradient path from connection to output
            x_with_connection = x_with_connection + torch.einsum('bsi,ijk,bsk->bsj', x_metric, connection, x_metric) * 1e-3

            # Normalize to preserve energy
            x_with_connection_norm = torch.sqrt(torch.sum(x_with_connection.abs() ** 2))
            x_with_connection = x_with_connection * torch.sqrt(initial_energy) / (x_with_connection_norm + 1e-8)
            print(f"x_with_connection requires_grad: {x_with_connection.requires_grad}")
            
            # Process through tiles
            tile_outputs = []
            for i, tile in enumerate(self.tiles):
                tile_output = tile(x_with_connection)
                print(f"Tile {i} output shape: {tile_output.shape}")
                print(f"Tile {i} output requires_grad: {tile_output.requires_grad}")
                tile_outputs.append(tile_output)
            
            # Stack tile outputs
            stacked_outputs = torch.stack(tile_outputs, dim=2)  # [batch_size, seq_len, num_heads, manifold_dim]
            print(f"After stacking tile outputs shape: {stacked_outputs.shape}")
            print(f"stacked_outputs requires_grad: {stacked_outputs.requires_grad}")
            
            # Apply mask if provided
            if mask is not None:
                # Expand mask to match stacked_outputs shape
                expanded_mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
                # Transpose stacked_outputs to match mask dimensions
                stacked_outputs = stacked_outputs.transpose(1, 2)  # [batch_size, num_heads, seq_len, manifold_dim]
                expanded_mask = expanded_mask.expand(-1, stacked_outputs.size(1), -1, stacked_outputs.size(-1))
                stacked_outputs = stacked_outputs.masked_fill(~expanded_mask, 0.0)
                # Transpose back to original shape
                stacked_outputs = stacked_outputs.transpose(1, 2)  # [batch_size, seq_len, num_heads, manifold_dim]
            
            # Mean over heads with energy preservation
            mean_output = stacked_outputs.mean(dim=2)  # [batch_size, seq_len, manifold_dim]
            mean_output = normalize_with_energy(mean_output, initial_energy)
            print(f"After mean over heads shape: {mean_output.shape}")
            print(f"mean_output requires_grad: {mean_output.requires_grad}")

            # Apply quantum attention layers with energy preservation
            quantum_output = mean_output
            for i, layer in enumerate(self.attention_layers):
                quantum_output = layer(quantum_output)
                quantum_output = normalize_with_energy(quantum_output, initial_energy)
                print(f"After quantum attention layer {i} shape: {quantum_output.shape}")
                print(f"quantum_output layer {i} requires_grad: {quantum_output.requires_grad}")
            
            # Handle complex tensor for layer norm with energy preservation
            if torch.is_complex(quantum_output):
                # Compute current energy before layer norm
                pre_norm_energy = torch.sum(quantum_output.abs() ** 2)
                
                # Apply layer norm to real and imaginary parts separately
                real_part = quantum_output.real.float()
                imag_part = quantum_output.imag.float()
                
                # Apply layer norm with correct shape
                layer_norm = self.quantum_bridge.layer_norm
                layer_norm.weight = nn.Parameter(layer_norm.weight[:self.manifold_dim])
                layer_norm.bias = nn.Parameter(layer_norm.bias[:self.manifold_dim])
                real_norm = F.layer_norm(
                    real_part,
                    normalized_shape=[self.manifold_dim],
                    weight=layer_norm.weight,
                    bias=layer_norm.bias
                )
                imag_norm = F.layer_norm(
                    imag_part,
                    normalized_shape=[self.manifold_dim],
                    weight=layer_norm.weight,
                    bias=layer_norm.bias
                )
                
                # Combine back to complex
                quantum_output_norm = torch.complex(real_norm, imag_norm)
                
                # Restore original energy
                quantum_output_norm = normalize_with_energy(quantum_output_norm, pre_norm_energy)
            else:
                # If real tensor, apply layer norm directly and preserve energy
                pre_norm_energy = torch.sum(quantum_output ** 2)
                layer_norm = self.quantum_bridge.layer_norm
                layer_norm.weight = nn.Parameter(layer_norm.weight[:self.manifold_dim])
                layer_norm.bias = nn.Parameter(layer_norm.bias[:self.manifold_dim])
                quantum_output_norm = F.layer_norm(
                    quantum_output.float(),
                    normalized_shape=[self.manifold_dim],
                    weight=layer_norm.weight,
                    bias=layer_norm.bias
                )
                quantum_output_norm = normalize_with_energy(quantum_output_norm, pre_norm_energy)

            # Add residual connection with energy preservation
            residual_weight = 0.1
            combined_output = quantum_output_norm + residual_weight * quantum_output
            combined_output = normalize_with_energy(combined_output, initial_energy)
            
            # Apply flow operation and unpack output
            flow_output, _ = self.flow(combined_output)
            if torch.is_complex(flow_output):
                flow_output = flow_output.real
            flow_output = normalize_with_energy(flow_output, initial_energy)
            print(f"flow_output requires_grad: {flow_output.requires_grad}")
            
            # Convert flow_output to complex before inverse projection
            if not torch.is_complex(flow_output):
                flow_output = torch.complex(flow_output, torch.zeros_like(flow_output))
            flow_output = flow_output.to(dtype=self.dtype)
            
            # Calculate energy before inverse projection
            pre_inverse_energy = torch.sum(flow_output.real ** 2 + flow_output.imag ** 2, 
                                         dim=-1, keepdim=True)
            
            # Inverse manifold projection with energy preservation
            output = self.manifold_proj_inv(flow_output)
            
            # Ensure output is complex
            if not torch.is_complex(output):
                output = torch.complex(output, torch.zeros_like(output))
            output = normalize_with_energy(output, pre_inverse_energy)
            print(f"After inverse manifold projection shape: {output.shape}")
            print(f"output requires_grad: {output.requires_grad}")
            
            # Add final residual connection with energy preservation
            x_flat_reshaped = self.x_flat.reshape(output.shape)
            
            # Ensure x_flat_reshaped is complex
            if not torch.is_complex(x_flat_reshaped):
                x_flat_reshaped = torch.complex(x_flat_reshaped, torch.zeros_like(x_flat_reshaped))
            x_flat_reshaped = x_flat_reshaped.to(dtype=self.dtype)
            
            # Calculate energies
            output_energy = torch.sum(output.real ** 2 + output.imag ** 2, dim=-1, keepdim=True)
            residual_energy = torch.sum(x_flat_reshaped.real ** 2 + x_flat_reshaped.imag ** 2, 
                                      dim=-1, keepdim=True)
            
            # Scale residual to match output energy
            residual_weight = 0.1
            scaled_residual = normalize_with_energy(x_flat_reshaped, output_energy) * residual_weight
            
            # Combine and normalize to preserve output energy
            output = output + scaled_residual
            output = normalize_with_energy(output, output_energy)
            
            # Create direct path for gradient flow while preserving energy
            if self.training:
                # Calculate connection scale while preserving complex structure
                connection_scale = torch.sum(connection.real ** 2 + connection.imag ** 2, dim=(-3, -2, -1), keepdim=True)
                output_with_connection = output * (1.0 + 1e-6 * connection_scale)
                output_with_connection = normalize_with_energy(output_with_connection, output_energy)
                
                # Calculate input scale while preserving complex structure
                x_flat_scale = torch.sum(x_flat_reshaped.real ** 2 + x_flat_reshaped.imag ** 2, dim=-1, keepdim=True)
                output_with_input = output_with_connection * (1.0 + 1e-6 * x_flat_scale)
                output = normalize_with_energy(output_with_input, output_energy)
            
            # Validate final output shape
            self._validate_shape(
                output,
                (batch_size, seq_len, self.hidden_dim),
                "final_output"
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            logger.error("Shape errors accumulated:")
            for error in self.shape_errors:
                logger.error(error)
            raise

    def prepare_attention_state(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> AttentionState:
        """Prepare attention state from input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            AttentionState object containing quantum state
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to manifold space
        x_manifold = self.manifold_proj(x)  # [batch_size, seq_len, manifold_dim]
        
        # Normalize quantum state
        quantum_state = self.normalize_complex_tensor(x_manifold)
        
        # Reshape for multi-head attention
        quantum_state = quantum_state.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Create state manager with config
        config = StateConfig(
            dim=self.hidden_dim,
            type=StateType.PURE,
            dtype=x.dtype
        )
        state_manager = StateManager(config, device=x.device)
        state_manager.states["quantum"] = quantum_state
        
        # Initialize geometric and manifold states
        geometric_state = torch.zeros_like(quantum_state)
        manifold_state = torch.zeros_like(quantum_state)
        
        # Create attention state with all required parameters
        state = AttentionState(
            state_manager=state_manager,
            geometric_state=geometric_state,
            manifold_state=manifold_state
        )
        
        # Apply mask if provided
        if mask is not None:
            state = self.apply_mask(state, mask)
            
        return state

    def apply_mask(self, state: AttentionState, mask: torch.Tensor) -> AttentionState:
        """Apply attention mask to the quantum state.
        
        Args:
            state: The current attention state
            mask: Boolean mask tensor of shape [batch_size, seq_length]
            
        Returns:
            Updated attention state with mask applied
        """
        if mask is None:
            return state
            
        # Get quantum state
        quantum_state = state.state_manager.states.get("quantum")
        if quantum_state is None:
            return state
            
        # Expand mask for multi-head attention
        expanded_mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
        expanded_mask = expanded_mask.expand(-1, quantum_state.size(1), -1, -1)  # [batch_size, num_heads, seq_len, 1]
        
        # Initialize attention scores if not present
        if state.attention_scores is None:
            # Create attention scores with -inf where mask is False
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, self.num_heads, mask.size(1), -1)  # [batch_size, num_heads, seq_len, seq_len]
            
            state.attention_scores = torch.full(
                (mask.size(0), self.num_heads, mask.size(1), mask.size(1)),
                float("-inf"),
                device=mask.device
            )
            # Set 0 where mask is True
            state.attention_scores = state.attention_scores.masked_fill(attention_mask, 0.0)
        else:
            # Apply mask to existing attention scores
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, self.num_heads, mask.size(1), -1)  # [batch_size, num_heads, seq_len, seq_len]
            state.attention_scores = state.attention_scores.masked_fill(~attention_mask, float("-inf"))
        
        # Apply mask to quantum state
        masked_quantum_state = quantum_state * expanded_mask
        
        # Update state
        state.state_manager.states["quantum"] = masked_quantum_state
        
        return state

    def _process_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Process attention using tensor operations."""
        # Handle complex numbers by taking the real part for softmax
        attention_scores = torch.matmul(x, x.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores.real, dim=-1)
        
        # Apply attention using complex values
        attention_output = torch.matmul(attention_weights, x)
        return attention_output

    def _compute_geometric_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute geometric features using tensor operations."""
        # Ensure metric requires gradients
        metric = self.metric.requires_grad_(True)
        
        # Use proper tensor operations with gradient tracking
        features = torch.matmul(x, metric)
        
        # Ensure features require gradients
        features.requires_grad_(True)
        
        return features

    def _apply_parallel_transport(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel transport to input features.
        
        Args:
            x: Input features of shape [batch_size, seq_len, manifold_dim]
            
        Returns:
            Transported features of shape [batch_size, seq_len, manifold_dim]
        """
        # Create target point on manifold (origin)
        y = torch.zeros_like(x)
        
        # Create tangent vector at x
        v = x - y
        
        # Apply parallel transport
        transported = self.transport(x, y, v)
        
        return transported

    def _compute_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum features from input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            
        Returns:
            Quantum features tensor
        """
        # Store original shape
        original_shape = x.shape
        
        # Reshape input to [batch_size * seq_len, head_dim] if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.reshape(-1, hidden_dim)
        
        # Project to manifold space
        x = x.reshape(-1, self.hidden_dim)  # Ensure correct input shape
        x = self.manifold_proj(x)  # Project to manifold space
        
        # Reshape back to 3D for tile processing
        if len(original_shape) == 3:
            x = x.reshape(original_shape[0], original_shape[1], -1)
        
        # Apply quantum attention through tiles
        for tile in self.tiles:
            # Get just the tensor from the tile output
            tile_output = tile(x)
            if hasattr(tile_output, 'coordinates'):
                x = tile_output.coordinates
            else:
                x = tile_output
        
        return x

    def compute_attention_patterns(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute attention patterns with shape validation."""
        try:
            # Validate shapes
            self._check_attention_shapes(query, key, value)
            
            batch_size, num_heads, seq_len, head_dim = query.shape
            
            # Reshape tensors to 3D for projection
            query_flat = query.reshape(batch_size * num_heads, seq_len, head_dim)  # [batch_size * num_heads, seq_len, head_dim]
            key_flat = key.reshape(batch_size * num_heads, seq_len, head_dim)
            value_flat = value.reshape(batch_size * num_heads, seq_len, head_dim)
            
            # Print shapes for debugging
            print(f"Input shapes - query_flat: {query_flat.shape}, key_flat: {key_flat.shape}, value_flat: {value_flat.shape}")
            print(f"Weight shapes - manifold_proj: {self.manifold_proj.weight.shape}")
            
            # Project to manifold space with correct dimensions
            query_manifold = self.manifold_proj(query_flat)  # [batch_size * num_heads, seq_len, manifold_dim]
            key_manifold = self.manifold_proj(key_flat)
            value_manifold = self.manifold_proj(value_flat)
            
            print(f"Manifold shapes - query: {query_manifold.shape}, key: {key_manifold.shape}, value: {value_manifold.shape}")
            
            # Reshape back to 4D with correct dimensions
            query_manifold = query_manifold.reshape(batch_size, num_heads, seq_len, -1)
            key_manifold = key_manifold.reshape(batch_size, num_heads, seq_len, -1)
            value_manifold = value_manifold.reshape(batch_size, num_heads, seq_len, -1)
            
            print(f"4D shapes - query: {query_manifold.shape}, key: {key_manifold.shape}, value: {value_manifold.shape}")
            
            # Compute attention scores with quantum geometric structure
            attention_scores = torch.matmul(query_manifold, key_manifold.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(query_manifold.size(-1))  # Scale by actual feature dimension
            
            # Apply softmax
            attention_weights = self.complex_softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            attention_output = torch.matmul(attention_weights, value_manifold)
            
            # Project back to original space
            attention_output_flat = attention_output.reshape(batch_size * num_heads, seq_len, -1)
            output = self.manifold_proj_inv(attention_output_flat)
            
            # Reshape back to 4D with original dimensions
            output = output.reshape(batch_size, num_heads, seq_len, head_dim)
            
            if not return_metrics:
                return output
                
            metrics = {
                'attention_weights': attention_weights,
                'attention_scores': attention_scores
            }
            
            return output, metrics
            
        except Exception as e:
            logger.error(f"Attention pattern computation failed: {str(e)}")
            raise

    def integrate_heads(self, head_states: List[torch.Tensor]) -> torch.Tensor:
        """Integrate multiple attention heads into a single representation.
        
        Args:
            head_states: List of head state tensors, each of shape [batch_size, seq_len, head_dim]
            
        Returns:
            Integrated tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Stack heads along the feature dimension
        stacked = torch.cat(head_states, dim=-1)
        
        # Project to hidden dimension
        integrated = self.to_out(stacked)
        return integrated



    def classical_to_quantum(self, x: torch.Tensor) -> QuantumState:
        """Convert classical input to quantum state.

        Args:
            x: Input tensor of shape (batch_size, manifold_dim) or (batch_size, seq_len, manifold_dim)
            
        Returns:
            Quantum state tensor
        """
        result = self.quantum_bridge.neural_to_quantum(x)
        # Handle both direct return and tuple return with validation
        if isinstance(result, tuple):
            return result[0]  # Extract just the QuantumState
        return result

    def quantum_to_classical(self, quantum_state: Union[torch.Tensor, QuantumState]) -> torch.Tensor:
        """Convert quantum state back to classical tensor.
        
        Args:
            quantum_state: Quantum state tensor or QuantumState object
            
        Returns:
            Classical tensor
        """
        # Ensure we have a QuantumState object
        if not isinstance(quantum_state, QuantumState):
            quantum_state = QuantumState(
                amplitudes=quantum_state,
                basis_labels=[str(i) for i in range(self.manifold_dim)],
                phase=torch.zeros(1, dtype=self.dtype, device=self.device)
            )
        return self.quantum_bridge.quantum_to_neural(quantum_state)

    def create_attention_parameters(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Create attention mechanism parameters.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary of attention parameters
        """
        # Create query, key, value projections with proper types
        q_weight = torch.randn(
            batch_size, self.num_heads, seq_len, self.hidden_dim // self.num_heads,
            dtype=self.dtype, device=self.device
        )
        k_weight = torch.randn(
            batch_size, self.num_heads, seq_len, self.hidden_dim // self.num_heads,
            dtype=self.dtype, device=self.device
        )
        v_weight = torch.randn(
            batch_size, self.num_heads, seq_len, self.hidden_dim // self.num_heads,
            dtype=self.dtype, device=self.device
        )
        
        # Initialize attention parameters with proper scaling
        scale = torch.sqrt(torch.tensor(self.hidden_dim // self.num_heads, dtype=self.dtype))
        params = {
            "query_weight": nn.Parameter(q_weight / scale),
            "key_weight": nn.Parameter(k_weight),
            "value_weight": nn.Parameter(v_weight),
            "attention_scale": scale,
        }
        
        return params

    def compute_metric_tensor(self, state: Union[AttentionState, torch.Tensor]) -> torch.Tensor:
        """Compute metric tensor for attention manifold.
        
        Args:
            state: Current attention state or tensor
            
        Returns:
            Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
        """
        # Handle both AttentionState and raw tensor inputs
        if isinstance(state, AttentionState):
            state_tensor = state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum"))
            # Print state tensor shape for debugging
            print(f"State tensor shape: {state_tensor.shape}")
            # Reshape to [1, 1, hidden_dim]
            state_tensor = state_tensor.reshape(1, 1, -1)
        else:
            state_tensor = state
            
        # Ensure state tensor has batch dimension
        if state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(0)
            
        batch_size = state_tensor.shape[0]
        seq_len = state_tensor.shape[1]
        
        # Initialize metric tensor with identity matrix for stability
        metric = torch.eye(
            self.manifold_dim,
            dtype=self.dtype,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute Jacobian of state map
        with torch.enable_grad():
            state_tensor = state_tensor.requires_grad_(True)
            
            # Create basis vectors for tangent space
            basis_vectors = []
            for i in range(self.manifold_dim):
                v = torch.zeros(batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device)
                v[:, :, i] = 1.0
                basis_vectors.append(v)
            
            # Compute Jacobian-vector products for each basis vector
            jvps = []
            for v in basis_vectors:
                # Compute JVP
                _, jvp = torch.autograd.functional.jvp(
                    lambda x: self._compute_quantum_features(x),
                    (state_tensor,),
                    (v,)
                )
                jvps.append(jvp)
            
            # Stack JVPs to form Jacobian
            jacobian = torch.stack(jvps, dim=1)  # [batch_size, manifold_dim, manifold_dim]
        
        # Compute metric as g_ij = sum_k (∂f_k/∂x_i)(∂f_k/∂x_j)
        metric_update = torch.einsum('...ki,...kj->...ij', jacobian, jacobian.conj())
        
        # Scale metric to prevent numerical instability
        scale = torch.max(torch.abs(metric_update))
        if scale > 1:
            metric_update = metric_update / scale
        
        # Add regularization term to ensure positive definiteness
        reg_term = torch.eye(self.manifold_dim, dtype=self.dtype, device=self.device).unsqueeze(0) * 1e-3
        metric = metric + metric_update + reg_term
        
        # Ensure Hermitian property
        metric = 0.5 * (metric + metric.transpose(-2, -1).conj())
        
        return metric

    def prepare_code_state(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare quantum code state for attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Quantum code state of shape [batch_size, seq_len, hidden_dim]
        """
        # Ensure proper dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.shape
        
        # Project to quantum code space with correct dimensions
        code_proj = self.pattern_proj(x.view(-1, self.manifold_dim))
        code_proj = code_proj.view(batch_size, seq_len, -1)
        
        # Apply quantum encoding through forward pass (self-attention style)
        attention_output = self.attention(
            code_proj, code_proj, code_proj,
            return_metrics=False
        )
        
        # Add geometric structure through forward pass
        flow_output, _ = self.flow(attention_output.view(-1, self.manifold_dim))
        flow_output = flow_output.view(batch_size, seq_len, -1)
        
        return flow_output

    def build_attention_complex(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, AttentionMetrics]:
        """Build quantum geometric attention complex.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len, hidden_dim]
            key: Key tensor of shape [batch_size, seq_len, hidden_dim]
            value: Value tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            Tuple containing:
            - Attention output tensor of shape [batch_size, seq_len, hidden_dim]
            - Attention metrics
        """
        batch_size, seq_len, _ = query.shape
        
        # Prepare attention state with proper dimensions
        state = self.prepare_attention_state(query, mask)
        
        # Project queries, keys and values to manifold space
        q_manifold = self.manifold_proj(query.view(-1, self.head_dim))
        k_manifold = self.manifold_proj(key.view(-1, self.head_dim))
        v_manifold = self.manifold_proj(value.view(-1, self.head_dim))
        
        # Reshape for attention computation
        q_manifold = q_manifold.view(batch_size, seq_len, self.num_heads, -1)
        k_manifold = k_manifold.view(batch_size, seq_len, self.num_heads, -1)
        v_manifold = v_manifold.view(batch_size, seq_len, self.num_heads, -1)
        
        # Create default key tensor if mask is None
        key_tensor = mask if mask is not None else state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum"))
        
        # Create metric context
        context = MetricContext(
            timestamp=0.0,
            device=query.device,
            batch_size=batch_size,
            sequence_length=seq_len,
            hidden_dim=self.hidden_dim,
            resolution=1.0
        )
        
        # Compute attention patterns with metrics
        attention_pattern, metrics = cast(
            Tuple[torch.Tensor, AttentionMetrics],
            self.compute_attention_patterns(
                state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")),
                key_tensor,
                value=key_tensor,
                return_metrics=True
            )
        )
        
        # Apply attention with proper dimension handling
        attention_output = torch.einsum(
            "bhqk,bkhd->bqhd",
            attention_pattern,
            v_manifold.transpose(1, 2)
        )
        
        # Combine heads and project back to hidden dimension
        attention_output = attention_output.reshape(batch_size, seq_len, -1)
        attention_output = self.to_out(attention_output)
        
        return attention_output, metrics



    def validate_quantum_state(self, state: torch.Tensor) -> None:
        """Validate quantum state and raise error if invalid.
        
        Args:
            state: Tensor to validate
            
        Raises:
            ValueError: If state is invalid
        """
        # Check complex type
        if not torch.is_complex(state):
            raise ValueError("Quantum state must be complex-valued")
            
        # Check normalization with proper tolerance
        norms = state.abs().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-7):
            raise ValueError(f"Quantum state must be normalized, got norms: {norms}")
            
        # Check shape compatibility
        if len(state.shape) not in [3, 4]:  # (batch, seq, dim) or (batch, heads, seq, dim)
            raise ValueError(f"Invalid quantum state shape: {state.shape}")
            
        # Check dimension compatibility
        if state.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected hidden dimension {self.hidden_dim}, got {state.shape[-1]}")
            
        # Check phase consistency
        phases = torch.angle(state)
        if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
            raise ValueError("Invalid phases in quantum state")
            
        # Check numerical stability
        if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
            raise ValueError("Quantum state contains NaN or Inf values")

    def _is_valid_quantum_state(self, state: torch.Tensor) -> bool:
        """Check if tensor represents a valid quantum state.
        
        Args:
            state: Tensor to check
            
        Returns:
            True if state is valid quantum state
        """
        # Check normalization
        norms = state.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5):
            return False
            
        # Check shape compatibility
        if len(state.shape) not in [3, 4]:  # (batch, seq, dim) or (batch, heads, seq, dim)
            return False
            
        # Check dimension compatibility
        if state.shape[-1] != self.hidden_dim:
            return False
            
        return True

    def construct_hamiltonian(self, attention_pattern: torch.Tensor) -> torch.Tensor:
        """Construct the Hamiltonian from the attention pattern.

        Args:
            attention_pattern: The attention pattern tensor of shape [..., manifold_dim]

        Returns:
            The Hamiltonian tensor of shape [..., manifold_dim, manifold_dim]
        """
        # Project attention pattern to manifold dimension if needed
        if attention_pattern.shape[-1] != self.manifold_dim:
            attention_pattern = self.manifold_proj(attention_pattern)

        # Construct Hamiltonian as outer product
        hamiltonian = torch.einsum('...i,...j->...ij', attention_pattern, attention_pattern)
        
        # Ensure Hermitian
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose(-1, -2))
        
        return hamiltonian

    def evolve_state(self, state: QuantumState, hamiltonian: torch.Tensor, t: float = 0.1) -> QuantumState:
        """Evolve quantum state under Hamiltonian.
        
        This method implements quantum time evolution using the Schrödinger equation:
        |ψ(t)⟩ = exp(-iHt/ℏ)|ψ(0)⟩
        
        Args:
            state: Initial quantum state |ψ(0)⟩
            hamiltonian: Hamiltonian operator H (must be Hermitian)
            t: Evolution time (defaults to 0.1)
            
        Returns:
            Evolved quantum state |ψ(t)⟩
        """
        if not isinstance(t, float):
            raise TypeError("Evolution time 't' must be a float")
            
        # Ensure complex types
        hamiltonian = hamiltonian.to(torch.complex128)
        state_vector = state.amplitudes.to(torch.complex128)
        
        # Single time evolution
        evolution_operator = torch.matrix_exp(-1j * hamiltonian * t)
        
        # Handle batch dimension properly
        if len(state_vector.shape) > 2:
            batch_size, seq_len, dim = state_vector.shape
            state_vector = state_vector.reshape(-1, dim)
            evolution_operator = evolution_operator.reshape(-1, dim, dim)
        
        # Apply evolution
        evolved_state = torch.matmul(state_vector.unsqueeze(1), evolution_operator).squeeze(1)
        
        # Compute phase correction for each state in the batch
        phase_corrections = []
        for i in range(evolved_state.shape[0]):
            # Get the first basis state for this batch
            basis_state = evolution_operator[i, 0] if len(evolution_operator.shape) > 2 else evolution_operator[0]
            # Compute overlap and phase correction
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

    def normalize_complex_tensor(self, tensor: torch.Tensor, target_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalize complex tensor while preserving phase and gradients."""
        # Compute norm along the last dimension
        current_norm = torch.sqrt(torch.sum(tensor.real ** 2 + tensor.imag ** 2, dim=-1, keepdim=True) + 1e-8)
        
        # Use target norm if provided, otherwise use ones
        if target_norm is None:
            target_norm = torch.ones_like(current_norm)
        
        # Compute scale factor
        scale = target_norm / current_norm
        
        # Scale tensor while preserving gradients
        return tensor * scale

    def geometric_attention_flow(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        dt: float = 0.1,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, FlowMetrics]]:
        """Apply geometric flow to attention patterns."""
        # Debug prints for gradient tracking
        print("\nStarting geometric_attention_flow:")
        print(f"Input requires_grad: {x.requires_grad}")
        
        # Ensure input requires gradients
        x = x.requires_grad_(True)
        
        # Initialize flow metrics
        metrics = []
        
        # Initialize flow state
        current = x
        
        # Prepare attention state
        state = self.prepare_attention_state(current, mask)
        
        # Apply mask if provided
        if mask is not None:
            state = self.apply_mask(state, mask)
        
        # Compute initial metric
        metric = self.compute_metric_tensor(state)
        print(f"Initial metric requires_grad: {metric.requires_grad}")
        
        # Ensure metric requires gradients
        metric = metric.requires_grad_(True)
        
        # Apply flow steps
        for step in range(num_steps):
            print(f"\nStep {step}:")
            # Compute Ricci tensor
            ricci = self.pattern_bundle.compute_ricci_tensor(metric)
            print(f"Ricci tensor requires_grad: {ricci.requires_grad}")
            
            # Take flow step
            new_metric, step_metrics = self.pattern_bundle.flow_step(metric, ricci, dt)
            print(f"New metric requires_grad: {new_metric.requires_grad}")
            
            # Debug print pattern bundle parameters
            for name, param in self.pattern_bundle.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradients")
            
            # Update metric
            metric = new_metric
            
            # Create new state with updated geometric state
            state = AttentionState(
                state_manager=state.state_manager,
                geometric_state=metric.reshape(metric.shape[0], -1),  # Flatten metric for geometric state
                manifold_state=state.manifold_state,  # Keep existing manifold state
                attention_scores=state.attention_scores,  # Keep existing attention scores
                attention_patterns=state.attention_patterns,  # Keep existing attention patterns
                entanglement_history=state.entanglement_history,  # Keep existing entanglement history
                metrics=state.metrics  # Keep existing metrics
            )
            
            # Process attention with updated state
            current = self._process_attention(current)
            print(f"Current output requires_grad: {current.requires_grad}")
            
            # Store metrics
            metrics.append(step_metrics)
        
        # Return final state and metrics if requested
        if return_metrics:
            # Compute average curvature from metrics
            curvature = torch.zeros_like(metric)
            parallel_transport = torch.zeros_like(metric)
            geodesic_distance = torch.zeros(metric.shape[0], metric.shape[1])
            energy = torch.zeros(metric.shape[0], metric.shape[1])
            
            for m in metrics:
                curvature = curvature + m.get('curvature', torch.zeros_like(curvature))
                parallel_transport = parallel_transport + m.get('parallel_transport', torch.zeros_like(parallel_transport))
                geodesic_distance = geodesic_distance + m.get('geodesic_distance', torch.zeros_like(geodesic_distance))
                energy = energy + m.get('energy', torch.zeros_like(energy))
            
            num_metrics = len(metrics)
            if num_metrics > 0:
                curvature = curvature / num_metrics
                parallel_transport = parallel_transport / num_metrics
                geodesic_distance = geodesic_distance / num_metrics
                energy = energy / num_metrics
            
            return current, FlowMetrics(
                curvature=curvature,
                parallel_transport=parallel_transport,
                geodesic_distance=geodesic_distance,
                energy=energy
            )
        
        return current

    def flow_step(self, state: AttentionState, dt: float) -> AttentionState:
        """Perform a single flow step.
        
        Args:
            state: Current attention state
            dt: Time step size
            
        Returns:
            Updated attention state
        """
        # Get quantum state
        quantum_state = state.state_manager.states.get("quantum")
        if quantum_state is None:
            return state
            
        # Get metric tensor
        metric = self.compute_metric_tensor(state)
        
        # Compute Ricci flow update
        ricci = compute_ricci_tensor(metric)
        metric = metric - dt * ricci
        
        # Update state with new metric
        state.geometric_state = metric
        
        return state

    def prepare_quantum_state(self, classical_input: torch.Tensor) -> torch.Tensor:
        """Convert classical input to quantum state.

        Args:
            classical_input: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            Quantum state tensor of shape [..., manifold_dim] where ... matches input dimensions
        """
        # Project to manifold space first
        manifold_input = self.manifold_proj(classical_input)
        # Project back to hidden dimension
        hidden_input = self.manifold_proj_inv(manifold_input)
        # Convert to quantum state
        quantum_state = self.classical_to_quantum(hidden_input)
        # Convert to complex64 to match linear layer dtype
        quantum_data = quantum_state.data.to(dtype=torch.complex64)
        # Project back to manifold space
        quantum_manifold = self.manifold_proj(quantum_data)
        # Normalize the quantum state
        norms = torch.norm(quantum_manifold, dim=-1, keepdim=True)
        quantum_manifold = quantum_manifold / (norms + 1e-8)
        return quantum_manifold

    def _validate_shape(self, tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str) -> None:
        """Validate tensor shape and log any mismatches."""
        if not self.validate_shapes:
            return
            
        if tensor.shape != expected_shape:
            error_msg = f"Shape mismatch in {name}: expected {expected_shape}, got {tensor.shape}"
            self.shape_errors.append(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    def _clear_shape_errors(self) -> None:
        """Clear accumulated shape errors."""
        self.shape_errors = []
        
    def get_shape_errors(self) -> List[str]:
        """Get list of accumulated shape errors."""
        return self.shape_errors
        
    @torch.jit.export
    def _check_input_shapes(self, x: torch.Tensor) -> None:
        """Validate input shapes with TorchScript support."""
        if not self.validate_shapes:
            return
            
        if len(x.shape) != 3:
            raise ValueError(f"Input must be 3D (batch_size, seq_len, hidden_dim), got shape {x.shape}")
            
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Input hidden dimension must be {self.hidden_dim}, got {x.size(-1)}")
            
    @torch.jit.export
    def _check_attention_shapes(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """Validate attention computation shapes with TorchScript support."""
        if not self.validate_shapes:
            return
            
        # Check batch size and num_heads match
        if not (query.size(0) == key.size(0) == value.size(0)):
            raise ValueError("Batch sizes must match")
            
        if not (query.size(1) == key.size(1) == value.size(1)):
            raise ValueError("Number of heads must match")
            
        # Check sequence lengths
        if not (key.size(2) == value.size(2)):
            raise ValueError("Key and value sequence lengths must match")
            
        # Check head dimensions
        head_dim = self.hidden_dim // self.num_heads
        if not (query.size(-1) == key.size(-1) == value.size(-1) == head_dim):
            raise ValueError(f"Head dimension must be {head_dim}")
            
    @torch.jit.export
    def _validate_quantum_state_shape(self, state: torch.Tensor) -> None:
        """Validate quantum state shape with TorchScript support."""
        if not self.validate_shapes:
            return
            
        if len(state.shape) not in [3, 4]:
            raise ValueError(f"Invalid quantum state shape: {state.shape}")
            
        if state.shape[-1] != self.manifold_dim:
            raise ValueError(f"Expected manifold dimension {self.manifold_dim}, got {state.shape[-1]}")

