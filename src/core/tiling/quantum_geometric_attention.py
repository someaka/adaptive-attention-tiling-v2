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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, cast

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
        dtype: torch.dtype = torch.float32,
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
        
        # Initialize manifold projections with float dtype
        self.manifold_proj = nn.Linear(hidden_dim, self.manifold_dim, dtype=dtype, device=device)
        self.manifold_proj_inv = nn.Linear(self.manifold_dim, hidden_dim, dtype=dtype, device=device)
        
        # Initialize quantum bridge with complex dtype
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            manifold_type=manifold_type,
            curvature=curvature,
            dtype=torch.complex64,  # Use complex dtype for quantum components
            device=device
        )
        
        # Initialize tiles with complex dtype
        self.tiles = nn.ModuleList([
            QuantumMotivicTile(
                size=tile_size,
                hidden_dim=self.manifold_dim,
                num_heads=1,
                dropout=dropout,
                resolution=1.0,
                cohomology_dim=self.manifold_dim,
                motive_rank=motive_rank,
                dtype=torch.complex64  # Use complex dtype for quantum components
            )
            for _ in range(num_heads)
        ])
        
        # Initialize attention layers with complex dtype
        self.attention_layers = nn.ModuleList([
            nn.Linear(self.manifold_dim, self.manifold_dim, dtype=torch.complex64, device=device)
            for _ in range(num_layers)
        ])
        
        # Initialize metric tensor with float dtype
        metric = torch.eye(self.manifold_dim, dtype=dtype, device=device)
        self.metric = nn.Parameter(metric, requires_grad=True)
        
        # Initialize flow with float dtype
        self.flow = GeometricFlow(
            hidden_dim=hidden_dim,
            manifold_dim=self.manifold_dim,
            motive_rank=motive_rank,
            num_charts=4,  # Default value
            integration_steps=10,  # Default value
            dt=0.1,  # Default value
            stability_threshold=1e-6,  # Default value
            dtype=dtype
        )
        
        # Initialize dropout
        self.dropout = nn.Dropout(dropout)
        
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
                
                # Initialize real part with Glorot/Xavier initialization
                real_weight = torch.randn(weight_shape, device=self.device) * std
                imag_weight = torch.randn(weight_shape, device=self.device) * std
                
                # Create complex weight tensor
                complex_weight = torch.complex(real_weight, imag_weight)
                
                # Ensure the weight is complex and has the correct dtype
                if not torch.is_complex(complex_weight):
                    complex_weight = complex_weight.to(dtype=torch.complex64)
                layer.weight = nn.Parameter(complex_weight)
                
                if layer.bias is not None:
                    real_bias = torch.randn(weight_shape[0], device=self.device) * std
                    imag_bias = torch.randn(weight_shape[0], device=self.device) * std
                    
                    # Create complex bias tensor
                    complex_bias = torch.complex(real_bias, imag_bias)
                    
                    # Ensure the bias is complex and has the correct dtype
                    if not torch.is_complex(complex_bias):
                        complex_bias = complex_bias.to(dtype=torch.complex64)
                    layer.bias = nn.Parameter(complex_bias)

        # Initialize all attention layers
        for layer in self.attention_layers:
            init_complex_linear(layer)

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize manifold projections with real weights first
        with torch.no_grad():
            # Create real orthogonal matrices
            real_weight = torch.empty_like(self.manifold_proj.weight.real)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(self.manifold_proj.weight.imag)
            nn.init.orthogonal_(imag_weight)
            
            # Combine into complex weight
            self.manifold_proj.weight.copy_(torch.complex(real_weight, imag_weight))
            self.manifold_proj.bias.zero_()
            
            # Same for inverse projection
            real_weight = torch.empty_like(self.manifold_proj_inv.weight.real)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(self.manifold_proj_inv.weight.imag)
            nn.init.orthogonal_(imag_weight)
            self.manifold_proj_inv.weight.copy_(torch.complex(real_weight, imag_weight))
            self.manifold_proj_inv.bias.zero_()
            
            # Initialize pattern projections similarly
            real_weight = torch.empty_like(self.pattern_proj.weight.real)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(self.pattern_proj.weight.imag)
            nn.init.orthogonal_(imag_weight)
            self.pattern_proj.weight.copy_(torch.complex(real_weight, imag_weight))
            self.pattern_proj.bias.zero_()
            
            real_weight = torch.empty_like(self.pattern_proj_inv.weight.real)
            nn.init.orthogonal_(real_weight)
            imag_weight = torch.empty_like(self.pattern_proj_inv.weight.imag)
            nn.init.orthogonal_(imag_weight)
            self.pattern_proj_inv.weight.copy_(torch.complex(real_weight, imag_weight))
            self.pattern_proj_inv.bias.zero_()
            
            # Initialize attention components with small normal values
            for layer in [self.query, self.key, self.value, self.to_qkv, self.to_out]:
                real_weight = torch.empty_like(layer.weight.real)
                imag_weight = torch.empty_like(layer.weight.imag)
                nn.init.normal_(real_weight, std=0.02)
                nn.init.normal_(imag_weight, std=0.02)
                layer.weight.copy_(torch.complex(real_weight, imag_weight))
                layer.bias.zero_()

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
        """Forward pass through quantum geometric attention."""
        print(f"\nInput shape: {x.shape}")
        batch_size, seq_len, _ = x.shape
        
        # Store original input shape for gradient flow
        self.x_flat = x.reshape(-1, x.shape[-1])  # Store x_flat as class attribute
        self.x_flat.requires_grad_(True)  # Ensure requires_grad is True
        self.x_flat.retain_grad()  # Retain gradients for x_flat
        print(f"x_flat requires_grad: {self.x_flat.requires_grad}")
        
        # Project input to manifold space using x_flat
        x_manifold = self.manifold_proj(self.x_flat.reshape(batch_size, seq_len, -1))  # [batch_size, seq_len, manifold_dim]
        print(f"After manifold projection shape: {x_manifold.shape}")
        print(f"x_manifold requires_grad: {x_manifold.requires_grad}")
        
        # Apply metric tensor
        metric = self.metric.to(dtype=x_manifold.dtype)
        metric.requires_grad_(True)  # Ensure metric requires gradients
        print(f"metric requires_grad: {metric.requires_grad}")
        x_metric = torch.einsum('bsi,ij->bsj', x_manifold, metric)
        print(f"x_metric requires_grad: {x_metric.requires_grad}")
        
        # Get connection and ensure consistent types
        connection = self.quantum_bridge.pattern_bundle.connection
        if connection.shape[-1] != self.manifold_dim:
            # Project connection to manifold dimension
            connection = connection[..., :self.manifold_dim, :self.manifold_dim, :self.manifold_dim]
        connection.requires_grad_(True)  # Ensure connection requires gradients
        connection.retain_grad()  # Retain gradients for connection
        print(f"connection requires_grad: {connection.requires_grad}")
        
        # Convert x_metric to complex if connection is complex
        if torch.is_complex(connection):
            x_metric = x_metric.to(dtype=torch.complex64)
        
        # Apply connection to input with gradient tracking
        x_with_connection = x_metric + torch.einsum('bsi,ijk->bsj', x_metric, connection)
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
            expanded_mask = expanded_mask.expand(-1, self.num_heads, -1, self.manifold_dim)  # [batch_size, num_heads, seq_len, manifold_dim]
            stacked_outputs = stacked_outputs.masked_fill(~expanded_mask, 0.0)
        
        # Mean over heads
        mean_output = stacked_outputs.mean(dim=2)  # [batch_size, seq_len, manifold_dim]
        print(f"After mean over heads shape: {mean_output.shape}")
        print(f"mean_output requires_grad: {mean_output.requires_grad}")
        
        # Apply quantum attention layers
        quantum_output = mean_output
        for i, layer in enumerate(self.attention_layers):
            quantum_output = layer(quantum_output)
            print(f"After quantum attention layer {i} shape: {quantum_output.shape}")
            print(f"quantum_output layer {i} requires_grad: {quantum_output.requires_grad}")
        
        # Apply flow operation and unpack output
        flow_output, _ = self.flow(quantum_output)
        if torch.is_complex(flow_output):
            flow_output = flow_output.real
        print(f"flow_output requires_grad: {flow_output.requires_grad}")
        
        # Inverse manifold projection
        output = self.manifold_proj_inv(flow_output)
        print(f"After inverse manifold projection shape: {output.shape}")
        print(f"output requires_grad: {output.requires_grad}")
        
        # Add residual connection for gradient stability using x_flat
        x_flat_reshaped = self.x_flat.reshape(output.shape)
        output = output + 0.1 * x_flat_reshaped  # Use x_flat directly in residual connection
        
        # Create a direct path for gradient flow
        if self.training:
            # Ensure gradients flow through connection
            connection_scale = torch.sum(connection * connection.conj()).real
            output = output * (1.0 + 1e-6 * connection_scale)
            
            # Create stronger gradient path for x_flat
            x_flat_scale = torch.sum(x_flat_reshaped * x_flat_reshaped).real
            output = output * (1.0 + 1e-6 * x_flat_scale)
            
            # Add gradient hooks for debugging
            def hook_connection(grad):
                print(f"\nConnection gradient shape: {grad.shape}")
                print(f"Connection gradient norm: {grad.norm().item()}")
                return grad
            
            def hook_x_flat(grad):
                print(f"\nx_flat gradient shape: {grad.shape}")
                print(f"x_flat gradient norm: {grad.norm().item()}")
                return grad
            
            connection.register_hook(hook_connection)
            self.x_flat.register_hook(hook_x_flat)
        
        return output

    def prepare_attention_state(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> AttentionState:
        """Prepare attention state with optional mask."""
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=self.dtype)
        
        # Initialize quantum and geometric states with correct dimensions
        batch_size, seq_len, _ = x.shape
        
        # Initialize quantum state with random values and normalize
        quantum_state = torch.randn(
            batch_size, self.num_heads, seq_len, self.manifold_dim,
            dtype=self.dtype, device=x.device
        )
        # Normalize quantum state along the last dimension
        quantum_state = quantum_state / quantum_state.norm(dim=-1, keepdim=True)
        
        # Initialize geometric state
        geometric_state = torch.zeros(
            batch_size, self.num_heads, seq_len, self.manifold_dim,
            dtype=self.dtype, device=x.device
        )
        
        # Initialize AttentionState
        state = AttentionState.initialize(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            device=x.device,
            dtype=self.dtype
        )
        
        # Store quantum state in state manager
        state.state_manager.states["quantum"] = quantum_state
        
        # Initialize geometric components
        state.geometric_state = geometric_state
        state.manifold_state = geometric_state.clone()
        
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
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute quantum geometric attention patterns.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask [batch_size, seq_len]
            return_metrics: Whether to return attention metrics
            
        Returns:
            - Attention patterns [batch_size, num_heads, seq_len, seq_len]
            - Optional metrics dictionary if return_metrics is True
        """
        # Initialize metrics
        metrics: Dict[str, Any] = {}
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Split into real and imaginary parts
        scores_real = scores.real
        scores_imag = scores.imag
        
        # Apply softmax to real and imaginary parts separately
        attn_real = F.softmax(scores_real, dim=-1)
        attn_imag = F.softmax(scores_imag, dim=-1)
        
        # Combine real and imaginary parts
        attn = torch.complex(attn_real, attn_imag)
        
        # Normalize to ensure row-wise sum is 1
        row_sums = torch.sum(attn, dim=-1, keepdim=True)
        attn = attn / row_sums
        
        if return_metrics:
            metrics['attention_scores'] = scores
            metrics['attention_weights'] = attn
            return attn, metrics
            
        return attn

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
        """Apply geometric attention flow to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            num_steps: Number of flow steps
            dt: Time step size
            return_metrics: Whether to return flow metrics
            
        Returns:
            Flowed tensor and optional metrics
        """
        # Initialize state
        state = self.prepare_attention_state(x, mask)

        # Initialize metrics
        if return_metrics:
            curvature_list = []
            parallel_transport_list = []
            geodesic_distance_list = []
            energy_list = []

        # Initialize flow
        current = x
        batch_size, seq_len, hidden_dim = current.shape

        # Get initial metric
        metric = self.compute_metric_tensor(state)  # [batch_size, manifold_dim, manifold_dim]

        # Ensure metric has correct shape for flow step
        if len(metric.shape) > 3:
            # Reshape to [batch_size * seq_len, manifold_dim, manifold_dim]
            metric = metric.reshape(-1, self.manifold_dim, self.manifold_dim)

        def project_to_manifold(tensor: torch.Tensor, target_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Project tensor to manifold space while preserving norm and gradients."""
            # Store input norm if target_norm not provided
            if target_norm is None:
                target_norm = torch.sqrt(torch.sum(tensor.real ** 2 + tensor.imag ** 2, dim=-1, keepdim=True) + 1e-8)
            
            # Project to manifold space
            manifold = self.manifold_proj(tensor)
            
            # Normalize while preserving gradients
            return self.normalize_complex_tensor(manifold, target_norm)

        def project_to_hidden(tensor: torch.Tensor, target_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Project tensor to hidden space while preserving norm and gradients."""
            # Store input norm if target_norm not provided
            if target_norm is None:
                target_norm = torch.sqrt(torch.sum(tensor.real ** 2 + tensor.imag ** 2, dim=-1, keepdim=True) + 1e-8)
            
            # Project to hidden space
            hidden = self.manifold_proj_inv(tensor)
            
            # Normalize while preserving gradients
            return self.normalize_complex_tensor(hidden, target_norm)

        # Get initial norms
        initial_hidden_norm = torch.sqrt(torch.sum(current.reshape(-1, hidden_dim).real ** 2 + 
                                         current.reshape(-1, hidden_dim).imag ** 2, dim=-1, keepdim=True))
        print(f"\nInitial hidden norm: {initial_hidden_norm.mean().item():.4f}")

        # Project initial state to manifold space
        initial_manifold = project_to_manifold(current.reshape(-1, hidden_dim), initial_hidden_norm)
        initial_manifold_norm = torch.sqrt(torch.sum(initial_manifold.real ** 2 + initial_manifold.imag ** 2, dim=-1, keepdim=True))
        print(f"Initial manifold norm: {initial_manifold_norm.mean().item():.4f}")

        # Apply flow steps
        for step in range(num_steps):
            print(f"\nStep {step + 1}/{num_steps}")
            
            # Project to manifold space
            current_manifold = project_to_manifold(current.reshape(-1, hidden_dim), initial_hidden_norm)
            manifold_norm = torch.sqrt(torch.sum(current_manifold.real ** 2 + current_manifold.imag ** 2, dim=-1, keepdim=True))
            print(f"Manifold norm: {manifold_norm.mean().item():.4f}")

            # Compute flow update
            flow_update, flow_metrics = self.flow(current_manifold.reshape(batch_size, seq_len, -1))
            flow_update = flow_update.reshape(-1, self.manifold_dim)
            flow_norm = torch.sqrt(torch.sum(flow_update.real ** 2 + flow_update.imag ** 2, dim=-1, keepdim=True))
            print(f"Flow update norm: {flow_norm.mean().item():.4f}")

            # Project flow update to hidden space
            flow_update = project_to_hidden(flow_update, initial_hidden_norm)
            flow_update = flow_update.reshape(batch_size, seq_len, hidden_dim)
            hidden_norm = torch.sqrt(torch.sum(flow_update.real ** 2 + flow_update.imag ** 2, dim=-1, keepdim=True))
            print(f"Hidden space norm: {hidden_norm.mean().item():.4f}")

            # Update current state with small step size
            current = current + dt * flow_update
            update_norm = torch.sqrt(torch.sum(current.real ** 2 + current.imag ** 2, dim=-1, keepdim=True))
            print(f"After update norm: {update_norm.mean().item():.4f}")

            # Normalize current state to match initial norm
            current = self.normalize_complex_tensor(current.reshape(-1, hidden_dim), initial_hidden_norm)
            current = current.reshape(batch_size, seq_len, hidden_dim)
            current_norm = torch.sqrt(torch.sum(current.real ** 2 + current.imag ** 2, dim=-1, keepdim=True))
            print(f"Final norm for step: {current_norm.mean().item():.4f}")

            # Ensure numerical stability
            if torch.isnan(current).any() or torch.isinf(current).any():
                print("Warning: Numerical instability detected, resetting to initial state")
                current = x  # Reset to initial state if instability detected
                break

            if return_metrics:
                # Compute metrics for this step
                # Expand metric for each sequence position
                metric_expanded = metric.unsqueeze(1).expand(batch_size, seq_len, self.manifold_dim, self.manifold_dim)
                
                # Compute metrics for this step
                curvature = compute_ricci_tensor(metric_expanded)  # [batch_size, seq_len, manifold_dim, manifold_dim]
                parallel_transport = compute_parallel_transport(current_manifold.reshape(batch_size, seq_len, -1), metric_expanded)  # [batch_size, seq_len, manifold_dim, manifold_dim]
                geodesic_distance = compute_geodesic_distance(current_manifold.reshape(batch_size, seq_len, -1), metric_expanded)  # [batch_size, seq_len]
                energy = compute_flow_energy(current_manifold.reshape(batch_size, seq_len, -1), metric_expanded)  # [batch_size, seq_len]

                curvature_list.append(curvature)
                parallel_transport_list.append(parallel_transport)
                geodesic_distance_list.append(geodesic_distance)
                energy_list.append(energy)

        # Final normalization
        current = self.normalize_complex_tensor(current.reshape(-1, hidden_dim), initial_hidden_norm)
        current = current.reshape(batch_size, seq_len, hidden_dim)
        final_norm = torch.sqrt(torch.sum(current.real ** 2 + current.imag ** 2, dim=-1, keepdim=True))
        print(f"\nFinal norm: {final_norm.mean().item():.4f}")

        if return_metrics:
            metrics = FlowMetrics(
                curvature=torch.stack(curvature_list)[-1],  # Take last step
                parallel_transport=torch.stack(parallel_transport_list)[-1],  # Take last step
                geodesic_distance=torch.stack(geodesic_distance_list)[-1],  # Take last step
                energy=torch.stack(energy_list)[-1]  # Take last step
            )
            return current, metrics

        return current
