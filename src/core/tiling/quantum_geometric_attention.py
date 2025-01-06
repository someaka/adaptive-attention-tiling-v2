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
        num_heads: int = 8,
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
            manifold_dim: Manifold dimension (defaults to hidden_dim // 2)
            num_layers: Number of attention layers
            tile_size: Size of attention tiles
            motive_rank: Rank of motivic structure
            dtype: Data type
            device: Device to use
            is_causal: Whether to use causal attention
        """
        super().__init__()

        # Initialize dimensions and parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.manifold_dim = manifold_dim if manifold_dim is not None else hidden_dim // 2
        self.dtype = dtype
        self.device = device or torch.device('cpu')

        # Initialize quantum attention tiles
        self.tiles = nn.ModuleList([
            QuantumMotivicTile(
                size=tile_size,
                hidden_dim=self.head_dim,
                num_heads=1,
                dropout=dropout,
                resolution=1.0,
                cohomology_dim=manifold_dim or hidden_dim // 2,
                motive_rank=motive_rank,
                dtype=dtype
            )
            for _ in range(num_heads)
        ])

        # Validate dtype is complex
        if not torch.is_complex(torch.empty(1, dtype=dtype)):
            raise ValueError("dtype must be complex (torch.complex64 or torch.complex128)")
            
        # Initialize HilbertSpace for quantum computations
        self.hilbert_space = HilbertSpace(
            dim=self.manifold_dim,
            dtype=self.dtype  # HilbertSpace will handle device internally
        )
        
        # Initialize state management and validation
        self.state_config = StateConfig(
            dim=self.manifold_dim,
            type=StateType.PURE,  # Start with pure states
            epsilon=1e-6,
            max_entanglement=1.0,
            dtype=self.dtype
        )
        self.state_manager = StateManager(self.state_config, device=self.device)
        
        # Initialize state validators
        self.state_validator = StateValidator(tolerance=1e-6)
        self.preparation_validator = StatePreparationValidator(tolerance=1e-6)
        
        # Initialize original scale buffer for quantum-classical interface
        self.register_buffer(
            'original_scale',
            torch.ones(1, 1, 1, dtype=self.dtype, device=self.device)
        )
        
        # Initialize neural quantum bridge for state conversion
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            manifold_type=manifold_type,
            curvature=curvature,
            dtype=dtype,
            device=device
        )
        
        # Initialize metric tensor
        self.register_parameter(
            'metric',
            nn.Parameter(torch.eye(self.manifold_dim, dtype=self.dtype, device=self.device))
        )
        
        # Initialize projection layers with correct dimensions
        self.manifold_proj = nn.Linear(
            self.head_dim,  # Input dimension is head_dim (hidden_dim // num_heads)
            self.manifold_dim,  # Output dimension is manifold_dim
            dtype=dtype,
            device=device
        )
        
        # Convert weights and bias to complex
        with torch.no_grad():
            self.manifold_proj.weight.data = self.manifold_proj.weight.data.to(dtype)
            if self.manifold_proj.bias is not None:
                self.manifold_proj.bias.data = self.manifold_proj.bias.data.to(dtype)
        
        self.manifold_proj_inv = nn.Linear(
            self.manifold_dim,  # Input dimension is manifold_dim
            self.head_dim,  # Output dimension is head_dim
            dtype=dtype,
            device=device
        )
        
        # Convert weights and bias to complex
        with torch.no_grad():
            self.manifold_proj_inv.weight.data = self.manifold_proj_inv.weight.data.to(dtype)
            if self.manifold_proj_inv.bias is not None:
                self.manifold_proj_inv.bias.data = self.manifold_proj_inv.bias.data.to(dtype)
        
        # Initialize pattern projection layers
        self.pattern_proj = nn.Linear(
            self.manifold_dim,
            self.head_dim,  # Changed to head_dim to match manifold_proj_inv output
            dtype=dtype,
            device=device
        )
        
        # Convert weights and bias to complex
        with torch.no_grad():
            self.pattern_proj.weight.data = self.pattern_proj.weight.data.to(dtype)
            if self.pattern_proj.bias is not None:
                self.pattern_proj.bias.data = self.pattern_proj.bias.data.to(dtype)
        
        self.pattern_proj_inv = nn.Linear(
            self.head_dim,  # Changed to head_dim to match pattern_proj output
            self.manifold_dim,
            dtype=dtype,
            device=device
        )
        
        # Convert weights and bias to complex
        with torch.no_grad():
            self.pattern_proj_inv.weight.data = self.pattern_proj_inv.weight.data.to(dtype)
            if self.pattern_proj_inv.bias is not None:
                self.pattern_proj_inv.bias.data = self.pattern_proj_inv.bias.data.to(dtype)

        # Initialize query, key, value transformations
        self.query = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.key = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.value = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        
        # Convert weights and bias to complex
        with torch.no_grad():
            self.query.weight.data = self.query.weight.data.to(dtype)
            if self.query.bias is not None:
                self.query.bias.data = self.query.bias.data.to(dtype)
            
            self.key.weight.data = self.key.weight.data.to(dtype)
            if self.key.bias is not None:
                self.key.bias.data = self.key.bias.data.to(dtype)
            
            self.value.weight.data = self.value.weight.data.to(dtype)
            if self.value.bias is not None:
                self.value.bias.data = self.value.bias.data.to(dtype)

        # Initialize dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize attention layers
        self.attention_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])

        # Initialize to_qkv and to_out projections
        expanded_dim = self.num_heads * hidden_dim
        self.to_qkv = nn.Linear(hidden_dim, 3 * expanded_dim, dtype=dtype, device=device)
        self.to_out = nn.Linear(expanded_dim, hidden_dim, dtype=dtype, device=device)

        # Initialize geometric flow with correct dimensions
        self.flow = GeometricFlow(
            hidden_dim=self.manifold_dim,  # Use manifold_dim for hidden_dim
            manifold_dim=self.manifold_dim,  # Use manifold_dim for manifold_dim
            motive_rank=motive_rank,
            num_charts=4,
            integration_steps=10,
            dt=0.1,
            stability_threshold=1e-6,
            dtype=dtype
        ).to(self.device)
        
        # Initialize quantum attention
        self.quantum_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype, device=self.device)
        )
        
        # Initialize arithmetic pattern
        self.arithmetic = ArithmeticPattern(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            motive_rank=motive_rank,
            dtype=dtype
        ).to(self.device)
        
        # Initialize manifold maps
        self.exp_map = HyperbolicExponential(dim=self.manifold_dim)
        self.log_map = HyperbolicLogarithm(dim=self.manifold_dim)
        self.transport = ParallelTransport(dim=self.manifold_dim)
        
        # Initialize weights
        self._init_weights()
        
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
            for layer in [self.to_qkv, self.to_out, self.query, self.key, self.value]:
                if hasattr(layer, 'weight'):
                    real_weight = torch.empty_like(layer.weight.real)
                    nn.init.normal_(real_weight, std=0.02)
                    imag_weight = torch.empty_like(layer.weight.imag)
                    nn.init.normal_(imag_weight, std=0.02)
                    layer.weight.copy_(torch.complex(real_weight, imag_weight))
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.zero_()
            
            # Initialize attention layers
            for layer in self.attention_layers:
                if hasattr(layer, 'weight'):
                    real_weight = torch.empty_like(layer.weight.real)
                    nn.init.normal_(real_weight, std=0.02)
                    imag_weight = torch.empty_like(layer.weight.imag)
                    nn.init.normal_(imag_weight, std=0.02)
                    layer.weight.copy_(torch.complex(real_weight, imag_weight))
                if hasattr(layer, 'bias') and layer.bias is not None:
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply quantum geometric attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len]
            return_metrics: Whether to return attention metrics
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim] and optional metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to manifold space
        x = x.to(torch.complex64)
        x_proj = self.manifold_proj(x.view(-1, self.head_dim)).view(batch_size, seq_len, -1)
        
        # Split into heads
        x_split = x_proj.view(batch_size, seq_len, self.num_heads, -1)
        x_split = x_split.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Process through attention layers
        attention_output: List[torch.Tensor] = []
        metrics_dict: Dict[str, Any] = {} if return_metrics else {}
        
        for i, layer in enumerate(self.attention_layers):
            # Compute attention patterns
            layer_output = layer(x_split, mask=mask, return_metrics=return_metrics)
            
            if return_metrics:
                assert isinstance(layer_output, tuple)
                output, layer_metrics = layer_output
                metrics_dict[f'layer_{i}'] = layer_metrics
            else:
                assert isinstance(layer_output, torch.Tensor)  # Type assertion for mypy
                output = layer_output
                
            assert isinstance(output, torch.Tensor)  # Type assertion for mypy
            attention_output.append(output)
            
        # Combine attention outputs
        combined = torch.stack(attention_output, dim=0)
        assert isinstance(combined, torch.Tensor)  # Type assertion for mypy
        combined = combined.mean(dim=0)
        assert isinstance(combined, torch.Tensor)  # Type assertion for mypy
        assert isinstance(combined, torch.Tensor)  # Type assertion for mypy
        
        # Project back to input space
        output = self.output_proj(combined.view(batch_size, seq_len, -1))
        
        if return_metrics:
            return output, metrics_dict
            
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
        # Use proper tensor operations
        features = F.linear(x, self.metric)
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
        # Reshape input to [batch_size * seq_len, head_dim] if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.reshape(-1, hidden_dim)
        
        # Split hidden_dim into num_heads x head_dim
        if x.shape[-1] == self.hidden_dim:
            x = x.view(-1, self.num_heads, self.head_dim)
            x = x.reshape(-1, self.head_dim)
        
        # Project to manifold space
        x = self.manifold_proj(x)
        
        # Apply quantum attention through tiles
        for tile in self.tiles:
            x = tile(x)
        
        return x

    def compute_attention_patterns(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute attention patterns from query and key tensors.

        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Optional value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask
            return_metrics: Whether to return attention metrics

        Returns:
            Attention patterns and optional metrics
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Project to manifold space
        query = query.to(torch.complex64)
        key = key.to(torch.complex64)

        # Project query and key to manifold
        query = self.manifold_proj(query.view(-1, head_dim)).view(batch_size, num_heads, seq_len, -1)
        key = self.manifold_proj(key.view(-1, head_dim)).view(batch_size, num_heads, seq_len, -1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Convert complex scores to real for softmax
        scores_real = scores.abs()
        attention_weights = torch.softmax(scores_real, dim=-1)

        # Compute output based on whether value is provided
        if value is not None:
            assert isinstance(value, torch.Tensor)  # Type assertion for mypy
            processed_value = value.to(torch.complex64)
            processed_value = self.manifold_proj(processed_value.view(-1, head_dim)).view(batch_size, num_heads, seq_len, -1)
            
            # Convert processed_value to real for matmul with attention_weights
            processed_value_real = torch.view_as_real(processed_value)  # [batch, heads, seq_len, head_dim, 2]
            processed_value_real = processed_value_real.permute(0, 1, 2, 4, 3)  # [batch, heads, seq_len, 2, head_dim]
            processed_value_real = processed_value_real.reshape(batch_size, num_heads, seq_len, -1)  # [batch, heads, seq_len, 2*head_dim]
            attention_output = torch.matmul(attention_weights, processed_value_real)
        else:
            attention_output = attention_weights

        # Return metrics if requested
        if return_metrics:
            metrics = {
                'attention_weights': attention_weights,
                'attention_scores': scores,
                'query_proj': query,
                'key_proj': key,
                'value_proj': processed_value if value is not None else None
            }
            return attention_output, metrics

        return attention_output

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
        else:
            state_tensor = state
            
        batch_size = state_tensor.shape[0]
        
        # Initialize metric tensor
        metric = torch.zeros(
            (batch_size, self.manifold_dim, self.manifold_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        # Compute Jacobian of state map
        with torch.enable_grad():
            state_tensor = state_tensor.requires_grad_(True)
            output = self._compute_quantum_features(state_tensor)
            
            # Create basis vectors for tangent space
            basis_vectors = torch.eye(
                self.manifold_dim,
                device=self.device,
                dtype=self.dtype
            ).reshape(-1, self.manifold_dim)  # [manifold_dim, manifold_dim]
            
            # Compute Jacobian-vector products for each basis vector
            jvps = []
            for v in basis_vectors:
                _, jvp = torch.autograd.functional.jvp(
                    lambda x: self._compute_quantum_features(x),
                    (state_tensor,),
                    (v.expand_as(state_tensor),)
                )
                jvps.append(jvp)
            
            # Stack JVPs to form Jacobian
            jacobian = torch.stack(jvps, dim=1)  # [batch_size, manifold_dim, manifold_dim]
        
        # Compute metric as g_ij = sum_k (∂f_k/∂x_i)(∂f_k/∂x_j)
        metric = torch.einsum('...ki,...kj->...ij', jacobian, jacobian.conj())
        
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
            transport_list = []
            geodesic_list = []
            energy_list = []
        
        # Compute initial metric
        metric = self.compute_metric_tensor(state)
        
        # Initialize flow
        current = x
        for step in range(num_steps):
            # Compute Ricci tensor
            ricci = compute_ricci_tensor(metric)
            
            # Update metric using Ricci flow
            metric = metric + dt * ricci
            
            # Project to manifold
            current = self.manifold_proj(current.view(-1, self.head_dim))
            
            # Apply flow step
            current = self.flow.flow_step(current, metric)
            
            # Project back
            current = self.manifold_proj_inv(current).view(*x.shape)
            
            if return_metrics:
                # Compute flow metrics
                curvature = torch.einsum('...ii', ricci)
                transport = compute_parallel_transport(current, metric)
                geodesic = compute_geodesic_distance(current, metric)
                energy = compute_flow_energy(current, metric)
                
                curvature_list.append(curvature)
                transport_list.append(transport)
                geodesic_list.append(geodesic)
                energy_list.append(energy)
        
        if return_metrics:
            # Create metric context
            context = MetricContext(
                timestamp=0.0,
                device=x.device,
                batch_size=x.shape[0],
                sequence_length=x.shape[1],
                hidden_dim=self.hidden_dim,
                resolution=1.0
            )
            
            # Compute flow metrics
            metrics = compute_flow_metrics(
                flow_path=current,
                metric_tensor=metric,
                context=context
            )
            return current, metrics
        
        return current

    def prepare_quantum_state(self, x: torch.Tensor) -> QuantumState:
        """Prepare quantum state from classical input.
        
        Args:
            x: Classical input tensor
            
        Returns:
            Quantum state representation
        """
        # Project to manifold space
        x_manifold = self.manifold_proj(x.view(-1, self.head_dim))
        
        # Apply quantum attention
        x_quantum = self.quantum_attention(x_manifold)
        
        # Normalize state
        state_norm = torch.norm(x_quantum, dim=-1, keepdim=True)
        quantum_state = x_quantum / (state_norm + 1e-8)
        
        # Create quantum state with proper initialization
        basis_size = quantum_state.shape[-1]
        basis_labels = [str(i) for i in range(basis_size)]
        phase = torch.zeros(1, dtype=self.dtype, device=self.device)
        
        # Validate quantum state
        self.validate_quantum_state(quantum_state)
        
        return QuantumState(
            amplitudes=quantum_state,
            basis_labels=basis_labels,
            phase=phase
        )

    def measure_quantum_state(
        self,
        state: QuantumState,
        observable: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Measure quantum state with optional observable.
        
        Args:
            state: Quantum state to measure
            observable: Optional observable operator
            
        Returns:
            Measurement outcome
        """
        if observable is None:
            # Default to computational basis measurement
            probs = torch.abs(state.amplitudes) ** 2
            return probs
        
        # Compute expectation value
        if observable.dim() == 2:
            # Single observable
            expectation = torch.einsum(
                '...i,ij,...j->...',
                state.amplitudes.conj(),
                observable,
                state.amplitudes
            )
        else:
            # Multiple observables
            expectation = torch.einsum(
                '...i,ijk,...k->...j',
                state.amplitudes.conj(),
                observable,
                state.amplitudes
            )
        return expectation.real

    def compute_berry_phase(
        self,
        state: QuantumState,
        hamiltonian: torch.Tensor,
        time_steps: int = 100
    ) -> torch.Tensor:
        """Compute Berry phase for cyclic evolution.
        
        Args:
            state: Initial quantum state
            hamiltonian: Time-dependent Hamiltonian
            time_steps: Number of time steps for evolution
            
        Returns:
            Berry phase
        """
        # Initialize phase accumulation
        phase = torch.zeros(1, dtype=self.dtype, device=self.device)
        
        # Create time points
        times = torch.linspace(0, 2*torch.pi, time_steps, device=self.device)
        dt = float(times[1] - times[0])  # Convert to float for evolve_state
        
        # Initialize state
        current_state = state
        
        for t in range(time_steps-1):
            # Evolve state with float time step
            next_state = self.evolve_state(current_state, hamiltonian, dt)
            
            # Compute Berry connection
            overlap = torch.vdot(current_state.amplitudes, next_state.amplitudes)
            connection = -torch.log(overlap / torch.abs(overlap)).imag
            
            # Accumulate phase
            phase += connection
            
            current_state = next_state
            
        return phase

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
        """
        # Initialize phase
        phase = torch.zeros(1, dtype=self.dtype, device=self.device)
        current_state = state
        
        # Use unit time step for each unitary evolution
        dt = 1.0
        
        for U in path:
            # Apply unitary with explicit float time step
            next_state = self.evolve_state(current_state, U, dt)
            
            # Compute connection
            overlap = torch.vdot(current_state.amplitudes, next_state.amplitudes)
            connection = torch.log(overlap / torch.abs(overlap)).imag
            
            # Accumulate phase
            phase += connection
            current_state = next_state
            
        return phase

    def compute_entanglement_metrics(self, state: QuantumState) -> Dict[str, torch.Tensor]:
        """Compute entanglement metrics for quantum state.
        
        Args:
            state: Quantum state to analyze
            
        Returns:
            Dictionary of entanglement metrics
        """
        # Get state vector and compute density matrix
        state_vector = state.amplitudes
        density_matrix = torch.einsum('...i,...j->...ij', state_vector, state_vector.conj())
        
        # Compute reduced density matrices
        n = state_vector.shape[-1]
        n_qubits = int(math.log2(n))
        
        metrics = {}
        
        # Compute von Neumann entropy
        eigenvals = torch.linalg.eigvalsh(density_matrix)
        entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-10), dim=-1)
        metrics['von_neumann_entropy'] = entropy
        
        # Compute purity
        purity = torch.einsum('...ij,...ji->...', density_matrix, density_matrix)
        metrics['purity'] = purity
        
        # Compute concurrence for 2-qubit states
        if n_qubits == 2:
            sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)
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
    
    def analyze_pattern_formation(
        self,
        pattern: torch.Tensor,
        time_steps: int = 100
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Analyze pattern formation dynamics.
        
        Args:
            pattern: Input pattern tensor of shape [batch_size, seq_len, hidden_dim]
            time_steps: Number of evolution steps to analyze
            
        Returns:
            Dictionary containing:
            - 'formation_trajectory': Evolution of pattern over time (Tensor)
            - 'stability_metrics': Stability analysis at each time step (Dict[str, Tensor])
            - 'coherence_metrics': Pattern coherence measures (Dict[str, Tensor])
            - 'symmetry_metrics': Symmetry analysis results (Dict[str, Tensor])
        """
        # Initialize pattern formation validator with appropriate thresholds
        validator = PatternFormationValidator(
            tolerance=1e-6,
            coherence_threshold=0.8,
            symmetry_threshold=0.9,
            defect_threshold=0.1,
            frequency_threshold=0.1,
            phase_threshold=0.1
        )
        
        # Initialize pattern dynamics with correct parameters
        dynamics = PatternDynamics(
            grid_size=pattern.shape[1],  # Use sequence length as grid size
            space_dim=2,  # 2D space for attention patterns
            hidden_dim=self.hidden_dim,
            dt=0.01,  # Small time step for stability
            num_modes=self.num_heads,  # Use number of heads as modes
            quantum_enabled=True  # Enable quantum features
        )
        
        # Validate and evolve pattern
        validation_result = validator.validate(
            dynamics=dynamics,
            initial=pattern,
            time_steps=time_steps
        )
        
        # Initialize default metrics with correct types
        default_metrics = {
            'formation_trajectory': torch.zeros_like(pattern),
            'stability_metrics': {'stability': torch.tensor(0.0, device=pattern.device)},
            'coherence_metrics': {'coherence': torch.tensor(0.0, device=pattern.device)},
            'symmetry_metrics': {'symmetry': torch.tensor(0.0, device=pattern.device)}
        }
        
        # Extract metrics from validation result
        if validation_result.data is not None:
            trajectory = validation_result.data.get('trajectory')
            if trajectory is not None:
                default_metrics['formation_trajectory'] = trajectory
            
            stability = validation_result.data.get('stability')
            if isinstance(stability, dict):
                default_metrics['stability_metrics'] = {
                    k: v for k, v in stability.items() 
                    if isinstance(v, torch.Tensor)
                }
            
            coherence = validation_result.data.get('coherence')
            if isinstance(coherence, dict):
                default_metrics['coherence_metrics'] = {
                    k: v for k, v in coherence.items() 
                    if isinstance(v, torch.Tensor)
                }
            
            symmetry = validation_result.data.get('symmetry')
            if isinstance(symmetry, dict):
                default_metrics['symmetry_metrics'] = {
                    k: v for k, v in symmetry.items() 
                    if isinstance(v, torch.Tensor)
                }
        
        return default_metrics
    
    def _validate_metrics(self, metrics: Dict[str, Any]) -> None:
        """Validate metrics dictionary structure and values with detailed checks.
        
        Args:
            metrics: Dictionary of metrics to validate
            
        Raises:
            ValueError: If metrics are invalid
        """
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics must be a dictionary, got {type(metrics)}")
            
        required_keys = {'entropy', 'complexity', 'stability'}
        if not all(k in metrics for k in required_keys):
            raise ValueError(f"Metrics must contain keys: {required_keys}")
            
        for k, v in metrics.items():
            # Check value type
            if v is not None and not isinstance(v, (int, float, torch.Tensor)):
                raise ValueError(f"Metric {k} must be numeric or None, got {type(v)}")
            
            # Check for invalid values
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Metric {k} has invalid value: {v}")
            elif isinstance(v, torch.Tensor):
                if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                    raise ValueError(f"Metric {k} contains NaN or Inf values")
            
            # Check value ranges for specific metrics
            if k == 'entropy' and v is not None:
                if isinstance(v, (int, float)) and (v < 0 or v > math.log(self.hidden_dim)):
                    raise ValueError(f"Invalid entropy value: {v}")
            elif k == 'complexity' and v is not None:
                if isinstance(v, (int, float)) and v < 0:
                    raise ValueError(f"Invalid complexity value: {v}")
            elif k == 'stability' and v is not None:
                if isinstance(v, (int, float)) and (v < 0 or v > 1):
                    raise ValueError(f"Invalid stability value: {v}")

    def _combine_tile_metrics(self, tile_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine metrics from multiple tiles with error handling.
        
        Args:
            tile_metrics: List of metric dictionaries from tiles
            
        Returns:
            Combined metrics dictionary
            
        Raises:
            ValueError: If metrics are invalid or inconsistent
        """
        if not tile_metrics:
            raise ValueError("No tile metrics provided")
            
        combined = {}
        
        try:
            # Combine numeric metrics by averaging
            numeric_keys = {'attention_entropy', 'output_norm', 'quantum_entropy', 'complexity'}
            for key in numeric_keys:
                values = [m[key] for m in tile_metrics if key in m]
                if values:
                    # Check for invalid values
                    if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in values):
                        raise ValueError(f"Invalid values for metric {key}")
                    combined[key] = sum(values) / len(values)
            
            # Combine attention metrics
            attention_keys = {'attention_scores', 'attention_probs'}
            for key in attention_keys:
                tensors = [m[key] for m in tile_metrics if key in m]
                if tensors:
                    # Validate tensor shapes
                    if not all(t.shape == tensors[0].shape for t in tensors):
                        raise ValueError(f"Inconsistent shapes for metric {key}")
                    combined[key] = torch.stack(tensors).mean(dim=0)
            
            # Combine quantum geometric metrics
            quantum_keys = {'quantum_geometric'}
            for key in quantum_keys:
                metrics = [m[key] for m in tile_metrics if key in m]
                if metrics:
                    # Validate metric structure
                    required_fields = {'entropy', 'complexity', 'stability'}
                    if not all(all(f in m for f in required_fields) for m in metrics):
                        raise ValueError(f"Missing required fields in {key}")
                        
                    combined[key] = {
                        field: sum(m[field] for m in metrics) / len(metrics)
                        for field in required_fields
                    }
                    
            return combined
            
        except Exception as e:
            raise ValueError(f"Error combining tile metrics: {str(e)}")

    def _update_quantum_state(self, state: QuantumState) -> Dict[str, Any]:
        """Update quantum state with current metrics and validation.
        
        Args:
            state: Quantum state to update
            
        Returns:
            Dictionary of computed metrics
        """
        # Validate state
        self.validate_quantum_state(state.amplitudes)
        
        # Compute quantum metrics
        metrics = self.compute_entanglement_metrics(state)
        
        # Return metrics instead of storing them
        return metrics

    def _process_tile_output(
        self,
        tile_output: Union[torch.Tensor, PatternSection],
        tile_metrics: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process output from a quantum tile.
        
        Args:
            tile_output: Output tensor or pattern section from tile
            tile_metrics: Metrics from tile processing
            
        Returns:
            Tuple of:
            - Processed classical tensor
            - Updated metrics dictionary
        """
        # Extract tensor from pattern section if needed
        output_tensor = tile_output.coordinates if isinstance(tile_output, PatternSection) else tile_output
        
        # Convert to classical representation
        classical_output = self.quantum_to_classical(output_tensor)
        
        # Validate and update metrics
        self._validate_metrics(tile_metrics)
        
        # Add quantum geometric metrics
        quantum_metrics = compute_attention_metrics(
            attention_patterns=classical_output,
            metric_context=MetricContext(
                timestamp=0.0,
                device=classical_output.device,
                batch_size=classical_output.shape[0],
                sequence_length=classical_output.shape[1],
                hidden_dim=self.hidden_dim,
                resolution=1.0
            )
        )
        
        updated_metrics = {
            **tile_metrics,
            'quantum_geometric': {
                'entropy': float(quantum_metrics.entropy.mean().item()),
                'complexity': float(quantum_metrics.complexity.mean().item()),
                'stability': float(quantum_metrics.pattern_stability.mean().item()) if quantum_metrics.pattern_stability is not None else None
            }
        }
        
        return classical_output, updated_metrics
    
    def _prepare_quantum_state(self, x: torch.Tensor) -> QuantumState:
        """Prepare quantum state from classical input with validation.
        
        Args:
            x: Classical input tensor
            
        Returns:
            Validated quantum state
            
        Raises:
            ValueError: If state preparation fails validation
        """
        # Convert to quantum state
        if not isinstance(x, QuantumState):
            if not torch.is_complex(x):
                x = x.to(torch.complex64)
            state = QuantumState(
                amplitudes=x,
                basis_labels=[str(i) for i in range(self.manifold_dim)],
                phase=torch.zeros(1, dtype=self.dtype, device=self.device)
            )
        else:
            state = x
            
        # Validate preparation
        validation_result = self.preparation_validator.validate_preparation(
            target=state,
            prepared=state  # Self-validation for initial preparation
        )
        
        if not validation_result.is_valid:
            # Attempt to correct the state
            state = self.preparation_validator.correct_state(
                state=state,
                error_type=validation_result.error_type or StateValidationErrorType.INVALID_NORM
            )
            
        return state
        
    def _measure_quantum_state(self, state: QuantumState) -> torch.Tensor:
        """Measure quantum state and project to classical representation.
        
        Args:
            state: Quantum state to measure
            
        Returns:
            Classical tensor representation
        """
        # Validate state before measurement
        properties = self.state_validator.validate_state(state)
        
        # Project state to classical representation
        if properties.is_pure:
            # For pure states, use direct projection
            classical = state.amplitudes
        else:
            # For mixed states, use density matrix eigendecomposition
            density_matrix = torch.matmul(
                state.amplitudes,
                state.amplitudes.conj().transpose(-2, -1)
            )
            eigenvals, eigenvecs = torch.linalg.eigh(density_matrix)
            # Use dominant eigenvector
            classical = eigenvecs[..., -1]
            
        return classical.real
        
    def _compute_density_matrix(self, state: QuantumState) -> torch.Tensor:
        """Compute density matrix representation of quantum state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Density matrix tensor
        """
        # Validate state
        properties = self.state_validator.validate_state(state)
        
        # Compute density matrix
        if properties.is_pure:
            # Pure state density matrix
            return torch.matmul(
                state.amplitudes.unsqueeze(-1),
                state.amplitudes.conj().unsqueeze(-2)
            )
        else:
            # Mixed state density matrix
            return torch.matmul(
                state.amplitudes,
                state.amplitudes.conj().transpose(-2, -1)
            )
            
    def _compute_von_neumann_entropy(self, state: Union[QuantumState, torch.Tensor]) -> torch.Tensor:
        """Compute von Neumann entropy of quantum state.
        
        Args:
            state: Either a QuantumState object or tensor of shape [..., manifold_dim]
            
        Returns:
            Von Neumann entropy tensor
            
        Note:
            For pure states S = 0
            For mixed states 0 < S ≤ log(d) where d is manifold dimension
        """
        # Ensure we have a QuantumState object
        if not isinstance(state, QuantumState):
            state = self._prepare_quantum_state(state)
            
        # Validate state
        properties = self.state_validator.validate_state(state)
        
        # Use HilbertSpace to compute entropy
        return self.hilbert_space.compute_entropy(state)
    
    def _geometric_update(self, x: torch.Tensor) -> torch.Tensor:
        """Updates based on manifold structure using geometric flow.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Updated tensor with same shape as input
        """
        # Project to manifold space if needed
        if x.shape[-1] != self.flow.manifold_dim:
            x_manifold = x[..., :self.flow.manifold_dim]
        else:
            x_manifold = x
        
        # Apply geometric flow and get metrics
        x_flowed, flow_metrics = self.flow(
            x_manifold,
            return_path=False  # We only need final state
        )
        
        # The flow already handles:
        # 1. Computing the metric via compute_metric()
        # 2. Computing Ricci tensor via compute_ricci_tensor()
        # 3. Performing integration steps with flow_step()
        # 4. Applying quantum corrections through arithmetic structure
        
        return x_flowed
    
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
            attention_scores = torch.zeros(batch_size, self.num_heads, seq_len, seq_len,
                                        device=mask.device, dtype=self.dtype)
            state.attention_scores = attention_scores
            
        # Create mask for attention scores
        # Expand mask to match attention scores shape [batch_size, num_heads, seq_len, seq_len]
        expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, -1, mask.size(1))
        
        # Apply mask by setting masked positions to -inf
        masked_scores = attention_scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Update state with masked scores
        state.attention_scores = masked_scores
        
        return state
    
