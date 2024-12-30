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
from src.core.quantum.state_space import QuantumState


@dataclass
class GeometricStructures:
    """Geometric structures for attention."""
    
    dim: int
    manifold_type: str
    curvature: float
    parallel_transport_method: str
    metric: torch.Tensor = field(default_factory=lambda: torch.eye(1, dtype=torch.float32))

    def __post_init__(self):
        """Initialize geometric operations."""
        # Initialize metric tensor with proper dimensions
        self.metric = torch.eye(self.dim, dtype=torch.float32)
            
        if self.manifold_type == "hyperbolic":
            self.exp_map = HyperbolicExponential(self.dim, self.curvature)
            self.log_map = HyperbolicLogarithm(self.dim, self.curvature)
        else:
            self.exp_map = EuclideanExponential(self.dim)
            self.log_map = EuclideanLogarithm(self.dim)
        self.transport = ParallelTransport(self.dim)
        
        # Validate metric tensor
        self._validate_metric()

    def _validate_metric(self):
        """Validate the metric tensor."""
        if self.metric.shape != (self.dim, self.dim):
            raise ValueError(f"Metric tensor must have shape ({self.dim}, {self.dim})")
        if not torch.allclose(self.metric, self.metric.T):
            raise ValueError("Metric tensor must be symmetric")
        try:
            torch.linalg.cholesky(self.metric)
        except RuntimeError:
            raise ValueError("Metric tensor must be positive definite")


class PatternDynamics:
    """Pattern dynamics for attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_patterns: int,
        temperature: float = 0.1,
        adaptation_rate: float = 0.01
    ):
        """Initialize pattern dynamics.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            num_patterns: Number of patterns to track
            temperature: Temperature for pattern adaptation
            adaptation_rate: Rate of pattern adaptation
        """
        self.dim = dim
        self.num_heads = num_heads
        self.num_patterns = num_patterns
        self.temperature = temperature
        self.adaptation_rate = adaptation_rate

        # Initialize pattern memory with proper dimensions
        self.patterns = nn.Parameter(torch.randn(num_patterns, dim, dim))
        self.pattern_scores = nn.Parameter(torch.zeros(num_patterns))

    def update_patterns(self, x: torch.Tensor) -> torch.Tensor:
        """Update pattern memory with new input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Updated pattern scores of shape (batch_size, seq_len, num_patterns)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute pattern similarities with proper dimension handling
        x_expanded = x.unsqueeze(2).unsqueeze(3)  # (batch, seq, 1, 1, dim)
        patterns_expanded = self.patterns.unsqueeze(0).unsqueeze(0)  # (1, 1, num_patterns, dim, dim)
        
        # Compute similarity as trace of matrix product
        similarities = torch.einsum('bsnij,bsnjk->bsn', x_expanded, patterns_expanded)
        
        # Update pattern scores with temperature
        scores = F.softmax(similarities / self.temperature, dim=-1)
        
        # Adapt patterns with learning rate
        pattern_updates = torch.einsum('bsn,bsij->nij', scores, x.unsqueeze(-1) @ x.unsqueeze(-2))
        self.patterns.data += self.adaptation_rate * pattern_updates
        
        return scores


@dataclass
class AttentionState:
    """State for quantum geometric attention."""
    
    quantum_state: torch.Tensor
    geometric_state: torch.Tensor
    attention_scores: Optional[torch.Tensor] = None


@dataclass
class AttentionMetrics:
    """Metrics for attention patterns."""
    
    entropy: torch.Tensor  # Shape: (batch_size,)
    complexity: torch.Tensor  # Shape: (batch_size,)
    stability: Optional[torch.Tensor] = None  # Shape: (batch_size,)
    sparsity: Optional[torch.Tensor] = None  # Shape: (batch_size,)


@dataclass
class FlowMetrics:
    """Metrics for geometric attention flow."""
    
    curvature: torch.Tensor  # Shape: (batch_size,)
    parallel_transport: torch.Tensor  # Shape: (batch_size, hidden_dim)
    geodesic_distance: torch.Tensor  # Shape: (batch_size,)
    energy: torch.Tensor  # Shape: (batch_size,)


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
        """
        super().__init__()

        # Initialize quantum attention tiles
        self.tiles = nn.ModuleList([
            QuantumMotivicTile(
                size=tile_size,
                hidden_dim=hidden_dim,
                num_heads=1,  # Each tile handles one head
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
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.manifold_dim = manifold_dim if manifold_dim is not None else 4
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        # Initialize original scale buffer for quantum-classical interface
        self.register_buffer(
            'original_scale',
            torch.ones(1, 1, 1, dtype=self.dtype, device=self.device)
        )
        self.device = device or torch.device('cpu')


                
        
        # Initialize metric tensor
        self.register_parameter(
            'metric',
            nn.Parameter(torch.eye(self.manifold_dim, dtype=self.dtype, device=self.device))
        )
        
        # Initialize projection layers with correct dimensions
        self.manifold_proj = nn.Linear(
            self.head_dim,  # Input dimension is head_dim
            self.manifold_dim,  # Output dimension is manifold_dim
            dtype=self.dtype,
            device=self.device
        )
        
        self.manifold_proj_inv = nn.Linear(
            self.manifold_dim,  # Input dimension is manifold_dim
            self.head_dim,  # Output dimension is head_dim
            dtype=self.dtype,
            device=self.device
        )
        
        # Initialize pattern projection layers
        self.pattern_proj = nn.Linear(
            self.manifold_dim,
            self.head_dim,  # Changed to head_dim to match manifold_proj_inv output
            dtype=self.dtype,
            device=self.device
        )
        
        self.pattern_proj_inv = nn.Linear(
            self.head_dim,  # Changed to head_dim to match pattern_proj output
            self.manifold_dim,
            dtype=self.dtype,
            device=self.device
        )

        # Initialize query, key, value transformations
        self.query = nn.Linear(self.manifold_dim, self.manifold_dim, dtype=self.dtype, device=self.device)
        self.key = nn.Linear(self.manifold_dim, self.manifold_dim, dtype=self.dtype, device=self.device)
        self.value = nn.Linear(self.manifold_dim, self.manifold_dim, dtype=self.dtype, device=self.device)

        # Initialize dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize attention layers
        self.attention_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype, device=self.device)
            for _ in range(num_layers)
        ])

        # Initialize to_qkv and to_out projections
        expanded_dim = self.num_heads * self.manifold_dim
        self.to_qkv = nn.Linear(expanded_dim, 3 * expanded_dim, dtype=self.dtype, device=self.device)
        self.to_out = nn.Linear(expanded_dim, hidden_dim, dtype=self.dtype, device=self.device)

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
        """Apply quantum geometric attention."""
        batch_size, seq_len, _ = x.shape
        
        # Ensure return_metrics is a boolean
        return_metrics_bool = bool(return_metrics) if not isinstance(return_metrics, bool) else return_metrics
        metrics = {} if return_metrics_bool else {}
        
        # Process through each attention layer
        current = x
        for i, attention_layer in enumerate(self.attention_layers):
            # Project to Q, K, V in-place
            qkv = self.to_qkv(current)  # [batch_size, seq_len, 3 * expanded_dim]
            q, k, v = qkv.chunk(3, dim=-1)  # each is [batch_size, seq_len, expanded_dim]
            
            # Reshape to separate heads: [batch_size, num_heads, seq_len, manifold_dim]
            q = q.view(batch_size, seq_len, self.num_heads, self.manifold_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.manifold_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.manifold_dim).transpose(1, 2)
            
            # Project to manifold dimension before applying flow
            q_manifold = self.manifold_proj(q.reshape(-1, self.head_dim))  # [batch_size * seq_len * num_heads, manifold_dim]
            k_manifold = self.manifold_proj(k.reshape(-1, self.head_dim))
            
            # Reshape for flow: [batch_size * seq_len * num_heads, manifold_dim]
            q_manifold = q_manifold.reshape(-1, self.manifold_dim)
            k_manifold = k_manifold.reshape(-1, self.manifold_dim)
            
            # Apply geometric flow to queries and keys
            q_flowed, q_metrics = self.flow(q_manifold)
            k_flowed, k_metrics = self.flow(k_manifold)
            
            # Project back to head_dim and reshape: [batch_size, num_heads, seq_len, head_dim]
            q_flowed = self.pattern_proj(q_flowed).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            k_flowed = self.pattern_proj(k_flowed).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            
            if return_metrics_bool:
                metrics[f'layer_{i}_q_flow'] = q_metrics
                metrics[f'layer_{i}_k_flow'] = k_metrics
            
            # Compute attention scores using complex inner product
            scores = torch.matmul(q_flowed, k_flowed.transpose(-2, -1).conj()) / math.sqrt(self.head_dim)
            
            # Apply mask if provided
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)  # Add head and query dimensions
                scores = scores.masked_fill(~mask, float('-inf'))
            
            # Apply attention
            attn = F.softmax(scores.real, dim=-1)  # Use real part for softmax
            attn = self.dropout_layer(attn)
            
            # Apply attention to values
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
            
            # Update current
            current = out
            
            if return_metrics_bool:
                metrics[f'layer_{i}_output'] = {
                    'norm': float(torch.linalg.vector_norm(out).item()),
                    'mean': float(out.mean().item()),
                    'std': float(out.std().item())
                }
        
        # Apply final projection
        out = self.to_out(current)
        
        if return_metrics_bool:
            metrics['final_output'] = {
                'norm': float(torch.linalg.vector_norm(out).item()),
                'mean': float(out.mean().item()),
                'std': float(out.std().item())
            }
            return out, metrics
        
        return out

    def prepare_attention_state(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> AttentionState:
        """Prepare attention state with optional mask."""
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=self.dtype)
        
        # Initialize quantum and geometric states with correct dimensions
        batch_size, seq_len, _ = x.shape
        quantum_state = torch.zeros(
            batch_size, self.num_heads, seq_len, self.head_dim,
            dtype=self.dtype, device=x.device
        )
        geometric_state = torch.zeros(
            batch_size, self.num_heads, seq_len, self.manifold_dim,
            dtype=self.dtype, device=x.device
        )
        
        return AttentionState(
            quantum_state=quantum_state,
            geometric_state=geometric_state,
            attention_scores=None
        )

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
            x: Input tensor of shape [batch_size, seq_len, manifold_dim]
            
        Returns:
            Quantum features of shape [batch_size, seq_len, manifold_dim]
        """
        # Project to manifold
        x = self.manifold_proj(x)
        
        # Compute query, key, value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attn = self.dropout_layer(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        
        # Project back to manifold
        out = self.manifold_proj_inv(out)
        
        return out

    def compute_attention_patterns(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, AttentionMetrics]]:
        """Compute attention patterns with optional metrics.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len, hidden_dim]
            key: Key tensor of shape [batch_size, seq_len, hidden_dim]
            value: Optional value tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask of shape [batch_size, seq_len]
            return_metrics: Whether to return attention metrics
            
        Returns:
            Tuple containing:
            - Attention patterns tensor of shape [batch_size, num_heads, seq_len, seq_len]
            - Optional AttentionMetrics
        """
        batch_size, seq_len, _ = query.shape
        
        # Prepare attention state with proper dimensions
        state = self.prepare_attention_state(query, mask)
        
        # Project to manifold space
        q_manifold = self.manifold_proj(query.view(-1, self.head_dim))
        q_manifold = q_manifold.view(batch_size, seq_len, self.num_heads, -1)
        
        k_manifold = self.manifold_proj(key.view(-1, self.head_dim))
        k_manifold = k_manifold.view(batch_size, seq_len, self.num_heads, -1)
        
        # Compute attention scores with proper scaling
        scores = torch.einsum('bhid,bhjd->bhij', q_manifold, k_manifold) * (self.scale)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add head and query dimensions
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention patterns
        attention_patterns = F.softmax(scores, dim=-1)
        
        if return_metrics:
            # Compute metrics using the attention patterns
            metrics = AttentionMetrics(
                entropy=self.compute_entropy(attention_patterns),
                complexity=self.compute_complexity(attention_patterns),
                stability=self.compute_stability(attention_patterns),
                sparsity=self.compute_sparsity(attention_patterns)
            )
            return attention_patterns, metrics
        
        if value is not None:
            attention_output = torch.einsum("bhij,bhjd->bhid", attention_patterns, self.manifold_proj(value.view(-1, self.head_dim)).view(batch_size, seq_len, self.num_heads, -1))
            if return_metrics:
                return attention_output, metrics
            return attention_output
            
        if return_metrics:
            return attention_patterns, metrics
        return attention_patterns

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

    def compute_entanglement_entropy(self, state: torch.Tensor, split_idx: int) -> torch.Tensor:
        """Compute entanglement entropy across a bipartition.
        
        Args:
            state: Quantum state tensor
            split_idx: Index to split the state
            
        Returns:
            Entanglement entropy
        """
        # Reshape state into matrix
        state_matrix = state.view(-1, 2**split_idx, 2**(state.shape[-1] - split_idx))
        
        # Compute singular value decomposition
        _, s, _ = torch.svd(state_matrix)
        
        # Compute entropy from singular values
        s_sq = s ** 2
        entropy = -torch.sum(s_sq * torch.log(s_sq + 1e-10))
        
        return entropy

    def compute_entropy(self, features: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention features."""
        probs = F.softmax(features, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def compute_complexity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute complexity of attention features."""
        return torch.norm(features, p=2, dim=-1)

    def compute_stability(self, features: torch.Tensor) -> torch.Tensor:
        """Compute stability of attention features."""
        return torch.var(features, dim=-1)

    def compute_sparsity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute sparsity of attention features."""
        return torch.count_nonzero(features, dim=-1).float() / features.shape[-1]

    def classical_to_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Convert classical input to quantum state.

        Args:
            x: Input tensor of shape (batch_size, manifold_dim) or (batch_size, seq_len, manifold_dim)
            
        Returns:
            Quantum state tensor
        """
        # Add sequence dimension if input is flat
        if x.dim() == 2:
            x_reshaped = x.unsqueeze(1)  # (batch_size, 1, manifold_dim)
        else:
            x_reshaped = x
            
        batch_size, seq_len, _ = x_reshaped.shape
        
        # Project to manifold space
        x_manifold = self.manifold_proj(x_reshaped.view(-1, self.manifold_dim))  # (batch_size * seq_len, manifold_dim)
        x_manifold = x_manifold.view(batch_size, seq_len, self.manifold_dim)  # (batch_size, seq_len, manifold_dim)
        
        # Project to pattern space
        x_pattern = self.pattern_proj(x_manifold)  # (batch_size, seq_len, hidden_dim)
        
        # Apply quantum attention
        x_quantum = self.quantum_attention(x_pattern)  # (batch_size, seq_len, hidden_dim)
        
        # Project back to manifold space
        x_manifold = self.pattern_proj_inv(x_quantum)  # (batch_size, seq_len, manifold_dim)
        
        # Project back to original space
        x_out = self.manifold_proj_inv(x_manifold.view(-1, self.manifold_dim))  # (batch_size * seq_len, manifold_dim)
        x_out = x_out.view(batch_size, seq_len, self.manifold_dim)  # (batch_size, seq_len, manifold_dim)
        
        # Remove sequence dimension if input was flat
        if x.dim() == 2:
            x_out = x_out.squeeze(1)  # (batch_size, manifold_dim)
            
        return x_out

    def quantum_to_classical(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Convert quantum state back to classical tensor.
        
        Args:
            quantum_state: Quantum state tensor from classical_to_quantum
            
        Returns:
            Classical tensor with original dimensions
        """
        # Handle both multi-head and flat input formats
        if len(quantum_state.shape) == 4:
            batch_size, num_heads, seq_len, _ = quantum_state.shape
            # Reshape to (batch_size * num_heads, seq_len, hidden_dim)
            x_reshaped = quantum_state.transpose(1, 2).reshape(batch_size * num_heads, seq_len, self.hidden_dim)
        else:
            x_reshaped = quantum_state
            
        assert torch.allclose(x_reshaped.abs().norm(dim=-1), torch.ones_like(x_reshaped.abs().norm(dim=-1))), "Input quantum state should have unit norm"
            
        # Restore original scale before projections
        x_scaled = x_reshaped * self.original_scale
        assert torch.allclose(x_scaled.norm(dim=-1), self.original_scale.squeeze(-1)), "Scale should be restored correctly"
            
        # Project back from quantum state space
        classical_proj = self.pattern_proj_inv(x_scaled)
        assert classical_proj.shape[-1] == self.manifold_dim, f"Expected manifold dim {self.manifold_dim}, got {classical_proj.shape[-1]}"
        
        # Project back from manifold
        classical_output = self.manifold_proj_inv(classical_proj)
        assert classical_output.shape[-1] == (self.head_dim if len(quantum_state.shape) == 4 else self.hidden_dim), "Incorrect output dimension"
        
        # Reshape back to multi-head format if needed
        if len(quantum_state.shape) == 4:
            classical_output = classical_output.view(batch_size, seq_len, num_heads, -1).transpose(1, 2)
            
        return classical_output

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
            state_tensor = state.quantum_state
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
        key_tensor = state.quantum_state if mask is None else mask
        
        # Compute attention patterns with metrics
        attention_pattern, metrics = cast(
            Tuple[torch.Tensor, AttentionMetrics],
            self.compute_attention_patterns(
                state.quantum_state,
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

    def _compute_attention_metrics(self, attention_pattern: torch.Tensor) -> AttentionMetrics:
        """Compute metrics for attention pattern.
        
        Args:
            attention_pattern: Attention pattern tensor
            
        Returns:
            Attention metrics
        """
        # Compute entropy
        entropy = -torch.sum(
            attention_pattern * torch.log(attention_pattern + 1e-10),
            dim=-1
        ).mean()
        
        # Compute complexity using singular values
        u, s, v = torch.svd(attention_pattern)
        complexity = torch.sum(s) / s[0]
        
        # Compute stability through gradient norm
        attention_pattern.requires_grad_(True)
        grad = torch.autograd.grad(
            attention_pattern.sum(), attention_pattern, create_graph=True
        )[0]
        stability = torch.norm(grad, dim=-1).mean()
        
        # Compute sparsity
        sparsity = torch.sum(attention_pattern > 0.1) / attention_pattern.numel()
        
        return AttentionMetrics(
            entropy=entropy,
            complexity=complexity,
            stability=stability,
            sparsity=sparsity
        )

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
            
        # Check normalization
        norms = state.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5):
            raise ValueError("Quantum state must be normalized")
            
        # Check shape compatibility
        if len(state.shape) not in [3, 4]:  # (batch, seq, dim) or (batch, heads, seq, dim)
            raise ValueError("Invalid quantum state shape")
            
        # Check dimension compatibility
        if state.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected hidden dimension {self.hidden_dim}, got {state.shape[-1]}")

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
            ricci = self.flow.compute_ricci_tensor(metric)
            
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
                transport = self._apply_parallel_transport(current)
                geodesic = self.flow.compute_geodesic_distance(x, current)
                energy = self.flow.compute_flow_energy(metric)
                
                curvature_list.append(curvature)
                transport_list.append(transport)
                geodesic_list.append(geodesic)
                energy_list.append(energy)
        
        if return_metrics:
            metrics = FlowMetrics(
                curvature=torch.stack(curvature_list, dim=0).mean(dim=0),
                parallel_transport=torch.stack(transport_list, dim=0).mean(dim=0),
                geodesic_distance=torch.stack(geodesic_list, dim=0).mean(dim=0),
                energy=torch.stack(energy_list, dim=0).mean(dim=0)
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


class QuantumGeometricTransformer(nn.Module):
    """Transformer using quantum geometric attention."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int = 8,
        tile_size: int = 64,
        manifold_type: str = "hyperbolic",
        curvature: float = -1.0,
    ):
        super().__init__()
        
        # Attention layers
        self.layers = nn.ModuleList(
            [
                QuantumGeometricAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    tile_size=tile_size,
                    manifold_type=manifold_type,
                    curvature=curvature,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, return_patterns: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        """Apply quantum geometric transformer.

        Args:
            x: Input tensor
            return_patterns: Whether to return pattern metrics

        Returns:
            Tuple containing:
            - Processed tensor
            - Optional list of pattern metrics from each layer
        """
        metrics_list: List[Dict[str, Any]] = []

        for layer in self.layers:
            # Apply attention with residual
            attended, metrics = layer(x, return_patterns=return_patterns)
            x = x + attended

            # Apply normalization
            x = self.norm(x)

            if return_patterns:
                metrics_list.append(metrics)

        if return_patterns:
            return x, metrics_list

        return x, None
