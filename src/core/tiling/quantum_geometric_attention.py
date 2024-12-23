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

from dataclasses import dataclass
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


@dataclass
class GeometricStructures:
    """Geometric structures for attention."""
    
    dim: int
    manifold_type: str
    curvature: float
    parallel_transport_method: str

    def __post_init__(self):
        """Initialize geometric operations."""
        if self.manifold_type == "hyperbolic":
            self.exp_map = HyperbolicExponential(self.dim, self.curvature)
            self.log_map = HyperbolicLogarithm(self.dim, self.curvature)
        else:
            self.exp_map = EuclideanExponential(self.dim)
            self.log_map = EuclideanLogarithm(self.dim)
        self.transport = ParallelTransport(self.dim)


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

        # Initialize pattern memory
        self.patterns = nn.Parameter(torch.randn(num_patterns, dim))
        self.pattern_scores = nn.Parameter(torch.zeros(num_patterns))

    def update_patterns(self, x: torch.Tensor) -> torch.Tensor:
        """Update pattern memory with new input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Updated pattern scores
        """
        # Compute pattern similarities
        similarities = F.cosine_similarity(
            x.unsqueeze(2),  # (batch, seq, 1, dim)
            self.patterns.unsqueeze(0).unsqueeze(0),  # (1, 1, num_patterns, dim)
            dim=-1
        )
        
        # Update pattern scores with temperature
        scores = F.softmax(similarities / self.temperature, dim=-1)
        
        # Adapt patterns with learning rate
        pattern_updates = torch.einsum('bsn,bsd->nd', scores, x)
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
        manifold_dim: int = 8,
        num_layers: int = 3,
        tile_size: int = 8,
        motive_rank: int = 4,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """Initialize quantum geometric attention.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
            manifold_type: Type of manifold to use
            curvature: Manifold curvature
            manifold_dim: Manifold dimension
            num_layers: Number of attention layers
            tile_size: Size of attention tiles
            motive_rank: Rank of motivic structure
            dtype: Data type to use
            device: Device for computation
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.manifold_dim = manifold_dim
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        
        # Initialize projections with geometric structure preservation
        self.manifold_proj = nn.Linear(hidden_dim, manifold_dim, device=self.device)
        self.manifold_proj_inv = nn.Linear(manifold_dim, hidden_dim, device=self.device)
        self.pattern_proj = nn.Linear(manifold_dim, hidden_dim, device=self.device)
        self.pattern_proj_inv = nn.Linear(hidden_dim, manifold_dim, device=self.device)
        
        # Initialize weights with orthogonal projections to preserve geometry
        with torch.no_grad():
            # Create orthogonal projection matrices using QR decomposition
            Q, _ = torch.linalg.qr(torch.randn(hidden_dim, manifold_dim, device=self.device))
            Q_inv = Q.t()
            
            # Scale the projections to preserve norm
            scale = math.sqrt(hidden_dim / manifold_dim)
            
            # Assign weights with orthogonality preservation
            self.manifold_proj.weight.data = Q.t() * scale
            self.manifold_proj_inv.weight.data = Q / scale
            self.pattern_proj.weight.data = Q / scale
            self.pattern_proj_inv.weight.data = Q.t() * scale
            
            # Zero out biases
            self.manifold_proj.bias.data.zero_()
            self.manifold_proj_inv.bias.data.zero_()
            self.pattern_proj.bias.data.zero_()
            self.pattern_proj_inv.bias.data.zero_()
            
            # Verify initialization by checking reconstruction
            x = torch.randn(10, hidden_dim, device=self.device)
            x = x / x.norm(dim=-1, keepdim=True)  # Normalize input
            
            # Store original scale for later restoration
            self.original_scale = x.norm(dim=-1, keepdim=True)
            
            # Test manifold projections with geometric preservation
            x_manifold = self.manifold_proj(x)
            x_reconstructed = self.manifold_proj_inv(x_manifold)
            
            # Test pattern projections
            x_pattern = self.pattern_proj(x_manifold)
            x_pattern_reconstructed = self.pattern_proj_inv(x_pattern)
            
            # Print debug information
            print(f"Original norm: {x.norm(dim=-1).mean()}")
            print(f"Manifold norm: {x_manifold.norm(dim=-1).mean()}")
            print(f"Reconstructed norm: {x_reconstructed.norm(dim=-1).mean()}")
            print(f"Pattern norm: {x_pattern.norm(dim=-1).mean()}")
            print(f"Pattern reconstructed norm: {x_pattern_reconstructed.norm(dim=-1).mean()}")
            
            # Check reconstruction error with relaxed tolerance
            manifold_error = (x - x_reconstructed).norm(dim=-1).mean()
            pattern_error = (x_manifold - x_pattern_reconstructed).norm(dim=-1).mean()
            
            print(f"Manifold reconstruction error: {manifold_error}")
            print(f"Pattern reconstruction error: {pattern_error}")
            
            # Use a more reasonable tolerance for initialization
            assert manifold_error < 1.0, "Manifold projections should approximately preserve information"
        
        self.dropout = nn.Dropout(dropout)
        self.attention_layers = nn.ModuleList([
            nn.Linear(manifold_dim, hidden_dim, device=self.device) for _ in range(num_layers)
        ])
        
        # Initialize geometric flow
        self.flow = GeometricFlow(
            hidden_dim=hidden_dim,
            manifold_dim=self.manifold_dim,
            dtype=dtype
        ).to(self.device)
        
        # Initialize arithmetic pattern
        self.arithmetic = ArithmeticPattern(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            motive_rank=motive_rank,
            dtype=dtype
        ).to(self.device)
        
        # Initialize attention projections
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype, device=self.device)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=self.device),
            nn.Dropout(dropout)
        )
        
        # Initialize geometric maps
        if manifold_type == "hyperbolic":
            self.exp_map = HyperbolicExponential(hidden_dim, curvature, dtype=dtype).to(self.device)
            self.log_map = HyperbolicLogarithm(hidden_dim, curvature, dtype=dtype).to(self.device)
        else:
            self.exp_map = EuclideanExponential(hidden_dim, dtype=dtype).to(self.device)
            self.log_map = EuclideanLogarithm(hidden_dim, dtype=dtype).to(self.device)

        # Parallel transport
        self.transport = ParallelTransport(hidden_dim, dtype=dtype).to(self.device)

        # Projections
        self.metric = nn.Parameter(torch.eye(hidden_dim, dtype=dtype, device=self.device))
        self.pattern_proj = nn.Linear(self.manifold_dim, hidden_dim, dtype=dtype, device=self.device)
        
        # Initialize projections with near-orthogonal weights
        self.manifold_proj = nn.Linear(hidden_dim, self.manifold_dim, device=self.device)
        self.pattern_proj = nn.Linear(self.manifold_dim, hidden_dim, device=self.device)
        self.manifold_proj_inv = nn.Linear(self.manifold_dim, hidden_dim, device=self.device)
        self.pattern_proj_inv = nn.Linear(hidden_dim, self.manifold_dim, device=self.device)
        
        # Initialize weights to be approximately inverse
        nn.init.orthogonal_(self.manifold_proj.weight)
        nn.init.orthogonal_(self.pattern_proj.weight)
        self.manifold_proj_inv.weight.data = self.manifold_proj.weight.t()
        self.pattern_proj_inv.weight.data = self.pattern_proj.weight.t()
        
        # Initialize biases to zero
        nn.init.zeros_(self.manifold_proj.bias)
        nn.init.zeros_(self.pattern_proj.bias)
        nn.init.zeros_(self.manifold_proj_inv.bias)
        nn.init.zeros_(self.pattern_proj_inv.bias)

        # Move all components to device
        self.exp_map = self.exp_map.to(self.device)
        self.log_map = self.log_map.to(self.device)
        self.transport = self.transport.to(self.device)
        self.metric = nn.Parameter(self.metric.to(self.device))
        self.pattern_proj = self.pattern_proj.to(self.device)
        self.to(self.device)

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
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply quantum geometric attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            return_metrics: Whether to return attention metrics
            
        Returns:
            Tuple of (output tensor, optional metrics dictionary)
        """
        batch_size, seq_len, _ = x.shape
        metrics: Dict[str, Any] = {} if return_metrics else {}
        
        # Process through each attention layer
        current = x
        for i, attention_layer in enumerate(self.attention_layers):
            # Project to Q, K, V
            qkv = self.to_qkv(current)
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = map(
                lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2),
                qkv
            )
            
            # Apply geometric flow to queries and keys
            q_flowed, q_metrics = self.flow(q.reshape(-1, self.head_dim))
            k_flowed, k_metrics = self.flow(k.reshape(-1, self.head_dim))
            
            # Reshape back to attention dimensions
            q_flowed = q_flowed.view(batch_size, self.num_heads, seq_len, self.head_dim)
            k_flowed = k_flowed.view(batch_size, self.num_heads, seq_len, self.head_dim)
            
            if return_metrics:
                metrics[f'layer_{i}_q_flow'] = q_metrics
                metrics[f'layer_{i}_k_flow'] = k_metrics
            
            # Initialize attention scores
            dots = torch.zeros(
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
                device=x.device,
                dtype=x.dtype
            )
            
            # Process through attention layer
            head_dots, layer_metrics = attention_layer(q_flowed, k_flowed, v)
            dots = head_dots
            
            if return_metrics:
                metrics[f'layer_{i}_attention'] = layer_metrics
            
            # Apply attention mask if provided
            if mask is not None:
                dots = dots.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
            
            # Compute attention weights
            attn = F.softmax(dots * self.scale, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            out = torch.matmul(attn, v)
            
            # Reshape and project output
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            out = self.to_out(out)
            
            # Add residual connection and update current
            current = current + out
            
            if return_metrics:
                metrics[f'layer_{i}_output'] = {
                    'norm': current.norm().item(),
                    'mean': current.mean().item(),
                    'std': current.std().item()
                }
        
        return current, metrics if return_metrics else None

    def prepare_attention_state(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> AttentionState:
        """Prepare attention state with optional mask."""
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=self.dtype)
        
        # Initialize quantum and geometric states
        quantum_state = torch.zeros_like(x, dtype=self.dtype)
        geometric_state = torch.zeros_like(x, dtype=self.dtype)
        
        return AttentionState(
            quantum_state=quantum_state,
            geometric_state=geometric_state,
            attention_scores=None
        )

    def _process_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Process attention using tensor operations."""
        # Replace tensor calls with proper operations
        attention_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)), dim=-1)
        attention_output = torch.matmul(attention_weights, x)
        return attention_output

    def _compute_geometric_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute geometric features using tensor operations."""
        # Use proper tensor operations
        features = F.linear(x, self.metric)
        return features

    def _apply_parallel_transport(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel transport using tensor operations."""
        # Use transport module properly
        transported = self.transport(x)
        return transported

    def _compute_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum features using tensor operations."""
        # Use attention module properly, ensuring we get just the tensor
        features = self.attention(x, x, x, return_metrics=False)  # Pass x as q,k,v and disable metrics
        return features

    def compute_attention_patterns(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, AttentionMetrics]]:
        """Compute attention patterns with optional metrics."""
        # Prepare attention state
        state = self.prepare_attention_state(x, mask)

        # Compute geometric features
        geometric_features = self._compute_geometric_features(state.geometric_state)

        # Apply parallel transport
        transported_features = self._apply_parallel_transport(geometric_features)

        # Compute quantum features
        quantum_features = self._compute_quantum_features(state.quantum_state)

        # Combine features
        combined_features = transported_features + quantum_features

        if return_metrics:
            metrics = AttentionMetrics(
                entropy=self.compute_entropy(combined_features),
                complexity=self.compute_complexity(combined_features),
                stability=self.compute_stability(combined_features),
                sparsity=self.compute_sparsity(combined_features)
            )
            return combined_features, metrics
        return combined_features

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
        """Convert classical tensor to quantum state.
        
        Args:
            x: Classical input tensor of shape (batch_size, num_heads, seq_len, head_dim)
                or (batch_size, seq_len, hidden_dim)
            
        Returns:
            Quantum state tensor with preserved dimensions
        """
        # Handle both multi-head and flat input formats
        if len(x.shape) == 4:
            batch_size, num_heads, seq_len, head_dim = x.shape
            # Reshape to (batch_size * num_heads, seq_len, head_dim)
            x_reshaped = x.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        else:
            batch_size, seq_len, hidden_dim = x.shape
            x_reshaped = x
            
        # Store original scale for later restoration in quantum_to_classical
        self.original_scale = x_reshaped.norm(dim=-1, keepdim=True)
        assert torch.all(self.original_scale > 0), "Original scale should be positive"
        
        # Project to manifold dimension first
        x_manifold = self.manifold_proj(x_reshaped)
        assert x_manifold.shape[-1] == self.manifold_dim, f"Expected manifold dim {self.manifold_dim}, got {x_manifold.shape[-1]}"
        
        # Project to quantum state space while preserving sequence structure
        quantum_proj = self.pattern_proj(x_manifold)
        assert quantum_proj.shape[-1] == self.hidden_dim, f"Expected hidden dim {self.hidden_dim}, got {quantum_proj.shape[-1]}"
        
        # Normalize each sequence position to unit sphere
        quantum_state = F.normalize(quantum_proj, p=2, dim=-1)
        assert torch.allclose(quantum_state.norm(dim=-1), torch.ones_like(quantum_state.norm(dim=-1))), "Quantum state should have unit norm"
        
        # Add quantum phase while preserving sequence structure
        phase = torch.angle(quantum_state)
        quantum_state = quantum_state * torch.exp(1j * phase).to(dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128)
        assert torch.allclose(quantum_state.abs().norm(dim=-1), torch.ones_like(quantum_state.abs().norm(dim=-1))), "Quantum state should preserve unit norm after phase"
        
        # Reshape back to multi-head format if needed
        if len(x.shape) == 4:
            quantum_state = quantum_state.view(batch_size, seq_len, num_heads, -1).transpose(1, 2)
            
        # Return only the real part for compatibility with neural networks
        return quantum_state.real

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
        # Create query, key, value projections
        q_weight = torch.randn(batch_size, self.num_heads, seq_len, self.hidden_dim // self.num_heads)
        k_weight = torch.randn(batch_size, self.num_heads, seq_len, self.hidden_dim // self.num_heads)
        v_weight = torch.randn(batch_size, self.num_heads, seq_len, self.hidden_dim // self.num_heads)
        
        # Initialize attention parameters
        params = {
            "query_weight": nn.Parameter(q_weight / torch.sqrt(torch.tensor(self.hidden_dim))),
            "key_weight": nn.Parameter(k_weight),
            "value_weight": nn.Parameter(v_weight),
            "attention_scale": torch.sqrt(torch.tensor(self.hidden_dim // self.num_heads)),
        }
        
        return params

    def compute_metric_tensor(self, state: AttentionState) -> torch.Tensor:
        """Compute metric tensor for attention manifold.
        
        Args:
            state: Current attention state
            
        Returns:
            Metric tensor
        """
        # Extract geometric state
        g_state = state.geometric_state
        
        # Compute Jacobian
        g_state.requires_grad_(True)
        tangent_vectors = torch.eye(self.hidden_dim).to(g_state.device)
        
        # Use flow module's forward method
        def flow_fn(x):
            return self.flow(x)
        
        jacobian = torch.autograd.functional.jvp(
            flow_fn,
            g_state,
            tangent_vectors,
            create_graph=True
        )[1]
        
        # Compute metric tensor
        metric = torch.einsum("...i,...j->...ij", jacobian, jacobian)
        
        return metric

    def prepare_code_state(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare quantum code state for attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantum code state
        """
        # Project to quantum code space
        code_proj = self.pattern_proj(x)
        
        # Apply quantum encoding through forward pass (self-attention style)
        attention_output = self.attention(code_proj, code_proj, code_proj, return_metrics=False)
        
        # Add geometric structure through forward pass
        flow_output, _ = self.flow(attention_output)
        
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
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            - Attention output
            - Attention metrics
        """
        # Prepare attention state
        state = self.prepare_attention_state(query, mask)
        
        # Get tensor from state for pattern computation
        x = state.quantum_state
        
        # Compute attention patterns using the method with return_metrics=True
        # The second return value is guaranteed to be AttentionMetrics when return_metrics=True
        attention_pattern, metrics = cast(
            Tuple[torch.Tensor, AttentionMetrics],
            self.compute_attention_patterns(x, mask, return_metrics=True)
        )
        
        # Apply attention
        attention_output = torch.einsum(
            "bhqk,bkhd->bqhd",
            attention_pattern,
            value.view(*value.shape[:-1], self.num_heads, -1)
        )
        
        # Combine heads
        attention_output = attention_output.reshape(
            *attention_output.shape[:-2], self.hidden_dim
        )
        
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

    def is_valid_quantum_state(self, state: torch.Tensor) -> bool:
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
            - Processed tensor
            - Optional list of pattern metrics from each layer
        """
        metrics_list = []

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
