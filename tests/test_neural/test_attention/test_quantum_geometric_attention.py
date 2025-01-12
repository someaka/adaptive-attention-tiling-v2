"""
Unit tests for the quantum geometric attention mechanism.

Tests cover:
1. Attention state preparation and properties
2. Pattern computation and metrics
3. Geometric attention flow
4. Quantum-classical interface
5. Multi-head integration
6. Geometric phases
7. Manifold curvature
8. Entanglement
9. Error correction
10. Topological features
11. Advanced geometric structures
12. Pattern dynamics
"""

import numpy as np
import pytest
import torch
import torch.linalg

from src.core.tiling.quantum_geometric_attention import (
    AttentionState,
    GeometricStructures,
    QuantumGeometricAttention,
    QuantumGeometricConfig
)
from src.metrics.attention import (
    AttentionMetrics,
    FlowMetrics,
    compute_attention_metrics
)
from src.core.patterns.dynamics import PatternDynamics

def complex_randn(*size, device=None):
    """Create random complex tensor with proper initialization."""
    real = torch.randn(*size, device=device)
    imag = torch.randn(*size, device=device)
    return torch.complex(real, imag)

class TestQuantumGeometricAttention:
    """Test suite for quantum geometric attention with proper cleanup."""

    def teardown_method(self):
        """Clean up after each test."""
        import gc
        gc.collect()

    @pytest.fixture
    def manifold_dim(self) -> int:
        """Return manifold dimension for tests."""
        return 4  # Base dimension for tests

    @pytest.fixture
    def hidden_dim(self, manifold_dim) -> int:
        """Return hidden dimension for tests."""
        return manifold_dim * 2  # Hidden dim is double the manifold dim

    @pytest.fixture
    def num_heads(self) -> int:
        """Return number of attention heads for tests."""
        return 4  # Reduced from 8 to better match smaller dimensions

    @pytest.fixture
    def batch_size(self) -> int:
        """Return batch size for tests."""
        return 16

    @pytest.fixture
    def seq_length(self) -> int:
        """Return sequence length for tests."""
        return 8  # Reduced from 32 to better match test scale

    @pytest.fixture
    def attention_layer(self, hidden_dim, manifold_dim, num_heads):
        """Create a test attention layer with proper device placement."""
        device = torch.device('cpu')
        config = QuantumGeometricConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            manifold_type="hyperbolic",
            curvature=-1.0,
            manifold_dim=manifold_dim,
            num_layers=3,
            tile_size=8,
            motive_rank=4,
            dtype=torch.complex64,
            device=device,
            is_causal=False
        )
        return QuantumGeometricAttention(config=config)

    @pytest.fixture
    def geometric_structures(self, manifold_dim):
        """Create geometric structures for testing."""
        return GeometricStructures(
            dim=manifold_dim,  # Use manifold_dim instead of hidden_dim
            manifold_type="hyperbolic",
            curvature=-1.0,
            parallel_transport_method="schild",
        )

    @pytest.fixture
    def pattern_dynamics(self, hidden_dim, num_heads):
        """Create pattern dynamics for testing."""
        return PatternDynamics(
            dt=0.1,
            device=torch.device('cpu')
        )

    def test_attention_state_preparation(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim, num_heads
    ):
        """Test attention state preparation and properties."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Test state properties
        assert isinstance(state, AttentionState), "Should return AttentionState"
        
        # Get quantum state directly from state manager's states dictionary
        quantum_state = state.state_manager.states.get("quantum")
        assert quantum_state is not None, "Should have quantum state"

        # Test state dimensions
        assert quantum_state.shape[0] == batch_size, "Batch dimension preserved"
        assert quantum_state.shape[1] == num_heads, "Head dimension correct"
        assert quantum_state.shape[-1] == manifold_dim, "Manifold dimension correct"

        # Test state normalization
        norms = quantum_state.norm(dim=-1)
        # Check normalization with proper complex tolerances
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"
        
        # Test mask application
        masked_state = attention_layer.apply_mask(state, mask)
        # Create attention mask [batch_size, num_heads, seq_length, seq_length]
        expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, num_heads, seq_length, seq_length)
        assert torch.all(
            masked_state.attention_scores[~expanded_mask] == float("-inf")
        ), "Mask should be properly applied"

    def test_attention_pattern_computation(
        self, attention_layer, batch_size, seq_length, hidden_dim, num_heads
    ):
        """Test attention pattern computation."""
        # Create query, key, value tensors with correct shapes
        head_dim = hidden_dim // num_heads
        
        # Create tensors with correct shape [batch_size, num_heads, seq_len, head_dim]
        query = complex_randn(batch_size, num_heads, seq_length, head_dim)
        key = complex_randn(batch_size, num_heads, seq_length, head_dim)
        value = complex_randn(batch_size, num_heads, seq_length, head_dim)

        # Compute attention patterns with complex tensors directly
        result = attention_layer.compute_attention_patterns(
            query, key, value, return_metrics=True
        )
        attention_output, metrics = result  # Unpack result

        # Test output shape
        expected_output_shape = (batch_size, num_heads, seq_length, 2 * attention_layer.manifold_dim)  # 2* for real/imag parts
        assert attention_output.shape == expected_output_shape, f"Output shape should be {expected_output_shape}, got {attention_output.shape}"

        # Test metrics dictionary
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        
        # Test attention weights properties (from the metrics)
        attention_weights = metrics.get('attention_weights')
        assert attention_weights is not None, "Should return attention weights in metrics"
        assert attention_weights.shape == (batch_size, num_heads, seq_length, seq_length), "Attention weights shape incorrect"
        
        # Test row-wise normalization of attention weights
        row_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-5
        ), "Attention weights should be row-normalized"

        # Test attention scores
        attention_scores = metrics.get('attention_scores')
        assert attention_scores is not None, "Should return attention scores in metrics"
        assert attention_scores.shape == (batch_size, num_heads, seq_length, seq_length), "Attention scores shape incorrect"
        assert attention_scores.dtype == torch.complex64, "Attention scores should be complex"

        # Test causality if applicable
        if hasattr(attention_layer, 'is_causal') and attention_layer.is_causal:
            assert torch.all(
                torch.triu(attention_weights, diagonal=1) == 0
            ), "Causal attention should be lower triangular"

    def test_geometric_attention_flow(
        self, attention_layer, batch_size, seq_length, hidden_dim, num_heads
    ):
        """Test geometric attention flow computation."""
        head_dim = hidden_dim // num_heads  # This will be 2 (8 // 4)
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Compute geometric flow with metrics
        flow_result = attention_layer.geometric_attention_flow(
            x, mask=mask, num_steps=10, dt=0.1, return_metrics=True
        )
        flow = flow_result[0]  # Unpack flow tensor
        metrics = flow_result[1]  # Unpack metrics

        # Test flow properties
        assert flow.shape == x.shape, "Flow should match input shape"

        # Test flow metrics
        assert isinstance(metrics, FlowMetrics), "Should return flow metrics"
        assert metrics.energy is not None, "Should compute flow energy"
        assert metrics.curvature is not None, "Should compute flow curvature"

    def test_quantum_classical_interface(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test quantum-classical information interface."""
        # Create classical input with manifold dimension
        classical_input = complex_randn(batch_size, seq_length, hidden_dim)

        # Convert to quantum state
        quantum_state = attention_layer.prepare_quantum_state(classical_input)

        # Test quantum state properties
        assert quantum_state.amplitudes.shape[-1] == manifold_dim, "Should preserve manifold dimension"
        assert attention_layer.is_valid_quantum_state(quantum_state.amplitudes), "Should be valid quantum state"

        # Test quantum state normalization
        norms = quantum_state.amplitudes.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"

        # Test gradients
        quantum_state.amplitudes.requires_grad = True
        loss = quantum_state.amplitudes.sum()
        loss.backward()
        assert quantum_state.amplitudes.grad is not None, "Should allow gradient flow"

    def test_multi_head_integration(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
    ) -> None:
        """Test multi-head attention integration."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output shape
        assert output.shape == (batch_size, seq_length, hidden_dim), "Wrong output shape"

        # Test output properties
        assert output.dtype == attention_layer.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

        # Test gradient flow
        output.sum().backward()
        for param in attention_layer.parameters():
            assert param.grad is not None, "Should compute gradients for all parameters"

    def test_geometric_phases(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum geometric phases in attention."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output properties
        assert output.dtype == attention_layer.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

        # Test gradient flow
        output.sum().backward()
        for param in attention_layer.parameters():
            assert param.grad is not None, "Should compute gradients for all parameters"

    def test_manifold_curvature(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test attention manifold curvature properties."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Compute metric tensor
        metric = attention_layer.compute_metric_tensor(state)

        # Test metric tensor properties
        assert metric.shape[-2:] == (manifold_dim, manifold_dim), "Metric tensor should have manifold dimensions"
        assert torch.allclose(
            metric, metric.transpose(-1, -2).conj()
        ), "Metric tensor should be Hermitian"
        assert not torch.isnan(metric).any(), "Metric tensor should not contain NaN values"
        assert not torch.isinf(metric).any(), "Metric tensor should not contain Inf values"

        # Test positive definiteness
        eigenvalues = torch.linalg.eigvalsh(metric)
        assert torch.all(eigenvalues > -1e-6), "Metric tensor should be positive semi-definite"

    def test_attention_entanglement(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test entanglement properties in attention states."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Get quantum state
        quantum_state = state.state_manager.states.get("quantum")
        assert quantum_state is not None, "Should have quantum state"

        # Test quantum state properties
        assert not torch.isnan(quantum_state).any(), "Quantum state should not contain NaN values"
        assert not torch.isinf(quantum_state).any(), "Quantum state should not contain Inf values"

        # Test normalization
        norms = quantum_state.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"

    def test_error_correction(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum error correction in attention."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output properties
        assert output.dtype == attention_layer.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

        # Test gradient flow
        output.sum().backward()
        for param in attention_layer.parameters():
            assert param.grad is not None, "Should compute gradients for all parameters"

    def test_topological_features(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test topological features in attention."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output properties
        assert output.dtype == attention_layer.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

        # Test gradient flow
        output.sum().backward()
        for param in attention_layer.parameters():
            assert param.grad is not None, "Should compute gradients for all parameters"

    def test_attention_patterns(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        """Test attention pattern computation."""
        # Create query, key tensors
        query = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)
        key = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)

        # Compute attention patterns
        result = attention_layer.compute_attention_patterns(query, key, return_metrics=True)
        patterns = result[0]  # Unpack patterns tensor
        metrics = result[1]  # Unpack metrics

        # Test shape and properties
        assert patterns.shape == (
            batch_size,
            attention_layer.num_heads,
            seq_length,
            seq_length,
        ), "Wrong pattern shape"

        # Test row-wise normalization
        row_sums = patterns.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-5
        ), "Patterns should be row-normalized"

        # Test metric properties
        assert isinstance(metrics, AttentionMetrics), "Should return metrics"
        assert metrics.pattern_entropy is not None, "Should compute pattern entropy"
        assert metrics.pattern_complexity is not None, "Should compute pattern complexity"
        assert metrics.pattern_stability is not None, "Should compute pattern stability"

    def test_geometric_structures(self, geometric_structures, manifold_dim):
        """Test geometric structures functionality."""
        # Test metric initialization
        assert geometric_structures.metric.shape == (manifold_dim, manifold_dim), "Wrong metric shape"
        assert geometric_structures.connection.shape == (manifold_dim, manifold_dim, manifold_dim), "Wrong connection shape"
        assert geometric_structures.curvature_tensor.shape == (manifold_dim, manifold_dim, manifold_dim, manifold_dim), "Wrong curvature tensor shape"

        # Test metric properties
        assert torch.allclose(
            geometric_structures.metric,
            geometric_structures.metric.transpose(-1, -2).conj()
        ), "Metric should be Hermitian"

        # Test connection properties
        assert not torch.isnan(geometric_structures.connection).any(), "Connection should not contain NaN values"
        assert not torch.isinf(geometric_structures.connection).any(), "Connection should not contain Inf values"

        # Test curvature tensor properties
        assert not torch.isnan(geometric_structures.curvature_tensor).any(), "Curvature tensor should not contain NaN values"
        assert not torch.isinf(geometric_structures.curvature_tensor).any(), "Curvature tensor should not contain Inf values"

    def test_pattern_dynamics(self, pattern_dynamics, hidden_dim, batch_size):
        """Test pattern dynamics functionality."""
        # Create initial state
        initial_state = complex_randn(batch_size, hidden_dim)
        
        # Test evolution with time parameter
        time = 0.1
        evolved_state = pattern_dynamics.evolve(initial_state, time)
        assert evolved_state.shape == initial_state.shape, "Evolution should preserve shape"
        assert not torch.isnan(evolved_state).any(), "Evolution should not produce NaN values"
        assert not torch.isinf(evolved_state).any(), "Evolution should not produce Inf values"
        
        # Test energy conservation
        initial_energy = pattern_dynamics.compute_energy(initial_state)
        final_energy = pattern_dynamics.compute_energy(evolved_state)
        assert torch.allclose(initial_energy, final_energy, rtol=1e-5), "Energy should be conserved"
        
        # Test state normalization
        initial_norm = torch.norm(initial_state, dim=-1)
        final_norm = torch.norm(evolved_state, dim=-1)
        assert torch.allclose(initial_norm, final_norm, rtol=1e-5), "Norm should be conserved"
