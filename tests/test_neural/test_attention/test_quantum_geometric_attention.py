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
"""


import numpy as np
import pytest
import torch

from src.neural.attention.quantum_geometric_attention import (
    AttentionMetrics,
    AttentionState,
    FlowMetrics,
    QuantumGeometricAttention,
)


class TestQuantumGeometricAttention:
    @pytest.fixture
    def hidden_dim(self):
        """Hidden dimension size."""
        return 64

    @pytest.fixture
    def num_heads(self):
        """Number of attention heads."""
        return 8

    @pytest.fixture
    def batch_size(self):
        """Batch size for testing."""
        return 16

    @pytest.fixture
    def seq_length(self):
        """Sequence length for testing."""
        return 32

    @pytest.fixture
    def attention_layer(self, hidden_dim, num_heads):
        """Create a test attention layer."""
        return QuantumGeometricAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            motive_rank=4,
            manifold_dim=8,
            num_layers=3,
            tile_size=16
        )

    def test_attention_state_preparation(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test attention state preparation and properties."""
        # Create input tensor
        x = torch.randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Test state properties
        assert isinstance(state, AttentionState), "Should return AttentionState"
        assert state.quantum_state is not None, "Should have quantum state"
        assert state.geometric_state is not None, "Should have geometric state"

        # Test state dimensions
        assert state.quantum_state.shape[0] == batch_size, "Batch dimension preserved"
        assert state.quantum_state.shape[1] == num_heads, "Head dimension correct"

        # Test state normalization
        norms = state.quantum_state.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5
        ), "Quantum states should be normalized"

        # Test mask application
        masked_state = attention_layer.apply_mask(state, mask)
        assert torch.all(
            masked_state.attention_scores[~mask] == float("-inf")
        ), "Mask should be properly applied"

    def test_attention_pattern_computation(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test attention pattern computation."""
        # Create query, key, value tensors
        query = torch.randn(batch_size, num_heads, seq_length, hidden_dim // num_heads)
        key = torch.randn(batch_size, num_heads, seq_length, hidden_dim // num_heads)
        value = torch.randn(batch_size, num_heads, seq_length, hidden_dim // num_heads)

        # Compute attention patterns
        patterns, metrics = attention_layer.compute_attention_patterns(
            query, key, value, scale=1.0
        )

        # Test pattern properties
        assert patterns.shape == (batch_size, num_heads, seq_length, seq_length), \
            "Pattern shape correct"

        # Test row-wise normalization
        row_sums = patterns.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-5
        ), "Patterns should be row-normalized"

        # Test metric properties
        assert isinstance(metrics, AttentionMetrics), "Should return metrics"
        assert metrics.entropy is not None, "Should compute entropy"
        assert metrics.complexity is not None, "Should compute complexity"

        # Test causality if applicable
        if attention_layer.is_causal:
            assert torch.all(
                torch.triu(patterns, diagonal=1) == 0
            ), "Causal attention should be lower triangular"

    def test_geometric_attention_flow(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test geometric attention flow computation."""
        # Create patterns and metric tensor
        patterns = torch.softmax(
            torch.randn(batch_size, num_heads, seq_length, seq_length),
            dim=-1
        )
        metric = torch.eye(hidden_dim).expand(batch_size, num_heads, -1, -1)

        # Compute geometric flow
        flow, metrics = attention_layer.geometric_attention_flow(patterns, metric)

        # Test flow properties
        assert flow.shape == patterns.shape, "Flow should match pattern shape"

        # Test conservation
        flow_divergence = attention_layer.compute_flow_divergence(flow)
        assert torch.allclose(
            flow_divergence, torch.zeros_like(flow_divergence), rtol=1e-4
        ), "Flow should be conserved"

        # Test metric compatibility
        assert attention_layer.check_metric_compatibility(flow, metric), \
            "Flow should be compatible with metric"

        # Test flow metrics
        assert isinstance(metrics, FlowMetrics), "Should return flow metrics"
        assert metrics.energy is not None, "Should compute flow energy"
        assert metrics.curvature is not None, "Should compute flow curvature"

    def test_quantum_classical_interface(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum-classical information interface."""
        # Create classical input
        classical_input = torch.randn(batch_size, seq_length, hidden_dim)

        # Convert to quantum state
        quantum_state = attention_layer.classical_to_quantum(classical_input)

        # Test quantum state properties
        assert attention_layer.is_valid_quantum_state(quantum_state), \
            "Should be valid quantum state"

        # Convert back to classical
        classical_output = attention_layer.quantum_to_classical(quantum_state)

        # Test information preservation
        assert torch.allclose(
            classical_input, classical_output, rtol=1e-4
        ), "Should preserve information through quantum-classical interface"

        # Test gradients
        quantum_state.requires_grad = True
        loss = classical_output.sum()
        loss.backward()
        assert quantum_state.grad is not None, "Should allow gradient flow"

    def test_multi_head_integration(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test multi-head attention integration."""
        # Create per-head inputs
        head_states = [
            torch.randn(batch_size, seq_length, hidden_dim // num_heads)
            for _ in range(num_heads)
        ]

        # Integrate heads
        integrated = attention_layer.integrate_heads(head_states)

        # Test integrated shape
        assert integrated.shape == (batch_size, seq_length, hidden_dim), \
            "Integration should preserve dimensions"

        # Test head separation
        separated = attention_layer.separate_heads(integrated)
        assert len(separated) == num_heads, "Should recover all heads"

        # Test head independence
        head_correlations = attention_layer.compute_head_correlations(separated)
        assert torch.all(
            head_correlations < 0.5
        ), "Heads should be approximately independent"

        # Test attention output
        output = attention_layer(torch.randn(batch_size, seq_length, hidden_dim))
        assert output.shape == (batch_size, seq_length, hidden_dim), \
            "Should maintain input shape"

        # Test gradient flow through heads
        output.sum().backward()
        for param in attention_layer.parameters():
            assert param.grad is not None, "Should compute gradients for all heads"

    def test_geometric_phases(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum geometric phases in attention."""
        # Create cyclic attention path
        def create_cyclic_path(steps: int = 100):
            """Create a cyclic path in attention parameter space."""
            t = torch.linspace(0, 2*np.pi, steps)
            return [
                attention_layer.create_attention_parameters(
                    torch.cos(ti), torch.sin(ti)
                )
                for ti in t
            ]

        # Compute Berry phase
        path = create_cyclic_path()
        berry_phase = attention_layer.compute_berry_phase(path)

        # Test phase quantization
        assert torch.abs(berry_phase - torch.round(berry_phase)) < 1e-4, \
            "Berry phase should be quantized"

        # Test parallel transport
        state = torch.randn(batch_size, seq_length, hidden_dim)
        transported = attention_layer.parallel_transport(state, path)

        # Test holonomy
        holonomy = attention_layer.compute_holonomy(transported)
        assert torch.allclose(
            holonomy @ holonomy.conj().transpose(-1, -2),
            torch.eye(hidden_dim),
            rtol=1e-4
        ), "Holonomy should be unitary"

        # Test Wilczek-Zee connection
        connection = attention_layer.wilczek_zee_connection(path)
        assert connection.shape == (len(path), hidden_dim, hidden_dim), \
            "Connection should have correct shape"

    def test_manifold_curvature(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test attention manifold curvature properties."""
        # Create local patch of attention states
        states = torch.randn(batch_size, seq_length, hidden_dim)

        # Compute metric tensor
        metric = attention_layer.compute_metric_tensor(states)
        assert metric.shape == (batch_size, hidden_dim, hidden_dim), \
            "Metric tensor should have correct shape"

        # Test Riemann curvature
        riemann = attention_layer.compute_riemann_tensor(states)
        # Test symmetries
        assert torch.allclose(
            riemann,
            -riemann.transpose(-1, -2),
            rtol=1e-4
        ), "Riemann tensor should be antisymmetric in last indices"

        # Test sectional curvature
        planes = torch.randn(batch_size, hidden_dim, 2)  # Random 2-planes
        sectional = attention_layer.compute_sectional_curvature(states, planes)
        assert sectional.shape == (batch_size,), \
            "Sectional curvature should be scalar"

        # Test Ricci curvature
        ricci = attention_layer.compute_ricci_tensor(states)
        assert ricci.shape == (batch_size, hidden_dim, hidden_dim), \
            "Ricci tensor should have correct shape"

        # Test scalar curvature
        scalar = attention_layer.compute_scalar_curvature(states)
        assert scalar.shape == (batch_size,), \
            "Scalar curvature should be scalar"

    def test_attention_entanglement(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test entanglement properties in attention states."""
        # Create attention state
        state = attention_layer.prepare_attention_state(
            torch.randn(batch_size, seq_length, hidden_dim),
            torch.ones(batch_size, seq_length).bool()
        )

        # Test bipartite entanglement
        def test_bipartite(split_idx: int):
            """Test entanglement across bipartition."""
            entropy = attention_layer.compute_entanglement_entropy(
                state.quantum_state, split_idx
            )
            assert entropy >= 0, "Entropy should be non-negative"
            assert entropy <= np.log(2) * min(
                split_idx, seq_length - split_idx
            ), "Entropy should satisfy area law"
            return entropy

        # Test various bipartitions
        entropies = [test_bipartite(i) for i in range(1, seq_length)]
        assert len(entropies) == seq_length - 1, "Should test all bipartitions"

        # Test mutual information
        mi = attention_layer.compute_mutual_information(
            state.quantum_state, [0, seq_length//2, -1]
        )
        assert mi >= 0, "Mutual information should be non-negative"

        # Test entanglement spectrum
        spectrum = attention_layer.compute_entanglement_spectrum(
            state.quantum_state, seq_length//2
        )
        assert torch.all(spectrum >= 0), "Spectrum should be non-negative"
        assert torch.allclose(
            spectrum.sum(), torch.tensor(1.0), rtol=1e-5
        ), "Spectrum should be normalized"

    def test_error_correction(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum error correction in attention."""
        # Create code state
        code_state = attention_layer.prepare_code_state(
            torch.randn(batch_size, seq_length, hidden_dim)
        )

        # Test encoding map
        logical_ops = attention_layer.get_logical_operators()
        assert all(
            attention_layer.check_operator_preservation(op, code_state)
            for op in logical_ops
        ), "Encoding should preserve logical operators"

        # Test error detection
        def test_error_detection(error):
            """Test detection of specific error."""
            corrupted = attention_layer.apply_error(code_state, error)
            syndrome = attention_layer.measure_syndrome(corrupted)
            detected = attention_layer.detect_error(syndrome)
            return detected == error

        # Test common errors
        assert test_error_detection("bit_flip"), "Should detect bit flips"
        assert test_error_detection("phase_flip"), "Should detect phase flips"

        # Test recovery
        def test_recovery(error):
            """Test recovery from specific error."""
            corrupted = attention_layer.apply_error(code_state, error)
            recovered = attention_layer.recover_state(corrupted)
            fidelity = attention_layer.compute_state_fidelity(
                code_state, recovered
            )
            return fidelity > 0.99

        assert test_recovery("bit_flip"), "Should recover from bit flips"
        assert test_recovery("phase_flip"), "Should recover from phase flips"

    def test_topological_features(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test topological features in attention."""
        # Create attention complex
        complex = attention_layer.build_attention_complex(
            torch.randn(batch_size, seq_length, hidden_dim)
        )

        # Test Betti numbers
        betti = attention_layer.compute_betti_numbers(complex)
        assert len(betti) > 0, "Should compute Betti numbers"
        assert betti[0] >= 1, "Should be connected"

        # Test persistent homology
        diagrams = attention_layer.compute_persistence_diagrams(complex)
        assert len(diagrams) > 0, "Should compute persistence diagrams"

        # Test topological features
        features = attention_layer.extract_topological_features(complex)
        assert features.shape == (batch_size, attention_layer.num_topological_features), \
            "Should extract topological features"

        # Test attention filtration
        def test_filtration_consistency(threshold) -> None:
            """Test consistency of attention filtration."""
            filtered = attention_layer.filter_attention_complex(complex, threshold)
            assert attention_layer.check_filtration_consistency(
                complex, filtered
            ), "Filtration should be consistent"

        test_filtration_consistency(0.5)

        # Test topological invariants
        invariants = attention_layer.compute_topological_invariants(complex)
        assert len(invariants) > 0, "Should compute topological invariants"

        # Test boundary map
        boundary = attention_layer.compute_boundary_map(complex)
        assert torch.matrix_rank(boundary) < boundary.shape[1], \
            "Should have non-trivial homology"
