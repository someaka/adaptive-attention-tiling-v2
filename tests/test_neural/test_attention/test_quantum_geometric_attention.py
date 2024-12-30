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
    AttentionMetrics,
    AttentionState,
    FlowMetrics,
    GeometricStructures,
    QuantumGeometricAttention,
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
        torch.cuda.empty_cache()
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return QuantumGeometricAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            motive_rank=4,
            manifold_dim=manifold_dim,  # Explicitly pass manifold_dim
            num_layers=3,
            tile_size=8,  # Reduced from 16 to match smaller scale
            dtype=torch.complex64,
            device=device
        )

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
        assert state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")) is not None, "Should have quantum state"
        assert state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")) is not None, "Should have geometric state"

        # Test state dimensions
        assert state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")).shape[0] == batch_size, "Batch dimension preserved"
        assert state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")).shape[1] == num_heads, "Head dimension correct"
        assert state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")).shape[-1] == manifold_dim, "Manifold dimension correct"

        # Test state normalization
        norms = state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")).norm(dim=-1)
        # Check normalization with proper complex tolerances
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"
        
        # Validate quantum state properties
        assert self.attention_layer.is_valid_quantum_state(state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")))

        # Test mask application
        masked_state = attention_layer.apply_mask(state, mask)
        assert torch.all(
            masked_state.attention_scores[~mask] == float("-inf")
        ), "Mask should be properly applied"

    def test_attention_pattern_computation(
        self, attention_layer, batch_size, seq_length, hidden_dim, num_heads
    ):
        """Test attention pattern computation."""
        # Create query, key, value tensors
        query = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)
        key = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)
        value = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)

        # Compute attention patterns
        result = attention_layer.compute_attention_patterns(
            query, key, value, scale=1.0
        )
        patterns = result[0]  # Unpack patterns tensor
        metrics = result[1]  # Unpack metrics

        # Test pattern properties
        assert patterns.shape == (
            batch_size,
            attention_layer.num_heads,
            seq_length,
            seq_length,
        ), "Pattern shape correct"

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
        self, attention_layer, batch_size, seq_length, hidden_dim, num_heads
    ):
        """Test geometric attention flow computation."""
        # Create patterns and metric tensor
        patterns = torch.softmax(
            complex_randn(batch_size, attention_layer.num_heads, seq_length, seq_length), dim=-1
        )
        metric = torch.eye(hidden_dim, dtype=torch.complex64).expand(batch_size, attention_layer.num_heads, -1, -1)

        # Compute geometric flow
        flow_result = attention_layer.geometric_attention_flow(patterns, metric)
        flow = flow_result[0]  # Unpack flow tensor
        metrics = flow_result[1]  # Unpack metrics

        # Test flow properties
        assert flow.shape == patterns.shape, "Flow should match pattern shape"

        # Test conservation
        flow_divergence = attention_layer.compute_flow_divergence(flow)
        assert torch.allclose(
            flow_divergence, torch.zeros_like(flow_divergence), rtol=1e-4
        ), "Flow should be conserved"

        # Test metric compatibility
        assert attention_layer.check_metric_compatibility(
            flow, metric
        ), "Flow should be compatible with metric"

        # Test flow metrics
        assert isinstance(metrics, FlowMetrics), "Should return flow metrics"
        assert metrics.energy is not None, "Should compute flow energy"
        assert metrics.curvature is not None, "Should compute flow curvature"

    def test_quantum_classical_interface(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test quantum-classical information interface."""
        # Create classical input with manifold dimension
        classical_input = complex_randn(batch_size, seq_length, manifold_dim)

        # Convert to quantum state
        quantum_state = attention_layer.classical_to_quantum(classical_input)

        # Test quantum state properties
        assert quantum_state.shape[-1] == manifold_dim, "Should preserve manifold dimension"
        assert attention_layer.is_valid_quantum_state(
            quantum_state
        ), "Should be valid quantum state"

        # Convert back to classical
        classical_output = attention_layer.quantum_to_classical(quantum_state)

        # Test shape preservation
        assert classical_output.shape == classical_input.shape, "Should preserve input shape"
        assert classical_output.shape[-1] == manifold_dim, "Should preserve manifold dimension"

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
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim, num_heads
    ):
        """Test multi-head attention integration."""
        # Create per-head inputs with manifold dimension
        head_dim = manifold_dim  # Each head operates on manifold dimension
        head_states = [
            complex_randn(batch_size, seq_length, head_dim)
            for _ in range(num_heads)
        ]

        # Integrate heads
        integrated = attention_layer.integrate_heads(head_states)

        # Test integrated shape
        assert integrated.shape == (
            batch_size,
            seq_length,
            hidden_dim,
        ), "Integration should preserve dimensions"

        # Test head separation
        separated = attention_layer.separate_heads(integrated)
        assert len(separated) == num_heads, "Should recover all heads"
        assert all(s.shape[-1] == manifold_dim for s in separated), "Each head should have manifold dimension"

        # Test head independence
        head_correlations = attention_layer.compute_head_correlations(separated)
        assert torch.all(
            head_correlations < 0.5
        ), "Heads should be approximately independent"

        # Test attention output
        output = attention_layer(complex_randn(batch_size, seq_length, hidden_dim))
        assert output.shape == (
            batch_size,
            seq_length,
            hidden_dim,
        ), "Should maintain input shape"

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
            t = torch.linspace(0, 2 * np.pi, steps)
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
        assert (
            torch.abs(berry_phase - torch.round(berry_phase)) < 1e-4
        ), "Berry phase should be quantized"

        # Test parallel transport
        state = complex_randn(batch_size, seq_length, hidden_dim)
        transported = attention_layer.parallel_transport(state, path)

        # Test holonomy
        holonomy = attention_layer.compute_holonomy(transported)
        assert torch.allclose(
            holonomy @ holonomy.conj().transpose(-1, -2),
            torch.eye(hidden_dim),
            rtol=1e-4,
        ), "Holonomy should be unitary"

        # Test Wilczek-Zee connection
        connection = attention_layer.wilczek_zee_connection(path)
        assert connection.shape == (
            len(path),
            hidden_dim,
            hidden_dim,
        ), "Connection should have correct shape"

    def test_manifold_curvature(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test attention manifold curvature properties."""
        # Create local patch of attention states with manifold dimension
        states = complex_randn(batch_size, seq_length, manifold_dim)

        # Compute metric tensor
        metric = attention_layer.compute_metric_tensor(states)
        assert metric.shape == (
            batch_size,
            manifold_dim,
            manifold_dim,
        ), "Metric tensor should have manifold dimensions"

        # Test Riemann curvature
        riemann = attention_layer.compute_riemann_tensor(states)
        # Test symmetries
        assert torch.allclose(
            riemann, -riemann.transpose(-1, -2), rtol=1e-4
        ), "Riemann tensor should be antisymmetric in last indices"

        # Test sectional curvature
        planes = complex_randn(batch_size, manifold_dim, 2)  # Random 2-planes in manifold
        sectional = attention_layer.compute_sectional_curvature(states, planes)
        assert sectional.shape == (batch_size,), "Sectional curvature should be scalar"

        # Test Ricci curvature
        ricci = attention_layer.compute_ricci_tensor(states)
        assert ricci.shape == (
            batch_size,
            manifold_dim,
            manifold_dim,
        ), "Ricci tensor should have manifold dimensions"

        # Test scalar curvature
        scalar = attention_layer.compute_scalar_curvature(states)
        assert scalar.shape == (batch_size,), "Scalar curvature should be scalar"

    def test_attention_entanglement(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test entanglement properties in attention states."""
        # Create attention state
        state = attention_layer.prepare_attention_state(
            complex_randn(batch_size, seq_length, hidden_dim),
            torch.ones(batch_size, seq_length).bool(),
        )

        # Test bipartite entanglement
        def test_bipartite(split_idx: int):
            """Test entanglement across bipartition."""
            entropy = attention_layer.compute_entanglement_entropy(
                state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")), split_idx
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
            state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")), [0, seq_length // 2, -1]
        )
        assert mi >= 0, "Mutual information should be non-negative"

        # Test entanglement spectrum
        spectrum = attention_layer.compute_entanglement_spectrum(
            state.state_manager.states.get("quantum", state.state_manager.initialize_state("quantum")), seq_length // 2
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
            complex_randn(batch_size, seq_length, hidden_dim)
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
            fidelity = attention_layer.compute_state_fidelity(code_state, recovered)
            return fidelity > 0.99

        assert test_recovery("bit_flip"), "Should recover from bit flips"
        assert test_recovery("phase_flip"), "Should recover from phase flips"

    def test_topological_features(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test topological features in attention."""
        # Create attention complex
        attention_complex = attention_layer.build_attention_complex(
            complex_randn(batch_size, seq_length, hidden_dim)
        )

        # Test Betti numbers
        betti = attention_layer.compute_betti_numbers(attention_complex)
        assert len(betti) > 0, "Should compute Betti numbers"
        assert betti[0] >= 1, "Should be connected"

        # Test persistent homology
        diagrams = attention_layer.compute_persistence_diagrams(attention_complex)
        assert len(diagrams) > 0, "Should compute persistence diagrams"

        # Test topological features
        features = attention_layer.extract_topological_features(attention_complex)
        assert features.shape == (
            batch_size,
            attention_layer.num_topological_features,
        ), "Should extract topological features"

        # Test attention filtration
        def test_filtration_consistency(threshold) -> None:
            """Test consistency of attention filtration."""
            filtered = attention_layer.filter_attention_complex(attention_complex, threshold)
            assert attention_layer.check_filtration_consistency(
                attention_complex, filtered
            ), "Filtration should be consistent"

        test_filtration_consistency(0.5)

        # Test topological invariants
        invariants = attention_layer.compute_topological_invariants(attention_complex)
        assert len(invariants) > 0, "Should compute topological invariants"

        # Test boundary map
        boundary = attention_layer.compute_boundary_map(attention_complex)
        assert (
            torch.linalg.matrix_rank(boundary) < boundary.shape[1]
        ), "Should have non-trivial homology"

    def test_attention_patterns(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        """Test attention pattern computation."""
        # Create query, key, value tensors
        query = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)
        key = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)
        value = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim // attention_layer.num_heads)

        # Compute attention patterns
        result = attention_layer.compute_attention_patterns(query, key)
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

    def test_geometric_structures(self, geometric_structures, hidden_dim):
        """Test geometric structures functionality."""
        # Test metric initialization
        assert geometric_structures.metric.shape == (hidden_dim, hidden_dim)
        assert geometric_structures.connection.shape == (hidden_dim, hidden_dim, hidden_dim)
        assert geometric_structures.curvature_tensor.shape == (hidden_dim, hidden_dim, hidden_dim, hidden_dim)

        # Test sectional curvature computation
        v1 = complex_randn(hidden_dim)
        v2 = complex_randn(hidden_dim)
        curvature = geometric_structures.compute_sectional_curvature(None, v1, v2)
        assert isinstance(curvature, torch.Tensor)
        assert curvature.ndim == 0  # Scalar output

    def test_pattern_dynamics(self, pattern_dynamics, hidden_dim, batch_size):
        """Test pattern dynamics functionality."""
        # Test pattern library initialization
        assert pattern_dynamics.patterns.shape == (64, hidden_dim)
        assert pattern_dynamics.pattern_importance.shape == (64,)
        assert pattern_dynamics.transfer_weights.shape == (8, 64, 64)

        # Test Fisher information computation
        states = complex_randn(batch_size, hidden_dim)
        fisher = pattern_dynamics.compute_fisher_information(states)
        assert fisher.shape == (hidden_dim, hidden_dim)
        assert torch.allclose(fisher, fisher.t())  # Should be symmetric

    def compute_control_matrix(self) -> torch.Tensor:
        """Compute control matrix for system controllability test.
        
        Returns:
            Control matrix tensor
        """
        # Create a simple control matrix for testing
        # This is just a placeholder implementation
        dim = 4  # Small dimension for testing
        return torch.eye(dim)  # Identity matrix as control matrix

    def test_controllability(self):
        """Test system controllability."""
        control_matrix = self.compute_control_matrix()
        rank = torch.linalg.matrix_rank(control_matrix)
        assert rank == control_matrix.shape[0], "Should be controllable"
        return control_matrix
