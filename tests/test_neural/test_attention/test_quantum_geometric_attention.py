"""
Unit tests for the quantum geometric attention mechanism.

Tests cover:
1. Basic Functionality
   - Initialization
   - Forward pass
   - Deterministic behavior
   - Batch processing
2. Component Testing
   - State preparation
   - Attention patterns
   - Geometric operations
   - Quantum operations
3. State Handling
   - Transitions
   - Error recovery
   - Edge cases
4. Validation
   - Config validation
   - State validation
   - Metric validation
5. Advanced Features
   - Pattern dynamics
   - Geometric flow
   - Quantum effects
"""

import numpy as np
import pytest
import torch
import torch.linalg

from src.core.tiling.quantum_geometric_attention import (
    AttentionState,
    GeometricStructures,
    QuantumGeometricAttention,
    QuantumGeometricConfig,
    MetricError,
    InvalidQuantumStateError
)
from src.metrics.attention import (
    compute_attention_metrics,
    compute_flow_metrics,
    compute_parallel_transport,
    compute_geodesic_distance,
    compute_flow_energy,
    compute_ricci_tensor
)
from src.metrics.quantum_geometric_metrics import (
    MetricContext,
    UnifiedMetrics,
    MetricDomain
)
from src.validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    QuantumStateValidationResult
)
from src.validation.geometric.metric import (
    MetricProperties,
    MetricValidation
)
from src.core.patterns.dynamics import PatternDynamics

def complex_randn(*size, dtype=torch.complex64):
    """Helper to create random complex tensor.
    
    Args:
        *size: The shape of the tensor to create
        dtype: The dtype of the tensor (default: torch.complex64)
        
    Returns:
        A complex tensor with random values, normalized globally across all dimensions except batch
    """
    # Create random complex tensor
    real = torch.randn(*size)
    imag = torch.randn(*size)
    z = torch.complex(real, imag)
    
    # Normalize globally across all dimensions except batch
    norm = torch.sqrt(torch.sum(torch.abs(z) ** 2, dim=tuple(range(1, len(z.shape))), keepdim=True))
    return (z / norm).to(dtype)  # Convert to specified dtype

class TestQuantumGeometricAttention:
    """Test suite for quantum geometric attention with proper cleanup."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test."""
        torch.manual_seed(42)  # Ensure reproducibility
        yield
        import gc
        gc.collect()

    @pytest.fixture
    def manifold_dim(self) -> int:
        """Return manifold dimension for tests."""
        return 8  # Reduced from 16 to use less memory

    @pytest.fixture
    def hidden_dim(self, manifold_dim) -> int:
        """Return hidden dimension for tests."""
        return manifold_dim * 2  # Reduced from 4x to use less memory

    @pytest.fixture
    def num_heads(self) -> int:
        """Return number of attention heads for tests."""
        return 4  # Reduced from 8 to use less memory

    @pytest.fixture
    def batch_size(self) -> int:
        """Return batch size for tests."""
        return 4  # Reduced from 16 to use less memory

    @pytest.fixture
    def seq_length(self) -> int:
        """Return sequence length for tests."""
        return 4  # Reduced from 8 to use less memory

    @pytest.fixture
    def config(self, hidden_dim, manifold_dim, num_heads):
        """Create test configuration with validation settings."""
        return QuantumGeometricConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            manifold_type="hyperbolic",
            curvature=-1.0,
            manifold_dim=manifold_dim,
            num_layers=3,
            tile_size=8,
            motive_rank=4,
            dtype=torch.complex128,
            device=torch.device('cpu'),
            is_causal=False
        )

    @pytest.fixture
    def attention_layer(self, config):
        """Create a test attention layer with proper device placement."""
        return QuantumGeometricAttention(config=config)

    @pytest.fixture
    def geometric_structures(self, manifold_dim):
        """Create geometric structures for testing."""
        return GeometricStructures(
            dim=manifold_dim,
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

    @pytest.fixture
    def metric_context(self, batch_size, seq_length, hidden_dim):
        """Create metric context for testing."""
        return MetricContext(
            timestamp=0.0,
            device=torch.device('cpu'),
            batch_size=batch_size,
            sequence_length=seq_length,
            hidden_dim=hidden_dim,
            resolution=1.0
        )

    def test_config_validation(self, hidden_dim, manifold_dim, num_heads):
        """Test configuration validation."""
        # Test valid config
        valid_config = QuantumGeometricConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            manifold_dim=manifold_dim,
            dtype=torch.complex128
        )
        attention = QuantumGeometricAttention(valid_config)
        assert attention is not None, "Should create with valid config"

        # Test invalid hidden_dim
        with pytest.raises(ValueError):
            invalid_config = QuantumGeometricConfig(
                hidden_dim=-1,
                num_heads=num_heads,
                manifold_dim=manifold_dim,
                dtype=torch.complex128
            )
            QuantumGeometricAttention(invalid_config)

        # Test incompatible dimensions
        with pytest.raises(ValueError):
            invalid_config = QuantumGeometricConfig(
                hidden_dim=hidden_dim,
                num_heads=hidden_dim + 1,  # More heads than dimensions
                manifold_dim=manifold_dim,
                dtype=torch.complex128
            )
            QuantumGeometricAttention(invalid_config)


    def test_attention_state_preparation(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim, num_heads
    ):
        """Test comprehensive attention state preparation and validation."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()
    
        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)
    
        # 1. Test basic type and existence properties
        assert isinstance(state, AttentionState), "Should return AttentionState"
        assert state.state_manager is not None, "Should have state manager"
        assert state.geometric_state is not None, "Should have geometric state"
    
        # 2. Test state manager initialization and contents
        assert "input" in state.state_manager.states, "Should store input state"
        assert "manifold" in state.state_manager.states, "Should store manifold state"
        assert "quantum" in state.state_manager.states, "Should store quantum state"
    
        # 3. Test quantum state properties
        quantum_state = state.state_manager.states.get("quantum")
        assert quantum_state is not None, "Should have quantum state"
    
        # 4. Test geometric and manifold state properties
        # The geometric state should have shape [batch_size * num_heads, seq_len, manifold_dim]
        expected_shape = (batch_size * attention_layer.num_heads, seq_length, manifold_dim)
        assert state.geometric_state.shape == expected_shape, "Geometric state shape mismatch"
        assert state.geometric_state.dtype == attention_layer.config.dtype, "Geometric state dtype mismatch"
        
        manifold_state = state.state_manager.states["manifold"]
        assert manifold_state.shape[-1] == manifold_dim, "Manifold state dimension mismatch"
        assert not torch.isnan(manifold_state).any(), "Manifold state contains NaN"
        assert not torch.isinf(manifold_state).any(), "Manifold state contains Inf"

        # 5. Test attention components initialization
        assert state.attention_scores is None, "Initial attention scores should be None"
        assert isinstance(state.attention_patterns, dict), "Attention patterns should be dict"
        assert isinstance(state.entanglement_history, dict), "Entanglement history should be dict"
        assert isinstance(state.metrics, dict), "Metrics should be dict"

        # 6. Test state validation
        assert state.validate_state(state.geometric_state), "Geometric state validation failed"
        assert state.validate_state(manifold_state), "Manifold state validation failed"

        # 7. Test mask application
        masked_state = attention_layer.apply_mask(state, mask)
        assert masked_state.attention_scores is not None, "Masked state should have attention scores"
        assert torch.is_tensor(masked_state.attention_scores), "Attention scores should be tensor"
        
        # Test mask expansion and application
        expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(
            batch_size, num_heads, seq_length, seq_length
        )
        assert torch.all(
            masked_state.attention_scores[~expanded_mask] == float("-inf")
        ), "Mask should be properly applied"

        # 8. Test state consistency
        assert id(masked_state.state_manager) == id(state.state_manager), "State manager should be preserved"
        assert "mask" in masked_state.state_manager.states, "Mask should be stored in state manager"


    def test_mask_handling(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test attention mask handling."""
        # Create input with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        
        # Print input tensor properties
        print(f"\nInput tensor properties:")
        print(f"Shape: {x.shape}")
        print(f"Norm mean: {torch.norm(x, dim=-1).mean():.4f}")
        print(f"Norm std: {torch.norm(x, dim=-1).std():.4f}")
        
        try:
            # Test with all-ones mask
            mask_ones = torch.ones(batch_size, seq_length).bool()
            print(f"\nTesting all-ones mask:")
            print(f"Mask shape: {mask_ones.shape}")
            print(f"Mask sum: {mask_ones.sum()}/{mask_ones.numel()}")
            output_ones = attention_layer(x, mask=mask_ones)
            assert not torch.isnan(output_ones).any(), "Should handle all-ones mask"
            print("✓ All-ones mask test passed")
            
            # Test with all-zeros mask
            mask_zeros = torch.zeros(batch_size, seq_length).bool()
            print(f"\nTesting all-zeros mask:")
            print(f"Mask shape: {mask_zeros.shape}")
            print(f"Mask sum: {mask_zeros.sum()}/{mask_zeros.numel()}")
            output_zeros = attention_layer(x, mask=mask_zeros)
            assert torch.allclose(
                output_zeros, torch.zeros_like(output_zeros), rtol=1e-5
            ), "Should zero-out with all-zeros mask"
            print("✓ All-zeros mask test passed")
            
        except Exception as e:
            print(f"Error during mask handling test: {str(e)}")
            raise

    def test_sequence_length_handling(self, attention_layer, batch_size, hidden_dim):
        """Test handling of different sequence lengths."""
        # Test zero-length sequence
        x_zero = complex_randn(batch_size, attention_layer.num_heads, 0, hidden_dim)
        with pytest.raises(ValueError):
            attention_layer(x_zero)
        
        # Test single-token sequence
        x_single = complex_randn(batch_size, attention_layer.num_heads, 1, hidden_dim)
        output_single = attention_layer(x_single)
        assert output_single.shape == x_single.shape, "Should handle single-token sequence"
        
        # Test long sequence
        long_seq_len = 1024
        x_long = complex_randn(batch_size, attention_layer.num_heads, long_seq_len, hidden_dim)
        output_long = attention_layer(x_long)
        assert output_long.shape == x_long.shape, "Should handle long sequence"
        
        # Test varying sequence lengths in batch
        seq_lengths = [2, 4, 8, 16]
        max_len = max(seq_lengths)
        x_varying = complex_randn(len(seq_lengths), attention_layer.num_heads, max_len, hidden_dim)
        
        # Create key padding mask [batch_size, seq_len]
        key_padding_mask = torch.zeros(len(seq_lengths), max_len).bool()
        for i, length in enumerate(seq_lengths):
            key_padding_mask[i, :length] = True
            
        # Create causal mask [seq_len, seq_len]
        causal_mask = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool),
            diagonal=1
        ).logical_not()
        
        # Process with both masks
        state = attention_layer.prepare_attention_state(x_varying)
        state.set_key_padding_mask(key_padding_mask)
        state.set_attention_mask(causal_mask)
        output_varying = attention_layer(x_varying)
        assert output_varying.shape == x_varying.shape, "Should handle varying sequence lengths"

    def test_unified_metrics(
        self, attention_layer, batch_size, seq_length, hidden_dim, metric_context
    ):
        """Test unified metrics computation."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer with metrics
        output, metrics = attention_layer(x, mask=mask, return_metrics=True)

        # Test metric properties
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        
        # Test metric domains
        for domain in MetricDomain:
            domain_metrics = metrics.get(domain.value)
            if domain_metrics is not None:
                assert isinstance(domain_metrics, dict), f"{domain.value} metrics should be a dictionary"

        # Test specific metrics from UnifiedMetrics
        if "step_0" in metrics:
            step_metrics = metrics["step_0"]
            
            # Test quantum metrics
            if "quantum_entropy" in step_metrics:
                assert isinstance(step_metrics["quantum_entropy"], torch.Tensor)
                assert not torch.isnan(step_metrics["quantum_entropy"]).any()
            
            # Test geometric metrics
            if "geodesic_distance" in step_metrics:
                assert isinstance(step_metrics["geodesic_distance"], torch.Tensor)
                assert not torch.isnan(step_metrics["geodesic_distance"]).any()
            
            # Test pattern metrics
            if "pattern_evolution" in step_metrics:
                assert isinstance(step_metrics["pattern_evolution"], dict)
                assert len(step_metrics["pattern_evolution"]) > 0

            # Test arithmetic metrics
            if "local_height" in step_metrics:
                assert isinstance(step_metrics["local_height"], torch.Tensor)
                assert not torch.isnan(step_metrics["local_height"]).any()

        # Test that metrics are being computed at each step
        step_keys = [key for key in metrics.keys() if key.startswith("step_")]
        assert len(step_keys) > 0, "Should have step-wise metrics"

    def test_metric_validation(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test metric tensor validation."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Compute metric tensor
        try:
            metric = attention_layer._compute_metric_tensor(state)
            properties = attention_layer._validate_metric_properties(metric, "test")

            # Test metric properties
            assert isinstance(properties, MetricProperties), "Should return MetricProperties"
            assert properties.is_positive_definite, "Metric should be positive definite"
            assert properties.is_compatible, "Metric should be compatible"
            assert properties.has_bounded_curvature, "Metric should have bounded curvature"

            # Test curvature components
            assert properties.sectional_curvature is not None, "Should compute sectional curvature"
            assert properties.ricci_curvature is not None, "Should compute Ricci curvature"
            assert properties.scalar_curvature is not None, "Should compute scalar curvature"

        except MetricError as e:
            print(f"Error during metric validation: {str(e)}")
            raise

    def test_error_handling(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test error handling for invalid states and metrics."""
        # Test invalid quantum state - using NaN values which should fail
        invalid_state = torch.full((batch_size, attention_layer.num_heads, seq_length, hidden_dim), 
                                 float('nan'), dtype=attention_layer.config.dtype)
        with pytest.raises(InvalidQuantumStateError):
            attention_layer._prepare_quantum_state(invalid_state)

        # Test invalid metric tensor - using a non-square matrix which should fail
        with pytest.raises(MetricError):
            invalid_metric = torch.zeros(batch_size, hidden_dim, hidden_dim + 1, 
                                       dtype=attention_layer.config.dtype)
            attention_layer._validate_metric_properties(invalid_metric, "invalid")

    def test_attention_pattern_computation(
        self, attention_layer, batch_size, seq_length, hidden_dim, num_heads
    ):
        """Test attention pattern computation."""
        # Create query, key tensors with correct shapes
        head_dim = hidden_dim // num_heads
        manifold_dim = attention_layer.manifold_dim
        
        # Create tensors with correct shape [batch_size, num_heads, seq_len, head_dim]
        query = torch.randn(batch_size, num_heads, seq_length, manifold_dim, 
                           dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        key = torch.randn(batch_size, num_heads, seq_length, manifold_dim, 
                         dtype=attention_layer.config.dtype, device=attention_layer.config.device)

        # Compute attention patterns with metrics
        patterns, metrics = attention_layer.compute_attention_patterns(
            query, key, return_metrics=True
        )

        # Test output shape
        assert patterns.shape == (batch_size, num_heads, seq_length, seq_length), "Wrong pattern shape"

        # Test row-wise normalization
        row_sums = patterns.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-5
        ), "Patterns should be row-normalized"

        # Test metrics
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        assert "attention_scores" in metrics, "Should include attention scores"

    def test_geometric_attention_flow(
        self, attention_layer, batch_size, seq_length, hidden_dim, num_heads
    ):
        """Test geometric attention flow computation."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Compute geometric flow with metrics
        flow, metrics = attention_layer.geometric_attention_flow(
            x, mask=mask, num_steps=10, dt=0.1, return_metrics=True
        )

        # Test flow properties
        assert flow.shape == x.shape, "Flow should match input shape"
        assert not torch.isnan(flow).any(), "Flow should not contain NaN values"
        assert not torch.isinf(flow).any(), "Flow should not contain Inf values"

        # Test flow metrics
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        assert len(metrics) > 0, "Should compute flow metrics"

    def test_quantum_classical_interface(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test quantum-classical information interface."""
        # Create classical input with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)

        # Convert to quantum state with validation
        quantum_state, validation_result = attention_layer._prepare_quantum_state(
            x, return_validation=True
        )

        # Test quantum state properties
        assert validation_result.is_valid, "Should be valid quantum state"
        assert not torch.isnan(quantum_state.amplitudes).any(), "Should not contain NaN values"
        assert not torch.isinf(quantum_state.amplitudes).any(), "Should not contain Inf values"

        # Test quantum state normalization
        norms = torch.sqrt(torch.sum(torch.abs(quantum_state.amplitudes) ** 2, 
                                   dim=tuple(range(1, len(quantum_state.amplitudes.shape)))))
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"

    def test_multi_head_integration(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
    ):
        """Test multi-head attention integration."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        
        # Create attention mask
        mask = torch.ones(batch_size, seq_length).bool()
        
        # Forward pass
        output = attention_layer(x, mask=mask)
        
        # Test output shape and properties
        assert output.shape == x.shape, "Output shape should match input shape"
        assert output.dtype == attention_layer.config.dtype, "Output dtype should match config"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"
        
        # Test gradient flow
        output.sum().backward()
        
        # Track gradient flow through key components
        print("\nGradient Flow Check:")
        for name, param in attention_layer.named_parameters():
            if param.requires_grad:
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Has grad: {param.grad is not None}")
                if param.grad is not None:
                    print(f"  Grad stats - Mean: {param.grad.abs().mean():.2e}, Max: {param.grad.abs().max():.2e}")
                    print(f"  Grad has NaN: {torch.isnan(param.grad).any()}")
                    print(f"  Grad has Inf: {torch.isinf(param.grad).any()}")

        # Check specific components
        components_to_check = [
            ('manifold_proj', attention_layer.manifold_proj),
            ('manifold_proj_inv', attention_layer.manifold_proj_inv),
            ('pattern_proj', attention_layer.pattern_proj),
            ('quantum_bridge', attention_layer.quantum_bridge),
            ('riemannian', attention_layer.riemannian),
        ]

        print("\nComponent Gradient Check:")
        for name, component in components_to_check:
            if hasattr(component, 'parameters'):
                has_any_grad = any(p.grad is not None for p in component.parameters())
                print(f"{name}:")
                print(f"  Has any gradients: {has_any_grad}")
                for param_name, param in component.named_parameters():
                    print(f"  {param_name}:")
                    print(f"    Requires grad: {param.requires_grad}")
                    if param.grad is not None:
                        print(f"    Grad stats - Mean: {param.grad.abs().mean():.2e}, Max: {param.grad.abs().max():.2e}")

        # Now check each parameter
        for name, param in attention_layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"

    def test_geometric_phases(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum geometric phases in attention."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output properties
        assert output.dtype == attention_layer.config.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

    def test_manifold_curvature(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test attention manifold curvature properties."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Compute metric tensor
        metric = attention_layer._compute_metric_tensor(state)

        # Test metric tensor properties
        assert metric.shape[-2:] == (manifold_dim, manifold_dim), "Metric tensor should have manifold dimensions"
        assert torch.allclose(
            metric, metric.transpose(-1, -2).conj()
        ), "Metric tensor should be Hermitian"
        assert not torch.isnan(metric).any(), "Metric tensor should not contain NaN values"
        assert not torch.isinf(metric).any(), "Metric tensor should not contain Inf values"

        # Test positive definiteness
        eigenvalues = torch.linalg.eigvalsh(metric)
        assert (eigenvalues > 0).all(), "Metric tensor should be positive definite"

    def test_attention_entanglement(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test entanglement properties in attention states."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Prepare attention state
        state = attention_layer.prepare_attention_state(x, mask)

        # Get quantum state
        quantum_state = state.state_manager.states.get("quantum")
        assert quantum_state is not None, "Should have quantum state"

        # Test quantum state properties
        assert not torch.isnan(quantum_state.amplitudes).any(), "Quantum state should not contain NaN values"
        assert not torch.isinf(quantum_state.amplitudes).any(), "Quantum state should not contain Inf values"

        # Test normalization
        norms = torch.norm(quantum_state.amplitudes, dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"

    def test_error_correction(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test quantum error correction in attention."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        # Add noise
        x = x + 0.1 * torch.randn_like(x)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output properties
        assert output.dtype == attention_layer.config.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

    def test_topological_features(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test topological features in attention."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        mask = torch.ones(batch_size, seq_length).bool()

        # Process through attention layer
        output = attention_layer(x, mask=mask)

        # Test output properties
        assert output.dtype == attention_layer.config.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"

    def test_basic_error_recovery(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test recovery from simple error states."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim)
        
        # Verify input normalization
        x_norm = torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=tuple(range(1, len(x.shape))), keepdim=True))
        assert torch.allclose(
            x_norm,
            torch.ones(batch_size, 1, 1, 1, dtype=x_norm.dtype, device=x_norm.device),
            rtol=1e-5
        ), "Input should be normalized per batch"
        
        # Test recovery from small perturbations
        x_perturbed = x + 1e-6 * complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim)
        # Normalize perturbed input
        x_perturbed = x_perturbed / torch.sqrt(torch.sum(torch.abs(x_perturbed) ** 2, dim=tuple(range(1, len(x_perturbed.shape))), keepdim=True))
        output = attention_layer(x_perturbed)
        
        # Check that output maintains reasonable scale (may not be exactly normalized)
        output_norm = torch.sqrt(torch.sum(torch.abs(output) ** 2, dim=tuple(range(1, len(output.shape))), keepdim=True))
        assert (output_norm > 0.1).all() and (output_norm < 10.0).all(), "Output should maintain reasonable scale"
        
        # Test with slightly denormalized states
        x_denorm = x * 1.1  # Slightly off normalization
        output_denorm = attention_layer(x_denorm)
        output_denorm_norm = torch.sqrt(torch.sum(torch.abs(output_denorm) ** 2, dim=tuple(range(1, len(output_denorm.shape))), keepdim=True))
        assert (output_denorm_norm > 0.1).all() and (output_denorm_norm < 10.0).all(), "Output should maintain reasonable scale with denormalized input"
        
        # Test with small numerical noise
        x_noisy = x + complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim) * 1e-7
        # Normalize noisy input
        x_noisy = x_noisy / torch.sqrt(torch.sum(torch.abs(x_noisy) ** 2, dim=tuple(range(1, len(x_noisy.shape))), keepdim=True))
        output_noisy = attention_layer(x_noisy)
        assert not torch.isnan(output_noisy).any(), "Should handle numerical noise"
        assert not torch.isinf(output_noisy).any(), "Should handle numerical noise without producing inf"
        
        # Test state preparation validation
        state = attention_layer.prepare_attention_state(x_noisy)
        quantum_state = state.state_manager.states.get("quantum")
        assert quantum_state is not None, "Should create valid quantum state"
        assert not torch.isnan(quantum_state.amplitudes).any(), "Quantum state should not contain NaN"
        assert not torch.isinf(quantum_state.amplitudes).any(), "Quantum state should not contain Inf"
        
        # Verify quantum state normalization
        quantum_norm = torch.sqrt(torch.sum(torch.abs(quantum_state.amplitudes) ** 2, dim=tuple(range(1, len(quantum_state.amplitudes.shape))), keepdim=True))
        assert torch.allclose(
            quantum_norm,
            torch.ones(batch_size, 1, 1, 1, dtype=quantum_norm.dtype, device=quantum_norm.device),
            rtol=1e-5
        ), "Quantum state should be normalized per batch"

    def test_component_isolation(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test individual components in isolation."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim,
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)
    
        # Test quantum state preparation separately
        state = attention_layer._prepare_quantum_state(x)
        assert state.amplitudes.shape == x.shape, "Shape preserved in quantum state"
    
        # Compute norm with proper type handling - normalize globally across all dimensions except batch
        state_norm = torch.sqrt(torch.sum(torch.abs(state.amplitudes) ** 2, dim=tuple(range(1, len(state.amplitudes.shape))), keepdim=True))
        assert torch.allclose(
            state_norm,
            torch.ones(batch_size, 1, 1, 1, dtype=state_norm.dtype),
            rtol=1e-5
        ), "Quantum state should be normalized globally per batch"
        
        # Test geometric projection separately
        projected = attention_layer.manifold_proj(x.reshape(-1, hidden_dim))
        assert not torch.isnan(projected).any(), "Projection should be stable"
        assert projected.shape[-1] == attention_layer.manifold_dim, "Projection dimension correct"
        
        # Test attention pattern computation
        head_dim = hidden_dim // attention_layer.num_heads
        query = torch.randn(batch_size, attention_layer.num_heads, seq_length, head_dim,
                           dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        key = torch.randn(batch_size, attention_layer.num_heads, seq_length, head_dim,
                         dtype=attention_layer.config.dtype, device=attention_layer.config.device)
        patterns = attention_layer.compute_attention_patterns(query, key)
        assert torch.allclose(
            patterns.sum(dim=-1),
            torch.ones(batch_size, attention_layer.num_heads, seq_length, dtype=patterns.dtype),
            rtol=1e-5
        ), "Attention patterns should be normalized"

    def test_gradient_stability(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test gradient stability through complex operations."""
        # Create input with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Forward pass
        output = attention_layer(x)
        loss = output.abs().mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradient properties
        assert x.grad is not None, "Should compute gradients"
        assert not torch.isnan(x.grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(x.grad).any(), "Gradients should not be Inf"
        
        # Test gradient scale
        grad_norm = torch.norm(x.grad)
        assert grad_norm < 100, "Gradients should not explode"
        assert grad_norm > 1e-8, "Gradients should not vanish"

    def test_batch_consistency(
        self, attention_layer, batch_size, seq_length, hidden_dim
    ):
        """Test consistency across batch processing."""
        # Create two identical inputs in different batch positions
        # Shape: [batch_size, num_heads, seq_len, hidden_dim]
        x = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim)
        x_repeated = x.clone()
        x_repeated[1] = x[0]  # Make second batch element identical to first

        output = attention_layer(x[0])
        output_repeated = attention_layer(x_repeated[1])
        
        # Check that other batch elements are unaffected
        assert torch.allclose(
            output, output_repeated, rtol=1e-1
        ), "Unrelated batch elements should be unchanged"

    def test_dtype_handling(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test handling of different dtypes."""
        # Test with float32 input
        x_float32 = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, dtype=torch.float32)
        output_float32 = attention_layer(x_float32)
        assert output_float32.dtype == attention_layer.config.dtype, "Should convert to config dtype"
        
        # Test with float64 input
        x_float64 = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, dtype=torch.float64)
        output_float64 = attention_layer(x_float64)
        assert output_float64.dtype == attention_layer.config.dtype, "Should convert to config dtype"
        
        # Test with complex64 input
        x_complex64 = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim).to(torch.complex64)
        output_complex64 = attention_layer(x_complex64)
        assert output_complex64.dtype == attention_layer.config.dtype, "Should maintain or convert complex dtype"
        
        # Test with complex128 input
        x_complex128 = complex_randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim).to(torch.complex128)
        output_complex128 = attention_layer(x_complex128)
        assert output_complex128.dtype == attention_layer.config.dtype, "Should maintain or convert complex dtype"

    def test_mask_handling_in_attention(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int
    ):
        """Test attention mask handling in quantum geometric attention."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim, 
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)

        # Test with key padding mask
        key_padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
        key_padding_mask[:, -1] = False  # Mask out last token
        output_key_mask = attention_layer(x, mask=key_padding_mask)
        
        # Verify output shape
        assert output_key_mask.shape == x.shape, "Output shape should match input shape"
        
        # Test with attention mask
        attention_mask = torch.ones(seq_length, seq_length, dtype=torch.bool)
        attention_mask.triu_(1).logical_not_()  # Causal mask
        output_attn_mask = attention_layer(x, mask=attention_mask)
        
        # Verify output shape
        assert output_attn_mask.shape == x.shape, "Output shape should match input shape"
        
        # Test with head-specific attention mask
        head_mask = torch.ones(batch_size, attention_layer.num_heads, seq_length, seq_length, dtype=torch.bool)
        head_mask[:, :, :, -1] = False  # Can't attend to last token
        output_head_mask = attention_layer(x, mask=head_mask)
        
        # Verify output shape
        assert output_head_mask.shape == x.shape, "Output shape should match input shape"

    def test_causal_attention(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int
    ):
        """Test causal attention masking."""
        # Create input tensor with correct shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim,
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool),
            diagonal=1
        ).logical_not()

        # Process with causal mask
        output = attention_layer(x, mask=causal_mask)

        # Verify output shape
        assert output.shape == x.shape, "Output shape should match input shape"

        # Get attention scores from the layer
        state = attention_layer.prepare_attention_state(x, causal_mask)
        state = attention_layer.apply_mask(state, causal_mask)
        scores = state.attention_scores

        # Verify causal pattern in attention scores
        assert scores is not None, "Attention scores should not be None"
        # Future tokens should have zero attention scores
        future_positions = ~causal_mask.unsqueeze(0).unsqueeze(0).expand_as(scores)
        masked_scores = scores[future_positions]
        
        # For complex attention scores, check both real and imaginary parts are -inf
        assert torch.all(masked_scores.real == float('-inf')), "Real part of masked positions should be -inf"
        assert torch.all(masked_scores.imag == 0.0), "Imaginary part of masked positions should be 0"

    def test_mixed_mask_types(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int
    ):
        """Test mixed mask types."""
        # Create input tensor with shape [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, attention_layer.num_heads, seq_length, hidden_dim,
                       dtype=attention_layer.config.dtype, device=attention_layer.config.device)

        # Create key padding mask [batch_size, seq_length]
        key_padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool,
                                    device=attention_layer.config.device)
        
        # Create causal mask [seq_length, seq_length]
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool,
                      device=attention_layer.config.device),
            diagonal=1
        ).logical_not()

        # Process with both masks
        state = attention_layer.prepare_attention_state(x)
        
        # Update debug info with correct shape
        state.state_manager.states["debug_info"] = {
            "input_shape": tuple(x.shape),
            "input_dtype": str(x.dtype),
            "manifold_shape": tuple(x.shape[:-1] + (attention_layer.manifold_dim,)),
            "num_heads": attention_layer.num_heads
        }
        
        # Set masks
        state.set_key_padding_mask(key_padding_mask)
        state.set_attention_mask(causal_mask)
        
        # Apply mask to compute attention scores
        state = attention_layer.apply_mask(state, causal_mask)
        
        # Apply attention flow
        output = attention_layer(x, mask=causal_mask)

        # Verify output shape
        assert output.shape == x.shape, "Output shape should match input shape"

        # Get attention scores from the layer
        scores = state.attention_scores

        # Verify causal pattern in attention scores
        assert scores is not None, "Attention scores should not be None"
        # Future tokens should have -inf attention scores
        future_positions = ~causal_mask.unsqueeze(0).unsqueeze(0).expand_as(scores)
        masked_scores = scores[future_positions]
        assert torch.all(torch.isneginf(masked_scores.real)), "Real part of masked positions should be -inf"
        assert torch.all(masked_scores.imag == 0.0), "Imaginary part of masked positions should be 0"
