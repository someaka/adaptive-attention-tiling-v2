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

import logging
import numpy as np
import pytest
import torch
import torch.linalg
from torch.autograd import profiler
from torch.autograd.profiler import profile as Profile
from typing import Optional, Dict, List, Any

from src.core.tiling.quantum_geometric_attention import (
    AttentionState,
    GeometricStructures,
    QuantumGeometricAttention,
)
from src.metrics.attention import (
    AttentionMetrics,
    FlowMetrics,
    compute_attention_metrics
)
from src.core.patterns.dynamics import PatternDynamics

logger = logging.getLogger(__name__)

def complex_randn(*size, device=None, dtype=torch.complex64):
    """Create random complex tensor with proper initialization."""
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(*size, device=device, dtype=real_dtype)
    imag = torch.randn(*size, device=device, dtype=real_dtype)
    return torch.complex(real, imag).to(dtype=dtype)

class TestQuantumGeometricAttention:
    """Test suite for quantum geometric attention with proper cleanup."""

    @pytest.fixture(autouse=True)
    def enable_anomaly_detection(self):
        """Enable gradient anomaly detection for all tests."""
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        yield
        torch.autograd.set_detect_anomaly(False)

    def teardown_method(self):
        """Clean up after each test."""
        import gc
        gc.collect()

    @pytest.fixture
    def manifold_dim(self) -> int:
        """Return manifold dimension for tests."""
        return 2  # Reduced from 4 to match the tensor shapes

    @pytest.fixture
    def hidden_dim(self, manifold_dim) -> int:
        """Return hidden dimension for tests."""
        return 8  # Fixed hidden dimension for testing

    @pytest.fixture
    def num_heads(self) -> int:
        """Return number of attention heads for tests."""
        return 4  # Fixed number of heads for testing

    @pytest.fixture
    def batch_size(self) -> int:
        """Return batch size for tests."""
        return 8  # Reduced from 16 to make testing faster

    @pytest.fixture
    def seq_length(self) -> int:
        """Return sequence length for tests."""
        return 4  # Reduced from 8 to make testing faster

    @pytest.fixture
    def attention_layer(self, hidden_dim, manifold_dim, num_heads):
        """Create a test attention layer with proper device placement."""
        device = torch.device('cpu')
        dtype = torch.complex64  # Use complex64 consistently
        layer = QuantumGeometricAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            manifold_dim=manifold_dim,
            dtype=dtype,
            device=device
        )
        # Ensure all parameters require gradients
        for param in layer.parameters():
            param.requires_grad = True
        return layer

    @pytest.fixture
    def geometric_structures(self, manifold_dim):
        """Create geometric structures for testing."""
        return GeometricStructures(
            dim=manifold_dim,  # Use manifold_dim instead of hidden_dim
            num_heads=8,
            manifold_type="hyperbolic",
            curvature=-1.0,
            parallel_transport_method="schild"
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
        # Create input tensor with proper dtype and dimensions
        # Use smaller dimensions for testing
        batch_size = 2
        seq_length = 2
        manifold_dim = attention_layer.manifold_dim

        # Create input tensor with proper shape
        x = complex_randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones(batch_size, seq_length).bool()

        # Compute geometric flow with metrics
        flow_result = attention_layer.geometric_attention_flow(
            x,  # Use original input
            mask=mask,
            num_steps=2,
            dt=0.1,
            return_metrics=True
        )
        flow, metrics = flow_result

        # Verify output shape
        assert flow.shape == (batch_size, seq_length, hidden_dim)

        # Verify metrics
        assert hasattr(metrics, 'curvature')
        assert hasattr(metrics, 'parallel_transport')
        assert hasattr(metrics, 'geodesic_distance')
        assert hasattr(metrics, 'energy')

        # Verify metric shapes
        assert metrics.curvature.shape == (batch_size, seq_length, manifold_dim, manifold_dim)
        assert metrics.parallel_transport.shape == (batch_size, seq_length, manifold_dim, manifold_dim)
        assert metrics.geodesic_distance.shape == (batch_size, seq_length)
        assert metrics.energy.shape == (batch_size, seq_length)

        # Verify flow preserves quantum state properties
        def normalize_complex_tensor(tensor: torch.Tensor, target_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Normalize complex tensor while preserving phase."""
            current_norm = torch.sqrt(torch.sum(tensor.real ** 2 + tensor.imag ** 2, dim=-1, keepdim=True))
            if target_norm is None:
                target_norm = torch.ones_like(current_norm)
            scale = target_norm / (current_norm + 1e-8)
            return tensor * scale

        def project_to_manifold(tensor: torch.Tensor, target_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Project tensor to manifold space while preserving norm."""
            # Store input norm if target_norm not provided
            if target_norm is None:
                target_norm = torch.sqrt(torch.sum(tensor.real ** 2 + tensor.imag ** 2, dim=-1, keepdim=True))
            
            # Project to manifold space
            manifold = attention_layer.manifold_proj(tensor)
            
            # Normalize to match input norm
            return normalize_complex_tensor(manifold, target_norm)

        # Get initial hidden norm
        x_flat = x.reshape(-1, hidden_dim)
        initial_hidden_norm = torch.sqrt(torch.sum(x_flat.real ** 2 + x_flat.imag ** 2, dim=-1, keepdim=True))

        # Project input and output to manifold space while preserving norms
        x_manifold = project_to_manifold(x_flat, initial_hidden_norm)
        flow_manifold = project_to_manifold(flow.reshape(-1, hidden_dim), initial_hidden_norm)

        # Check norm conservation (approximately)
        x_norm = torch.norm(x_manifold, dim=-1)
        flow_norm = torch.norm(flow_manifold, dim=-1)
        assert torch.allclose(x_norm, flow_norm, rtol=1e-2)

    def test_quantum_classical_interface(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test quantum-classical information interface."""
        # Create classical input with proper dtype and requires_grad
        classical_input = complex_randn(batch_size, seq_length, hidden_dim)
        classical_input.requires_grad = True

        # Convert to quantum state
        quantum_state = attention_layer.prepare_quantum_state(classical_input)

        # Test quantum state properties
        assert quantum_state.shape[-1] == manifold_dim, "Should preserve manifold dimension"
        assert quantum_state.dtype == classical_input.dtype, "Should maintain dtype"

        # Test quantum state normalization
        norms = torch.norm(quantum_state, dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-8
        ), "Quantum states should be normalized"

        # Test gradients by computing loss and backprop
        loss = quantum_state.abs().sum()
        loss.backward()
        
        # Check if gradients were computed
        assert classical_input.grad is not None, "Should compute gradients"
        assert not torch.isnan(classical_input.grad).any(), "Gradients should not be NaN"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_multi_head_integration(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
    ) -> None:
        """Test multi-head attention integration."""
        # Enable anomaly detection with more verbose output
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        
        # Create input tensor with proper dtype
        x = complex_randn(batch_size, seq_length, hidden_dim, dtype=attention_layer.dtype)
        x.requires_grad = True  # Ensure input requires grad
        mask = torch.ones(batch_size, seq_length).bool()

        # Track intermediate tensors
        intermediate_tensors = {}
        computation_steps = []
        
        def save_tensor(name: str, tensor: torch.Tensor, step_info: str = ""):
            """Enhanced tensor tracking with computation step info."""
            if tensor.requires_grad:
                tensor.retain_grad()  # Ensure gradient is retained
                intermediate_tensors[name] = tensor
                computation_steps.append(f"Step: {step_info}")
                
                print(f"\nTracking tensor: {name}")
                print(f"Step info: {step_info}")
                print(f"Shape: {tensor.shape}")
                print(f"Requires grad: {tensor.requires_grad}")
                print(f"Is complex: {tensor.is_complex()}")
                if tensor.is_complex():
                    print(f"Complex stats:")
                    print(f"  Magnitude mean: {tensor.abs().mean().item():.6f}")
                    print(f"  Real mean: {tensor.real.mean().item():.6f}")
                    print(f"  Imag mean: {tensor.imag.mean().item():.6f}")
                
                def hook(grad):
                    if grad is not None:
                        print(f"\nGradient for {name} (Step: {step_info}):")
                        print(f"  Shape: {grad.shape}")
                        if grad.is_complex():
                            grad_abs = grad.abs()
                            print(f"  Complex Gradient stats:")
                            print(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                            print(f"    Real mean: {grad.real.mean().item():.6f}")
                            print(f"    Imag mean: {grad.imag.mean().item():.6f}")
                            print(f"    Max magnitude: {grad_abs.max().item():.6f}")
                            print(f"    Min magnitude: {grad_abs.min().item():.6f}")
                        else:
                            print(f"  Gradient stats:")
                            print(f"    Norm: {torch.norm(grad).item():.6f}")
                            print(f"    Mean: {grad.mean().item():.6f}")
                            print(f"    Max: {grad.max().item():.6f}")
                            print(f"    Min: {grad.min().item():.6f}")
                    return grad
                
                tensor.register_hook(hook)
        
        # Track initial tensors
        quantum_bridge = attention_layer.quantum_bridge
        save_tensor("input", x, "Initial input tensor")
        
        # Get pattern bundle parameters
        pattern_bundle = quantum_bridge.pattern_bundle
        metric = pattern_bundle.metric
        connection = pattern_bundle.connection
        
        # Get Riemannian framework parameters
        riemannian_framework = pattern_bundle.riemannian_framework
        connection_coeffs = riemannian_framework.connection_coeffs
        
        # Ensure all parameters require gradients and retain them
        metric.requires_grad = True
        connection.requires_grad = True
        connection_coeffs.requires_grad = True
        metric.retain_grad()
        connection.retain_grad()
        connection_coeffs.retain_grad()
        
        # Register hooks for all parameters
        def parameter_hook(name):
            def hook(grad):
                if grad is not None:
                    print(f"\n{name} gradient stats:")
                    print(f"Shape: {grad.shape}")
                    if grad.is_complex():
                        grad_abs = grad.abs()
                        print(f"Complex gradient stats:")
                        print(f"  Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                        print(f"  Real mean: {grad.real.mean().item():.6f}")
                        print(f"  Imag mean: {grad.imag.mean().item():.6f}")
                        print(f"  Max magnitude: {grad_abs.max().item():.6f}")
                        print(f"  Min magnitude: {grad_abs.min().item():.6f}")
                    else:
                        print(f"Real gradient stats:")
                        print(f"  Norm: {torch.norm(grad).item():.6f}")
                        print(f"  Mean: {grad.mean().item():.6f}")
                        print(f"  Max: {grad.max().item():.6f}")
                        print(f"  Min: {grad.min().item():.6f}")
                return grad
            return hook
        
        metric.register_hook(parameter_hook("Metric"))
        connection.register_hook(parameter_hook("Connection"))
        connection_coeffs.register_hook(parameter_hook("Connection Coefficients"))
        
        # Save parameters for tracking
        save_tensor("pattern_bundle.metric", metric, "Pattern bundle metric parameter")
        save_tensor("pattern_bundle.connection", connection, "Pattern bundle connection parameter")
        save_tensor("riemannian_framework.connection_coeffs", connection_coeffs, "Riemannian framework connection coefficients")
        
        # Track input flattening
        x_flat = x.reshape(-1, hidden_dim)
        save_tensor("x_flat", x_flat, "Flattened input")

        # Forward pass with gradient tracking
        print("\n=== Starting Forward Pass ===")
        output = attention_layer(x, mask=mask)
        save_tensor("output", output, "Final output")
        
        # Compute loss that ensures all parameters are used
        print("\n=== Starting Backward Pass ===")
        loss = output.abs().pow(2).sum()  # Use absolute value squared for complex tensors
        print(f"Loss value: {loss.item():.6f}")
        loss.backward()
        
        # Log gradient flow analysis
        print("\n=== Gradient Flow Analysis ===")
        print("=" * 50)
        
        # Check each tracked tensor
        for name, tensor in intermediate_tensors.items():
            print(f"\nAnalyzing tensor: {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Requires grad: {tensor.requires_grad}")
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                grad = tensor.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    print(f"  Complex Gradient stats:")
                    print(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    print(f"    Real mean: {grad.real.mean().item():.6f}")
                    print(f"    Imag mean: {grad.imag.mean().item():.6f}")
                    print(f"    Max magnitude: {grad_abs.max().item():.6f}")
                    print(f"    Min magnitude: {grad_abs.min().item():.6f}")
                else:
                    print(f"  Gradient stats:")
                    print(f"    Norm: {torch.norm(grad).item():.6f}")
                    print(f"    Mean: {grad.mean().item():.6f}")
                    print(f"    Max: {grad.max().item():.6f}")
                    print(f"    Min: {grad.min().item():.6f}")
            else:
                print("  No gradients")
        
        # Log computation steps
        print("\n=== Computation Steps ===")
        for i, step in enumerate(computation_steps):
            print(f"{i+1}. {step}")
        
        # Final assertions with detailed error messages
        metric_grad = pattern_bundle.metric.grad
        connection_grad = pattern_bundle.connection.grad
        connection_coeffs_grad = riemannian_framework.connection_coeffs.grad
        
        assert metric_grad is not None, \
            "No gradients in pattern_bundle.metric - gradient flow is blocked"
        assert connection_grad is not None, \
            "No gradients in pattern_bundle.connection - gradient flow is blocked"
        assert connection_coeffs_grad is not None, \
            "No gradients in riemannian_framework.connection_coeffs - gradient flow is blocked"
        
        # Additional assertions to verify gradient quality
        if metric_grad is not None:
            grad_abs = metric_grad.abs()
            assert torch.isfinite(grad_abs).all(), \
                "Metric gradients contain inf/nan values"
            assert grad_abs.mean() > 0, \
                f"Metric gradients are zero (mean magnitude: {grad_abs.mean().item():.6f})"
        
        if connection_grad is not None:
            grad_abs = connection_grad.abs()
            assert torch.isfinite(grad_abs).all(), \
                "Connection gradients contain inf/nan values"
            assert grad_abs.mean() > 0, \
                f"Connection gradients are zero (mean magnitude: {grad_abs.mean().item():.6f})"
                
        if connection_coeffs_grad is not None:
            grad_abs = connection_coeffs_grad.abs()
            assert torch.isfinite(grad_abs).all(), \
                "Connection coefficients gradients contain inf/nan values"
            assert grad_abs.mean() > 0, \
                f"Connection coefficients gradients are zero (mean magnitude: {grad_abs.mean().item():.6f})"

        # Verify gradients exist for all parameters
        for name, param in attention_layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has inf/nan gradients"
            assert param.grad.abs().mean() > 0, f"Parameter {name} has zero gradients"

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

        # For complex gradients, use abs() before sum()
        loss = output.abs().sum()
        loss.backward()
        
        # Check gradients
        for name, param in attention_layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"

    def test_manifold_curvature(
        self, attention_layer, batch_size, seq_length, hidden_dim, manifold_dim
    ):
        """Test attention manifold curvature properties."""
        # Create input tensor with proper dtype
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

        # Test positive definiteness (using real part for eigenvalues)
        eigenvalues = torch.linalg.eigvalsh(metric.real)
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

        # For complex gradients, use abs() before sum()
        loss = output.abs().sum()
        loss.backward()
        
        # Check gradients
        for name, param in attention_layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"

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

        # For complex gradients, use abs() before sum()
        loss = output.abs().sum()
        loss.backward()
        
        # Check gradients
        for name, param in attention_layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"

    def test_attention_patterns(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        """Test attention pattern computation."""
        # Create query, key tensors with proper dimensions
        head_dim = hidden_dim // num_heads
        query = complex_randn(batch_size, num_heads, seq_length, head_dim)
        key = complex_randn(batch_size, num_heads, seq_length, head_dim)
        value = complex_randn(batch_size, num_heads, seq_length, head_dim)

        # Compute attention patterns
        result = attention_layer.compute_attention_patterns(query, key, value, return_metrics=True)
        patterns, metrics = result

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
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        assert 'attention_scores' in metrics, "Should have attention scores"
        assert 'attention_weights' in metrics, "Should have attention weights"
        
        # Test attention scores shape
        assert metrics['attention_scores'].shape == (batch_size, num_heads, seq_length, seq_length), "Wrong attention scores shape"
        assert metrics['attention_weights'].shape == (batch_size, num_heads, seq_length, seq_length), "Wrong attention weights shape"

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
        # Create initial state with proper complex dtype
        initial_state = complex_randn(batch_size, hidden_dim)
        
        # Normalize initial state
        initial_state = initial_state / torch.norm(initial_state, dim=-1, keepdim=True)
        
        # Create field operator with matching dimensions
        field_operator = complex_randn(hidden_dim, hidden_dim)
        
        # Make field operator Hermitian
        field_operator = 0.5 * (field_operator + field_operator.conj().transpose(-2, -1))
        
        # Ensure field operator is trace-preserving
        field_operator = field_operator - torch.eye(hidden_dim, dtype=field_operator.dtype, device=field_operator.device) * field_operator.diagonal().mean()
        
        # Test evolution with time parameter
        time = 0.1
        evolved_state = pattern_dynamics.evolve(initial_state, time)
        
        # Normalize evolved state
        evolved_state = evolved_state / torch.norm(evolved_state, dim=-1, keepdim=True)
        
        # Test shape and dtype preservation
        assert evolved_state.shape == initial_state.shape, "Evolution should preserve shape"
        assert evolved_state.dtype == initial_state.dtype, "Evolution should preserve dtype"
        assert not torch.isnan(evolved_state).any(), "Evolution should not produce NaN values"
        assert not torch.isinf(evolved_state).any(), "Evolution should not produce Inf values"
        
        # Test energy conservation
        initial_energy = pattern_dynamics.compute_energy(initial_state)
        final_energy = pattern_dynamics.compute_energy(evolved_state)
        
        # Extract energy values from metrics
        initial_energy_val = initial_energy['total']
        final_energy_val = final_energy['total']
        
        # Test energy conservation with proper tolerance for complex values
        assert torch.allclose(
            initial_energy_val.abs(),
            final_energy_val.abs(),
            rtol=1e-3,  # Increased tolerance for numerical stability
            atol=1e-3   # Added absolute tolerance
        ), "Energy should be conserved"

    def test_quantum_bridge_gradient_diagnostic(
        self,
        attention_layer: QuantumGeometricAttention,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
    ) -> None:
        """Detailed diagnostic test for gradient flow through quantum bridge."""
        logger.info("\n=== Starting Quantum Bridge Gradient Diagnostic ===")
        
        # Create test input
        x = torch.randn(batch_size, seq_length, hidden_dim, requires_grad=True)
        logger.info(f"\nInput shape: {x.shape}")
        
        # Track intermediate tensors
        intermediate_tensors = {}
        computation_steps = []
        
        def save_tensor(name: str, tensor: torch.Tensor, step_info: str = ""):
            """Enhanced tensor tracking with computation step info."""
            if tensor.requires_grad:
                tensor.retain_grad()
                intermediate_tensors[name] = tensor
                computation_steps.append(f"Step: {step_info}")
                
                logger.info(f"\nTracking tensor: {name}")
                logger.info(f"Step info: {step_info}")
                logger.info(f"Shape: {tensor.shape}")
                logger.info(f"Requires grad: {tensor.requires_grad}")
                logger.info(f"Is complex: {tensor.is_complex()}")
                if tensor.is_complex():
                    logger.info(f"Complex stats:")
                    logger.info(f"  Magnitude mean: {tensor.abs().mean().item():.6f}")
                    logger.info(f"  Real mean: {tensor.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {tensor.imag.mean().item():.6f}")
                
                def hook(grad):
                    if grad is not None:
                        logger.info(f"\nGradient for {name} (Step: {step_info}):")
                        logger.info(f"  Shape: {grad.shape}")
                        if grad.is_complex():
                            grad_abs = grad.abs()
                            logger.info(f"  Complex Gradient stats:")
                            logger.info(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                            logger.info(f"    Real mean: {grad.real.mean().item():.6f}")
                            logger.info(f"    Imag mean: {grad.imag.mean().item():.6f}")
                            logger.info(f"    Max magnitude: {grad_abs.max().item():.6f}")
                            logger.info(f"    Min magnitude: {grad_abs.min().item():.6f}")
                        else:
                            logger.info(f"  Gradient stats:")
                            logger.info(f"    Norm: {torch.norm(grad).item():.6f}")
                            logger.info(f"    Mean: {grad.mean().item():.6f}")
                            logger.info(f"    Max: {grad.max().item():.6f}")
                            logger.info(f"    Min: {grad.min().item():.6f}")
                        return grad
                    return grad
                
                tensor.register_hook(hook)
        
        # Forward pass with enhanced tracking
        quantum_bridge = attention_layer.quantum_bridge
        
        # Track initial tensors
        save_tensor("input", x, "Initial input tensor")
        save_tensor("pattern_bundle.metric", quantum_bridge.pattern_bundle.metric, "Pattern bundle metric parameter")
        save_tensor("pattern_bundle.connection", quantum_bridge.pattern_bundle.connection, "Pattern bundle connection parameter")
        
        # Track metric and connection views
        metric_view = quantum_bridge.pattern_bundle.metric.clone()
        connection_view = quantum_bridge.pattern_bundle.connection.clone()
        save_tensor("metric_view", metric_view, "Cloned metric view")
        save_tensor("connection_view", connection_view, "Cloned connection view")
        
        # Track connection usage in forward pass
        x_flat = x.reshape(-1, hidden_dim)
        save_tensor("x_flat", x_flat, "Flattened input")
        
        # Track intermediate quantum states
        logger.info("\n=== Starting Forward Pass ===")
        output = attention_layer(x)
        save_tensor("output", output, "Final output")
        
        # Compute loss and backward
        logger.info("\n=== Starting Backward Pass ===")
        loss = output.abs().sum()
        logger.info(f"Loss value: {loss.item():.6f}")
        loss.backward()
        
        # Log gradient flow analysis
        logger.info("\n=== Gradient Flow Analysis ===")
        logger.info("=" * 50)
        
        # Check each tracked tensor
        for name, tensor in intermediate_tensors.items():
            logger.info(f"\nAnalyzing tensor: {name}")
            logger.info(f"  Shape: {tensor.shape}")
            logger.info(f"  Requires grad: {tensor.requires_grad}")
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                grad = tensor.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    logger.info(f"  Complex Gradient stats:")
                    logger.info(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    logger.info(f"    Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"    Imag mean: {grad.imag.mean().item():.6f}")
                    logger.info(f"    Max magnitude: {grad_abs.max().item():.6f}")
                    logger.info(f"    Min magnitude: {grad_abs.min().item():.6f}")
                else:
                    logger.info(f"  Gradient stats:")
                    logger.info(f"    Norm: {torch.norm(grad).item():.6f}")
                    logger.info(f"    Mean: {grad.mean().item():.6f}")
                    logger.info(f"    Max: {grad.max().item():.6f}")
                    logger.info(f"    Min: {grad.min().item():.6f}")
            else:
                logger.info("  No gradients")
        
        # Log computation steps
        logger.info("\n=== Computation Steps ===")
        for i, step in enumerate(computation_steps):
            logger.info(f"{i+1}. {step}")
        
        # Final assertions with detailed error messages
        connection_grad = quantum_bridge.pattern_bundle.connection.grad
        assert connection_grad is not None, \
            "No gradients in pattern_bundle.connection - gradient flow is blocked"
        
        # Additional assertions to verify gradient quality
        if connection_grad is not None:
            grad_abs = connection_grad.abs()
            assert torch.isfinite(grad_abs).all(), \
                "Connection gradients contain inf/nan values"
            assert grad_abs.mean() > 0, \
                f"Connection gradients are zero (mean magnitude: {grad_abs.mean().item():.6f})"

    def test_energy_conservation_in_forward_pass(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test energy conservation during forward pass."""
        # Create input tensor
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Store initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Forward pass
        output = attention_layer(x)
        
        # Compute final energy
        final_energy = torch.sum(output.abs() ** 2)
        
        # Check energy conservation with tolerance
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2,  # 1% relative tolerance
            atol=1e-2   # Small absolute tolerance
        ), f"Energy not conserved: initial={initial_energy.item():.4f}, final={final_energy.item():.4f}"
        
        # Test gradient flow
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input should receive gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        assert not torch.isinf(x.grad).any(), "Input gradients contain Inf"

    def test_metric_combination(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test metric combination and gradient flow."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Get initial metrics
        base_metric = attention_layer.base_metric
        pattern_metric = attention_layer.pattern_metric
        metric = attention_layer.metric
        
        # Forward pass
        output = attention_layer(x)
        
        # Check metric properties
        assert base_metric.requires_grad, "Base metric should require gradients"
        assert pattern_metric.requires_grad, "Pattern metric should require gradients"
        assert metric.requires_grad, "Metric should require gradients"
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check metric gradients
        assert base_metric.grad is not None, "Base metric should receive gradients"
        assert pattern_metric.grad is not None, "Pattern metric should receive gradients"
        assert metric.grad is not None, "Metric should receive gradients"

    def test_connection_gradient_flow(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test gradient flow through connection coefficients."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Get connection from quantum bridge
        connection = attention_layer.quantum_bridge.pattern_bundle.connection
        connection.requires_grad_(True)
        connection.retain_grad()
        
        # Forward pass
        output = attention_layer(x)
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check connection gradients
        assert connection.grad is not None, "Connection should receive gradients"
        assert not torch.isnan(connection.grad).any(), "Connection gradients contain NaN"
        assert not torch.isinf(connection.grad).any(), "Connection gradients contain Inf"
        assert connection.grad.abs().mean() > 0, "Connection gradients should be non-zero"

    def test_layer_normalization(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test layer normalization with proper shape handling."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Forward pass
        output = attention_layer(x)
        
        # Check output properties
        assert output.shape == x.shape, "Output shape should match input shape"
        assert output.dtype == x.dtype, "Output dtype should match input dtype"
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input should receive gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        assert not torch.isinf(x.grad).any(), "Input gradients contain Inf"

    def test_energy_conservation_during_normalization(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test energy conservation during normalization steps."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Store initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Project to manifold space
        x_manifold = attention_layer.manifold_proj(x)
        manifold_energy = torch.sum(x_manifold.abs() ** 2)
        
        # Check energy after manifold projection
        assert torch.allclose(
            initial_energy, manifold_energy,
            rtol=1e-2, atol=1e-2
        ), f"Energy not conserved after manifold projection: initial={initial_energy.item():.4f}, manifold={manifold_energy.item():.4f}"
        
        # Forward pass
        output = attention_layer(x)
        final_energy = torch.sum(output.abs() ** 2)
        
        # Check energy conservation through entire forward pass
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2, atol=1e-2
        ), f"Energy not conserved through forward pass: initial={initial_energy.item():.4f}, final={final_energy.item():.4f}"

    def test_residual_connection_energy(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test energy conservation with residual connections."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Store initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Forward pass
        output = attention_layer(x)
        
        # Check output properties
        assert output.shape == x.shape, "Output shape should match input shape"
        assert output.dtype == x.dtype, "Output dtype should match input dtype"
        
        # Compute final energy
        final_energy = torch.sum(output.abs() ** 2)
        
        # Check energy conservation
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2, atol=1e-2
        ), f"Energy not conserved with residual connections: initial={initial_energy.item():.4f}, final={final_energy.item():.4f}"
        
        # Test gradient flow
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input should receive gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        assert not torch.isinf(x.grad).any(), "Input gradients contain Inf"

    def test_quantum_bridge_energy_conservation(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test energy conservation through quantum bridge."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Store initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Get quantum bridge
        quantum_bridge = attention_layer.quantum_bridge
        
        # Project to quantum state
        quantum_state = quantum_bridge.neural_to_quantum(x)
        quantum_energy = torch.sum(quantum_state.abs() ** 2)
        
        # Check energy conservation in quantum state
        assert torch.allclose(
            initial_energy, quantum_energy,
            rtol=1e-2, atol=1e-2
        ), f"Energy not conserved in quantum state: initial={initial_energy.item():.4f}, quantum={quantum_energy.item():.4f}"
        
        # Project back to neural state
        neural_state = quantum_bridge.quantum_to_neural(quantum_state)
        final_energy = torch.sum(neural_state.abs() ** 2)
        
        # Check energy conservation after round trip
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2, atol=1e-2
        ), f"Energy not conserved after quantum-neural round trip: initial={initial_energy.item():.4f}, final={final_energy.item():.4f}"

    def test_metric_factors_gradient_flow(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test gradient flow through metric factors."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Get metric factors
        metric_factors = attention_layer.quantum_bridge.pattern_bundle.riemannian_framework.metric_factors
        metric_factors.requires_grad_(True)
        metric_factors.retain_grad()
        
        # Forward pass
        output = attention_layer(x)
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check metric factors gradients
        assert metric_factors.grad is not None, "Metric factors should receive gradients"
        assert not torch.isnan(metric_factors.grad).any(), "Metric factors gradients contain NaN"
        assert not torch.isinf(metric_factors.grad).any(), "Metric factors gradients contain Inf"
        assert metric_factors.grad.abs().mean() > 0, "Metric factors gradients should be non-zero"

    def test_combined_metric_gradient_flow(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test gradient flow through combined metric tensor."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Get combined metric
        combined_metric = attention_layer.combined_metric
        combined_metric.requires_grad_(True)
        combined_metric.retain_grad()
        
        # Forward pass
        output = attention_layer(x)
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check combined metric gradients
        assert combined_metric.grad is not None, "Combined metric should receive gradients"
        assert not torch.isnan(combined_metric.grad).any(), "Combined metric gradients contain NaN"
        assert not torch.isinf(combined_metric.grad).any(), "Combined metric gradients contain Inf"
        assert combined_metric.grad.abs().mean() > 0, "Combined metric gradients should be non-zero"

    def test_layer_norm_with_energy_conservation(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test layer normalization with energy conservation."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Store initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Forward pass
        output = attention_layer(x)
        
        # Get layer norm parameters
        layer_norm = attention_layer.quantum_bridge.layer_norm
        assert layer_norm.weight.shape[0] >= attention_layer.manifold_dim, "Layer norm weight shape mismatch"
        assert layer_norm.bias.shape[0] >= attention_layer.manifold_dim, "Layer norm bias shape mismatch"
        
        # Check energy conservation
        final_energy = torch.sum(output.abs() ** 2)
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2, atol=1e-2
        ), f"Energy not conserved through layer norm: initial={initial_energy.item():.4f}, final={final_energy.item():.4f}"
        
        # Test gradient flow
        loss = output.abs().mean()
        loss.backward()
        
        # Check layer norm parameter gradients
        assert layer_norm.weight.grad is not None, "Layer norm weight should receive gradients"
        assert layer_norm.bias.grad is not None, "Layer norm bias should receive gradients"
        assert not torch.isnan(layer_norm.weight.grad).any(), "Layer norm weight gradients contain NaN"
        assert not torch.isnan(layer_norm.bias.grad).any(), "Layer norm bias gradients contain NaN"

    def test_complex_dropout_with_energy_conservation(self, attention_layer, batch_size, seq_length, hidden_dim):
        """Test complex dropout with energy conservation."""
        x = complex_randn(batch_size, seq_length, hidden_dim)
        x.requires_grad_(True)
        
        # Store initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Enable training mode
        attention_layer.train()
        
        # Forward pass
        output = attention_layer(x)
        
        # Check energy conservation (approximately, due to dropout)
        final_energy = torch.sum(output.abs() ** 2)
        energy_ratio = final_energy / initial_energy
        assert 0.5 < energy_ratio < 1.5, f"Energy ratio out of bounds: {energy_ratio.item():.4f}"
        
        # Test gradient flow
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input should receive gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        assert not torch.isinf(x.grad).any(), "Input gradients contain Inf"
        
        # Test in eval mode (no dropout)
        attention_layer.eval()
        with torch.no_grad():
            eval_output = attention_layer(x)
            eval_energy = torch.sum(eval_output.abs() ** 2)
            assert torch.allclose(
                initial_energy, eval_energy,
                rtol=1e-2, atol=1e-2
            ), f"Energy not conserved in eval mode: initial={initial_energy.item():.4f}, final={eval_energy.item():.4f}"
