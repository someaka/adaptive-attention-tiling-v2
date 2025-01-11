"""Unit tests for quantum geometric attention mechanism.

This module provides comprehensive testing for the quantum geometric attention mechanism,
covering core functionality, gradient flow, energy conservation, and geometric properties.
Tests are organized by component and functionality, ensuring thorough coverage while
maintaining clarity and maintainability.

Test Categories:
1. Core Attention Mechanics
2. Geometric Properties
3. Energy Conservation
4. Gradient Flow
5. Error Handling
"""

import logging
import pytest
import torch
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from src.core.tiling.quantum_geometric_attention import (
    AttentionState,
    GeometricStructures,
    QuantumGeometricAttention,
)
from src.metrics.attention import AttentionMetrics
from src.core.patterns.dynamics import PatternDynamics

logger = logging.getLogger(__name__)

# Test Configuration
@dataclass
class TestConfig:
    batch_size: int = 2
    seq_length: int = 4
    hidden_dim: int = 16
    num_heads: int = 2
    manifold_dim: int = 8
    device: torch.device = torch.device('cpu')
    dtype: torch.dtype = torch.complex64

TEST_CONFIG = TestConfig()

def complex_randn(*size: int, device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Create random complex tensor with proper initialization."""
    device = device or TEST_CONFIG.device
    dtype = dtype or TEST_CONFIG.dtype
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(*size, device=device, dtype=real_dtype)
    imag = torch.randn(*size, device=device, dtype=real_dtype)
    return torch.complex(real, imag).to(dtype=dtype)

@pytest.fixture(scope="class")
def test_config() -> TestConfig:
    """Provide test configuration."""
    return TEST_CONFIG

@pytest.fixture(scope="function")
def attention_layer(test_config: TestConfig) -> QuantumGeometricAttention:
    """Create attention layer for testing."""
    layer = QuantumGeometricAttention(
        hidden_dim=test_config.hidden_dim,
        num_heads=test_config.num_heads,
        manifold_dim=test_config.manifold_dim,
        dtype=test_config.dtype,
        device=test_config.device
    )
    for param in layer.parameters():
        param.requires_grad_(True)
    return layer

@pytest.fixture(scope="function")
def test_input(test_config: TestConfig) -> torch.Tensor:
    """Create test input tensor."""
    return complex_randn(
        test_config.batch_size, 
        test_config.seq_length, 
        test_config.hidden_dim,
        device=test_config.device,
        dtype=test_config.dtype
    )

@pytest.fixture(scope="function")
def attention_mask(test_config: TestConfig) -> torch.Tensor:
    """Create attention mask."""
    return torch.ones(
        test_config.batch_size,
        test_config.seq_length,
        dtype=torch.bool,
        device=test_config.device
    )

class TestQuantumGeometricAttention:
    """Test suite for quantum geometric attention mechanism."""

    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Setup test environment."""
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        yield
        torch.autograd.set_detect_anomaly(False)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def verify_gradients(self, tensor: torch.Tensor, name: str):
        """Helper to verify gradient properties."""
        assert tensor.grad is not None, f"{name} should have gradients"
        assert torch.isfinite(tensor.grad).all(), f"{name} has inf/nan gradients"
        assert tensor.grad.abs().mean() > 0, f"{name} has zero gradients"

    def verify_tensor_properties(self, tensor: torch.Tensor, name: str):
        """Helper to verify tensor properties."""
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains infinite values"
        assert tensor.dtype == TEST_CONFIG.dtype, f"{name} has incorrect dtype"

    def test_attention_state(self, attention_layer, test_input, attention_mask):
        """Test attention state preparation and properties."""
        state = attention_layer.prepare_attention_state(test_input, attention_mask)
        
        # Verify state properties
        assert isinstance(state, AttentionState)
        quantum_state = state.state_manager.states.get("quantum")
        assert quantum_state is not None
        
        # Verify shapes
        expected_shape = (
            TEST_CONFIG.batch_size,
            TEST_CONFIG.num_heads,
            TEST_CONFIG.seq_length,
            TEST_CONFIG.manifold_dim
        )
        assert quantum_state.shape == expected_shape
        
        # Verify normalization
        norms = quantum_state.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)
        
        # Verify tensor properties
        self.verify_tensor_properties(quantum_state, "quantum_state")

    def test_attention_patterns(self, attention_layer: QuantumGeometricAttention, test_config: TestConfig):
        """Test attention pattern computation."""
        # Setup dimensions
        head_dim = test_config.hidden_dim // test_config.num_heads
        
        # Create inputs
        query = complex_randn(
            test_config.batch_size,
            test_config.num_heads,
            test_config.seq_length,
            head_dim
        )
        key = complex_randn(
            test_config.batch_size,
            test_config.num_heads,
            test_config.seq_length,
            head_dim
        )
        value = complex_randn(
            test_config.batch_size,
            test_config.num_heads,
            test_config.seq_length,
            head_dim
        )

        # Project to manifold space
        def project_to_manifold(x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(-1, head_dim)
            proj = attention_layer.manifold_proj(flat)
            return proj.reshape(
                test_config.batch_size,
                test_config.num_heads,
                test_config.seq_length,
                test_config.manifold_dim
            )

        query_m = project_to_manifold(query)
        key_m = project_to_manifold(key)
        value_m = project_to_manifold(value)

        # Compute patterns
        patterns, metrics = attention_layer.compute_attention_patterns(
            query_m, key_m, value_m,
            return_metrics=True
        )

        # Verify shapes
        expected_shape = (
            test_config.batch_size,
            test_config.num_heads,
            test_config.seq_length,
            test_config.seq_length
        )
        assert patterns.shape == expected_shape

        # Verify normalization
        row_sums = patterns.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5)

        # Verify metrics
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['attention_scores', 'attention_weights'])
        assert all(m.shape == expected_shape for m in [
            metrics['attention_scores'],
            metrics['attention_weights']
        ])

    def test_geometric_flow(self, attention_layer, test_input, attention_mask, test_config):
        """Test geometric flow computation."""
        flow, metrics = attention_layer.geometric_attention_flow(
            test_input,
            mask=attention_mask,
            num_steps=2,
            dt=0.1,
            return_metrics=True
        )

        # Verify shapes
        assert flow.shape == test_input.shape
        assert metrics.curvature.shape == (
            test_config.batch_size,
            test_config.seq_length,
            test_config.manifold_dim,
            test_config.manifold_dim
        )
        assert metrics.parallel_transport.shape == (
            test_config.batch_size,
            test_config.seq_length,
            test_config.manifold_dim,
            test_config.manifold_dim
        )
        assert metrics.geodesic_distance.shape == (
            test_config.batch_size,
            test_config.seq_length
        )

        # Verify tensor properties
        self.verify_tensor_properties(flow, "flow")
        self.verify_tensor_properties(metrics.curvature, "curvature")
        self.verify_tensor_properties(metrics.parallel_transport, "parallel_transport")

    def test_energy_conservation(self, attention_layer, test_input):
        """Test energy conservation through quantum bridge."""
        # Initial energy
        initial_energy = torch.sum(test_input.abs() ** 2)
        
        # Forward pass
        output = attention_layer(test_input)
        final_energy = torch.sum(output.abs() ** 2)
        
        # Verify energy conservation
        assert torch.allclose(initial_energy, final_energy, rtol=1e-2), (
            f"Energy not conserved: initial={initial_energy.item():.4f}, "
            f"final={final_energy.item():.4f}"
        )

    def test_gradient_flow(self, attention_layer, test_input):
        """Test gradient flow through all components."""
        test_input.requires_grad_(True)
        
        # Track components
        components = {
            "base_metric": attention_layer.base_metric,
            "pattern_metric": attention_layer.pattern_metric,
            "combined_metric": attention_layer.combined_metric,
            "pattern_bundle_metric": attention_layer.quantum_bridge.pattern_bundle.metric,
            "connection": attention_layer.quantum_bridge.pattern_bundle.connection,
            "metric_factors": attention_layer.quantum_bridge.pattern_bundle.riemannian_framework.metric_factors
        }
        
        # Enable gradients
        for tensor in components.values():
            tensor.requires_grad_(True)
            tensor.retain_grad()
        
        # Forward and backward
        output = attention_layer(test_input)
        loss = output.abs().mean()
        loss.backward()
        
        # Verify gradients
        for name, tensor in components.items():
            self.verify_gradients(tensor, name)

    def test_error_handling(self, attention_layer: QuantumGeometricAttention):
        """Test error handling for invalid inputs."""
        # Invalid shape
        with pytest.raises(ValueError, match="Invalid input shape"):
            invalid_shape = complex_randn(
                TEST_CONFIG.batch_size,
                TEST_CONFIG.seq_length + 1,
                TEST_CONFIG.hidden_dim
            )
            attention_layer(invalid_shape)
        
        # NaN values
        with pytest.raises(ValueError, match="Input contains NaN values"):
            nan_input = complex_randn(
                TEST_CONFIG.batch_size,
                TEST_CONFIG.seq_length,
                TEST_CONFIG.hidden_dim
            )
            nan_input[0, 0] = float('nan')
            attention_layer(nan_input)
        
        # Infinite values
        with pytest.raises(ValueError, match="Input contains infinite values"):
            inf_input = complex_randn(
                TEST_CONFIG.batch_size,
                TEST_CONFIG.seq_length,
                TEST_CONFIG.hidden_dim
            )
            inf_input[0, 0] = float('inf')
            attention_layer(inf_input)
