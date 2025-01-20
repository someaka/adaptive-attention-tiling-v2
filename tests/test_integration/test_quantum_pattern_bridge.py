"""Tests for quantum-pattern bridge integration.

This module implements tests for the conversion between quantum states and patterns,
verifying the preservation of quantum properties and geometric structure during conversion.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, NamedTuple

from src.core.quantum.types import QuantumState
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.neural.attention.pattern.pattern_dynamics import PatternDynamics
from src.neural.attention.pattern.quantum import QuantumGeometricTensor
from src.utils.test_helpers import assert_manifold_properties
from src.validation.geometric.metric import ConnectionValidator, ConnectionValidation
from tests.utils.config_loader import load_test_config


class BridgeComponents(NamedTuple):
    """Components needed for quantum bridge tests."""
    bridge: NeuralQuantumBridge
    dynamics: PatternDynamics


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Load test configuration based on environment."""
    return load_test_config()


@pytest.fixture
def bridge_components(test_config) -> BridgeComponents:
    """Setup quantum bridge components with proper configuration."""
    hidden_dim = test_config["quantum_bridge"]["hidden_dim"]
    manifold_dim = test_config["quantum_bridge"]["manifold_dim"]
    dtype = getattr(torch, test_config["quantum_bridge"]["dtype"])
    
    bridge = NeuralQuantumBridge(hidden_dim=hidden_dim, manifold_dim=manifold_dim, dtype=dtype)
    dynamics = PatternDynamics(hidden_dim=hidden_dim)
    
    return BridgeComponents(bridge=bridge, dynamics=dynamics)


@pytest.fixture
def test_pattern(test_config) -> torch.Tensor:
    """Create a normalized test pattern."""
    batch_size = test_config["quantum_bridge"]["batch_size"]
    hidden_dim = test_config["quantum_bridge"]["hidden_dim"]
    dtype = getattr(torch, test_config["quantum_bridge"]["dtype"])
    
    pattern = torch.randn(batch_size, hidden_dim, dtype=dtype)
    return F.normalize(pattern, dim=-1)


@pytest.fixture
def quantum_state(bridge_components, test_pattern) -> QuantumState:
    """Create a quantum state from test pattern."""
    return bridge_components.dynamics._to_quantum_state(test_pattern)


class TestStateConversion:
    """Tests for quantum state conversion."""
    
    def test_pattern_to_quantum(self, bridge_components, test_pattern, test_config):
        """Test conversion from pattern to quantum state."""
        quantum_state = bridge_components.dynamics._to_quantum_state(test_pattern)
        
        # Verify quantum state properties
        assert isinstance(quantum_state, QuantumState)
        assert quantum_state.amplitudes.dtype == torch.complex128
        assert torch.allclose(
            quantum_state.norm(),
            torch.ones(test_pattern.shape[0], dtype=torch.float64),
            atol=test_config["quantum_bridge_tests"]["test_tolerances"]["norm"]
        )
        
        # Verify phase preservation
        phase = torch.angle(quantum_state.amplitudes)
        assert torch.all(torch.isfinite(phase)), "Phase should be well-defined"
    
    def test_quantum_to_pattern(self, bridge_components, test_pattern, quantum_state, test_config):
        """Test conversion from quantum state back to classical pattern."""
        reconstructed = bridge_components.bridge.quantum_to_neural(quantum_state)
        
        # Verify shape and dtype
        assert reconstructed.shape == test_pattern.shape
        assert reconstructed.dtype == test_pattern.dtype
        
        # Verify reconstruction quality using absolute values for complex tensors
        cosine_sim = F.cosine_similarity(torch.abs(test_pattern), torch.abs(reconstructed))
        assert torch.all(cosine_sim > test_config["quantum_bridge_tests"]["test_tolerances"]["cosine_similarity"])
        
        # Verify norm preservation
        assert torch.allclose(
            torch.norm(reconstructed, dim=-1),
            torch.norm(test_pattern, dim=-1),
            atol=test_config["quantum_bridge_tests"]["test_tolerances"]["norm"]
        )


class TestQuantumEvolution:
    """Tests for quantum state evolution."""
    
    def test_attention_evolution(self, bridge_components, quantum_state, test_config):
        """Test quantum evolution with attention mechanism."""
        batch_size = test_config["quantum_bridge"]["batch_size"]
        hidden_dim = test_config["quantum_bridge"]["hidden_dim"]
        
        # Create attention pattern
        attention = torch.randn(batch_size, hidden_dim, hidden_dim, dtype=torch.complex128)
        attention = attention / torch.norm(attention, dim=(-2, -1), keepdim=True)
        
        # Evolve with attention
        evolved = bridge_components.bridge.evolve_quantum_state_with_attention(
            quantum_state,
            attention_pattern=attention,
            time=test_config["quantum_bridge_tests"]["evolution_time"]
        )
        
        # Verify quantum properties
        assert isinstance(evolved, QuantumState)
        assert evolved.amplitudes.dtype == torch.complex128
        assert torch.allclose(
            evolved.norm(),
            torch.ones(batch_size, dtype=torch.float64),
            atol=test_config["quantum_bridge_tests"]["test_tolerances"]["norm"]
        )
        
        # Verify geometric properties
        metric_tensor = bridge_components.bridge.pattern_bundle.riemannian_framework.compute_metric(evolved.amplitudes)
        assert_manifold_properties(
            metric_tensor.values,
            test_config["quantum_bridge_tests"]["test_tolerances"]["manifold"]
        )
        
        # Verify entanglement preservation
        initial_entropy = quantum_state.entropy()
        evolved_entropy = evolved.entropy()
        assert torch.allclose(
            initial_entropy,
            evolved_entropy,
            atol=test_config["quantum_bridge_tests"]["test_tolerances"]["entropy"]
        )
    
    def test_multi_head_evolution(self, bridge_components, test_config):
        """Test quantum evolution with multi-head attention."""
        # Use smaller dimensions for memory efficiency
        batch_size = 2
        num_heads = 4
        seq_len = 4
        hidden_dim = 32  # Reduced from test_config value
        dtype = getattr(torch, test_config["quantum_bridge"]["dtype"])
        
        # Create multi-head pattern
        pattern = torch.randn(batch_size, num_heads, seq_len, hidden_dim, dtype=dtype)
        pattern = F.normalize(pattern, p=2, dim=-1)
        
        # Create attention pattern - match state_dim which is hidden_dim//2 due to real/imag split
        state_dim = hidden_dim // 2
        
        # Create a simpler attention pattern that's just sequence-to-sequence
        attention = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.complex128)
        attention = attention / torch.norm(attention, dim=(-2, -1), keepdim=True)
        
        # Convert and evolve
        quantum_state = bridge_components.dynamics._to_quantum_state(pattern)
        evolved = bridge_components.bridge.evolve_quantum_state_with_attention(
            quantum_state,
            attention_pattern=attention,
            time=test_config["quantum_bridge_tests"]["evolution_time"]
        )
        
        # Verify per-head properties
        assert evolved.amplitudes.shape[:2] == (batch_size, num_heads)
        norms = evolved.norm()
        assert norms.shape == (batch_size, num_heads)
        assert torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=test_config["quantum_bridge_tests"]["test_tolerances"]["norm"]
        )
        
        # Verify geometric properties per head
        metric_tensor = bridge_components.bridge.pattern_bundle.riemannian_framework.compute_metric(evolved.amplitudes)
        assert_manifold_properties(
            metric_tensor.values,
            test_config["quantum_bridge_tests"]["test_tolerances"]["manifold"]
        )


class TestGeometricProperties:
    """Tests for geometric properties and transformations."""
    
    def test_scale_bridging(self, bridge_components, test_pattern, test_config):
        """Test bridging between different scales."""
        batch_size = test_config["quantum_bridge"]["batch_size"]
        source_scale = test_config["quantum_bridge_tests"]["source_scale"]
        target_scale = test_config["quantum_bridge_tests"]["target_scale"]
        
        scaled = bridge_components.bridge.bridge_scales(
            state=test_pattern,
            source_scale=source_scale,
            target_scale=target_scale
        )
        
        # Verify scale transition
        assert scaled.shape == test_pattern.shape
        assert torch.allclose(
            torch.norm(scaled, dim=-1),
            torch.ones(batch_size, dtype=torch.float64) * target_scale / source_scale,
            atol=float(test_config["quantum_bridge_tests"]["test_tolerances"]["scale"])
        )
        
        # Verify geometric structure preservation
        metric_tensor = bridge_components.bridge.pattern_bundle.riemannian_framework.compute_metric(scaled)
        assert_manifold_properties(
            metric_tensor.values,
            float(test_config["quantum_bridge_tests"]["test_tolerances"]["manifold"])
        )
    
    def test_pattern_bundle_gradient(self, bridge_components, test_pattern, test_config):
        """Test gradient flow through pattern bundle."""
        test_pattern.requires_grad_(True)
        
        # Forward pass
        quantum_state = bridge_components.bridge.neural_to_quantum(test_pattern)
        evolved = bridge_components.bridge.evolve_quantum_state_with_attention(
            quantum_state,
            time=test_config["quantum_bridge_tests"]["evolution_time"]
        )
        evolved_pattern = bridge_components.bridge.quantum_to_neural(evolved)
        
        # Backward pass - use real part for loss computation
        loss = evolved_pattern.real.sum()  # Only use real part for loss
        loss.backward()
        
        # Verify gradients
        assert test_pattern.grad is not None, "Pattern should have gradients"
        assert bridge_components.bridge.pattern_bundle.metric.grad is not None, "Metric should have gradients"
        assert bridge_components.bridge.pattern_bundle.connection.grad is not None, "Connection should have gradients"
        
        # Verify gradient properties
        assert torch.all(torch.isfinite(test_pattern.grad)), "Pattern gradients should be finite"
        assert torch.all(torch.isfinite(bridge_components.bridge.pattern_bundle.metric.grad)), "Metric gradients should be finite"
        assert torch.all(torch.isfinite(bridge_components.bridge.pattern_bundle.connection.grad)), "Connection gradients should be finite"


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_invalid_inputs(self, bridge_components, test_config):
        """Test handling of invalid inputs."""
        hidden_dim = test_config["quantum_bridge"]["hidden_dim"]
        dtype = getattr(torch, test_config["quantum_bridge"]["dtype"])
        
        # Test invalid dimensions
        with pytest.raises(ValueError, match="dimension"):
            invalid_pattern = torch.randn(8, hidden_dim * 2, dtype=dtype)
            bridge_components.bridge.neural_to_quantum(invalid_pattern)
        
        # Test invalid scale factors
        with pytest.raises(ValueError, match="must be positive"):
            pattern = torch.randn(2, hidden_dim, dtype=dtype)
            bridge_components.bridge.bridge_scales(pattern, source_scale=-1.0, target_scale=1.0)
        
        # Test non-finite values
        with pytest.raises(ValueError, match="non-finite"):
            pattern = torch.full((2, hidden_dim), float('inf'), dtype=dtype)
            bridge_components.bridge.neural_to_quantum(pattern)
    
    def test_edge_cases(self, bridge_components, test_config):
        """Test handling of edge cases."""
        hidden_dim = test_config["quantum_bridge"]["hidden_dim"]
        dtype = getattr(torch, test_config["quantum_bridge"]["dtype"])
        
        # Test zero-norm pattern
        with pytest.raises(ValueError, match="zero"):
            zero_pattern = torch.zeros(2, hidden_dim, dtype=dtype)
            bridge_components.bridge.neural_to_quantum(zero_pattern)
        
        # Test tiny values
        tiny_pattern = torch.full((2, hidden_dim), 1e-10, dtype=dtype)
        tiny_pattern = F.normalize(tiny_pattern, dim=-1)
        quantum_state = bridge_components.bridge.neural_to_quantum(tiny_pattern)
        assert isinstance(quantum_state, QuantumState), "Should handle tiny values"
  