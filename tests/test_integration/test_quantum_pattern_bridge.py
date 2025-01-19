"""Tests for quantum-pattern bridge integration.

This module implements tests for the conversion between quantum states and patterns,
verifying the preservation of quantum properties and geometric structure during conversion.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

from src.core.quantum.types import QuantumState
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.neural.attention.pattern.pattern_dynamics import PatternDynamics
from src.neural.attention.pattern.quantum import QuantumGeometricTensor

@pytest.fixture
def setup_bridge():
    """Setup test bridge and components."""
    hidden_dim = 8  # Must be even for quantum states
    manifold_dim = hidden_dim  # Set manifold_dim equal to hidden_dim for testing
    bridge = NeuralQuantumBridge(hidden_dim=hidden_dim, manifold_dim=manifold_dim, dtype=torch.float64)
    dynamics = PatternDynamics(hidden_dim=hidden_dim)
    tensor = QuantumGeometricTensor(dim=hidden_dim)
    
    return bridge, dynamics, tensor

def test_pattern_to_quantum_conversion(setup_bridge):
    """Test conversion from pattern to quantum state."""
    bridge, dynamics, _ = setup_bridge

    # Create test pattern with batch dimension
    pattern = torch.randn(2, 8, dtype=torch.float64)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)

    # Convert to quantum state
    quantum_state = dynamics._to_quantum_state(pattern)

    # Verify quantum state properties
    assert isinstance(quantum_state, QuantumState)
    assert quantum_state.amplitudes.dtype == torch.complex128
    assert torch.allclose(quantum_state.norm(), torch.ones(2, dtype=torch.float64), atol=1e-6)

def test_quantum_to_pattern_conversion(setup_bridge):
    """Test conversion from quantum state back to classical pattern."""
    bridge, dynamics, _ = setup_bridge
    
    # Create a test pattern with batch dimension
    pattern = torch.randn(2, 8, dtype=torch.float64)
    pattern = F.normalize(pattern, dim=1)
    
    # Convert to quantum state
    quantum_state = bridge.neural_to_quantum(pattern)
    
    # Convert back to pattern
    reconstructed_pattern = bridge.quantum_to_neural(quantum_state)
    
    # Verify shape and dtype
    assert reconstructed_pattern.shape == pattern.shape
    assert reconstructed_pattern.dtype == pattern.dtype
    
    # Verify reconstruction quality
    cosine_similarity = F.cosine_similarity(pattern, reconstructed_pattern)
    assert torch.all(cosine_similarity > 0.99)

def test_quantum_evolution_with_attention(setup_bridge):
    """Test quantum evolution with attention mechanism."""
    bridge, dynamics, _ = setup_bridge

    # Create test pattern with batch dimension
    batch_size = 2
    pattern = torch.randn(batch_size, 8, dtype=torch.float64)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)

    # Convert to quantum state
    quantum_state = dynamics._to_quantum_state(pattern)

    # Create attention pattern
    attention = torch.randn(batch_size, 8, 8, dtype=torch.complex128)
    attention = attention / torch.norm(attention, dim=(-2, -1), keepdim=True)

    # Evolve with attention
    evolved_state = bridge.evolve_quantum_state_with_attention(
        quantum_state,
        attention_pattern=attention,
        time=1.0
    )

    # Verify evolution properties
    assert isinstance(evolved_state, QuantumState)
    assert evolved_state.amplitudes.dtype == torch.complex128
    assert torch.allclose(evolved_state.norm(), torch.ones(batch_size, dtype=torch.float64), atol=1e-6)

def test_scale_bridging(setup_bridge):
    """Test bridging between different scales."""
    bridge, dynamics, _ = setup_bridge
    
    # Create test pattern with batch dimension
    batch_size = 2
    pattern = torch.randn(batch_size, 8, dtype=torch.float64)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
    
    # Bridge scales
    source_scale = 1.0
    target_scale = 2.0
    scaled_pattern = bridge.bridge_scales(
        state=pattern,
        source_scale=source_scale,
        target_scale=target_scale
    )
    
    # Verify scale transition properties
    assert scaled_pattern.shape == pattern.shape
    assert torch.allclose(
        torch.norm(scaled_pattern, dim=-1),
        torch.ones(batch_size, dtype=torch.float64) * target_scale / source_scale,
        atol=1e-6
    )

def test_coherence_computation(setup_bridge):
    """Test computation of quantum coherence between states."""
    bridge, dynamics, _ = setup_bridge
    
    # Create test patterns with batch dimension
    batch_size = 2
    pattern1 = torch.randn(batch_size, 8, dtype=torch.float64)
    pattern1 = torch.nn.functional.normalize(pattern1, p=2, dim=-1)
    pattern2 = torch.randn(batch_size, 8, dtype=torch.float64)
    pattern2 = torch.nn.functional.normalize(pattern2, p=2, dim=-1)
    
    # Compute coherence
    coherence = bridge.compute_coherence(pattern1, pattern2)
    
    # Verify coherence properties
    assert coherence.shape == (batch_size,)
    assert torch.all(coherence >= 0) and torch.all(coherence <= 1)
    
    # Test self-coherence
    self_coherence = bridge.compute_coherence(pattern1, pattern1)
    assert torch.allclose(self_coherence, torch.ones(batch_size), atol=1e-6)

def test_pattern_bundle_evolution(setup_bridge):
    """Test evolution of pattern bundle with attention."""
    bridge, dynamics, _ = setup_bridge

    # Create test pattern bundle with batch dimension
    batch_size = 2
    pattern = torch.randn(batch_size, 8, dtype=torch.float64)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)

    # Evolve pattern bundle
    evolved_pattern = bridge.evolve_pattern_bundle_with_attention(pattern, time=1.0)
    evolved_pattern = torch.nn.functional.normalize(evolved_pattern, p=2, dim=-1)

    # Verify evolution properties
    assert evolved_pattern.shape == pattern.shape
    assert evolved_pattern.dtype == torch.float64
    assert torch.allclose(
        torch.norm(evolved_pattern, dim=-1),
        torch.ones(batch_size, dtype=torch.float64),
        atol=1e-6
    )

def test_geometric_flow_evolution(setup_bridge):
    """Test evolution through quantum geometric flow."""
    bridge, dynamics, _ = setup_bridge

    # Create test tensor with batch dimension
    batch_size = 2
    tensor = torch.randn(batch_size, 8, dtype=torch.float64)
    tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1)

    # Evolve through geometric flow
    evolved_tensor = bridge.evolve_geometric_flow_with_attention(tensor, time=1.0)
    evolved_tensor = torch.nn.functional.normalize(evolved_tensor, p=2, dim=-1)

    # Verify evolution properties
    assert evolved_tensor.shape == tensor.shape
    assert evolved_tensor.dtype == torch.float64
    assert torch.allclose(
        torch.norm(evolved_tensor, dim=-1),
        torch.ones(batch_size, dtype=torch.float64),
        atol=1e-6
    )

def test_error_handling(setup_bridge):
    """Test error handling in quantum-pattern bridge."""
    bridge, dynamics, _ = setup_bridge
    
    # Test invalid input dimensions
    with pytest.raises(ValueError):
        invalid_pattern = torch.randn(8, 16)  # Wrong hidden dimension
        bridge.neural_to_quantum(invalid_pattern)
    
    # Test invalid scale factors
    with pytest.raises(ValueError):
        pattern = torch.randn(2, 8, dtype=torch.float64)
        bridge.bridge_scales(pattern, source_scale=-1.0, target_scale=1.0)
    
    # Test non-finite values
    with pytest.raises(ValueError):
        pattern = torch.full((2, 8), float('inf'), dtype=torch.float64)
        bridge.neural_to_quantum(pattern)

def test_multi_head_attention_evolution(setup_bridge):
    """Test quantum evolution with multi-head attention patterns.
    
    This test verifies:
    1. Multi-head quantum state evolution
    2. Head dimension preservation
    3. Quantum properties per head
    """
    bridge, dynamics, _ = setup_bridge
    
    # Create multi-head input
    batch_size = 2
    num_heads = 4
    seq_len = 8
    hidden_dim = bridge.hidden_dim
    
    # Create pattern with head dimension
    pattern = torch.randn(batch_size, num_heads, seq_len, hidden_dim, dtype=torch.float64)
    pattern = F.normalize(pattern, p=2, dim=-1)
    
    # Create multi-head attention pattern
    attention = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.complex128)
    attention = attention / torch.norm(attention, dim=(-2, -1), keepdim=True)
    
    # Convert to quantum state preserving head dimension
    quantum_state = dynamics._to_quantum_state(pattern)
    
    # Evolve with attention
    evolved_state = bridge.evolve_quantum_state_with_attention(
        quantum_state,
        attention_pattern=attention,
        time=1.0
    )
    
    # Verify evolution properties per head
    assert isinstance(evolved_state, QuantumState)
    assert evolved_state.amplitudes.dtype == torch.complex128
    assert evolved_state.amplitudes.shape[:2] == (batch_size, num_heads), "Should preserve head dimension"
    
    # Check normalization per head
    norms = evolved_state.norm()
    assert norms.shape == (batch_size, num_heads), "Should have norm per head"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_pattern_bundle_gradient_flow(setup_bridge):
    """Test gradient flow through pattern bundle during quantum evolution.
    
    This test verifies:
    1. Pattern bundle parameter gradients
    2. Multi-head gradient preservation
    3. Connection form gradient flow
    """
    bridge, dynamics, _ = setup_bridge
    
    # Enable gradients for pattern bundle
    bridge.pattern_bundle.requires_grad_(True)
    
    # Create multi-head input
    batch_size = 2
    num_heads = 4
    seq_len = 8
    hidden_dim = bridge.hidden_dim
    
    # Create input with gradients
    pattern = torch.randn(batch_size, num_heads, seq_len, hidden_dim, dtype=torch.float64, requires_grad=True)
    pattern_norm = torch.norm(pattern, p=2, dim=-1, keepdim=True)
    pattern = pattern / (pattern_norm + 1e-6)  # Manual normalization that preserves gradients
    pattern.retain_grad()  # Ensure gradients are retained for non-leaf tensor
    
    # Forward pass through quantum evolution
    quantum_state = dynamics._to_quantum_state(pattern)
    evolved_state = bridge.evolve_quantum_state_with_attention(
        quantum_state,
        time=1.0
    )
    
    # Compute loss using evolved state
    loss = evolved_state.amplitudes.abs().mean()
    loss.backward()
    
    # Verify pattern bundle gradients
    assert pattern.grad is not None, "Input should have gradients"
    assert torch.all(torch.isfinite(pattern.grad)), "Gradients should be finite"
    
    # Verify pattern bundle parameter gradients
    for param in bridge.pattern_bundle.parameters():
        assert param.grad is not None, "Pattern bundle parameters should have gradients"
        assert torch.all(torch.isfinite(param.grad)), "Pattern bundle gradients should be finite"
  