"""Tests for quantum-pattern bridge integration.

This module implements tests for the conversion between quantum states and patterns,
verifying the preservation of quantum properties and geometric structure during conversion.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Dict, Any

from src.core.quantum.types import QuantumState
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.attention.pattern.quantum import QuantumGeometricTensor

@pytest.fixture
def setup_bridge():
    """Setup test bridge and components."""
    hidden_dim = 8  # Must be even for quantum states
    bridge = NeuralQuantumBridge(hidden_dim=hidden_dim)
    dynamics = PatternDynamics(hidden_dim=hidden_dim)
    tensor = QuantumGeometricTensor(dim=hidden_dim)
    
    return bridge, dynamics, tensor

def test_pattern_to_quantum_conversion(setup_bridge):
    """Test conversion from classical pattern to quantum state."""
    bridge, dynamics, _ = setup_bridge
    
    # Create test pattern
    pattern = torch.randn(8, dtype=torch.float32)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
    
    # Convert to quantum state
    quantum_state = dynamics._to_quantum_state(pattern)
    
    # Verify quantum state properties
    assert isinstance(quantum_state, QuantumState)
    assert torch.allclose(quantum_state.norm(), torch.tensor(1.0), atol=1e-6)
    assert len(quantum_state.basis_labels) == pattern.shape[-1]
    assert quantum_state.amplitudes.shape == pattern.shape
    
    # Verify phase information
    assert quantum_state.phase is not None
    assert quantum_state.phase.shape == pattern.shape

def test_quantum_to_pattern_conversion(setup_bridge):
    """Test conversion from quantum state to classical pattern."""
    bridge, dynamics, _ = setup_bridge
    
    # Create test quantum state
    amplitudes = torch.randn(8, dtype=torch.complex64)
    amplitudes = amplitudes / torch.norm(amplitudes)
    phase = torch.angle(amplitudes)
    basis_labels = [f"basis_{i}" for i in range(8)]
    
    quantum_state = QuantumState(
        amplitudes=amplitudes,
        basis_labels=basis_labels,
        phase=phase
    )
    
    # Convert to classical pattern
    pattern = dynamics._from_quantum_state(quantum_state)
    
    # Verify pattern properties
    assert isinstance(pattern, torch.Tensor)
    assert pattern.dtype == torch.float32
    assert pattern.shape == amplitudes.shape
    assert torch.allclose(torch.norm(pattern), torch.tensor(1.0), atol=1e-6)

def test_fidelity_preservation(setup_bridge):
    """Test preservation of fidelity during conversion cycle."""
    bridge, dynamics, _ = setup_bridge
    
    # Create initial pattern
    pattern = torch.randn(8, dtype=torch.float32)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
    
    # Convert to quantum and back
    quantum_state = dynamics._to_quantum_state(pattern)
    reconstructed = dynamics._from_quantum_state(quantum_state)
    
    # Compute fidelity
    fidelity = torch.abs(torch.dot(pattern, reconstructed))
    
    # Verify high fidelity
    assert fidelity > 0.99, f"Low fidelity: {fidelity}"

def test_quantum_evolution_consistency(setup_bridge):
    """Test consistency of quantum evolution with pattern dynamics."""
    bridge, dynamics, tensor = setup_bridge
    
    # Create test pattern
    pattern = torch.randn(8, dtype=torch.float32)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
    
    # Evolve through quantum channel
    quantum_state = dynamics._to_quantum_state(pattern)
    evolved_state = bridge.evolve_quantum_state(quantum_state)
    evolved_pattern = dynamics._from_quantum_state(evolved_state)
    
    # Verify evolution properties
    assert torch.allclose(torch.norm(evolved_pattern), torch.tensor(1.0), atol=1e-6)
    assert evolved_pattern.shape == pattern.shape
    
    # Verify quantum geometric properties are preserved
    Q1 = tensor.compute_tensor(quantum_state)
    Q2 = tensor.compute_tensor(evolved_state)
    
    # Check metric preservation
    g1 = Q1.real
    g2 = Q2.real
    assert torch.allclose(torch.trace(g1), torch.trace(g2), atol=1e-5)
    
    # Check symplectic preservation
    omega1 = Q1.imag
    omega2 = Q2.imag
    assert torch.allclose(torch.trace(omega1), torch.trace(omega2), atol=1e-5)

def test_scale_transition(setup_bridge):
    """Test quantum-pattern bridge across scale transitions."""
    bridge, dynamics, _ = setup_bridge
    
    # Create test pattern
    pattern = torch.randn(8, dtype=torch.float32)
    pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
    
    # Define scales
    source_scale = 1.0
    target_scale = 2.0
    
    # Bridge scales through quantum channel
    scaled_pattern = bridge.bridge_scales(
        state=pattern,
        source_scale=source_scale,
        target_scale=target_scale
    )
    
    # Verify scale transition properties
    assert scaled_pattern.shape == pattern.shape
    assert torch.allclose(torch.norm(scaled_pattern), torch.tensor(1.0), atol=1e-6)
    
    # Verify scale consistency
    scale_ratio = target_scale / source_scale
    pattern_ratio = torch.norm(scaled_pattern) / torch.norm(pattern)
    assert abs(scale_ratio - pattern_ratio) < 1e-6

def test_error_bounds(setup_bridge):
    """Test error bounds in quantum-pattern conversion."""
    bridge, dynamics, _ = setup_bridge
    
    # Create test patterns with noise
    n_samples = 100
    patterns = []
    errors = []
    
    for _ in range(n_samples):
        # Create noisy pattern
        pattern = torch.randn(8, dtype=torch.float32)
        pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
        noise = torch.randn_like(pattern) * 0.01
        noisy_pattern = torch.nn.functional.normalize(pattern + noise, p=2, dim=-1)
        
        # Convert through quantum channel
        quantum_state = dynamics._to_quantum_state(noisy_pattern)
        reconstructed = dynamics._from_quantum_state(quantum_state)
        
        # Compute error
        error = torch.norm(reconstructed - pattern)
        
        patterns.append(pattern)
        errors.append(error.item())
    
    # Verify error bounds
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    assert mean_error < 0.1, f"High mean error: {mean_error}"
    assert std_error < 0.05, f"High error variance: {std_error}" 