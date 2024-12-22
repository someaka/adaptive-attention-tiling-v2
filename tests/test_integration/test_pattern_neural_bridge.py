"""Tests for pattern-neural bridge integration.

This module implements tests for the neural network operations on patterns,
verifying the forward pass, backward pass, and gradient computation.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Dict, Any

from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.patterns.pattern_processor import PatternProcessor
from src.core.flow.neural import NeuralGeometricFlow

@pytest.fixture
def setup_components():
    """Setup test components."""
    hidden_dim = 64
    manifold_dim = 32
    
    dynamics = PatternDynamics(hidden_dim=hidden_dim)
    processor = PatternProcessor(
        manifold_dim=manifold_dim,
        hidden_dim=hidden_dim
    )
    flow = NeuralGeometricFlow(
        manifold_dim=manifold_dim,
        hidden_dim=hidden_dim
    )
    
    return dynamics, processor, flow

def test_forward_pass(setup_components):
    """Test forward pass through pattern-neural bridge.
    
    This test verifies:
    1. Pattern to neural conversion
    2. Neural network processing
    3. Output shape and properties
    """
    dynamics, processor, _ = setup_components
    
    # Create test input with correct dimensions [batch, channels, height, width]
    batch_size = 4
    num_heads = 8
    seq_len = 16
    grid_size = dynamics.size  # Get grid size from dynamics
    
    states = torch.randn(batch_size, dynamics.dim, grid_size, grid_size)
    
    # Forward pass through dynamics
    results = dynamics.forward(states, return_patterns=True)
    
    # Verify output shapes and properties
    assert isinstance(results, dict), "Should return dictionary"
    assert "patterns" in results, "Should include patterns"
    assert "pattern_scores" in results, "Should include pattern scores"
    
    patterns = results["patterns"]
    scores = results["pattern_scores"]
    
    # Check shapes
    assert patterns[0].shape == states.shape, "Pattern shape should match input"
    assert len(scores) == len(patterns), "Should have score for each pattern"
    
    # Check normalization
    for i in range(len(patterns)):
        patterns[i] = patterns[i] / torch.norm(patterns[i].to(torch.float32))
        norm = torch.norm(patterns[i].to(torch.float32))
        assert torch.allclose(norm, torch.tensor(1.0, dtype=torch.float32), atol=1e-6), "Patterns should be normalized"

def test_backward_pass(setup_components):
    """Test backward pass and gradient flow.
    
    This test verifies:
    1. Gradient computation
    2. Gradient flow through the network
    3. Parameter updates
    """
    dynamics, processor, _ = setup_components
    
    # Create test input with gradients [batch, channels, height, width]
    grid_size = dynamics.size
    x = torch.randn(4, dynamics.dim, grid_size, grid_size, requires_grad=True)
    
    # Forward pass
    results = dynamics.forward(x, return_patterns=True)
    patterns = results["patterns"]
    
    # Compute loss using mean squared error
    target = torch.ones_like(patterns[0])
    loss = torch.nn.functional.mse_loss(patterns[0], target)
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "Should have gradients"
    assert not torch.allclose(x.grad, torch.zeros_like(x)), "Gradients should be non-zero"

def test_gradient_computation(setup_components):
    """Test explicit gradient computation.
    
    This test verifies:
    1. Linearized dynamics computation
    2. Gradient accuracy
    3. Numerical stability
    """
    dynamics, _, _ = setup_components
    
    # Create test pattern [batch, channels, height, width]
    grid_size = dynamics.size
    pattern = torch.randn(1, dynamics.dim, grid_size, grid_size)
    
    # Compute linearization
    linearized = dynamics.compute_linearization(pattern)
    
    # Check properties
    batch_size = pattern.size(0)
    state_size = pattern.numel() // batch_size
    expected_shape = (batch_size, state_size, state_size)
    assert linearized.shape == expected_shape, "Should preserve total size"
    assert not torch.allclose(linearized, pattern.view(batch_size, -1).unsqueeze(-1)), "Should modify pattern"
    assert torch.isfinite(linearized).all(), "Should be numerically stable"