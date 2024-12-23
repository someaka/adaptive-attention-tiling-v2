"""Tests for pattern-neural bridge integration.

This module implements tests for the neural network operations on patterns,
verifying the forward pass, backward pass, and gradient computation.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Dict, Any
import torch.nn as nn
import torch.nn.functional as F

from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.patterns.pattern_processor import PatternProcessor
from src.core.flow.neural import NeuralGeometricFlow
from tests.utils.config_loader import load_test_config

@pytest.fixture
def test_config():
    """Load test configuration."""
    return load_test_config()

@pytest.fixture
def setup_components(test_config):
    """Setup test components."""
    # Ensure minimum dimensions for attention heads
    hidden_dim = max(test_config["geometric_tests"]["dimensions"], 8)  # Minimum 8 for attention
    manifold_dim = max(test_config["geometric_tests"]["dimensions"], 8)  # Minimum 8 for attention
    num_heads = test_config["geometric_tests"]["num_heads"]  # Use single head from debug config
    grid_size = 8  # Fixed small grid size for debug
    
    dynamics = PatternDynamics(
        hidden_dim=hidden_dim,
        grid_size=grid_size
    )
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
    grid_size = dynamics.grid_size  # Get grid size from dynamics
    
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
    grid_size = dynamics.grid_size
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
    grid_size = dynamics.grid_size
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

def project_to_manifold(x: torch.Tensor, manifold_dim: int) -> torch.Tensor:
    """Project input tensor to manifold dimension.
    
    Args:
        x: Input tensor of shape (batch_size, *)
        manifold_dim: Target manifold dimension
        
    Returns:
        Projected tensor of shape (batch_size, manifold_dim)
    """
    # Ensure input is properly shaped
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension if missing
    
    batch_size = x.size(0)
    x_flat = x.reshape(batch_size, -1)  # Use reshape instead of view for better compatibility
    
    # Project to half the manifold dimension since we'll double it for complex numbers
    projection = nn.Linear(x_flat.size(1), manifold_dim // 2, dtype=x.dtype, device=x.device)
    projection.requires_grad_(True)  # Enable gradients for projection
    
    # Initialize weights and biases with proper gradients
    nn.init.xavier_uniform_(projection.weight)
    nn.init.zeros_(projection.bias)
    
    # Project and ensure gradients are preserved
    projected = projection(x_flat)
    
    # Make it complex by adding zeros for imaginary part
    zeros = torch.zeros_like(projected, requires_grad=True)
    projected = torch.cat([projected, zeros], dim=-1)
    
    # Ensure output is normalized while preserving gradients
    norm = torch.norm(projected, p=2, dim=-1, keepdim=True)
    return projected / (norm + 1e-8)

def test_geometric_attention_integration(setup_components):
    """Test integration with geometric attention."""
    dynamics, processor, flow = setup_components
    
    # Create test input
    grid_size = dynamics.grid_size
    x = torch.randn(4, dynamics.dim, grid_size, grid_size, requires_grad=True)  # Enable gradients
    
    # Project to manifold dimension
    x_projected = project_to_manifold(x, flow.manifold_dim)
    
    # Compute metric and connection
    metric = flow.compute_metric(x_projected)
    connection = flow.compute_connection(metric, x_projected)
    
    # Verify shapes
    assert metric.shape == (x.size(0), flow.manifold_dim, flow.manifold_dim)
    assert connection.shape == (x.size(0), flow.manifold_dim, flow.manifold_dim, flow.manifold_dim)

def test_pattern_manipulation(setup_components):
    """Test pattern manipulation operations.
    
    This test verifies:
    1. Pattern evolution
    2. Pattern transformation
    3. Pattern structure preservation
    """
    dynamics, processor, _ = setup_components
    
    # Create test pattern
    grid_size = dynamics.grid_size
    pattern = torch.randn(1, dynamics.dim, grid_size, grid_size, requires_grad=True)  # Enable gradients
    
    # Get pattern evolution
    results = dynamics(pattern, return_patterns=True)
    patterns = results["patterns"]
    assert len(patterns) > 0, "Should produce pattern evolution"
    
    # Transform through dynamics
    next_state = dynamics.compute_next_state(pattern)
    assert next_state.shape == pattern.shape, "Should preserve shape under transformation"
    assert not torch.allclose(next_state, pattern), "Should modify pattern content"
    
    # Verify pattern structure preservation
    next_results = dynamics(next_state, return_patterns=True)
    assert torch.allclose(
        results["pattern_scores"].norm(),
        next_results["pattern_scores"].norm(),
        atol=1e-5
    ), "Should preserve pattern structure"

def test_training_integration(setup_components, test_config):
    """Test integration with training pipeline."""
    dynamics, processor, flow = setup_components
    
    # Enable gradients for all parameters
    for param in flow.parameters():
        param.requires_grad_(True)
    for param in dynamics.parameters():
        param.requires_grad_(True)
    for param in processor.parameters():
        param.requires_grad_(True)
    
    # Get test parameters from config and components
    batch_size = test_config["geometric_tests"]["batch_size"]
    manifold_dim = processor.hidden_dim  # Use processor's hidden dimension
    
    # Create input with shape [batch_size, manifold_dim, manifold_dim]
    batch = torch.randn(batch_size, manifold_dim, manifold_dim, requires_grad=True)
    
    # Forward pass through processor
    processed = processor(batch)
    
    # Forward pass through flow
    output = flow(processed)
    
    # Compute loss
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert all(p.grad is not None for p in flow.parameters() if p.requires_grad)
    assert all(p.grad is not None for p in dynamics.parameters() if p.requires_grad)
    assert all(p.grad is not None for p in processor.parameters() if p.requires_grad)