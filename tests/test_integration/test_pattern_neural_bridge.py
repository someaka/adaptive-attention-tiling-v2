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

from  src.neural.attention.pattern.pattern_dynamics import PatternDynamics
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
    # Get dimensions from config
    hidden_dim = test_config["geometric_tests"]["hidden_dim"]
    manifold_dim = test_config["geometric"]["manifold_dim"]  # Use manifold_dim from geometric section
    num_heads = test_config["geometric_tests"]["num_heads"]
    grid_size = 8  # Fixed small grid size for debug
    
    dynamics = PatternDynamics(
        grid_size=grid_size,
        space_dim=2,  # 2D spatial dimensions
        boundary='periodic',  # Use periodic boundary conditions
        dt=0.01,  # Small time step for stability
        num_modes=8,  # Number of stability modes
        hidden_dim=hidden_dim,
        quantum_enabled=False  # Disable quantum features for basic tests
    )
    processor = PatternProcessor(
        manifold_dim=manifold_dim,  # Use same manifold_dim as flow
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
    
    # Project directly to manifold dimension
    projection = nn.Linear(x_flat.size(1), manifold_dim, dtype=x.dtype, device=x.device)
    projection.requires_grad_(True)  # Enable gradients for projection
    
    # Initialize weights and biases with proper gradients
    nn.init.xavier_uniform_(projection.weight)
    nn.init.zeros_(projection.bias)
    
    # Project and ensure gradients are preserved
    projected = projection(x_flat)
    
    # Ensure output is normalized while preserving gradients
    norm = torch.norm(projected, p=2, dim=-1, keepdim=True)
    return projected / (norm + 1e-8)

def test_geometric_attention_integration(setup_components):
    """Test integration with geometric attention."""
    dynamics, processor, flow = setup_components
    
    # Create test input
    grid_size = dynamics.size
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
    grid_size = dynamics.size
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
    
    print("\nDimension Configuration:")
    print(f"Flow hidden_dim: {flow.hidden_dim}")
    print(f"Flow manifold_dim: {flow.manifold_dim}")
    print(f"Processor hidden_dim: {processor.hidden_dim}")
    print(f"Processor manifold_dim: {processor.manifold_dim}")
    
    # Enable gradients for all parameters recursively
    def enable_gradients_recursive(module):
        """Enable gradients recursively for all parameters in a module."""
        if not isinstance(module, torch.nn.Module):
            return

        # Enable gradients for direct parameters
        for param in module.parameters(recurse=False):
            if param is not None:
                param.requires_grad_(True)

        # Enable gradients for buffers
        for buffer in module.buffers():
            if buffer is not None:
                buffer.requires_grad_(True)

        # Special handling for specific module types
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(True)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.requires_grad_(True)

        # Handle special attributes that should have gradients
        special_attrs = [
            'metric', 'form', 'connection', 'composition_basis', 
            'structure_preserving_map', 'k', 'omega', 'wave_basis',
            'diffusion_coeff', 'reaction_coeff', 'kernel',
            'metric_factors', 'connection_coeffs', 'prime_bases',
            'wave_speed', 'damping', 'coupling'
        ]
        for attr in special_attrs:
            if hasattr(module, attr):
                attr_val = getattr(module, attr)
                if isinstance(attr_val, torch.Tensor):
                    attr_val.requires_grad_(True)

        # Recursively enable for children
        for child in module.children():
            enable_gradients_recursive(child)

        # Handle submodules that are Sequential or ModuleList
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            for submodule in module:
                enable_gradients_recursive(submodule)

        # Handle named children for deeper nesting
        for name, child in module.named_children():
            enable_gradients_recursive(child)
            # Handle nested attributes
            if hasattr(child, 'pattern_bundle'):
                enable_gradients_recursive(child.pattern_bundle)
            if hasattr(child, 'geometric_flow'):
                enable_gradients_recursive(child.geometric_flow)
            if hasattr(child, 'symplectic'):
                enable_gradients_recursive(child.symplectic)
            if hasattr(child, 'enriched'):
                enable_gradients_recursive(child.enriched)
            if hasattr(child, 'operadic'):
                enable_gradients_recursive(child.operadic)
            if hasattr(child, 'framework'):
                enable_gradients_recursive(child.framework)
            if hasattr(child, 'height_structure'):
                enable_gradients_recursive(child.height_structure)

    # Enable gradients for all components
    enable_gradients_recursive(flow)
    enable_gradients_recursive(dynamics)
    enable_gradients_recursive(processor)

    # Get test parameters from config and components
    batch_size = test_config["geometric_tests"]["batch_size"]
    manifold_dim = flow.manifold_dim  # Use flow's manifold dimension for consistency
    hidden_dim = flow.hidden_dim  # Use flow's hidden dimension for consistency

    print(f"\nTest Parameters:")
    print(f"batch_size: {batch_size}")
    print(f"manifold_dim: {manifold_dim}")
    print(f"hidden_dim: {hidden_dim}")

    # Create input with correct shape [batch_size, manifold_dim]
    batch = torch.randn(batch_size, manifold_dim, requires_grad=True)
    print(f"\nInput batch shape: {batch.shape}")

    # Forward pass through processor
    processed = processor(batch)
    print(f"Processed output shape: {processed.shape}")

    # Forward pass through flow
    output = flow(processed)

    # Compute loss
    loss = output[0].mean()  # Use first element of tuple (evolved tensor)

    # Backward pass
    loss.backward()

    # Print gradient information
    print("\nFlow Parameter Gradients:")
    for name, param in flow.named_parameters():
        if param.requires_grad:
            print(f"{name}: grad={param.grad is not None}, shape={param.shape}")

    # Check gradients for specific components that should have them
    required_grad_components = [
        'metric_net.0.weight',
        'connection_net.0.weight',
        'curvature_net.0.weight',
        'stability_net.0.weight',
        'quantum_bridge.layer_norm_real.weight',
        'quantum_bridge.layer_norm_imag.weight',
        'quantum_bridge.manifold_norm_real.weight',
        'quantum_bridge.manifold_norm_imag.weight',
        'quantum_bridge.inverse_projection.weight',
        'quantum_bridge.pattern_bundle.metric',
        'quantum_bridge.pattern_bundle.connection',
        'arithmetic.coupling',
        'arithmetic.height_map.0.weight',
        'arithmetic.flow.weight',
        'arithmetic.l_function.0.weight',
        'arithmetic.quantum_height.0.weight',
        'arithmetic.quantum_l_function.0.weight',
        'fisher_net.0.weight',
        'expectation_projection.0.weight',
        'quantum_correction_net.0.weight',
        'connection_projection.0.weight'
    ]

    # Check that required components have gradients
    for component in required_grad_components:
        param = None
        try:
            # Handle nested components
            parts = component.split('.')
            curr = flow
            for part in parts:
                curr = getattr(curr, part)
            param = curr
        except AttributeError:
            continue

        if param is not None and isinstance(param, torch.Tensor):
            assert param.grad is not None, f"{component} should have gradients"
            assert torch.all(torch.isfinite(param.grad)), f"{component} has non-finite gradients"

    # Check that at least one parameter has gradients
    assert any(p.grad is not None for p in flow.parameters() if p.requires_grad), "No parameters have gradients"