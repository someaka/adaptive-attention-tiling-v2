"""Tests for the neural quantum bridge."""

import pytest
import torch
import numpy as np

from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.core.quantum.types import QuantumState


@pytest.fixture
def bridge():
    """Create test bridge."""
    return NeuralQuantumBridge(
        hidden_dim=64,
        num_heads=8,
        dropout=0.1
    )


@pytest.fixture
def test_state():
    """Create test neural state."""
    return torch.randn(8, 64)  # [batch_size, hidden_dim]


class TestNeuralQuantumBridge:
    """Tests for NeuralQuantumBridge."""
    
    def test_forward_pass_and_gradients(self, bridge, test_state):
        """Test forward pass and gradient flow through the bridge."""
        # Enable gradient tracking
        x = test_state.clone().requires_grad_(True)
        
        # Forward pass
        output = bridge(x)
        
        # Check output properties
        assert output.shape == x.shape, "Output shape should match input"
        assert torch.all(torch.isfinite(output)), "Output should be finite"
        
        # Test gradient flow
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradient properties
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not be NaN"
        assert not torch.isinf(x.grad).any(), "Input gradients should not be Inf"
        
        # Check layer norm gradients for real part
        assert bridge.layer_norm_real.weight.grad is not None, "Real layer norm weight should have gradients"
        assert bridge.layer_norm_real.bias.grad is not None, "Real layer norm bias should have gradients"
        assert not torch.isnan(bridge.layer_norm_real.weight.grad).any(), "Real layer norm gradients should not be NaN"
        assert not torch.isinf(bridge.layer_norm_real.weight.grad).any(), "Real layer norm gradients should not be Inf"
        
        # Check layer norm gradients for imaginary part
        assert bridge.layer_norm_imag.weight.grad is not None, "Imaginary layer norm weight should have gradients"
        assert bridge.layer_norm_imag.bias.grad is not None, "Imaginary layer norm bias should have gradients"
        assert not torch.isnan(bridge.layer_norm_imag.weight.grad).any(), "Imaginary layer norm gradients should not be NaN"
        assert not torch.isinf(bridge.layer_norm_imag.weight.grad).any(), "Imaginary layer norm gradients should not be Inf"
        
        # Check gradient magnitudes
        grad_norm = torch.norm(x.grad)
        real_layer_norm_grad = torch.norm(bridge.layer_norm_real.weight.grad)
        imag_layer_norm_grad = torch.norm(bridge.layer_norm_imag.weight.grad)
        assert grad_norm > 1e-8, "Input gradients should not vanish"
        assert real_layer_norm_grad > 1e-8, "Real layer norm gradients should not vanish"
        assert imag_layer_norm_grad > 1e-8, "Imaginary layer norm gradients should not vanish"
        assert grad_norm < 100, "Input gradients should not explode"
        assert real_layer_norm_grad < 100, "Real layer norm gradients should not explode"
        assert imag_layer_norm_grad < 100, "Imaginary layer norm gradients should not explode"
    
    def test_bridge_scales(self, bridge, test_state):
        """Test scale bridging functionality."""
        # Test upscaling
        source_scale = 1.0
        target_scale = 2.0
        
        upscaled = bridge.bridge_scales(
            test_state,
            source_scale,
            target_scale
        )
        
        assert upscaled.shape == test_state.shape
        assert torch.all(torch.isfinite(upscaled))
        
        # Test downscaling
        source_scale = 2.0
        target_scale = 1.0
        
        downscaled = bridge.bridge_scales(
            test_state,
            source_scale,
            target_scale
        )
        
        assert downscaled.shape == test_state.shape
        assert torch.all(torch.isfinite(downscaled))
        
        # Test scale preservation
        source_scale = 1.0
        target_scale = 1.0
        
        preserved = bridge.bridge_scales(
            test_state,
            source_scale,
            target_scale
        )
        
        assert preserved.shape == test_state.shape
        assert torch.all(torch.isfinite(preserved))
        
    def test_compute_coherence(self, bridge):
        """Test quantum coherence computation."""
        # Create test states
        state1 = torch.randn(8, 64)  # [batch_size, hidden_dim]
        state2 = torch.randn(8, 64)  # [batch_size, hidden_dim]
        
        # Compute coherence
        coherence = bridge.compute_coherence(state1, state2)
        
        # Check properties
        assert torch.is_tensor(coherence)
        assert torch.all(torch.isfinite(coherence))
        assert torch.all(coherence >= 0)  # Coherence should be non-negative
        assert torch.all(coherence <= 1)  # Coherence should be bounded by 1
        
        # Test self-coherence
        self_coherence = bridge.compute_coherence(state1, state1)
        assert torch.allclose(self_coherence, torch.ones_like(self_coherence))
        
        # Test symmetry
        coherence_12 = bridge.compute_coherence(state1, state2)
        coherence_21 = bridge.compute_coherence(state2, state1)
        assert torch.allclose(coherence_12, coherence_21)
        
    def test_invalid_inputs(self, bridge):
        """Test bridge behavior with invalid inputs."""
        # Test invalid scale values
        state = torch.randn(8, 64)  # [batch_size, hidden_dim]
        
        with pytest.raises(ValueError):
            bridge.bridge_scales(
                state,
                source_scale=-1.0,  # Invalid negative scale
                target_scale=1.0
            )
            
        with pytest.raises(ValueError):
            bridge.bridge_scales(
                state,
                source_scale=1.0,
                target_scale=0.0  # Invalid zero scale
            )
            
        # Test invalid state shapes
        invalid_state = torch.randn(8, 16)  # Wrong dimension - should be hidden_dim (64)
        
        with pytest.raises(ValueError, match="Input tensor must have hidden dimension"):
            bridge.bridge_scales(
                invalid_state,
                source_scale=1.0,
                target_scale=2.0
            )
            
        with pytest.raises(ValueError, match="Input tensor must have hidden dimension"):
            bridge.compute_coherence(
                state,
                invalid_state  # Mismatched dimensions
            ) 