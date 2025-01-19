"""Tests for the neural quantum bridge."""

import pytest
import torch
import numpy as np
import torch.nn.functional as F

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
    
    def test_scale_normalization_gradients(self, bridge, test_state):
        """Test gradient flow through scale normalization."""
        # Enable gradient tracking
        x = test_state.clone().requires_grad_(True)
        
        # Track scale normalization gradients
        scale_grads = []
        def hook_fn(grad):
            scale_grads.append(grad.clone())
            return grad
            
        # Forward pass
        output = bridge(x)
        output.register_hook(hook_fn)
        
        # Compute loss that depends on scale
        loss = (output.abs().mean() - 1.0) ** 2  # Loss sensitive to scaling
        loss.backward()
        
        # Check gradient properties
        assert len(scale_grads) > 0, "Scale normalization should have gradients"
        assert not torch.isnan(scale_grads[0]).any(), "Scale gradients should not be NaN"
        assert not torch.isinf(scale_grads[0]).any(), "Scale gradients should not be Inf"
        assert torch.norm(scale_grads[0]) > 1e-8, "Scale gradients should not vanish"
        assert torch.norm(scale_grads[0]) < 100, "Scale gradients should not explode"
        
        # Check that pattern bundle metric receives gradients
        assert bridge.pattern_bundle.metric.grad is not None, "Pattern bundle metric should have gradients"
        assert torch.norm(bridge.pattern_bundle.metric.grad) > 1e-8, "Pattern bundle metric gradients should not vanish"

    def test_multi_head_gradient_flow(self, bridge):
        """Test gradient flow with multi-head inputs.
        
        This test verifies:
        1. Multi-head quantum state preparation and validation
        2. Independent processing of different attention heads
        3. Gradient flow through quantum operations
        4. Type consistency throughout computation chain
        """
        # Create multi-head input with proper type matching bridge's dtype
        batch_size = 2
        num_heads = 4
        seq_len = 8
        hidden_dim = bridge.hidden_dim
        
        # Input shape: [batch_size, num_heads, seq_len, hidden_dim]
        x = torch.randn(batch_size, num_heads, seq_len, hidden_dim, dtype=bridge.dtype, requires_grad=True)
        
        # Store original norm for later comparison
        original_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        
        # Forward pass through bridge
        output = bridge(x)
        
        # Verify output maintains multi-head structure and type
        assert output.shape == x.shape, "Output shape should match input"
        assert output.dtype == bridge.dtype, "Bridge output should match input dtype (neural space)"
        
        # Test head independence by perturbing one head
        perturbed_x = x.clone()
        head_idx = 1
        perturbed_x[:, head_idx] += 0.1  # Perturb second head
        perturbed_output = bridge(perturbed_x)
        
        # Only the perturbed head should change significantly
        output_diff = (perturbed_output - output).abs().mean(dim=(0, 2, 3))  # Average across batch, seq, hidden
        assert output_diff[head_idx] > output_diff.mean(), "Perturbed head should change more than others"
        unperturbed_diff = output_diff[torch.arange(num_heads) != head_idx]
        assert torch.all(unperturbed_diff < output_diff[head_idx]), "Unperturbed heads should change less"
        
        # Convert to quantum state and validate
        quantum_state = bridge.neural_to_quantum(x, return_validation=True)
        assert isinstance(quantum_state, tuple), "Expected (state, validation) tuple"
        state, validation = quantum_state
        
        # Verify state validation passed
        assert validation.is_valid, f"Quantum state validation failed: {validation.error_type}"
        assert validation.data["metrics"]["fidelity"] > 0.99, f"Low quantum state fidelity: {validation.data['metrics']['fidelity']}"
        
        # Verify state properties
        assert state.layout["type"] == "attention", "Wrong state layout type"
        assert state.layout["batch_size"] == batch_size
        assert state.layout["num_heads"] == num_heads
        assert state.layout["seq_length"] == seq_len
        assert state.layout["dim"] == hidden_dim
        
        # Get amplitudes and verify normalization per head
        amplitudes = state.amplitudes
        reshaped_amplitudes = amplitudes.reshape(batch_size, num_heads, seq_len, -1)
        norms = torch.norm(reshaped_amplitudes, dim=-1)  # Norm per head/sequence element
        assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5), "Amplitudes not normalized per head"
        
        # Compute loss that depends on head structure
        loss = output.abs().mean()
        loss.backward()
        
        # Verify input gradients
        assert x.grad is not None, "Input should have gradients"
        assert x.grad.shape == (batch_size, num_heads, seq_len, hidden_dim)
        
        # Verify gradient independence between heads
        grad_per_head = x.grad.abs().mean(dim=(0, 2, 3))  # Average across batch, seq, hidden
        grad_std = grad_per_head.std()
        assert grad_std > 1e-6, "Gradients should vary between heads"
        
        # Convert back to neural representation with original shape
        neural_output = bridge.quantum_to_neural(state)
        neural_output = neural_output.reshape(batch_size, num_heads, seq_len, hidden_dim)
        
        # Verify neural output maintains structure
        assert neural_output.shape == x.shape, "Neural output shape mismatch"
        assert neural_output.dtype == bridge.dtype, "Neural output dtype mismatch"
        
        # Verify norm preservation
        output_norm = torch.linalg.vector_norm(neural_output, dim=-1, keepdim=True)
        assert torch.allclose(output_norm, original_norm, rtol=1e-4), "Output norm should match input norm"

    def test_connection_parameter_gradients(self, bridge):
        """Test gradient flow through pattern bundle connection parameter.
        
        This test verifies:
        1. Connection parameter gradient computation
        2. Gradient flow through quantum evolution
        3. Gradient magnitude and stability
        """
        # Enable gradients for pattern bundle connection
        bridge.pattern_bundle.connection.requires_grad_(True)
        
        # Create input with batch and head dimensions
        batch_size = 2
        num_heads = 4
        hidden_dim = bridge.hidden_dim
        
        x = torch.randn(batch_size, num_heads, hidden_dim, requires_grad=True)
        
        # Track initial connection parameter
        initial_connection = bridge.pattern_bundle.connection.clone()
        
        # Forward pass through bridge
        output = bridge(x)
        
        # Convert complex output to real by taking magnitude
        if torch.is_complex(output):
            output = torch.abs(output)
            
        # Compute loss using real values
        loss = output.abs().mean()
        loss.backward()
        
        # Verify connection parameter gradients
        assert bridge.pattern_bundle.connection.grad is not None, "Connection should have gradients"
        assert not torch.allclose(
            bridge.pattern_bundle.connection.grad,
            torch.zeros_like(bridge.pattern_bundle.connection.grad)
        ), "Connection gradients should be non-zero"
        
        # Verify gradient properties
        grad = bridge.pattern_bundle.connection.grad
        assert torch.all(torch.isfinite(grad)), "Connection gradients should be finite"
        assert grad.shape == bridge.pattern_bundle.connection.shape, "Gradient shape mismatch" 