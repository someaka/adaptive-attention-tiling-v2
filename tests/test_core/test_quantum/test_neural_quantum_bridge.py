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