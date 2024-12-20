"""Tests for the scale transition system."""

import pytest
import torch
import numpy as np
from typing import Dict

from src.core.scale_transition import (
    ScaleTransitionConfig,
    ScaleTransitionLayer,
    ScaleTransitionSystem
)


@pytest.fixture
def config() -> ScaleTransitionConfig:
    """Create test configuration."""
    return ScaleTransitionConfig(
        min_scale=0.25,
        max_scale=4.0,
        num_scales=4,
        dim=64,
        use_quantum_bridge=True
    )


@pytest.fixture
def transition_layer(config: ScaleTransitionConfig) -> ScaleTransitionLayer:
    """Create test transition layer."""
    return ScaleTransitionLayer(config)


@pytest.fixture
def transition_system(config: ScaleTransitionConfig) -> ScaleTransitionSystem:
    """Create test transition system."""
    return ScaleTransitionSystem(config)


class TestScaleTransitionLayer:
    """Tests for ScaleTransitionLayer."""
    
    def test_initialization(self, transition_layer: ScaleTransitionLayer, config: ScaleTransitionConfig) -> None:
        """Test layer initialization."""
        assert len(transition_layer.scale_up) == config.num_scales - 1
        assert len(transition_layer.scale_down) == config.num_scales - 1
        assert transition_layer.quantum_bridge is not None
        
    def test_transition_up(self, transition_layer: ScaleTransitionLayer) -> None:
        """Test upward scale transition."""
        state = torch.randn(8, 64)  # [batch_size, dim]
        source_scale = 1.0
        target_scale = 2.0
        
        transitioned = transition_layer.transition_up(
            state,
            source_scale,
            target_scale
        )
        
        assert transitioned.shape == state.shape
        assert torch.all(torch.isfinite(transitioned))
        
    def test_transition_down(self, transition_layer: ScaleTransitionLayer) -> None:
        """Test downward scale transition."""
        state = torch.randn(8, 64)
        source_scale = 2.0
        target_scale = 1.0
        
        transitioned = transition_layer.transition_down(
            state,
            source_scale,
            target_scale
        )
        
        assert transitioned.shape == state.shape
        assert torch.all(torch.isfinite(transitioned))
        
    def test_invalid_transitions(self, transition_layer: ScaleTransitionLayer) -> None:
        """Test invalid scale transitions."""
        state = torch.randn(8, 64)
        
        # Test invalid upward transition
        with pytest.raises(ValueError):
            transition_layer.transition_up(
                state,
                source_scale=2.0,
                target_scale=1.0
            )
            
        # Test invalid downward transition
        with pytest.raises(ValueError):
            transition_layer.transition_down(
                state,
                source_scale=1.0,
                target_scale=2.0
            )
            
        # Test too large scale difference
        with pytest.raises(ValueError):
            transition_layer.transition_up(
                state,
                source_scale=1.0,
                target_scale=32.0
            )


class TestScaleTransitionSystem:
    """Tests for ScaleTransitionSystem."""
    
    def test_connect_scales(self, transition_system: ScaleTransitionSystem) -> None:
        """Test connecting states at different scales."""
        # Create test states and scales
        states = [torch.randn(8, 64) for _ in range(4)]
        scales = [1.0, 2.0, 4.0, 8.0]
        
        connected = transition_system.connect_scales(states, scales)
        
        assert len(connected) == len(states)
        for state in connected:
            assert state.shape == states[0].shape
            assert torch.all(torch.isfinite(state))
            
    def test_validate_transitions(self, transition_system: ScaleTransitionSystem) -> None:
        """Test transition validation metrics."""
        # Create test states and scales
        states = [torch.randn(8, 64) for _ in range(3)]
        scales = [1.0, 2.0, 4.0]
        
        metrics = transition_system.validate_transitions(states, scales)
        
        # Check metrics
        assert "scale_consistency" in metrics
        assert "information_preservation" in metrics
        if transition_system.config.use_quantum_bridge:
            assert "quantum_coherence" in metrics
        
        for metric_name, values in metrics.items():
            assert values.shape == (len(states) - 1,)
            assert torch.all(torch.isfinite(values))
                
    def test_invalid_inputs(self, transition_system: ScaleTransitionSystem) -> None:
        """Test system behavior with invalid inputs."""
        # Test mismatched states and scales
        states = [torch.randn(8, 64) for _ in range(3)]
        scales = [1.0, 2.0]  # One scale missing
        
        with pytest.raises(ValueError):
            transition_system.connect_scales(states, scales)
            
        with pytest.raises(ValueError):
            transition_system.validate_transitions(states, scales)
            
    def test_disabled_quantum_bridge(self) -> None:
        """Test system behavior with quantum bridge disabled."""
        config = ScaleTransitionConfig(
            min_scale=0.25,
            max_scale=4.0,
            num_scales=4,
            dim=64,
            use_quantum_bridge=False
        )
        system = ScaleTransitionSystem(config)
        
        # Create test states and scales
        states = [torch.randn(8, 64) for _ in range(3)]
        scales = [1.0, 2.0, 4.0]
        
        metrics = system.validate_transitions(states, scales)
        
        # Check that quantum coherence is not present
        assert "quantum_coherence" not in metrics