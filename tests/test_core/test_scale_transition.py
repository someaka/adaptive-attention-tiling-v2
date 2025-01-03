"""Tests for the scale transition system."""

import pytest
import torch
import torch.nn.functional as F
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
        dim=16,
        use_quantum_bridge=False,
        hidden_dim=16,
        dtype=torch.float32
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
        assert transition_layer.quantum_bridge is None
        
    def test_transition_up(self, transition_layer: ScaleTransitionLayer) -> None:
        """Test upward scale transition."""
        state = torch.randn(2, 16)
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
        state = torch.randn(2, 16)
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
        state = torch.randn(2, 16)  # Match config dimensions
        
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
        states = [torch.randn(2, 16) for _ in range(4)]  # Match config dimensions
        scales = [1.0, 2.0, 4.0, 8.0]
        
        connected = transition_system.connect_scales(states, scales)
        
        assert len(connected) == len(states)
        for state in connected:
            assert state.shape == states[0].shape
            assert torch.all(torch.isfinite(state))
            
    def test_validate_transitions(self, transition_system: ScaleTransitionSystem) -> None:
        """Test transition validation metrics."""
        # Create test states and scales
        states = [torch.randn(2, 16) for _ in range(3)]  # Match config dimensions
        scales = [1.0, 2.0, 4.0]
        
        metrics = transition_system.validate_transitions(states, scales)
        
        # Check metrics
        assert "scale_consistency" in metrics
        assert "information_preservation" in metrics
        if transition_system.config.use_quantum_bridge:
            assert "quantum_coherence" in metrics
        
        for metric_name, values in metrics.items():
            assert values.shape == (len(states) - 1,)  # Should be (2,) for 3 states
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

    def test_scale_transition_accuracy(self, transition_system: ScaleTransitionSystem) -> None:
        """Test accuracy of scale transitions at different scales."""
        # Create test states at different scales
        batch_size = 2
        dim = transition_system.config.dim
        base_state = torch.randn(batch_size, dim)
        
        scales = [0.5, 1.0, 2.0]
        states = {
            scale: F.normalize(base_state * scale, p=2, dim=-1)
            for scale in scales
        }
        
        # Test transitions between all pairs of scales
        for source_scale in scales:
            for target_scale in scales:
                if source_scale != target_scale:
                    # Perform transition
                    source_state = states[source_scale]
                    transitioned = transition_system.transition_layer(
                        source_state,
                        source_scale,
                        target_scale
                    )
                    
                    # Check scale-appropriate properties
                    scale_ratio = target_scale / source_scale
                    expected_norm = torch.linalg.vector_norm(source_state, dim=-1) * scale_ratio
                    actual_norm = torch.linalg.vector_norm(transitioned, dim=-1)
                    
                    assert torch.allclose(
                        actual_norm, 
                        expected_norm, 
                        rtol=1e-4
                    ), f"Scale transition from {source_scale} to {target_scale} did not preserve expected norm"

    def test_scale_transition_reversibility(self, transition_system: ScaleTransitionSystem) -> None:
        """Test reversibility of scale transitions."""
        # Create test state
        batch_size = 2
        dim = transition_system.config.dim
        original_state = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        
        # Test round-trip transitions
        scale_pairs = [
            (1.0, 2.0),
            (1.0, 0.5)
        ]
        
        for scale1, scale2 in scale_pairs:
            # Forward transition
            intermediate = transition_system.transition_layer(
                original_state,
                scale1,
                scale2
            )
            
            # Backward transition
            recovered = transition_system.transition_layer(
                intermediate,
                scale2,
                scale1
            )
            
            # Check recovery accuracy
            recovery_error = torch.linalg.vector_norm(recovered - original_state, dim=-1)
            assert torch.all(recovery_error < 1e-4), f"Scale transition {scale1}->{scale2}->{scale1} not reversible"

    def test_scale_transition_stability(self, transition_system: ScaleTransitionSystem) -> None:
        """Test stability of scale transitions under perturbations."""
        # Create test state
        batch_size = 2
        dim = transition_system.config.dim
        base_state = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        
        # Test stability under perturbations
        scales = [0.5, 1.0, 2.0]
        perturbation_sizes = [1e-4, 1e-3]
        
        for source_scale in scales:
            for target_scale in scales:
                if source_scale != target_scale:
                    for eps in perturbation_sizes:
                        # Create perturbed state
                        noise = torch.randn_like(base_state) * eps
                        perturbed_state = F.normalize(base_state + noise, p=2, dim=-1)
                        
                        # Transition both states
                        base_transitioned = transition_system.transition_layer(
                            base_state,
                            source_scale,
                            target_scale
                        )
                        
                        perturbed_transitioned = transition_system.transition_layer(
                            perturbed_state,
                            source_scale,
                            target_scale
                        )
                        
                        # Check stability
                        base_diff = torch.linalg.vector_norm(
                            base_transitioned - perturbed_transitioned,
                            dim=-1
                        )
                        assert torch.all(base_diff < eps * 10), f"Scale transition not stable for perturbation {eps}"


@pytest.mark.dependency(depends=["TestStateSpace"])
class TestScaleTransition:
    """Tests for ScaleTransition."""
    
    def test_initialization(self, transition_layer: ScaleTransitionLayer, config: ScaleTransitionConfig) -> None:
        """Test layer initialization."""
        assert len(transition_layer.scale_up) == config.num_scales - 1
        assert len(transition_layer.scale_down) == config.num_scales - 1
        assert transition_layer.quantum_bridge is None
        
    def test_transition_up(self, transition_layer: ScaleTransitionLayer) -> None:
        """Test upward scale transition."""
        state = torch.randn(2, 16)
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
        state = torch.randn(2, 16)
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
        state = torch.randn(2, 16)  # Match config dimensions
        
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


@pytest.mark.dependency(depends=["TestStateSpace", "TestScaleTransition"])
class TestTransitionAccuracy:
    """Tests for ScaleTransitionSystem."""
    
    def test_connect_scales(self, transition_system: ScaleTransitionSystem) -> None:
        """Test connecting states at different scales."""
        # Create test states and scales
        states = [torch.randn(2, 16) for _ in range(4)]  # Match config dimensions
        scales = [1.0, 2.0, 4.0, 8.0]
        
        connected = transition_system.connect_scales(states, scales)
        
        assert len(connected) == len(states)
        for state in connected:
            assert state.shape == states[0].shape
            assert torch.all(torch.isfinite(state))
            
    def test_validate_transitions(self, transition_system: ScaleTransitionSystem) -> None:
        """Test transition validation metrics."""
        # Create test states and scales
        states = [torch.randn(2, 16) for _ in range(3)]  # Match config dimensions
        scales = [1.0, 2.0, 4.0]
        
        metrics = transition_system.validate_transitions(states, scales)
        
        # Check metrics
        assert "scale_consistency" in metrics
        assert "information_preservation" in metrics
        if transition_system.config.use_quantum_bridge:
            assert "quantum_coherence" in metrics
        
        for metric_name, values in metrics.items():
            assert values.shape == (len(states) - 1,)  # Should be (2,) for 3 states
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

    def test_scale_transition_accuracy(self, transition_system: ScaleTransitionSystem) -> None:
        """Test accuracy of scale transitions at different scales."""
        # Create test states at different scales
        batch_size = 2
        dim = transition_system.config.dim
        base_state = torch.randn(batch_size, dim)
        
        scales = [0.5, 1.0, 2.0]
        states = {
            scale: F.normalize(base_state * scale, p=2, dim=-1)
            for scale in scales
        }
        
        # Test transitions between all pairs of scales
        for source_scale in scales:
            for target_scale in scales:
                if source_scale != target_scale:
                    # Perform transition
                    source_state = states[source_scale]
                    transitioned = transition_system.transition_layer(
                        source_state,
                        source_scale,
                        target_scale
                    )
                    
                    # Check scale-appropriate properties
                    scale_ratio = target_scale / source_scale
                    expected_norm = torch.linalg.vector_norm(source_state, dim=-1) * scale_ratio
                    actual_norm = torch.linalg.vector_norm(transitioned, dim=-1)
                    
                    assert torch.allclose(
                        actual_norm, 
                        expected_norm, 
                        rtol=1e-4
                    ), f"Scale transition from {source_scale} to {target_scale} did not preserve expected norm"

    def test_scale_transition_reversibility(self, transition_system: ScaleTransitionSystem) -> None:
        """Test reversibility of scale transitions."""
        # Create test state
        batch_size = 2
        dim = transition_system.config.dim
        original_state = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        
        # Test round-trip transitions
        scale_pairs = [
            (1.0, 2.0),
            (1.0, 0.5)
        ]
        
        for scale1, scale2 in scale_pairs:
            # Forward transition
            intermediate = transition_system.transition_layer(
                original_state,
                scale1,
                scale2
            )
            
            # Backward transition
            recovered = transition_system.transition_layer(
                intermediate,
                scale2,
                scale1
            )
            
            # Check recovery accuracy
            recovery_error = torch.linalg.vector_norm(recovered - original_state, dim=-1)
            assert torch.all(recovery_error < 1e-4), f"Scale transition {scale1}->{scale2}->{scale1} not reversible"

    def test_scale_transition_stability(self, transition_system: ScaleTransitionSystem) -> None:
        """Test stability of scale transitions under perturbations."""
        # Create test state
        batch_size = 2
        dim = transition_system.config.dim
        base_state = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        
        # Test stability under perturbations
        scales = [0.5, 1.0, 2.0]
        perturbation_sizes = [1e-4, 1e-3]
        
        for source_scale in scales:
            for target_scale in scales:
                if source_scale != target_scale:
                    for eps in perturbation_sizes:
                        # Create perturbed state
                        noise = torch.randn_like(base_state) * eps
                        perturbed_state = F.normalize(base_state + noise, p=2, dim=-1)
                        
                        # Transition both states
                        base_transitioned = transition_system.transition_layer(
                            base_state,
                            source_scale,
                            target_scale
                        )
                        
                        perturbed_transitioned = transition_system.transition_layer(
                            perturbed_state,
                            source_scale,
                            target_scale
                        )
                        
                        # Check stability
                        base_diff = torch.linalg.vector_norm(
                            base_transitioned - perturbed_transitioned,
                            dim=-1
                        )
                        assert torch.all(base_diff < eps * 10), f"Scale transition not stable for perturbation {eps}"


@pytest.mark.dependency(depends=["TestStateSpace", "TestScaleTransition", "TestTransitionAccuracy"])
class TestQuantumPropertyPreservation:
    """Tests for QuantumPropertyPreservation."""
    
    def test_quantum_property_preservation(self, transition_system: ScaleTransitionSystem) -> None:
        """Test preservation of quantum properties during scale transitions."""
        # Create quantum-like test state (normalized with phase)
        batch_size = 8
        dim = transition_system.config.dim
        amplitudes = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        phase = torch.exp(1j * torch.rand(batch_size, dim))
        quantum_state = amplitudes * phase
        
        # Test scale transition
        source_scale = 1.0
        target_scale = 2.0
        
        transitioned = transition_system.transition_layer(
            quantum_state,
            source_scale,
            target_scale
        )
        
        # Check normalization preservation
        orig_norm = torch.linalg.vector_norm(quantum_state, dim=-1)
        trans_norm = torch.linalg.vector_norm(transitioned, dim=-1)
        assert torch.allclose(
            trans_norm / orig_norm,
            torch.tensor(target_scale / source_scale),
            rtol=1e-4
        ), "Scale transition did not preserve quantum state normalization"
        
        # Check phase coherence
        phase_correlation = torch.abs(torch.sum(
            torch.conj(quantum_state) * transitioned,
            dim=-1
        ))
        assert torch.all(phase_correlation > 0.9), "Scale transition disrupted phase coherence"