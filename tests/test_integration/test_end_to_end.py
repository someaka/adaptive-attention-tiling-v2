"""End-to-end integration tests for the neural quantum system."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention, AttentionState
from src.core.tiling.quantum_attention_tile import QuantumMotivicTile
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.core.quantum.state_space import QuantumState
from src.core.patterns.fiber_types import LocalChart as PatternSection
from src.core.tiling.state_manager import StateManager, StateConfig, StateType
from src.validation.quantum.state import QuantumStateValidationResult
from typing import Tuple, Union


@pytest.fixture
def hidden_dim():
    """Hidden dimension for tests."""
    return 64


@pytest.fixture
def num_heads():
    """Number of attention heads."""
    return 8


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 16


@pytest.fixture
def attention_layer(hidden_dim, num_heads):
    """Create quantum geometric attention layer."""
    return QuantumGeometricAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1
    )


@pytest.fixture
def attention_tile(hidden_dim):
    """Create quantum motivic attention tile."""
    return QuantumMotivicTile(
        size=hidden_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        resolution=1.0,
        cohomology_dim=8,
        motive_rank=4
    )


@pytest.fixture
def neural_bridge(hidden_dim, num_heads):
    """Create neural quantum bridge."""
    return NeuralQuantumBridge(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1
    )


@pytest.fixture
def test_input(batch_size, hidden_dim):
    """Create test input tensor."""
    return torch.randn(batch_size, hidden_dim)


@pytest.fixture
def state_manager():
    """Create state manager for geometric computations."""
    config = StateConfig(
        dim=64,
        type=StateType.PURE,
        epsilon=1e-6,
        max_entanglement=1.0
    )
    return StateManager(config=config)


@pytest.mark.dependency(depends=["TestStateSpace", "TestScaleTransition", "TestTransitionAccuracy"])
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_attention_quantum_flow(
        self,
        attention_layer: QuantumGeometricAttention,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test flow from attention through quantum evolution and back.
        
        Flow: Input -> Attention -> Quantum State -> Evolution -> Neural State
        """
        # 1. Apply attention
        attention_out = attention_layer(
            test_input, test_input, test_input
        )
        assert attention_out.shape == test_input.shape
        
        # 2. Convert to quantum state
        quantum_state = neural_bridge.neural_to_quantum(attention_out)
        assert isinstance(quantum_state, QuantumState)
        assert torch.allclose(quantum_state.norm(), torch.tensor(1.0))
        
        # 3. Evolve quantum state
        evolved_state = neural_bridge.evolve_quantum_state(quantum_state)
        if isinstance(evolved_state, tuple):
            evolved_state = evolved_state[0]  # Extract state from validation result
        assert isinstance(evolved_state, QuantumState)
        assert torch.allclose(evolved_state.norm(), torch.tensor(1.0))
        
        # 4. Convert back to neural
        final_out = neural_bridge.quantum_to_neural(evolved_state)
        assert final_out.shape == test_input.shape
        
        # Check information preservation
        attention_norm = torch.linalg.vector_norm(attention_out, dim=-1)
        final_norm = torch.linalg.vector_norm(final_out, dim=-1)
        norm_ratio = final_norm / attention_norm
        assert torch.allclose(norm_ratio, torch.tensor(1.0), rtol=1e-1)

    def test_pattern_evolution_flow(
        self,
        attention_tile: QuantumMotivicTile,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test pattern evolution through the system.
        
        Flow: Input -> Pattern -> Quantum Evolution -> Scale Transition -> Pattern
        """
        # 1. Create initial pattern section
        pattern = PatternSection(
            coordinates=test_input,
            dimension=test_input.shape[-1],
            transition_maps={}
        )
        
        # 2. Process through attention tile
        tile_output = attention_tile(pattern)
        assert isinstance(tile_output, PatternSection)
        
        # 3. Evolve pattern through bridge
        evolved_pattern, metrics = neural_bridge.evolve_pattern_bundle(
            tile_output,
            time=1.0,
            scale_factor=2.0
        )
        
        assert isinstance(evolved_pattern, PatternSection)
        assert "quantum_evolution" in metrics
        assert "transport" in metrics
        assert "scale" in metrics
        assert "validation" in metrics
        
        # Check pattern properties preserved
        assert evolved_pattern.dimension == pattern.dimension
        coord_correlation = F.cosine_similarity(
            evolved_pattern.coordinates.flatten(),
            pattern.coordinates.flatten(),
            dim=0
        )
        assert coord_correlation > 0.1  # Should be different but related

    def test_scale_aware_attention(
        self,
        attention_layer: QuantumGeometricAttention,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor,
        state_manager: StateManager
    ):
        """Test attention behavior across different scales.
        
        Flow: Input -> (Attention at scale 1) -> Scale Up -> (Attention at scale 2)
        """
        # 1. Initial attention at scale 1
        attention_scale1 = attention_layer(
            test_input, test_input, test_input
        )
        
        # 2. Create pattern and scale up
        pattern = PatternSection(
            coordinates=attention_scale1,
            dimension=attention_scale1.shape[-1],
            transition_maps={}
        )
        
        scaled_pattern, metrics = neural_bridge.evolve_pattern_bundle(
            pattern,
            time=1.0,
            scale_factor=2.0
        )
        
        # 3. Attention at new scale
        attention_scale2 = attention_layer(
            scaled_pattern.coordinates,
            scaled_pattern.coordinates,
            scaled_pattern.coordinates
        )
        
        # Check scale-appropriate properties
        scale1_norm = torch.linalg.vector_norm(attention_scale1, dim=-1)
        scale2_norm = torch.linalg.vector_norm(attention_scale2, dim=-1)
        norm_ratio = scale2_norm / scale1_norm
        assert torch.all(norm_ratio > 1.5), "Scale transition should affect attention magnitude"
        
        # Check attention patterns are scale-covariant
        attention_diff = torch.linalg.vector_norm(
            F.normalize(attention_scale2, dim=-1) - 
            F.normalize(attention_scale1, dim=-1),
            dim=-1
        )
        assert torch.all(attention_diff < 0.5), "Attention patterns should be similar up to scale"

    def test_geometric_preservation_flow(
        self,
        attention_layer: QuantumGeometricAttention,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor,
        state_manager: StateManager
    ):
        """Test preservation of geometric properties through the flow."""
        # 1. Initial geometric properties
        input_quantum = state_manager.initialize_state("input", test_input.shape[-1])
        input_geometric = state_manager.initialize_state("input_geom", test_input.shape[-1])
        input_state = AttentionState(
            quantum_state=input_quantum,
            geometric_state=input_geometric
        )
        input_metric = attention_layer.compute_metric_tensor(input_state)
        
        # 2. Through attention
        attention_out = attention_layer(
            test_input, test_input, test_input
        )
        attention_quantum = state_manager.initialize_state("attention", attention_out.shape[-1])
        attention_geometric = state_manager.initialize_state("attention_geom", attention_out.shape[-1])
        attention_state = AttentionState(
            quantum_state=attention_quantum,
            geometric_state=attention_geometric
        )
        attention_metric = attention_layer.compute_metric_tensor(attention_state)
        
        # 3. To quantum and evolve
        quantum_state = neural_bridge.neural_to_quantum(attention_out)
        if isinstance(quantum_state, tuple):
            quantum_state = quantum_state[0]  # Extract state from validation result
        evolved_state = neural_bridge.evolve_quantum_state(quantum_state)
        
        # 4. Scale transition
        pattern = PatternSection(
            coordinates=neural_bridge.quantum_to_neural(evolved_state),
            dimension=test_input.shape[-1],
            transition_maps={}
        )
        
        scaled_pattern, _ = neural_bridge.evolve_pattern_bundle(
            pattern,
            time=1.0,
            scale_factor=2.0
        )
        
        final_quantum = state_manager.initialize_state("final", scaled_pattern.coordinates.shape[-1])
        final_geometric = state_manager.initialize_state("final_geom", scaled_pattern.coordinates.shape[-1])
        final_state = AttentionState(
            quantum_state=final_quantum,
            geometric_state=final_geometric
        )
        final_metric = attention_layer.compute_metric_tensor(final_state)
        
        # Check geometric properties
        # - Metric signature should be preserved
        input_eigvals = torch.linalg.eigvals(input_metric)
        final_eigvals = torch.linalg.eigvals(final_metric)
        assert torch.allclose(
            torch.sign(input_eigvals.real),
            torch.sign(final_eigvals.real)
        )
        
        # - Geometric features should be preserved up to scale
        input_features = attention_layer._compute_geometric_features(input_state.geometric_state)
        final_features = attention_layer._compute_geometric_features(final_state.geometric_state)
        feature_ratio = torch.linalg.vector_norm(final_features) / torch.linalg.vector_norm(input_features)
        assert feature_ratio > 1.0, "Geometric features should scale up"