"""End-to-end integration tests for the neural quantum system."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import gc
import psutil
import os
import yaml
from pathlib import Path
import logging
import time
import traceback
import tracemalloc
from typing import Dict, Any, Optional, cast, Union, Tuple

# Remove global tracemalloc.start() since we'll initialize it in the profile_memory_usage function

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention, AttentionState
from src.core.tiling.quantum_attention_tile import QuantumMotivicTile
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.core.patterns.fiber_types import LocalChart as PatternSection
from src.core.tiling.state_manager import StateManager, StateConfig, StateType
from src.validation.quantum.state import QuantumStateValidationResult
from src.core.quantum.types import QuantumState




# Configure logger
logger = logging.getLogger(__name__)

# Load test configuration
with open("tests/test_integration/test_config.yaml", "r") as f:
    TEST_CONFIG = yaml.safe_load(f)

# Get active regime configuration
ACTIVE_REGIME = TEST_CONFIG["active_regime"]
REGIME_CONFIG = TEST_CONFIG["regimes"][ACTIVE_REGIME]
COMMON_CONFIG = TEST_CONFIG["common"]

# Merge active regime config with common config
TEST_CONFIG = {**COMMON_CONFIG, **REGIME_CONFIG}

logger.info(f"Running tests with {ACTIVE_REGIME} regime")


def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def cleanup_memory():
    """Clean up memory based on configuration."""
    if TEST_CONFIG["clear_memory"]:
        gc.collect()


@pytest.fixture
def hidden_dim():
    """Hidden dimension for tests."""
    return TEST_CONFIG["hidden_dim"]


@pytest.fixture
def num_heads():
    """Number of attention heads."""
    return TEST_CONFIG["num_heads"]


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return TEST_CONFIG["batch_size"]


@pytest.fixture
def attention_layer(hidden_dim, num_heads):
    """Create quantum geometric attention layer."""
    logger.info(f"Initializing attention layer with hidden_dim={hidden_dim}, num_heads={num_heads}")
    layer = QuantumGeometricAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=TEST_CONFIG["dropout"],
        manifold_type=TEST_CONFIG["manifold_type"],
        curvature=TEST_CONFIG["curvature"],
        manifold_dim=TEST_CONFIG["manifold_dim"],
        num_layers=TEST_CONFIG["num_layers"],
        tile_size=TEST_CONFIG["tile_size"],
        motive_rank=TEST_CONFIG["motive_rank"]
    )
    logger.info(f"Attention layer initialized with manifold_dim={TEST_CONFIG['manifold_dim']}")
    return layer


@pytest.fixture
def attention_tile(hidden_dim):
    """Create quantum motivic attention tile."""
    return QuantumMotivicTile(
        size=hidden_dim,
        hidden_dim=hidden_dim,
        num_heads=TEST_CONFIG["num_heads"],
        dropout=TEST_CONFIG["dropout"],
        resolution=1.0,
        cohomology_dim=TEST_CONFIG["manifold_dim"],
        motive_rank=TEST_CONFIG["motive_rank"]
    )


@pytest.fixture
def neural_bridge(hidden_dim, num_heads):
    """Create neural quantum bridge."""
    return NeuralQuantumBridge(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=TEST_CONFIG["dropout"],
        manifold_type=TEST_CONFIG["manifold_type"],
        curvature=TEST_CONFIG["curvature"]
    )


@pytest.fixture
def test_input(batch_size, hidden_dim):
    """Create test input tensor."""
    return torch.randn(batch_size, TEST_CONFIG["seq_len"], hidden_dim)  # Shape: [batch_size, seq_len, hidden_dim]


@pytest.fixture
def state_manager():
    """Create state manager for geometric computations."""
    config = StateConfig(
        dim=TEST_CONFIG["hidden_dim"],
        type=StateType.PURE,
        epsilon=TEST_CONFIG["epsilon"],
        max_entanglement=1.0  # Default value since we removed it from config
    )
    return StateManager(config=config)


def log_tensor_info(name: str, tensor: torch.Tensor):
    """Log tensor shape, dtype, and basic statistics."""
    if TEST_CONFIG["debug"]["log_shapes"]:
        if tensor.dtype in [torch.complex64, torch.complex128]:
            # For complex tensors, compute stats on absolute values
            abs_tensor = torch.abs(tensor)
            logger.debug(
                f"{name} - Shape: {tensor.shape}, Dtype: {tensor.dtype}, "
                f"Mean Abs: {abs_tensor.mean().item():.3f}, Std Abs: {abs_tensor.std().item():.3f}, "
                f"Min Abs: {abs_tensor.min().item():.3f}, Max Abs: {abs_tensor.max().item():.3f}"
            )
        else:
            logger.debug(
                f"{name} - Shape: {tensor.shape}, Dtype: {tensor.dtype}, "
                f"Mean: {tensor.mean().item():.3f}, Std: {tensor.std().item():.3f}, "
                f"Min: {tensor.min().item():.3f}, Max: {tensor.max().item():.3f}"
            )


def check_tensor_valid(name: str, tensor: torch.Tensor) -> bool:
    """Check if tensor contains NaN or Inf values."""
    if not TEST_CONFIG["debug"]["check_nan"]:
        return True
        
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        logger.error(f"{name} contains {'NaN' if has_nan else 'Inf'} values!")
        if TEST_CONFIG["debug"]["break_on_nan"]:
            raise ValueError(f"{name} contains invalid values")
        return False
    return True


def profile_memory_usage():
    """Profile current memory usage."""
    if not TEST_CONFIG["debug"]["profile_memory"]:
        return {}
        
    # Initialize tracemalloc if not already started
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        
    try:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            'current': get_memory_usage(),
            'peak': max(stat.size for stat in top_stats) / 1024 / 1024 if top_stats else 0
        }
    except Exception as e:
        logger.warning(f"Memory profiling failed: {str(e)}")
        return {
            'current': get_memory_usage(),
            'peak': 0
        }
    finally:
        # Stop tracing to clean up
        tracemalloc.stop()


def save_debug_state(name: str, state: Dict[str, Any]):
    """Save debug state for analysis."""
    if not TEST_CONFIG["debug"]["save_states"]:
        return
        
    debug_dir = Path("debug_states")
    debug_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    state_file = debug_dir / f"{name}_{timestamp}.pt"
    
    # Convert tensors to CPU before saving
    cpu_state = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in state.items()
    }
    
    torch.save(cpu_state, state_file)
    logger.debug(f"Saved debug state to {state_file}")


def get_quantum_state_tensor(state: Union[QuantumState, Tuple[QuantumState, Any]]) -> torch.Tensor:
    """Extract tensor from quantum state or validation result tuple."""
    if isinstance(state, tuple):
        state = state[0]  # Extract state from validation result
    # Access the tensor representation directly
    tensor = state.amplitudes if hasattr(state, 'amplitudes') else state
    return cast(torch.Tensor, tensor)  # Ensure return type is Tensor


@pytest.mark.dependency(depends=["TestStateSpace", "TestScaleTransition", "TestTransitionAccuracy"])
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_attention_quantum_flow(
        self,
        attention_layer: QuantumGeometricAttention,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test flow from attention through quantum evolution and back."""
        logger.info("Starting attention quantum flow test")
        
        # Log initial tensor info with more detail
        logger.info(f"Input tensor shape: {test_input.shape}")
        logger.info(f"Input tensor device: {test_input.device}")
        logger.info(f"Input tensor dtype: {test_input.dtype}")
        
        # Log memory usage before starting
        if TEST_CONFIG["track_memory"]:
            initial_memory = get_memory_usage()
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        cleanup_memory()
        
        try:
            # 1. Apply attention
            logger.info("Applying attention layer")
            start_time = time.time()
            
            # Log attention layer configuration
            logger.info(f"Attention layer head_dim: {attention_layer.head_dim}")
            logger.info(f"Attention layer num_heads: {attention_layer.num_heads}")
            logger.info(f"Attention layer manifold_dim: {attention_layer.manifold_dim}")
            
            attention_out = attention_layer(
                test_input,
                mask=None,
                return_metrics=TEST_CONFIG["debug"]["log_metrics"]
            )
            
            if isinstance(attention_out, tuple):
                attention_out, metrics = attention_out
                if TEST_CONFIG["debug"]["log_metrics"]:
                    logger.info(f"Attention metrics: {metrics}")
            
            logger.info(f"Attention computation took {time.time() - start_time:.2f}s")
            logger.info(f"Attention output shape: {attention_out.shape}")
            
            # Log memory after attention
            if TEST_CONFIG["track_memory"]:
                attention_memory = get_memory_usage()
                logger.info(f"Memory after attention: {attention_memory:.2f} MB")
                logger.info(f"Memory increase: {attention_memory - initial_memory:.2f} MB")

            if TEST_CONFIG["debug"]["save_states"]:
                save_debug_state("post_attention", {
                    'input': test_input,
                    'output': attention_out
                })
            
            assert attention_out.shape == test_input.shape
            
            # Clear intermediate tensors
            del attention_layer
            cleanup_memory()
            
            # 2. Convert to quantum state
            logger.debug("Converting to quantum state")
            start_time = time.time()
            
            quantum_state = neural_bridge.neural_to_quantum(attention_out)
            quantum_tensor = get_quantum_state_tensor(quantum_state)
            
            logger.debug(f"Quantum conversion took {time.time() - start_time:.2f}s")
            log_tensor_info("quantum_state", quantum_tensor)
            check_tensor_valid("quantum_state", quantum_tensor)
            
            if TEST_CONFIG["debug"]["save_states"]:
                save_debug_state("post_quantum", {
                    'input': attention_out,
                    'quantum_state': quantum_tensor
                })
            
            assert isinstance(quantum_state, QuantumState)
            assert torch.allclose(quantum_state.norm(), torch.tensor(1.0, dtype=quantum_state.norm().dtype))
            
            # Check quantum state properties
            assert quantum_tensor.shape[-1] == 4  # manifold_dim is 4
            
            # Clear intermediate tensors
            del attention_out
            cleanup_memory()
            
            # 3. Evolve quantum state
            logger.debug("Evolving quantum state")
            start_time = time.time()
            flow_step = 0
            
            def flow_callback(step_state):
                nonlocal flow_step
                flow_step += 1
                if TEST_CONFIG["debug"]["log_flow"]:
                    logger.debug(f"Flow step {flow_step}")
                    step_tensor = get_quantum_state_tensor(step_state)
                    log_tensor_info(f"flow_state_{flow_step}", step_tensor)
                if flow_step >= TEST_CONFIG["debug"]["max_flow_steps"]:
                    raise TimeoutError("Maximum flow steps exceeded")
            
            evolved_state = neural_bridge.evolve_quantum_state(quantum_state, time=1.0)
            
            logger.debug(f"Quantum evolution took {time.time() - start_time:.2f}s")
            
            if isinstance(evolved_state, tuple):
                evolved_state = evolved_state[0]
            
            evolved_tensor = get_quantum_state_tensor(evolved_state)
            log_tensor_info("evolved_state", evolved_tensor)
            check_tensor_valid("evolved_state", evolved_tensor)
            
            if TEST_CONFIG["debug"]["save_states"]:
                save_debug_state("post_evolution", {
                    'input': quantum_tensor,
                    'evolved_state': evolved_tensor
                })
            
            assert isinstance(evolved_state, QuantumState)
            assert torch.allclose(evolved_state.norm(), torch.tensor(1.0, dtype=evolved_state.norm().dtype))
            
            # Clear intermediate tensors
            del quantum_state
            cleanup_memory()
            
            # 4. Convert back to neural
            logger.debug("Converting back to neural state")
            start_time = time.time()
            
            final_out = neural_bridge.quantum_to_neural(evolved_state)
            
            logger.debug(f"Neural conversion took {time.time() - start_time:.2f}s")
            log_tensor_info("final_out", final_out)
            check_tensor_valid("final_out", final_out)
            
            if TEST_CONFIG["debug"]["save_states"]:
                save_debug_state("final", {
                    'input': evolved_tensor,
                    'output': final_out
                })
            
            assert final_out.shape == test_input.shape
            
            # Clear final tensors
            del evolved_state
            cleanup_memory()
            
            # Check information preservation
            attention_norm = torch.linalg.vector_norm(test_input, dim=-1)
            final_norm = torch.linalg.vector_norm(final_out, dim=-1)
            norm_ratio = final_norm / attention_norm
            assert torch.allclose(norm_ratio, torch.tensor(1.0, dtype=norm_ratio.dtype), rtol=TEST_CONFIG["evolution"]["rtol"])
            
            logger.info("Attention quantum flow test completed successfully")
            
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        finally:
            if TEST_CONFIG["debug"]["profile_memory"]:
                memory_stats = profile_memory_usage()
                logger.info(f"Memory stats: {memory_stats}")
                tracemalloc.stop()
            
            cleanup_memory()

    def test_pattern_evolution_flow(
        self,
        attention_tile: QuantumMotivicTile,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test pattern evolution through the system."""
        try:
            # Initialize memory profiling if enabled
            if TEST_CONFIG["debug"]["profile_memory"]:
                initial_stats = profile_memory_usage()
                logger.info(f"Initial memory stats: {initial_stats}")
            
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
            
        finally:
            if TEST_CONFIG["debug"]["profile_memory"]:
                final_stats = profile_memory_usage()
                logger.info(f"Final memory stats: {final_stats}")
            cleanup_memory()

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
            test_input,
            mask=None,
            return_metrics=False
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
            mask=None,
            return_metrics=False
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
        input_state = AttentionState.initialize(
            hidden_dim=test_input.shape[-1],
            num_heads=attention_layer.num_heads,
            device=test_input.device,
            dtype=test_input.dtype
        )
        input_metric = attention_layer.compute_metric_tensor(input_state)
        
        # 2. Through attention
        attention_out = attention_layer(
            test_input,
            mask=None,
            return_metrics=False
        )
        attention_quantum = state_manager.initialize_state("attention", attention_out.shape[-1])
        attention_geometric = state_manager.initialize_state("attention_geom", attention_out.shape[-1])
        attention_state = AttentionState.initialize(
            hidden_dim=attention_out.shape[-1],
            num_heads=attention_layer.num_heads,
            device=attention_out.device,
            dtype=attention_out.dtype
        )
        attention_metric = attention_layer.compute_metric_tensor(attention_state)
        
        # 3. To quantum and evolve
        quantum_state = neural_bridge.neural_to_quantum(attention_out)
        if isinstance(quantum_state, tuple):
            quantum_state = quantum_state[0]  # Extract state from validation result
        evolved_state = neural_bridge.evolve_quantum_state(quantum_state, time=1.0)
        
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
        final_state = AttentionState.initialize(
            hidden_dim=scaled_pattern.coordinates.shape[-1],
            num_heads=attention_layer.num_heads,
            device=scaled_pattern.coordinates.device,
            dtype=scaled_pattern.coordinates.dtype
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