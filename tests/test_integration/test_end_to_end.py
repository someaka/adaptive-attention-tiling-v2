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

from src.core.tiling.quantum_geometric_attention import (
    QuantumGeometricAttention,
    AttentionState,
    QuantumGeometricConfig
)
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
    config = QuantumGeometricConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=TEST_CONFIG["dropout"],
        manifold_type=TEST_CONFIG["manifold_type"],
        curvature=TEST_CONFIG["curvature"],
        manifold_dim=TEST_CONFIG["manifold_dim"],
        num_layers=TEST_CONFIG["num_layers"],
        tile_size=TEST_CONFIG["tile_size"],
        motive_rank=TEST_CONFIG["motive_rank"],
        dtype=torch.float32,  # Changed to float32 as base type
        device=torch.device('cpu'),
        is_causal=False
    )
    layer = QuantumGeometricAttention(config=config)
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
        manifold_dim=TEST_CONFIG["manifold_dim"],
        num_heads=num_heads,
        dropout=TEST_CONFIG["dropout"],
        manifold_type=TEST_CONFIG["manifold_type"],
        curvature=TEST_CONFIG["curvature"],
        dtype=torch.float32,  # Changed to float32 as base type
        device=torch.device('cpu')
    )

@pytest.fixture
def test_input(batch_size, hidden_dim):
    """Create test input tensor."""
    # Create input with batch and sequence dimensions
    x = torch.randn(batch_size, TEST_CONFIG["seq_len"], hidden_dim)
    # Normalize along hidden dimension
    return F.normalize(x, p=2, dim=-1)

@pytest.fixture
def state_manager():
    """Create state manager for geometric computations."""
    config = StateConfig(
        dim=TEST_CONFIG["hidden_dim"],
        type=StateType.PURE,
        epsilon=TEST_CONFIG["epsilon"],
        max_entanglement=1.0,
        dtype=torch.float32  # Added dtype specification
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
        
        try:
            # Convert input to complex if needed
            if not torch.is_complex(test_input):
                test_input = test_input.to(torch.complex128)

            # 1. Apply attention
            attention_out = attention_layer(
                test_input,
                mask=None,
                return_metrics=TEST_CONFIG["debug"]["log_metrics"]
            )
            
            if isinstance(attention_out, tuple):
                attention_out, metrics = attention_out
            
            assert attention_out.shape == test_input.shape
            
            # 2. Convert to quantum state with attention
            quantum_state = neural_bridge.neural_to_quantum(attention_out)
            if isinstance(quantum_state, tuple):
                quantum_state, validation = quantum_state
                assert validation.is_valid, "Quantum state validation failed"
            
            # Create attention pattern with matching dtype
            batch_size = test_input.shape[0]
            attention_pattern = torch.randn(
                batch_size,
                attention_layer.config.hidden_dim,
                attention_layer.config.hidden_dim,
                dtype=quantum_state.amplitudes.dtype,
                device=test_input.device
            )
            attention_pattern = attention_pattern / torch.norm(
                attention_pattern,
                dim=(-2, -1),
                keepdim=True
            )
            
            # 3. Evolve with attention
            evolved_state = neural_bridge.evolve_quantum_state_with_attention(
                quantum_state,
                attention_pattern=attention_pattern,
                time=1.0
            )
            
            assert isinstance(evolved_state, QuantumState)
            assert torch.allclose(
                evolved_state.norm(),
                torch.ones(batch_size, dtype=torch.float64),
                atol=1e-6
            )
            
            # 4. Convert back to neural
            final_out = neural_bridge.quantum_to_neural(evolved_state)
            
            assert final_out.shape == test_input.shape
            assert torch.allclose(
                torch.norm(final_out, dim=-1),
                torch.ones(batch_size, dtype=torch.float64),
                atol=1e-6
            )
            
        finally:
            cleanup_memory()

    def test_scale_bridging(
        self,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test bridging between different scales."""
        try:
            # Convert input to complex if needed
            if not torch.is_complex(test_input):
                test_input = test_input.to(torch.complex128)

            # Bridge scales
            source_scale = 1.0
            target_scale = 2.0
            scaled_pattern = neural_bridge.bridge_scales(
                state=test_input,
                source_scale=source_scale,
                target_scale=target_scale
            )
            
            # Verify scale transition properties
            assert scaled_pattern.shape == test_input.shape
            scale_factor = target_scale / source_scale
            assert torch.allclose(
                torch.norm(scaled_pattern, dim=-1),
                torch.ones(test_input.shape[0], dtype=test_input.dtype).real * scale_factor,
                atol=1e-6
            )
            
        finally:
            cleanup_memory()

    def test_quantum_coherence(
        self,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test quantum coherence computation."""
        try:
            # Convert input to complex if needed
            if not torch.is_complex(test_input):
                test_input = test_input.to(torch.complex128)

            # Create two test patterns with single batch element
            pattern1 = test_input[:1]  # Take only first batch element
            pattern2 = torch.randn_like(pattern1)
            pattern2 = F.normalize(pattern2, p=2, dim=-1)

            # Compute coherence
            coherence = neural_bridge.compute_coherence(pattern1, pattern2)

            # Verify coherence properties
            assert coherence.shape == (1,), f"Expected shape (1,) but got {coherence.shape}"
            assert torch.all(coherence >= 0), "Coherence should be non-negative"
            assert torch.all(coherence <= 1), "Coherence should be bounded by 1"

            # Test self-coherence
            self_coherence = neural_bridge.compute_coherence(pattern1, pattern1)
            assert torch.allclose(self_coherence, torch.ones_like(self_coherence), atol=1e-6), "Self-coherence should be 1"
            
        finally:
            cleanup_memory()

    def test_geometric_flow(
        self,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test geometric flow evolution."""
        try:
            # Convert input to complex if needed
            if not torch.is_complex(test_input):
                test_input = test_input.to(torch.complex128)

            # Evolve through geometric flow
            evolved_tensor = neural_bridge.evolve_geometric_flow_with_attention(
                test_input,
                time=1.0
            )

            # Verify evolution properties
            assert evolved_tensor.shape == test_input.shape
            assert torch.allclose(
                torch.norm(evolved_tensor, dim=-1),
                torch.ones(test_input.shape[0], dtype=test_input.dtype).real,
                atol=1e-6
            )

            # Verify no NaN or Inf values
            assert not torch.isnan(evolved_tensor).any(), "Evolution produced NaN values"
            assert not torch.isinf(evolved_tensor).any(), "Evolution produced Inf values"

        finally:
            cleanup_memory()

    def test_error_handling(
        self,
        neural_bridge: NeuralQuantumBridge
    ):
        """Test error handling in neural quantum bridge."""
        try:
            # Test invalid input dimensions
            with pytest.raises(ValueError):
                invalid_pattern = torch.randn(8)  # 1D tensor instead of required 2D or 3D
                neural_bridge.neural_to_quantum(invalid_pattern)
            
            # Test invalid scale factors
            with pytest.raises(ValueError):
                pattern = torch.randn(2, 8)
                pattern = F.normalize(pattern, p=2, dim=-1)
                neural_bridge.bridge_scales(
                    pattern,
                    source_scale=-1.0,
                    target_scale=1.0
                )
            
            # Test non-finite values
            with pytest.raises(ValueError):
                invalid_pattern = torch.full((2, 8), float('inf'))
                neural_bridge.neural_to_quantum(invalid_pattern)
                
        finally:
            cleanup_memory()