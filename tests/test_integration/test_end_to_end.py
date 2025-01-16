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
        dtype=torch.float32,  # Base type for neural operations
        device=torch.device('cpu'),
        is_causal=False
    )
    layer = QuantumGeometricAttention(config=config)
    logger.info(f"Attention layer initialized with manifold_dim={TEST_CONFIG['manifold_dim']}")
    return layer

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
        dtype=torch.float32,  # Base type for neural operations
        device=torch.device('cpu')
    )

@pytest.fixture
def test_input(batch_size, hidden_dim):
    """Create test input tensor."""
    # Create input with batch and sequence dimensions
    x = torch.randn(batch_size, TEST_CONFIG["seq_len"], hidden_dim)
    # Normalize along hidden dimension and ensure float32
    return F.normalize(x, p=2, dim=-1).to(dtype=torch.float32)

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
            # Ensure input is float32
            test_input = test_input.to(dtype=torch.float32)

            # Convert geometric tensor to real before manifold projection
            attention_out = attention_layer(
                test_input,
                mask=None,
                return_metrics=TEST_CONFIG["debug"]["log_metrics"]
            )
            
            if isinstance(attention_out, tuple):
                attention_out, metrics = attention_out
            
            # Ensure output is float32
            attention_out = attention_out.real.to(dtype=torch.float32)
            assert attention_out.shape == test_input.shape
            
            # Convert to quantum state
            quantum_state = neural_bridge.neural_to_quantum(attention_out)
            if isinstance(quantum_state, tuple):
                quantum_state, validation = quantum_state
                assert validation.is_valid, "Quantum state validation failed"
            
            # Create attention pattern with complex128 dtype
            batch_size = test_input.shape[0]
            attention_pattern = torch.randn(
                batch_size,
                attention_layer.config.hidden_dim,
                attention_layer.config.hidden_dim,
                dtype=torch.complex128,
                device=test_input.device
            )
            attention_pattern = attention_pattern / torch.norm(
                attention_pattern,
                dim=(-2, -1),
                keepdim=True
            )
            
            # Evolve with attention
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
            
            # Convert back to neural (will handle dtype conversion internally)
            final_out = neural_bridge.quantum_to_neural(evolved_state)
            final_out = final_out.real.to(dtype=torch.float32)
            
            assert final_out.shape == test_input.shape
            assert final_out.dtype == torch.float32
            assert torch.allclose(
                torch.norm(final_out, dim=-1),
                torch.ones(batch_size, dtype=torch.float32),
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
            # Ensure input is float32
            test_input = test_input.to(dtype=torch.float32)

            # Bridge scales
            source_scale = 1.0
            target_scale = 2.0
            scaled_pattern = neural_bridge.bridge_scales(
                state=test_input,
                source_scale=source_scale,
                target_scale=target_scale
            )
            
            # Convert output to float32
            scaled_pattern = scaled_pattern.real.to(dtype=torch.float32)
            
            # Verify scale transition properties with float32
            assert scaled_pattern.shape == test_input.shape
            assert scaled_pattern.dtype == torch.float32
            scale_factor = target_scale / source_scale
            assert torch.allclose(
                torch.norm(scaled_pattern, dim=-1),
                torch.ones(test_input.shape[0], dtype=torch.float32) * scale_factor,
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
            # Ensure input is float32
            test_input = test_input.to(dtype=torch.float32)

            # Create two test patterns with single batch element
            pattern1 = test_input[:1]  # Take only first batch element
            pattern2 = torch.randn_like(pattern1)
            pattern2 = F.normalize(pattern2, p=2, dim=-1)

            # Compute coherence
            coherence = neural_bridge.compute_coherence(pattern1, pattern2)

            # Verify coherence properties
            assert coherence.shape == (1,)
            assert coherence.dtype == torch.float32
            assert torch.all(coherence >= 0)
            assert torch.all(coherence <= 1)

            # Test self-coherence
            self_coherence = neural_bridge.compute_coherence(pattern1, pattern1)
            assert torch.allclose(
                self_coherence,
                torch.ones_like(self_coherence, dtype=torch.float32),
                atol=1e-6
            )
            
        finally:
            cleanup_memory()

    def test_geometric_flow(
        self,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ):
        """Test geometric flow evolution."""
        try:
            # Ensure input is float32
            test_input = test_input.to(dtype=torch.float32)

            # Evolve through geometric flow
            evolved_tensor = neural_bridge.evolve_geometric_flow_with_attention(
                test_input,
                time=1.0
            )

            # Convert output to float32
            evolved_tensor = evolved_tensor.real.to(dtype=torch.float32)

            # Verify evolution properties with float32
            assert evolved_tensor.shape == test_input.shape
            assert evolved_tensor.dtype == torch.float32
            assert torch.allclose(
                torch.norm(evolved_tensor, dim=-1),
                torch.ones(test_input.shape[0], dtype=torch.float32),
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
                invalid_pattern = torch.randn(8, dtype=torch.float32)  # 1D tensor instead of required 2D or 3D
                neural_bridge.neural_to_quantum(invalid_pattern)
            
            # Test invalid scale factors
            with pytest.raises(ValueError):
                pattern = torch.randn(2, 8, dtype=torch.float32)
                pattern = F.normalize(pattern, p=2, dim=-1)
                neural_bridge.bridge_scales(
                    pattern,
                    source_scale=-1.0,
                    target_scale=1.0
                )
            
            # Test non-finite values
            with pytest.raises(ValueError):
                invalid_pattern = torch.full((2, 8), float('inf'), dtype=torch.float32)
                neural_bridge.neural_to_quantum(invalid_pattern)
                
        finally:
            cleanup_memory()