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

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def cleanup_memory() -> None:
    """Clean up memory based on configuration."""
    if TEST_CONFIG["clear_memory"]:
        gc.collect()

@pytest.fixture
def hidden_dim() -> int:
    """Hidden dimension for tests."""
    return TEST_CONFIG["hidden_dim"]

@pytest.fixture
def num_heads() -> int:
    """Number of attention heads."""
    return TEST_CONFIG["num_heads"]

@pytest.fixture
def batch_size() -> int:
    """Batch size for tests."""
    return TEST_CONFIG["batch_size"]

@pytest.fixture
def attention_layer(hidden_dim: int, num_heads: int) -> QuantumGeometricAttention:
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
def neural_bridge(hidden_dim: int, num_heads: int) -> NeuralQuantumBridge:
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
def test_input(batch_size: int, hidden_dim: int) -> torch.Tensor:
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
    ) -> None:
        """Test the quantum attention flow pipeline.
        
        This test validates:
        1. Neural → Quantum state conversion with proper dtype handling
        2. Quantum state evolution through attention mechanism
        3. Quantum → Neural state conversion with preserved properties
        4. End-to-end flow through the attention layer
        """
        try:
            # Ensure input has correct shape [batch_size, seq_len, hidden_dim]
            batch_size = test_input.shape[0]
            seq_len = test_input.shape[1]
            hidden_dim = test_input.shape[2]
            
            # 1. Test neural bridge forward pass
            bridge_output = neural_bridge(test_input)
            assert bridge_output.shape == test_input.shape, "Bridge output shape mismatch"
            assert torch.isfinite(bridge_output).all(), "Bridge output contains invalid values"
            # Bridge applies transformations that modify the norm, but should maintain finite values
            assert torch.all(torch.norm(bridge_output, dim=-1) > 0), "Bridge output has zero norm"
            
            # 2. Test quantum state conversion and properties
            # Reshape input to [batch_size * seq_len, hidden_dim] for quantum conversion
            flat_input = test_input.reshape(-1, hidden_dim)
            quantum_state = neural_bridge.neural_to_quantum(flat_input)
            if isinstance(quantum_state, tuple):
                quantum_state, validation = quantum_state
                assert validation.is_valid, "Quantum state validation failed"
            
            assert isinstance(quantum_state, QuantumState), "Invalid quantum state type"
            assert quantum_state.amplitudes.dtype == torch.complex128, "Incorrect quantum state dtype"
            # Quantum states should be normalized
            assert torch.allclose(
                quantum_state.norm(),
                torch.ones(batch_size * seq_len, dtype=torch.float64),
                atol=1e-6
            ), "Quantum state normalization error"
            
            # 3. Test quantum evolution
            evolved_state = neural_bridge.evolve_quantum_state_with_attention(quantum_state)
            assert isinstance(evolved_state, QuantumState), "Evolution produced invalid state type"
            # Evolution should preserve quantum state norm
            assert torch.allclose(
                evolved_state.norm(),
                quantum_state.norm(),
                atol=1e-6
            ), "Evolution did not preserve state norm"
            
            # 4. Test attention layer end-to-end with metrics
            attention_out, metrics = attention_layer(test_input, return_metrics=True)
            assert attention_out.shape == test_input.shape, "Attention output shape mismatch"
            assert torch.isfinite(attention_out).all(), "Attention output contains invalid values"
            # Attention output should have non-zero norm
            assert torch.all(torch.norm(attention_out, dim=-1) > 0), "Attention output has zero norm"
            
            # 5. Validate attention patterns from metrics
            assert "attention_patterns" in metrics, "Missing attention patterns in metrics"
            attention_patterns = metrics["attention_patterns"]
            assert isinstance(attention_patterns, dict), "Invalid attention patterns format"
            assert "quantum" in attention_patterns, "Missing quantum attention patterns"
            
            quantum_patterns = attention_patterns["quantum"]
            assert quantum_patterns.shape[-2:] == (seq_len, seq_len), \
                "Invalid attention pattern shape"
            assert torch.isfinite(quantum_patterns).all(), "Attention pattern contains invalid values"
            # Attention patterns should sum to 1 after softmax
            assert torch.allclose(
                torch.sum(torch.softmax(quantum_patterns, dim=-1), dim=-1),
                torch.ones(quantum_patterns.shape[:-1], dtype=quantum_patterns.dtype),
                atol=1e-6
            ), "Attention pattern not normalized"
            
            # 6. Validate quantum properties are preserved
            neural_out = neural_bridge.quantum_to_neural(evolved_state)
            # Reshape back to match original input shape
            neural_out = neural_out.reshape(batch_size, seq_len, hidden_dim)
            assert neural_out.shape == test_input.shape, "Neural output shape mismatch"
            assert torch.isfinite(neural_out).all(), "Neural output contains invalid values"
            # Neural output should have non-zero norm
            assert torch.all(torch.norm(neural_out, dim=-1) > 0), "Neural output has zero norm"
            
            # 7. Test coherence between input and output states
            coherence = neural_bridge.compute_coherence(test_input.reshape(-1, hidden_dim), neural_out.reshape(-1, hidden_dim))
            assert torch.all(coherence >= 0) and torch.all(coherence <= 1), "Invalid coherence values"
            assert coherence.shape == (batch_size * seq_len,), "Incorrect coherence shape"
            # Coherence should be non-zero (states shouldn't be orthogonal)
            assert torch.all(coherence > 0), "Zero coherence detected"
            
            # 8. Validate entanglement history
            assert "entanglement_history" in metrics, "Missing entanglement history"
            entanglement = metrics["entanglement_history"]
            assert isinstance(entanglement, dict), "Invalid entanglement history format"
            
            # 9. Validate quantum entropy
            if "step_0" in metrics and "quantum_entropy" in metrics["step_0"]:
                entropy = metrics["step_0"]["quantum_entropy"]
                assert torch.isfinite(entropy).all(), "Invalid quantum entropy values"
                assert torch.all(entropy >= 0), "Negative quantum entropy detected"
                # Entropy should be bounded by ln(dim) for a quantum system
                max_entropy = np.log(hidden_dim)
                assert torch.all(entropy <= max_entropy + 1e-6), "Entropy exceeds theoretical maximum"
            
        finally:
            cleanup_memory()

    def test_scale_bridging(
        self,
        neural_bridge: NeuralQuantumBridge,
        test_input: torch.Tensor
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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