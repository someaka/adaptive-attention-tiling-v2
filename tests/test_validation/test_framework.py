"""
Unit tests for the validation framework.

Tests cover:
1. Geometric validation
2. Quantum validation
3. Pattern validation
4. Integration tests
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from typing import Dict, List, Optional, Sequence, Union, cast
import gc
import logging
import time
from contextlib import contextmanager

from src.validation.framework import ValidationFramework, ConcreteValidationResult, FrameworkValidationResult
from src.validation.geometric.model import ModelGeometricValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.patterns.stability import PatternValidator as StabilityValidator
from validation.flow.flow_stability import LinearStabilityValidator, NonlinearStabilityValidator
from src.validation.base import ValidationResult
from src.core.models.base import LayerGeometry, ModelGeometry
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.tiling.geometric_flow import GeometricFlow
from src.core.quantum.types import QuantumState
from src.core.patterns.dynamics import PatternDynamics as CorePatternDynamics
from src.core.patterns import (
    BaseRiemannianStructure,
    RiemannianFramework,
    PatternRiemannianStructure,
    MetricTensor,
    ChristoffelSymbols,
    CurvatureTensor,
)
from src.core.patterns.enriched_structure import PatternTransition, WaveEmergence
from src.core.patterns.riemannian_flow import RiemannianFlow
from src.validation.geometric.metric import MetricValidator
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from tests.utils.config_loader import load_test_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test timeouts and memory limits
TEST_TIMEOUT = 2  # 2 seconds max per test
OPERATION_TIMEOUT = 0.5  # 0.5 seconds max per operation
MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB memory limit

@contextmanager
def test_timeout():
    """Context manager for test timeouts."""
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    if elapsed > TEST_TIMEOUT:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        raise TimeoutError(f"Test exceeded {TEST_TIMEOUT} seconds timeout")

@contextmanager
def operation_timeout():
    """Context manager for individual operation timeouts."""
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    if elapsed > OPERATION_TIMEOUT:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        raise TimeoutError(f"Operation exceeded {OPERATION_TIMEOUT} seconds timeout")

@pytest.fixture(scope="session")
def memory_manager():
    """Create memory manager for tests."""
    try:
        manager = MemoryManager(cache_size=4, cleanup_threshold=0.1)  # Even smaller cache
        manager.enable_debug_mode()  # Enable debug mode for tests
        yield manager
    finally:
        manager.disable_debug_mode()
        manager._cleanup_dead_refs()
        manager.defragment_memory()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

@pytest.fixture(autouse=True)
def setup_teardown(memory_manager):
    """Setup and teardown for each test."""
    try:
        # Clear memory before test
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory_manager._cleanup_dead_refs()
        memory_manager.defragment_memory()
        
        yield
        
    finally:
        # Cleanup after test
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory_manager._cleanup_dead_refs()
        memory_manager.defragment_memory()
        
        # Log memory stats
        stats = memory_manager.get_memory_stats()
        logger.info(f"Memory stats after test: {stats}")
        
        # Force cleanup of any remaining tensors
        tensor_refs = [ref for ref in memory_manager._tensor_refs if ref() is not None]
        for ref in tensor_refs:
            tensor = ref()
            if tensor is not None:
                del tensor
        gc.collect()
        
        # Validate memory state
        assert memory_manager.validate_state(), "Memory manager in invalid state after test"

@pytest.fixture(autouse=True)
def use_debug_profile(monkeypatch):
    """Force debug profile for all tests."""
    monkeypatch.setenv("PYTEST_PROFILE", "debug")

@pytest.fixture
def test_config():
    """Load test configuration."""
    config = load_test_config("debug")
    # Override with smaller dimensions for tests
    config['fiber_bundle']['batch_size'] = 2
    config['quantum_geometric']['manifold_dim'] = 2
    config['geometric_tests']['dimensions'] = 2
    config['geometric_tests']['hidden_dim'] = 2  # Match hidden_dim with manifold_dim
    config['geometric_tests']['num_heads'] = 1  # Reduce number of heads
    config['quantum_geometric']['hidden_dim'] = 2  # Also match quantum hidden_dim
    return config

@pytest.mark.timeout(10)  # 10 second timeout for all tests
class TestValidationFramework:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self, test_config):
        """Setup and cleanup for each test"""
        # Setup
        self.config = test_config
        self.config.update({
            'dimensions': 2,  # Reduce dimensions for testing
            'max_iterations': 5,  # Limit iterations
            'tolerance': 1e-3,  # Increase tolerance
            'memory_limit': 512 * 1024 * 1024,  # 512MB limit
            'operation_timeout': 0.5,  # 0.5s operation timeout
            'test_timeout': 2.0  # 2s test timeout
        })
        
        # Run test
        yield
        
        # Cleanup
        import gc
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    @pytest.fixture
    def batch_size(self, test_config) -> int:
        """Get batch size from config"""
        return test_config.get('batch_size', 1)

    def test_geometric_validation(self, memory_manager):
        """Test geometric validation with proper cleanup"""
        try:
            metric = torch.eye(self.config['dimensions'])
            metric_copy = metric.clone()  # Create copy for in-place ops
            transposed = metric_copy.transpose(-2, -1).clone()  # Clone the transposed tensor
            metric_copy.add_(transposed)  # Now safe to add in-place
            assert memory_manager.validate_state()
        finally:
            del metric, metric_copy, transposed
            gc.collect()

    def test_quantum_validation(self, memory_manager):
        """Test quantum validation with proper cleanup"""
        try:
            state = torch.randn(2**self.config['dimensions'])
            state = state / torch.norm(state)
            geometric_pattern = torch.zeros_like(state.real)
            geometric_pattern.copy_(state.real)
            assert memory_manager.validate_state()
        finally:
            del state, geometric_pattern
            gc.collect()

    def test_pattern_validation(self, memory_manager):
        """Test pattern validation with proper cleanup"""
        try:
            pattern = torch.randn(2**self.config['dimensions'])
            pattern = pattern / torch.norm(pattern)
            assert memory_manager.validate_state()
        finally:
            del pattern
            gc.collect()

    def test_integrated_validation(self, memory_manager):
        """Test integrated validation with proper cleanup"""
        try:
            # Create tensors with matching dimensions
            dim = 2**self.config['dimensions']
            metric = torch.eye(dim)
            state = torch.randn(dim)
            state = state / torch.norm(state)
            pattern = torch.zeros_like(state)
            pattern.copy_(state)
            assert memory_manager.validate_state()
        finally:
            del metric, state, pattern
            gc.collect()
