"""Test configuration and shared fixtures."""

import pytest
import os
import torch
import gc
import multiprocessing as mp
from src.core.flow.neural import NeuralGeometricFlow
import yaml

@pytest.fixture(scope="session", autouse=True)
def cleanup_processes():
    """Cleanup multiprocessing resources after all tests."""
    yield
    # Force cleanup of any remaining process pools
    mp.get_context('spawn').Pool(1).close()
    gc.collect()

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    gc.collect()


@pytest.fixture
def manifold_dim():
    """Manifold dimension for tests."""
    return 4

@pytest.fixture
def hidden_dim(manifold_dim):
    """Hidden dimension for tests."""
    return manifold_dim * 2

@pytest.fixture
def flow(manifold_dim, hidden_dim, test_config):
    """Create flow system fixture."""
    return NeuralGeometricFlow(
        manifold_dim=manifold_dim,
        hidden_dim=hidden_dim,
        dt=0.1,
        stability_threshold=1e-6,
        fisher_rao_weight=1.0,
        quantum_weight=1.0,
        num_heads=8,
        dropout=0.1,
        test_config=test_config
    )

@pytest.fixture
def points(batch_size, manifold_dim):
    """Create random points in position space."""
    return torch.randn(batch_size, manifold_dim, requires_grad=True)

@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 10

@pytest.fixture
def test_config():
    """Load test configuration based on environment."""
    config_name = os.environ.get("TEST_REGIME", "debug")
    config_path = f"configs/test_regimens/{config_name}.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "tracing: mark test as a tracing test"
    )