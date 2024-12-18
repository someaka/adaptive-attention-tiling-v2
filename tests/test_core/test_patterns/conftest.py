"""
Fixtures for fiber bundle tests.

This module provides shared fixtures for testing both the base mathematical
implementation and pattern-specific implementation of fiber bundles.
"""

import os
import pytest
import torch
import yaml

from src.core.patterns.fiber_bundle import BaseFiberBundle
from src.core.tiling.patterns.pattern_fiber_bundle import PatternFiberBundle


@pytest.fixture
def test_config():
    """Load test configuration based on environment."""
    config_name = os.environ.get("TEST_REGIME", "debug")
    config_path = f"configs/test_regimens/{config_name}.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def base_manifold(test_config):
    """Create a test base manifold."""
    dim = test_config["geometric_tests"]["dimensions"]
    batch_size = test_config["geometric_tests"]["batch_size"]
    dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
    return torch.randn(batch_size, dim, dtype=dtype)


@pytest.fixture
def fiber_dim():
    """Dimension of the fiber."""
    return 3  # Standard SO(3) fiber dimension


@pytest.fixture
def base_bundle(test_config):
    """Create base implementation instance."""
    dim = test_config["geometric_tests"]["dimensions"]
    fiber_dim = 3  # Standard SO(3) fiber dimension
    total_dim = dim + fiber_dim
    return BaseFiberBundle(
        base_dim=dim,
        fiber_dim=fiber_dim,
        structure_group="SO3"
    )


@pytest.fixture
def pattern_bundle(test_config):
    """Create pattern implementation instance."""
    dim = test_config["geometric_tests"]["dimensions"]
    fiber_dim = 3  # Standard SO(3) fiber dimension
    total_dim = dim + fiber_dim
    return PatternFiberBundle(
        base_dim=dim,
        fiber_dim=fiber_dim,
        structure_group="O(n)",
        device=torch.device("cpu")
    )


@pytest.fixture
def structure_group():
    """Create a structure group for the bundle."""
    return torch.eye(3)  # SO(3) structure group


@pytest.fixture
def dtype(test_config):
    """Get the configured dtype."""
    return getattr(torch, test_config["geometric_tests"]["dtype"])


@pytest.fixture
def batch_size(test_config):
    """Get the configured batch size."""
    return test_config["geometric_tests"]["batch_size"]


@pytest.fixture
def total_space(base_bundle, batch_size, dtype):
    """Create a point in the total space."""
    return torch.randn(batch_size, base_bundle.total_dim, dtype=dtype)


@pytest.fixture
def tangent_vector(base_bundle, batch_size, dtype):
    """Create a tangent vector."""
    return torch.randn(batch_size, base_bundle.total_dim, dtype=dtype)


@pytest.fixture
def test_path(dtype):
    """Create a test path (circular loop)."""
    t = torch.linspace(0, 2 * torch.pi, 100, dtype=dtype)
    return torch.stack([torch.cos(t), torch.sin(t)], dim=1) 