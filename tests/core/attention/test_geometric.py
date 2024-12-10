"""Tests for geometric structures and operations in hyperbolic space.

This module tests:
- Minkowski inner product calculations
- Hyperbolic exponential and logarithm maps
- Parallel transport operations
- Geometric structure initialization and operations
"""

import torch
import pytest
from src.core.attention.geometric import (
    GeometricStructures,
    HyperbolicExponential,
    HyperbolicLogarithm,
    ParallelTransport
)

@pytest.fixture
def dim():
    return 4

@pytest.fixture
def batch_size():
    return 8

@pytest.fixture
def geometric_structures(dim):
    return GeometricStructures(
        dim=dim,
        num_heads=8,
        manifold_type="hyperbolic",
        curvature=-1.0
    )

@pytest.fixture
def hyperbolic_exp(dim):
    return HyperbolicExponential(dim)

@pytest.fixture
def hyperbolic_log(dim):
    return HyperbolicLogarithm(dim)

def test_minkowski_inner_product(hyperbolic_exp, dim, batch_size):
    """Test Minkowski inner product computation."""
    # Create test vectors
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    
    # Compute inner product
    inner = hyperbolic_exp.minkowski_inner(x, y)
    
    # Test shape
    assert inner.shape == (batch_size,)
    
    # Test signature (-,+,+,...)
    time_part = -x[..., 0] * y[..., 0]
    space_part = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    expected = time_part + space_part
    
    # Compare with manual calculation
    assert torch.allclose(inner, expected, rtol=1e-5)

def test_project_to_hyperboloid(hyperbolic_exp, dim, batch_size):
    """Test projection onto hyperboloid."""
    # Create test points
    x = torch.randn(batch_size, dim)
    
    # Project to hyperboloid
    x_proj = hyperbolic_exp.project_to_hyperboloid(x)
    
    # Test shape
    assert x_proj.shape == (batch_size, dim)
    
    # Test that points lie on hyperboloid (<x,x> = -1)
    inner = hyperbolic_exp.minkowski_inner(x_proj, x_proj)
    assert torch.allclose(inner, torch.full_like(inner, -1.0), rtol=1e-5)
    
    # Test that time component is positive
    assert torch.all(x_proj[..., 0] > 0)

def test_exp_log_inverse(hyperbolic_exp, hyperbolic_log, dim, batch_size):
    """Test that exp and log are inverse operations."""
    # Create test point and tangent vector
    x = torch.randn(batch_size, dim)
    v = torch.randn(batch_size, dim)
    
    # Project x to hyperboloid and v to tangent space
    x = hyperbolic_exp.project_to_hyperboloid(x)
    v = hyperbolic_exp.project_to_tangent(x, v)
    
    # Apply exp then log
    y = hyperbolic_exp(x, v)
    v_recovered = hyperbolic_log(x, y)
    
    # Test recovery of tangent vector
    assert torch.allclose(v, v_recovered, rtol=1e-4, atol=1e-4)

def test_parallel_transport(geometric_structures, dim, batch_size):
    """Test parallel transport operations."""
    # Create test points and vector
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    v = torch.randn(batch_size, dim)
    
    # Transport vector
    v_transported = geometric_structures.parallel_transport_batch(x, y, v)
    
    # Test shape
    assert v_transported.shape == (batch_size, dim)
    
    # Test that transport preserves norm (approximately)
    v_norm = torch.norm(v, dim=-1)
    v_transported_norm = torch.norm(v_transported, dim=-1)
    assert torch.allclose(v_norm, v_transported_norm, rtol=1e-4)

def test_geodesic_distance(geometric_structures, dim, batch_size):
    """Test computation of geodesic distances."""
    # Create test points
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    
    # Compute distance
    dist = geometric_structures.compute_geodesic_distance(x, y)
    
    # Test shape and positivity
    assert dist.shape == (batch_size,)
    assert torch.all(dist >= 0)
    
    # Test symmetry
    dist_reverse = geometric_structures.compute_geodesic_distance(y, x)
    assert torch.allclose(dist, dist_reverse, rtol=1e-5)
    
    # Test identity
    dist_same = geometric_structures.compute_geodesic_distance(x, x)
    assert torch.allclose(dist_same, torch.zeros_like(dist_same), atol=1e-5)
