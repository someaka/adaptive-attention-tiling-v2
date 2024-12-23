"""Tests for geometric structures and operations in hyperbolic space.

This module tests:
- Minkowski inner product calculations
- Hyperbolic exponential and logarithm maps
- Parallel transport operations
- Geometric structure initialization and operations
"""

import gc
import torch
import pytest
from src.core.attention.geometric import (
    GeometricStructures,
    HyperbolicExponential,
    HyperbolicLogarithm,
    ParallelTransport
)

def cleanup_tensors():
    """Clean up tensors and free memory."""
    gc.collect()

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    cleanup_tensors()

@pytest.fixture
def dim():
    return 2

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def geometric_structures(dim):
    struct = GeometricStructures(
        dim=dim,
        num_heads=1,  # Reduced from 8
        manifold_type="hyperbolic",
        curvature=-1.0
    )
    yield struct
    del struct
    cleanup_tensors()

@pytest.fixture
def hyperbolic_exp(dim):
    exp = HyperbolicExponential(dim)
    yield exp
    del exp
    cleanup_tensors()

@pytest.fixture
def hyperbolic_log(dim):
    log = HyperbolicLogarithm(dim)
    yield log
    del log
    cleanup_tensors()

@pytest.mark.dependency(name="test_minkowski_inner_product")
@pytest.mark.order(1)
@pytest.mark.geometric
@pytest.mark.level1
def test_minkowski_inner_product(hyperbolic_exp, dim, batch_size):
    """Test Minkowski inner product computation. Level 1: Depends on basic tensor operations."""
    # Create test vectors with explicit cleanup
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
    
    # Cleanup
    del x, y, inner, time_part, space_part, expected
    cleanup_tensors()

@pytest.mark.dependency(name="test_project_to_hyperboloid", depends=["test_minkowski_inner_product"])
@pytest.mark.order(2)
@pytest.mark.geometric
@pytest.mark.level1
def test_project_to_hyperboloid(hyperbolic_exp, dim, batch_size):
    """Test projection onto hyperboloid. Level 1: Depends on basic tensor operations."""
    # Create test points with explicit cleanup
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
    
    # Cleanup
    del x, x_proj, inner
    cleanup_tensors()

@pytest.mark.dependency(name="test_exp_log_inverse", depends=["test_minkowski_inner_product", "test_project_to_hyperboloid"])
@pytest.mark.order(3)
@pytest.mark.geometric
@pytest.mark.level2
def test_exp_log_inverse(hyperbolic_exp, hyperbolic_log, dim, batch_size):
    """Test that exp and log are inverse operations. Level 2: Depends on projection and inner product."""
    # Create test point and tangent vector with smaller magnitudes
    x = torch.randn(batch_size, dim) * 0.1  # Scale down the base point
    v = torch.randn(batch_size, dim) * 0.1  # Scale down the tangent vector
    
    # Project x to hyperboloid and v to tangent space
    x = hyperbolic_exp.project_to_hyperboloid(x)
    v = hyperbolic_exp.project_to_tangent(x, v)
    
    # Print intermediate values for debugging
    print(f"\nOriginal vector v: {v}")
    
    # Apply exp then log
    y = hyperbolic_exp(x, v)
    print(f"Point after exp map y: {y}")
    
    v_recovered = hyperbolic_log(x, y)
    print(f"Recovered vector v: {v_recovered}")
    print(f"Difference: {torch.abs(v - v_recovered)}")
    
    # Test recovery of tangent vector with looser tolerances
    assert torch.allclose(v, v_recovered, rtol=1e-3, atol=1e-3)

@pytest.mark.dependency(name="test_parallel_transport", depends=["test_minkowski_inner_product", "test_project_to_hyperboloid"])
@pytest.mark.order(4)
@pytest.mark.geometric
@pytest.mark.level2
def test_parallel_transport(geometric_structures, dim, batch_size):
    """Test parallel transport operations. Level 2: Depends on projection and inner product."""
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

@pytest.mark.dependency(name="test_geodesic_distance", depends=["test_minkowski_inner_product", "test_project_to_hyperboloid"])
@pytest.mark.order(5)
@pytest.mark.geometric
@pytest.mark.level2
def test_geodesic_distance(geometric_structures, dim, batch_size):
    """Test computation of geodesic distances. Level 2: Depends on inner product and projection."""
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
    print("Forward distance:", dist)
    print("Reverse distance:", dist_reverse)
    print("Difference:", torch.abs(dist - dist_reverse))
    assert torch.allclose(dist, dist_reverse, rtol=1e-5)
    
    # Test identity
    dist_same = geometric_structures.compute_geodesic_distance(x, x)
    print("Identity distance:", dist_same)
    assert torch.allclose(dist_same, torch.zeros_like(dist_same), atol=1e-5)
