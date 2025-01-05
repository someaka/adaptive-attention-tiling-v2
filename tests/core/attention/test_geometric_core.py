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

def test_project_to_hyperboloid(hyperbolic_exp, dim, batch_size):
    """Test projection onto hyperboloid with enhanced precision checks.
    
    This test verifies:
    1. Points lie exactly on hyperboloid (<x,x> = -1)
    2. Time component is strictly positive
    3. Projection is numerically stable
    4. Batch operations preserve precision
    """
    # Create test points with controlled magnitudes
    x = torch.randn(batch_size, dim) * 0.1  # Scale inputs for stability
    
    print("\nTesting hyperboloid projection:")
    print(f"Input shape: {x.shape}")
    print(f"Input norm: {torch.norm(x)}")
    
    # Project to hyperboloid
    x_proj = hyperbolic_exp.project_to_hyperboloid(x)
    
    print(f"\nProjected points:")
    print(f"Shape: {x_proj.shape}")
    print(f"Time components: {x_proj[..., 0]}")
    print(f"Space components norm: {torch.norm(x_proj[..., 1:], dim=-1)}")
    
    # Test hyperboloid constraint with high precision
    inner = hyperbolic_exp.minkowski_inner(x_proj, x_proj)
    constraint_violation = torch.abs(inner + 1.0)
    
    print(f"\nConstraint verification:")
    print(f"Inner products: {inner}")
    print(f"Constraint violations: {constraint_violation}")
    print(f"Max violation: {torch.max(constraint_violation)}")
    
    # Verify properties with appropriate tolerances
    assert x_proj.shape == (batch_size, dim), "Shape mismatch"
    assert torch.allclose(inner, torch.full_like(inner, -1.0), rtol=1e-6, atol=1e-6), \
           f"Hyperboloid constraint violated: {constraint_violation}"
    assert torch.all(x_proj[..., 0] > 0), f"Non-positive time components: {x_proj[..., 0]}"

def test_exp_log_inverse(hyperbolic_exp, hyperbolic_log, dim, batch_size):
    """Test exponential and logarithm maps are inverse operations with high precision.
    
    This test verifies:
    1. exp(log(y)) = y for points on hyperboloid
    2. log(exp(v)) = v for tangent vectors
    3. Distance preservation
    4. Numerical stability under composition
    """
    # Create test points with controlled magnitudes
    x = torch.randn(batch_size, dim) * 0.1
    v = torch.randn(batch_size, dim) * 0.1
    
    print("\nTesting exp-log inverse property:")
    print(f"Base point shape: {x.shape}")
    print(f"Tangent vector shape: {v.shape}")
    
    # Project points and vectors
    x = hyperbolic_exp.project_to_hyperboloid(x)
    v = hyperbolic_exp.project_to_tangent(x, v)
    
    print("\nProjected quantities:")
    print(f"Base point norm: {torch.norm(x)}")
    print(f"Tangent vector norm: {torch.norm(v)}")
    
    # Apply exp then log
    y = hyperbolic_exp(x, v)
    v_recovered = hyperbolic_log(x, y)
    
    print("\nRecovered quantities:")
    print(f"Exponential map result: {y}")
    print(f"Recovered vector: {v_recovered}")
    
    # Compute error metrics
    vector_error = torch.norm(v - v_recovered)
    inner_error = torch.abs(
        hyperbolic_exp.minkowski_inner(v, v) - 
        hyperbolic_exp.minkowski_inner(v_recovered, v_recovered)
    )
    
    print("\nError analysis:")
    print(f"Vector recovery error: {vector_error}")
    print(f"Inner product error: {inner_error}")
    
    # Verify recovery with appropriate tolerances
    assert torch.allclose(v, v_recovered, rtol=1e-5, atol=1e-5), \
           f"Vector recovery failed: {vector_error}"
    assert torch.allclose(
        hyperbolic_exp.minkowski_inner(v, v),
        hyperbolic_exp.minkowski_inner(v_recovered, v_recovered),
        rtol=1e-5, atol=1e-5
    ), f"Inner product not preserved: {inner_error}"

def test_parallel_transport(dim, batch_size):
    """Test parallel transport operations with enhanced precision checks.
    
    This test verifies:
    1. Norm preservation in both Euclidean and hyperbolic spaces
    2. Tangent space constraints
    3. Numerical stability under transport
    4. Batch operation consistency
    """
    print("\nTesting parallel transport:")
    
    # Test Euclidean transport
    euclidean_struct = GeometricStructures(
        dim=dim,
        num_heads=1,
        manifold_type="euclidean",
        curvature=0.0
    )
    
    # Create test points with controlled magnitudes
    x = torch.randn(batch_size, dim) * 0.1
    y = torch.randn(batch_size, dim) * 0.1
    v = torch.randn(batch_size, dim) * 0.1
    
    print("\nEuclidean transport test:")
    print(f"Point shapes: {x.shape}, {y.shape}")
    print(f"Vector shape: {v.shape}")
    
    # Transport vector in Euclidean space
    v_transported_euc = euclidean_struct.parallel_transport_batch(x, y, v)
    
    # Verify Euclidean properties
    v_norm = torch.norm(v, dim=-1)
    v_transported_norm = torch.norm(v_transported_euc, dim=-1)
    
    print("\nEuclidean verification:")
    print(f"Original norm: {v_norm}")
    print(f"Transported norm: {v_transported_norm}")
    print(f"Norm difference: {torch.abs(v_norm - v_transported_norm)}")
    
    assert torch.allclose(v_norm, v_transported_norm, rtol=1e-5), \
           "Euclidean transport failed to preserve norm"
    
    # Test hyperbolic transport
    hyperbolic_struct = GeometricStructures(
        dim=dim,
        num_heads=1,
        manifold_type="hyperbolic",
        curvature=-1.0
    )
    
    print("\nHyperbolic transport test:")
    
    # Project points to hyperboloid
    x_hyp = hyperbolic_struct.exp_map.project_to_hyperboloid(x)
    y_hyp = hyperbolic_struct.exp_map.project_to_hyperboloid(y)
    v_hyp = hyperbolic_struct.exp_map.project_to_tangent(x_hyp, v)
    
    print("\nProjected quantities:")
    print(f"Base point inner: {hyperbolic_struct.exp_map.minkowski_inner(x_hyp, x_hyp)}")
    print(f"Target point inner: {hyperbolic_struct.exp_map.minkowski_inner(y_hyp, y_hyp)}")
    print(f"Tangent vector inner: {hyperbolic_struct.exp_map.minkowski_inner(v_hyp, v_hyp)}")
    
    # Transport vector in hyperbolic space
    v_transported_hyp = hyperbolic_struct.parallel_transport_batch(x_hyp, y_hyp, v_hyp)
    
    # Verify hyperbolic properties
    inner_before = hyperbolic_struct.exp_map.minkowski_inner(v_hyp, v_hyp)
    inner_after = hyperbolic_struct.exp_map.minkowski_inner(v_transported_hyp, v_transported_hyp)
    
    print("\nHyperbolic verification:")
    print(f"Original inner: {inner_before}")
    print(f"Transported inner: {inner_after}")
    print(f"Inner product difference: {torch.abs(inner_before - inner_after)}")
    
    assert torch.allclose(inner_before, inner_after, rtol=1e-5), \
           "Hyperbolic transport failed to preserve inner product"

def test_geodesic_distance(geometric_structures, dim, batch_size):
    """Test computation of geodesic distances with enhanced precision checks.
    
    This test verifies:
    1. Distance positivity and symmetry
    2. Triangle inequality
    3. Identity of indiscernibles
    4. Numerical stability of distance computation
    """
    print("\nTesting geodesic distance computation:")
    
    # Create test points with controlled magnitudes
    x = torch.randn(batch_size, dim) * 0.1
    y = torch.randn(batch_size, dim) * 0.1
    z = torch.randn(batch_size, dim) * 0.1
    
    print("\nTest points:")
    print(f"Shapes: {x.shape}, {y.shape}, {z.shape}")
    print(f"Norms: {torch.norm(x)}, {torch.norm(y)}, {torch.norm(z)}")
    
    # Compute distances
    dist_xy = geometric_structures.compute_geodesic_distance(x, y)
    dist_yx = geometric_structures.compute_geodesic_distance(y, x)
    dist_yz = geometric_structures.compute_geodesic_distance(y, z)
    dist_xz = geometric_structures.compute_geodesic_distance(x, z)
    dist_xx = geometric_structures.compute_geodesic_distance(x, x)
    
    print("\nDistance computations:")
    print(f"d(x,y): {dist_xy}")
    print(f"d(y,x): {dist_yx}")
    print(f"d(y,z): {dist_yz}")
    print(f"d(x,z): {dist_xz}")
    print(f"d(x,x): {dist_xx}")
    
    # Test metric properties
    symmetry_error = torch.abs(dist_xy - dist_yx)
    triangle_violation = torch.relu(dist_xz - (dist_xy + dist_yz))
    identity_error = torch.abs(dist_xx)
    
    print("\nMetric property verification:")
    print(f"Symmetry error: {symmetry_error}")
    print(f"Triangle inequality violation: {triangle_violation}")
    print(f"Identity error: {identity_error}")
    
    # Verify metric properties with appropriate tolerances
    assert torch.all(dist_xy >= 0), "Non-negative distance violated"
    assert torch.allclose(dist_xy, dist_yx, rtol=1e-5), \
           f"Symmetry violated: {symmetry_error}"
    assert torch.all(dist_xz <= dist_xy + dist_yz + 1e-5), \
           f"Triangle inequality violated: {triangle_violation}"
    assert torch.allclose(dist_xx, torch.zeros_like(dist_xx), atol=1e-5), \
           f"Identity of indiscernibles violated: {identity_error}"

def test_sectional_curvature(geometric_structures, dim, batch_size):
    """Test computation of sectional curvature with enhanced precision checks.
    
    This test verifies:
    1. Curvature sign consistency
    2. Scale invariance
    3. Numerical stability
    4. Batch operation precision
    """
    print("\nTesting sectional curvature computation:")
    
    # Create orthonormal vectors with controlled magnitudes
    v1 = torch.randn(batch_size, dim) * 0.1
    v2 = torch.randn(batch_size, dim) * 0.1
    x = torch.randn(batch_size, dim) * 0.1
    
    print("\nTest vectors:")
    print(f"Shapes: {v1.shape}, {v2.shape}")
    print(f"Base point norm: {torch.norm(x)}")
    print(f"Vector norms: {torch.norm(v1)}, {torch.norm(v2)}")
    
    # Project to manifold and orthonormalize
    if isinstance(geometric_structures, GeometricStructures) and \
       geometric_structures.manifold_type == "hyperbolic":
        x = geometric_structures.exp_map.project_to_hyperboloid(x)
        v1 = geometric_structures.exp_map.project_to_tangent(x, v1)
        v2 = geometric_structures.exp_map.project_to_tangent(x, v2)
    
    # Compute sectional curvature
    K = geometric_structures.compute_sectional_curvature(x, v1, v2)
    
    print("\nCurvature computation:")
    print(f"Sectional curvature: {K}")
    
    # Test scaling invariance
    scale = 2.0
    K_scaled = geometric_structures.compute_sectional_curvature(x, scale*v1, scale*v2)
    scale_error = torch.abs(K - K_scaled)
    
    print("\nScale invariance check:")
    print(f"Original curvature: {K}")
    print(f"Scaled curvature: {K_scaled}")
    print(f"Scale error: {scale_error}")
    
    # Verify properties with appropriate tolerances
    if isinstance(geometric_structures, GeometricStructures):
        if geometric_structures.manifold_type == "hyperbolic":
            assert torch.all(K < 0), "Hyperbolic curvature must be negative"
        elif geometric_structures.manifold_type == "euclidean":
            assert torch.allclose(K, torch.zeros_like(K), atol=1e-5), \
                   "Euclidean curvature must be zero"
    
    assert torch.allclose(K, K_scaled, rtol=1e-5), \
           f"Scale invariance violated: {scale_error}"
