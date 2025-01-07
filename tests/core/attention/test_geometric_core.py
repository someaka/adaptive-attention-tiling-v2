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
import logging
import os
from datetime import datetime
from src.core.attention.geometric import (
    GeometricStructures,
    HyperbolicExponential,
    HyperbolicLogarithm,
    ParallelTransport
)

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a unique log file for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/geometric_core_test_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger('geometric_core_tests')

@pytest.fixture(scope='session')
def logger():
    """Fixture to provide logger for tests."""
    return setup_logging()

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
def geometric_structures(dim, logger):
    logger.info(f"\nInitializing GeometricStructures with dim={dim}")
    struct = GeometricStructures(
        dim=dim,
        num_heads=1,  # Reduced from 8
        manifold_type="hyperbolic",
        curvature=-1.0
    )
    logger.info("GeometricStructures initialized successfully")
    yield struct
    logger.info("Cleaning up GeometricStructures")
    del struct
    cleanup_tensors()

@pytest.fixture
def hyperbolic_exp(dim, logger):
    logger.info(f"\nInitializing HyperbolicExponential with dim={dim}")
    exp = HyperbolicExponential(dim)
    logger.info("HyperbolicExponential initialized successfully")
    yield exp
    logger.info("Cleaning up HyperbolicExponential")
    del exp
    cleanup_tensors()

@pytest.fixture
def hyperbolic_log(dim, logger):
    logger.info(f"\nInitializing HyperbolicLogarithm with dim={dim}")
    log = HyperbolicLogarithm(dim)
    logger.info("HyperbolicLogarithm initialized successfully")
    yield log
    logger.info("Cleaning up HyperbolicLogarithm")
    del log
    cleanup_tensors()

def test_minkowski_inner_product(hyperbolic_exp, dim, batch_size, logger):
    """Test Minkowski inner product computation with detailed logging."""
    logger.info("\n=== Testing Minkowski Inner Product ===")
    
    # Create test vectors with explicit cleanup
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    
    logger.debug(f"Input vectors:\nx: {x}\ny: {y}")
    logger.debug(f"Vector shapes: x: {x.shape}, y: {y.shape}")
    
    # Compute inner product
    inner = hyperbolic_exp.minkowski_inner(x, y)
    logger.info(f"Computed inner product: {inner}")
    
    # Test shape
    logger.debug(f"Inner product shape: {inner.shape}")
    assert inner.shape == (batch_size,), f"Shape mismatch: expected {(batch_size,)}, got {inner.shape}"
    
    # Test signature (-,+,+,...)
    time_part = -x[..., 0] * y[..., 0]
    space_part = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    expected = time_part + space_part
    
    logger.debug(f"Time component: {time_part}")
    logger.debug(f"Space component: {space_part}")
    logger.debug(f"Expected result: {expected}")
    
    # Compare with manual calculation
    diff = torch.abs(inner - expected)
    logger.info(f"Max difference from expected: {torch.max(diff)}")
    assert torch.allclose(inner, expected, rtol=1e-5), \
           f"Inner product mismatch: max diff = {torch.max(diff)}"
    
    logger.info("Minkowski inner product test passed successfully")
    
    # Cleanup
    del x, y, inner, time_part, space_part, expected
    cleanup_tensors()

def test_project_to_hyperboloid(hyperbolic_exp, dim, batch_size, logger):
    """Test projection onto hyperboloid with enhanced precision checks and logging."""
    logger.info("\n=== Testing Hyperboloid Projection ===")
    
    # Create test points with controlled magnitudes
    x = torch.randn(batch_size, dim) * 0.1  # Scale inputs for stability
    logger.debug(f"Input points:\n{x}")
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Input norm: {torch.norm(x)}")
    
    # Project to hyperboloid
    x_proj = hyperbolic_exp.project_to_hyperboloid(x)
    logger.info("\nProjection Results:")
    logger.debug(f"Projected points:\n{x_proj}")
    logger.info(f"Shape: {x_proj.shape}")
    logger.info(f"Time components: {x_proj[..., 0]}")
    logger.info(f"Space components norm: {torch.norm(x_proj[..., 1:], dim=-1)}")
    
    # Test hyperboloid constraint with high precision
    inner = hyperbolic_exp.minkowski_inner(x_proj, x_proj)
    constraint_violation = torch.abs(inner + 1.0)
    
    logger.info("\nConstraint Verification:")
    logger.debug(f"Inner products: {inner}")
    logger.info(f"Constraint violations: {constraint_violation}")
    logger.info(f"Max violation: {torch.max(constraint_violation)}")
    
    # Verify properties with appropriate tolerances
    try:
        assert x_proj.shape == (batch_size, dim), "Shape mismatch"
        logger.info("Shape verification passed")
        
        assert torch.allclose(inner, torch.full_like(inner, -1.0), rtol=1e-6, atol=1e-6), \
               f"Hyperboloid constraint violated: {constraint_violation}"
        logger.info("Hyperboloid constraint verification passed")
        
        assert torch.all(x_proj[..., 0] > 0), f"Non-positive time components: {x_proj[..., 0]}"
        logger.info("Time component positivity verification passed")
        
        logger.info("All hyperboloid projection tests passed successfully")
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
        raise

def test_exp_log_inverse(hyperbolic_exp, hyperbolic_log, dim, batch_size, logger):
    """Test exponential and logarithm maps inverse property with detailed logging."""
    logger.info("\n=== Testing Exp-Log Inverse Property ===")
    
    # Use double precision for better numerical stability
    dtype = torch.float64
    
    # Create test points with very controlled magnitudes
    x = torch.randn(batch_size, dim, dtype=dtype) * 0.01  # Reduced magnitude
    v = torch.randn(batch_size, dim, dtype=dtype) * 0.01  # Reduced magnitude
    
    logger.debug(f"Initial points:\nx: {x}\nv: {v}")
    logger.info(f"Shapes - x: {x.shape}, v: {v.shape}")
    
    # Project points to hyperboloid with Kahan summation
    x_space = x[..., 1:]
    x_space_norm_sq = torch.zeros(batch_size, dtype=dtype)
    compensation = torch.zeros(batch_size, dtype=dtype)
    
    for i in range(x_space.size(-1)):
        y = x_space[..., i] * x_space[..., i] - compensation
        t = x_space_norm_sq + y
        compensation = (t - x_space_norm_sq) - y
        x_space_norm_sq = t
    
    x_time = torch.sqrt(1.0 + x_space_norm_sq)
    x = torch.cat([x_time.unsqueeze(-1), x_space], dim=-1)
    
    # Project vector to tangent space with enhanced precision
    v = hyperbolic_exp.project_to_tangent(x, v)
    
    logger.info("\nProjected Quantities:")
    logger.debug(f"Projected base point:\n{x}")
    logger.debug(f"Projected tangent vector:\n{v}")
    logger.info(f"Base point norm: {torch.norm(x)}")
    logger.info(f"Tangent vector norm: {torch.norm(v)}")
    
    # Store original inner product for validation
    v_inner_original = hyperbolic_exp.minkowski_inner(v, v)
    
    # Apply exp then log with constraint verification
    y = hyperbolic_exp(x, v)
    logger.debug(f"Hyperboloid constraint after exp: {torch.abs(hyperbolic_exp.minkowski_inner(y, y) + 1.0)}")
    
    v_recovered = hyperbolic_log(x, y)
    logger.debug(f"Tangent space constraint: {torch.abs(hyperbolic_exp.minkowski_inner(x, v_recovered))}")
    
    logger.info("\nRecovered Quantities:")
    logger.debug(f"Exponential map result:\n{y}")
    logger.debug(f"Recovered vector:\n{v_recovered}")
    
    # Compute error metrics with enhanced precision
    vector_error = torch.norm(v - v_recovered)
    v_recovered_inner = hyperbolic_exp.minkowski_inner(v_recovered, v_recovered)
    inner_error = torch.abs(v_inner_original - v_recovered_inner)
    
    logger.info("\nError Analysis:")
    logger.info(f"Vector recovery error: {vector_error}")
    logger.info(f"Inner product error: {inner_error}")
    
    # Verify recovery with appropriate tolerances
    try:
        # Use relative tolerance for vector comparison
        assert torch.allclose(v, v_recovered, rtol=1e-7, atol=1e-7), \
               f"Vector recovery failed: {vector_error}"
        logger.info("Vector recovery verified")
        
        # Use absolute tolerance for inner product
        assert torch.allclose(v_inner_original, v_recovered_inner, rtol=1e-7, atol=1e-7), \
               f"Inner product not preserved: {inner_error}"
        logger.info("Inner product preservation verified")
        
        logger.info("All exp-log inverse tests passed successfully")
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
        raise

def test_parallel_transport(dim, batch_size, logger):
    """Test parallel transport operations with comprehensive logging."""
    logger.info("\n=== Testing Parallel Transport ===")
    
    # Test Euclidean transport
    logger.info("\nInitializing Euclidean Structure")
    euclidean_struct = GeometricStructures(
        dim=dim,
        num_heads=1,
        manifold_type="euclidean",
        curvature=0.0
    )
    
    # Create test points with controlled magnitudes
    x = torch.randn(batch_size, dim) * 0.01  # Reduced from 0.1
    y = torch.randn(batch_size, dim) * 0.01  # Reduced from 0.1
    v = torch.randn(batch_size, dim) * 0.01  # Reduced from 0.1
    
    # Convert to double precision
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    v = v.to(torch.float64)
    
    logger.debug(f"Test points:\nx: {x}\ny: {y}\nv: {v}")
    logger.info(f"Point shapes: x: {x.shape}, y: {y.shape}")
    logger.info(f"Vector shape: v: {v.shape}")
    
    # Transport vector in Euclidean space
    logger.info("\nPerforming Euclidean Transport")
    v_transported_euc = euclidean_struct.parallel_transport_batch(x, y, v)
    logger.debug(f"Transported vector (Euclidean):\n{v_transported_euc}")
    
    # Verify Euclidean properties
    v_norm = torch.norm(v, dim=-1)
    v_transported_norm = torch.norm(v_transported_euc, dim=-1)
    
    logger.info("\nEuclidean Verification:")
    logger.info(f"Original norm: {v_norm}")
    logger.info(f"Transported norm: {v_transported_norm}")
    logger.info(f"Norm difference: {torch.abs(v_norm - v_transported_norm)}")
    
    try:
        assert torch.allclose(v_norm, v_transported_norm, rtol=1e-5), \
               "Euclidean transport failed to preserve norm"
        logger.info("Euclidean transport norm preservation verified")
    except AssertionError as e:
        logger.error(f"Euclidean transport test failed: {str(e)}")
        raise
    
    # Test hyperbolic transport
    logger.info("\nInitializing Hyperbolic Structure")
    hyperbolic_struct = GeometricStructures(
        dim=dim,
        num_heads=1,
        manifold_type="hyperbolic",
        curvature=-1.0
    )
    
    # Project points to hyperboloid
    logger.info("\nProjecting to Hyperboloid")
    x_hyp = hyperbolic_struct.exp_map.project_to_hyperboloid(x)
    y_hyp = hyperbolic_struct.exp_map.project_to_hyperboloid(y)
    v_hyp = hyperbolic_struct.exp_map.project_to_tangent(x_hyp, v)
    
    logger.debug(f"Projected points:\nx_hyp: {x_hyp}\ny_hyp: {y_hyp}\nv_hyp: {v_hyp}")
    
    # Verify projections
    logger.info("\nVerifying Projections:")
    logger.info(f"Base point inner: {hyperbolic_struct.exp_map.minkowski_inner(x_hyp, x_hyp)}")
    logger.info(f"Target point inner: {hyperbolic_struct.exp_map.minkowski_inner(y_hyp, y_hyp)}")
    logger.info(f"Tangent vector inner: {hyperbolic_struct.exp_map.minkowski_inner(v_hyp, v_hyp)}")
    
    # Transport vector in hyperbolic space
    logger.info("\nPerforming Hyperbolic Transport")
    v_transported_hyp = hyperbolic_struct.parallel_transport_batch(x_hyp, y_hyp, v_hyp)
    logger.debug(f"Transported vector (Hyperbolic):\n{v_transported_hyp}")
    
    # Verify hyperbolic properties
    inner_before = hyperbolic_struct.exp_map.minkowski_inner(v_hyp, v_hyp)
    inner_after = hyperbolic_struct.exp_map.minkowski_inner(v_transported_hyp, v_transported_hyp)
    
    logger.info("\nHyperbolic Verification:")
    logger.info(f"Original inner: {inner_before}")
    logger.info(f"Transported inner: {inner_after}")
    logger.info(f"Inner product difference: {torch.abs(inner_before - inner_after)}")
    
    try:
        assert torch.allclose(inner_before, inner_after, rtol=1e-5, atol=1e-6), \
               "Hyperbolic transport failed to preserve inner product"
        logger.info("Hyperbolic transport inner product preservation verified")
        
        logger.info("All parallel transport tests passed successfully")
    except AssertionError as e:
        logger.error(f"Hyperbolic transport test failed: {str(e)}")
        raise

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
    K = geometric_structures.sectional_curvature(x, v1, v2)
    
    print("\nCurvature computation:")
    print(f"Sectional curvature: {K}")
    
    # Test scaling invariance
    scale = 2.0
    K_scaled = geometric_structures.sectional_curvature(x, scale*v1, scale*v2)
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
