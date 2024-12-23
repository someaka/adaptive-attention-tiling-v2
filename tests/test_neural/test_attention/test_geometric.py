"""
Unit tests for geometric operations and manifold structures.

Tests cover:
1. Geometric structures and tensors
2. Hyperbolic operations
3. Euclidean operations
4. Parallel transport methods
5. Quantum geometric integration
"""

import gc
import numpy as np
import pytest
import torch
import psutil
import os

from src.core.attention.geometric import (
    GeometricStructures,
    HyperbolicExponential,
    HyperbolicLogarithm,
    EuclideanExponential,
    EuclideanLogarithm,
    ParallelTransport,
)
from tests.utils.config_loader import load_test_config


def print_memory_usage(tag: str):
    """Print current memory usage."""
    gc.collect()
    print(f"\n[{tag}]")


def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print detailed information about a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Memory: {tensor.element_size() * tensor.nelement() / 1024:.2f} KB")
    if len(tensor.shape) > 0:
        print(f"  Norm: {torch.norm(tensor)}")
    print(f"  Values: {tensor}")


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    print_memory_usage("Before config load")
    config = load_test_config()
    print_memory_usage("After config load")
    return config


class TestGeometricStructures:
    """Test suite for geometric structures."""

    @pytest.fixture
    def dim(self, test_config) -> int:
        """Return dimension for tests."""
        return test_config["geometric_tests"]["dimensions"]

    @pytest.fixture
    def num_heads(self, test_config) -> int:
        """Return number of attention heads for tests."""
        return test_config["geometric_tests"]["num_heads"]

    @pytest.fixture
    def batch_size(self, test_config) -> int:
        """Return batch size for tests."""
        return test_config["geometric_tests"]["batch_size"]

    @pytest.fixture
    def dtype(self, test_config) -> torch.dtype:
        """Return data type for tests."""
        return getattr(torch, test_config["geometric_tests"]["dtype"])

    @pytest.fixture
    def geometric_structures(self, dim, num_heads, dtype):
        """Create geometric structures for testing."""
        print_memory_usage("Before creating geometric structures")
        struct = GeometricStructures(
            dim=dim,
            num_heads=num_heads,
            manifold_type="hyperbolic",
            curvature=-1.0,
            parallel_transport_method="schild",
        ).to(dtype=dtype)
        print_memory_usage("After creating geometric structures")
        return struct

    def test_metric_initialization(self, geometric_structures, dim):
        """Test metric tensor initialization."""
        print_memory_usage("Start metric test")
        
        metric = geometric_structures.metric
        print(f"\nMetric tensor shape: {metric.shape}")
        print(f"Metric tensor dtype: {metric.dtype}")
        
        assert metric.shape == (dim, dim)
        assert torch.allclose(metric, torch.eye(dim, dtype=metric.dtype))
        
        print_memory_usage("End metric test")

    def test_connection_initialization(self, geometric_structures, dim):
        """Test connection coefficients initialization."""
        print_memory_usage("Start connection test")
        
        connection = geometric_structures.connection
        print(f"\nConnection tensor shape: {connection.shape}")
        print(f"Connection tensor dtype: {connection.dtype}")
        
        assert connection.shape == (dim, dim, dim)
        assert torch.allclose(connection, torch.zeros(dim, dim, dim, dtype=connection.dtype))
        
        print_memory_usage("End connection test")

    def test_curvature_tensor(self, geometric_structures, dim):
        """Test curvature tensor initialization and properties."""
        print_memory_usage("Start curvature test")
        
        # Initialize curvature tensor with non-zero values that satisfy symmetries
        metric = geometric_structures.metric
        print(f"\nInitial metric shape: {metric.shape}")
        
        g_ik = metric.unsqueeze(1).unsqueeze(3)
        g_jl = metric.unsqueeze(0).unsqueeze(2)
        g_il = metric.unsqueeze(1).unsqueeze(2)
        g_jk = metric.unsqueeze(0).unsqueeze(3)
        
        # Set curvature tensor
        K = geometric_structures.curvature
        geometric_structures.curvature_tensor.data = K * (g_ik * g_jl - g_il * g_jk)
        
        # Test symmetries
        anti_sym_12 = geometric_structures.curvature_tensor + geometric_structures.curvature_tensor.permute(1, 0, 2, 3)
        anti_sym_34 = geometric_structures.curvature_tensor + geometric_structures.curvature_tensor.permute(0, 1, 3, 2)
        
        assert torch.allclose(anti_sym_12, torch.zeros_like(anti_sym_12), atol=1e-6)
        assert torch.allclose(anti_sym_34, torch.zeros_like(anti_sym_34), atol=1e-6)
        
        print_memory_usage("End curvature test")

    def test_sectional_curvature(self, geometric_structures, dim):
        """Test sectional curvature computation."""
        print_memory_usage("Start sectional test")
        
        x = torch.randn(dim, dtype=geometric_structures.metric.dtype)
        v1 = torch.randn(dim, dtype=geometric_structures.metric.dtype)
        v2 = torch.randn(dim, dtype=geometric_structures.metric.dtype)
        
        # Make v2 orthogonal to v1
        v2 = v2 - torch.dot(v1, v2) * v1 / torch.dot(v1, v1)
        
        curvature = geometric_structures.compute_sectional_curvature(x, v1, v2)
        
        assert curvature.shape == ()  # Scalar output
        assert not torch.isnan(curvature)
        
        print_memory_usage("End sectional test")

    def test_geodesic_distance(self, geometric_structures, dim):
        """Test geodesic distance computation."""
        print_memory_usage("Start geodesic test")
        
        x = torch.randn(dim, dtype=geometric_structures.metric.dtype)
        x = x / (torch.norm(x) + 1e-8)  # Normalize with epsilon
        
        # Test distance to self
        self_distance = geometric_structures.compute_geodesic_distance(x, x)
        assert torch.allclose(self_distance, torch.tensor(0.0), atol=1e-3)
        
        # Test with different point
        y = torch.randn(dim, dtype=geometric_structures.metric.dtype)
        y = y / (torch.norm(y) + 1e-8)
        
        distance = geometric_structures.compute_geodesic_distance(x, y)
        assert distance.shape == ()
        assert distance >= 0
        
        print_memory_usage("End geodesic test")


class TestHyperbolicOperations:
    """Test hyperbolic geometric operations."""
    
    @pytest.fixture
    def test_scales(self, test_config) -> list[float]:
        """Return test scales for vector operations."""
        return test_config["hyperbolic_tests"]["vector_scales"]

    @pytest.fixture
    def test_norms(self, test_config) -> list[float]:
        """Return test norms for vector operations."""
        return test_config["hyperbolic_tests"]["test_norms"]

    @pytest.fixture
    def precision(self, test_config) -> torch.dtype:
        """Return precision for tests."""
        return getattr(torch, test_config["hyperbolic_tests"]["precision"])

    def test_hyperbolic_distance_formula(self, precision):
        """Test the hyperbolic distance formula directly."""
        print_memory_usage("Start hyperbolic distance test")
        
        # Create points in the Poincare ball
        x = torch.tensor([-0.3, 0.04, 0.004], dtype=precision)
        y = torch.tensor([-0.32, 0.042, 0.005], dtype=precision)
        
        # Project to hyperboloid (t² - x² - y² - z² = 1)
        t_x = torch.sqrt(1 + torch.sum(x * x))
        t_y = torch.sqrt(1 + torch.sum(y * y))
        
        x = torch.cat([t_x.unsqueeze(0), x])
        y = torch.cat([t_y.unsqueeze(0), y])
        
        # Compute Minkowski inner product
        inner = -x[0]*y[0] + torch.sum(x[1:] * y[1:])
        
        # Compute distance
        dist = torch.acosh(-inner)
        
        assert dist >= 0
        assert inner <= -1 + 1e-6
        
        print_memory_usage("End hyperbolic distance test")

    def test_exp_map_properties(self, precision):
        """Test mathematical properties of exponential map."""
        print_memory_usage("Start exp map test")
        
        # Create a point on the hyperboloid
        x_spatial = torch.tensor([0.1, -0.2, 0.1], dtype=precision)
        x_t = torch.sqrt(1 + torch.sum(x_spatial * x_spatial))
        x = torch.cat([x_t.unsqueeze(0), x_spatial])
        
        # Create a small tangent vector
        v_spatial = torch.tensor([0.001, -0.002, 0.001], dtype=precision)
        v_t = torch.sum(x[1:] * v_spatial) / x[0]  # Ensure orthogonality
        v = torch.cat([v_t.unsqueeze(0), v_spatial])
        
        exp_map = HyperbolicExponential(dim=4)
        y = exp_map(x, v)
        
        # Verify result lies on hyperboloid
        hyperboloid_constraint = -y[0]*y[0] + torch.sum(y[1:] * y[1:])
        assert torch.allclose(hyperboloid_constraint, torch.tensor(-1.0, dtype=precision), atol=1e-6)
        
        print_memory_usage("End exp map test")

    def test_log_map_properties(self, precision):
        """Test mathematical properties of logarithm map."""
        print_memory_usage("Start log map test")
        
        # Create two points on the hyperboloid
        x_spatial = torch.tensor([0.1, -0.2, 0.1], dtype=precision)
        x_t = torch.sqrt(1 + torch.sum(x_spatial * x_spatial))
        x = torch.cat([x_t.unsqueeze(0), x_spatial])
        
        y_spatial = torch.tensor([0.12, -0.22, 0.11], dtype=precision)
        y_t = torch.sqrt(1 + torch.sum(y_spatial * y_spatial))
        y = torch.cat([y_t.unsqueeze(0), y_spatial])
        
        log_map = HyperbolicLogarithm(dim=4)
        v = log_map(x, y)
        
        # Compute hyperbolic distance
        inner = -x[0]*y[0] + torch.sum(x[1:] * y[1:])
        dist = torch.acosh(-inner)
        
        # Length of logarithm should equal hyperbolic distance
        v_norm = torch.sqrt(torch.sum(v[1:] * v[1:]))
        print(f"Distance: {dist:.6f}, Vector norm: {v_norm:.6f}, Diff: {abs(v_norm - dist):.6f}")
        
        assert torch.allclose(v_norm, dist, atol=1e-6)
        
        # Verify v is in tangent space (orthogonal to x in Minkowski sense)
        inner = -x[0]*v[0] + torch.sum(x[1:] * v[1:])
        print(f"Orthogonality: {inner:.6f}")
        
        assert torch.allclose(inner, torch.tensor(0.0, dtype=precision), atol=1e-6)
        
        print_memory_usage("End log map test")

    def test_exp_log_inverse(self, test_scales, precision):
        """Test that exp and log are inverses of each other."""
        print_memory_usage("Start exp-log inverse test")
        
        exp_map = HyperbolicExponential(dim=4)
        log_map = HyperbolicLogarithm(dim=4)
        
        for scale in test_scales:
            print(f"\nTesting with scale {scale}:")
            
            # Generate random point on hyperboloid
            x_spatial = torch.randn(3, dtype=precision) * 0.1
            x_t = torch.sqrt(1 + torch.sum(x_spatial * x_spatial))
            x = torch.cat([x_t.unsqueeze(0), x_spatial])
            
            # Generate random tangent vector (orthogonal to x)
            v_spatial = torch.randn(3, dtype=precision) * scale
            v_t = torch.sum(x[1:] * v_spatial) / x[0]  # Ensure orthogonality
            v = torch.cat([v_t.unsqueeze(0), v_spatial])
            
            print(f"Base point x: {x}")
            print(f"Initial vector v: {v}")
            
            # Apply exp then log
            y = exp_map(x, v)
            v_recovered = log_map(x, y)
            
            print(f"Exp map result y: {y}")
            print(f"Recovered vector v: {v_recovered}")
            print(f"Difference: {torch.norm(v - v_recovered)}")
            
            # Check that we recover the original vector
            assert torch.allclose(v, v_recovered, atol=0.001, rtol=0.01)
        
        print_memory_usage("End exp-log inverse test")

    def test_exp_log_consistency(self, test_norms, precision):
        """Test consistency between exponential and logarithm maps."""
        print_memory_usage("Start exp-log consistency test")
        
        exp_map = HyperbolicExponential(dim=4)
        log_map = HyperbolicLogarithm(dim=4)
        
        for norm_val in test_norms:
            print(f"\nTesting with norm {norm_val}:")
            
            # Generate random point on hyperboloid
            x_spatial = torch.randn(3, dtype=precision) * 0.1
            x_t = torch.sqrt(1 + torch.sum(x_spatial * x_spatial))
            x = torch.cat([x_t.unsqueeze(0), x_spatial])
            
            # Generate random direction and scale to desired norm
            v_dir = torch.randn(3, dtype=precision)
            v_dir = v_dir / torch.norm(v_dir)
            v_spatial = v_dir * norm_val
            v_t = torch.sum(x[1:] * v_spatial) / x[0]  # Ensure orthogonality
            v = torch.cat([v_t.unsqueeze(0), v_spatial])
            
            print(f"Base point x: {x}")
            print(f"Initial vector v: {v}")
            
            # Apply exp then log
            y = exp_map(x, v)
            v_recovered = log_map(x, y)
            
            print(f"Exp map result y: {y}")
            print(f"Recovered vector v: {v_recovered}")
            
            # Verify norms are preserved
            v_norm = torch.norm(v[1:])
            v_recovered_norm = torch.norm(v_recovered[1:])
            
            print(f"Original norm: {v_norm}")
            print(f"Recovered norm: {v_recovered_norm}")
            print(f"Difference: {abs(v_norm - v_recovered_norm)}")
            
            assert torch.allclose(v_norm, v_recovered_norm, atol=0.001)
        
        print_memory_usage("End exp-log consistency test")


class TestEuclideanOperations:
    """Test suite for Euclidean operations."""

    @pytest.fixture
    def dim(self, test_config) -> int:
        """Return dimension for tests."""
        return test_config["euclidean_tests"]["dimensions"]

    @pytest.fixture
    def test_batch_size(self, test_config) -> int:
        """Return batch size for tests."""
        return test_config["euclidean_tests"]["test_batch_size"]

    @pytest.fixture
    def exp_map(self, dim):
        """Create Euclidean exponential map."""
        return EuclideanExponential(dim=dim)

    @pytest.fixture
    def log_map(self, dim):
        """Create Euclidean logarithm map."""
        return EuclideanLogarithm(dim=dim)

    def test_exp_log_inverse(self, exp_map, log_map, dim):
        """Test exponential and logarithm maps are inverse operations."""
        print_memory_usage("Start Euclidean exp-log test")
        
        x = torch.randn(dim, dtype=torch.float32)
        v = torch.randn(dim, dtype=torch.float32)
        
        # Test exp(log(y)) = y
        y = exp_map(x, v)
        v_recovered = log_map(x, y)
        
        assert torch.allclose(v, v_recovered)
        
        print_memory_usage("End Euclidean exp-log test")

    def test_exp_zero_vector(self, exp_map, dim):
        """Test exponential map with zero vector."""
        print_memory_usage("Start Euclidean zero vector test")
        
        x = torch.randn(dim, dtype=torch.float32)
        v = torch.zeros(dim, dtype=torch.float32)
        
        result = exp_map(x, v)
        assert torch.allclose(result, x)
        
        print_memory_usage("End Euclidean zero vector test")

    def test_log_same_point(self, log_map, dim):
        """Test logarithm map with same point."""
        print_memory_usage("Start Euclidean same point test")
        
        x = torch.randn(dim, dtype=torch.float32)
        result = log_map(x, x)
        
        assert torch.allclose(result, torch.zeros(dim, dtype=torch.float32))
        
        print_memory_usage("End Euclidean same point test")


class TestParallelTransport:
    """Test suite for parallel transport."""

    @pytest.fixture
    def dim(self, test_config) -> int:
        """Return dimension for tests."""
        return test_config["parallel_transport"]["dimensions"]

    @pytest.fixture
    def methods(self, test_config) -> list[str]:
        """Return transport methods to test."""
        return test_config["parallel_transport"]["methods"]

    @pytest.fixture
    def num_test_cases(self, test_config) -> int:
        """Return number of test cases to run."""
        return test_config["parallel_transport"]["test_cases"]

    @pytest.fixture
    def transport_schild(self, dim):
        """Create parallel transport with Schild's ladder."""
        return ParallelTransport(dim=dim, method="schild")

    @pytest.fixture
    def transport_pole(self, dim):
        """Create parallel transport with pole ladder."""
        return ParallelTransport(dim=dim, method="pole")

    def test_schild_ladder(self, transport_schild, dim):
        """Test Schild's ladder parallel transport."""
        print_memory_usage("Start Schild test")
        
        v = torch.randn(dim, dtype=torch.float32)
        x = torch.randn(dim, dtype=torch.float32)
        y = torch.randn(dim, dtype=torch.float32)
        
        # Test transport preserves vector norm approximately
        transported = transport_schild(v, x, y)
        
        assert torch.allclose(torch.norm(transported), torch.norm(v), rtol=1e-4)
        
        print_memory_usage("End Schild test")

    def test_pole_ladder(self, transport_pole, dim):
        """Test pole ladder parallel transport."""
        print_memory_usage("Start pole test")
        
        v = torch.randn(dim, dtype=torch.float32)
        x = torch.randn(dim, dtype=torch.float32)
        y = torch.randn(dim, dtype=torch.float32)
        
        # Test transport preserves vector norm approximately
        transported = transport_pole(v, x, y)
        
        assert torch.allclose(torch.norm(transported), torch.norm(v), rtol=1e-4)
        
        print_memory_usage("End pole test")

    def test_transport_zero_vector(self, transport_schild, transport_pole, dim):
        """Test parallel transport of zero vector."""
        print_memory_usage("Start zero vector transport test")
        
        v = torch.zeros(dim, dtype=torch.float32)
        x = torch.randn(dim, dtype=torch.float32)
        y = torch.randn(dim, dtype=torch.float32)
        
        # Both methods should preserve zero vector
        schild_result = transport_schild(v, x, y)
        pole_result = transport_pole(v, x, y)
        
        assert torch.allclose(schild_result, torch.zeros(dim, dtype=torch.float32))
        assert torch.allclose(pole_result, torch.zeros(dim, dtype=torch.float32))
        
        print_memory_usage("End zero vector transport test")

    def test_transport_same_point(self, transport_schild, transport_pole, dim):
        """Test parallel transport to same point."""
        print_memory_usage("Start same point transport test")
        
        v = torch.randn(dim, dtype=torch.float32)
        x = torch.randn(dim, dtype=torch.float32)
        
        # Both methods should return original vector when transporting to same point
        schild_result = transport_schild(v, x, x)
        pole_result = transport_pole(v, x, x)
        
        assert torch.allclose(schild_result, v)
        assert torch.allclose(pole_result, v)
        
        print_memory_usage("End same point transport test")
