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
        return test_config["geometric"]["dimensions"]

    @pytest.fixture
    def num_heads(self, test_config) -> int:
        """Return number of attention heads for tests."""
        return test_config["geometric"]["num_heads"]

    @pytest.fixture
    def batch_size(self, test_config) -> int:
        """Return batch size for tests."""
        return test_config["fiber_bundle"]["batch_size"]

    @pytest.fixture
    def dtype(self, test_config) -> torch.dtype:
        """Return data type for tests."""
        return getattr(torch, test_config["fiber_bundle"]["dtype"])

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
        
        curvature = geometric_structures.sectional_curvature(x, v1, v2)
        
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
    def test_scales(self) -> list[float]:
        """Return test scales for vector operations."""
        return [0.1, 0.5, 1.0, 2.0]  # Standard test scales

    @pytest.fixture
    def test_norms(self) -> list[float]:
        """Return test norms for vector operations."""
        return [0.1, 1.0, 5.0, 10.0]  # Standard test norms

    @pytest.fixture
    def precision(self, test_config) -> torch.dtype:
        """Return precision for tests."""
        return getattr(torch, test_config["fiber_bundle"]["dtype"])

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
        dim = 3
        exp_map = HyperbolicExponential(dim, dtype=precision)
        log_map = HyperbolicLogarithm(dim, dtype=precision)
        geom = GeometricStructures(dim=dim, num_heads=1, manifold_type="hyperbolic", curvature=-1.0)
        
        # Create points on hyperboloid
        x = torch.tensor([1.2, 0.3, 0.4], dtype=precision)
        y = torch.tensor([1.5, 0.5, 0.0], dtype=precision)
        x = exp_map.project_to_hyperboloid(x)
        y = exp_map.project_to_hyperboloid(y)
        
        print_test_case("Test Points",
            x=x,
            y=y
        )
        verify_hyperboloid_constraint(exp_map, x, "base point")
        verify_hyperboloid_constraint(exp_map, y, "target point")
        
        # Compute logarithm map
        v = log_map.forward(x, y)
        print_test_case("Logarithm Result",
            v=v,
            norm=torch.norm(v)
        )
        verify_tangent_space(exp_map, x, v, "log vector")
        
        # Verify distance preservation
        dist = geom.compute_geodesic_distance(x.unsqueeze(0), y.unsqueeze(0))
        v_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v.unsqueeze(0), v.unsqueeze(0))))
        print_test_case("Distance Check",
            distance=dist,
            vector_norm=v_norm,
            difference=torch.abs(dist - v_norm)
        )
        assert torch.allclose(dist, v_norm, atol=1e-7)

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
    """Test Euclidean geometric operations."""

    @pytest.fixture
    def dim(self, test_config) -> int:
        """Return dimension for tests."""
        return test_config["geometric"]["dimensions"]

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
    """Test parallel transport operations."""

    @pytest.fixture
    def dim(self, test_config) -> int:
        """Return dimension for tests."""
        return test_config["geometric"]["dimensions"]

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
        """Create Schild's ladder transport."""
        exp_map = HyperbolicExponential(dim)
        transport = ParallelTransport(dim=dim, method="schild", exp_map=exp_map)
        return transport

    @pytest.fixture
    def transport_pole(self, dim):
        """Create pole ladder transport."""
        exp_map = HyperbolicExponential(dim)
        transport = ParallelTransport(dim=dim, method="pole", exp_map=exp_map)
        return transport

    def test_schild_ladder(self, transport_schild, dim):
        """Test Schild's ladder parallel transport."""
        print("\n=== Testing Schild's Ladder Transport ===")
        print_memory_usage("Start Schild test")

        # Create test points and vectors with controlled magnitudes
        v = torch.randn(dim, dtype=torch.float32) * 0.1
        x = torch.randn(dim, dtype=torch.float32) * 0.5
        y = torch.randn(dim, dtype=torch.float32) * 0.5
        
        print(f"\nInitial values:")
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"v: {v}")
        print(f"Initial norms - x: {torch.norm(x)}, y: {torch.norm(y)}, v: {torch.norm(v)}")

        # Project points and vector with detailed logging
        print("\nProjecting to hyperboloid and tangent space...")
        x = transport_schild.exp_map.project_to_hyperboloid(x)
        y = transport_schild.exp_map.project_to_hyperboloid(y)
        v = transport_schild.exp_map.project_to_tangent(x, v)
        
        print(f"\nAfter projection:")
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"v: {v}")
        print(f"Projected norms - x: {torch.norm(x)}, y: {torch.norm(y)}, v: {torch.norm(v)}")
        
        # Verify initial conditions
        verify_hyperboloid_constraint(transport_schild.exp_map, x, "initial x")
        verify_hyperboloid_constraint(transport_schild.exp_map, y, "initial y")
        verify_tangent_space(transport_schild.exp_map, x, v, "initial v")

        # Perform transport with detailed analysis
        print("\nPerforming parallel transport...")
        transported = transport_schild(v, x, y)
        
        # Verify transport properties
        properties = verify_parallel_transport_properties(transport_schild, x, y, v, transported, "Schild's ladder")
        
        # If verification fails, analyze the failure
        if (abs(properties['y_tangent']) >= 1e-6 or 
            abs(properties['norm_ratio'] - 1.0) >= 1e-5):
            analysis = analyze_transport_failure(transport_schild, x, y, v, transported, "Schild's ladder")
        
        # Original assertions with more context
        inner = transport_schild.exp_map.minkowski_inner(y, transported)
        if torch.abs(inner) >= 1e-6:
            print(f"\nTangent space violation:")
            print(f"  Inner product <y,transported>: {inner}")
            print(f"  Threshold: 1e-6")
            assert False, f"Transported vector not in tangent space: {inner}"

        norm_ratio = torch.norm(transported) / torch.norm(v)
        if torch.abs(norm_ratio - 1.0) >= 1e-5:
            print(f"\nNorm preservation violation:")
            print(f"  Norm ratio: {norm_ratio}")
            print(f"  Expected: 1.0 ± 1e-5")
            assert False, f"Transport did not preserve norm: {norm_ratio}"

    def test_pole_ladder(self, transport_pole, dim):
        """Test pole ladder parallel transport."""
        print("\n=== Testing Pole Ladder Transport ===")
        print_memory_usage("Start pole test")

        # Create test points and vectors
        v = torch.randn(dim, dtype=torch.float32) * 0.1  # Smaller vectors for stability
        x = torch.randn(dim, dtype=torch.float32)
        y = torch.randn(dim, dtype=torch.float32)
        
        print(f"\nInitial values:")
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"v: {v}")
        print(f"Initial norms - x: {torch.norm(x)}, y: {torch.norm(y)}, v: {torch.norm(v)}")

        # Project points to hyperboloid and vector to tangent space
        x = transport_pole.exp_map.project_to_hyperboloid(x)
        y = transport_pole.exp_map.project_to_hyperboloid(y)
        v = transport_pole.exp_map.project_to_tangent(x, v)
        
        print(f"\nAfter projection:")
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"v: {v}")
        print(f"Projected norms - x: {torch.norm(x)}, y: {torch.norm(y)}, v: {torch.norm(v)}")
        print(f"Hyperboloid constraints - x: {transport_pole.exp_map.minkowski_inner(x, x) + 1}, y: {transport_pole.exp_map.minkowski_inner(y, y) + 1}")
        print(f"Tangent constraint - <x,v>: {transport_pole.exp_map.minkowski_inner(x, v)}")

        # Test transport preserves vector norm approximately
        transported = transport_pole(v, x, y)
        
        print(f"\nTransport results:")
        print(f"Transported vector: {transported}")
        print(f"Transport norm: {torch.norm(transported)}")
        print(f"Original norm: {torch.norm(v)}")

        # Verify transported vector is in tangent space at y
        inner = transport_pole.exp_map.minkowski_inner(y, transported)
        print(f"Tangent space check - <y,transported>: {inner}")
        assert torch.abs(inner) < 1e-6, f"Transported vector not in tangent space: {inner}"

        # Verify norm preservation
        norm_ratio = torch.norm(transported) / torch.norm(v)
        print(f"Norm ratio (transported/original): {norm_ratio}")
        assert torch.abs(norm_ratio - 1.0) < 1e-5, f"Transport did not preserve norm: {norm_ratio}"

    def test_transport_zero_vector(self, transport_schild, transport_pole, dim):
        """Test parallel transport of zero vector."""
        print("\n=== Testing Zero Vector Transport ===")
        print_memory_usage("Start zero vector transport test")

        # Create test points and zero vector
        v = torch.zeros(dim, dtype=torch.float32)
        x = torch.randn(dim, dtype=torch.float32) * 0.5  # Smaller magnitude for stability
        y = torch.randn(dim, dtype=torch.float32) * 0.5
        
        print(f"\nInitial setup:")
        print(f"Zero vector v: {v}")
        print(f"Point x: {x}")
        print(f"Point y: {y}")
        print(f"Initial norms - x: {torch.norm(x)}, y: {torch.norm(y)}, v: {torch.norm(v)}")

        # Project points with verification
        print("\nProjecting points to hyperboloid...")
        x = transport_schild.exp_map.project_to_hyperboloid(x)
        y = transport_schild.exp_map.project_to_hyperboloid(y)
        
        verify_hyperboloid_constraint(transport_schild.exp_map, x, "initial x")
        verify_hyperboloid_constraint(transport_schild.exp_map, y, "initial y")
        
        # Transport with both methods
        print("\nPerforming parallel transport...")
        schild_result = transport_schild(v, x, y)
        pole_result = transport_pole(v, x, y)
        
        print(f"\nTransport results:")
        print(f"Schild's ladder result: {schild_result}")
        print(f"Pole ladder result: {pole_result}")
        print(f"Norms - Schild: {torch.norm(schild_result)}, Pole: {torch.norm(pole_result)}")
        
        # Verify results component by component
        print("\nComponent-wise analysis:")
        for i in range(dim):
            print(f"Component {i}:")
            print(f"  Schild: {schild_result[i]}")
            print(f"  Pole: {pole_result[i]}")
            print(f"  Magnitude - Schild: {abs(schild_result[i])}, Pole: {abs(pole_result[i])}")
        
        # Check if results are close to zero
        schild_max = torch.max(torch.abs(schild_result))
        pole_max = torch.max(torch.abs(pole_result))
        print(f"\nMaximum absolute values:")
        print(f"  Schild: {schild_max}")
        print(f"  Pole: {pole_max}")
        
        if schild_max >= 1e-6:
            print(f"\nSchild's ladder failed zero vector preservation:")
            print(f"  Maximum component: {schild_max}")
            print(f"  Threshold: 1e-6")
            assert False, f"Schild's ladder failed to preserve zero vector: {schild_result}"
        
        if pole_max >= 1e-6:
            print(f"\nPole ladder failed zero vector preservation:")
            print(f"  Maximum component: {pole_max}")
            print(f"  Threshold: 1e-6")
            assert False, f"Pole ladder failed to preserve zero vector: {pole_result}"

    def test_transport_same_point(self, transport_schild, transport_pole, dim):
        """Test parallel transport to same point."""
        print("\n=== Testing Same Point Transport ===")
        print_memory_usage("Start same point transport test")

        # Create test points and vector with controlled magnitudes
        v = torch.randn(dim, dtype=torch.float32) * 0.1
        x = torch.randn(dim, dtype=torch.float32) * 0.5
        
        print(f"\nInitial setup:")
        print(f"Point x: {x}")
        print(f"Vector v: {v}")
        print(f"Initial norms - x: {torch.norm(x)}, v: {torch.norm(v)}")

        # Project point and vector with verification
        print("\nProjecting to hyperboloid and tangent space...")
        x = transport_schild.exp_map.project_to_hyperboloid(x)
        v = transport_schild.exp_map.project_to_tangent(x, v)
        
        verify_hyperboloid_constraint(transport_schild.exp_map, x, "point x")
        verify_tangent_space(transport_schild.exp_map, x, v, "vector v")
        
        # Transport with both methods
        print("\nPerforming parallel transport...")
        schild_result = transport_schild(v, x, x)
        pole_result = transport_pole(v, x, x)
        
        print(f"\nTransport results:")
        print(f"Original vector: {v}")
        print(f"Schild result: {schild_result}")
        print(f"Pole result: {pole_result}")
        
        # Detailed comparison
        print("\nComponent-wise comparison:")
        for i in range(dim):
            print(f"Component {i}:")
            print(f"  Original: {v[i]}")
            print(f"  Schild: {schild_result[i]}")
            print(f"  Pole: {pole_result[i]}")
            print(f"  Differences - Schild: {abs(v[i] - schild_result[i])}, Pole: {abs(v[i] - pole_result[i])}")
        
        # Verify results with detailed error messages
        if not torch.allclose(schild_result, v, rtol=1e-5, atol=1e-5):
            diff = torch.norm(schild_result - v)
            print(f"\nSchild's ladder failed same-point transport:")
            print(f"  L2 difference: {diff}")
            print(f"  Component-wise difference: {schild_result - v}")
            print(f"  Relative tolerance: 1e-5")
            print(f"  Absolute tolerance: 1e-5")
            assert False, f"Schild's ladder failed to preserve vector at same point: {diff}"
        
        if not torch.allclose(pole_result, v, rtol=1e-5, atol=1e-5):
            diff = torch.norm(pole_result - v)
            print(f"\nPole ladder failed same-point transport:")
            print(f"  L2 difference: {diff}")
            print(f"  Component-wise difference: {pole_result - v}")
            print(f"  Relative tolerance: 1e-5")
            print(f"  Absolute tolerance: 1e-5")
            assert False, f"Pole ladder failed to preserve vector at same point: {diff}"

def print_test_case(name: str, **values):
    """Print test case details."""
    print(f"\n=== {name} ===")
    for key, val in values.items():
        if isinstance(val, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {val.shape}")
            print(f"  Values: {val}")
            print(f"  Dtype: {val.dtype}")
            if len(val.shape) > 0:
                print(f"  Norm (L2): {torch.norm(val)}")
                if val.shape[-1] > 1:
                    print(f"  Time component: {val[..., 0]}")
                    print(f"  Space components: {val[..., 1:]}")
        else:
            print(f"{key}: {val}")
    print("=" * 50)

def verify_hyperboloid_constraint(exp_map: HyperbolicExponential, x: torch.Tensor, tag: str, atol: float = 1e-6):
    """Verify point lies on hyperboloid."""
    constraint = exp_map.minkowski_inner(x, x) + 1.0
    print(f"\nHyperboloid constraint check ({tag}):")
    print(f"  Constraint value: {constraint}")
    print(f"  Time component: {x[..., 0]}")
    print(f"  Space components norm: {torch.norm(x[..., 1:])}")
    assert torch.allclose(constraint, torch.zeros_like(constraint), atol=atol), \
        f"Point {tag} not on hyperboloid: {constraint}"
    assert torch.all(x[..., 0] > 0), f"Time component not positive for {tag}: {x[..., 0]}"

def verify_tangent_space(exp_map: HyperbolicExponential, x: torch.Tensor, v: torch.Tensor, tag: str, atol: float = 1e-6):
    """Verify vector lies in tangent space."""
    inner = exp_map.minkowski_inner(x, v)
    print(f"\nTangent space check ({tag}):")
    print(f"  Inner product: {inner}")
    print(f"  Vector norm: {torch.norm(v)}")
    assert torch.allclose(inner, torch.zeros_like(inner), atol=atol), \
        f"Vector {tag} not in tangent space: {inner}"

def verify_parallel_transport_properties(transport, x, y, v, transported, tag: str):
    """Verify mathematical properties of parallel transport."""
    print(f"\n=== Verifying {tag} Transport Properties ===")
    
    # Check hyperboloid constraints
    x_constraint = transport.exp_map.minkowski_inner(x, x) + 1
    y_constraint = transport.exp_map.minkowski_inner(y, y) + 1
    print(f"Hyperboloid constraints:")
    print(f"  x: {x_constraint} (should be ≈ 0)")
    print(f"  y: {y_constraint} (should be ≈ 0)")
    
    # Check tangent space constraints
    x_tangent = transport.exp_map.minkowski_inner(x, v)
    y_tangent = transport.exp_map.minkowski_inner(y, transported)
    print(f"Tangent space constraints:")
    print(f"  <x,v>: {x_tangent} (should be ≈ 0)")
    print(f"  <y,transported>: {y_tangent} (should be ≈ 0)")
    
    # Check norm preservation
    v_norm = torch.norm(v)
    transported_norm = torch.norm(transported)
    norm_ratio = transported_norm / v_norm if v_norm > 0 else float('inf')
    print(f"Norm preservation:")
    print(f"  Original norm: {v_norm}")
    print(f"  Transported norm: {transported_norm}")
    print(f"  Ratio: {norm_ratio} (should be ≈ 1)")
    
    # Check vector components
    print(f"Vector components:")
    print(f"  Original v: {v}")
    print(f"  Transported v: {transported}")
    print(f"  Component-wise difference: {transported - v}")
    
    return {
        'x_constraint': x_constraint,
        'y_constraint': y_constraint,
        'x_tangent': x_tangent,
        'y_tangent': y_tangent,
        'v_norm': v_norm,
        'transported_norm': transported_norm,
        'norm_ratio': norm_ratio
    }

def analyze_transport_failure(transport, x, y, v, transported, tag: str):
    """Analyze why transport might be failing."""
    print(f"\n=== Analyzing {tag} Transport Failure ===")
    
    # Create GeometricStructures instance for distance computation
    geom = GeometricStructures(dim=x.shape[-1], num_heads=1, manifold_type="hyperbolic", curvature=-1.0)
    
    # Check if points are too far apart
    distance = geom.compute_geodesic_distance(x.unsqueeze(0), y.unsqueeze(0))
    print(f"Distance between points: {distance}")
    if distance > 10:
        print("WARNING: Points may be too far apart for stable transport")
    
    # Check numerical stability
    x_norm = torch.norm(x)
    y_norm = torch.norm(y)
    print(f"Point norms:")
    print(f"  ||x||: {x_norm}")
    print(f"  ||y||: {y_norm}")
    if x_norm > 100 or y_norm > 100:
        print("WARNING: Point norms are large, may cause numerical instability")
    
    # Check vector magnitude
    v_norm = torch.norm(v)
    print(f"Vector magnitude: {v_norm}")
    if v_norm > 1:
        print("WARNING: Vector magnitude > 1, may cause instability")
    
    # Analyze transport components
    t_components = {
        'time': transported[0],
        'space': transported[1:],
        'space_norm': torch.norm(transported[1:])
    }
    print(f"Transported vector components:")
    print(f"  Time component: {t_components['time']}")
    print(f"  Space components: {t_components['space']}")
    print(f"  Space norm: {t_components['space_norm']}")
    
    return {
        'distance': distance,
        'x_norm': x_norm,
        'y_norm': y_norm,
        'v_norm': v_norm,
        't_components': t_components
    }

def test_hyperbolic_operations():
    """Test hyperbolic exponential and logarithm maps with enhanced precision."""
    dim = 3
    exp_map = HyperbolicExponential(dim, dtype=torch.float64)
    log_map = HyperbolicLogarithm(dim, dtype=torch.float64)
    geom = GeometricStructures(dim=dim, num_heads=1, manifold_type="hyperbolic", curvature=-1.0)
    
    # Test points
    print_test_case("Base Points",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64),
        y=torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64)
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    y = torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    y = exp_map.project_to_hyperboloid(y)
    verify_hyperboloid_constraint(exp_map, x, "base point x")
    verify_hyperboloid_constraint(exp_map, y, "base point y")
    
    # Test exp-log cycle
    v = log_map.forward(x, y)
    print_test_case("Log Map Result",
        v=v,
        norm=torch.norm(v)
    )
    verify_tangent_space(exp_map, x, v, "log map result")
    
    y_recovered = exp_map.forward(x, v)
    print_test_case("Exp Map Result",
        y_original=y,
        y_recovered=y_recovered,
        difference=y - y_recovered
    )
    verify_hyperboloid_constraint(exp_map, y_recovered, "recovered point")
    assert torch.allclose(y, y_recovered, atol=1e-7)
    
    # Test distance preservation
    dist_direct = geom.compute_geodesic_distance(x.unsqueeze(0), y.unsqueeze(0))
    v_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v.unsqueeze(0), v.unsqueeze(0))))
    print_test_case("Distance Check",
        direct_distance=dist_direct,
        vector_norm=v_norm,
        difference=torch.abs(dist_direct - v_norm)
    )
    assert torch.allclose(dist_direct, v_norm, atol=1e-7)

def test_euclidean_operations():
    """Test Euclidean exponential and logarithm maps."""
    dim = 3
    exp_map = EuclideanExponential(dim, dtype=torch.float64)
    log_map = EuclideanLogarithm(dim, dtype=torch.float64)
    
    # Test points
    print_test_case("Euclidean Points",
        x=torch.tensor([0.0, 0.3, 0.4], dtype=torch.float64),
        y=torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    )
    x = torch.tensor([0.0, 0.3, 0.4], dtype=torch.float64)
    y = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    
    # Test exp-log cycle
    v = log_map.forward(x, y)
    print_test_case("Euclidean Log Result",
        v=v,
        norm=torch.norm(v)
    )
    
    y_recovered = exp_map.forward(x, v)
    print_test_case("Euclidean Exp Result",
        y_original=y,
        y_recovered=y_recovered,
        difference=y - y_recovered
    )
    assert torch.allclose(y, y_recovered, atol=1e-7)
    
    # Verify Euclidean properties
    assert torch.allclose(v, y - x, atol=1e-7)
    assert torch.allclose(y_recovered, x + v, atol=1e-7)

def test_mixed_geometry():
    """Test interaction between hyperbolic and Euclidean operations."""
    dim = 3
    hyp_exp = HyperbolicExponential(dim, dtype=torch.float64)
    hyp_log = HyperbolicLogarithm(dim, dtype=torch.float64)
    euc_exp = EuclideanExponential(dim, dtype=torch.float64)
    euc_log = EuclideanLogarithm(dim, dtype=torch.float64)
    geom = GeometricStructures(dim=dim, num_heads=1, manifold_type="hyperbolic", curvature=-1.0)
    
    # Test points
    print_test_case("Mixed Geometry Points",
        hyp_x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64),
        euc_x=torch.tensor([0.0, 0.3, 0.4], dtype=torch.float64)
    )
    hyp_x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    euc_x = torch.tensor([0.0, 0.3, 0.4], dtype=torch.float64)
    
    # Project hyperbolic point
    hyp_x = hyp_exp.project_to_hyperboloid(hyp_x)
    verify_hyperboloid_constraint(hyp_exp, hyp_x, "hyperbolic point")
    
    # Test small tangent vectors
    hyp_v = torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)
    euc_v = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    
    # Project hyperbolic vector to tangent space
    hyp_v = hyp_exp.project_to_tangent(hyp_x, hyp_v)
    verify_tangent_space(hyp_exp, hyp_x, hyp_v, "hyperbolic vector")
    
    # Apply exponential maps
    hyp_y = hyp_exp.forward(hyp_x, hyp_v)
    euc_y = euc_exp.forward(euc_x, euc_v)
    
    print_test_case("Mixed Exponential Results",
        hyp_result=hyp_y,
        euc_result=euc_y
    )
    verify_hyperboloid_constraint(hyp_exp, hyp_y, "hyperbolic exp result")
    
    # Verify distances
    hyp_dist = geom.compute_geodesic_distance(hyp_x.unsqueeze(0), hyp_y.unsqueeze(0))
    euc_dist = torch.norm(euc_y - euc_x)
    print_test_case("Distance Comparison",
        hyperbolic_distance=hyp_dist,
        euclidean_distance=euc_dist
    )
    
    # Verify vector norms
    hyp_v_norm = torch.sqrt(torch.abs(hyp_exp.minkowski_inner(hyp_v.unsqueeze(0), hyp_v.unsqueeze(0))))
    euc_v_norm = torch.norm(euc_v)
    print_test_case("Vector Norm Comparison",
        hyperbolic_norm=hyp_v_norm,
        euclidean_norm=euc_v_norm
    )
    
    # Verify distance-norm relationships
    assert torch.allclose(hyp_dist, hyp_v_norm, atol=1e-7)
    assert torch.allclose(euc_dist, euc_v_norm, atol=1e-7)
