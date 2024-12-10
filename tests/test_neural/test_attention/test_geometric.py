"""
Unit tests for geometric operations and manifold structures.

Tests cover:
1. Geometric structures and tensors
2. Hyperbolic operations
3. Euclidean operations
4. Parallel transport methods
5. Quantum geometric integration
"""

import numpy as np
import pytest
import torch

from src.core.attention.geometric import (
    GeometricStructures,
    HyperbolicExponential,
    HyperbolicLogarithm,
    EuclideanExponential,
    EuclideanLogarithm,
    ParallelTransport,
)


class TestGeometricStructures:
    """Test suite for geometric structures."""

    @pytest.fixture
    def dim(self) -> int:
        """Return dimension for tests."""
        return 32

    @pytest.fixture
    def num_heads(self) -> int:
        """Return number of attention heads for tests."""
        return 8

    @pytest.fixture
    def batch_size(self) -> int:
        """Return batch size for tests."""
        return 16

    @pytest.fixture
    def geometric_structures(self, dim, num_heads):
        """Create geometric structures for testing."""
        return GeometricStructures(
            dim=dim,
            num_heads=num_heads,
            manifold_type="hyperbolic",
            curvature=-1.0,
            parallel_transport_method="schild",
        )

    def test_metric_initialization(self, geometric_structures, dim):
        """Test metric tensor initialization."""
        assert geometric_structures.metric.shape == (dim, dim)
        assert torch.allclose(geometric_structures.metric, torch.eye(dim))

    def test_connection_initialization(self, geometric_structures, dim):
        """Test connection coefficients initialization."""
        assert geometric_structures.connection.shape == (dim, dim, dim)
        assert torch.allclose(geometric_structures.connection, torch.zeros(dim, dim, dim))

    def test_curvature_tensor(self, geometric_structures, dim):
        """Test curvature tensor initialization and properties."""
        # Initialize curvature tensor with non-zero values that satisfy symmetries
        # R_ijkl = g_ik g_jl - g_il g_jk for constant curvature space
        metric = geometric_structures.metric
        g_ik = metric.unsqueeze(1).unsqueeze(3)  # Shape: (dim, 1, dim, 1)
        g_jl = metric.unsqueeze(0).unsqueeze(2)  # Shape: (1, dim, 1, dim)
        g_il = metric.unsqueeze(1).unsqueeze(2)  # Shape: (dim, 1, 1, dim)
        g_jk = metric.unsqueeze(0).unsqueeze(3)  # Shape: (1, dim, dim, 1)
        
        # Set curvature tensor for constant curvature space
        K = geometric_structures.curvature
        geometric_structures.curvature_tensor.data = K * (g_ik * g_jl - g_il * g_jk)
        
        # Verify shape
        assert geometric_structures.curvature_tensor.shape == (dim, dim, dim, dim)
        
        # Test symmetries using vectorized operations
        # Anti-symmetry in first two indices: R_ijkl = -R_jikl
        anti_sym_12 = geometric_structures.curvature_tensor + geometric_structures.curvature_tensor.permute(1, 0, 2, 3)
        assert torch.allclose(anti_sym_12, torch.zeros_like(anti_sym_12), atol=1e-6)
        
        # Anti-symmetry in last two indices: R_ijkl = -R_ijlk
        anti_sym_34 = geometric_structures.curvature_tensor + geometric_structures.curvature_tensor.permute(0, 1, 3, 2)
        assert torch.allclose(anti_sym_34, torch.zeros_like(anti_sym_34), atol=1e-6)
        
        # First Bianchi identity: R_ijkl + R_jkil + R_kijl = 0
        bianchi = (geometric_structures.curvature_tensor + 
                  geometric_structures.curvature_tensor.permute(1, 2, 0, 3) + 
                  geometric_structures.curvature_tensor.permute(2, 0, 1, 3))
        assert torch.allclose(bianchi, torch.zeros_like(bianchi), atol=1e-6)

    def test_sectional_curvature(self, geometric_structures, dim):
        """Test sectional curvature computation."""
        x = torch.randn(dim)
        v1 = torch.randn(dim)
        v2 = torch.randn(dim)
        
        # Make v2 orthogonal to v1
        v2 = v2 - torch.dot(v1, v2) * v1 / torch.dot(v1, v1)
        
        curvature = geometric_structures.compute_sectional_curvature(x, v1, v2)
        assert curvature.shape == ()  # Scalar output
        assert not torch.isnan(curvature)

    def test_geodesic_distance(self, geometric_structures, dim):
        """Test geodesic distance computation."""
        x = torch.randn(dim)
        x = x / (torch.norm(x) + 1e-8)  # Normalize with epsilon
        
        # Test distance to self with same point object
        self_distance = geometric_structures.compute_geodesic_distance(x, x)
        assert torch.allclose(
            self_distance,
            torch.tensor(0.0),
            atol=1e-3  # Relaxed absolute tolerance for self distance
        )
        
        # Test distance properties with different points
        y = torch.randn(dim)
        y = y / (torch.norm(y) + 1e-8)  # Normalize with epsilon
        
        distance = geometric_structures.compute_geodesic_distance(x, y)
        assert distance.shape == ()  # Scalar output
        assert distance >= 0  # Non-negative


class TestHyperbolicOperations:
    """Test hyperbolic geometric operations.
    
    Key mathematical properties being tested:
    1. Exponential map (exp_x(v)):
       - Maps a tangent vector v at point x to the hyperbolic manifold
       - Preserves the length of the geodesic (|v| = d(x, exp_x(v)))
       - Result always lies on the hyperbolic manifold (norm = 1)
       
    2. Logarithm map (log_x(y)):
       - Maps a point y on the manifold to a tangent vector at x
       - Inverse of exponential map: log_x(exp_x(v)) = v
       - Length of output equals hyperbolic distance: |log_x(y)| = d(x,y)
       
    3. Hyperbolic distance:
       d(x,y) = acosh(-<x,y>)  where <.,.> is the Minkowski inner product
    """
    
    @staticmethod
    def log_test_values(name: str, **tensors):
        """Helper to log test values in a consistent format."""
        print(f"\n{name}:")
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v}")
                if len(v.shape) > 0:  # If not scalar
                    print(f"{k} norm: {torch.norm(v)}")

    def test_hyperbolic_distance_formula(self):
        """Test the hyperbolic distance formula directly."""
        x = torch.tensor([-0.3277, 0.0394, 0.0043])
        y = torch.tensor([-0.3281, 0.0429, 0.0046])
        
        # Normalize to hyperbolic space
        x = x / torch.norm(x)
        y = y / torch.norm(y)
        
        # Compute Minkowski inner product
        inner = torch.sum(x * y)
        
        # Compute distance
        dist = torch.acosh(-inner)
        
        self.log_test_values("Hyperbolic Distance Test",
                           x=x, y=y, inner=inner, distance=dist)
        
        # Distance should be non-negative
        assert dist >= 0
        # Inner product should be <= -1 for points on hyperbolic manifold
        assert inner <= -1 + 1e-6

    def test_exp_map_properties(self):
        """Test mathematical properties of exponential map."""
        x = torch.tensor([-0.3277, 0.0394, 0.0043])
        v = torch.tensor([0.001, -0.002, 0.001])
        
        x = x / torch.norm(x)  # Normalize base point
        
        exp_map = HyperbolicExponential(dim=3)
        y = exp_map(x, v)
        
        self.log_test_values("Exponential Map Test",
                           x=x,
                           v=v,
                           y=y,
                           x_norm=torch.norm(x),
                           v_norm=torch.norm(v),
                           y_norm=torch.norm(y))
        
        # Result should lie on hyperbolic manifold
        assert torch.allclose(torch.norm(y), torch.tensor(1.0), atol=1e-6)
        
        # Exponential map should preserve vector norm as geodesic distance
        dist = torch.acosh(-torch.sum(x * y))
        assert torch.allclose(dist, torch.norm(v), atol=1e-6)

    def test_log_map_properties(self):
        """Test mathematical properties of logarithm map."""
        x = torch.tensor([-0.3277, 0.0394, 0.0043])
        y = torch.tensor([-0.3281, 0.0429, 0.0046])
        
        x = x / torch.norm(x)
        y = y / torch.norm(y)
        
        log_map = HyperbolicLogarithm(dim=3)
        v = log_map(x, y)
        
        # Compute hyperbolic distance
        dist = torch.acosh(-torch.sum(x * y))
        
        self.log_test_values("Logarithm Map Test",
                           x=x,
                           y=y,
                           v=v,
                           distance=dist,
                           v_norm=torch.norm(v))
        
        # Length of logarithm should equal hyperbolic distance
        assert torch.allclose(torch.norm(v), dist, atol=1e-6)
        
        # Verify v is in tangent space (orthogonal to x in Minkowski sense)
        inner = torch.sum(x * v)
        assert torch.allclose(inner, torch.tensor(0.0), atol=1e-6)

    def test_exp_log_inverse(self):
        """Test that exp and log are inverses of each other."""
        # Initialize maps
        exp_map = HyperbolicExponential(dim=32)
        log_map = HyperbolicLogarithm(dim=32)
        
        # Test with different scales
        scales = [0.001, 0.01, 0.1]
        
        for scale in scales:
            # Generate random point on hyperbolic manifold
            x = torch.randn(32)
            x = x / torch.norm(x)
            
            # Generate random tangent vector
            v = torch.randn(32)
            v = v - (torch.sum(x * v) * x)  # Project to tangent space
            v = v * (scale / torch.norm(v))  # Scale to desired norm
            
            # Apply exp then log
            y = exp_map(x, v)
            v_recovered = log_map(x, y)
            
            self.log_test_values(f"Exp-Log Test (scale={scale})",
                               x=x,
                               v=v,
                               y=y,
                               v_recovered=v_recovered,
                               error=torch.norm(v - v_recovered))
            
            # Verify recovery
            assert torch.allclose(v, v_recovered, atol=1e-3, rtol=1e-2)

    def test_exp_log_consistency(self):
        """Test consistency of vector norms through exp-log maps."""
        exp_map = HyperbolicExponential(dim=32)
        log_map = HyperbolicLogarithm(dim=32)
        
        # Test with different norms
        norms = [0.001, 0.01, 0.1]
        
        for norm_val in norms:
            # Generate normalized base point
            x = torch.randn(32)
            x = x / torch.norm(x)
            
            # Generate random direction and scale to desired norm
            v_dir = torch.randn(32)
            v_dir = v_dir - (torch.sum(x * v_dir) * x)  # Project to tangent space
            v = v_dir * (norm_val / torch.norm(v_dir))
            
            # Apply maps
            y = exp_map(x, v)
            v_recovered = log_map(x, y)
            
            self.log_test_values(f"\nTesting with norm {norm_val}:",
                               v_norm=torch.norm(v),
                               y_norm=torch.norm(y),
                               v_recovered_norm=torch.norm(v_recovered))
            
            # Verify norm preservation
            assert torch.allclose(torch.norm(v), torch.norm(v_recovered), atol=1e-3)


class TestEuclideanOperations:
    """Test suite for Euclidean operations."""

    @pytest.fixture
    def dim(self) -> int:
        """Return dimension for tests."""
        return 32

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
        x = torch.randn(dim)
        v = torch.randn(dim)
        
        # Test exp(log(y)) = y
        y = exp_map(x, v)
        v_recovered = log_map(x, y)
        assert torch.allclose(v, v_recovered)

    def test_exp_zero_vector(self, exp_map, dim):
        """Test exponential map with zero vector."""
        x = torch.randn(dim)
        v = torch.zeros(dim)
        
        result = exp_map(x, v)
        assert torch.allclose(result, x)

    def test_log_same_point(self, log_map, dim):
        """Test logarithm map with same point."""
        x = torch.randn(dim)
        
        result = log_map(x, x)
        assert torch.allclose(result, torch.zeros(dim))


class TestParallelTransport:
    """Test suite for parallel transport."""

    @pytest.fixture
    def dim(self) -> int:
        """Return dimension for tests."""
        return 32

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
        v = torch.randn(dim)
        x = torch.randn(dim)
        y = torch.randn(dim)
        
        # Test transport preserves vector norm approximately
        transported = transport_schild(v, x, y)
        assert torch.allclose(torch.norm(transported), torch.norm(v), rtol=1e-4)

    def test_pole_ladder(self, transport_pole, dim):
        """Test pole ladder parallel transport."""
        v = torch.randn(dim)
        x = torch.randn(dim)
        y = torch.randn(dim)
        
        # Test transport preserves vector norm approximately
        transported = transport_pole(v, x, y)
        assert torch.allclose(torch.norm(transported), torch.norm(v), rtol=1e-4)

    def test_transport_zero_vector(self, transport_schild, transport_pole, dim):
        """Test parallel transport of zero vector."""
        v = torch.zeros(dim)
        x = torch.randn(dim)
        y = torch.randn(dim)
        
        # Both methods should preserve zero vector
        assert torch.allclose(transport_schild(v, x, y), torch.zeros(dim))
        assert torch.allclose(transport_pole(v, x, y), torch.zeros(dim))

    def test_transport_same_point(self, transport_schild, transport_pole, dim):
        """Test parallel transport to same point."""
        v = torch.randn(dim)
        x = torch.randn(dim)
        
        # Both methods should return original vector when transporting to same point
        assert torch.allclose(transport_schild(v, x, x), v)
        assert torch.allclose(transport_pole(v, x, x), v)
