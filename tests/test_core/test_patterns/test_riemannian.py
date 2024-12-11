"""
Unit tests for the Riemannian framework implementation.

Tests cover:
1. Metric tensor properties
2. Christoffel symbols
3. Covariant derivatives
4. Geodesic flows
5. Curvature computations
6. Holonomy and parallel transport
7. Sectional curvature
8. Ricci flow
9. Riemannian metric properties
10. Riemannian curvature tensors
11. Geodesic properties and computations
"""


import numpy as np
import pytest
import torch

from src.core.patterns.riemannian import (
    PatternRiemannianStructure,
    RiemannianFramework,
    ChristoffelSymbols,
    CurvatureTensor,
)


class TestRiemannianStructure:
    @pytest.fixture
    def manifold_dim(self):
        """Dimension of test manifold."""
        return 4

    @pytest.fixture
    def riemannian_structure(self, manifold_dim):
        """Create a test Riemannian structure."""
        return PatternRiemannianStructure(dim=manifold_dim)

    def test_metric_tensor(self, riemannian_structure, manifold_dim):
        """Test metric tensor properties."""
        point = torch.randn(manifold_dim)
        vectors = (torch.randn(manifold_dim), torch.randn(manifold_dim))

        # Compute metric
        g = riemannian_structure.metric_tensor(point, vectors)

        # Test symmetry
        g_transpose = riemannian_structure.metric_tensor(point, vectors[::-1])
        assert torch.allclose(g, g_transpose, rtol=1e-5), "Metric should be symmetric"

        # Test positive definiteness
        v = torch.randn(manifold_dim)
        assert (
            riemannian_structure.metric_tensor(point, (v, v)) >= 0
        ), "Metric should be positive definite"

    def test_christoffel_symbols(self, riemannian_structure, manifold_dim):
        """Test Christoffel symbols properties."""
        chart = torch.randn(manifold_dim)
        christoffel = riemannian_structure.christoffel_symbols(chart)

        # Test shape
        assert christoffel.shape == (
            manifold_dim,
            manifold_dim,
            manifold_dim,
        ), "Christoffel symbols should have correct shape"

        # Test symmetry in lower indices
        assert torch.allclose(
            christoffel.transpose(-1, -2), christoffel, rtol=1e-5
        ), "Christoffel symbols should be symmetric in lower indices"

    def test_covariant_derivative(self, riemannian_structure, manifold_dim):
        """Test covariant derivative properties."""
        vector_field = torch.randn(manifold_dim)
        direction = torch.randn(manifold_dim)

        # Compute covariant derivative
        cov_deriv = riemannian_structure.covariant_derivative(vector_field, direction)

        # Test linearity in direction
        scalar = 2.0
        assert torch.allclose(
            riemannian_structure.covariant_derivative(vector_field, scalar * direction),
            scalar * cov_deriv,
            rtol=1e-5,
        ), "Covariant derivative should be linear in direction"

    def test_geodesic_flow(self, riemannian_structure, manifold_dim):
        """Test geodesic flow properties."""
        initial_point = torch.randn(manifold_dim)
        initial_velocity = torch.randn(manifold_dim)

        # Generate geodesic
        flow = riemannian_structure.geodesic_flow(initial_point, initial_velocity)

        # Test constant speed
        speeds = []
        for t in torch.linspace(0, 1, 10):
            point = flow(t)
            velocity = flow.velocity(t)
            speed = riemannian_structure.metric_tensor(point, (velocity, velocity))
            speeds.append(speed)

        assert np.allclose(
            speeds, speeds[0], rtol=1e-4
        ), "Geodesic should have constant speed"

    def test_curvature_tensor(self, riemannian_structure, manifold_dim):
        """Test curvature tensor properties."""
        point = torch.randn(manifold_dim)
        R = riemannian_structure.curvature_tensor(point)

        # Test shape
        assert (
            R.shape == (manifold_dim,) * 4
        ), "Curvature tensor should have correct shape"

        # Test antisymmetry
        assert torch.allclose(
            R.transpose(0, 1), -R, rtol=1e-5
        ), "Curvature should be antisymmetric in first two indices"

        assert torch.allclose(
            R.transpose(2, 3), -R, rtol=1e-5
        ), "Curvature should be antisymmetric in last two indices"

        # Test first Bianchi identity
        bianchi1 = R + R.roll(1, 0).roll(1, 1) + R.roll(2, 0).roll(2, 1)
        assert torch.allclose(
            bianchi1, torch.zeros_like(R), rtol=1e-5
        ), "First Bianchi identity should hold"

        # Test second Bianchi identity
        nabla_R = riemannian_structure.covariant_derivative_curvature(point)
        bianchi2 = nabla_R + nabla_R.roll(2, 0) + nabla_R.roll(4, 0)
        assert torch.allclose(
            bianchi2, torch.zeros_like(nabla_R), rtol=1e-5
        ), "Second Bianchi identity should hold"

    def test_sectional_curvature(self, riemannian_structure, manifold_dim):
        """Test sectional curvature computations."""
        point = torch.randn(manifold_dim)

        # Generate orthonormal vectors
        v1 = torch.randn(manifold_dim)
        v1 = v1 / torch.norm(v1)
        v2 = torch.randn(manifold_dim)
        v2 = v2 - (v2 @ v1) * v1
        v2 = v2 / torch.norm(v2)

        # Compute sectional curvature
        K = riemannian_structure.sectional_curvature(point, v1, v2)
        assert isinstance(K, torch.Tensor)

        # Test bounds in constant curvature case
        if hasattr(riemannian_structure, "constant_curvature"):
            k = riemannian_structure.constant_curvature
            assert torch.allclose(K, k * torch.ones_like(K), rtol=1e-5)

    def test_ricci_flow(self, riemannian_structure, manifold_dim):
        """Test Ricci flow evolution."""
        # Initial metric
        g0 = torch.eye(manifold_dim) + 0.1 * torch.randn(manifold_dim, manifold_dim)
        g0 = (g0 + g0.T) / 2  # Ensure symmetry

        # Evolve metric under Ricci flow
        t = torch.linspace(0, 1, 100)
        g_t = riemannian_structure.evolve_ricci_flow(g0, t)

        # Test metric properties along flow
        for g in g_t:
            # Symmetry
            assert torch.allclose(g, g.transpose(-1, -2), rtol=1e-5)

            # Positive definiteness
            eigenvals = torch.linalg.eigvalsh(g)
            assert torch.all(eigenvals > 0)

        # Test convergence in constant curvature case
        if hasattr(riemannian_structure, "constant_curvature"):
            g_final = g_t[-1]
            ricci_final = riemannian_structure.ricci_tensor(g_final)
            assert torch.allclose(
                ricci_final,
                riemannian_structure.constant_curvature * g_final,
                rtol=1e-4,
            )

    def test_parallel_transport(self, riemannian_structure, manifold_dim):
        """Test parallel transport and holonomy."""
        # Generate a loop
        t = torch.linspace(0, 2 * np.pi, 100)
        loop = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        if manifold_dim > 2:
            loop = torch.cat([loop, torch.zeros(len(t), manifold_dim - 2)], dim=1)

        # Initial vector to transport
        v0 = torch.randn(manifold_dim)
        v0 = v0 / torch.norm(v0)

        # Parallel transport
        v_t = riemannian_structure.parallel_transport(v0, loop)

        # Test preservation of length
        for v in v_t:
            assert torch.allclose(torch.norm(v), torch.norm(v0), rtol=1e-5)

        # Compute holonomy
        holonomy = v_t[-1] @ v0.T
        assert torch.allclose(
            holonomy @ holonomy.T, torch.eye(manifold_dim), rtol=1e-5
        ), "Holonomy should be orthogonal"

    def test_killing_fields(self, riemannian_structure, manifold_dim):
        """Test Killing vector fields."""

        # Generate vector field
        def vector_field(x):
            return torch.randn(manifold_dim)

        # Test Killing equation
        point = torch.randn(manifold_dim)
        lie_derivative = riemannian_structure.lie_derivative_metric(point, vector_field)

        # A vector field is Killing if its Lie derivative of the metric vanishes
        is_killing = torch.allclose(
            lie_derivative, torch.zeros_like(lie_derivative), rtol=1e-5
        )

        if hasattr(riemannian_structure, "isometries"):
            # If we know the isometry group dimension
            killing_dim = riemannian_structure.isometries.dimension
            rank = torch.linalg.matrix_rank(lie_derivative)
            assert rank <= killing_dim, "Too many independent Killing fields"

    def test_metric_properties(self, riemannian_manifold, batch_size, manifold_dim):
        """Test Riemannian metric properties."""

        # Initialize metric
        def test_metric():
            """Test basic metric properties."""
            g = riemannian_manifold.get_metric()

            # Test symmetry
            assert torch.allclose(g, g.transpose(-1, -2)), "Metric should be symmetric"

            # Test positive definiteness
            v = torch.randn(batch_size, manifold_dim)
            assert torch.all(
                riemannian_manifold.inner_product(v, v) > 0
            ), "Metric should be positive definite"

            return g

        g = test_metric()

        # Test metric compatibility
        def test_compatibility():
            """Test metric compatibility with connection."""
            # Get Levi-Civita connection
            nabla = riemannian_manifold.get_levi_civita()

            # Test metric compatibility
            X = riemannian_manifold.random_vector_field()
            Y = riemannian_manifold.random_vector_field()
            Z = riemannian_manifold.random_vector_field()

            nabla_g = riemannian_manifold.covariant_derivative(
                lambda v: riemannian_manifold.inner_product(Y(v), Z(v)), X
            )

            assert torch.allclose(
                nabla_g, torch.zeros_like(nabla_g), atol=1e-5
            ), "Connection should be metric compatible"

            return nabla

        nabla = test_compatibility()

        # Test frame bundle
        def test_frame_bundle():
            """Test orthonormal frame bundle."""
            # Get orthonormal frame
            frame = riemannian_manifold.get_orthonormal_frame()

            # Test orthonormality
            gram = riemannian_manifold.compute_gram_matrix(frame)
            assert torch.allclose(
                gram, torch.eye(manifold_dim)
            ), "Frame should be orthonormal"

            # Test structure equations
            theta = riemannian_manifold.get_canonical_form(frame)
            omega = riemannian_manifold.get_connection_form(frame)

            d_theta = riemannian_manifold.exterior_derivative(theta)
            assert torch.allclose(
                d_theta, -omega.wedge(theta)
            ), "Should satisfy first structure equation"

            return frame

        frame = test_frame_bundle()

    def test_curvature_tensors(self, riemannian_manifold, batch_size, manifold_dim):
        """Test Riemannian curvature tensors."""

        # Test Riemann tensor
        def test_riemann():
            """Test Riemann curvature tensor."""
            Riem = riemannian_manifold.get_riemann_tensor()

            # Test symmetries
            X = riemannian_manifold.random_vector_field()
            Y = riemannian_manifold.random_vector_field()
            Z = riemannian_manifold.random_vector_field()
            W = riemannian_manifold.random_vector_field()

            # Antisymmetry in first two arguments
            assert torch.allclose(
                Riem(X, Y, Z, W), -Riem(Y, X, Z, W)
            ), "Should be antisymmetric in first two"

            # Antisymmetry in last two arguments
            assert torch.allclose(
                Riem(X, Y, Z, W), -Riem(X, Y, W, Z)
            ), "Should be antisymmetric in last two"

            # First Bianchi identity
            bianchi1 = Riem(X, Y, Z, W) + Riem(Y, Z, X, W) + Riem(Z, X, Y, W)
            assert torch.allclose(
                bianchi1, torch.zeros_like(bianchi1)
            ), "Should satisfy first Bianchi"

            return Riem

        Riem = test_riemann()

        # Test Ricci tensor
        def test_ricci():
            """Test Ricci curvature tensor."""
            Ric = riemannian_manifold.get_ricci_tensor()

            # Test symmetry
            assert torch.allclose(
                Ric, Ric.transpose(-1, -2)
            ), "Ricci should be symmetric"

            # Test second Bianchi identity
            nabla = riemannian_manifold.get_levi_civita()
            bianchi2 = riemannian_manifold.covariant_derivative(Ric, nabla)
            assert torch.allclose(
                torch.trace(bianchi2),
                0.5
                * riemannian_manifold.gradient(riemannian_manifold.scalar_curvature()),
            ), "Should satisfy second Bianchi"

            return Ric

        Ric = test_ricci()

        # Test sectional curvature
        def test_sectional():
            """Test sectional curvature."""
            # Get orthonormal 2-plane
            e1 = torch.randn(batch_size, manifold_dim)
            e2 = torch.randn(batch_size, manifold_dim)
            e1, e2 = riemannian_manifold.gram_schmidt([e1, e2])

            K = riemannian_manifold.sectional_curvature(e1, e2)

            if riemannian_manifold.has_constant_curvature():
                # Should be constant
                K_values = K.unique()
                assert len(K_values) == 1, "Should have constant curvature"

            return K

        K = test_sectional()

    def test_geodesics(self, riemannian_manifold, batch_size, manifold_dim):
        """Test geodesic properties and computations."""

        # Test geodesic equation
        def test_geodesic_equation():
            """Test geodesic differential equation."""
            # Initial conditions
            p = riemannian_manifold.random_point()
            v = riemannian_manifold.random_tangent_vector(p)

            # Compute geodesic
            gamma = riemannian_manifold.compute_geodesic(p, v)

            # Test that it satisfies geodesic equation
            acceleration = riemannian_manifold.covariant_derivative(
                gamma.velocity, gamma.velocity
            )
            assert torch.allclose(
                acceleration, torch.zeros_like(acceleration)
            ), "Should satisfy geodesic equation"

            return gamma

        gamma = test_geodesic_equation()

        # Test exponential map
        def test_exponential():
            """Test properties of exponential map."""
            p = riemannian_manifold.random_point()
            v = riemannian_manifold.random_tangent_vector(p)

            # Test exp(0) = id
            assert torch.allclose(
                riemannian_manifold.exponential_map(p, torch.zeros_like(v)), p
            ), "exp(0) should be identity"

            # Test differential of exp at 0
            w = 0.001 * v
            exp_w = riemannian_manifold.exponential_map(p, w)
            assert torch.allclose(
                (exp_w - p) / 0.001, v, atol=1e-3
            ), "Differential at 0 should be identity"

            return exp_w

        exp_w = test_exponential()

        # Test Jacobi fields
        def test_jacobi():
            """Test Jacobi field equation."""
            # Get geodesic
            p = riemannian_manifold.random_point()
            v = riemannian_manifold.random_tangent_vector(p)
            gamma = riemannian_manifold.compute_geodesic(p, v)

            # Get Jacobi field
            J = riemannian_manifold.compute_jacobi_field(gamma)

            # Test Jacobi equation
            R = riemannian_manifold.get_riemann_tensor()
            nabla = riemannian_manifold.get_levi_civita()

            jacobi_op = riemannian_manifold.covariant_derivative(
                lambda t: riemannian_manifold.covariant_derivative(J, gamma.velocity)(
                    t
                ),
                gamma.velocity,
            ) + R(gamma.velocity, J, gamma.velocity)

            assert torch.allclose(
                jacobi_op, torch.zeros_like(jacobi_op)
            ), "Should satisfy Jacobi equation"

            return J

        J = test_jacobi()
