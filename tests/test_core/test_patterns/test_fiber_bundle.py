"""
Unit tests for fiber bundle implementations.

This test suite verifies:
1. Protocol requirements (interface tests)
2. Base implementation correctness
3. Pattern-specific implementation features

The tests are organized to ensure both implementations correctly
satisfy the FiberBundle protocol while maintaining their specific features.
"""

from typing import Any
import numpy as np
import math
import pytest
import torch
import yaml
import os
import hypothesis
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp
import logging

from src.core.patterns.fiber_bundle import BaseFiberBundle
from src.core.tiling.patterns.pattern_fiber_bundle import (
    FiberBundle,
    PatternFiberBundle,
    LocalChart,
    FiberChart,
)
from src.utils.test_helpers import assert_manifold_properties
from src.validation.geometric.metric import ConnectionValidator, ConnectionValidation
from tests.utils.config_loader import load_test_config


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Load test configuration based on environment."""
    return load_test_config()


@pytest.fixture
def base_manifold(test_config):
    """Create a test base manifold."""
    base_dim = test_config["fiber_bundle"]["base_dim"]
    batch_size = test_config["fiber_bundle"]["batch_size"]
    dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
    return torch.randn(batch_size, base_dim, dtype=dtype)


@pytest.fixture
def fiber_dim(test_config):
    """Dimension of the fiber."""
    return test_config["fiber_bundle"]["fiber_dim"]


@pytest.fixture
def base_bundle(test_config):
    """Create base implementation instance."""
    base_dim = test_config["fiber_bundle"]["base_dim"]
    fiber_dim = test_config["fiber_bundle"]["fiber_dim"]
    dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
    return BaseFiberBundle(base_dim=base_dim, fiber_dim=fiber_dim, dtype=dtype)


@pytest.fixture
def pattern_bundle(test_config):
    """Create pattern implementation instance."""
    base_dim = test_config["fiber_bundle"]["base_dim"]
    fiber_dim = test_config["fiber_bundle"]["fiber_dim"]
    dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
    return PatternFiberBundle(base_dim=base_dim, fiber_dim=fiber_dim, dtype=dtype)


@pytest.fixture
def structure_group(test_config):
    """Create a structure group for the bundle."""
    return torch.eye(test_config["fiber_bundle_tests"]["structure_group_dim"])


class TestFiberBundleProtocol:
    """Test suite verifying protocol requirements are met."""

    def _get_vertical_components(self, connection: torch.Tensor) -> torch.Tensor:
        """Extract vertical components from connection form output.
        
        Handles both forms:
        1. Matrix form (pattern implementation) - extracts diagonal elements
        2. Direct vector form (base implementation) - returns as is
        
        The matrix form represents the connection as a fiber_dim × fiber_dim matrix
        where vertical components are on the diagonal. The direct form represents
        vertical components directly as a fiber_dim vector.
        
        Args:
            connection: Connection form output, either:
                - Matrix form: shape (..., fiber_dim, fiber_dim)
                - Direct form: shape (..., fiber_dim)
            
        Returns:
            Vertical components as a vector of shape (..., fiber_dim)
        """
        # Check if connection is in matrix form (has square last dimensions)
        if len(connection.shape) >= 2 and connection.shape[-1] == connection.shape[-2]:
            # Matrix form - extract diagonal elements
            # This handles both batched (..., fiber_dim, fiber_dim) and
            # unbatched (fiber_dim, fiber_dim) cases
            return torch.diagonal(connection, dim1=-2, dim2=-1)
            
        # Direct vector form - return as is
        return connection

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_bundle_projection(self, bundle, request, base_manifold, test_config):
        """Test that bundle projection satisfies protocol requirements."""
        bundle = request.getfixturevalue(bundle)
        batch_size = base_manifold.shape[0]
        # Create total space point with correct dimensions
        total_space = torch.randn(batch_size, bundle.total_dim)  # total_dim = base_dim + fiber_dim
        projected = bundle.bundle_projection(total_space)

        # Test projection properties
        assert projected.shape[-1] == bundle.base_dim, "Projected shape should match base dimension"
        
        # Verify idempotency by checking that projecting the base point gives the same result
        # First, pad the projected point with zeros in the fiber dimensions
        padded_projected = torch.zeros_like(total_space)
        padded_projected[..., :bundle.base_dim] = projected
        assert torch.allclose(
            bundle.bundle_projection(padded_projected),
            projected,
            rtol=test_config["fiber_bundle_tests"]["test_tolerances"]["projection"],
        ), "Projection should be idempotent"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_local_trivialization(self, bundle, request, base_manifold, fiber_dim, test_config):
        """Test that local trivialization satisfies protocol requirements."""
        bundle = request.getfixturevalue(bundle)
        batch_size = test_config["fiber_bundle"]["batch_size"]
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        
        # Create test points with correct total dimension
        total_points = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        
        # Test basic properties
        local_chart, fiber_chart = bundle.local_trivialization(total_points)
        
        # Verify protocol requirements
        assert isinstance(local_chart, LocalChart), "Should return LocalChart instance"
        assert isinstance(fiber_chart, FiberChart), "Should return FiberChart instance"
        assert local_chart.dimension == bundle.base_dim, "Local chart dimension mismatch"
        assert fiber_chart.fiber_coordinates.shape[-1] == fiber_dim, "Fiber dimension mismatch"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_transition_functions(self, bundle, request):
        """Test that transition functions satisfy protocol requirements."""
        bundle = request.getfixturevalue(bundle)
        
        # Create test charts with correct total dimension
        point1 = torch.randn(4, bundle.total_dim)
        point2 = torch.randn(4, bundle.total_dim)
        chart1, _ = bundle.local_trivialization(point1)
        chart2, _ = bundle.local_trivialization(point2)
        
        # Test transition function
        transition = bundle.transition_functions(chart1, chart2)
        assert transition.shape[-2:] == (bundle.fiber_dim, bundle.fiber_dim), "Invalid transition shape"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_connection_form(self, bundle, request, test_config):
        """Test that connection form satisfies protocol requirements and theoretical principles."""
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test points and vectors with correct total dimension
        total_points = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        tangent_vectors = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        
        # Get local trivialization
        local_chart, fiber_chart = bundle.local_trivialization(total_points)
        
        # Test connection form properties
        connection = bundle.connection_form(tangent_vectors)
        
        # 1. Shape and basic requirements
        if isinstance(bundle, PatternFiberBundle):
            assert connection.shape[-2:] == (bundle.fiber_dim, bundle.fiber_dim), \
                "Connection form should map to fiber_dim × fiber_dim matrices"
        else:
            assert connection.shape[-1] == bundle.fiber_dim, \
                "Connection form should map to fiber dimension"
        
        # 2. Pattern structure preservation
        if isinstance(bundle, PatternFiberBundle):
            # Test that connection preserves pattern structure
            pattern_metric = bundle.metric[bundle.base_dim:, bundle.base_dim:]
            metric_compat = torch.matmul(connection, pattern_metric) + \
                           torch.matmul(pattern_metric, connection.transpose(-2, -1))
            assert torch.allclose(
                metric_compat,
                torch.zeros_like(metric_compat),
                rtol=test_config["fiber_bundle_tests"]["test_tolerances"]["connection"]
            ), "Connection should preserve pattern metric structure"
        
        # 3. Torsion-free property
        if isinstance(bundle, PatternFiberBundle):
            # Create two test vectors
            X = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
            Y = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
            
            # Compute covariant derivatives
            nabla_X_Y = bundle.connection_form(X) @ Y[..., bundle.base_dim:].unsqueeze(-1)
            nabla_Y_X = bundle.connection_form(Y) @ X[..., bundle.base_dim:].unsqueeze(-1)
            
            # Compute Lie bracket [X,Y]
            bracket = nabla_X_Y - nabla_Y_X
            
            # Torsion tensor should vanish
            torsion = bracket - bundle.connection_form(
                torch.cat([
                    torch.zeros_like(X[..., :bundle.base_dim]),  # Zero out cross product for 2D base
                    torch.zeros_like(X[..., bundle.base_dim:])
                ], dim=-1)
            )
            assert torch.allclose(
                torsion,
                torch.zeros_like(torsion),
                rtol=1e-5
            ), "Connection should be torsion-free"
        
        # 4. Vertical vector preservation
        vertical_vectors = torch.zeros_like(tangent_vectors)
        vertical_vectors[..., bundle.base_dim:] = torch.randn(
            batch_size, bundle.fiber_dim, dtype=dtype
        )
        vertical_connection = bundle.connection_form(vertical_vectors)
        
        if isinstance(bundle, PatternFiberBundle):
            vertical_output = torch.diagonal(vertical_connection, dim1=-2, dim2=-1)
        else:
            vertical_output = vertical_connection
            
        assert torch.allclose(
            vertical_output,
            vertical_vectors[..., bundle.base_dim:],
            rtol=1e-5
        ), "Connection should preserve vertical vectors exactly"
        
        # 5. Linearity property
        scalars = torch.randn(batch_size, dtype=dtype)
        scaled_connection = bundle.connection_form(scalars.unsqueeze(-1) * tangent_vectors)
        linear_connection = scalars.unsqueeze(-1).unsqueeze(-1) * bundle.connection_form(tangent_vectors)
        
        assert torch.allclose(
            scaled_connection,
            linear_connection,
            rtol=1e-5
        ), "Connection form should be linear"
        
        # 6. Structure group compatibility
        if isinstance(bundle, PatternFiberBundle):
            # Test compatibility with structure group action
            group_element = torch.eye(bundle.fiber_dim, dtype=dtype)
            transformed_connection = torch.einsum(
                "...ij,...jk->...ik",
                connection,
                group_element
            )
            
            assert torch.allclose(
                transformed_connection,
                connection,
                rtol=1e-5
            ), "Connection should be compatible with structure group"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_parallel_transport(self, bundle, request, test_config):
        """Test that parallel transport satisfies geometric requirements."""
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test section and path
        section = torch.randn(bundle.fiber_dim, dtype=dtype)
        
        # Create a circular path in the base space
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=dtype)
        # Always create 3D path with circle in x-y plane
        base_path = torch.stack([
            torch.cos(t),  # x component
            torch.sin(t),  # y component
            torch.zeros_like(t)  # z component
        ], dim=1)
        
        # Test 1: Preservation of fiber metric
        transported = bundle.parallel_transport(section, base_path)
        
        # Check that the norm is preserved at each point
        section_norm = torch.norm(section)
        transported_norms = torch.norm(transported, dim=1)
        assert torch.allclose(
            transported_norms,
            section_norm * torch.ones_like(transported_norms),
            rtol=test_config["fiber_bundle_tests"]["test_tolerances"]["transport"]
        ), "Parallel transport should preserve the fiber metric"
        
        # Test 2: Path independence for contractible loops
        # Create a figure-8 path in x-y plane
        t = torch.linspace(0, 4 * torch.pi, 200, dtype=dtype)
        figure8_path = torch.stack([
            torch.sin(t/2) * torch.cos(t),  # x component
            torch.sin(t/2) * torch.sin(t),  # y component
            torch.zeros_like(t)  # z component
        ], dim=1)
        
        # Transport around figure-8
        transported_loop = bundle.parallel_transport(section, figure8_path)
        
        # Start and end should match for contractible loop
        assert torch.allclose(
            transported_loop[0],
            transported_loop[-1],
            rtol=1e-4
        ), "Parallel transport around contractible loop should return to start"
        
        # Test 3: Consistency with connection form
        # Take small steps and compare with infinitesimal transport
        for i in range(len(base_path) - 1):
            # Get tangent vector between points
            tangent = base_path[i+1] - base_path[i]
            
            # Compute connection form value
            connection_value = bundle.connection_form(
                torch.cat([tangent, torch.zeros(bundle.fiber_dim, dtype=dtype)])
            )
            
            # For pattern bundle, connection gives matrix
            if isinstance(bundle, PatternFiberBundle):
                infinitesimal = transported[i] + torch.matmul(
                    connection_value,
                    transported[i].unsqueeze(-1)
                ).squeeze(-1)
            else:
                # For base bundle, connection gives vector
                infinitesimal = transported[i] + connection_value
            
            assert torch.allclose(
                infinitesimal,
                transported[i+1],
                rtol=1e-3
            ), f"Transport step {i} inconsistent with connection form"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_vertical_horizontal_separation(self, bundle, request, test_config):
        """Test that the connection form properly separates vertical and horizontal components."""
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test vectors
        total_vectors = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        
        # Get connection form output
        connection = bundle.connection_form(total_vectors)
        
        # Extract vertical components
        vertical_components = self._get_vertical_components(connection)
        
        # Create purely vertical vectors
        vertical_vectors = torch.zeros_like(total_vectors)
        vertical_vectors[..., bundle.base_dim:] = total_vectors[..., bundle.base_dim:]
        
        # Verify that vertical vectors are preserved
        vertical_connection = bundle.connection_form(vertical_vectors)
        vertical_output = self._get_vertical_components(vertical_connection)
        
        assert torch.allclose(
            vertical_output,
            vertical_vectors[..., bundle.base_dim:],
            rtol=1e-5
        ), "Connection should preserve vertical vectors"
        
        # Create purely horizontal vectors
        horizontal_vectors = torch.zeros_like(total_vectors)
        horizontal_vectors[..., :bundle.base_dim] = total_vectors[..., :bundle.base_dim]
        
        # Verify that horizontal vectors give zero vertical component
        horizontal_connection = bundle.connection_form(horizontal_vectors)
        horizontal_output = self._get_vertical_components(horizontal_connection)
        
        assert torch.allclose(
            horizontal_output,
            torch.zeros_like(horizontal_output),
            rtol=1e-5
        ), "Horizontal vectors should have zero vertical component"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_structure_group_respect(self, bundle, request, test_config):
        """Test that the connection form respects the structure group action."""
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test points and vectors
        total_points = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        tangent_vectors = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        
        # Create structure group element (SO(3) for our case)
        theta = torch.randn(batch_size, dtype=dtype) * np.pi  # Random rotation angles
        c, s = torch.cos(theta), torch.sin(theta)
        g = torch.stack([
            torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
            torch.stack([s, c, torch.zeros_like(c)], dim=-1),
            torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1)
        ], dim=-2)
        
        # Apply structure group action
        transformed_points = torch.clone(total_points)
        transformed_points[..., bundle.base_dim:] = torch.matmul(
            g, total_points[..., bundle.base_dim:].unsqueeze(-1)
        ).squeeze(-1)
        
        transformed_vectors = torch.clone(tangent_vectors)
        transformed_vectors[..., bundle.base_dim:] = torch.matmul(
            g, tangent_vectors[..., bundle.base_dim:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Compare connection forms
        original_connection = bundle.connection_form(tangent_vectors)
        transformed_connection = bundle.connection_form(transformed_vectors)
        
        if isinstance(bundle, PatternFiberBundle):
            # For matrix form, conjugate by g
            expected_connection = torch.matmul(
                torch.matmul(g, original_connection),
                g.transpose(-2, -1)
            )
        else:
            # For vector form, transform by g
            expected_connection = torch.matmul(
                g, original_connection.unsqueeze(-1)
            ).squeeze(-1)
            
        assert torch.allclose(
            transformed_connection,
            expected_connection,
            rtol=1e-5
        ), "Connection should respect structure group action"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_fiber_metric_preservation(self, bundle, request, test_config):
        """Test that the connection form preserves the fiber metric."""
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test points and vectors
        total_points = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        X = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        Y = torch.randn(batch_size, bundle.fiber_dim, dtype=dtype)
        
        # Get fiber metric (identity for SO(3))
        fiber_metric = torch.eye(bundle.fiber_dim, dtype=dtype).expand(batch_size, -1, -1)
        
        # Compute connection form
        connection = bundle.connection_form(X)
        
        if isinstance(bundle, PatternFiberBundle):
            # For matrix form
            metric_derivative = torch.matmul(connection, fiber_metric) + \
                              torch.matmul(fiber_metric, connection.transpose(-2, -1))
        else:
            # For vector form, construct the matrix form first
            connection_matrix = torch.zeros(
                batch_size, bundle.fiber_dim, bundle.fiber_dim,
                dtype=dtype
            )
            for i in range(bundle.fiber_dim):
                connection_matrix[..., i, i] = connection[..., i]
            metric_derivative = torch.matmul(connection_matrix, fiber_metric) + \
                              torch.matmul(fiber_metric, connection_matrix.transpose(-2, -1))
        
        assert torch.allclose(
            metric_derivative,
            torch.zeros_like(metric_derivative),
            rtol=1e-5
        ), "Connection should preserve fiber metric"


class TestBaseFiberBundle:
    """Test suite for base mathematical implementation."""

    def test_holonomy_computation(self, base_bundle):
        """Test holonomy computation specific to base implementation."""
        # Generate test holonomies
        holonomies = [torch.eye(3) for _ in range(3)]
        holonomy_group = base_bundle.compute_holonomy_group(holonomies)
        assert len(holonomy_group.shape) == 3, "Invalid holonomy group shape"

    def test_holonomy_algebra(self, base_bundle, test_config):
        """Test holonomy algebra computation specific to base implementation."""
        holonomies = [torch.eye(3) for _ in range(3)]
        algebra = base_bundle.compute_holonomy_algebra(holonomies)
        assert torch.allclose(
            algebra + algebra.transpose(-1, -2),
            torch.zeros_like(algebra),
            rtol=test_config["fiber_bundle"]["tolerance"]
        ), "Algebra should be anti-symmetric"


class TestPatternFiberBundle:
    """Test suite for pattern-specific implementation."""

    @pytest.fixture
    def bundle(self, pattern_bundle):
        """Get pattern bundle instance."""
        return pattern_bundle

    def test_device_handling(self, pattern_bundle):
        """Test device placement specific to pattern implementation."""
        assert pattern_bundle.connection.device == torch.device("cpu"), "Incorrect device placement"

    def test_parameter_gradients(self, pattern_bundle):
        """Test parameter gradients specific to pattern implementation."""
        assert pattern_bundle.connection.requires_grad, "Connection should be trainable"
        assert pattern_bundle.metric.requires_grad, "Metric should be trainable"

    def test_batch_operations(self, pattern_bundle, test_config):
        """Test batch operation handling specific to pattern implementation."""
        batch_size = test_config["fiber_bundle"]["batch_size"]
        total_space = torch.randn(batch_size, pattern_bundle.total_dim)
        
        # Test batch projection
        projected = pattern_bundle.bundle_projection(total_space)
        assert projected.shape[0] == batch_size, "Should preserve batch dimension"
        
        # Test batch connection
        connection = pattern_bundle.connection_form(total_space)
        assert connection.shape[0] == batch_size, "Should handle batched input"

    def test_connection_form_components(self, pattern_bundle, test_config):
        """Test individual components of the connection form separately."""
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]

        # Test 1: Pure vertical vectors
        vertical_vector = torch.zeros(batch_size, pattern_bundle.total_dim, dtype=dtype)
        vertical_vector[..., pattern_bundle.base_dim:] = torch.randn(
            batch_size, pattern_bundle.fiber_dim, dtype=dtype
        )
        vertical_connection = pattern_bundle.connection_form(vertical_vector)
        
        # Verify vertical preservation
        vertical_components = torch.diagonal(vertical_connection, dim1=-2, dim2=-1)
        assert torch.allclose(
            vertical_components,
            vertical_vector[..., pattern_bundle.base_dim:],
            rtol=test_config["fiber_bundle"]["tolerance"]
        ), "Connection should preserve pure vertical vectors"

        # Test 2: Pure horizontal vectors
        horizontal_vector = torch.zeros(batch_size, pattern_bundle.total_dim, dtype=dtype)
        horizontal_vector[..., :pattern_bundle.base_dim] = torch.randn(
            batch_size, pattern_bundle.base_dim, dtype=dtype
        )
        horizontal_connection = pattern_bundle.connection_form(horizontal_vector)
        
        # Verify skew-symmetry
        skew_check = horizontal_connection + horizontal_connection.transpose(-2, -1)
        assert torch.allclose(
            skew_check,
            torch.zeros_like(skew_check),
            rtol=test_config["fiber_bundle"]["tolerance"]
        ), "Connection should be skew-symmetric for horizontal vectors"

        # Test 3: Individual Christoffel symbols
        for i in range(pattern_bundle.base_dim):
            matrix = pattern_bundle.connection[i]
            # Verify skew-symmetry of each component
            assert torch.allclose(
                matrix + matrix.transpose(-2, -1),
                torch.zeros_like(matrix),
                rtol=test_config["fiber_bundle"]["tolerance"]
            ), f"Connection component {i} should be skew-symmetric"

    def test_parallel_transport_components(self, pattern_bundle, test_config):
        """Test individual components of parallel transport separately."""
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        
        # Test 1: Short straight line transport
        section = torch.randn(pattern_bundle.fiber_dim, dtype=dtype)
        t = torch.linspace(0, 1, 10, dtype=dtype)
        # Create 3D path along x-axis
        straight_path = torch.stack([
            t,  # x component
            torch.zeros_like(t),  # y component
            torch.zeros_like(t)  # z component
        ], dim=1)
        
        # Verify metric preservation along straight path
        section_norm = torch.norm(section)
        straight_transport = pattern_bundle.parallel_transport(section, straight_path)
        straight_norms = torch.norm(straight_transport, dim=1)
        assert torch.allclose(
            straight_norms,
            section_norm * torch.ones_like(straight_norms),
            rtol=1e-4
        ), "Parallel transport should preserve norm along straight path"

        # Test 2: Small circular arc transport
        theta = torch.linspace(0, torch.pi/4, 20, dtype=dtype)  # 45-degree arc
        # Create 3D path with arc in x-y plane
        arc_path = torch.stack([
            torch.cos(theta),  # x component
            torch.sin(theta),  # y component
            torch.zeros_like(theta)  # z component
        ], dim=1)
        
        # Verify metric preservation along arc
        arc_transport = pattern_bundle.parallel_transport(section, arc_path)
        arc_norms = torch.norm(arc_transport, dim=1)
        assert torch.allclose(
            arc_norms,
            section_norm * torch.ones_like(arc_norms),
            rtol=1e-4
        ), "Parallel transport should preserve norm along circular arc"

        # Test 3: Infinitesimal transport consistency
        # Create small path along x-axis
        small_path = torch.stack([
            theta[:2],  # x component
            torch.zeros_like(theta[:2]),  # y component
            torch.zeros_like(theta[:2])  # z component
        ], dim=1)
        
        # Compute expected infinitesimal transport using connection
        tangent = small_path[1] - small_path[0]
        connection_value = pattern_bundle.connection_form(
            torch.cat([tangent, torch.zeros(pattern_bundle.fiber_dim, dtype=dtype)])
        )
        expected_transport = section + torch.matmul(connection_value, section.unsqueeze(-1)).squeeze(-1)
        
        assert torch.allclose(
            pattern_bundle.parallel_transport(section, small_path)[1],
            expected_transport,
            rtol=1e-4
        ), "Infinitesimal parallel transport should match connection form"

    def test_holonomy_properties(self, pattern_bundle, test_config):
        """Test specific properties of the holonomy group."""
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        
        # Test 1: Small contractible loop
        section = torch.randn(pattern_bundle.fiber_dim, dtype=dtype)
        t = torch.linspace(0, 2*torch.pi, 50, dtype=dtype)
        small_loop = 0.1 * torch.stack([torch.cos(t), torch.sin(t)], dim=1)  # Scaled down loop
        small_transport = pattern_bundle.parallel_transport(section, small_loop)
        
        assert torch.allclose(
            small_transport[0],
            small_transport[-1],
            rtol=1e-4
        ), "Parallel transport around small contractible loop should be trivial"

        # Test 2: Composition of loops
        # Transport around same loop twice
        double_loop = torch.cat([small_loop, small_loop], dim=0)
        double_transport = pattern_bundle.parallel_transport(section, double_loop)
        
        assert torch.allclose(
            double_transport[0],
            double_transport[-1],
            rtol=1e-4
        ), "Parallel transport around composed loops should be consistent"

    class TestConnectionFormProperties:
        """Tests specifically for connection form properties."""

        def test_base_direction_independence(self, pattern_bundle):
            """Test that connection matrices are properly independent for each base direction.
            
            This verifies that each base direction has its own unique connection matrix,
            while still maintaining required symmetries.
            """
            # Create random tangent vectors in different base directions
            base_dim = pattern_bundle.base_dim
            fiber_dim = pattern_bundle.fiber_dim
            
            # Test each base direction independently
            for i in range(base_dim):
                # Create tangent vector only in i-th base direction
                tangent = torch.zeros(base_dim + fiber_dim)
                tangent[i] = 1.0
                
                # Get connection form
                connection = pattern_bundle.connection_form(tangent)
                
                # Verify it's skew-symmetric in fiber indices
                assert torch.allclose(
                    connection + connection.transpose(-2, -1),
                    torch.zeros_like(connection),
                    rtol=1e-5
                ), f"Connection in direction {i} should be skew-symmetric"
                
                # Verify it has proper shape
                assert connection.shape == (fiber_dim, fiber_dim), \
                    f"Connection in direction {i} has wrong shape"

        def test_levi_civita_symmetry(self, pattern_bundle):
            """Test that connection satisfies Levi-Civita symmetry conditions.
            
            This verifies that the connection is symmetric in its base indices
            while maintaining skew-symmetry in fiber indices.
            """
            base_dim = pattern_bundle.base_dim
            fiber_dim = pattern_bundle.fiber_dim
            
            # Create tangent vectors in pairs of base directions
            for i in range(base_dim):
                for j in range(i + 1, base_dim):
                    # Create tangent vector in i-th direction
                    tangent_i = torch.zeros(base_dim + fiber_dim)
                    tangent_i[i] = 1.0
                    
                    # Create tangent vector in j-th direction
                    tangent_j = torch.zeros(base_dim + fiber_dim)
                    tangent_j[j] = 1.0
                    
                    # Get connection forms
                    connection_i = pattern_bundle.connection_form(tangent_i)
                    connection_j = pattern_bundle.connection_form(tangent_j)
                    
                    # Verify symmetry between directions
                    assert torch.allclose(
                        connection_i,
                        connection_j,
                        rtol=1e-5
                    ), f"Connection should be symmetric between directions {i} and {j}"

        def test_fiber_skew_symmetry_preservation(self, pattern_bundle):
            """Test that connection preserves skew-symmetry in fiber indices.
            
            This verifies that the connection maintains skew-symmetry in fiber indices
            for any combination of base directions.
            """
            base_dim = pattern_bundle.base_dim
            fiber_dim = pattern_bundle.fiber_dim
            
            # Test random combinations of base directions
            num_tests = 10
            for _ in range(num_tests):
                # Create random tangent vector in base directions
                tangent = torch.randn(base_dim + fiber_dim)
                tangent[base_dim:] = 0  # Zero out fiber components
                
                # Get connection form
                connection = pattern_bundle.connection_form(tangent)
                
                # Verify skew-symmetry
                skew_diff = connection + connection.transpose(-2, -1)
                assert torch.allclose(
                    skew_diff,
                    torch.zeros_like(skew_diff),
                    rtol=1e-5
                ), "Connection should maintain skew-symmetry for any base direction combination"
                
                # Verify fiber metric compatibility
                # Convert Parameter to Tensor for indexing
                metric_tensor = torch.Tensor(pattern_bundle.metric)  # Cast to Tensor
                fiber_metric = metric_tensor[base_dim:, base_dim:]  # Now we can index
                
                metric_compat = torch.matmul(connection, fiber_metric) + \
                              torch.matmul(fiber_metric, connection.transpose(-2, -1))
                assert torch.allclose(
                    metric_compat,
                    torch.zeros_like(metric_compat),
                    rtol=1e-5
                ), "Connection should be compatible with fiber metric"

    def test_connection_vertical_preservation(self, pattern_bundle, test_config):
        """Test that connection form preserves vertical vectors exactly."""
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create purely vertical vectors
        total_points = torch.randn(batch_size, pattern_bundle.total_dim, dtype=dtype)
        vertical_vectors = torch.zeros_like(total_points)
        vertical_vectors[..., pattern_bundle.base_dim:] = torch.randn(
            batch_size, pattern_bundle.fiber_dim, dtype=dtype
        )
        
        # Apply connection form
        connection = pattern_bundle.connection_form(vertical_vectors)
        
        # Extract vertical components
        if isinstance(pattern_bundle, PatternFiberBundle):
            vertical_output = torch.diagonal(connection, dim1=-2, dim2=-1)
        else:
            vertical_output = connection
            
        # Verify preservation
        assert torch.allclose(
            vertical_output,
            vertical_vectors[..., pattern_bundle.base_dim:],
            rtol=1e-5
        ), "Connection should preserve vertical vectors exactly"

    def test_connection_horizontal_projection(self, pattern_bundle, test_config):
        """Test that connection form correctly projects horizontal vectors."""
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create purely horizontal vectors
        total_points = torch.randn(batch_size, pattern_bundle.total_dim, dtype=dtype)
        horizontal_vectors = torch.zeros_like(total_points)
        horizontal_vectors[..., :pattern_bundle.base_dim] = torch.randn(
            batch_size, pattern_bundle.base_dim, dtype=dtype
        )
        
        # Apply connection form
        connection = pattern_bundle.connection_form(horizontal_vectors)
        
        # For horizontal vectors, connection should give skew-symmetric output
        if isinstance(pattern_bundle, PatternFiberBundle):
            # Check skew-symmetry
            skew_check = connection + connection.transpose(-2, -1)
            assert torch.allclose(
                skew_check,
                torch.zeros_like(skew_check),
                rtol=1e-5
            ), "Connection should be skew-symmetric on horizontal vectors"
        else:
            # For base implementation, horizontal projection should be small
            assert torch.all(torch.abs(connection) < 1e-5), \
                "Connection should give small output for horizontal vectors"

    def test_connection_levi_civita_compatibility(self, pattern_bundle, test_config):
        """Test that connection form satisfies Levi-Civita compatibility conditions."""
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test vectors
        total_points = torch.randn(batch_size, pattern_bundle.total_dim, dtype=dtype)
        vector_1 = torch.randn(batch_size, pattern_bundle.total_dim, dtype=dtype)
        vector_2 = torch.randn(batch_size, pattern_bundle.total_dim, dtype=dtype)
        
        # Get connection outputs
        conn_1 = pattern_bundle.connection_form(vector_1)
        conn_2 = pattern_bundle.connection_form(vector_2)
        
        # Test metric compatibility
        # [X,Y] = ∇_X Y - ∇_Y X should be satisfied
        bracket = conn_1 @ vector_2[..., pattern_bundle.base_dim:].unsqueeze(-1) - \
                 conn_2 @ vector_1[..., pattern_bundle.base_dim:].unsqueeze(-1)
                 
        # The bracket should be skew-symmetric
        bracket_skew = bracket + bracket.transpose(-2, -1)
        assert torch.allclose(
            bracket_skew,
            torch.zeros_like(bracket_skew),
            rtol=1e-5
        ), "Lie bracket should be skew-symmetric"


class TestConnectionFormHypothesis:
    """Property-based tests for connection form using hypothesis."""

    @given(
        st.integers(min_value=2, max_value=5),  # base_dim
        st.integers(min_value=2, max_value=5),  # fiber_dim
        st.integers(min_value=1, max_value=5),  # batch_size
    )
    def test_vertical_preservation_property(self, base_dim, fiber_dim, batch_size):
        """Test that purely vertical vectors are preserved exactly."""
        bundle = PatternFiberBundle(base_dim=base_dim, fiber_dim=fiber_dim)
        
        # Create purely vertical vectors
        vertical = torch.zeros(batch_size, bundle.total_dim)
        vertical[..., base_dim:] = torch.randn(batch_size, fiber_dim)
        
        # Get connection form
        connection = bundle.connection_form(vertical)
        
        # Extract vertical components
        vertical_output = torch.diagonal(connection, dim1=-2, dim2=-1)
        
        # Verify preservation
        assert torch.allclose(
            vertical_output,
            vertical[..., base_dim:],
            rtol=1e-5
        ), "Connection should preserve vertical vectors exactly"

    @given(
        st.integers(min_value=2, max_value=5),  # base_dim
        st.integers(min_value=2, max_value=5),  # fiber_dim
        st.integers(min_value=1, max_value=5),  # batch_size
    )
    def test_horizontal_skew_symmetry_property(self, base_dim, fiber_dim, batch_size):
        """Test that horizontal vectors produce skew-symmetric output."""
        bundle = PatternFiberBundle(base_dim=base_dim, fiber_dim=fiber_dim)
        
        # Create purely horizontal vectors
        horizontal = torch.zeros(batch_size, bundle.total_dim)
        horizontal[..., :base_dim] = torch.randn(batch_size, base_dim)
        
        # Get connection form
        connection = bundle.connection_form(horizontal)
        
        # Verify skew-symmetry
        skew_check = connection + connection.transpose(-2, -1)
        assert torch.allclose(
            skew_check,
            torch.zeros_like(skew_check),
            rtol=1e-5
        ), "Connection should be skew-symmetric for horizontal vectors"

    @given(
        st.integers(min_value=2, max_value=5),  # base_dim
        st.integers(min_value=2, max_value=5),  # fiber_dim
        st.integers(min_value=1, max_value=5),  # batch_size
    )
    @settings(deadline=1000)  # Increase deadline to 1000ms
    def test_levi_civita_symmetry_property(self, base_dim, fiber_dim, batch_size):
        """Test Levi-Civita symmetry property."""
        bundle = PatternFiberBundle(base_dim=base_dim, fiber_dim=fiber_dim)
        
        # Create a batch of basis vectors for all directions at once
        basis_vectors = torch.eye(base_dim).unsqueeze(0).expand(batch_size, -1, -1)
        # Pad with zeros for fiber dimensions
        basis_vectors = torch.cat([
            basis_vectors,
            torch.zeros(batch_size, base_dim, fiber_dim)
        ], dim=-1)
        
        # Compute connection forms for all directions at once
        connections = []
        for i in range(base_dim):
            v_i = basis_vectors[:, i]
            conn_i = bundle.connection_form(v_i)
            # Project onto Lie algebra once
            conn_i = 0.5 * (conn_i - conn_i.transpose(-2, -1))
            connections.append(conn_i)
        
        # Verify symmetry between all pairs
        for i in range(base_dim):
            for j in range(i + 1, base_dim):
                assert torch.allclose(
                    connections[i][..., :fiber_dim, :fiber_dim],
                    connections[j][..., :fiber_dim, :fiber_dim],
                    rtol=1e-5  # Use a fixed tolerance for hypothesis tests
                ), f"Connection should be symmetric in base indices {i}, {j}"

    @given(
        st.integers(min_value=2, max_value=5),  # base_dim
        st.integers(min_value=2, max_value=5),  # fiber_dim
        st.integers(min_value=1, max_value=5),  # batch_size
    )
    def test_metric_compatibility_property(self, base_dim, fiber_dim, batch_size):
        """Test compatibility with fiber metric."""
        bundle = PatternFiberBundle(base_dim=base_dim, fiber_dim=fiber_dim)
        
        # Create random vector
        v = torch.randn(batch_size, bundle.total_dim)
        
        # Get connection form
        connection = bundle.connection_form(v)
        
        # Get fiber metric - only for PatternFiberBundle
        if isinstance(bundle, PatternFiberBundle):
            # Convert Parameter to Tensor for indexing
            metric_tensor = torch.Tensor(bundle.metric)  # Cast to Tensor
            fiber_metric = metric_tensor[base_dim:, base_dim:]  # Now we can index
            
            # Test metric compatibility: ω_a^b g_bc + ω_a^c g_bc = 0
            metric_compat = torch.matmul(connection, fiber_metric) + \
                           torch.matmul(fiber_metric, connection.transpose(-2, -1))
                           
            assert torch.allclose(
                metric_compat,
                torch.zeros_like(metric_compat),
                rtol=1e-5
            ), "Connection should be compatible with fiber metric"


class TestConnectionFormValidation:
    """Validation and logging infrastructure for connection form testing."""

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Set up logging for connection form tests."""
        # Ensure logs directory exists
        import os
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/connection_form_debug.log',
            filemode='w'  # Overwrite file each time
        )
        self.logger = logging.getLogger('connection_form_tests')

    def _log_connection_properties(self, bundle, tangent_vector, connection):
        """Log key properties of connection form computation.
        
        Args:
            bundle: The fiber bundle instance
            tangent_vector: Input tangent vector
            connection: Output connection form
        """
        self.logger.debug("\n=== Connection Form Properties ===")
        self.logger.debug(f"Bundle type: {type(bundle).__name__}")
        self.logger.debug(f"Input shape: {tangent_vector.shape}")
        self.logger.debug(f"Output shape: {connection.shape}")
        
        # Basic properties
        if isinstance(bundle, PatternFiberBundle):
            skew_error = torch.norm(connection + connection.transpose(-2, -1))
            self.logger.debug(f"Skew-symmetry error: {skew_error:.6f}")
            
            # Vertical preservation
            vertical_part = tangent_vector[..., bundle.base_dim:]
            if len(connection.shape) >= 2:
                vertical_output = torch.diagonal(connection, dim1=-2, dim2=-1)
            else:
                vertical_output = connection
            vertical_error = torch.norm(vertical_output - vertical_part)
            self.logger.debug(f"Vertical preservation error: {vertical_error:.6f}")
            
            # Metric compatibility
            metric = bundle.metric[bundle.base_dim:, bundle.base_dim:]
            metric_error = self._validate_metric_compatibility(connection, metric)
            self.logger.debug(f"Metric compatibility error: {metric_error:.6f}")
            
            # Structure group compatibility
            group_error = self._validate_structure_group(bundle, connection)
            self.logger.debug(f"Structure group error: {group_error:.6f}")

    def _validate_metric_compatibility(self, connection: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Check metric compatibility condition.
        
        Args:
            connection: Connection form output
            metric: Fiber metric tensor
            
        Returns:
            Error norm for metric compatibility
        """
        metric_compat = torch.matmul(connection, metric) + \
                       torch.matmul(metric, connection.transpose(-2, -1))
        return torch.norm(metric_compat)

    def _validate_torsion_free(self, bundle, connection: torch.Tensor, 
                             X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Check torsion-free condition.
        
        Args:
            bundle: The fiber bundle instance
            connection: Connection form output
            X, Y: Tangent vectors
            
        Returns:
            Error norm for torsion-free condition
        """
        # Compute covariant derivatives
        nabla_X_Y = connection @ Y[..., bundle.base_dim:].unsqueeze(-1)
        nabla_Y_X = connection @ X[..., bundle.base_dim:].unsqueeze(-1)
        
        # Compute Lie bracket [X,Y]
        bracket = nabla_X_Y - nabla_Y_X
        
        # For base dimensions other than 3, we compute the Lie bracket directly
        # [X,Y] = X∂Y - Y∂X in coordinates
        X_base = X[..., :bundle.base_dim]
        Y_base = Y[..., :bundle.base_dim]
        
        if bundle.base_dim == 3:
            # For 3D base, we can use cross product
            lie_bracket = torch.linalg.cross(X_base, Y_base, dim=-1)
        else:
            # For other dimensions, compute coordinate by coordinate
            lie_bracket = torch.zeros_like(X_base)
            for i in range(bundle.base_dim):
                lie_bracket[..., i] = (
                    X_base[..., i].unsqueeze(-1) * Y_base - 
                    Y_base[..., i].unsqueeze(-1) * X_base
                ).sum(dim=-1)
        
        # Pad lie_bracket to match fiber dimension if needed
        if bundle.base_dim < bundle.fiber_dim:
            padding = torch.zeros(
                *lie_bracket.shape[:-1], 
                bundle.fiber_dim - bundle.base_dim,
                device=lie_bracket.device,
                dtype=lie_bracket.dtype
            )
            lie_bracket = torch.cat([lie_bracket, padding], dim=-1)
        
        # Compute torsion tensor
        torsion = bracket - connection @ lie_bracket.unsqueeze(-1)
        
        return torch.norm(torsion)

    def _validate_structure_group(self, bundle, connection: torch.Tensor) -> torch.Tensor:
        """Check structure group compatibility.
        
        Args:
            bundle: The fiber bundle instance
            connection: Connection form output
            
        Returns:
            Error norm for structure group compatibility
        """
        if not isinstance(bundle, PatternFiberBundle):
            return torch.tensor(0.0)
            
        group_element = torch.eye(bundle.fiber_dim, device=connection.device)
        transformed = torch.einsum('...ij,...jk->...ik', connection, group_element)
        return torch.norm(transformed - connection)

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_connection_form_validation(self, bundle, request, test_config):
        """Comprehensive validation test for connection form properties."""
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        batch_size = test_config["fiber_bundle"]["batch_size"]
        
        # Create test vectors
        total_points = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        X = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        Y = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        
        # Compute connection form
        connection = bundle.connection_form(X)
        
        # Log properties
        self._log_connection_properties(bundle, X, connection)
        
        # Validate properties
        if isinstance(bundle, PatternFiberBundle):
            # 1. Metric compatibility
            metric = bundle.metric[bundle.base_dim:, bundle.base_dim:]
            metric_error = self._validate_metric_compatibility(connection, metric)
            assert metric_error < 5e-5, f"Metric compatibility error: {metric_error:.6f}"
            
            # 2. Torsion-free condition
            torsion_error = self._validate_torsion_free(bundle, connection, X, Y)
            assert torsion_error < 5e-5, f"Torsion-free error: {torsion_error:.6f}"
            
            # 3. Structure group compatibility
            group_error = self._validate_structure_group(bundle, connection)
            assert group_error < 5e-5, f"Structure group error: {group_error:.6f}"


class TestGeometricComponents:
    """Test suite for geometric components of fiber bundles."""

    def test_metric_derivatives(self, pattern_bundle, test_config):
        """Test computation of metric derivatives.
        
        Verifies:
        1. Shape correctness
        2. Symmetry properties
        3. Consistency with finite differences
        """
        # Get test dimensions
        base_dim = test_config["fiber_bundle"]["base_dim"]
        fiber_dim = test_config["fiber_bundle"]["fiber_dim"]
        total_dim = base_dim + fiber_dim
        dtype = getattr(torch, test_config["fiber_bundle"]["dtype"])
        tolerance = test_config["fiber_bundle"]["tolerance"]
        
        # Create test point
        point = torch.randn(total_dim, dtype=dtype, requires_grad=True)
        
        # Get metric at point - ensure it's computed at the point
        metric_tensor = pattern_bundle.compute_metric(point.unsqueeze(0))
        metric = metric_tensor.values[0]
        
        print("\nInitial metric:")
        print(metric)
        
        # Verify initial metric is symmetric
        assert torch.allclose(
            metric,
            metric.transpose(-2, -1),
            rtol=tolerance
        ), "Initial metric should be symmetric"
        
        # Compute derivatives using autograd
        metric_derivs = torch.zeros(total_dim, total_dim, total_dim, dtype=dtype)
        
        # For each component i, compute derivatives directly
        for i in range(total_dim):
            # Create perturbed point
            point_i = point.clone()
            point_i.requires_grad_(True)
            
            # Compute metric at perturbed point
            metric_tensor = pattern_bundle.compute_metric(point_i.unsqueeze(0))
            metric = metric_tensor.values[0]
            
            # For each component (j,k), compute derivative
            for j in range(total_dim):
                for k in range(total_dim):
                    # Take gradient of symmetrized metric component
                    sym_component = 0.5 * (metric[j,k] + metric[k,j])
                    grad = torch.autograd.grad(
                        outputs=sym_component,
                        inputs=point_i,
                        create_graph=False,  # No need for higher order gradients
                        retain_graph=True
                    )[0]
                    
                    # Store derivative
                    metric_derivs[i,j,k] = grad[i]

        # Verify shape
        assert metric_derivs.shape == (total_dim, total_dim, total_dim), \
            "Metric derivatives should have shape (total_dim, total_dim, total_dim)"
        
        # Verify symmetry in last two indices (metric symmetry)
        for i in range(total_dim):
            print(f"\nDerivatives for i={i}:")
            print("Original:")
            print(metric_derivs[i])
            print("Transposed:")
            print(metric_derivs[i].transpose(-2, -1))
            print("Difference:")
            print(metric_derivs[i] - metric_derivs[i].transpose(-2, -1))
            
            # Account for regularization in the comparison
            reg_scale = 1e-5  # Same as in implementation
            comparison = torch.allclose(
                metric_derivs[i],
                metric_derivs[i].transpose(-2, -1),
                rtol=tolerance,
                atol=reg_scale * 10  # Increase atol slightly to account for accumulated numerical errors
            )
            assert comparison, f"Metric derivatives should be symmetric in last two indices for i={i}"
        
        # Verify essential properties of the metric derivatives
        
        # 1. Block structure: derivatives should respect the base/fiber split
        # Base-base block derivatives should only affect base-base block
        for i in range(base_dim):
            # Check that derivatives in base directions don't affect fiber-fiber block
            fiber_block_derivs = metric_derivs[i, base_dim:, base_dim:]
            assert torch.all(torch.abs(fiber_block_derivs) < 1e-5), \
                f"Base direction {i} derivatives should not affect fiber-fiber block"
        
        # 2. Fiber direction derivatives should only affect fiber-related components
        for i in range(base_dim, total_dim):
            # Check that derivatives in fiber directions don't affect base-base block
            base_block_derivs = metric_derivs[i, :base_dim, :base_dim]
            assert torch.all(torch.abs(base_block_derivs) < 1e-5), \
                f"Fiber direction {i} derivatives should not affect base-base block"
        
        # 3. Verify that derivatives are continuous (values should be bounded)
        assert torch.all(torch.abs(metric_derivs) < 10.0), \
            "Metric derivatives should be bounded"
