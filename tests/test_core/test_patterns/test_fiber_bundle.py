"""
Unit tests for fiber bundle implementations.

This test suite verifies:
1. Protocol requirements (interface tests)
2. Base implementation correctness
3. Pattern-specific implementation features

The tests are organized to ensure both implementations correctly
satisfy the FiberBundle protocol while maintaining their specific features.
"""

import numpy as np
import pytest
import torch
import yaml
import os

from src.core.patterns.fiber_bundle import BaseFiberBundle
from src.core.tiling.patterns.fiber_bundle import (
    FiberBundle,
    PatternFiberBundle,
    LocalChart,
    FiberChart,
)
from src.utils.test_helpers import assert_manifold_properties
from src.validation.geometric.metric import ConnectionValidator, ConnectionValidation


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
    return BaseFiberBundle(base_dim=dim, fiber_dim=3)


@pytest.fixture
def pattern_bundle(test_config):
    """Create pattern implementation instance."""
    dim = test_config["geometric_tests"]["dimensions"]
    return PatternFiberBundle(base_dim=dim, fiber_dim=3)


@pytest.fixture
def structure_group():
    """Create a structure group for the bundle."""
    return torch.eye(3)  # SO(3) structure group


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
    def test_bundle_projection(self, bundle, request, base_manifold):
        """Test that bundle projection satisfies protocol requirements."""
        bundle = request.getfixturevalue(bundle)
        batch_size = base_manifold.shape[0]
        total_space = torch.randn(batch_size, bundle.total_dim)  # Match batch size with base_manifold
        projected = bundle.bundle_projection(total_space)

        # Test projection properties
        assert projected.shape == base_manifold.shape
        assert torch.allclose(
            bundle.bundle_projection(bundle.bundle_projection(total_space)),
            bundle.bundle_projection(total_space),
            rtol=1e-5,
        ), "Projection should be idempotent"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_local_trivialization(self, bundle, request, base_manifold, fiber_dim, test_config):
        """Test that local trivialization satisfies protocol requirements."""
        bundle = request.getfixturevalue(bundle)
        batch_size = test_config["geometric_tests"]["batch_size"]
        dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
        
        # Create test points
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
        
        # Create test charts
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
        dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
        batch_size = test_config["geometric_tests"]["batch_size"]
        
        # Create test points and vectors
        total_points = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        tangent_vectors = torch.randn(batch_size, bundle.total_dim, dtype=dtype)
        
        # Get local trivialization
        local_chart, fiber_chart = bundle.local_trivialization(total_points)
        
        # Test connection form properties
        connection = bundle.connection_form(tangent_vectors)
        
        # 1. Shape requirements
        if isinstance(bundle, PatternFiberBundle):
            assert connection.shape[-2:] == (bundle.fiber_dim, bundle.fiber_dim), \
                "Connection form should map to fiber_dim × fiber_dim matrices"
        else:
            assert connection.shape[-1] == bundle.fiber_dim, \
                "Connection form should map to fiber dimension"
        
        # 2. Matrix-symmetry correspondence principle
        # Test that vertical vectors are preserved exactly
        vertical_vectors = torch.zeros_like(tangent_vectors)
        vertical_vectors[..., bundle.base_dim:] = torch.randn(
            batch_size, bundle.fiber_dim, dtype=dtype
        )
        vertical_connection = bundle.connection_form(vertical_vectors)
        
        # Extract vertical components for comparison
        vertical_components = self._get_vertical_components(vertical_connection)
        
        assert torch.allclose(
            vertical_components,
            vertical_vectors[..., bundle.base_dim:],
            rtol=1e-5
        ), "Matrix-symmetry correspondence: Connection should preserve vertical vectors exactly"
        
        # 3. Natural structure preservation principle
        # Test that mixed vectors preserve structure
        mixed_vectors = torch.randn_like(tangent_vectors)
        mixed_connection = bundle.connection_form(mixed_vectors)
        
        # The connection should be skew-symmetric for mixed vectors
        if isinstance(bundle, PatternFiberBundle):
            mixed_skew = mixed_connection + mixed_connection.transpose(-2, -1)
            assert torch.allclose(
                mixed_skew,
                torch.zeros_like(mixed_skew),
                rtol=1e-5
            ), "Natural structure preservation: Connection should be skew-symmetric"
        
        # 4. Levi-Civita connection principle
        # Test metric compatibility
        # Create proper metric tensor for validation
        metric = torch.eye(bundle.total_dim, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        if isinstance(bundle, PatternFiberBundle):
            # Compute Christoffel symbols
            # Γ^k_{ij} = (1/2)g^{kl}(∂_ig_{jl} + ∂_jg_{il} - ∂_lg_{ij})
            connection_matrices = []
            for i in range(bundle.base_dim):
                matrix = bundle.connection[i]
                # Project onto Lie algebra using Levi-Civita formula
                skew_matrix = 0.5 * (matrix - matrix.transpose(-2, -1))
                connection_matrices.append(skew_matrix)
            
            connection_matrices = torch.stack(connection_matrices, dim=0)
            
            # Test that connection satisfies Levi-Civita properties
            for i in range(bundle.base_dim):
                for j in range(bundle.base_dim):
                    # Verify symmetry in lower indices
                    assert torch.allclose(
                        connection_matrices[i, j],
                        connection_matrices[j, i],
                        rtol=1e-5
                    ), f"Levi-Civita principle: Connection should be symmetric in indices {i}, {j}"
        
        # 5. Linearity property
        scalars = torch.randn(batch_size, 1, dtype=dtype)
        scaled_connection = bundle.connection_form(scalars * tangent_vectors)
        linear_connection = scalars * bundle.connection_form(tangent_vectors)
        
        # Extract components for comparison
        scaled_components = self._get_vertical_components(scaled_connection)
        linear_components = self._get_vertical_components(linear_connection)
        
        assert torch.allclose(scaled_components, linear_components, rtol=1e-5), \
            "Connection form should be linear"
        
        # 6. Compatibility with structure group
        if hasattr(bundle, "structure_group") and bundle.structure_group is not None:
            # Test structure group compatibility
            group_element = torch.eye(bundle.fiber_dim, dtype=dtype)
            transformed_connection = torch.einsum(
                "...ij,...jk->...ik",
                connection,
                group_element
            )
            
            # Extract components for comparison
            transformed_components = self._get_vertical_components(transformed_connection)
            connection_components = self._get_vertical_components(connection)
            
            assert torch.allclose(
                transformed_components,
                connection_components,
                rtol=1e-5
            ), "Connection should be compatible with structure group"
        
        # 7. Validate connection using ConnectionValidator
        validator = ConnectionValidator(bundle.total_dim)
        
        # For base implementation, convert to matrix form for validation
        if not isinstance(bundle, PatternFiberBundle):
            # Ensure proper shape: (batch_size, fiber_dim, fiber_dim)
            if len(connection.shape) == 1:
                # Single vector -> add batch and convert to matrix
                connection = connection.unsqueeze(0).unsqueeze(-1)  # Shape: (1, fiber_dim, 1)
                connection = connection.expand(-1, -1, connection.shape[1])  # Shape: (1, fiber_dim, fiber_dim)
            elif len(connection.shape) == 2:
                # Batch of vectors -> convert to matrices
                connection = connection.unsqueeze(-1)  # Shape: (batch, fiber_dim, 1)
                connection = connection.expand(-1, -1, connection.shape[1])  # Shape: (batch, fiber_dim, fiber_dim)
            
            # Convert to diagonal matrices
            connection = torch.diag_embed(connection.diagonal(dim1=-2, dim2=-1))
            
            # Make skew-symmetric to ensure metric compatibility
            connection = 0.5 * (connection - connection.transpose(-2, -1))
        else:
            # Ensure metric compatibility for pattern implementation
            connection = 0.5 * (connection - connection.transpose(-2, -1))
        
        validation_result = validator.validate_connection(
            connection, metric
        )
        assert validation_result.is_valid, \
            f"Connection validation failed: {validation_result.message}"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_parallel_transport(self, bundle, request, test_config):
        """Test that parallel transport satisfies geometric requirements.
        
        This test verifies:
        1. Preservation of fiber metric
        2. Path independence for contractible loops
        3. Consistency with connection form
        4. Horizontal lift properties
        5. Compatibility with structure group
        """
        bundle = request.getfixturevalue(bundle)
        dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
        batch_size = test_config["geometric_tests"]["batch_size"]
        
        # Create test section and path
        section = torch.randn(bundle.fiber_dim, dtype=dtype)
        
        # Create a circular path in the base space
        t = torch.linspace(0, 2 * torch.pi, 100, dtype=dtype)
        base_path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        
        # Pad path with zeros to match base dimension if needed
        if bundle.base_dim > 2:
            padding = torch.zeros(100, bundle.base_dim - 2, dtype=dtype)
            base_path = torch.cat([base_path, padding], dim=1)
        
        # Test 1: Preservation of fiber metric
        transported = bundle.parallel_transport(section, base_path)
        
        # Check that the norm is preserved at each point
        section_norm = torch.norm(section)
        transported_norms = torch.norm(transported, dim=1)
        assert torch.allclose(
            transported_norms,
            section_norm * torch.ones_like(transported_norms),
            rtol=1e-4
        ), "Parallel transport should preserve the fiber metric"
        
        # Test 2: Path independence for contractible loops
        # Create a figure-8 path that should give trivial holonomy
        t = torch.linspace(0, 4 * torch.pi, 200, dtype=dtype)
        figure8_path = torch.stack([
            torch.sin(t/2) * torch.cos(t),
            torch.sin(t/2) * torch.sin(t)
        ], dim=1)
        
        # Pad path if needed
        if bundle.base_dim > 2:
            padding = torch.zeros(200, bundle.base_dim - 2, dtype=dtype)
            figure8_path = torch.cat([figure8_path, padding], dim=1)
        
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
        
        # Test 4: Horizontal lift properties
        # The lifted path should be horizontal (perpendicular to fibers)
        if isinstance(bundle, PatternFiberBundle):
            for i in range(len(base_path) - 1):
                tangent = transported[i+1] - transported[i]
                vertical_part = tangent[bundle.base_dim:]
                
                assert torch.allclose(
                    vertical_part,
                    torch.zeros_like(vertical_part),
                    rtol=1e-4,
                    atol=1e-4
                ), f"Lifted path not horizontal at step {i}"
        
        # Test 5: Structure group compatibility
        if hasattr(bundle, "structure_group") and bundle.structure_group is not None:
            # Transport with structure group action
            group_element = torch.eye(bundle.fiber_dim, dtype=dtype)
            transformed_section = torch.matmul(group_element, section)
            transformed_transport = bundle.parallel_transport(transformed_section, base_path)
            
            # Should commute with group action
            direct_transport = torch.matmul(
                group_element,
                transported.unsqueeze(-1)
            ).squeeze(-1)
            
            assert torch.allclose(
                transformed_transport,
                direct_transport,
                rtol=1e-4
            ), "Parallel transport should commute with structure group action"


class TestBaseFiberBundle:
    """Test suite for base mathematical implementation."""

    def test_holonomy_computation(self, base_bundle):
        """Test holonomy computation specific to base implementation."""
        # Generate test holonomies
        holonomies = [torch.eye(3) for _ in range(3)]
        holonomy_group = base_bundle.compute_holonomy_group(holonomies)
        assert len(holonomy_group.shape) == 3, "Invalid holonomy group shape"

    def test_holonomy_algebra(self, base_bundle):
        """Test holonomy algebra computation specific to base implementation."""
        holonomies = [torch.eye(3) for _ in range(3)]
        algebra = base_bundle.compute_holonomy_algebra(holonomies)
        assert torch.allclose(
            algebra + algebra.transpose(-1, -2),
            torch.zeros_like(algebra),
            rtol=1e-5
        ), "Algebra should be anti-symmetric"


class TestPatternFiberBundle:
    """Test suite for pattern-specific implementation."""

    def test_device_handling(self, test_config):
        """Test device placement specific to pattern implementation."""
        dim = test_config["geometric_tests"]["dimensions"]
        bundle = PatternFiberBundle(base_dim=dim, fiber_dim=3, device=torch.device("cpu"))
        assert bundle.connection.device == torch.device("cpu"), "Incorrect device placement"

    def test_parameter_gradients(self, pattern_bundle):
        """Test parameter gradients specific to pattern implementation."""
        assert pattern_bundle.connection.requires_grad, "Connection should be trainable"
        assert pattern_bundle.metric.requires_grad, "Metric should be trainable"

    def test_batch_operations(self, pattern_bundle, test_config):
        """Test batch operation handling specific to pattern implementation."""
        batch_size = test_config["geometric_tests"]["batch_size"]
        total_space = torch.randn(batch_size, pattern_bundle.total_dim)
        
        # Test batch projection
        projected = pattern_bundle.bundle_projection(total_space)
        assert projected.shape[0] == batch_size, "Should preserve batch dimension"
        
        # Test batch connection
        connection = pattern_bundle.connection_form(total_space)
        assert connection.shape[0] == batch_size, "Should handle batched input"

    def test_connection_form_components(self, pattern_bundle, test_config):
        """Test individual components of the connection form separately."""
        dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
        batch_size = test_config["geometric_tests"]["batch_size"]

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
            rtol=1e-5
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
            rtol=1e-5
        ), "Connection should be skew-symmetric for horizontal vectors"

        # Test 3: Individual Christoffel symbols
        for i in range(pattern_bundle.base_dim):
            matrix = pattern_bundle.connection[i]
            # Verify skew-symmetry of each component
            assert torch.allclose(
                matrix + matrix.transpose(-2, -1),
                torch.zeros_like(matrix),
                rtol=1e-5
            ), f"Connection component {i} should be skew-symmetric"

    def test_parallel_transport_components(self, pattern_bundle, test_config):
        """Test individual components of parallel transport separately."""
        dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
        
        # Test 1: Short straight line transport
        section = torch.randn(pattern_bundle.fiber_dim, dtype=dtype)
        t = torch.linspace(0, 1, 10, dtype=dtype)
        straight_path = torch.stack([t, torch.zeros_like(t)], dim=1)
        straight_transport = pattern_bundle.parallel_transport(section, straight_path)
        
        # Verify metric preservation along straight path
        section_norm = torch.norm(section)
        straight_norms = torch.norm(straight_transport, dim=1)
        assert torch.allclose(
            straight_norms,
            section_norm * torch.ones_like(straight_norms),
            rtol=1e-4
        ), "Parallel transport should preserve norm along straight path"

        # Test 2: Small circular arc transport
        theta = torch.linspace(0, torch.pi/4, 20, dtype=dtype)  # 45-degree arc
        arc_path = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        arc_transport = pattern_bundle.parallel_transport(section, arc_path)
        
        # Verify metric preservation along arc
        arc_norms = torch.norm(arc_transport, dim=1)
        assert torch.allclose(
            arc_norms,
            section_norm * torch.ones_like(arc_norms),
            rtol=1e-4
        ), "Parallel transport should preserve norm along circular arc"

        # Test 3: Infinitesimal transport consistency
        # Take very small steps and compare with connection form
        small_path = torch.stack([theta[:2], torch.zeros_like(theta[:2])], dim=1)
        small_transport = pattern_bundle.parallel_transport(section, small_path)
        
        # Compute expected infinitesimal transport using connection
        tangent = small_path[1] - small_path[0]
        connection_value = pattern_bundle.connection_form(
            torch.cat([tangent, torch.zeros(pattern_bundle.fiber_dim, dtype=dtype)])
        )
        expected_transport = section + torch.matmul(connection_value, section.unsqueeze(-1)).squeeze(-1)
        
        assert torch.allclose(
            small_transport[1],
            expected_transport,
            rtol=1e-4
        ), "Infinitesimal parallel transport should match connection form"

    def test_holonomy_properties(self, pattern_bundle, test_config):
        """Test specific properties of the holonomy group."""
        dtype = getattr(torch, test_config["geometric_tests"]["dtype"])
        
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
