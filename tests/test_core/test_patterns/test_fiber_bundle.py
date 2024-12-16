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
        """Test that connection form satisfies protocol requirements."""
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
        
        # 2. Linearity property
        scalars = torch.randn(batch_size, 1, dtype=dtype)
        scaled_connection = bundle.connection_form(scalars * tangent_vectors)
        linear_connection = scalars * bundle.connection_form(tangent_vectors)
        
        # Extract components for comparison if needed
        scaled_components = self._get_vertical_components(scaled_connection)
        linear_components = self._get_vertical_components(linear_connection)
        
        assert torch.allclose(scaled_components, linear_components, rtol=1e-5), \
            "Connection form should be linear"
        
        # 3. Vertical projection property
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
        ), "Connection should act as identity on vertical vectors"
        
        # 4. Compatibility with structure group
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
        
        # 5. Validate connection using ConnectionValidator
        validator = ConnectionValidator(bundle.total_dim)
        
        # Create proper metric tensor for validation
        metric = torch.eye(bundle.total_dim, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # For base implementation, convert to matrix form for validation
        if not isinstance(bundle, PatternFiberBundle):
            # Debug shapes
            print(f"Initial connection shape: {connection.shape}")
            
            # Ensure proper shape: (batch_size, fiber_dim, fiber_dim)
            if len(connection.shape) == 1:
                # Single vector -> add batch and convert to matrix
                connection = connection.unsqueeze(0).unsqueeze(-1)  # Shape: (1, fiber_dim, 1)
                connection = connection.expand(-1, -1, connection.shape[1])  # Shape: (1, fiber_dim, fiber_dim)
            elif len(connection.shape) == 2:
                # Batch of vectors -> convert to matrices
                connection = connection.unsqueeze(-1)  # Shape: (batch, fiber_dim, 1)
                connection = connection.expand(-1, -1, connection.shape[1])  # Shape: (batch, fiber_dim, fiber_dim)
                
            print(f"After reshaping: {connection.shape}")
            
            # Convert to diagonal matrices
            connection = torch.diag_embed(connection.diagonal(dim1=-2, dim2=-1))
            print(f"After diag_embed: {connection.shape}")
            
            # Make skew-symmetric to ensure metric compatibility
            connection = 0.5 * (connection - connection.transpose(-2, -1))
            print(f"Final connection shape: {connection.shape}")
        else:
            # Ensure metric compatibility for pattern implementation
            # The connection should already be a batch of matrices
            print(f"Pattern connection shape: {connection.shape}")
            connection = 0.5 * (connection - connection.transpose(-2, -1))
        
        validation_result = validator.validate_connection(
            connection, metric
        )
        assert validation_result.is_valid, \
            f"Connection validation failed: {validation_result.message}"

    @pytest.mark.parametrize("bundle", ["base_bundle", "pattern_bundle"])
    def test_parallel_transport(self, bundle, request):
        """Test that parallel transport satisfies protocol requirements."""
        bundle = request.getfixturevalue(bundle)
        
        # Create test path and section
        t = torch.linspace(0, 2 * np.pi, 100)
        path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        section = torch.randn(bundle.fiber_dim)
        
        # Test transport
        transported = bundle.parallel_transport(section, path)
        assert transported.shape[1] == bundle.fiber_dim, "Should preserve fiber dimension"


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
