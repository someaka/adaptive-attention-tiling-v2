"""
Unit tests for the fiber bundle pattern space implementation.

Tests cover:
1. Bundle structure and properties
2. Local trivialization
3. Connection forms
4. Parallel transport
5. Holonomy groups
6. Principal bundle structure
7. Associated bundles
"""

import pytest
import numpy as np
import torch
from typing import Tuple, Any, List
from scipy.integrate import solve_ivp

from src.core.patterns.fiber_bundle import FiberBundle
from src.utils.test_helpers import assert_manifold_properties

class TestFiberBundle:
    @pytest.fixture
    def base_manifold(self):
        """Create a test base manifold."""
        return torch.randn(4, 8)  # 4D base manifold embedded in 8D
        
    @pytest.fixture
    def fiber_bundle(self):
        """Create a test fiber bundle instance."""
        return FiberBundle(base_dim=4, fiber_dim=3)
        
    @pytest.fixture
    def structure_group(self):
        """Create a structure group for the bundle."""
        return torch.eye(3)  # SO(3) structure group

    def test_bundle_projection(self, fiber_bundle, base_manifold):
        """Test that bundle projection preserves base manifold structure."""
        total_space = torch.randn(4, 11)  # 4D base + 3D fiber = 11D total
        projected = fiber_bundle.bundle_projection(total_space)
        
        # Test projection properties
        assert projected.shape == base_manifold.shape
        assert torch.allclose(
            fiber_bundle.bundle_projection(fiber_bundle.bundle_projection(total_space)),
            fiber_bundle.bundle_projection(total_space),
            rtol=1e-5
        ), "Projection should be idempotent"

    def test_local_trivialization(self, fiber_bundle):
        """Test local trivialization maps."""
        point = torch.randn(4, 11)
        local_chart, fiber_chart = fiber_bundle.local_trivialization(point)
        
        # Test chart properties
        assert local_chart.shape[-1] == 4, "Local chart should match base dimension"
        assert fiber_chart.shape[-1] == 3, "Fiber chart should match fiber dimension"
        
        # Test compatibility
        reconstructed = fiber_bundle.reconstruct_from_charts(local_chart, fiber_chart)
        assert torch.allclose(point, reconstructed, rtol=1e-5), "Charts should allow faithful reconstruction"

    def test_connection_form(self, fiber_bundle):
        """Test connection form properties."""
        tangent_vector = torch.randn(4, 11)
        connection = fiber_bundle.connection_form(tangent_vector)
        
        # Test connection properties
        assert connection.shape[-1] == 3, "Connection should map to fiber dimension"
        assert_manifold_properties(connection)
        
        # Test curvature
        curvature = fiber_bundle.compute_curvature(connection)
        assert curvature is not None, "Curvature should be computable"

    def test_parallel_transport(self, fiber_bundle):
        """Test parallel transport along paths."""
        # Create a circular path
        t = torch.linspace(0, 2*np.pi, 100)
        path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        
        # Create a section to transport
        initial_section = torch.randn(3)  # 3D fiber
        
        # Transport the section
        transported = fiber_bundle.parallel_transport(initial_section, path)
        
        # Test transport properties
        assert transported.shape[1] == 3, "Transport should preserve fiber dimension"
        assert torch.allclose(
            transported[-1], transported[0], rtol=1e-4
        ), "Transport around closed loop should be close to identity (up to holonomy)"

    def test_holonomy_group(self, fiber_bundle):
        """Test holonomy group computation and properties."""
        # Generate a family of loops
        def generate_loop(t: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
            return radius * torch.stack([torch.cos(t), torch.sin(t)], dim=1)
            
        t = torch.linspace(0, 2*np.pi, 100)
        loops = [generate_loop(t, r) for r in [0.5, 1.0, 1.5]]
        
        # Compute holonomy for each loop
        holonomies = []
        for loop in loops:
            initial_frame = torch.eye(3)
            transported_frame = fiber_bundle.parallel_transport(initial_frame, loop)
            holonomy = transported_frame[-1] @ initial_frame.inverse()
            holonomies.append(holonomy)
            
        # Test holonomy group properties
        holonomy_group = fiber_bundle.compute_holonomy_group(holonomies)
        assert torch.allclose(
            holonomy_group.determinant(),
            torch.ones(1),
            rtol=1e-5
        ), "Holonomy group should preserve orientation"
        
        # Test holonomy algebra
        holonomy_algebra = fiber_bundle.compute_holonomy_algebra(holonomies)
        assert torch.allclose(
            holonomy_algebra + holonomy_algebra.transpose(-1, -2),
            torch.zeros_like(holonomy_algebra),
            rtol=1e-5
        ), "Holonomy algebra should be anti-symmetric"

    def test_principal_bundle(self, fiber_bundle, structure_group):
        """Test principal bundle structure."""
        # Test right action of structure group
        point = torch.randn(4, 11)
        transformed = fiber_bundle.right_action(point, structure_group)
        assert transformed.shape == point.shape
        
        # Test equivariance of connection
        vector = torch.randn(4, 11)
        connection = fiber_bundle.connection_form(vector)
        transformed_connection = fiber_bundle.connection_form(
            fiber_bundle.right_action(vector, structure_group)
        )
        assert torch.allclose(
            transformed_connection,
            structure_group.inverse() @ connection @ structure_group,
            rtol=1e-5
        ), "Connection should be equivariant"
        
        # Test structure group orbit
        orbit = fiber_bundle.compute_orbit(point)
        assert len(orbit.shape) == 3, "Orbit should be a family of points"
        assert orbit.shape[-1] == point.shape[-1]

    def test_associated_bundles(self, fiber_bundle, structure_group):
        """Test associated bundle constructions."""
        # Create associated vector bundle
        vector_bundle = fiber_bundle.construct_associated_bundle(
            representation_dim=2
        )
        assert hasattr(vector_bundle, 'transition_functions')
        
        # Test induced connection
        point = torch.randn(4, 11)
        vector = torch.randn(4, 2)  # Vector in associated bundle
        induced_connection = vector_bundle.induced_connection(
            fiber_bundle.connection_form(point)
        )
        assert induced_connection.shape[-2:] == (2, 2)
        
        # Test parallel transport in associated bundle
        t = torch.linspace(0, 2*np.pi, 100)
        path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        initial_vector = torch.randn(2)
        transported = vector_bundle.parallel_transport(initial_vector, path)
        assert transported.shape[1] == 2

    def test_bundle_metrics(self, fiber_bundle):
        """Test bundle metric properties."""
        # Generate bundle metric
        total_metric = fiber_bundle.generate_bundle_metric()
        assert total_metric.shape[-2:] == (11, 11)
        
        # Test metric compatibility with projection
        horizontal = fiber_bundle.horizontal_projection(torch.randn(4, 11))
        vertical = fiber_bundle.vertical_projection(torch.randn(4, 11))
        assert torch.allclose(
            horizontal @ total_metric @ vertical.transpose(-1, -2),
            torch.zeros_like(horizontal),
            rtol=1e-5
        ), "Horizontal and vertical spaces should be orthogonal"
        
        # Test metric compatibility with connection
        connection_metric = fiber_bundle.compute_connection_metric()
        assert torch.allclose(
            connection_metric @ connection_metric.transpose(-1, -2),
            torch.eye(connection_metric.shape[-1]),
            rtol=1e-5
        ), "Connection metric should be orthogonal"

    def test_bundle_structure(
        self, fiber_bundle, base_manifold, fiber_dim
    ):
        """Test fundamental fiber bundle structure."""
        # Test total space structure
        def test_total_space():
            """Test properties of total space."""
            total_space = fiber_bundle.get_total_space()
            assert total_space.dim == base_manifold.dim + fiber_dim, \
                "Total space dimension should be sum"
            # Test fiber projection
            projection = fiber_bundle.get_projection()
            assert projection(total_space).dim == base_manifold.dim, \
                "Projection should map to base"
            return total_space, projection
            
        total_space, projection = test_total_space()
        
        # Test local trivialization
        def test_trivialization():
            """Test local trivialization properties."""
            # Get local chart
            chart = fiber_bundle.get_local_chart()
            local_triv = fiber_bundle.get_local_trivialization(chart)
            
            # Test fiber preservation
            point = total_space.sample_point()
            fiber_point = local_triv(point)
            assert fiber_bundle.is_fiber_preserving(
                local_triv
            ), "Should preserve fibers"
            
            # Test transition functions
            chart2 = fiber_bundle.get_overlapping_chart(chart)
            transition = fiber_bundle.get_transition_function(
                chart, chart2
            )
            assert fiber_bundle.is_smooth_transition(
                transition
            ), "Transition should be smooth"
            
            return local_triv, transition
            
        local_triv, transition = test_trivialization()
        
        # Test bundle sections
        def test_sections():
            """Test properties of bundle sections."""
            section = fiber_bundle.get_local_section()
            assert fiber_bundle.is_section(
                section
            ), "Should be valid section"
            
            # Test smoothness
            assert fiber_bundle.is_smooth_section(
                section
            ), "Section should be smooth"
            
            # Test section space
            section_space = fiber_bundle.get_section_space()
            assert section in section_space, \
                "Section should be in section space"
            
            return section, section_space
            
        section, section_space = test_sections()

    def test_connection_forms(
        self, fiber_bundle, base_manifold, fiber_dim
    ):
        """Test connection forms and horizontal distribution."""
        # Test connection form
        def test_connection():
            """Test properties of connection form."""
            omega = fiber_bundle.get_connection_form()
            
            # Test vertical space annihilation
            vertical = fiber_bundle.get_vertical_space()
            assert all(
                torch.allclose(
                    omega(v), torch.zeros_like(omega(v))
                ) for v in vertical.basis
            ), "Should annihilate vertical vectors"
            
            # Test equivariance
            g = fiber_bundle.get_structure_group_element()
            Ad_g = fiber_bundle.get_adjoint_action(g)
            assert torch.allclose(
                omega(fiber_bundle.group_action(g, vertical.basis[0])),
                Ad_g @ omega(vertical.basis[0])
            ), "Should be equivariant"
            
            return omega
            
        omega = test_connection()
        
        # Test horizontal distribution
        def test_horizontal():
            """Test horizontal distribution properties."""
            H = fiber_bundle.get_horizontal_distribution()
            
            # Test complementarity
            V = fiber_bundle.get_vertical_space()
            direct_sum = fiber_bundle.direct_sum(H, V)
            assert torch.allclose(
                direct_sum.dim,
                fiber_bundle.get_total_space().dim
            ), "Should be complementary"
            
            # Test parallelism
            assert fiber_bundle.is_parallel_distribution(
                H
            ), "Should be parallel"
            
            return H
            
        H = test_horizontal()
        
        # Test curvature
        def test_curvature():
            """Test curvature of connection."""
            Omega = fiber_bundle.get_curvature_form()
            
            # Test Cartan structure equation
            d_omega = fiber_bundle.exterior_derivative(omega)
            bracket_term = fiber_bundle.wedge_product(
                omega, omega
            )
            assert torch.allclose(
                Omega, d_omega + 0.5 * bracket_term
            ), "Should satisfy structure equation"
            
            # Test Bianchi identity
            d_Omega = fiber_bundle.exterior_derivative(Omega)
            assert torch.allclose(
                d_Omega, torch.zeros_like(d_Omega)
            ), "Should satisfy Bianchi"
            
            return Omega
            
        Omega = test_curvature()

    def test_parallel_transport(
        self, fiber_bundle, base_manifold, fiber_dim
    ):
        """Test parallel transport and holonomy."""
        # Initialize path
        def get_test_path():
            """Get closed path in base manifold."""
            return base_manifold.get_closed_path()
            
        path = get_test_path()
        
        # Test parallel transport
        def test_transport():
            """Test parallel transport properties."""
            # Get fiber point
            point = fiber_bundle.get_fiber_point()
            
            # Transport along path
            transported = fiber_bundle.parallel_transport(
                point, path
            )
            
            # Test horizontality
            velocity = fiber_bundle.compute_velocity(transported)
            assert all(
                fiber_bundle.is_horizontal(v)
                for v in velocity
            ), "Transport should be horizontal"
            
            return transported
            
        transported = test_transport()
        
        # Test holonomy
        def test_holonomy():
            """Test holonomy properties."""
            # Compute holonomy
            hol = fiber_bundle.compute_holonomy(path)
            
            # Test group property
            assert fiber_bundle.is_structure_group_element(
                hol
            ), "Should be in structure group"
            
            # Test concatenation
            path2 = base_manifold.get_closed_path()
            hol2 = fiber_bundle.compute_holonomy(path2)
            concat_hol = fiber_bundle.compute_holonomy(
                base_manifold.concatenate_paths(path, path2)
            )
            assert torch.allclose(
                concat_hol,
                fiber_bundle.group_product(hol, hol2)
            ), "Should respect concatenation"
            
            return hol
            
        holonomy = test_holonomy()
        
        # Test holonomy group
        def test_holonomy_group():
            """Test properties of holonomy group."""
            # Generate holonomy group
            hol_group = fiber_bundle.generate_holonomy_group()
            
            # Test Ambrose-Singer
            if fiber_bundle.is_connected():
                # Holonomy algebra should be generated by curvature
                curvature_algebra = fiber_bundle.generate_curvature_algebra()
                assert fiber_bundle.is_subalgebra(
                    fiber_bundle.holonomy_algebra(),
                    curvature_algebra
                ), "Should satisfy Ambrose-Singer"
            
            # Test reduction principle
            if fiber_bundle.admits_reduction():
                reduction = fiber_bundle.compute_holonomy_reduction()
                assert fiber_bundle.is_reduced_bundle(
                    reduction
                ), "Should give valid reduction"
                
            return hol_group
            
        hol_group = test_holonomy_group()
