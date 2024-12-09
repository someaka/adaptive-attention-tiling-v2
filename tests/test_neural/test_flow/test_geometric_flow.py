"""
Unit tests for the geometric flow system.

Tests cover:
1. Ricci tensor computation
2. Flow evolution steps
3. Singularity detection
4. Flow normalization
5. Geometric invariants
"""

import numpy as np
import pytest
import torch

from src.neural.flow.geometric_flow import (
    FlowMetrics,
    GeometricFlow,
    RicciTensor,
    Singularity,
)


class TestGeometricFlow:
    """Test suite for geometric flow computations."""

    @pytest.fixture
    def manifold_dim(self):
        """Return manifold dimension for tests."""
        return 4

    @pytest.fixture
    def batch_size(self):
        """Batch size for testing."""
        return 8

    @pytest.fixture
    def flow_system(self, manifold_dim):
        """Create a test geometric flow system."""
        return GeometricFlow(
            dim=manifold_dim,
            flow_type="ricci",
            timestep=0.01,
            normalize_method="volume",
        )

    def test_ricci_tensor(self, flow_system, manifold_dim, batch_size):
        """Test Ricci tensor computation."""
        # Create test metric and connection
        metric = torch.eye(manifold_dim).expand(batch_size, -1, -1)
        metric = metric + 0.1 * torch.randn(batch_size, manifold_dim, manifold_dim)
        metric = 0.5 * (metric + metric.transpose(-1, -2))  # Ensure symmetry

        connection = flow_system.compute_connection(metric)

        # Compute Ricci tensor
        ricci = flow_system.compute_ricci_tensor(metric, connection)

        # Test tensor properties
        assert isinstance(ricci, RicciTensor), "Should return RicciTensor"
        assert ricci.shape == (
            batch_size,
            manifold_dim,
            manifold_dim,
        ), "Should have correct shape"

        # Test symmetry
        assert torch.allclose(
            ricci, ricci.transpose(-1, -2), rtol=1e-5
        ), "Ricci tensor should be symmetric"

        # Test scaling property
        scaled_metric = 2 * metric
        scaled_ricci = flow_system.compute_ricci_tensor(
            scaled_metric, flow_system.compute_connection(scaled_metric)
        )
        assert torch.allclose(
            scaled_ricci, 0.5 * ricci, rtol=1e-4
        ), "Ricci tensor should scale correctly"

        # Test contracted Bianchi identity
        div_ricci = flow_system.compute_divergence(ricci, metric)
        assert torch.allclose(
            div_ricci, torch.zeros_like(div_ricci), rtol=1e-4
        ), "Should satisfy contracted Bianchi identity"

    def test_flow_step(self, flow_system, manifold_dim):
        """Test geometric flow evolution step."""
        # Create test metric
        metric = torch.eye(manifold_dim) + 0.1 * torch.randn(manifold_dim, manifold_dim)
        metric = 0.5 * (metric + metric.transpose(-1, -2))

        # Compute Ricci tensor
        ricci = flow_system.compute_ricci_tensor(
            metric, flow_system.compute_connection(metric)
        )

        # Evolve flow
        evolved_metric, metrics = flow_system.flow_step(metric, ricci, timestep=0.01)

        # Test metric properties
        assert torch.allclose(
            evolved_metric, evolved_metric.transpose(-1, -2), rtol=1e-5
        ), "Evolution should preserve symmetry"

        # Test positive definiteness
        eigenvals = torch.linalg.eigvalsh(evolved_metric)
        assert torch.all(
            eigenvals > 0
        ), "Evolution should preserve positive definiteness"

        # Test flow metrics
        assert isinstance(metrics, FlowMetrics), "Should return flow metrics"
        assert metrics.scalar_curvature is not None, "Should compute scalar curvature"
        assert metrics.volume_form is not None, "Should compute volume form"

        # Test short-time existence
        short_time = flow_system.evolve_flow(metric, time_span=[0, 0.1], steps=10)
        assert len(short_time) > 0, "Should have short-time existence"

        # Test energy monotonicity
        energies = [flow_system.compute_energy(m) for m in short_time]
        assert all(
            e1 >= e2 for e1, e2 in zip(energies[:-1], energies[1:])
        ), "Energy should be monotonic"

    def test_singularity_detection(self, flow_system, manifold_dim):
        """Test singularity detection in geometric flow."""
        # Create test metric with potential singularity
        metric = torch.eye(manifold_dim)
        metric[0, 0] = 0.1  # Near-singular metric

        # Detect singularities
        singularities = flow_system.detect_singularities(metric, threshold=0.2)

        # Test singularity properties
        assert len(singularities) > 0, "Should detect singularities"
        for sing in singularities:
            assert isinstance(sing, Singularity), "Should return Singularity objects"
            assert sing.type in [
                "Type-I",
                "Type-II",
            ], "Should classify singularity type"

        # Test singularity formation time
        formation_time = flow_system.estimate_singularity_time(metric)
        assert formation_time > 0, "Should estimate positive formation time"

        # Test neck detection
        necks = flow_system.detect_necks(metric)
        assert all(
            n.radius > 0 for n in necks
        ), "Neck regions should have positive radius"

        # Test surgery procedure if applicable
        if hasattr(flow_system, "perform_surgery"):
            surgered_metric = flow_system.perform_surgery(metric, necks[0])
            assert (
                torch.linalg.matrix_rank(surgered_metric) == manifold_dim
            ), "Surgery should preserve rank"

    def test_flow_normalization(self, flow_system, manifold_dim):
        """Test geometric flow normalization."""
        # Create test flow
        metric = torch.eye(manifold_dim) + 0.1 * torch.randn(manifold_dim, manifold_dim)
        metric = 0.5 * (metric + metric.transpose(-1, -2))

        # Test different normalization methods
        for method in ["ricci", "volume", "yamabe"]:
            normalized = flow_system.normalize_flow(metric, normalization=method)

            # Test normalization properties
            if method == "volume":
                volume = flow_system.compute_volume(normalized)
                assert torch.allclose(
                    volume, torch.tensor(1.0), rtol=1e-4
                ), "Volume normalization should preserve unit volume"

            elif method == "yamabe":
                scalar_curv = flow_system.compute_scalar_curvature(normalized)
                assert torch.allclose(
                    scalar_curv.mean(), torch.tensor(0.0), rtol=1e-4
                ), "Yamabe normalization should normalize scalar curvature"

        # Test normalization compatibility
        flow = flow_system.evolve_flow(metric, time_span=[0, 1], steps=10)
        normalized_flow = flow_system.normalize_flow_sequence(flow)
        assert len(normalized_flow) == len(flow), "Should preserve flow length"

    def test_geometric_invariants(self, flow_system, manifold_dim):
        """Test geometric invariant computation along flow."""
        # Create test metric
        metric = torch.eye(manifold_dim) + 0.1 * torch.randn(manifold_dim, manifold_dim)
        metric = 0.5 * (metric + metric.transpose(-1, -2))

        # Evolve flow
        flow = flow_system.evolve_flow(metric, time_span=[0, 1], steps=10)

        # Test scalar invariants
        for metric in flow:
            # Euler characteristic (if applicable)
            if manifold_dim % 2 == 0:
                euler = flow_system.compute_euler_characteristic(metric)
                assert isinstance(
                    euler, torch.Tensor
                ), "Should compute Euler characteristic"

            # Signature (if applicable)
            if manifold_dim % 4 == 0:
                signature = flow_system.compute_signature(metric)
                assert isinstance(signature, torch.Tensor), "Should compute signature"

        # Test Pontryagin classes
        pontryagin = flow_system.compute_pontryagin_classes(metric)
        assert len(pontryagin) > 0, "Should compute Pontryagin classes"

        # Test Chern classes if complex structure exists
        if hasattr(flow_system, "compute_chern_classes"):
            chern = flow_system.compute_chern_classes(metric)
            assert len(chern) > 0, "Should compute Chern classes"

    def test_ricci_flow(self, flow_system, batch_size, manifold_dim):
        """Test Ricci flow properties and evolution."""
        # Initialize metric
        metric = flow_system.initialize_metric(batch_size, manifold_dim)

        # Test Ricci tensor computation
        def test_ricci_tensor() -> torch.Tensor:
            """Test Ricci tensor properties."""
            ricci = flow_system.compute_ricci_tensor(metric)
            assert ricci.shape == metric.shape, "Should preserve shape"
            return ricci

        test_ricci_tensor()

        # Test scalar curvature
        def test_scalar_curvature() -> torch.Tensor:
            """Test scalar curvature computation."""
            scalar = flow_system.compute_scalar_curvature(metric)
            assert scalar.shape[:-2] == metric.shape[:-2], "Should reduce last two dims"
            return scalar

        test_scalar_curvature()

        # Test flow evolution
        def test_flow_evolution(metric: torch.Tensor, time: float = 1.0) -> list:
            """Test Ricci flow evolution."""
            trajectory = flow_system.evolve_metric(metric, final_time=time)
            assert len(trajectory) > 0, "Should produce trajectory"
            return trajectory

        test_flow_evolution(metric)

        # Test normalized flow
        def test_normalized_flow() -> list:
            """Test normalized Ricci flow."""
            norm_trajectory = flow_system.evolve_normalized_metric(metric, steps=100)
            assert len(norm_trajectory) > 0, "Should produce trajectory"
            return norm_trajectory

        test_normalized_flow()

    def test_mean_curvature_flow(self, flow_system, batch_size, manifold_dim):
        """Test mean curvature flow properties."""
        # Initialize surface embedding
        surface = flow_system.initialize_surface(batch_size, manifold_dim)

        # Test mean curvature vector
        def test_mean_curvature() -> torch.Tensor:
            """Test mean curvature computation."""
            mean_curv = flow_system.compute_mean_curvature(surface)
            assert mean_curv.shape[-1] == manifold_dim + 1, "Should be ambient vector"
            return mean_curv

        test_mean_curvature()

        # Test flow evolution
        def test_surface_evolution(surface: torch.Tensor, time: float = 1.0) -> list:
            """Test mean curvature flow evolution."""
            trajectory = flow_system.evolve_surface(surface, final_time=time)
            assert len(trajectory) > 0, "Should produce trajectory"
            return trajectory

        trajectory = test_surface_evolution(surface)

        # Test isoperimetric ratio
        def test_isoperimetric() -> None:
            """Test isoperimetric properties."""
            for surface in trajectory:
                area = flow_system.compute_surface_area(surface)
                volume = flow_system.compute_enclosed_volume(surface)
                ratio = (area**3) / (36 * np.pi * volume**2)
                assert ratio >= 1, "Should satisfy isoperimetric inequality"

        test_isoperimetric()

    def test_singularity_analysis(self, flow_system, batch_size, manifold_dim):
        """Test singularity formation and analysis."""
        # Initialize near singular metric
        metric = flow_system.initialize_near_singular_metric(batch_size, manifold_dim)

        # Test singularity detection
        def test_detect_singularities() -> list:
            """Test singularity detection methods."""
            singular_points = flow_system.detect_singular_points(metric)
            assert len(singular_points) > 0, "Should detect singularities"
            return singular_points

        singular_points = test_detect_singularities()

        # Test blow-up analysis
        def test_blow_up() -> None:
            """Test blow-up behavior near singularities."""
            for point in singular_points:
                # Compute blow-up sequence
                sequence = flow_system.compute_blow_up_sequence(metric, point)
                assert len(sequence) > 0, "Should compute blow-up"
                # Test convergence
                limit = flow_system.analyze_blow_up_limit(sequence)
                assert limit is not None, "Should have well-defined limit"

        test_blow_up()

        # Test surgery
        def test_surgery() -> torch.Tensor:
            """Test geometric surgery near singularities."""
            pre_surgery = flow_system.evolve_metric(metric, steps=100)
            post_surgery = flow_system.perform_surgery(pre_surgery[-1])
            assert post_surgery.shape == pre_surgery[-1].shape, "Should preserve shape"
            return post_surgery

        test_surgery()

        # Test neck pinching
        def test_neck_pinching() -> list:
            """Test neck pinching singularities."""
            neck = flow_system.initialize_neck_pinch(batch_size)
            pinch_trajectory = flow_system.evolve_metric(neck, final_time=1.0)
            assert len(pinch_trajectory) > 0, "Should produce trajectory"
            return pinch_trajectory

        test_neck_pinching()
