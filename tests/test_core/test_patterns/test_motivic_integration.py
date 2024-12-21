"""Tests for motivic integration system."""

import pytest
import torch
from src.core.patterns.motivic_integration import MotivicIntegrationSystem


@pytest.fixture
def integration_system():
    """Create a motivic integration system instance."""
    return MotivicIntegrationSystem(
        manifold_dim=4,
        hidden_dim=4,
        motive_rank=2,
        num_primes=3
    )


@pytest.fixture
def test_pattern():
    """Create a test pattern tensor."""
    return torch.randn(10, 4)  # Batch size 10, hidden dim 4


class TestMotivicIntegrationSystem:
    """Test motivic integration system functionality."""

    def test_initialization(self, integration_system):
        """Test that system initializes correctly."""
        assert integration_system.manifold_dim == 4
        assert integration_system.hidden_dim == 4
        assert integration_system.motive_rank == 2
        assert integration_system.num_primes == 3

    def test_compute_measure(self, integration_system, test_pattern):
        """Test measure computation."""
        measure, metrics = integration_system.compute_measure(test_pattern)
        min_dim = max(4, 2 * integration_system.motive_rank)
        assert measure.shape == (test_pattern.shape[0], min_dim)
        assert isinstance(metrics, dict)

    def test_compute_integral(self, integration_system, test_pattern):
        """Test integral computation."""
        integral, metrics = integration_system.compute_integral(test_pattern)
        assert integral.shape == (test_pattern.shape[0],)
        assert isinstance(metrics, dict)
        assert "measure_norm" in metrics

    def test_evolve_integral(self, integration_system, test_pattern):
        """Test integral evolution."""
        integrals, metrics = integration_system.evolve_integral(
            test_pattern,
            time_steps=3
        )
        assert integrals.shape == (3, test_pattern.shape[0])
        assert isinstance(metrics, dict)
        assert len(metrics) == 3

    def test_stability_metrics(self, integration_system, test_pattern):
        """Test stability metric computation."""
        metrics = integration_system.compute_stability_metrics(
            test_pattern,
            num_perturbations=3,
            perturbation_scale=0.1
        )
        assert isinstance(metrics, dict)
        assert "mean_variation" in metrics
        assert "max_variation" in metrics

    def test_batch_processing(self, integration_system):
        """Test processing of different batch sizes."""
        # Test single sample
        single = torch.randn(1, 4)
        measure, _ = integration_system.compute_measure(single)
        min_dim = max(4, 2 * integration_system.motive_rank)
        assert measure.shape == (1, min_dim)

        # Test batch
        batch = torch.randn(5, 4)
        measure, _ = integration_system.compute_measure(batch)
        assert measure.shape == (5, min_dim)

    def test_quantum_effects(self, integration_system, test_pattern):
        """Test quantum correction effects."""
        # Compare with and without quantum corrections
        classical_integral, _ = integration_system.compute_integral(
            test_pattern,
            with_quantum=False
        )
        quantum_integral, _ = integration_system.compute_integral(
            test_pattern,
            with_quantum=True
        )
        
        # Quantum effects should make some difference
        assert not torch.allclose(classical_integral, quantum_integral)

    def test_evolution_stability(self, integration_system, test_pattern):
        """Test stability of integral evolution."""
        # Evolve with different time steps
        short_evolution, _ = integration_system.evolve_integral(
            test_pattern,
            time_steps=2
        )
        long_evolution, _ = integration_system.evolve_integral(
            test_pattern,
            time_steps=4
        )
        
        # Evolution should be stable
        assert torch.allclose(
            short_evolution[-1],
            long_evolution[1],
            rtol=1e-2
        )

    def test_perturbation_sensitivity(self, integration_system, test_pattern):
        """Test sensitivity to perturbations."""
        # Compare different perturbation scales
        small_metrics = integration_system.compute_stability_metrics(
            test_pattern,
            perturbation_scale=0.01
        )
        large_metrics = integration_system.compute_stability_metrics(
            test_pattern,
            perturbation_scale=0.1
        )
        
        # Larger perturbations should cause more variation
        assert small_metrics["mean_variation"] < large_metrics["mean_variation"]