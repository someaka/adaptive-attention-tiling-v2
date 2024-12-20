"""Tests for motivic integration system."""

import pytest
import torch
from src.core.patterns.motivic_integration import MotivicIntegrationSystem


@pytest.fixture
def integration_system():
    """Create a motivic integration system instance."""
    return MotivicIntegrationSystem(
        manifold_dim=2,
        hidden_dim=4,
        motive_rank=2,
        num_primes=4,
        monte_carlo_steps=5,
        num_samples=50
    )


@pytest.fixture
def test_pattern():
    """Create a test pattern tensor."""
    return torch.randn(10, 4)  # Batch size 10, hidden dim 4


class TestMotivicIntegrationSystem:
    """Test motivic integration system functionality."""

    def test_initialization(self, integration_system):
        """Test that system initializes correctly."""
        assert integration_system.manifold_dim == 2
        assert integration_system.hidden_dim == 4
        assert integration_system.motive_rank == 2
        assert integration_system.num_primes == 4
        assert integration_system.monte_carlo_steps == 5
        assert integration_system.num_samples == 50

    def test_compute_measure(self, integration_system, test_pattern):
        """Test measure computation."""
        # Test without quantum corrections
        measure, metrics = integration_system.compute_measure(
            test_pattern,
            with_quantum=False
        )
        assert measure.shape == (10, 2)  # Batch size 10, motive rank 2
        assert 'measure_norm' in metrics
        assert 'cohomology_degree' in metrics
        assert 'metric_determinant' in metrics
        assert 'quantum_correction' in metrics
        assert metrics['quantum_correction'] == 1.0
        
        # Test with quantum corrections
        measure, metrics = integration_system.compute_measure(
            test_pattern,
            with_quantum=True
        )
        assert measure.shape == (10, 2)
        assert metrics['quantum_correction'] != 1.0

    def test_compute_integral(self, integration_system, test_pattern):
        """Test integral computation."""
        integral, metrics = integration_system.compute_integral(test_pattern)
        assert integral.shape == (10,)  # Batch size 10
        assert 'domain_volume' in metrics
        assert 'integral_mean' in metrics
        assert 'integral_std' in metrics
        assert 'measure_norm' in metrics
        assert 'cohomology_degree' in metrics
        assert 'metric_determinant' in metrics
        assert 'quantum_correction' in metrics

    def test_evolve_integral(self, integration_system, test_pattern):
        """Test integral evolution."""
        integrals, metrics = integration_system.evolve_integral(
            test_pattern,
            time_steps=3
        )
        assert len(integrals) == 3
        assert all(i.shape == (10,) for i in integrals)  # All batch size 10
        assert 'initial_integral' in metrics
        assert 'final_integral' in metrics
        assert 'integral_change' in metrics
        assert 'max_integral' in metrics
        assert 'min_integral' in metrics
        assert 'mean_measure_norm' in metrics
        assert 'mean_domain_volume' in metrics

    def test_stability_metrics(self, integration_system, test_pattern):
        """Test stability metric computation."""
        metrics = integration_system.compute_stability_metrics(
            test_pattern,
            num_perturbations=3,
            perturbation_scale=0.1
        )
        assert 'mean_integral_change' in metrics
        assert 'max_integral_change' in metrics
        assert 'integral_std' in metrics
        assert 'perturbation_correlation' in metrics
        assert -1 <= metrics['perturbation_correlation'] <= 1

    def test_batch_processing(self, integration_system):
        """Test processing of different batch sizes."""
        # Test single sample
        single = torch.randn(1, 4)
        measure, _ = integration_system.compute_measure(single)
        assert measure.shape == (1, 2)
        
        # Test large batch
        batch = torch.randn(100, 4)
        measure, _ = integration_system.compute_measure(batch)
        assert measure.shape == (100, 2)

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
        
        # Quantum corrections should make a difference
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
            time_steps=5
        )
        
        # Check that early steps match
        assert torch.allclose(
            short_evolution[0],
            long_evolution[0],
            rtol=1e-5
        )
        assert torch.allclose(
            short_evolution[1],
            long_evolution[1],
            rtol=1e-5
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
        
        # Larger perturbations should cause bigger changes
        assert (
            large_metrics['mean_integral_change'] >
            small_metrics['mean_integral_change']
        ) 