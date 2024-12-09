"""Test suite for information density analyzer."""

from __future__ import annotations

import torch

from src.core.metrics.density_analyzer import InformationDensityAnalyzer


def test_gradient_density() -> None:
    """Test gradient-based density estimation."""
    analyzer = InformationDensityAnalyzer(window_size=3, smoothing_factor=0.5)
    sequence = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.long)

    density = analyzer.compute_gradient_density(sequence)
    assert density.shape == sequence.shape
    assert torch.all(density >= 0)
    assert torch.all(density <= 1)

    # Check that constant regions have zero density
    assert density[0, 1].item() == 0  # Middle of first constant region
    assert density[0, 4].item() == 0  # Middle of second constant region
    assert density[0, 7].item() == 0  # Middle of third constant region

    # Check that transitions have maximum density
    assert density[0, 2].item() == 1  # First transition
    assert density[0, 5].item() == 1  # Second transition


def test_entropy_density() -> None:
    """Test entropy-based density estimation."""
    analyzer = InformationDensityAnalyzer(window_size=3, smoothing_factor=0.5)
    sequence = torch.tensor([[1, 1, 1, 2, 3, 2, 3, 2, 3]], dtype=torch.long)

    density = analyzer.compute_entropy_density(sequence)
    assert density.shape == sequence.shape
    assert torch.all(density >= 0)
    assert torch.all(density <= 1)

    # Check that constant regions have low entropy
    assert torch.mean(density[0, 0:3]).item() < 0.5  # First constant region

    # Check that alternating regions have high entropy
    assert torch.mean(density[0, 4:9]).item() > 0.4  # Alternating region


def test_smoothing() -> None:
    """Test density smoothing."""
    analyzer = InformationDensityAnalyzer(window_size=3, smoothing_factor=0.5)
    density = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float)

    smoothed = analyzer.smooth_density(density)
    assert smoothed.shape == density.shape
    assert torch.all(smoothed >= 0)
    assert torch.all(smoothed <= 1)

    # Check that smoothing reduces variation
    assert torch.std(smoothed) < torch.std(density)


def test_multi_scale_analysis() -> None:
    """Test multi-scale density analysis."""
    analyzer = InformationDensityAnalyzer(window_size=3, smoothing_factor=0.5)
    sequence = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.long)

    densities = analyzer.analyze_multi_scale(sequence, num_scales=3)
    assert densities.shape == (1, 3, sequence.shape[1])
    assert torch.all(densities >= 0)
    assert torch.all(densities <= 1)

    # Check that larger scales have smoother density
    for i in range(1, 3):
        assert torch.std(densities[:, i, :]) <= torch.std(densities[:, i - 1, :])


def test_edge_cases() -> None:
    """Test handling of edge cases."""
    analyzer = InformationDensityAnalyzer(window_size=3, smoothing_factor=0.5)

    # Single token
    sequence = torch.tensor([[1]], dtype=torch.long)
    density = analyzer.compute_gradient_density(sequence)
    assert density.shape == sequence.shape
    assert torch.all(density >= 0)
    assert torch.all(density <= 1)

    # All same token
    sequence = torch.ones(1, 10, dtype=torch.long)
    density = analyzer.compute_gradient_density(sequence)
    assert density.shape == sequence.shape
    assert torch.all(density >= 0)
    assert torch.all(density <= 1)
    assert torch.mean(density).item() == 0  # Should have zero density

    # Random sequence
    sequence = torch.randint(0, 1000, (1, 20))
    density = analyzer.compute_gradient_density(sequence)
    assert density.shape == sequence.shape
    assert torch.all(density >= 0)
    assert torch.all(density <= 1)
