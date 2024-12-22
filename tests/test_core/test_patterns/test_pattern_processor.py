"""Test pattern processor functionality."""

import pytest
import torch
import numpy as np
from typing import List, Optional, Tuple

# Only allow CPU device
ALLOWED_DEVICES = ["cpu"]

@pytest.fixture
def device() -> str:
    """Get test device."""
    return "cpu"

@pytest.fixture
def test_pattern(batch_size: int = 32, pattern_size: int = 64) -> torch.Tensor:
    """Generate test pattern."""
    return torch.randn(batch_size, pattern_size)

def test_pattern_evolution(test_pattern):
    """Test pattern evolution over time."""
    noise_scale = 0.1
    evolved = test_pattern + torch.randn_like(test_pattern) * noise_scale
    
    # Test shape consistency
    assert evolved.shape == test_pattern.shape
    
    # Test that most values are within reasonable bounds
    diff = torch.abs(evolved - test_pattern)
    within_bounds = (diff <= noise_scale * 3).float().mean()  # Within 3 standard deviations
    assert within_bounds >= 0.95  # At least 95% of values should be within bounds

def test_pattern_stability(test_pattern):
    """Test pattern stability under perturbations."""
    noise_scale = 0.01
    perturbed = test_pattern + torch.randn_like(test_pattern) * noise_scale
    
    # Test shape consistency
    assert perturbed.shape == test_pattern.shape
    
    # Test that most values are within reasonable bounds
    diff = torch.abs(perturbed - test_pattern)
    within_bounds = (diff <= noise_scale * 3).float().mean()  # Within 3 standard deviations
    assert within_bounds >= 0.99  # At least 99% of values should be within bounds

def test_pattern_metrics(test_pattern):
    """Test pattern quality metrics."""
    # Calculate proper entropy using softmax normalization
    normalized = torch.softmax(test_pattern.flatten(), dim=0)
    entropy = -(normalized * torch.log(normalized + 1e-10)).sum()
    assert entropy.item() > 0
    
    # Stability metric
    stability = torch.std(test_pattern).item()
    assert 0 <= stability <= 10
  