"""Tests for pattern wavelength computation."""

import torch
import pytest
import numpy as np
from src.validation.patterns.formation import SpatialValidator

def test_wavelength_simple_pattern():
    """Test wavelength computation with a simple sinusoidal pattern."""
    # Create a simple sinusoidal pattern with known wavelength
    size = 32
    wavelength = 8  # 8 pixels per cycle
    x = torch.linspace(0, 2*np.pi, size)
    y = torch.linspace(0, 2*np.pi, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create pattern with wavelength of 8 pixels
    pattern = torch.sin(2*np.pi*X/wavelength) + torch.sin(2*np.pi*Y/wavelength)
    pattern = pattern.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    validator = SpatialValidator()
    computed_wavelength = validator._compute_wavelength(pattern)
    
    print(f"Pattern shape: {pattern.shape}")
    print(f"Expected wavelength: {wavelength}")
    print(f"Computed wavelength: {computed_wavelength}")
    
    assert torch.abs(computed_wavelength - wavelength) < 1.0

def test_wavelength_complex_pattern():
    """Test wavelength computation with a more complex pattern."""
    # Create a pattern with multiple wavelengths
    size = 32
    wavelength1 = 8
    wavelength2 = 16
    x = torch.linspace(0, 2*np.pi, size)
    y = torch.linspace(0, 2*np.pi, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create pattern with two wavelengths, stronger amplitude on wavelength1
    pattern = 2*torch.sin(2*np.pi*X/wavelength1) + torch.sin(2*np.pi*Y/wavelength2)
    pattern = pattern.unsqueeze(0).unsqueeze(0)
    
    validator = SpatialValidator()
    computed_wavelength = validator._compute_wavelength(pattern)
    
    print(f"Pattern shape: {pattern.shape}")
    print(f"Expected dominant wavelength: {wavelength1}")
    print(f"Computed wavelength: {computed_wavelength}")
    
    # Should detect the dominant wavelength (wavelength1)
    assert torch.abs(computed_wavelength - wavelength1) < 1.0

def test_wavelength_batch():
    """Test wavelength computation with batched patterns."""
    size = 32
    batch_size = 2
    channels = 3
    wavelength = 8
    
    x = torch.linspace(0, 2*np.pi, size)
    y = torch.linspace(0, 2*np.pi, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create batch of patterns
    patterns = []
    for _ in range(batch_size):
        channel_patterns = []
        for _ in range(channels):
            # Add random phase shift to make patterns different
            phase = torch.rand(1) * 2 * np.pi
            pattern = torch.sin(2*np.pi*X/wavelength + phase) + torch.sin(2*np.pi*Y/wavelength + phase)
            channel_patterns.append(pattern)
        patterns.append(torch.stack(channel_patterns))
    pattern_batch = torch.stack(patterns)
    
    validator = SpatialValidator()
    computed_wavelengths = validator._compute_wavelength(pattern_batch)
    
    print(f"Pattern batch shape: {pattern_batch.shape}")
    print(f"Computed wavelengths shape: {computed_wavelengths.shape}")
    print(f"Computed wavelengths: {computed_wavelengths}")
    
    # All patterns should have same wavelength
    assert computed_wavelengths.shape == (batch_size, channels)
    assert torch.all(torch.abs(computed_wavelengths - wavelength) < 1.0)

def test_wavelength_noise():
    """Test wavelength computation with noisy pattern."""
    size = 32
    wavelength = 8
    noise_level = 0.2
    
    x = torch.linspace(0, 2*np.pi, size)
    y = torch.linspace(0, 2*np.pi, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create pattern with noise
    pattern = torch.sin(2*np.pi*X/wavelength) + torch.sin(2*np.pi*Y/wavelength)
    noise = torch.randn_like(pattern) * noise_level
    noisy_pattern = pattern + noise
    noisy_pattern = noisy_pattern.unsqueeze(0).unsqueeze(0)
    
    validator = SpatialValidator()
    computed_wavelength = validator._compute_wavelength(noisy_pattern)
    
    print(f"Pattern shape: {noisy_pattern.shape}")
    print(f"Noise level: {noise_level}")
    print(f"Expected wavelength: {wavelength}")
    print(f"Computed wavelength: {computed_wavelength}")
    
    # Should still detect wavelength within tolerance despite noise
    assert torch.abs(computed_wavelength - wavelength) < 2.0  # Larger tolerance for noise
