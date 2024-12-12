"""Diagnostic tests for wavelength computation."""

import torch
import numpy as np
from src.validation.patterns.formation import SpatialValidator

def create_test_pattern(size: int, wavelength: float) -> torch.Tensor:
    """Create a test pattern with exact wavelength.
    
    Args:
        size: Size of the pattern (both height and width)
        wavelength: Desired wavelength in pixels
        
    Returns:
        Pattern tensor of shape (1, 1, size, size)
    """
    # Create spatial coordinates in pixels
    x = torch.arange(size, dtype=torch.float32)
    
    # Create pattern with exactly wavelength pixels per cycle
    pattern = torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1)
    
    # Repeat pattern across all rows
    pattern = pattern.repeat(1, 1, size, 1)
    return pattern

def test_frequency_grid_diagnostic():
    """Test frequency grid creation and properties."""
    size = 32
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    
    print("\nFrequency Grid Analysis:")
    print(f"Grid shape: {freqs.shape}")
    print(f"Min frequency: {freqs.min()}")
    print(f"Max frequency: {freqs.max()}")
    print(f"Positive frequencies: {freqs[1:size//2]}")  # Skip DC (0)
    print(f"Negative frequencies: {freqs[size//2:]}")
    
    # Check Nyquist frequency
    nyquist = freqs[size//2]
    print(f"Nyquist frequency: {nyquist}")
    assert torch.isclose(nyquist, torch.tensor(-0.5)), "Incorrect Nyquist frequency"

def test_wavelength_computation_diagnostic():
    """Test wavelength computation with detailed output."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    validator = SpatialValidator()
    
    # Step 1: Create frequency grids
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    
    print("\nFrequency Grid Info:")
    print(f"X frequencies range: [{freqs_x.min()}, {freqs_x.max()}]")
    print(f"Y frequencies range: [{freqs_y.min()}, {freqs_y.max()}]")
    print(f"First few X frequencies: {freqs_x[0, :8]}")  # Print first 8 frequencies
    
    # Step 2: Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    print("\nPower Spectrum Info:")
    print(f"Power shape: {power.shape}")
    print(f"Max power value: {power.max()}")
    
    # Get the indices of top 5 power values
    power_flat = power.reshape(-1)
    top_powers, top_indices = torch.topk(power_flat, 5)
    print("\nTop 5 Power Values:")
    for i, (p, idx) in enumerate(zip(top_powers, top_indices)):
        y, x = idx // size, idx % size
        freq_x = freqs_x[y, x].item()
        freq_y = freqs_y[y, x].item()
        print(f"{i+1}. Power: {p:.2f}, Position: ({y}, {x}), Frequency: ({freq_x:.4f}, {freq_y:.4f})")
    
    # Step 3: Create frequency masks
    nyquist = 0.5
    mask_x = (freqs_x.abs() > 0) & (freqs_x.abs() <= nyquist)
    mask_y = (freqs_y.abs() > 0) & (freqs_y.abs() <= nyquist)
    mask = mask_x | mask_y
    
    print("\nMask Info:")
    print(f"Valid X frequencies: {mask_x.sum()}")
    print(f"Valid Y frequencies: {mask_y.sum()}")
    print(f"Combined valid frequencies: {mask.sum()}")
    
    # Step 4: Get valid frequencies and power
    power_valid = power.reshape(-1)[mask.reshape(-1)]
    freqs_x_valid = freqs_x[mask]
    freqs_y_valid = freqs_y[mask]
    
    # Find top 5 valid frequencies
    top_valid_powers, top_valid_indices = torch.topk(power_valid, 5)
    print("\nTop 5 Valid Power Values:")
    for i, (p, idx) in enumerate(zip(top_valid_powers, top_valid_indices)):
        freq_x = freqs_x_valid[idx].item()
        freq_y = freqs_y_valid[idx].item()
        print(f"{i+1}. Power: {p:.2f}, Frequency: ({freq_x:.4f}, {freq_y:.4f})")
    
    # Step 5: Find peak frequency
    peak_idx = torch.argmax(power_valid)
    peak_freq_x = freqs_x_valid[peak_idx]
    peak_freq_y = freqs_y_valid[peak_idx]
    
    print("\nPeak Frequency Info:")
    print(f"Peak X frequency: {peak_freq_x}")
    print(f"Peak Y frequency: {peak_freq_y}")
    
    # Step 6: Compute wavelength
    peak_freq = max(peak_freq_x.abs(), peak_freq_y.abs())
    computed_wavelength = 1.0 / peak_freq
    
    print("\nWavelength Info:")
    print(f"Peak frequency: {peak_freq}")
    print(f"Computed wavelength: {computed_wavelength}")
    print(f"Expected wavelength: {wavelength}")
    
    # Final validation
    assert torch.isclose(torch.tensor(computed_wavelength), torch.tensor(wavelength), atol=1.0), \
        f"Wavelength mismatch: computed={computed_wavelength}, expected={wavelength}"

def test_batch_processing_diagnostic():
    """Test batch processing with detailed output."""
    size = 32
    wavelengths = [8.0, 16.0]
    batch_size = len(wavelengths)
    
    # Create batch of patterns
    patterns = []
    for wavelength in wavelengths:
        pattern = create_test_pattern(size, wavelength)
        patterns.append(pattern)
    pattern_batch = torch.cat(patterns, dim=0)
    
    print("\nBatch Processing Info:")
    print(f"Batch shape: {pattern_batch.shape}")
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern_batch)
    
    print(f"Input wavelengths: {wavelengths}")
    print(f"Computed wavelengths shape: {computed.shape}")
    print(f"Computed wavelengths: {computed}")
    
    # Check each batch element
    for i, (expected, computed_tensor) in enumerate(zip(wavelengths, computed)):
        assert torch.isclose(computed_tensor[0], torch.tensor(expected), atol=1.0), \
            f"Batch {i} wavelength mismatch: computed={computed_tensor[0]}, expected={expected}"
