"""Unit tests for wavelength computation components."""

import torch
import pytest
import logging
import numpy as np
import math
from src.validation.patterns.formation import SpatialValidator

logger = logging.getLogger(__name__)

def create_test_pattern(size: int, wavelength: float) -> torch.Tensor:
    """Create a test pattern with exact wavelength.
    
    Args:
        size: Size of the pattern (both height and width)
        wavelength: Desired wavelength in pixels
        
    Returns:
        Pattern tensor of shape (1, 1, size, size)
    """
    # Create spatial coordinates normalized to [0, 1]
    x = torch.linspace(0, 1, size, dtype=torch.float32)
    y = torch.linspace(0, 1, size, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create pattern with exact spatial frequency
    # wavelength = size/cycles, so cycles = size/wavelength
    cycles = size / wavelength
    pattern = torch.sin(2 * np.pi * cycles * X) + torch.sin(2 * np.pi * cycles * Y)
    
    # Normalize to [-1, 1]
    pattern = pattern / pattern.abs().max()
    
    # Add batch and channel dimensions
    pattern = pattern.unsqueeze(0).unsqueeze(0)
    
    return pattern

def test_pattern_creation():
    """Test that pattern creation works as expected."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Check shape
    assert pattern.shape == (1, 1, size, size)
    
    # Check periodicity - count zero crossings
    x_slice = pattern[0, 0, 0, :]
    zero_crossings = torch.where(x_slice[:-1] * x_slice[1:] < 0)[0]
    num_cycles = len(zero_crossings) / 2  # Two crossings per cycle
    measured_wavelength = size / num_cycles
    
    print(f"Expected wavelength: {wavelength}")
    print(f"Measured wavelength: {measured_wavelength}")
    assert abs(measured_wavelength - wavelength) < 1.0

def test_fft_frequencies():
    """Test FFT frequency computation."""
    size = 32
    wavelength = 8.0  # 8 pixels per cycle
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Get frequency grid
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_2d = torch.sqrt(freqs_x[None, :].pow(2) + freqs_y[:, None].pow(2))
    
    # Find peak frequency
    peak_idx = torch.argmax(power.reshape(-1))
    peak_y, peak_x = peak_idx // size, peak_idx % size
    peak_freq = freqs_2d[peak_y, peak_x]
    
    # Expected frequency is 1/wavelength
    expected_freq = 1.0 / wavelength
    
    print(f"Expected frequency: {expected_freq}")
    print(f"Peak frequency: {peak_freq}")
    
    assert torch.allclose(peak_freq, torch.tensor(expected_freq), atol=0.01)

def test_wavelength_computation():
    """Test complete wavelength computation pipeline."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    print(f"Input wavelength: {wavelength}")
    print(f"Computed wavelength: {computed}")
    
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=1.0)

def test_wavelength_batch_consistency():
    """Test wavelength computation is consistent across batch dimension."""
    size = 32
    wavelength = 8.0
    batch_size = 3
    
    # Create batch of identical patterns
    pattern = create_test_pattern(size, wavelength)
    pattern_batch = pattern.repeat(batch_size, 1, 1, 1)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern_batch)
    
    # All wavelengths should be identical
    assert torch.allclose(computed, computed[0].expand_as(computed), atol=1e-6)
    assert torch.allclose(computed[0], torch.tensor([wavelength], dtype=torch.float32), atol=1.0)

def test_frequency_grid_creation():
    """Test creation of frequency grids."""
    validator = SpatialValidator()
    size = 8
    
    # Create pattern
    pattern = torch.zeros((1, 1, size, size))
    
    # Get frequency grids
    freqs_x = torch.fft.fftfreq(pattern.shape[-1])
    freqs_y = torch.fft.fftfreq(pattern.shape[-2])
    
    # Test frequency grid shapes
    assert freqs_x.shape == (size,)
    assert freqs_y.shape == (size,)
    
    # Test frequency values for size=8
    expected_freqs = torch.tensor([0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125])
    assert torch.allclose(freqs_x, expected_freqs, atol=1e-6)
    assert torch.allclose(freqs_y, expected_freqs, atol=1e-6)

def test_2d_frequency_grid():
    """Test creation of 2D frequency grid."""
    validator = SpatialValidator()
    size = 4
    
    # Create frequency grids
    freqs_x = torch.fft.fftfreq(size)
    freqs_y = torch.fft.fftfreq(size)
    
    # Create 2D grid
    freqs_2d = torch.sqrt(freqs_x[None, :].pow(2) + freqs_y[:, None].pow(2))
    
    # Test grid shape
    assert freqs_2d.shape == (size, size)
    
    # Test specific values
    expected_grid = torch.tensor([
        [0.0, 0.25, 0.5, 0.25],
        [0.25, 0.3536, 0.5590, 0.3536],
        [0.5, 0.5590, 0.7071, 0.5590],
        [0.25, 0.3536, 0.5590, 0.3536]
    ])
    assert torch.allclose(freqs_2d, expected_grid, atol=1e-4)

def test_power_spectrum():
    """Test computation of power spectrum."""
    validator = SpatialValidator()
    size = 8
    
    # Create simple sinusoidal pattern
    x = torch.linspace(0, 2*np.pi, size)
    pattern = torch.sin(x).view(1, 1, 1, -1).repeat(1, 1, size, 1)
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Test power shape
    assert power.shape == pattern.shape
    
    # Test power is real and non-negative
    assert torch.all(power >= 0)
    assert not torch.any(torch.isnan(power))

def test_dominant_frequency_single():
    """Test finding dominant frequency in a simple pattern."""
    size = 8
    wavelength = 4  # 4 pixels per cycle
    
    # Create test pattern with known wavelength
    x = torch.linspace(0, 2*np.pi, size)
    pattern = torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1).repeat(1, 1, size, 1)
    
    validator = SpatialValidator()
    
    # Get frequency grids
    freqs_x = torch.fft.fftfreq(size)
    freqs_y = torch.fft.fftfreq(size)
    freqs_2d = torch.sqrt(freqs_x[None, :].pow(2) + freqs_y[:, None].pow(2))
    
    # Compute power spectrum
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Get positive frequencies
    mask = (freqs_2d > 0)
    positive_freqs = freqs_2d[mask]
    positive_power = power.reshape(*power.shape[:-2], -1)[..., mask.reshape(-1)]
    
    # Find dominant frequency
    dominant_idx = torch.argmax(positive_power, dim=-1, keepdim=True)
    dominant_freq = torch.gather(positive_freqs.expand(*power.shape[:-2], -1), -1, dominant_idx)
    
    # Expected frequency for wavelength=4 is 0.25
    expected_freq = 0.25
    assert torch.allclose(dominant_freq, torch.tensor([[expected_freq]]), atol=1e-2)

def test_wavelength_conversion():
    """Test conversion from frequency to wavelength."""
    size = 32
    test_freqs = torch.tensor([[0.125, 0.25, 0.5]])  # Known frequencies
    
    # Convert to wavelengths
    wavelengths = 1.0 / (test_freqs * size) * size
    
    # Expected wavelengths in pixels
    expected = torch.tensor([[8., 4., 2.]])  # 1/0.125=8, 1/0.25=4, 1/0.5=2 cycles
    assert torch.allclose(wavelengths, expected)

def test_end_to_end_known_pattern():
    """Test complete wavelength computation with known pattern."""
    size = 32
    wavelength = 8.0  # 8 pixels per cycle
    
    # Create test pattern
    x = torch.linspace(0, 2*np.pi, size)
    pattern = torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1).repeat(1, 1, size, 1)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    # Print debug info
    print(f"Pattern shape: {pattern.shape}")
    print(f"Input wavelength: {wavelength}")
    print(f"Computed wavelength: {computed}")
    
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=1.0)

def test_wavelength_batch_consistency():
    """Test wavelength computation is consistent across batch dimension."""
    size = 32
    wavelength = 8.0
    batch_size = 3
    
    # Create batch of identical patterns
    x = torch.linspace(0, 2*np.pi, size)
    single_pattern = torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1).repeat(1, 1, size, 1)
    pattern_batch = single_pattern.repeat(batch_size, 1, 1, 1)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern_batch)
    
    # All wavelengths should be identical
    assert torch.allclose(computed, computed[0].expand_as(computed), atol=1e-6)
    assert torch.allclose(computed[0], torch.tensor([wavelength], dtype=torch.float32), atol=1.0)

def test_wavelength_debug():
    """Debug test to examine wavelength computation steps."""
    size = 32
    wavelength = 8.0  # 8 pixels per cycle
    pattern = create_test_pattern(size, wavelength)
    
    # Print pattern info
    print("\nPattern Info:")
    print(f"Size: {size}")
    print(f"Target wavelength: {wavelength}")
    x_slice = pattern[0, 0, 0, :]
    print(f"First row values: {x_slice[:16]}")  # Print first half of first row
    
    # Compute FFT and power spectrum
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Create frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_2d = torch.sqrt(freqs_x[None, :].pow(2) + freqs_y[:, None].pow(2))
    
    print("\nFrequency Info:")
    print(f"X frequencies: {freqs_x[:8]}")  # Print first few frequencies
    print(f"Y frequencies: {freqs_y[:8]}")
    
    # Find peak in power spectrum
    power_2d = power[0, 0]  # Remove batch and channel dims
    peak_idx = torch.argmax(power_2d)
    peak_y, peak_x = peak_idx // size, peak_idx % size
    peak_freq = freqs_2d[peak_y, peak_x]
    
    print("\nPower Spectrum Info:")
    print(f"Peak power location: ({peak_y}, {peak_x})")
    print(f"Peak frequency value: {peak_freq}")
    print(f"Expected frequency: {1.0/wavelength}")
    
    # Convert to wavelength
    computed_wavelength = 1.0 / peak_freq if peak_freq > 0 else float('inf')
    print(f"\nComputed wavelength: {computed_wavelength}")
    
    # Verify result
    assert abs(computed_wavelength - wavelength) < 1.0

def test_fft_power_debug():
    """Debug test to examine FFT power spectrum in detail."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern[0, 0])  # Remove batch and channel dims
    power = torch.abs(fft).pow(2)
    
    # Get frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Print frequency information
    print("\nFrequency Grid Info:")
    print(f"X frequencies: {freqs_x}")
    print(f"Expected frequency: {1.0/wavelength}")
    
    # Find peaks in power spectrum
    power_1d = power[0, :]  # Look at first row only
    peaks = []
    for i in range(1, size-1):
        if power_1d[i] > power_1d[i-1] and power_1d[i] > power_1d[i+1]:
            peaks.append((i, freqs_x[i], power_1d[i].item()))
    
    print("\nPower Spectrum Peaks:")
    for idx, freq, pwr in peaks:
        print(f"Index: {idx}, Frequency: {freq:.4f}, Power: {pwr:.4f}")
        if abs(abs(freq) - 1.0/wavelength) < 0.01:
            print("  ^ This is our target frequency!")
    
    # Find global maximum in power spectrum
    max_idx = torch.argmax(power)
    max_y, max_x = max_idx // size, max_idx % size
    max_freq_x = freqs_x[max_x]
    max_freq_y = freqs_y[max_y]
    max_freq = torch.sqrt(max_freq_x**2 + max_freq_y**2)
    
    print("\nGlobal Maximum:")
    print(f"Location: ({max_y}, {max_x})")
    print(f"Frequency components: x={max_freq_x:.4f}, y={max_freq_y:.4f}")
    print(f"Combined frequency: {max_freq:.4f}")
    print(f"Implied wavelength: {1.0/max_freq:.4f}")

def test_wavelength_computation_debug():
    """Debug test to examine wavelength computation steps."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Get the validator's computation
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    # Print intermediate values
    print("\nPattern Info:")
    print(f"Shape: {pattern.shape}")
    print(f"Target wavelength: {wavelength}")
    
    # Compute FFT manually
    fft = torch.fft.fft2(pattern[0, 0])
    power = torch.abs(fft).pow(2)
    
    # Get frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_2d = torch.sqrt(freqs_x[None, :].pow(2) + freqs_y[:, None].pow(2))
    
    # Find peak in power spectrum
    peak_idx = torch.argmax(power)
    peak_y, peak_x = peak_idx // size, peak_idx % size
    peak_freq = freqs_2d[peak_y, peak_x]
    
    print("\nFrequency Analysis:")
    print(f"Peak location: ({peak_y}, {peak_x})")
    print(f"Peak frequency value: {peak_freq}")
    print(f"Expected frequency: {1.0/wavelength}")
    
    # Convert to wavelength
    manual_wavelength = 1.0 / peak_freq if peak_freq > 0 else float('inf')
    print(f"\nWavelength Computation:")
    print(f"Manual wavelength: {manual_wavelength}")
    print(f"Validator wavelength: {computed.item()}")
    
    # Verify result
    assert abs(manual_wavelength - wavelength) < 1.0

def test_pattern_frequency_mapping():
    """Test the relationship between input pattern frequency and FFT frequencies."""
    size = 32
    test_wavelengths = [4.0, 8.0, 16.0]  # Test multiple wavelengths
    
    print("\nPattern Frequency Mapping Test")
    print("------------------------------")
    
    for wavelength in test_wavelengths:
        pattern = create_test_pattern(size, wavelength)
        
        # Analyze input pattern
        x_slice = pattern[0, 0, 0, :]
        
        # Find peaks instead of zero crossings for more accurate wavelength measurement
        peaks = torch.where((x_slice[1:-1] > x_slice[:-2]) & (x_slice[1:-1] > x_slice[2:]))[0] + 1
        if len(peaks) > 1:
            avg_peak_distance = torch.mean(torch.diff(peaks.float()))
            measured_wavelength = avg_peak_distance.item()
        else:
            measured_wavelength = size  # Handle case with only one peak
        
        # Compute FFT
        fft = torch.fft.fft2(pattern)
        power = torch.abs(fft).pow(2)
        
        # Get frequency grid
        freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
        freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
        freqs_2d = torch.sqrt(freqs_x[None, :].pow(2) + freqs_y[:, None].pow(2))
        
        # Find peak frequency
        peak_idx = torch.argmax(power.reshape(-1))
        peak_y, peak_x = peak_idx // size, peak_idx % size
        peak_freq = freqs_2d[peak_y, peak_x]
        
        # Convert peak frequency to wavelength
        computed_wavelength = 1.0 / peak_freq
        
        print(f"\nWavelength: {wavelength}")
        if len(peaks) > 1:
            print(f"Peak distances: {torch.diff(peaks.float()).tolist()}")
        print(f"Measured wavelength from pattern: {measured_wavelength}")
        print(f"Peak frequency index: ({peak_y}, {peak_x})")
        print(f"Peak frequency value: {peak_freq}")
        print(f"Computed wavelength from FFT: {computed_wavelength}")
        print(f"Expected frequency: {1.0/wavelength}")
        
        # Verify the relationship
        assert abs(measured_wavelength - wavelength) < 1.0, f"Pattern wavelength mismatch: {measured_wavelength} vs {wavelength}"
        assert abs(computed_wavelength - wavelength) < 1.0, f"FFT wavelength mismatch: {computed_wavelength} vs {wavelength}"

def test_frequency_grid_detail():
    """Test frequency grid creation in detail."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Create frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Create 2D frequency grids for x and y separately
    freqs_x_2d = freqs_x[None, :].expand(size, size)
    freqs_y_2d = freqs_y[:, None].expand(size, size)
    
    # Print frequency ranges
    print(f"X frequency grid shape: {freqs_x_2d.shape}")
    print(f"Y frequency grid shape: {freqs_y_2d.shape}")
    print(f"X frequency range: [{freqs_x_2d.min()}, {freqs_x_2d.max()}]")
    print(f"Y frequency range: [{freqs_y_2d.min()}, {freqs_y_2d.max()}]")
    print(f"Expected frequency: {1.0/wavelength}")
    
    # Check frequency grid properties
    assert freqs_x_2d.shape == (size, size)
    assert freqs_y_2d.shape == (size, size)
    assert torch.allclose(freqs_x_2d.min(), torch.tensor(-0.5))
    assert torch.allclose(freqs_x_2d.max(), torch.tensor(0.5))
    assert torch.allclose(freqs_y_2d.min(), torch.tensor(-0.5))
    assert torch.allclose(freqs_y_2d.max(), torch.tensor(0.5))

def test_power_spectrum_detail():
    """Test power spectrum computation in detail."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Create frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x_2d = freqs_x[None, :].expand(size, size)
    freqs_y_2d = freqs_y[:, None].expand(size, size)
    
    # Create masks for valid frequency ranges
    mask_x = (torch.abs(freqs_x_2d) > 1.0/size) & (torch.abs(freqs_x_2d) <= 0.5)
    mask_y = (torch.abs(freqs_y_2d) > 1.0/size) & (torch.abs(freqs_y_2d) <= 0.5)
    mask = mask_x & mask_y
    
    # Get frequencies and power in valid range
    freqs_x_valid = freqs_x_2d[mask]
    freqs_y_valid = freqs_y_2d[mask]
    power_valid = power.reshape(*power.shape[:-2], -1)[..., mask.reshape(-1)]
    
    # Find peak frequency
    peak_idx = torch.argmax(power_valid, dim=-1)
    peak_freq_x = freqs_x_valid[peak_idx]
    peak_freq_y = freqs_y_valid[peak_idx]
    peak_freq = torch.maximum(torch.abs(peak_freq_x), torch.abs(peak_freq_y))
    
    print(f"Power spectrum shape: {power.shape}")
    print(f"Number of valid frequencies: {freqs_x_valid.shape}")
    print(f"Peak frequency x: {peak_freq_x}")
    print(f"Peak frequency y: {peak_freq_y}")
    print(f"Peak frequency: {peak_freq}")
    print(f"Expected frequency: {1.0/wavelength}")
    print(f"Peak wavelength: {1.0/peak_freq}")
    
    # Check power spectrum properties
    assert power.shape[-2:] == (size, size)
    assert freqs_x_valid.numel() > 0
    assert torch.allclose(1.0/peak_freq, torch.tensor(wavelength), atol=1.0)

def test_frequency_mask_detail():
    """Test frequency masking in detail."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Create frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x_2d = freqs_x[None, :].expand(size, size)
    freqs_y_2d = freqs_y[:, None].expand(size, size)
    
    # Create masks with different thresholds
    mask_x = (torch.abs(freqs_x_2d) > 1.0/size) & (torch.abs(freqs_x_2d) <= 0.5)
    mask_y = (torch.abs(freqs_y_2d) > 1.0/size) & (torch.abs(freqs_y_2d) <= 0.5)
    mask = mask_x & mask_y
    
    print(f"Total grid points: {freqs_x_2d.numel()}")
    print(f"Points in x frequency mask: {mask_x.sum()}")
    print(f"Points in y frequency mask: {mask_y.sum()}")
    print(f"Points in combined mask: {mask.sum()}")
    
    # Expected frequency should be within masked range
    target_freq = 1.0/wavelength
    assert target_freq > 1.0/size, "Target frequency below low cutoff"
    assert target_freq <= 0.5, "Target frequency above Nyquist"
    
    # Check if target frequency is in masked range
    freq_x_valid = freqs_x_2d[mask]
    freq_y_valid = freqs_y_2d[mask]
    min_freq = torch.minimum(freq_x_valid.min(), freq_y_valid.min())
    max_freq = torch.maximum(freq_x_valid.max(), freq_y_valid.max())
    print(f"Valid frequency range: [{min_freq}, {max_freq}]")
    print(f"Target frequency: {target_freq}")
    assert target_freq >= min_freq and target_freq <= max_freq

def test_fft_frequency_grid_validation():
    """Test the creation and properties of FFT frequency grid."""
    size = 32
    pattern = create_test_pattern(size=size, wavelength=8.0)
    
    # Create frequency grids
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Basic properties
    assert len(freqs_x) == size, f"Expected {size} frequencies, got {len(freqs_x)}"
    assert freqs_x[0] == 0.0, f"First frequency should be 0, got {freqs_x[0]}"
    assert freqs_x[size//2] == -0.5, f"Middle frequency should be -0.5, got {freqs_x[size//2]}"
    
    # Frequency spacing
    df = freqs_x[1] - freqs_x[0]
    expected_df = 1.0/size
    assert abs(df - expected_df) < 1e-6, f"Expected frequency spacing {expected_df}, got {df}"
    
    # Maximum positive frequency
    max_pos_freq = freqs_x[1:size//2].max()
    assert abs(max_pos_freq - 0.5 + df) < 1e-6, f"Expected max positive frequency 0.5-df, got {max_pos_freq}"
    
    print(f"Frequency grid properties:")
    print(f"Size: {size}")
    print(f"Spacing: {df}")
    print(f"Max positive freq: {max_pos_freq}")
    print(f"Nyquist freq: {0.5}")

def test_power_spectrum_validation():
    """Test power spectrum computation and peak detection."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size=size, wavelength=wavelength)
    
    # Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Expected frequency
    expected_freq = 1.0 / wavelength
    
    # Find peak in power spectrum
    peak_idx = torch.argmax(power.reshape(-1))
    peak_y, peak_x = peak_idx // size, peak_idx % size
    
    # Convert indices to frequencies
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    peak_freq_x = freqs_x[peak_x]
    peak_freq_y = freqs_y[peak_y]
    
    print(f"Power spectrum validation:")
    print(f"Pattern wavelength: {wavelength}")
    print(f"Expected frequency: {expected_freq}")
    print(f"Peak x frequency: {peak_freq_x}")
    print(f"Peak y frequency: {peak_freq_y}")
    print(f"Power spectrum shape: {power.shape}")
    
    # Check if peak frequency matches expected
    peak_freq = max(abs(peak_freq_x), abs(peak_freq_y))
    assert abs(peak_freq - expected_freq) < 0.01, f"Expected frequency {expected_freq}, got {peak_freq}"

def test_wavelength_conversion_validation():
    """Test conversion from frequency to wavelength."""
    size = 32
    input_wavelength = 8.0
    
    # Create test pattern
    pattern = create_test_pattern(size=size, wavelength=input_wavelength)
    
    # Expected frequency
    expected_freq = 1.0 / input_wavelength
    
    # Compute FFT and find peak
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    peak_idx = torch.argmax(power.reshape(-1))
    peak_y, peak_x = peak_idx // size, peak_idx % size
    
    # Get frequencies
    freqs_x = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_y = torch.fft.fftfreq(size, dtype=torch.float32)
    peak_freq_x = freqs_x[peak_x]
    peak_freq_y = freqs_y[peak_y]
    
    # Convert back to wavelength
    peak_freq = max(abs(peak_freq_x), abs(peak_freq_y))
    computed_wavelength = 1.0 / peak_freq if peak_freq != 0 else float('inf')
    
    print(f"Wavelength conversion validation:")
    print(f"Input wavelength: {input_wavelength}")
    print(f"Peak frequency: {peak_freq}")
    print(f"Computed wavelength: {computed_wavelength}")
    
    assert abs(computed_wavelength - input_wavelength) < 1.0, \
        f"Expected wavelength {input_wavelength}, got {computed_wavelength}"

def test_pattern_creation_exact_wavelength():
    """Test that created patterns have exactly the specified wavelength."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Method 1: Count peaks to verify wavelength
    x = pattern[0, 0, size//2, :]
    
    # Find peaks instead of zero crossings for more accurate wavelength measurement
    peaks = torch.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]
    
    expected_peaks = size / wavelength
    actual_peaks = len(peaks)
    
    logger.info(f"Expected peaks: {expected_peaks}")
    logger.info(f"Actual peaks: {actual_peaks}")
    logger.info(f"Peak positions: {peaks}")
    
    assert abs(actual_peaks - expected_peaks) <= 1, f"Wrong number of peaks: {actual_peaks} vs {expected_peaks}"

def test_frequency_grid_properties():
    """Test properties of the frequency grid used in wavelength computation."""
    size = 32
    freqs = torch.fft.fftfreq(size)
    
    # Test frequency range
    assert freqs[0] == 0, "First frequency should be DC (0)"
    assert freqs[size//2] == -0.5, "Middle frequency should be -0.5"
    assert freqs[1] == 1/size, "First positive frequency should be 1/size"
    
    # Test symmetry
    pos_freqs = freqs[1:size//2]
    neg_freqs = freqs[size//2+1:]
    assert torch.allclose(pos_freqs, -neg_freqs.flip(0)), \
        "Positive and negative frequencies should be symmetric"

def test_fft_peak_detection():
    """Test that FFT peak detection correctly identifies dominant frequency."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT manually
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Get frequency grid
    freqs = torch.fft.fftfreq(size)
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    freqs_2d = torch.sqrt(freqs_x.pow(2) + freqs_y.pow(2))
    
    # Find peak excluding DC
    mask = freqs_2d > 0
    peak_freq = freqs_2d[mask][torch.argmax(power.reshape(-1)[mask.reshape(-1)])]
    
    expected_freq = 1.0 / wavelength
    assert torch.allclose(peak_freq, torch.tensor(expected_freq), atol=1e-6), \
        f"Peak frequency incorrect: expected {expected_freq}, got {peak_freq}"

def test_wavelength_computation_stages():
    """Test each stage of wavelength computation to ensure correctness."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Stage 1: FFT computation
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    assert power.shape == pattern.shape, \
        f"Power spectrum shape mismatch: {power.shape} vs {pattern.shape}"
    
    # Stage 2: Peak detection
    freqs = torch.fft.fftfreq(size)
    freqs_2d = torch.sqrt(
        freqs[None, :].expand(size, size).pow(2) + 
        freqs[:, None].expand(size, size).pow(2)
    )
    mask = freqs_2d > 0
    peak_freq = freqs_2d[mask][torch.argmax(power.reshape(-1)[mask.reshape(-1)])]
    
    # Stage 3: Wavelength conversion
    computed_wavelength = 1.0 / peak_freq
    assert abs(computed_wavelength - wavelength) < 0.1, \
        f"Final wavelength incorrect: expected {wavelength}, got {computed_wavelength}"

def test_frequency_grid_creation():
    """Test frequency grid properties in detail."""
    size = 32
    validator = SpatialValidator()
    
    # Create frequency grid
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    freqs_2d = torch.sqrt(freqs_x.pow(2) + freqs_y.pow(2))
    
    logger.info(f"Frequency grid shape: {freqs_2d.shape}")
    logger.info(f"Max frequency: {freqs_2d.max():.4f}")
    logger.info(f"Min frequency: {freqs_2d.min():.4f}")
    logger.info(f"Nyquist frequency: {freqs.max():.4f}")
    
    # Test Nyquist frequency
    assert torch.isclose(freqs.max(), torch.tensor(0.5)), "Incorrect Nyquist frequency"
    
    # Test frequency spacing
    freq_spacing = freqs[1] - freqs[0]
    logger.info(f"Frequency spacing: {freq_spacing:.4f}")
    assert torch.isclose(freq_spacing, torch.tensor(1/size)), "Incorrect frequency spacing"

def test_pattern_creation_accuracy():
    """Test if created pattern has exact wavelength."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Count peaks manually
    x_slice = pattern[0, 0, size//2, :]
    peaks = torch.where((x_slice[1:-1] > x_slice[:-2]) & 
                       (x_slice[1:-1] > x_slice[2:]))[0]
    
    expected_peaks = size / wavelength
    actual_peaks = len(peaks)
    
    logger.info(f"Expected peaks: {expected_peaks}")
    logger.info(f"Actual peaks: {actual_peaks}")
    logger.info(f"Peak positions: {peaks}")
    
    assert abs(actual_peaks - expected_peaks) <= 1, f"Wrong number of peaks: {actual_peaks} vs {expected_peaks}"

def test_wavelength_computation_steps():
    """Test each step of wavelength computation."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    validator = SpatialValidator()
    
    # Get FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Create frequency grid
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    freqs_2d = torch.sqrt(freqs_x.pow(2) + freqs_y.pow(2))
    
    # Create mask and get valid frequencies
    mask = freqs_2d > 0
    power_valid = power.reshape(*power.shape[:-2], -1)[..., mask.reshape(-1)]
    freqs_valid = freqs_2d[mask]
    
    # Find peak
    peak_idx = torch.argmax(power_valid, dim=-1)
    peak_freq = freqs_valid[peak_idx]
    computed_wavelength = 1.0 / peak_freq
    
    logger.info(f"Power spectrum max: {power_valid.max().item():.4f}")
    logger.info(f"Peak frequency: {peak_freq.item():.4f}")
    logger.info(f"Computed wavelength: {computed_wavelength.item():.4f}")
    logger.info(f"Expected wavelength: {wavelength:.4f}")
    
    assert torch.isclose(computed_wavelength, torch.tensor(wavelength), atol=1.0), \
        f"Wavelength mismatch: {computed_wavelength:.4f} vs {wavelength:.4f}"

def test_batch_processing():
    """Test wavelength computation with batch processing."""
    size = 32
    wavelengths = [8.0, 16.0]
    patterns = torch.stack([create_test_pattern(size, w) for w in wavelengths])
    validator = SpatialValidator()
    
    computed = validator._compute_wavelength(patterns)
    expected = torch.tensor(wavelengths)
    
    logger.info(f"Batch shape: {patterns.shape}")
    logger.info(f"Computed wavelengths: {computed}")
    logger.info(f"Expected wavelengths: {expected}")
    
    assert torch.allclose(computed, expected, atol=1.0), \
        f"Batch wavelength mismatch: {computed} vs {expected}"

def test_wavelength_computation_diagnostic():
    """Diagnostic test for wavelength computation."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    validator = SpatialValidator()
    
    # Get FFT
    N = pattern.shape[-1]
    freqs = torch.fft.fftfreq(N, dtype=torch.float32)
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Log frequency components
    logger.info(f"FFT frequencies: {freqs}")
    logger.info(f"Power spectrum shape: {power.shape}")
    logger.info(f"Max power location: {torch.argmax(power.reshape(-1))}")
    logger.info(f"Power at DC: {power[0,0,0,0]}")
    
    # Compute wavelength
    wavelength_computed = validator._compute_wavelength(pattern)
    logger.info(f"Input wavelength: {wavelength}")
    logger.info(f"Computed wavelength: {wavelength_computed}")
    
    # Test each step
    freqs_2d = torch.sqrt(freqs[None, :].expand(N, N).pow(2) + 
                         freqs[:, None].expand(N, N).pow(2))
    mask = (freqs_2d > 0) & (freqs_2d <= freqs.max())
    logger.info(f"Valid frequencies: {freqs_2d[mask]}")
    logger.info(f"Number of valid frequencies: {mask.sum()}")

def test_pattern_creation_exact():
    """Test if pattern creation gives exact wavelengths."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Count peaks in middle row
    x_slice = pattern[0, 0, size//2, :]
    peaks = torch.where((x_slice[1:-1] > x_slice[:-2]) & 
                       (x_slice[1:-1] > x_slice[2:]))[0]
    
    # Expected number of cycles
    expected_cycles = size / wavelength
    actual_cycles = len(peaks)
    
    logger.info(f"Pattern size: {size}")
    logger.info(f"Target wavelength: {wavelength}")
    logger.info(f"Expected cycles: {expected_cycles}")
    logger.info(f"Actual cycles: {actual_cycles}")
    logger.info(f"Peak positions: {peaks}")
    
    assert abs(actual_cycles - expected_cycles) <= 0.5, \
        f"Wrong number of cycles: {actual_cycles} vs {expected_cycles}"
