"""Comprehensive tests for pattern analysis and validation.

This module provides a complete test suite for pattern analysis, including:
1. Pattern Creation and Validation
2. Wavelength Computation
3. FFT and Frequency Analysis
4. Pattern Flow and Stability
"""

import torch
import pytest
import logging
import numpy as np
import math
from src.validation.patterns.formation import SpatialValidator
from src.validation.patterns.stability import PatternStabilityValidator
from src.validation.geometric.flow import FlowValidator, FlowValidationResult
from src.neural.attention.pattern.dynamics import PatternDynamics
from typing import Optional

logger = logging.getLogger(__name__)

# =====================================
# Pattern Creation and Utility Functions
# =====================================

def create_test_pattern(size: int, wavelength: float) -> torch.Tensor:
    """Create a test pattern with exact wavelength.
    
    Args:
        size: Size of the pattern (both height and width)
        wavelength: Desired wavelength in pixels
        
    Returns:
        Pattern tensor of shape (1, 1, size, size)
    """
    x = torch.arange(size, dtype=torch.float32)
    pattern = torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1)
    pattern = pattern.repeat(1, 1, size, 1)
    return pattern

def create_complex_pattern(size: int, wavelengths: list, amplitudes: Optional[list] = None) -> torch.Tensor:
    """Create a pattern with multiple wavelengths.
    
    Args:
        size: Size of the pattern
        wavelengths: List of wavelengths
        amplitudes: Optional list of amplitudes, defaults to equal amplitudes
        
    Returns:
        Pattern tensor
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(wavelengths)
    
    x = torch.arange(size, dtype=torch.float32)
    pattern = torch.zeros(1, 1, size, size)
    
    for wavelength, amplitude in zip(wavelengths, amplitudes):
        component = amplitude * torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1)
        component = component.repeat(1, 1, size, 1)
        pattern = pattern + component
        
    return pattern

# =====================================
# Basic Pattern Tests
# =====================================

def test_pattern_creation():
    """Test that pattern creation works as expected."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Check shape
    assert pattern.shape == (1, 1, size, size)
    
    # Check wavelength by measuring distance between peaks
    x_slice = pattern[0, 0, 0, :]
    peaks = torch.where((x_slice[1:-1] > x_slice[:-2]) & (x_slice[1:-1] > x_slice[2:]))[0] + 1
    peak_distances = torch.diff(peaks.float())
    assert torch.allclose(peak_distances, torch.tensor(wavelength), atol=0.1)

def test_complex_pattern():
    """Test creation and analysis of complex patterns."""
    size = 32
    wavelengths = [8.0, 16.0]
    amplitudes = [2.0, 1.0]  # First wavelength should dominate
    pattern = create_complex_pattern(size, wavelengths, amplitudes)
    
    logger.info(f"Pattern shape: {pattern.shape}")
    logger.info(f"Pattern mean: {pattern.mean():.4f}")
    logger.info(f"Pattern std: {pattern.std():.4f}")
    logger.info(f"Pattern min/max: {pattern.min():.4f}/{pattern.max():.4f}")
    
    # Expected frequencies for wavelengths
    expected_freqs = [1.0/w for w in wavelengths]
    logger.info(f"Expected frequencies: {expected_freqs}")
    
    validator = SpatialValidator()
    
    # Get intermediate values for debugging
    N = pattern.shape[-1]
    freqs = torch.fft.fftfreq(N, dtype=torch.float32)
    logger.info(f"Frequency grid shape: {freqs.shape}")
    logger.info(f"Frequency values: {freqs}")
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Zero out DC component for better peak finding
    power[..., 0, 0] = 0
    
    # Find peaks in flattened power spectrum
    power_flat = power[0, 0].reshape(-1)
    peak_values, peak_indices = torch.topk(power_flat, 5)
    
    logger.info("Power spectrum peaks (DC removed):")
    for i, (val, idx) in enumerate(zip(peak_values, peak_indices)):
        # Convert flat index back to 2D
        y_idx = idx // N
        x_idx = idx % N
        freq_x = freqs[x_idx]
        freq_y = freqs[y_idx]
        total_freq = max(abs(freq_x), abs(freq_y))
        implied_wavelength = 1.0 / total_freq if total_freq != 0 else float('inf')
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, wavelength={implied_wavelength:.2f}")
    
    # Get computed wavelength
    computed = validator._compute_wavelength(pattern)
    logger.info(f"Computed wavelength shape: {computed.shape}")
    logger.info(f"Computed wavelength value: {computed}")
    logger.info(f"Expected wavelength: {wavelengths[0]}")
    
    # Check if wavelength matches expected
    assert torch.allclose(computed, torch.tensor([[wavelengths[0]]], dtype=torch.float32), atol=1.0)

def test_complex_pattern_creation():
    """Test that complex pattern is created with correct wavelengths."""
    size = 32
    wavelengths = [8.0, 16.0]
    amplitudes = [2.0, 1.0]
    pattern = create_complex_pattern(size, wavelengths, amplitudes)
    
    # Log pattern statistics
    logger.info(f"Pattern mean: {pattern.mean()}")
    logger.info(f"Pattern std: {pattern.std()}")
    logger.info(f"Pattern min: {pattern.min()}")
    logger.info(f"Pattern max: {pattern.max()}")
    
    # Look at a single row to verify periodicity
    row = pattern[0,0,0]
    logger.info(f"First row values:\n{row}")
    
    # Compute autocorrelation to find periodicity
    row_fft = torch.fft.fft(row)
    power = torch.abs(row_fft).pow(2)
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Find peaks in power spectrum (excluding DC)
    power[0] = 0  # Zero out DC component
    peak_indices = torch.topk(power, 4).indices  # Get top 4 peaks
    peak_freqs = freqs[peak_indices]
    
    logger.info("Frequency peaks (excluding DC):")
    for i, (idx, freq) in enumerate(zip(peak_indices, peak_freqs)):
        logger.info(f"Peak {i+1}: index={idx}, freq={freq:.4f}, implied wavelength={1.0/abs(freq) if freq != 0 else float('inf'):.2f}")

def test_pattern_wavelength_scaling():
    """Test that pattern wavelength is correctly scaled."""
    size = 32
    wavelength = 8.0  # Should see 4 complete cycles in the pattern
    
    # Create pattern with correct scaling
    x = torch.linspace(0, 2*np.pi*size/wavelength, size)  # Scale to get desired wavelength
    y = torch.linspace(0, 2*np.pi*size/wavelength, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    pattern = torch.sin(X) + torch.sin(Y)
    pattern = pattern.unsqueeze(0).unsqueeze(0)
    
    # Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    power[..., 0, 0] = 0  # Zero out DC
    
    # Find peaks
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    power_flat = power[0, 0].reshape(-1)
    peak_values, peak_indices = torch.topk(power_flat, 4)
    
    logger.info("Pattern with correct wavelength scaling:")
    for i, (val, idx) in enumerate(zip(peak_values, peak_indices)):
        y_idx = idx // size
        x_idx = idx % size
        freq_x = freqs[x_idx]
        freq_y = freqs[y_idx]
        total_freq = max(abs(freq_x), abs(freq_y))
        implied_wavelength = 1.0 / total_freq if total_freq != 0 else float('inf')
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, wavelength={implied_wavelength:.2f}")
    
    # The dominant frequency should correspond to wavelength=8
    peak_idx = torch.argmax(power_flat)
    y_idx = peak_idx // size
    x_idx = peak_idx % size
    freq = max(abs(freqs[x_idx]), abs(freqs[y_idx]))
    computed_wavelength = 1.0 / freq
    
    assert abs(computed_wavelength - wavelength) < 1.0, \
        f"Computed wavelength {computed_wavelength:.2f} does not match expected {wavelength:.2f}"

# =====================================
# Frequency Analysis Tests
# =====================================

def test_frequency_grid():
    """Test frequency grid creation and properties."""
    size = 32
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Test basic properties
    assert freqs.shape == (size,)
    assert freqs[0] == 0.0  # DC component
    assert freqs[size//2] == -0.5  # Nyquist frequency
    assert torch.allclose(freqs[1:size//2].max(), torch.tensor(0.5 - 1/size))

def test_power_spectrum():
    """Test power spectrum computation."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern[0, 0])
    power = torch.abs(fft).pow(2)
    
    # Create frequency grid
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    
    # Create frequency mask
    nyquist = 0.5
    mask_x = (freqs_x.abs() > 0) & (freqs_x.abs() <= nyquist)
    mask_y = (freqs_y.abs() > 0) & (freqs_y.abs() <= nyquist)
    mask = mask_x | mask_y
    
    # Find peak frequency
    masked_power = power.clone()
    masked_power[~mask] = 0
    peak_idx = torch.argmax(masked_power)
    peak_y, peak_x = peak_idx // size, peak_idx % size
    peak_freq = max(abs(freqs_x[peak_y, peak_x]), abs(freqs_y[peak_y, peak_x]))
    
    # Check frequency
    expected_freq = 1.0 / wavelength
    assert torch.allclose(peak_freq, torch.tensor(expected_freq), atol=0.01)

# =====================================
# Wavelength Computation Tests
# =====================================

def test_wavelength_computation():
    """Test complete wavelength computation pipeline."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=1.0)

def test_wavelength_batch():
    """Test wavelength computation with batched input."""
    size = 32
    wavelengths = [8.0, 16.0]
    
    patterns = [create_test_pattern(size, w) for w in wavelengths]
    pattern_batch = torch.cat(patterns, dim=0)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern_batch)
    
    expected = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1)
    assert torch.allclose(computed, expected, atol=1.0)

def test_wavelength_noise():
    """Test wavelength computation with noisy patterns."""
    size = 32
    wavelength = 8.0
    noise_level = 0.2
    
    pattern = create_test_pattern(size, wavelength)
    noise = torch.randn_like(pattern) * noise_level
    noisy_pattern = pattern + noise
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(noisy_pattern)
    
    # Allow larger tolerance for noisy pattern
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=2.0)

def test_wavelength_high_frequency():
    """Test wavelength computation with high frequency pattern (small wavelength)."""
    size = 64
    wavelength = 4.0  # Small wavelength (high frequency)
    pattern = create_test_pattern(size, wavelength)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    # Check if computed wavelength matches expected
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=0.5)
    
    # Additional validation by checking pattern periodicity
    x_slice = pattern[0, 0, 0, :]
    peaks = torch.where((x_slice[1:-1] > x_slice[:-2]) & (x_slice[1:-1] > x_slice[2:]))[0] + 1
    measured_wavelength = torch.mean(torch.diff(peaks.float()))
    assert torch.allclose(measured_wavelength, torch.tensor(wavelength), atol=0.5)

def test_wavelength_low_frequency():
    """Test wavelength computation with low frequency pattern (large wavelength)."""
    size = 128
    wavelength = 32.0  # Large wavelength (low frequency)
    pattern = create_test_pattern(size, wavelength)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    # Check if computed wavelength matches expected
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=1.0)
    
    # Additional validation by checking pattern periodicity
    x_slice = pattern[0, 0, 0, :]
    peaks = torch.where((x_slice[1:-1] > x_slice[:-2]) & (x_slice[1:-1] > x_slice[2:]))[0] + 1
    measured_wavelength = torch.mean(torch.diff(peaks.float()))
    assert torch.allclose(measured_wavelength, torch.tensor(wavelength), atol=1.0)

def test_wavelength_computation_diagnostic():
    """Detailed diagnostic test for wavelength computation."""
    # Create a simple pattern with known wavelength
    size = 32
    wavelength = 8.0  # Should correspond to frequency 0.125
    pattern = create_test_pattern(size, wavelength)
    
    # Step 1: Verify pattern creation
    logger.info(f"Pattern shape: {pattern.shape}")
    logger.info(f"Pattern values (first row):\n{pattern[0,0,0]}")
    
    # Step 2: Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    logger.info(f"FFT shape: {fft.shape}")
    
    # Step 3: Get frequency grids
    N = pattern.shape[-1]
    freqs = torch.fft.fftfreq(N, dtype=torch.float32)
    freqs_x, freqs_y = torch.meshgrid(freqs, freqs, indexing='ij')
    logger.info(f"Expected frequency for wavelength {wavelength}: {1.0/wavelength}")
    logger.info(f"Available frequencies:\n{freqs}")
    
    # Step 4: Create frequency masks
    nyquist = 0.5
    mask_x = (freqs_x.abs() > 0) & (freqs_x.abs() <= nyquist)
    mask_y = (freqs_y.abs() > 0) & (freqs_y.abs() <= nyquist)
    mask = mask_x | mask_y
    
    # Step 5: Get valid frequencies and power
    batch_shape = power.shape[:-2]
    power_valid = power.reshape(*batch_shape, -1)[..., mask.reshape(-1)]
    freqs_x_valid = freqs_x[mask]
    freqs_y_valid = freqs_y[mask]
    
    # Log the top peaks in the valid power spectrum
    peak_values, peak_indices = torch.topk(power_valid[0,0], 5)
    logger.info("Top 5 peaks in valid power spectrum:")
    for i, (val, idx) in enumerate(zip(peak_values, peak_indices)):
        freq_x = freqs_x_valid[idx]
        freq_y = freqs_y_valid[idx]
        wavelength = 1.0/float(torch.maximum(freq_x.abs(), freq_y.abs()).item())
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, implied wavelength={wavelength:.2f}")
    
    # Step 6: Find peak frequency
    peak_idx = torch.argmax(power_valid, dim=-1)
    peak_freq_x = freqs_x_valid[peak_idx]
    peak_freq_y = freqs_y_valid[peak_idx]
    peak_freqs = torch.maximum(peak_freq_x.abs(), peak_freq_y.abs())
    
    # Log peak frequencies
    logger.info("Top 5 peaks in valid power spectrum:")
    for i, (val, idx) in enumerate(zip(peak_values, peak_indices)):
        freq_x = freqs_x_valid[idx]
        freq_y = freqs_y_valid[idx]
        wavelength = 1.0/float(torch.maximum(freq_x.abs(), freq_y.abs()).item())
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, implied wavelength={wavelength:.2f}")
    
    return 1.0/peak_freqs.item()

# =====================================
# Pattern Flow Integration Tests
# =====================================

@pytest.fixture
def setup_test_parameters():
    """Set up test parameters."""
    return {
        'batch_size': 1,
        'grid_size': 16,
        'space_dim': 2,
        'time_steps': 20,
        'tolerance': 1e-4,
        'energy_threshold': 1e-6,
        'dt': 0.1
    }

@pytest.fixture
def pattern_validator(setup_test_parameters):
    """Create pattern stability validator."""
    return PatternStabilityValidator(
        tolerance=setup_test_parameters['tolerance'],
        max_time=setup_test_parameters['time_steps']
    )

@pytest.fixture
def flow_validator(setup_test_parameters):
    """Create flow validator."""
    return FlowValidator(
        energy_threshold=1e-6,
        monotonicity_threshold=1e-4,
        singularity_threshold=1.0,
        max_iterations=1000,
        tolerance=setup_test_parameters['tolerance']
    )

def test_pattern_flow_stability(setup_test_parameters, pattern_validator, flow_validator):
    """Test pattern stability under flow."""
    params = setup_test_parameters
    
    # Create pattern
    pattern = torch.randn(
        params['batch_size'],
        params['space_dim'],
        params['grid_size'],
        params['grid_size']
    ) * 0.1
    
    # Create dynamics
    dynamics = PatternDynamics(
        grid_size=params['grid_size'],
        space_dim=params['space_dim'],
        dt=params['dt']
    )
    
    # Validate stability
    stability_result = pattern_validator.validate(
        dynamics,
        pattern,
        parameter_name='dt'
    )
    assert stability_result.is_valid, "Pattern should be stable"
    
    # Evolve pattern
    evolution = dynamics.evolve_pattern(
        pattern,
        diffusion_coefficient=0.1,
        steps=params['time_steps']
    )
    flow = torch.stack(evolution)
    
    # Validate flow
    flow_result = flow_validator.validate_long_time_existence(flow)
    assert flow_result.is_valid, "Flow should exist for long time"

def test_pattern_flow_energy(setup_test_parameters, flow_validator):
    """Test energy conservation in pattern flow."""
    params = setup_test_parameters
    
    # Create pattern and evolve
    pattern = torch.randn(
        params['batch_size'],
        params['space_dim'],
        params['grid_size'],
        params['grid_size']
    ) * 0.1
    
    dynamics = PatternDynamics(
        grid_size=params['grid_size'],
        space_dim=params['space_dim'],
        dt=params['dt']
    )
    
    evolution = dynamics.evolve_pattern(
        pattern,
        diffusion_coefficient=0.1,
        steps=params['time_steps']
    )
    flow = torch.stack(evolution)
    
    # Validate energy conservation
    energy_result = flow_validator.validate_energy_conservation(flow)
    assert energy_result.is_valid, "Flow should conserve energy"
    assert energy_result.data['energy_variation'] < params['energy_threshold']

# =====================================
# Debug and Diagnostic Tests
# =====================================

def test_wavelength_diagnostic():
    """Detailed diagnostic test for wavelength computation."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Get frequency grid
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    logger.info(f"Frequency grid: {freqs}")
    logger.info(f"Expected frequency: {1/wavelength}")
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Find peak
    power_flat = power.reshape(-1)
    peak_idx = torch.argmax(power_flat)
    freq_idx = peak_idx % size
    peak_freq = freqs[freq_idx]
    
    logger.info(f"Peak frequency: {peak_freq}")
    logger.info(f"Computed wavelength: {1/abs(peak_freq) if peak_freq != 0 else float('inf')}")
    
    # Validate
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    logger.info(f"Validator computed wavelength: {computed}")
    
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=1.0)
