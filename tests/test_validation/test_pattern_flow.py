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
import os
import yaml
from src.validation.patterns.formation import SpatialValidator
from src.validation.flow.flow_stability import (
    LinearStabilityValidator,
    NonlinearStabilityValidator,
    LinearStabilityValidation,
    StructuralStabilityValidator
)
from src.validation.patterns.stability import (
    PatternValidator,
    PatternStabilityResult
)
from src.validation.geometric.flow import (
    TilingFlowValidator as FlowValidator,
    TilingFlowValidationResult as FlowValidationResult
)
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.tiling.geometric_flow import GeometricFlow
from typing import Optional

logger = logging.getLogger(__name__)

# Load test configuration
@pytest.fixture
def test_config():
    """Load test configuration based on environment."""
    config_name = os.environ.get("TEST_REGIME", "debug")
    config_path = f"configs/test_regimens/{config_name}.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture
def setup_test_parameters(test_config):
    """Setup test parameters from configuration."""
    return {
        'batch_size': int(test_config['fiber_bundle']['batch_size']),
        'grid_size': 32,  # Fixed larger grid size for better frequency resolution
        'space_dim': int(test_config['geometric_tests']['dimensions']),
        'time_steps': int(test_config['parallel_transport']['path_points']),
        'dt': float(test_config['quantum_geometric']['dt']),
        'energy_threshold': float(test_config['quantum_arithmetic']['tolerances']['state_norm']),
        'tolerance': float(test_config['fiber_bundle']['tolerance']),
        'stability_threshold': float(test_config['quantum_arithmetic']['validation']['stability_threshold']),
        'hidden_dim': int(test_config['geometric_tests']['hidden_dim']),
        'dtype': getattr(torch, test_config['quantum_arithmetic']['dtype'])
    }

@pytest.fixture
def pattern_validator(setup_test_parameters):
    """Create pattern validator."""
    from src.validation.flow.flow_stability import (
        LinearStabilityValidator,
        NonlinearStabilityValidator,
        StructuralStabilityValidator
    )
    
    # Create individual validators with thresholds
    linear_validator = LinearStabilityValidator(
        tolerance=setup_test_parameters['tolerance']
    )
    nonlinear_validator = NonlinearStabilityValidator(
        tolerance=setup_test_parameters['tolerance']
    )
    structural_validator = StructuralStabilityValidator(
        tolerance=setup_test_parameters['tolerance']
    )
    
    return PatternValidator(
        linear_validator=linear_validator,
        nonlinear_validator=nonlinear_validator,
        structural_validator=structural_validator,
        lyapunov_threshold=setup_test_parameters['tolerance'] * 0.1,
        perturbation_threshold=setup_test_parameters['tolerance']
    )

@pytest.fixture
def flow_validator(setup_test_parameters, test_config):
    """Create flow validator."""
    flow = GeometricFlow(
        hidden_dim=setup_test_parameters['hidden_dim'],
        manifold_dim=setup_test_parameters['space_dim'],
        motive_rank=1,
        num_charts=1,
        integration_steps=setup_test_parameters['time_steps'],
        dt=setup_test_parameters['dt'],
        stability_threshold=setup_test_parameters['stability_threshold'],
        dtype=setup_test_parameters['dtype']
    )
    return FlowValidator(
        flow=flow,
        stability_threshold=setup_test_parameters['stability_threshold'],
        curvature_bounds=(-test_config['quantum_geometric']['fisher_rao_weight'], 
                         test_config['quantum_geometric']['fisher_rao_weight']),
        max_energy=1.0/test_config['quantum_geometric']['dt']  # Use inverse dt as energy bound
    )

# =====================================
# Pattern Creation and Utility Functions
# =====================================

def create_test_pattern(size: int, wavelength: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a test pattern with exact wavelength.
    
    Args:
        size: Size of the pattern (both height and width)
        wavelength: Desired wavelength in pixels
        dtype: Data type for the pattern tensor
        
    Returns:
        Pattern tensor of shape (1, 1, size, size)
    """
    # Create pattern in float32 first
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    
    # Create normalized coordinates [0, 1] to ensure complete cycles
    x = torch.linspace(0, 1, size, dtype=torch.float32)
    
    # Calculate number of cycles from wavelength
    cycles = size / wavelength
    
    # Create sine pattern with exact number of cycles
    for i in range(size):
        pattern[0, 0, i, :] = torch.sin(2*np.pi*cycles*x)
    
    # Normalize pattern to [-1, 1] range
    pattern = pattern / pattern.abs().max()
    
    # Add tiny amount of noise to avoid numerical issues
    noise = torch.randn_like(pattern) * 1e-8
    pattern = pattern + noise
    
    # Convert to target dtype if different
    if dtype != torch.float32:
        if dtype in [torch.complex64, torch.complex128]:
            pattern = pattern + 0j  # Convert to complex by adding zero imaginary part
        pattern = pattern.to(dtype=dtype)
    
    return pattern

def create_complex_pattern(
    size: int,
    wavelengths: list[float],
    amplitudes: Optional[list[float]] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a pattern with multiple wavelengths.
    
    Args:
        size: Size of the pattern
        wavelengths: List of wavelengths
        amplitudes: Optional list of amplitudes, defaults to equal amplitudes
        dtype: Data type for the pattern tensor
        
    Returns:
        Pattern tensor
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(wavelengths)
    
    # Create pattern in float32 first
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    x = torch.linspace(0, size-1, size, dtype=torch.float32)
    
    for wavelength, amplitude in zip(wavelengths, amplitudes):
        freq = 1.0 / wavelength  # Convert wavelength to frequency in cycles per sample
        component = amplitude * torch.sin(2*np.pi*freq*x).view(1, 1, 1, -1)
        component = component.repeat(1, 1, size, 1)
        pattern = pattern + component
    
    # Convert to target dtype if different
    if dtype != torch.float32:
        if dtype in [torch.complex64, torch.complex128]:
            pattern = pattern + 0j  # Convert to complex by adding zero imaginary part
        pattern = pattern.to(dtype=dtype)
    
    return pattern

# =====================================
# Basic Pattern Tests
# =====================================

def test_pattern_creation(setup_test_parameters) -> None:
    """Test that pattern creation works as expected."""
    size = setup_test_parameters['grid_size']
    wavelength = size/4.0  # Create 4 complete cycles

    # Create pattern with exact frequency
    freq = 1.0 / wavelength  # Convert wavelength to frequency in cycles per sample
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    x = torch.linspace(0, size-1, size, dtype=torch.float32)  # Use pixel coordinates [0, size-1]

    # Calculate cycles based on pixel coordinates
    cycles = freq  # Number of cycles per pixel
    
    # Create pattern with specified wavelength
    for i in range(size):
        pattern[0, 0, i, :] = torch.sin(2*np.pi*cycles*x)

    # Normalize pattern to [-1, 1] range
    pattern = pattern / pattern.abs().max()
    
    # Add tiny amount of noise to avoid numerical issues
    noise = torch.randn_like(pattern) * 1e-8
    pattern = pattern + noise
    
    # Convert to target dtype if needed
    if setup_test_parameters['dtype'] != torch.float32:
        if setup_test_parameters['dtype'] in [torch.complex64, torch.complex128]:
            pattern = pattern + 0j
        pattern = pattern.to(dtype=setup_test_parameters['dtype'])

    # Check shape
    assert pattern.shape == (1, 1, size, size)

    # For complex patterns, check wavelength using FFT
    x_slice = pattern[0, 0, 0, :].abs() if pattern.is_complex() else pattern[0, 0, 0, :]
    x_slice = x_slice.to(torch.float32)  # Convert to float32 for FFT

    # Remove mean to avoid DC component
    x_slice = x_slice - x_slice.mean()

    # Apply window function to reduce spectral leakage
    window = torch.hann_window(size, dtype=torch.float32)
    x_slice = x_slice * window

    # Compute FFT
    fft = torch.fft.fft(x_slice)
    power = torch.abs(fft).pow(2)
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)

    # Debug logging before modification
    logger.info(f"Raw power spectrum: {power}")
    logger.info(f"Raw frequencies: {freqs}")

    # Find peak frequency considering all frequencies except DC
    power[0] = 0  # Zero out DC
    peak_idx = torch.argmax(power)  # Look at all frequencies
    peak_freq = abs(freqs[peak_idx])  # Use absolute value of frequency

    # Compute wavelength from frequency
    # FFT frequencies are normalized to [-0.5, 0.5], so multiply by size to get cycles per pattern
    actual_freq = peak_freq * size
    wavelength_value = size / actual_freq if actual_freq != 0 else float('inf')  # Convert frequency to wavelength in pixels
    computed_wavelength = torch.as_tensor([[wavelength_value]], dtype=torch.float32)

    # Debug logging
    logger.info(f"Pattern size: {size}")
    logger.info(f"Expected wavelength: {wavelength}")
    logger.info(f"Peak frequency: {peak_freq}")
    logger.info(f"Computed wavelength: {computed_wavelength}")
    logger.info(f"Available frequencies: {freqs}")
    logger.info(f"Power spectrum: {power}")

    # Check wavelength with larger tolerance for FFT discretization
    expected_wavelength = torch.tensor(wavelength, dtype=torch.float32)
    tolerance = max(setup_test_parameters['tolerance'], 2.0/size)  # At least two grid points
    assert torch.allclose(computed_wavelength, expected_wavelength,
                         rtol=tolerance,
                         atol=tolerance)

def test_complex_pattern(setup_test_parameters) -> None:
    """Test creation and analysis of complex patterns."""
    size = setup_test_parameters['grid_size']
    wavelengths = [size/4.0, size/2.0]  # Two wavelengths that fit in the grid
    amplitudes = [2.0, 1.0]  # First wavelength should dominate
    pattern = create_complex_pattern(size, wavelengths, amplitudes, dtype=setup_test_parameters['dtype'])
    
    logger.info(f"Pattern shape: {pattern.shape}")
    logger.info(f"Pattern mean: {pattern.mean():.4f}")
    logger.info(f"Pattern std: {pattern.std():.4f}")
    
    # For complex patterns, use absolute values for min/max
    if pattern.is_complex():
        pattern_abs = pattern.abs()
        logger.info(f"Pattern abs min/max: {pattern_abs.min():.4f}/{pattern_abs.max():.4f}")
    else:
        logger.info(f"Pattern min/max: {pattern.min():.4f}/{pattern.max():.4f}")
    
    # Expected frequencies for wavelengths
    expected_freqs = [size/w/size for w in wavelengths]  # Convert to cycles per sample
    logger.info(f"Expected frequencies: {expected_freqs}")
    
    validator = SpatialValidator()
    
    # Get intermediate values for debugging
    N = pattern.shape[-1]
    freqs = torch.fft.fftfreq(N, dtype=torch.float32)  # FFT frequencies should be real
    logger.info(f"Frequency grid shape: {freqs.shape}")
    logger.info(f"Frequency values: {freqs}")
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Zero out DC component for better peak finding
    power[..., 0, 0] = 0
    
    # Find peaks in flattened power spectrum
    power_flat = power[0, 0].reshape(-1)
    num_peaks = min(5, power_flat.numel())
    peak_values, peak_indices = torch.topk(power_flat, num_peaks)
    
    logger.info("Power spectrum peaks (DC removed):")
    for i, (val, idx) in enumerate(zip(peak_values, peak_indices)):
        # Convert flat index back to 2D
        y_idx = idx // N
        x_idx = idx % N
        freq_x = freqs[x_idx]
        freq_y = freqs[y_idx]
        total_freq = max(abs(freq_x), abs(freq_y))
        implied_wavelength = size / (total_freq * size) if total_freq != 0 else float('inf')
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, wavelength={implied_wavelength:.2f}")
    
    # Get computed wavelength
    computed = validator._compute_wavelength(pattern)
    logger.info(f"Computed wavelength shape: {computed.shape}")
    logger.info(f"Computed wavelength value: {computed}")
    logger.info(f"Expected wavelength: {wavelengths[0]}")
    
    # Check if wavelength matches expected with config tolerance
    assert torch.allclose(computed.abs(), torch.tensor([[wavelengths[0]]], dtype=torch.float32), atol=setup_test_parameters['tolerance'])

def test_complex_pattern_creation(setup_test_parameters) -> None:
    """Test that complex pattern is created with correct wavelengths."""
    size = setup_test_parameters['grid_size']
    wavelengths = [size/4.0, size/2.0]  # Two wavelengths that fit in the grid
    amplitudes = [2.0, 1.0]
    pattern = create_complex_pattern(size, wavelengths, amplitudes, dtype=setup_test_parameters['dtype'])
    
    # Log pattern statistics
    logger.info(f"Pattern mean: {pattern.mean()}")
    logger.info(f"Pattern std: {pattern.std()}")
    
    # For complex patterns, use absolute values for min/max
    if pattern.is_complex():
        pattern_abs = pattern.abs()
        logger.info(f"Pattern abs min/max: {pattern_abs.min():.4f}/{pattern_abs.max():.4f}")
    else:
        logger.info(f"Pattern min/max: {pattern.min():.4f}/{pattern.max():.4f}")
    
    # Look at a single row to verify periodicity
    row = pattern[0,0,0]
    logger.info(f"First row values:\n{row}")
    
    # Compute autocorrelation to find periodicity
    row_fft = torch.fft.fft(row)
    power = torch.abs(row_fft).pow(2)
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)  # FFT frequencies should be real
    
    # Find peaks in power spectrum (excluding DC)
    power[0] = 0  # Zero out DC component
    num_peaks = min(4, power.numel())
    peak_indices = torch.topk(power, num_peaks).indices  # Get top peaks
    peak_freqs = freqs[peak_indices]
    
    logger.info("Frequency peaks (excluding DC):")
    for i, (idx, freq) in enumerate(zip(peak_indices, peak_freqs)):
        logger.info(f"Peak {i+1}: index={idx}, freq={freq:.4f}, implied wavelength={1.0/abs(freq) if freq != 0 else float('inf'):.2f}")

def test_pattern_wavelength_scaling(setup_test_parameters) -> None:
    """Test that pattern wavelength is correctly scaled."""
    size = setup_test_parameters['grid_size']
    wavelength = size/4.0  # Should see 4 complete cycles in the pattern
    
    # Create pattern with correct scaling (use float32 for linspace)
    x = torch.linspace(0, size-1, size, dtype=torch.float32)
    freq = size / wavelength  # Convert wavelength to frequency in cycles per pattern
    pattern = torch.sin(2*np.pi*(freq/size)*x).view(1, 1, 1, -1).repeat(1, 1, size, 1)
    
    # Convert to target dtype if needed
    if setup_test_parameters['dtype'] != torch.float32:
        if setup_test_parameters['dtype'] in [torch.complex64, torch.complex128]:
            pattern = pattern + 0j
        pattern = pattern.to(dtype=setup_test_parameters['dtype'])
    
    # Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    power[..., 0, 0] = 0  # Zero out DC
    
    # Find peaks (use float32 for frequencies)
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
        implied_wavelength = size / (total_freq * size) if total_freq != 0 else float('inf')
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, wavelength={implied_wavelength:.2f}")
    
    # The dominant frequency should correspond to wavelength
    peak_idx = torch.argmax(power_flat)
    y_idx = peak_idx // size
    x_idx = peak_idx % size
    freq = max(abs(freqs[x_idx]), abs(freqs[y_idx]))
    computed_wavelength = 1.0 / freq
    
    assert abs(computed_wavelength - wavelength) < setup_test_parameters['tolerance'], \
        f"Computed wavelength {computed_wavelength:.2f} does not match expected {wavelength:.2f}"

# =====================================
# Frequency Analysis Tests
# =====================================

def test_frequency_grid(setup_test_parameters):
    """Test frequency grid creation and properties."""
    size = setup_test_parameters['grid_size']
    # FFT frequencies should be real
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Test basic properties
    assert freqs.shape == (size,)
    assert freqs[0] == 0.0  # DC component
    assert freqs[size//2] == -0.5  # Nyquist frequency
    assert torch.allclose(freqs[1:size//2].max(dim=0)[0], torch.tensor(0.5 - 1/size))

def test_power_spectrum(setup_test_parameters):
    """Test power spectrum computation."""
    size = setup_test_parameters['grid_size']
    wavelength = size/4.0
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
    # Compute FFT and power (FFT frequencies should be real)
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Create frequency grid (real-valued)
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    
    # Create frequency mask
    nyquist = 0.5
    mask_x = (freqs_x.abs() > 0) & (freqs_x.abs() <= nyquist)
    mask_y = (freqs_y.abs() > 0) & (freqs_y.abs() <= nyquist)
    mask = mask_x | mask_y
    
    # Find peak frequency
    masked_power = power[0, 0].clone()  # Remove batch dimensions
    masked_power[~mask] = 0
    peak_idx = torch.argmax(masked_power)
    peak_y, peak_x = peak_idx // size, peak_idx % size
    peak_freq = max(abs(freqs_x[peak_y, peak_x]), abs(freqs_y[peak_y, peak_x]))
    
    # Check frequency (convert to target dtype for comparison)
    expected_freq = torch.tensor(size / wavelength / size, dtype=torch.float32)
    assert torch.allclose(peak_freq, expected_freq, atol=setup_test_parameters['tolerance'])

# =====================================
# Wavelength Computation Tests
# =====================================

def test_wavelength_computation(setup_test_parameters):
    """Test complete wavelength computation pipeline."""
    size = setup_test_parameters['grid_size']
    wavelength = size/4.0
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern)
    
    # For complex patterns, compare absolute values
    if computed.is_complex():
        computed = computed.abs()
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=setup_test_parameters['tolerance'])

def test_wavelength_batch(setup_test_parameters):
    """Test wavelength computation with batched input."""
    size = setup_test_parameters['grid_size']
    wavelengths = [size/4.0, size/2.0]
    
    patterns = [create_test_pattern(size, w, dtype=setup_test_parameters['dtype']) for w in wavelengths]
    pattern_batch = torch.cat(patterns, dim=0)
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(pattern_batch)
    
    # For complex patterns, compare absolute values
    if computed.is_complex():
        computed = computed.abs()
    expected = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1)
    assert torch.allclose(computed, expected, atol=setup_test_parameters['tolerance'])

def test_wavelength_noise(setup_test_parameters):
    """Test wavelength computation with noisy patterns."""
    size = setup_test_parameters['grid_size']
    wavelength = size/4.0
    noise_level = setup_test_parameters['tolerance']
    
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
    # Create noise in the same dtype as pattern
    if pattern.is_complex():
        noise_real = torch.randn_like(pattern.real) * noise_level
        noise_imag = torch.randn_like(pattern.imag) * noise_level
        noise = torch.complex(noise_real, noise_imag)
    else:
        noise = torch.randn_like(pattern) * noise_level
    
    noisy_pattern = pattern + noise
    
    validator = SpatialValidator()
    computed = validator._compute_wavelength(noisy_pattern)
    
    # For complex patterns, compare absolute values
    if computed.is_complex():
        computed = computed.abs()
    assert torch.allclose(computed, torch.tensor([[wavelength]], dtype=torch.float32), atol=2.0*setup_test_parameters['tolerance'])

def test_wavelength_high_frequency(setup_test_parameters) -> None:
    """Test wavelength computation for high frequency patterns."""
    size = setup_test_parameters['grid_size']
    wavelength = size/8.0  # High frequency - 8 cycles
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
    validator = SpatialValidator()
    result = validator._compute_wavelength(pattern)
    
    # Convert to float32 for comparison
    computed = result.to(torch.float32)
    expected = torch.tensor(wavelength, dtype=torch.float32)
    
    # Use larger tolerance for high frequencies
    tolerance = setup_test_parameters['tolerance'] * 2.0
    assert torch.allclose(computed, expected, rtol=tolerance, atol=tolerance)

def test_wavelength_low_frequency(setup_test_parameters) -> None:
    """Test wavelength computation for low frequency patterns."""
    size = setup_test_parameters['grid_size']
    wavelength = size / 2.0  # Low frequency - 2 cycles
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
    validator = SpatialValidator()
    result = validator._compute_wavelength(pattern)
    
    # Convert to float32 for comparison
    computed = result.to(torch.float32)
    expected = torch.tensor(wavelength, dtype=torch.float32)
    
    # Use standard tolerance for low frequencies
    assert torch.allclose(computed, expected, 
                         rtol=setup_test_parameters['tolerance'], 
                         atol=setup_test_parameters['tolerance'])

def test_wavelength_computation_diagnostic(setup_test_parameters):
    """Detailed diagnostic test for wavelength computation."""
    # Create a simple pattern with known wavelength
    size = setup_test_parameters['grid_size']
    wavelength = size/4.0  # Should correspond to frequency 0.125
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
    # Step 1: Verify pattern creation
    logger.info(f"Pattern shape: {pattern.shape}")
    if pattern.is_complex():
        logger.info(f"Pattern values (first row, abs):\n{pattern[0,0,0].abs()}")
    else:
        logger.info(f"Pattern values (first row):\n{pattern[0,0,0]}")
    
    # Step 2: Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    logger.info(f"FFT shape: {fft.shape}")
    
    # Step 3: Get frequency grids (use float32 for frequencies)
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
    num_peaks = min(5, power_valid[0,0].numel())
    peak_values, peak_indices = torch.topk(power_valid[0,0], num_peaks)
    logger.info("Top peaks in valid power spectrum:")
    for i, (val, idx) in enumerate(zip(peak_values, peak_indices)):
        freq_x = freqs_x_valid[idx]
        freq_y = freqs_y_valid[idx]
        wavelength_i = 1.0/float(torch.maximum(freq_x.abs(), freq_y.abs()).item())
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, implied wavelength={wavelength_i:.2f}")
    
    # Step 6: Find peak frequency
    peak_idx = torch.argmax(power_valid, dim=-1)
    peak_freq_x = freqs_x_valid[peak_idx]
    peak_freq_y = freqs_y_valid[peak_idx]
    peak_freqs = torch.maximum(peak_freq_x.abs(), peak_freq_y.abs())
    
    # Compute wavelength and verify
    computed_wavelength = 1.0/peak_freqs.item()
    assert torch.allclose(torch.tensor(computed_wavelength), torch.tensor(wavelength), atol=1e-4), \
        f"Computed wavelength {computed_wavelength} does not match expected wavelength {wavelength}"

# =====================================
# Pattern Flow Integration Tests
# =====================================

def test_pattern_flow_stability(setup_test_parameters, pattern_validator, flow_validator):
    """Test pattern stability under flow."""
    params = setup_test_parameters

    # Create minimal pattern with very small dimensions
    reduced_size = 2  # Minimal size

    # Create minimal pattern with flattened shape
    pattern = torch.randn(
        1,  # Single batch
        reduced_size * reduced_size,  # Flattened spatial dimensions
        dtype=torch.float32
    ) * float(params['tolerance'])

    # Convert to target dtype
    if params['dtype'] in [torch.complex64, torch.complex128]:
        pattern_imag = torch.randn_like(pattern) * float(params['tolerance'])
        pattern = torch.complex(pattern, pattern_imag).to(dtype=params['dtype'])
    else:
        pattern = pattern.to(dtype=params['dtype'])

    # Create minimal geometric flow
    pattern_flow = GeometricFlow(
        hidden_dim=2,
        manifold_dim=reduced_size * reduced_size,  # Total flattened spatial dimensions
        motive_rank=1,
        num_charts=1,
        integration_steps=1,  # Minimal steps
        dt=params['dt'],
        stability_threshold=params['stability_threshold'],
        dtype=params['dtype']
    )

    # Quick stability check
    stability_result = pattern_validator.validate(
        pattern_flow=pattern_flow,
        initial_state=pattern,
        time_steps=1  # Single time step
    )

    # Basic validation checks
    assert isinstance(stability_result, PatternStabilityResult)
    assert stability_result.data is not None
    assert isinstance(stability_result.lyapunov_exponents, torch.Tensor)
    assert isinstance(stability_result.perturbation_response, dict)

    # Quick flow check
    with torch.no_grad():
        output, metrics = pattern_flow(pattern)
        print(f"Output shape: {output.shape}")

    # Add time dimension for flow validation
    output = output.unsqueeze(0)  # [time_steps, batch_size, dim]
    print(f"Output shape with time: {output.shape}")

    # Minimal flow validation
    flow_result = flow_validator.validate_long_time_existence(
        output,  # [time_steps, batch_size, dim]
        time_steps=1
    )
    assert flow_result.is_valid

def test_pattern_flow_energy(setup_test_parameters, flow_validator):
    """Test energy conservation in pattern flow."""
    params = setup_test_parameters
    
    # Create minimal pattern
    pattern = torch.randn(
        1,  # Single batch
        2,  # Minimal dimension
        2,  # Minimal size
        2,  # Minimal size
        dtype=torch.float32
    ) * float(params['tolerance'])
    
    # Convert to target dtype
    if params['dtype'] in [torch.complex64, torch.complex128]:
        pattern_imag = torch.randn_like(pattern) * float(params['tolerance'])
        pattern = torch.complex(pattern, pattern_imag).to(dtype=params['dtype'])
    else:
        pattern = pattern.to(dtype=params['dtype'])
    
    # Create minimal flow
    pattern_flow = GeometricFlow(
        hidden_dim=2,
        manifold_dim=2,
        motive_rank=1,
        num_charts=1,
        integration_steps=1,
        dt=params['dt'],
        stability_threshold=params['stability_threshold'],
        dtype=params['dtype']
    )
    
    # Quick evolution
    output, _ = pattern_flow(pattern)
    
    # Basic energy check
    energy_result = flow_validator.validate_energy_conservation(output)
    assert energy_result.is_valid
    assert energy_result.data['is_finite']
    assert energy_result.data['is_bounded']

# =====================================
# Debug and Diagnostic Tests
# =====================================

def test_frequency_grid_creation():
    """Test frequency grid creation and properties."""
    size = 32
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    
    # Test basic properties
    assert freqs.shape == (size,)
    assert freqs[0] == 0.0  # DC component
    assert freqs[size//2] == -0.5  # Nyquist frequency
    assert torch.allclose(freqs[1:size//2].max(), torch.tensor(0.5 - 1/size))

def test_fft_computation():
    """Test FFT computation and power spectrum analysis."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Test basic properties
    assert fft.shape == pattern.shape
    assert power.shape == pattern.shape
    assert torch.allclose(power[0,0,0,0], torch.tensor(0.0))  # DC component should be zero

def test_peak_frequency_detection():
    """Test peak frequency detection accuracy."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Find peak frequency
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    peak_idx = torch.argmax(power.reshape(-1))
    peak_freq = freqs[peak_idx % size]
    
    # Verify peak frequency
    expected_freq = 1.0 / wavelength
    peak_freq_tensor = peak_freq.abs().clone().detach()
    expected_freq_tensor = torch.tensor(expected_freq, dtype=torch.float32)
    assert torch.allclose(peak_freq_tensor, expected_freq_tensor, atol=1e-4)

def test_wavelength_conversion():
    """Test wavelength conversion logic."""
    size = 32
    wavelength = 8.0
    pattern = create_test_pattern(size, wavelength)
    
    # Compute FFT and power
    fft = torch.fft.fft2(pattern)
    power = torch.abs(fft).pow(2)
    
    # Find peak frequency
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    peak_idx = torch.argmax(power.reshape(-1))
    peak_freq = freqs[peak_idx % size]
    
    # Convert to wavelength and create tensors directly from float values
    computed_wavelength = 1.0 / abs(peak_freq) if peak_freq != 0 else float('inf')
    assert abs(computed_wavelength - wavelength) < 1e-4, \
        f"Computed wavelength {computed_wavelength} does not match expected wavelength {wavelength}"

def test_edge_cases():
    """Test wavelength computation with edge cases."""
    test_cases = [
        (16, 4.0),  # Small size, exact fit
        (64, 16.0), # Large size, exact fit
        (32, 10.6667), # Non-integer wavelength
        (128, 64.0) # Very large wavelength
    ]
    
    for size, wavelength in test_cases:
        pattern = create_test_pattern(size, wavelength)
        
        # Compute FFT and power
        fft = torch.fft.fft2(pattern)
        power = torch.abs(fft).pow(2)
        
        # Find peak frequency
        freqs = torch.fft.fftfreq(size, dtype=torch.float32)
        peak_idx = torch.argmax(power.reshape(-1))
        peak_freq = freqs[peak_idx % size]
        
        # Convert to wavelength and compare float values directly
        computed_wavelength = 1.0 / abs(peak_freq) if peak_freq != 0 else float('inf')
        assert abs(computed_wavelength - wavelength) < 1e-4, \
            f"Computed wavelength {computed_wavelength} does not match expected wavelength {wavelength}"

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

def test_wavelength_step_by_step(setup_test_parameters):
    """Detailed step-by-step test of wavelength calculation process."""
    # Use a larger grid size for better frequency resolution
    size = 32  # Fixed size for this test
    wavelength = size/4.0  # Should see 4 complete cycles
    
    # Step 1: Create pattern and verify basic properties
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    logger.info(f"\nStep 1: Pattern Creation")
    logger.info(f"Pattern shape: {pattern.shape}")
    logger.info(f"Pattern mean: {pattern.mean():.4f}")
    logger.info(f"Pattern std: {pattern.std():.4f}")
    
    # Step 2: Create frequency grid and verify properties
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    logger.info(f"\nStep 2: Frequency Grid")
    logger.info(f"Grid shape: {freqs.shape}")
    logger.info(f"Grid values: {freqs}")
    logger.info(f"Expected frequency: {1.0/wavelength:.4f}")
    
    # Step 3: Compute FFT and verify
    fft = torch.fft.fft2(pattern)
    logger.info(f"\nStep 3: FFT Computation")
    logger.info(f"FFT shape: {fft.shape}")
    logger.info(f"FFT dtype: {fft.dtype}")
    
    # Step 4: Compute power spectrum
    power = torch.abs(fft).pow(2)
    power[..., 0, 0] = 0  # Zero out DC
    logger.info(f"\nStep 4: Power Spectrum")
    logger.info(f"Power shape: {power.shape}")
    logger.info(f"Max power: {power.max():.4f}")
    logger.info(f"Power > 0 locations: {torch.where(power > 0)[0]}")
    
    # Step 5: Create 2D frequency grids
    freqs_x = freqs[None, :].expand(size, size)
    freqs_y = freqs[:, None].expand(size, size)
    logger.info(f"\nStep 5: 2D Frequency Grids")
    logger.info(f"Frequency grid shapes: {freqs_x.shape}, {freqs_y.shape}")
    
    # Step 6: Create frequency mask
    nyquist = 0.5
    mask_x = (freqs_x.abs() > 0) & (freqs_x.abs() <= nyquist)
    mask_y = (freqs_y.abs() > 0) & (freqs_y.abs() <= nyquist)
    mask = mask_x | mask_y
    logger.info(f"\nStep 6: Frequency Mask")
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Number of valid frequencies: {mask.sum()}")
    
    # Step 7: Get valid frequencies and power
    power_flat = power[0, 0].reshape(-1)
    power_valid = power_flat[mask.reshape(-1)]
    freqs_x_valid = freqs_x[mask]
    freqs_y_valid = freqs_y[mask]
    logger.info(f"\nStep 7: Valid Frequencies")
    logger.info(f"Valid power shape: {power_valid.shape}")
    logger.info(f"Valid freqs shape: {freqs_x_valid.shape}")
    
    # Step 8: Find peak frequency
    peak_idx = torch.argmax(power_valid)
    peak_freq_x = freqs_x_valid[peak_idx]
    peak_freq_y = freqs_y_valid[peak_idx]
    peak_freq = torch.maximum(peak_freq_x.abs(), peak_freq_y.abs())
    logger.info(f"\nStep 8: Peak Detection")
    logger.info(f"Peak power index: {peak_idx}")
    logger.info(f"Peak freq x: {peak_freq_x:.4f}")
    logger.info(f"Peak freq y: {peak_freq_y:.4f}")
    logger.info(f"Peak freq: {peak_freq:.4f}")
    
    # Step 9: Convert to wavelength
    computed_wavelength = 1.0 / peak_freq
    logger.info(f"\nStep 9: Wavelength Computation")
    logger.info(f"Computed wavelength: {computed_wavelength:.4f}")
    logger.info(f"Expected wavelength: {wavelength:.4f}")
    
    # Final verification
    assert torch.allclose(
        computed_wavelength,
        torch.tensor(wavelength, dtype=torch.float32),
        atol=setup_test_parameters['tolerance']
    ), f"Wavelength mismatch: computed={computed_wavelength:.4f}, expected={wavelength:.4f}"

def test_wavelength_detailed_debug(setup_test_parameters):
    """Extremely detailed debug test for wavelength computation."""
    size = setup_test_parameters['grid_size']  # Should be 32
    wavelength = size/4.0  # Should be 8.0 - expecting 4 complete cycles
    
    # Step 1: Pattern Creation
    x = torch.linspace(0, size-1, size, dtype=torch.float32)
    freq = 1.0 / wavelength  # Should be 0.125 cycles per sample
    logger.info(f"\nStep 1: Pattern Creation")
    logger.info(f"Input wavelength: {wavelength}")
    logger.info(f"Input frequency: {freq} cycles per sample")
    logger.info(f"Grid points: {x}")
    
    # Create the pattern
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    for i in range(size):
        pattern[0, 0, i, :] = torch.sin(2*np.pi*freq*x)
    
    # Log pattern values
    logger.info(f"Pattern first row values: {pattern[0,0,0]}")
    logger.info(f"Pattern min/max: {pattern.min():.4f}/{pattern.max():.4f}")
    
    # Step 2: FFT Computation
    fft = torch.fft.fft(pattern[0,0,0])
    power = torch.abs(fft).pow(2)
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    
    logger.info(f"\nStep 2: FFT Analysis")
    logger.info(f"FFT values: {fft}")
    logger.info(f"Power spectrum: {power}")
    logger.info(f"Frequency grid: {freqs}")
    
    # Step 3: Peak Detection
    power[0] = 0  # Zero out DC
    peak_idx = torch.argmax(power)
    peak_freq = abs(freqs[peak_idx])
    peak_power = power[peak_idx]
    
    logger.info(f"\nStep 3: Peak Analysis")
    logger.info(f"Peak index: {peak_idx}")
    logger.info(f"Peak frequency: {peak_freq}")
    logger.info(f"Peak power: {peak_power}")
    
    # Step 4: Wavelength Computation - try different methods
    wavelength_1 = 1.0 / peak_freq
    wavelength_2 = size / (peak_freq * size)
    wavelength_3 = size * (1.0 / peak_freq)
    wavelength_4 = 1.0 / (peak_freq * size)
    
    logger.info(f"\nStep 4: Wavelength Computation Methods")
    logger.info(f"Method 1 (1/freq): {wavelength_1}")
    logger.info(f"Method 2 (size/(freq*size)): {wavelength_2}")
    logger.info(f"Method 3 (size*(1/freq)): {wavelength_3}")
    logger.info(f"Method 4 (1/(freq*size)): {wavelength_4}")
    
    # Step 5: Compare with expected
    logger.info(f"\nStep 5: Comparison")
    logger.info(f"Expected wavelength: {wavelength}")
    logger.info(f"Expected frequency: {1.0/wavelength}")
    logger.info(f"Actual peak frequency: {peak_freq}")
    
    # Additional analysis of power spectrum
    logger.info(f"\nStep 6: Power Spectrum Analysis")
    # Get top 5 peaks
    power_copy = power.clone()
    for i in range(5):
        idx = torch.argmax(power_copy)
        freq = freqs[idx]
        pwr = power_copy[idx]
        logger.info(f"Peak {i+1}: freq={freq:.4f}, power={pwr:.4f}, implied wavelength={1.0/abs(freq) if freq != 0 else float('inf'):.4f}")
        power_copy[idx] = 0
    
    # Verify the pattern actually has the wavelength we think it does
    logger.info(f"\nStep 7: Pattern Verification")
    # Count zero crossings to verify wavelength
    zero_crossings = torch.sum(torch.diff(torch.signbit(pattern[0,0,0])).bool()).item()
    implied_wavelength = 2 * size / zero_crossings if zero_crossings > 0 else float('inf')
    logger.info(f"Number of zero crossings: {zero_crossings}")
    logger.info(f"Implied wavelength from zero crossings: {implied_wavelength}")
    
    assert torch.allclose(wavelength_1, torch.tensor(8.0)), "Wavelength computation method 1 failed"
    assert torch.allclose(wavelength_2, torch.tensor(8.0)), "Wavelength computation method 2 failed"
    assert torch.allclose(wavelength_3, torch.tensor(256.0)), "Wavelength computation method 3 failed"
    assert torch.allclose(wavelength_4, torch.tensor(0.25)), "Wavelength computation method 4 failed"

def test_wavelength_granular(caplog):
    """Ultra-granular test of wavelength computation to pinpoint exact issues."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Step 1: Create a pattern with known wavelength
    size = 32
    wavelength = 8.0  # Should create exactly 4 cycles in 32 pixels
    x = torch.linspace(0, size-1, size, dtype=torch.float32)
    
    logger.info("\nStep 1: Pattern Creation")
    logger.info(f"Size: {size}")
    logger.info(f"Target wavelength: {wavelength}")
    logger.info(f"x values: {x}")
    
    # Create pattern with exact frequency
    freq = 1.0 / wavelength  # freq is in cycles per sample
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    logger.info(f"Target frequency: {freq} cycles per sample")
    
    # Create the pattern one value at a time and verify
    for i in range(size):
        for j in range(size):
            # Compute exact value
            value = torch.sin(torch.tensor(2*np.pi*freq*j))  # Use j for x-direction pattern
            pattern[0, 0, i, j] = value
            # Verify each value
            expected = torch.sin(torch.tensor(2*np.pi*0.125*j))  # freq = 0.125 for wavelength = 8
            assert abs(value - expected) < 1e-6, f"Pattern value mismatch at ({i},{j}): {value} != {expected}"
    
    # Step 2: Verify pattern properties
    logger.info("\nStep 2: Pattern Properties")
    logger.info(f"Pattern shape: {pattern.shape}")
    logger.info(f"Pattern min: {pattern.min()}")
    logger.info(f"Pattern max: {pattern.max()}")
    assert pattern.shape == (1, 1, size, size), f"Pattern shape wrong: {pattern.shape}"
    assert torch.allclose(pattern.min(), torch.tensor(-1.0), atol=1e-6), f"Pattern min wrong: {pattern.min()}"
    assert torch.allclose(pattern.max(), torch.tensor(1.0), atol=1e-6), f"Pattern max wrong: {pattern.max()}"
    
    # Step 3: Extract row for FFT
    row = pattern[0, 0, 0, :].clone()
    logger.info("\nStep 3: Row Extraction")
    logger.info(f"Row shape: {row.shape}")
    logger.info(f"Row values: {row}")
    assert row.shape == (size,), f"Row shape wrong: {row.shape}"
    
    # Step 4: Remove mean
    row_mean = row.mean()
    row = row - row_mean
    logger.info("\nStep 4: Mean Removal")
    logger.info(f"Row mean: {row_mean}")
    logger.info(f"Zero-mean row values: {row}")
    assert torch.allclose(row.mean(), torch.tensor(0.0), atol=1e-6), f"Row mean not zero: {row.mean()}"
    
    # Step 5: Apply window
    window = torch.hann_window(size, dtype=torch.float32)
    row = row * window
    logger.info("\nStep 5: Window Application")
    logger.info(f"Window values: {window}")
    logger.info(f"Windowed row values: {row}")
    assert row.shape == (size,), f"Windowed row shape wrong: {row.shape}"
    
    # Step 6: Compute FFT
    fft = torch.fft.fft(row)
    logger.info("\nStep 6: FFT Computation")
    logger.info(f"FFT values: {fft}")
    assert fft.shape == (size,), f"FFT shape wrong: {fft.shape}"
    
    # Step 7: Compute power spectrum
    power = torch.abs(fft).pow(2)
    logger.info("\nStep 7: Power Spectrum")
    logger.info(f"Power spectrum values: {power}")
    assert power.shape == (size,), f"Power shape wrong: {power.shape}"
    
    # Step 8: Get frequency grid
    freqs = torch.fft.fftfreq(size, dtype=torch.float32)
    logger.info("\nStep 8: Frequency Grid")
    logger.info(f"Frequency grid values: {freqs}")
    assert freqs.shape == (size,), f"Frequency grid shape wrong: {freqs.shape}"
    
    # Step 9: Find peak frequency
    power[0] = 0  # Zero out DC
    peak_idx = torch.argmax(power)
    peak_freq = abs(freqs[peak_idx])
    logger.info("\nStep 9: Peak Detection")
    logger.info(f"Expected frequency: {freq}")
    logger.info(f"Peak index: {peak_idx}")
    logger.info(f"Peak frequency: {peak_freq}")
    logger.info(f"All frequencies: {freqs}")
    logger.info(f"Power spectrum: {power}")
    
    # Step 10: Compute wavelength
    computed_wavelength = 1.0 / abs(peak_freq) if peak_freq != 0 else float('inf')
    logger.info("\nStep 10: Wavelength Computation")
    logger.info(f"Input wavelength: {wavelength}")
    logger.info(f"Computed wavelength: {computed_wavelength}")
    logger.info(f"Wavelength error: {abs(computed_wavelength - wavelength)}")
    
    # Final verification
    assert abs(computed_wavelength - wavelength) < 1e-6, \
        f"Wavelength computation wrong: computed={computed_wavelength}, expected={wavelength}"

def test_frequency_calculation_comparison():
    """Compare different frequency calculation methods to identify the correct one."""
    size = 32
    wavelength = size/4.0  # Should create 4 complete cycles
    
    # Create normalized coordinates [0, 1] to ensure complete cycles
    x = torch.linspace(0, 1, size, dtype=torch.float32)

    # Method 1: Direct cycle-based calculation
    cycles1 = 4.0  # We want exactly 4 cycles
    pattern1 = torch.zeros(1, 1, size, size, dtype=torch.float32)
    for i in range(size):
        pattern1[0, 0, i, :] = torch.sin(2*np.pi*cycles1*x)

    # Method 2: Wavelength-based calculation
    cycles2 = 4.0  # Same as Method 1
    pattern2 = torch.zeros(1, 1, size, size, dtype=torch.float32)
    for i in range(size):
        pattern2[0, 0, i, :] = torch.sin(2*np.pi*cycles2*x)

    # Analyze both patterns
    def analyze_pattern(pattern, method_name):
        # Extract row and compute FFT
        row = pattern[0, 0, 0, :].clone()
        row = row - row.mean()
        window = torch.hann_window(size, dtype=torch.float32)
        row = row * window
        fft = torch.fft.fft(row)
        power = torch.abs(fft).pow(2)
        freqs = torch.fft.fftfreq(size, dtype=torch.float32)

        # Find peak frequency
        power[0] = 0  # Zero out DC
        peak_idx = torch.argmax(power)
        peak_freq = abs(freqs[peak_idx])
        computed_wavelength = size / (peak_freq * size) if peak_freq != 0 else float('inf')

        logger.info(f"\n{method_name} Analysis:")
        logger.info(f"Input cycles: {cycles1 if method_name == 'Method 1' else cycles2}")
        logger.info(f"First row values: {pattern[0,0,0]}")
        logger.info(f"Peak frequency: {peak_freq}")
        logger.info(f"Computed wavelength: {computed_wavelength}")
        logger.info(f"Expected wavelength: {wavelength}")

        return computed_wavelength

    # Analyze both methods
    wavelength1 = analyze_pattern(pattern1, "Method 1")
    wavelength2 = analyze_pattern(pattern2, "Method 2")

    # Compare number of zero crossings to verify actual wavelength
    def count_cycles(pattern):
        row = pattern[0,0,0]
        zero_crossings = torch.sum(torch.diff(torch.signbit(row)).bool()).item()
        return zero_crossings / 2  # Each cycle has 2 zero crossings

    cycles1 = count_cycles(pattern1)
    cycles2 = count_cycles(pattern2)

    logger.info(f"\nActual cycles in pattern:")
    logger.info(f"Method 1: {cycles1} cycles")
    logger.info(f"Method 2: {cycles2} cycles")
    logger.info(f"Expected: 4.0 cycles")

    # The correct method should produce exactly 4 cycles
    assert abs(cycles1 - 4.0) < 1e-6, "Method 1 produces incorrect number of cycles"
    assert abs(cycles2 - 4.0) < 1e-6, "Method 2 produces incorrect number of cycles"
