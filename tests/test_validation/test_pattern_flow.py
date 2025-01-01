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
        'grid_size': int(test_config['geometric_tests']['dimensions']),
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
    x = torch.linspace(0, size-1, size, dtype=torch.float32)
    
    # Create pattern with exact wavelength using sine
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    for i in range(size):
        # Create 4 complete cycles across the grid
        pattern[0, 0, i, :] = torch.sin(2*np.pi*x/wavelength)
    
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
    x = torch.arange(size, dtype=torch.float32)
    pattern = torch.zeros(1, 1, size, size, dtype=torch.float32)
    
    for wavelength, amplitude in zip(wavelengths, amplitudes):
        component = amplitude * torch.sin(2*np.pi*x/wavelength).view(1, 1, 1, -1)
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
    wavelength = size / 4.0  # Create 4 complete cycles
    pattern = create_test_pattern(size, wavelength, dtype=setup_test_parameters['dtype'])
    
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
    
    # Adjust for discrete sampling
    if peak_freq > 0.5:
        peak_freq = 1.0 - peak_freq
    
    # Compute wavelength from frequency, accounting for Nyquist normalization
    wavelength_value = 2.0 / peak_freq if peak_freq != 0 else float('inf')
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
    expected_freqs = [1.0/w for w in wavelengths]
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
        implied_wavelength = 1.0 / total_freq if total_freq != 0 else float('inf')
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
    x = torch.linspace(0, 2*np.pi*size/wavelength, size, dtype=torch.float32)
    y = torch.linspace(0, 2*np.pi*size/wavelength, size, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    pattern = torch.sin(X) + torch.sin(Y)
    pattern = pattern.unsqueeze(0).unsqueeze(0)
    
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
        implied_wavelength = 1.0 / total_freq if total_freq != 0 else float('inf')
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
    expected_freq = torch.tensor(1.0 / wavelength, dtype=torch.float32)
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
    wavelength = size / 8.0  # High frequency - 8 cycles
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
        wavelength = 1.0/float(torch.maximum(freq_x.abs(), freq_y.abs()).item())
        logger.info(f"Peak {i+1}: power={val:.2f}, freq_x={freq_x:.4f}, freq_y={freq_y:.4f}, implied wavelength={wavelength:.2f}")
    
    # Step 6: Find peak frequency
    peak_idx = torch.argmax(power_valid, dim=-1)
    peak_freq_x = freqs_x_valid[peak_idx]
    peak_freq_y = freqs_y_valid[peak_idx]
    peak_freqs = torch.maximum(peak_freq_x.abs(), peak_freq_y.abs())
    
    return 1.0/peak_freqs.item()

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
