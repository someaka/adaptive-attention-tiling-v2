"""Tests for U(1) covariant utilities.

This module tests the U(1) covariant operations implemented in u1_utils.py:
1. Normalization with phase tracking
2. U(1)-covariant inner product
3. Phase evolution tracking
4. Phase composition
5. Winding number computation
"""

import pytest
import torch
import numpy as np
from typing import cast, Tuple
from src.core.quantum.u1_utils import (
    normalize_with_phase,
    u1_inner_product,
    track_phase_evolution,
    compose_phases,
    compute_winding_number
)

@pytest.fixture
def dtype():
    """Complex data type for quantum computations."""
    return torch.complex64

def test_normalize_with_phase(dtype):
    """Test normalization with phase tracking."""
    # Test case 1: Simple vector
    x = torch.tensor([1.0 + 1j, 2.0 + 2j], dtype=dtype)
    result = cast(Tuple[torch.Tensor, torch.Tensor], normalize_with_phase(x, return_phase=True))
    normalized, phase = result
    
    # Check normalization
    assert torch.allclose(torch.norm(normalized), torch.tensor(1.0)), "Vector should be normalized"
    
    # Check phase preservation
    reconstructed = normalized * torch.exp(1j * phase)
    angle_preserved = torch.allclose(
        torch.angle(x[0]/x[1]), 
        torch.angle(reconstructed[0]/reconstructed[1]),
        rtol=1e-5
    )
    assert angle_preserved, "Relative phase between components should be preserved"
    
    # Test case 2: Zero vector
    x_zero = torch.zeros(2, dtype=dtype)
    normalized_zero = cast(torch.Tensor, normalize_with_phase(x_zero))
    assert torch.allclose(normalized_zero, x_zero), "Zero vector should remain zero"
    
    # Test case 3: Real input
    x_real = torch.tensor([1.0, 2.0])
    normalized_real = cast(torch.Tensor, normalize_with_phase(x_real))
    assert normalized_real.is_complex(), "Output should be complex even for real input"
    assert torch.allclose(torch.norm(normalized_real), torch.tensor(1.0)), "Real input should be normalized"

def test_u1_inner_product(dtype):
    """Test U(1)-covariant inner product."""
    # Test case 1: Parallel vectors
    x = torch.tensor([1.0 + 1j, 2.0 + 2j], dtype=dtype)
    y = 2 * x
    inner = u1_inner_product(x, y)
    assert torch.abs(inner) == pytest.approx(1.0), "Parallel vectors should have unit inner product"
    
    # Test case 2: Orthogonal vectors
    x = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=dtype)
    y = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=dtype)
    inner = u1_inner_product(x, y)
    assert torch.abs(inner) == pytest.approx(0.0), "Orthogonal vectors should have zero inner product"
    
    # Test case 3: Phase invariance
    x = torch.tensor([1.0 + 1j, 2.0 + 2j], dtype=dtype)
    phase = torch.tensor(0.5)
    y = torch.exp(1j * phase) * x
    inner = u1_inner_product(x, y)
    assert torch.abs(inner) == pytest.approx(1.0), "Inner product should be phase invariant"

def test_track_phase_evolution(dtype):
    """Test phase evolution tracking."""
    # Test case 1: Simple rotation
    initial = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=dtype)
    angle = torch.tensor(np.pi / 4)
    # Use torch functions instead of numpy for complex number creation
    evolved = torch.stack([
        torch.complex(torch.cos(angle), torch.sin(angle)),
        torch.tensor(0.0 + 0j, dtype=dtype)
    ]).to(dtype)
    corrected, phase_diff = track_phase_evolution(initial, evolved)
    
    assert torch.abs(phase_diff - angle) < 1e-5, "Should correctly track phase difference"
    assert torch.allclose(corrected, initial), "Should recover initial state after phase correction"
    
    # Test case 2: Multiple components
    initial = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=dtype) / np.sqrt(2)
    evolved = torch.exp(1j * angle) * initial
    corrected, phase_diff = track_phase_evolution(initial, evolved)
    
    assert torch.abs(phase_diff - angle) < 1e-5, "Should track phase for multiple components"
    assert torch.allclose(corrected, initial), "Should recover initial state for multiple components"

def test_compose_phases():
    """Test phase composition."""
    # Test case 1: Simple addition
    phase1 = torch.tensor(0.5)
    phase2 = torch.tensor(0.7)
    composed = compose_phases(phase1, phase2)
    expected = torch.angle(torch.exp(1j * (phase1 + phase2)))
    assert torch.abs(composed - expected) < 1e-5, "Should correctly compose phases"
    
    # Test case 2: Wrapping around
    phase1 = torch.tensor(2.0)
    phase2 = torch.tensor(2.0)
    composed = compose_phases(phase1, phase2)
    assert -np.pi <= composed <= np.pi, "Composed phase should be in [-π, π]"
    
    # Test case 3: Opposite phases
    phase1 = torch.tensor(np.pi/2)
    phase2 = torch.tensor(-np.pi/2)
    composed = compose_phases(phase1, phase2)
    assert torch.abs(composed) < 1e-5, "Opposite phases should cancel"

def test_compute_winding_number(dtype):
    """Test winding number computation."""
    # Test case 1: No winding
    t = torch.linspace(0, 2*np.pi, 100, dtype=torch.float32)
    phase = torch.tensor(0.5, dtype=torch.float32)
    state = torch.exp(1j * phase) * torch.ones(100, dtype=dtype)
    winding = compute_winding_number(state)
    assert torch.abs(winding) < 1e-5, "Constant phase should have zero winding"
    
    # Test case 2: Single winding
    state = torch.exp(1j * t).to(dtype)
    winding = compute_winding_number(state)
    assert torch.abs(winding - 1.0) < 1e-5, "Should detect single winding"
    
    # Test case 3: Multiple windings
    state = torch.exp(2j * t).to(dtype)
    winding = compute_winding_number(state)
    assert torch.abs(winding - 2.0) < 1e-5, "Should detect multiple windings"
    
    # Test case 4: Negative winding
    state = torch.exp(-1j * t).to(dtype)
    winding = compute_winding_number(state)
    assert torch.abs(winding + 1.0) < 1e-5, "Should detect negative winding"
    
    # Test case 5: Constant vector
    state = torch.ones(10, dtype=dtype)
    winding = compute_winding_number(state)
    assert torch.abs(winding) < 1e-5, "Constant vector should have zero winding" 