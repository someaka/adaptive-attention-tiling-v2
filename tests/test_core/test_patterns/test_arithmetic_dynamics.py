"""Tests for arithmetic dynamics module.

This module tests the arithmetic dynamics implementation, including:
- ArithmeticDynamics
- ArithmeticPattern
- ModularFormComputer
"""

import pytest
import torch
from torch import Tensor
from typing import Tuple, Dict, Any

from src.core.patterns.arithmetic_dynamics import (
    ArithmeticDynamics,
    ArithmeticPattern,
    ModularFormComputer
)

@pytest.fixture
def arithmetic_dynamics():
    """Create a basic ArithmeticDynamics instance for testing."""
    return ArithmeticDynamics(
        hidden_dim=64,
        motive_rank=4,
        num_primes=8,
        height_dim=4,
        quantum_weight=0.1
    )

@pytest.fixture
def arithmetic_pattern():
    """Create a basic ArithmeticPattern instance for testing."""
    return ArithmeticPattern(
        input_dim=64,
        hidden_dim=64,
        motive_rank=4,
        num_layers=3
    )

@pytest.fixture
def modular_form_computer():
    """Create a basic ModularFormComputer instance for testing."""
    return ModularFormComputer(
        hidden_dim=64,
        weight=2,
        level=1,
        num_coeffs=10
    )

def test_arithmetic_dynamics_initialization(arithmetic_dynamics):
    """Test that ArithmeticDynamics initializes correctly."""
    assert arithmetic_dynamics.hidden_dim == 64
    assert arithmetic_dynamics.motive_rank == 4
    assert arithmetic_dynamics.num_primes == 8
    assert arithmetic_dynamics.height_dim == 4
    assert arithmetic_dynamics.quantum_weight == 0.1
    assert isinstance(arithmetic_dynamics.height_map, torch.nn.Linear)
    assert isinstance(arithmetic_dynamics.flow, torch.nn.Linear)

def test_arithmetic_dynamics_forward(arithmetic_dynamics):
    """Test the forward pass of ArithmeticDynamics."""
    batch_size = 2
    seq_len = 3
    x = torch.randn(batch_size, seq_len, arithmetic_dynamics.hidden_dim)
    
    output, metrics = arithmetic_dynamics(x, steps=1)
    
    assert output.shape == x.shape
    assert isinstance(metrics, dict)
    assert 'height' in metrics
    assert 'l_value' in metrics
    assert 'flow_magnitude' in metrics

def test_arithmetic_pattern_initialization(arithmetic_pattern):
    """Test that ArithmeticPattern initializes correctly."""
    assert arithmetic_pattern.input_dim == 64
    assert arithmetic_pattern.hidden_dim == 64
    assert arithmetic_pattern.motive_rank == 4
    assert arithmetic_pattern.num_layers == 3
    assert len(arithmetic_pattern.layers) == 3
    assert isinstance(arithmetic_pattern.pattern_proj, torch.nn.Linear)

def test_arithmetic_pattern_forward(arithmetic_pattern):
    """Test the forward pass of ArithmeticPattern."""
    batch_size = 2
    seq_len = 3
    x = torch.randn(batch_size, seq_len, arithmetic_pattern.input_dim)
    
    output, metrics = arithmetic_pattern(x)
    
    assert output.shape == x.shape
    assert isinstance(metrics, list)
    assert len(metrics) == arithmetic_pattern.num_layers
    for layer_metrics in metrics:
        assert 'layer_norm' in layer_metrics
        assert 'layer_mean' in layer_metrics
        assert 'layer_std' in layer_metrics

def test_modular_form_initialization(modular_form_computer):
    """Test that ModularFormComputer initializes correctly."""
    assert modular_form_computer.hidden_dim == 64
    assert modular_form_computer.weight == 2
    assert modular_form_computer.level == 1
    assert modular_form_computer.num_coeffs == 10
    assert isinstance(modular_form_computer.coeff_net, torch.nn.Sequential)
    assert isinstance(modular_form_computer.symmetry_net, torch.nn.Sequential)

def test_modular_form_forward(modular_form_computer):
    """Test the forward pass of ModularFormComputer."""
    batch_size = 2
    x = torch.randn(batch_size, modular_form_computer.hidden_dim)
    
    q_coeffs, metrics = modular_form_computer(x)
    
    assert q_coeffs.shape == (batch_size, modular_form_computer.num_coeffs)
    assert isinstance(metrics, dict)
    assert 'weight' in metrics
    assert 'level' in metrics
    assert 'q_norm' in metrics

def test_height_computation(arithmetic_dynamics):
    """Test height computation with quantum corrections."""
    batch_size = 2
    x = torch.randn(batch_size, arithmetic_dynamics.hidden_dim)
    
    height = arithmetic_dynamics.compute_height(x)
    
    assert height.shape == (batch_size, arithmetic_dynamics.height_dim)
    assert not torch.isnan(height).any()
    assert not torch.isinf(height).any()

def test_l_function_computation(arithmetic_dynamics):
    """Test L-function computation with quantum corrections."""
    batch_size = 2
    x = torch.randn(batch_size, arithmetic_dynamics.hidden_dim)
    
    l_value = arithmetic_dynamics.compute_l_function(x)
    
    assert l_value.shape == (batch_size, arithmetic_dynamics.motive_rank)
    assert not torch.isnan(l_value).any()
    assert not torch.isinf(l_value).any()

def test_quantum_correction(arithmetic_dynamics):
    """Test quantum correction computation."""
    batch_size = 2
    metric = torch.randn(batch_size, arithmetic_dynamics.hidden_dim)
    
    correction = arithmetic_dynamics.compute_quantum_correction(metric)
    
    assert correction.shape == (batch_size, 2)  # Projects to 2D measure space
    assert not torch.isnan(correction).any()
    assert not torch.isinf(correction).any()

def test_quantum_metric(arithmetic_dynamics):
    """Test quantum metric computation."""
    batch_size = 2
    x = torch.randn(batch_size, arithmetic_dynamics.hidden_dim)
    
    metric = arithmetic_dynamics.compute_quantum_metric(x)
    
    assert metric.shape == (batch_size, arithmetic_dynamics.hidden_dim)
    assert not torch.isnan(metric).any()
    assert not torch.isinf(metric).any()

def test_modular_form_q_expansion(modular_form_computer):
    """Test q-expansion computation."""
    batch_size = 2
    x = torch.randn(batch_size, modular_form_computer.hidden_dim)
    
    q_coeffs = modular_form_computer.compute_q_expansion(x)
    
    assert q_coeffs.shape == (batch_size, modular_form_computer.num_coeffs)
    assert q_coeffs.dtype == torch.complex64
    assert not torch.isnan(q_coeffs).any()
    assert not torch.isinf(q_coeffs).any()

def test_modular_form_symmetries(modular_form_computer):
    """Test modular symmetry parameter computation."""
    batch_size = 2
    x = torch.randn(batch_size, modular_form_computer.hidden_dim)
    
    symmetries = modular_form_computer.compute_symmetries(x)
    
    assert 'translation' in symmetries
    assert 'inversion' in symmetries
    assert symmetries['translation'].shape == (batch_size,)
    assert symmetries['inversion'].shape == (batch_size,)
    assert not torch.isnan(symmetries['translation']).any()
    assert not torch.isnan(symmetries['inversion']).any() 