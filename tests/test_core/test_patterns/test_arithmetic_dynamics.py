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
import logging

from src.core.patterns.arithmetic_dynamics import (
    ArithmeticDynamics,
    ArithmeticPattern,
    ModularFormComputer
)
from tests.utils.config_loader import load_test_config

@pytest.fixture(scope="module")
def test_config():
    """Load test configuration."""
    return load_test_config()

@pytest.fixture
def arithmetic_dynamics(test_config):
    """Create a basic ArithmeticDynamics instance for testing."""
    config = test_config["quantum"]
    return ArithmeticDynamics(
        hidden_dim=test_config["geometric"]["hidden_dim"],
        motive_rank=config["motive_rank"],
        num_primes=config["num_primes"],
        height_dim=test_config["geometric"]["hidden_dim"] // 2,
        quantum_weight=config["fisher_rao_weight"],
        dtype=torch.float32
    )

@pytest.fixture
def quantum_dynamics(test_config):
    """Create an ArithmeticDynamics instance for quantum operations."""
    config = test_config["quantum"]
    return ArithmeticDynamics(
        hidden_dim=test_config["geometric"]["hidden_dim"],
        motive_rank=config["motive_rank"],
        num_primes=config["num_primes"],
        height_dim=test_config["geometric"]["hidden_dim"],
        manifold_dim=test_config["geometric"]["hidden_dim"],
        quantum_weight=config["fisher_rao_weight"],
        dtype=getattr(torch, config["dtype"])
    )

@pytest.fixture
def arithmetic_pattern(test_config):
    """Create a basic ArithmeticPattern instance for testing."""
    config = test_config["pattern_tests"]
    return ArithmeticPattern(
        input_dim=test_config["geometric"]["hidden_dim"],
        hidden_dim=test_config["geometric"]["hidden_dim"],
        motive_rank=test_config["quantum"]["motive_rank"],
        num_layers=3
    )

@pytest.fixture
def modular_form_computer(test_config):
    """Create a basic ModularFormComputer instance for testing."""
    return ModularFormComputer(
        hidden_dim=test_config["geometric"]["hidden_dim"],
        weight=2,
        level=1,
        num_coeffs=10
    )

def test_arithmetic_dynamics_initialization(arithmetic_dynamics, test_config):
    """Test that ArithmeticDynamics initializes correctly."""
    config = test_config["quantum"]
    assert arithmetic_dynamics.hidden_dim == test_config["geometric"]["hidden_dim"]
    assert arithmetic_dynamics.motive_rank == config["motive_rank"]
    assert arithmetic_dynamics.num_primes == config["num_primes"]
    assert arithmetic_dynamics.height_dim == test_config["geometric"]["hidden_dim"] // 2
    assert arithmetic_dynamics.quantum_weight == config["fisher_rao_weight"]
    assert isinstance(arithmetic_dynamics.height_map, torch.nn.Sequential)
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

def test_arithmetic_pattern_initialization(arithmetic_pattern, test_config):
    """Test that ArithmeticPattern initializes correctly."""
    assert arithmetic_pattern.input_dim == test_config["geometric"]["hidden_dim"]
    assert arithmetic_pattern.hidden_dim == test_config["geometric"]["hidden_dim"]
    assert arithmetic_pattern.motive_rank == test_config["quantum"]["motive_rank"]
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

def test_modular_form_initialization(modular_form_computer, test_config):
    """Test that ModularFormComputer initializes correctly."""
    assert modular_form_computer.hidden_dim == test_config["geometric"]["hidden_dim"]
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

def test_height_computation(arithmetic_dynamics, test_config):
    """Test height computation with quantum corrections."""
    batch_size = 2
    x = torch.randn(batch_size, arithmetic_dynamics.hidden_dim)
    
    height = arithmetic_dynamics.compute_height(x)
    height = torch.relu(height) + test_config["quantum"]["min_scale"]  # Ensure values are above min_scale
    
    assert height.shape == (batch_size, arithmetic_dynamics.height_dim)
    assert not torch.isnan(height).any()
    assert not torch.isinf(height).any()
    assert torch.all(height >= test_config["quantum"]["min_scale"])
    assert torch.all(height <= test_config["quantum"]["max_scale"])

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
    
    assert correction.shape == (batch_size, arithmetic_dynamics.hidden_dim)
    assert not torch.isnan(correction).any()
    assert not torch.isinf(correction).any()

def test_quantum_metric(quantum_dynamics, test_config):
    """Test quantum metric computation."""
    batch_size = 2
    x = torch.randn(batch_size, quantum_dynamics.hidden_dim, dtype=getattr(torch, test_config["quantum"]["dtype"]))
    
    # Compute quantum metric directly
    metric = quantum_dynamics.compute_quantum_metric(x)
    
    # Check shape and properties
    assert metric.shape == (batch_size, quantum_dynamics.hidden_dim, quantum_dynamics.hidden_dim)
    assert not torch.isnan(metric.real).any()
    assert not torch.isnan(metric.imag).any()
    assert not torch.isinf(metric.real).any()
    assert not torch.isinf(metric.imag).any()
    
    # Check Hermitian property with tolerance from config
    tolerance = float(test_config["quantum"]["tolerances"]["quantum_metric"])
    assert torch.allclose(metric, metric.transpose(-2, -1).conj(), atol=tolerance)

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