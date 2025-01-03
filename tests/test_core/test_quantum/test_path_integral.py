"""Tests for quantum path integral functionality."""

import numpy as np
import pytest
import torch

from src.core.quantum.path_integral import (
    ActionFunctional,
    PathSampler,
    Propagator,
    StationaryPhase,
    HilbertSpace,
    QuantumState,
)


def test_action_functional():
    """Test action functional computation."""
    # Initialize
    hilbert_space = HilbertSpace(dim=2)
    action = ActionFunctional(hilbert_space)
    
    # Test path
    path = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    
    # Compute action
    result = action.compute_action(path)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0  # Scalar output
    
    # Test stationary points
    boundary = (path[0], path[-1])
    stationary = action.stationary_points(boundary)
    assert stationary.points.shape[0] == 100  # Default num_points
    assert stationary.points.shape[1] == 2  # Dimension
    assert torch.allclose(stationary.points[0], boundary[0])
    assert torch.allclose(stationary.points[-1], boundary[1])


def test_path_sampler():
    """Test path sampling functionality."""
    # Initialize
    hilbert_space = HilbertSpace(dim=2)
    action = ActionFunctional(hilbert_space)
    sampler = PathSampler(action, num_paths=10)
    
    # Sample paths
    start = torch.zeros(2)
    end = torch.ones(2)
    paths = sampler.sample_paths(start, end)
    
    assert len(paths) == 10  # num_paths
    for path in paths:
        assert torch.allclose(path.points[0], start)
        assert torch.allclose(path.points[-1], end)
        assert path.weight.is_complex()


def test_propagator():
    """Test quantum propagator."""
    # Initialize
    hilbert_space = HilbertSpace(dim=2)
    action = ActionFunctional(hilbert_space)
    sampler = PathSampler(action)
    propagator = Propagator(sampler, hilbert_space)
    
    # Initial state
    initial_state = QuantumState(
        amplitudes=torch.tensor([1.0, 0.0], dtype=torch.complex64),
        basis_labels=["0", "1"],
        phase=torch.tensor(0.0)
    )
    
    # Final points
    final_points = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    
    # Propagate
    final_state = propagator.propagate(initial_state, final_points)
    assert isinstance(final_state, QuantumState)
    assert final_state.amplitudes.shape == initial_state.amplitudes.shape
    
    # Convert sum to float32 before comparison
    total_prob = torch.sum(torch.abs(final_state.amplitudes)**2).to(torch.float32)
    assert torch.allclose(total_prob, torch.tensor(1.0, dtype=torch.float32), atol=1e-6)


def test_stationary_phase():
    """Test stationary phase approximation."""
    # Initialize
    hilbert_space = HilbertSpace(dim=2)
    action = ActionFunctional(hilbert_space)
    stationary = StationaryPhase(action)
    
    # Find classical path
    boundary = (torch.zeros(2), torch.ones(2))
    classical_path = stationary.find_classical_path(boundary)
    
    # Test path properties
    assert torch.allclose(classical_path.points[0], boundary[0])
    assert torch.allclose(classical_path.points[-1], boundary[1])
    
    # Test quantum corrections
    corrections = stationary.quantum_corrections(classical_path)
    assert isinstance(corrections, torch.Tensor)
    assert corrections.is_complex()
