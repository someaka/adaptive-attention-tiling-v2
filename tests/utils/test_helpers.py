"""Test helper functions and utilities."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from src.core.quantum.state_space import QuantumState


def assert_manifold_properties(metric_tensor: torch.Tensor, tolerance: float = 1e-6) -> None:
    """Assert that a metric tensor satisfies manifold properties.
    
    Args:
        metric_tensor: The metric tensor to validate
        tolerance: Numerical tolerance for comparisons
    """
    # Check symmetry
    assert torch.allclose(metric_tensor, metric_tensor.transpose(-2, -1), atol=tolerance), \
        "Metric tensor must be symmetric"
    
    # Check positive definiteness
    eigenvals = torch.linalg.eigvalsh(metric_tensor)
    assert torch.all(eigenvals > -tolerance), \
        "Metric tensor must be positive definite"


def generate_test_quantum_state(num_qubits: int, batch_size: int = 1) -> QuantumState:
    """Generate a test quantum state.
    
    Args:
        num_qubits: Number of qubits
        batch_size: Batch size
        
    Returns:
        Test quantum state
    """
    dim = 2 ** num_qubits
    state = torch.randn(batch_size, dim, dtype=torch.complex64)
    state = state / torch.norm(state, dim=-1, keepdim=True)
    return QuantumState(state)


def generate_test_density_matrix(num_qubits: int, pure: bool = True) -> torch.Tensor:
    """Generate a test density matrix.
    
    Args:
        num_qubits: Number of qubits
        pure: Whether to generate a pure state
        
    Returns:
        Test density matrix
    """
    dim = 2 ** num_qubits
    if pure:
        state = torch.randn(dim, dtype=torch.complex64)
        state = state / torch.norm(state)
        return state.outer(state)
    else:
        # Generate mixed state by mixing random pure states
        num_states = 3
        weights = torch.rand(num_states)
        weights = weights / weights.sum()
        
        rho = torch.zeros((dim, dim), dtype=torch.complex64)
        for i in range(num_states):
            state = torch.randn(dim, dtype=torch.complex64)
            state = state / torch.norm(state)
            rho += weights[i] * state.outer(state)
        return rho


def assert_quantum_state_properties(state: QuantumState, tolerance: float = 1e-6) -> None:
    """Assert that a quantum state satisfies basic properties.
    
    Args:
        state: Quantum state to validate
        tolerance: Numerical tolerance for comparisons
    """
    # Check normalization
    norm = torch.norm(state.amplitudes, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=tolerance), \
        "State must be normalized"
    
    # Check density matrix properties
    rho = state.density_matrix()
    
    # Hermiticity
    assert torch.allclose(rho, rho.conj().transpose(-2, -1), atol=tolerance), \
        "Density matrix must be Hermitian"
    
    # Trace one
    trace = torch.trace(rho)
    assert torch.allclose(trace.real, torch.tensor(1.0), atol=tolerance), \
        "Density matrix must have trace one"
    assert torch.allclose(trace.imag, torch.tensor(0.0), atol=tolerance), \
        "Density matrix trace must be real"
    
    # Positive semidefinite
    eigenvals = torch.linalg.eigvalsh(rho)
    assert torch.all(eigenvals > -tolerance), \
        "Density matrix must be positive semidefinite"
