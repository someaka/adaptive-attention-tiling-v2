"""Test helper functions and utilities."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import psutil
import os

from src.core.quantum.state_space import QuantumState


def measure_memory_usage(func: Callable) -> Callable:
    """Decorator to measure memory usage of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that measures memory
    """
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run function
        result = func(*args, **kwargs)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Print memory usage
        print(f"Memory usage for {func.__name__}:")
        print(f"Initial: {initial_memory:.2f} MB")
        print(f"Final: {final_memory:.2f} MB")
        print(f"Difference: {final_memory - initial_memory:.2f} MB")
        
        return result
    return wrapper


def benchmark_forward_backward(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable,
    num_iterations: int = 100,
) -> Tuple[float, float]:
    """Benchmark forward and backward passes.
    
    Args:
        model: Model to benchmark
        input_data: Input data
        target: Target data
        loss_fn: Loss function
        num_iterations: Number of iterations
        
    Returns:
        Average forward time, average backward time
    """
    forward_times = []
    backward_times = []
    
    for _ in range(num_iterations):
        # Forward pass
        start = time.time()
        output = model(input_data)
        forward_time = time.time() - start
        forward_times.append(forward_time)
        
        # Backward pass
        loss = loss_fn(output, target)
        start = time.time()
        loss.backward()
        backward_time = time.time() - start
        backward_times.append(backward_time)
        
        model.zero_grad()
    
    return np.mean(forward_times), np.mean(backward_times)


def assert_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-6) -> None:
    """Assert that two tensors are equal within a tolerance.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        tolerance: Numerical tolerance for comparisons
    """
    assert tensor1.shape == tensor2.shape, f"Tensor shapes don't match: {tensor1.shape} vs {tensor2.shape}"
    assert torch.allclose(tensor1, tensor2, atol=tolerance), \
        f"Tensors not equal within tolerance {tolerance}"


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


def generate_random_tensor(shape: Tuple[int, ...], requires_grad: bool = True) -> torch.Tensor:
    """Generate a random tensor for testing.
    
    Args:
        shape: Shape of tensor to generate
        requires_grad: Whether tensor requires gradients
        
    Returns:
        Random tensor
    """
    tensor = torch.randn(*shape, requires_grad=requires_grad)
    return tensor


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
