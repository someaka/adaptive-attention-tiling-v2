"""Test helper functions and utilities."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import psutil
import os

from ..core.quantum.state_space import QuantumState


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
        input_data: Input data tensor
        target: Target tensor
        loss_fn: Loss function
        num_iterations: Number of iterations
        
    Returns:
        Tuple of (forward time, backward time) in seconds
    """
    model.train()
    
    # Warmup
    for _ in range(10):
        output = model(input_data)
        loss = loss_fn(output, target)
        loss.backward()
    
    # Benchmark forward pass
    forward_times = []
    backward_times = []
    
    for _ in range(num_iterations):
        start = time.time()
        output = model(input_data)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start)
        
        start = time.time()
        loss = loss_fn(output, target)
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start)
        
    return np.mean(forward_times), np.mean(backward_times)


def assert_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-6):
    """Assert that two tensors are equal within a tolerance.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        tolerance: Numerical tolerance for comparisons
    """
    assert torch.allclose(tensor1, tensor2, rtol=tolerance, atol=tolerance), \
        f"Tensors not equal within tolerance {tolerance}"


def assert_manifold_properties(metric_tensor: torch.Tensor, tolerance: float = 1e-6):
    """Assert that a metric tensor satisfies manifold properties.
    
    Args:
        metric_tensor: The metric tensor to validate
        tolerance: Numerical tolerance for comparisons
    """
    # Check symmetry
    assert torch.allclose(metric_tensor, metric_tensor.transpose(-1, -2), rtol=tolerance, atol=tolerance), \
        "Metric tensor is not symmetric"
        
    # Check positive definiteness
    eigenvals = torch.linalg.eigvalsh(metric_tensor)
    assert torch.all(eigenvals > -tolerance), \
        "Metric tensor is not positive definite"


def generate_test_quantum_state(num_qubits: int, batch_size: int = 1) -> QuantumState:
    """Generate a test quantum state.
    
    Args:
        num_qubits: Number of qubits
        batch_size: Batch size
        
    Returns:
        Test quantum state
    """
    state_dim = 2 ** num_qubits
    state = torch.randn(batch_size, state_dim, dtype=torch.complex64)
    state = state / torch.norm(state, dim=-1, keepdim=True)
    return QuantumState(state, num_qubits)


def generate_test_density_matrix(num_qubits: int, pure: bool = True) -> torch.Tensor:
    """Generate a test density matrix.
    
    Args:
        num_qubits: Number of qubits
        pure: Whether to generate a pure state
        
    Returns:
        Test density matrix
    """
    state_dim = 2 ** num_qubits
    
    if pure:
        # Generate pure state
        state = torch.randn(state_dim, dtype=torch.complex64)
        state = state / torch.norm(state)
        density = state.outer(state.conj())
    else:
        # Generate mixed state
        eigenvals = torch.rand(state_dim)
        eigenvals = eigenvals / eigenvals.sum()
        
        # Random unitary
        unitary = torch.randn(state_dim, state_dim, dtype=torch.complex64)
        unitary = torch.matrix_exp(unitary - unitary.conj().T)
        
        # Construct density matrix
        density = unitary @ torch.diag(eigenvals) @ unitary.conj().T
        
    return density


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


def assert_quantum_state_properties(state: QuantumState, tolerance: float = 1e-6):
    """Assert that a quantum state satisfies basic properties.
    
    Args:
        state: Quantum state to validate
        tolerance: Numerical tolerance for comparisons
    """
    # Check normalization
    norms = torch.norm(state.data, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=tolerance, atol=tolerance), \
        "Quantum state is not normalized"
        
    # Check shape
    expected_dim = 2 ** state.num_qubits
    assert state.data.shape[-1] == expected_dim, \
        f"Invalid state dimension {state.data.shape[-1]}, expected {expected_dim}"
        
    # Check dtype
    assert state.data.dtype == torch.complex64, \
        f"Invalid dtype {state.data.dtype}, expected torch.complex64"
