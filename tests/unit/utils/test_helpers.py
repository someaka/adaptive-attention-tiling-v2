"""Test helper utilities for common testing operations."""

from __future__ import annotations

import gc
import logging
import time
from typing import Callable, TypeVar, Any, Protocol, Optional, Tuple

import numpy as np
import psutil
import torch
from torch import nn

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=torch.Tensor)


def get_memory_stats() -> tuple[int, int]:
    """Get memory stats.
    
    Returns:
        Tuple of (allocated, reserved) memory in bytes
    """
    try:
        vm = psutil.virtual_memory()
        return vm.used, vm.total
    except Exception as e:
        logger.warning("Failed to get memory stats: %s", e)
        return 0, 0


def run_memory_test(fn, device="cpu") -> tuple[int, int, Any]:
    """Run a memory test.
    
    Args:
        fn: Function to test
        device: Device to run on
        
    Returns:
        tuple: (memory allocated, memory reserved, function result) in bytes
    """
    gc.collect()
    start_allocated, start_reserved = get_memory_stats()
    
    result = fn()
    
    end_allocated, end_reserved = get_memory_stats()
    
    return (end_allocated - start_allocated, end_reserved - start_reserved, result)


def measure_memory_usage(fn, device="cpu") -> tuple[Any, dict[str, int]]:
    """Measure memory usage of a function.
    
    Args:
        fn: Function to test
        device: Device to run on
        
    Returns:
        tuple: (result, memory_stats)
    """
    gc.collect()
    start_allocated, start_reserved = get_memory_stats()
    
    result = fn()
    
    end_allocated, end_reserved = get_memory_stats()
    
    return result, {
        "allocated_diff": end_allocated - start_allocated,
        "reserved_diff": end_reserved - start_reserved,
        "peak_reserved": end_reserved
    }


def assert_manifold_properties(tensor: torch.Tensor) -> bool:
    """Assert that a tensor has valid manifold properties."""
    if not isinstance(tensor, torch.Tensor):
        return False
    
    if tensor.dim() < 2:
        return False
    
    if torch.isnan(tensor).any():
        return False
    
    if torch.isinf(tensor).any():
        return False
    
    return True


def generate_random_tensor(
    shape: tuple[int, ...],
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    *,  # Force requires_grad to be keyword-only
    requires_grad: bool = False,
) -> torch.Tensor:
    """Generate a random tensor with specified properties."""
    tensor = torch.randn(*shape, device=device, dtype=dtype)
    tensor.requires_grad_(requires_grad)
    return tensor


def assert_tensor_equal(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that two tensors are equal within tolerance."""
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        avg_diff = torch.mean(diff).item()
        error_msg = f"Tensors not equal. Max diff: {max_diff}, Avg diff: {avg_diff}"
        raise AssertionError(error_msg)


def check_memory_usage(fn: Callable[[], Any]) -> tuple[int, int, Any]:
    """Check memory usage before and after function execution.

    Args:
        fn: Function to check memory usage for

    Returns:
        tuple: (memory allocated, memory reserved, function result) in bytes
    """
    gc.collect()
    start_allocated, start_reserved = get_memory_stats()

    result = fn()

    end_allocated, end_reserved = get_memory_stats()

    return (end_allocated - start_allocated, end_reserved - start_reserved, result)


def benchmark_forward_backward(
    model: nn.Module,
    input_shape: tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cpu",
) -> tuple[float, float]:
    """Benchmark forward and backward passes of a model.

    Returns
    -------
        tuple[float, float]: Average forward time, average backward time in seconds

    """
    forward_times: list[float] = []
    backward_times: list[float] = []

    # Generate random input
    x = generate_random_tensor(input_shape, device=device, requires_grad=True)

    # Warmup rounds
    for _ in range(warmup_iterations):
        out = model(x)
        loss = out.mean()
        loss.backward()  # type: ignore[no-untyped-call]

    # Benchmark rounds
    for _ in range(num_iterations):
        # Forward pass
        start_time = time.perf_counter()
        out = model(x)
        forward_times.append(time.perf_counter() - start_time)

        # Backward pass
        loss = out.mean()
        start_time = time.perf_counter()
        loss.backward()  # type: ignore[no-untyped-call]
        backward_times.append(time.perf_counter() - start_time)

    return float(np.mean(forward_times)), float(np.mean(backward_times))


def measure_time(fn: Callable) -> Tuple[float, Any]:
    """Measure execution time of a function.
    
    Args:
        fn: Function to measure
        
    Returns:
        tuple: (execution time in seconds, function result)
    """
    start_time = time.time()
    result = fn()
    end_time = time.time()
    
    return end_time - start_time, result


def run_performance_test(fn: Callable, 
                        warmup_iters: int = 3,
                        test_iters: int = 10,
                        device: str = "cpu") -> dict:
    """Run a performance test.
    
    Args:
        fn: Function to test
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations
        device: Device to run on
        
    Returns:
        dict: Performance metrics
    """
    # Run warmup iterations
    for _ in range(warmup_iters):
        fn()
        
    # Run test iterations and measure memory
    result, memory_stats = measure_memory_usage(fn, device)
    
    # Measure time
    time_taken, _ = measure_time(fn)
    
    return {
        "time": time_taken,
        "memory": memory_stats,
        "iterations": test_iters
    }
