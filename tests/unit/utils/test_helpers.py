"""Test helper utilities for common testing operations."""

from __future__ import annotations

import gc
import logging
import time
from typing import Callable, TypeVar, Any, Protocol, Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=torch.Tensor)


def has_vulkan() -> bool:
    """Check if Vulkan is available."""
    return hasattr(torch, 'vulkan') and getattr(torch, 'vulkan').is_available()


def vulkan_sync() -> None:
    """Synchronize Vulkan device if available."""
    if has_vulkan():
        getattr(torch, 'vulkan').synchronize()


def vulkan_empty_cache() -> None:
    """Empty Vulkan cache if available."""
    if has_vulkan():
        getattr(torch, 'vulkan').empty_cache()


def vulkan_memory_stats() -> tuple[int, int]:
    """Get Vulkan memory stats if available.
    
    Returns:
        tuple[int, int]: (allocated memory, reserved memory) in bytes
    """
    if has_vulkan():
        try:
            vulkan = getattr(torch, 'vulkan')
            allocated = vulkan.memory_allocated()
            reserved = vulkan.max_memory_allocated()
            return allocated, reserved
        except RuntimeError as e:
            logger.warning("Failed to get Vulkan memory stats: %s", e)
    return 0, 0


def assert_manifold_properties(tensor: torch.Tensor) -> bool:
    """
    Assert that a tensor satisfies basic manifold properties.
    
    Args:
        tensor: Tensor to check
        
    Returns:
        bool: True if properties are satisfied
    """
    # Check smoothness (finite values)
    assert torch.all(torch.isfinite(tensor)).item(), "Tensor contains non-finite values"
    
    # Check differentiability (gradients can be computed)
    if tensor.requires_grad:
        try:
            loss = tensor.sum()
            loss.backward()
            if tensor.grad is not None:
                assert torch.all(torch.isfinite(tensor.grad)).item(), "Non-finite gradients"
        except Exception as e:
            raise AssertionError(f"Differentiability check failed: {e}")
    
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
    vulkan_empty_cache()
    vulkan_sync()

    start_allocated, start_reserved = vulkan_memory_stats()

    result = fn()

    vulkan_sync()
    end_allocated, end_reserved = vulkan_memory_stats()

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
        if device == "vulkan":
            vulkan_sync()
            start_time = time.perf_counter()
            out = model(x)
            vulkan_sync()
            forward_times.append(time.perf_counter() - start_time)
        else:
            start_time = time.perf_counter()
            out = model(x)
            forward_times.append(time.perf_counter() - start_time)

        # Backward pass
        loss = out.mean()
        if device == "vulkan":
            vulkan_sync()
            start_time = time.perf_counter()
            loss.backward()  # type: ignore[no-untyped-call]
            vulkan_sync()
            backward_times.append(time.perf_counter() - start_time)
        else:
            start_time = time.perf_counter()
            loss.backward()  # type: ignore[no-untyped-call]
            backward_times.append(time.perf_counter() - start_time)

    return float(np.mean(forward_times)), float(np.mean(backward_times))


def measure_memory_usage(func: Callable[[], torch.Tensor]) -> tuple[int, int]:
    """Measure memory usage of a function.

    Args:
    ----
        func: Function to measure memory usage of.

    Returns:
    -------
        Tuple of (allocated memory, reserved memory).

    """
    # Run garbage collection to get accurate memory measurements
    gc.collect()
    vulkan_empty_cache()
    vulkan_sync()

    # Get initial memory stats
    start_allocated, start_reserved = vulkan_memory_stats()

    # Run the function
    func()
    vulkan_sync()

    # Get final memory stats
    end_allocated, end_reserved = vulkan_memory_stats()

    # For Vulkan, ensure we capture the peak memory usage
    if has_vulkan():
        end_reserved = max(end_reserved, getattr(torch, 'vulkan').max_memory_allocated())

    return (end_allocated - start_allocated, end_reserved - start_reserved)
