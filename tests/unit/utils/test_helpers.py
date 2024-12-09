"""Test helper utilities for common testing operations."""

from __future__ import annotations

import gc
import logging
import time
from typing import Callable, TypeVar

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=torch.Tensor)


def generate_random_tensor(
    shape: tuple[int, ...],
    device: str = "cpu",
    dtype: torch.dtype | None = None,
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


def synchronize_device():
    """Synchronize device operations."""
    if hasattr(torch, "vulkan") and torch.vulkan.is_available():
        try:
            torch.vulkan.synchronize()
        except RuntimeError as e:
            logger.warning("Failed to synchronize device: %s", e)


def empty_cache():
    """Empty device memory cache."""
    if hasattr(torch, "vulkan") and torch.vulkan.is_available():
        try:
            torch.vulkan.empty_cache()
        except RuntimeError as e:
            logger.warning("Failed to empty cache: %s", e)


def get_memory_stats():
    """Get current memory statistics.

    Returns:
        tuple: (allocated memory, reserved memory) in bytes
    """
    if hasattr(torch, "vulkan") and torch.vulkan.is_available():
        try:
            allocated = torch.vulkan.memory_allocated()
            reserved = (
                torch.vulkan.max_memory_allocated()
            )  # Use max allocated as reserved for Vulkan
        except RuntimeError as e:
            logger.warning("Failed to get memory stats: %s", e)
            return 0, 0
        else:
            return allocated, reserved
    return 0, 0  # Return 0 for CPU-only mode


def check_memory_usage(fn):
    """Check memory usage before and after function execution.

    Args:
        fn: Function to check memory usage for

    Returns:
        tuple: (memory allocated, memory reserved) in bytes
    """
    empty_cache()
    synchronize_device()

    start_allocated, start_reserved = get_memory_stats()

    result = fn()

    synchronize_device()
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
        if device == "vulkan":
            synchronize_device()
            start_time = time.perf_counter()
            out = model(x)
            synchronize_device()
            forward_times.append(time.perf_counter() - start_time)
        else:
            start_time = time.perf_counter()
            out = model(x)
            forward_times.append(time.perf_counter() - start_time)

        # Backward pass
        loss = out.mean()
        if device == "vulkan":
            synchronize_device()
            start_time = time.perf_counter()
            loss.backward()  # type: ignore[no-untyped-call]
            synchronize_device()
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
    empty_cache()
    synchronize_device()  # Add synchronization before measurement

    # Get initial memory stats
    start_allocated, start_reserved = get_memory_stats()

    # Run the function
    func()
    synchronize_device()  # Add synchronization after function execution

    # Get final memory stats
    end_allocated, end_reserved = get_memory_stats()

    # For Vulkan, ensure we capture the peak memory usage
    if hasattr(torch, "vulkan") and torch.vulkan.is_available():
        end_reserved = max(end_reserved, torch.vulkan.max_memory_allocated())

    return (end_allocated - start_allocated, end_reserved - start_reserved)
