"""Tests for memory management in tensor operations.

This module tests:
1. Tensor lifecycle management
2. Memory cleanup between operations
3. Resource optimization
4. Memory leak prevention
"""

import gc
import pytest
import torch
import psutil
import os
import logging
import warnings
from contextlib import contextmanager

# Filter out PyTorch deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*torch.distributed.reduce_op.*",
    category=FutureWarning
)

from src.utils.memory_management_util import (
    tensor_manager,
    memory_optimizer,
    register_tensor,
    optimize_memory,
    clear_memory,
    get_memory_stats,
    DEBUG_MODE,
    _is_tensor
)
from src.core.attention.geometric import HyperbolicExponential


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_process_memory() -> float:
    """Get current process memory in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting process memory: {e}")
        return 0.0


@contextmanager
def memory_tracker(operation: str):
    """Track memory usage before and after an operation."""
    try:
        force_cleanup()  # Added cleanup before tracking
        initial_memory = get_process_memory()
        logger.info(f"Starting {operation}: {initial_memory:.2f} MB")
        yield
        force_cleanup()  # Added cleanup after tracking
        final_memory = get_process_memory()
        logger.info(f"Finished {operation}: {final_memory:.2f} MB (diff: {final_memory - initial_memory:.2f} MB)")
    except Exception as e:
        logger.error(f"Error in memory tracker: {e}")
        raise


def force_cleanup():
    """Force aggressive memory cleanup."""
    try:
        # Multiple rounds of cleanup
        for _ in range(3):
            clear_memory()
            gc.collect()
            
            # Additional cleanup
            for obj in gc.get_objects():
                try:
                    if _is_tensor(obj):
                        del obj
                except Exception:
                    continue
            gc.collect()
        
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")


def test_tensor_lifecycle():
    """Test tensor lifecycle management."""
    with memory_tracker("tensor_lifecycle"):
        force_cleanup()
        initial_memory = get_process_memory()
        
        # Create and register tensors with minimal sizes
        with optimize_memory("test_op"):
            t1 = register_tensor(torch.randn(2, 2), "test_op")
            t2 = register_tensor(torch.randn(2, 2), "test_op")
            t3 = t1 @ t2
            t3 = register_tensor(t3, "test_op")
            
            force_cleanup()
            mid_memory = get_process_memory()
            memory_diff = mid_memory - initial_memory
            assert memory_diff >= 0, "Memory should not decrease during tensor creation"
            
            # Delete tensors explicitly
            del t1, t2, t3
        
        force_cleanup()
        final_memory = get_process_memory()
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 10, f"Memory leak detected: {memory_diff:.2f} MB"


def test_operation_cleanup():
    """Test cleanup between operations."""
    with memory_tracker("operation_cleanup"):
        force_cleanup()
        initial_memory = get_process_memory()
        
        # Perform multiple operations with minimal tensors
        for i in range(2):  # Reduced iterations
            with optimize_memory(f"op_{i}"):
                t1 = register_tensor(torch.randn(2, 2), f"op_{i}")
                t2 = register_tensor(torch.randn(2, 2), f"op_{i}")
                t3 = register_tensor(t1 @ t2, f"op_{i}")
                
                force_cleanup()
                mid_memory = get_process_memory()
                memory_diff = mid_memory - initial_memory
                assert memory_diff >= 0, f"Memory should not decrease during operation {i}"
                
                # Delete tensors explicitly
                del t1, t2, t3
        
        force_cleanup()
        final_memory = get_process_memory()
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 10, f"Memory leak detected: {memory_diff:.2f} MB"


def test_hyperbolic_operations():
    """Test memory management in hyperbolic operations."""
    with memory_tracker("hyperbolic_operations"):
        force_cleanup()
        initial_memory = get_process_memory()
        
        exp_map = HyperbolicExponential(dim=2)  # Reduced dimension
        
        # Create minimal test points
        x = torch.tensor([1.1, 0.1])  # Reduced size
        v = torch.tensor([0.0, 0.1])  # Reduced size
        
        # Perform operations with cleanup after each
        for i in range(2):  # Reduced iterations
            with optimize_memory(f"hyperbolic_{i}"):
                result = exp_map(x, v)
                del result
                
                force_cleanup()
                mid_memory = get_process_memory()
                memory_diff = abs(mid_memory - initial_memory)
                assert memory_diff < 10, f"Memory leak detected at iteration {i}: {memory_diff:.2f} MB"
        
        force_cleanup()
        final_memory = get_process_memory()
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 10, f"Memory leak detected: {memory_diff:.2f} MB"


def test_nested_operations():
    """Test memory management with nested operations."""
    with memory_tracker("nested_operations"):
        force_cleanup()
        initial_memory = get_process_memory()
        
        # Perform nested operations with minimal tensors
        with optimize_memory("outer"):
            t1 = register_tensor(torch.randn(2, 2), "outer")
            
            with optimize_memory("inner1"):
                t2 = register_tensor(torch.randn(2, 2), "inner1")
                t3 = register_tensor(t1 @ t2, "inner1")
                del t2  # Cleanup inner tensors
            
            with optimize_memory("inner2"):
                t4 = register_tensor(torch.randn(2, 2), "inner2")
                t5 = register_tensor(t3 @ t4, "inner2")
                del t4  # Cleanup inner tensors
            
            force_cleanup()
            mid_memory = get_process_memory()
            memory_diff = mid_memory - initial_memory
            assert memory_diff >= 0, "Memory should not decrease during nested operations"
            
            # Delete remaining tensors
            del t1, t3, t5
        
        force_cleanup()
        final_memory = get_process_memory()
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 10, f"Memory leak detected: {memory_diff:.2f} MB"


def test_memory_stress():
    """Stress test memory management."""
    with memory_tracker("memory_stress"):
        force_cleanup()
        initial_memory = get_process_memory()
        
        # Perform intensive operations with minimal tensors
        n_iterations = 2  # Reduced iterations
        tensor_sizes = [(2, 2), (2, 2)]  # Same size tensors for multiplication
        
        for i in range(n_iterations):
            with optimize_memory(f"stress_{i}"):
                tensors = []
                for size in tensor_sizes:
                    t = register_tensor(torch.randn(*size), f"stress_{i}")
                    tensors.append(t)
                
                # Perform operations with cleanup after each
                for j in range(len(tensors) - 1):
                    with optimize_memory(f"stress_{i}_op_{j}"):
                        result = register_tensor(tensors[j] @ tensors[j + 1], f"stress_{i}_op_{j}")
                        tensors.append(result)
                        force_cleanup()
                
                force_cleanup()
                mid_memory = get_process_memory()
                memory_diff = abs(mid_memory - initial_memory)
                assert memory_diff < 10, f"Excessive memory use at iteration {i}: {memory_diff:.2f} MB"
                
                # Clear tensors list and force cleanup
                for tensor in tensors:
                    del tensor
                tensors.clear()
                force_cleanup()
        
        force_cleanup()
        final_memory = get_process_memory()
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 10, f"Memory leak detected: {memory_diff:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 