"""Shared memory management utilities for scale classes."""

from contextlib import contextmanager
from src.core.performance.cpu.memory_management import MemoryManager
from src.utils.memory_management import optimize_memory

# Global memory manager instance
memory_manager = MemoryManager()

@contextmanager
def memory_efficient_computation(operation: str):
    """Context manager for memory-efficient computations."""
    with optimize_memory(operation):
        yield 