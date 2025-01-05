"""Minimal standalone memory management test."""

import gc
import torch
import logging
from contextlib import contextmanager

from src.utils.memory_management_util import (
    register_tensor,
    optimize_memory,
    clear_memory
)

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def test_scope(name: str):
    """Simple test scope context manager."""
    logger.info(f"Starting test: {name}")
    clear_memory()
    try:
        yield
    finally:
        clear_memory()
        logger.info(f"Finished test: {name}")

def test_basic_tensor():
    """Test basic tensor operations."""
    with test_scope("basic_tensor"):
        # Create minimal tensor
        with optimize_memory("test"):
            t = register_tensor(torch.randn(2, 2), "test")
            logger.info(f"Created tensor: {t}")
            del t
        
        # Force cleanup
        clear_memory()

def test_matrix_multiply():
    """Test matrix multiplication."""
    with test_scope("matrix_multiply"):
        # Create minimal tensors
        with optimize_memory("test"):
            t1 = register_tensor(torch.randn(2, 2), "test")
            t2 = register_tensor(torch.randn(2, 2), "test")
            result = register_tensor(t1 @ t2, "test")
            logger.info(f"Matrix multiply result: {result}")
            del t1, t2, result
        
        # Force cleanup
        clear_memory()

def main():
    """Run all tests."""
    # Initial cleanup
    clear_memory()
    gc.collect()
    
    try:
        # Run tests
        test_basic_tensor()
        test_matrix_multiply()
        
        logger.info("All tests passed!")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 