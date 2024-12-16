"""Memory management utilities for tensor operations.

This module provides utilities for:
1. Tensor lifecycle management
2. Memory tracking and optimization
3. Resource cleanup
4. Safe tensor operations
"""

import gc
import weakref
import logging
import warnings
from typing import Optional, Dict, Set, Any
from contextlib import contextmanager

import torch

# Filter PyTorch internal deprecation warnings
# These warnings come from PyTorch's internal type checking mechanisms when using isinstance() with torch.Tensor
# The warnings suggest using torch.is_tensor(), but this doesn't fully solve the issue since PyTorch's own
# internal code still uses the deprecated behavior. These warnings are scheduled to be addressed in future
# PyTorch versions, but for now we filter them to avoid noise in the logs.
warnings.filterwarnings('ignore', category=UserWarning, message='isinstance\\(x, torch\\.Tensor\\)')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_tensor(obj: Any) -> bool:
    """Safe tensor type check using PyTorch's recommended method."""
    try:
        return torch.is_tensor(obj)
    except Exception:
        return False


class TensorManager:
    """Manages tensor lifecycle and memory cleanup."""
    
    def __init__(self):
        self._tensors: Set[int] = set()  # Track tensor ids
        self._tensor_refs: Dict[int, weakref.ref] = {}  # Weak references to tensors
        self._operation_tensors: Dict[str, Set[int]] = {}  # Tensors by operation
        self._enabled = True
        self._max_tensor_size = 1024 * 1024  # 1MB default max tensor size
    
    def set_max_tensor_size(self, size_bytes: int) -> None:
        """Set maximum allowed tensor size in bytes."""
        self._max_tensor_size = size_bytes
    
    def register_tensor(self, tensor: torch.Tensor, operation: Optional[str] = None) -> torch.Tensor:
        """Register a tensor for lifecycle management."""
        try:
            if not self._enabled:
                return tensor
            
            # Check tensor size
            tensor_size = tensor.element_size() * tensor.nelement()
            if tensor_size > self._max_tensor_size:
                logger.warning(
                    f"Tensor size ({tensor_size} bytes) exceeds maximum allowed size "
                    f"({self._max_tensor_size} bytes). Consider using smaller tensors."
                )
            
            tensor_id = id(tensor)
            self._tensors.add(tensor_id)
            self._tensor_refs[tensor_id] = weakref.ref(tensor, lambda _: self._cleanup_tensor(tensor_id))
            
            if operation:
                if operation not in self._operation_tensors:
                    self._operation_tensors[operation] = set()
                self._operation_tensors[operation].add(tensor_id)
            
            return tensor
            
        except (RuntimeError, MemoryError) as e:
            logger.error(f"Failed to register tensor: {e}")
            # Try to free memory
            self.clear_all()
            raise
    
    def _cleanup_tensor(self, tensor_id: int) -> None:
        """Clean up a tensor when it's no longer needed."""
        try:
            self._tensors.discard(tensor_id)
            self._tensor_refs.pop(tensor_id, None)
            
            for op_tensors in self._operation_tensors.values():
                op_tensors.discard(tensor_id)
                
        except Exception as e:
            logger.error(f"Error during tensor cleanup: {e}")
    
    def cleanup_operation(self, operation: str) -> None:
        """Clean up all tensors associated with an operation."""
        try:
            if operation in self._operation_tensors:
                tensor_ids = self._operation_tensors[operation].copy()
                for tensor_id in tensor_ids:
                    ref = self._tensor_refs.get(tensor_id)
                    if ref is not None:
                        tensor = ref()
                        if tensor is not None and _is_tensor(tensor):
                            del tensor
                self._operation_tensors[operation].clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Error during operation cleanup: {e}")
    
    @contextmanager
    def operation_scope(self, operation: str):
        """Context manager for tensor operations."""
        try:
            yield
        except Exception as e:
            logger.error(f"Error in operation scope: {e}")
            raise
        finally:
            self.cleanup_operation(operation)
    
    @contextmanager
    def disable(self):
        """Temporarily disable tensor management."""
        prev_state = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = prev_state
    
    def clear_all(self) -> None:
        """Clear all managed tensors."""
        try:
            operations = list(self._operation_tensors.keys())
            for operation in operations:
                self.cleanup_operation(operation)
            self._tensors.clear()
            self._tensor_refs.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during complete cleanup: {e}")


class MemoryOptimizer:
    """Optimizes memory usage for tensor operations."""
    
    def __init__(self):
        self.tensor_manager = TensorManager()
        self._max_memory_usage = 1024 * 1024 * 1024  # 1GB default max memory
    
    def set_max_memory_usage(self, size_bytes: int) -> None:
        """Set maximum allowed memory usage in bytes."""
        self._max_memory_usage = size_bytes
    
    @contextmanager
    def optimize(self, operation: str):
        """Context manager for optimized tensor operations."""
        try:
            # Check current memory usage
            current_memory = self.get_memory_stats()['allocated']
            if current_memory > self._max_memory_usage:
                logger.warning(
                    f"Memory usage ({current_memory} bytes) exceeds maximum allowed "
                    f"({self._max_memory_usage} bytes). Attempting cleanup."
                )
                self.clear_memory()
            
            with self.tensor_manager.operation_scope(operation):
                yield
                
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
            self.clear_memory()
            raise
    
    def clear_memory(self) -> None:
        """Clear all cached memory."""
        try:
            self.tensor_manager.clear_all()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during memory clearing: {e}")
    
    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            stats = {
                'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'allocated': 0, 'cached': 0}


# Global instances with reasonable limits for 16GB system
tensor_manager = TensorManager()
tensor_manager.set_max_tensor_size(100 * 1024 * 1024)  # 100MB max tensor size

memory_optimizer = MemoryOptimizer()
memory_optimizer.set_max_memory_usage(8 * 1024 * 1024 * 1024)  # 8GB max memory usage (half of system memory)

# Debug mode for fine-grained control when needed
DEBUG_MODE = False  # Disable aggressive cleanup by default

def register_tensor(tensor: torch.Tensor, operation: Optional[str] = None) -> torch.Tensor:
    """Register a tensor with the global tensor manager."""
    if DEBUG_MODE:
        # Force cleanup before registration in debug mode
        clear_memory()
    return tensor_manager.register_tensor(tensor, operation)


@contextmanager
def optimize_memory(operation: str):
    """Context manager for optimized memory usage."""
    if DEBUG_MODE:
        # Force cleanup before operation in debug mode
        clear_memory()
    with memory_optimizer.optimize(operation):
        try:
            yield
        finally:
            if DEBUG_MODE:
                # Force cleanup after operation in debug mode
                clear_memory()


def clear_memory() -> None:
    """Clear all cached memory."""
    memory_optimizer.clear_memory()
    if DEBUG_MODE:
        # Force garbage collection in debug mode
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Additional cleanup
        for obj in gc.get_objects():
            try:
                if _is_tensor(obj):  # Use safer tensor check
                    del obj
            except Exception:
                continue
        gc.collect()


def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics."""
    return memory_optimizer.get_memory_stats() 

def force_cleanup():
    """Force aggressive memory cleanup."""
    try:
        # Multiple rounds of cleanup
        for _ in range(3):
            clear_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Additional cleanup
            for obj in gc.get_objects():
                try:
                    # Use type comparison instead of isinstance to avoid the deprecation warning
                    if type(obj).__module__ == 'torch' and type(obj).__name__ == 'Tensor':
                        del obj
                except Exception:
                    continue
            gc.collect()
        
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")
  