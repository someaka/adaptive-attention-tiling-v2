"""Base memory management interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    
    allocated_memory: int  # Current allocated memory in bytes
    peak_memory: int  # Peak memory usage in bytes
    fragmentation_ratio: float  # Memory fragmentation ratio
    operation_type: str  # Type of operation (allocate, free, transfer)


class MemoryError(Exception):
    """Base exception for memory management errors."""
    pass


class MemoryManagerBase(ABC):
    """Abstract base class for memory management."""

    def __init__(self):
        self._allocated_memory = 0
        self._peak_memory = 0
        self._metrics: List[MemoryMetrics] = []

    @abstractmethod
    def allocate_tensor(self, size: Union[Tuple[int, ...], torch.Size], dtype: Any) -> Any:
        """Allocate memory for a tensor.
        
        Args:
            size: Tensor dimensions
            dtype: Data type
            
        Returns:
            Allocated tensor/buffer
        """
        pass

    @abstractmethod
    def free_tensor(self, tensor: Any) -> None:
        """Free tensor memory.
        
        Args:
            tensor: Tensor to free
        """
        pass

    @abstractmethod
    def copy_to_device(self, src: Any, dst: Any) -> None:
        """Copy data to device memory.
        
        Args:
            src: Source data
            dst: Destination buffer/tensor
        """
        pass

    @abstractmethod
    def copy_from_device(self, src: Any, dst: Any) -> None:
        """Copy data from device memory.
        
        Args:
            src: Source buffer/tensor
            dst: Destination data
        """
        pass

    def get_allocated_memory(self) -> int:
        """Get current allocated memory in bytes."""
        return self._allocated_memory

    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self._peak_memory

    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        return 0.0  # Override in implementations

    def get_metrics(self) -> List[MemoryMetrics]:
        """Get memory usage metrics history."""
        return self._metrics

    def record_metric(self, operation_type: str) -> None:
        """Record a memory metric.
        
        Args:
            operation_type: Type of operation
        """
        self._metrics.append(MemoryMetrics(
            allocated_memory=self._allocated_memory,
            peak_memory=self._peak_memory,
            fragmentation_ratio=self.get_fragmentation_ratio(),
            operation_type=operation_type
        ))

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up memory resources."""
        pass 