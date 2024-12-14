"""Benchmark metrics for performance evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import time
import torch

from ..performance.vulkan.memory.memory_pool import MemoryPoolManager


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    name: str
    size: int
    avg_time: float
    avg_memory: float = 0.0
    throughput: float = 0.0
    batch_size: Optional[int] = None
    stability: Optional[float] = None
    accuracy: Optional[float] = None
    efficiency: Optional[float] = None
    sequential_time: Optional[float] = None
    batch_time: Optional[float] = None
    convergence_rate: Optional[float] = None


@dataclass
class BenchmarkMetrics:
    """Collection of benchmark metrics."""
    
    # Operation metrics
    operations: List[OperationMetrics] = field(default_factory=list)
    
    # Timing metrics
    forward_time: float = 0.0
    backward_time: float = 0.0
    total_time: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0  # items/second
    flops: float = 0.0  # floating point operations
    efficiency: float = 0.0  # % of peak performance
    
    # Additional metrics
    num_parameters: int = 0
    batch_size: int = 0
    device: str = 'cpu'

    # Vulkan memory tracking
    memory_pool: Optional[MemoryPoolManager] = None
    
    def record_operation(self, name: str, **kwargs):
        """Record metrics for an operation.
        
        Args:
            name: Operation name
            **kwargs: Operation metrics
        """
        metrics = OperationMetrics(name=name, **kwargs)
        self.operations.append(metrics)
        
    @classmethod
    def from_model(cls, model: torch.nn.Module, input_size: tuple) -> 'BenchmarkMetrics':
        """Create metrics from model.
        
        Args:
            model: Model to benchmark
            input_size: Input tensor size
            
        Returns:
            Benchmark metrics
        """
        metrics = cls()
        
        # Count parameters
        metrics.num_parameters = sum(p.numel() for p in model.parameters())
        
        # Get device
        metrics.device = next(model.parameters()).device.type
        
        # Get batch size
        metrics.batch_size = input_size[0]
        
        return metrics
        
    def update_timing(self, forward_time: float, backward_time: float):
        """Update timing metrics.
        
        Args:
            forward_time: Forward pass time
            backward_time: Backward pass time
        """
        self.forward_time = forward_time
        self.backward_time = backward_time
        self.total_time = forward_time + backward_time
        
    def update_memory(self):
        """Update memory metrics."""
        if self.memory_pool is not None:
            # Get total allocated memory across all pools
            total_allocated = 0
            peak_allocated = 0
            
            for pool in self.memory_pool.pools.values():
                total_allocated += pool.used_size
                peak_allocated = max(peak_allocated, pool.total_size)
            
            self.memory_allocated_mb = total_allocated / (1024 * 1024)  # Convert to MB
            self.peak_memory_mb = peak_allocated / (1024 * 1024)  # Convert to MB
            
    def compute_throughput(self):
        """Compute throughput metrics."""
        if self.total_time > 0:
            self.throughput = self.batch_size / self.total_time
            
    def to_dict(self) -> Dict[str, Union[float, int, Dict[str, Any]]]:
        """Convert metrics to dictionary.
        
        Returns:
            Dictionary of metrics
        """
        metrics_dict = {
            'forward_time': self.forward_time,
            'backward_time': self.backward_time,
            'total_time': self.total_time,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_allocated_mb': self.memory_allocated_mb,
            'throughput': self.throughput,
            'flops': self.flops,
            'efficiency': self.efficiency,
            'num_parameters': self.num_parameters,
            'batch_size': self.batch_size,
            'operations': {}
        }
        
        # Add operation metrics
        operations_dict = {}
        for op in self.operations:
            op_dict = {k: v for k, v in vars(op).items() if v is not None}
            operations_dict[f"{op.name}_{op.size}"] = op_dict
        metrics_dict['operations'] = operations_dict
            
        return metrics_dict
