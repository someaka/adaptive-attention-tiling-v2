"""Benchmark metrics for performance evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import torch


@dataclass
class BenchmarkMetrics:
    """Collection of benchmark metrics."""
    
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
        if torch.cuda.is_available():
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
    def compute_throughput(self):
        """Compute throughput metrics."""
        if self.total_time > 0:
            self.throughput = self.batch_size / self.total_time
            
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary.
        
        Returns:
            Dictionary of metrics
        """
        return {
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
        }
