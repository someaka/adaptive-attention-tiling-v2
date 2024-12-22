"""Benchmark metrics collection and analysis utilities."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Optional
from statistics import mean, stdev
import json
from pathlib import Path


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    name: str
    times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    additional_metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_measurement(self, **kwargs):
        """Add a measurement to the metrics."""
        if 'time' in kwargs:
            self.times.append(float(kwargs['time']))
        if 'memory' in kwargs:
            self.memory_usage.append(float(kwargs['memory']))
        if 'size' in kwargs:
            self.sizes.append(int(kwargs['size']))
        if 'batch_size' in kwargs:
            self.batch_sizes.append(int(kwargs['batch_size']))
            
        # Store any additional metrics
        for key, value in kwargs.items():
            if key not in ['time', 'memory', 'size', 'batch_size']:
                if key not in self.additional_metrics:
                    self.additional_metrics[key] = []
                self.additional_metrics[key].append(float(value))

    def get_summary(self) -> Dict[str, Union[str, int, float]]:
        """Get summary statistics."""
        summary: Dict[str, Union[str, int, float]] = {
            'name': self.name,
            'measurements': len(self.times)
        }
        
        if self.times:
            summary.update({
                'avg_time': float(mean(self.times)),
                'std_time': float(stdev(self.times) if len(self.times) > 1 else 0),
                'min_time': float(min(self.times)),
                'max_time': float(max(self.times))
            })
            
        if self.memory_usage:
            summary.update({
                'avg_memory': float(mean(self.memory_usage)),
                'peak_memory': float(max(self.memory_usage))
            })
            
        if self.sizes:
            summary.update({
                'min_size': int(min(self.sizes)),
                'max_size': int(max(self.sizes))
            })
            
        if self.batch_sizes:
            summary.update({
                'min_batch': int(min(self.batch_sizes)),
                'max_batch': int(max(self.batch_sizes))
            })
            
        # Add summaries for additional metrics
        for key, values in self.additional_metrics.items():
            if values:
                summary[f'avg_{key}'] = float(mean(values))
                if len(values) > 1:
                    summary[f'std_{key}'] = float(stdev(values))
                summary[f'min_{key}'] = float(min(values))
                summary[f'max_{key}'] = float(max(values))
                
        return summary


class BenchmarkMetrics:
    """Collection of benchmark metrics."""
    
    def __init__(self):
        """Initialize benchmark metrics collection."""
        self.operations: Dict[str, OperationMetrics] = {}
        
    def record_operation(self, name: str, **kwargs):
        """Record metrics for an operation.
        
        Args:
            name: Name of the operation
            **kwargs: Metrics to record (time, memory, size, etc.)
        """
        if name not in self.operations:
            self.operations[name] = OperationMetrics(name=name)
        self.operations[name].add_measurement(**kwargs)
        
    def get_summary(self) -> Dict[str, Dict[str, Union[str, int, float]]]:
        """Get summary of all metrics."""
        return {
            name: metrics.get_summary()
            for name, metrics in self.operations.items()
        }
        
    def save_to_file(self, filepath: str):
        """Save metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics
        """
        summary = self.get_summary()
        Path(filepath).write_text(json.dumps(summary, indent=2))
        
    def print_summary(self):
        """Print human-readable summary of metrics."""
        summary = self.get_summary()
        for op_name, op_metrics in summary.items():
            print(f"\n{op_name}:")
            for metric, value in op_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}") 