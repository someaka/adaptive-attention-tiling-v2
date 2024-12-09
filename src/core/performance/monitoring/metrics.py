"""Performance Metrics Collection Framework.

This module provides tools for collecting and analyzing performance metrics
across CPU operations, memory usage, and system resources.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import psutil


@dataclass
class PerformanceMetrics:
    """Container for various performance metrics."""

    execution_time: float
    cpu_usage: float
    memory_usage: float
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    operation_type: str = "unknown"
    timestamp: float = time.time()


class MetricsCollector:
    """Collects and analyzes performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.process = psutil.Process()

    def record_metric(
        self,
        operation_type: str,
        execution_time: float,
        cache_stats: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record performance metrics for an operation."""
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()

        metric = PerformanceMetrics(
            execution_time=execution_time,
            cpu_usage=cpu_percent,
            memory_usage=memory_info.rss / 1024**2,  # Convert to MB
            cache_hits=cache_stats.get("hits") if cache_stats else None,
            cache_misses=cache_stats.get("misses") if cache_stats else None,
            operation_type=operation_type,
        )

        self.metrics[operation_type].append(metric)

    def get_metrics(
        self,
        operation_type: Optional[str] = None,
        time_range: Optional[tuple[float, float]] = None,
    ) -> List[PerformanceMetrics]:
        """Get metrics, optionally filtered by operation type and time range."""
        metrics = (
            self.metrics[operation_type]
            if operation_type
            else [m for metrics in self.metrics.values() for m in metrics]
        )

        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics if start_time <= m.timestamp <= end_time]

        return metrics

    def get_summary(self, operation_type: Optional[str] = None) -> Dict[str, float]:
        """Get summary statistics for metrics."""
        metrics = self.get_metrics(operation_type)
        if not metrics:
            return {}

        execution_times = [m.execution_time for m in metrics]
        cpu_usages = [m.cpu_usage for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]

        return {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "avg_cpu_usage": sum(cpu_usages) / len(cpu_usages),
            "avg_memory_usage": sum(memory_usages) / len(memory_usages),
            "peak_memory_usage": max(memory_usages),
        }

    def get_bottlenecks(self, threshold: float = 0.9) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        for op_type, metrics in self.metrics.items():
            if not metrics:
                continue

            # Check execution time variance
            times = [m.execution_time for m in metrics]
            avg_time = sum(times) / len(times)
            max_time = max(times)

            if max_time > avg_time * (1 + threshold):
                bottlenecks.append(f"{op_type}: High execution time variance")

            # Check memory usage
            memory_usages = [m.memory_usage for m in metrics]
            if max(memory_usages) > psutil.virtual_memory().total * threshold:
                bottlenecks.append(f"{op_type}: High memory usage")

            # Check CPU usage
            cpu_usages = [m.cpu_usage for m in metrics]
            if sum(cpu_usages) / len(cpu_usages) > 90:
                bottlenecks.append(f"{op_type}: High CPU usage")

        return bottlenecks

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()

    def export_metrics(self, format: str = "dict") -> Union[Dict, str]:
        """Export metrics in specified format."""
        if format == "dict":
            return {
                op: [vars(m) for m in metrics] for op, metrics in self.metrics.items()
            }
        if format == "csv":
            # Create CSV string
            headers = [
                "operation_type",
                "execution_time",
                "cpu_usage",
                "memory_usage",
                "cache_hits",
                "cache_misses",
                "timestamp",
            ]
            rows = [",".join(headers)]

            for metrics in self.get_metrics():
                row = [
                    metrics.operation_type,
                    str(metrics.execution_time),
                    str(metrics.cpu_usage),
                    str(metrics.memory_usage),
                    str(metrics.cache_hits or ""),
                    str(metrics.cache_misses or ""),
                    str(metrics.timestamp),
                ]
                rows.append(",".join(row))

            return "\n".join(rows)
        raise ValueError(f"Unsupported format: {format}")
