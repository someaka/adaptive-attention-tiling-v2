"""Infrastructure metrics implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    
    cpu_utilization: float  # CPU utilization percentage
    memory_used: int  # Memory used in bytes
    memory_available: int  # Memory available in bytes
    disk_used: int  # Disk space used in bytes
    disk_available: int  # Disk space available in bytes
    network_sent: int  # Network bytes sent
    network_received: int  # Network bytes received
    
    def get_summary(self) -> str:
        """Get human-readable summary of metrics.
        
        Returns:
            Summary string
        """
        summary = [
            "Resource Metrics:",
            f"  CPU: {self.cpu_utilization:.1f}%",
            f"  Memory: {self.memory_used / 1024**2:.1f}MB used, {self.memory_available / 1024**2:.1f}MB available",
            f"  Disk: {self.disk_used / 1024**2:.1f}MB used, {self.disk_available / 1024**2:.1f}MB available", 
            f"  Network: {self.network_sent / 1024:.1f}KB sent, {self.network_received / 1024:.1f}KB received"
        ]
        return "\n".join(summary)


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""
    
    compute_time: float  # Computation time in seconds
    memory_time: float  # Memory transfer time in seconds
    sync_time: float  # Synchronization time in seconds
    total_time: float  # Total execution time in seconds


@dataclass
class InfrastructureMetrics:
    """Complete infrastructure metrics."""
    
    resources: ResourceMetrics  # Resource utilization
    performance: PerformanceMetrics  # Performance metrics
    device_info: Dict[str, str]  # Device information
    error_log: List[str]  # Error messages
    
    def __post_init__(self):
        """Validate metrics."""
        if self.resources.memory_used > self.resources.memory_available:
            self.error_log.append(
                f"Memory usage ({self.resources.memory_used}) exceeds available memory "
                f"({self.resources.memory_available})"
            )
            
        total_time = (
            self.performance.compute_time +
            self.performance.memory_time +
            self.performance.sync_time
        )
        if not torch.allclose(
            torch.tensor(total_time),
            torch.tensor(self.performance.total_time),
            rtol=1e-5
        ):
            self.error_log.append(
                f"Total time mismatch: sum of components ({total_time}) != "
                f"total ({self.performance.total_time})"
            )
            
    def get_summary(self) -> str:
        """Get human-readable summary of metrics."""
        summary = []
        
        # Resource utilization
        summary.append("Resource Utilization:")
        summary.append(f"  Memory: {self.resources.memory_used/1e6:.1f}MB / "
                      f"{self.resources.memory_available/1e6:.1f}MB")
        if self.resources.cpu_utilization is not None:
            summary.append(f"  CPU: {self.resources.cpu_utilization:.1f}%")
            
        # Performance timing
        summary.append("\nPerformance Timing:")
        summary.append(f"  Compute: {self.performance.compute_time*1000:.1f}ms")
        summary.append(f"  Memory: {self.performance.memory_time*1000:.1f}ms")
        summary.append(f"  Sync: {self.performance.sync_time*1000:.1f}ms")
        summary.append(f"  Total: {self.performance.total_time*1000:.1f}ms")
        
        # Device info
        summary.append("\nDevice Information:")
        for key, value in self.device_info.items():
            summary.append(f"  {key}: {value}")
            
        # Errors if any
        if self.error_log:
            summary.append("\nErrors:")
            for error in self.error_log:
                summary.append(f"  - {error}")
                
        return "\n".join(summary)
