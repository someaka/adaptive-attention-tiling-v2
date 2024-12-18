"""Performance Interfaces.

This module defines interfaces for performance monitoring and optimization:
1. Performance Metrics - System performance measurements
2. Resource Management - Resource allocation and monitoring
3. Optimization - Performance optimization strategies
"""

from typing import Protocol, TypeVar, Dict, List, Optional, Any, Tuple, TypeVarTuple, Unpack
from typing_extensions import runtime_checkable
import torch
from dataclasses import dataclass
from pathlib import Path

from .metrics import MetricResult

# Invariant type variable for protocols that use T in parameter positions
T = TypeVar('T', bound=torch.Tensor)
# Covariant type variable for protocols that only return T
T_co = TypeVar('T_co', bound=torch.Tensor, covariant=True)
Ts = TypeVarTuple('Ts')

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    disk_io: Optional[Dict[str, float]] = None
    network_io: Optional[Dict[str, float]] = None

@dataclass
class PerformanceProfile:
    """Performance profile container."""
    operation_name: str
    execution_time: float
    resource_usage: ResourceMetrics
    throughput: float
    efficiency: float
    bottlenecks: List[str]
    optimization_suggestions: List[str]

@runtime_checkable
class IPerformanceMonitor(Protocol[T_co]):
    """Performance monitoring interface."""
    
    def start_monitoring(self, operation_name: str) -> None:
        """Start monitoring performance.
        
        Args:
            operation_name: Name of operation to monitor
        """
        ...
    
    def stop_monitoring(self) -> PerformanceProfile:
        """Stop monitoring and get profile.
        
        Returns:
            Performance profile
        """
        ...
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource metrics.
        
        Returns:
            Resource metrics
        """
        ...
    
    def profile_operation(self, 
                         operation: Any,
                         *args: Unpack[Ts],
                         **kwargs: Any) -> Tuple[Any, PerformanceProfile]:
        """Profile operation execution.
        
        Args:
            operation: Operation to profile
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result and performance profile
        """
        ...

@runtime_checkable
class IResourceManager(Protocol):
    """Resource management interface."""
    
    def allocate_resources(self, requirements: Dict[str, float]) -> bool:
        """Attempt to allocate resources.
        
        Args:
            requirements: Resource requirements
            
        Returns:
            Whether allocation succeeded
        """
        ...
    
    def release_resources(self, resource_id: str) -> None:
        """Release allocated resources.
        
        Args:
            resource_id: Resource allocation ID
        """
        ...
    
    def get_available_resources(self) -> Dict[str, float]:
        """Get available resource amounts.
        
        Returns:
            Available resources
        """
        ...
    
    def optimize_allocation(self, 
                          current: Dict[str, float],
                          target: Dict[str, float]) -> Dict[str, float]:
        """Optimize resource allocation.
        
        Args:
            current: Current allocation
            target: Target allocation
            
        Returns:
            Optimized allocation
        """
        ...

@runtime_checkable
class IPerformanceOptimizer(Protocol[T_co]):
    """Performance optimization interface."""
    
    def analyze_bottlenecks(self, profile: PerformanceProfile) -> List[str]:
        """Analyze performance bottlenecks.
        
        Args:
            profile: Performance profile
            
        Returns:
            List of bottlenecks
        """
        ...
    
    def suggest_optimizations(self, 
                            profile: PerformanceProfile) -> List[Dict[str, Any]]:
        """Suggest performance optimizations.
        
        Args:
            profile: Performance profile
            
        Returns:
            List of optimization suggestions
        """
        ...
    
    def optimize_operation(self,
                         operation: Any,
                         *args: Unpack[Ts],
                         **kwargs: Any) -> Any:
        """Optimize operation execution.
        
        Args:
            operation: Operation to optimize
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Optimized operation
        """
        ...
    
    def auto_tune(self,
                 operation: Any,
                 search_space: Dict[str, List[Any]],
                 metric: str) -> Dict[str, Any]:
        """Auto-tune operation parameters.
        
        Args:
            operation: Operation to tune
            search_space: Parameter search space
            metric: Optimization metric
            
        Returns:
            Optimal parameters
        """
        ...

@runtime_checkable
class IPerformanceManager(Protocol[T]):
    """Combined performance management interface."""
    
    monitor: IPerformanceMonitor[T]
    resources: IResourceManager
    optimizer: IPerformanceOptimizer[T]
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize performance management.
        
        Args:
            config: Configuration dictionary
        """
        ...
    
    def optimize_pipeline(self,
                        operations: List[Any],
                        data: T) -> Tuple[T, List[PerformanceProfile]]:
        """Optimize pipeline execution.
        
        Args:
            operations: Pipeline operations
            data: Input data
            
        Returns:
            Output data and performance profiles
        """
        ...
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report.
        
        Returns:
            Performance report
        """
        ... 