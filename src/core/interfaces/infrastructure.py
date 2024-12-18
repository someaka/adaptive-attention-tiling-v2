"""Infrastructure Interfaces.

This module defines interfaces for system infrastructure:
1. Data Management - Data loading, saving, and processing
2. Computation Management - Computation scheduling and resource management
3. Monitoring - System monitoring and logging
"""

from typing import Protocol, TypeVar, Dict, List, Optional, Any, Union
from typing_extensions import runtime_checkable
import torch
from pathlib import Path
from dataclasses import dataclass

T = TypeVar('T', bound=torch.Tensor)
PathLike = Union[str, Path]

@dataclass
class DataConfig:
    """Data configuration container."""
    source_path: PathLike
    batch_size: int
    shuffle: bool
    preprocessing: List[str]
    augmentation: Optional[Dict[str, Any]] = None

@dataclass
class ComputeConfig:
    """Computation configuration container."""
    device: str
    precision: str
    num_workers: int
    distributed: bool
    optimization_level: int

@dataclass
class MonitoringConfig:
    """Monitoring configuration container."""
    log_dir: PathLike
    metrics: List[str]
    save_frequency: int
    profile: bool

@runtime_checkable
class IDataManager(Protocol[T]):
    """Data management interface."""
    
    def load_dataset(self, config: DataConfig) -> T:
        """Load dataset from source.
        
        Args:
            config: Data configuration
            
        Returns:
            Loaded dataset
        """
        ...
    
    def save_data(self, data: T, path: PathLike) -> None:
        """Save data to disk.
        
        Args:
            data: Data to save
            path: Save path
        """
        ...
    
    def preprocess_data(self, data: T, steps: List[str]) -> T:
        """Preprocess data.
        
        Args:
            data: Raw data
            steps: Preprocessing steps
            
        Returns:
            Preprocessed data
        """
        ...
    
    def augment_data(self, data: T, config: Dict[str, Any]) -> T:
        """Augment data.
        
        Args:
            data: Input data
            config: Augmentation configuration
            
        Returns:
            Augmented data
        """
        ...
    
    def create_dataloader(self, 
                         data: T,
                         batch_size: int,
                         shuffle: bool) -> Any:
        """Create data loader.
        
        Args:
            data: Input data
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            Data loader
        """
        ...

@runtime_checkable
class IComputeManager(Protocol):
    """Computation management interface."""
    
    def initialize_compute(self, config: ComputeConfig) -> None:
        """Initialize compute resources.
        
        Args:
            config: Compute configuration
        """
        ...
    
    def schedule_computation(self, 
                           function: Any,
                           inputs: Dict[str, Any],
                           priority: Optional[int] = None) -> Any:
        """Schedule computation.
        
        Args:
            function: Function to execute
            inputs: Function inputs
            priority: Execution priority
            
        Returns:
            Computation handle
        """
        ...
    
    def allocate_resources(self, 
                          requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate compute resources.
        
        Args:
            requirements: Resource requirements
            
        Returns:
            Allocated resources
        """
        ...
    
    def release_resources(self, resources: Dict[str, Any]) -> None:
        """Release compute resources.
        
        Args:
            resources: Resources to release
        """
        ...
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor resource usage.
        
        Returns:
            Resource usage metrics
        """
        ...

@runtime_checkable
class ISystemMonitor(Protocol):
    """System monitoring interface."""
    
    def initialize_monitoring(self, config: MonitoringConfig) -> None:
        """Initialize monitoring.
        
        Args:
            config: Monitoring configuration
        """
        ...
    
    def log_metric(self, 
                  name: str,
                  value: float,
                  step: Optional[int] = None) -> None:
        """Log metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        ...
    
    def log_artifacts(self, 
                     artifacts: Dict[str, Any],
                     artifact_type: str) -> None:
        """Log artifacts.
        
        Args:
            artifacts: Artifacts to log
            artifact_type: Type of artifacts
        """
        ...
    
    def start_profiling(self) -> None:
        """Start system profiling."""
        ...
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop system profiling.
        
        Returns:
            Profiling results
        """
        ...
    
    def system_metrics(self) -> Dict[str, float]:
        """Get system metrics.
        
        Returns:
            System metrics
        """
        ...
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate monitoring report.
        
        Returns:
            Monitoring report
        """
        ... 