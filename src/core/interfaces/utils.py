"""Utility Interfaces.

This module defines utility interfaces:
1. Configuration - System configuration management
2. Serialization - Data serialization and deserialization
3. Visualization - Data and pattern visualization
"""

from typing import Protocol, TypeVar, Dict, List, Optional, Any, Union
from typing_extensions import runtime_checkable
import torch
from pathlib import Path
from dataclasses import dataclass

from .pattern_space import IFiberBundle
from .neural_pattern import IPatternNetwork
from .quantum import IQuantumState
from .crystal import ICrystal

T = TypeVar('T', bound=torch.Tensor)
PathLike = Union[str, Path]

@dataclass
class ConfigurationSpec:
    """Configuration specification."""
    name: str
    type: str
    default: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

@runtime_checkable
class IConfigurationManager(Protocol):
    """Configuration management interface."""
    
    def load_config(self, path: PathLike) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            path: Configuration file path
            
        Returns:
            Configuration dictionary
        """
        ...
    
    def save_config(self, config: Dict[str, Any], path: PathLike) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Save path
        """
        ...
    
    def validate_config(self, 
                       config: Dict[str, Any],
                       specs: Dict[str, ConfigurationSpec]) -> List[str]:
        """Validate configuration against specs.
        
        Args:
            config: Configuration to validate
            specs: Configuration specifications
            
        Returns:
            List of validation errors
        """
        ...
    
    def merge_configs(self,
                     base: Dict[str, Any],
                     override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        ...

@runtime_checkable
class ISerializationManager(Protocol[T]):
    """Data serialization interface."""
    
    def serialize_tensor(self, tensor: T) -> bytes:
        """Serialize tensor to bytes.
        
        Args:
            tensor: Tensor to serialize
            
        Returns:
            Serialized bytes
        """
        ...
    
    def deserialize_tensor(self, data: bytes) -> T:
        """Deserialize tensor from bytes.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized tensor
        """
        ...
    
    def save_checkpoint(self,
                       network: IPatternNetwork[T],
                       path: PathLike) -> None:
        """Save network checkpoint.
        
        Args:
            network: Network to save
            path: Save path
        """
        ...
    
    def load_checkpoint(self,
                       path: PathLike) -> IPatternNetwork[T]:
        """Load network checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Loaded network
        """
        ...
    
    def export_model(self,
                    network: IPatternNetwork[T],
                    format: str,
                    path: PathLike) -> None:
        """Export model to specified format.
        
        Args:
            network: Network to export
            format: Export format
            path: Export path
        """
        ...

@runtime_checkable
class IVisualizationManager(Protocol[T]):
    """Data visualization interface."""
    
    def visualize_pattern(self,
                         pattern: T,
                         save_path: Optional[PathLike] = None) -> None:
        """Visualize pattern.
        
        Args:
            pattern: Pattern to visualize
            save_path: Optional save path
        """
        ...
    
    def visualize_attention(self,
                          attention_weights: T,
                          save_path: Optional[PathLike] = None) -> None:
        """Visualize attention weights.
        
        Args:
            attention_weights: Attention weights
            save_path: Optional save path
        """
        ...
    
    def visualize_geometry(self,
                          bundle: IFiberBundle[T],
                          save_path: Optional[PathLike] = None) -> None:
        """Visualize geometric structure.
        
        Args:
            bundle: Fiber bundle
            save_path: Optional save path
        """
        ...
    
    def visualize_quantum_state(self,
                              state: IQuantumState[T],
                              save_path: Optional[PathLike] = None) -> None:
        """Visualize quantum state.
        
        Args:
            state: Quantum state
            save_path: Optional save path
        """
        ...
    
    def visualize_crystal(self,
                         crystal: ICrystal[T],
                         save_path: Optional[PathLike] = None) -> None:
        """Visualize crystal structure.
        
        Args:
            crystal: Crystal structure
            save_path: Optional save path
        """
        ...
    
    def plot_metrics(self,
                    metrics: Dict[str, List[float]],
                    save_path: Optional[PathLike] = None) -> None:
        """Plot metrics history.
        
        Args:
            metrics: Metrics history
            save_path: Optional save path
        """
        ...
    
    def generate_report(self,
                       data: Dict[str, Any],
                       template: str,
                       save_path: PathLike) -> None:
        """Generate visualization report.
        
        Args:
            data: Report data
            template: Report template
            save_path: Save path
        """
        ... 