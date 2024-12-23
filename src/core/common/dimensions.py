"""Dimension management utilities.

This module provides utilities for managing tensor dimensions and transitions
between different dimensional spaces in the neural geometric flow system.

The module includes:
1. DimensionConfig - Configuration for dimension management
2. Custom tensor types (QuantumTensor, GeometricTensor)
3. DimensionManager - Core class for dimension management
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, TypeVar, Type, cast

import torch
import torch.nn as nn
from torch import Tensor
import torch.jit

# Type variables for generic tensor operations
T = TypeVar('T', bound=Tensor)
TensorType = TypeVar('TensorType', bound=Type[Tensor])


@dataclass(frozen=True)
class DimensionConfig:
    """Configuration for dimension management.
    
    This immutable configuration class defines the dimensions for different
    aspects of the neural geometric flow system.
    
    Attributes:
        attention_depth: Depth of attention mechanism
        quantum_dim: Dimension of quantum state space
        geometric_dim: Dimension of geometric manifold
        flow_dim: Dimension of flow space
        emergence_dim: Dimension of emergent patterns
    """
    attention_depth: int
    quantum_dim: int
    geometric_dim: int
    flow_dim: int
    emergence_dim: int
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if any(dim <= 0 for dim in self.__dict__.values()):
            raise ValueError("All dimensions must be positive")
    
    def compute_minimal_dimension(self) -> int:
        """Compute minimal required dimension.
        
        Returns:
            Minimum dimension required across all components
        """
        return max(
            self.attention_depth,
            self.quantum_dim,
            self.geometric_dim,
            self.flow_dim,
            self.emergence_dim
        )
    
    @classmethod
    def from_test_config(cls, test_config: Optional[dict] = None) -> 'DimensionConfig':
        """Create configuration from test config.
        
        Args:
            test_config: Optional test configuration dictionary
            
        Returns:
            DimensionConfig instance with test dimensions
        """
        if test_config is None:
            # Default minimal configuration
            return cls(
                attention_depth=1,
                quantum_dim=2,
                geometric_dim=2,
                flow_dim=2,
                emergence_dim=2
            )
            
        # Get dimensions from test config
        manifold_dim = test_config.get("manifold_dim", 4)
        
        return cls(
            attention_depth=1,
            quantum_dim=manifold_dim,
            geometric_dim=manifold_dim,
            flow_dim=manifold_dim,
            emergence_dim=manifold_dim
        )


class QuantumTensor(Tensor):
    """Tensor subclass for quantum states and operations."""
    
    @staticmethod
    def __new__(cls, tensor: Tensor) -> 'QuantumTensor':
        """Create a new QuantumTensor instance."""
        return tensor.as_subclass(cls)
    
    @classmethod
    def __torch_function__(
        cls,
        func: Type[Tensor],
        types: Tuple[Type[Tensor], ...],
        args: Tuple = (),
        kwargs: Optional[dict] = None
    ) -> Tensor:
        """Override torch function behavior.
        
        Args:
            func: Torch function to override
            types: Tuple of tensor types
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Result of function application
        """
        kwargs = kwargs or {}
        return super().__torch_function__(func, types, args, kwargs)


class GeometricTensor(Tensor):
    """Tensor subclass for geometric operations."""
    
    @staticmethod
    def __new__(cls, tensor: Tensor) -> 'GeometricTensor':
        """Create a new GeometricTensor instance."""
        return tensor.as_subclass(cls)
    
    @classmethod
    def __torch_function__(
        cls,
        func: Type[Tensor],
        types: Tuple[Type[Tensor], ...],
        args: Tuple = (),
        kwargs: Optional[dict] = None
    ) -> Tensor:
        """Override torch function behavior.
        
        Args:
            func: Torch function to override
            types: Tuple of tensor types
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Result of function application
        """
        kwargs = kwargs or {}
        return super().__torch_function__(func, types, args, kwargs)


class DimensionManager(nn.Module):
    """Manages tensor dimensions and transitions.
    
    This class provides utilities for:
    1. Dimension validation
    2. Transitions between spaces
    3. Structure preservation during transitions
    4. Tensor type conversion
    
    Attributes:
        config: Dimension configuration
        min_dim: Minimum dimension across all components
    """
    
    def __init__(self, config: DimensionConfig):
        """Initialize dimension manager.
        
        Args:
            config: Dimension configuration
        """
        super().__init__()
        self.config = config
        self.min_dim = config.compute_minimal_dimension()
        
    def verify_dimension(self, tensor: Tensor) -> bool:
        """Verify tensor has sufficient dimension.
        
        Args:
            tensor: Input tensor to verify
            
        Returns:
            True if dimension is valid
            
        Raises:
            ValueError: If dimension is insufficient
        """
        if tensor.shape[-1] < self.min_dim:
            raise ValueError(
                f"Tensor dimension {tensor.shape[-1]} below minimum {self.min_dim}"
            )
        return True
        
    def project(
        self,
        tensor: Tensor,
        target_dim: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Project tensor to target dimension while preserving structure.
        
        This method uses orthogonal projection to preserve geometric structure
        during dimension changes.
        
        Args:
            tensor: Input tensor to project
            target_dim: Target dimension
            dtype: Optional dtype for projection
            device: Optional device for projection
            
        Returns:
            Projected tensor with preserved structure
        """
        if tensor.shape[-1] == target_dim:
            return tensor
            
        # Verify dimension compatibility
        self.verify_dimension(tensor)
        
        # Create and initialize projection layer
        projection = self._create_projection_layer(
            tensor.shape[-1],
            target_dim,
            dtype or tensor.dtype,
            device or tensor.device
        )
        
        # Project tensor
        original_shape = tensor.shape[:-1]
        tensor = tensor.reshape(-1, tensor.shape[-1])
        result = projection(tensor)
        
        return result.reshape(*original_shape, target_dim)
        
    def reshape_to_flat(self, tensor: Tensor) -> Tensor:
        """Reshape tensor to flat representation.
        
        Args:
            tensor: Input tensor to flatten
            
        Returns:
            Flattened tensor
        """
        return tensor.reshape(tensor.size(0), -1)
        
    def reshape_to_metric(self, tensor: Tensor, batch_size: int) -> Tensor:
        """Reshape tensor to metric tensor format.
        
        Args:
            tensor: Input tensor to reshape
            batch_size: Batch size for reshaping
            
        Returns:
            Reshaped metric tensor
        """
        manifold_dim = int(tensor.size(-1) ** 0.5)
        return tensor.reshape(batch_size, manifold_dim, manifold_dim)
        
    def reshape_to_connection(self, tensor: Tensor, batch_size: int) -> Tensor:
        """Reshape tensor to connection tensor format.
        
        Args:
            tensor: Input tensor to reshape
            batch_size: Batch size for reshaping
            
        Returns:
            Reshaped connection tensor
            
        Raises:
            ValueError: If tensor size is not a perfect cube
        """
        total_size = tensor.size(-1)
        manifold_dim = int(round(total_size ** (1/3)))  # Cube root
        
        # Verify dimensions
        if manifold_dim ** 3 != total_size:
            raise ValueError(
                f"Tensor size {total_size} is not a perfect cube for connection tensor. "
                f"Nearest cube root is {manifold_dim}"
            )
            
        return tensor.reshape(batch_size, manifold_dim, manifold_dim, manifold_dim)
        
    def validate_and_project(
        self,
        tensor: Tensor,
        target_dim: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> Tensor:
        """Validate tensor dimensions and project if needed.
        
        This method combines validation and projection into a single operation
        for convenience and safety.
        
        Args:
            tensor: Input tensor to project
            target_dim: Target dimension
            dtype: Data type for projection
            device: Device for projection
            
        Returns:
            Validated and projected tensor
            
        Raises:
            ValueError: If tensor dimensions are invalid
        """
        # Validate basic shape
        if len(tensor.shape) != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")
            
        # Verify minimum dimension
        self.verify_dimension(tensor)
            
        # Project if needed
        if tensor.size(-1) != target_dim:
            projection = self._create_projection_layer(
                tensor.size(-1),
                target_dim,
                dtype,
                device
            )
            tensor = projection(tensor)
            
        return tensor.to(dtype=dtype, device=device)

    def to_quantum_tensor(self, tensor: Tensor) -> QuantumTensor:
        """Convert tensor to QuantumTensor type.
        
        Args:
            tensor: Input tensor to convert
            
        Returns:
            Converted QuantumTensor
        """
        # Create a new tensor with the same data
        new_tensor = tensor.clone()
        # Convert to QuantumTensor
        return QuantumTensor(new_tensor.detach().requires_grad_(tensor.requires_grad))

    def to_geometric_tensor(self, tensor: Tensor) -> GeometricTensor:
        """Convert tensor to GeometricTensor type.
        
        Args:
            tensor: Input tensor to convert
            
        Returns:
            Converted GeometricTensor
        """
        # Create a new tensor with the same data
        new_tensor = tensor.clone()
        # Convert to GeometricTensor
        return GeometricTensor(new_tensor.detach().requires_grad_(tensor.requires_grad))
        
    def _create_projection_layer(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> nn.Linear:
        """Create and initialize a projection layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            dtype: Data type for layer
            device: Device for layer
            
        Returns:
            Initialized projection layer
        """
        projection = nn.Linear(
            in_features,
            out_features,
            dtype=dtype,
            device=device
        )
        nn.init.orthogonal_(projection.weight)
        nn.init.zeros_(projection.bias)
        return projection