"""Device management utilities."""

import torch
from typing import Optional, Union

def get_device() -> torch.device:
    """Get the default device for tensor operations."""
    # Force CPU-only mode
    return torch.device('cpu')

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to the default device."""
    # Force CPU-only mode
    return tensor.to(device='cpu')

def create_identity(size: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Create an identity matrix on the default device.
    
    Args:
        size: Size of the identity matrix
        dtype: Optional dtype for the matrix. If None, uses default dtype.
        
    Returns:
        Identity matrix on CPU with specified dtype
    """
    # Force CPU-only mode
    return torch.eye(size, dtype=dtype, device='cpu')