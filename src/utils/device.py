"""Device management utilities."""

import torch
from typing import Optional, Tuple

def get_device(prefer_vulkan: bool = False) -> Tuple[torch.device, bool]:
    """Get the best available device for computation.
    Currently forcing CPU usage until Vulkan support is complete.
    
    Args:
        prefer_vulkan: Whether to try Vulkan first (currently ignored)
        
    Returns:
        Tuple of (device, is_vulkan_available)
    """
    # Force CPU for now
    return torch.device('cpu'), False

def to_device(tensor: torch.Tensor, device: Optional[torch.device] = None, force_cpu: bool = True) -> torch.Tensor:
    """Move tensor to device, with graceful fallback to CPU.
    Currently forcing CPU usage until Vulkan support is complete.
    
    Args:
        tensor: Input tensor
        device: Target device (currently ignored)
        force_cpu: If True, always returns CPU tensor
        
    Returns:
        Tensor on CPU
    """
    return tensor.cpu()

def create_identity(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create an identity matrix.
    Currently always creates on CPU until Vulkan support is complete.
    
    Args:
        size: Size of the identity matrix
        device: Target device (currently ignored)
        
    Returns:
        Identity matrix on CPU
    """
    return torch.eye(size)  # Always on CPU