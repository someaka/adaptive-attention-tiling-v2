"""Configuration for tiling components."""

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TilingConfig:
    """Configuration for tiling operations."""
    
    # Basic tiling parameters
    tile_size: Tuple[int, ...] = (8, 8)
    stride: Optional[Tuple[int, ...]] = None
    padding: str = 'same'
    
    # Advanced options
    overlap: int = 0
    dilation: int = 1
    groups: int = 1
    
    # Memory management
    max_memory_gb: float = 4.0
    optimize_layout: bool = True
    
    # Performance tuning
    num_warps: int = 4
    num_stages: int = 2
    vectorize: bool = True

CONFIG = TilingConfig()
