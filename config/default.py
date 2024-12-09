from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Default configuration."""

    backend: Literal["cpu", "vulkan"] = "cpu"
    tile_size: int = 1024
    precision: Literal["float32", "float16"] = "float32"
