"""Base test utilities and fixtures for holographic tests."""

import pytest
import torch
from typing import Tuple, NamedTuple, Union
from src.core.crystal.scale_classes.holographiclift import HolographicLifter
from src.core.crystal.scale_classes.ml.models import HolographicNet
from src.core.crystal.scale_classes.ml.config import HolographicTestConfig


class HolographicTestData(NamedTuple):
    """Container for test data using NamedTuple for immutability and efficiency."""
    boundary_field: torch.Tensor
    radial_points: torch.Tensor
    bulk_field: torch.Tensor
    uv_data: torch.Tensor  # Pre-computed for efficiency
    ir_data: torch.Tensor  # Pre-computed for efficiency


class TestHolographicBase:
    """Base class containing common test utilities."""
    
    @pytest.fixture(scope="class")
    def config(self):
        """Provide test configuration."""
        return HolographicTestConfig()
    
    @staticmethod
    def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        """Compute relative error between tensors."""
        norm_expected = torch.norm(expected)
        return torch.norm(actual - expected).item() / norm_expected.item() if norm_expected > 0 else float('inf')
    
    @staticmethod
    def create_test_data(model: Union[HolographicNet, HolographicLifter],
                        shape: Tuple[int, ...] = (4, 4),  # Smaller default shape
                        n_radial: int = 10,  # Fewer radial points by default
                        z_range: Tuple[float, float] = (0.1, 10.0),
                        random: bool = True) -> HolographicTestData:
        """Create and cache test data."""
        boundary = (torch.randn if random else torch.ones)(*shape, dtype=model.dtype)
        radial = torch.linspace(z_range[0], z_range[1], n_radial, dtype=model.dtype)
        
        # Handle both HolographicNet and HolographicLifter
        if isinstance(model, HolographicNet):
            bulk = model.holographic_lift(boundary, radial)
        else:
            bulk = model.holographic_lift(boundary, radial)
        
        return HolographicTestData(
            boundary_field=boundary,
            radial_points=radial,
            bulk_field=bulk,
            uv_data=bulk[0],  # Pre-compute for efficiency
            ir_data=bulk[-1]
        )
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate tensor properties with descriptive errors."""
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains infinite values")
        if torch.norm(tensor) == 0:
            raise ValueError(f"{name} has zero norm") 