"""Tests for the attention tiling functionality."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

from src.core.backends.base import AttentionTile, ResolutionStrategy, ResourceProfile
from tests.utils.test_helpers import (
    assert_tensor_equal,
    benchmark_forward_backward,
    generate_random_tensor,
    measure_memory_usage,
)

try:
    import torch_vulkan  # type: ignore
    HAS_VULKAN = torch_vulkan.is_available()
except ImportError:
    HAS_VULKAN = False


class TestAttentionTile(nn.Module):
    """Concrete implementation of AttentionTile protocol for testing."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        manifold_dim: int = 32,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.device = device
        self.dtype = dtype
        self._state: Optional[torch.Tensor] = None
        self._neighbors: List[AttentionTile] = []
        self._resolution = 1.0
        self._metrics: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.process(x)

    @property
    def state(self) -> Optional[torch.Tensor]:
        return self._state

    @state.setter
    def state(self, value: Optional[torch.Tensor]) -> None:
        if value is not None:
            value = value.to(device=self.device, dtype=self.dtype)
        self._state = value

    @property
    def neighbors(self) -> List[AttentionTile]:
        return self._neighbors

    @property
    def resolution(self) -> float:
        return self._resolution

    @resolution.setter
    def resolution(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Resolution must be between 0.0 and 1.0")
        self._resolution = value

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention with current resolution."""
        query = query.to(device=self.device, dtype=self.dtype)
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        if mask is not None:
            mask = mask.to(device=self.device)
        
        batch_size, seq_len, _ = query.shape
        return torch.randn(
            batch_size, seq_len, self.hidden_dim,
            device=self.device, dtype=self.dtype
        )

    def process(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input tensor with current resolution."""
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        batch_size, seq_len, _ = inputs.shape
        outputs = self.compute_attention(inputs, inputs, inputs)
        self._metrics["compute_cost"] = seq_len * self.hidden_dim * self.resolution
        self._metrics["full_compute_cost"] = seq_len * self.hidden_dim
        return outputs

    def adapt_resolution(
        self, 
        strategy: ResolutionStrategy,
        scale_factor: float,
    ) -> None:
        """Adapt resolution based on strategy and scale factor."""
        old_resolution = self.resolution
        if strategy == ResolutionStrategy.FIXED:
            new_resolution = scale_factor
        elif strategy == ResolutionStrategy.ADAPTIVE:
            new_resolution = old_resolution * scale_factor
        else:  # DYNAMIC
            new_resolution = min(1.0, max(0.1, scale_factor))
        
        self.resolution = new_resolution

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if HAS_VULKAN:
            # Vulkan-specific memory tracking
            return {
                "allocated_memory": 1000.0,  # Example values
                "cached_memory": 500.0,
                "peak_memory": 1500.0
            }

        # CPU memory tracking
        return {
            "allocated_memory": 0.0,
            "cached_memory": 0.0,
            "peak_memory": 0.0
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        self._metrics.update({
            "gradient_norm": 1.0,
            "pattern_diff": 0.5,
            "flow": 0.7
        })
        return self._metrics

    def compress_state(self) -> torch.Tensor:
        """Compress current state."""
        if self._state is None:
            raise ValueError("No state to compress")
        compressed_size = int(self._state.shape[1] * self.resolution)
        return self._state[:, :compressed_size, :]

    def expand_state(self, target_size: int) -> torch.Tensor:
        """Expand compressed state to target size."""
        if self._state is None:
            raise ValueError("No state to expand")
        return torch.nn.functional.interpolate(
            self._state.transpose(1, 2),
            size=target_size,
            mode='linear'
        ).transpose(1, 2)


def test_random_seed_consistency() -> None:
    """Test that random seeds produce consistent results."""
    torch.manual_seed(42)
    tile1 = TestAttentionTile(input_dim=128, hidden_dim=128)
    inputs1 = torch.randn(1, 64, 128)
    output1 = tile1.process(inputs1)

    torch.manual_seed(42)
    tile2 = TestAttentionTile(input_dim=128, hidden_dim=128)
    inputs2 = torch.randn(1, 64, 128)
    output2 = tile2.process(inputs2)

    assert_tensor_equal(output1, output2)


@pytest.mark.parametrize("device", ["cpu"])
def test_device_support(device: str) -> None:
    """Test tile functionality on different devices."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128, device=device)
    inputs = torch.randn(1, 64, 128, device=device)
    outputs = tile.process(inputs)
    assert outputs.device.type == device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_support(dtype: torch.dtype) -> None:
    """Test tile functionality with different dtypes."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128, dtype=dtype)
    inputs = torch.randn(1, 64, 128, dtype=dtype)
    outputs = tile.process(inputs)
    assert outputs.dtype == dtype


def test_memory_tracking() -> None:
    """Test memory usage tracking."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    
    def _run_process() -> Tuple[float, float]:
        inputs = torch.randn(1, 64, 128)
        outputs = tile.process(inputs)
        stats = tile.get_memory_stats()
        return stats["allocated_memory"], stats["peak_memory"]

    allocated, peak = _run_process()
    assert peak >= allocated, "Peak memory should be >= allocated memory"


@pytest.mark.benchmark
def test_performance_benchmarking() -> None:
    """Test performance benchmarking capabilities."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    inputs = torch.randn(1, 64, 128, requires_grad=True)
    target = torch.randn(1, 64, 128)

    forward_time, backward_time = benchmark_forward_backward(
        model=tile,
        input_data=inputs,
        target=target,
        loss_fn=torch.nn.functional.mse_loss
    )
    assert forward_time > 0.0
    assert backward_time > 0.0


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "vulkan: mark test as requiring Vulkan")
    config.addinivalue_line("markers", "benchmark: mark test as performance benchmark")
    config.addinivalue_line("markers", "todo: mark test as incomplete/TODO")


def test_attention_tile_initialization() -> None:
    """Test basic initialization of AttentionTile."""
    # Test valid initialization
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    assert tile.input_dim == 128
    assert tile.hidden_dim == 128
    assert tile.state is None
    assert len(tile.neighbors) == 0
    assert tile.resolution == 1.0


def test_attention_tile_process() -> None:
    """Test processing of input sequences."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    inputs = torch.randn(1, 64, 128)

    # Test full resolution
    outputs = tile.process(inputs)
    assert outputs.shape == (1, 64, 128)

    # Test reduced resolution
    tile.resolution = 0.5
    outputs_reduced = tile.process(inputs)
    assert outputs_reduced.shape == (1, 64, 128)

    # Lower resolution should use less computation
    metrics = tile.get_metrics()
    assert metrics["compute_cost"] < metrics["full_compute_cost"]


def test_attention_tile_adapt_resolution() -> None:
    """Test dynamic resolution adaptation."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)

    # Test adaptation to lower resolution
    old_resolution = tile.resolution
    tile.adapt_resolution(ResolutionStrategy.ADAPTIVE, 0.5)
    assert tile.resolution < old_resolution
    assert tile.resolution >= 0.0

    # Test adaptation to higher resolution
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    tile.resolution = 0.5
    initial_resolution = tile.resolution

    # Apply high density multiple times
    for _ in range(5):
        tile.adapt_resolution(ResolutionStrategy.DYNAMIC, 0.9)

    assert tile.resolution > initial_resolution
    assert tile.resolution <= 1.0


def test_attention_tile_neighbors() -> None:
    """Test neighbor list functionality."""
    tile1 = TestAttentionTile(input_dim=128, hidden_dim=128)
    tile2 = TestAttentionTile(input_dim=128, hidden_dim=128)

    # Test adding neighbor
    tile1._neighbors.append(tile2)
    tile2._neighbors.append(tile1)
    assert tile2 in tile1.neighbors
    assert tile1 in tile2.neighbors

    # Test removing neighbor
    tile1._neighbors.remove(tile2)
    tile2._neighbors.remove(tile1)
    assert tile2 not in tile1.neighbors
    assert tile1 not in tile2.neighbors


def test_state_management() -> None:
    """Test state compression, expansion, and transfer."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)

    # Create sample state
    original_state = torch.randn(1, 32, 64)  # [batch, seq, state_dim]
    tile.state = original_state

    # Test compression
    compressed = tile.compress_state()
    assert compressed.shape[1] < original_state.shape[1]

    # Test expansion
    expanded = tile.expand_state(target_size=32)
    assert expanded.shape == original_state.shape

    # Test compression/expansion preserves relative values
    compressed_then_expanded = tile.expand_state(target_size=32)
    assert torch.allclose(original_state, compressed_then_expanded, rtol=0.2)


def test_advanced_resolution_adaptation() -> None:
    """Test different resolution adaptation strategies."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    tile.resolution = 0.5

    # Test different strategies
    strategies = {
        ResolutionStrategy.FIXED: (0.7, 0.7),
        ResolutionStrategy.ADAPTIVE: (0.7, 0.35),
        ResolutionStrategy.DYNAMIC: (0.7, 0.7),
    }

    for strategy, (scale_factor, expected) in strategies.items():
        tile = TestAttentionTile(input_dim=128, hidden_dim=128)
        tile.resolution = 0.5
        tile.adapt_resolution(strategy, scale_factor)

        if strategy == ResolutionStrategy.FIXED:
            assert tile.resolution == expected, f"Failed for strategy {strategy}"
        else:
            assert abs(tile.resolution - expected) < 0.2, f"Failed for strategy {strategy}"

    # Test memory stats
    stats = tile.get_memory_stats()
    assert "allocated_memory" in stats
    assert "cached_memory" in stats
    assert "peak_memory" in stats


def test_information_flow() -> None:
    """Test information flow tracking and control."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    inputs = torch.randn(1, 32, 64)

    _ = tile.process(inputs)
    metrics = tile.get_metrics()
    assert isinstance(metrics["gradient_norm"], float)
    assert isinstance(metrics["pattern_diff"], float)
    assert isinstance(metrics["flow"], float)
    assert all(
        isinstance(v, float) and v >= 0
        for v in [metrics["gradient_norm"], metrics["pattern_diff"], metrics["flow"]]
    )


def test_error_handling() -> None:
    """Test error handling and edge cases."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    
    # Test invalid resolution
    with pytest.raises(ValueError):
        tile.resolution = -0.1
    with pytest.raises(ValueError):
        tile.resolution = 1.5
        
    # Test state operations without state
    with pytest.raises(ValueError):
        tile.compress_state()
    with pytest.raises(ValueError):
        tile.expand_state(32)
        
    # Test invalid input dimensions
    with pytest.raises(RuntimeError):
        tile.process(torch.randn(1, 64))  # Missing dimension
    with pytest.raises(RuntimeError):
        tile.process(torch.randn(1, 64, 256))  # Wrong input dimension


@pytest.mark.benchmark
def test_resource_profiling() -> None:
    """Test resource profiling capabilities."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    
    # Profile memory usage
    inputs = torch.randn(1, 64, 128)
    
    # Get initial memory state
    initial_stats = tile.get_memory_stats()
    
    # Process and track memory
    outputs = tile.process(inputs)
    process_stats = tile.get_memory_stats()
    
    # Verify memory tracking
    assert process_stats["allocated_memory"] >= initial_stats["allocated_memory"]
    assert process_stats["peak_memory"] >= initial_stats["peak_memory"]
    
    # Profile computation cost
    metrics = tile.get_metrics()
    assert "compute_cost" in metrics
    assert metrics["compute_cost"] > 0
    assert metrics["compute_cost"] <= metrics["full_compute_cost"]


def test_attention_computation() -> None:
    """Test attention computation correctness."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    
    # Test with different sequence lengths
    for seq_len in [16, 32, 64]:
        query = torch.randn(1, seq_len, 128)
        key = torch.randn(1, seq_len, 128)
        value = torch.randn(1, seq_len, 128)
        
        # Test without mask
        output = tile.compute_attention(query, key, value)
        assert output.shape == (1, seq_len, 128)
        
        # Test with attention mask
        mask = torch.ones(1, seq_len, seq_len).bool()
        mask[:, :, seq_len//2:] = False  # Mask out second half
        output_masked = tile.compute_attention(query, key, value, mask)
        assert output_masked.shape == (1, seq_len, 128)
        
        # Verify masked attention has different values
        assert not torch.allclose(output, output_masked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_device_support() -> None:
    """Test attention tile functionality across devices."""
    # Create tiles on different devices
    cpu_tile = TestAttentionTile(input_dim=128, hidden_dim=128, device="cpu")
    gpu_tile = TestAttentionTile(input_dim=128, hidden_dim=128, device="cuda")
    
    # Test data transfer
    inputs = torch.randn(1, 64, 128)
    cpu_output = cpu_tile.process(inputs)
    gpu_output = gpu_tile.process(inputs)
    
    # Results should be similar regardless of device
    cpu_output_gpu = cpu_output.cuda()
    assert torch.allclose(cpu_output_gpu, gpu_output, rtol=1e-5)
    
    # Test state transfer between devices
    state = torch.randn(1, 32, 64)
    cpu_tile.state = state
    gpu_tile.state = state
    
    assert torch.allclose(
        cpu_tile.state.cuda(),
        gpu_tile.state,
        rtol=1e-5
    )


def test_resolution_impact() -> None:
    """Test the impact of resolution on computation and accuracy."""
    tile = TestAttentionTile(input_dim=128, hidden_dim=128)
    inputs = torch.randn(1, 64, 128)
    
    # Get baseline results at full resolution
    tile.resolution = 1.0
    baseline_output = tile.process(inputs)
    baseline_metrics = tile.get_metrics()
    
    # Test different resolutions
    resolutions = [0.25, 0.5, 0.75]
    for res in resolutions:
        tile.resolution = res
        output = tile.process(inputs)
        metrics = tile.get_metrics()
        
        # Verify computation cost scales with resolution
        assert abs(metrics["compute_cost"] - 
                  baseline_metrics["compute_cost"] * res) < 1e-5
        
        # Verify output maintains reasonable similarity
        similarity = torch.nn.functional.cosine_similarity(
            baseline_output.flatten(),
            output.flatten(),
            dim=0
        )
        assert similarity > 0.5  # Outputs should maintain some similarity
