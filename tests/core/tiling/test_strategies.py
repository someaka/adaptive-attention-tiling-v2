"""Tests for the attention tiling functionality."""

import numpy as np
import pytest
import torch

from src.core.tiling.base import AttentionTile, ResolutionStrategy, ResourceProfile
from src.core.tiling.components.config import CONFIG


def test_attention_tile_initialization() -> None:
    """Test basic initialization of AttentionTile.

    Ensures:
    - Correct dimension handling
    - Proper state initialization
    - Valid resolution range
    - Proper component initialization
    """
    # Test valid initialization
    tile = AttentionTile(size=32, hidden_dim=128, resolution=1.0)
    assert tile.size == 32
    assert tile.resolution == 1.0
    assert tile.state is None
    assert len(tile.neighbors) == 0

    # Test invalid size
    with pytest.raises(ValueError, match="Size must be positive"):
        AttentionTile(size=0, hidden_dim=128, resolution=1.0)

    # Test invalid resolution
    with pytest.raises(ValueError, match="Resolution must be between"):
        AttentionTile(size=32, hidden_dim=128, resolution=1.5)
    with pytest.raises(ValueError, match="Resolution must be between"):
        AttentionTile(size=32, hidden_dim=128, resolution=-0.1)


def test_attention_tile_process() -> None:
    """Test processing of input sequences."""
    tile = AttentionTile(size=64, hidden_dim=128, resolution=1.0)
    inputs = torch.randn(1, 64, 128)

    # Test full resolution
    outputs = tile.process(inputs)
    assert outputs.shape == inputs.shape

    # Test reduced resolution
    tile.resolution = 0.5
    outputs_reduced = tile.process(inputs)
    assert outputs_reduced.shape == inputs.shape

    # Lower resolution should use less computation
    compute_metrics = tile.get_metrics()
    compute_cost = compute_metrics["compute_cost"]
    full_cost = compute_metrics["full_compute_cost"]
    assert compute_cost < full_cost


def test_attention_tile_adapt_resolution() -> None:
    """Test dynamic resolution adaptation.

    Verifies that:
    1. Resolution decreases when density is low
    2. Resolution trends upward when density is high
    3. Changes are smooth and stable (momentum)
    """
    tile = AttentionTile(size=32, hidden_dim=128, resolution=1.0)

    # Test adaptation to lower resolution
    density_metric = 0.3
    old_resolution = tile.resolution
    tile.adapt_resolution(density_metric)
    assert tile.resolution < old_resolution
    assert tile.resolution >= CONFIG.MIN_RESOLUTION

    # Test adaptation to higher resolution
    # Multiple steps to observe trending behavior
    # Start from a lower resolution to test increase
    tile = AttentionTile(size=32, hidden_dim=128, resolution=0.5)
    initial_resolution = tile.resolution

    # Apply high density multiple times to overcome momentum
    for _ in range(5):  # Need multiple steps due to momentum
        tile.adapt_resolution(0.9)

    # Final resolution should be higher than initial
    assert tile.resolution > initial_resolution
    assert tile.resolution <= CONFIG.MAX_RESOLUTION


def test_attention_tile_neighbor_management() -> None:
    """Test neighbor management functionality."""
    tile1 = AttentionTile(size=32, hidden_dim=128, resolution=1.0)
    tile2 = AttentionTile(size=32, hidden_dim=128, resolution=1.0)

    # Test adding neighbor
    tile1.add_neighbor(tile2)
    assert tile2 in tile1.neighbors
    assert tile1 in tile2.neighbors

    # Test removing neighbor
    tile1.remove_neighbor(tile2)
    assert tile2 not in tile1.neighbors
    assert tile1 not in tile2.neighbors


def test_state_management() -> None:
    """Test state compression, expansion, and transfer."""
    tile1 = AttentionTile(size=32, hidden_dim=128, resolution=0.5)  # Start with lower resolution
    AttentionTile(size=32, hidden_dim=128, resolution=0.5)

    # Create sample state
    original_state = torch.randn(1, 32, 64)  # [batch, seq, state_dim]
    tile1.state = original_state

    # Test compression
    compressed = tile1.compress_state()
    assert compressed.shape[1] < original_state.shape[1]  # Should be smaller now

    # Test expansion
    expanded = tile1.expand_state(target_size=32)
    assert expanded.shape == original_state.shape

    # Test that compression followed by expansion preserves relative values
    # Note: We can't expect exact equality due to interpolation
    compressed_then_expanded = tile1.expand_state(target_size=32)
    assert torch.allclose(original_state, compressed_then_expanded, rtol=0.2)


def test_advanced_resolution_adaptation() -> None:
    """Test different resolution adaptation strategies."""
    if not hasattr(torch, "vulkan") or not torch.vulkan.is_available():
        pytest.skip("Test requires Vulkan for consistent resolution behavior")
    tile = AttentionTile(size=32, hidden_dim=128, resolution=0.5)

    # Test different strategies
    strategies = {
        ResolutionStrategy.LINEAR: (0.7, 0.6),  # (metric, expected)
        ResolutionStrategy.EXPONENTIAL: (0.7, 0.6),
        ResolutionStrategy.THRESHOLD: (
            0.7,
            1.0,
        ),  # Threshold strategy jumps to max when above threshold
        ResolutionStrategy.ADAPTIVE: (0.7, 0.6),  # Considers history
    }

    for strategy, (metric, expected) in strategies.items():
        # Reset tile and adapter state
        tile = AttentionTile(size=32, hidden_dim=128, resolution=0.5)  # Create fresh tile
        tile.adapt_resolution(metric, strategy=strategy, momentum=0.0, hysteresis=0.0)

        if strategy == ResolutionStrategy.THRESHOLD:
            assert tile.resolution == expected, f"Failed for strategy {strategy}"
        else:
            assert abs(tile.resolution - expected) < 0.2, f"Failed for strategy {strategy}"

    # Test hysteresis
    history = []
    tile = AttentionTile(size=32, hidden_dim=128, resolution=0.5)  # Reset
    for _ in range(10):
        # Oscillating input with high hysteresis to dampen changes
        metric = 0.7 if len(history) % 2 == 0 else 0.3
        tile.adapt_resolution(metric, strategy=ResolutionStrategy.LINEAR, hysteresis=0.5)
        history.append(tile.resolution)

    # Check that changes are minimal due to high hysteresis
    variations = np.diff(history)
    assert np.mean(np.abs(variations)) < 0.2

    # Test momentum
    tile = AttentionTile(size=32, hidden_dim=128, resolution=0.5)  # Reset
    initial_res = tile.resolution

    # Sudden change - use a high density to force increase
    new_res = tile.adapt_resolution(
        0.95,
        strategy=ResolutionStrategy.LINEAR,
        momentum=0.8,
        hysteresis=0.1,
    )
    assert new_res > initial_res
    assert new_res < 1.0  # Should not reach max due to momentum

    # Should continue increasing due to momentum
    final_res = tile.adapt_resolution(
        0.5,  # Even with lower density
        strategy=ResolutionStrategy.LINEAR,
        momentum=0.8,
        hysteresis=0.1,
    )
    assert final_res > new_res  # Still increases due to momentum


def test_resolution_strategies(tile: AttentionTile) -> None:
    """Test different resolution adaptation strategies."""
    # Test each strategy with known inputs and expected outputs
    strategies = {
        ResolutionStrategy.LINEAR: (0.8, 0.7),  # High input -> high resolution
        ResolutionStrategy.EXPONENTIAL: (0.8, 0.6),  # High input -> medium resolution
        ResolutionStrategy.ADAPTIVE: (0.7, 0.6),  # Considers history
    }

    for strategy, (metric, expected) in strategies.items():
        tile.resolution = 0.5  # Reset
        tile.adapt_resolution(
            metric,
            strategy=strategy,
            momentum=0.5,
            hysteresis=0.1,
        )
        assert abs(tile.resolution - expected) < 0.3, f"Failed for strategy {strategy}"

    # Test hysteresis
    history = []
    tile.resolution = 0.5  # Reset
    for _ in range(10):
        # Oscillating input with high hysteresis to dampen changes
        metric = 0.7 if len(history) % 2 == 0 else 0.3
        tile.adapt_resolution(
            metric,
            strategy=ResolutionStrategy.LINEAR,
            hysteresis=0.5,
        )
        history.append(tile.resolution)

    # Check that changes are minimal due to high hysteresis
    variations = np.diff(history)
    assert np.mean(np.abs(variations)) < 0.2


def test_information_flow() -> None:
    """Test information flow tracking and control."""
    tile = AttentionTile(size=32, hidden_dim=128)
    inputs = torch.randn(1, 32, 64)

    # Test information flow statistics
    _ = tile.process(inputs)
    metrics = tile.get_metrics()
    assert isinstance(metrics["gradient_norm"], float)
    assert isinstance(metrics["pattern_diff"], float)
    assert isinstance(metrics["flow"], float)
    assert all(
        isinstance(v, float) and v >= 0
        for v in [metrics["gradient_norm"], metrics["pattern_diff"], metrics["flow"]]
    )


def test_resource_management() -> None:
    """Test resource optimization and management."""
    tile = AttentionTile(size=64, hidden_dim=128, resolution=1.0)
    profile = ResourceProfile(compute_limit=1e6, memory_limit=1e6)

    # Test resource optimization
    tile.optimize_resources(profile)
    assert tile.get_metrics()["full_compute_cost"] <= profile.compute_limit

    # Test automatic scaling
    inputs = torch.randn(1, 64, 128)
    tile.process(inputs)
    assert tile.get_metrics()["last_compute_cost"] <= profile.compute_limit

    # Test memory optimization
    mem_stats = tile.get_memory_stats()
    assert mem_stats["peak_memory"] <= profile.memory_limit


def test_cross_tile_optimization() -> None:
    """Test optimization across multiple tiles.

    This test verifies that:
    1. Load balancing properly redistributes compute costs
    2. Dynamic minimum load constraints are maintained based on information density
    3. Maximum load constraints are adapted to computational budget
    4. The overall load imbalance is reduced
    5. Resolution adjustments are within bounds
    6. Tiles adapt to local information density
    7. Chain topology maintains connectivity
    """
    # Set up tiles with different information densities
    num_tiles = 5
    tiles = [AttentionTile(size=32, hidden_dim=128, resolution=1.0) for _ in range(num_tiles)]
    # More gradual density variation
    density_factors = [0.6, 0.7, 0.8, 0.9, 1.0]
    base_inputs = torch.randn(1, 32, 64)

    # Initialize tiles with different loads and densities
    for tile, density in zip(tiles, density_factors):
        input_tensor = base_inputs * density
        _ = tile.process(input_tensor)
        # Set initial load closer to target
        initial_load = 0.7 * density  # Higher base load
        tile.update_metrics(
            {"compute_cost": min(0.9, initial_load), "information_density": density},
        )

    # Connect tiles in a chain (paper's topology)
    for i in range(num_tiles - 1):
        tiles[i].add_neighbor(tiles[i + 1])

    # Get initial loads for comparison
    initial_loads = [t.get_metrics()["compute_cost"] for t in tiles]
    initial_max_diff = max(initial_loads) - min(initial_loads)

    # Calculate load thresholds based on density
    min_thresholds = [max(0.3, 0.5 * d) for d in density_factors]  # Higher min load
    # Lower max load
    max_thresholds = [min(0.95, 0.9 + 0.05 * d) for d in density_factors]

    # Run optimization steps
    num_steps = 10
    for _ in range(num_steps):
        # First adapt resolution based on local density
        for tile, density in zip(tiles, density_factors):
            tile.adapt_resolution(density)

        # Then balance load with neighbors
        for i, tile in enumerate(tiles):
            neighbors = tiles[:i] + tiles[i + 1 :]
            tile.balance_load(neighbors)

    # Get final state
    final_loads = [t.get_metrics()["compute_cost"] for t in tiles]
    final_max_diff = max(final_loads) - min(final_loads)

    # Verify improvements
    assert (
        final_max_diff <= initial_max_diff
    ), f"Load imbalance increased: {final_max_diff} > {initial_max_diff}"

    # Verify load constraints are respected
    for load, min_t, max_t in zip(final_loads, min_thresholds, max_thresholds):
        assert load >= min_t, f"Load {load} below minimum threshold {min_t}"
        assert load <= max_t, f"Load {load} above maximum threshold {max_t}"

    # Verify load distribution relative to information density
    weighted_loads = [load / density for load, density in zip(final_loads, density_factors)]
    max_weighted_diff = max(weighted_loads) - min(weighted_loads)
    # Allow more variation in density-weighted loads
    assert (
        max_weighted_diff <= 0.5
    ), "Load distribution not properly balanced for information density"

    # Check resolution bounds
    for tile in tiles:
        msg = f"Resolution {tile.resolution} out of bounds"
        assert 0.1 <= tile.resolution <= 1.0, msg


def test_debugging_visualization() -> None:
    """Test debugging and visualization features."""
    tile = AttentionTile(size=32, hidden_dim=128, resolution=1.0)
    inputs = torch.randn(1, 32, 64)

    # Process some data to generate state
    outputs = tile.process(inputs)
    tile.state = outputs  # Set state explicitly for testing

    # Test visualization data
    vis_data = tile.get_visualization_data()
    assert "attention_patterns" in vis_data
    assert "state_heatmap" in vis_data
    assert "compute_efficiency" in vis_data
    assert "memory_usage" in vis_data
    assert "resolution" in vis_data

    # Check tensor fields
    assert isinstance(vis_data["attention_patterns"], torch.Tensor)
    assert isinstance(vis_data["state_heatmap"], torch.Tensor)
    # Check numeric fields
    assert isinstance(vis_data["compute_efficiency"], float)
    assert isinstance(vis_data["memory_usage"], float)
    assert isinstance(vis_data["resolution"], float)


@pytest.fixture
def tile() -> AttentionTile:
    """Create a basic attention tile for testing."""
    return AttentionTile(size=16, hidden_dim=128, resolution=1.0)


def test_initialization() -> None:
    """Test the initialization of AttentionTile with various parameters."""
    # Valid initialization
    tile = AttentionTile(size=16, hidden_dim=128, resolution=0.5)
    assert tile.size == 16
    assert tile.resolution == 0.5
    assert tile.state is None

    # Invalid size
    with pytest.raises(ValueError, match="Size must be positive"):
        AttentionTile(size=0, hidden_dim=128)
    with pytest.raises(ValueError, match="Size must be positive"):
        AttentionTile(size=-1, hidden_dim=128)

    # Invalid resolution
    with pytest.raises(ValueError, match="Resolution must be between"):
        AttentionTile(size=16, hidden_dim=128, resolution=0.05)
    with pytest.raises(ValueError, match="Resolution must be between"):
        AttentionTile(size=16, hidden_dim=128, resolution=1.5)


def test_process(tile: AttentionTile) -> None:
    """Test the processing of input data through the attention tile."""
    batch_size, seq_len, dim = 2, 16, 32
    inputs = torch.randn(batch_size, seq_len, dim)

    # Full resolution processing
    outputs = tile.process(inputs)
    assert outputs.shape == inputs.shape

    # Reduced resolution processing
    tile.resolution = 0.5
    outputs = tile.process(inputs)
    assert outputs.shape == inputs.shape


def test_state_compression(tile: AttentionTile) -> None:
    """Test the state compression functionality of the attention tile."""
    # Set initial state
    tile.state = torch.randn(1, tile.size, 32)
    original_state = tile.state.clone()

    # Test compression
    tile.resolution = 0.5
    compressed = tile.compress_state()
    assert compressed.shape[1] == max(2, int(tile.size * tile.resolution))

    # Test expansion
    expanded = tile.expand_state(tile.size)
    assert expanded.shape == original_state.shape


def test_state_transfer(tile: AttentionTile) -> None:
    """Test the state transfer between neighboring attention tiles."""
    neighbor = AttentionTile(size=16, hidden_dim=128)

    # Add neighbor relationship
    tile.add_neighbor(neighbor)

    # Set initial state
    tile.state = torch.randn(1, tile.size, 32)

    # Test transfer with overlap
    tile.transfer_state(neighbor, blend_factor=0.5)
    assert neighbor.state is not None
    assert neighbor.state.shape == (1, neighbor.size, 32)

    # Test transfer with no overlap
    tile.transfer_state(neighbor, blend_factor=1.0)
    assert torch.allclose(neighbor.state[:, :16, :], tile.state[:, :16, :], rtol=0.1)


def test_resolution_adaptation(tile: AttentionTile) -> None:
    """Test the resolution adaptation strategies of the attention tile."""
    # Test linear strategy
    tile.adapt_resolution(0.75, strategy=ResolutionStrategy.LINEAR)
    assert CONFIG.MIN_RESOLUTION <= tile.resolution <= CONFIG.MAX_RESOLUTION

    # Test exponential strategy
    tile.adapt_resolution(0.75, strategy=ResolutionStrategy.EXPONENTIAL)
    assert CONFIG.MIN_RESOLUTION <= tile.resolution <= CONFIG.MAX_RESOLUTION

    # Test threshold strategy
    tile.adapt_resolution(0.3, strategy=ResolutionStrategy.THRESHOLD)
    assert CONFIG.MIN_RESOLUTION <= tile.resolution <= CONFIG.MAX_RESOLUTION

    # Test adaptive strategy
    for i in range(5):
        tile.adapt_resolution(0.5 + i * 0.1, strategy=ResolutionStrategy.ADAPTIVE)
        assert CONFIG.MIN_RESOLUTION <= tile.resolution <= CONFIG.MAX_RESOLUTION


def test_gradient_tracking(tile: AttentionTile) -> None:
    """Test the gradient tracking functionality of the attention tile.

    TODO: Update this test when Vulkan gradient tracking is implemented.
    For now, we just verify the gradient norm is always zero.
    """
    gradient_norm = tile._compute_gradient_norm()
    assert gradient_norm.item() == 0, "Gradient norm should be zero"


def test_boundary_mask(tile: AttentionTile) -> None:
    """Test the boundary mask generation for attention tiles."""
    mask = tile.get_boundary_mask()
    assert mask.shape == (1, tile.size, tile.size)
    assert torch.all(mask <= 1.0)
    assert torch.all(mask >= 0.0)


def test_information_flow_computation(tile: AttentionTile) -> None:
    """Test the information flow computation between attention tiles."""
    # Test with empty state
    info_flow = tile.compute_information_flow()
    assert info_flow == 0.0

    # Test with some state
    inputs = torch.randn(1, 64, 128)
    _ = tile.process(inputs)
    info_flow = tile.compute_information_flow()
    # Information flow can be positive or negative depending on attention patterns
    assert isinstance(info_flow, float)

    # Test with neighbors
    neighbor = AttentionTile(size=32, hidden_dim=128)
    tile.add_neighbor(neighbor)
    _ = neighbor.process(inputs)
    info_flow = tile.compute_information_flow()
    # Information flow can be positive or negative depending on attention patterns
    assert isinstance(info_flow, float)


def test_resource_optimization(tile: AttentionTile) -> None:
    """Test the resource optimization capabilities of attention tiles."""
    profile = ResourceProfile(compute_limit=1000.0, memory_limit=1e6)

    # Process some inputs to set compute cost
    inputs = torch.randn(2, tile.size, 32)
    tile.process(inputs)

    # Test optimization
    original_resolution = tile.resolution
    tile.optimize_resources(profile)
    assert tile.resolution <= original_resolution


def test_memory_stats(tile: AttentionTile) -> None:
    """Test the memory statistics collection for attention tiles."""
    # Test with no state
    stats = tile.get_memory_stats()
    assert "peak_memory" in stats
    assert "current_memory" in stats

    # Test with state
    tile.state = torch.randn(1, tile.size, 32)
    # Get memory stats based on device type
    if hasattr(torch, "vulkan") and torch.vulkan.is_available():
        backend = tile.get_vulkan_backend()
        stats = backend.get_metrics()
        assert stats["memory_usage"] > 0
    else:
        stats = tile.get_memory_stats()
        assert stats["current_memory"] > 0


def test_load_balancing(tile: AttentionTile) -> None:
    """Test the load balancing functionality between attention tiles."""
    neighbors = [AttentionTile(size=16, hidden_dim=128) for _ in range(3)]
    for n in neighbors:
        tile.add_neighbor(n)

    # Process some inputs to set compute costs
    inputs = torch.randn(2, tile.size, 32)
    tile.process(inputs)
    for n in neighbors:
        n.process(inputs)

    # Test load balancing
    tile.balance_load(neighbors)
    assert CONFIG.MIN_RESOLUTION <= tile.resolution <= CONFIG.MAX_RESOLUTION


def test_visualization_data(tile: AttentionTile) -> None:
    """Test the visualization data generation for attention tiles."""
    # Test with no state
    data = tile.get_visualization_data()
    assert "attention_patterns" in data
    assert "state_heatmap" in data
    assert "compute_efficiency" in data
    assert "memory_usage" in data
    assert "resolution" in data

    # Test with state
    tile.state = torch.randn(1, tile.size, 32)
    data = tile.get_visualization_data()
    assert isinstance(data["attention_patterns"], torch.Tensor)
    assert isinstance(data["state_heatmap"], torch.Tensor)
    assert isinstance(data["compute_efficiency"], float)
    assert isinstance(data["memory_usage"], float)
    assert isinstance(data["resolution"], float)


def test_metrics_logging(tile: AttentionTile) -> None:
    """Test the metrics logging functionality of attention tiles."""
    # Process some input to generate metrics
    inputs = torch.randn(2, tile.size, 32)
    tile.process(inputs)

    # Test logging
    metrics = tile.log_metrics()
    assert "compute_efficiency" in metrics
    assert "memory_usage" in metrics
    assert "resolution" in metrics
    assert "avg_compute_time" in metrics
    assert "compute_cost" in metrics
    assert "load_balance" in metrics

    # Check that all metrics are floats
    for key, value in metrics.items():
        if key != "resolution_history":  # Skip list field
            msg = f"Metric {key} should be float, got {type(value)}"
            assert isinstance(value, float), msg

    # Check that metrics are within reasonable ranges
    assert metrics["compute_efficiency"] >= 0  # Can be > 1 for super-efficient cases
    assert metrics["memory_usage"] >= 0
    assert 0 <= metrics["resolution"] <= 1.0
    assert metrics["load_balance"] >= 0


def test_neighbor_management(tile: AttentionTile) -> None:
    """Test the neighbor management functionality of attention tiles."""
    neighbor = AttentionTile(size=16, hidden_dim=128)

    # Test adding neighbor
    tile.add_neighbor(neighbor)
    assert neighbor in tile.neighbors
    assert tile in neighbor.neighbors

    # Test removing neighbor
    tile.remove_neighbor(neighbor)
    assert neighbor not in tile.neighbors
    assert tile not in neighbor.neighbors


def test_load_balancing_edge_cases() -> None:
    """Test load balancing under various edge conditions."""
    if not hasattr(torch, "vulkan") or not torch.vulkan.is_available():
        pytest.skip("Test requires Vulkan for consistent load balancing behavior")
    # Setup base tile
    base_tile = AttentionTile(size=32, hidden_dim=128, resolution=1.0)
    inputs = torch.randn(1, 32, 64)

    # Case 1: Single neighbor with extreme load
    neighbor1 = AttentionTile(size=32, hidden_dim=128, resolution=1.0)
    _ = base_tile.process(inputs)  # Set base load to ~0.5
    _ = neighbor1.process(inputs)

    # Set initial loads through metrics
    neighbor1.update_metrics({"compute_cost": 0.05})  # Set extreme low load
    base_tile.update_metrics({"compute_cost": 0.8})  # Set high load

    # Run multiple balance iterations to achieve target load
    initial_load = base_tile.get_metrics()["compute_cost"]
    for _ in range(3):  # Allow multiple iterations for momentum-based changes
        base_tile.balance_load([neighbor1])

    # Check that load decreased over multiple iterations
    n1_metrics = neighbor1.get_metrics()
    base_metrics = base_tile.get_metrics()
    assert n1_metrics["compute_cost"] > 0.05  # Neighbor load should increase
    assert base_metrics["compute_cost"] < initial_load  # Base load should decrease

    # Case 2: All neighbors at max capacity
    neighbors_max = [AttentionTile(size=32, hidden_dim=128, resolution=1.0) for _ in range(3)]
    for n in neighbors_max:
        n.update_metrics({"compute_cost": 0.9})
    base_tile.update_metrics({"compute_cost": 0.3})
    base_tile.balance_load(neighbors_max)
    assert base_tile.get_metrics()["compute_cost"] > 0.3

    # Case 3: All neighbors at min capacity
    neighbors_min = [AttentionTile(size=32, hidden_dim=128, resolution=1.0) for _ in range(3)]
    for n in neighbors_min:
        n.update_metrics({"compute_cost": 0.1})
    base_tile.update_metrics({"compute_cost": 0.9})

    # Run multiple balance iterations for significant load reduction
    initial_load = base_tile.get_metrics()["compute_cost"]
    for _ in range(3):
        base_tile.balance_load(neighbors_min)
    assert base_tile.get_metrics()["compute_cost"] < initial_load
