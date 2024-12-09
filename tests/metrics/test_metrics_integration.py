"""Integration tests for advanced metrics with AttentionTile."""

import pytest
import torch

from src.core.common.enums import ResolutionStrategy
from src.core.tiling.base.attention_tile import AttentionTile


@pytest.fixture
def attention_tile():
    """Create a basic attention tile for testing."""
    return AttentionTile(size=16, resolution=1.0, hidden_dim=32)


@pytest.fixture
def input_tensor():
    """Create a sample input tensor."""
    batch_size = 2
    seq_len = 16
    hidden_dim = 32
    return torch.randn(batch_size, seq_len, hidden_dim)


class TestMetricsIntegration:
    """Test integration of advanced metrics with AttentionTile."""

    def test_metrics_initialization(self, attention_tile):
        """Test that advanced metrics are properly initialized."""
        metrics = attention_tile.get_metrics()
        assert "ifq" in metrics
        assert "cer" in metrics
        assert "ae" in metrics
        assert all(isinstance(metrics[k], float) for k in ["ifq", "cer", "ae"])

    def test_metrics_after_processing(self, attention_tile, input_tensor):
        """Test metrics update after processing input."""
        # Process input
        output = attention_tile._process_impl(input_tensor, update_metrics=True)
        metrics = attention_tile.get_metrics()

        # Check metrics were updated
        assert metrics["ifq"] >= 0.0
        assert metrics["cer"] >= 0.0
        assert metrics["ae"] >= 0.0
        assert output.shape == input_tensor.shape

    def test_metrics_history(self, attention_tile, input_tensor):
        """Test metrics history accumulation."""
        # Process multiple times
        for _ in range(3):
            attention_tile._process_impl(input_tensor, update_metrics=True)

        # Check metrics log
        assert len(attention_tile._metrics_log) >= 3
        assert all("ifq" in m for m in attention_tile._metrics_log[-3:])
        assert all("cer" in m for m in attention_tile._metrics_log[-3:])
        assert all("ae" in m for m in attention_tile._metrics_log[-3:])

    def test_metrics_with_resolution_change(self, attention_tile, input_tensor):
        """Test metrics behavior with resolution changes."""
        # Initial processing
        attention_tile._process_impl(input_tensor, update_metrics=True)
        initial_metrics = attention_tile.get_metrics()

        # Change resolution
        attention_tile.resolution = 0.5
        attention_tile._process_impl(input_tensor, update_metrics=True)
        new_metrics = attention_tile.get_metrics()

        # Metrics should reflect the change
        assert len(attention_tile._metrics["resolution_history"]) >= 2
        assert (
            new_metrics["cer"] != initial_metrics["cer"]
        )  # CER should change with resolution

    def test_metrics_with_neighbors(self, attention_tile, input_tensor):
        """Test metrics with neighboring tiles."""
        # Create neighbor tile
        neighbor = AttentionTile(size=16, resolution=1.0, hidden_dim=32)
        attention_tile.add_neighbor(neighbor)

        # Process both tiles
        attention_tile._process_impl(input_tensor, update_metrics=True)
        neighbor._process_impl(input_tensor, update_metrics=True)

        # Check information flow metrics
        metrics = attention_tile.get_metrics()
        assert metrics["flow"] >= 0.0  # Should have some information flow
        assert metrics["ifq"] >= 0.0  # IFQ should account for neighbor interaction

    def test_metrics_during_adaptation(self, attention_tile, input_tensor):
        """Test metrics during resolution adaptation."""
        initial_metrics = {}
        adaptation_metrics = []

        # Initial state
        attention_tile._process_impl(input_tensor, update_metrics=True)
        initial_metrics = attention_tile.get_metrics().copy()

        # Simulate adaptation
        for _ in range(5):
            # Process with current resolution
            attention_tile._process_impl(input_tensor, update_metrics=True)

            # Adapt resolution based on density
            attention_tile.adapt_resolution(
                density_metric=0.5,
                strategy=ResolutionStrategy.ADAPTIVE,
            )

            # Store metrics
            adaptation_metrics.append(attention_tile.get_metrics().copy())

        # Verify adaptation is reflected in metrics
        assert len(adaptation_metrics) == 5
        assert adaptation_metrics[-1]["ae"] != initial_metrics["ae"]  # AE should change

    def test_load_balancing_metrics(self, attention_tile, input_tensor):
        """Test metrics during load balancing."""
        # Create neighbors with different loads
        neighbors = [
            AttentionTile(size=16, resolution=r, hidden_dim=32) for r in [0.8, 1.0, 0.6]
        ]
        for n in neighbors:
            attention_tile.add_neighbor(n)

        # Initial processing
        attention_tile._process_impl(input_tensor, update_metrics=True)
        initial_metrics = attention_tile.get_metrics().copy()

        # Process neighbors
        for n in neighbors:
            n._process_impl(input_tensor, update_metrics=True)

        # Balance load
        attention_tile.balance_load(neighbors)
        attention_tile._process_impl(input_tensor, update_metrics=True)
        balanced_metrics = attention_tile.get_metrics()

        # Check metrics reflect load balancing
        assert "load_distribution" in balanced_metrics
        assert balanced_metrics["cer"] != initial_metrics["cer"]

    def test_stress_conditions(self, attention_tile):
        """Test metrics under stress conditions."""
        # Test with very small input
        tiny_input = torch.randn(1, 2, 32)
        attention_tile._process_impl(tiny_input, update_metrics=True)

        # Check metrics excluding list/matrix metrics
        metrics = attention_tile.get_metrics()
        for key, value in metrics.items():
            if not isinstance(value, list):  # Skip list metrics like state_heatmap
                assert value >= 0.0, f"Metric {key} should be non-negative"

        # Test with larger input
        large_input = torch.randn(4, 16, 32)
        attention_tile._process_impl(large_input, update_metrics=True)
        metrics = attention_tile.get_metrics()
        for key, value in metrics.items():
            if not isinstance(value, list):  # Skip list metrics
                assert value >= 0.0, f"Metric {key} should be non-negative"

        # Test with different hidden dimensions
        diff_dim_input = torch.randn(2, 16, 16)  # Different hidden dim
        attention_tile._process_impl(diff_dim_input, update_metrics=True)
        metrics = attention_tile.get_metrics()
        for key, value in metrics.items():
            if not isinstance(value, list):  # Skip list metrics
                assert value >= 0.0, f"Metric {key} should be non-negative"
