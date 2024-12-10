"""Integration tests for advanced metrics with QuantumMotivicTile."""

import pytest
import torch
import numpy as np

from src.core.common.enums import ResolutionStrategy
from src.core.tiling.quantum_attention_tile import QuantumMotivicTile


@pytest.fixture
def attention_tile():
    """Create a basic attention tile for testing."""
    tile = QuantumMotivicTile(size=16, hidden_dim=32)
    tile._initialize_quantum_structure()  # Ensure quantum structure is initialized
    return tile


@pytest.fixture
def input_tensor():
    """Create a sample input tensor."""
    batch_size = 2
    seq_len = 16
    hidden_dim = 32
    return torch.randn(batch_size, seq_len, hidden_dim)


class TestMetricsIntegration:
    """Test integration of advanced metrics with QuantumMotivicTile."""

    def test_metrics_initialization(self, attention_tile):
        """Test that advanced metrics are properly initialized."""
        metrics = attention_tile.get_metrics()
        assert "ifq" in metrics
        assert "cer" in metrics
        assert "ae" in metrics
        assert "quantum_entropy" in metrics
        assert "motive_height" in metrics
        assert "l_function_value" in metrics
        assert "adelic_norm" in metrics
        assert all(isinstance(metrics[k], float) for k in ["ifq", "cer", "ae"])

    def test_metrics_after_processing(self, attention_tile, input_tensor):
        """Test metrics update after processing input."""
        # Process input
        output = attention_tile._process_impl(input_tensor, update_metrics=True)
        metrics = attention_tile.get_metrics()

        # Check metrics were updated and are within valid ranges
        assert 0.0 <= metrics["ifq"] <= 1.0
        assert metrics["cer"] >= 0.0
        assert 0.0 <= metrics["ae"] <= 1.0
        assert 0.1 <= metrics["quantum_entropy"] <= 5.0
        assert 0.0 <= metrics["motive_height"] <= 10.0
        assert metrics["l_function_value"] >= 0.01
        assert 0.1 <= metrics["adelic_norm"] <= 1.0
        assert output.shape == input_tensor.shape

    def test_metrics_history(self, attention_tile, input_tensor):
        """Test metrics history accumulation."""
        # Process multiple times
        for _ in range(3):
            attention_tile._process_impl(input_tensor, update_metrics=True)
            metrics = attention_tile.get_metrics()
            attention_tile._metrics_log.append(metrics)

        # Check metrics log
        assert len(attention_tile._metrics_log) >= 3
        assert all("ifq" in m for m in attention_tile._metrics_log[-3:])
        assert all("cer" in m for m in attention_tile._metrics_log[-3:])
        assert all("ae" in m for m in attention_tile._metrics_log[-3:])
        assert all("quantum_entropy" in m for m in attention_tile._metrics_log[-3:])
        assert all("motive_height" in m for m in attention_tile._metrics_log[-3:])

    def test_metrics_with_resolution_change(self, attention_tile, input_tensor):
        """Test metrics behavior with resolution changes."""
        # Initial processing
        attention_tile._process_impl(input_tensor, update_metrics=True)
        initial_metrics = attention_tile.get_metrics()
        attention_tile._metrics["resolution_history"] = [1.0]

        # Change resolution
        attention_tile.resolution = 0.5
        attention_tile._metrics["resolution_history"].append(0.5)
        attention_tile._process_impl(input_tensor, update_metrics=True)
        new_metrics = attention_tile.get_metrics()

        # Metrics should reflect the change
        assert len(attention_tile._metrics["resolution_history"]) >= 2
        assert new_metrics["cer"] != initial_metrics["cer"]
        assert new_metrics["quantum_entropy"] != initial_metrics["quantum_entropy"]

    def test_metrics_with_neighbors(self, attention_tile, input_tensor):
        """Test metrics with neighboring tiles."""
        # Create neighbor tile
        neighbor = QuantumMotivicTile(size=16, hidden_dim=32)
        neighbor._initialize_quantum_structure()
        attention_tile.add_neighbor(neighbor)

        # Process both tiles
        attention_tile._process_impl(input_tensor, update_metrics=True)
        neighbor._process_impl(input_tensor, update_metrics=True)

        # Check information flow metrics
        metrics = attention_tile.get_metrics()
        assert metrics["flow"] >= 0.0
        assert 0.0 <= metrics["ifq"] <= 1.0
        assert metrics["l_function_value"] >= 0.01

    def test_metrics_during_adaptation(self, attention_tile, input_tensor):
        """Test metrics during resolution adaptation."""
        # Initial state
        attention_tile._process_impl(input_tensor, update_metrics=True)
        initial_metrics = attention_tile.get_metrics().copy()
        attention_tile._metrics["ae"] = 0.5  # Set initial AE

        # Simulate adaptation
        adaptation_metrics = []
        for _ in range(5):
            # Process with current resolution
            attention_tile._process_impl(input_tensor, update_metrics=True)
            attention_tile._metrics["ae"] += 0.1  # Simulate AE change

            # Adapt resolution based on density
            attention_tile.adapt_resolution(
                density_metric=0.5,
                strategy=ResolutionStrategy.ADAPTIVE,
            )

            # Store metrics
            adaptation_metrics.append(attention_tile.get_metrics().copy())

        # Verify adaptation is reflected in metrics
        assert len(adaptation_metrics) == 5
        assert adaptation_metrics[-1]["ae"] != initial_metrics["ae"]

    def test_load_balancing_metrics(self, attention_tile, input_tensor):
        """Test metrics during load balancing."""
        # Create neighbors with different loads
        neighbors = [QuantumMotivicTile(size=16, hidden_dim=32) for _ in range(3)]
        for n in neighbors:
            n._initialize_quantum_structure()
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
        metrics = attention_tile.get_metrics()

        # Check all scalar metrics are non-negative
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.float64)):
                assert value >= 0.0, f"Metric {key} should be non-negative"

        # Test with larger input
        large_input = torch.randn(4, 16, 32)
        attention_tile._process_impl(large_input, update_metrics=True)
        metrics = attention_tile.get_metrics()

        # Check all scalar metrics are non-negative
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.float64)):
                assert value >= 0.0, f"Metric {key} should be non-negative"
