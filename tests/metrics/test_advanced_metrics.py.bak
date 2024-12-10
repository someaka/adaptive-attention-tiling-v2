"""Tests for advanced metrics."""

import numpy as np
import pytest

from src.core.common.constants import (
    IFQ_DENSITY_WEIGHT,
    IFQ_EDGE_WEIGHT,
    IFQ_FLOW_WEIGHT,
    IFQ_PATTERN_WEIGHT,
)
from src.core.metrics.advanced_metrics import AdvancedMetricsAnalyzer


@pytest.fixture
def analyzer():
    """Create a metrics analyzer instance."""
    return AdvancedMetricsAnalyzer()


class TestInformationFlowQuality:
    """Test Information Flow Quality metric."""

    def test_ifq_perfect_score(self, analyzer):
        """Test IFQ calculation with perfect components."""
        ifq = analyzer.compute_ifq(
            pattern_stability=1.0,
            cross_tile_flow=1.0,
            edge_utilization=1.0,
            info_density=1.0,
        )
        assert ifq == pytest.approx(1.0)

    def test_ifq_zero_score(self, analyzer):
        """Test IFQ calculation with zero components."""
        ifq = analyzer.compute_ifq(
            pattern_stability=0.0,
            cross_tile_flow=0.0,
            edge_utilization=0.0,
            info_density=0.0,
        )
        assert ifq == pytest.approx(0.0)

    def test_ifq_weighted_components(self, analyzer):
        """Test IFQ weights are applied correctly."""
        ifq = analyzer.compute_ifq(
            pattern_stability=1.0,
            cross_tile_flow=0.0,
            edge_utilization=0.5,
            info_density=0.25,
        )
        expected = (
            IFQ_PATTERN_WEIGHT * 1.0
            + IFQ_FLOW_WEIGHT * 0.0
            + IFQ_EDGE_WEIGHT * 0.5
            + IFQ_DENSITY_WEIGHT * 0.25
        )
        assert ifq == pytest.approx(expected)

    def test_ifq_input_clipping(self, analyzer):
        """Test IFQ handles out-of-range inputs."""
        ifq = analyzer.compute_ifq(
            pattern_stability=1.5,  # Should be clipped to 1.0
            cross_tile_flow=-0.5,  # Should be clipped to 0.0
            edge_utilization=1.0,
            info_density=0.5,
        )
        assert 0 <= ifq <= 1


class TestComputationalEfficiencyRatio:
    """Test Computational Efficiency Ratio metric."""

    def test_cer_basic_calculation(self, analyzer):
        """Test basic CER calculation."""
        cer = analyzer.compute_cer(
            information_transferred=0.8,
            compute_cost=1.0,
            memory_usage=1024 * 1024,  # 1MB
            resolution=1.0,
        )
        assert cer > 0

    def test_cer_zero_compute_cost(self, analyzer):
        """Test CER handles zero compute cost."""
        cer = analyzer.compute_cer(
            information_transferred=0.5,
            compute_cost=0.0,
            memory_usage=1024,
            resolution=0.5,
        )
        assert cer > 0  # Should use MIN_COMPUTE instead of 0

    def test_cer_efficiency_scaling(self, analyzer):
        """Test CER scales with efficiency factors."""
        base_cer = analyzer.compute_cer(
            information_transferred=1.0,
            compute_cost=1.0,
            memory_usage=1024 * 1024,
            resolution=1.0,
        )

        # Higher compute cost should lower CER
        high_compute_cer = analyzer.compute_cer(
            information_transferred=1.0,
            compute_cost=2.0,
            memory_usage=1024 * 1024,
            resolution=1.0,
        )
        assert high_compute_cer < base_cer

        # Higher information transfer should increase CER
        high_info_cer = analyzer.compute_cer(
            information_transferred=2.0,
            compute_cost=1.0,
            memory_usage=1024 * 1024,
            resolution=1.0,
        )
        assert high_info_cer > base_cer


class TestAdaptationEffectiveness:
    """Test Adaptation Effectiveness metric."""

    def test_ae_empty_history(self, analyzer):
        """Test AE handles empty history."""
        ae = analyzer.compute_ae(
            resolution_history=[],
            load_variance_history=[],
        )
        assert ae == 0.0

    def test_ae_perfect_stability(self, analyzer):
        """Test AE with perfectly stable history."""
        ae = analyzer.compute_ae(
            resolution_history=[0.5] * 10,  # Constant resolution
            load_variance_history=[0.0] * 10,  # Perfect load balance
        )
        assert 0.6 < ae < 0.8  # Good but not perfect (we want some adaptivity)

    def test_ae_with_target(self, analyzer):
        """Test AE with target resolution."""
        # Resolution gradually approaches target
        resolutions = np.linspace(0.3, 0.5, 10)
        ae = analyzer.compute_ae(
            resolution_history=resolutions.tolist(),
            load_variance_history=[0.1] * 10,
            target_resolution=0.5,
        )
        assert 0 < ae < 1

    def test_ae_components(self, analyzer):
        """Test individual AE components."""
        # Create history with some controlled variation
        resolutions = [0.5 + 0.1 * np.sin(i) for i in range(10)]
        variances = [0.1 + 0.05 * np.cos(i) for i in range(10)]

        ae = analyzer.compute_ae(
            resolution_history=resolutions,
            load_variance_history=variances,
        )

        # Should penalize oscillation but reward responsiveness
        assert 0.2 < ae < 0.8


def test_metrics_history(analyzer):
    """Test metrics history management."""
    metrics = {"test": 1.0}
    analyzer.add_metrics(metrics)
    assert len(analyzer.get_history()) == 1

    analyzer.clear_history()
    assert len(analyzer.get_history()) == 0
