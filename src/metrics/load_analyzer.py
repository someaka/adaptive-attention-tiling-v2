"""Load analyzer module."""

import logging
from typing import Any

import numpy as np
import torch

from src.core.common.constants import (
    LOAD_VARIANCE_THRESHOLD,
    RELATIVE_LOAD_HIGH,
    RELATIVE_LOAD_LOW,
)
from src.core.tiling.base import AttentionTile

logger = logging.getLogger(__name__)

# Constants
MIN_VARIANCE_SAMPLES = 2
CONVERGENCE_WINDOW = 5


class LoadBalanceAnalyzer:
    """Analyze load balancing metrics from attention tiles."""

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._history: list[dict] = []
        self.variance_threshold = LOAD_VARIANCE_THRESHOLD
        self.overload_threshold = RELATIVE_LOAD_HIGH
        self.underload_threshold = RELATIVE_LOAD_LOW

    @property
    def history(self) -> list[dict]:
        """Get the history of metrics snapshots."""
        return self._history

    def add_metrics(self, metrics: dict) -> None:
        """Add metrics snapshot to history."""
        self._history.append(metrics)

    def get_load_distribution(self) -> tuple[float, float, float]:
        """Get load distribution statistics over time.

        Returns:
            Tuple of (mean_load, std_load, max_load_diff)

        """
        if not self._history:
            return 0.0, 0.0, 0.0

        # Collect all loads
        loads = []
        for metrics in self._history:
            if "load_balance" in metrics:
                lb = metrics["load_balance"]
                loads.append(lb["local_load"])
                if "neighbor_loads" in lb:
                    if isinstance(lb["neighbor_loads"], list):
                        loads.extend(lb["neighbor_loads"])
                    else:
                        loads.append(lb["neighbor_loads"])

        if not loads:
            return 0.0, 0.0, 0.0

        loads_tensor = torch.tensor(loads, dtype=torch.float32)
        return (
            float(loads_tensor.mean()),
            float(loads_tensor.std()),
            float(torch.max(loads_tensor) - torch.min(loads_tensor)),
        )

    def get_convergence_rate(self) -> float:
        """Get rate of change in load variance."""
        recent = self._history[-10:]  # Look at last 10 samples

        variances = [
            metrics["load_balance"]["load_variance"]
            for metrics in recent
            if "load_balance" in metrics
        ]

        if len(variances) < MIN_VARIANCE_SAMPLES:
            return 0.0

        # Compute rate of change
        return (variances[-1] - variances[0]) / len(variances)

    def get_stability_score(self) -> float:
        """Get stability score based on load differences."""
        recent = self._history[-10:]  # Look at last 10 samples

        max_diffs = [
            metrics["load_balance"]["max_load_diff"]
            for metrics in recent
            if "load_balance" in metrics
        ]

        if not max_diffs:
            return 1.0  # Perfect stability if no history

        # Lower differences = higher stability
        avg_diff = sum(max_diffs) / len(max_diffs)
        return max(0.0, 1.0 - avg_diff)

    def analyze_network(self, tiles: list[AttentionTile]) -> dict[str, Any]:
        """Analyze the load distribution across a network of tiles.

        Args:
            tiles: List of attention tiles to analyze

        Returns:
            Dictionary containing network statistics

        """
        loads = [t._last_compute_cost for t in tiles]
        neighbor_counts = [len(t._neighbors) for t in tiles]

        load_stats = {
            "mean": float(np.mean(loads)),
            "std": float(np.std(loads)),
            "min": float(np.min(loads)),
            "max": float(np.max(loads)),
            "variance": float(np.var(loads)),
        }

        # Calculate balance score (1 - normalized variance)
        max_variance = 1.0  # Maximum possible variance for normalized loads
        balance_score = 1.0 - min(load_stats["variance"] / max_variance, 1.0)

        network_stats = {
            "num_tiles": len(tiles),
            "avg_neighbors": float(np.mean(neighbor_counts)),
            "load_stats": load_stats,
            "connectivity": float(np.mean(neighbor_counts) / max(1, len(tiles) - 1)),
            "balance_score": float(balance_score),
        }

        return {
            "load_stats": load_stats,
            "network_stats": network_stats,
        }

    def get_recommendations(self, tiles: list[AttentionTile]) -> list[str]:
        """Generate recommendations for load balancing based on network analysis."""
        stats = self.analyze_network(tiles)
        recommendations = []

        # Get variance either from history or current analysis
        variance = 0.0
        if self._history and "load_balance" in self._history[-1]:
            variance = self._history[-1]["load_balance"]["load_variance"]
        else:
            variance = stats["load_stats"]["variance"]

        # Check for high variance in loads
        if variance > LOAD_VARIANCE_THRESHOLD:
            recommendations.append(
                f"High load variance ({variance:.2f}) detected. "
                "Consider adjusting tile resolutions to balance loads."
            )

        # Check for overloaded tiles
        if stats["load_stats"]["max"] > RELATIVE_LOAD_HIGH:
            recommendations.append(
                "Some tiles are overloaded "
                "Consider reducing resolution or adjusting load balancing parameters."
            )

        # Check for underutilized tiles
        if stats["load_stats"]["min"] < RELATIVE_LOAD_LOW:
            recommendations.append(
                "Some tiles are underutilized "
                "Consider increasing resolution or adjusting load balancing parameters."
            )

        # Check convergence
        if len(self._history) >= CONVERGENCE_WINDOW:
            recent_variance = [
                h["load_balance"]["load_variance"]
                for h in self._history[-CONVERGENCE_WINDOW:]
            ]
            if all(v > LOAD_VARIANCE_THRESHOLD for v in recent_variance):
                recommendations.append(
                    "Persistent high variance in load distribution. "
                    "Consider more aggressive load balancing."
                )

        return recommendations
