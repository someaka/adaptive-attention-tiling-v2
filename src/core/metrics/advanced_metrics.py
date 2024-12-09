"""Advanced Metrics for Pattern Analysis.

This module implements advanced metrics for analyzing pattern behavior:
- Information flow quality
- Pattern stability
- Cross-tile analysis
- Edge utilization
"""

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class InformationFlowMetrics:
    """Metrics for information flow analysis."""

    pattern_stability: float
    cross_tile_flow: float
    edge_utilization: float
    info_density: float

    def compute_ifq(self) -> float:
        """Compute Information Flow Quality."""
        return (
            self.pattern_stability
            * self.cross_tile_flow
            * self.edge_utilization
            * self.info_density
        )


class AdvancedMetricsAnalyzer:
    """Analyzer for advanced pattern metrics."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "ifq": [],  # Information Flow Quality
            "stability": [],  # Pattern Stability
            "flow": [],  # Cross-tile Flow
            "edge": [],  # Edge Utilization
            "density": [],  # Information Density
        }

    def compute_pattern_stability(
        self, pattern: torch.Tensor, window_size: int = 10
    ) -> float:
        """Compute pattern stability over time window."""
        if len(self.history["stability"]) < window_size:
            stability = float(torch.std(pattern))
        else:
            history_tensor = torch.tensor(self.history["stability"][-window_size:])
            stability = float(torch.std(torch.cat([history_tensor, pattern.view(-1)])))

        self.history["stability"].append(stability)
        return stability

    def compute_cross_tile_flow(self, pattern: torch.Tensor, tile_size: int) -> float:
        """Compute information flow between tiles."""
        # Reshape into tiles
        batch_size, seq_len, hidden_dim = pattern.shape
        num_tiles = seq_len // tile_size
        tiles = pattern.view(batch_size, num_tiles, tile_size, hidden_dim)

        # Compute flow between adjacent tiles
        flow = torch.mean(torch.abs(tiles[:, 1:] - tiles[:, :-1]))
        self.history["flow"].append(float(flow))
        return float(flow)

    def compute_edge_utilization(
        self, pattern: torch.Tensor, edge_threshold: float = 0.1
    ) -> float:
        """Compute edge attention utilization."""
        edge_mask = torch.abs(pattern) > edge_threshold
        utilization = float(torch.mean(edge_mask.float()))
        self.history["edge"].append(utilization)
        return utilization

    def compute_info_density(self, pattern: torch.Tensor) -> float:
        """Compute information density."""
        density = float(torch.mean(torch.abs(pattern)))
        self.history["density"].append(density)
        return density

    def compute_ifq(
        self,
        pattern: torch.Tensor,
        tile_size: int = 64,
        window_size: int = 10,
        edge_threshold: float = 0.1,
    ) -> InformationFlowMetrics:
        """Compute all information flow metrics."""
        metrics = InformationFlowMetrics(
            pattern_stability=self.compute_pattern_stability(pattern, window_size),
            cross_tile_flow=self.compute_cross_tile_flow(pattern, tile_size),
            edge_utilization=self.compute_edge_utilization(pattern, edge_threshold),
            info_density=self.compute_info_density(pattern),
        )

        ifq = metrics.compute_ifq()
        self.history["ifq"].append(ifq)

        return metrics
