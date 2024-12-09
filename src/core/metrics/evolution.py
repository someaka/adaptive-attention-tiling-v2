"""Evolution Metrics for Pattern Analysis.

This module implements evolution metrics for analyzing pattern dynamics:
- L-function computation
- Flow evolution
- Orbit analysis
- Ergodic averages
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn


@dataclass
class EvolutionMetrics:
    """Metrics for pattern evolution."""

    l_values: torch.Tensor
    flow_metrics: torch.Tensor
    orbit_stats: Dict[str, float]
    ergodic_avg: torch.Tensor


class LFunctionComputation:
    """Compute L-functions for pattern analysis."""

    def __init__(self, hidden_dim: int, rank: int = 4, num_factors: int = 8):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_factors = num_factors

        # L-function network
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, rank * num_factors),
            nn.ReLU(),
            nn.Linear(rank * num_factors, rank),
        )

    def compute_l_values(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute L-function values."""
        return self.network(pattern)


class FlowEvolution:
    """Analyze pattern flow evolution."""

    def __init__(self, hidden_dim: int, flow_dim: int = 4):
        self.hidden_dim = hidden_dim
        self.flow_dim = flow_dim

        # Flow computation
        self.flow = nn.Linear(hidden_dim, flow_dim)

        # Evolution tracking
        self.history: List[torch.Tensor] = []

    def compute_flow(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute flow metrics."""
        flow_metrics = self.flow(pattern)
        self.history.append(flow_metrics)
        return flow_metrics

    def get_evolution_stats(self, window_size: int = 10) -> Dict[str, torch.Tensor]:
        """Get evolution statistics."""
        if len(self.history) < 2:
            return {
                "velocity": torch.zeros(self.flow_dim),
                "acceleration": torch.zeros(self.flow_dim),
                "stability": torch.zeros(self.flow_dim),
            }

        history_tensor = torch.stack(self.history[-window_size:])
        velocity = history_tensor[1:] - history_tensor[:-1]
        acceleration = velocity[1:] - velocity[:-1]

        return {
            "velocity": velocity.mean(0),
            "acceleration": acceleration.mean(0),
            "stability": history_tensor.std(0),
        }


class OrbitAnalysis:
    """Analyze pattern orbits."""

    def __init__(self, hidden_dim: int, orbit_dim: int = 2):
        self.hidden_dim = hidden_dim
        self.orbit_dim = orbit_dim

        # Orbit projection
        self.projection = nn.Linear(hidden_dim, orbit_dim)

        # Orbit history
        self.history: List[torch.Tensor] = []

    def analyze_orbit(self, pattern: torch.Tensor) -> Dict[str, float]:
        """Analyze pattern orbit."""
        orbit_point = self.projection(pattern)
        self.history.append(orbit_point)

        if len(self.history) < 2:
            return {
                "period": 0.0,
                "radius": float(torch.norm(orbit_point)),
                "stability": 0.0,
            }

        history_tensor = torch.stack(self.history)

        # Compute orbit statistics
        radius = float(torch.norm(orbit_point))
        period = self._estimate_period(history_tensor)
        stability = float(torch.std(history_tensor[-10:]))

        return {"period": period, "radius": radius, "stability": stability}

    def _estimate_period(self, orbit: torch.Tensor) -> float:
        """Estimate orbit period using autocorrelation."""
        if len(orbit) < 4:
            return 0.0

        # Compute autocorrelation
        mean = orbit.mean(0)
        std = orbit.std(0)
        normalized = (orbit - mean) / (std + 1e-8)

        acf = torch.zeros(len(orbit) // 2)
        for lag in range(len(acf)):
            correlation = torch.corrcoef(
                normalized[lag:].flatten(),
                normalized[: -lag if lag > 0 else None].flatten(),
            )[0, 1]
            acf[lag] = correlation

        # Find first peak after lag 0
        peaks = ((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])).nonzero()
        return float(peaks[0] + 1) if len(peaks) > 0 else 0.0


class ErgodicAnalysis:
    """Analyze ergodic properties of patterns."""

    def __init__(self, hidden_dim: int, num_observables: int = 4):
        self.hidden_dim = hidden_dim
        self.num_observables = num_observables

        # Observable functions
        self.observables = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_observables)
            ]
        )

        # History of observations
        self.history: List[torch.Tensor] = []

    def compute_ergodic_average(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute ergodic averages of observables."""
        # Compute current observables
        current = torch.cat([obs(pattern) for obs in self.observables])
        self.history.append(current)

        # Compute time average
        time_avg = torch.stack(self.history).mean(0)

        return time_avg


class EvolutionAnalyzer:
    """Complete evolution analysis system."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 4,
        flow_dim: int = 4,
        orbit_dim: int = 2,
        num_observables: int = 4,
    ):
        self.l_function = LFunctionComputation(hidden_dim, rank)
        self.flow = FlowEvolution(hidden_dim, flow_dim)
        self.orbit = OrbitAnalysis(hidden_dim, orbit_dim)
        self.ergodic = ErgodicAnalysis(hidden_dim, num_observables)

    def analyze_evolution(self, pattern: torch.Tensor) -> EvolutionMetrics:
        """Perform complete evolution analysis."""
        return EvolutionMetrics(
            l_values=self.l_function.compute_l_values(pattern),
            flow_metrics=self.flow.compute_flow(pattern),
            orbit_stats=self.orbit.analyze_orbit(pattern),
            ergodic_avg=self.ergodic.compute_ergodic_average(pattern),
        )
