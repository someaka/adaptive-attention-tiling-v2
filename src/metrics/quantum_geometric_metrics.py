"""Quantum Geometric Metrics Framework.

This module implements the core metrics infrastructure for:
- Quantum cohomological measurements
- Arithmetic height computations
- Geometric flow characteristics
- Pattern dynamics analysis

The focus is on capturing the deep mathematical structure
rather than visualization - providing a rich API for future
visualization projects to build upon.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Type variables for generic metrics
T = TypeVar("T")
MetricValue = TypeVar("MetricValue", float, torch.Tensor, np.ndarray)


class MetricDomain(Enum):
    """Domains for different types of metrics."""

    QUANTUM = "quantum"
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    PATTERN = "pattern"
    RESOURCE = "resource"
    ATTENTION = "attention"  # Added for attention-specific metrics


# @dataclass
# class ValidationConfig:
#     """Configuration for validation metrics."""
#     # Quantum parameters
#     cohomology_dim: int = 8
#     motive_rank: int = 4

#     # Geometric parameters
#     manifold_dim: int = 32
#     num_charts: int = 4

#     # Arithmetic parameters
#     num_primes: int = 8
#     height_dim: int = 4

#     # Pattern parameters
#     pattern_dim: int = 64
#     stability_threshold: float = 1e-6

#     # Resource parameters
#     max_memory_gb: float = 16.0
#     max_compute_time_ms: float = 1000.0

#     def to_dict(self) -> Dict:
#         """Convert config to dictionary."""
#         return {
#             "cohomology_dim": self.cohomology_dim,
#             "motive_rank": self.motive_rank,
#             "manifold_dim": self.manifold_dim,
#             "num_charts": self.num_charts,
#             "num_primes": self.num_primes,
#             "height_dim": self.height_dim,
#             "pattern_dim": self.pattern_dim,
#             "stability_threshold": self.stability_threshold,
#             "max_memory_gb": self.max_memory_gb,
#             "max_compute_time_ms": self.max_compute_time_ms
#         }

#     @classmethod
#     def from_dict(cls, config_dict: Dict) -> "ValidationConfig":
#         """Create config from dictionary."""
#         return cls(**config_dict)


@dataclass
class MetricContext:
    """Context information for metric computation."""

    timestamp: float
    device: torch.device
    batch_size: int
    sequence_length: int
    hidden_dim: int
    resolution: float


class BaseMetric(Generic[T]):
    """Base class for all metrics."""

    def __init__(self, name: str, domain: MetricDomain):
        self.name = name
        self.domain = domain
        self.history: List[T] = []

    def update(self, value: T, context: MetricContext) -> None:
        """Update metric with new value."""
        self.history.append(value)

    def compute(self) -> T:
        """Compute current metric value."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset metric history."""
        self.history.clear()


class QuantumMetrics:
    """Quantum cohomological metrics."""

    def __init__(self, hidden_dim: int, motive_rank: int):
        self.hidden_dim = hidden_dim
        self.motive_rank = motive_rank

    def compute_cohomology_class(
        self, attention_patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute cohomology class of attention patterns."""
        # Project to cohomology space
        batch_size = attention_patterns.size(0)
        seq_len = attention_patterns.size(1)
        hidden_dim = attention_patterns.size(2)

        # Reshape to preserve memory layout
        reshaped = attention_patterns.reshape(batch_size * seq_len, hidden_dim)

        # Project to motive space
        if not hasattr(self, "motive_rank"):
            self.motive_rank = 1

        return reshaped.reshape(batch_size, -1)

    def compute_quantum_entropy(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute quantum entropy of patterns."""
        probs = torch.softmax(patterns, dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    def compute_motive_height(
        self, cohomology: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute motivic height."""
        return torch.norm(cohomology, p=2, dim=-1)


class ArithmeticMetrics:
    """Arithmetic dynamics metrics."""

    def __init__(self, num_primes: int = 8, motive_rank: int = 4, hidden_dim: int = 64):
        self.num_primes = num_primes
        self.motive_rank = motive_rank
        self.hidden_dim = hidden_dim

        # Initialize prime bases
        self.prime_bases = nn.Parameter(torch.randn(num_primes, motive_rank))

        # Motive space projection
        self.motive_proj = nn.Linear(hidden_dim, motive_rank)

    def compute_local_height(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute local height at finite places."""
        batch_size, seq_len, hidden_dim = patterns.shape

        # Project to motive space
        patterns_flat = patterns.reshape(-1, hidden_dim)
        motive_coords = self.motive_proj(patterns_flat)

        # Project to prime bases
        prime_coords = torch.matmul(motive_coords, self.prime_bases.T)
        prime_coords = prime_coords.view(batch_size, seq_len, self.num_primes)

        # Compute local height as p-adic norm
        local_height = torch.norm(prime_coords, dim=-1)

        return local_height

    def compute_global_height(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute global height combining all places."""
        local_height = self.compute_local_height(patterns, context)
        global_height = torch.mean(local_height, dim=-1)
        return global_height

    def compute_height_distribution(
        self, patterns: torch.Tensor, context: MetricContext, num_bins: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distribution of heights."""
        local_height = self.compute_local_height(patterns, context)

        # Compute histogram
        min_height = local_height.min()
        max_height = local_height.max()

        bins = torch.linspace(min_height, max_height, num_bins + 1)
        heights = local_height.reshape(-1)

        # Count heights in each bin
        counts = torch.zeros(num_bins)
        for i in range(num_bins):
            mask = (heights >= bins[i]) & (heights < bins[i + 1])
            counts[i] = mask.sum()

        return bins[:-1], counts

    def compute_l_function(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute L-function values."""
        # Simplified L-function computation
        return torch.norm(patterns, p=2, dim=-1)

    def compute_adelic_norm(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute adelic norm."""
        return torch.norm(patterns, p=float("inf"), dim=-1)


class GeometricMetrics:
    """Geometric flow metrics."""

    def __init__(self, manifold_dim: int, hidden_dim: int):
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim

    def compute_geodesic_distance(
        self, flow_path: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute geodesic distance along flow."""
        if flow_path is None or len(flow_path) < 2:
            return torch.tensor(0.0)

        # Compute distances between consecutive points
        distances = []
        for i in range(len(flow_path) - 1):
            dist = torch.norm(flow_path[i + 1] - flow_path[i], dim=-1).mean()
            distances.append(dist)

        if not distances:
            return torch.tensor(0.0)

        return torch.stack(distances).sum()

    def compute_parallel_transport(
        self, vector: torch.Tensor, connection: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute parallel transport of vector."""
        return vector - torch.einsum("bijkl,bl->bi", connection, vector)

    def compute_curvature(
        self, flow_path: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute curvature of flow."""
        if flow_path is None or len(flow_path) < 3:
            return torch.tensor(0.0)

        # Compute second derivatives
        velocities = []
        for i in range(len(flow_path) - 1):
            vel = flow_path[i + 1] - flow_path[i]
            velocities.append(vel)

        accelerations = []
        for i in range(len(velocities) - 1):
            acc = velocities[i + 1] - velocities[i]
            accelerations.append(acc)

        if not accelerations:
            return torch.tensor(0.0)

        # Compute curvature as norm of acceleration
        curvature = torch.stack(
            [torch.norm(acc, dim=-1).mean() for acc in accelerations]
        ).mean()

        return curvature


class PatternMetrics:
    """Pattern-based metrics."""

    def compute_pattern_entropy(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute pattern entropy metric."""
        # Normalize patterns
        probs = F.softmax(patterns.view(patterns.shape[0], -1), dim=-1)

        # Compute entropy and normalize by log(dim) to bound between 0 and 1
        max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1) / max_entropy

        return entropy

    def compute_pattern_complexity(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute pattern complexity metric.

        Combines pattern entropy with temporal gradient information to measure
        the overall complexity of pattern evolution.

        Args:
            patterns: Pattern tensor of shape (batch_size, time_steps, seq_len, hidden_dim)
            context: Metric computation context

        Returns:
            Complexity metric of shape (batch_size,) normalized between 0 and 1
        """
        batch_size = patterns.shape[0]

        # Reshape patterns if needed
        if len(patterns.shape) == 4:  # Already in (batch, time, seq, hidden)
            patterns_4d = patterns
        else:  # Need to add time dimension
            patterns_4d = patterns.unsqueeze(1)  # (batch, 1, seq, hidden)

        # Use combination of entropy and gradient information
        entropy = self.compute_pattern_entropy(patterns_4d, context)  # (batch_size,)

        # Compute temporal gradients across time dimension if we have multiple timesteps
        if patterns_4d.shape[1] > 1:
            gradients = torch.norm(
                patterns_4d[:, 1:]
                - patterns_4d[:, :-1],  # (batch_size, time-1, seq_len, hidden_dim)
                dim=(-2, -1),  # Average over seq_len and hidden_dim
            ).mean(
                dim=1
            )  # Average over time, result is (batch_size,)
        else:
            # For single timestep, use spatial gradients
            spatial_diff = (
                patterns_4d[:, :, 1:] - patterns_4d[:, :, :-1]
            )  # (batch, 1, seq-1, hidden)
            gradients = torch.norm(spatial_diff, dim=(-2, -1)).squeeze(1)  # (batch,)

        # Normalize gradients by pattern magnitude to get relative change
        pattern_magnitude = torch.norm(patterns_4d, dim=(-2, -1)).mean(
            dim=1
        )  # (batch,)
        gradients = gradients / (pattern_magnitude + 1e-10)

        # Apply sigmoid to bound gradients between 0 and 1
        gradients = torch.sigmoid(gradients)

        # Combine metrics - both are now (batch_size,) and normalized between 0 and 1
        complexity = entropy * gradients

        # Apply tanh and scale to ensure final complexity is between 0 and 1
        complexity = 0.5 * (torch.tanh(complexity) + 1.0)

        return complexity

    def compute_pattern_stability(
        self, patterns: torch.Tensor, context: MetricContext
    ) -> torch.Tensor:
        """Compute pattern stability metric."""
        # Use inverse of gradient magnitude
        gradients = torch.norm(patterns[:, 1:] - patterns[:, :-1], dim=-1).mean(dim=1)

        stability = 1.0 / (gradients + 1e-10)
        return stability


class UnifiedMetrics:
    """Unified metrics framework combining all perspectives."""

    def __init__(
        self,
        hidden_dim: int = 64,  # Changed from 256 to 64
        num_heads: int = 8,
        num_primes: int = 8,
        motive_rank: int = 4,
        manifold_dim: int = 32,
        num_bins: int = 100,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_primes = num_primes
        self.motive_rank = motive_rank
        self.manifold_dim = manifold_dim
        self.num_bins = num_bins

        # Initialize component metrics
        self.quantum = QuantumMetrics(hidden_dim=hidden_dim, motive_rank=motive_rank)
        self.arithmetic = ArithmeticMetrics(
            num_primes=num_primes, motive_rank=motive_rank, hidden_dim=hidden_dim
        )
        self.geometric = GeometricMetrics(
            manifold_dim=manifold_dim, hidden_dim=hidden_dim
        )
        self.pattern = PatternMetrics()

    def compute_all_metrics(
        self, data: Dict[str, torch.Tensor], context: MetricContext
    ) -> Dict[str, torch.Tensor]:
        """Compute all metrics for given data."""
        metrics = {}

        # Quantum metrics
        if "attention_patterns" in data:
            metrics["cohomology_class"] = self.quantum.compute_cohomology_class(
                data["attention_patterns"], context
            )
            metrics["quantum_entropy"] = self.quantum.compute_quantum_entropy(
                data["attention_patterns"], context
            )

        # Arithmetic metrics
        if "patterns" in data:
            metrics["local_height"] = self.arithmetic.compute_local_height(
                data["patterns"], context
            )
            metrics["l_function"] = self.arithmetic.compute_l_function(
                data["patterns"], context
            )
            metrics["adelic_norm"] = self.arithmetic.compute_adelic_norm(
                data["patterns"], context
            )

        # Geometric metrics
        if "flow_path" in data and data["flow_path"] is not None:
            metrics["geodesic_distance"] = self.geometric.compute_geodesic_distance(
                data["flow_path"], context
            )
            metrics["curvature"] = self.geometric.compute_curvature(
                data["flow_path"], context
            )

        # Pattern evolution metrics
        if "pattern_history" in data:
            pattern_history = data["pattern_history"]
            if not isinstance(pattern_history, list):
                pattern_history = [pattern_history]  # Convert single tensor to list
            metrics["pattern_evolution"] = self.compute_pattern_evolution(
                pattern_history, context
            )

        return metrics

    def compute_pattern_evolution(
        self, pattern_history: List[torch.Tensor], context: MetricContext
    ) -> torch.Tensor:
        """Compute pattern evolution metrics."""
        if not pattern_history or len(pattern_history) < 2:
            return torch.tensor(0.0)

        # Compute differences between consecutive patterns
        diffs = []
        for i in range(len(pattern_history) - 1):
            if pattern_history[i] is not None and pattern_history[i + 1] is not None:
                diff = torch.norm(
                    pattern_history[i + 1] - pattern_history[i], dim=-1
                ).mean()
                diffs.append(diff)

        if not diffs:
            return torch.tensor(0.0)

        return torch.stack(diffs)
