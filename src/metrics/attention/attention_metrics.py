"""Attention metrics for quantum geometric attention.

This module provides metrics for analyzing attention patterns in the
quantum geometric attention framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from ..quantum_geometric_metrics import (
    MetricDomain,
    BaseMetric,
    MetricContext,
    GeometricMetrics,
    PatternMetrics
)
from .flow_metrics import (
    compute_flow_energy,
    compute_geodesic_distance,
    compute_parallel_transport
)
from ...core.patterns.formation import BifurcationMetrics, BifurcationAnalyzer
from ...core.patterns.symplectic import SymplecticStructure


@dataclass
class AttentionMetrics:
    """Metrics for attention patterns.
    
    Contains geometric, flow, and pattern metrics for attention analysis:
    - Geometric metrics: metric_tensor, curvature_tensor, connection
    - Flow metrics: flow_energy, flow_divergence, geometric_flow
    - Pattern metrics: pattern_entropy, pattern_complexity, pattern_stability
    """
    
    # Geometric metrics
    metric_tensor: Tensor  # Riemannian metric tensor
    curvature_tensor: Tensor  # Riemann/Ricci curvature tensor
    connection: Tensor  # Parallel transport connection
    
    # Flow metrics
    flow_energy: Tensor  # Energy of attention flow
    flow_divergence: Tensor  # Divergence of flow field
    geometric_flow: Tensor  # Ricci flow metrics
    
    # Pattern metrics
    pattern_entropy: Tensor  # Pattern formation entropy
    pattern_complexity: Tensor  # Structural complexity
    pattern_stability: Tensor  # Dynamic stability

    @property
    def entropy(self) -> Tensor:
        """Alias for pattern_entropy."""
        return self.pattern_entropy

    @property
    def complexity(self) -> Tensor:
        """Alias for pattern_complexity."""
        return self.pattern_complexity


def compute_attention_metrics(
    attention_patterns: Tensor,
    metric_context: MetricContext,
) -> AttentionMetrics:
    """Compute attention metrics for given attention patterns.
    
    Args:
        attention_patterns: Attention pattern tensor
        metric_context: Context for metric computation
        
    Returns:
        AttentionMetrics containing computed metrics
    """
    # Initialize metrics computers
    geometric_metrics = GeometricMetrics(
        manifold_dim=metric_context.hidden_dim,
        hidden_dim=metric_context.hidden_dim
    )
    pattern_metrics = PatternMetrics()
    bifurcation_analyzer = BifurcationAnalyzer()
    
    # Compute geometric metrics
    metric_tensor = torch.eye(metric_context.hidden_dim)  # Base metric
    curvature_tensor = geometric_metrics.compute_curvature(attention_patterns, metric_context)
    connection = compute_parallel_transport(attention_patterns, metric_tensor)
    
    # Compute flow metrics
    flow_energy = compute_flow_energy(attention_patterns, metric_tensor)
    flow_divergence = torch.zeros_like(attention_patterns)  # Placeholder
    geometric_flow = compute_geodesic_distance(attention_patterns, metric_tensor)
    
    # Compute pattern metrics
    pattern_entropy = pattern_metrics.compute_pattern_entropy(attention_patterns, metric_context)
    pattern_complexity = pattern_metrics.compute_pattern_complexity(attention_patterns, metric_context)
    
    # Compute stability metrics using bifurcation analysis
    stability_metrics = bifurcation_analyzer._compute_stability_metrics(attention_patterns)
    pattern_stability = torch.tensor(stability_metrics.stability_margin)
    
    return AttentionMetrics(
        metric_tensor=metric_tensor,
        curvature_tensor=curvature_tensor,
        connection=connection,
        flow_energy=flow_energy,
        flow_divergence=flow_divergence,
        geometric_flow=geometric_flow,
        pattern_entropy=pattern_entropy,
        pattern_complexity=pattern_complexity,
        pattern_stability=pattern_stability
    )


class AttentionMetricTracker(BaseMetric[AttentionMetrics]):
    """Tracker for attention metrics."""

    def __init__(self):
        """Initialize attention metric tracker."""
        super().__init__("attention", MetricDomain.ATTENTION)
        self.reset()

    def update(self, value: AttentionMetrics, context: MetricContext) -> None:
        """Update metrics with new values."""
        self.history.append(value)

    def compute(self) -> AttentionMetrics:
        """Compute aggregate metrics."""
        if not self.history:
            raise ValueError("No metrics have been accumulated")
            
        # Average tensors across history
        metrics_dict = {
            "metric_tensor": torch.stack([m.metric_tensor for m in self.history]).mean(0),
            "curvature_tensor": torch.stack([m.curvature_tensor for m in self.history]).mean(0),
            "connection": torch.stack([m.connection for m in self.history]).mean(0),
            "flow_energy": torch.stack([m.flow_energy for m in self.history]).mean(0),
            "flow_divergence": torch.stack([m.flow_divergence for m in self.history]).mean(0),
            "geometric_flow": torch.stack([m.geometric_flow for m in self.history]).mean(0),
            "pattern_entropy": torch.stack([m.pattern_entropy for m in self.history]).mean(0),
            "pattern_complexity": torch.stack([m.pattern_complexity for m in self.history]).mean(0),
            "pattern_stability": torch.stack([m.pattern_stability for m in self.history]).mean(0)
        }
        
        return AttentionMetrics(**metrics_dict)

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.history: List[AttentionMetrics] = [] 