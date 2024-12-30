"""Bifurcation metrics for quantum geometric attention.

This module provides metrics for analyzing bifurcations in the
quantum geometric attention framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from ..quantum_geometric_metrics import MetricDomain, BaseMetric, MetricContext


@dataclass
class BifurcationMetrics:
    """Metrics for bifurcation analysis."""
    stability_margin: torch.Tensor  # Shape: (batch_size,)
    max_eigenvalue: torch.Tensor  # Shape: (batch_size,)
    symplectic_invariant: torch.Tensor  # Shape: (batch_size,)
    quantum_metric: torch.Tensor  # Shape: (batch_size, hidden_dim, hidden_dim)
    pattern_height: torch.Tensor  # Shape: (batch_size,)
    geometric_flow: torch.Tensor  # Shape: (batch_size, hidden_dim)


class BifurcationMetricTracker(BaseMetric[BifurcationMetrics]):
    """Tracker for bifurcation metrics."""

    def __init__(self):
        """Initialize bifurcation metric tracker."""
        super().__init__("bifurcation", MetricDomain.PATTERN)
        self.reset()

    def update(self, value: BifurcationMetrics, context: MetricContext) -> None:
        """Update metrics with new values."""
        self.history.append(value)

    def compute(self) -> BifurcationMetrics:
        """Compute aggregate metrics."""
        if not self.history:
            raise ValueError("No metrics have been accumulated")
            
        # Average tensors across history
        metrics_dict = {
            "stability_margin": torch.stack([m.stability_margin for m in self.history]).mean(0),
            "max_eigenvalue": torch.stack([m.max_eigenvalue for m in self.history]).mean(0),
            "symplectic_invariant": torch.stack([m.symplectic_invariant for m in self.history]).mean(0),
            "quantum_metric": torch.stack([m.quantum_metric for m in self.history]).mean(0),
            "pattern_height": torch.stack([m.pattern_height for m in self.history]).mean(0),
            "geometric_flow": torch.stack([m.geometric_flow for m in self.history]).mean(0)
        }
        
        return BifurcationMetrics(**metrics_dict)

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.history = []


def compute_bifurcation_metrics(
    attention_patterns: torch.Tensor,
    metric_context: MetricContext
) -> BifurcationMetrics:
    """Compute bifurcation metrics from attention patterns.
    
    Args:
        attention_patterns: Attention pattern tensor
        metric_context: Metric computation context
        
    Returns:
        Computed bifurcation metrics
    """
    # Compute stability margin using eigenvalue analysis
    eigenvalues = torch.linalg.eigvals(attention_patterns)
    stability_margin = torch.min(torch.abs(eigenvalues.real), dim=-1)[0]
    max_eigenvalue = torch.max(torch.abs(eigenvalues), dim=-1)[0]
    
    # Compute symplectic invariant
    symplectic_form = torch.zeros_like(attention_patterns[..., :2, :2])
    symplectic_form[..., 0, 1] = 1.0
    symplectic_form[..., 1, 0] = -1.0
    symplectic_invariant = torch.einsum('...ij,...jk,...ki->...', 
                                      attention_patterns, 
                                      symplectic_form, 
                                      attention_patterns)
    
    # Compute quantum metric using geometric structure
    quantum_metric = torch.einsum('...i,...j->...ij', 
                                attention_patterns, 
                                attention_patterns)
    
    # Compute pattern height as norm
    pattern_height = torch.norm(attention_patterns, dim=(-2, -1))
    
    # Compute geometric flow as gradient
    geometric_flow = torch.autograd.grad(
        attention_patterns.sum(), 
        attention_patterns, 
        create_graph=True
    )[0].mean(dim=-2)
    
    return BifurcationMetrics(
        stability_margin=stability_margin,
        max_eigenvalue=max_eigenvalue,
        symplectic_invariant=symplectic_invariant,
        quantum_metric=quantum_metric,
        pattern_height=pattern_height,
        geometric_flow=geometric_flow
    ) 