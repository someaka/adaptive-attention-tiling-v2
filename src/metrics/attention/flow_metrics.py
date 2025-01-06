"""Flow metrics for quantum geometric attention.

This module provides metrics for analyzing geometric flow in the
quantum geometric attention framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from ..quantum_geometric_metrics import MetricDomain, BaseMetric, MetricContext


@dataclass
class FlowMetrics:
    """Metrics for geometric attention flow."""
    
    curvature: torch.Tensor  # Shape: (batch_size,)
    parallel_transport: torch.Tensor  # Shape: (batch_size, hidden_dim)
    geodesic_distance: torch.Tensor  # Shape: (batch_size,)
    energy: torch.Tensor  # Shape: (batch_size,)


class FlowMetricTracker(BaseMetric[FlowMetrics]):
    """Tracker for flow metrics."""

    def __init__(self):
        """Initialize flow metric tracker."""
        super().__init__("flow", MetricDomain.GEOMETRIC)
        self.reset()

    def update(self, value: FlowMetrics, context: MetricContext) -> None:
        """Update metrics with new values.
        
        Args:
            value: New flow metrics
            context: Metric computation context
        """
        # Accumulate metrics
        self._curvature.append(value.curvature)
        self._parallel_transport.append(value.parallel_transport)
        self._geodesic_distance.append(value.geodesic_distance)
        self._energy.append(value.energy)

    def compute(self) -> FlowMetrics:
        """Compute aggregate metrics.
        
        Returns:
            Aggregated flow metrics
        """
        if not self._curvature:
            raise ValueError("No metrics have been accumulated")

        # Stack and average metrics
        curvature = torch.stack(self._curvature).mean(dim=0)
        parallel_transport = torch.stack(self._parallel_transport).mean(dim=0)
        geodesic_distance = torch.stack(self._geodesic_distance).mean(dim=0)
        energy = torch.stack(self._energy).mean(dim=0)

        return FlowMetrics(
            curvature=curvature,
            parallel_transport=parallel_transport,
            geodesic_distance=geodesic_distance,
            energy=energy
        )

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self._curvature = []
        self._parallel_transport = []
        self._geodesic_distance = []
        self._energy = []


def compute_flow_metrics(
    flow_path: torch.Tensor,
    metric_tensor: torch.Tensor,
    context: MetricContext
) -> FlowMetrics:
    """Compute flow metrics from flow path and metric tensor.
    
    Args:
        flow_path: Flow path tensor
        metric_tensor: Metric tensor
        context: Metric computation context
        
    Returns:
        Computed flow metrics
    """
    # Compute curvature
    ricci = compute_ricci_tensor(metric_tensor)
    curvature = torch.einsum('...ii', ricci)

    # Compute parallel transport
    transport = compute_parallel_transport(flow_path, metric_tensor)

    # Compute geodesic distance
    distance = compute_geodesic_distance(flow_path, metric_tensor)

    # Compute energy
    energy = compute_flow_energy(flow_path, metric_tensor)

    return FlowMetrics(
        curvature=curvature,
        parallel_transport=transport,
        geodesic_distance=distance,
        energy=energy
    )


def compute_ricci_tensor(metric: torch.Tensor) -> torch.Tensor:
    """Compute Ricci tensor from metric tensor.

    Args:
        metric: Metric tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]

    Returns:
        Ricci tensor with same shape as input
    """
    # Handle both 3D and 4D inputs
    orig_shape = metric.shape
    if len(orig_shape) == 4:
        batch_size, seq_len, dim, _ = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, dim, dim)
    else:
        batch_size, dim, _ = orig_shape
        seq_len = None

    # Initialize Christoffel symbols
    christoffel = torch.zeros(batch_size if seq_len is None else batch_size * seq_len,
                            dim, dim, dim, dtype=metric.dtype, device=metric.device)

    # Compute inverse metric
    ginv = torch.inverse(metric)

    # Compute metric derivatives (approximated)
    eps = 1e-6
    for k in range(dim):
        # Create perturbation in k direction
        h = torch.zeros_like(metric)
        h[..., k, :] = eps

        # Compute finite difference approximation
        dg_k = (metric + h - metric) / eps

        # Store in Christoffel symbols
        for i in range(dim):
            for j in range(dim):
                # Γ^i_jk = 1/2 g^il (∂_j g_kl + ∂_k g_jl - ∂_l g_jk)
                christoffel[:, i, j, k] = 0.5 * torch.sum(
                    ginv[:, i, :] * (
                        dg_k[:, j, :] +
                        dg_k[:, j, :] -
                        dg_k[:, j, :]
                    ),
                    dim=-1
                )

    # Compute Riemann tensor
    riemann = torch.zeros_like(christoffel)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
                    riemann[:, i, j, k] += (
                        christoffel[:, i, j, l] * christoffel[:, l, k, k] -
                        christoffel[:, i, j, k] * christoffel[:, l, l, k]
                    )

    # Contract to get Ricci tensor
    ricci = torch.zeros_like(metric)
    for i in range(dim):
        for j in range(dim):
            # Contract k index: R^k_ikj -> R_ij
            ricci[:, i, j] = torch.sum(riemann[:, :, i, j], dim=1)

    # Reshape back to original shape if needed
    if seq_len is not None:
        ricci = ricci.reshape(batch_size, seq_len, dim, dim)

    return ricci


def compute_parallel_transport(
    path: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """Compute parallel transport along path.
    
    Args:
        path: Path tensor [batch_size, seq_len, manifold_dim] or [batch_size, manifold_dim]
        metric: Metric tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
        
    Returns:
        Parallel transport tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
    """
    # Handle both 2D and 3D inputs
    orig_shape = path.shape
    if len(orig_shape) == 3:
        batch_size, seq_len, manifold_dim = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim]
        path = path.reshape(-1, manifold_dim)
        # Reshape metric to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, manifold_dim, manifold_dim)
    else:
        batch_size, manifold_dim = orig_shape
        seq_len = None
    
    # Compute Christoffel symbols
    christoffel = compute_christoffel_symbols(metric)
    
    # Initialize transport tensor
    transport = torch.zeros_like(metric)
    
    # Compute transport components
    for i in range(manifold_dim):
        for j in range(manifold_dim):
            transport[..., i, j] = path[..., i] * path[..., j]
    
    # Apply metric compatibility
    transport = torch.einsum('...ij,...jk->...ik', transport, metric)
    
    # Reshape back to original shape if needed
    if seq_len is not None:
        transport = transport.reshape(batch_size, seq_len, manifold_dim, manifold_dim)
    
    return transport


def compute_geodesic_distance(
    path: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """Compute geodesic distance along path.
    
    Args:
        path: Path tensor [batch_size, seq_len, manifold_dim] or [batch_size, manifold_dim]
        metric: Metric tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
        
    Returns:
        Geodesic distance [batch_size, seq_len] or [batch_size]
    """
    # Handle both 2D and 3D inputs
    orig_shape = path.shape
    if len(orig_shape) == 3:
        batch_size, seq_len, manifold_dim = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim]
        path = path.reshape(-1, manifold_dim)
        # Reshape metric to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, manifold_dim, manifold_dim)
    else:
        batch_size, manifold_dim = orig_shape
        seq_len = None
    
    # Compute metric inner product
    distance = torch.sqrt(torch.einsum(
        'bi,bij,bj->b',
        path,
        metric,
        path
    ))
    
    # Reshape back to original shape if needed
    if seq_len is not None:
        distance = distance.reshape(batch_size, seq_len)
    
    return distance


def compute_flow_energy(
    path: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """Compute energy of flow path.
    
    Args:
        path: Flow path tensor [batch_size, seq_len, manifold_dim] or [batch_size, manifold_dim]
        metric: Metric tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
        
    Returns:
        Flow energy [batch_size, seq_len] or [batch_size]
    """
    # Handle both 2D and 3D inputs
    orig_shape = path.shape
    if len(orig_shape) == 3:
        batch_size, seq_len, manifold_dim = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim]
        path = path.reshape(-1, manifold_dim)
        # Reshape metric to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, manifold_dim, manifold_dim)
    else:
        batch_size, manifold_dim = orig_shape
        seq_len = None
    
    # Compute energy using metric inner product
    energy = torch.einsum(
        'bi,bij,bj->b',
        path,
        metric,
        path
    )
    
    # Reshape back to original shape if needed
    if seq_len is not None:
        energy = energy.reshape(batch_size, seq_len)
    
    return energy 


def compute_ricci_curvature(
    metric: torch.Tensor,
    christoffel: torch.Tensor
) -> torch.Tensor:
    """Compute Ricci curvature tensor.
    
    Args:
        metric: Metric tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
        christoffel: Christoffel symbols [batch_size, seq_len, manifold_dim, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim, manifold_dim]
        
    Returns:
        Ricci curvature tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
    """
    # Handle both 2D and 3D inputs
    orig_shape = metric.shape
    if len(orig_shape) == 4:
        batch_size, seq_len, manifold_dim, _ = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, manifold_dim, manifold_dim)
        # Reshape christoffel to [batch_size * seq_len, manifold_dim, manifold_dim, manifold_dim]
        christoffel = christoffel.reshape(-1, manifold_dim, manifold_dim, manifold_dim)
    else:
        batch_size, manifold_dim, _ = orig_shape
        seq_len = None
    
    # Compute Riemann curvature tensor components
    riemann = torch.zeros_like(metric).unsqueeze(-1)
    for i in range(manifold_dim):
        for j in range(manifold_dim):
            for k in range(manifold_dim):
                for l in range(manifold_dim):
                    # R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
                    term1 = christoffel[..., i, j, k] * christoffel[..., k, l, j]
                    term2 = christoffel[..., i, j, l] * christoffel[..., l, k, j]
                    riemann[..., i, j] += term1 - term2
    
    # Contract to get Ricci tensor
    ricci = torch.einsum('...ijik->...jk', riemann)
    
    # Reshape back to original shape if needed
    if seq_len is not None:
        ricci = ricci.reshape(batch_size, seq_len, manifold_dim, manifold_dim)
    
    return ricci 


def compute_christoffel_symbols(
    metric: torch.Tensor
) -> torch.Tensor:
    """Compute Christoffel symbols from metric tensor.
    
    Args:
        metric: Metric tensor [batch_size, manifold_dim, manifold_dim] or [batch_size, seq_len, manifold_dim, manifold_dim]
        
    Returns:
        Christoffel symbols [batch_size, manifold_dim, manifold_dim, manifold_dim] or [batch_size, seq_len, manifold_dim, manifold_dim, manifold_dim]
    """
    # Get dimensions
    orig_shape = metric.shape
    if len(orig_shape) == 4:
        batch_size, seq_len, manifold_dim, _ = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, manifold_dim, manifold_dim)
    else:
        batch_size, manifold_dim, _ = orig_shape
        seq_len = None
    
    # Compute inverse metric
    ginv = torch.inverse(metric)  # [batch_size (* seq_len), manifold_dim, manifold_dim]
    
    # Initialize Christoffel symbols
    christoffel = torch.zeros(batch_size if seq_len is None else batch_size * seq_len,
                            manifold_dim, manifold_dim, manifold_dim,
                            dtype=metric.dtype, device=metric.device)
    
    # Compute metric derivatives (approximated)
    eps = 1e-6
    for k in range(manifold_dim):
        # Create perturbation in k direction
        h = torch.zeros_like(metric)
        h[..., k, :] = eps
        
        # Compute finite difference approximation
        dg_k = (metric + h - metric) / eps
        
        # Store in Christoffel symbols
        for i in range(manifold_dim):
            for j in range(manifold_dim):
                # Γ^i_jk = 1/2 g^il (∂_j g_kl + ∂_k g_jl - ∂_l g_jk)
                christoffel[:, i, j, k] = 0.5 * torch.sum(
                    ginv[:, i, :] * (
                        dg_k[:, j, :] +
                        dg_k[:, j, :] -
                        dg_k[:, j, :]
                    ),
                    dim=-1
                )
    
    # Reshape back to original shape if needed
    if seq_len is not None:
        christoffel = christoffel.reshape(batch_size, seq_len, manifold_dim, manifold_dim, manifold_dim)
    
    return christoffel 