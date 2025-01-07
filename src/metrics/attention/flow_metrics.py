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
        Ricci tensor [batch_size, seq_len, manifold_dim, manifold_dim] or [batch_size, manifold_dim, manifold_dim]
    """
    # Handle both 2D and 3D inputs
    orig_shape = metric.shape
    print(f"Original metric shape: {orig_shape}")
    
    if len(orig_shape) == 4:
        batch_size, seq_len, manifold_dim, _ = orig_shape
        # Reshape to [batch_size * seq_len, manifold_dim, manifold_dim]
        metric = metric.reshape(-1, manifold_dim, manifold_dim)
    else:
        batch_size, manifold_dim, _ = orig_shape
        seq_len = None
    
    # Compute inverse metric
    ginv = torch.inverse(metric)  # [batch_size (* seq_len), manifold_dim, manifold_dim]
    print(f"Inverse metric shape: {ginv.shape}")
    
    # Compute metric derivatives (approximated)
    eps = 1e-6
    dg = []
    batch_size_seq = metric.shape[0]  # This is batch_size * seq_len if seq_len exists
    
    for k in range(manifold_dim):
        # Initialize gradient tensor with correct shape
        grad_k = torch.zeros_like(metric)
        
        # Compute finite difference for each batch element
        for b in range(batch_size_seq):
            if k < manifold_dim - 1:
                # Forward difference for non-last components
                grad_k[b] = (metric[b, k+1:] - metric[b, k:-1]) / eps
            else:
                # Backward difference for last component
                grad_k[b] = (metric[b, :1] - metric[b, -1:]) / eps
        
        print(f"Gradient {k} shape: {grad_k.shape}")
        dg.append(grad_k)

    # Stack derivatives
    dg = torch.stack(dg, dim=1)  # [batch_size, dim, dim, dim]
    print(f"Stacked derivatives shape: {dg.shape}")

    # Compute Christoffel symbols using vectorized operations
    christoffel = 0.5 * torch.einsum(
        'bim,bnjk->binjk',  # Fixed einsum equation to match tensor dimensions
        ginv,
        dg + dg.transpose(-2, -1) - dg.transpose(-1, -2)
    )
    print(f"Christoffel symbols shape: {christoffel.shape}")

    # Compute Riemann tensor using vectorized operations
    riemann = (
        torch.einsum('bimjk,bmnlp->binjlp', christoffel, christoffel) -  # Fixed einsum equation
        torch.einsum('bimlk,bmnjp->binjlp', christoffel, christoffel)    # Fixed einsum equation
    )
    print(f"Riemann tensor shape: {riemann.shape}")

    # Contract to get Ricci tensor
    ricci = torch.einsum('binjnj->bij', riemann)  # Fixed einsum equation
    print(f"Ricci tensor shape: {ricci.shape}")

    # Reshape back to original shape if needed
    if seq_len is not None:
        ricci = ricci.reshape(batch_size, seq_len, manifold_dim, manifold_dim)
        print(f"Final reshaped Ricci tensor shape: {ricci.shape}")

    return ricci


def compute_parallel_transport(
    path: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """Compute parallel transport along path.
    
    Args:
        path: Path tensor of shape [batch_size, manifold_dim]
        metric: Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
        
    Returns:
        Transport tensor of shape [batch_size, manifold_dim, manifold_dim]
    """
    print(f"\nPath shape: {path.shape}")
    print(f"Metric shape: {metric.shape}")
    
    # Get manifold dimension from metric tensor
    manifold_dim = metric.shape[-1]
    
    # Initialize transport tensor with correct dimensions
    transport = torch.eye(manifold_dim, dtype=path.dtype, device=path.device)
    transport = transport.unsqueeze(0).expand(path.shape[0], -1, -1)
    print(f"Transport shape: {transport.shape}")
    
    # Compute transport
    transport = torch.einsum('...ij,...jk->...ik', transport, metric)
    
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
    print(f"\nPath shape in geodesic: {path.shape}")
    print(f"Metric shape in geodesic: {metric.shape}")
    
    # Get dimensions
    if len(path.shape) == 3:
        batch_size, seq_len, path_dim = path.shape
    else:
        # If path is [N, manifold_dim], treat N as batch_size
        batch_size, path_dim = path.shape
        seq_len = 1  # Default to 1 if no sequence dimension
        # Add sequence dimension
        path = path.unsqueeze(1)

    # Get manifold dimension from metric
    manifold_dim = metric.shape[-1]
    
    # Project path to manifold dimension if needed
    if path_dim != manifold_dim:
        # Take first manifold_dim components
        path = path[..., :manifold_dim]
    
    # Ensure metric has correct shape and batch size
    if len(metric.shape) == 3:
        # Expand metric to include sequence dimension
        metric = metric.unsqueeze(1)
    # Expand metric to match batch size if needed
    if metric.shape[0] == 1 and batch_size > 1:
        metric = metric.expand(batch_size, -1, -1, -1)

    # Compute metric inner product for each sequence position
    distance = torch.zeros(batch_size, seq_len, device=path.device, dtype=path.dtype)

    for b in range(batch_size):
        for s in range(seq_len):
            # Extract vectors and matrix for this position
            p = path[b, s]  # [manifold_dim]
            # Use metric[0] if batch size is 1, otherwise use metric[b]
            m = metric[0 if metric.shape[0] == 1 else b, s if len(metric.shape) > 3 else 0]  # [manifold_dim, manifold_dim]
            # Compute p^T * M * p
            distance[b, s] = torch.sqrt(torch.abs(p.conj() @ m @ p))

    # If original path had no sequence dimension, remove it from output
    if len(path.shape) == 2:
        distance = distance.squeeze(1)

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
    print(f"\nPath shape in energy: {path.shape}")
    print(f"Metric shape in energy: {metric.shape}")
    
    # Get dimensions
    if len(path.shape) == 3:
        batch_size, seq_len, path_dim = path.shape
    else:
        # If path is [N, manifold_dim], treat N as batch_size
        batch_size, path_dim = path.shape
        seq_len = 1  # Default to 1 if no sequence dimension
        # Add sequence dimension
        path = path.unsqueeze(1)

    # Get manifold dimension from metric
    manifold_dim = metric.shape[-1]
    
    # Project path to manifold dimension if needed
    if path_dim != manifold_dim:
        # Take first manifold_dim components
        path = path[..., :manifold_dim]
    
    # Ensure metric has correct shape and batch size
    if len(metric.shape) == 3:
        # Expand metric to include sequence dimension
        metric = metric.unsqueeze(1)
    # Expand metric to match batch size if needed
    if metric.shape[0] == 1 and batch_size > 1:
        metric = metric.expand(batch_size, -1, -1, -1)

    # Compute energy using metric inner product for each sequence position
    energy = torch.zeros(batch_size, seq_len, device=path.device, dtype=path.dtype)

    for b in range(batch_size):
        for s in range(seq_len):
            # Extract vectors and matrix for this position
            p = path[b, s]  # [manifold_dim]
            # Use metric[0] if batch size is 1, otherwise use metric[b]
            m = metric[0 if metric.shape[0] == 1 else b, s if len(metric.shape) > 3 else 0]  # [manifold_dim, manifold_dim]
            # Compute p^T * M * p
            energy[b, s] = torch.abs(p.conj() @ m @ p)

    # If original path had no sequence dimension, remove it from output
    if len(path.shape) == 2:
        energy = energy.squeeze(1)

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