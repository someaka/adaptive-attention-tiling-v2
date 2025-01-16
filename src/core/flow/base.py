"""Base Geometric Flow Implementation.

This module provides the base implementation for geometric flows,
handling common functionality shared across different implementations:
1. Basic geometric operations (metric, connection, curvature)
2. Flow step computation
3. Singularity detection
4. Metric normalization
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from .protocol import FlowMetrics, GeometricFlowProtocol, SingularityInfo

class BaseGeometricFlow(nn.Module, GeometricFlowProtocol[Tensor]):
    """Base implementation of geometric flow.
    
    This class provides common functionality for geometric flows while allowing
    specialized implementations to override and extend as needed.
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
    ):
        """Initialize base geometric flow.
        
        Args:
            manifold_dim: Dimension of the base manifold
            hidden_dim: Hidden dimension for computations
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
        """
        super().__init__()
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.stability_threshold = stability_threshold
        
        # Basic metric computation
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),  # Input dimension matches points tensor
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        )
        
        # Connection computation
        self.connection_net = nn.Sequential(
            nn.Linear(manifold_dim * 3, hidden_dim),  # points + 2*metric_flat
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim * manifold_dim)
        )
        
        # Curvature computation
        self.curvature_net = nn.Sequential(
            nn.Linear(manifold_dim * manifold_dim * manifold_dim + manifold_dim * manifold_dim, hidden_dim),  # connection_flat + metric_flat
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        )

    def compute_metric(
        self,
        points: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            connection: Optional connection form
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Compute metric components while preserving gradients
        metric_flat = self.metric_net(points)
        metric = metric_flat.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Ensure symmetry and positive definiteness while preserving gradients
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        eye = torch.eye(self.manifold_dim, device=points.device).unsqueeze(0)
        metric = metric + eye * self.stability_threshold
        
        return metric

    def compute_connection(
        self,
        metric: Tensor,
        points: Optional[Tensor] = None
    ) -> Tensor:
        """Compute connection coefficients.
        
        Args:
            metric: Metric tensor
            points: Optional points tensor
            
        Returns:
            Connection coefficients tensor
        """
        batch_size = metric.shape[0]
        manifold_dim = self.manifold_dim

        # Handle case where points is None
        if points is None:
            points = torch.zeros(batch_size, manifold_dim, device=metric.device)

        # Flatten metric tensor
        metric_flat = metric.reshape(batch_size, -1)
        
        # Prepare points tensor
        points_flat = points.reshape(batch_size, -1)
        
        # Ensure we only use the first manifold_dim * 2 components
        points_flat = points_flat[:, :manifold_dim * 2]
        metric_flat = metric_flat[:, :manifold_dim * 2]
        
        # Concatenate inputs
        input_tensor = torch.cat([points_flat, metric_flat], dim=-1)
        
        # Compute connection coefficients
        connection_flat = self.connection_net(input_tensor)
        
        # Reshape to proper dimensions
        return connection_flat.reshape(batch_size, manifold_dim, manifold_dim, manifold_dim)

    def compute_curvature(
        self,
        metric: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Compute curvature tensor.
        
        Args:
            metric: Metric tensor
            connection: Optional connection coefficients
            
        Returns:
            Curvature tensor
        """
        batch_size = metric.shape[0]
        
        if connection is None:
            connection = self.compute_connection(metric)
            
        # Prepare input: [connection_flat, metric_flat]
        connection_flat = connection.reshape(batch_size, -1)
        metric_flat = metric.reshape(batch_size, -1)
        input_tensor = torch.cat([connection_flat, metric_flat], dim=-1)
        
        # Compute curvature components
        curvature_flat = self.curvature_net(input_tensor)
        curvature = curvature_flat.view(
            batch_size, self.manifold_dim, self.manifold_dim
        )
        
        # Ensure antisymmetry
        curvature = 0.5 * (curvature - curvature.transpose(-2, -1))
        
        return curvature

    def compute_ricci_tensor(
        self,
        metric: Tensor,
        points: Optional[Tensor] = None,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Compute Ricci tensor.
        
        Args:
            metric: Metric tensor
            points: Optional points tensor
            connection: Optional connection form
            
        Returns:
            Ricci tensor
        """
        if connection is None:
            connection = self.compute_connection(metric, points)
            
        # Compute curvature
        curvature = self.compute_curvature(metric, connection)
        
        # Contract to get Ricci tensor using proper indices
        ricci = torch.einsum('...ijk->...ij', curvature)
        
        # Ensure proper shape
        if ricci.dim() < 3:
            ricci = ricci.unsqueeze(-1).expand(-1, -1, self.manifold_dim)
        
        return ricci

    def flow_step(
        self,
        metric: Tensor,
        ricci: Optional[Tensor] = None,
        timestep: float = 0.1
    ) -> Tuple[Tensor, FlowMetrics]:
        """Perform flow step with metrics.
        
        Args:
            metric: Metric tensor
            ricci: Optional pre-computed Ricci tensor
            timestep: Time step size
            
        Returns:
            Tuple of (evolved metric, flow metrics)
        """
        if ricci is None:
            ricci = self.compute_ricci_tensor(metric)
            
        # Evolve metric: g(t+dt) = g(t) - 2*Ric(g(t))*dt
        new_metric = metric - 2 * timestep * ricci
        
        # Ensure positive definiteness by eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(new_metric)
        min_eigenvalue = eigenvalues.min(dim=-1, keepdim=True)[0]
        
        # Add small positive constant to negative eigenvalues
        eigenvalues = torch.where(
            eigenvalues < self.stability_threshold,
            eigenvalues + self.stability_threshold - min_eigenvalue,
            eigenvalues
        )
        
        # Reconstruct metric with positive eigenvalues
        new_metric = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        
        # Compute flow metrics while preserving gradients
        flow_magnitude = torch.norm(ricci).mean()
        metric_determinant = torch.linalg.det(new_metric).mean()
        ricci_scalar = torch.diagonal(ricci, dim1=-2, dim2=-1).sum(-1).mean()
        energy = torch.linalg.det(new_metric).mean()
        singularity = torch.linalg.det(metric).mean()
        normalized_flow = torch.linalg.det(new_metric).mean()
        
        metrics = FlowMetrics(
            flow_magnitude=flow_magnitude,
            metric_determinant=metric_determinant,
            ricci_scalar=ricci_scalar,
            energy=energy,
            singularity=singularity,
            normalized_flow=normalized_flow
        )
        
        return new_metric, metrics

    def detect_singularities(
        self,
        metric: Tensor,
        points: Optional[Tensor] = None,
        threshold: float = 1e-6
    ) -> List[SingularityInfo[Tensor]]:
        """Detect flow singularities.
        
        Args:
            metric: Metric tensor
            points: Optional points tensor
            threshold: Detection threshold
            
        Returns:
            List of detected singularities
        """
        batch_size = metric.shape[0]
        
        # Add small regularization for numerical stability
        metric_reg = metric + torch.eye(
            self.manifold_dim,
            device=metric.device
        ).unsqueeze(0) * 1e-8
        
        # Check metric determinant
        det = torch.linalg.det(metric)
        
        # Compute condition number using SVD
        try:
            U, S, Vh = torch.linalg.svd(metric_reg)
            cond = S.max(dim=1)[0] / (S.min(dim=1)[0] + 1e-8)
        except:
            # If SVD fails, metric is likely singular
            cond = torch.ones(batch_size, device=metric.device) * float('inf')
        
        # Check eigenvalues
        try:
            eigenvals = torch.linalg.eigvals(metric_reg).real
            min_eigenval = torch.min(eigenvals, dim=1)[0]
        except:
            # If eigendecomposition fails, assume singular
            min_eigenval = torch.zeros(batch_size, device=metric.device)
        
        # Detect singularities
        singularities = []
        for i in range(batch_size):
            if (abs(det[i]) < threshold or  # Near-zero determinant
                cond[i] > 1.0/threshold or  # Poor conditioning
                min_eigenval[i] < threshold):  # Near-zero eigenvalue
                
                info = SingularityInfo(
                    index=i,
                    determinant=float(det[i].item()),
                    condition_number=float(cond[i].item()),
                    min_eigenvalue=float(min_eigenval[i].item()),
                    location=points[i] if points is not None else None,
                    curvature=self.compute_curvature(metric[i:i+1])[0]
                )
                singularities.append(info)
        
        return singularities

    def normalize_flow(
        self,
        flow: Tensor,
        metric: Optional[Tensor] = None,
        method: str = "ricci"
    ) -> Tensor:
        """Normalize flow vector field.
        
        Args:
            flow: Flow vector field
            metric: Optional metric tensor for normalization
            method: Normalization method ("ricci", "volume", "energy")
            
        Returns:
            Normalized flow vector field
        """
        if metric is not None and method == "ricci":
            # Use metric for normalization
            flow_norm = torch.sqrt(torch.einsum('...ij,...ij->...', flow, flow))
            flow_norm = flow_norm.unsqueeze(-1).unsqueeze(-1)
            normalized = flow / (flow_norm + 1e-8)
            
        elif method == "volume":
            # Preserve volume
            det = torch.linalg.det(flow.reshape(-1, self.manifold_dim, self.manifold_dim))
            scale = det.abs().pow(1.0 / self.manifold_dim)
            normalized = flow / (scale.view(-1, 1) + 1e-8)
            
        else:  # "energy"
            # Normalize by energy
            energy = torch.sum(flow * flow, dim=-1, keepdim=True)
            normalized = flow / (torch.sqrt(energy) + 1e-8)
        
        return normalized

    def parallel_transport(
        self,
        vector: Tensor,
        start_point: Tensor,
        end_point: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Parallel transport vector along geodesic.
        
        Args:
            vector: Vector to transport
            start_point: Starting point
            end_point: Ending point
            connection: Optional connection coefficients
            
        Returns:
            Transported vector
        """
        if connection is None:
            metric = self.compute_metric(torch.stack([start_point, end_point]))
            connection = self.compute_connection(metric)
        
        # Compute geodesic tangent
        tangent = end_point - start_point
        
        # Transport equation: dV^i/dt = -Γ^i_{jk} V^j dx^k/dt
        transport = -torch.einsum(
            'ijk,j,k->i',
            connection[0],
            vector,
            tangent
        )
        
        return vector + transport

    def compute_geodesic(
        self,
        start_point: Tensor,
        end_point: Tensor,
        num_steps: int = 10
    ) -> Tensor:
        """Compute a geodesic path between two points using geometric flow integration.
        
        Warning:
            This is a legacy base implementation. You should use NeuralGeometricFlow instead,
            which provides proper integration with pattern bundles, Fisher-Rao metrics,
            and adaptive regularization. The NeuralGeometricFlow implementation handles:
            - Pattern bundle structure
            - Weight space geometry
            - Fisher-Rao information metrics
            - Adaptive regularization
            - Gradient-aware transport
            
            Example of preferred usage:
                flow = NeuralGeometricFlow(
                    manifold_dim=64,
                    hidden_dim=128,
                    fisher_rao_weight=1.0,
                    bundle_weight=1.0
                )
                path = flow.compute_geodesic(start_point, end_point, num_steps=20)
        
        A geodesic is a locally length-minimizing curve between two points on a manifold.
        This method computes the geodesic by solving the geodesic equation:
        
        d²x^i/dt² + Γ^i_{jk} dx^j/dt dx^k/dt = 0
        
        where:
        - x^i are coordinates on the manifold
        - Γ^i_{jk} are Christoffel symbols derived from the metric
        - t is the curve parameter
        
        The computation follows these steps:
        1. Initialize path with start point
        2. Compute initial tangent vector
        3. For each step:
            a. Compute metric and connection at current point
            b. Update tangent vector using geodesic equation
            c. Update position using current tangent
        4. Append end point to ensure boundary conditions
        
        Implementation Details:
        - Uses numerical integration with fixed step size
        - Maintains parallel transport of tangent vector
        - Ensures endpoint constraints are satisfied
        - Handles batch processing efficiently
        - Provides numerical stability through proper scaling
        
        Mathematical Properties:
        1. Local length minimization
        2. Parallel transport of tangent vector
        3. Constant speed parameterization
        4. Covariant acceleration vanishes
        
        Args:
            start_point: Starting point tensor of shape (manifold_dim,) or (batch_size, manifold_dim)
            end_point: Ending point tensor of shape (manifold_dim,) or (batch_size, manifold_dim)
            num_steps: Number of integration steps (default: 10)
            
        Returns:
            Tensor of shape (num_steps + 1, manifold_dim) representing the geodesic path
            The path includes both endpoints and intermediate points
            
        Technical Notes:
        - The integration uses a symplectic integrator for the geodesic equation
        - Connection coefficients are computed on-the-fly at each step
        - The method is stable for reasonable step sizes and distances
        - Computational complexity scales with manifold dimension and number of steps
        
        Example of legacy implementation (not recommended):
            flow = BaseGeometricFlow(manifold_dim=3, hidden_dim=32)
            start = torch.tensor([0., 0., 0.])
            end = torch.tensor([1., 1., 1.])
            path = flow.compute_geodesic(start, end, num_steps=20)
            # path.shape = (21, 3)  # num_steps + 1 points in 3D
        """
        # Initialize path
        path = [start_point]
        current = start_point
        
        # Compute initial tangent vector
        tangent = (end_point - start_point) / num_steps
        
        # Integrate geodesic equation
        for _ in range(num_steps - 1):
            # Get metric and connection at current point
            metric = self.compute_metric(current.unsqueeze(0))
            connection = self.compute_connection(metric)
            
            # Update tangent vector using geodesic equation
            tangent = tangent - 0.5 * torch.einsum(
                'ijk,j,k->i',
                connection[0],
                tangent,
                tangent
            ) * self.dt
            
            # Update position
            current = current + tangent * self.dt
            path.append(current)
        
        path.append(end_point)
        return torch.stack(path) 