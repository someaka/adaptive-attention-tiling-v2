"""Neural Geometric Flow Implementation.

This module implements geometric flows for neural attention:
- Ricci tensor computation
- Flow step implementation
- Singularity detection and handling
- Flow normalization
- Energy conservation
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


@dataclass
class FlowMetrics:
    """Metrics for geometric flow analysis."""

    ricci_scalar: torch.Tensor  # Scalar curvature
    energy: torch.Tensor  # Flow energy
    singularity: torch.Tensor  # Singularity measure
    normalized_flow: torch.Tensor  # Normalized flow vector


@dataclass
class SingularityInfo:
    """Information about a geometric singularity."""
    
    location: torch.Tensor  # Point where singularity occurs
    curvature: torch.Tensor  # Curvature tensor at singularity
    resolution: torch.Tensor  # Resolution matrix for fixing singularity
    
    def is_removable(self) -> bool:
        """Check if singularity is removable via resolution."""
        # Check if resolution matrix is invertible
        try:
            torch.linalg.inv(self.resolution)
            return True
        except:
            return False
            
    def get_blowup_rate(self) -> float:
        """Get rate of curvature blowup near singularity."""
        return float(torch.max(torch.abs(self.curvature)))


class RicciTensorNetwork(nn.Module):
    """Neural computation of Ricci tensor."""

    def __init__(self, manifold_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.manifold_dim = manifold_dim

        # Metric computation
        self.metric_network = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim),
        )

        # Christoffel symbols
        self.christoffel_network = nn.Sequential(
            nn.Linear(manifold_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim),
        )

        # Ricci tensor
        self.ricci_network = nn.Sequential(
            nn.Linear(manifold_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, manifold_dim * manifold_dim),
        )

    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points."""
        batch_metrics = self.metric_network(points)
        return batch_metrics.view(-1, self.manifold_dim, self.manifold_dim)

    def compute_christoffel(
        self, points: torch.Tensor, metric: torch.Tensor
    ) -> torch.Tensor:
        """Compute Christoffel symbols."""
        batch_size = points.shape[0]
        symbols = torch.zeros(
            batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim
        )

        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    input_tensor = torch.cat(
                        [
                            points,
                            metric[:, i, j].unsqueeze(1),
                            metric[:, j, k].unsqueeze(1),
                            metric[:, k, i].unsqueeze(1),
                        ],
                        dim=1,
                    )
                    symbols[:, i, j, k] = self.christoffel_network(
                        input_tensor
                    ).squeeze()

        return symbols

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Ricci tensor at points."""
        # Compute metric
        metric = self.compute_metric(points)

        # Compute Christoffel symbols
        christoffel = self.compute_christoffel(points, metric)

        # Compute Ricci tensor
        batch_size = points.shape[0]
        ricci = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim)

        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                input_tensor = torch.cat(
                    [
                        points,
                        metric[:, i, j].unsqueeze(1),
                        christoffel[:, i, j].reshape(batch_size, -1),
                    ],
                    dim=1,
                )
                ricci[:, i, j] = self.ricci_network(input_tensor).squeeze()

        return ricci


class RicciTensor:
    """Wrapper class for Ricci tensor to ensure proper tensor operations."""
    
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        
    @classmethod
    def __instancecheck__(cls, instance):
        """Check if instance is a RicciTensor."""
        return hasattr(instance, 'tensor') and isinstance(instance.tensor, torch.Tensor)
        
    @classmethod
    def __subclasscheck__(cls, subclass):
        """Check if subclass is a RicciTensor subclass."""
        return issubclass(subclass, cls)
        
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Handle torch function calls on RicciTensor."""
        if kwargs is None:
            kwargs = {}
        args = [a.tensor if isinstance(a, RicciTensor) else a for a in args]
        ret = func(*args, **kwargs)
        return RicciTensor(ret) if isinstance(ret, torch.Tensor) else ret
        
    def __truediv__(self, other):
        """Handle division."""
        if isinstance(other, (int, float)):
            return RicciTensor(self.tensor / other)
        return RicciTensor(self.tensor / other.tensor)
        
    def __mul__(self, other):
        """Handle multiplication."""
        if isinstance(other, (int, float)):
            return RicciTensor(self.tensor * other)
        return RicciTensor(self.tensor * other.tensor)
        
    def __rmul__(self, other):
        """Handle right multiplication."""
        return self.__mul__(other)
        
    def __getattr__(self, name):
        """Forward attribute access to tensor."""
        try:
            return getattr(self.tensor, name)
        except AttributeError:
            raise AttributeError(f"'RicciTensor' object has no attribute '{name}'")


class FlowStepNetwork(nn.Module):
    """Network for computing flow steps."""

    def __init__(self, manifold_dim: int, hidden_dim: int = 128):
        """Initialize network.
        
        Args:
            manifold_dim: Dimension of manifold
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Input size accounts for points and Ricci tensor
        input_dim = manifold_dim + manifold_dim * manifold_dim
        output_dim = manifold_dim
        
        # Network for computing flow vector field
        self.flow_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Network for computing flow energy
        self.energy_network = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def normalize_flow(self, flow: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Normalize flow to preserve volume and maintain stability.
        
        Args:
            flow: Flow vector field
            metric: Metric tensor
            
        Returns:
            Normalized flow vector field
        """
        batch_size = flow.shape[0]
        n = flow.shape[1]
        
        # Ensure metric is positive definite
        epsilon = 1e-6
        identity = torch.eye(n, device=flow.device).unsqueeze(0).expand(batch_size, -1, -1)
        metric = metric + epsilon * identity
        
        # Compute metric determinant with stability
        det = torch.det(metric)
        det = torch.clamp(det, min=epsilon)
        sqrt_det = torch.sqrt(det)
        
        # Compute inverse metric with stability
        try:
            metric_inv = torch.linalg.inv(metric)
        except:
            # If inversion fails, use more stable pseudo-inverse
            metric_inv = torch.linalg.pinv(metric)
        
        # Compute covariant divergence properly
        div = torch.zeros(batch_size, device=flow.device)
        for i in range(n):
            for j in range(n):
                div += metric_inv[:, i, j] * flow[:, j]
        
        # Normalize flow with bounded scale factor
        scale = div / (n * sqrt_det)
        scale = torch.clamp(scale, min=-1.0, max=1.0)
        scale = scale.unsqueeze(1)
        
        # Apply normalization with metric compatibility
        metric_diag = torch.diagonal(metric, dim1=1, dim2=2)
        normalized_flow = flow - scale * metric_diag
        
        # Ensure flow magnitude is bounded
        flow_norm = torch.norm(normalized_flow, dim=1, keepdim=True)
        max_norm = 1e3
        scale_factor = torch.where(
            flow_norm > max_norm,
            max_norm / flow_norm,
            torch.ones_like(flow_norm)
        )
        normalized_flow = normalized_flow * scale_factor
        
        return normalized_flow

    def compute_flow_vector(self, points: torch.Tensor, ricci: RicciTensor) -> torch.Tensor:
        """Compute flow vector field at given points with stability.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor at points
            
        Returns:
            Flow vector field of shape (batch_size, manifold_dim)
        """
        # Prepare input with proper scaling
        points_scaled = points / (torch.norm(points, dim=1, keepdim=True) + 1e-6)
        ricci_scaled = ricci.tensor / (torch.norm(ricci.tensor, dim=(1,2), keepdim=True) + 1e-6)
        
        # Flatten and concatenate inputs
        ricci_flat = ricci_scaled.reshape(ricci.tensor.shape[0], -1)
        input_tensor = torch.cat([points_scaled, ricci_flat], dim=1)
        
        # Compute flow with activation bounds
        flow = torch.tanh(self.flow_network(input_tensor))
        
        # Scale flow to reasonable magnitude
        flow = flow * 1e2  # Limit initial magnitude
        
        return flow

    def compute_energy(self, points: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Compute flow energy.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            flow: Flow vector field of shape (batch_size, manifold_dim)
            
        Returns:
            Energy scalar
        """
        # Compute energy using points and flow
        points_flat = points.reshape(points.shape[0], -1)
        energy = self.energy_network(points_flat)
        
        # Add regularization term based on flow magnitude
        flow_norm = torch.norm(flow, dim=-1, keepdim=True)
        energy = energy + 0.1 * flow_norm.mean()
        
        return energy.squeeze()

    def forward(
        self, points: torch.Tensor, ricci: RicciTensor, dt: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute flow step.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor at points
            dt: Time step size
            
        Returns:
            Tuple of:
            - Updated points after flow step
            - Flow energy 
        """
        # Compute flow vector field
        flow = self.compute_flow_vector(points, ricci)
        
        # Update points using Euler integration
        new_points = points + dt * flow
        
        # Compute energy of flow
        energy = self.compute_energy(points, flow)
        
        return new_points, energy


class SingularityDetector(nn.Module):
    """Detection and analysis of flow singularities."""

    def __init__(self, manifold_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.manifold_dim = manifold_dim

        # Singularity detection
        self.detector = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 4),  # [type, severity, x, y]
        )

        # Resolution computation
        self.resolver = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim),
        )

    def detect_singularities(
        self, points: torch.Tensor, flow: torch.Tensor
    ) -> List[SingularityInfo]:
        """Detect singularities in flow."""
        singularities = []

        # Combine points and flow
        input_tensor = torch.cat([points, flow], dim=1)
        detection = self.detector(input_tensor)

        # Analyze detection results
        for i in range(len(points)):
            if detection[i, 1] > 0.5:  # Severity threshold
                resolution = self.resolver(input_tensor[i : i + 1])

                singularities.append(
                    SingularityInfo(
                        location=points[i],
                        curvature=detection[i, 0],
                        resolution=resolution,
                    )
                )

        return singularities

    def _classify_singularity(self, type_idx: torch.Tensor) -> str:
        """Classify type of singularity.
        
        Args:
            type_idx: Type index from singularity detector
            
        Returns:
            String describing singularity type
        """
        types = ["removable", "essential", "conical", "cusp"]
        type_idx = torch.argmax(type_idx).item()
        return types[type_idx]


class FlowNormalizer(nn.Module):
    """Normalization of geometric flows."""

    def __init__(self, manifold_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.manifold_dim = manifold_dim

        # Normalization network
        self.normalizer = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, manifold_dim),
        )

        # Scale factor computation
        self.scale_computer = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def normalize_flow(self, metric: torch.Tensor) -> torch.Tensor:
        """Normalize metric to preserve volume.
        
        Args:
            metric: Metric tensor of shape (batch_size, n, n)
            
        Returns:
            Normalized metric tensor
        """
        # Compute determinant for each batch element
        det = torch.det(metric)
        
        # Compute scaling factor to normalize volume
        scale = det.pow(-1/(2*metric.shape[1]))
        
        # Reshape scale for broadcasting
        scale = scale.view(-1, 1, 1)
        
        # Scale metric to preserve volume
        return scale * metric

    def normalize_flow_vector(self, flow: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """Normalize flow vector field."""
        # Compute normalization scale
        scale = self.scale_computer(flow)

        # Normalize flow
        input_tensor = torch.cat([flow, energy.expand(-1, 1)], dim=1)
        normalized = self.normalizer(input_tensor)

        return normalized * scale


class GeometricFlow:
    """Complete geometric flow system for neural attention."""

    def __init__(
        self,
        manifold_dim: int,
        flow_type: str = "ricci",
        timestep: float = 0.01,
        normalize_method: str = "volume",
        hidden_dim: int = 128,
    ):
        """Initialize geometric flow system.
        
        Args:
            manifold_dim: Dimension of the manifold
            flow_type: Type of geometric flow ("ricci" or "mean_curvature")
            timestep: Time step for flow evolution
            normalize_method: Flow normalization method ("volume" or "yamabe")
            hidden_dim: Hidden dimension for neural networks
        """
        self.manifold_dim = manifold_dim
        self.flow_type = flow_type
        self.timestep = timestep
        self.normalize_method = normalize_method
        self.dt = timestep
        
        # Initialize components
        self.ricci = RicciTensorNetwork(manifold_dim, hidden_dim)
        self.flow_step = FlowStepNetwork(manifold_dim, hidden_dim // 2)
        self.singularity = SingularityDetector(manifold_dim, hidden_dim // 4)
        self.normalizer = FlowNormalizer(manifold_dim, hidden_dim // 4)
        
        # Energy conservation threshold
        self.energy_threshold = 1e-6
        self.points = None
        
    def set_points(self, points: torch.Tensor):
        """Set points tensor for flow computation.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
        """
        self.points = points
        
    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor
            
        Returns:
            Metric tensor
        """
        self.set_points(points)
        batch_size = points.shape[0]
        n = points.shape[1]
        
        # Initialize metric tensor
        metric = torch.zeros(batch_size, n, n, device=points.device)
        
        # Compute metric components using inner products
        for i in range(n):
            for j in range(n):
                metric[:,i,j] = torch.sum(
                    points[:,i].unsqueeze(1) * points[:,j].unsqueeze(1),
                    dim=-1
                )
                
        return metric

    def compute_connection(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel symbols (connection coefficients).
        
        Args:
            metric: Metric tensor of shape (batch_size, n, n)
            
        Returns:
            Connection coefficients of shape (batch_size, n, n, n)
        """
        batch_size = metric.shape[0]
        n = self.manifold_dim
        
        # Add small diagonal term to prevent singularity
        eye = torch.eye(n, device=metric.device).unsqueeze(0).expand(batch_size, -1, -1)
        regularized_metric = metric + 1e-6 * eye
        
        # Initialize connection coefficients
        connection = torch.zeros(
            batch_size, n, n, n,
            device=metric.device,
            requires_grad=False  # Initialize without grad to avoid in-place issues
        )
        
        # Compute inverse metric safely using pseudoinverse
        metric_inv = torch.linalg.pinv(regularized_metric)
        
        # Compute metric derivatives using finite differences
        eps = 1e-6
        metric_deriv = torch.zeros(batch_size, n, n, n, device=metric.device)
        
        for k in range(n):
            # Forward difference
            points_plus = self.points.clone()
            points_plus.data[:, k] += eps  # Use .data to avoid in-place grad issues
            metric_plus = self.compute_metric(points_plus)
            
            # Backward difference
            points_minus = self.points.clone()
            points_minus.data[:, k] -= eps  # Use .data to avoid in-place grad issues
            metric_minus = self.compute_metric(points_minus)
            
            # Central difference
            metric_deriv[:, :, :, k] = (metric_plus - metric_minus) / (2 * eps)
        
        # Compute Christoffel symbols
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Γ^k_ij = 1/2 g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
                    for l in range(n):
                        connection[:, k, i, j] = connection[:, k, i, j] + 0.5 * metric_inv[:, k, l] * (
                            metric_deriv[:, j, l, i] +
                            metric_deriv[:, i, l, j] -
                            metric_deriv[:, i, j, l]
                        )
        
        # Make connection require grad after computations
        connection.requires_grad_(True)
        return connection

    def compute_mean_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute mean curvature from metric tensor.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Mean curvature tensor
        """
        batch_size = metric.shape[0]
        n = metric.shape[1]
        
        # Compute Christoffel symbols
        christoffel = self.compute_connection(metric)
        
        # Compute mean curvature
        mean_curv = torch.zeros(batch_size, device=metric.device)
        for i in range(n):
            for j in range(n):
                mean_curv += torch.diagonal(metric, dim1=1, dim2=2)[:, i] * christoffel[:, i, j, j]
                
        return mean_curv / (n * (n-1))

    def compute_ricci_tensor(self, metric: torch.Tensor) -> RicciTensor:
        """Compute Ricci tensor from metric.
        
        Args:
            metric: Metric tensor of shape (batch_size, n, n)
            
        Returns:
            Ricci tensor
        """
        batch_size = metric.shape[0]
        n = self.manifold_dim
        
        # Compute Christoffel symbols
        christoffel = self.compute_connection(metric)
        
        # Initialize Ricci tensor
        ricci = torch.zeros_like(metric)
        
        # Compute Ricci tensor components
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        # R_ij = R^k_ikj
                        term1 = christoffel[:, k, i, j] * christoffel[:, l, k, l]
                        term2 = christoffel[:, k, i, l] * christoffel[:, l, k, j]
                        ricci[:, i, j] += term1 - term2
                        
        # Ensure tensor is symmetric
        ricci = 0.5 * (ricci + ricci.transpose(1, 2))
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        ricci = ricci + epsilon * torch.eye(n, device=metric.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        return RicciTensor(ricci)

    def compute_flow_vector(self, points: torch.Tensor, ricci: RicciTensor) -> torch.Tensor:
        """Compute flow vector field at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor at points
            
        Returns:
            Flow vector field of shape (batch_size, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Flatten tensors for network input
        ricci_flat = ricci.tensor.reshape(batch_size, -1)
        points_flat = points.reshape(batch_size, -1)
        network_input = torch.cat([points_flat, ricci_flat], dim=1)
        
        # Compute flow vector field
        flow = self.flow_step.flow_network(network_input)
        
        # Scale flow to match metric scale
        metric = self.compute_metric(points)
        metric_scale = torch.sqrt(torch.abs(torch.det(metric)))
        flow = flow / (metric_scale.unsqueeze(-1) + 1e-8)
        
        # Normalize flow to preserve volume
        flow = self.flow_step.normalize_flow(flow, metric)
        
        # Ensure output shape matches points shape
        flow = flow.reshape(batch_size, self.manifold_dim)
        
        return flow

    def flow_step(
        self, metric: torch.Tensor, ricci: RicciTensor, timestep: float = None
    ) -> Tuple[torch.Tensor, FlowMetrics]:
        """Perform one step of geometric flow evolution.
        
        Args:
            metric: Current metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            ricci: Current Ricci tensor of shape (batch_size, manifold_dim, manifold_dim)
            timestep: Time step size (optional)
            
        Returns:
            Tuple of (evolved metric, flow metrics)
        """
        if timestep is None:
            timestep = self.timestep
            
        # Choose flow type and ensure proper scaling
        if self.flow_type == "ricci":
            # Scale Ricci tensor by metric determinant for better stability
            det = torch.det(metric)
            scale = torch.sqrt(torch.abs(det) + 1e-8).unsqueeze(-1).unsqueeze(-1)
            flow = -ricci.tensor / scale
        else:  # mean_curvature
            flow = self.compute_mean_curvature(metric)
            
        # Evolve metric with normalized flow
        evolved_metric = metric + timestep * flow
        
        # Compute flow metrics
        scalar_curv = self.compute_scalar_curvature(evolved_metric)
        energy = self.compute_energy(evolved_metric)
        singularity = self.detect_singularities(evolved_metric)[0].severity
        normalized_flow = self.normalizer.normalize_flow(evolved_metric)
        
        metrics = FlowMetrics(
            ricci_scalar=scalar_curv,
            energy=energy,
            singularity=singularity,
            normalized_flow=normalized_flow
        )
        
        return evolved_metric, metrics

    def detect_singularities(self, metric: torch.Tensor, threshold: float = 0.1) -> List[SingularityInfo]:
        """Detect singularities in the metric.
        
        Args:
            metric: Metric tensor
            threshold: Detection threshold
            
        Returns:
            List of detected singularities
        """
        if self.points is None:
            raise ValueError("Points must be set before detecting singularities")
            
        # Compute curvature
        ricci = self.compute_ricci_tensor(metric)
        curv = torch.norm(ricci.tensor, dim=(-2,-1))
        
        # Find points with high curvature
        singular_mask = curv > threshold
        singular_points = self.points[singular_mask]
        singular_curvs = ricci.tensor[singular_mask]
        
        # Create resolution matrices
        n = self.manifold_dim
        resolutions = torch.eye(n).expand(len(singular_points), n, n)
        
        # Create singularity info objects
        singularities = []
        for i in range(len(singular_points)):
            singularities.append(SingularityInfo(
                location=singular_points[i],
                curvature=singular_curvs[i],
                resolution=resolutions[i]
            ))
            
        return singularities

    def normalize_flow(self, metric: torch.Tensor) -> torch.Tensor:
        """Normalize metric to preserve volume.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Normalized metric tensor
        """
        return self.normalizer.normalize_flow(metric)

    def compute_volume(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute volume form from metric.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Volume form
        """
        return torch.sqrt(torch.abs(torch.linalg.det(metric)))

    def compute_scalar_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute scalar curvature from metric.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Scalar curvature
        """
        ricci = self.compute_ricci_tensor(metric)
        return torch.einsum('...ii->...', ricci.tensor)

    def compute_energy(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute energy functional.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Energy value
        """
        ricci = self.compute_ricci_tensor(metric)
        return torch.einsum('...ij,...ij->...', ricci.tensor, ricci.tensor)

    def evolve_flow(self, metric: torch.Tensor, time_span: List[float], steps: int) -> List[torch.Tensor]:
        """Evolve metric along geometric flow.
        
        Args:
            metric: Initial metric tensor
            time_span: [start_time, end_time]
            steps: Number of evolution steps
            
        Returns:
            List of evolved metrics
        """
        dt = (time_span[1] - time_span[0]) / steps
        metrics = [metric]
        
        current_metric = metric
        for _ in range(steps):
            # Compute Ricci tensor
            ricci = self.compute_ricci_tensor(current_metric)
            
            # Evolution step
            current_metric, _ = self.flow_step(current_metric, ricci, timestep=dt)
            
            # Normalize if needed
            if self.normalize_method:
                current_metric = self.normalize_flow(current_metric)
                
            metrics.append(current_metric)
            
        return metrics

    def initialize_metric(self, batch_size: int, manifold_dim: int) -> torch.Tensor:
        """Initialize metric tensor.
        
        Args:
            batch_size: Batch size
            manifold_dim: Manifold dimension
            
        Returns:
            Initial metric tensor
        """
        # Initialize as perturbation of identity metric
        metric = torch.eye(manifold_dim).unsqueeze(0).expand(batch_size, -1, -1)
        perturbation = 0.1 * torch.randn(batch_size, manifold_dim, manifold_dim)
        metric = metric + 0.5 * (perturbation + perturbation.transpose(-1, -2))
        return metric

    def initialize_surface(self, batch_size: int, manifold_dim: int) -> torch.Tensor:
        """Initialize surface metric for mean curvature flow.
        
        Args:
            batch_size: Batch size
            manifold_dim: Manifold dimension
            
        Returns:
            Surface metric tensor
        """
        # Initialize conformal factor
        phi = torch.randn(batch_size, manifold_dim)
        conformal_factor = torch.exp(phi).unsqueeze(-1).unsqueeze(-1)
        
        # Create base metric
        metric = torch.eye(manifold_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        metric = metric.requires_grad_(True)
        
        # Apply conformal factor
        return conformal_factor * metric

    def initialize_near_singular_metric(self, batch_size: int, manifold_dim: int) -> torch.Tensor:
        """Initialize metric tensor near singularity.
        
        Args:
            batch_size: Batch size
            manifold_dim: Manifold dimension
            
        Returns:
            Near-singular metric tensor
        """
        metric = self.initialize_metric(batch_size, manifold_dim)
        # Create near-singular point by scaling one component
        metric[:, 0, 0] *= 0.1
        return metric

    def detect_necks(self, metric: torch.Tensor) -> List[SingularityInfo]:
        """Detect neck-like regions in the metric.
        
        Args:
            metric: Metric tensor
            
        Returns:
            List of detected neck regions
        """
        # Compute second fundamental form
        ricci = self.compute_ricci_tensor(metric)
        
        # Look for regions of high mean curvature
        mean_curv = torch.einsum('...ii->...', ricci.tensor) / self.manifold_dim
        necks = []
        
        for i in range(len(metric)):
            if torch.abs(mean_curv[i]) > 1.0:
                necks.append(
                    SingularityInfo(
                        location=metric[i],
                        type="neck",
                        severity=float(torch.abs(mean_curv[i])),
                        resolution=torch.eye(self.manifold_dim, device=metric.device)
                    )
                )
                
        return necks

    def estimate_singularity_time(self, metric: torch.Tensor) -> float:
        """Estimate time until singularity formation.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Estimated time until singularity
        """
        # Use Ricci flow evolution to estimate singularity time
        ricci = self.compute_ricci_tensor(metric)
        scalar_curv = torch.einsum('...ii', ricci.tensor)
        
        # Hamilton's estimate
        min_scalar = float(torch.min(scalar_curv))
        if min_scalar >= 0:
            return float('inf')
        return -1 / (2 * min_scalar)

    def normalize_flow_sequence(self, metrics: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize a sequence of metrics.
        
        Args:
            metrics: List of metric tensors
            
        Returns:
            List of normalized metric tensors
        """
        return [self.normalize_flow(metric) for metric in metrics]

    def step(self, metric: torch.Tensor) -> Tuple[torch.Tensor, FlowMetrics]:
        """Perform a single flow step.

        Args:
            metric: Current metric tensor of shape (batch_size, manifold_dim, manifold_dim)

        Returns:
            Tuple of (evolved metric, flow metrics)
        """
        # Ensure metric requires grad
        if not metric.requires_grad:
            metric.requires_grad_(True)

        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Perform flow step
        evolved_metric, metrics = self.flow_step(metric, ricci)
        
        # Normalize if needed
        if self.normalize_method:
            evolved_metric = self.normalize_flow(evolved_metric)
            
        return evolved_metric, metrics

    def evolve(
        self, points: torch.Tensor, num_steps: int = 100, dt: float = 0.01
    ) -> Tuple[List[torch.Tensor], List[FlowMetrics], List[SingularityInfo]]:
        """Evolve points along geometric flow."""
        trajectories = [points]
        metrics = []
        all_singularities = []
        current = points

        for _ in range(num_steps):
            # Compute Ricci tensor
            ricci = self.ricci(current)

            # Perform flow step
            new_points, energy = self.flow_step(current, RicciTensor(ricci), dt)

            # Detect singularities
            flow = new_points - current
            singularities = self.singularity.detect_singularities(current, flow)

            # Normalize flow if needed
            if len(singularities) > 0:
                flow = self.normalizer.normalize_flow_vector(flow, energy)
                new_points = current + dt * flow

            # Compute metrics
            metrics.append(
                FlowMetrics(
                    ricci_scalar=torch.trace(ricci).mean(),
                    energy=energy.mean(),
                    singularity=torch.tensor(len(singularities)),
                    normalized_flow=flow.norm(),
                )
            )

            # Update state
            current = new_points
            trajectories.append(current)
            all_singularities.extend(singularities)

            # Check energy conservation
            if energy.mean() < self.energy_threshold:
                break

        return trajectories, metrics, all_singularities

    def detect_singular_points(self, metric: torch.Tensor) -> List[SingularityInfo]:
        """Alias for detect_singularities for backward compatibility."""
        return self.detect_singularities(metric)

    def compute_blow_up_sequence(self, metric: torch.Tensor, point: SingularityInfo) -> List[torch.Tensor]:
        """Compute blow-up sequence near singularity.
        
        Args:
            metric: Metric tensor
            point: Singularity information
        
        Returns:
            List of rescaled metrics approaching singularity
        """
        # Initialize sequence
        sequence = []
        current = metric.clone()
        
        # Compute scale factors
        scales = torch.logspace(-3, 3, 20)
        
        # Generate sequence
        for scale in scales:
            # Center at singularity
            centered = current - point.location
            
            # Rescale metric
            rescaled = scale * centered
            
            # Add to sequence
            sequence.append(rescaled)
        
        return sequence

    def _classify_singularity(self, type_idx: torch.Tensor) -> str:
        """Classify type of singularity.
        
        Args:
            type_idx: Type index from singularity detector
            
        Returns:
            String describing singularity type
        """
        types = ["removable", "essential", "conical", "cusp"]
        type_idx = torch.argmax(type_idx).item()
        return types[type_idx]
