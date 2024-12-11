"""Neural Geometric Flow Implementation.

This module implements geometric flows for neural attention:
- Ricci tensor computation
- Flow step implementation
- Singularity detection and handling
- Flow normalization
- Energy conservation
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

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
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Ricci tensor
        self.ricci_network = nn.Sequential(
            nn.Linear(manifold_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, manifold_dim * manifold_dim),
        )

    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        n = self.manifold_dim
        
        # Use metric network to compute components
        metric_components = self.metric_network(points)
        
        # Reshape to proper metric tensor shape
        metric = metric_components.view(batch_size, n, n)
        
        # Add small regularization for numerical stability
        eps = 1e-6
        eye = torch.eye(n, device=points.device)
        metric = metric + eps * eye.unsqueeze(0)
        
        # Ensure metric is symmetric
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        return metric

    def compute_christoffel(self, points: torch.Tensor, metric: torch.Tensor):
        """Compute Christoffel symbols."""
        batch_size = points.shape[0]
        symbols = torch.zeros(
            batch_size,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            device=points.device,
        )

        # Compute Christoffel symbols using neural network
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    # Construct input tensor with correct shape
                    # [metric_ij, metric_jk, metric_ki]
                    metric_ij = metric[:, i, j].view(-1, 1)
                    metric_jk = metric[:, j, k].view(-1, 1)
                    metric_ki = metric[:, k, i].view(-1, 1)
                    
                    # Concatenate along dimension 1 to get shape [batch_size, 3]
                    input_tensor = torch.cat([metric_ij, metric_jk, metric_ki], dim=1)
                    
                    # Compute Christoffel symbol - output shape is [batch_size, 1]
                    symbols[:, i, j, k] = self.christoffel_network(input_tensor)[:, 0]

        return symbols

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Ricci tensor at points."""
        # Compute metric
        metric = self.compute_metric(points)

        # Compute Christoffel symbols
        christoffel = self.compute_christoffel(points, metric)

        # Compute Ricci tensor
        batch_size = points.shape[0]
        ricci = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, device=points.device)

        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                # Construct input tensor of shape [batch_size, manifold_dim * 4]
                # [points, metric_row_i, metric_row_j, christoffel_ij]
                input_tensor = torch.cat(
                    [
                        points,  # [batch_size, manifold_dim]
                        metric[:, i],  # [batch_size, manifold_dim]
                        metric[:, j],  # [batch_size, manifold_dim]
                        christoffel[:, i, j].reshape(batch_size, self.manifold_dim)  # [batch_size, manifold_dim]
                    ],
                    dim=1,
                )
                # Compute Ricci tensor components
                ricci_output = self.ricci_network(input_tensor).reshape(batch_size, self.manifold_dim, self.manifold_dim)
                # Make it symmetric to satisfy geometric constraints
                ricci_output = 0.5 * (ricci_output + ricci_output.transpose(-1, -2))
                # Add contribution to Ricci tensor
                ricci += ricci_output

        # Normalize by number of terms
        ricci = ricci / (self.manifold_dim * self.manifold_dim)
        
        return ricci


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
        # Compute flow magnitude using metric
        flow_norm = torch.sqrt(torch.einsum('...ij,...ij->...', flow, flow))
        flow_norm = flow_norm.unsqueeze(-1).unsqueeze(-1)
        
        # Add small epsilon to avoid division by zero
        normalized_flow = flow / (flow_norm + 1e-8)
        
        # Scale to reasonable magnitude
        normalized_flow = normalized_flow * 0.1
        
        return normalized_flow

    def compute_flow_vector(self, points: torch.Tensor, ricci: torch.Tensor) -> torch.Tensor:
        """Compute flow vector field at given points with stability.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor
            
        Returns:
            Flow vector field of shape (batch_size, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Reshape Ricci tensor for network input
        ricci_flat = ricci.reshape(batch_size, -1)
        ricci_flat = ricci_flat[:, :points.shape[1]]  # Take first manifold_dim components
        
        # Concatenate points with Ricci components
        flow_input = torch.cat([points, ricci_flat], dim=-1)
        
        # Compute flow through network
        flow = torch.tanh(self.flow_network(flow_input))
        
        # Scale flow to reasonable magnitude
        flow = flow * 1e2  # Limit initial magnitude
        
        return flow

    def compute_energy(self, states: torch.Tensor) -> torch.Tensor:
        """Compute energy functional.
        
        Args:
            states: States tensor of shape (batch_size, phase_dim)
            
        Returns:
            Energy tensor of shape (batch_size,)
        """
        # Split into position and momentum
        pos = states[..., :self.manifold_dim]
        mom = states[..., self.manifold_dim:]
        
        # Compute kinetic energy (from momentum)
        kinetic = 0.5 * torch.sum(mom * mom, dim=-1)
        
        # Compute potential energy (using position)
        metric = self.compute_metric(pos)
        potential = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)
        
        # Total energy is sum of kinetic and potential
        return kinetic + potential

    def forward(
        self, points: torch.Tensor, ricci: torch.Tensor, dt: float = 0.01
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
        energy = self.compute_energy(torch.cat([points, flow], dim=-1))
        
        return new_points, energy


class SingularityDetector(nn.Module):
    """Detection and analysis of flow singularities."""

    def __init__(self, manifold_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.manifold_dim = manifold_dim

        # Singularity detection - input is [points, flattened_flow]
        input_dim = manifold_dim + manifold_dim * manifold_dim
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 4),  # [type, severity, x, y]
        )

        # Resolution computation - input is [points, flattened_flow]
        self.resolver = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim),
        )

    def detect_singularities(
        self, metric: torch.Tensor, flow: Optional[torch.Tensor] = None
    ) -> List[SingularityInfo]:
        """Detect singularities in the metric tensor and optionally in the flow field.
        
        Args:
            metric: Metric tensor of shape [batch_size, dim, dim]
            flow: Optional flow tensor of shape [batch_size, dim] or [batch_size, dim, dim]
            
        Returns:
            List of detected singularities
        """
        batch_size = metric.shape[0]
        singularities = []
        
        # Method 1: Check metric degeneracy
        det = torch.det(metric)
        eps = 1e-6
        
        # Create points from diagonal of metric
        points = torch.diagonal(metric, dim1=1, dim2=2)  # Shape: [batch_size, dim]
        
        for i in range(batch_size):
            if abs(det[i]) < eps:
                sing = SingularityInfo(
                    location=points[i],  # Use diagonal elements as location
                    curvature=1.0 - abs(det[i]),
                    resolution=torch.eye(self.manifold_dim, device=metric.device)
                )
                singularities.append(sing)
                
        # Method 2: Check flow field if provided
        if flow is not None:
            if len(flow.shape) == 3:
                # Convert 3D flow to 2D
                flat_flow = flow.view(batch_size, -1)
            else:
                flat_flow = flow
                
            # Concatenate metric and flow for detection
            input_tensor = torch.cat([metric.view(batch_size, -1), flat_flow], dim=1)
            
            # Detect potential singularities
            detection = self.detector(input_tensor)
            
            # Check each point for singularities
            for i in range(batch_size):
                if torch.any(detection[i] > 0.5):
                    # Get resolved position
                    resolved = self.resolver(input_tensor[i:i+1])
                    
                    # Create singularity info
                    info = SingularityInfo(
                        location=points[i],  # Use diagonal elements as location
                        curvature=detection[i].max().item(),
                        resolution=resolved[0]
                    )
                    singularities.append(info)
                
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


class GeometricFlow(nn.Module):
    """Geometric flow on a manifold."""

    def __init__(self, phase_dim: int = 4, hidden_dim: int = 128):
        """Initialize geometric flow.
        
        Args:
            phase_dim: Dimension of phase space (position + momentum)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.phase_dim = phase_dim
        self.manifold_dim = phase_dim // 2  # Position space dimension
        self.hidden_dim = hidden_dim
        
        # Networks
        self.ricci = RicciTensorNetwork(self.manifold_dim, hidden_dim)  # Use position space dim
        self.flow_network = nn.Sequential(
            nn.Linear(self.manifold_dim * 2, hidden_dim),  # Use position space dim
            nn.ReLU(),
            nn.Linear(hidden_dim, self.manifold_dim)  # Use position space dim
        )
        
        # Hamiltonian system for energy conservation
        from .hamiltonian import HamiltonianSystem
        self.hamiltonian = HamiltonianSystem(phase_dim, hidden_dim)  # Use full phase space dim
        
        # Singularity detection and normalization
        self.singularity = SingularityDetector(self.manifold_dim, hidden_dim // 4)  # Use position space dim
        self.normalizer = FlowNormalizer(self.manifold_dim, hidden_dim // 4)  # Use position space dim
        
        self._points = None
        self._metric = None
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass computing flow evolution.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim) for geometric flow
                   or (batch_size, phase_dim) for Hamiltonian evolution
            
        Returns:
            Evolved points tensor of shape (batch_size, phase_dim)
        """
        if points.shape[-1] == self.manifold_dim:
            # Position components only - do geometric flow
            metric = self.compute_metric(points)
            ricci = self.compute_ricci_tensor(metric, points)
            flow_vector = self.compute_flow(points, ricci)
            
            # Update position using flow
            new_pos = points + flow_vector
            
            # Create momentum for Hamiltonian evolution
            zeros = torch.zeros_like(points)
            phase_points = torch.cat([new_pos, zeros], dim=-1)  # [pos, zeros] for initial momentum
            
            # Get new momentum from Hamiltonian evolution
            evolved_points = self.hamiltonian(phase_points)  # Pass full phase space points
            return evolved_points
            
        elif points.shape[-1] == self.phase_dim:
            # Full phase space points - do Hamiltonian evolution only
            return self.hamiltonian(points)
            
        else:
            raise ValueError(f"Points must have shape (batch_size, {self.manifold_dim}) for geometric flow "
                           f"or (batch_size, {self.phase_dim}) for Hamiltonian evolution")
        
    @property
    def points(self) -> torch.Tensor:
        """Get points on manifold."""
        if self._points is None:
            raise ValueError("Points must be set before accessing")
        return self._points
        
    @points.setter 
    def points(self, value: torch.Tensor):
        """Set points on manifold.
        
        Args:
            value: Points tensor of shape (batch_size, manifold_dim)
        """
        if value.dim() != 2:
            raise ValueError(f"Points must be 2D tensor, got shape {value.shape}")
        if value.shape[1] != self.manifold_dim:
            raise ValueError(f"Points must have shape (batch_size, {self.manifold_dim})")
        self._points = value

    @property
    def metric(self) -> torch.Tensor:
        """Get metric tensor."""
        if self._metric is None:
            raise ValueError("Metric must be set before accessing")
        return self._metric
        
    @metric.setter
    def metric(self, value: torch.Tensor):
        """Set metric tensor.
        
        Args:
            value: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        if value.dim() != 3:
            raise ValueError(f"Metric must be 3D tensor, got shape {value.shape}")
        if value.shape[1:] != (self.manifold_dim, self.manifold_dim):
            raise ValueError(f"Metric must have shape (batch_size, {self.manifold_dim}, {self.manifold_dim})")
        self._metric = value

    def compute_connection(self, metric: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel connection.
        
        Args:
            metric: Metric tensor
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Connection tensor
        """
        batch_size = metric.shape[0]
        n = self.manifold_dim
        eps = 1e-6
        
        # Initialize tensors
        connection = torch.zeros(batch_size, n, n, n, device=metric.device)
        metric_deriv = torch.zeros(batch_size, n, n, n, device=metric.device)
        
        # Compute metric derivatives
        eye = torch.eye(n, device=metric.device).expand(batch_size, n, n)
        for k in range(n):
            # Forward difference for derivatives
            points_plus = points + eps * eye[:, :, k]
            metric_plus = self.compute_metric(points_plus)
            metric_deriv[:, k] = (metric_plus - metric) / eps
            
        # Compute inverse metric
        metric_inv = torch.linalg.pinv(metric)
        
        # Add small regularization to avoid singularities
        regularized_metric = metric + eps * torch.eye(n, device=metric.device)
        
        # Compute Christoffel symbols
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Combine terms
                    connection[:, k, i, j] = 0.5 * torch.sum(
                        metric_inv[:, k, :] * (
                            metric_deriv[:, i, :, j] +
                            metric_deriv[:, j, :, i] -
                            metric_deriv[:, :, i, j]
                        ), dim=1
                    )
        
        return connection

    def compute_ricci_tensor(self, metric: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Compute Ricci tensor.
        
        Args:
            metric: Metric tensor
            points: Points tensor
            
        Returns:
            Ricci tensor
        """
        batch_size = metric.shape[0]
        n = self.manifold_dim
        
        # Initialize Ricci tensor
        ricci = torch.zeros(batch_size, n, n, device=metric.device)
        
        # Compute Christoffel symbols
        christoffel = self.compute_connection(metric, points)
        
        # Contract indices to get Ricci tensor
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        ricci[:, i, j] += (
                            christoffel[:, k, i, l] * christoffel[:, l, j, k] -
                            christoffel[:, k, i, j] * christoffel[:, l, l, k]
                        )
        
        return ricci

    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        n = self.manifold_dim
        
        # Use metric network to compute components
        metric_components = self.ricci.metric_network(points)
        
        # Reshape to proper metric tensor shape
        metric = metric_components.view(batch_size, n, n)
        
        # Add small regularization for numerical stability
        eps = 1e-6
        eye = torch.eye(n, device=points.device)
        metric = metric + eps * eye.unsqueeze(0)
        
        # Ensure metric is symmetric
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        return metric

    def compute_flow(self, points: torch.Tensor, ricci: torch.Tensor) -> torch.Tensor:
        """Compute flow vector field.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor
            
        Returns:
            Flow vector field
        """
        batch_size = points.shape[0]
        
        # Reshape Ricci tensor for network input
        ricci_flat = ricci.reshape(batch_size, -1)
        ricci_flat = ricci_flat[:, :points.shape[1]]  # Take first manifold_dim components
        
        # Concatenate points with Ricci components
        flow_input = torch.cat([points, ricci_flat], dim=-1)
        
        # Compute flow through network
        flow = self.flow_network(flow_input)
        
        # Normalize flow
        flow_norm = torch.norm(flow, dim=-1, keepdim=True)
        flow = flow / (flow_norm + 1e-8)
        
        return flow

    def flow_step(self, metric: torch.Tensor, ricci: torch.Tensor, timestep: Optional[float] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Perform one step of geometric flow.
        
        Args:
            metric: Metric tensor of shape [batch_size, dim, dim]
            ricci: Ricci tensor
            timestep: Time step size (optional)
            
        Returns:
            Tuple of (evolved metric, [initial_metric, evolved_metric])
        """
        batch_size = metric.shape[0]
        dt = timestep if timestep is not None else 0.01

        # Compute flow vector field
        flow = -ricci  # Ricci flow

        # Normalize flow
        flow = self.normalize_flow(flow, metric)
        
        # Evolve metric
        evolved_metric = metric + dt * flow
        
        # Ensure positive definiteness
        eigenvals = torch.linalg.eigvalsh(evolved_metric)
        min_eigenval = eigenvals.min()
        if min_eigenval < 0:
            # Add small positive diagonal term to ensure positive definiteness
            evolved_metric = evolved_metric + (-min_eigenval + 1e-6) * torch.eye(
                self.manifold_dim, device=metric.device
            ).unsqueeze(0).repeat(batch_size, 1, 1)

        # Return initial and evolved metrics
        return evolved_metric, [metric, evolved_metric]

    def detect_singularities(
        self, metric: torch.Tensor, flow: Optional[torch.Tensor] = None
    ) -> List[SingularityInfo]:
        """Detect singularities in the metric tensor and optionally in the flow field.
        
        Args:
            metric: Metric tensor of shape [batch_size, dim, dim]
            flow: Optional flow tensor of shape [batch_size, dim] or [batch_size, dim, dim]
            
        Returns:
            List of detected singularities
        """
        return self.singularity.detect_singularities(metric, flow)

    def evolve(
        self, points: torch.Tensor, num_steps: int = 100, dt: float = 0.01
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[SingularityInfo]]]:
        """Evolve points along geometric flow.
        
        Args:
            points: Points tensor of shape [batch_size, dim]
            num_steps: Number of evolution steps
            dt: Time step size
            
        Returns:
            Tuple of:
            - List of point trajectories
            - List of metrics
            - List of singularities at each step
        """
        trajectories = [points]
        metrics = []
        all_singularities = []
        current = points.clone()

        for _ in range(num_steps):
            # Compute metric and Ricci tensor
            metric = self.compute_metric(current)
            ricci = self.compute_ricci_tensor(metric, current)

            # Evolve one step
            new_points, step_metrics = self.flow_step(current, ricci, dt)

            # Detect singularities
            singularities = self.detect_singularities(metric)

            # Update state
            current = new_points
            trajectories.append(current)
            metrics.append(step_metrics)
            all_singularities.append(singularities)

        return trajectories, metrics, all_singularities

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
        ricci = self.compute_ricci_tensor(metric, self.points)
        
        # Perform flow step
        evolved_metric, metrics = self.flow_step(metric, ricci)
        
        return evolved_metric, metrics

    def compute_flow_vector(self, points: torch.Tensor, ricci: torch.Tensor) -> torch.Tensor:
        """Legacy method for computing flow vector field.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor
            
        Returns:
            Flow vector field
        """
        return self.compute_flow(points, ricci)

    def compute_mean_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute mean curvature from metric tensor.
        
        Args:
            metric: Metric tensor
            
        Returns:
            Mean curvature tensor
        """
        batch_size = metric.shape[0]
        n = self.manifold_dim
        
        # Compute Christoffel symbols
        christoffel = self.compute_connection(metric, self.points)
        
        # Compute mean curvature
        mean_curv = torch.zeros(batch_size, device=metric.device)
        for i in range(n):
            for j in range(n):
                mean_curv += torch.diagonal(metric, dim1=1, dim2=2)[:, i] * christoffel[:, i, j, j]
                
        return mean_curv / (n * (n-1))

    def detect_singularities(self, metric: torch.Tensor) -> List[SingularityInfo]:
        """Detect singularities in metric tensor.
        
        Args:
            metric: Metric tensor
            
        Returns:
            List of detected singularities
        """
        return self.singularity.detect_singularities(metric)

    def normalize_flow(self, flow: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Normalize flow vector field.
        
        Args:
            flow: Flow vector field
            metric: Metric tensor
            
        Returns:
            Normalized flow vector field
        """
        # Compute flow magnitude using metric
        flow_norm = torch.sqrt(torch.einsum('...ij,...ij->...', flow, flow))
        flow_norm = flow_norm.unsqueeze(-1).unsqueeze(-1)
        
        # Add small epsilon to avoid division by zero
        normalized_flow = flow / (flow_norm + 1e-8)
        
        # Scale to reasonable magnitude
        normalized_flow = normalized_flow * 0.1
        
        return normalized_flow

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

    def compute_energy(self, states: torch.Tensor) -> torch.Tensor:
        """Compute energy functional.
        
        Args:
            states: States tensor of shape (batch_size, phase_dim)
            
        Returns:
            Energy tensor of shape (batch_size,)
        """
        # Split into position and momentum
        pos = states[..., :self.manifold_dim]
        mom = states[..., self.manifold_dim:]
        
        # Compute kinetic energy (from momentum)
        kinetic = 0.5 * torch.sum(mom * mom, dim=-1)
        
        # Compute potential energy (using position)
        metric = self.compute_metric(pos)
        potential = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)
        
        # Total energy is sum of kinetic and potential
        return kinetic + potential
