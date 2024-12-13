"""Neural Geometric Flow Implementation.

This module implements geometric flows for neural attention:
- Ricci tensor computation
- Flow step implementation
- Singularity detection and handling
- Flow normalization
- Energy conservation
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FlowMetrics:
    """Metrics for geometric flow analysis."""
    
    flow_magnitude: torch.Tensor  # Magnitude of flow vector field
    metric_determinant: torch.Tensor  # Determinant of metric tensor
    ricci_scalar: torch.Tensor  # Scalar curvature
    energy: torch.Tensor  # Flow energy
    singularity: torch.Tensor  # Singularity measure
    normalized_flow: torch.Tensor  # Normalized flow vector


class SingularityInfo:
    """Information about a geometric singularity."""
    
    def __init__(self, index: int, determinant: float, condition_number: float, min_eigenvalue: float, 
                 location: Optional[torch.Tensor] = None, curvature: Optional[torch.Tensor] = None):
        self.index = index
        self.determinant = determinant
        self.condition_number = condition_number
        self.min_eigenvalue = min_eigenvalue
        self.location = location if location is not None else torch.zeros(2)
        self.curvature = curvature if curvature is not None else torch.zeros(2, 2)

    def is_removable(self) -> bool:
        """Check if singularity is removable via resolution."""
        return self.condition_number < 1e6 and self.min_eigenvalue > -1e-4
        
    def get_blowup_rate(self) -> float:
        """Get rate of curvature blowup near singularity."""
        return max(self.condition_number, 1.0 / abs(self.min_eigenvalue + 1e-8))


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
        
        # Compute base metric components using neural network
        metric_components = self.metric_network(points)
        
        # Flatten the metric components to correct size
        metric_components = metric_components.view(batch_size, -1)[:, :self.manifold_dim * self.manifold_dim]
        
        # Reshape into matrix form
        metric = metric_components.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Ensure symmetry
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Make positive definite using matrix exponential
        # First normalize to prevent overflow
        metric = metric / (torch.norm(metric, dim=(-2, -1), keepdim=True) + 1e-8)
        metric = torch.matrix_exp(metric)
        
        # Add regularization
        eye = torch.eye(self.manifold_dim, device=points.device).unsqueeze(0)
        metric = metric + eye * 1e-3
        
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

        # Metric computation network
        self.metric_network = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        )

    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        manifold_dim = points.shape[1]
        
        # Compute metric components
        metric_flat = self.metric_network(points)
        
        # Reshape to metric tensor
        return metric_flat.view(batch_size, manifold_dim, manifold_dim)

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
        self, metric: torch.Tensor, points: torch.Tensor, threshold: float = 1e-6
    ) -> List[SingularityInfo]:
        """Detect singularities in the metric at given points.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            points: Points tensor of shape (batch_size, manifold_dim)
            threshold: Threshold for singularity detection
            
        Returns:
            List of SingularityInfo objects for detected singularities
        """
        singularities = []
        batch_size = metric.shape[0]
        
        # Add small regularization for numerical stability
        metric_reg = metric + torch.eye(self.manifold_dim, device=metric.device).unsqueeze(0) * 1e-8
        
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
        
        for i in range(batch_size):
            if (abs(det[i]) < threshold or  # Near-zero determinant
                cond[i] > 1.0/threshold or  # Poor conditioning
                min_eigenval[i] < threshold):  # Near-zero eigenvalue
                
                # Create singularity info
                info = SingularityInfo(
                    index=i,
                    determinant=det[i].item(),
                    condition_number=cond[i].item(),
                    min_eigenvalue=min_eigenval[i].item()
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
        types: list[str] = ["removable", "essential", "conical", "cusp"]
        idx: int = int(torch.argmax(type_idx, dim=-1).item())
        return types[idx]


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

    def compute_connection(self, metric: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel connection coefficients.
        
        Args:
            metric: Metric tensor (batch_size x dim x dim)
            position: Points tensor (batch_size x dim)
            
        Returns:
            Connection coefficients (batch_size x dim x dim x dim)
        """
        batch_size = metric.shape[0]
        dim = metric.shape[1]
        
        # Initialize tensors
        connection = torch.zeros(batch_size, dim, dim, dim, device=metric.device)
        metric_deriv = torch.zeros(batch_size, dim, dim, dim, device=metric.device)
        
        # Ensure position has correct shape (batch_size x dim)
        if len(position.shape) > 2:
            position = position.reshape(batch_size, -1)[:, :dim]
        elif position.shape[1] > dim:
            position = position[:, :dim]
            
        # Compute metric derivatives using finite differences
        eps = 1e-6
        eye = torch.eye(dim, device=metric.device)
        
        for k in range(dim):
            # Create offset points for finite differences
            offset = eps * eye[k].view(1, -1).expand(batch_size, -1)
            points_plus = position + offset
            points_minus = position - offset
            
            # Compute metric at offset points
            metric_plus = self.compute_metric(points_plus)
            metric_minus = self.compute_metric(points_minus)
            
            # Compute derivative
            metric_deriv[:, k] = (metric_plus - metric_minus) / (2 * eps)
        
        # Compute connection coefficients
        metric_inv = torch.inverse(metric)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # Compute terms for Christoffel symbols
                    term1 = metric_deriv[:, k, :, j]  # ∂_k g_{mj}
                    term2 = metric_deriv[:, j, :, k]  # ∂_j g_{mk}
                    term3 = metric_deriv[:, :, j, k]  # ∂_m g_{jk}
                    
                    # Contract with inverse metric
                    connection[:, i, j, k] = 0.5 * torch.sum(
                        metric_inv[:, i, :] * (term1 + term2 - term3),
                        dim=1
                    )
        
        return connection

    def compute_ricci_tensor(self, metric: torch.Tensor, points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Ricci curvature tensor.
        
        Args:
            metric: Metric tensor (batch_size x dim x dim)
            points: Optional points tensor (batch_size x dim)
            
        Returns:
            Ricci tensor (batch_size x dim x dim)
        """
        batch_size = metric.shape[0]
        dim = metric.shape[1]
        
        # If points not provided, use stored points
        if points is None:
            points = self._points
            if points is None:
                points = torch.zeros(batch_size, dim, device=metric.device)
        
        # Ensure points have correct shape (batch_size x dim)
        if len(points.shape) > 2:
            points = points.reshape(batch_size, -1)[:, :dim]
        elif points.shape[1] > dim:
            points = points[:, :dim]
            
        # Compute connection coefficients
        christoffel = self.compute_connection(metric, points)
        
        # Initialize Riemann tensor
        riemann = torch.zeros(batch_size, dim, dim, dim, dim, device=metric.device)
        
        # Compute Riemann curvature tensor components
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        # R^i_{jkl} = \partial_k \Gamma^i_{jl} - \partial_l \Gamma^i_{jk} + ...
                        term1 = torch.sum(christoffel[:, i, :, k] * christoffel[:, :, j, l], dim=1)
                        term2 = torch.sum(christoffel[:, i, :, l] * christoffel[:, :, j, k], dim=1)
                        riemann[:, i, j, k, l] = term1 - term2
        
        # Contract to get Ricci tensor
        ricci = torch.zeros(batch_size, dim, dim, device=metric.device)
        metric_inv = torch.inverse(metric)
        
        for i in range(dim):
            for j in range(dim):
                # Contract Riemann tensor with metric to get Ricci tensor
                ricci[:, i, j] = torch.sum(
                    metric_inv[:, :, :] * riemann[:, :, i, :, j],
                    dim=(1, 2)
                )
                
        return ricci

    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim) or (batch_size, phase_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        # Extract position coordinates if full phase space is provided
        if points.shape[1] == self.phase_dim:
            position = points[:, :self.manifold_dim]
        else:
            position = points
            
        batch_size = position.shape[0]
        
        # Initialize metric tensor
        metric = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, 
                           device=points.device)
        
        # Get metric components from network
        metric_components = self.ricci.metric_network(position)  # Shape: (batch_size, manifold_dim * manifold_dim)
        
        # Flatten the metric components to correct size
        metric_components = metric_components.view(batch_size, -1)[:, :self.manifold_dim * self.manifold_dim]
        
        # Reshape into matrix form
        metric = metric_components.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Ensure symmetry
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Make positive definite
        eye = torch.eye(self.manifold_dim, device=points.device).unsqueeze(0)
        metric = metric + eye * 1e-4
        metric = torch.matrix_exp(metric)
        
        return metric

    def compute_flow(self, points: torch.Tensor, ricci: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute flow vector field.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor of shape (batch_size, manifold_dim) or (batch_size, manifold_dim, manifold_dim)
            metric: Optional metric tensor
            
        Returns:
            Flow vector field of shape (batch_size, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Ensure Ricci tensor has correct shape
        if len(ricci.shape) == 2:
            # Convert vector to diagonal matrix
            ricci_matrix = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, 
                                     device=points.device)
            ricci_matrix.diagonal(dim1=1, dim2=2)[:] = ricci
            ricci = ricci_matrix
        
        # Compute metric at current points
        if metric is None:
            metric = self.compute_metric(points)
        
        # Add regularization for numerical stability
        eye = torch.eye(self.manifold_dim, device=points.device).unsqueeze(0)
        metric_reg = metric + eye * 1e-6
        
        # Compute inverse metric
        metric_inv = torch.linalg.inv(metric_reg)
        
        # Contract tensors to get flow vector
        flow = torch.einsum('bij,bjk->bik', metric_inv, ricci)
        flow_vector = torch.diagonal(flow, dim1=1, dim2=2)
        
        # Normalize flow
        flow_norm = torch.norm(flow_vector, dim=1, keepdim=True)
        flow_vector = flow_vector / (flow_norm + 1e-8)
        
        # Scale flow to prevent instability
        flow_vector = flow_vector * 0.1
        
        return flow_vector

    def flow_step(
            self, 
            input_tensor: torch.Tensor, 
            ricci: Optional[torch.Tensor] = None, 
            dt: Union[float, torch.Tensor] = 0.01
        ) -> Tuple[torch.Tensor, FlowMetrics]:
        """Perform one step of geometric flow evolution.
        
        The geometric flow system operates in two spaces:
        1. Position space (manifold_dim = phase_dim/2 dimensions)
           - Points: (batch_size, manifold_dim)
           - Metric: (batch_size, manifold_dim, manifold_dim)
           - Ricci: (batch_size, manifold_dim, manifold_dim)
        
        2. Phase space (phase_dim dimensions)
           - States: (batch_size, phase_dim)
           - Hamiltonian: (batch_size,)
        
        This method handles two types of evolution:
        1. Metric evolution (Ricci flow):
           g(t+dt) = g(t) - 2*Ric(g(t))*dt
        
        2. Point evolution (geometric flow):
           x(t+dt) = x(t) + v(t)*dt
           where v(t) is the normalized flow vector
        
        Args:
            input_tensor: Either:
                - Points tensor of shape (batch_size, manifold_dim) for point evolution
                - Metric tensor of shape (batch_size, manifold_dim, manifold_dim) for metric evolution
            ricci: Optional Ricci tensor for metric evolution of shape (batch_size, manifold_dim, manifold_dim)
                  If None, will be computed from the metric
            dt: Time step size (scalar or tensor)
            
        Returns:
            Tuple of (evolved tensor, flow metrics):
            - For metric evolution: (evolved_metric, flow_metrics)
              evolved_metric: (batch_size, manifold_dim, manifold_dim)
            - For point evolution: (evolved_points, flow_metrics)
              evolved_points: (batch_size, manifold_dim)
        """
        # Convert dt to tensor if it's a float
        if isinstance(dt, float):
            dt = torch.tensor(dt, device=input_tensor.device)
        
        # Determine if input is points or metric based on shape
        is_metric = len(input_tensor.shape) == 3
        batch_size = input_tensor.shape[0]
        manifold_dim = self.manifold_dim  # Position space dimension
        
        if is_metric:
            # Metric evolution using Ricci flow
            metric = input_tensor  # Shape: (batch_size, manifold_dim, manifold_dim)
            assert metric.shape == (batch_size, manifold_dim, manifold_dim), \
                f"Expected metric shape {(batch_size, manifold_dim, manifold_dim)}, got {metric.shape}"
            
            # If Ricci tensor not provided, compute it
            if ricci is None:
                # Initialize points if not provided
                if self.points is None:
                    points = torch.zeros(batch_size, manifold_dim, device=metric.device)
                else:
                    # Ensure points are in position space
                    points = self.points
                    if points.shape[1] == self.phase_dim:
                        points = points[:, :manifold_dim]  # Take position components
                
                ricci = self.compute_ricci_tensor(metric, points)
            
            assert ricci.shape == (batch_size, manifold_dim, manifold_dim), \
                f"Expected Ricci shape {(batch_size, manifold_dim, manifold_dim)}, got {ricci.shape}"
            
            # Evolve metric using Ricci flow: g(t+dt) = g(t) - 2*Ric(g(t))*dt
            evolved_metric = metric - 2 * dt * ricci
            
            # Compute flow metrics
            metrics = FlowMetrics(
                flow_magnitude=torch.norm(ricci.view(metric.shape[0], -1), dim=1),
                metric_determinant=torch.linalg.det(evolved_metric),
                ricci_scalar=torch.diagonal(ricci, dim1=1, dim2=2).sum(dim=1),
                energy=torch.linalg.det(evolved_metric),
                singularity=torch.linalg.det(metric),
                normalized_flow=torch.linalg.det(evolved_metric)
            )
            
            return evolved_metric, metrics
            
        else:
            # Point evolution using flow vector field
            points = input_tensor  # Shape: (batch_size, manifold_dim)
            assert points.shape == (batch_size, manifold_dim), \
                f"Expected points shape {(batch_size, manifold_dim)}, got {points.shape}"
            
            # Compute metric from points
            metric = self.compute_metric(points)  # Shape: (batch_size, manifold_dim, manifold_dim)
            
            # Compute Ricci tensor and flow vector
            ricci_tensor = self.compute_ricci_tensor(metric, points)
            flow_vector = self.compute_flow(points, ricci_tensor, metric)  # Shape: (batch_size, manifold_dim)
            
            # Normalize flow vector to preserve volume
            flow_vector = self.normalize_flow(flow_vector, metric)
            
            # Update position using flow with scalar dt
            evolved_points = points + dt * flow_vector
            
            # Compute flow metrics
            metrics = FlowMetrics(
                flow_magnitude=torch.norm(flow_vector, dim=1),
                metric_determinant=torch.linalg.det(metric),
                ricci_scalar=torch.diagonal(ricci_tensor, dim1=1, dim2=2).sum(dim=1),
                energy=torch.linalg.det(metric),
                singularity=torch.linalg.det(metric),
                normalized_flow=torch.linalg.det(metric)
            )
            
            return evolved_points, metrics

    def detect_singularities(
        self, 
        metric: torch.Tensor, 
        points: Optional[torch.Tensor] = None,
        threshold: float = 1e-6
    ) -> List[SingularityInfo]:
        """Detect singularities in the metric at given points.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            points: Optional points tensor of shape (batch_size, manifold_dim)
            threshold: Threshold for singularity detection
            
        Returns:
            List of SingularityInfo objects for detected singularities
        """
        batch_size = metric.shape[0]
        
        # Use SingularityDetector if points are provided
        if points is not None:
            return self.singularity.detect_singularities(metric, points, threshold)
            
        # Otherwise do direct metric analysis
        det = torch.linalg.det(metric)
        U, S, Vh = torch.linalg.svd(metric)
        
        # Get condition number and eigenvalues 
        cond = S.max(dim=-1)[0] / S.min(dim=-1)[0]
        eigenvals = torch.linalg.eigvals(metric).real
        min_eigenval = eigenvals.min(dim=-1)[0]
        
        # Compute Ricci curvature
        ricci = self.compute_ricci_tensor(metric)
        
        # Create masks for different singularity conditions
        det_mask = (torch.abs(det) < threshold).to(torch.bool)
        cond_mask = (cond > 1/threshold).to(torch.bool) 
        eigenval_mask = (min_eigenval < threshold).to(torch.bool)
        curv_mask = torch.zeros_like(det_mask)
        
        # Combine masks
        singular_mask = det_mask | cond_mask | eigenval_mask | curv_mask
        
        # Get indices of singular points
        singular_indices = torch.where(singular_mask)[0]
        
        # Create singularity info objects
        singularities = []
        for idx in singular_indices:
            location = self._points[idx] if self._points is not None else torch.zeros(self.manifold_dim)
            info = SingularityInfo(
                index=int(idx.item()),  # Explicitly cast to int
                determinant=float(det[idx].item()),
                condition_number=float(cond[idx].item()),
                min_eigenvalue=float(min_eigenval[idx].item()),
                location=location,
                curvature=ricci[idx]
            )
            singularities.append(info)
            
        return singularities

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
            new_points, step_metrics = self.flow_step_points(current, dt)

            # Detect singularities
            singularities = self.detect_singularities(metric, current)

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
        # For each metric tensor, normalize the flow using itself as the metric
        return [self.normalize_flow(metric, metric) for metric in metrics]

    def step(self, metric: torch.Tensor) -> Tuple[torch.Tensor, FlowMetrics]:
        """Perform a single flow step.

        Args:
            metric: Current metric tensor of shape (batch_size, manifold_dim, manifold_dim)

        Returns:
            Tuple of (evolved metric, flow metrics)
        """
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Evolve metric using Ricci flow
        evolved_metric = metric - 2 * ricci * 0.01  # Fixed small dt
        
        # Compute flow metrics
        metrics = FlowMetrics(
            flow_magnitude=torch.norm(ricci.view(metric.shape[0], -1), dim=1),
            metric_determinant=torch.linalg.det(evolved_metric),
            ricci_scalar=torch.diagonal(ricci, dim1=1, dim2=2).sum(dim=1),
            energy=torch.linalg.det(evolved_metric),
            singularity=torch.linalg.det(metric),
            normalized_flow=torch.linalg.det(evolved_metric)
        )
        
        return evolved_metric, metrics

    def compute_flow_vector(self, points: torch.Tensor, ricci: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Legacy method for computing flow vector field.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            ricci: Ricci tensor
            metric: Optional metric tensor
            
        Returns:
            Flow vector field
        """
        return self.compute_flow(points, ricci, metric)

    def compute_mean_curvature(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute mean curvature from metric tensor.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            
        Returns:
            Mean curvature tensor of shape (batch_size, manifold_dim)
        """
        if self._points is None:
            raise ValueError("Points must be set before accessing")
            
        batch_size = metric.shape[0]
        
        # Compute Christoffel symbols
        christoffel = self.compute_connection(metric, self._points)
        
        # Compute mean curvature components
        mean_curvature = torch.zeros((batch_size, self.manifold_dim), device=metric.device)
        
        # H^i = g^{jk} Γ^i_{jk}
        metric_inv = torch.inverse(metric)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    mean_curvature[:, i] += metric_inv[:, j, k] * christoffel[:, i, j, k]
                    
        return mean_curvature

    def flow_step_points(
        self, 
        points: torch.Tensor, 
        dt: Union[float, torch.Tensor] = 0.01
    ) -> Tuple[torch.Tensor, FlowMetrics]:
        """Perform one step of geometric flow evolution.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim) or (batch_size, phase_dim)
            dt: Time step size (scalar or tensor)
            
        Returns:
            Tuple of (evolved points, flow metrics)
        """
        # Convert dt to tensor if float
        if isinstance(dt, float):
            dt = torch.tensor(dt, device=points.device)

        # Compute metric and flow
        metric = self.compute_metric(points)
        ricci = self.compute_ricci_tensor(metric, points)
        flow_vector = self.compute_flow(points, ricci, metric)
        
        # Normalize flow
        normalized_flow = self.normalize_flow(flow_vector, metric)
        
        # Evolve points
        evolved_points = points + normalized_flow * dt
        
        # Compute metrics
        metrics = FlowMetrics(
            flow_magnitude=torch.norm(flow_vector, dim=1),
            metric_determinant=torch.linalg.det(metric),
            ricci_scalar=torch.diagonal(ricci, dim1=1, dim2=2).sum(dim=1),
            energy=torch.linalg.det(metric),
            singularity=torch.linalg.det(metric),
            normalized_flow=torch.linalg.det(metric)
        )
        
        return evolved_points, metrics

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
        if type_idx == 0:
            return "removable"
        elif type_idx == 1:
            return "essential"
        elif type_idx == 2:
            return "pole"
        else:
            return "unknown"
