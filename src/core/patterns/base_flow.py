"""Base implementation of geometric flow.

This module provides the base implementation of geometric flow,
serving as a foundation for more specific flow implementations.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
from torch import nn

from ..types import GeometricFlowProtocol

class BaseGeometricFlow(nn.Module, GeometricFlowProtocol):
    """Base implementation of geometric flow.
    
    This class provides core functionality for geometric flows,
    implementing the GeometricFlowProtocol with basic versions
    of required methods.
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize base geometric flow.
        
        Args:
            manifold_dim: Dimension of the manifold
            hidden_dim: Hidden dimension for flow computations
            num_layers: Number of layers in flow network
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
            dtype: Data type for tensors
        """
        super().__init__()
        
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim or manifold_dim * 4
        self.num_layers = num_layers
        self.dt = dt
        self.stability_threshold = stability_threshold
        self.dtype = dtype
        
        # Initialize metric network for real and imaginary parts separately
        metric_output_dim = manifold_dim * manifold_dim
        
        # Real part network
        real_layers = []
        real_layers.append(nn.Linear(manifold_dim, self.hidden_dim, dtype=torch.float32))
        real_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            real_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float32))
            real_layers.append(nn.Tanh())
        real_layers.append(nn.Linear(self.hidden_dim, metric_output_dim, dtype=torch.float32))
        self.real_metric_net = nn.Sequential(*real_layers)
        
        # Imaginary part network
        imag_layers = []
        imag_layers.append(nn.Linear(manifold_dim, self.hidden_dim, dtype=torch.float32))
        imag_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            imag_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float32))
            imag_layers.append(nn.Tanh())
        imag_layers.append(nn.Linear(self.hidden_dim, metric_output_dim, dtype=torch.float32))
        self.imag_metric_net = nn.Sequential(*imag_layers)
        
        # Initialize weights
        for net in [self.real_metric_net, self.imag_metric_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Initialize basic flow network
        self.flow_layers = nn.ModuleList([
            nn.Linear(manifold_dim, self.hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, manifold_dim, dtype=torch.float32)
        ])
    
    def compute_ricci_tensor(
        self,
        metric: torch.Tensor,
        connection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Base implementation of Ricci tensor computation.
        
        Args:
            metric: Metric tensor at current point
            connection: Optional connection form
            
        Returns:
            Ricci curvature tensor
        """
        batch_size = metric.shape[0]
        
        # Basic Ricci computation (to be overridden by specific implementations)
        ricci = torch.zeros_like(metric)
        
        if connection is not None:
            # Use connection to modify Ricci tensor
            ricci = ricci + torch.einsum('...ij,...jk->...ik', connection, metric)
        
        return ricci
    
    def flow_step(
        self,
        metric: torch.Tensor,
        ricci: torch.Tensor,
        timestep: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform one step of geometric flow.
        
        Args:
            metric: Current metric tensor
            ricci: Ricci curvature tensor
            timestep: Integration time step
            
        Returns:
            Tuple of (new_metric, flow_metrics)
        """
        # Compute eigendecomposition of metric
        eigvals, eigvecs = torch.linalg.eigh(metric)
        
        # Ensure metric stays positive definite
        min_eigval = torch.min(torch.abs(eigvals))
        if min_eigval <= self.stability_threshold:
            # Add small positive constant to maintain positive definiteness
            eigvals = eigvals + (-min_eigval + self.stability_threshold)
            metric = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1).conj()
            
        # Scale timestep based on metric and Ricci norms
        # Compute norms while preserving batch dimensions
        metric_flat = metric.reshape(-1, metric.shape[-1] * metric.shape[-1])
        ricci_flat = ricci.reshape(-1, ricci.shape[-1] * ricci.shape[-1])
        
        metric_norm = torch.norm(metric_flat, dim=1, keepdim=True)
        ricci_norm = torch.norm(ricci_flat, dim=1, keepdim=True)
        
        # Compute adaptive timestep with proper broadcasting
        adaptive_timestep = timestep * torch.minimum(
            torch.ones_like(metric_norm),
            0.1 * metric_norm / (ricci_norm + self.stability_threshold)
        )
        
        # Reshape adaptive timestep for broadcasting with metric and flow tensors
        adaptive_timestep = adaptive_timestep.view(*metric.shape[:-2], 1, 1)
        
        # Compute flow step with positivity preservation
        flow = -2 * ricci
        
        # Ensure flow has the same shape as metric
        flow = flow.reshape(*metric.shape)
        
        # Compute eigendecomposition of flow
        flow_eigvals, flow_eigvecs = torch.linalg.eigh(flow)
        
        # Limit the magnitude of negative eigenvalues
        max_neg_eigval = -self.stability_threshold / adaptive_timestep
        flow_eigvals = torch.clamp(torch.abs(flow_eigvals), min=max_neg_eigval) * torch.exp(1j * torch.angle(flow_eigvals))
        
        # Reconstruct flow with limited negative eigenvalues
        flow = flow_eigvecs @ torch.diag_embed(flow_eigvals) @ flow_eigvecs.transpose(-2, -1).conj()
        
        # Update metric
        new_metric = metric + flow * adaptive_timestep
        
        # Project back to positive definite cone if needed
        eigvals, eigvecs = torch.linalg.eigh(new_metric)
        min_eigval = torch.min(torch.abs(eigvals))
        if min_eigval <= self.stability_threshold:
            eigvals = torch.clamp(torch.abs(eigvals), min=self.stability_threshold) * torch.exp(1j * torch.angle(eigvals))
            new_metric = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1).conj()
        
        # Normalize metric if needed
        new_metric_flat = new_metric.reshape(-1, new_metric.shape[-1] * new_metric.shape[-1])
        new_metric_norm = torch.norm(new_metric_flat, dim=1, keepdim=True).view(-1, 1, 1)
        
        if torch.any(new_metric_norm > 1.0 / self.stability_threshold):
            new_metric = new_metric * (1.0 / self.stability_threshold) / new_metric_norm
        
        # Ensure symmetry
        new_metric = 0.5 * (new_metric + new_metric.transpose(-2, -1).conj())
        
        # Add small positive constant to diagonal for stability
        new_metric = new_metric + torch.eye(new_metric.shape[-1], device=new_metric.device, dtype=new_metric.dtype).unsqueeze(0) * self.stability_threshold
        
        # Compute basic flow metrics
        metrics = {
            'metric_norm': torch.norm(new_metric).item(),
            'ricci_norm': torch.norm(ricci).item(),
            'step_size': adaptive_timestep.mean().item(),
            'min_eigval': torch.min(torch.abs(eigvals)).item(),
            'max_eigval': torch.max(torch.abs(eigvals)).item(),
            'condition_number': (torch.max(torch.abs(eigvals)) / torch.clamp(torch.min(torch.abs(eigvals)), min=self.stability_threshold)).item()
        }
        
        return new_metric, metrics
    
    def detect_singularities(
        self,
        flow: torch.Tensor,
        threshold: float = 1e-6
    ) -> List[Dict[str, Any]]:
        """Detect basic singularities in the flow.
        
        Args:
            flow: The geometric flow tensor
            threshold: Detection threshold
            
        Returns:
            List of detected singularities with metadata
        """
        singularities = []
        
        # Basic singularity detection based on metric degeneration
        eigenvals = torch.linalg.eigvals(flow).real
        min_eigenval = torch.min(eigenvals, dim=-1)[0]
        
        # Find points where metric becomes degenerate
        singular_points = torch.where(min_eigenval < threshold)[0]
        
        for idx in singular_points:
            singularities.append({
                'position': idx.item(),
                'eigenvalue': min_eigenval[idx].item(),
                'type': 'metric_degeneration'
            })
        
        return singularities
    
    def normalize_flow(
        self,
        flow: torch.Tensor,
        normalization: str = "ricci"
    ) -> torch.Tensor:
        """Normalize the geometric flow.
        
        Args:
            flow: Flow tensor to normalize
            normalization: Type of normalization
            
        Returns:
            Normalized flow tensor
        """
        if normalization == "ricci":
            # Normalize by Ricci scalar
            scalar = torch.diagonal(flow, dim1=-2, dim2=-1).sum(-1, keepdim=True)
            normalized = flow / (scalar.abs() + self.stability_threshold)
        else:
            # Default L2 normalization
            norm = torch.norm(flow, dim=(-2, -1), keepdim=True)
            normalized = flow / (norm + self.stability_threshold)
        
        return normalized
    
    def forward(
        self,
        x: torch.Tensor,
        return_path: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply geometric flow to input tensor.
        
        Args:
            x: Input tensor
            return_path: Whether to return intermediate flow path
            
        Returns:
            Tuple of (flowed_tensor, flow_metrics)
        """
        # Initialize flow path if requested
        path: List[torch.Tensor] = [x] if return_path else []
        
        # Compute initial metric
        metric = self.compute_metric(x)
        
        # Apply flow layers
        current = x
        for layer in self.flow_layers:
            current = layer(current)
            if return_path:
                path.append(current)
        
        metrics = {
            'initial_metric_norm': torch.norm(metric).item(),
            'final_metric_norm': torch.norm(self.compute_metric(current)).item()
        }
        
        if return_path:
            metrics['path'] = path
            
        return current, metrics
    
    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from input.
        
        Args:
            x: Input tensor
            
        Returns:
            Metric tensor
        """
        print("\nIn compute_metric:")
        print(f"Input requires_grad: {x.requires_grad}")
        
        # Ensure input requires gradients
        x = x.requires_grad_(True)
        
        # Debug print metric network parameters
        for name, param in self.real_metric_net.named_parameters():
            print(f"real_metric_net {name} requires_grad: {param.requires_grad}")
            if param.grad is not None:
                print(f"real_metric_net {name} has gradients")
        
        # Compute real and imaginary parts of metric
        real_metric = self.real_metric_net(x)
        print(f"real_metric requires_grad: {real_metric.requires_grad}")
        
        imag_metric = self.imag_metric_net(x)
        print(f"imag_metric requires_grad: {imag_metric.requires_grad}")
        
        # Reshape to square matrices
        real_metric = real_metric.view(-1, self.manifold_dim, self.manifold_dim)
        imag_metric = imag_metric.view(-1, self.manifold_dim, self.manifold_dim)
        
        # Make metrics symmetric
        real_metric = 0.5 * (real_metric + real_metric.transpose(-2, -1))
        imag_metric = 0.5 * (imag_metric + imag_metric.transpose(-2, -1))
        
        # Add small positive constant to diagonal for stability
        real_metric = real_metric + torch.eye(self.manifold_dim, device=x.device, dtype=x.dtype) * 1e-6
        
        # Combine into complex metric
        metric = torch.complex(real_metric, imag_metric)
        print(f"Final metric requires_grad: {metric.requires_grad}")
        
        # Ensure metric requires gradients
        metric = metric.requires_grad_(True)
        
        return metric 