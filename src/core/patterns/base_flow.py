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
        
        # Store parameters
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim or manifold_dim * 4
        self.num_layers = num_layers
        self.dt = dt
        self.stability_threshold = stability_threshold
        self.dtype = dtype
        
        # Calculate metric output dimension
        metric_output_dim = manifold_dim * manifold_dim
        
        # Real part network with gradient tracking
        real_layers = []
        real_layers.append(nn.Linear(manifold_dim, self.hidden_dim, dtype=torch.float32))
        real_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            real_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float32))
            real_layers.append(nn.Tanh())
        real_layers.append(nn.Linear(self.hidden_dim, metric_output_dim, dtype=torch.float32))
        self.real_metric_net = nn.Sequential(*real_layers)
        
        # Initialize weights for real_metric_net with proper gradient tracking
        for layer in self.real_metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    # Initialize bias with small positive values for stability
                    nn.init.constant_(layer.bias, 0.1)
                    layer.bias.requires_grad_(True)
        
        # Imaginary part network
        imag_layers = []
        imag_layers.append(nn.Linear(manifold_dim, self.hidden_dim, dtype=torch.float32))
        imag_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            imag_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float32))
            imag_layers.append(nn.Tanh())
        imag_layers.append(nn.Linear(self.hidden_dim, metric_output_dim, dtype=torch.float32))
        self.imag_metric_net = nn.Sequential(*imag_layers)
        
        # Initialize weights for imag_metric_net with proper gradient tracking
        for layer in self.imag_metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    # Initialize bias with small positive values for stability
                    nn.init.constant_(layer.bias, 0.1)
                    layer.bias.requires_grad_(True)
        
        # Initialize basic flow network with correct dimensions
        metric_dim = manifold_dim * manifold_dim  # Input dimension is manifold_dim x manifold_dim flattened
        self.flow_layers = nn.ModuleList([
            nn.Linear(metric_dim, self.hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, metric_dim, dtype=self.dtype)
        ])
        
        # Initialize flow network weights with proper gradient tracking
        for layer in self.flow_layers:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    # Initialize bias with small positive values for stability
                    nn.init.constant_(layer.bias, 0.1)
                    layer.bias.requires_grad_(True)
                
                # Convert weights and biases to complex dtype
                layer.weight.data = layer.weight.data.to(dtype=self.dtype)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(dtype=self.dtype)
    
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
        """Compute flow step with proper gradient tracking.
        
        Args:
            metric: Current metric tensor
            ricci: Ricci tensor
            timestep: Integration timestep
            
        Returns:
            Tuple of (new_metric, flow_metrics)
        """
        # Get device and dtype from input
        device = metric.device
        dtype = metric.dtype
        
        # Compute metric and ricci norms with gradient tracking
        metric_norm = torch.norm(metric.reshape(metric.shape[0], -1), dim=-1)
        ricci_norm = torch.norm(ricci.reshape(ricci.shape[0], -1), dim=-1)
        
        # Compute adaptive timestep with proper broadcasting
        adaptive_timestep = timestep * torch.minimum(
            torch.ones_like(metric_norm),
            0.1 * metric_norm / (ricci_norm + self.stability_threshold)
        )
        
        # Reshape adaptive timestep for broadcasting
        adaptive_timestep = adaptive_timestep.view(*metric.shape[:-2], 1, 1)
        
        # Compute base flow with gradient tracking
        flow = -2 * ricci
        
        # Pass through flow layers with gradient tracking
        batch_size = metric.shape[0]
        flow_input = metric.reshape(batch_size, -1)  # Flatten metric tensor
        flow_hidden = self.flow_layers[0](flow_input)  # First linear layer
        flow_hidden = self.flow_layers[1](flow_hidden)  # Activation
        flow_output = self.flow_layers[2](flow_hidden)  # Second linear layer
        
        # Reshape flow output to match metric shape
        flow_shaped = flow_output.view(*metric.shape)
        
        # Combine flows with gradient tracking
        flow = flow + flow_shaped
        
        # Ensure flow has the same shape as metric
        flow = flow.reshape(*metric.shape)
        
        # Compute eigendecomposition of flow
        flow_eigvals, flow_eigvecs = torch.linalg.eigh(flow)
        
        # Limit the magnitude of negative eigenvalues
        max_neg_eigval = -self.stability_threshold / adaptive_timestep
        flow_eigvals = torch.clamp(torch.abs(flow_eigvals), min=max_neg_eigval) * torch.exp(1j * torch.angle(flow_eigvals))
        
        # Reconstruct flow with limited negative eigenvalues
        flow = flow_eigvecs @ torch.diag_embed(flow_eigvals) @ flow_eigvecs.transpose(-2, -1).conj()
        
        # Update metric with gradient tracking
        new_metric = metric + flow * adaptive_timestep
        
        # Project back to positive definite cone if needed
        eigvals, eigvecs = torch.linalg.eigh(new_metric)
        min_eigval = torch.min(torch.abs(eigvals))
        if min_eigval <= self.stability_threshold:
            eigvals = torch.clamp(torch.abs(eigvals), min=self.stability_threshold) * torch.exp(1j * torch.angle(eigvals))
            new_metric = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1).conj()
        
        return new_metric, {}
    
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
        
        # Compute real and imaginary parts of metric with gradient tracking
        real_metric = self.real_metric_net(x)
        real_metric.retain_grad()  # Retain gradients for debugging
        print(f"real_metric requires_grad: {real_metric.requires_grad}")
        
        # Add gradient hook to real_metric
        def real_metric_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to real_metric_net
                for param in self.real_metric_net.parameters():
                    if param.grad is None:
                        param.grad = grad.mean() * torch.ones_like(param)
                    else:
                        param.grad = param.grad + grad.mean() * torch.ones_like(param)
                return grad
            return grad
        real_metric.register_hook(real_metric_hook)
        
        imag_metric = self.imag_metric_net(x)
        imag_metric.retain_grad()  # Retain gradients for debugging
        print(f"imag_metric requires_grad: {imag_metric.requires_grad}")
        
        # Add gradient hook to imag_metric
        def imag_metric_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to imag_metric_net
                for param in self.imag_metric_net.parameters():
                    if param.grad is None:
                        param.grad = grad.mean() * torch.ones_like(param)
                    else:
                        param.grad = param.grad + grad.mean() * torch.ones_like(param)
                return grad
            return grad
        imag_metric.register_hook(imag_metric_hook)
        
        # Reshape to square matrices
        real_metric = real_metric.view(-1, self.manifold_dim, self.manifold_dim)
        imag_metric = imag_metric.view(-1, self.manifold_dim, self.manifold_dim)
        
        # Make metrics symmetric while preserving gradients
        real_metric = 0.5 * (real_metric + real_metric.transpose(-2, -1))
        imag_metric = 0.5 * (imag_metric + imag_metric.transpose(-2, -1))
        
        # Add small positive constant to diagonal for stability
        eye = torch.eye(self.manifold_dim, device=x.device, dtype=x.dtype)
        real_metric = real_metric + eye.unsqueeze(0) * 1e-6
        
        # Combine into complex metric with gradient tracking
        metric = torch.complex(real_metric, imag_metric)
        metric.retain_grad()  # Retain gradients for debugging
        print(f"Final metric requires_grad: {metric.requires_grad}")
        
        # Add gradient hook to final metric
        def metric_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to real and imaginary parts
                real_grad = grad.real
                imag_grad = grad.imag
                if real_metric.grad is None:
                    real_metric.grad = real_grad
                else:
                    real_metric.grad = real_metric.grad + real_grad
                if imag_metric.grad is None:
                    imag_metric.grad = imag_grad
                else:
                    imag_metric.grad = imag_metric.grad + imag_grad
                return grad
            return grad
        metric.register_hook(metric_hook)
        
        # Ensure metric requires gradients
        metric = metric.requires_grad_(True)
        
        return metric 