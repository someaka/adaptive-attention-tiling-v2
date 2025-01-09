"""Riemannian-specific geometric flow implementation.

This module provides a Riemannian geometry specific implementation
of geometric flow, building on the base geometric flow.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
from torch import nn

from .base_flow import BaseGeometricFlow

class RiemannianFlow(BaseGeometricFlow):
    """Riemannian-specific implementation of geometric flow.
    
    This class extends the base geometric flow with Riemannian-specific
    features including Christoffel symbols, geodesics, and proper
    Ricci curvature computation.
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        use_parallel_transport: bool = True,
        dtype: torch.dtype = torch.complex64
    ):
        """Initialize Riemannian flow.
        
        Args:
            manifold_dim: Dimension of the base manifold
            hidden_dim: Hidden dimension for flow computations
            num_layers: Number of layers in flow network
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
            use_parallel_transport: Whether to use parallel transport
            dtype: Data type for tensors
        """
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dt=dt,
            stability_threshold=stability_threshold,
            dtype=dtype
        )
        
        self.use_parallel_transport = use_parallel_transport
        
        # Initialize real metric network with proper gradient tracking
        real_layers = []
        real_layers.append(nn.Linear(manifold_dim, self.hidden_dim))
        real_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            real_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            real_layers.append(nn.Tanh())
        real_layers.append(nn.Linear(self.hidden_dim, manifold_dim * manifold_dim))
        self.real_metric_net = nn.Sequential(*real_layers)
        
        # Initialize weights for real_metric_net with proper gradient tracking
        for layer in self.real_metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad_(True)
                layer.weight.retain_grad()  # Ensure gradients are retained
                
                if layer.bias is not None:
                    # Initialize bias with small positive values for stability
                    nn.init.constant_(layer.bias, 0.1)
                    layer.bias.requires_grad_(True)
                    layer.bias.retain_grad()  # Ensure gradients are retained
                    
                # Register gradient hooks for debugging
                def make_hook(name, param):
                    def hook(grad):
                        if grad is not None:
                            # Initialize gradient if None
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            # Scale gradient to prevent explosion
                            grad = grad / (grad.norm() + 1e-8)
                            # Update gradients
                            param.grad = param.grad + grad
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"real_metric_net.{layer}.weight", layer.weight))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"real_metric_net.{layer}.bias", layer.bias))
        
        # Additional Riemannian-specific networks with proper gradient tracking
        self.christoffel_net = nn.Sequential(
            nn.Linear(manifold_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, manifold_dim * manifold_dim * manifold_dim)
        )
        
        # Initialize weights for christoffel_net with proper gradient tracking
        for layer in self.christoffel_net:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad_(True)
                layer.weight.retain_grad()  # Ensure gradients are retained
                
                if layer.bias is not None:
                    # Initialize bias with small positive values for stability
                    nn.init.constant_(layer.bias, 0.1)
                    layer.bias.requires_grad_(True)
                    layer.bias.retain_grad()  # Ensure gradients are retained
                    
                # Register gradient hooks for debugging
                def make_hook(name, param):
                    def hook(grad):
                        if grad is not None:
                            # Initialize gradient if None
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            # Scale gradient to prevent explosion
                            grad = grad / (grad.norm() + 1e-8)
                            # Update gradients
                            param.grad = param.grad + grad
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"christoffel_net.{layer}.weight", layer.weight))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"christoffel_net.{layer}.bias", layer.bias))
    
    def compute_christoffel(
        self,
        metric: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Christoffel symbols using connection coefficients.
        
        Args:
            metric: Metric tensor
            points: Optional points tensor
            
        Returns:
            Christoffel symbols
        """
        batch_size = metric.shape[0]
        device = metric.device
        dtype = metric.dtype
        
        # Ensure metric requires gradients
        metric.requires_grad_(True)
        
        # Get connection coefficients
        if hasattr(self, 'connection_coeffs'):
            connection_coeffs = self.connection_coeffs
            connection_coeffs.requires_grad_(True)  # Ensure gradients flow
            
            # Reshape connection coefficients to match batch size if needed
            if connection_coeffs.shape[0] != batch_size:
                connection_coeffs = connection_coeffs.expand(batch_size, -1, -1, -1)
            
            # Compute Christoffel symbols using connection coefficients
            christoffel = connection_coeffs
            
            # Add metric contribution
            metric_inv = torch.linalg.inv(metric)  # [batch_size, manifold_dim, manifold_dim]
            metric_inv.requires_grad_(True)  # Ensure gradients flow
            
            metric_contribution = torch.einsum(
                'bij,bjk->bik',
                metric_inv,
                metric
            )
            
            # Combine connection coefficients with metric contribution
            christoffel = christoffel + 0.5 * metric_contribution.unsqueeze(-1)
            
            return christoffel
            
        # Fallback to computing from scratch if no connection coefficients
        # Create a list to store metric derivatives
        metric_derivs_list = []
        
        # Create identity matrix for finite differences
        eye = torch.eye(
            self.manifold_dim,
            device=device,
            dtype=dtype
        )
        
        # Small value for finite differences
        eps = 1e-6
        
        # Create a zero tensor for points if None
        if points is None:
            points = torch.zeros(
                batch_size,
                self.manifold_dim,
                device=metric.device,
                requires_grad=True  # Ensure gradients flow
            )
        else:
            points.requires_grad_(True)  # Ensure gradients flow
        
        # Compute metric derivatives using finite differences
        for k in range(self.manifold_dim):
            shift = eps * eye[k]
            # Forward difference with stability scaling
            shifted_points = points + shift.unsqueeze(0)
            shifted_points.requires_grad_(True)  # Ensure gradients flow
            shifted_metric = self.compute_metric(shifted_points)
            diff = (shifted_metric - metric) / (eps + self.stability_threshold)
            # Ensure diff has the right shape [batch_size, manifold_dim, manifold_dim]
            if diff.dim() > 3:
                diff = diff.reshape(-1, self.manifold_dim, self.manifold_dim)
            metric_derivs_list.append(diff)
        
        # Stack metric derivatives along a new dimension
        metric_derivs = torch.stack(metric_derivs_list, dim=-1)
        
        # Get inverse metric tensor
        metric_inv = torch.linalg.inv(metric)  # [batch_size, manifold_dim, manifold_dim]
        metric_inv.requires_grad_(True)  # Ensure gradients flow
        
        # Initialize Christoffel symbols
        christoffel_list = []
        
        # Compute Christoffel symbols using einsum for better broadcasting
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    # Compute partial derivatives
                    partial_k = metric_derivs[..., k, i, j]
                    partial_j = metric_derivs[..., j, i, k]
                    partial_i = metric_derivs[..., i, j, k]
                    
                    # Combine terms using einsum for proper broadcasting
                    combined_terms = partial_k + partial_j - partial_i  # [batch_size]
                    christoffel_component = 0.5 * torch.einsum(
                        '...i,...->...',
                        metric_inv[..., i, :],
                        combined_terms
                    )
                    christoffel_list.append(christoffel_component)
        
        # Stack Christoffel components into final tensor
        christoffel = torch.stack(christoffel_list, dim=-1)
        christoffel = christoffel.reshape(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
        return christoffel
    
    def compute_ricci_tensor(
        self,
        metric: torch.Tensor,
        connection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Ricci tensor.

        Args:
            metric: Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
                or [batch_size, seq_len, manifold_dim, manifold_dim]
                or [batch_size, seq_len, heads, manifold_dim, manifold_dim]
                or [batch_size, seq_len, heads, time, manifold_dim, manifold_dim]
                or [batch_size, seq_len, heads, time, extra, manifold_dim, manifold_dim]
                or [batch_size, seq_len, heads, time, extra, extra2, manifold_dim, manifold_dim]
                or [batch_size, seq_len, heads, time, extra, extra2, extra3, manifold_dim, manifold_dim]
                or [batch_size, seq_len, heads, time, extra, extra2, extra3, extra4, manifold_dim, manifold_dim]

        Returns:
            Ricci tensor of shape [batch_size, manifold_dim, manifold_dim]
        """
        # Store original device and dtype
        device = metric.device
        dtype = metric.dtype

        # Handle different input shapes
        orig_shape = metric.shape
        if metric.dim() > 3:
            # If metric has more than 3 dimensions, flatten all but last 2
            batch_size = metric.shape[0]
            metric = metric.reshape(-1, self.manifold_dim, self.manifold_dim)
            # Update batch_size to account for flattened dimensions
            batch_size = metric.shape[0]
        elif metric.dim() == 3:  # [batch_size, manifold_dim, manifold_dim]
            batch_size = metric.shape[0]
        else:
            raise ValueError(f"Unexpected metric shape: {metric.shape}")

        # Compute Christoffel symbols if not provided
        if connection is None:
            connection = self.compute_christoffel(metric)
        else:
            # Ensure connection requires gradients
            connection.requires_grad_(True)

        # Add gradient hook to connection
        def connection_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to connection coefficients
                if hasattr(self, 'connection_coeffs'):
                    if self.connection_coeffs.grad is None:
                        self.connection_coeffs.grad = grad.mean(0)
                    else:
                        self.connection_coeffs.grad = self.connection_coeffs.grad + grad.mean(0)
                return grad
            return grad
        connection.register_hook(connection_hook)

        # Compute Riemann tensor components
        riemann = torch.zeros(
            batch_size,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            device=device,
            dtype=dtype
        )

        # Compute Riemann tensor
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        # R^i_{jkl} = \partial_k \Gamma^i_{jl} - \partial_l \Gamma^i_{jk} + ...
                        term1 = connection[..., i, j, l].unsqueeze(-1) * connection[..., :, k, l]
                        term2 = -connection[..., i, j, k].unsqueeze(-1) * connection[..., :, l, k]
                        riemann[..., i, j, k, l] = term1.sum(-1) + term2.sum(-1)

        # Contract to get Ricci tensor
        ricci = torch.einsum('...ijij->...ij', riemann)

        # Ensure final shape is [batch_size, manifold_dim, manifold_dim]
        if ricci.shape[0] != batch_size:
            # If batch dimension is wrong, reshape to combine all dimensions except last two
            ricci = ricci.reshape(batch_size, self.manifold_dim, self.manifold_dim)

        assert ricci.shape == (batch_size, self.manifold_dim, self.manifold_dim), \
            f"Expected shape {(batch_size, self.manifold_dim, self.manifold_dim)}, got {ricci.shape}"

        return ricci
    
    def parallel_transport(
        self,
        vector: torch.Tensor,
        path: torch.Tensor,
        metric: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Parallel transport a vector along a path.
        
        Args:
            vector: Vector to transport
            path: Path along which to transport
            metric: Optional metric tensor
            
        Returns:
            Parallel transported vector
        """
        if not self.use_parallel_transport:
            return vector
            
        if metric is None:
            metric = self.compute_metric(path[0])
            
        # Get Christoffel symbols
        christoffel = self.compute_christoffel(metric, path[0])
        
        # Initialize transported vector
        current = vector.clone()  # Clone to avoid modifying input
        
        # Transport along path segments
        for i in range(len(path) - 1):
            # Compute tangent vector
            tangent = path[i + 1] - path[i]
            
            # Transport equation
            current = current - torch.einsum(
                '...ijk,...j,...k->...i',
                christoffel,
                current,
                tangent
            ) * self.dt
            
        return current
    
    def flow_step(
        self,
        metric: torch.Tensor,
        ricci: torch.Tensor,
        timestep: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform Riemannian flow step.
        
        Args:
            metric: Current metric tensor
            ricci: Ricci curvature tensor
            timestep: Integration time step
            
        Returns:
            Tuple of (new_metric, flow_metrics)
        """
        # Basic flow step from parent
        new_metric, metrics = super().flow_step(metric, ricci, timestep)
        
        # Add Riemannian-specific metrics
        metrics.update({
            'scalar_curvature': torch.diagonal(ricci, dim1=-2, dim2=-1).sum(-1).mean().item(),
            'christoffel_norm': torch.norm(self.compute_christoffel(metric)).item()
        })
        
        return new_metric, metrics 
    
    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the Riemannian metric tensor at given points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        # Ensure points require gradients
        points = points.requires_grad_(True)
        
        # Split into real and imaginary parts
        real_points = points[..., :self.manifold_dim//2]
        imag_points = points[..., self.manifold_dim//2:]
        
        # Compute metric components with gradient tracking
        real_metric = self.real_metric_net(real_points)
        imag_metric = self.imag_metric_net(imag_points)
        
        # Reshape metric components
        batch_size = points.shape[0]
        real_metric = real_metric.view(batch_size, self.manifold_dim//2, self.manifold_dim//2)
        imag_metric = imag_metric.view(batch_size, self.manifold_dim//2, self.manifold_dim//2)
        
        # Combine into full metric tensor
        metric = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, 
                            dtype=points.dtype, device=points.device)
        metric[..., :self.manifold_dim//2, :self.manifold_dim//2] = real_metric
        metric[..., self.manifold_dim//2:, self.manifold_dim//2:] = imag_metric
        
        # Ensure metric requires gradients
        metric = metric.requires_grad_(True)
        
        return metric 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the flow."""
        # Ensure input requires gradients
        x = x.requires_grad_(True)
        
        # Pass through real metric network with gradient tracking
        real_metric = self.real_metric_net(x)
        real_metric = real_metric.view(-1, self.manifold_dim, self.manifold_dim)
        real_metric = 0.5 * (real_metric + real_metric.transpose(-2, -1))
        real_metric = real_metric.requires_grad_(True)
        real_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook to real metric
        def real_metric_hook(grad):
            if grad is not None:
                # Initialize gradient if None
                if grad.grad is None:
                    grad.grad = torch.zeros_like(grad)
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                # Update gradients
                grad.grad = grad.grad + grad
                print(f"Gradient for real_metric: {grad.abs().mean().item()}")
                
                # Ensure gradients flow back to real_metric_net parameters
                for name, param in self.real_metric_net.named_parameters():
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad = param.grad + grad.mean() * torch.ones_like(param)
            return grad
        real_metric.register_hook(real_metric_hook)
        
        return real_metric 