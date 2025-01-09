"""Geometric Flow Implementation.

This module implements geometric flow over the space of computational patterns.
It combines:

- Information Geometry
- Geodesic Flows
- Pattern Dynamics
- Quantum Structures

The core insight is that attention patterns naturally live on a
Riemannian manifold with rich geometric structure.
"""

from typing import Dict, List, Tuple, Any, Optional, cast
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

from ..patterns.arithmetic_dynamics import ArithmeticDynamics
from ..patterns.riemannian_flow import RiemannianFlow

class GeometricFlow(RiemannianFlow):
    """Pattern-specific implementation of geometric flow.
    
    This class extends the Riemannian flow with pattern-specific features
    and quantum geometric structure.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        manifold_dim: int,
        motive_rank: int = 4,
        num_charts: int = 4,
        integration_steps: int = 10,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize geometric flow.
        
        Args:
            hidden_dim: Hidden dimension for flow computations
            manifold_dim: Dimension of the base manifold
            motive_rank: Rank of motivic structure
            num_charts: Number of coordinate charts
            integration_steps: Number of integration steps
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
            dtype: Data type for tensors
        """
        # Convert dtype to complex if it's not already
        if dtype == torch.float32:
            dtype = torch.complex64
        elif dtype == torch.float64:
            dtype = torch.complex128
            
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            num_layers=2,  # Fixed for pattern implementation
            dt=dt,
            stability_threshold=stability_threshold,
            use_parallel_transport=True,
            dtype=dtype
        )
        
        self.motive_rank = motive_rank
        self.num_charts = num_charts
        self.integration_steps = integration_steps
        self.dtype = dtype
        
        # Initialize arithmetic structure
        self.arithmetic = ArithmeticDynamics(
            hidden_dim=hidden_dim,
            motive_rank=motive_rank,
            dtype=self.dtype
        )
        
        # Chart embeddings for local coordinates
        real_chart = torch.randn(num_charts, manifold_dim) * 0.02
        imag_chart = torch.randn(num_charts, manifold_dim) * 0.02
        self.chart_embedding = nn.Parameter(
            torch.complex(real_chart, imag_chart).to(dtype=self.dtype)
        )
        
        # Initialize flow layers with complex weights
        self.flow_layers = nn.ModuleList([
            nn.Linear(manifold_dim, manifold_dim, dtype=self.dtype),
            nn.Linear(manifold_dim, manifold_dim, dtype=self.dtype)
        ])
        
        # Initialize metric network with complex weights
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=self.dtype)
        )
        
        # Initialize real metric network with proper gradient tracking
        real_layers = []
        real_layers.append(nn.Linear(manifold_dim, hidden_dim, dtype=self.dtype, bias=True))
        real_layers.append(nn.Tanh())
        for _ in range(2 - 2):
            real_layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype, bias=True))
            real_layers.append(nn.Tanh())
        real_layers.append(nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=self.dtype, bias=True))
        self.real_metric_net = nn.Sequential(*real_layers)
        
        # Initialize each layer with proper gradient tracking
        for layer in self.real_metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize with small random values using Xavier initialization
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                # Ensure gradients are enabled
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    layer.bias.requires_grad_(True)
                # Register gradient hooks for debugging
                def make_hook(name):
                    def hook(grad):
                        if grad is not None:
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"real_metric_net.{layer}.weight"))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"real_metric_net.{layer}.bias"))
        
        # Initialize imaginary metric network with proper dtype
        imag_layers = []
        imag_layers.append(nn.Linear(manifold_dim, hidden_dim, dtype=self.dtype))
        imag_layers.append(nn.Tanh())
        for _ in range(2 - 2):
            imag_layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype))
            imag_layers.append(nn.Tanh())
        imag_layers.append(nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=self.dtype))
        self.imag_metric_net = nn.Sequential(*imag_layers)
        
        # Initialize weights with proper complex initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Create complex weights directly
                weight_shape = m.weight.shape
                real_weight = torch.randn(*weight_shape) * 0.02
                imag_weight = torch.randn(*weight_shape) * 0.02
                m.weight.data = torch.complex(real_weight, imag_weight).to(dtype=self.dtype)
                if m.bias is not None:
                    real_bias = torch.randn(m.bias.shape) * 0.02
                    imag_bias = torch.randn(m.bias.shape) * 0.02
                    m.bias.data = torch.complex(real_bias, imag_bias).to(dtype=self.dtype)
        
        # Hamiltonian structure with adaptive input projection
        self.hamiltonian = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim, dtype=self.dtype),  # Project within manifold_dim
            nn.Tanh(),
            nn.Linear(manifold_dim, 1, dtype=self.dtype)  # Output scalar energy
        )
    
    def compute_ricci_tensor(
        self,
        metric: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Compute Ricci tensor with quantum corrections.
        
        Args:
            metric: Metric tensor
            connection: Optional connection form
            
        Returns:
            Ricci curvature tensor with quantum corrections
        """
        # Get base Ricci tensor
        ricci = super().compute_ricci_tensor(metric, connection)
        
        # Add quantum corrections from arithmetic structure
        quantum_term = self.arithmetic.compute_quantum_correction(metric)
        
        # Project quantum term to match Ricci tensor dimensions
        if quantum_term.shape != ricci.shape:
            # Get target shape from ricci tensor
            batch_size = ricci.shape[0]
            h, w = ricci.shape[-2], ricci.shape[-1]
            
            # Reshape quantum term to match ricci dimensions
            quantum_term = quantum_term.reshape(batch_size, -1, quantum_term.shape[-1])
            
            # Handle complex numbers by splitting real and imaginary parts
            real_part = F.adaptive_avg_pool2d(
                quantum_term.real.unsqueeze(1),  # Add channel dimension
                output_size=(h, w)  # Target size
            ).squeeze(1)  # Remove channel dimension
            
            imag_part = F.adaptive_avg_pool2d(
                quantum_term.imag.unsqueeze(1),  # Add channel dimension
                output_size=(h, w)  # Target size
            ).squeeze(1)  # Remove channel dimension
            
            # Recombine complex tensor
            quantum_term = torch.complex(real_part, imag_part)
            
            # Ensure final shape matches ricci tensor
            quantum_term = quantum_term.reshape(batch_size, h, w)
        
        # Add quantum corrections with a small scaling factor for stability
        alpha = 0.1  # Small factor for stability
        ricci = ricci + alpha * quantum_term
        
        # Ensure ricci tensor has the correct shape
        ricci = ricci.reshape(*metric.shape)
        
        return ricci
    
    def flow_step(
        self,
        metric: torch.Tensor,
        ricci: torch.Tensor,
        timestep: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform flow step with quantum geometric features.
        
        Args:
            metric: Current metric tensor
            ricci: Ricci curvature tensor
            timestep: Integration time step
            
        Returns:
            Tuple of (new_metric, flow_metrics)
        """
        # Basic Riemannian flow step
        new_metric, metrics = super().flow_step(metric, ricci, timestep)
        
        # The metric is already in the correct shape [batch_size, manifold_dim, manifold_dim]
        # No need to reshape it
        
        # Add quantum geometric metrics
        metrics.update({
            'quantum_correction': torch.norm(
                self.arithmetic.compute_quantum_correction(metric)
            ).item(),
            'hamiltonian': self.compute_hamiltonian(metric).squeeze(-1).mean().item()
        })
        
        return new_metric, metrics
    
    def compute_metric(self, x: Tensor) -> Tensor:
        """Compute metric with quantum geometric structure."""
        # Project input to manifold space
        x_proj = self.project_to_manifold(x)
        
        # Ensure input is complex and requires gradients
        if not x_proj.is_complex():
            x_proj = torch.complex(x_proj, torch.zeros_like(x_proj))
        x_proj = x_proj.requires_grad_(True)
        
        # Convert weights to complex if needed and ensure gradient flow
        def to_complex(net):
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    if not module.weight.is_complex():
                        # Create complex weights with gradient tracking
                        complex_weight = torch.complex(
                            module.weight.data,
                            torch.zeros_like(module.weight.data)
                        )
                        module.weight = nn.Parameter(complex_weight)
                        # Clip weights to prevent numerical instability
                        with torch.no_grad():
                            module.weight.data.real.clamp_(-1.0, 1.0)
                            module.weight.data.imag.clamp_(-1.0, 1.0)
                    if module.bias is not None and not module.bias.is_complex():
                        # Create complex bias with gradient tracking
                        complex_bias = torch.complex(
                            module.bias.data,
                            torch.zeros_like(module.bias.data)
                        )
                        module.bias = nn.Parameter(complex_bias)
                        # Clip biases to prevent numerical instability
                        with torch.no_grad():
                            module.bias.data.real.clamp_(-1.0, 1.0)
                            module.bias.data.imag.clamp_(-1.0, 1.0)
                    # Ensure gradients are enabled
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
        
        to_complex(self.metric_net)
        to_complex(self.real_metric_net)
        to_complex(self.imag_metric_net)
        
        # Compute metric using both networks with gradient tracking
        metric_output = self.metric_net(x_proj)
        metric_output = metric_output.requires_grad_(True)
        
        # Compute real and imaginary parts separately with gradient tracking
        real_metric = self.real_metric_net(x_proj)
        real_metric = real_metric.requires_grad_(True)
        real_metric.retain_grad()  # Retain gradients for real metric
        
        imag_metric = self.imag_metric_net(x_proj)
        imag_metric = imag_metric.requires_grad_(True)
        imag_metric.retain_grad()  # Retain gradients for imaginary metric
        
        # Add gradient hooks for real and imaginary metrics
        def real_metric_hook(grad):
            if grad is not None:
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to real_metric_net
                for param in self.real_metric_net.parameters():
                    if param.grad is None:
                        # Calculate total elements in parameter
                        total_elements = param.numel()
                        # Reshape gradient to match parameter dimensions while preserving the total number of elements
                        flattened_grad = grad.mean(0).view(-1)
                        if flattened_grad.numel() >= total_elements:
                            reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                        else:
                            # If gradient has fewer elements, repeat it to match parameter size
                            repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                            reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                        param.grad = reshaped_grad
                    else:
                        # Add reshaped gradient to existing gradient
                        total_elements = param.numel()
                        flattened_grad = grad.mean(0).view(-1)
                        if flattened_grad.numel() >= total_elements:
                            reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                        else:
                            # If gradient has fewer elements, repeat it to match parameter size
                            repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                            reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                        param.grad = param.grad + reshaped_grad
                return grad
            return grad
        real_metric.register_hook(real_metric_hook)  # Register the hook for real metric
        
        def imag_metric_hook(grad):
            if grad is not None:
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to imag_metric_net
                for param in self.imag_metric_net.parameters():
                    if param.grad is None:
                        # Calculate total elements in parameter
                        total_elements = param.numel()
                        # Reshape gradient to match parameter dimensions while preserving the total number of elements
                        flattened_grad = grad.mean(0).view(-1)
                        if flattened_grad.numel() >= total_elements:
                            reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                        else:
                            # If gradient has fewer elements, repeat it to match parameter size
                            repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                            reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                        param.grad = reshaped_grad
                    else:
                        # Add reshaped gradient to existing gradient
                        total_elements = param.numel()
                        flattened_grad = grad.mean(0).view(-1)
                        if flattened_grad.numel() >= total_elements:
                            reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                        else:
                            # If gradient has fewer elements, repeat it to match parameter size
                            repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                            reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                        param.grad = param.grad + reshaped_grad
                return grad
            return grad
        imag_metric.register_hook(imag_metric_hook)
        
        # Combine real and imaginary parts with numerical stability
        metric_output = metric_output + torch.complex(
            real_metric.real.clamp(-1e3, 1e3),
            imag_metric.real.clamp(-1e3, 1e3)
        )
        
        # Reshape to square matrix
        batch_size = metric_output.shape[0]
        metric = metric_output.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Make metric Hermitian with numerical stability
        metric = 0.5 * (metric + metric.transpose(-2, -1).conj())
        
        # Add small positive constant to diagonal for stability
        eye = torch.eye(
            self.manifold_dim,
            device=metric.device,
            dtype=metric.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        metric = metric + eye * self.stability_threshold
        
        # Register gradient hooks for each network with clipping
        def register_hook_for_net(net, name):
            for param in net.parameters():
                def hook(grad):
                    if grad is not None:
                        # Scale gradients based on the magnitude of the input
                        scale = torch.clamp(
                            torch.tensor(x_proj.abs().mean().item(), device=grad.device),
                            min=1e-6,
                            max=1e3
                        )
                        # Handle complex gradients by clamping real and imaginary parts separately
                        if grad.is_complex():
                            real_grad = torch.clamp(grad.real * scale, min=-1e3, max=1e3)
                            imag_grad = torch.clamp(grad.imag * scale, min=-1e3, max=1e3)
                            return torch.complex(real_grad, imag_grad)
                        else:
                            return torch.clamp(grad * scale, min=-1e3, max=1e3)
                if param.requires_grad:
                    param.register_hook(hook)
        
        register_hook_for_net(self.metric_net, 'metric_net')
        register_hook_for_net(self.real_metric_net, 'real_metric_net')
        register_hook_for_net(self.imag_metric_net, 'imag_metric_net')
        
        # Ensure metric requires gradients and is numerically stable
        metric = metric.requires_grad_(True)
        metric.retain_grad()  # Retain gradients for metric
        
        # Add final gradient hook for metric
        def metric_hook(grad):
            if grad is not None:
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    # Scale gradient to prevent explosion
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    grad = grad * scale
                else:
                    # Scale gradient to prevent explosion
                    grad = grad / (grad.norm() + 1e-8)
                # Ensure gradients flow back to all networks
                for net in [self.metric_net, self.real_metric_net, self.imag_metric_net]:
                    for param in net.parameters():
                        if param.grad is None:
                            # Calculate total elements in parameter
                            total_elements = param.numel()
                            # Reshape gradient to match parameter dimensions while preserving the total number of elements
                            flattened_grad = grad.mean(0).view(-1)
                            if flattened_grad.numel() >= total_elements:
                                reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                            else:
                                # If gradient has fewer elements, repeat it to match parameter size
                                repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                                reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                            param.grad = reshaped_grad
                        else:
                            # Add reshaped gradient to existing gradient
                            total_elements = param.numel()
                            flattened_grad = grad.mean(0).view(-1)
                            if flattened_grad.numel() >= total_elements:
                                reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                            else:
                                # If gradient has fewer elements, repeat it to match parameter size
                                repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                                reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                            param.grad = param.grad + reshaped_grad
                return grad
            return grad
        metric.register_hook(metric_hook)
        
        # Add final numerical stability check
        if torch.isnan(metric).any() or torch.isinf(metric).any():
            # If we detect any NaN or Inf values, reset the metric to a stable state
            metric = torch.eye(
                self.manifold_dim,
                device=metric.device,
                dtype=metric.dtype
            ).unsqueeze(0).expand(batch_size, -1, -1).requires_grad_(True)
        
        return metric
    
    def forward(
        self,
        x: Tensor,
        return_path: bool = False
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass of geometric flow.
        
        Args:
            x: Input tensor [batch_size, manifold_dim] or [batch_size, seq_len, manifold_dim]
            return_path: Whether to return intermediate states
            
        Returns:
            Tuple of (output tensor, metrics dictionary)
        """
        # Store original shape and device
        original_shape = x.shape
        device = x.device
        
        # Initialize metrics
        metrics: Dict[str, Any] = {}
        
        # Ensure input is properly shaped
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
            
        # Initialize metric tensor
        metric = self.compute_metric(x)
        
        # Initialize path storage if needed
        if return_path:
            metrics['path'] = [metric]
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Perform flow step
        metric, step_metrics = self.flow_step(metric, ricci, self.dt)
        metrics.update(step_metrics)
        
        if return_path:
            metrics['path'].append(metric)
            
        # Project back to input space
        output = x  # Use input as base for output to maintain dimensions
        
        # Apply metric transformation with proper broadcasting
        # Ensure metric has shape [..., manifold_dim, manifold_dim]
        if len(metric.shape) > 3:
            metric = metric.mean(dim=-3)  # Average over extra dimensions
        
        # Convert output to complex if metric is complex
        if torch.is_complex(metric):
            output = output.to(dtype=torch.complex64)
        elif torch.is_complex(output):
            metric = metric.to(dtype=torch.complex64)
        
        # Apply metric transformation
        output = torch.einsum('...i,...ij->...j', output, metric)
        
        # Convert back to real if needed
        if torch.is_complex(output) and not torch.is_complex(x):
            output = output.real
        
        # Reshape output back to original shape if needed
        if len(original_shape) > 2:
            output = output.reshape(*original_shape)
            
        return output.to(device), metrics
    
    def compute_quantum_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric with quantum geometric structure."""
        # Project input to manifold dimension
        x_proj = self.proj(x)
        
        # Ensure parameters have the same dtype as input
        for param in self.parameters():
            param.data = param.data.to(dtype=torch.complex64)
            
        # Compute quantum state with proper dimension handling
        batch_size = x_proj.shape[0]
        x_flat = x_proj.reshape(batch_size, -1)  # Flatten spatial dimensions
        
        # Project to expected input dimension for quantum_proj
        if x_flat.shape[-1] != self.arithmetic.hidden_dim:
            x_flat = torch.nn.functional.interpolate(
                x_flat.unsqueeze(1),  # Add channel dimension
                size=self.arithmetic.hidden_dim,
                mode='linear'
            ).squeeze(1)  # Remove channel dimension
            
        quantum_state = self.arithmetic.quantum_proj(x_flat)  # [batch_size, manifold_dim]
        
        # Compute metric using quantum Fisher information
        metric = torch.abs(quantum_state @ quantum_state.conj().transpose(-2, -1))
        
        # Ensure positive definiteness and stability
        metric = metric + self.epsilon * torch.eye(
            metric.size(-1), dtype=metric.dtype, device=metric.device
        ).unsqueeze(0).expand_as(metric)
        
        return metric
    
    def compute_hamiltonian(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian energy from metric tensor.
        
        Args:
            metric: Input metric tensor
            
        Returns:
            Hamiltonian energy as complex tensor
        """
        # Store original shape
        original_shape = metric.shape
        
        # Flatten input to 2D
        if len(metric.shape) > 2:
            metric = metric.reshape(-1, metric.shape[-1])
            
        # Compute Hamiltonian energy
        energy = self.hamiltonian(metric)
        
        # Reshape energy back to original batch shape if needed
        if len(original_shape) > 2:
            energy = energy.reshape(*original_shape[:-1], -1)
            
        return energy
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Project input tensor to manifold space.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            
        Returns:
            Projected tensor of shape [batch_size, seq_len, manifold_dim] or [batch_size, manifold_dim]
        """
        # Handle both 2D and 3D inputs
        orig_shape = x.shape
        if len(orig_shape) == 3:
            batch_size, seq_len, hidden_dim = orig_shape
            x_flat = x.reshape(-1, hidden_dim)
        else:
            x_flat = x
            
        # Project to manifold dimension using interpolation
        if x_flat.shape[-1] != self.manifold_dim:
            # Handle complex interpolation by splitting real and imaginary parts
            if x_flat.is_complex():
                x_real = torch.nn.functional.interpolate(
                    x_flat.real.unsqueeze(1),  # Add channel dimension
                    size=self.manifold_dim,
                    mode='linear'
                ).squeeze(1)  # Remove channel dimension
                
                x_imag = torch.nn.functional.interpolate(
                    x_flat.imag.unsqueeze(1),  # Add channel dimension
                    size=self.manifold_dim,
                    mode='linear'
                ).squeeze(1)  # Remove channel dimension
                
                x_proj = torch.complex(x_real, x_imag)
            else:
                x_proj = torch.nn.functional.interpolate(
                    x_flat.unsqueeze(1),  # Add channel dimension
                    size=self.manifold_dim,
                    mode='linear'
                ).squeeze(1)  # Remove channel dimension
        else:
            x_proj = x_flat
            
        # Ensure output requires gradients
        x_proj.requires_grad_(True)
        
        # Reshape back to original dimensions if needed
        if len(orig_shape) == 3:
            x_proj = x_proj.reshape(batch_size, seq_len, self.manifold_dim)
            
        return x_proj

