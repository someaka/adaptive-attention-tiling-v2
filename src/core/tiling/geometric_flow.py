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
        super().__init__(manifold_dim=manifold_dim)
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.motive_rank = motive_rank
        self.num_charts = num_charts
        self.integration_steps = integration_steps
        self.dt = dt
        self.stability_threshold = stability_threshold
        self.dtype = dtype
        
        # Initialize base metric as identity matrix with small perturbation
        base_metric = torch.eye(manifold_dim, dtype=self.dtype)
        base_metric = base_metric + torch.randn_like(base_metric) * 0.01  # Small random perturbation
        self.base_metric = nn.Parameter(base_metric, requires_grad=True)
        
        # Initialize arithmetic dynamics
        self.arithmetic = ArithmeticDynamics(
            hidden_dim=manifold_dim,
            motive_rank=motive_rank,
            manifold_dim=manifold_dim,
            dtype=self.dtype
        )
        
        # Add gradient hook for base metric
        def base_metric_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                if grad.is_complex():
                    grad_abs = grad.abs()
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    scale = torch.clamp(scale.real, min=1e-8, max=1e3)
                    grad = grad * scale
                else:
                    scale = 1.0 / (grad.norm() + 1e-8)
                    scale = torch.clamp(scale, min=1e-8, max=1e3)
                    grad = grad * scale
                return grad
            return grad
        self.base_metric.register_hook(base_metric_hook)
        
        # Initialize real metric network with proper gradient tracking
        self.real_metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        ).to(dtype=torch.float32)
        
        # Enable gradients for the entire network
        self.real_metric_net.requires_grad_(True)
        
        # Initialize each layer with proper dtype and gradient tracking
        for layer in self.real_metric_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.to(dtype=torch.float32)
                layer.weight.requires_grad_(True)
                layer.weight.retain_grad()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(dtype=torch.float32)
                    layer.bias.requires_grad_(True)
                    layer.bias.retain_grad()
                
                # Register gradient hooks for debugging
                def make_hook(name, param):
                    def hook(grad):
                        if grad is not None:
                            # Initialize gradient if None
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            # Scale gradient to prevent explosion
                            grad_abs = grad.abs()
                            scale = 1.0 / (grad_abs.norm() + 1e-8)
                            scale = torch.clamp(scale, min=1e-8, max=1e3)
                            grad = grad * scale
                            # Update gradients
                            param.grad = param.grad + grad
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                            return grad  # Return the modified gradient
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"real_metric_net.{layer}.weight", layer.weight))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"real_metric_net.{layer}.bias", layer.bias))
        
        # Initialize imaginary metric network with proper dtype
        self.imag_metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        ).to(dtype=torch.float32)
        
        # Enable gradients for the entire network
        self.imag_metric_net.requires_grad_(True)
        
        # Initialize weights with proper dtype and gradient tracking
        for layer in self.imag_metric_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.to(dtype=torch.float32)
                layer.weight.requires_grad_(True)
                layer.weight.retain_grad()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(dtype=torch.float32)
                    layer.bias.requires_grad_(True)
                    layer.bias.retain_grad()
                
                # Register gradient hooks for debugging
                def make_hook(name, param):
                    def hook(grad):
                        if grad is not None:
                            # Initialize gradient if None
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            # Scale gradient to prevent explosion
                            grad_abs = grad.abs()
                            scale = 1.0 / (grad_abs.norm() + 1e-8)
                            scale = torch.clamp(scale, min=1e-8, max=1e3)
                            grad = grad * scale
                            # Update gradients
                            param.grad = param.grad + grad
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                            return grad  # Return the modified gradient
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"imag_metric_net.{layer}.weight", layer.weight))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"imag_metric_net.{layer}.bias", layer.bias))
        
        # Hamiltonian structure with adaptive input projection
        self.hamiltonian = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim),  # Project within manifold_dim
            nn.Tanh(),
            nn.Linear(manifold_dim, 1)  # Output scalar energy
        )
        
        # Initialize hamiltonian layers with proper dtype
        for layer in self.hamiltonian:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.to(dtype=self.dtype)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(dtype=self.dtype)
    
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
    
    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from points."""
        # Debug input
        print(f"\nGeometric Flow Metric Computation Debug:")
        print(f"Input points shape: {points.shape}")
        print(f"Input points requires_grad: {points.requires_grad}")
        print(f"Input points grad_fn: {points.grad_fn}")
        print(f"Input points dtype: {points.dtype}")
        
        # Ensure points have correct dtype
        points = points.to(dtype=self.dtype)
        
        # Split into real and imaginary parts
        real_points = points.real if points.is_complex() else points
        imag_points = points.imag if points.is_complex() else torch.zeros_like(points, dtype=self.dtype)
        
        print(f"\nReal/Imag Points Debug:")
        print(f"Real points shape: {real_points.shape}")
        print(f"Real points requires_grad: {real_points.requires_grad}")
        print(f"Real points dtype: {real_points.dtype}")
        print(f"Imag points shape: {imag_points.shape}")
        print(f"Imag points requires_grad: {imag_points.requires_grad}")
        print(f"Imag points dtype: {imag_points.dtype}")
        
        # Compute metric components
        real_metric = self.real_metric_net(real_points)
        imag_metric = self.imag_metric_net(imag_points)
        
        print(f"\nMetric Components Debug:")
        print(f"Real metric shape: {real_metric.shape}")
        print(f"Real metric requires_grad: {real_metric.requires_grad}")
        print(f"Real metric grad_fn: {real_metric.grad_fn}")
        print(f"Real metric dtype: {real_metric.dtype}")
        print(f"Imag metric shape: {imag_metric.shape}")
        print(f"Imag metric requires_grad: {imag_metric.requires_grad}")
        print(f"Imag metric grad_fn: {imag_metric.grad_fn}")
        print(f"Imag metric dtype: {imag_metric.dtype}")
        
        # Reshape metric components
        batch_size = points.shape[0]
        real_metric = real_metric.view(batch_size, self.manifold_dim, self.manifold_dim)
        imag_metric = imag_metric.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        print(f"\nReshaped Components Debug:")
        print(f"Reshaped real metric shape: {real_metric.shape}")
        print(f"Reshaped real metric dtype: {real_metric.dtype}")
        print(f"Reshaped imag metric shape: {imag_metric.shape}")
        print(f"Reshaped imag metric dtype: {imag_metric.dtype}")
        
        # Combine into full metric tensor
        if points.is_complex():
            metric = torch.complex(real_metric, imag_metric)
        else:
            metric = real_metric
        
        print(f"\nFinal Metric Debug:")
        print(f"Final metric shape: {metric.shape}")
        print(f"Final metric requires_grad: {metric.requires_grad}")
        print(f"Final metric grad_fn: {metric.grad_fn}")
        print(f"Final metric dtype: {metric.dtype}")
        print(f"Final metric mean: {metric.abs().mean().item()}")
        print(f"Final metric max: {metric.abs().max().item()}")
        
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
        
        # Check input for NaN
        if torch.isnan(x).any():
            print("NaN detected in input")
            print(f"Input shape: {x.shape}")
            print(f"Input mean: {x.abs().mean().item()}")
        
        # Initialize metrics
        metrics: Dict[str, Any] = {}
        
        # Ensure input is properly shaped
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
            
        # Initialize metric tensor with gradient tracking
        metric = self.compute_metric(x)
        
        # Check metric for NaN
        if torch.isnan(metric).any():
            print("NaN detected in metric")
            print(f"Metric shape: {metric.shape}")
            print(f"Metric mean: {metric.abs().mean().item()}")
        
        metric = metric.requires_grad_(True)
        metric.retain_grad()
        
        # Add gradient hook for metric tensor with NaN checks
        def metric_hook(grad):
            if grad is not None:
                # Check for NaN in incoming gradient
                if torch.isnan(grad).any():
                    print("NaN detected in metric gradient")
                    print(f"Gradient shape: {grad.shape}")
                    print(f"Gradient mean before scaling: {grad.abs().mean().item()}")
                
                # Handle complex gradients
                if grad.is_complex():
                    grad_abs = grad.abs()
                    scale = 1.0 / (grad_abs.norm() + 1e-8)
                    scale = torch.clamp(scale.real, min=1e-8, max=1e3)
                    grad = grad * scale
                else:
                    scale = 1.0 / (grad.norm() + 1e-8)
                    scale = torch.clamp(scale, min=1e-8, max=1e3)
                    grad = grad * scale
                
                # Check for NaN after scaling
                if torch.isnan(grad).any():
                    print("NaN detected in metric gradient after scaling")
                    print(f"Scale value: {scale.item()}")
                
                # Ensure gradients flow back to all networks
                for net in [self.real_metric_net, self.imag_metric_net]:
                    for param in net.parameters():
                        # Ensure parameter requires gradients
                        param.requires_grad_(True)
                        
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        
                        # Calculate total elements in parameter
                        total_elements = param.numel()
                        
                        # Reshape gradient to match parameter dimensions
                        flattened_grad = grad.mean(0).view(-1)
                        
                        # Extract real part if gradient is complex
                        if torch.is_complex(flattened_grad):
                            flattened_grad = flattened_grad.real
                        
                        # Check for NaN in flattened gradient
                        if torch.isnan(flattened_grad).any():
                            print(f"NaN detected in flattened gradient for {param.shape}")
                        
                        if flattened_grad.numel() >= total_elements:
                            reshaped_grad = flattened_grad[:total_elements].reshape(param.shape)
                        else:
                            # If gradient has fewer elements, repeat it to match parameter size
                            repeated_grad = flattened_grad.repeat(total_elements // flattened_grad.numel() + 1)
                            reshaped_grad = repeated_grad[:total_elements].reshape(param.shape)
                        
                        # Check for NaN in reshaped gradient
                        if torch.isnan(reshaped_grad).any():
                            print(f"NaN detected in reshaped gradient for {param.shape}")
                            print(f"Original grad shape: {grad.shape}")
                            print(f"Flattened grad shape: {flattened_grad.shape}")
                            print(f"Reshaped grad shape: {reshaped_grad.shape}")
                        
                        # Add reshaped gradient to existing gradient
                        param.grad = param.grad + reshaped_grad
                        
                        # Check final gradient and print debug information
                        if param.grad is not None:
                            print(f"Parameter shape: {param.shape}")
                            print(f"Gradient shape: {param.grad.shape}")
                            print(f"Gradient norm: {param.grad.norm().item()}")
                            print(f"Gradient requires_grad: {param.grad.requires_grad}")
                            
                            if torch.isnan(param.grad).any():
                                print(f"NaN detected in final gradient for {param.shape}")
                        return grad
                    return grad
            return grad
        metric.register_hook(metric_hook)
        
        # Initialize path storage if needed
        if return_path:
            metrics['path'] = [metric]
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Check Ricci tensor for NaN
        if torch.isnan(ricci).any():
            print("NaN detected in Ricci tensor")
            print(f"Ricci shape: {ricci.shape}")
            print(f"Ricci mean: {ricci.abs().mean().item()}")
        
        # Perform flow step
        metric, step_metrics = self.flow_step(metric, ricci, self.dt)
        metrics.update(step_metrics)
        
        # Check metric after flow step
        if torch.isnan(metric).any():
            print("NaN detected in metric after flow step")
            print(f"Metric shape: {metric.shape}")
            print(f"Metric mean: {metric.abs().mean().item()}")
        
        if return_path:
            metrics['path'].append(metric)
            
        # Project back to input space
        output = x  # Use input as base for output to maintain dimensions
        
        # Apply metric transformation with proper broadcasting
        if len(metric.shape) > 3:
            metric = metric.mean(dim=-3)  # Average over extra dimensions
        
        # Convert output to complex if metric is complex
        if torch.is_complex(metric):
            output = output.to(dtype=torch.complex64)
        elif torch.is_complex(output):
            metric = metric.to(dtype=torch.complex64)
        
        # Apply metric transformation
        output = torch.einsum('...i,...ij->...j', output, metric)
        
        # Check output for NaN
        if torch.isnan(output).any():
            print("NaN detected in output")
            print(f"Output shape: {output.shape}")
            print(f"Output mean: {output.abs().mean().item()}")
        
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

