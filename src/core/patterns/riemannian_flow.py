"""Riemannian-specific geometric flow implementation.

This module provides a Riemannian geometry specific implementation
of geometric flow, building on the base geometric flow.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
from torch import nn
import logging
import numpy as np
import torch.nn.functional as F
import copy

from .base_flow import BaseGeometricFlow

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_tensor(tensor: torch.Tensor, name: str) -> None:
    """Helper function to log tensor statistics."""
    if tensor is None:
        logger.debug(f"{name} is None")
        return
    
    # Helper function to safely check grad
    def check_grad(t):
        if not t.requires_grad:
            return False
        if t.is_leaf:
            return t.grad is not None
        if hasattr(t, '_grad_cache'):
            return t._grad_cache is not None
        return False
    
    # Helper function to safely get mean/std/min/max for complex tensors
    def get_stats(t):
        if t.is_complex():
            return {
                'mean': (t.real.abs().mean().item(), t.imag.abs().mean().item()),
                'std': (t.real.abs().std().item(), t.imag.abs().std().item()),
                'min': (t.real.abs().min().item(), t.imag.abs().min().item()),
                'max': (t.real.abs().max().item(), t.imag.abs().max().item()),
                'has_nan': (torch.isnan(t.real).any().item(), torch.isnan(t.imag).any().item()),
                'has_inf': (torch.isinf(t.real).any().item(), torch.isinf(t.imag).any().item())
            }
        else:
            return {
                'mean': t.abs().mean().item(),
                'std': t.abs().std().item(),
                'min': t.abs().min().item(),
                'max': t.abs().max().item(),
                'has_nan': torch.isnan(t).any().item(),
                'has_inf': torch.isinf(t).any().item()
            }
    
    stats = {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'requires_grad': tensor.requires_grad,
        'is_complex': tensor.is_complex(),
        'is_leaf': tensor.is_leaf,
        'has_grad': check_grad(tensor),
        **get_stats(tensor)
    }
    
    logger.debug(f"\nTensor {name} stats:\n" + "\n".join(f"{k}: {v}" for k, v in stats.items()))

def register_gradient_hook(tensor: torch.Tensor, name: str) -> None:
    """Helper function to register gradient hooks with logging."""
    if tensor is not None and tensor.requires_grad:
        def hook(grad):
            if grad is not None:
                debug_tensor(grad, f"{name}_grad")
            return grad
        tensor.register_hook(hook)

class RiemannianFlow(BaseGeometricFlow):
    """Riemannian flow for geometric computations."""
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
    ):
        """Initialize Riemannian flow."""
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dt=dt,
            stability_threshold=stability_threshold,
            dtype=dtype
        )
        
        # Store configuration
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize metric networks with correct dimensions
        metric_hidden_dim = manifold_dim * 2  # Increase hidden dimension for better expressivity
        metric_output_dim = manifold_dim * manifold_dim  # Output dimension for metric tensor
        
        # Real part network with proper dtype
        real_layers = []
        real_layers.append(nn.Linear(manifold_dim, metric_hidden_dim))
        real_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            real_layers.append(nn.Linear(metric_hidden_dim, metric_hidden_dim))
            real_layers.append(nn.Tanh())
        real_layers.append(nn.Linear(metric_hidden_dim, metric_output_dim))
        self.real_metric_net = nn.Sequential(*real_layers)
        
        # Imaginary part network with proper dtype
        imag_layers = []
        imag_layers.append(nn.Linear(manifold_dim, metric_hidden_dim))
        imag_layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            imag_layers.append(nn.Linear(metric_hidden_dim, metric_hidden_dim))
            imag_layers.append(nn.Tanh())
        imag_layers.append(nn.Linear(metric_hidden_dim, metric_output_dim))
        self.imag_metric_net = nn.Sequential(*imag_layers)
        
        # Initialize parameters with proper gradients and scaling
        for param in self.parameters():
            param.requires_grad_(True)
            if len(param.shape) > 1:  # Weight matrices
                # Custom orthogonal initialization for complex tensors
                rows = param.size(0)
                cols = param.numel() // rows
                flattened = param.new_empty((rows, cols)).normal_(0, 1)
                if rows < cols:
                    flattened.t_()
                q, r = torch.linalg.qr(flattened)
                # Use sgn instead of sign for complex support
                d = torch.diag(r, 0)
                ph = torch.sgn(d)
                q *= ph
                if rows < cols:
                    q.t_()
                param.data = q.reshape(param.shape)
            else:  # Bias vectors
                nn.init.zeros_(param)
        
        # Move to device and set dtype
        self.to(device=self.device)
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad_(True)
    
    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from input points."""
        # Debug input
        print("\nInput points shape:", x.shape)
        print("Input requires grad:", x.requires_grad)
        print("Input grad fn:", x.grad_fn)

        # Split into real and imaginary parts while preserving gradients
        points_real = x.real
        points_imag = x.imag

        # Debug shapes
        print("\nReal points shape:", points_real.shape)
        print("Imag points shape:", points_imag.shape)

        # Ensure gradients are enabled
        with torch.set_grad_enabled(True):
            # Compute real and imaginary metrics
            real_metric = self.real_metric_net(points_real)
            imag_metric = self.imag_metric_net(points_imag)

            print("\nReal metric requires grad:", real_metric.requires_grad)
            print("Real metric grad fn:", real_metric.grad_fn)
            print("Imag metric requires grad:", imag_metric.requires_grad)
            print("Imag metric grad fn:", imag_metric.grad_fn)

            # Reshape to square matrices while preserving gradients
            real_metric = real_metric.view(-1, self.manifold_dim, self.manifold_dim)
            imag_metric = imag_metric.view(-1, self.manifold_dim, self.manifold_dim)

            # Combine into complex metric
            metric = torch.complex(real_metric, imag_metric)

            # Make metric Hermitian while preserving gradients
            metric = 0.5 * (metric + metric.transpose(-2, -1).conj())

            # Project eigenvalues to be positive
            eigenvals, eigenvecs = torch.linalg.eigh(metric)
            eigenvals = torch.clamp(eigenvals.real, min=1e-6)  # Use only real part and ensure positive eigenvalues
            
            # Convert eigenvalues to complex
            eigenvals = torch.complex(eigenvals, torch.zeros_like(eigenvals))
            
            # Create diagonal matrix with complex eigenvalues
            diag_eigenvals = torch.diag_embed(eigenvals)

            # Reconstruct metric with positive eigenvalues
            metric = torch.matmul(torch.matmul(eigenvecs, diag_eigenvals), eigenvecs.transpose(-2, -1).conj())

            # Apply flow layers to metric
            batch_size = metric.shape[0]
            metric = metric.reshape(batch_size, -1)  # Flatten the metric tensor
            for layer in self.flow_layers:
                metric = layer(metric)
                metric = metric.requires_grad_(True)  # Ensure gradients are preserved
            metric = metric.reshape(batch_size, self.manifold_dim, self.manifold_dim)  # Reshape back to square matrix

            # Make metric Hermitian again after flow layers
            metric = 0.5 * (metric + metric.transpose(-2, -1).conj())

            # Project eigenvalues to be positive again
            eigenvals, eigenvecs = torch.linalg.eigh(metric)
            eigenvals = torch.clamp(eigenvals.real, min=1e-6)  # Use only real part and ensure positive eigenvalues
            
            # Convert eigenvalues to complex
            eigenvals = torch.complex(eigenvals, torch.zeros_like(eigenvals))
            
            # Create diagonal matrix with complex eigenvalues
            diag_eigenvals = torch.diag_embed(eigenvals)

            # Reconstruct metric with positive eigenvalues
            metric = torch.matmul(torch.matmul(eigenvecs, diag_eigenvals), eigenvecs.transpose(-2, -1).conj())

            print("\nOutput metric shape:", metric.shape)
            print("Output requires grad:", metric.requires_grad)
            print("Output grad fn:", metric.grad_fn)

            return metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass ensuring gradient flow."""
        # Ensure input requires gradients
        x = x.requires_grad_(True)
        
        # Compute metric with gradient tracking
        metric = self.compute_metric(x)
        
        # Add gradient hook for debugging
        def metric_hook(grad):
            if grad is not None:
                print("\nMetric gradient in forward:")
                print(f"Shape: {grad.shape}")
                print(f"Norm: {grad.norm().item()}")
                print(f"Mean: {grad.abs().mean().item()}")
            return grad
        metric.register_hook(metric_hook)
        
        return metric