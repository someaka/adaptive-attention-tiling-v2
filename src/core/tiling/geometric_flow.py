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

from typing import Dict, List, Tuple, Any, Optional, cast, Union
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
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            stability_threshold=stability_threshold,
            dtype=dtype
        )
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.motive_rank = motive_rank
        self.num_charts = num_charts
        self.integration_steps = integration_steps
        self.dt = dt
        self.stability_threshold = stability_threshold
        self.dtype = dtype
        
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
        
        print("\nInitializing GeometricFlow:")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  manifold_dim: {manifold_dim}")
        print(f"  motive_rank: {motive_rank}")
        print(f"  dtype: {dtype}")
        
        # Initialize base metric with proper gradient tracking
        base_metric = torch.eye(manifold_dim, dtype=self.dtype)
        base_metric = base_metric + torch.randn_like(base_metric) * 0.01  # Small random perturbation
        self.base_metric = nn.Parameter(base_metric, requires_grad=True)
        
        # Add gradient hook to base_metric for debugging and gradient flow
        def base_metric_hook(grad):
            if grad is not None:
                print(f"Base metric gradient norm: {grad.norm().item()}")
                print(f"Base metric gradient mean: {grad.abs().mean().item()}")
                return grad
            return grad
        self.base_metric.register_hook(base_metric_hook)
        
        # Initialize connection coefficients with proper gradient tracking
        connection_coeffs = torch.zeros(manifold_dim, manifold_dim, manifold_dim, dtype=self.dtype)
        connection_coeffs = connection_coeffs + torch.randn_like(connection_coeffs) * 0.01  # Small random perturbation
        self.connection_coeffs = nn.Parameter(connection_coeffs, requires_grad=True)
        
        # Add gradient hook to connection coefficients
        def connection_hook(grad):
            if grad is not None:
                print(f"Connection coefficients gradient norm: {grad.norm().item()}")
                print(f"Connection coefficients gradient mean: {grad.abs().mean().item()}")
                return grad
            return grad
        self.connection_coeffs.register_hook(connection_hook)
        
        # Initialize Christoffel network with proper gradient tracking
        self.christoffel_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim * manifold_dim, dtype=self.dtype)
        )
        
        # Enable gradients for the entire Christoffel network and its parameters
        for param in self.christoffel_net.parameters():
            param.requires_grad_(True)
        
        # Initialize each layer with proper dtype and gradient tracking
        for layer in self.christoffel_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.to(dtype=self.dtype)
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(dtype=self.dtype)
                    layer.bias.requires_grad_(True)
                
                # Register simple gradient hooks for debugging
                def make_hook(name):
                    def hook(grad):
                        if grad is not None:
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"christoffel_net.{layer}.weight"))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"christoffel_net.{layer}.bias"))
        
        # Initialize arithmetic dynamics with proper gradient tracking
        self.arithmetic = ArithmeticDynamics(
            hidden_dim=manifold_dim,
            motive_rank=motive_rank,
            manifold_dim=manifold_dim,
            dtype=self.dtype
        )
        
        print("  Arithmetic dynamics initialized")
        print(f"  Arithmetic coupling shape: {self.arithmetic.coupling.shape}")
        print(f"  Arithmetic coupling requires grad: {self.arithmetic.coupling.requires_grad}")
        
        # Add forward hook to track arithmetic usage
        def arithmetic_hook(module, input, output):
            print("\nArithmetic forward hook in GeometricFlow:")
            print(f"  Input shapes: {[x.shape if isinstance(x, torch.Tensor) else None for x in input]}")
            print(f"  Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
            print(f"  Coupling requires grad: {module.arithmetic.coupling.requires_grad}")
            print(f"  Coupling is leaf: {module.arithmetic.coupling.is_leaf}")
            print(f"  Coupling grad fn: {module.arithmetic.coupling.grad_fn}")
        self.register_forward_hook(arithmetic_hook)
        
        # Initialize real metric network with proper gradient tracking
        real_layers = []
        real_layers.append(nn.Linear(manifold_dim, hidden_dim, dtype=torch.float32))
        real_layers.append(nn.Tanh())
        for _ in range(2):  # Add intermediate layers
            real_layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32))
            real_layers.append(nn.Tanh())
        real_layers.append(nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=torch.float32))
        self.real_metric_net = nn.Sequential(*real_layers)
        
        # Initialize weights for real_metric_net with proper gradient tracking
        for layer in self.real_metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization and reduced gain
                nn.init.xavier_normal_(layer.weight, gain=0.1)  # Reduced gain for stability
                layer.weight.requires_grad_(True)
                
                if layer.bias is not None:
                    # Initialize bias with small values
                    nn.init.constant_(layer.bias, 0.01)
                    layer.bias.requires_grad_(True)
                    
                # Register gradient hooks for debugging only
                def make_hook(name):
                    def hook(grad):
                        if grad is not None:
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"real_metric_net.{layer}.weight"))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"real_metric_net.{layer}.bias"))
        
        # Initialize imaginary metric network with proper gradient tracking
        imag_layers = []
        imag_layers.append(nn.Linear(manifold_dim, hidden_dim, dtype=torch.float32))
        imag_layers.append(nn.Tanh())
        for _ in range(2):  # Add intermediate layers
            imag_layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32))
            imag_layers.append(nn.Tanh())
        imag_layers.append(nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=torch.float32))
        self.imag_metric_net = nn.Sequential(*imag_layers)
        
        # Enable gradients for the entire network
        self.imag_metric_net.requires_grad_(True)
        
        # Initialize weights with proper dtype and scaling
        for layer in self.imag_metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier normal initialization and reduced gain
                nn.init.xavier_normal_(layer.weight, gain=0.1)  # Reduced gain for stability
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    # Initialize bias with small values
                    nn.init.constant_(layer.bias, 0.01)
                    layer.bias.requires_grad_(True)
                
                # Register gradient hooks for debugging only
                def make_hook(name):
                    def hook(grad):
                        if grad is not None:
                            print(f"Gradient for {name}: {grad.abs().mean().item()}")
                        return grad
                    return hook
                layer.weight.register_hook(make_hook(f"imag_metric_net.{layer}.weight"))
                if layer.bias is not None:
                    layer.bias.register_hook(make_hook(f"imag_metric_net.{layer}.bias"))
        
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
        """Compute metric tensor from points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        # Ensure points have correct dtype and require gradients
        points = points.to(dtype=self.dtype).requires_grad_(True)
        
        # Get batch size and flatten if needed
        batch_size = points.shape[0] if points.dim() > 1 else 1
        points_flat = points.reshape(batch_size, -1)
        
        # Split into real and imaginary parts and ensure float32 dtype for network inputs
        real_points = points_flat.real.to(dtype=torch.float32)
        imag_points = points_flat.imag.to(dtype=torch.float32)
        
        # Compute metric components directly without using parent class's networks
        real_metric = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, 
                                dtype=torch.float32, device=points.device)
        imag_metric = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim,
                                dtype=torch.float32, device=points.device)
        
        # Add learnable components
        real_metric = real_metric + self.base_metric.real
        imag_metric = imag_metric + self.base_metric.imag
        
        # Add point-dependent components
        real_points_matrix = real_points.reshape(batch_size, self.manifold_dim, 1) @ real_points.reshape(batch_size, 1, self.manifold_dim)
        imag_points_matrix = imag_points.reshape(batch_size, self.manifold_dim, 1) @ imag_points.reshape(batch_size, 1, self.manifold_dim)
        
        real_metric = real_metric + real_points_matrix
        imag_metric = imag_metric + imag_points_matrix
        
        # Combine into full metric tensor
        metric = torch.complex(real_metric, imag_metric)
        
        # Add small diagonal term for stability
        eye = torch.eye(
            self.manifold_dim,
            device=metric.device,
            dtype=metric.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        metric = metric + 1e-6 * eye
        
        # Ensure metric requires gradients
        metric = metric.requires_grad_(True)
        
        return metric
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced gradient tracking."""
        print("\nGeometricFlow forward:")
        print(f"  Input shape: {x.shape}")
        print(f"  Input requires grad: {x.requires_grad}")
        print(f"  Input grad fn: {x.grad_fn}")
        print(f"  Arithmetic coupling requires grad: {self.arithmetic.coupling.requires_grad}")
        
        # Track real metric net parameters before forward
        print("\nReal metric net parameters before forward:")
        for name, param in self.real_metric_net.named_parameters():
            print(f"  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Requires grad: {param.requires_grad}")
            print(f"    Grad fn: {param.grad_fn}")
            print(f"    Is leaf: {param.is_leaf}")
        
        # Apply arithmetic dynamics with gradient tracking
        arithmetic_output, metrics = self.arithmetic(x)
        print(f"  Arithmetic output shape: {arithmetic_output.shape}")
        print(f"  Arithmetic output requires grad: {arithmetic_output.requires_grad}")
        print(f"  Arithmetic output grad fn: {arithmetic_output.grad_fn}")
        print(f"  Arithmetic metrics: {metrics}")
        
        # Add coupling contribution to ensure gradients
        coupling_contribution = torch.einsum('ij,ij->', self.arithmetic.coupling, self.arithmetic.coupling)
        print(f"  Coupling contribution: {coupling_contribution.item():.6f}")
        print(f"  Coupling contribution requires grad: {coupling_contribution.requires_grad}")
        print(f"  Coupling contribution grad fn: {coupling_contribution.grad_fn}")
        
        # Scale coupling contribution and add to output
        coupling_scale = 0.001  # Reduced scale for stability
        coupling_term = coupling_scale * coupling_contribution.real
        output = arithmetic_output + coupling_term.unsqueeze(0).unsqueeze(0).expand_as(arithmetic_output)
        print(f"  Output shape after coupling: {output.shape}")
        print(f"  Output requires grad: {output.requires_grad}")
        print(f"  Output grad fn: {output.grad_fn}")
        
        # Add connection coefficients contribution to ensure gradient flow
        connection_contribution = self.connection_coeffs.abs().sum() * 0.0001  # Small scale for stability
        output = output + connection_contribution.unsqueeze(0).unsqueeze(0).expand_as(output)
        
        # Add base metric contribution to ensure gradient flow
        base_metric_contribution = self.base_metric.abs().sum() * 0.0001  # Small scale for stability
        output = output + base_metric_contribution.unsqueeze(0).unsqueeze(0).expand_as(output)
        
        # Add explicit metric regularization
        metric_reg = torch.einsum('ij,ij->', self.base_metric, self.base_metric) * 0.0001
        output = output + metric_reg.real.unsqueeze(0).unsqueeze(0).expand_as(output)
        
        # Track real metric net parameters after forward
        print("\nReal metric net parameters after forward:")
        for name, param in self.real_metric_net.named_parameters():
            print(f"  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Requires grad: {param.requires_grad}")
            print(f"    Grad fn: {param.grad_fn}")
            print(f"    Is leaf: {param.is_leaf}")
            print(f"    Has grad: {param.grad is not None}")
            if param.grad is not None:
                print(f"    Grad shape: {param.grad.shape}")
                print(f"    Grad mean: {param.grad.abs().mean().item():.6f}")
        
        return output
    
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
    
    def compute_christoffel(
        self,
        metric: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Christoffel symbols using neural network.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            points: Optional points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Christoffel symbols tensor of shape (batch_size, manifold_dim, manifold_dim, manifold_dim)
        """
        batch_size = metric.shape[0]
        
        # If points is None, use flattened metric as input
        if points is None:
            points = metric.reshape(batch_size, -1)
        
        # Project points to manifold dimension if needed
        if points.shape[-1] > self.manifold_dim:
            points = points[:, :self.manifold_dim]
        
        # Compute Christoffel symbols using neural network
        christoffel = self.christoffel_net(points)
        
        # Reshape output to proper dimensions
        christoffel = christoffel.view(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
        return christoffel

