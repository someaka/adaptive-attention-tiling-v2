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
from dataclasses import dataclass

from ..patterns.arithmetic_dynamics import ArithmeticDynamics
from ..patterns.riemannian_flow import RiemannianFlow

@dataclass
class FlowParams:
    """Parameters for geometric flow."""
    dt: float
    stability_threshold: float
    stress_energy_weight: float = 0.1
    use_quantum_features: bool = False

class ComplexReLU(nn.Module):
    """ReLU activation for complex numbers."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return torch.complex(
                F.relu(x.real),
                F.relu(x.imag)
            )
        return F.relu(x)

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
        dtype: torch.dtype = torch.float32,
        use_quantum_features: bool = False
    ):
        """Initialize geometric flow.
        
        Args:
            hidden_dim: Hidden dimension for flow computations
            manifold_dim: Dimension of the base manifold
            motive_rank: Rank of motive force
            num_charts: Number of coordinate charts
            integration_steps: Number of integration steps
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
            dtype: Data type for tensors
            use_quantum_features: Whether to use quantum features
        """
        super().__init__(manifold_dim=manifold_dim)
        
        # Initialize parameters
        self.params = FlowParams(
            dt=dt,
            stability_threshold=stability_threshold,
            stress_energy_weight=0.1,
            use_quantum_features=use_quantum_features
        )
        
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.motive_rank = motive_rank
        self.num_charts = num_charts
        self.integration_steps = integration_steps
        self.dtype = dtype
        self.use_quantum_features = use_quantum_features
        self.quantum_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        
        # Initialize flow layers with proper dtype
        layer_dtype = self.quantum_dtype if use_quantum_features else dtype
        self.flow_layers = nn.ModuleList([
            nn.Linear(manifold_dim, hidden_dim, dtype=layer_dtype),
            ComplexReLU(),
            nn.Linear(hidden_dim, manifold_dim, dtype=layer_dtype)
        ])
        
        # Initialize metric network with proper dtype
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim, dtype=layer_dtype),
            ComplexReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=layer_dtype)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if use_quantum_features:
                    # Initialize complex weights with correct base dtype and unitary structure
                    real_weight = torch.empty(m.weight.shape, dtype=dtype)
                    imag_weight = torch.empty(m.weight.shape, dtype=dtype)
                    
                    # Initialize real part with Xavier uniform
                    nn.init.xavier_uniform_(real_weight)
                    
                    # Initialize imaginary part with small random values
                    nn.init.xavier_uniform_(imag_weight)
                    imag_weight.mul_(0.1)  # Scale down imaginary part for stability
                    
                    # Ensure weight matrix is approximately unitary for stability
                    weight = torch.complex(real_weight, imag_weight)
                    u, _, v = torch.linalg.svd(weight, full_matrices=False)
                    weight = torch.matmul(u, v)  # Construct closest unitary matrix
                    m.weight.data = weight.to(dtype=layer_dtype)
                    
                    if m.bias is not None:
                        # Initialize bias with small complex values
                        real_bias = torch.empty(m.bias.shape, dtype=dtype)
                        imag_bias = torch.empty(m.bias.shape, dtype=dtype)
                        nn.init.uniform_(real_bias, -0.1, 0.1)
                        nn.init.uniform_(imag_bias, -0.01, 0.01)  # Smaller imaginary part
                        m.bias.data = torch.complex(real_bias, imag_bias).to(dtype=layer_dtype)
                else:
                    # Standard initialization for real weights
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    # Ensure correct dtype
                    m.weight.data = m.weight.data.to(dtype=layer_dtype)
                    if m.bias is not None:
                        m.bias.data = m.bias.data.to(dtype=layer_dtype)
    
    def _init_quantum_features(self):
        """Initialize weights for quantum features."""
        if not self.use_quantum_features:
            raise RuntimeError("Attempting to initialize quantum features when not enabled")
            
        # Convert flow layers to complex dtype
        for layer in self.flow_layers:
            if isinstance(layer, nn.Linear):
                # Initialize complex weights
                real_weight = nn.init.xavier_uniform_(torch.empty_like(layer.weight.real))
                imag_weight = torch.zeros_like(layer.weight.real)
                layer.weight.data = torch.complex(real_weight, imag_weight)
                
                if layer.bias is not None:
                    real_bias = nn.init.zeros_(torch.empty_like(layer.bias.real))
                    imag_bias = torch.zeros_like(layer.bias.real)
                    layer.bias.data = torch.complex(real_bias, imag_bias)
        
        # Convert metric network to complex dtype
        for layer in self.metric_net:
            if isinstance(layer, nn.Linear):
                # Initialize complex weights
                real_weight = nn.init.xavier_uniform_(torch.empty_like(layer.weight.real))
                imag_weight = torch.zeros_like(layer.weight.real)
                layer.weight.data = torch.complex(real_weight, imag_weight)
                
                if layer.bias is not None:
                    real_bias = nn.init.zeros_(torch.empty_like(layer.bias.real))
                    imag_bias = torch.zeros_like(layer.bias.real)
                    layer.bias.data = torch.complex(real_bias, imag_bias)
    
    def compute_hamiltonian(self, metric: torch.Tensor, include_quantum: bool = True) -> torch.Tensor:
        """Compute Hamiltonian energy of the metric tensor.
        
        This method computes both classical and quantum contributions to the
        Hamiltonian energy. The quantum contribution is only included if
        quantum features are enabled and include_quantum is True.
        
        Args:
            metric: Metric tensor of shape [batch_size, n, n]
            include_quantum: Whether to include quantum energy contribution
            
        Returns:
            Hamiltonian energy tensor
            
        Raises:
            ValueError: If metric tensor contains invalid values
        """
        if not torch.isfinite(metric).all():
            raise ValueError("Metric tensor contains non-finite values")
            
        # Compute classical terms
        det = torch.linalg.det(metric).real  # Take real part for energy
        trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(-1)
        
        # Compute Ricci scalar (simplified)
        ricci_scalar = -torch.log(det.abs() + self.params.stability_threshold)
        
        # Combine classical terms for energy
        energy = (
            ricci_scalar +  # Curvature term
            0.1 * trace +   # Kinetic term
            0.01 * det      # Volume term
        )
        
        # Add quantum contribution if enabled
        if self.use_quantum_features and include_quantum and hasattr(self, 'quantum_net'):
            quantum_energy = self.quantum_net(metric).squeeze(-1).real
            energy = energy + self.params.stress_energy_weight * quantum_energy
            
        return energy.unsqueeze(-1)  # Add dimension for consistency
    
    def compute_quantum_energy(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute quantum energy of the metric tensor.
        
        Args:
            metric: Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Quantum energy tensor
            
        Raises:
            RuntimeError: If quantum features are not enabled
        """
        if not self.use_quantum_features:
            raise RuntimeError("Quantum features not enabled")
            
        if not hasattr(self, 'quantum_net'):
            raise RuntimeError("Quantum network not initialized")
            
        # Handle complex input
        if torch.is_complex(metric):
            metric = metric.real
            
        # Compute quantum energy
        with torch.set_grad_enabled(True):
            energy = self.quantum_net(metric)
            
        # Ensure real output
        if torch.is_complex(energy):
            energy = energy.real
            
        return energy.squeeze(-1).mean()
    
    def compute_quantum_metric(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute metric with quantum geometric structure.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantum metric tensor if quantum features are enabled,
            otherwise returns None
        """
        if not self.use_quantum_features or not hasattr(self, 'arithmetic'):
            return None
            
        # Project input to manifold dimension
        x_proj = self.proj(x)
        
        # Ensure parameters have the same dtype as input
        for param in self.parameters():
            param.data = param.data.to(dtype=self.quantum_dtype)
            
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
        metric = metric + self.stability_threshold * torch.eye(
            metric.size(-1), dtype=metric.dtype, device=metric.device
        ).unsqueeze(0).expand_as(metric)
        
        return metric
    
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
        
        # Add quantum corrections if enabled
        if self.use_quantum_features and self.arithmetic is not None:
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
        
        # Add quantum geometric metrics if enabled
        if self.use_quantum_features and hasattr(self, 'arithmetic'):
            metrics.update({
                'quantum_correction': torch.norm(
                    self.arithmetic.compute_quantum_correction(metric)
                ).item(),
                'quantum_energy': self.compute_quantum_energy(metric).item()
            })
        
        return new_metric, metrics
    
    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the Riemannian metric tensor at the given points."""
        # Get raw metric from network
        metric = self.metric_net(points)
        metric = metric.view(points.shape[0], self.manifold_dim, self.manifold_dim)
        
        # Ensure symmetry
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Add positive definite regularization
        eye = torch.eye(self.manifold_dim, dtype=metric.dtype, device=metric.device)
        eye = eye.expand(points.shape[0], -1, -1)
        
        # Project onto positive definite cone
        eigenvals = torch.linalg.eigvalsh(metric)
        min_eigenval = eigenvals.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        shift = torch.relu(-min_eigenval) + self.stability_threshold
        metric = metric + shift * eye
        
        return metric
    
    def compute_ricci(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the Ricci tensor at the given points."""
        # First compute metric
        metric = self.compute_metric(points)
        
        # Get Ricci tensor from parent class
        ricci = super().compute_ricci_tensor(points)
        
        # Ensure symmetry
        ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
        
        return ricci
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with geometric flow.
        
        Args:
            x: Input tensor of shape [batch_size, time_steps, ...] or [batch_size, ...]
               where ... represents spatial dimensions.
               
        Returns:
            Tuple of:
            - Output tensor with same shape as input
            - Dictionary containing metrics like energy and stability
        """
        # Store original dtype and shape
        orig_dtype = x.dtype
        orig_shape = x.shape
        is_complex_dtype = torch.is_complex(x)
        
        # Convert to working dtype while preserving complex values
        if self.use_quantum_features:
            x = x.to(dtype=self.quantum_dtype)
        else:
            if torch.is_complex(x):
                x = x.real
            x = x.to(dtype=self.dtype)
        
        # Check if we have time dimension
        has_time_dim = len(x.shape) > 2
        if has_time_dim:
            time_steps = x.size(1)
            # Flatten time dimension into batch
            x = x.reshape(-1, *x.shape[2:])
        else:
            time_steps = 1
            x = x.unsqueeze(0)
            
        # Handle dimensionality by projecting to manifold space instead of truncating
        if x.size(-1) > self.manifold_dim:
            # Project to manifold dimension using learned projection
            if not hasattr(self, 'dim_reduction'):
                # Initialize projection matrix if not exists
                in_dim = x.size(-1)
                if torch.is_complex(x):
                    real_proj = torch.randn(in_dim, self.manifold_dim) * 0.02
                    imag_proj = torch.randn(in_dim, self.manifold_dim) * 0.02
                    proj = torch.complex(real_proj, imag_proj)
                    self.dim_reduction = nn.Parameter(proj.to(dtype=x.dtype))
                else:
                    self.dim_reduction = nn.Parameter(
                        torch.randn(in_dim, self.manifold_dim).to(dtype=x.dtype) * 0.02
                    )
            
            # Apply projection
            x = x @ self.dim_reduction
            
        # Compute metrics
        metrics = {}
        metric = self.compute_metric(x)
        metrics['energy'] = self._compute_energy(metric)
        metrics['stability'] = self._compute_stability(metric)
        
        # Apply flow layers
        output = x
        for layer in self.flow_layers:
            output = layer(output)
            
        # Project back to original dimension if needed
        if orig_shape[-1] > self.manifold_dim:
            # Use transpose of reduction matrix for inverse projection
            output = output @ self.dim_reduction.transpose(-2, -1)
            
        # Reshape output back to original shape
        if has_time_dim:
            output = output.reshape(*orig_shape)
        else:
            output = output.squeeze(0)
            
        # Convert back to original dtype while preserving complex values
        if torch.is_complex(output) and not is_complex_dtype:
            # If original was real but we produced complex, take magnitude
            output = output.abs()
        output = output.to(dtype=orig_dtype)
        
        return output, metrics
    
    def _compute_energy(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute energy of metric tensor.
        
        Args:
            metric: Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Energy tensor of shape [batch_size]
        """
        # Add stability threshold to diagonal
        eye = torch.eye(
            self.manifold_dim,
            device=metric.device,
            dtype=metric.dtype
        ).unsqueeze(0).expand(metric.size(0), -1, -1)
        metric_stable = metric + self.params.stability_threshold * eye

        # Compute determinant and trace
        det = torch.linalg.det(metric_stable).abs()
        tr = torch.diagonal(metric_stable, dim1=-2, dim2=-1).sum(-1)

        # Compute condition number
        if torch.is_complex(metric):
            eigenvalues = torch.linalg.eigvals(metric_stable)
            eigenvalues_abs = eigenvalues.abs()
            condition = eigenvalues_abs.max(-1)[0] / (eigenvalues_abs.min(-1)[0] + self.params.stability_threshold)
        else:
            eigenvalues = torch.linalg.eigvalsh(metric_stable)
            condition = eigenvalues.max(-1)[0] / (eigenvalues.min(-1)[0] + self.params.stability_threshold)

        # Compute energy terms
        det_energy = -torch.log(det + self.params.stability_threshold)
        tr_energy = 0.5 * tr
        condition_energy = torch.log(condition + self.params.stability_threshold)

        # Combine energy terms with stress energy weight
        energy = det_energy + tr_energy + self.params.stress_energy_weight * condition_energy
        return energy
        
    def _compute_stability(self, metric: Tensor) -> Tensor:
        """Compute stability of metric tensor.
        
        Args:
            metric: Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Stability scalar measuring numerical conditioning
            
        Raises:
            ValueError: If metric tensor contains invalid values
        """
        if not torch.isfinite(metric).all():
            raise ValueError("Metric tensor contains non-finite values")
            
        # Add stability term
        metric = metric + self.params.stability_threshold * torch.eye(
            metric.size(-1), device=metric.device, dtype=metric.dtype
        ).unsqueeze(0)
        
        # Compute eigenvalues safely
        try:
            eigenvals = torch.linalg.eigvalsh(metric)  # Use eigvalsh for Hermitian matrices
        except RuntimeError:
            # If eigenvalue computation fails, return worst stability
            return torch.zeros(metric.size(0), device=metric.device)
            
        # Get real part if complex
        if torch.is_complex(eigenvals):
            eigenvals = eigenvals.real
            
        # Compute condition number
        max_eigenval = eigenvals.max(dim=-1).values
        min_eigenval = eigenvals.min(dim=-1).values
        condition_number = max_eigenval / (min_eigenval + self.params.stability_threshold)
        
        # Stability is inverse of condition number
        stability = 1.0 / (condition_number + self.params.stability_threshold)
        
        return stability

