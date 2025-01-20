"""Neural Geometric Flow Implementation.

This module provides a neural network-specific implementation of geometric flows,
building on top of pattern formation dynamics and adding neural-specific features.
Implements the vertical integration between pattern, geometric, and quantum layers.
"""

from __future__ import annotations

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast, TypeVar, Generic
from dataclasses import dataclass

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Type checking imports
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

# Local imports - core functionality
from .pattern import PatternFormationFlow
from .protocol import FlowMetrics, QuantumFlowMetrics, SingularityInfo as BaseSingularityInfo

# Local imports - quantum components
from ..quantum.neural_quantum_bridge import NeuralQuantumBridge
from ..quantum.types import QuantumState
from ...validation.quantum.state import QuantumStateValidationResult

# Local imports - dimension management
from src.core.common.dimensions import DimensionManager, DimensionConfig

# Enable torchtyping
patch_typeguard()

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
BatchTensor = TensorType["batch_size", "manifold_dim"]
MetricTensor = TensorType["batch_size", "manifold_dim", "manifold_dim"]
ConnectionTensor = TensorType["batch_size", "manifold_dim", "manifold_dim", "manifold_dim"]
CurvatureTensor = TensorType["batch_size", "manifold_dim", "manifold_dim"]

@dataclass
class SingularityInfo:
    """Information about a detected singularity."""
    index: int
    point: Optional[torch.Tensor]
    eigenvalues: torch.Tensor

def enable_gradients(module):
    """Enable gradients for all parameters in a module and its children."""
    if not isinstance(module, torch.nn.Module):
        return
        
    for param in module.parameters():
        param.requires_grad_(True)
        
    for child in module.children():
        enable_gradients(child)

class ComplexReLU(nn.Module):
    """ReLU activation for complex numbers.
    
    Applies ReLU separately to real and imaginary parts.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_complex():
            return torch.complex(
                F.relu(x.real),
                F.relu(x.imag)
            )
        return F.relu(x)

class NeuralGeometricFlow(PatternFormationFlow):
    """Neural network-specific implementation of geometric flow.
    
    This class implements the vertical integration between:
    1. Pattern Processing Layer (inherited from PatternFormationFlow)
    2. Geometric Processing Layer (neural weight space geometry)
    3. Quantum Integration Layer (quantum state management and evolution)
    
    Key Integration Points:
    1. Pattern → Geometric
       - Pattern field mapping
       - Geometric flow preservation
       - Scale connection handling
       
    2. Geometric → Quantum
       - Quantum state preparation
       - Geometric phase tracking
       - Entanglement management
       
    3. Horizontal Integration
       - Information transport
       - Structure preservation
       - Resource allocation
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: int,
        motive_rank: int = 4,
        num_primes: int = 8,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        fisher_rao_weight: float = 1.0,
        quantum_weight: float = 1.0,
        num_heads: int = 8,
        dropout: float = 0.1,
        quantum_correction_strength: float = 0.1,
        phase_tracking_enabled: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        stability_term: float = 0.1,
        test_config: Optional[dict] = None
    ):
        """Initialize neural geometric flow with quantum integration.
        
        Args:
            manifold_dim: Dimension of the base manifold
            hidden_dim: Hidden dimension for computations
            motive_rank: Rank of motivic structure (default: 4)
            num_primes: Number of prime bases (default: 8)
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
            fisher_rao_weight: Weight for Fisher-Rao metric contribution
            quantum_weight: Weight for quantum contribution
            num_heads: Number of attention heads for quantum bridge
            dropout: Dropout rate for quantum bridge
            quantum_correction_strength: Strength of quantum corrections
            phase_tracking_enabled: Whether to track geometric phases
            dtype: Data type for tensors
            device: Device for computation
            stability_term: Coefficient for stability term in connection computation
            test_config: Optional test configuration dictionary
        """
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            stability_threshold=stability_threshold,
            motive_rank=motive_rank,
            num_primes=num_primes,
            dtype=dtype
        )
        
        # Store configuration
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim
        self.motive_rank = motive_rank
        self.num_primes = num_primes
        self.fisher_rao_weight = fisher_rao_weight
        self.quantum_weight = quantum_weight
        self.quantum_correction_strength = quantum_correction_strength
        self.phase_tracking_enabled = phase_tracking_enabled
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        self.stability_term = stability_term
        
        # Calculate dimensions
        self.projection_dim = hidden_dim * 2
        self.connection_dim = manifold_dim * manifold_dim * manifold_dim  # Fix: Only need 3 dimensions
        
        # Calculate number of parameters for lower triangular matrix
        n = manifold_dim
        self.n_metric_params = n * (n + 1) // 2  # Number of elements in lower triangular matrix
        
        # Initialize quantum bridge with proper configuration
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=hidden_dim,  # Use the flow's hidden dimension
            num_heads=num_heads,
            dropout=dropout,
            dtype=self.dtype,
            manifold_type="hyperbolic",
            curvature=-1.0
        )
        
        # Initialize quantum bridge projection layer
        self.quantum_bridge_projection = nn.Linear(
            manifold_dim * manifold_dim,  # Input is flattened metric
            hidden_dim,  # Project to hidden dim
            dtype=self.dtype,
            bias=True
        )
        
        # Custom orthogonal initialization for complex numbers
        if self.dtype.is_complex:
            # Initialize real and imaginary parts separately
            real_weight = torch.empty(hidden_dim, manifold_dim * manifold_dim, dtype=torch.float64)
            imag_weight = torch.empty(hidden_dim, manifold_dim * manifold_dim, dtype=torch.float64)
            
            # Use Xavier initialization for both parts
            nn.init.xavier_uniform_(real_weight)
            nn.init.xavier_uniform_(imag_weight)
            
            # Scale down imaginary part for stability
            imag_weight *= 0.1
            
            # Combine into complex weight
            complex_weight = torch.complex(real_weight, imag_weight)
            
            # Make approximately unitary using QR decomposition
            q, r = torch.linalg.qr(complex_weight)  # No need to transpose here
            d = torch.diag(r, 0)
            ph = torch.sgn(d)  # Use sgn instead of sign for complex numbers
            q = q * ph.unsqueeze(-1)  # Broadcast phase along the last dimension
            
            # Set the weight
            with torch.no_grad():
                self.quantum_bridge_projection.weight.copy_(q)
                self.quantum_bridge_projection.bias.zero_()
        else:
            nn.init.orthogonal_(self.quantum_bridge_projection.weight)
            nn.init.zeros_(self.quantum_bridge_projection.bias)

        # Initialize dimension manager for operadic transitions
        self.dim_manager = DimensionManager(
            DimensionConfig.from_test_config(test_config)
        )
        
        # Initialize networks with proper dimensioning and gradients
        self._init_fisher_rao_networks()
        self._init_quantum_networks()
        self._init_connection_networks()
        
        # Enable gradients for all parameters recursively
        enable_gradients(self)
        
        # Ensure specific components have gradients enabled
        if hasattr(self, 'quantum_bridge'):
            enable_gradients(self.quantum_bridge)
            if hasattr(self.quantum_bridge, 'pattern_bundle'):
                enable_gradients(self.quantum_bridge.pattern_bundle)
                
        if hasattr(self, 'arithmetic'):
            enable_gradients(self.arithmetic)
            for name, module in self.arithmetic.named_children():
                enable_gradients(module)
                
        if hasattr(self, 'pattern_formation'):
            enable_gradients(self.pattern_formation)
            if hasattr(self.pattern_formation, 'symplectic'):
                enable_gradients(self.pattern_formation.symplectic)
                
        if hasattr(self, 'pattern_evolution'):
            enable_gradients(self.pattern_evolution)
            if hasattr(self.pattern_evolution, 'framework'):
                enable_gradients(self.pattern_evolution.framework)
                
        if hasattr(self, 'operadic'):
            enable_gradients(self.operadic)
            
        if hasattr(self, 'wave'):
            enable_gradients(self.wave)
            
        if hasattr(self, 'reaction_net'):
            enable_gradients(self.reaction_net)
            
        if hasattr(self, 'diffusion_net'):
            enable_gradients(self.diffusion_net)
            
        if hasattr(self, 'control_net'):
            enable_gradients(self.control_net)
            
        if hasattr(self, 'fisher_net'):
            enable_gradients(self.fisher_net)
            
        if hasattr(self, 'expectation_projection'):
            enable_gradients(self.expectation_projection)
            
        if hasattr(self, 'metric_projection'):
            enable_gradients(self.metric_projection)
            
        if hasattr(self, 'quantum_correction_net'):
            enable_gradients(self.quantum_correction_net)
            
        if hasattr(self, 'connection_projection'):
            enable_gradients(self.connection_projection)
        
        # quantum_correction_net is initialized in _init_quantum_networks
        
    def _handle_dimension(
        self,
        tensor: Tensor,
        target_dim: Optional[int] = None,
        preserve_gradients: bool = True
    ) -> Tensor:
        """Handle dimension transitions using operadic structure.
        
        Args:
            tensor: Input tensor to transform
            target_dim: Optional target dimension (defaults to manifold_dim)
            preserve_gradients: Whether to preserve gradient flow
            
        Returns:
            Transformed tensor with correct dimensions
            
        Raises:
            ValueError: If tensor dimension is invalid
        """
        if target_dim is None:
            target_dim = self.manifold_dim
            
        if tensor.shape[-1] == target_dim:
            return tensor
            
        # Verify dimension compatibility
        self.dim_manager.verify_dimension(tensor)
        
        # Handle dimension transition
        try:
            with torch.set_grad_enabled(preserve_gradients):
                # Save original shape
                original_shape = tensor.shape[:-1]
                
                # Project using learned transformation
                projection = nn.Linear(
                    tensor.shape[-1],
                    target_dim,
                    dtype=self.dtype,
                    device=self.device
                )
                
                # Initialize with proper scaling
                nn.init.orthogonal_(projection.weight)
                nn.init.zeros_(projection.bias)
                
                # Apply projection
                tensor = tensor.reshape(-1, tensor.shape[-1])
                result = projection(tensor)
                
                return result.reshape(*original_shape, target_dim)
                
        except Exception as e:
            raise ValueError(f"Failed to transform dimensions: {str(e)}") from e

    def _init_fisher_rao_networks(self):
        """Initialize networks for Fisher-Rao metric computation."""
        activation = ComplexReLU() if self.dtype.is_complex else nn.ReLU()
        
        self.fisher_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation,
            nn.Linear(self.hidden_dim, self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
    def _init_quantum_networks(self):
        """Initialize networks for quantum corrections."""
        manifold_squared = self.manifold_dim * self.manifold_dim
        activation = ComplexReLU() if self.dtype.is_complex else nn.ReLU()
        
        # Projection networks with proper dimensioning
        self.expectation_projection = nn.Sequential(
            nn.Linear(manifold_squared, self.manifold_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.manifold_dim, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation
        )
        
        self.metric_projection = nn.Sequential(
            nn.Linear(manifold_squared, self.manifold_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.manifold_dim, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation
        )
        
        # Correction network with residual connection and proper dimensioning
        self.quantum_correction_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.manifold_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.manifold_dim, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation,
            nn.Linear(self.manifold_dim, self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
    def _init_connection_networks(self):
        """Initialize networks for connection computation."""
        input_dim = self.manifold_dim
        if hasattr(self, 'point_dim'):
            input_dim = self.point_dim
            
        activation = ComplexReLU() if self.dtype.is_complex else nn.ReLU()
        
        # Connection projection with proper dimensioning
        self.connection_projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation
        )
        
        # Connection network with proper dimensioning - output all components
        self.connection_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim * 2, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation,
            nn.Linear(self.hidden_dim * 2, self.manifold_dim * self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
        # Initialize connection network with small random values
        with torch.no_grad():
            final_layer = self.connection_net[-1]
            nn.init.xavier_uniform_(final_layer.weight, gain=0.01)  # Use small gain for stability
            nn.init.zeros_(final_layer.bias)  # Keep bias at zero initially
        
        # Stability network with proper dimensioning
        stability_input_dim = self.manifold_dim + self.manifold_dim * self.manifold_dim + 1
        self.stability_net = nn.Sequential(
            nn.Linear(stability_input_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            activation,
            nn.Linear(self.hidden_dim, self.manifold_dim * self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )

    def _to_real(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to real tensor by taking real part."""
        if tensor.is_complex():
            return tensor.real.to(dtype=self.dtype)
        return tensor.to(dtype=self.dtype)

    def _to_tensor_type(self, tensor: torch.Tensor, tensor_type: type) -> TensorType:
        """Convert torch.Tensor to appropriate TensorType."""
        # We'll just cast the tensor directly without checking its type
        return cast(tensor_type, tensor)

    def compute_fisher_rao_metric(self, points: BatchTensor) -> MetricTensor:
        """Compute Fisher-Rao information metric efficiently."""
        batch_size = points.shape[0]
        
        # Compute score function with gradient
        with torch.set_grad_enabled(True):
            points_detached = points.detach().requires_grad_(True)
            score = self.fisher_net(points_detached)
            
            # Compute Fisher-Rao metric using autograd
            fisher_metric = torch.empty(
                (batch_size, self.manifold_dim, self.manifold_dim),
                device=points.device,
                dtype=points.dtype
            )
            
            for i in range(self.manifold_dim):
                grad_outputs = torch.zeros_like(score)
                grad_outputs[:, i] = 1.0
                gradients = torch.autograd.grad(
                    score,
                    points_detached,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                fisher_metric[:, i] = gradients
        
        return self._to_tensor_type(fisher_metric, MetricTensor)

    def prepare_quantum_state(
        self,
        points: Tensor,
        return_validation: bool = True
    ) -> Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]]:
        """Prepare quantum state from neural points.
        
        Implements the Geometric → Quantum vertical integration by:
        1. Converting neural patterns to quantum states
        2. Validating state preparation
        3. Tracking geometric phases if enabled
        
        Args:
            points: Neural pattern points
            return_validation: Whether to return validation metrics
            
        Returns:
            Quantum state and optional validation result
        """
        return self.quantum_bridge.neural_to_quantum(points, return_validation)

    def compute_quantum_corrections(
        self,
        state: Union[QuantumState, Tuple[QuantumState, QuantumStateValidationResult]],
        metric: Union[torch.Tensor, MetricTensor]
    ) -> MetricTensor:
        """Compute quantum corrections to the metric tensor.
        
        Args:
            state: Quantum state or tuple of (state, validation)
            metric: Base metric tensor
            
        Returns:
            Corrected metric tensor
        """
        # Guard tensor dimensions
        torch.jit.annotate(List[int], list(metric.shape))
        batch_size = metric.size(0)
        
        # Always convert metric to MetricTensor
        metric_tensor = self._to_tensor_type(metric, MetricTensor)
        
        # Extract quantum state if tuple
        if isinstance(state, tuple):
            state = state[0]
            
        # Get quantum state tensor and convert to QuantumTensor
        quantum_state = state.density_matrix()
        quantum_state = self.dim_manager.to_quantum_tensor(quantum_state)
        
        if len(quantum_state.shape) == 3:
            quantum_state = quantum_state.mean(dim=1)
            
        # Convert to real tensors and validate dimensions
        quantum_state = self._to_real(quantum_state)
        metric_real = self._to_real(metric_tensor)
        
        # Project quantum state to manifold dimension
        quantum_state_flat = quantum_state.reshape(batch_size, -1)
        quantum_state_proj = quantum_state_flat[:, :self.manifold_dim]
        
        # Apply correction network
        corrections = self.quantum_correction_net(quantum_state_proj)
        
        # Reshape output to metric tensor shape and convert to GeometricTensor
        metric_shape = corrections.reshape(batch_size, self.manifold_dim, self.manifold_dim)
        geometric_tensor = self.dim_manager.to_geometric_tensor(metric_shape)
        
        return self._to_tensor_type(geometric_tensor, MetricTensor)

    def compute_metric(self, points: torch.Tensor) -> MetricTensor:
        """Compute Riemannian metric tensor.
        
        Ensures positive definiteness through regularization and proper scaling.
        """
        batch_size = points.shape[0]
        
        # Project points through metric network
        metric_flat = self.metric_net(points)
        metric = metric_flat.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Log initial metric properties
        print(f"Initial metric stats:")
        print(f"- Shape: {metric.shape}")
        print(f"- Mean: {metric.mean().item():.6f}")
        print(f"- Std: {metric.std().item():.6f}")
        print(f"- Min: {metric.min().item():.6f}")
        print(f"- Max: {metric.max().item():.6f}")
        
        # Ensure symmetry
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Scale the metric to have reasonable eigenvalues
        metric_norm = torch.linalg.norm(metric, dim=(-2, -1), keepdim=True)
        metric = metric / (metric_norm + self.stability_threshold)
        
        # Add positive definite regularization with adaptive strength
        eye = torch.eye(self.manifold_dim, device=points.device, dtype=points.dtype)
        eigenvals = torch.linalg.eigvalsh(metric)
        min_eigenval = eigenvals.min(dim=-1)[0]
        
        # Use larger regularization when eigenvalues are negative
        reg_strength = torch.where(
            min_eigenval < self.stability_threshold,
            torch.ones_like(min_eigenval),
            self.stability_threshold * torch.ones_like(min_eigenval)
        )
        reg_strength = reg_strength.view(-1, 1, 1)
        
        metric = metric + reg_strength * eye.unsqueeze(0)
        
        # Log after regularization
        print(f"\nAfter regularization:")
        eigenvals = torch.linalg.eigvalsh(metric)
        print(f"- Min eigenvalue: {eigenvals.min().item():.6f}")
        print(f"- Max eigenvalue: {eigenvals.max().item():.6f}")
        print(f"- Determinant: {torch.linalg.det(metric).min().item():.6f}")
        
        # Project onto positive definite cone if still needed
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        min_eigenval = torch.min(eigenvalues, dim=-1)[0]
        needs_projection = min_eigenval < self.stability_threshold
        
        if torch.any(needs_projection):
            print(f"\nNeeds projection: {needs_projection.sum().item()} metrics")
            eigenvals = torch.clamp(eigenvalues, min=self.stability_threshold)
            metric = torch.matmul(
                torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
                eigenvectors.transpose(-2, -1)
            )
            
            # Log after projection
            print(f"\nAfter projection:")
            eigenvals = torch.linalg.eigvalsh(metric)
            print(f"- Min eigenvalue: {eigenvals.min().item():.6f}")
            print(f"- Max eigenvalue: {eigenvals.max().item():.6f}")
            print(f"- Determinant: {torch.linalg.det(metric).min().item():.6f}")
        
        return self._to_tensor_type(metric, MetricTensor)

    def flow_step(
        self,
        metric: Union[torch.Tensor, MetricTensor],
        ricci: Optional[Union[torch.Tensor, MetricTensor]] = None,
        timestep: float = 0.1,
        attention_pattern: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, QuantumFlowMetrics]:
        """Perform neural network-aware flow step with quantum integration."""
        # Convert input tensors to appropriate types
        metric_tensor = self._to_tensor_type(metric, MetricTensor)
        ricci_tensor = self._to_tensor_type(ricci, MetricTensor) if ricci is not None else None
        
        # Validate input dimensions
        batch_size = metric.shape[0]
        self._validate_dimensions(
            metric,
            [batch_size, self.manifold_dim, self.manifold_dim],
            "metric"
        )
        if ricci is not None:
            self._validate_dimensions(
                ricci,
                [batch_size, self.manifold_dim, self.manifold_dim],
                "ricci"
            )
        
        # Get pattern flow step from parent
        new_metric, base_metrics = super().flow_step(metric_tensor, ricci_tensor, timestep)
        
        # Scale metric to preserve volume while maintaining gradient flow
        # Add minimal regularization only for stability
        eye = torch.eye(
            self.manifold_dim,
            device=metric.device,
            dtype=metric.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        reg_metric = new_metric + self.stability_threshold * eye
        
        # Compute determinant after flow step
        det_before = torch.linalg.det(metric_tensor)
        det_after = torch.linalg.det(reg_metric)
        
        # Compute scale factor more precisely
        scale = torch.pow(torch.abs(det_before) / (torch.abs(det_after) + self.stability_threshold), 1.0/self.manifold_dim)
        
        # Apply less restrictive clamping using torch.clamp
        scale = torch.clamp(scale, min=0.01, max=100.0)
        scale = scale.view(-1, 1, 1)  # Reshape for broadcasting
        
        # Scale metric to preserve volume
        new_metric = reg_metric * scale
        
        # Ensure symmetry
        new_metric = 0.5 * (new_metric + new_metric.transpose(-2, -1))
        
        # Project onto positive definite cone with minimal eigenvalue bound
        eigenvalues, eigenvectors = torch.linalg.eigh(new_metric)
        eigenvalues = torch.clamp(eigenvalues, min=self.stability_threshold)
        new_metric = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        
        # Initialize quantum metrics efficiently
        device = metric.device
        dtype = metric.dtype
        
        quantum_metrics = {
            'quantum_entropy': torch.zeros((), device=device, dtype=dtype),
            'berry_phase': None,
            'mean_curvature': None,
            'quantum_corrections': None
        }
        
        # Quantum evolution and metrics computation
        if hasattr(self, 'quantum_bridge'):
            # Prepare initial state efficiently
            # Project metric to quantum state space using proper dimensionality
            metric_flat = new_metric.reshape(batch_size, -1)  # [batch_size, manifold_dim * manifold_dim]
            
            # Use the properly initialized projection layer
            metric_projected = self.quantum_bridge_projection(metric_flat)
            
            # Normalize the projected tensor
            metric_projected = F.normalize(metric_projected, p=2, dim=-1)
            
            initial_state = self.prepare_quantum_state(
                metric_projected,
                return_validation=False
            )
            
            if not isinstance(initial_state, tuple):
                # Evolve quantum state with attention pattern
                evolved_state = self.quantum_bridge.evolve_quantum_state_with_attention(
                    initial_state,
                    attention_pattern=attention_pattern,
                    time=timestep
                )
                
                # Compute quantum metrics efficiently
                inner_product = evolved_state.inner_product(initial_state)
                quantum_metrics['quantum_entropy'] = -torch.abs(inner_product).log()
                
                if self.phase_tracking_enabled:
                    quantum_metrics['berry_phase'] = torch.angle(inner_product)
                
                # Compute geometric quantities efficiently
                quantum_metrics['mean_curvature'] = torch.diagonal(
                    new_metric,
                    dim1=-2,
                    dim2=-1
                ).mean()
                
                # Get quantum corrections with gradient tracking
                quantum_metrics['quantum_corrections'] = self.compute_quantum_corrections(
                    evolved_state,
                    self._to_tensor_type(new_metric, MetricTensor)
                )
        
        # Create and return flow metrics
        # Convert float metrics to tensors while preserving gradients
        flow_magnitude = torch.as_tensor(float(base_metrics.flow_magnitude), device=device, dtype=dtype).requires_grad_(True)
        flow_magnitude = flow_magnitude.expand(batch_size)

        metric_determinant = torch.as_tensor(float(base_metrics.metric_determinant), device=device, dtype=dtype).requires_grad_(True)
        metric_determinant = metric_determinant.expand(batch_size)

        ricci_scalar = torch.as_tensor(float(base_metrics.ricci_scalar), device=device, dtype=dtype).requires_grad_(True)
        ricci_scalar = ricci_scalar.expand(batch_size)

        energy = torch.as_tensor(float(base_metrics.energy), device=device, dtype=dtype).requires_grad_(True)
        energy = energy.expand(batch_size)

        singularity = torch.as_tensor(float(base_metrics.singularity), device=device, dtype=dtype).requires_grad_(True)
        singularity = singularity.expand(batch_size)

        return new_metric, QuantumFlowMetrics(
            flow_magnitude=flow_magnitude,
            metric_determinant=metric_determinant,
            ricci_scalar=ricci_scalar,
            energy=energy,
            singularity=singularity,
            normalized_flow=base_metrics.normalized_flow,
            quantum_entropy=quantum_metrics['quantum_entropy'],
            berry_phase=quantum_metrics['berry_phase'],
            mean_curvature=quantum_metrics['mean_curvature'],
            quantum_corrections=quantum_metrics['quantum_corrections']
        )

    def parallel_transport(
        self,
        vector: Tensor,
        start_point: Tensor,
        end_point: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Parallel transport with neural network and quantum awareness."""
        # Get pattern-aware transport from parent
        transported = super().parallel_transport(
            vector, start_point, end_point, connection
        )
        
        # Scale by gradient ratio for neural networks
        start_norm = torch.norm(start_point, dim=-1, keepdim=True)
        end_norm = torch.norm(end_point, dim=-1, keepdim=True)
        scale = torch.sqrt(end_norm / (start_norm + 1e-8))
        transported = transported * scale
        
        # Add quantum geometric contribution if enabled
        if self.phase_tracking_enabled:
            start_state = self.prepare_quantum_state(start_point, return_validation=False)
            end_state = self.prepare_quantum_state(end_point, return_validation=False)
            
            if isinstance(start_state, QuantumState) and isinstance(end_state, QuantumState):
                # Compute transport phase using inner product
                phase = torch.angle(end_state.inner_product(start_state))
                transported = transported * torch.exp(1j * phase).real
        
        return transported

    def compute_connection(
        self,
        metric: MetricTensor,
        points: Optional[Union[torch.Tensor, BatchTensor]] = None
    ) -> ConnectionTensor:
        """Compute Christoffel symbols of the Levi-Civita connection."""
        batch_size = metric.shape[0]
    
        # If points is None, use metric as input to connection network
        if points is None:
            points = metric.view(batch_size, -1)
    
        # Convert points to BatchTensor
        points_tensor = self._to_tensor_type(points, BatchTensor)
    
        # Project points to manifold dimension
        points_proj = points_tensor[:, :self.manifold_dim]
        points_proj = self._to_tensor_type(points_proj, BatchTensor)
        
        # Project through connection projection first
        points_hidden = self.connection_projection(points_proj)
        
        # Then compute connection through main network
        network_output = self.connection_net(points_hidden)
        
        # Reshape to proper dimensions (batch_size, i, j, k)
        connection = network_output.view(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
        # Ensure proper symmetry in lower indices (required for Levi-Civita connection)
        connection = 0.5 * (connection + connection.transpose(-2, -1))
        
        # Add stability term to prevent degeneracy
        eye = torch.eye(self.manifold_dim, device=points.device, dtype=points.dtype)
        connection = connection + self.stability_threshold * eye.unsqueeze(0).unsqueeze(-1)
        
        return self._to_tensor_type(connection, ConnectionTensor)

    def _flow_metrics_to_dict(self, flow_metrics: QuantumFlowMetrics) -> Dict[str, Any]:
        """Convert flow metrics to dictionary, handling optional fields elegantly."""
        metrics = {
            'flow_magnitude': flow_metrics.flow_magnitude,
            'metric_determinant': flow_metrics.metric_determinant,
            'ricci_scalar': flow_metrics.ricci_scalar,
            'energy': flow_metrics.energy,
            'singularity': flow_metrics.singularity,
            'normalized_flow': flow_metrics.normalized_flow,
            'quantum_entropy': flow_metrics.quantum_entropy,
        }
        
        # Add optional metrics if present
        optional_metrics = {
            'berry_phase': flow_metrics.berry_phase,
            'mean_curvature': flow_metrics.mean_curvature,
            'quantum_corrections_norm': (
                torch.norm(flow_metrics.quantum_corrections)
                if flow_metrics.quantum_corrections is not None
                else None
            )
        }
        metrics.update({k: v for k, v in optional_metrics.items() if v is not None})
        
        return metrics

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through neural geometric flow."""
        # Initialize metrics dictionary
        metrics: Dict[str, Any] = {}
        
        # Convert input to BatchTensor while preserving gradients
        x_batch = x.clone()
        
        # Project to manifold dimension if needed
        batch_size = x_batch.shape[0]
        if x_batch.shape[-1] != self.manifold_dim:
            x_batch = self.dim_manager.project(
                x_batch,
                target_dim=self.manifold_dim,
                dtype=self.dtype,
                device=self.device
            )
        
        # 1. Compute geometric quantities
        metric = self.compute_metric(x_batch)
        connection = self.compute_connection(metric, x_batch)
        
        metrics.update({
            'metric_norm': torch.norm(metric),
            'connection_norm': torch.norm(connection)
        })
        
        # 2. Apply quantum corrections if enabled
        if self.quantum_weight > 0:
            # Project to hidden dimension for quantum state preparation
            if x.shape[-1] != self.hidden_dim:
                x_hidden = torch.nn.functional.pad(x, (0, self.hidden_dim - x.shape[-1]))
            else:
                x_hidden = x
            
            # Prepare quantum state and compute corrections
            quantum_state = self.prepare_quantum_state(x_hidden)
            quantum_correction = self.compute_quantum_corrections(quantum_state, metric)
            
            # Record quantum metrics
            metrics['quantum_correction_norm'] = torch.norm(quantum_correction)
            
            # Apply corrections to input
            correction_flat = self.dim_manager.reshape_to_flat(quantum_correction)
            correction_proj = self.dim_manager.project(
                correction_flat,
                target_dim=self.manifold_dim,
                dtype=self.dtype,
                device=self.device
            )
            x_batch = x_batch + self.quantum_weight * correction_proj
            
            # Recompute metric with quantum corrections
            metric = self.compute_metric(x_batch)
            
        # 3. Evolve the system
        new_metric, flow_metrics = self.flow_step(metric)
        
        # Project evolved metric back to manifold using Cholesky decomposition
        try:
            # Compute Cholesky decomposition
            L = torch.linalg.cholesky(new_metric)
            
            # Use lower triangular part as coordinates
            x_evolved = L.diagonal(dim1=-2, dim2=-1)
            
            # Add off-diagonal elements
            for i in range(1, self.manifold_dim):
                x_evolved = torch.cat([x_evolved, L[..., i, :i]], dim=-1)
                
        except:
            # If Cholesky fails, use simpler approach
            x_evolved = torch.diagonal(new_metric, dim1=-2, dim2=-1)
        
        # Ensure output has correct shape
        if x_evolved.shape[-1] != self.manifold_dim:
            x_evolved = self.dim_manager.project(
                x_evolved,
                target_dim=self.manifold_dim,
                dtype=self.dtype,
                device=self.device
            )
        
        # Add flow metrics to output
        metrics.update(self._flow_metrics_to_dict(flow_metrics))
        
        # Add regularization terms to ensure gradient flow through all components
        reg_terms = []
        
        # Add regularization for quantum bridge components
        if hasattr(self, 'quantum_bridge'):
            reg_terms.extend([
                torch.norm(self.quantum_bridge.layer_norm_real.weight),
                torch.norm(self.quantum_bridge.layer_norm_imag.weight),
                torch.norm(self.quantum_bridge.manifold_norm_real.weight),
                torch.norm(self.quantum_bridge.manifold_norm_imag.weight),
                torch.norm(self.quantum_bridge.inverse_projection.weight)
            ])
            
            if hasattr(self.quantum_bridge, 'pattern_bundle'):
                reg_terms.extend([
                    torch.norm(self.quantum_bridge.pattern_bundle.metric),
                    torch.norm(self.quantum_bridge.pattern_bundle.connection)
                ])
        
        # Add regularization for arithmetic components
        if hasattr(self, 'arithmetic'):
            reg_terms.extend([
                torch.norm(self.arithmetic.coupling),
                torch.norm(getattr(self.arithmetic.height_map, '0').weight),
                torch.norm(self.arithmetic.flow.weight),
                torch.norm(getattr(self.arithmetic.l_function, '0').weight),
                torch.norm(getattr(self.arithmetic.quantum_height, '0').weight),
                torch.norm(getattr(self.arithmetic.quantum_l_function, '0').weight)
            ])
        
        # Add regularization for other components
        if isinstance(self.fisher_net, nn.Module):
            reg_terms.append(torch.norm(getattr(self.fisher_net, '0').weight))
        if isinstance(self.expectation_projection, nn.Module):
            reg_terms.append(torch.norm(getattr(self.expectation_projection, '0').weight))
        if isinstance(self.metric_projection, nn.Module):
            reg_terms.append(torch.norm(getattr(self.metric_projection, '0').weight))
        if isinstance(self.quantum_correction_net, nn.Module):
            reg_terms.append(torch.norm(getattr(self.quantum_correction_net, '0').weight))
        if isinstance(self.connection_projection, nn.Module):
            reg_terms.append(torch.norm(getattr(self.connection_projection, '0').weight))
        
        # Combine regularization terms
        if reg_terms:
            reg_loss = torch.stack(reg_terms).sum() * 1e-6  # Small weight to not affect main objective
            x_evolved = x_evolved + reg_loss  # Add regularization to affect gradients
        
        return x_evolved, metrics

    def _validate_dimensions(
        self,
        tensor: Tensor,
        expected_shape: List[int],
        name: str
    ) -> None:
        """Validate tensor dimensions efficiently.
        
        Args:
            tensor: Input tensor to validate
            expected_shape: List of expected dimensions
            name: Name of tensor for error messages
        """
        actual_shape = tensor.shape
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Expected {name} to have {len(expected_shape)} dimensions, "
                f"got {len(actual_shape)}"
            )
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"Expected {name} dimension {i} to be {expected}, "
                    f"got {actual}"
                )

    def compute_stability_term(
        self,
        points: BatchTensor,
        metric: MetricTensor
    ) -> ConnectionTensor:
        """Compute stability term for connection computation."""
        batch_size = points.shape[0]
        
        # Ensure points has correct shape
        if points.shape[-1] != self.manifold_dim:
            points = self._to_tensor_type(
                points.view(batch_size, -1)[:, :self.manifold_dim],
                BatchTensor
            )
        
        # Flatten metric tensor correctly
        metric_flat = metric.view(batch_size, -1)
        if metric_flat.shape[-1] != self.manifold_dim * self.manifold_dim:
            # Pad or truncate to correct size
            target_size = self.manifold_dim * self.manifold_dim
            if metric_flat.shape[-1] < target_size:
                padding = torch.zeros(batch_size, target_size - metric_flat.shape[-1], device=metric.device)
                metric_flat = torch.cat([metric_flat, padding], dim=-1)
            else:
                metric_flat = metric_flat[:, :target_size]
        metric_flat = self._to_tensor_type(metric_flat, BatchTensor)
        
        # Compute stability features
        stability_features = torch.cat([
            points,
            metric_flat,
            torch.ones(batch_size, 1, device=points.device)
        ], dim=-1)
        
        # Project using stability network
        stability = self.stability_net(stability_features)
        
        # Reshape using dimension manager and convert to ConnectionTensor
        connection = self.dim_manager.reshape_to_connection(stability, batch_size)
        return self._to_tensor_type(connection, ConnectionTensor)

    def _init_networks(self):
        """Initialize neural networks with proper dimensioning."""
        # Activation function
        activation = ComplexReLU() if self.dtype.is_complex else nn.ReLU()
        
        # Metric network with proper dimensioning
        self.metric_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            activation,
            nn.Linear(self.hidden_dim, self.hidden_dim * 2, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim * 2),
            activation,
            nn.Linear(self.hidden_dim * 2, self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
        # Initialize final layer to approximate identity
        with torch.no_grad():
            final_layer = self.metric_net[-1]
            final_layer.weight.data.zero_()
            final_layer.bias.data.zero_()
            # Set diagonal elements to small positive values
            for i in range(min(self.manifold_dim, final_layer.bias.shape[0])):
                idx = i * self.manifold_dim + i
                if idx < final_layer.bias.shape[0]:
                    final_layer.bias.data[idx] = 0.1
        
        # Connection network with proper dimensioning - output all components
        self.connection_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim * 2, dtype=torch.float64 if self.dtype.is_complex else self.dtype),
            activation,
            nn.Linear(self.hidden_dim * 2, self.manifold_dim * self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
        # Initialize connection network with small random values
        with torch.no_grad():
            final_layer = self.connection_net[-1]
            nn.init.xavier_uniform_(final_layer.weight, gain=0.01)  # Use small gain for stability
            nn.init.zeros_(final_layer.bias)  # Keep bias at zero initially
        
        # Stability network with proper dimensioning
        stability_input_dim = self.manifold_dim + self.manifold_dim * self.manifold_dim + 1
        self.stability_net = nn.Sequential(
            nn.Linear(stability_input_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            activation,
            nn.Linear(self.hidden_dim, self.manifold_dim * self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )

    def compute_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
        
        # Compute metric components - output size should be manifold_dim * manifold_dim
        metric_components = self.metric_net(x)
        metric_components = metric_components[:, :self.manifold_dim * self.manifold_dim]
        
        # Reshape to metric tensor
        metric = metric_components.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Ensure metric is symmetric and positive definite
        metric = 0.5 * (metric + metric.transpose(-1, -2))
        
        # Add stability term
        metric.diagonal(dim1=-2, dim2=-1).add_(self.stability_threshold)
        
        # Project onto positive definite cone
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        eigenvalues = torch.clamp(eigenvalues, min=self.stability_threshold)
        metric = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        
        return metric

    def compute_flow_field(self, x: torch.Tensor) -> torch.Tensor:
        """Compute flow field from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Flow field tensor of shape (batch_size, seq_len, manifold_dim)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
            seq_len = 1
        
        # Compute metric tensor
        metric = self.compute_metric_tensor(x)
        
        # Compute flow field using metric
        flow_field = self.compute_flow(metric)  # Default time=0.0
        
        # Reshape back to sequence format if needed
        if seq_len > 1:
            flow_field = flow_field.view(batch_size, seq_len, -1)
        
        return flow_field

    def compute_divergence(self, flow_field: torch.Tensor) -> torch.Tensor:
        """Compute divergence of flow field.
        
        Args:
            flow_field: Flow field tensor of shape (batch_size, seq_len, manifold_dim)
            
        Returns:
            Divergence tensor of shape (batch_size, seq_len)
        """
        # Compute divergence using autograd
        batch_size = flow_field.shape[0]
        seq_len = flow_field.shape[1] if flow_field.dim() == 3 else 1
        
        # Reshape if needed
        if flow_field.dim() == 3:
            flow_field = flow_field.view(-1, self.manifold_dim)
        
        # Compute divergence
        divergence = torch.zeros(batch_size * seq_len, device=flow_field.device)
        
        for i in range(self.manifold_dim):
            divergence += torch.autograd.grad(
                flow_field[:, i].sum(),
                flow_field,
                create_graph=True,
                retain_graph=True
            )[0][:, i]
        
        # Reshape back if needed
        if seq_len > 1:
            divergence = divergence.view(batch_size, seq_len)
        
        return divergence

    def compute_curvature_flow(self, x: torch.Tensor) -> torch.Tensor:
        """Compute curvature flow from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Curvature flow tensor of shape (batch_size, seq_len, manifold_dim)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
            seq_len = 1
        
        # Compute metric tensor
        metric = self.compute_metric_tensor(x)
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Compute curvature flow
        flow = self.compute_flow(metric)  # Default time=0.0
        
        # Reshape back to sequence format if needed
        if seq_len > 1:
            flow = flow.view(batch_size, seq_len, -1)
        
        return flow

    def compute_ricci_flow(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Ricci flow from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Ricci flow tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
        
        # Compute metric tensor
        metric = self.compute_metric_tensor(x)
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        return ricci

    def compute_scalar_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar curvature from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Scalar curvature tensor of shape (batch_size,)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
        
        # Compute metric tensor
        metric = self.compute_metric_tensor(x)
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Compute scalar curvature as trace of Ricci tensor
        scalar = torch.diagonal(ricci, dim1=-2, dim2=-1).sum(-1)
        
        return scalar

    def compute_scalar_curvature_from_metric(
        self,
        metric: Union[torch.Tensor, MetricTensor],
        ricci: Optional[Union[torch.Tensor, MetricTensor]] = None
    ) -> torch.Tensor:
        """Compute scalar curvature from metric tensor.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            ricci: Optional pre-computed Ricci tensor
            
        Returns:
            Scalar curvature tensor of shape (batch_size,)
        """
        if ricci is None:
            ricci = self.compute_ricci_tensor(metric)
            
        # Compute scalar curvature as trace of Ricci tensor
        scalar = torch.diagonal(ricci, dim1=-2, dim2=-1).sum(-1)
        
        return scalar

    def evolve_flow(self, x: torch.Tensor, time_steps: int = 10) -> torch.Tensor:
        """Evolve flow for given number of time steps.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            time_steps: Number of time steps to evolve
            
        Returns:
            Evolved tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Initialize evolved tensor
        evolved = x
        
        # Evolve for given number of steps
        for _ in range(time_steps):
            # Compute metric tensor
            metric = self.compute_metric_tensor(evolved)
            
            # Compute Ricci tensor
            ricci = self.compute_ricci_tensor(metric, evolved)
            
            # Take flow step
            evolved_metric, _ = self.flow_step(metric, ricci)
            
            # Project back to input space
            evolved = self.project_to_input_space(evolved_metric)
        
        return evolved

    def compute_flow_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute flow energy from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Flow energy tensor of shape (batch_size,)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
        
        # Compute metric tensor
        metric = self.compute_metric_tensor(x)
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric)
        
        # Compute flow vector
        flow = self.compute_flow(metric)  # Default time=0.0
        
        # Compute energy as norm of flow vector
        energy = torch.norm(flow, dim=-1)
        
        return energy

    def compute_volume_form(self, x: torch.Tensor) -> torch.Tensor:
        """Compute volume form from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Volume form tensor of shape (batch_size,)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
        
        # Compute metric tensor
        metric = self.compute_metric_tensor(x)
        
        # Compute volume form as square root of metric determinant
        volume = torch.sqrt(torch.linalg.det(metric))
        
        return volume

    def project_to_input_space(self, metric: torch.Tensor) -> torch.Tensor:
        """Project metric tensor back to input space.
        
        Args:
            metric: Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
            
        Returns:
            Input space tensor of shape (batch_size, hidden_dim)
        """
        batch_size = metric.shape[0]
        
        # Flatten metric tensor
        metric_flat = metric.view(batch_size, -1)
        
        # Project to input space
        x = self.metric_net[0](metric_flat)  # Use first layer of metric network in reverse
        
        return x

    def integrate_flow(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate flow field to get evolved state.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Integrated tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Project to manifold space if needed
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
        else:
            batch_size = x.shape[0]
            seq_len = 1
        
        # Compute flow field
        flow_field = self.compute_flow_field(x)
        
        # Integrate flow field using Euler method
        dt = self.dt
        integrated = x + dt * flow_field
        
        # Reshape back to sequence format if needed
        if seq_len > 1:
            integrated = integrated.view(batch_size, seq_len, -1)
        
        return integrated

    def compute_flow(self, metric: torch.Tensor, time: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
        """Compute the geometric flow for the given metric tensor.
    
        Args:
            metric: The metric tensor to compute flow for.
            time: The time parameter for flow evolution. Can be a float or tensor.
    
        Returns:
            The computed flow vector with shape (batch_size, manifold_dim).
        """
        # Compute Ricci tensor from metric
        ricci = self.compute_ricci_tensor(metric)
    
        # Compute flow vector using Ricci tensor
        # Take the diagonal elements as the flow vector components
        flow_vector = -torch.diagonal(ricci, dim1=-2, dim2=-1)  # Shape: [batch_size, manifold_dim]
    
        # Apply time scaling if time tensor has any non-zero values
        if isinstance(time, torch.Tensor):
            if torch.nonzero(time).numel() > 0:
                # Reshape time tensor to match flow_vector dimensions
                if time.dim() == 3:  # [batch_size, manifold_dim, manifold_dim]
                    # Take diagonal to match flow_vector shape
                    time = torch.diagonal(time, dim1=-2, dim2=-1)  # Shape: [batch_size, manifold_dim]
                elif time.dim() == 2:  # [batch_size, manifold_dim]
                    pass  # Already in correct shape
                else:
                    time = time.view(flow_vector.shape)
                flow_vector = flow_vector * time
        elif time != 0:
            flow_vector = flow_vector * time
    
        return flow_vector

    def compute_curvature(
        self,
        metric: Union[torch.Tensor, MetricTensor],
        connection: Optional[Union[torch.Tensor, ConnectionTensor]] = None
    ) -> Union[torch.Tensor, CurvatureTensor]:
        """Compute Riemann curvature tensor.
        
        The Riemann curvature tensor R^i_{jkl} is computed using the connection coefficients
        (Christoffel symbols) Γ^i_{jk}. Since we don't have derivatives, we only compute:
        
        R^i_{jkl} = Γ^i_{mk} Γ^m_{jl} - Γ^i_{ml} Γ^m_{jk}
        
        Args:
            metric: Metric tensor of shape (batch_size, n, n)
            connection: Optional connection form of shape (batch_size, n, n, n)
            
        Returns:
            Riemann curvature tensor R^i_{jkl} of shape (batch_size, n, n, n, n)
        """
        batch_size = metric.shape[0]
        
        if connection is None:
            connection = self.compute_connection(self._to_tensor_type(metric, MetricTensor))
            
        # Initialize curvature tensor R^i_{jkl}
        curvature = torch.zeros(
            batch_size,
            self.manifold_dim,  # i index (contravariant)
            self.manifold_dim,  # j index (covariant)
            self.manifold_dim,  # k index (covariant)
            self.manifold_dim,  # l index (covariant)
            device=metric.device,
            dtype=metric.dtype
        )
        
        # Compute first term: Γ^i_{mk} Γ^m_{jl}
        # For each i,j,k,l we sum over m:
        # connection[..., i, m, k] * connection[..., m, j, l]
        term1 = torch.einsum('...imk,...mjl->...ijkl', connection, connection)
        
        # Compute second term: -Γ^i_{ml} Γ^m_{jk}
        # For each i,j,k,l we sum over m:
        # connection[..., i, m, l] * connection[..., m, j, k]
        term2 = torch.einsum('...iml,...mjk->...ijkl', connection, connection)
        
        # Combine terms: R^i_{jkl} = term1 - term2
        curvature = term1 - term2
        
        return self._to_tensor_type(curvature, CurvatureTensor)

    def compute_ricci_tensor(
        self,
        metric: Union[torch.Tensor, MetricTensor],
        points: Optional[Union[torch.Tensor, BatchTensor]] = None,
        connection: Optional[Union[torch.Tensor, ConnectionTensor]] = None
    ) -> Union[torch.Tensor, MetricTensor]:
        """Compute Ricci tensor.
        
        Args:
            metric: Metric tensor of shape (batch_size, n, n)
            points: Optional points tensor
            connection: Optional connection form of shape (batch_size, n, n, n)
            
        Returns:
            Ricci tensor of shape (batch_size, n, n)
        """
        if connection is None:
            connection = self.compute_connection(self._to_tensor_type(metric, MetricTensor), points)
            
        # Compute Riemann curvature tensor (5D: batch, i, j, k, l)
        riemann = self.compute_curvature(metric, connection)
        
        # Contract to get Ricci tensor
        # Ric_{jl} = R^i_{jil}
        # Contract i indices
        ricci = torch.einsum('...ijil->...jl', riemann)
        
        # Ensure symmetry
        ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
        
        return self._to_tensor_type(ricci, MetricTensor)

    def compute_mean_curvature(
        self,
        metric: Union[torch.Tensor, MetricTensor],
        points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mean curvature from metric tensor.
        
        Args:
            metric: Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
            points: Optional points tensor
            
        Returns:
            Mean curvature tensor of shape [batch_size, manifold_dim]
        """
        # Convert to tensor type if needed
        metric_tensor = self._to_tensor_type(metric, MetricTensor)
        
        # Compute Ricci tensor
        ricci = self.compute_ricci_tensor(metric_tensor, points)
        
        # Mean curvature is trace of Ricci tensor divided by manifold dimension
        mean_curvature = torch.diagonal(ricci, dim1=-2, dim2=-1) / self.manifold_dim
        
        return mean_curvature

    def detect_singularities(
        self,
        metric: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        threshold: float = 1e-6
    ) -> BaseSingularityInfo:
        """Detect flow singularities.
        
        Args:
            metric: Metric tensor
            points: Optional points tensor
            threshold: Detection threshold
            
        Returns:
            Singularity information
        """
        batch_size = metric.shape[0]
        
        # Add small regularization for numerical stability
        metric_reg = metric + torch.eye(
            self.manifold_dim,
            device=metric.device,
            dtype=metric.dtype
        ).unsqueeze(0) * 1e-8
        
        # Check metric determinant
        det = torch.linalg.det(metric_reg)
        
        # Compute condition number using SVD
        try:
            U, S, Vh = torch.linalg.svd(metric_reg)
            cond = S.max(dim1=-1)[0] / (S.min(dim1=-1)[0] + 1e-8)
        except:
            # If SVD fails, metric is likely singular
            cond = torch.ones(batch_size, device=metric.device) * float('inf')
        
        # Check eigenvalues
        try:
            eigenvals = torch.linalg.eigvals(metric_reg).real
            min_eigenval = torch.min(eigenvals, dim=-1)[0]
        except:
            # If eigendecomposition fails, assume singular
            min_eigenval = torch.zeros(batch_size, device=metric.device)
        
        # Find first singularity
        for i in range(batch_size):
            if (abs(det[i]) < threshold or  # Near-zero determinant
                cond[i] > 1.0/threshold or  # Poor conditioning
                min_eigenval[i] < threshold):  # Near-zero eigenvalue
                
                return BaseSingularityInfo(
                    index=i,
                    determinant=float(det[i].item()),
                    condition_number=float(cond[i].item()),
                    min_eigenvalue=float(min_eigenval[i].item()),
                    location=points[i] if points is not None else None,
                    curvature=self.compute_curvature(metric[i:i+1])[0]
                )
                    
        # If no singularity found, return first point as non-singular
        return BaseSingularityInfo(
            index=0,
            determinant=float(det[0].item()),
            condition_number=float(cond[0].item()),
            min_eigenvalue=float(min_eigenval[0].item()),
            location=points[0] if points is not None else None,
            curvature=self.compute_curvature(metric[0:1])[0]
        )

    def compute_ricci(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Ricci tensor.
        
        This method computes the full Ricci tensor, not just the scalar.
        """
        # First compute the metric
        metric = self.compute_metric(points)
        
        # Compute connection
        connection = self.compute_connection(metric, points)
        
        # Then compute the Ricci tensor using the parent class method
        return self.compute_ricci_tensor(metric, connection)