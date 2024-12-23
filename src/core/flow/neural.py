"""Neural Geometric Flow Implementation.

This module provides a neural network-specific implementation of geometric flows,
building on top of pattern formation dynamics and adding neural-specific features.
Implements the vertical integration between pattern, geometric, and quantum layers.
"""

from __future__ import annotations

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

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
from .protocol import FlowMetrics, QuantumFlowMetrics

# Local imports - quantum components
from ..quantum.neural_quantum_bridge import NeuralQuantumBridge
from ..quantum.types import QuantumState
from ...validation.quantum.state import QuantumStateValidationResult

# Local imports - dimension management
from ..common.dimensions import DimensionManager, DimensionConfig

# Enable torchtyping
patch_typeguard()

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
BatchTensor = TensorType["batch_size", "manifold_dim"]
MetricTensor = TensorType["batch_size", "manifold_dim", "manifold_dim"]
ConnectionTensor = TensorType["batch_size", "manifold_dim", "manifold_dim", "manifold_dim"]

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
            stability_threshold=stability_threshold
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
        self.connection_dim = manifold_dim * manifold_dim * manifold_dim
        
        # Initialize metric network
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=self.dtype, device=self.device)
        )
        
        # Initialize quantum bridge with proper configuration
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            dtype=self.dtype,
            device=self.device,
            manifold_type="hyperbolic",
            curvature=-1.0
        )
        
        # Initialize dimension manager for operadic transitions
        self.dim_manager = DimensionManager(
            DimensionConfig.from_test_config(test_config)
        )
        
        # Initialize neural networks with proper dimensioning
        self._init_fisher_rao_networks()
        self._init_quantum_networks()
        self._init_connection_networks()
        
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
        self.fisher_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
    def _init_quantum_networks(self):
        """Initialize networks for quantum corrections."""
        manifold_squared = self.manifold_dim * self.manifold_dim
        
        # Projection networks with proper dimensioning
        self.expectation_projection = nn.Sequential(
            nn.Linear(manifold_squared, self.projection_dim // 2, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.projection_dim // 2),
            nn.ReLU()
        )
        
        self.metric_projection = nn.Sequential(
            nn.Linear(manifold_squared, self.projection_dim // 2, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.projection_dim // 2),
            nn.ReLU()
        )
        
        # Correction network with residual connection and proper dimensioning
        self.quantum_correction_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )
        
    def _init_connection_networks(self):
        """Initialize networks for connection computation."""
        input_dim = self.manifold_dim * self.manifold_dim
        if hasattr(self, 'point_dim'):
            input_dim += self.point_dim
            
        # Connection projection with proper dimensioning
        self.connection_projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Connection network with proper dimensioning
        self.connection_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.connection_dim, dtype=self.dtype, device=self.device)
        )
        
        # Stability network with proper dimensioning
        stability_input_dim = self.manifold_dim + self.manifold_dim * self.manifold_dim + 1
        self.stability_net = nn.Sequential(
            nn.Linear(stability_input_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.connection_dim, dtype=self.dtype, device=self.device)
        )

    def _to_real(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to real tensor by taking real part."""
        if tensor.is_complex():
            return tensor.real
        return tensor

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
        
        # Use dimension manager for reshaping and projection
        quantum_state_flat = quantum_state.reshape(batch_size, -1)
        
        # Project quantum state with dimension validation
        quantum_state_proj = self.dim_manager.validate_and_project(
            quantum_state_flat,
            target_dim=self.hidden_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        # Apply correction network
        corrections = self.quantum_correction_net(quantum_state_proj)
        
        # Reshape output to metric tensor shape and convert to GeometricTensor
        metric_shape = corrections.reshape(batch_size, self.manifold_dim, self.manifold_dim)
        geometric_tensor = self.dim_manager.to_geometric_tensor(metric_shape)
        
        return self._to_tensor_type(geometric_tensor, MetricTensor)

    def compute_metric(self, points: BatchTensor) -> MetricTensor:
        """Compute Riemannian metric tensor at given points."""
        batch_size = points.shape[0]
        
        # Use dimension manager for validation and projection
        points_proj = self.dim_manager.validate_and_project(
            points,
            target_dim=self.manifold_dim,
            dtype=self.dtype,
            device=self.device
        )
        points_proj = self._to_tensor_type(points_proj, BatchTensor)
        
        # Pre-allocate output tensor
        metric = torch.empty(
            (batch_size, self.manifold_dim, self.manifold_dim),
            device=self.device,
            dtype=self.dtype
        )
        
        # Base metric computation
        base_metric = self.metric_net(points_proj)
        metric.copy_(self.dim_manager.reshape_to_metric(base_metric, batch_size))
        
        # Quantum corrections
        if self.phase_tracking_enabled:
            with torch.no_grad():
                # Project points to hidden dimension for quantum state preparation
                points_hidden = torch.zeros(
                    (batch_size, self.hidden_dim),
                    device=self.device,
                    dtype=self.dtype
                )
                points_hidden[..., :self.manifold_dim] = points_proj
                quantum_state = self.prepare_quantum_state(points_hidden, return_validation=False)
            
            corrections = self.compute_quantum_corrections(
                quantum_state, 
                self._to_tensor_type(metric, MetricTensor)
            )
            metric.add_(corrections, alpha=self.quantum_correction_strength)
            
        # Fisher-Rao contribution
        fisher_metric = self.compute_fisher_rao_metric(points_proj)
        metric.add_(fisher_metric, alpha=self.fisher_rao_weight)
        
        # Ensure metric properties
        metric.add_(metric.transpose(-2, -1).clone())
        metric.mul_(0.5)
        
        # Add stability term
        metric.diagonal(dim1=-2, dim2=-1).add_(self.stability_threshold)
        
        # Force symmetry
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Project onto positive definite cone
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        eigenvalues = torch.clamp(eigenvalues, min=self.stability_threshold)
        metric = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        
        return self._to_tensor_type(metric, MetricTensor)

    def flow_step(
        self,
        metric: Union[torch.Tensor, MetricTensor],
        ricci: Optional[Union[torch.Tensor, MetricTensor]] = None,
        timestep: float = 0.1
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
        
        # Apply neural weight space normalization efficiently
        with torch.no_grad():
            norm = torch.diagonal(new_metric, dim1=-2, dim2=-1).sum(-1)
            norm = torch.sqrt(norm).unsqueeze(-1).unsqueeze(-1).add_(1e-8)
            new_metric.div_(norm)
        
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
            with torch.no_grad():
                initial_state = self.prepare_quantum_state(
                    metric.view(-1, self.manifold_dim),
                    return_validation=False
                )
                
                if not isinstance(initial_state, tuple):
                    # Evolve quantum state
                    evolved_state = self.quantum_bridge.evolve_quantum_state(
                        initial_state,
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
        return new_metric, QuantumFlowMetrics(
            flow_magnitude=base_metrics.flow_magnitude,
            metric_determinant=base_metrics.metric_determinant,
            ricci_scalar=base_metrics.ricci_scalar,
            energy=base_metrics.energy,
            singularity=base_metrics.singularity,
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
        
        # Use dimension manager for validation
        points_proj = self.dim_manager.validate_and_project(
            points_tensor,
            target_dim=self.manifold_dim,
            dtype=self.dtype,
            device=self.device
        )
        points_proj = self._to_tensor_type(points_proj, BatchTensor)
        
        # Compute base connection
        network_output = self.connection_net(points_proj)
        
        # Reshape using dimension manager
        connection = self.dim_manager.reshape_to_connection(network_output, batch_size)
        
        # Add stability term
        if self.stability_term > 0:
            stability = self.compute_stability_term(points_proj, metric)
            connection = connection + stability * self.stability_term
            
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
        
        # Convert input to BatchTensor
        x_batch = self._to_tensor_type(x, BatchTensor)
        
        # 1. Compute geometric quantities
        metric = self.compute_metric(x_batch)
        connection = self.compute_connection(metric, x_batch)
        
        metrics.update({
            'metric_norm': torch.norm(metric),
            'connection_norm': torch.norm(connection)
        })
        
        # 2. Apply quantum corrections if enabled
        if self.quantum_weight > 0:
            # Prepare quantum state and compute corrections
            quantum_state = self.prepare_quantum_state(x_batch)
            quantum_correction = self.compute_quantum_corrections(quantum_state, metric)
            
            # Record quantum metrics
            metrics['quantum_correction_norm'] = torch.norm(quantum_correction)
            
            # Apply corrections to input
            batch_size, manifold_dim = x.shape
            correction_flat = self.dim_manager.reshape_to_flat(quantum_correction)
            correction_proj = self.dim_manager.project(
                correction_flat,
                target_dim=manifold_dim,
                dtype=self.dtype,
                device=self.device
            )
            x_batch = x_batch + self.quantum_weight * correction_proj
        
        # 3. Evolve the system
        x_evolved, flow_metrics = self.flow_step(metric)
        
        # Add flow metrics to output
        metrics.update(self._flow_metrics_to_dict(flow_metrics))
        
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
        
        # Use dimension manager for flattening
        metric_flat = self.dim_manager.reshape_to_flat(metric)
        
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
        """Initialize all neural networks."""
        # Metric network
        self.metric_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
        )

        # Connection network
        self.connection_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.hidden_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.connection_dim, dtype=self.dtype, device=self.device)
        )

        # Initialize other networks
        self._init_connection_networks()
        self._init_quantum_networks()

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
        
        # Compute metric components
        metric_components = self.metric_net(x)
        
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

    def compute_flow_vector(self, points: torch.Tensor, ricci: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # Compute metric at current points if not provided
        if metric is None:
            metric = self.compute_metric_tensor(points)
        
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

    def compute_flow(self, metric: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        """Compute the geometric flow for the given metric tensor.
        
        Args:
            metric: The metric tensor to compute flow for.
            time: The time parameter for flow evolution.
            
        Returns:
            The computed flow tensor.
        """
        # Compute Ricci tensor from metric
        ricci = self.compute_ricci_tensor(metric)
        
        # Get points from metric if needed
        if self.points is None:
            batch_size = metric.shape[0]
            points = torch.zeros(batch_size, self.manifold_dim, device=metric.device)
        else:
            points = self.points
        
        # Compute flow vector using Ricci tensor
        flow_vector = self.compute_flow_vector(points, ricci, metric)
        
        # Apply time scaling
        flow_vector = flow_vector * float(time) if time != 0.0 else flow_vector
        
        return flow_vector