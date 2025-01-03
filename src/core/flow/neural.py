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
        
        # Calculate number of parameters for lower triangular matrix
        n = manifold_dim
        self.n_metric_params = n * (n + 1) // 2  # Number of elements in lower triangular matrix
        
        # Initialize metric network with proper output dimension for manifold_dim x manifold_dim metric
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim, dtype=self.dtype, device=self.device)
        )
        
        # Initialize weights to ensure positive definiteness
        nn.init.orthogonal_(self.metric_net[0].weight)
        nn.init.zeros_(self.metric_net[0].bias)
        nn.init.orthogonal_(self.metric_net[-1].weight)
        nn.init.constant_(self.metric_net[-1].bias, 0.1)  # Small positive bias for stability
        
        # Initialize quantum bridge with proper configuration
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=max(manifold_dim, num_heads * 8),  # Ensure hidden_dim is divisible by num_heads
            num_heads=num_heads,
            dropout=dropout,
            dtype=self.dtype,
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
            nn.Linear(manifold_squared, self.manifold_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.manifold_dim),
            nn.ReLU()
        )
        
        self.metric_projection = nn.Sequential(
            nn.Linear(manifold_squared, self.manifold_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.manifold_dim),
            nn.ReLU()
        )
        
        # Correction network with residual connection and proper dimensioning
        self.quantum_correction_net = nn.Sequential(
            nn.Linear(self.manifold_dim, self.manifold_dim, dtype=self.dtype, device=self.device),
            nn.LayerNorm(self.manifold_dim),
            nn.ReLU(),
            nn.Linear(self.manifold_dim, self.manifold_dim * self.manifold_dim, dtype=self.dtype, device=self.device)
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
        
        # Project quantum state to manifold dimension
        quantum_state_flat = quantum_state.reshape(batch_size, -1)
        quantum_state_proj = quantum_state_flat[:, :self.manifold_dim]
        
        # Apply correction network
        corrections = self.quantum_correction_net(quantum_state_proj)
        
        # Reshape output to metric tensor shape and convert to GeometricTensor
        metric_shape = corrections.reshape(batch_size, self.manifold_dim, self.manifold_dim)
        geometric_tensor = self.dim_manager.to_geometric_tensor(metric_shape)
        
        return self._to_tensor_type(geometric_tensor, MetricTensor)

    def compute_metric(
        self,
        points: torch.Tensor,
        connection: Optional[torch.Tensor] = None
    ) -> MetricTensor:
        """Compute metric tensor at points.
        
        Args:
            points: Points tensor of shape [batch_size, manifold_dim]
            connection: Optional connection coefficients
            
        Returns:
            Metric tensor of shape [batch_size, manifold_dim, manifold_dim]
        """
        # Validate input dimensions
        batch_size = points.shape[0]
        self._validate_dimensions(
            points,
            [batch_size, self.manifold_dim],
            "points"
        )
        
        # Project points through metric network to get metric components
        metric_features = self.metric_net(points)
        
        # Reshape to metric tensor
        metric = metric_features.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Make metric symmetric
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Add initial regularization term
        eye = torch.eye(
            self.manifold_dim,
            device=points.device,
            dtype=points.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        metric = metric + 1e-3 * eye
        
        # Project onto positive definite cone with strong minimum eigenvalue
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        eigenvalues = torch.clamp(eigenvalues, min=1e-2)  # Increased minimum eigenvalue
        metric = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        
        # Add final regularization to ensure stability
        metric = metric + 1e-3 * eye
        
        # Normalize by determinant to ensure unit volume
        det = torch.linalg.det(metric).unsqueeze(-1).unsqueeze(-1)
        metric = metric / (det + 1e-8).pow(1/self.manifold_dim)
        
        # Final projection to ensure minimum eigenvalue after normalization
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)  # Ensure minimum eigenvalue after normalization
        metric = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        
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
        
        # Scale metric to preserve volume
        with torch.no_grad():
            # Add regularization to prevent ill-conditioning
            eye = torch.eye(
                self.manifold_dim,
                device=metric.device,
                dtype=metric.dtype
            ).unsqueeze(0).expand(batch_size, -1, -1)
            new_metric = new_metric + 1e-3 * eye  # Increased regularization
            
            # Compute scale factor with clamping to prevent extreme values
            det_before = torch.linalg.det(metric_tensor)
            det_after = torch.linalg.det(new_metric)
            scale = (det_before / (det_after + 1e-8)).pow(1/self.manifold_dim)
            scale = torch.clamp(scale, min=0.1, max=10.0)  # Prevent extreme scaling
            scale = scale.view(-1, 1, 1)  # Reshape for broadcasting
            new_metric = new_metric * scale
            
            # Ensure final metric is symmetric
            new_metric = 0.5 * (new_metric + new_metric.transpose(-2, -1))
            
            # Project onto positive definite cone with increased minimum eigenvalue
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(new_metric)
                eigenvalues = torch.clamp(eigenvalues, min=1e-2)  # Increased minimum eigenvalue
                new_metric = torch.matmul(
                    torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
                    eigenvectors.transpose(-2, -1)
                )
            except RuntimeError:
                # If eigendecomposition fails, add more regularization
                new_metric = new_metric + 1e-2 * eye
                eigenvalues, eigenvectors = torch.linalg.eigh(new_metric)
                eigenvalues = torch.clamp(eigenvalues, min=1e-2)
                new_metric = torch.matmul(
                    torch.matmul(eigenvectors, torch.diag_embed(eigenvalues)),
                    eigenvectors.transpose(-2, -1)
                )
            
            # Verify and fix volume preservation
            det_final = torch.linalg.det(new_metric)
            if not torch.allclose(det_before, det_final, rtol=1e-4):
                # Apply one more correction if needed
                scale = (det_before / (det_final + 1e-8)).pow(1/self.manifold_dim)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                scale = scale.view(-1, 1, 1)
                new_metric = new_metric * scale
                
                # Ensure final metric is symmetric
                new_metric = 0.5 * (new_metric + new_metric.transpose(-2, -1))
        
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
                # Project metric to hidden dimension before preparing quantum state
                metric_flat = new_metric.reshape(batch_size, -1)  # [batch_size, manifold_dim * manifold_dim]
                metric_padded = torch.zeros(batch_size, self.quantum_bridge.hidden_dim, device=metric.device, dtype=metric.dtype)
                metric_padded[:, :metric_flat.shape[1]] = metric_flat
                
                initial_state = self.prepare_quantum_state(
                    metric_padded,
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
        flow_magnitude = torch.tensor(base_metrics.flow_magnitude, device=device, dtype=dtype)
        flow_magnitude = flow_magnitude.expand(batch_size)

        metric_determinant = torch.tensor(base_metrics.metric_determinant, device=device, dtype=dtype)
        metric_determinant = metric_determinant.expand(batch_size)

        ricci_scalar = torch.tensor(base_metrics.ricci_scalar, device=device, dtype=dtype)
        ricci_scalar = ricci_scalar.expand(batch_size)

        energy = torch.tensor(base_metrics.energy, device=device, dtype=dtype)
        energy = energy.expand(batch_size)

        singularity = torch.tensor(base_metrics.singularity, device=device, dtype=dtype)
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
        points_proj = self._to_tensor_type(points_proj, BatchTensor)  # Convert back to BatchTensor
        
        # Compute base connection
        network_output = self.connection_net(points_proj)
        
        # Reshape using dimension manager
        connection = network_output.view(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
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
        """Compute curvature tensor.
        
        Args:
            metric: Metric tensor
            connection: Optional connection coefficients
            
        Returns:
            Curvature tensor
        """
        batch_size = metric.shape[0]
        
        if connection is None:
            connection = self.compute_connection(self._to_tensor_type(metric, MetricTensor))
        
        # Prepare input: [connection_flat, metric_flat]
        connection_flat = connection.reshape(batch_size, -1)
        metric_flat = metric.reshape(batch_size, -1)
        
        # Calculate expected dimensions
        connection_dim = self.manifold_dim * self.manifold_dim * self.manifold_dim
        metric_dim = self.manifold_dim * self.manifold_dim
        
        # Pad or truncate tensors to match expected dimensions
        if connection_flat.shape[-1] < connection_dim:
            padding = torch.zeros(batch_size, connection_dim - connection_flat.shape[-1], device=connection_flat.device)
            connection_flat = torch.cat([connection_flat, padding], dim=-1)
        else:
            connection_flat = connection_flat[:, :connection_dim]
        
        if metric_flat.shape[-1] < metric_dim:
            padding = torch.zeros(batch_size, metric_dim - metric_flat.shape[-1], device=metric_flat.device)
            metric_flat = torch.cat([metric_flat, padding], dim=-1)
        else:
            metric_flat = metric_flat[:, :metric_dim]
        
        # Concatenate tensors
        input_tensor = torch.cat([connection_flat, metric_flat], dim=-1)
        
        # Compute curvature components
        curvature_flat = self.curvature_net(input_tensor)
        curvature = curvature_flat.view(
            batch_size, self.manifold_dim, self.manifold_dim
        )
        
        # Ensure antisymmetry
        curvature = 0.5 * (curvature - curvature.transpose(-2, -1))
        
        return self._to_tensor_type(curvature, CurvatureTensor)

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
    ) -> BaseSingularityInfo[torch.Tensor]:
        """Detect flow singularities.
        
        Args:
            metric: Metric tensor
            points: Optional points tensor
            threshold: Detection threshold
            
        Returns:
            Singularity information
        """
        batch_size = metric.shape[0]
        
        # Reshape metric if needed
        if len(metric.shape) == 2:
            metric = metric.view(batch_size, self.manifold_dim, self.manifold_dim)
        
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
            cond = S.max(dim=1)[0] / (S.min(dim=1)[0] + 1e-8)
        except:
            # If SVD fails, metric is likely singular
            cond = torch.ones(batch_size, device=metric.device) * float('inf')
        
        # Check eigenvalues
        try:
            eigenvals = torch.linalg.eigvals(metric_reg).real
            min_eigenval = torch.min(eigenvals, dim=1)[0]
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
                    curvature=self.compute_curvature(metric[i:i+1])[0])
        
        # If no singularity found, return first point as non-singular
        return BaseSingularityInfo(
            index=0,
            determinant=float(det[0].item()),
            condition_number=float(cond[0].item()),
            min_eigenvalue=float(min_eigenval[0].item()),
            location=points[0] if points is not None else None,
            curvature=self.compute_curvature(metric[0:1])[0])

    def compute_ricci(self, points: BatchTensor) -> CurvatureTensor:
        """Compute Ricci tensor from points.
        
        Args:
            points: Points tensor of shape [batch_size, manifold_dim]
            
        Returns:
            Ricci tensor of shape [batch_size, manifold_dim, manifold_dim]
        """
        # Compute metric tensor first
        metric = self.compute_metric(points)
        
        # Compute connection coefficients
        connection = self.compute_connection(metric, points)
        
        batch_size = points.shape[0]
        n = self.manifold_dim
        
        # Initialize Ricci tensor
        ricci = torch.zeros(batch_size, n, n, device=points.device, dtype=points.dtype)
        
        # Compute Ricci tensor components
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        # R_{ij} = R^k_{ikj}
                        # where R^l_{ijk} = ∂_i Γ^l_{jk} - ∂_j Γ^l_{ik} + Γ^m_{jk}Γ^l_{im} - Γ^m_{ik}Γ^l_{jm}
                        
                        # First term: ∂_i Γ^l_{jk}
                        term1 = torch.autograd.grad(
                            connection[:, j, k, l],
                            points,
                            grad_outputs=torch.ones_like(connection[:, j, k, l]),
                            create_graph=True,
                            retain_graph=True
                        )[0][:, i]
                        
                        # Second term: -∂_j Γ^l_{ik}
                        term2 = -torch.autograd.grad(
                            connection[:, i, k, l],
                            points,
                            grad_outputs=torch.ones_like(connection[:, i, k, l]),
                            create_graph=True,
                            retain_graph=True
                        )[0][:, j]
                        
                        # Third term: Γ^m_{jk}Γ^l_{im}
                        term3 = torch.sum(connection[:, j, k, :] * connection[:, i, :, l], dim=-1)
                        
                        # Fourth term: -Γ^m_{ik}Γ^l_{jm}
                        term4 = -torch.sum(connection[:, i, k, :] * connection[:, j, :, l], dim=-1)
                        
                        # Sum all terms
                        ricci[:, i, j] += term1 + term2 + term3 + term4
        
        # Make Ricci tensor symmetric
        ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
        
        return self._to_tensor_type(ricci, CurvatureTensor)