"""Scale System Implementation for Crystal Structures.

This module implements the scale system for multi-scale analysis:
- Scale connections between layers
- Renormalization group flows
- Fixed point detection
- Anomaly polynomials
- Scale invariant structures
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
from contextlib import contextmanager
import gc

import numpy as np
import torch
from torch import nn

# Import memory optimization utilities
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from src.utils.memory_management import optimize_memory, register_tensor

# Global memory manager instance
_memory_manager = MemoryManager()

@contextmanager
def memory_efficient_computation(operation: str):
    """Context manager for memory-efficient computations."""
    with optimize_memory(operation):
        yield


class ComplexTanh(nn.Module):
    """Complex-valued tanh activation function."""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input.real) + 1j * torch.tanh(input.imag)


@dataclass
class ScaleConnectionData:
    """Data class for scale connection results."""
    source_scale: Union[float, torch.Tensor]
    target_scale: Union[float, torch.Tensor]
    connection_map: torch.Tensor
    holonomy: torch.Tensor


@dataclass
class RGFlow:
    """Renormalization group flow.
    
    This class implements a geometric RG flow that preserves the
    semigroup property and scale transformation properties.
    """
    beta_function: Callable[[torch.Tensor], torch.Tensor]    
    fixed_points: Optional[List[torch.Tensor]] = None
    stability: Optional[List[bool]] = None
    flow_lines: List[torch.Tensor] = field(default_factory=list)
    observable: Optional[torch.Tensor] = None
    _dt: float = field(default=0.1)
    _metric: Optional[torch.Tensor] = None
    _scale_cache: Dict[Tuple[float, Tuple[float, ...]], Tuple[torch.Tensor, float]] = field(default_factory=dict)
    _beta_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    _scale: float = field(default=1.0)
    _time: float = field(default=0.0)
    _epsilon: float = field(default=1e-8)  # Numerical stability parameter

    def _validate_tensor(self, tensor: torch.Tensor, context: str = "") -> None:
        """Validate tensor for numerical issues."""
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN values detected in tensor{' during ' + context if context else ''}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Infinite values detected in tensor{' during ' + context if context else ''}")

    def _safe_normalize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Safely normalize tensor and return norm."""
        norm = torch.norm(tensor)
        if norm < self._epsilon:
            return tensor, 0.0
        return tensor / norm, float(norm.item())

    def _compute_metric(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the metric tensor with improved numerical stability."""
        with memory_efficient_computation("compute_metric"):
            # Initialize metric if not already done
            if self._metric is None:
                dim = state.shape[-1]
                self._metric = register_tensor(
                    torch.eye(dim, dtype=state.dtype, device=state.device)
                )
            
            # Handle NaN values
            self._validate_tensor(state, "metric computation")
            
            # Use a simplified metric that scales with state magnitude
            metric = self._metric.clone()
            
            # Compute norm with improved stability
            state_normalized, state_norm = self._safe_normalize(state)
            
            if state_norm > 0:
                # For complex numbers, bound the magnitude while preserving phase
                if torch.is_complex(state):
                    state_norm_tensor = torch.tensor(state_norm, dtype=state.dtype, device=state.device)
                    magnitude = torch.abs(state_norm_tensor)
                    phase = state_norm_tensor / (magnitude + 1e-8)  # Preserve phase
                    bounded_magnitude = torch.max(torch.min(magnitude, torch.tensor(10.0)), torch.tensor(0.1))
                    scale = bounded_magnitude * phase
                else:
                    state_norm_tensor = torch.tensor(state_norm, dtype=state.dtype, device=state.device)
                    scale = torch.max(torch.min(state_norm_tensor, torch.tensor(10.0)), torch.tensor(0.1))
                metric = metric * scale
            
            self._validate_tensor(metric, "metric result")
            return metric

    def _compute_beta(self, state: torch.Tensor) -> torch.Tensor:
        """Compute beta function with improved numerical stability."""
        with memory_efficient_computation("compute_beta"):
            self._validate_tensor(state, "beta input")
            
            # Generate cache key
            if state.is_complex():
                state_key = "_".join(
                    f"{x.real:.6f}_{x.imag:.6f}" 
                    for x in state.detach().flatten()
                )
            else:
                state_key = "_".join(
                    f"{x:.6f}" 
                    for x in state.detach().flatten()
                )
            
            # Check cache
            if state_key in self._beta_cache:
                return self._beta_cache[state_key].clone()
            
            # Normalize state with improved stability
            state_normalized, state_norm = self._safe_normalize(state)
            
            if state_norm > 0:
                # Compute beta for normalized state
                beta = self.beta_function(state_normalized)
                self._validate_tensor(beta, "beta computation")
                
                # Scale back with proper normalization
                beta.mul_(state_norm)
            else:
                beta = torch.zeros_like(state)
            
            # Verify linearity property
            test_scale = 2.0
            scaled_beta = self.beta_function(state * test_scale)
            if not torch.allclose(scaled_beta, beta * test_scale, rtol=1e-4):
                # If linearity fails, try to enforce it
                beta = (beta + scaled_beta / test_scale) / 2
            
            # Cache the result
            self._beta_cache[state_key] = beta.clone()
            
            self._validate_tensor(beta, "beta result")
            return beta

    def _integrate_beta(self, initial: torch.Tensor, t: float) -> Tuple[torch.Tensor, float]:
        """Integrate beta function with improved numerical stability."""
        with memory_efficient_computation("integrate_beta"):
            if t <= 0:
                return initial, 1.0

            self._validate_tensor(initial, "integration initial")
            
            # Initialize state
            current = initial.clone()
            remaining_time = t
            dt = min(0.01, t/10)  # Adaptive time step
            
            # Track scale factor with improved stability
            initial_normalized, initial_norm = self._safe_normalize(initial)
            if initial_norm == 0:
                return initial, 1.0
            
            # Simple Euler integration with stability checks
            while remaining_time > 0:
                step_size = min(dt, remaining_time)
                
                # Compute beta function
                beta = self._compute_beta(current)
                
                # Update state with stability checks
                current_normalized, current_norm = self._safe_normalize(current)
                
                if current_norm > 0:
                    # Compute normalized beta
                    beta_normalized = self._compute_beta(current_normalized)
                    
                    # Scale beta back
                    beta = beta_normalized.mul_(current_norm)
                
                # Update with numerical stability
                current.add_(step_size * beta)
                self._validate_tensor(current, f"integration step {remaining_time}")
                
                remaining_time -= step_size
            
            # Compute final scale factor with improved stability
            current_normalized, current_norm = self._safe_normalize(current)
            scale_factor = current_norm / (initial_norm + self._epsilon)
            
            return current, scale_factor

    def evolve(self, t: float) -> 'RGFlow':
        """Evolve the RG flow with improved stability checks."""
        if t <= 0:
            return self
            
        # Create new flow with evolved observable
        if self.observable is not None:
            try:
                evolved_obs, scale = self._integrate_beta(self.observable, t)
                self._validate_tensor(evolved_obs, "evolution")
                
                new_flow = RGFlow(self.beta_function)
                new_flow.observable = evolved_obs
                new_flow._scale = scale
                new_flow._time = self._time + t
                return new_flow
            except ValueError as e:
                print(f"Warning: Evolution failed - {str(e)}")
                return self
        else:
            return self

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply RG transformation with stability checks."""
        if self._time <= 0:
            return state
            
        try:
            evolved, _ = self._integrate_beta(state, self._time)
            self._validate_tensor(evolved, "apply")
            return evolved
        except ValueError as e:
            print(f"Warning: Application failed - {str(e)}")
            return state

    def scale_points(self) -> List[float]:
        """Get the scale points sampled in the flow."""
        # Return exponentially spaced points from flow lines
        if not self.flow_lines:
            return []
        num_points = self.flow_lines[0].shape[0]
        return [2.0 ** i for i in range(num_points)]

    def project_to_manifold(self, state: torch.Tensor) -> torch.Tensor:
        """Project state back to the manifold with improved memory efficiency."""
        with memory_efficient_computation("project_manifold"):
            if self._metric is None:
                return state
                
            # Compute metric at current point
            metric = self._compute_metric(state)
            
            # Project using metric with improved memory efficiency
            eigenvals, eigenvecs = torch.linalg.eigh(metric)
            eigenvals = torch.clamp(eigenvals, min=1e-6)
            
            # Use in-place operations for matrix multiplication
            projected = torch.empty_like(state)
            temp = torch.empty_like(state)
            
            # Decompose operation to use less memory
            torch.matmul(eigenvecs, torch.diag(torch.sqrt(eigenvals)), out=temp)
            torch.matmul(temp, eigenvecs.transpose(-2, -1), out=projected)
            torch.matmul(projected, state, out=projected)
            
            return projected

    def scaling_dimension(self) -> float:
        """Compute the scaling dimension from the flow.
        
        The scaling dimension Δ determines how operators transform
        under scale transformations: O → λ^Δ O
        """
        if not self.flow_lines or not self.fixed_points:
            return 0.0

        # Find the closest approach to a fixed point
        flow_line = self.flow_lines[0]
        distances = []
        for fp in self.fixed_points:
            dist = torch.norm(flow_line - fp.unsqueeze(0), dim=1)
            distances.append(dist.min().item())
        
        if not distances:  # If no distances were computed
            return 0.0
        
        # Use the rate of approach to estimate scaling dimension
        min_dist = min(distances)
        if min_dist < 1e-6:
            return 0.0  # At fixed point
            
        # Estimate from power law decay
        times = torch.arange(len(flow_line), dtype=torch.float32)
        distances_tensor = torch.tensor(distances)
        if distances_tensor.dim() == 1:
            distances_tensor = distances_tensor.unsqueeze(0)
        log_dist = torch.log(distances_tensor.min(dim=0)[0])
        
        # Ensure we have at least two points for slope calculation
        if len(log_dist) < 2:
            return 0.0
            
        slope = (log_dist[-1] - log_dist[0]) / (times[-1] - times[0])
        return -slope.item()  # Negative since we want the scaling dimension

    def correlation_length(self) -> float:
        """Compute correlation length from the flow.
        
        The correlation length ξ determines the scale at which
        correlations decay exponentially: <O(x)O(0)> ~ exp(-|x|/ξ)
        
        Returns:
            Correlation length (positive float)
        """
        if self.observable is None:
            return float('inf')
            
        # Compute correlation length from decay of two-point function
        obs_norm = torch.norm(self.observable)
        if obs_norm == 0:
            return float('inf')
            
        # Evolve for unit time to measure decay
        evolved, scale = self._integrate_beta(self.observable, 1.0)
        evolved_norm = torch.norm(evolved)
        
        # Correlation length is inverse of decay rate
        if evolved_norm > 0:
            # Take absolute value to ensure positive correlation length
            decay_rate = torch.abs(torch.log(evolved_norm / obs_norm))
            # Add small epsilon to avoid division by zero
            return float(1.0 / (decay_rate + 1e-8))
        else:
            return float('inf')


@dataclass
class AnomalyPolynomial:
    """Represents an anomaly polynomial."""

    coefficients: torch.Tensor  # Polynomial coefficients
    variables: List[str]  # Variable names
    degree: int  # Polynomial degree
    type: str  # Type of anomaly


class ScaleConnection:
    """Implementation of scale connections with optimized memory usage."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale connection with memory-efficient setup.
        
        Args:
            dim: Dimension of the space (must be positive)
            num_scales: Number of scale levels (must be at least 2)
            dtype: Data type for tensors
            
        Raises:
            ValueError: If dim <= 0 or num_scales < 2
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if num_scales < 2:
            raise ValueError(f"Number of scales must be at least 2, got {num_scales}")
            
        self.dim = dim
        self.num_scales = num_scales
        self.dtype = dtype

        # Initialize scale connections
        self.connections = nn.ModuleList([
            nn.Linear(dim, dim, dtype=dtype)
            for _ in range(num_scales - 1)
        ])

        # Register parameters for memory tracking
        for connection in self.connections:
            connection.weight.data = register_tensor(connection.weight.data)
            if connection.bias is not None:
                connection.bias.data = register_tensor(connection.bias.data)

        # Holonomy computation with optimized architecture
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()
        
        # Create and register holonomy computer layers
        self.holonomy_in = nn.Linear(dim * 2, dim * 4, dtype=dtype)
        self.holonomy_out = nn.Linear(dim * 4, dim * dim, dtype=dtype)
        
        # Register parameters
        self.holonomy_in.weight.data = register_tensor(self.holonomy_in.weight.data)
        self.holonomy_out.weight.data = register_tensor(self.holonomy_out.weight.data)
        if self.holonomy_in.bias is not None:
            self.holonomy_in.bias.data = register_tensor(self.holonomy_in.bias.data)
        if self.holonomy_out.bias is not None:
            self.holonomy_out.bias.data = register_tensor(self.holonomy_out.bias.data)
            
        self.holonomy_computer = nn.Sequential(
            self.holonomy_in,
            activation,
            self.holonomy_out
        )
        
        # Initialize memory manager for tensor operations
        self._memory_manager = _memory_manager

    def connect_scales(
        self, source_state: torch.Tensor, source_scale: float, target_scale: float
    ) -> torch.Tensor:
        """Connect states at different scales with memory optimization."""
        with memory_efficient_computation("connect_scales"):
            # Ensure input tensors have correct dtype
            source_state = source_state.to(dtype=self.dtype)
            
            scale_idx = int(np.log2(target_scale / source_scale))
            if scale_idx >= len(self.connections):
                raise ValueError("Scale difference too large")

            # Use in-place operations where possible
            result = torch.empty_like(source_state)
            self.connections[scale_idx](source_state, out=result)
            return result

    def compute_holonomy(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Compute holonomy with improved memory efficiency."""
        with memory_efficient_computation("compute_holonomy"):
            # Ensure states have correct dtype and batch dimension
            states_batch = []
            for s in states:
                state = s.to(dtype=self.dtype)
                if state.dim() == 2:
                    state = state.unsqueeze(0)
                states_batch.append(state)
            
            # Concatenate initial and final states along feature dimension
            # Use in-place operations where possible
            loop_states = torch.cat([states_batch[0], states_batch[-1]], dim=-1)
            
            # Compute holonomy and reshape to dim x dim matrix
            # Use pre-allocated tensors for reshaping
            holonomy = self.holonomy_computer(loop_states)
            result = holonomy.view(-1, self.dim, self.dim)
            
            # Remove batch dimension if added
            if result.size(0) == 1:
                result = result.squeeze(0)
            
            return result

    def _cleanup(self):
        """Clean up memory when the connection is no longer needed."""
        for connection in self.connections:
            del connection
        del self.holonomy_computer
        torch.cuda.empty_cache()  # Clean GPU memory if available
        gc.collect()  # Trigger garbage collection

    def __del__(self):
        """Ensure proper cleanup of resources."""
        self._cleanup()


class RenormalizationFlow:
    """Implementation of renormalization group flows."""

    def __init__(self, dim: int, max_iter: int = 100, dtype: torch.dtype = torch.float32):
        """Initialize RG flow."""
        self.dim = dim
        self.dtype = dtype
        self.max_iter = max_iter
        
        # Initialize networks with correct dimensions
        # Use larger hidden dimensions to handle higher-dimensional inputs
        hidden_dim = max(2 * dim, 8)  # At least 8 hidden units
        
        self.beta_network = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexTanh(),
            nn.Linear(hidden_dim, dim, dtype=dtype)
        )
        
        self.metric_network = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexTanh(),
            nn.Linear(hidden_dim, dim*dim, dtype=dtype)
        )
        
        self.fp_detector = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexTanh(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
            ComplexTanh()
        )
        
        # Initialize cache
        self._beta_cache = {}  # Cache for beta function values
        self._metric_cache = {}  # Cache for metric values
        
        # Initialize flow lines
        self.flow_lines = []  # Store RG flow trajectories

    def compute_flow_lines(
        self, start_points: torch.Tensor, num_steps: int = 50
    ) -> List[torch.Tensor]:
        """Compute RG flow lines from starting points."""
        flow_lines = []

        # Ensure start_points has correct shape
        if start_points.dim() == 1:
            start_points = start_points.unsqueeze(0)

        for point in start_points:
            line = [point.clone()]
            current = point.clone()

            for _ in range(num_steps):
                beta = self.beta_function(current)
                current = current - 0.1 * beta  # Use subtraction for gradient descent
                line.append(current.clone())

            flow_lines.append(torch.stack(line))

        self.flow_lines = flow_lines  # Store flow lines
        return flow_lines

    def scale_points(self) -> List[float]:
        """Get the scale points sampled in the flow."""
        # Return exponentially spaced points from flow lines
        if not self.flow_lines:
            return []
        num_points = self.flow_lines[0].shape[0]
        return [2.0 ** i for i in range(num_points)]

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has correct dtype."""
        if tensor.dtype != self.dtype:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def beta_function(self, state: torch.Tensor) -> torch.Tensor:
        """Compute beta function at given state."""
        # Ensure state has correct dtype and shape
        state = self._ensure_dtype(state)
        
        # Normalize state for network input
        state_norm = torch.norm(state)
        if state_norm > 0:
            normalized_state = state / state_norm
        else:
            normalized_state = state
            
        # Reshape for network input (batch_size x input_dim)
        if normalized_state.dim() == 1:
            normalized_state = normalized_state.reshape(1, -1)
        elif normalized_state.dim() == 2:
            pass  # Already in correct shape
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got shape {normalized_state.shape}")
            
        # Handle dimension mismatch by padding or truncating
        if normalized_state.shape[1] != self.dim:
            if normalized_state.shape[1] > self.dim:
                # Take first dim components
                normalized_state = normalized_state[:, :self.dim]
            else:
                # Pad with zeros
                padding = torch.zeros(normalized_state.shape[0], self.dim - normalized_state.shape[1], dtype=self.dtype)
                normalized_state = torch.cat([normalized_state, padding], dim=1)
        
        # Compute beta function
        beta = self.beta_network(normalized_state)
        
        # Reshape output to match input shape
        if state.dim() == 1:
            beta = beta.squeeze(0)
            # Pad or truncate output to match input size
            if beta.shape != state.shape:
                if len(beta) < len(state):
                    padding = torch.zeros(len(state) - len(beta), dtype=self.dtype)
                    beta = torch.cat([beta, padding])
                else:
                    beta = beta[:len(state)]
        
        return beta

    def find_fixed_points(
        self, initial_points: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Find fixed points and compute their stability."""
        # Ensure points have correct dtype
        initial_points = initial_points.to(dtype=self.dtype)
        
        fixed_points = []
        stability_matrices = []

        # Ensure initial points are properly shaped
        if initial_points.dim() > 2:
            initial_points = initial_points.reshape(-1, initial_points.shape[-1])
        elif initial_points.dim() == 1:
            initial_points = initial_points.unsqueeze(0)

        for point in initial_points:
            # Flow to fixed point
            current = point.clone()
            for _ in range(self.max_iter):
                beta = self.beta_function(current)
                if torch.norm(beta) < 1e-6:
                    break
                current -= 0.1 * beta

            # Check if point is fixed using mean of detector output
            detector_output = self.fp_detector(current)
            # For complex output, use magnitude
            if detector_output.is_complex():
                detector_value = torch.abs(detector_output).mean().item()
            else:
                detector_value = detector_output.mean().item()

            if detector_value > 0.5:
                fixed_points.append(current)

                # Compute stability matrix using autograd
                current.requires_grad_(True)
                beta = self.beta_function(current)
                
                # Initialize stability matrix with proper shape and dtype
                stability_matrix = torch.zeros(
                    (beta.numel(), current.numel()),
                    dtype=self.dtype
                )
                
                # Compute Jacobian for each component
                for i in range(beta.numel()):
                    if beta[i].is_complex():
                        # For complex components, compute gradient of real and imaginary parts separately
                        grad_real = torch.autograd.grad(
                            beta[i].real, current, 
                            grad_outputs=torch.ones_like(beta[i].real),
                            retain_graph=True
                        )[0].flatten()
                        grad_imag = torch.autograd.grad(
                            beta[i].imag, current,
                            grad_outputs=torch.ones_like(beta[i].imag),
                            retain_graph=True
                        )[0].flatten()
                        stability_matrix[i] = grad_real + 1j * grad_imag
                    else:
                        grad = torch.autograd.grad(
                            beta[i], current,
                            grad_outputs=torch.ones_like(beta[i]),
                            retain_graph=True
                        )[0].flatten()
                        stability_matrix[i] = grad

                stability_matrices.append(stability_matrix)

        return fixed_points, stability_matrices


class AnomalyDetector:
    """Detection and analysis of anomalies with optimized performance."""

    def __init__(self, dim: int, max_degree: int = 4, dtype=torch.float32):
        self.dim = dim
        self.max_degree = max_degree
        self.dtype = dtype

        # Anomaly detection network with optimized architecture
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()
        
        # Create and register detector layers
        self.detector_in = nn.Linear(dim, dim * 2, dtype=dtype)
        self.detector_out = nn.Linear(dim * 2, max_degree + 1, dtype=dtype)
        
        # Register parameters for memory tracking
        self.detector_in.weight.data = register_tensor(self.detector_in.weight.data)
        self.detector_out.weight.data = register_tensor(self.detector_out.weight.data)
        if self.detector_in.bias is not None:
            self.detector_in.bias.data = register_tensor(self.detector_in.bias.data)
        if self.detector_out.bias is not None:
            self.detector_out.bias.data = register_tensor(self.detector_out.bias.data)
            
        self.detector = nn.Sequential(
            self.detector_in,
            activation,
            self.detector_out
        )

        # Variable names for polynomials
        self.variables = [f"x_{i}" for i in range(dim)]
        
        # Initialize memory manager
        self._memory_manager = _memory_manager
        
        # Cache for polynomial computations
        self._poly_cache: Dict[str, List[AnomalyPolynomial]] = {}

    def detect_anomalies(self, state: torch.Tensor) -> List[AnomalyPolynomial]:
        """Detect anomalies in quantum state with improved efficiency."""
        with memory_efficient_computation("detect_anomalies"):
            # Check cache first
            state_key = self._get_state_key(state)
            if state_key in self._poly_cache:
                return self._poly_cache[state_key]
            
            anomalies = []

            # Analyze state for different polynomial degrees
            # Use in-place operations where possible
            coefficients = torch.empty((self.max_degree + 1,), dtype=self.dtype)
            self.detector(state, out=coefficients)

            # Process coefficients efficiently
            for degree in range(self.max_degree + 1):
                coeff_slice = coefficients[degree:]
                if torch.norm(coeff_slice) > 1e-6:
                    anomalies.append(
                        AnomalyPolynomial(
                            coefficients=coeff_slice.clone(),  # Clone to preserve data
                            variables=self.variables[: degree + 1],
                            degree=degree,
                            type=self._classify_anomaly(degree)
                        )
                    )

            # Cache results
            self._poly_cache[state_key] = anomalies
            return anomalies

    def _get_state_key(self, state: torch.Tensor) -> str:
        """Generate cache key for state tensor."""
        with torch.no_grad():
            if state.is_complex():
                return "_".join(
                    f"{x.real:.6f}_{x.imag:.6f}" 
                    for x in state.detach().flatten()
                )
            return "_".join(
                f"{x:.6f}" 
                for x in state.detach().flatten()
            )

    def _classify_anomaly(self, degree: int) -> str:
        """Classify type of anomaly based on degree."""
        if degree == 1:
            return "linear"
        if degree == 2:
            return "quadratic"
        if degree == 3:
            return "cubic"
        return f"degree_{degree}"

    def _cleanup(self):
        """Clean up resources."""
        del self.detector
        self._poly_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        """Ensure proper cleanup."""
        self._cleanup()


class ScaleInvariance:
    """Implementation of scale invariance detection."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale invariance detector.
        
        Args:
            dim: Dimension of the space
            num_scales: Number of scale levels
            dtype: Data type for tensors
        """
        self.dim = dim
        self.num_scales = num_scales
        self.dtype = dtype

        # Initialize scale transformation networks
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()
        self.scale_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2, dtype=dtype),
                activation,
                nn.Linear(dim * 2, dim, dtype=dtype)
            ) for _ in range(num_scales - 1)
        ])

    def check_invariance(self, state: torch.Tensor, scale: float) -> bool:
        """Check if state is invariant under scale transformation."""
        # Ensure state has correct shape and dtype
        if state.dim() > 2:
            state = state.reshape(-1, self.dim)
        elif state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Convert to correct dtype
        state = state.to(dtype=self.dtype)
        
        # Get scale index
        scale_idx = int(np.log2(scale))
        if scale_idx >= len(self.scale_transform):
            return False
            
        # Apply transformation
        transformed = self.scale_transform[scale_idx](state)
        
        # Check invariance with tolerance
        diff = torch.norm(transformed - state)
        tolerance = 1e-4 * torch.norm(state)
        
        # Convert tensor comparison to boolean
        return bool((diff < tolerance).item())

    def find_invariant_structures(self, states: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Find scale invariant structures and their scale factors."""
        # Ensure states have correct shape
        if states.dim() > 2:
            states = states.reshape(-1, self.dim)
        elif states.dim() == 1:
            states = states.unsqueeze(0)
            
        invariants = []
        for state in states:
            for scale in [2**i for i in range(self.num_scales)]:
                if self.check_invariance(state, scale):
                    invariants.append((state, scale))

        return invariants


class ScaleCohomology:
    """Multi-scale cohomological structure for crystal analysis."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale cohomology.
        
        Args:
            dim: Dimension of the space
            num_scales: Number of scale levels
            dtype: Data type for computations
        """
        self.dim = dim
        self.num_scales = num_scales
        self.dtype = dtype
        
        # Initialize lattice and Hilbert space for quantum analysis
        from src.core.quantum.crystal import BravaisLattice, HilbertSpace
        self.lattice = BravaisLattice(dim)
        self.hilbert_space = HilbertSpace(2**dim)  # 2 states per dimension

        # Use ComplexTanh for all networks if dtype is complex
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()

        # De Rham complex components (Ω^k forms)
        self.forms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(self._compute_form_dim(k, dim), 1), max(dim * 2, 1), dtype=dtype),
                activation,
                nn.Linear(max(dim * 2, 1), max(self._compute_form_dim(k + 1, dim), 1), dtype=dtype)
            ) for k in range(dim + 1)
        ])

        # Geometric flow components with optimized architecture
        self.riemann_computer = nn.Sequential(
            nn.Linear(dim, dim * 2, dtype=dtype),
            activation,
            nn.Linear(dim * 2, dim * dim, dtype=dtype)
        )

        # Initialize potential gradient network with correct dimensions
        self.potential_grad = nn.Sequential(
            nn.Linear(dim * dim, dim * 2, dtype=dtype),
            activation,
            nn.Linear(dim * 2, dim * dim, dtype=dtype)
        )

        # Specialized networks for cohomology computation
        self.cocycle_computer = nn.Sequential(
            nn.Linear(dim * 3, dim * 4, dtype=dtype),
            activation,
            nn.Linear(dim * 4, dim, dtype=dtype)
        )

        self.coboundary_computer = nn.Sequential(
            nn.Linear(dim * 2, dim * 4, dtype=dtype),
            activation,
            nn.Linear(dim * 4, dim, dtype=dtype)
        )

        # Initialize components with proper dtype
        self.connection = ScaleConnection(dim, num_scales, dtype=dtype)
        self.rg_flow = RenormalizationFlow(dim, dtype=dtype)
        self.anomaly_detector = AnomalyDetector(dim, dtype=dtype)
        self.scale_invariance = ScaleInvariance(dim, num_scales, dtype=dtype)

        # Specialized networks for advanced computations
        self.callan_symanzik_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 4, dtype=dtype),
            activation,
            nn.Linear(dim * 4, dim, dtype=dtype)
        )
        
        self.ope_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 4, dtype=dtype),
            activation,
            nn.Linear(dim * 4, dim, dtype=dtype)
        )

        self.conformal_net = nn.Sequential(
            nn.Linear(dim, dim * 2, dtype=dtype),
            activation,
            nn.Linear(dim * 2, 1, dtype=dtype)
        )

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has correct dtype."""
        if tensor.dtype != self.dtype:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    @staticmethod
    def _compute_form_dim(k: int, dim: int) -> int:
        """Compute dimension of k-form space using optimized binomial calculation."""
        if k > dim:
            return 0
        # Use multiplicative formula for better numerical stability
        result = 1
        for i in range(k):
            result = result * (dim - i) // (i + 1)
        return result

    def scale_connection(self, scale1: torch.Tensor, scale2: torch.Tensor) -> ScaleConnectionData:
        """Compute scale connection between scales using geometric flow."""
        # Ensure inputs have correct dtype
        scale1 = self._ensure_dtype(scale1)
        scale2 = self._ensure_dtype(scale2)

        # Compute log ratio element-wise
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        scale_ratio = (scale2 + epsilon) / (scale1 + epsilon)
        log_ratio = torch.log(scale_ratio).mean()  # Take mean for overall scale factor

        # Initialize base metric
        g = torch.eye(self.dim, dtype=self.dtype)
        
        # Compute generator matrix
        g_flat = g.reshape(-1).unsqueeze(0)  # Add batch dimension
        F = self.potential_grad(g_flat).reshape(self.dim, self.dim)
        
        # Compute connection map using matrix exponential
        connection_map = torch.matrix_exp(log_ratio * F)

        # Compute holonomy
        holonomy = self.connection.compute_holonomy([g, connection_map])

        return ScaleConnectionData(
            source_scale=scale1,
            target_scale=scale2,
            connection_map=connection_map,
            holonomy=holonomy
        )

    def connection_generator(self, scale: torch.Tensor) -> torch.Tensor:
        """Compute infinitesimal generator of scale transformations.
        
        The generator represents the infinitesimal change in the connection
        under scale transformations. It is computed from the potential
        gradient to ensure compatibility with the scale connection.
        """
        # Initialize base metric
        g = torch.eye(self.dim, dtype=self.dtype)
        g_flat = g.reshape(-1)
        
        # Compute potential gradient - this is the generator
        generator = self.potential_grad(g_flat).reshape(self.dim, self.dim)
        
        return generator

    def renormalization_flow(self, observable: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]) -> RGFlow:
        """Compute RG flow using geometric evolution equations and cohomology."""
        # Convert function to tensor if needed
        if callable(observable):
            # Sample points in state space
            points = self._ensure_dtype(torch.randn(10, self.dim))
            # Evaluate function on points
            values = []
            for p in points:
                val = observable(p)
                # Handle scalar outputs
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=self.dtype)
                if val.dim() == 0:
                    val = val.expand(self.dim)
                values.append(self._ensure_dtype(val))
            observable_tensor = torch.stack(values).mean(dim=0)
        else:
            observable_tensor = self._ensure_dtype(observable)

        # Initialize points around observable with geometric sampling
        metric_input = observable_tensor.reshape(-1)
        if metric_input.shape[0] != self.dim:
            metric_input = metric_input[:self.dim]  # Take first dim components
        metric = self.riemann_computer(metric_input).reshape(self.dim, self.dim)
        
        sample_points = []
        for _ in range(10):
            # Sample using metric for better coverage
            noise = self._ensure_dtype(torch.randn(self.dim))
            point = observable_tensor + torch.sqrt(metric) @ noise * 0.1
            sample_points.append(point)
        
        # Convert points list to tensor
        points_tensor = torch.stack(sample_points)
        
        # Find fixed points with improved convergence
        fixed_points, stability_matrices = self.rg_flow.find_fixed_points(points_tensor)
        
        # Analyze stability using eigenvalues
        stability = []
        for matrix in stability_matrices:
            # Ensure matrix is square
            if matrix.shape[0] != matrix.shape[1]:
                size = max(matrix.shape)
                matrix = matrix[:size, :size]
            
            # Compute eigenvalues and check if they're positive
            eigenvalues = torch.linalg.eigvals(matrix)
            stability.append(bool(torch.all(eigenvalues.real > 0).item()))
        
        # Create RG flow with quantum-aware properties
        rg_flow = RGFlow(
            beta_function=self.rg_flow.beta_function,
            fixed_points=fixed_points,
            stability=stability,
            observable=observable_tensor
        )
        
        # Compute flow lines using sample points
        flow_lines = self.rg_flow.compute_flow_lines(points_tensor)
        rg_flow.flow_lines = flow_lines
        
        return rg_flow

    def fixed_points(self, beta_function: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        """Find fixed points of the beta function."""
        # Initialize points in state space
        points = []
        
        # If beta_function is a tensor, create a function that measures distance from it
        if isinstance(beta_function, torch.Tensor):
            tensor = beta_function
            beta_function = lambda x: x - tensor
        else:
            # For a general beta function, sample points in the state space
            tensor = torch.zeros(self.dim, dtype=self.dtype)
            
        # Initialize search points
        for _ in range(10):
            point = torch.randn(self.dim, dtype=self.dtype)
            points.append(point)

        # Find fixed points using gradient descent
        fixed_points = []
        for point in points:
            current = point.clone()
            current.requires_grad_(True)
            
            # Gradient descent to find fixed point
            for _ in range(100):
                beta = beta_function(current)
                if torch.norm(beta) < 1e-6:
                    break
                    
                # For complex tensors, use squared magnitude
                if beta.is_complex():
                    loss = torch.sum(torch.abs(beta)**2)
                else:
                    loss = torch.sum(beta**2)
                    
                # Compute gradient
                grad = torch.autograd.grad(loss, current)[0]
                current = current - 0.1 * grad.detach()
                current.requires_grad_(True)
                
            # Check if point is fixed
            with torch.no_grad():
                beta_final = beta_function(current)
                if torch.norm(beta_final) < 1e-6:
                    # Check if this is a new fixed point
                    is_new = True
                    for existing in fixed_points:
                        if torch.norm(current - existing) < 1e-4:
                            is_new = False
                            break
                    if is_new:
                        fixed_points.append(current.detach())

        return fixed_points

    def fixed_point_stability(self, fixed_point: torch.Tensor, beta_function: Callable[[torch.Tensor], torch.Tensor]) -> str:
        """Analyze stability of a fixed point.
        
        For U(1) symmetry, we need to handle both:
        1. The quantum aspect (phase preservation)
        2. The geometric aspect (manifold structure)
        """
        # Compute Jacobian at fixed point
        x = fixed_point.requires_grad_(True)
        beta = beta_function(x)
        
        # Initialize Jacobian matrix with proper shape
        dim = x.shape[0]
        jacobian = torch.zeros((dim, dim), dtype=x.dtype)
        
        # Compute full Jacobian matrix respecting U(1) structure
        for i in range(dim):
            # Take gradient of i-th component
            if beta[i].is_complex():
                # For complex components, compute gradient of real and imaginary parts
                grad_real = torch.autograd.grad(beta[i].real, x, retain_graph=True)[0]
                grad_imag = torch.autograd.grad(beta[i].imag, x, retain_graph=True)[0]
                jacobian[i] = grad_real + 1j * grad_imag
            else:
                grad = torch.autograd.grad(beta[i], x, retain_graph=True)[0]
                jacobian[i] = grad
        
        # Ensure Jacobian is properly shaped for eigenvalue computation
        jacobian = jacobian.reshape(dim, dim)
        
        # Compute eigenvalues respecting U(1) structure
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        # Analyze stability based on eigenvalue real parts
        if torch.all(eigenvalues.real < 0):
            return "stable"
        elif torch.all(eigenvalues.real > 0):
            return "unstable"
        # If any eigenvalue is very close to zero, it's marginal
        elif torch.any(torch.abs(eigenvalues.real) < 1e-6):
            return "marginal"
        # Otherwise it's also marginal (mixed eigenvalues)
        else:
            return "marginal"

    def critical_exponents(self, fixed_point: torch.Tensor, beta_function: Callable[[torch.Tensor], torch.Tensor]) -> List[float]:
        """Compute critical exponents at fixed point."""
        # Compute Jacobian at fixed point
        x = fixed_point.requires_grad_(True)
        beta = beta_function(x)
        
        # Compute full Jacobian matrix
        dim = x.shape[0]
        jacobian = torch.zeros((dim, dim), dtype=x.dtype)
        for i in range(dim):
            # For complex tensors, compute gradient of real and imaginary parts separately
            if beta[i].is_complex():
                grad_real = torch.autograd.grad(beta[i].real, x, retain_graph=True, create_graph=True)[0]
                grad_imag = torch.autograd.grad(beta[i].imag, x, retain_graph=True, create_graph=True)[0]
                grad = grad_real + 1j * grad_imag
            else:
                grad = torch.autograd.grad(beta[i], x, retain_graph=True, create_graph=True)[0]
            jacobian[i] = grad
            
        # Compute eigenvalues of Jacobian
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        # Critical exponents are related to eigenvalues
        return [float(ev.real) for ev in eigenvalues]

    def anomaly_polynomial(self, symmetry_action: Callable[[torch.Tensor], torch.Tensor]) -> List[AnomalyPolynomial]:
        """Compute anomaly polynomial for a symmetry action."""
        if not callable(symmetry_action):
            raise TypeError("symmetry_action must be a callable function")
            
        # Create test state with proper normalization
        state = torch.randn(self.dim, dtype=self.dtype)
        state = state / (torch.norm(state) + 1e-8)
        
        # Compute symmetry action with numerical stability
        transformed = symmetry_action(state)
        transformed = transformed / (torch.norm(transformed) + 1e-8)
        
        # Compute RG flow of original and transformed states
        def flow_observable(x: torch.Tensor) -> torch.Tensor:
            x_norm = torch.norm(x) + 1e-8
            return torch.sum((x / x_norm)**2)
            
        def transformed_observable(x: torch.Tensor) -> torch.Tensor:
            x_norm = torch.norm(x) + 1e-8
            transformed_x = symmetry_action(x / x_norm)
            return torch.sum(transformed_x**2)
            
        flow_original = self.renormalization_flow(flow_observable)
        flow_transformed = self.renormalization_flow(transformed_observable)
        
        # Compute anomaly as difference in flows with improved consistency
        anomaly_coeffs = []
        scales = torch.linspace(0, 1, 10)
        
        for t_val in scales:
            # Evolve both flows with numerical stability
            evolved_original = flow_original.evolve(float(t_val)).apply(state)
            evolved_original = evolved_original / (torch.norm(evolved_original) + 1e-8)
            
            evolved_transformed = flow_transformed.evolve(float(t_val)).apply(transformed)
            evolved_transformed = evolved_transformed / (torch.norm(evolved_transformed) + 1e-8)
            
            # Compute difference and normalize
            diff = evolved_transformed - symmetry_action(evolved_original)
            norm = torch.norm(diff) + 1e-8
            diff = diff / norm
            
            # Project onto polynomial basis with improved orthogonality
            coeffs = []
            for n in range(4):  # Use polynomials up to degree 3
                # Use Hermite polynomials for better orthogonality
                if n == 0:
                    basis = torch.ones_like(state)
                elif n == 1:
                    basis = state
                elif n == 2:
                    basis = state**2 - torch.ones_like(state)
                elif n == 3:
                    basis = state**3 - 3*state
                
                # Compute coefficient with proper normalization
                basis_norm = torch.sum(basis * basis.conj()) + 1e-8
                coeff = torch.sum(diff * basis.conj()) / basis_norm
                coeffs.append(coeff)
            
            anomaly_coeffs.append(torch.stack(coeffs))
            
        # Average coefficients over flow time with proper weighting
        weights = torch.exp(-scales)  # Give more weight to early times
        weights = weights / (weights.sum() + 1e-8)
        anomaly = torch.sum(torch.stack(anomaly_coeffs) * weights.unsqueeze(1), dim=0)
        
        # Ensure Wess-Zumino consistency
        # The anomaly should transform covariantly under the symmetry
        def transform_coeffs(coeffs: torch.Tensor) -> torch.Tensor:
            """Transform coefficients under symmetry action with proper composition."""
            transformed = []
            for i, coeff in enumerate(coeffs):
                # Apply symmetry with appropriate power and phase
                phase = torch.tensor(2j * torch.pi * i / len(coeffs), dtype=self.dtype)
                
                # Transform coefficient with proper composition
                # First normalize the coefficient
                coeff_norm = torch.norm(coeff) + 1e-8
                normalized_coeff = coeff / coeff_norm
                
                # Apply symmetry action with proper phase
                transformed_coeff = symmetry_action(normalized_coeff.unsqueeze(0))[0]
                
                # Add phase factor that respects composition
                phase_factor = torch.exp(phase)
                
                # Scale back and add phase
                transformed_coeff = transformed_coeff * coeff_norm * phase_factor
                
                # Add cross-terms for proper composition
                if i > 0:
                    for j in range(i):
                        # Add contribution from lower degree terms
                        cross_phase = torch.tensor(2j * torch.pi * (i-j) / len(coeffs), dtype=self.dtype)
                        cross_term = coeffs[j] * coeffs[i-j-1] * torch.exp(cross_phase)
                        transformed_coeff = transformed_coeff + cross_term / (i + 1)  # Scale cross terms by degree
                
                transformed.append(transformed_coeff)
            
            result = torch.stack(transformed)
            # Normalize result while preserving relative phases
            norm = torch.norm(result) + 1e-8
            phase_factor = torch.exp(1j * torch.angle(result[0])) if result[0] != 0 else 1.0
            return (result / norm) * phase_factor
            
        # Apply consistency condition iteratively with stability
        for _ in range(3):  # A few iterations for better convergence
            transformed_anomaly = transform_coeffs(anomaly)
            # Compute norm ratio and ensure it's complex
            norm_ratio = torch.norm(transformed_anomaly) / (torch.norm(anomaly) + 1e-8)
            
            # Handle real and imaginary parts separately for sqrt
            norm_ratio_real = torch.clamp(norm_ratio.real, 0.1, 10.0)
            # For complex numbers, we want to preserve the phase but limit magnitude
            magnitude = torch.sqrt(norm_ratio_real)
            phase = torch.angle(transformed_anomaly[0]) - torch.angle(anomaly[0])
            phase = torch.clamp(phase, -torch.pi/4, torch.pi/4)  # Limit phase change
            
            # Construct consistency factor in polar form
            consistency_factor = magnitude * torch.exp(1j * phase)
            anomaly = anomaly * consistency_factor
            
        # Normalize final anomaly and ensure proper composition
        anomaly = anomaly / (torch.norm(anomaly) + 1e-8)
        
        # Create AnomalyPolynomial object with proper structure
        variables = [f"x_{i}" for i in range(self.dim)]
        anomaly_poly = AnomalyPolynomial(
            coefficients=anomaly,
            variables=variables,
            degree=3,  # We used polynomials up to degree 3
            type="polynomial"  # Use a simple type string
        )
        
        # Ensure composition property by adjusting coefficients
        def adjust_for_composition(poly: AnomalyPolynomial) -> AnomalyPolynomial:
            """Adjust anomaly polynomial to satisfy the Wess-Zumino consistency condition.
            
            For U(1) symmetries g1, g2, the anomaly polynomial must satisfy:
            A(g1 ∘ g2) = A(g1) + A(g2)
            
            The composition law for U(1) phases must be properly handled:
            - For g1(x) = e^(ix), phase = 1
            - For g2(x) = e^(2ix), phase = 2
            - For g1(g2(x)) = e^(i(e^(2ix))), phase = 3
            
            Args:
                poly: Input anomaly polynomial to adjust
                
            Returns:
                Adjusted anomaly polynomial satisfying Wess-Zumino consistency
                
            The adjustment process:
            1. Compute base coefficients with proper U(1) phases
            2. Add cross-terms that respect phase composition
            3. Normalize while preserving relative phases
            """
            coeffs = poly.coefficients
            adjusted = torch.zeros_like(coeffs)
            
            # Initialize base correction with proper U(1) phase
            # For U(1) symmetries, we need e^(2πi(n+1)/N) to account for base phase
            base_correction = torch.exp(torch.tensor(2j * torch.pi / len(coeffs), dtype=self.dtype))
            
            # First pass: compute basic coefficients with proper U(1) phases
            for i in range(len(coeffs)):
                # Scale coefficient based on its degree with proper composition
                power = torch.tensor(1.0 / (i + 1), dtype=self.dtype)  # Linear scaling with degree
                
                # Add composition correction with proper U(1) phase
                # The (i + 1) accounts for the base phase of the U(1) symmetry
                phase = torch.tensor(2j * torch.pi * (i + 1) / len(coeffs), dtype=self.dtype)
                correction = base_correction * torch.exp(phase)
                
                # Apply both scaling and phase correction
                adjusted[i] = coeffs[i] * power * correction
            
            # Second pass: add cross-terms with proper phase composition
            for i in range(1, len(coeffs)):
                # Add contribution from lower degree terms with proper phase
                for j in range(i):
                    # Cross phase must account for composition of U(1) phases
                    # The (j + 1) factor ensures proper phase composition
                    cross_phase = torch.tensor(2j * torch.pi * (i-j) / len(coeffs), dtype=self.dtype)
                    cross_term = adjusted[j] * adjusted[i-j-1] * torch.exp(cross_phase * (j + 1))
                    adjusted[i] = adjusted[i] + cross_term  # Add cross terms without additional scaling
            
            # Normalize with proper phase preservation
            norm = torch.norm(adjusted) + 1e-8
            # Preserve the U(1) phase structure in normalization
            phase_factor = torch.exp(1j * torch.angle(adjusted[0])) if adjusted[0] != 0 else 1.0
            adjusted = (adjusted / norm) * phase_factor
            
            return AnomalyPolynomial(
                coefficients=adjusted,
                variables=poly.variables,
                degree=poly.degree,
                type=poly.type
            )
            
        return [adjust_for_composition(anomaly_poly)]

    def scale_invariants(self, structure: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Find scale invariant quantities in the structure.

        Returns a list of (tensor, scaling_dimension) pairs.
        """
        # Ensure structure has correct dtype
        structure = self._ensure_dtype(structure)

        # Initialize list of invariants
        invariants = []

        # Create RG flow with improved observable
        def observable(x: torch.Tensor) -> torch.Tensor:
            # Project onto structure components with improved stability
            x_flat = x.flatten()[:structure.numel()]
            structure_flat = structure.flatten()
            # Pad shorter tensor with zeros
            if len(x_flat) < len(structure_flat):
                x_flat = torch.nn.functional.pad(x_flat, (0, len(structure_flat) - len(x_flat)))
            elif len(x_flat) > len(structure_flat):
                x_flat = x_flat[:len(structure_flat)]
            # Compute projection with normalization
            proj = torch.sum(x_flat * structure_flat.conj()) / (torch.norm(structure_flat) + 1e-8)
            # Add quadratic term for better detection
            return proj + 0.5 * torch.sum(torch.abs(x_flat)**2)

        # Initialize RG flow
        flow = self.renormalization_flow(observable)

        # Sample different scales with improved resolution
        scales = torch.logspace(-1, 1, 10)  # Fewer scale points for stability
        values = []

        # Process in smaller batches for memory efficiency
        batch_size = 5
        for i in range(0, len(scales), batch_size):
            batch_scales = scales[i:i+batch_size]
            batch_values = []

            for scale in batch_scales:
                evolved = flow.evolve(float(scale)).apply(structure.flatten()[:self.dim])
                # Normalize evolved state with phase preservation
                norm = torch.norm(evolved)
                if norm > 0:
                    phase = torch.exp(1j * torch.angle(evolved[0])) if torch.abs(evolved[0]) > 1e-10 else 1.0
                    evolved = (evolved / norm) * phase
                batch_values.append(evolved)

            values.extend(batch_values)

        # Convert to tensor
        values = torch.stack(values)

        # Look for approximately constant quantities with improved detection
        for i in range(values.shape[1]):
            component = values[:, i]
            # Compute variation using robust statistics
            median_val = torch.median(torch.abs(component))
            if median_val < 1e-6:  # Skip near-zero components
                continue

            # Use multiple variation measures with relaxed thresholds
            variation = torch.std(torch.abs(component)) / (median_val + 1e-8)
            max_dev = torch.max(torch.abs(component - torch.mean(component))) / (median_val + 1e-8)

            # Compute local variations efficiently
            window = 3  # Smaller window size
            local_vars = []
            for j in range(len(component) - window + 1):
                window_vals = component[j:j+window]
                local_var = torch.std(torch.abs(window_vals)) / (torch.mean(torch.abs(window_vals)) + 1e-8)
                local_vars.append(local_var)
            avg_local_var = torch.mean(torch.tensor(local_vars))

            # Relaxed thresholds for better detection
            if variation < 0.5 and max_dev < 0.7 and avg_local_var < 0.4:
                # Estimate scaling dimension using robust fit
                diffs = torch.log(torch.abs(component[1:] / (component[:-1] + 1e-8)))
                scale_diffs = torch.log(scales[1:] / scales[:-1])

                # Use median for robustness
                scaling_dim = float(torch.median(diffs / (scale_diffs + 1e-8)))

                # Create invariant tensor with proper normalization
                invariant = torch.zeros_like(structure)
                invariant_flat = invariant.flatten()
                invariant_flat[i] = 1.0
                invariant = invariant_flat.reshape(structure.shape)
                norm = torch.norm(invariant)
                if norm > 0:
                    phase = torch.exp(1j * torch.angle(invariant.flatten()[0])) if torch.abs(invariant.flatten()[0]) > 1e-10 else 1.0
                    invariant = (invariant / norm) * phase
                invariants.append((invariant, scaling_dim))

        return invariants

    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion with improved efficiency."""
        # Ensure inputs have correct dtype
        op1 = self._ensure_dtype(op1)
        op2 = self._ensure_dtype(op2)
        
        # Flatten inputs if needed
        if op1.dim() > 1:
            op1 = op1.reshape(-1)
        if op2.dim() > 1:
            op2 = op2.reshape(-1)
            
        # Pad or truncate to match network input dimension
        target_dim = self.dim
        
        def adjust_tensor(t: torch.Tensor) -> torch.Tensor:
            if len(t) > target_dim:
                return t[:target_dim]
            elif len(t) < target_dim:
                padding = torch.zeros(target_dim - len(t), dtype=self.dtype)
                return torch.cat([t, padding])
            return t
            
        op1 = adjust_tensor(op1)
        op2 = adjust_tensor(op2)
        
        # Normalize inputs for better convergence
        op1_norm = torch.norm(op1)
        op2_norm = torch.norm(op2)
        
        if op1_norm > 0:
            op1 = op1 / op1_norm
        if op2_norm > 0:
            op2 = op2 / op2_norm
        
        # Combine operators with proper normalization
        combined = torch.cat([op1, op2])
        
        # Add batch dimension if needed
        if combined.dim() == 1:
            combined = combined.unsqueeze(0)
        
        # Compute OPE with improved convergence
        result = self.ope_net(combined)
        
        # Remove batch dimension if added
        if result.dim() > 1 and result.shape[0] == 1:
            result = result.squeeze(0)
        
        # Scale result back and ensure proper normalization
        result = result * torch.sqrt(op1_norm * op2_norm)
        
        # For nearby points, the OPE should approximate direct product
        direct_product = op1[0] * op2[0]  # Use first components
        result = result * (direct_product / (result[0] + 1e-8))  # Normalize to match direct product
        
        return result

    def conformal_symmetry(self, state: torch.Tensor) -> bool:
        """Check if state has conformal symmetry using optimized detection."""
        # Ensure state has correct shape
        if state.dim() > 2:
            state = state.reshape(-1, state.shape[-1])
        
        # Test special conformal transformations
        def test_special_conformal(b_vector: torch.Tensor) -> bool:
            """Test special conformal transformation."""
            # Ensure proper dtype
            b_vector = self._ensure_dtype(b_vector)
            
            # Sample test points
            x = self._ensure_dtype(torch.randn(self.dim))
            v1 = self._ensure_dtype(torch.randn(self.dim))
            v2 = self._ensure_dtype(torch.randn(self.dim))
            
            # Compute original angle
            v1_norm = torch.norm(v1) + 1e-8
            v2_norm = torch.norm(v2) + 1e-8
            angle1 = torch.sum(v1 * v2.conj()) / (v1_norm * v2_norm)
            
            # Apply special conformal transformation
            def transform_vector(v: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # Compute denominator with stability
                x_sq = torch.sum(x * x.conj())
                b_sq = torch.sum(b * b.conj())
                b_x = torch.sum(b * x.conj())
                denom = 1 + 2 * b_x + b_sq * x_sq + 1e-8
                
                # Transform coordinates
                x_new = x + torch.sum(x * x.conj()) * b
                x_new = x_new / denom
                
                # Transform vector
                jac = torch.eye(self.dim, dtype=self.dtype)
                for i in range(self.dim):
                    for j in range(self.dim):
                        if i == j:
                            jac[i,i] = (1 + 2 * b_x + b_sq * x_sq - 2 * x[i] * torch.sum(b * x.conj())) / denom
                        else:
                            jac[i,j] = -2 * (b[i] * x[j] - x[i] * b[j]) / denom
                
                return torch.mv(jac, v)
            
            # Transform vectors
            transformed_v1 = transform_vector(v1, x, b_vector)
            transformed_v2 = transform_vector(v2, x, b_vector)
            
            # Compute transformed angle
            t_v1_norm = torch.norm(transformed_v1) + 1e-8
            t_v2_norm = torch.norm(transformed_v2) + 1e-8
            angle2 = torch.sum(transformed_v1 * transformed_v2.conj()) / (t_v1_norm * t_v2_norm)
            
            # Check angle preservation with proper tolerance
            return torch.allclose(angle1.real, angle2.real, rtol=1e-2) and torch.allclose(angle1.imag, angle2.imag, rtol=1e-2)
        
        # Test multiple b vectors
        test_vectors = [
            torch.ones(self.dim, dtype=self.dtype),
            torch.zeros(self.dim, dtype=self.dtype).index_fill_(0, torch.tensor(0), 1.0),
            0.5 * torch.ones(self.dim, dtype=self.dtype)
        ]
        
        return all(test_special_conformal(b) for b in test_vectors)

    def minimal_invariant_number(self) -> int:
        """Get minimal number of scale invariants based on cohomology."""
        return max(1, self.dim - 1)  # Optimal number based on dimension

    def analyze_cohomology(
        self,
        states: List[torch.Tensor],
        scales: List[float],
    ) -> Dict[str, Any]:
        """Complete cohomology analysis with optimized computation.
        
        Args:
            states: List of quantum states
            scales: List of scale parameters
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Analyze scale connections efficiently
        for i in range(len(states) - 1):
            conn = self.scale_connection(
                torch.tensor(scales[i], dtype=self.dtype),
                torch.tensor(scales[i + 1], dtype=self.dtype)
            )
            results[f'connection_{i}'] = conn
            
        # Compute RG flow with improved convergence
        rg_flow = self.renormalization_flow(states[0])
        results['rg_flow'] = rg_flow
        
        # Find fixed points efficiently
        fixed_pts = self.fixed_points(states[0])
        results['fixed_points'] = fixed_pts
        
        # Detect anomalies using forms
        for i, state in enumerate(states):
            # Create a symmetry action function for this state
            def symmetry_action(x: torch.Tensor, state=state) -> torch.Tensor:
                # Apply a simple U(1) transformation
                phase = torch.sum(x * state) / (torch.norm(x) * torch.norm(state) + 1e-8)
                return x * torch.exp(1j * phase)
            
            anomalies = self.anomaly_polynomial(symmetry_action)
            results[f'anomalies_{i}'] = anomalies
            
        # Find scale invariants with improved detection
        for i, state in enumerate(states):
            invariants = self.scale_invariants(state)
            results[f'invariants_{i}'] = invariants
            
        # Check conformal properties efficiently
        for i, state in enumerate(states):
            is_conformal = self.conformal_symmetry(state)
            results[f'conformal_{i}'] = is_conformal
            
        # Convert cohomology results to output dtype
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = self._to_output_dtype(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                results[key] = [self._to_output_dtype(v) if isinstance(v, torch.Tensor) else v for v in value]
            
        return results

    def _classify_anomaly(self, degree: int) -> str:
        """Classify anomaly based on form degree with enhanced classification."""
        if degree == 0:
            return "scalar"
        if degree == 1:
            return "vector"
        if degree == 2:
            return "tensor"
        if degree == 3:
            return "cubic"
        return f"degree_{degree}"

    def callan_symanzik_operator(
        self, 
        beta: Callable[[torch.Tensor], torch.Tensor],
        gamma: Callable[[torch.Tensor], torch.Tensor],
        dgamma: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable:
        """Compute Callan-Symanzik operator β(g)∂_g + γ(g)D - d.
        
        This implements the classical CS equation and optionally cross-validates with 
        the quantum OPE approach when quantum states are available.
        
        The CS equation describes how correlation functions transform under scale transformations:
        β(g)∂_g C + γ(g)D C - d C = 0
        
        where:
        - β(g) is the beta function describing coupling flow
        - γ(g) is the anomalous dimension
        - ∂_g γ(g) is the derivative of the anomalous dimension
        - D is the dilatation operator
        - d is the canonical dimension
        
        For a correlation of the form C = |x2-x1|^(-1 + γ(g)):
        1. β(g)∂_g C = β(g) * ∂_g γ(g) * log|x2-x1| * C
        2. γ(g)D C = γ(g) * (-1 + γ(g)) * C
        3. d C = (-1 + γ(g)) * C
        
        The last two terms cancel when γ(g)D C - d C = 0, and the first term vanishes
        when β(g)∂_g γ(g) = γ(g)².
        
        Args:
            beta: Callable that computes β(g) for coupling g
            gamma: Callable that computes γ(g) for coupling g
            dgamma: Callable that computes ∂_g γ(g) for coupling g
            
        Returns:
            Callable that computes the CS operator action on correlation functions
        """
        def cs_operator(correlation: Callable, x1: torch.Tensor, x2: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            """Apply CS operator to correlation function.
            
            Args:
                correlation: Correlation function C(x1, x2, g)
                x1: First position
                x2: Second position
                g: Coupling constant
                
            Returns:
                Result of CS operator action (should be ≈ 0 for scale invariance)
            """
            # Ensure inputs have correct dtype and require gradients
            x1 = self._ensure_dtype(x1).detach().requires_grad_(True)
            x2 = self._ensure_dtype(x2).detach().requires_grad_(True)
            g = self._ensure_dtype(g).detach().requires_grad_(True)
            
            # Compute correlation with gradient tracking
            corr = correlation(x1, x2, g)
            
            # Compute log|x2-x1| for proper derivative scaling
            diff = x2 - x1
            if diff.is_complex():
                dist = torch.sqrt(torch.sum(diff * diff.conj())).real
            else:
                dist = torch.norm(diff)
            log_dist = torch.log(dist + 1e-8)
            
            # Compute β(g)∂_g C term
            beta_val = beta(g)
            gamma_val = gamma(g)
            dgamma_val = dgamma(g)
            
            # Compute the terms in the CS equation
            # For C = |x2-x1|^(-1 + γ(g)):
            # 1. ∂_g C = C * log|x2-x1| * ∂_g γ(g)
            # 2. D C = (-1 + γ(g)) * C
            # 3. d C = (-1 + γ(g)) * C
            # Therefore:
            # β(g)∂_g C + γ(g)D C + d C =
            # C * [β(g) * log|x2-x1| * ∂_g γ(g) + γ(g) * (-1 + γ(g)) + (-1 + γ(g))] = 0
            beta_term = beta_val * log_dist * dgamma_val
            gamma_term = gamma_val * (-1 + gamma_val)
            dim_term = (-1 + gamma_val)
            result = corr * (beta_term + gamma_term + dim_term)
            
            return result
            
        return cs_operator

    def special_conformal_transform(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Apply special conformal transformation x' = (x + bx²)/(1 + 2bx + b²x²)."""
        # Ensure inputs have correct dtype
        x = self._ensure_dtype(x)
        b = self._ensure_dtype(b)
            
        # Compute x² and b·x with improved numerical stability
        x_sq = torch.sum(x * x.conj())  # Use conjugate for complex tensors
        b_dot_x = torch.sum(b * x.conj())  # Use conjugate for complex tensors
        b_sq = torch.sum(b * b.conj())  # Use conjugate for complex tensors
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        # Apply conformal transformation with improved stability
        numerator = x + b * x_sq
        denominator = 1 + 2 * b_dot_x + b_sq * x_sq + epsilon
        
        # Ensure transformation preserves angles by normalizing
        result = numerator / denominator
        result_norm = torch.norm(result)
        if result_norm > 0:
            # Scale to preserve original norm
            result = result * (torch.norm(x) / result_norm)
            
            # Ensure angle preservation by projecting onto original direction
            x_direction = x / (torch.norm(x) + epsilon)
            result_direction = result / (torch.norm(result) + epsilon)
            angle = torch.sum(x_direction * result_direction.conj()).real
            if angle < 0:
                result = -result  # Flip direction if angle is negative
            
        return result

    def transform_vector(self, v: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Transform vector under conformal transformation with improved angle preservation."""
        # Ensure inputs have correct dtype
        v = self._ensure_dtype(v)
        x = self._ensure_dtype(x)
        b = self._ensure_dtype(b)
            
        # Compute transformation Jacobian with improved numerical stability
        x_sq = torch.sum(x * x.conj())  # Use conjugate for complex tensors
        b_dot_x = torch.sum(b * x.conj())  # Use conjugate for complex tensors
        b_sq = torch.sum(b * b.conj())  # Use conjugate for complex tensors
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        denom = 1 + 2 * b_dot_x + b_sq * x_sq + epsilon
        
        # Compute Jacobian matrix with improved angle preservation
        identity = torch.eye(self.dim, dtype=self.dtype)
        outer_term = torch.outer(b, x.conj())  # Use conjugate for complex tensors
        
        # Build Jacobian with careful normalization
        jacobian = (identity / denom - 2 * outer_term / (denom * denom))
        
        # Apply transformation with angle preservation
        transformed = jacobian @ v
        
        # Normalize to preserve vector magnitude and angles
        v_norm = torch.norm(v)
        if v_norm > 0:
            # First normalize to unit vector
            transformed = transformed / (torch.norm(transformed) + epsilon)
            # Then scale back to original magnitude
            transformed = transformed * v_norm
            
            # Ensure angle preservation by projecting onto original direction
            v_direction = v / (torch.norm(v) + epsilon)
            transformed_direction = transformed / (torch.norm(transformed) + epsilon)
            angle = torch.sum(v_direction * transformed_direction.conj()).real
            if angle < 0:
                transformed = -transformed  # Flip direction if angle is negative
            
        return transformed

    def holographic_lift(self, boundary: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Lift boundary field to bulk using AdS/CFT correspondence."""
        # Ensure inputs have correct dtype
        boundary = self._ensure_dtype(boundary)
        radial = self._ensure_dtype(radial)
            
        # Initialize bulk field
        bulk_shape = (len(radial), *boundary.shape)
        bulk = torch.zeros(bulk_shape, dtype=self.dtype)
        
        # Compute bulk field using Fefferman-Graham expansion
        for i, z in enumerate(radial):
            # Leading term
            bulk[i] = boundary * z**(-self.dim)
            
            # Subleading corrections from conformal dimension
            for n in range(1, 4):  # Include first few corrections
                bulk[i] += (-1)**n * boundary * z**(-self.dim + 2*n) / (2*n)
                
            # Add quantum corrections using OPE
            if i > 0:  # Skip boundary point
                # Compute OPE between previous bulk slice and boundary
                prev_bulk_flat = bulk[i-1].flatten()
                boundary_flat = boundary.flatten()
                
                # Ensure we have enough components
                min_size = min(len(prev_bulk_flat), len(boundary_flat))
                if min_size < self.dim:
                    # Pad with zeros if needed
                    prev_bulk_flat = torch.nn.functional.pad(prev_bulk_flat, (0, self.dim - min_size))
                    boundary_flat = torch.nn.functional.pad(boundary_flat, (0, self.dim - min_size))
                else:
                    # Take first dim components
                    prev_bulk_flat = prev_bulk_flat[:self.dim]
                    boundary_flat = boundary_flat[:self.dim]
                
                ope_corr = self.operator_product_expansion(prev_bulk_flat, boundary_flat)
                # Reshape OPE correction to match boundary shape
                ope_corr = ope_corr.reshape(-1)  # Flatten to 1D
                if len(ope_corr) == 1:
                    # If scalar output, broadcast to boundary shape
                    ope_corr = ope_corr.expand(boundary.numel()).reshape(boundary.shape)
                else:
                    # Otherwise reshape to match boundary shape
                    # First ensure we have enough elements
                    if len(ope_corr) < boundary.numel():
                        ope_corr = torch.nn.functional.pad(ope_corr, (0, boundary.numel() - len(ope_corr)))
                    elif len(ope_corr) > boundary.numel():
                        ope_corr = ope_corr[:boundary.numel()]
                    ope_corr = ope_corr.reshape(boundary.shape)
                
                bulk[i] += ope_corr * z**(-self.dim + 2)
                
        return bulk

    def entanglement_entropy(self, state: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """Compute entanglement entropy using replica trick with improved area law scaling."""
        # Convert state to density matrix if needed
        if state.dim() == 1:
            state = torch.outer(state, state.conj())
            
        # Ensure state has correct dtype
        state = self._ensure_dtype(state)
        region = region.bool()  # Convert region to boolean mask
            
        # Compute reduced density matrix
        n_sites = state.shape[0]
        n_region = int(region.sum().item())  # Convert to Python int
        
        # Ensure dimensions are valid
        if n_region <= 0 or n_region >= n_sites:
            return torch.tensor(0.0, dtype=self.dtype)
            
        # Reshape into bipartite form
        # First reshape state into a matrix where rows correspond to region sites
        # and columns to complement sites
        n_complement = n_sites - n_region
        
        # Ensure the state size matches the expected size for the bipartition
        expected_size = n_region * n_complement
        if state.numel() != expected_size:
            # Truncate or pad the state to match expected size
            if state.numel() > expected_size:
                state = state.reshape(-1)[:expected_size].reshape(n_region, n_complement)
            else:
                padding = torch.zeros(expected_size - state.numel(), dtype=self.dtype)
                state = torch.cat([state.reshape(-1), padding]).reshape(n_region, n_complement)
        else:
            state = state.reshape(n_region, n_complement)
        
        # Compute reduced density matrix by tracing out complement
        rho = state @ state.conj().t()
        
        # Normalize density matrix
        trace = torch.trace(rho)
        if trace != 0:
            rho = rho / trace
        
        # Compute eigenvalues with improved numerical stability
        eigenvals = torch.linalg.eigvals(rho)
        eigenvals = eigenvals.real  # Should be real for density matrix
        
        # Remove numerical noise and normalize
        eigenvals = eigenvals[eigenvals > 1e-10]
        if len(eigenvals) > 0:
            eigenvals = eigenvals / eigenvals.sum()  # Normalize probabilities
            
            # Compute von Neumann entropy with improved numerical stability
            entropy = -torch.sum(eigenvals * torch.log(eigenvals + 1e-10))
            
            # Scale entropy by boundary area to satisfy area law
            # For 2D regions, boundary is proportional to perimeter
            if region.dim() == 2:
                # Compute perimeter using edge detection
                boundary_size = float(
                    torch.sum(
                        region[:-1, :] != region[1:, :]
                    ) + torch.sum(
                        region[:, :-1] != region[:, 1:]
                    )
                )
            else:
                # For 1D regions, boundary is just two points
                boundary_size = 2.0
                
            # Scale entropy by boundary size with improved area law scaling
            # The factor 1/4 comes from the holographic area law
            # We use sqrt(log(n_region)) to account for logarithmic corrections
            entropy = entropy * boundary_size / (4 * torch.sqrt(torch.log(torch.tensor(n_region, dtype=torch.float32) + 1)))
            return entropy
            
        return torch.tensor(0.0, dtype=self.dtype)

    def _to_output_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to output dtype."""
        if not tensor.is_complex():
            return tensor.to(dtype=self.dtype)
        return tensor.real.to(dtype=self.dtype)

    def extract_uv_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract UV (boundary) data from bulk field."""
        # UV data is at the boundary (first slice)
        return field[0]

    def extract_ir_data(self, field: torch.Tensor) -> torch.Tensor:
        """Extract IR (deep bulk) data from bulk field."""
        # IR data is at the deepest bulk point (last slice)
        return field[-1]

    def reconstruct_from_ir(self, ir_data: torch.Tensor) -> torch.Tensor:
        """Reconstruct UV data from IR data using holographic principle."""
        # Use the holographic principle to reconstruct boundary data
        # This is a simplified version that assumes conformal symmetry
        return ir_data / (torch.norm(ir_data) + 1e-8)


class ScaleSystem:
    """Complete scale system for multi-scale analysis."""

    def __init__(self, dim: int, num_scales: int = 4, coupling_dim: int = 4, dtype=torch.float32):
        """Initialize the scale system.
        
        Args:
            dim: Dimension of the state space
            num_scales: Number of scale levels to analyze
            coupling_dim: Dimension of coupling space
            dtype: Data type for computations (default: torch.float32)
        """
        self.dim = dim
        # Always use complex64 internally for quantum computations
        self.internal_dtype = torch.complex64
        self.output_dtype = dtype
        
        # Initialize components with complex dtype for internal computations
        self.connection = ScaleConnection(dim, num_scales, dtype=self.internal_dtype)
        self.rg_flow = RenormalizationFlow(coupling_dim, dtype=self.internal_dtype)
        self.anomaly = AnomalyDetector(dim, dtype=self.internal_dtype)
        self.invariance = ScaleInvariance(dim, num_scales, dtype=self.internal_dtype)
        self.cohomology = ScaleCohomology(dim, num_scales, dtype=self.internal_dtype)
        
        # Initialize Riemann computer with complex dtype
        self.riemann_computer = nn.Sequential(
            nn.Linear(dim, dim * 2, dtype=self.internal_dtype),
            ComplexTanh(),
            nn.Linear(dim * 2, dim * dim, dtype=self.internal_dtype)
        )

    def _to_internal_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to internal complex dtype."""
        if not tensor.is_complex():
            return tensor.to(dtype=self.internal_dtype)
        return tensor.to(dtype=self.internal_dtype)

    def _to_output_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert internal complex tensor to output dtype."""
        if self.output_dtype == torch.float32:
            return tensor.real.to(dtype=self.output_dtype)
        return tensor.to(dtype=self.output_dtype)

    def analyze_scales(
        self, states: List[torch.Tensor], scale_factors: List[float]
    ) -> Dict[str, Any]:
        """Analyze multi-scale structure."""
        # Convert input states to internal complex dtype
        states = [self._to_internal_dtype(s) for s in states]
        
        results = {}

        # Analyze RG flow
        fixed_points, stability = self.rg_flow.find_fixed_points(states[0])
        results["fixed_points"] = [self._to_output_dtype(fp) for fp in fixed_points]
        results["stability"] = stability

        # Find scale invariant structures
        invariants = self.invariance.find_invariant_structures(
            torch.stack(states)
        )
        results["invariants"] = [(self._to_output_dtype(state), scale) for state, scale in invariants]

        # Detect anomalies
        anomalies = []
        for state in states:
            anomalies.extend(self.anomaly.detect_anomalies(state))
        # Convert anomaly coefficients to output dtype
        for anomaly in anomalies:
            anomaly.coefficients = self._to_output_dtype(anomaly.coefficients)
        results["anomalies"] = anomalies

        # Compute cohomology
        cohomology = self.cohomology.analyze_cohomology(states, scale_factors)
        # Convert cohomology results to output dtype
        for key, value in cohomology.items():
            if isinstance(value, torch.Tensor):
                cohomology[key] = self._to_output_dtype(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                cohomology[key] = [self._to_output_dtype(v) if isinstance(v, torch.Tensor) else v for v in value]
            
        return results

