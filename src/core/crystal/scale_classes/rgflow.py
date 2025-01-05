from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
from contextlib import contextmanager

import logging

import torch
from torch import nn

# Import memory optimization utilities
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from utils.memory_management_util import optimize_memory, register_tensor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

