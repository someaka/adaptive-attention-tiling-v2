"""Pattern stability validation implementation.

This module validates pattern stability:
- Linear stability analysis
- Nonlinear stability analysis
- Perturbation response
- Lyapunov analysis
- Mode decomposition
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol, runtime_checkable
import torch
import torch.nn as nn
import numpy as np

from src.validation.base import ValidationResult
from src.validation.flow.stability import (
    LinearStabilityValidator,
    NonlinearStabilityValidator,
    StructuralStabilityValidator
)
from src.core.tiling.geometric_flow import PatternFlow as CorePatternFlow


@runtime_checkable
class GeometricFlow(Protocol):
    """Protocol defining required methods for geometric flow."""
    
    def compute_metric(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from flow."""
        ...
        
    def compute_ricci_tensor(self, metric: torch.Tensor, connection: torch.Tensor) -> torch.Tensor:
        """Compute Ricci tensor from metric and connection."""
        ...
        
    def flow_step(
        self,
        metric: torch.Tensor,
        ricci: torch.Tensor,
        timestep: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evolve metric using Ricci flow."""
        ...
        
    def detect_singularities(
        self,
        flow: torch.Tensor,
        threshold: float = 1e-6
    ) -> List[Dict[str, torch.Tensor]]:
        """Detect singularities in the flow."""
        ...
        
    def normalize_flow(
        self,
        flow: torch.Tensor,
        normalization: str = "ricci"
    ) -> torch.Tensor:
        """Normalize flow vector field."""
        ...


@dataclass
class StabilityMetrics:
    """Stability metrics for pattern validation."""

    linear_stability: float
    """Linear stability measure."""

    nonlinear_stability: float
    """Nonlinear stability measure."""

    lyapunov_stability: float
    """Lyapunov stability measure."""

    def compute_overall_stability(self) -> float:
        """Compute overall stability score."""
        return (
            0.4 * self.linear_stability +
            0.4 * self.nonlinear_stability +
            0.2 * self.lyapunov_stability
        )

    def classify_stability(self) -> Dict[str, Any]:
        """Classify stability type."""
        overall = self.compute_overall_stability()
        
        if overall > 0.8:
            category = "strongly_stable"
            confidence = min(1.0, (overall - 0.8) * 5)
        elif overall > 0.6:
            category = "stable"
            confidence = min(1.0, (overall - 0.6) * 5)
        elif overall > 0.4:
            category = "marginally_stable"
            confidence = min(1.0, (overall - 0.4) * 5)
        else:
            category = "unstable"
            confidence = min(1.0, (0.4 - overall) * 2.5)
            
        return {
            "category": category,
            "confidence": float(confidence)
        }


class PatternFlow(CorePatternFlow, GeometricFlow):
    """Pattern-specific geometric flow implementation for stability validation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, manifold_dim: int = 32):
        super().__init__(input_dim, hidden_dim, manifold_dim)

        # Additional network for Ricci tensor computation
        self.ricci_net = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim),
        )

        # Additional network for metric computation
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim),
        )

    def compute_metric(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor from flow."""
        batch_size = flow.shape[0]
        
        # Project flow to metric space
        metric_components = self.metric_net(flow)
        metric = metric_components.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Ensure metric is symmetric and positive definite
        metric = 0.5 * (metric + metric.transpose(-1, -2))
        metric = metric + torch.eye(self.manifold_dim).to(flow.device) * 1e-6
        
        return metric

    def compute_ricci_tensor(self, metric: torch.Tensor, connection: torch.Tensor) -> torch.Tensor:
        """Compute Ricci tensor from metric and connection."""
        batch_size = metric.shape[0]
        
        # Flatten metric and connection for network input
        metric_flat = metric.reshape(batch_size, -1)
        connection_flat = connection.reshape(batch_size, -1)
        combined = torch.cat([metric_flat, connection_flat], dim=-1)
        
        # Compute Ricci tensor components
        ricci_components = self.ricci_net(combined)
        ricci = ricci_components.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Ensure symmetry
        ricci = 0.5 * (ricci + ricci.transpose(-1, -2))
        
        return ricci

    def flow_step(
        self,
        metric: torch.Tensor,
        ricci: torch.Tensor,
        timestep: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evolve metric using Ricci flow."""
        # Evolve metric using Ricci flow equation
        evolved_metric = metric - 2 * timestep * ricci
        
        # Compute flow metrics
        metrics = {
            "flow_magnitude": torch.norm(ricci.reshape(metric.shape[0], -1), dim=1),
            "metric_determinant": torch.linalg.det(evolved_metric),
            "ricci_scalar": torch.diagonal(ricci, dim1=1, dim2=2).sum(dim=1),
            "energy": torch.linalg.det(evolved_metric),
            "singularity": torch.linalg.det(metric),
            "normalized_flow": torch.linalg.det(evolved_metric)
        }
        
        return evolved_metric, metrics

    def detect_singularities(
        self,
        flow: torch.Tensor,
        threshold: float = 1e-6
    ) -> List[Dict[str, torch.Tensor]]:
        """Detect singularities in the flow."""
        batch_size = flow.shape[0]
        singularities = []
        
        # Compute flow derivatives
        flow_grad = torch.gradient(flow, dim=-1)[0]
        
        # Check for singular points (where flow vanishes)
        flow_norm = torch.norm(flow, dim=-1)
        singular_points = flow_norm < threshold
        
        # Check for derivative singularities
        grad_norm = torch.norm(flow_grad.reshape(batch_size, -1), dim=-1)
        derivative_singularities = grad_norm > 1.0 / threshold
        
        # Collect singularity information
        for i in range(batch_size):
            if singular_points[i].any() or derivative_singularities[i]:
                singularities.append({
                    "position": flow[i],
                    "type": "vanishing" if singular_points[i].any() else "derivative",
                    "strength": flow_norm[i] if singular_points[i].any() else grad_norm[i]
                })
                
        return singularities

    def normalize_flow(
        self,
        flow: torch.Tensor,
        normalization: str = "ricci"
    ) -> torch.Tensor:
        """Normalize flow vector field."""
        if normalization == "ricci":
            # Normalize by Ricci scalar
            metric = self.compute_metric(flow)
            ricci_scalar = torch.diagonal(
                self.compute_ricci_tensor(
                    metric,
                    torch.zeros_like(flow)  # Dummy connection
                ),
                dim1=1, dim2=2
            ).sum(dim=1, keepdim=True)
            normalized = flow / (torch.abs(ricci_scalar) + 1e-8)
            
        elif normalization == "metric":
            # Normalize by metric determinant
            metric = self.compute_metric(flow)
            det = torch.linalg.det(metric).unsqueeze(-1)
            normalized = flow / (torch.abs(det) + 1e-8)
            
        else:  # Default L2 normalization
            norm = torch.norm(flow, dim=-1, keepdim=True)
            normalized = flow / (norm + 1e-8)
            
        return normalized


class LyapunovValidator:
    """Validator for Lyapunov stability analysis."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def compute_lyapunov_exponents(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute Lyapunov exponents for trajectories."""
        batch_size = trajectories.shape[0]
        time_steps = trajectories.shape[-1]
        
        # Compute finite-time Lyapunov exponents
        exponents = []
        for i in range(batch_size):
            traj = trajectories[i]
            # Compute growth rates between consecutive points
            diffs = torch.diff(traj, dim=-1)
            growth_rates = torch.log(torch.norm(diffs, dim=0) + 1e-8)
            # Average over time
            exponent = growth_rates.mean()
            exponents.append(exponent)
            
        return torch.stack(exponents)

    def classify_stability(self, trajectories: torch.Tensor) -> Dict[str, Any]:
        """Classify stability based on Lyapunov analysis."""
        exponents = self.compute_lyapunov_exponents(trajectories)
        max_exponent = exponents.max().item()
        
        if max_exponent < -self.tolerance:
            stability_type = "asymptotically_stable"
            confidence = min(1.0, abs(max_exponent) / (10 * self.tolerance))
        elif abs(max_exponent) < self.tolerance:
            stability_type = "marginally_stable"
            confidence = min(1.0, self.tolerance / abs(max_exponent) if max_exponent != 0 else 1.0)
        else:
            stability_type = "unstable"
            confidence = min(1.0, max_exponent / (10 * self.tolerance))
            
        return {
            "type": stability_type,
            "confidence": float(confidence)
        }

    def estimate_attractor_dimension(self, trajectories: torch.Tensor) -> float:
        """Estimate attractor dimension using Kaplan-Yorke dimension."""
        exponents = self.compute_lyapunov_exponents(trajectories)
        sorted_exponents = torch.sort(exponents, descending=True)[0]
        
        # Find Kaplan-Yorke dimension
        partial_sum = 0.0
        for i, exp in enumerate(sorted_exponents):
            partial_sum += exp.item()
            if partial_sum < 0:
                # Interpolate between dimensions
                prev_sum = partial_sum - exp.item()
                dim = i + prev_sum / abs(exp.item())
                return float(dim)
                
        return float(len(sorted_exponents))

    def estimate_predictability_time(self, trajectories: torch.Tensor) -> float:
        """Estimate predictability time based on largest Lyapunov exponent."""
        exponents = self.compute_lyapunov_exponents(trajectories)
        max_exponent = exponents.max().item()
        
        if max_exponent <= 0:
            return float('inf')
        
        # Use inverse of max Lyapunov exponent as characteristic time
        return float(1.0 / max_exponent)


class PatternStabilityValidator:
    """Complete pattern stability validation system."""

    def __init__(
        self,
        linear_threshold: float = 0.1,
        nonlinear_threshold: float = 0.2,
        lyapunov_threshold: float = 0.01,
        tolerance: float = 1e-6,
    ):
        self.linear_threshold = linear_threshold
        self.nonlinear_threshold = nonlinear_threshold
        self.lyapunov_threshold = lyapunov_threshold
        self.tolerance = tolerance
        
        # Initialize pattern flow
        self.pattern_flow: Optional[GeometricFlow] = None  # Lazy initialization since we need input dimension
        
        # Initialize validators
        self.linear_validator = LinearStabilityValidator(tolerance, linear_threshold)
        self.nonlinear_validator = NonlinearStabilityValidator(tolerance)
        self.lyapunov_validator = LyapunovValidator(tolerance)

    def _ensure_flow(self, patterns: torch.Tensor) -> None:
        """Ensure pattern flow is initialized with correct dimensions."""
        if self.pattern_flow is None:
            input_dim = patterns.shape[-1]
            self.pattern_flow = PatternFlow(input_dim=input_dim)

    def validate_stability(
        self, time_series: torch.Tensor, parameters: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Validate pattern stability."""
        self._ensure_flow(time_series)
        
        # Compute stability metrics
        metrics = self.compute_stability_metrics(time_series)
        
        # Prepare validation result
        result = {
            "linear": metrics.linear_stability,
            "nonlinear": metrics.nonlinear_stability,
            "lyapunov": metrics.lyapunov_stability,
            "overall_score": metrics.compute_overall_stability()
        }
        
        # Add parameter dependence if provided
        if parameters is not None:
            param_sensitivity = self._compute_parameter_sensitivity(time_series, parameters)
            result["parameter_dependence"] = float(param_sensitivity)
            
        return result

    def compute_stability_metrics(self, patterns: torch.Tensor) -> StabilityMetrics:
        """Compute stability metrics for patterns."""
        self._ensure_flow(patterns)
        
        if self.pattern_flow is None:
            raise ValueError("Pattern flow initialization failed")
            
        # Compute linear stability
        linear_result = self.linear_validator.validate_stability(
            flow=self.pattern_flow,  # type: ignore[arg-type]
            state=patterns
        )
        linear_stability = float(1.0 - linear_result.growth_rates.max().item())
        
        # Compute nonlinear stability
        nonlinear_result = self.nonlinear_validator.validate_stability(
            flow=self.pattern_flow,  # type: ignore[arg-type]
            state=patterns
        )
        nonlinear_stability = float(nonlinear_result.basin_size)
        
        # Compute Lyapunov stability
        lyapunov_exponents = self.lyapunov_validator.compute_lyapunov_exponents(patterns)
        lyapunov_stability = float(1.0 / (1.0 + torch.exp(lyapunov_exponents.max()).item()))
        
        return StabilityMetrics(
            linear_stability=linear_stability,
            nonlinear_stability=nonlinear_stability,
            lyapunov_stability=lyapunov_stability
        )

    def decompose_modes(self, patterns: torch.Tensor) -> List[torch.Tensor]:
        """Decompose patterns into spatial modes."""
        # Use SVD for mode decomposition
        U, S, V = torch.linalg.svd(patterns.reshape(patterns.shape[0], -1))
        
        # Reconstruct modes
        modes = []
        for i in range(min(5, len(S))):  # Keep top 5 modes
            mode = (U[:, i:i+1] @ V[i:i+1, :]).reshape(patterns.shape)
            modes.append(mode)
            
        return modes

    def analyze_mode_stability(self, modes: List[torch.Tensor]) -> List[float]:
        """Analyze stability of individual modes."""
        stabilities = []
        for mode in modes:
            # Compute mode energy
            energy = torch.norm(mode)
            # Normalize stability to [0, 1]
            stability = float(1.0 / (1.0 + torch.exp(-energy)).item())
            stabilities.append(stability)
            
        return stabilities

    def analyze_mode_interactions(self, modes: List[torch.Tensor]) -> torch.Tensor:
        """Analyze interactions between modes."""
        n_modes = len(modes)
        interactions = torch.zeros((n_modes, n_modes))
        
        for i in range(n_modes):
            for j in range(n_modes):
                # Compute normalized correlation
                correlation = torch.sum(modes[i] * modes[j]) / (
                    torch.norm(modes[i]) * torch.norm(modes[j])
                )
                interactions[i, j] = correlation
                
        return interactions

    def _compute_parameter_sensitivity(
        self, patterns: torch.Tensor, parameters: torch.Tensor
    ) -> float:
        """Compute sensitivity to parameter variations."""
        # Compute correlation between patterns and parameters
        pattern_var = torch.var(patterns.reshape(patterns.shape[0], -1), dim=1)
        param_var = torch.var(parameters, dim=0)
        
        correlation = torch.corrcoef(
            torch.stack([pattern_var, param_var])
        )[0, 1]
        
        return float(abs(correlation).item())
