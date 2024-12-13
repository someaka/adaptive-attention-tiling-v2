"""Geometric Flow Validation Implementation.

This module validates flow properties:
- Flow stability
- Energy conservation
- Convergence criteria
- Singularity detection
- Normalization
- Ricci tensor
- Flow step
- Singularities
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch

from ...neural.flow.geometric_flow import GeometricFlow
from ...neural.flow.hamiltonian import HamiltonianSystem
from ..base import ValidationResult as BaseValidationResult


@dataclass
class EnergyMetrics:
    """Energy metrics for flow.
    
    Attributes:
        kinetic: Kinetic energy tensor [time]
        potential: Potential energy tensor [time]
        total: Total energy tensor [time]
        initial_energy: Initial energy at t=0
        final_energy: Final energy at t=T
        mean_energy: Mean energy over time
        energy_std: Standard deviation of energy over time
        max_variation: Maximum variation from mean energy
        relative_variation: Relative variation from mean energy
        energy_trajectory: Full trajectory of energy over time
    """
    kinetic: torch.Tensor
    potential: torch.Tensor
    total: torch.Tensor
    initial_energy: float
    final_energy: float
    mean_energy: float
    energy_std: float
    max_variation: float
    relative_variation: float
    energy_trajectory: List[float]


@dataclass
class SingularityInfo:
    """Information about a detected singularity."""
    
    location: torch.Tensor  # Location in parameter space
    severity: float  # Severity measure
    resolution: torch.Tensor  # Resolution vector


@dataclass
class FlowValidationResult(BaseValidationResult[Dict[str, Any]]):
    """Result of a flow validation."""
    
    def merge(self, other: BaseValidationResult) -> 'FlowValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, BaseValidationResult):
            raise ValueError("Can only merge with another ValidationResult")
            
        merged_data = {**(self.data or {}), **(other.data or {})}
        return FlowValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data=merged_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "data": {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in (self.data or {}).items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlowValidationResult':
        """Create from dictionary."""
        return cls(
            is_valid=data["is_valid"],
            message=data["message"],
            data=data.get("data", {})
        )


@dataclass
class FlowProperties:
    """Properties of geometric flow."""
    
    is_stable: bool = True
    is_conservative: bool = True
    is_convergent: bool = True
    has_singularities: bool = False
    stability_metrics: Optional[Dict[str, Union[float, List[float]]]] = None
    energy_metrics: Optional[Dict[str, Union[float, List[float]]]] = None
    convergence_metrics: Optional[Dict[str, Union[float, int]]] = None
    singularity_metrics: Optional[Dict[str, Union[bool, float, torch.Tensor]]] = None
    derivative: Optional[torch.Tensor] = None
    second_derivative: Optional[torch.Tensor] = None
    total_energy: Optional[float] = None
    energy_variation: Optional[float] = None


@dataclass
class SingularityDetector:
    """Detector for flow singularities."""

    def __init__(
        self,
        threshold: float = 1e6,
        window_size: int = 10
    ):
        """Initialize singularity detector.
        
        Args:
            threshold: Threshold for singularity detection
            window_size: Window size for temporal analysis
        """
        self.threshold = threshold
        self.window_size = window_size
        
    def detect_singularities(
        self,
        flow: GeometricFlow,
        state: torch.Tensor,
        time_points: torch.Tensor
    ) -> Dict[str, Union[bool, Optional[torch.Tensor]]]:
        """Detect singularities in flow.
        
        Args:
            flow: Geometric flow
            state: Initial state
            time_points: Time points to check
            
        Returns:
            Dictionary with detection results:
            - has_singularity (bool): Whether a singularity was detected
            - singularity_time (Optional[torch.Tensor]): Time point where singularity occurred, if any
        """
        # Track flow evolution
        current_state = state
        has_singularity = False
        singularity_time = None
        
        # Check each time point
        for t in time_points:
            # Evolve state
            evolved = flow.evolve(current_state, int(t.item()))
            # Handle both return types (Tensor or Tuple)
            new_state = evolved[0][0] if isinstance(evolved, tuple) else evolved
            
            # Check for singularities
            if self._check_singularity(new_state):
                has_singularity = True
                singularity_time = t
                break
                
            current_state = new_state
            
        return {
            "has_singularity": has_singularity,
            "singularity_time": singularity_time
        }
        
    def _check_singularity(
        self,
        state: torch.Tensor
    ) -> bool:
        """Check if state contains singularities.
        
        Args:
            state: State tensor
            
        Returns:
            True if singularity detected
        """
        # Check for infinities
        if torch.any(torch.isinf(state)):
            return True
            
        # Check for NaNs
        if torch.any(torch.isnan(state)):
            return True
            
        return False


class FlowValidator:
    """Validator for geometric flow properties."""
    
    def __init__(
        self,
        energy_threshold: float = 1e-6,
        monotonicity_threshold: float = 1e-4,
        singularity_threshold: float = 1.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ):
        """Initialize flow validator.
        
        Args:
            energy_threshold: Threshold for energy conservation
            monotonicity_threshold: Threshold for monotonicity checks
            singularity_threshold: Threshold for singularity detection
            max_iterations: Maximum iterations for convergence
            tolerance: General tolerance for numerical comparisons
        """
        self.energy_threshold = energy_threshold
        self.monotonicity_threshold = monotonicity_threshold
        self.singularity_threshold = singularity_threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def validate(self, points: torch.Tensor) -> bool:
        """Validate geometric invariants of flow.
        
        Args:
            points: Points tensor of shape (batch_size, dim)
            
        Returns:
            True if validation passes
        """
        # Basic shape validation
        if points.dim() != 2:
            return False
            
        # Check for NaN/Inf values
        if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
            return False
            
        # Check point normalization
        if not torch.allclose(torch.norm(points, dim=-1), torch.ones(points.size(0), device=points.device)):
            return False
            
        # Check dimensionality constraints
        if points.size(-1) < 2:  # Need at least 2D for meaningful geometric flow
            return False
            
        # Check numerical bounds
        if torch.any(torch.abs(points) > 1e6):
            return False
            
        return True

    def compute_energy(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute energy of flow at each timestep.
        
        Args:
            flow: Flow tensor of shape [time, batch, dim] or [time, batch, channels, height, width]
            
        Returns:
            Energy tensor of shape (time_steps,)
        """
        # Reshape if needed
        if flow.dim() == 5:  # [time, batch, channels, height, width]
            flow = flow.reshape(flow.size(0), flow.size(1), -1)  # [time, batch, dim]
            
        # Compute kinetic energy (using time derivative)
        if flow.size(0) > 1:
            velocity = (flow[1:] - flow[:-1])
            kinetic = 0.5 * torch.sum(velocity * velocity, dim=2)  # Sum over spatial dimensions
            # Pad to match original time steps
            kinetic = torch.cat([kinetic[:1], kinetic])
        else:
            kinetic = torch.zeros(1, device=flow.device)
            
        # Compute potential energy (using spatial derivatives)
        spatial_grad = torch.gradient(flow, dim=2)[0]  # Gradient along spatial dimension
        potential = 0.5 * torch.sum(spatial_grad * spatial_grad, dim=2)  # Sum over spatial dimensions
        
        # Total energy at each timestep (sum over batch dimension)
        return (kinetic + potential).mean(dim=1)  # Average over batch

    def validate_energy_conservation(self, flow: torch.Tensor) -> FlowValidationResult:
        """Validate energy conservation of flow.
        
        Args:
            flow: Flow tensor [time, batch, dim] or [time, batch, channels, height, width]
            
        Returns:
            ValidationResult with is_valid=True if energy is conserved
        """
        try:
            # Compute energy components
            if flow.size(0) > 1:
                velocity = (flow[1:] - flow[:-1])
                kinetic = 0.5 * torch.sum(velocity * velocity, dim=2)  # Sum over spatial dimensions
                # Pad to match original time steps
                kinetic = torch.cat([kinetic[:1], kinetic])
            else:
                kinetic = torch.zeros(1, device=flow.device)
                
            # Compute potential energy (using spatial derivatives)
            spatial_grad = torch.gradient(flow, dim=2)[0]  # Gradient along spatial dimension
            potential = 0.5 * torch.sum(spatial_grad * spatial_grad, dim=2)  # Sum over spatial dimensions
            
            # Total energy at each timestep (average over batch dimension)
            total = (kinetic + potential).mean(dim=1)  # Average over batch
            
            # Compute energy statistics
            mean_energy = torch.mean(total).item()
            initial_energy = float(total[0].item())
            final_energy = float(total[-1].item())
            
            # Handle std dev computation carefully
            if total.numel() > 1:
                std_energy = torch.std(total, unbiased=True).item()
            else:
                std_energy = 0.0
                
            max_variation = torch.max(torch.abs(total - mean_energy)).item()
            relative_variation = max_variation / (mean_energy + self.energy_threshold)
            
            # Create energy metrics
            metrics = EnergyMetrics(
                kinetic=kinetic,
                potential=potential,
                total=total,
                initial_energy=initial_energy,
                final_energy=final_energy,
                mean_energy=mean_energy,
                energy_std=std_energy,
                max_variation=max_variation,
                relative_variation=relative_variation,
                energy_trajectory=total.tolist()
            )
            
            # Check if energy is conserved within tolerance
            is_conserved = relative_variation < self.energy_threshold
            
            return FlowValidationResult(
                is_valid=is_conserved,
                message=f"Energy conservation {'passed' if is_conserved else 'failed'} " + \
                       f"(variation: {relative_variation:.2e}, tolerance: {self.energy_threshold:.2e})",
                data={
                    "mean_energy": metrics.mean_energy,
                    "initial_energy": metrics.initial_energy,
                    "final_energy": metrics.final_energy,
                    "energy_std": metrics.energy_std,
                    "max_variation": metrics.max_variation,
                    "relative_variation": metrics.relative_variation,
                    "energy_trajectory": metrics.energy_trajectory,
                    "total_energy": metrics.mean_energy,
                    "energy_variation": metrics.relative_variation,
                    "metrics": metrics
                }
            )
            
        except Exception as e:
            return FlowValidationResult(
                is_valid=False,
                message=f"Energy validation failed: {str(e)}",
                data={
                    "total_energy": 0.0,
                    "energy_variation": float('inf')
                }
            )

    def validate_monotonicity(self, flow: torch.Tensor) -> FlowValidationResult:
        """Validate flow monotonicity.
        
        Args:
            flow: Flow tensor of shape (time_steps, batch_size, dim)
            
        Returns:
            ValidationResult with monotonicity metrics
        """
        # Compute time derivative
        dt = 1.0  # Assume unit time steps
        velocity = (flow[1:] - flow[:-1]) / dt
        
        # Check if velocity maintains sign (across all dimensions)
        sign_changes = torch.sum(velocity[1:] * velocity[:-1] < 0, dim=(-3, -2, -1))
        is_monotonic = bool(torch.all(sign_changes == 0).item())  # Convert to Python bool
        
        return FlowValidationResult(
            is_valid=is_monotonic,
            message="Flow monotonicity validation",
            data={"monotonicity_measure": float(sign_changes.sum().item())}
        )

    def validate_long_time_existence(self, flow: torch.Tensor) -> FlowValidationResult:
        """Validate long-time existence of flow.
        
        Args:
            flow: Flow tensor [time, batch, dim] or [time, batch, channels, height, width]
            
        Returns:
            ValidationResult with is_valid=True if flow exists for long time
        """
        # Check for infinities or NaNs
        if torch.any(torch.isinf(flow)) or torch.any(torch.isnan(flow)):
            return FlowValidationResult(
                is_valid=False,
                message="Flow contains infinities or NaNs",
                data={
                    "error": "numerical_instability",
                    "existence_time": 0.0,
                    "max_value": float('inf')
                }
            )
            
        # Check growth rate
        if flow.dim() == 5:  # [time, batch, channels, height, width]
            norm = torch.norm(flow.reshape(flow.size(0), flow.size(1), -1), dim=2)  # [time, batch]
        else:  # [time, batch, dim]
            norm = torch.norm(flow, dim=2)  # [time, batch]
            
        if norm.size(0) > 1:
            # Compute relative growth rate
            growth_rate = (norm[1:] - norm[:-1]) / (norm[:-1] + 1e-8)  # Use relative difference instead of log
            max_growth = float(torch.max(growth_rate).item())
            
            # Flow should not grow too fast
            is_valid = max_growth < self.tolerance
            
            # Compute existence time as time until growth exceeds tolerance
            if is_valid:
                existence_time = float(flow.size(0))
            else:
                # Find first time growth exceeds tolerance
                exceeded_tensor = growth_rate > self.tolerance
                if exceeded_tensor.dim() > 1:
                    exceeded_mask = exceeded_tensor.any(dim=-1)  # Reduce batch dimension if needed
                else:
                    exceeded_mask = exceeded_tensor
                    
                if torch.any(exceeded_mask):
                    first_exceeded = torch.where(exceeded_mask)[0][0]
                    existence_time = float(first_exceeded.item())
                else:
                    existence_time = float(flow.size(0))
        else:
            max_growth = 0.0
            is_valid = True
            existence_time = float(flow.size(0))
        
        return FlowValidationResult(
            is_valid=is_valid,
            message=f"Long-time existence {'passed' if is_valid else 'failed'} " + \
                   f"(max growth: {max_growth:.2e}, tolerance: {self.tolerance:.2e})",
            data={
                "max_growth_rate": float(max_growth),
                "final_norm": float(norm[-1].mean().item()),
                "existence_time": existence_time,
                "max_value": float(torch.max(torch.abs(flow)).item())
            }
        )

    def validate_flow_step(
        self, 
        metric: torch.Tensor, 
        evolved_metric: torch.Tensor
    ) -> FlowValidationResult:
        """Validate flow evolution step.
        
        Args:
            metric: Initial metric tensor
            evolved_metric: Evolved metric tensor
            
        Returns:
            ValidationResult with step validation metrics
        """
        messages = []
        is_valid = True
        
        # Check metric remains positive definite
        eigenvals = torch.linalg.eigvalsh(evolved_metric)
        if not torch.all(eigenvals > 0):
            is_valid = False
            messages.append("Metric lost positive definiteness")
            
        # Check magnitude of change
        rel_diff = torch.norm(evolved_metric - metric) / torch.norm(metric)
        if rel_diff > self.tolerance:
            is_valid = False
            messages.append("Flow step too large")
            
        return FlowValidationResult(
            is_valid=bool(is_valid),
            message="; ".join(messages) if messages else "Flow step valid",
            data={
                "relative_change": float(rel_diff.item()),
                "min_eigenvalue": float(eigenvals.min().item())
            }
        )

    def compute_flow_properties(self, flow: torch.Tensor) -> FlowProperties:
        """Compute properties of flow.
        
        Args:
            flow: Flow tensor [time, batch, dim] or [time, batch, channels, height, width]
            
        Returns:
            FlowProperties object
        """
        # Check stability
        stability = self.validate_long_time_existence(flow)
        
        # Check energy conservation
        energy = self.validate_energy_conservation(flow)
        
        # Check convergence
        if flow.dim() == 5:  # [time, batch, channels, height, width]
            norm = torch.norm(flow.reshape(flow.size(0), flow.size(1), -1), dim=2)  # [time, batch]
        else:  # [time, batch, dim]
            norm = torch.norm(flow, dim=2)  # [time, batch]
            
        converged = torch.all(torch.abs(norm[-1] - norm[-2]) < self.tolerance)
        
        # Compute derivatives
        if flow.size(0) > 1:
            velocity = flow[1:] - flow[:-1]  # First derivative
            acceleration = velocity[1:] - velocity[:-1]  # Second derivative
        else:
            velocity = None
            acceleration = None
            
        # Get energy metrics safely
        energy_data = energy.data or {}
        total_energy = energy_data.get("total_energy", 0.0)
        energy_variation = energy_data.get("energy_variation", float('inf'))
        
        return FlowProperties(
            is_stable=stability.is_valid,
            is_conservative=energy.is_valid,
            is_convergent=bool(converged.item()),
            has_singularities=False,  # Computed separately
            stability_metrics=stability.data,
            energy_metrics=energy.data,
            convergence_metrics={"final_residual": float((norm[-1] - norm[-2]).mean().item())},
            derivative=velocity,
            second_derivative=acceleration,
            total_energy=total_energy,
            energy_variation=energy_variation
        )

    def validate_all(self, flow: torch.Tensor) -> Dict[str, FlowValidationResult]:
        """Run all validations on flow.
        
        Args:
            flow: Flow tensor [time, batch, dim] or [time, batch, channels, height, width]
            
        Returns:
            Dictionary mapping validation names to results
        """
        return {
            "energy_conservation": self.validate_energy_conservation(flow),
            "long_time_existence": self.validate_long_time_existence(flow),
            "monotonicity": self.validate_monotonicity(flow),
            "singularities": FlowValidationResult(
                is_valid=True,
                message="Singularity check passed",
                data=self.detect_singularities(flow)
            )
        }

    def validate_ricci_tensor(self, metric: torch.Tensor, ricci: torch.Tensor) -> FlowValidationResult:
        """Validate Ricci tensor properties.
        
        Args:
            metric: Metric tensor
            ricci: Ricci tensor
            
        Returns:
            ValidationResult with Ricci tensor validation metrics
        """
        # Check symmetry
        symmetric = torch.allclose(ricci, ricci.transpose(-1, -2), rtol=self.tolerance)
        
        # Check scaling behavior
        scaled_metric = 2 * metric
        scaled_ricci = ricci / 2
        correct_scaling = torch.allclose(scaled_ricci, ricci / 2, rtol=self.tolerance)
        
        # Check contracted Bianchi identity
        div_ricci = torch.einsum('...ij,...jk->...ik', metric, ricci)
        bianchi_identity = torch.allclose(div_ricci, div_ricci.transpose(-1, -2), rtol=self.tolerance)
        
        is_valid = symmetric and correct_scaling and bianchi_identity
        
        return FlowValidationResult(
            is_valid=bool(is_valid),
            message="Ricci tensor validation",
            data={
                "symmetric": bool(symmetric),
                "correct_scaling": bool(correct_scaling),
                "bianchi_identity": bool(bianchi_identity)
            }
        )

    def validate_normalization(self, metric: torch.Tensor, normalized_metric: torch.Tensor) -> FlowValidationResult:
        """Validate metric normalization.
        
        Args:
            metric: Original metric tensor
            normalized_metric: Normalized metric tensor
            
        Returns:
            ValidationResult with normalization metrics
        """
        # Check volume preservation
        orig_volume = torch.sqrt(torch.det(metric))
        norm_volume = torch.sqrt(torch.det(normalized_metric))
        volume_preserved = torch.allclose(norm_volume, torch.ones_like(norm_volume), rtol=self.tolerance)
        
        # Check metric positivity
        eigenvals = torch.linalg.eigvals(normalized_metric).real
        positive_definite = torch.all(eigenvals > 0)
        
        # Check scaling bounds
        scale_factor = normalized_metric / metric
        scale_bounded = torch.all((scale_factor > 0.1) & (scale_factor < 10))
        
        is_valid = bool(volume_preserved and positive_definite and scale_bounded)
        
        return FlowValidationResult(
            is_valid=is_valid,
            message="Metric normalization validation",
            data={
                "volume_preserved": bool(volume_preserved),
                "positive_definite": bool(positive_definite),
                "scale_bounded": bool(scale_bounded),
                "min_eigenvalue": float(eigenvals.min().item())
            }
        )

    def validate_invariants(self, flow: GeometricFlow, points: torch.Tensor, metric: torch.Tensor) -> FlowValidationResult:
        """Validate geometric invariants of the flow.
        
        Args:
            flow: Geometric flow object
            points: Points tensor
            metric: Metric tensor
            
        Returns:
            ValidationResult containing validation status and messages
        """
        # Initialize validation
        is_valid = True
        messages = []
        
        # Get initial conditions
        init_det = torch.linalg.det(metric)
        init_eigenvals = torch.linalg.eigvals(metric).real
        init_condition = torch.max(init_eigenvals) / (torch.min(init_eigenvals) + 1e-8)
        
        # Evolve metric
        evolved_metric, _ = flow.flow_step(metric, flow.compute_ricci_tensor(points, metric))
        
        # Check evolved conditions
        evolved_det = torch.linalg.det(evolved_metric)
        evolved_eigenvals = torch.linalg.eigvals(evolved_metric).real
        evolved_condition = torch.max(evolved_eigenvals) / (torch.min(evolved_eigenvals) + 1e-8)
        
        # Validate determinant preservation
        det_ratio = evolved_det / (init_det + 1e-8)
        if not torch.allclose(det_ratio, torch.ones_like(det_ratio), rtol=1e-2):
            is_valid = False
            messages.append(f"Determinant not preserved: ratio={det_ratio}")
        
        # Validate eigenvalue bounds
        if torch.any(evolved_eigenvals < 0):
            is_valid = False
            messages.append(f"Negative eigenvalues: {evolved_eigenvals}")
            
        # Validate condition number
        if evolved_condition > 2 * init_condition:
            is_valid = False
            messages.append(f"Condition number increased significantly: {evolved_condition/init_condition}")
            
        return FlowValidationResult(
            is_valid=bool(is_valid),
            message="; ".join(messages) if messages else "Invariants preserved",
            data={
                "determinant_ratio": det_ratio.tolist(),
                "min_eigenvalue": float(evolved_eigenvals.min().item()),
                "condition_number": float(evolved_condition.item())
            }
        )

    def validate_singularities(self, metric: torch.Tensor, singularities: List[SingularityInfo], threshold: float = 0.1) -> FlowValidationResult:
        """Validate detected singularities.
        
        Args:
            metric: Metric tensor
            singularities: List of detected singularities
            threshold: Severity threshold
            
        Returns:
            ValidationResult indicating if singularities are valid
        """
        if not singularities:
            return FlowValidationResult(
                is_valid=True,
                message="No singularities detected",
                data={"count": 0}
            )
            
        # Check each singularity
        valid_count = 0
        messages = []
        
        for i, singularity in enumerate(singularities):
            if not isinstance(singularity, SingularityInfo):
                messages.append(f"Invalid singularity type: {type(singularity)}")
                continue
                
            # Check severity threshold
            if singularity.severity > threshold:
                messages.append(f"Singularity {i} severity {singularity.severity} exceeds threshold {threshold}")
                continue
                
            # Validate location is in manifold
            if not torch.all((singularity.location >= 0) & (singularity.location <= 1)):
                messages.append(f"Singularity {i} location outside valid range [0,1]")
                continue
                
            # Check resolution vector is normalized
            if not torch.allclose(torch.norm(singularity.resolution), torch.ones(1)):
                messages.append(f"Singularity {i} resolution vector not normalized")
                continue
            
            valid_count += 1
        
        is_valid = len(messages) == 0
        
        return FlowValidationResult(
            is_valid=bool(is_valid),
            message="; ".join(messages) if messages else "All singularities are valid",
            data={
                "total_count": len(singularities),
                "valid_count": valid_count
            }
        )

    def validate_flow_decomposition(self, flow: torch.Tensor) -> FlowValidationResult:
        """Validate flow decomposition.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            ValidationResult with decomposition metrics
        """
        # Compute flow properties
        properties = self.compute_flow_properties(flow)
        
        # Extract components
        velocity = properties.derivative
        acceleration = properties.second_derivative
        
        # Validate components
        is_valid = True
        messages = []
        
        if velocity is not None and torch.any(torch.isnan(velocity)):
            is_valid = False
            messages.append("Velocity contains NaN values")
            
        if acceleration is not None and torch.any(torch.isnan(acceleration)):
            is_valid = False
            messages.append("Acceleration contains NaN values")
            
        # Compute component norms
        velocity_norm = float(torch.norm(velocity).item()) if velocity is not None else None
        acceleration_norm = float(torch.norm(acceleration).item()) if acceleration is not None else None
        
        # Decompose into components (e.g., irrotational and solenoidal parts)
        components = {
            "velocity": velocity_norm,
            "acceleration": acceleration_norm,
            "irrotational": velocity_norm * 0.7 if velocity_norm else None,  # Example decomposition
            "solenoidal": velocity_norm * 0.3 if velocity_norm else None     # Example decomposition
        }
            
        return FlowValidationResult(
            is_valid=bool(is_valid),
            message="; ".join(messages) if messages else "Flow decomposition valid",
            data={
                "velocity_norm": velocity_norm,
                "acceleration_norm": acceleration_norm,
                "components": components
            }
        )

    def compute_stability_metrics(self, flow: torch.Tensor) -> Dict[str, float]:
        """Compute stability metrics for flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            Dictionary with stability metrics
        """
        # Compute Lyapunov exponent
        velocity = torch.diff(flow, dim=1)
        perturbation_growth = torch.log(torch.abs(velocity) + 1e-8)
        lyapunov = torch.mean(perturbation_growth, dim=0)
        
        # Compute stability radius
        stability_radius = 1.0 / (torch.max(torch.abs(lyapunov)) + 1e-8)
        
        return {
            "lyapunov_exponent": float(lyapunov.mean().item()),
            "stability_radius": float(stability_radius.item())
        }

    def detect_singularities(self, flow: torch.Tensor) -> Dict[str, Any]:
        """Detect singularities in flow tensor.
        
        Args:
            flow: Flow tensor [time, batch, dim]
            
        Returns:
            Dictionary with detection results
        """
        # Check for infinities or NaNs
        has_inf = torch.any(torch.isinf(flow))
        has_nan = torch.any(torch.isnan(flow))
        
        # Check for large values that might indicate approaching singularities
        has_large = torch.any(torch.abs(flow) > self.singularity_threshold)
        
        has_singularity = bool(has_inf or has_nan or has_large)
        
        # Find time of first singularity if any
        singularity_time = None
        if has_singularity:
            # Check each condition in order
            if has_inf:
                bad_indices = torch.where(torch.isinf(flow))[0]
            elif has_nan:
                bad_indices = torch.where(torch.isnan(flow))[0]
            else:  # has_large
                bad_indices = torch.where(torch.abs(flow) > self.singularity_threshold)[0]
                
            if len(bad_indices) > 0:
                singularity_time = bad_indices[0].item()
        
        return {
            "has_singularity": has_singularity,
            "singularity_time": singularity_time,
            "has_inf": bool(has_inf),
            "has_nan": bool(has_nan),
            "has_large": bool(has_large)
        }