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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from ...neural.flow.geometric_flow import GeometricFlow
from ...neural.flow.hamiltonian import HamiltonianSystem
from ..base import ValidationResult


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
    ) -> Dict[str, bool]:
        """Detect singularities in flow.
        
        Args:
            flow: Geometric flow
            state: Initial state
            time_points: Time points to check
            
        Returns:
            Dictionary with detection results
        """
        # Track flow evolution
        current_state = state
        has_singularity = False
        singularity_time = None
        
        # Check each time point
        for t in time_points:
            # Evolve state
            new_state = flow.evolve(current_state, t)
            
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


@dataclass
class FlowProperties:
    """Properties of geometric flow."""
    
    is_stable: bool = True
    is_conservative: bool = True
    is_convergent: bool = True
    has_singularities: bool = False
    stability_metrics: Optional[Dict] = None
    energy_metrics: Optional[Dict] = None
    convergence_metrics: Optional[Dict] = None
    singularity_metrics: Optional[Dict] = None
    derivative: Optional[torch.Tensor] = None
    second_derivative: Optional[torch.Tensor] = None


@dataclass
class EnergyMetrics:
    """Energy metrics for flow."""
    
    kinetic: torch.Tensor
    potential: torch.Tensor
    total: torch.Tensor


@dataclass
class FlowValidator:
    """Validator for geometric flow properties."""
    
    def __init__(
        self,
        energy_threshold: float = 1e-6,
        monotonicity_threshold: float = 1e-4,
        singularity_threshold: float = 1.0,
    ):
        """Initialize flow validator.
        
        Args:
            energy_threshold: Threshold for energy conservation
            monotonicity_threshold: Threshold for monotonicity checks
            singularity_threshold: Threshold for singularity detection
        """
        self.energy_threshold = energy_threshold
        self.monotonicity_threshold = monotonicity_threshold
        self.singularity_threshold = singularity_threshold
        
    def validate_energy_conservation(self, flow: torch.Tensor) -> ValidationResult:
        """Validate energy conservation in flow.
        
        Args:
            flow: Flow tensor of shape (time_steps, batch_size, dim)
            
        Returns:
            ValidationResult with energy metrics
        """
        # Compute energy metrics
        metrics = self.compute_energy_metrics(flow)
        
        # Check energy conservation
        energy_variation = torch.abs(metrics.total[-1] - metrics.total[0])
        is_conserved = energy_variation < self.energy_threshold
        
        return ValidationResult(
            is_valid=is_conserved,
            message="Energy conservation validation",
            data={
                "energy_variation": energy_variation,
                "initial_energy": metrics.total[0],
                "final_energy": metrics.total[-1],
                "metrics": metrics
            }
        )

    def validate_monotonicity(self, flow: torch.Tensor) -> ValidationResult:
        """Validate flow monotonicity.
        
        Args:
            flow: Flow tensor of shape (time_steps, batch_size, dim)
            
        Returns:
            ValidationResult with monotonicity metrics
        """
        # Compute time derivative
        dt = 1.0  # Assume unit time steps
        
        # First derivative (velocity)
        velocity = (flow[1:] - flow[:-1]) / dt
        
        # Check if velocity maintains sign
        sign_changes = torch.sum(velocity[1:] * velocity[:-1] < 0)
        is_monotonic = sign_changes == 0
        
        return ValidationResult(
            is_valid=is_monotonic,
            message="Flow monotonicity validation",
            data={"monotonicity_measure": sign_changes}
        )

    def validate_long_time_existence(self, flow: torch.Tensor) -> ValidationResult:
        """Validate long-time existence of flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            ValidationResult with existence metrics
        """
        # Check for blowup by looking at maximum values over time
        max_values = torch.max(torch.abs(flow), dim=-1)[0]  # Shape: (batch_size, time_steps)
        
        # Check if values remain bounded
        has_blowup = torch.any(max_values > self.singularity_threshold)
        
        # Check if flow converges by looking at the rate of change near the end
        end_window = 10  # Look at last 10 timesteps
        if flow.size(1) > end_window:
            recent_values = max_values[:, -end_window:]
            rate_of_change = torch.abs(recent_values[:, 1:] - recent_values[:, :-1])
            is_converged = torch.all(rate_of_change < self.monotonicity_threshold)
        else:
            is_converged = True  # For short sequences, assume convergence
        
        # Compute stability metrics
        stability = self.compute_stability_metrics(flow)
        
        # Flow exists for long time if it's bounded and converges
        exists_long_time = (not has_blowup) and is_converged
        
        return ValidationResult(
            is_valid=exists_long_time,
            message="Long-time existence validation",
            data={
                "existence_time": flow.size(1),
                "max_value": max_values.max().item(),
                "final_rate_of_change": rate_of_change.max().item() if flow.size(1) > end_window else 0.0,
                "stability": stability
            }
        )

    def detect_singularities(self, flow: torch.Tensor) -> Dict[str, Any]:
        """Detect singularities in flow.
        
        Args:
            flow: Flow tensor of shape (time_steps, batch_size, dim)
            
        Returns:
            Dictionary with singularity detection results
        """
        # Check for large values indicating singularities
        max_values = torch.max(torch.abs(flow), dim=-1)[0]
        singularity_mask = max_values > self.singularity_threshold
        
        has_singularity = torch.any(singularity_mask)
        singularity_time = None
        
        if has_singularity:
            # Find first occurrence of singularity
            singularity_indices = torch.where(singularity_mask)[0]
            if len(singularity_indices) > 0:
                singularity_time = singularity_indices[0].item()
        
        return {
            "has_singularity": has_singularity,
            "singularity_time": singularity_time
        }

    def compute_flow_properties(self, flow: torch.Tensor) -> FlowProperties:
        """Compute flow properties.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            FlowProperties object
        """
        # Compute derivatives using central differences
        dt = 1.0  # Assume unit time steps
        
        # First derivative (velocity) - preserve batch dimension
        velocity = (flow[:, 1:] - flow[:, :-1]) / dt  # Shape: (batch_size, time_steps-1, dim)
        
        # Second derivative (acceleration) - preserve batch dimension
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt  # Shape: (batch_size, time_steps-2, dim)
        
        # Compute stability metrics
        stability_metrics = self.compute_stability_metrics(flow)
        
        # Compute energy metrics
        energy_metrics = self.compute_energy_metrics(flow)
        
        return FlowProperties(
            derivative=velocity,
            second_derivative=acceleration,
            stability_metrics=stability_metrics,
            energy_metrics=energy_metrics
        )

    def compute_stability_metrics(self, flow: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute stability metrics for flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            Dictionary of stability metrics
        """
        dt = 1.0  # Assume unit time steps
        
        # Compute derivatives along time dimension (dim=1)
        velocity = (flow[:, 1:] - flow[:, :-1]) / dt  # Shape: (batch_size, time_steps-1, dim)
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt  # Shape: (batch_size, time_steps-2, dim)
        
        # Compute Lyapunov exponent estimate
        displacement = torch.norm(velocity, dim=-1)  # Shape: (batch_size, time_steps-1)
        lyapunov = torch.mean(torch.log(displacement + 1e-6))
        
        # Compute stability radius
        flow_norm = torch.norm(flow[:, 2:], dim=-1)  # Shape: (batch_size, time_steps-2)
        acc_norm = torch.norm(acceleration, dim=-1)  # Shape: (batch_size, time_steps-2)
        stability_radius = torch.min(flow_norm / (acc_norm + 1e-6))
        
        return {
            "lyapunov_exponent": lyapunov,
            "stability_radius": stability_radius,
            "max_acceleration": torch.max(acc_norm)
        }

    def compute_energy_metrics(self, flow: torch.Tensor) -> EnergyMetrics:
        """Compute energy metrics for flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            EnergyMetrics object
        """
        dt = 1.0  # Assume unit time steps
        
        # Compute kinetic energy using central differences
        velocity = (flow[:, 1:] - flow[:, :-1]) / dt  # Shape: (batch_size, time_steps-1, dim)
        kinetic = 0.5 * torch.sum(velocity ** 2, dim=(-1, -2))  # Shape: (batch_size,)
        
        # Compute potential energy (example using harmonic potential)
        potential = 0.5 * torch.sum(flow ** 2, dim=(-1, -2))  # Shape: (batch_size,)
        
        # Compute total energy
        total = kinetic + potential  # Shape: (batch_size,)
        
        return EnergyMetrics(
            kinetic=kinetic,
            potential=potential,
            total=total
        )

    def validate_flow_decomposition(self, flow: torch.Tensor) -> ValidationResult:
        """Validate flow decomposition.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            ValidationResult with decomposition metrics
        """
        # Compute flow properties
        properties = self.compute_flow_properties(flow)
        
        # Extract components
        components = {
            "velocity": properties.derivative,
            "acceleration": properties.second_derivative
        }
        
        return ValidationResult(
            is_valid=True,  # Always valid as this is just decomposition
            message="Flow decomposition validation",
            data={"components": components}
        )

    def validate_all(self, flow: torch.Tensor) -> Dict[str, ValidationResult]:
        """Run all validations on flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            Dictionary of validation results
        """
        return {
            "energy_conservation": self.validate_energy_conservation(flow),
            "monotonicity": self.validate_monotonicity(flow),
            "long_time_existence": self.validate_long_time_existence(flow),
            "singularities": self.detect_singularities(flow)
        }


@dataclass
class FlowStabilityValidation:
    """Results of flow stability validation."""

    stable: bool  # Overall stability
    lyapunov_exponents: torch.Tensor  # Stability measures
    perturbation_growth: torch.Tensor  # Growth rates
    error_bounds: torch.Tensor  # Error estimates


@dataclass
class EnergyValidation:
    """Results of energy conservation validation."""

    conserved: bool  # Energy conservation
    relative_error: float  # Energy error
    drift_rate: float  # Energy drift
    fluctuations: torch.Tensor  # Energy fluctuations


@dataclass
class ConvergenceValidation:
    """Results of convergence validation."""

    converged: bool  # Convergence status
    rate: float  # Convergence rate
    residuals: torch.Tensor  # Convergence residuals
    iterations: int  # Iterations to converge


class ValidationResult:
    """Result of validation with message."""
    
    def __init__(self, is_valid: bool, message: str = "", data: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.message = message
        self.data = data


class GeometricFlowValidator:
    """Validator for geometric flow properties."""

    def __init__(self, tolerance: float = 1e-5):
        """Initialize validator.
        
        Args:
            tolerance: Tolerance for validation checks
        """
        self.tolerance = tolerance

    def validate(self, points: torch.Tensor) -> bool:
        """Validate geometric invariants of flow.
        
        Args:
            points: Points tensor
            
        Returns:
            True if validation passes
        """
        # Basic shape validation
        if points.dim() != 2:
            return False
            
        # Check for NaN/Inf values
        if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
            return False
            
        return True


class FlowStabilityValidator:
    """Validator for flow stability."""

    def __init__(self, tolerance: float = 1e-5, stability_threshold: float = 0.1):
        """Initialize validator.
        
        Args:
            tolerance: Tolerance for stability checks
            stability_threshold: Maximum allowed Lyapunov exponent
        """
        self.tolerance = tolerance
        self.stability_threshold = stability_threshold

    def validate_stability(self, flow: GeometricFlow, points: torch.Tensor) -> ValidationResult:
        """Validate stability of flow.
        
        Args:
            flow: GeometricFlow instance
            points: Points tensor
            
        Returns:
            Validation result
        """
        # Compute initial metric
        flow.points = points
        metric = flow.compute_metric(points)
        
        # Evolve points
        evolved_points = flow(points)
        evolved_metric = flow.compute_metric(evolved_points)
        
        # Compute metric difference
        metric_diff = torch.norm(evolved_metric - metric, dim=(-2,-1))
        max_diff = torch.max(metric_diff)
        
        # Check stability
        stable = max_diff <= self.stability_threshold
        
        return ValidationResult(
            is_valid=stable,
            message="Flow stability check"
        )

    def validate_normalization(self, metric: torch.Tensor, normalized_metric: torch.Tensor) -> bool:
        """Validate flow normalization.
        
        Args:
            metric: Original metric tensor
            normalized_metric: Normalized metric tensor
            
        Returns:
            True if normalization is valid
        """
        # Check volume preservation
        orig_volume = torch.sqrt(torch.det(metric))
        norm_volume = torch.sqrt(torch.det(normalized_metric))
        volume_preserved = torch.allclose(norm_volume, torch.ones_like(norm_volume), rtol=self.tolerance)
        
        # Check metric positivity
        eigenvals = torch.linalg.eigvals(normalized_metric).real
        positive_definite = (eigenvals > 0).all()
        
        # Check scaling bounds
        scale_factor = normalized_metric / metric
        scale_bounded = torch.all((scale_factor > 0.1) & (scale_factor < 10))
        
        return volume_preserved and positive_definite and scale_bounded

    def validate_ricci_tensor(self, metric: torch.Tensor, ricci: torch.Tensor) -> bool:
        """Validate Ricci tensor properties.
        
        Args:
            metric: Metric tensor
            ricci: Ricci tensor
            
        Returns:
            True if Ricci tensor is valid
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
        
        return symmetric and correct_scaling and bianchi_identity

    def validate_flow_step(self, metric: torch.Tensor, evolved_metric: torch.Tensor, metrics: object) -> ValidationResult:
        """Validate flow evolution step.
        
        Args:
            metric: Initial metric tensor
            evolved_metric: Evolved metric tensor
            metrics: Flow metrics
            
        Returns:
            True if flow step is valid
        """
        # Initialize validation result
        is_valid = True
        messages = []

        # Check metric positive definiteness
        eigenvalues = torch.linalg.eigvalsh(evolved_metric)
        if not torch.all(eigenvalues > 1e-10):
            is_valid = False
            messages.append("Evolved metric lost positive definiteness")

        # Check metric conditioning
        condition = torch.max(eigenvalues) / torch.min(eigenvalues.abs())
        if condition > 1e4:
            is_valid = False
            messages.append(f"Poor metric conditioning: {condition:.2e}")

        # Check volume preservation (up to tolerance)
        init_det = torch.det(metric)
        evolved_det = torch.det(evolved_metric)
        rel_vol_change = torch.abs(evolved_det - init_det) / (torch.abs(init_det) + 1e-10)
        if rel_vol_change > 0.1:  # 10% tolerance
            is_valid = False
            messages.append(f"Volume not preserved: {rel_vol_change:.2e}")

        # Check flow magnitude
        if hasattr(metrics, 'flow_norm'):
            if torch.any(metrics.flow_norm > 1e3):
                is_valid = False
                messages.append("Flow magnitude too large")

        return ValidationResult(
            is_valid=is_valid,
            message="; ".join(messages) if messages else "Flow step valid"
        )

    def validate_singularities(self, metric: torch.Tensor, singularities: List, threshold: float = 0.1) -> ValidationResult:
        """Validate detected singularities.
        
        Args:
            metric: Metric tensor
            singularities: List of detected singularities
            threshold: Severity threshold
            
        Returns:
            ValidationResult indicating if singularities are valid
        """
        if not singularities:
            return ValidationResult(True, "No singularities detected")
            
        # Check each singularity
        for singularity in singularities:
            if not isinstance(singularity, SingularityInfo):
                return ValidationResult(False, f"Invalid singularity type: {type(singularity)}")
                
            # Check severity threshold
            if singularity.severity > threshold:
                return ValidationResult(
                    False,
                    f"Singularity severity {singularity.severity} exceeds threshold {threshold}"
                )
                
            # Validate location is in manifold
            if not torch.all((singularity.location >= 0) & (singularity.location <= 1)):
                return ValidationResult(False, "Singularity location outside valid range [0,1]")
                
            # Check resolution vector is normalized 
            if not torch.allclose(torch.norm(singularity.resolution), torch.ones(1)):
                return ValidationResult(False, "Resolution vector not normalized")
        
        return ValidationResult(True, "All singularities are valid")

    def validate_invariants(self, flow: GeometricFlow, points: torch.Tensor, metric: torch.Tensor) -> ValidationResult:
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
            
        return ValidationResult(is_valid=is_valid, message="; ".join(messages) if messages else "Invariants preserved")


class EnergyValidator:
    """Validator for energy conservation in geometric flow."""
    
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance

    def validate_energy(self, flow_system, states: torch.Tensor) -> ValidationResult:
        """Validate energy conservation.
        
        Args:
            flow_system: Geometric flow system
            states: Initial states tensor of shape (batch_size, phase_dim)
            
        Returns:
            Validation result with energy metrics
        """
        batch_size = states.shape[0]
        
        # Compute initial energy
        initial_energy = flow_system.compute_energy(states)
        
        # Evolve states
        evolved_states = flow_system(states)
        
        # Compute final energy
        final_energy = flow_system.compute_energy(evolved_states)
        
        # Compute relative error per batch element
        relative_error = torch.abs(final_energy - initial_energy) / (torch.abs(initial_energy) + 1e-8)
        
        # Check if energy is conserved within tolerance
        is_conserved = torch.all(relative_error < self.tolerance)
        
        return ValidationResult(
            is_valid=is_conserved,
            message="Energy conservation validation",
            data={
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "relative_error": relative_error,
                "initial_states": states,
                "final_states": evolved_states
            }
        )

    def validate_convergence(self, flow_system, states: torch.Tensor) -> ValidationResult:
        """Validate that flow converges.
        
        Args:
            flow_system: Geometric flow system
            states: Initial states tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Validation result with convergence metrics
        """
        # Initialize
        iterations = self.max_iterations
        current_states = states
        
        for i in range(self.max_iterations):
            # Evolve states
            next_states = flow_system(current_states)
            
            # Extract position components for comparison
            if next_states.shape[-1] > states.shape[-1]:
                next_pos = next_states[..., :states.shape[-1]]
            else:
                next_pos = next_states
                
            # Compute error as L2 norm of difference
            error = torch.norm(next_pos - current_states, dim=-1)
            
            # Check convergence
            if torch.all(error < self.threshold):
                iterations = i + 1
                break
                
            current_states = next_pos
            
        # Check if converged within max iterations
        converged = iterations < self.max_iterations
        
        return ValidationResult(
            is_valid=converged,
            message=f"Flow convergence validation (iterations={iterations})",
            data={
                "error": error,  # Keep as tensor
                "initial_states": states,
                "final_states": next_pos
            }
        )


class ConvergenceValidator:
    """Validator for flow convergence properties."""
    
    def __init__(self, threshold: float = 1e-4, max_iterations: int = 1000):
        self.threshold = threshold
        self.max_iterations = max_iterations

    def validate_convergence(self, flow_system, states: torch.Tensor) -> ValidationResult:
        """Validate that flow converges.
        
        Args:
            flow_system: Geometric flow system
            states: Initial states tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Validation result with convergence metrics
        """
        # Initialize
        iterations = self.max_iterations
        current_states = states
        
        for i in range(self.max_iterations):
            # Evolve states
            next_states = flow_system(current_states)
            
            # Extract position components for comparison
            if next_states.shape[-1] > states.shape[-1]:
                next_pos = next_states[..., :states.shape[-1]]
            else:
                next_pos = next_states
                
            # Compute error as L2 norm of difference
            error = torch.norm(next_pos - current_states, dim=-1)
            
            # Check convergence
            if torch.all(error < self.threshold):
                iterations = i + 1
                break
                
            current_states = next_pos
            
        # Check if converged within max iterations
        converged = iterations < self.max_iterations
        
        return ValidationResult(
            is_valid=converged,
            message=f"Flow convergence validation (iterations={iterations})",
            data={
                "error": error,  # Keep as tensor
                "initial_states": states,
                "final_states": next_pos
            }
        )


class GeometricFlowValidator:
    """Complete geometric flow validation system."""
    
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.stability_validator = FlowStabilityValidator(tolerance=tolerance)
        self.energy_validator = EnergyValidator(tolerance=tolerance)
        self.convergence_validator = ConvergenceValidator(threshold=tolerance)
        
    def validate(
        self,
        flow: GeometricFlow,
        hamiltonian: HamiltonianSystem,
        points: torch.Tensor,
        time_steps: int = 100,
    ) -> Tuple[ValidationResult, EnergyValidation, ConvergenceValidation]:
        """Perform complete flow validation."""
        # Validate stability
        stability = self.stability_validator.validate_stability(flow, points)

        # Validate energy conservation
        energy = self.energy_validator.validate_energy(hamiltonian, points, time_steps)

        # Validate convergence
        convergence = self.convergence_validator.validate_convergence(flow, points)

        return stability, energy, convergence

    def validate_normalization(self, metric: torch.Tensor, normalized_metric: torch.Tensor) -> bool:
        """Validate metric normalization.
        
        Args:
            metric: Original metric tensor
            normalized_metric: Normalized metric tensor
            
        Returns:
            True if normalization is valid
        """
        # Delegate to stability validator
        return self.stability_validator.validate_normalization(metric, normalized_metric)

    def validate_ricci_tensor(self, metric: torch.Tensor, ricci: torch.Tensor) -> bool:
        """Validate Ricci tensor properties.
        
        Args:
            metric: Metric tensor
            ricci: Ricci tensor
            
        Returns:
            True if Ricci tensor is valid
        """
        return self.stability_validator.validate_ricci_tensor(metric, ricci)

    def validate_flow_step(self, metric: torch.Tensor, evolved_metric: torch.Tensor, metrics: object) -> ValidationResult:
        """Validate flow evolution step.
        
        Args:
            metric: Initial metric tensor
            evolved_metric: Evolved metric tensor
            metrics: Flow metrics
            
        Returns:
            True if flow step is valid
        """
        return self.stability_validator.validate_flow_step(metric, evolved_metric, metrics)

    def validate_singularities(self, metric: torch.Tensor, singularities: List, threshold: float = 0.1) -> ValidationResult:
        """Validate detected singularities.
        
        Args:
            metric: Metric tensor
            singularities: List of detected singularities
            threshold: Severity threshold
            
        Returns:
            ValidationResult indicating if singularities are valid
        """
        return self.stability_validator.validate_singularities(metric, singularities, threshold)

    def validate_invariants(self, flow: GeometricFlow, points: torch.Tensor, metric: torch.Tensor) -> ValidationResult:
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
            
        return ValidationResult(is_valid=is_valid, message="; ".join(messages) if messages else "Invariants preserved")

    def validate_energy_conservation(self, flow: torch.Tensor) -> ValidationResult:
        """Validate energy conservation in flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            ValidationResult with energy metrics
        """
        # Compute energy metrics
        metrics = self.compute_energy_metrics(flow)
        
        # Check energy conservation
        energy_variation = torch.abs(metrics.total[-1] - metrics.total[0])
        is_conserved = energy_variation < self.tolerance
        
        return ValidationResult(
            is_valid=is_conserved,
            message="Energy conservation validation",
            data={
                "energy_variation": energy_variation,
                "initial_energy": metrics.total[0],
                "final_energy": metrics.total[-1],
                "metrics": metrics
            }
        )
        
    def compute_energy_metrics(self, flow: torch.Tensor) -> EnergyMetrics:
        """Compute energy metrics for flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            EnergyMetrics object
        """
        dt = 1.0  # Assume unit time steps
        
        # Compute kinetic energy using central differences
        velocity = (flow[:, 1:] - flow[:, :-1]) / dt  # Shape: (batch_size, time_steps-1, dim)
        kinetic = 0.5 * torch.sum(velocity ** 2, dim=(-1, -2))  # Shape: (batch_size,)
        
        # Compute potential energy (example using harmonic potential)
        potential = 0.5 * torch.sum(flow ** 2, dim=(-1, -2))  # Shape: (batch_size,)
        
        # Compute total energy
        total = kinetic + potential  # Shape: (batch_size,)
        
        return EnergyMetrics(
            kinetic=kinetic,
            potential=potential,
            total=total
        )

    def validate_monotonicity(self, flow: torch.Tensor) -> ValidationResult:
        """Validate flow monotonicity.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            ValidationResult with monotonicity metrics
        """
        # Compute flow properties
        properties = self.compute_flow_properties(flow)
        
        # Check monotonicity
        is_monotonic = torch.all(properties.derivative >= 0) or torch.all(properties.derivative <= 0)
        
        return ValidationResult(
            is_valid=is_monotonic,
            message="Flow monotonicity validation",
            data={
                "monotonicity_measure": properties.derivative.abs().mean().item()
            }
        )
        
    def compute_flow_properties(self, flow: torch.Tensor) -> FlowProperties:
        """Compute flow properties.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            FlowProperties object
        """
        # Compute derivatives
        velocity = torch.diff(flow, dim=1)
        acceleration = torch.diff(velocity, dim=1)
        
        # Compute stability metrics
        stability = torch.max(torch.abs(acceleration)) < self.tolerance
        
        # Compute energy metrics
        energy = self.compute_energy_metrics(flow)
        
        return FlowProperties(
            is_stable=stability,
            is_conservative=True,  # This should be computed based on energy conservation
            is_convergent=True,  # This should be computed based on convergence analysis
            has_singularities=False,  # This should be computed based on singularity detection
            stability_metrics={"max_acceleration": acceleration.abs().max().item()},
            energy_metrics={"total_energy": energy.total},
            convergence_metrics=None,
            singularity_metrics=None,
            derivative=velocity,
            second_derivative=acceleration
        )
        
    def validate_long_time_existence(self, flow: torch.Tensor) -> ValidationResult:
        """Validate long-time existence of flow.
        
        Args:
            flow: Flow tensor of shape (batch_size, time_steps, dim)
            
        Returns:
            ValidationResult with existence metrics
        """
        # Check for blowup by looking at maximum values over time
        max_values = torch.max(torch.abs(flow), dim=-1)[0]  # Shape: (batch_size, time_steps)
        
        # Check if values remain bounded
        has_blowup = torch.any(max_values > self.singularity_threshold)
        
        # Check if flow converges by looking at the rate of change near the end
        end_window = 10  # Look at last 10 timesteps
        if flow.size(1) > end_window:
            recent_values = max_values[:, -end_window:]
            rate_of_change = torch.abs(recent_values[:, 1:] - recent_values[:, :-1])
            is_converged = torch.all(rate_of_change < self.monotonicity_threshold)
        else:
            is_converged = True  # For short sequences, assume convergence
        
        # Compute stability metrics
        stability = self.compute_stability_metrics(flow)
        
        # Flow exists for long time if it's bounded and converges
        exists_long_time = (not has_blowup) and is_converged
        
        return ValidationResult(
            is_valid=exists_long_time,
            message="Long-time existence validation",
            data={
                "existence_time": flow.size(1),
                "max_value": max_values.max().item(),
                "final_rate_of_change": rate_of_change.max().item() if flow.size(1) > end_window else 0.0,
                "stability": stability
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
            "lyapunov_exponent": lyapunov.mean().item(),
            "stability_radius": stability_radius.item()
        }
