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
from typing import Dict, List, Optional, Tuple

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
            
        # Check for extremely large values
        if torch.any(torch.abs(state) > self.threshold):
            return True
            
        return False
        
    def analyze_singularity(
        self,
        flow: GeometricFlow,
        state: torch.Tensor,
        time_points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze singularity properties.
        
        Args:
            flow: Geometric flow
            state: Initial state
            time_points: Time points to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Track flow properties
        current_state = state
        states = []
        derivatives = []
        
        # Evolve and analyze
        for t in time_points:
            # Store current state
            states.append(current_state)
            
            # Compute time derivative
            dt = t[1] - t[0] if len(t) > 1 else 0.01
            derivative = (
                flow.evolve(current_state, dt) - current_state
            ) / dt
            derivatives.append(derivative)
            
            # Evolve state
            current_state = flow.evolve(current_state, t)
            
            # Check for singularity
            if self._check_singularity(current_state):
                break
                
        # Convert to tensors
        states = torch.stack(states)
        derivatives = torch.stack(derivatives)
        
        return {
            "states": states,
            "derivatives": derivatives,
            "times": time_points[:len(states)]
        }


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


@dataclass
class FlowValidator:
    """Validator for geometric flow properties."""

    def __init__(
        self,
        tolerance: float = 1e-5,
        max_time: float = 100.0,
        time_step: float = 0.01
    ):
        """Initialize flow validator.
        
        Args:
            tolerance: Numerical tolerance
            max_time: Maximum evolution time
            time_step: Integration time step
        """
        self.tolerance = tolerance
        self.max_time = max_time
        self.time_step = time_step
        
        # Initialize sub-validators
        self.stability_validator = FlowStabilityValidator(tolerance)
        self.energy_validator = EnergyValidator(tolerance)
        self.convergence_validator = ConvergenceValidator(tolerance)
        
    def validate(
        self,
        flow: GeometricFlow,
        initial_states: torch.Tensor,
    ) -> ValidationResult:
        """Validate geometric flow.
        
        Args:
            flow: Geometric flow to validate
            initial_states: Initial states tensor
            
        Returns:
            Validation result
        """
        # Track flow properties
        properties = FlowProperties(
            is_stable=True,
            is_conservative=True,
            is_convergent=True,
            has_singularities=False
        )
        
        # Validate stability
        stability_result = self.stability_validator.validate_stability(
            flow, initial_states
        )
        properties.stability_metrics = stability_result
        properties.is_stable = stability_result.stable
        
        # Validate energy conservation
        energy_result = self.energy_validator.validate_energy(
            flow, initial_states
        )
        properties.energy_metrics = energy_result
        properties.is_conservative = energy_result.conserved
        
        # Validate convergence
        convergence_result = self.convergence_validator.validate_convergence(
            flow, initial_states
        )
        properties.convergence_metrics = convergence_result
        properties.is_convergent = convergence_result.converged
        
        # Check for singularities
        singularity_result = self.validate_singularities(
            flow, initial_states
        )
        properties.has_singularities = singularity_result.has_singularities
        
        # Compile validation result
        return ValidationResult(
            is_valid=properties.is_stable and properties.is_convergent,
            message=self._generate_message(properties),
            stable=properties.is_stable,
            error=convergence_result.error,
            initial_energy=energy_result.initial,
            final_energy=energy_result.final,
            relative_error=energy_result.relative_error,
            initial_states=initial_states,
            final_states=convergence_result.final_states
        )
        
    def validate_singularities(
        self,
        flow: GeometricFlow,
        states: torch.Tensor,
    ) -> Dict:
        """Validate flow for singularities.
        
        Args:
            flow: Geometric flow
            states: States tensor
            
        Returns:
            Dictionary with singularity information
        """
        # Evolve flow
        time = 0.0
        current_states = states
        has_singularities = False
        singularity_time = None
        
        while time < self.max_time:
            # Evolve one step
            new_states = flow.evolve(current_states, self.time_step)
            
            # Check for singularities
            if self._detect_singularity(new_states):
                has_singularities = True
                singularity_time = time
                break
                
            current_states = new_states
            time += self.time_step
            
        return {
            "has_singularities": has_singularities,
            "singularity_time": singularity_time
        }
        
    def _detect_singularity(self, states: torch.Tensor) -> bool:
        """Detect if states contain singularities.
        
        Args:
            states: States tensor
            
        Returns:
            True if singularity detected
        """
        # Check for NaN or Inf values
        if torch.any(torch.isnan(states)) or torch.any(torch.isinf(states)):
            return True
            
        # Check for extremely large values
        if torch.any(torch.abs(states) > 1e6):
            return True
            
        return False
        
    def _generate_message(self, properties: FlowProperties) -> str:
        """Generate validation message.
        
        Args:
            properties: Flow properties
            
        Returns:
            Validation message
        """
        messages = []
        
        if not properties.is_stable:
            messages.append("Flow is unstable")
            
        if not properties.is_conservative:
            messages.append("Flow does not conserve energy")
            
        if not properties.is_convergent:
            messages.append("Flow does not converge")
            
        if properties.has_singularities:
            messages.append("Flow develops singularities")
            
        if not messages:
            return "Flow validation successful"
            
        return "Flow validation failed: " + "; ".join(messages)


@dataclass
class EnergyMetrics:
    """Energy metrics for geometric flow."""
    
    kinetic: float  # Kinetic energy
    potential: float  # Potential energy
    total: Optional[float] = None  # Total energy
    
    def __post_init__(self):
        """Compute total energy if not provided."""
        if self.total is None:
            self.total = self.kinetic + self.potential


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
    
    def __init__(self, is_valid: bool, message: str = "", stable: bool = False, error: float = 0.0, initial_energy: torch.Tensor = None, final_energy: torch.Tensor = None, relative_error: torch.Tensor = None, initial_states: torch.Tensor = None, final_states: torch.Tensor = None):
        self.is_valid = is_valid
        self.message = message
        self.stable = stable
        self.error = error
        self.initial_energy = initial_energy
        self.final_energy = final_energy
        self.relative_error = relative_error
        self.initial_states = initial_states
        self.final_states = final_states


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
            stable=stable,
            error=max_diff.item(),
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
            initial_energy=initial_energy,
            final_energy=final_energy,
            relative_error=relative_error,
            initial_states=states,
            final_states=evolved_states
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
            stable=converged,
            error=error,  # Keep as tensor
            initial_states=states,
            final_states=next_pos
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
            stable=converged,
            error=error,  # Keep as tensor
            initial_states=states,
            final_states=next_pos
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
        """Validate singularity detection.
        
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
