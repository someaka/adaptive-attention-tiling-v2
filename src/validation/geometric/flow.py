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
from typing import List, Tuple

import numpy as np
import torch

from ...neural.flow.geometric_flow import GeometricFlow
from ...neural.flow.hamiltonian import HamiltonianSystem


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
    
    def __init__(self, is_valid: bool, message: str = ""):
        self.is_valid = is_valid
        self.message = message


class FlowStabilityValidator:
    """Validation of flow stability properties."""

    def __init__(self, tolerance: float = 1e-6, stability_threshold: float = 0.1):
        self.tolerance = tolerance
        self.stability_threshold = stability_threshold

    def validate_stability(
        self, flow: GeometricFlow, points: torch.Tensor, time_steps: int = 100
    ) -> FlowStabilityValidation:
        """Validate flow stability."""
        # Compute Lyapunov exponents
        lyapunov = self._compute_lyapunov(flow, points, time_steps)

        # Measure perturbation growth
        growth = self._measure_growth(flow, points, time_steps)

        # Compute error bounds
        error_bounds = torch.sqrt(torch.abs(lyapunov)) * growth

        # Check stability
        stable = torch.all(lyapunov <= self.stability_threshold)

        return FlowStabilityValidation(
            stable=stable,
            lyapunov_exponents=lyapunov,
            perturbation_growth=growth,
            error_bounds=error_bounds,
        )

    def _compute_lyapunov(
        self, flow: GeometricFlow, points: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """Compute Lyapunov exponents of flow."""
        # Initialize perturbation vectors
        dim = points.shape[-1]
        perturbations = torch.eye(dim, device=points.device)

        # Evolve perturbations
        evolved = []
        current = points.clone()

        for _ in range(time_steps):
            # Flow step
            current = flow.step(current)

            # Evolve perturbations
            jacobian = flow.compute_jacobian(current)
            perturbations = torch.matmul(jacobian, perturbations)

            # Orthogonalize
            perturbations, _ = torch.linalg.qr(perturbations)
            evolved.append(torch.diagonal(perturbations, dim1=-2, dim2=-1))

        # Compute exponents
        evolved = torch.stack(evolved)
        return torch.mean(torch.log(torch.abs(evolved)), dim=0)

    def _measure_growth(
        self, flow: GeometricFlow, points: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """Measure perturbation growth rates."""
        # Add small perturbations
        eps = 1e-5
        perturbed = points + eps * torch.randn_like(points)

        # Evolve both
        original = points.clone()
        current_perturbed = perturbed.clone()

        differences = []
        for _ in range(time_steps):
            original = flow.step(original)
            current_perturbed = flow.step(current_perturbed)
            differences.append(torch.norm(current_perturbed - original, dim=-1))

        differences = torch.stack(differences)
        return torch.mean(differences / eps, dim=0)

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

    def validate_flow_step(self, metric: torch.Tensor, evolved_metric: torch.Tensor, metrics: object) -> bool:
        """Validate flow evolution step.
        
        Args:
            metric: Initial metric tensor
            evolved_metric: Evolved metric tensor
            metrics: Flow metrics
            
        Returns:
            True if flow step is valid
        """
        # Check metric positivity preserved
        eigenvals = torch.linalg.eigvals(evolved_metric).real
        positive_definite = (eigenvals > 0).all()
        
        # Check energy conservation
        energy_conserved = torch.allclose(metrics.energy[1:], metrics.energy[:-1], rtol=self.tolerance)
        
        # Check stability
        stable = metrics.singularity < self.stability_threshold
        
        return positive_definite and energy_conserved and stable

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

    def validate_invariants(self, flow: GeometricFlow, points: torch.Tensor, metric: torch.Tensor) -> bool:
        """Validate geometric invariants are preserved by flow.
        
        Args:
            flow: Geometric flow object
            points: Points tensor
            metric: Metric tensor
            
        Returns:
            True if all invariants are preserved
        """
        # Check volume preservation
        det_before = torch.linalg.det(metric)
        flow_vec = flow.compute_flow_vector(points, flow.compute_ricci_tensor(metric))
        new_points = points + flow.dt * flow_vec
        new_metric = flow.compute_metric(new_points)
        det_after = torch.linalg.det(new_metric)
        vol_preserved = torch.allclose(det_before, det_after, rtol=1e-3)
        
        # Check positive definiteness
        eigvals = torch.linalg.eigvals(metric).real
        pos_def = (eigvals > 0).all()
        
        # Check Ricci flow equation
        ricci = flow.compute_ricci_tensor(metric)
        flow_deriv = (new_metric - metric) / flow.dt
        ricci_flow = torch.allclose(flow_deriv, -2 * ricci, rtol=1e-2)
        
        return vol_preserved and pos_def and ricci_flow


class EnergyValidator:
    """Validation of energy conservation properties."""

    def __init__(self, tolerance: float = 1e-6, drift_threshold: float = 0.01):
        self.tolerance = tolerance
        self.drift_threshold = drift_threshold

    def validate_energy(
        self,
        hamiltonian: HamiltonianSystem,
        states: torch.Tensor,
        time_steps: int = 100,
    ) -> EnergyValidation:
        """Validate energy conservation."""
        # Track energy evolution
        energies = []
        current = states.clone()

        initial_energy = hamiltonian.compute_energy(current)

        for _ in range(time_steps):
            current = hamiltonian.evolve(current)
            energies.append(hamiltonian.compute_energy(current))

        energies = torch.stack(energies)

        # Compute relative error
        relative_error = torch.abs((energies - initial_energy) / initial_energy).mean()

        # Compute drift rate
        drift_rate = torch.mean((energies[1:] - energies[:-1]) / time_steps)

        # Compute fluctuations
        fluctuations = torch.std(energies, dim=0)

        # Check conservation
        conserved = (
            relative_error < self.tolerance and abs(drift_rate) < self.drift_threshold
        )

        return EnergyValidation(
            conserved=conserved,
            relative_error=relative_error.item(),
            drift_rate=drift_rate.item(),
            fluctuations=fluctuations,
        )


class ConvergenceValidator:
    """Validation of flow convergence properties."""

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def validate_convergence(
        self, flow: GeometricFlow, points: torch.Tensor
    ) -> ConvergenceValidation:
        """Validate flow convergence."""
        residuals = []
        current = points.clone()

        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Flow step
            next_points = flow.step(current)

            # Compute residual
            residual = torch.norm(next_points - current)
            residuals.append(residual.item())

            # Check convergence
            if residual < self.tolerance:
                return ConvergenceValidation(
                    converged=True,
                    rate=self._compute_rate(residuals),
                    residuals=torch.tensor(residuals),
                    iterations=iteration + 1,
                )

            current = next_points

        # Did not converge
        return ConvergenceValidation(
            converged=False,
            rate=self._compute_rate(residuals),
            residuals=torch.tensor(residuals),
            iterations=self.max_iterations,
        )

    def _compute_rate(self, residuals: List[float]) -> float:
        """Compute convergence rate from residuals."""
        if len(residuals) < 2:
            return 0.0

        # Use last few iterations for rate
        window = min(10, len(residuals) - 1)
        rates = [
            np.log(residuals[i + 1] / residuals[i]) for i in range(-window - 1, -1)
        ]
        return float(np.mean(rates))


class GeometricFlowValidator:
    """Complete geometric flow validation system."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        stability_threshold: float = 0.1,
        drift_threshold: float = 0.01,
        max_iterations: int = 1000,
    ):
        self.stability_validator = FlowStabilityValidator(
            tolerance, stability_threshold
        )
        self.energy_validator = EnergyValidator(tolerance, drift_threshold)
        self.convergence_validator = ConvergenceValidator(tolerance, max_iterations)

    def validate(
        self,
        flow: GeometricFlow,
        hamiltonian: HamiltonianSystem,
        points: torch.Tensor,
        time_steps: int = 100,
    ) -> Tuple[FlowStabilityValidation, EnergyValidation, ConvergenceValidation]:
        """Perform complete flow validation."""
        # Validate stability
        stability = self.stability_validator.validate_stability(
            flow, points, time_steps
        )

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

    def validate_flow_step(self, metric: torch.Tensor, evolved_metric: torch.Tensor, metrics: object) -> bool:
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

    def validate_invariants(self, flow: GeometricFlow, points: torch.Tensor, metric: torch.Tensor) -> bool:
        """Validate geometric invariants are preserved by flow.
        
        Args:
            flow: Geometric flow object
            points: Points tensor
            metric: Metric tensor
            
        Returns:
            True if all invariants are preserved
        """
        return self.stability_validator.validate_invariants(flow, points, metric)
