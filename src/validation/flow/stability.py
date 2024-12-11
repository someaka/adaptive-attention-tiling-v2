"""Flow Stability Validation Implementation.

This module validates flow stability properties:
- Linear stability analysis
- Nonlinear stability
- Structural stability
- Bifurcation analysis
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from ...neural.flow.geometric_flow import GeometricFlow


@dataclass
class LinearStabilityValidation:
    """Results of linear stability validation."""

    stable: bool  # Linear stability
    eigenvalues: torch.Tensor  # Stability eigenvalues
    eigenvectors: torch.Tensor  # Stability modes
    growth_rates: torch.Tensor  # Modal growth rates


@dataclass
class NonlinearStabilityValidation:
    """Results of nonlinear stability validation."""

    stable: bool  # Nonlinear stability
    lyapunov_function: float  # Energy functional
    basin_size: float  # Stability basin
    perturbation_bound: float  # Maximum perturbation


@dataclass
class StructuralStabilityValidation:
    """Results of structural stability validation."""

    stable: bool  # Structural stability
    sensitivity: float  # Parameter sensitivity
    robustness: float  # Perturbation robustness
    bifurcation_distance: float  # Distance to bifurcation


class LinearStabilityValidator:
    """Validation of linear stability properties."""

    def __init__(self, tolerance: float = 1e-6, stability_threshold: float = 0.0):
        self.tolerance = tolerance
        self.stability_threshold = stability_threshold

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> LinearStabilityValidation:
        """Validate linear stability."""
        # Compute Jacobian at state
        jacobian = flow.compute_jacobian(state)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(jacobian)

        # Get real parts for stability
        real_parts = torch.real(eigenvalues)

        # Compute growth rates
        growth_rates = torch.exp(real_parts)

        # Check stability
        stable = torch.all(real_parts <= self.stability_threshold)

        return LinearStabilityValidation(
            stable=stable,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            growth_rates=growth_rates,
        )


class NonlinearStabilityValidator:
    """Validation of nonlinear stability properties."""

    def __init__(self, tolerance: float = 1e-6, basin_samples: int = 100):
        self.tolerance = tolerance
        self.basin_samples = basin_samples

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 100
    ) -> NonlinearStabilityValidation:
        """Validate nonlinear stability."""
        # Compute Lyapunov function with conditioning
        lyapunov = self._compute_lyapunov(flow, state)
        
        # Estimate stability basin with bounds
        basin = self._estimate_basin(flow, state)
        
        # Find perturbation bound with safety margin
        bound = self._find_perturbation_bound(flow, state, time_steps)
        
        # Check overall stability
        stable = (
            lyapunov < 1.0 and  # Energy bounded
            basin > 0.01 and  # Reasonable basin size
            bound > 1e-3  # Meaningful perturbation tolerance
        )
        
        return NonlinearStabilityValidation(
            stable=stable,
            lyapunov_function=lyapunov,
            basin_size=basin,
            perturbation_bound=bound,
        )

    def _compute_lyapunov(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Compute Lyapunov function."""
        # Use energy as Lyapunov function
        energy = flow.compute_energy(state)
        
        # Add regularization for stability
        reg_energy = energy + 1e-6 * torch.sum(state ** 2)
        
        return float(reg_energy.item())

    def _estimate_basin(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Estimate stability basin size."""
        # Sample perturbations
        perturbs = torch.randn_like(state).unsqueeze(0).repeat(self.basin_samples, 1)
        scales = torch.logspace(-3, 0, self.basin_samples).unsqueeze(1)
        
        perturbed = state.unsqueeze(0) + scales * perturbs
        
        # Check stability for each perturbation
        stable_mask = torch.zeros(self.basin_samples, dtype=torch.bool)
        
        for i in range(self.basin_samples):
            energy_i = flow.compute_energy(perturbed[i])
            stable_mask[i] = energy_i < 2.0 * flow.compute_energy(state)
        
        # Find largest stable perturbation
        max_stable = scales[stable_mask][-1] if torch.any(stable_mask) else 0.0
        
        return float(max_stable.item())

    def _find_perturbation_bound(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> float:
        """Find maximum stable perturbation."""
        # Binary search for perturbation bound
        left = 1e-6
        right = 1.0
        
        for _ in range(10):  # Binary search iterations
            mid = (left + right) / 2
            perturb = mid * torch.randn_like(state)
            
            # Check if perturbation remains stable
            current = state + perturb
            stable = True
            
            for _ in range(time_steps):
                current = flow.evolve(current)
                if torch.any(torch.isnan(current)) or torch.any(torch.isinf(current)):
                    stable = False
                    break
                    
                energy = flow.compute_energy(current)
                if energy > 2.0 * flow.compute_energy(state):
                    stable = False
                    break
            
            if stable:
                left = mid
            else:
                right = mid
        
        return float(left)  # Conservative bound


class StructuralStabilityValidator:
    """Validation of structural stability properties."""

    def __init__(self, tolerance: float = 1e-6, parameter_range: float = 0.1):
        self.tolerance = tolerance
        self.parameter_range = parameter_range

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 100
    ) -> StructuralStabilityValidation:
        """Validate structural stability."""
        # Compute parameter sensitivity
        sensitivity = self._compute_sensitivity(flow, state)

        # Measure robustness
        robustness = self._measure_robustness(flow, state, time_steps)

        # Estimate bifurcation distance
        bifurcation = self._estimate_bifurcation(flow, state)

        # Check stability
        stable = (
            sensitivity < 1.0 / self.tolerance
            and robustness > self.tolerance
            and bifurcation > self.tolerance
        )

        return StructuralStabilityValidation(
            stable=stable,
            sensitivity=sensitivity.item(),
            robustness=robustness.item(),
            bifurcation_distance=bifurcation.item(),
        )

    def _compute_sensitivity(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> torch.Tensor:
        """Compute parameter sensitivity."""
        # Get nominal parameters
        params = flow.get_parameters()

        sensitivities = []
        for param in params:
            # Compute parameter gradient
            param.requires_grad_(True)
            output = flow.step(state)
            grad = torch.autograd.grad(output.sum(), param, create_graph=True)[0]
            param.requires_grad_(False)

            # Normalize sensitivity
            sensitivity = torch.norm(grad) * torch.norm(param) / torch.norm(output)
            sensitivities.append(sensitivity)

        return torch.stack(sensitivities).mean()

    def _measure_robustness(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """Measure robustness to perturbations."""
        # Get nominal parameters
        params = flow.get_parameters()

        # Test parameter perturbations
        magnitudes = torch.logspace(-3, 0, 10) * self.parameter_range
        robust_magnitude = torch.tensor(0.0)

        for mag in magnitudes:
            stable = True

            # Try random perturbations
            for _ in range(10):
                # Perturb parameters
                perturbed_params = [p + mag * torch.randn_like(p) for p in params]

                # Set perturbed parameters
                flow.set_parameters(perturbed_params)

                # Check stability
                current = state.clone()
                for _ in range(time_steps):
                    current = flow.step(current)
                    if torch.norm(current - state) > 10 * mag:
                        stable = False
                        break

                if not stable:
                    break

            # Restore parameters
            flow.set_parameters(params)

            if stable:
                robust_magnitude = mag
            else:
                break

        return robust_magnitude

    def _estimate_bifurcation(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> torch.Tensor:
        """Estimate distance to bifurcation."""
        # Get nominal parameters
        params = flow.get_parameters()

        # Compute eigenvalues at different parameter values
        eigenvalues = []

        for eps in torch.linspace(0, self.parameter_range, 10):
            # Perturb parameters
            perturbed_params = [p + eps * torch.randn_like(p) for p in params]

            # Set perturbed parameters
            flow.set_parameters(perturbed_params)

            # Compute stability
            jacobian = flow.compute_jacobian(state)
            eigs = torch.real(torch.linalg.eigvals(jacobian))
            eigenvalues.append(eigs)

            # Restore parameters
            flow.set_parameters(params)

        eigenvalues = torch.stack(eigenvalues)

        # Find where eigenvalues cross stability boundary
        crossings = torch.where(
            torch.sign(eigenvalues[:-1].max(dim=-1)[0])
            != torch.sign(eigenvalues[1:].max(dim=-1)[0])
        )[0]

        if len(crossings) > 0:
            return self.parameter_range * crossings[0].float() / 10
        return torch.tensor(self.parameter_range)


class StabilityValidator:
    """Complete stability validation system."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        stability_threshold: float = 0.0,
        basin_samples: int = 100,
        parameter_range: float = 0.1,
    ):
        self.linear_validator = LinearStabilityValidator(tolerance, stability_threshold)
        self.nonlinear_validator = NonlinearStabilityValidator(tolerance, basin_samples)
        self.structural_validator = StructuralStabilityValidator(
            tolerance, parameter_range
        )

    def validate(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 100
    ) -> Tuple[
        LinearStabilityValidation,
        NonlinearStabilityValidation,
        StructuralStabilityValidation,
    ]:
        """Perform complete stability validation."""
        # Validate linear stability
        linear = self.linear_validator.validate_stability(flow, state)

        # Validate nonlinear stability
        nonlinear = self.nonlinear_validator.validate_stability(flow, state, time_steps)

        # Validate structural stability
        structural = self.structural_validator.validate_stability(
            flow, state, time_steps
        )

        return linear, nonlinear, structural
