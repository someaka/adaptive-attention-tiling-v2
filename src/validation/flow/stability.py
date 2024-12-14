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

from src.core.tiling.geometric_flow import GeometricFlow


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
        # Compute Jacobian at state using autograd
        state.requires_grad_(True)
        jacobian_tuple = torch.autograd.functional.jacobian(flow.forward, state)
        state.requires_grad_(False)

        # Convert tuple to tensor and reshape to 2D matrix if needed
        if isinstance(jacobian_tuple, tuple):
            jacobian = torch.stack(list(jacobian_tuple))
        
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        state_size = state.numel() // batch_size
        jacobian = jacobian.view(batch_size, state_size, state_size)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(jacobian)

        # Get real parts for stability
        real_parts = torch.real(eigenvalues)

        # Compute growth rates
        growth_rates = torch.exp(real_parts)

        # Check stability - convert tensor to bool
        stable = bool(torch.all(real_parts <= self.stability_threshold).item())

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
        stable = bool(
            lyapunov < 1.0 and  # Energy bounded
            basin > 0.01 and  # Reasonable basin size
            bound > 1e-3  # Meaningful perturbation tolerance
        )
        
        return NonlinearStabilityValidation(
            stable=stable,
            lyapunov_function=float(lyapunov),
            basin_size=float(basin),
            perturbation_bound=float(bound),
        )

    def _compute_lyapunov(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Compute Lyapunov function."""
        # Use energy as Lyapunov function
        _, metrics = flow.forward(state)
        energy = metrics.get("energy", 0.0)
        
        # Add regularization for stability
        reg_energy = energy + 1e-6 * torch.sum(state ** 2)
        
        return float(reg_energy)

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
        
        base_output, base_metrics = flow.forward(state)
        base_energy = base_metrics.get("energy", 0.0)
        
        for i in range(self.basin_samples):
            output, metrics = flow.forward(perturbed[i])
            energy_i = metrics.get("energy", 0.0)
            stable_mask[i] = energy_i < 2.0 * base_energy
        
        # Find largest stable perturbation
        max_stable = scales[stable_mask][-1] if torch.any(stable_mask) else torch.tensor(0.0)
        
        return float(max_stable.item())

    def _find_perturbation_bound(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> float:
        """Find maximum stable perturbation."""
        # Binary search for perturbation bound
        left = 1e-6
        right = 1.0
        
        base_output, base_metrics = flow.forward(state)
        base_energy = base_metrics.get("energy", 0.0)
        
        for _ in range(10):  # Binary search iterations
            mid = (left + right) / 2
            perturb = mid * torch.randn_like(state)
            
            # Check if perturbation remains stable
            current = state + perturb
            stable = True
            
            for _ in range(time_steps):
                output, metrics = flow.forward(current)
                next_state = output
                if torch.any(torch.isnan(next_state)) or torch.any(torch.isinf(next_state)):
                    stable = False
                    break
                    
                energy = metrics.get("energy", 0.0)
                if energy > 2.0 * base_energy:
                    stable = False
                    break
                current = next_state
            
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
        stable = bool(
            sensitivity < 1.0 / self.tolerance
            and robustness > self.tolerance
            and bifurcation > self.tolerance
        )

        return StructuralStabilityValidation(
            stable=stable,
            sensitivity=float(sensitivity),
            robustness=float(robustness),
            bifurcation_distance=float(bifurcation),
        )

    def _compute_sensitivity(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> torch.Tensor:
        """Compute parameter sensitivity."""
        # Get nominal parameters
        params = list(flow.parameters())  # Use parameters() instead of get_parameters()

        sensitivities = []
        for param in params:
            # Compute parameter gradient
            param.requires_grad_(True)
            output, _ = flow.forward(state)
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
        params = list(flow.parameters())  # Use parameters() instead of get_parameters()
        original_params = [p.clone().detach() for p in params]  # Store original parameters

        # Test parameter perturbations
        magnitudes = torch.logspace(-3, 0, 10) * self.parameter_range
        robust_magnitude = torch.tensor(0.0)

        for mag in magnitudes:
            stable = True

            # Try random perturbations
            for _ in range(10):
                # Perturb parameters
                for param, orig_param in zip(params, original_params):
                    param.data = orig_param + mag * torch.randn_like(orig_param)

                # Check stability
                current = state.clone()
                for _ in range(time_steps):
                    output, _ = flow.forward(current)
                    next_state = output
                    if torch.norm(next_state - state) > 10 * mag:
                        stable = False
                        break
                    current = next_state

                if not stable:
                    break

            # Restore parameters
            for param, orig_param in zip(params, original_params):
                param.data = orig_param.clone()

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
        params = list(flow.parameters())  # Use parameters() instead of get_parameters()
        original_params = [p.clone().detach() for p in params]  # Store original parameters

        # Compute eigenvalues at different parameter values
        eigenvalues = []

        for eps in torch.linspace(0, self.parameter_range, 10):
            # Perturb parameters
            for param, orig_param in zip(params, original_params):
                param.data = orig_param + eps * torch.randn_like(orig_param)

            # Compute stability using autograd
            state.requires_grad_(True)
            jacobian_tuple = torch.autograd.functional.jacobian(flow.forward, state)
            state.requires_grad_(False)

            # Convert tuple to tensor if needed
            if isinstance(jacobian_tuple, tuple):
                jacobian = torch.stack(list(jacobian_tuple))
            
            batch_size = state.size(0) if len(state.shape) > 1 else 1
            state_size = state.numel() // batch_size
            jacobian = jacobian.view(batch_size, state_size, state_size)

            eigs = torch.real(torch.linalg.eigvals(jacobian))
            eigenvalues.append(eigs)

            # Restore parameters
            for param, orig_param in zip(params, original_params):
                param.data = orig_param.clone()

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
