"""Flow Stability Validation Implementation.

This module validates flow stability properties:
- Linear stability analysis
- Nonlinear stability
- Structural stability
- Bifurcation analysis
"""

from dataclasses import dataclass
from typing import Tuple, Any, Dict, Optional, Protocol, ClassVar, Type

import torch
import torch.nn as nn
from ..base import ValidationResult

from src.core.tiling.geometric_flow import GeometricFlow


class StabilityValidatorProtocol(Protocol):
    """Base protocol for stability validators."""
    
    tolerance: float
    
    def compute_stability(self, dynamics: Any, pattern: torch.Tensor) -> 'LinearStabilityValidation':
        """Compute stability analysis of pattern under dynamics."""
        ...


@dataclass
class StabilityValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation result for stability analysis."""
    
    def __init__(self, is_valid: bool, message: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(is_valid, message, data)
    
    def merge(self, other: ValidationResult) -> 'StabilityValidationResult':
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
        return StabilityValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data={**(self.data or {}), **(other.data or {})}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StabilityValidationResult':
        """Create from dictionary.
        
        Args:
            data: Dictionary containing validation data
            
        Returns:
            New StabilityValidationResult instance
            
        Raises:
            ValueError: If required fields are missing
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
            
        required_fields = {"is_valid", "message"}
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(data.keys())}")
            
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data")
        )


@dataclass
class LinearStabilityValidation:
    """Results of linear stability analysis."""
    stable: bool
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    growth_rates: torch.Tensor


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


class StabilityValidatorBase:
    """Base class for stability validators."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def compute_stability(self, dynamics: Any, pattern: torch.Tensor) -> LinearStabilityValidation:
        """Compute stability analysis of pattern under dynamics."""
        raise NotImplementedError


class LinearStabilityValidator(StabilityValidatorBase):
    """Validation of linear stability properties."""

    def __init__(self, tolerance: float = 1e-6, stability_threshold: float = 0.0):
        super().__init__(tolerance)
        self.stability_threshold = stability_threshold

    def compute_stability(
        self, flow: nn.Module, state: torch.Tensor
    ) -> LinearStabilityValidation:
        """Compute stability analysis of flow at given state."""
        try:
            # Define wrapper function that returns tensor from dict if needed
            def forward_fn(x):
                output = flow.forward(x)
                if isinstance(output, dict):
                    # Extract the main tensor from the output dictionary
                    # Assuming it's the first tensor value in the dict
                    return next(v for v in output.values() if isinstance(v, torch.Tensor))
                return output
            
            # Compute Jacobian using wrapped function
            jacobian_tuple = torch.autograd.functional.jacobian(forward_fn, state)
            
            # Convert tuple to tensor if needed
            if isinstance(jacobian_tuple, tuple):
                jacobian = jacobian_tuple[0]
            else:
                jacobian = jacobian_tuple
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = torch.linalg.eig(jacobian)
            
            # Check stability based on eigenvalue real parts
            real_parts = eigenvals.real
            is_stable = bool((real_parts <= self.stability_threshold).all())
            
            # Compute growth rates
            growth_rates = torch.exp(real_parts)
            
            return LinearStabilityValidation(
                stable=is_stable,
                eigenvalues=eigenvals,
                eigenvectors=eigenvecs,
                growth_rates=growth_rates
            )
            
        except Exception as e:
            # Return validation result with empty tensors on error
            device = state.device if hasattr(state, 'device') else 'cpu'
            empty_tensor = torch.zeros(1, device=device)
            return LinearStabilityValidation(
                stable=False,
                eigenvalues=empty_tensor,
                eigenvectors=empty_tensor,
                growth_rates=empty_tensor
            )

    def validate_stability(
        self, dynamics: Any, pattern: torch.Tensor
    ) -> StabilityValidationResult:
        """Validate stability of pattern under dynamics.
        
        Args:
            dynamics: Pattern dynamics to validate
            pattern: Pattern tensor to validate
            
        Returns:
            ValidationResult with stability analysis
        """
        # Compute stability analysis
        stability = self.compute_stability(dynamics, pattern)
        
        # Create validation message
        message = f"Pattern {'is' if stability.stable else 'is not'} stable under dynamics"
        if not stability.stable:
            max_eigenval = float(stability.eigenvalues.abs().max())
            message += f" (max eigenvalue: {max_eigenval:.2e})"
            
        # Return ValidationResult with stability data
        return StabilityValidationResult(
            is_valid=stability.stable,
            message=message,
            data={
                'eigenvalues': stability.eigenvalues,
                'eigenvectors': stability.eigenvectors,
                'growth_rates': stability.growth_rates,
                'stability_threshold': self.stability_threshold
            }
        )


class NonlinearStabilityValidator:
    """Validation of nonlinear stability properties."""

    def __init__(self, tolerance: float = 1e-6, basin_samples: int = 10):
        self.tolerance = tolerance
        self.basin_samples = basin_samples

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 100
    ) -> StabilityValidationResult:
        """Validate nonlinear stability."""
        # Compute Lyapunov function with conditioning
        lyapunov = self._compute_lyapunov(flow, state)
        
        # Estimate stability basin with bounds
        basin = self._estimate_basin(flow, state)
        
        # Find perturbation bound with safety margin
        bound = self._find_perturbation_bound(flow, state, time_steps)
        
        # Check overall stability
        is_stable = bool(
            lyapunov < 1.0 and  # Energy bounded
            basin > 0.01 and  # Reasonable basin size
            bound > 1e-3  # Meaningful perturbation tolerance
        )
        
        # Create validation message
        message = f"Nonlinear stability validation {'passed' if is_stable else 'failed'}"
        if not is_stable:
            message += f" (lyapunov={lyapunov:.2e}, basin={basin:.2e}, bound={bound:.2e})"
        
        return StabilityValidationResult(
            is_valid=is_stable,
            message=message,
            data={
                'lyapunov_function': float(lyapunov),
                'basin_size': float(basin),
                'perturbation_bound': float(bound)
            }
        )

    def _compute_lyapunov(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Compute Lyapunov function."""
        # Use energy as Lyapunov function
        _, metrics_dict = flow.forward(state)
        energy = float(metrics_dict.get("energy", 0.0)) if isinstance(metrics_dict, dict) else 0.0
        
        # For complex state, use squared magnitude
        if state.is_complex():
            reg_term = torch.sum(state.abs() ** 2)
        else:
            reg_term = torch.sum(state ** 2)
            
        # Add regularization term
        energy = energy + 0.1 * float(reg_term)
        
        return energy

    def _estimate_basin(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Estimate size of stability basin."""
        # Sample perturbations of increasing magnitude
        perturbations = [0.1 * i for i in range(self.basin_samples)]
        
        for eps in perturbations:
            # Add random perturbation
            perturbed = state + eps * torch.randn_like(state)
            
            # Check stability under perturbation
            _, metrics_dict = flow.forward(perturbed)
            energy = float(metrics_dict.get("energy", float('inf'))) if isinstance(metrics_dict, dict) else float('inf')
            
            # If energy exceeds threshold, we've left stability basin
            if energy > 10.0:
                return eps
                
        return perturbations[-1]

    def _find_perturbation_bound(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> float:
        """Find maximum stable perturbation magnitude."""
        # Binary search for perturbation bound
        left, right = 0.0, 1.0
        
        for _ in range(10):  # 10 binary search steps
            mid = (left + right) / 2
            perturbed = state + mid * torch.randn_like(state)
            
            # Evolve perturbed state
            current = perturbed
            stable = True
            
            for _ in range(time_steps):
                # One evolution step
                current, metrics_dict = flow.forward(current)
                energy = float(metrics_dict.get("energy", float('inf'))) if isinstance(metrics_dict, dict) else float('inf')
                
                # Check stability
                if energy > 10.0:
                    stable = False
                    break
                    
            if stable:
                left = mid  # Try larger perturbation
            else:
                right = mid  # Try smaller perturbation
                
        return left  # Return largest stable perturbation


class StructuralStabilityValidator:
    """Validation of structural stability properties."""

    def __init__(self, tolerance: float = 1e-6, parameter_range: float = 0.1):
        self.tolerance = tolerance
        self.parameter_range = parameter_range

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 100
    ) -> StabilityValidationResult:
        """Validate structural stability."""
        # Compute parameter sensitivity
        sensitivity = self._compute_sensitivity(flow, state)

        # Measure robustness
        robustness = self._measure_robustness(flow, state, time_steps)

        # Estimate bifurcation distance
        bifurcation = self._estimate_bifurcation(flow, state)

        # Check stability
        is_stable = bool(
            sensitivity < 1.0 / self.tolerance
            and robustness > self.tolerance
            and bifurcation > self.tolerance
        )

        # Create validation message
        message = f"Structural stability validation {'passed' if is_stable else 'failed'}"
        if not is_stable:
            message += f" (sensitivity={sensitivity:.2e}, robustness={robustness:.2e}, bifurcation={bifurcation:.2e})"

        return StabilityValidationResult(
            is_valid=is_stable,
            message=message,
            data={
                'sensitivity': float(sensitivity),
                'robustness': float(robustness),
                'bifurcation_distance': float(bifurcation)
            }
        )

    def _compute_sensitivity(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> torch.Tensor:
        """Compute parameter sensitivity."""
        # Get parameters
        params = list(flow.parameters())
        
        # Forward pass
        output, _ = flow.forward(state)
        
        # Split complex output into real and imaginary parts
        output_real = output.real
        output_imag = output.imag
        
        # Compute gradients for real and imaginary parts
        sensitivity = 0.0
        for param in params:
            # Real part gradient
            grad_real = torch.autograd.grad(
                output_real.sum(), param, create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            
            # Imaginary part gradient
            grad_imag = torch.autograd.grad(
                output_imag.sum(), param, create_graph=True,
                allow_unused=True
            )[0]
            
            # Handle None gradients
            if grad_real is None:
                grad_real = torch.zeros_like(param)
            if grad_imag is None:
                grad_imag = torch.zeros_like(param)
            
            # Combine gradients
            sensitivity += torch.norm(grad_real + 1j * grad_imag).item()
        
        return torch.tensor(sensitivity / len(params))

    def _measure_robustness(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """Measure robustness to perturbations."""
        # Get nominal parameters
        params = list(flow.parameters())
        original_params = [p.clone().detach() for p in params]

        # Test parameter perturbations with fewer samples
        magnitudes = torch.logspace(-3, 0, 5) * self.parameter_range
        robust_magnitude = torch.tensor(0.0)

        for mag in magnitudes:
            stable = True

            # Try fewer random perturbations
            for _ in range(3):
                # Perturb parameters
                for param, orig_param in zip(params, original_params):
                    param.data = orig_param + mag * torch.randn_like(orig_param)

                # Check stability with fewer steps
                current = state.clone()
                for _ in range(min(time_steps, 3)):
                    output, _ = flow.forward(current)
                    next_state = output
                    # Project current to same shape as next_state for comparison
                    if current.shape != next_state.shape:
                        current_proj = current[..., :next_state.shape[-1]]
                    else:
                        current_proj = current
                    if torch.norm(next_state - current_proj) > 10 * mag:
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
        params = list(flow.parameters())
        original_params = [p.clone().detach() for p in params]

        # Define wrapper functions that return real and imaginary parts
        def forward_real(x):
            output, _ = flow.forward(x)
            return output.real

        def forward_imag(x):
            output, _ = flow.forward(x)
            return output.imag

        # Compute eigenvalues at different parameter values with fewer samples
        eigenvalues = []

        for eps in torch.linspace(0, self.parameter_range, 5):
            # Perturb parameters
            for param, orig_param in zip(params, original_params):
                param.data = orig_param + eps * torch.randn_like(orig_param)

            # Compute stability using autograd for real and imaginary parts
            state_detached = state.detach()
            state_detached.requires_grad_(True)
            jacobian_real = torch.autograd.functional.jacobian(forward_real, state_detached)
            jacobian_imag = torch.autograd.functional.jacobian(forward_imag, state_detached)
            state_detached.requires_grad_(False)

            # Convert tuple to tensor if needed and combine real and imaginary parts
            if isinstance(jacobian_real, tuple):
                jacobian_real = torch.stack(list(jacobian_real))
            if isinstance(jacobian_imag, tuple):
                jacobian_imag = torch.stack(list(jacobian_imag))
            
            jacobian = jacobian_real + 1j * jacobian_imag
            
            # Get the correct shape for the Jacobian
            batch_size = state.size(0)
            manifold_dim = flow.manifold_dim
            # Flatten spatial dimensions and reshape to manifold dimensions
            jacobian = jacobian.reshape(batch_size, -1, manifold_dim)[:, :manifold_dim, :]

            eigs = torch.real(torch.linalg.eigvals(jacobian))
            eigenvalues.append(eigs)

            # Restore parameters
            for param, orig_param in zip(params, original_params):
                param.data = orig_param.clone()

        eigenvalues = torch.stack(eigenvalues)

        # Find where eigenvalues cross stability boundary
        crossings = torch.where(
            torch.sgn(eigenvalues[:-1].max(dim=-1)[0])
            != torch.sgn(eigenvalues[1:].max(dim=-1)[0])
        )[0]

        if len(crossings) > 0:
            return self.parameter_range * crossings[0].float() / 5
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
        self.tolerance = tolerance
        self.stability_threshold = stability_threshold
        self.linear_validator = LinearStabilityValidator(tolerance, stability_threshold)
        self.nonlinear_validator = NonlinearStabilityValidator(tolerance, basin_samples)
        self.structural_validator = StructuralStabilityValidator(
            tolerance, parameter_range
        )

    def compute_stability(
        self, dynamics: Any, pattern: torch.Tensor
    ) -> LinearStabilityValidation:
        """Compute stability analysis of pattern under dynamics."""
        return self.linear_validator.compute_stability(dynamics, pattern)

    def validate(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 100
    ) -> Tuple[
        StabilityValidationResult,
        StabilityValidationResult,
        StabilityValidationResult,
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

    def validate_stability(self, dynamics: Any, pattern: torch.Tensor) -> StabilityValidationResult:
        """Validate stability of pattern under dynamics.
        
        Args:
            dynamics: Pattern dynamics to validate
            pattern: Pattern tensor to validate
            
        Returns:
            ValidationResult with stability analysis
        """
        # Compute stability analysis
        stability = self.compute_stability(dynamics, pattern)
        
        # Check if all eigenvalues are within stability threshold
        is_stable = bool((stability.eigenvalues.abs() <= self.tolerance).all())
        
        # Create validation message
        message = f"Pattern {'is' if is_stable else 'is not'} stable under dynamics"
        if not is_stable:
            max_eigenval = float(stability.eigenvalues.abs().max())
            message += f" (max eigenvalue: {max_eigenval:.2e})"
            
        # Return ValidationResult with stability data
        return StabilityValidationResult(
            is_valid=is_stable,
            message=message,
            data={
                'eigenvalues': stability.eigenvalues,
                'eigenvectors': stability.eigenvectors,
                'stability_threshold': self.tolerance
            }
        )
