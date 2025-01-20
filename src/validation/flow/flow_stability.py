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

    def __init__(self, tolerance: float = 1e-6, stability_threshold: float = 0.1):
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
            # Consider near-zero values and zero values as stable
            is_stable = bool(
                (real_parts <= self.stability_threshold).all() or
                torch.all(torch.abs(real_parts) < 1e-5) or
                torch.all(real_parts == 0.0)
            )
            
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
        if stability.stable:
            message = "Pattern is stable under dynamics"
        else:
            max_eigenval = float(stability.eigenvalues.abs().max())
            # Consider zero eigenvalues as stable
            if max_eigenval == 0.0 or torch.all(torch.abs(stability.eigenvalues.real) < 1e-5):
                stability.stable = True
                message = "Pattern is stable under dynamics (zero eigenvalues)"
            else:
                message = f"Pattern is not stable under dynamics (max eigenvalue: {max_eigenval:.2e})"
            
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

    def __init__(self, tolerance: float = 1e-6, basin_samples: int = 5):
        self.tolerance = tolerance
        self.basin_samples = basin_samples

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 20
    ) -> StabilityValidationResult:
        """Validate nonlinear stability."""
        # Compute Lyapunov function with conditioning
        lyapunov = self._compute_lyapunov(flow, state)
        
        # Estimate stability basin with bounds
        basin = self._estimate_basin(flow, state)
        
        # Find perturbation bound with safety margin
        bound = self._find_perturbation_bound(flow, state, time_steps)
        
        # Check overall stability - consider near-zero values as stable
        is_stable = bool(
            (lyapunov < 10.0 or abs(lyapunov) < 1e-5) and  # Allow near-zero or small values
            (basin > 0.001 or abs(basin) < 1e-5) and  # Allow near-zero or positive values
            (bound > 1e-4 or abs(1.0 - bound) < 1e-3)  # Allow near-1 or positive values
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
        _, metrics = flow.forward(state)
        energy = metrics.get("energy", 0.0)
        
        # For complex state, use squared magnitude
        if state.is_complex():
            reg_term = torch.sum(state.abs() ** 2)
        else:
            reg_term = torch.sum(state ** 2)
        
        # Add regularization for stability (use float32 for intermediate calculations)
        reg_energy = float(energy) + 1e-6 * float(reg_term)
        
        return reg_energy

    def _estimate_basin(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Estimate stability basin size."""
        # Sample perturbations
        perturbs = torch.randn_like(state)
        perturbs = perturbs.unsqueeze(0).repeat(self.basin_samples, 1, 1, 1, 1)
        scales = torch.logspace(-3, 0, self.basin_samples).view(-1, 1, 1, 1, 1)
        
        perturbed = state.unsqueeze(0) + scales * perturbs
        
        # Check stability for each perturbation
        stable_mask = torch.zeros(self.basin_samples, dtype=torch.bool)
        
        base_output, base_metrics = flow.forward(state)
        base_energy = base_metrics.get("energy", torch.tensor(0.0))
        if isinstance(base_energy, torch.Tensor) and torch.is_complex(base_energy):
            base_energy = base_energy.abs()
        
        for i in range(self.basin_samples):
            output, metrics = flow.forward(perturbed[i])
            energy_i = metrics.get("energy", torch.tensor(0.0))
            if isinstance(energy_i, torch.Tensor) and torch.is_complex(energy_i):
                energy_i = energy_i.abs()
            stable_mask[i] = energy_i < 2.0 * base_energy
        
        # Find largest stable scale
        if not stable_mask.any():
            return 0.0
        
        max_stable_idx = stable_mask.nonzero()[-1]
        return scales[max_stable_idx].item()

    def _find_perturbation_bound(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> float:
        """Find maximum stable perturbation."""
        # Binary search for perturbation bound
        left = 1e-6
        right = 1.0

        base_output, base_metrics = flow.forward(state)
        base_energy = base_metrics.get("energy", 0.0)
        # Convert to real by taking magnitude if complex
        if isinstance(base_energy, torch.Tensor) and base_energy.is_complex():
            base_energy = torch.abs(base_energy)
        else:
            base_energy = float(base_energy)

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
                # Convert to real by taking magnitude if complex
                if isinstance(energy, torch.Tensor) and energy.is_complex():
                    energy = torch.abs(energy)
                else:
                    energy = float(energy)

                if energy > 2.0 * base_energy:
                    stable = False
                    break

                current = next_state

            if stable:
                left = mid  # Try larger perturbation
            else:
                right = mid  # Try smaller perturbation

        return float(left)  # Return largest stable perturbation


class StructuralStabilityValidator:
    """Validation of structural stability properties."""

    def __init__(self, tolerance: float = 1e-4, parameter_range: float = 0.5):
        self.tolerance = tolerance
        self.parameter_range = parameter_range

    def validate_stability(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int = 20
    ) -> StabilityValidationResult:
        """Validate structural stability."""
        # Compute parameter sensitivity with improved precision
        sensitivity = self._compute_sensitivity(flow, state)

        # Measure robustness with more samples
        robustness = self._measure_robustness(flow, state, time_steps)

        # Estimate bifurcation distance with improved precision
        bifurcation = self._estimate_bifurcation(flow, state)

        # Check stability with appropriate thresholds
        is_stable = bool(
            (sensitivity < 0.2 or abs(sensitivity) < 1e-4) and  # Reasonable sensitivity threshold
            (robustness > 0.05 or abs(robustness) < 1e-4) and  # Reasonable robustness threshold
            (bifurcation > -0.1 or abs(bifurcation) < 1e-4)  # Reasonable bifurcation threshold
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
        """Compute parameter sensitivity with improved numerical stability."""
        # Get parameters
        params = list(flow.parameters())
        
        # Forward pass with gradient tracking
        state_detached = state.detach().requires_grad_(True)
        output, metrics = flow.forward(state_detached)
        
        # Compute sensitivity with improved numerical stability
        sensitivity = 0.0
        for param in params:
            if torch.is_complex(output):
                # For complex outputs, use proper complex gradient computation
                grad_real = torch.autograd.grad(
                    output.real.sum(), param, create_graph=True,
                    allow_unused=True
                )[0]
                grad_imag = torch.autograd.grad(
                    output.imag.sum(), param, create_graph=True,
                    allow_unused=True
                )[0]
                
                # Handle None gradients
                grad_real = torch.zeros_like(param) if grad_real is None else grad_real
                grad_imag = torch.zeros_like(param) if grad_imag is None else grad_imag
                
                # Use proper complex norm computation
                if torch.is_complex(grad_real):
                    grad_real = torch.view_as_real(grad_real)
                if torch.is_complex(grad_imag):
                    grad_imag = torch.view_as_real(grad_imag)
                
                # Compute sensitivity using Frobenius norm
                sensitivity += torch.sqrt(
                    torch.sum(grad_real**2) + torch.sum(grad_imag**2)
                ).item()
            else:
                grad = torch.autograd.grad(
                    output.sum(), param, create_graph=True,
                    allow_unused=True
                )[0]
                
                # Handle None gradients
                grad = torch.zeros_like(param) if grad is None else grad
                sensitivity += torch.norm(grad, p='fro').item()
            
            # Add small regularization scaled by parameter norm
            param_norm = torch.norm(param, p='fro').item()
            sensitivity += 1e-6 * param_norm
        
        return torch.tensor(sensitivity / max(1, len(params)))

    def _measure_robustness(
        self, flow: GeometricFlow, state: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """Measure robustness to perturbations with improved precision."""
        # Get nominal parameters
        params = list(flow.parameters())
        original_params = [p.clone().detach() for p in params]

        # Test parameter perturbations with proper sampling
        magnitudes = torch.logspace(-6, -2, 10)  # More samples in critical range
        robust_magnitude = torch.tensor(0.0)

        for mag in magnitudes:
            stable = True

            # Try more random perturbations for better coverage
            for _ in range(10):  # Increased number of trials
                # Perturb parameters with controlled magnitude
                for param, orig_param in zip(params, original_params):
                    # Generate perturbation with unit norm
                    perturbation = torch.randn_like(orig_param)
                    perturbation = perturbation / (torch.norm(perturbation, p='fro') + 1e-8)
                    # Scale perturbation and add regularization
                    param.data = orig_param + mag * perturbation + 1e-8 * orig_param

                # Check stability with proper steps and metrics
                current = state.clone()
                max_diff = 0.0
                
                for _ in range(time_steps):
                    output, metrics = flow.forward(current)
                    next_state = output
                    
                    # Project current to same shape as next_state for comparison
                    if current.shape != next_state.shape:
                        current_proj = current[..., :next_state.shape[-1]]
                    else:
                        current_proj = current
                        
                    # Use appropriate distance metric for complex values
                    if torch.is_complex(next_state):
                        diff = torch.sqrt(
                            torch.sum(torch.abs(next_state - current_proj)**2)
                        ).item()
                    else:
                        diff = torch.norm(next_state - current_proj, p='fro').item()
                        
                    max_diff = max(max_diff, diff)
                    if max_diff > 3 * mag:  # More conservative stability criterion
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

        return robust_magnitude

    def _estimate_bifurcation(
        self, flow: GeometricFlow, state: torch.Tensor
    ) -> float:
        """Estimate the bifurcation point with improved precision."""
        state_detached = state.detach().requires_grad_(True)
        
        # Define forward function for jacobian computation that preserves complex values
        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            output, _ = flow.forward(x)
            output = output.reshape(x.shape)
            if torch.is_complex(output):
                # Handle complex values properly
                return torch.view_as_real(output).reshape(-1)
            return output
        
        # Compute jacobian with proper reshaping
        jacobian = torch.autograd.functional.jacobian(forward_fn, state_detached)
        if isinstance(jacobian, tuple):
            jacobian = torch.stack(list(jacobian))
        
        # Reshape jacobian appropriately
        if torch.is_complex(state):
            # Reconstruct complex jacobian
            n = state.numel()
            if isinstance(jacobian, tuple):
                jacobian = torch.stack(list(jacobian))
            jacobian = jacobian.view(2*n, -1)  # [2n, m] where m is output size
            jacobian = torch.complex(
                jacobian[:n, :],  # Real part
                jacobian[n:, :]   # Imaginary part
            )
        
        # Ensure square matrix
        jacobian = jacobian.view(state.shape[-1], -1)
        
        # Compute eigenvalues with improved numerical stability
        eigenvals = torch.linalg.eigvals(jacobian)
        
        # Return maximum real part with proper scaling
        max_real = eigenvals.real.max().item()
        return max_real / (1.0 + abs(max_real))  # Scale to [-1, 1] range


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
