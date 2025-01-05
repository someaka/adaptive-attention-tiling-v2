"""Pattern stability validation implementation.

This module validates pattern stability:
- Linear stability analysis
- Nonlinear stability analysis
- Perturbation response
- Lyapunov analysis
- Mode decomposition
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import numpy as np

from src.validation.base import ValidationResult
from validation.flow.flow_stability import (
    LinearStabilityValidator,
    NonlinearStabilityValidator,
    StructuralStabilityValidator,
    LinearStabilityValidation,
    NonlinearStabilityValidation,
    StructuralStabilityValidation
)
from src.core.tiling.geometric_flow import GeometricFlow


@dataclass
class PatternStabilityResult(ValidationResult[Dict[str, Any]]):
    """Results from pattern stability validation."""

    linear_stability: Optional[bool] = None
    nonlinear_stability: Optional[bool] = None
    structural_stability: Optional[bool] = None
    lyapunov_exponents: Optional[torch.Tensor] = None
    perturbation_response: Optional[Dict[str, torch.Tensor]] = None

    def merge(self, other: ValidationResult) -> 'PatternStabilityResult':
        """Merge with another validation result."""
        if not isinstance(other, PatternStabilityResult):
            raise TypeError("Can only merge with another PatternStabilityResult")
            
        merged_data = {}
        if self.data is not None:
            merged_data.update(self.data)
        if other.data is not None:
            merged_data.update(other.data)
            
        return PatternStabilityResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data=merged_data,
            linear_stability=self.linear_stability or other.linear_stability,
            nonlinear_stability=self.nonlinear_stability or other.nonlinear_stability,
            structural_stability=self.structural_stability or other.structural_stability,
            lyapunov_exponents=self.lyapunov_exponents or other.lyapunov_exponents,
            perturbation_response=self.perturbation_response or other.perturbation_response
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": self.data,
            "linear_stability": self.linear_stability,
            "nonlinear_stability": self.nonlinear_stability,
            "structural_stability": self.structural_stability,
            "lyapunov_exponents": self.lyapunov_exponents.tolist() if self.lyapunov_exponents is not None else None,
            "perturbation_response": {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in (self.perturbation_response or {}).items()
            }
        }
        return base_dict
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternStabilityResult':
        """Create from dictionary representation."""
        lyapunov = None
        if data.get("lyapunov_exponents") is not None:
            lyapunov = torch.tensor(data["lyapunov_exponents"])
            
        perturbation = None
        if data.get("perturbation_response") is not None:
            perturbation = {
                k: torch.tensor(v) if isinstance(v, list) else v
                for k, v in data["perturbation_response"].items()
            }
            
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data"),
            linear_stability=data.get("linear_stability"),
            nonlinear_stability=data.get("nonlinear_stability"),
            structural_stability=data.get("structural_stability"),
            lyapunov_exponents=lyapunov,
            perturbation_response=perturbation
        )


class PatternValidator:
    """Validates pattern stability properties."""

    def __init__(
        self,
        linear_validator: Optional[LinearStabilityValidator] = None,
        nonlinear_validator: Optional[NonlinearStabilityValidator] = None,
        structural_validator: Optional[StructuralStabilityValidator] = None,
        lyapunov_threshold: float = 0.1,
        perturbation_threshold: float = 0.1
    ):
        """Initialize pattern validator.
        
        Args:
            linear_validator: Validator for linear stability
            nonlinear_validator: Validator for nonlinear stability
            structural_validator: Validator for structural stability
            lyapunov_threshold: Threshold for Lyapunov stability
            perturbation_threshold: Threshold for perturbation response
        """
        self.linear_validator = linear_validator or LinearStabilityValidator()
        self.nonlinear_validator = nonlinear_validator or NonlinearStabilityValidator()
        self.structural_validator = structural_validator or StructuralStabilityValidator()
        self.lyapunov_threshold = lyapunov_threshold
        self.perturbation_threshold = perturbation_threshold

    def validate(
        self,
        pattern_flow: GeometricFlow,
        initial_state: torch.Tensor,
        time_steps: int = 100
    ) -> PatternStabilityResult:
        """Validate pattern stability.
        
        Args:
            pattern_flow: Pattern flow to validate
            initial_state: Initial state tensor
            time_steps: Number of time steps for evolution
            
        Returns:
            Validation results
        """
        # Linear stability analysis
        linear_result = self.linear_validator.validate_stability(pattern_flow, initial_state)
        
        # Nonlinear stability analysis
        nonlinear_result = self.nonlinear_validator.validate_stability(pattern_flow, initial_state, time_steps)
        
        # Structural stability analysis
        structural_result = self.structural_validator.validate_stability(pattern_flow, initial_state, time_steps)
        
        # Compute Lyapunov exponents
        lyapunov_exponents = self._compute_lyapunov_exponents(pattern_flow, initial_state, time_steps)
        
        # Test perturbation response
        perturbation_response = self._test_perturbation_response(pattern_flow, initial_state)
        
        # Combine results
        is_valid = bool(
            linear_result.is_valid
            and nonlinear_result.is_valid
            and structural_result.is_valid
            and torch.all(lyapunov_exponents < self.lyapunov_threshold)
            and all(v < self.perturbation_threshold for v in perturbation_response.values())
        )
        
        # Extract stability data from validation results
        linear_data = linear_result.data or {}
        nonlinear_data = nonlinear_result.data or {}
        structural_data = structural_result.data or {}
        
        # Create combined message
        max_perturbation = max(float(v.max()) for v in perturbation_response.values())
        message = "; ".join([
            linear_result.message,
            nonlinear_result.message,
            structural_result.message,
            f"Lyapunov exponents: {lyapunov_exponents.max():.2e}",
            f"Perturbation response: {max_perturbation:.2e}"
        ])
        
        return PatternStabilityResult(
            is_valid=is_valid,
            message=message,
            data={
                "linear_result": linear_data,
                "nonlinear_result": nonlinear_data,
                "structural_result": structural_data,
                "lyapunov_exponents": lyapunov_exponents.tolist(),
                "perturbation_response": {
                    k: v.tolist() for k, v in perturbation_response.items()
                }
            },
            linear_stability=linear_result.is_valid,
            nonlinear_stability=nonlinear_result.is_valid,
            structural_stability=structural_result.is_valid,
            lyapunov_exponents=lyapunov_exponents,
            perturbation_response=perturbation_response
        )

    def _compute_lyapunov_exponents(
        self, pattern_flow: GeometricFlow, initial_state: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """Compute Lyapunov exponents for the pattern flow."""
        # Create forward function for Jacobian computation
        def forward_fn(state):
            output, _ = pattern_flow(state)
            # Return real part for complex tensors
            return output.real if output.is_complex() else output
        
        # Get initial state
        state = initial_state.detach().requires_grad_(True)
        
        # Compute output with gradients enabled
        output = forward_fn(state)
        
        # Compute gradient of output norm with respect to input
        output_norm = torch.linalg.vector_norm(output, dim=-1)  # Only compute norm along last dimension
        output_norm = output_norm.sum()  # Make sure we have a scalar for gradient computation
        grad = torch.autograd.grad(output_norm, state, create_graph=False)[0]  # Disable create_graph
        
        # Compute local stability measure using vector norm
        stability = torch.linalg.vector_norm(grad, dim=-1)  # Only compute norm along last dimension
        
        # Return stability measure as Lyapunov exponent approximation
        return stability.unsqueeze(-1)  # Add dimension to match expected shape

    def _test_perturbation_response(
        self,
        pattern_flow: GeometricFlow,
        initial_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test response to different types of perturbations."""
        responses = {}
        dtype = initial_state.dtype
        device = initial_state.device
        
        # Create noise in appropriate dtype
        def create_noise(mode: str) -> torch.Tensor:
            if dtype in [torch.complex64, torch.complex128]:
                if mode == "uniform":
                    real = torch.rand_like(initial_state, dtype=torch.float32) * 0.01
                    imag = torch.rand_like(initial_state, dtype=torch.float32) * 0.01
                    return torch.complex(real, imag).to(dtype=dtype)
                else:  # gaussian
                    real = torch.randn_like(initial_state, dtype=torch.float32) * 0.01
                    imag = torch.randn_like(initial_state, dtype=torch.float32) * 0.01
                    return torch.complex(real, imag).to(dtype=dtype)
            else:
                if mode == "uniform":
                    return torch.rand_like(initial_state) * 0.01
                else:  # gaussian
                    return torch.randn_like(initial_state) * 0.01
        
        # Test small random perturbations
        noise = create_noise("gaussian")
        perturbed = initial_state + noise
        evolved_orig, _ = pattern_flow(initial_state)
        evolved_pert, _ = pattern_flow(perturbed)
        
        # Compute response using appropriate norm
        if dtype in [torch.complex64, torch.complex128]:
            responses["random"] = torch.norm(evolved_pert - evolved_orig).abs() / torch.norm(initial_state).abs()
        else:
            responses["random"] = torch.norm(evolved_pert - evolved_orig) / torch.norm(initial_state)
        
        # Test structured perturbations
        for mode in ["uniform", "gaussian"]:
            noise = create_noise(mode)
            perturbed = initial_state + noise
            evolved_pert, _ = pattern_flow(perturbed)
            
            # Compute response using appropriate norm
            if dtype in [torch.complex64, torch.complex128]:
                responses[mode] = torch.norm(evolved_pert - evolved_orig).abs() / torch.norm(initial_state).abs()
            else:
                responses[mode] = torch.norm(evolved_pert - evolved_orig) / torch.norm(initial_state)
            
        return responses
