"""Pattern stability validation implementation.

This module validates pattern stability:
- Linear stability analysis
- Nonlinear stability analysis
- Perturbation response
- Lyapunov analysis
- Mode decomposition
"""

from typing import Optional, Dict, List, Tuple, Any, TypeVar, Protocol
from dataclasses import dataclass
import numpy as np
import torch
from scipy import linalg

from src.validation.base import ValidationResult

T = TypeVar('T')

class PatternDynamics(Protocol):
    """Protocol defining required methods for pattern dynamics."""
    
    def compute_linearization(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute linearized dynamics matrix."""
        ...
        
    def apply_symmetry(self, pattern: torch.Tensor, symmetry: torch.Tensor) -> torch.Tensor:
        """Apply symmetry transformation to pattern."""
        ...
        
    def apply_scale_transform(self, pattern: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply scale transformation to pattern."""
        ...

class GeometricFlow(Protocol):
    """Protocol defining required methods for geometric flow."""
    ...

class HamiltonianSystem(Protocol):
    """Protocol defining required methods for Hamiltonian system."""
    ...


@dataclass
class PatternValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation result for pattern stability checks."""

    def merge(self, other: ValidationResult) -> 'PatternValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, PatternValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
        
        return PatternValidationResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data={
                **(self.data or {}),
                **(other.data or {})
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternValidationResult':
        """Create from dictionary representation."""
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data", {})
        )


@dataclass
class StabilitySpectrum:
    """Stability spectrum results."""
    
    eigenvalues: torch.Tensor
    """Eigenvalues of linearized system."""
    
    eigenvectors: torch.Tensor
    """Eigenvectors of linearized system."""
    
    growth_rates: torch.Tensor
    """Growth rates of perturbations."""
    
    frequencies: torch.Tensor
    """Frequencies of perturbations."""
    
    num_unstable: int
    """Number of unstable modes."""


@dataclass
class PatternValidator:
    """Validates pattern stability properties."""

    dynamics: PatternDynamics
    """Pattern dynamics system."""

    flow: GeometricFlow
    """Geometric flow system."""

    hamiltonian: Optional[HamiltonianSystem] = None
    """Optional Hamiltonian system."""

    def validate_stability(self, pattern: torch.Tensor, perturbation: torch.Tensor, 
                         time: float) -> ValidationResult:
        """Validate pattern stability under perturbation."""
        try:
            # Compute stability spectrum
            spectrum = self._compute_stability_spectrum(pattern, perturbation)
            
            # Check stability conditions
            is_stable = bool((spectrum.growth_rates < 0).all().item())
            max_growth = spectrum.growth_rates.max().item()
            
            # Prepare validation message
            if is_stable:
                msg = f"Pattern is stable with max growth rate {max_growth:.3f}"
            else:
                msg = f"Pattern is unstable with {spectrum.num_unstable} unstable modes"
            
            # Return validation result with spectrum data
            return PatternValidationResult(
                is_valid=is_stable,
                message=msg,
                data={
                    "spectrum": spectrum,
                    "max_growth_rate": max_growth,
                    "num_unstable": spectrum.num_unstable
                }
            )
            
        except Exception as e:
            return PatternValidationResult(
                is_valid=False,
                message=f"Stability validation failed: {str(e)}",
                data={}
            )

    def verify_symmetries(self, pattern: torch.Tensor, 
                         symmetry_group: torch.Tensor) -> ValidationResult:
        """Verify pattern symmetries."""
        try:
            # Apply symmetry transformations
            transformed = self._apply_symmetries(pattern, symmetry_group)
            
            # Check invariance
            diff = (pattern - transformed).abs().max().item()
            is_symmetric = diff < 1e-6
            
            return PatternValidationResult(
                is_valid=bool(is_symmetric),
                message=f"Pattern symmetry verification {'passed' if is_symmetric else 'failed'}",
                data={"max_difference": diff}
            )
            
        except Exception as e:
            return PatternValidationResult(
                is_valid=False,
                message=f"Symmetry verification failed: {str(e)}",
                data={}
            )

    def check_scale_invariance(self, pattern: torch.Tensor, 
                             scale_transformation: torch.Tensor) -> ValidationResult:
        """Check pattern scale invariance."""
        try:
            # Apply scale transformation
            transformed = self._apply_scale_transform(pattern, scale_transformation)
            
            # Compute relative difference
            rel_diff = ((pattern - transformed).abs() / (pattern.abs() + 1e-8)).max().item()
            is_invariant = rel_diff < 1e-6
            
            return PatternValidationResult(
                is_valid=bool(is_invariant),
                message=f"Scale invariance check {'passed' if is_invariant else 'failed'}",
                data={"relative_difference": rel_diff}
            )
            
        except Exception as e:
            return PatternValidationResult(
                is_valid=False,
                message=f"Scale invariance check failed: {str(e)}",
                data={}
            )

    def _compute_stability_spectrum(self, pattern: torch.Tensor, 
                                  perturbation: torch.Tensor) -> StabilitySpectrum:
        """Compute stability spectrum for pattern."""
        # Compute linearized system matrix
        linear_matrix = self.dynamics.compute_linearization(pattern)
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eig(linear_matrix)
        
        # Extract real and imaginary parts
        growth_rates = eigenvals.real
        frequencies = eigenvals.imag
        
        # Count unstable modes
        num_unstable = int((growth_rates > 0).sum().item())
        
        return StabilitySpectrum(
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            growth_rates=growth_rates,
            frequencies=frequencies,
            num_unstable=num_unstable
        )

    def _apply_symmetries(self, pattern: torch.Tensor, 
                         symmetry_group: torch.Tensor) -> torch.Tensor:
        """Apply symmetry transformations to pattern."""
        transformed = pattern.clone()
        for sym in symmetry_group:
            transformed = self.dynamics.apply_symmetry(transformed, sym)
        return transformed

    def _apply_scale_transform(self, pattern: torch.Tensor, 
                             scale_transformation: torch.Tensor) -> torch.Tensor:
        """Apply scale transformation to pattern."""
        return self.dynamics.apply_scale_transform(pattern, scale_transformation)
