"""Hamiltonian Flow Validation Implementation.

This module validates Hamiltonian flow properties:
- Energy conservation
- Symplectic structure preservation
- Phase space volume preservation
- Poincaré recurrence
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import torch
from torch import Tensor

from ..base import ValidationResult
from ...core.patterns.symplectic import SymplecticStructure
from ...neural.flow.hamiltonian import HamiltonianSystem


def convert_tensor_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensor data to serializable format."""
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            result[key] = convert_tensor_data(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


@dataclass
class HamiltonianFlowValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation results for Hamiltonian flow.
    
    This class handles validation of:
    - Energy conservation
    - Symplectic structure preservation
    - Phase space volume preservation
    - Poincaré recurrence
    """
    
    def __init__(self, is_valid: bool, message: str, data: Optional[Dict[str, Any]] = None):
        """Initialize Hamiltonian flow validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional validation data containing flow metrics and tensors
        """
        super().__init__(is_valid, message, data)
    
    def merge(self, other: ValidationResult) -> 'HamiltonianFlowValidationResult':
        """Merge with another validation result.
        
        Args:
            other: Another validation result to merge with
            
        Returns:
            New HamiltonianFlowValidationResult combining both results
            
        Raises:
            TypeError: If other is not a ValidationResult
        """
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge metrics dictionaries carefully
        merged_data = {**(self.data or {})}
        other_data = other.data or {}
        
        # Special handling for flow metrics
        for key, value in other_data.items():
            if key in merged_data and isinstance(value, dict):
                if isinstance(merged_data[key], dict):
                    merged_data[key].update(value)
                else:
                    merged_data[key] = value
            else:
                merged_data[key] = value
        
        return HamiltonianFlowValidationResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data=merged_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper tensor handling."""
        return {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": convert_tensor_data(self.data or {})
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HamiltonianFlowValidationResult':
        """Create from dictionary.
        
        Args:
            data: Dictionary containing validation data
            
        Returns:
            New HamiltonianFlowValidationResult instance
            
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
            data=data.get("data", {})
        )

    def __str__(self) -> str:
        """String representation with tensor summary."""
        tensor_summaries = []
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    tensor_summaries.append(f"{key}: {self._tensor_repr(value)}")
                elif isinstance(value, dict):
                    nested_tensors = [
                        f"{k}: {self._tensor_repr(v)}" 
                        for k, v in value.items() 
                        if isinstance(v, torch.Tensor)
                    ]
                    if nested_tensors:
                        tensor_summaries.append(f"{key}: {{{', '.join(nested_tensors)}}}")
        
        tensor_info = f" [{', '.join(tensor_summaries)}]" if tensor_summaries else ""
        return f"HamiltonianFlowValidationResult(valid={self.is_valid}, message='{self.message}'{tensor_info})"

    def _tensor_repr(self, tensor: Optional[Tensor], max_elements: int = 8) -> str:
        """Create a shortened string representation of tensors."""
        if tensor is None:
            return "None"
        shape = list(tensor.shape)
        if len(shape) == 0:
            return f"tensor({tensor.item():.4f})"
        if sum(shape) <= max_elements:
            return str(tensor)
        return f"tensor(shape={shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f})"


@dataclass
class HamiltonianValidation:
    """Results of Hamiltonian validation."""

    conserved: bool  # Energy conservation
    relative_error: float  # Energy error
    poisson_bracket: float  # Bracket computation
    recurrence_time: float  # Poincaré recurrence


@dataclass
class SymplecticValidation:
    """Results of symplectic validation."""

    preserved: bool  # Structure preservation
    volume_error: float  # Volume preservation error
    form_error: float  # Symplectic form error
    structure_drift: float  # Structure drift rate


@dataclass
class PhaseSpaceValidation:
    """Results of phase space validation."""

    ergodic: bool  # Ergodicity property
    mixing_rate: float  # Mixing time scale
    entropy: float  # KS entropy
    lyapunov: Tensor  # Lyapunov spectrum


class HamiltonianFlowValidator:
    """Validation of Hamiltonian flow properties."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.max_energy_drift = 0.01  # Maximum allowed energy drift
        self.max_symplectic_error = 0.01  # Maximum symplectic form error

    def validate_energy_conservation(
        self, flow: HamiltonianSystem, states: Tensor, time_steps: int = 100
    ) -> HamiltonianFlowValidationResult:
        """Validate energy conservation along flow."""
        is_valid = True
        messages = []

        # Compute initial energy
        initial_energy = flow.compute_energy(states)
        
        # Evolve system
        current_states = states
        energies = [initial_energy]
        
        for _ in range(time_steps):
            current_states = flow.evolve(current_states)
            current_energy = flow.compute_energy(current_states)
            energies.append(current_energy)
        
        energies = torch.stack(energies)
        
        # Check energy drift
        energy_drift = torch.abs(energies - initial_energy) / (torch.abs(initial_energy) + 1e-10)
        max_drift = torch.max(energy_drift)
        
        if max_drift > self.max_energy_drift:
            is_valid = False
            messages.append(f"Energy drift too large: {max_drift:.2e}")
            
        # Check energy fluctuations
        energy_std = torch.std(energies)
        if energy_std > self.tolerance:
            is_valid = False
            messages.append(f"Energy fluctuations too large: {energy_std:.2e}")

        return HamiltonianFlowValidationResult(
            is_valid=is_valid,
            message="; ".join(messages) if messages else "Energy conserved",
            data={
                "energy": {
                    "initial": initial_energy,
                    "history": energies,
                    "drift": max_drift,
                    "std": energy_std
                }
            }
        )

    def validate_symplectic_form(
        self, flow: HamiltonianSystem, states: Tensor
    ) -> HamiltonianFlowValidationResult:
        """Validate preservation of symplectic form."""
        is_valid = True
        messages = []

        # Get phase space dimension
        dim = states.shape[1] // 2

        # Construct symplectic form
        structure = SymplecticStructure(dim=2*dim)
        form = structure.compute_form(states)

        # Compute flow Jacobian
        with torch.enable_grad():
            states.requires_grad_(True)
            evolved = flow.evolve(states)
            jac_tuple = torch.autograd.functional.jacobian(flow.evolve, states)
            # Convert tuple to tensor if necessary
            jac = jac_tuple[0] if isinstance(jac_tuple, tuple) else jac_tuple

        # Check symplectic condition
        symplectic_error = torch.norm(
            torch.matmul(torch.matmul(jac, form.matrix), jac.transpose(-1, -2)) - form.matrix
        )
        if symplectic_error > self.max_symplectic_error:
            is_valid = False
            messages.append(f"Symplectic form not preserved: {symplectic_error:.2e}")

        return HamiltonianFlowValidationResult(
            is_valid=is_valid, 
            message="; ".join(messages) if messages else "Symplectic form preserved",
            data={
                "symplectic": {
                    "error": symplectic_error,
                    "jacobian": jac,
                    "form": form.matrix
                }
            }
        )

    def validate_flow(
        self, flow: HamiltonianSystem, states: Tensor, time_steps: int = 100
    ) -> HamiltonianFlowValidationResult:
        """Perform complete Hamiltonian flow validation."""
        # Check energy conservation
        energy_valid = self.validate_energy_conservation(flow, states, time_steps)
        if not energy_valid.is_valid:
            return energy_valid

        # Check symplectic form
        symplectic_valid = self.validate_symplectic_form(flow, states)
        if not symplectic_valid.is_valid:
            return symplectic_valid

        # Merge results
        return energy_valid.merge(symplectic_valid)


class HamiltonianValidator:
    """Validation of Hamiltonian conservation properties."""

    def __init__(self, tolerance: float = 1e-6, recurrence_threshold: float = 0.1):
        self.tolerance = tolerance
        self.recurrence_threshold = recurrence_threshold

    def validate_hamiltonian(
        self, system: HamiltonianSystem, state: Tensor, time_steps: int = 1000
    ) -> HamiltonianFlowValidationResult:
        """Validate Hamiltonian conservation."""
        # Track energy evolution
        initial_energy = system.compute_energy(state)
        energies = [initial_energy]
        current = state.clone()

        # Evolve system
        for _ in range(time_steps):
            current = system.evolve(current)
            energies.append(system.compute_energy(current))

        energies = torch.stack(energies)

        # Compute relative error
        relative_error = torch.abs((energies - initial_energy) / initial_energy).mean()

        # Compute Poisson bracket with Hamiltonian
        structure = SymplecticStructure(dim=state.shape[-1])
        poisson = structure.poisson_bracket(
            initial_energy,
            system.compute_energy(current),
            current,
        )

        # Estimate recurrence time
        recurrence = self._estimate_recurrence(energies, self.recurrence_threshold)

        is_valid = bool(relative_error < self.tolerance)
        message = "Hamiltonian conserved" if is_valid else f"Hamiltonian not conserved: relative error {relative_error:.2e}"

        return HamiltonianFlowValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "hamiltonian": {
                    "conserved": is_valid,
                    "relative_error": float(relative_error.item()),
                    "poisson_bracket": float(poisson.item()),
                    "recurrence_time": recurrence,
                    "energy_history": energies
                }
            }
        )

    def _estimate_recurrence(self, energies: Tensor, threshold: float) -> float:
        """Estimate Poincaré recurrence time."""
        initial = energies[0]
        diffs = torch.abs(energies - initial)
        recurrences = torch.where(diffs < threshold * torch.abs(initial))[0]

        if len(recurrences) > 1:
            return float(torch.mean(torch.diff(recurrences.float())))
        return float("inf")


class SymplecticValidator:
    """Validation of symplectic structure preservation."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_symplectic(
        self,
        system: HamiltonianSystem,
        structure: SymplecticStructure,
        state: Tensor,
        time_steps: int = 100,
    ) -> HamiltonianFlowValidationResult:
        """Validate symplectic preservation."""
        # Initial volume and form
        initial_volume = structure.compute_volume(state)
        initial_form = structure.compute_form(state)

        # Track evolution
        current = state.clone()
        volumes = [initial_volume]
        forms = [initial_form]

        for _ in range(time_steps):
            current = system.evolve(current)
            volumes.append(structure.compute_volume(current))
            forms.append(structure.compute_form(current))

        volumes = torch.stack(volumes)
        forms = torch.stack([f.matrix for f in forms])

        # Compute errors
        volume_error = torch.abs((volumes - initial_volume) / initial_volume).mean()
        form_error = torch.norm(forms - initial_form.matrix) / torch.norm(initial_form.matrix)
        structure_drift = torch.mean(torch.abs(forms[1:] - forms[:-1])) / time_steps

        is_valid = bool(volume_error < self.tolerance)
        message = (
            "Symplectic structure preserved" if is_valid 
            else f"Symplectic structure not preserved: volume error {volume_error:.2e}"
        )

        return HamiltonianFlowValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "symplectic": {
                    "preserved": is_valid,
                    "volume_error": float(volume_error.item()),
                    "form_error": float(form_error.item()),
                    "structure_drift": float(structure_drift.item()),
                    "volume_history": volumes,
                    "form_history": forms
                }
            }
        )


class PhaseSpaceValidator:
    """Validation of phase space properties."""

    def __init__(self, bins: int = 50, mixing_threshold: float = 0.1):
        self.bins = bins
        self.mixing_threshold = mixing_threshold

    def validate_phase_space(
        self, system: HamiltonianSystem, state: Tensor, time_steps: int = 1000
    ) -> HamiltonianFlowValidationResult:
        """Validate phase space properties."""
        # Track trajectory
        trajectory = [state.clone()]
        current = state.clone()

        for _ in range(time_steps):
            current = system.evolve(current)
            trajectory.append(current.clone())

        trajectory = torch.stack(trajectory)

        # Check ergodicity
        ergodic = self._check_ergodicity(trajectory)

        # Compute mixing rate
        mixing = self._compute_mixing(trajectory)

        # Compute KS entropy
        entropy = self._compute_entropy(trajectory)

        # Compute Lyapunov spectrum
        lyapunov = self._compute_lyapunov(system, state, time_steps)

        is_valid = bool(ergodic)
        message = (
            "Phase space properties valid" if is_valid
            else "Phase space properties invalid: non-ergodic behavior detected"
        )

        return HamiltonianFlowValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "phase_space": {
                    "ergodic": ergodic,
                    "mixing_rate": mixing,
                    "entropy": entropy,
                    "lyapunov": lyapunov,
                    "trajectory": trajectory
                }
            }
        )

    def _check_ergodicity(self, trajectory: Tensor) -> bool:
        """Check if trajectory appears ergodic."""
        # Compute phase space density
        density = torch.histogramdd(
            trajectory.reshape(-1, trajectory.shape[-1]), bins=self.bins
        )[0]

        # Check uniformity of coverage
        density = density / density.sum()
        uniform = 1.0 / (self.bins ** trajectory.shape[-1])

        return bool(torch.abs(density - uniform).mean() < self.mixing_threshold)

    def _compute_mixing(self, trajectory: Tensor) -> float:
        """Compute mixing rate from autocorrelation."""
        # Compute autocorrelation
        mean = torch.mean(trajectory, dim=0)
        centered = trajectory - mean

        norm = torch.sum(centered[0] ** 2)
        if norm == 0:
            return float("inf")

        correlations = []
        for t in range(len(trajectory)):
            corr = torch.sum(centered[0] * centered[t]) / norm
            correlations.append(corr.item())

        # Find mixing time
        threshold = np.exp(-1)
        for t, corr in enumerate(correlations):
            if corr < threshold:
                return float(t)

        return float("inf")

    def _compute_entropy(self, trajectory: Tensor) -> float:
        """Compute Kolmogorov-Sinai entropy estimate."""
        # Use correlation sum method
        r = 0.1  # Scale parameter
        N = len(trajectory)

        # Compute pairwise distances
        dists = torch.cdist(trajectory, trajectory)

        # Count close pairs
        counts = torch.sum(dists < r, dim=1).float()

        # Estimate correlation dimension
        correlations = torch.log(counts / N)
        entropy = torch.mean(correlations).item()

        return max(0.0, -entropy)

    def _compute_lyapunov(
        self, system: HamiltonianSystem, state: Tensor, time_steps: int
    ) -> Tensor:
        """Compute Lyapunov spectrum."""
        dim = state.shape[-1]
        perturbations = torch.eye(dim, device=state.device)

        # Track perturbation growth
        current = state.clone()
        for _ in range(time_steps):
            # Evolve state and perturbations
            current = system.evolve(current)
            with torch.enable_grad():
                current.requires_grad_(True)
                evolved = system.evolve(current)
                jac_tuple = torch.autograd.functional.jacobian(system.evolve, current)
                # Convert tuple to tensor if necessary
                jacobian = jac_tuple[0] if isinstance(jac_tuple, tuple) else jac_tuple
            perturbations = torch.matmul(jacobian, perturbations)

            # Orthogonalize
            perturbations, _ = torch.linalg.qr(perturbations)

        # Compute Lyapunov exponents
        return (
            torch.log(torch.abs(torch.diagonal(perturbations, dim1=-2, dim2=-1)))
            / time_steps
        )


class FlowValidator:
    """Complete flow validation system."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        recurrence_threshold: float = 0.1,
        mixing_threshold: float = 0.1,
        bins: int = 50,
    ):
        self.hamiltonian_validator = HamiltonianValidator(
            tolerance, recurrence_threshold
        )
        self.symplectic_validator = SymplecticValidator(tolerance)
        self.phase_space_validator = PhaseSpaceValidator(bins, mixing_threshold)

    def validate(
        self,
        system: HamiltonianSystem,
        structure: SymplecticStructure,
        state: Tensor,
        time_steps: int = 1000,
    ) -> HamiltonianFlowValidationResult:
        """Perform complete flow validation."""
        # Validate Hamiltonian properties
        hamiltonian = self.hamiltonian_validator.validate_hamiltonian(
            system, state, time_steps
        )

        # Validate symplectic properties
        symplectic = self.symplectic_validator.validate_symplectic(
            system, structure, state, time_steps
        )

        # Validate phase space properties
        phase_space = self.phase_space_validator.validate_phase_space(
            system, state, time_steps
        )

        # Merge all results
        result = hamiltonian.merge(symplectic)
        result = result.merge(phase_space)

        return result
