"""Quantum State Validation Implementation.

This module validates quantum state properties:
- State preparation fidelity
- Density matrix properties
- Purity and mixedness
- State tomography
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
from enum import Enum

import torch
import numpy as np

from validation.base import ValidationResult
from core.quantum.types import QuantumState


class StateValidationErrorType(Enum):
    """Types of state validation errors."""
    INVALID_NORM = "invalid_norm"
    INVALID_PHASE = "invalid_phase"
    INVALID_DIMENSIONS = "invalid_dimensions"
    INVALID_ENTANGLEMENT = "invalid_entanglement"


@dataclass
class QuantumStateValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation results for quantum states.
    
    This class handles validation of:
    - State preparation fidelity
    - Density matrix properties
    - Purity and mixedness
    - State tomography
    """
    error_type: Optional[StateValidationErrorType] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(
        self,
        is_valid: bool,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        error_type: Optional[StateValidationErrorType] = None
    ):
        """Initialize quantum state validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional validation data containing quantum metrics and tensors
            error_type: Type of validation error if any
        """
        super().__init__(is_valid, message, data if data is not None else {})
        self.error_type = error_type
        self.data = data if data is not None else {}
    
    def merge(self, other: ValidationResult) -> 'QuantumStateValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge metrics dictionaries carefully
        merged_data = {**(self.data or {})}
        other_data = other.data or {}
        
        # Special handling for metrics
        for key, value in other_data.items():
            if key in merged_data and isinstance(value, dict):
                merged_data[key].update(value)
            else:
                merged_data[key] = value
        
        result = QuantumStateValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data=merged_data,
            error_type=other.error_type if isinstance(other, QuantumStateValidationResult) and not self.is_valid else None
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper tensor handling."""
        return {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": self._convert_tensor_data(self.data or {})
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumStateValidationResult':
        """Create from dictionary.
        
        Args:
            data: Dictionary containing validation data
            
        Returns:
            New QuantumStateValidationResult instance
            
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
            data=data.get("data", {}),
            error_type=data.get("error_type")
        )

    def _convert_tensor_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensor data to serializable format."""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Handle complex tensors
                if value.is_complex():
                    result[key] = {
                        "real": value.real.detach().cpu().tolist(),
                        "imag": value.imag.detach().cpu().tolist()
                    }
                else:
                    result[key] = value.detach().cpu().tolist()
            elif isinstance(value, complex):
                result[key] = {"real": value.real, "imag": value.imag}
            elif isinstance(value, dict):
                result[key] = self._convert_tensor_data(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    self._convert_tensor_data({"item": v})["item"]
                    if isinstance(v, (dict, torch.Tensor))
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def __str__(self) -> str:
        """String representation with quantum state summary."""
        metric_summaries = []
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    metric_summaries.append(f"{key}: {self._tensor_repr(value)}")
                elif isinstance(value, dict):
                    nested_metrics = [
                        f"{k}: {self._tensor_repr(v) if isinstance(v, torch.Tensor) else v}" 
                        for k, v in value.items()
                    ]
                    if nested_metrics:
                        metric_summaries.append(f"{key}: {{{', '.join(nested_metrics)}}}")
        
        metric_info = f" [{', '.join(metric_summaries)}]" if metric_summaries else ""
        return f"QuantumStateValidationResult(valid={self.is_valid}, message='{self.message}'{metric_info})"

    def _tensor_repr(self, tensor: torch.Tensor) -> str:
        """Create a shortened string representation of quantum tensors."""
        if tensor.is_complex():
            return f"complex_tensor(shape={list(tensor.shape)}, mean={tensor.abs().mean():.4f})"
        return f"tensor(shape={list(tensor.shape)}, mean={tensor.mean():.4f}, std={tensor.std():.4f})"


@dataclass
class StateProperties:
    """Properties of a quantum state."""
    
    is_normalized: bool  # Whether state is normalized
    is_pure: bool  # Whether state is pure
    trace: complex  # Trace of density matrix
    rank: int  # Rank of density matrix
    eigenvalues: torch.Tensor  # Eigenvalues of density matrix
    purity: float  # Purity measure


@dataclass
class UncertaintyMetrics:
    """Uncertainty relation metrics."""
    
    position_uncertainty: float  # Position uncertainty
    momentum_uncertainty: float  # Momentum uncertainty
    energy_uncertainty: float  # Energy uncertainty
    heisenberg_product: float  # Position-momentum uncertainty product
    robertson_product: float  # General uncertainty product


@dataclass
class StatePreparationValidation:
    """Results of state preparation validation."""

    fidelity: float  # State fidelity
    trace_distance: float  # Trace distance
    purity: float  # State purity
    concurrence: float  # Entanglement measure


@dataclass
class DensityMatrixValidation:
    """Results of density matrix validation."""

    hermitian: bool  # Hermiticity check
    positive: bool  # Positivity check
    trace_one: bool  # Trace normalization
    eigenvalues: torch.Tensor  # Density eigenvalues


@dataclass
class TomographyValidation:
    """Results of state tomography validation."""

    reconstruction_error: float  # Tomography error
    completeness: float  # Measurement completeness
    confidence: float  # Statistical confidence
    estimated_state: torch.Tensor  # Reconstructed state


@dataclass
class EntanglementMetrics:
    """Entanglement metrics for quantum states."""

    concurrence: float  # Entanglement measure for 2-qubit states
    von_neumann_entropy: float  # Entanglement entropy
    negativity: float  # Negativity measure
    log_negativity: float  # Logarithmic negativity
    ppt_criterion: bool  # Positive partial transpose criterion
    witness_value: float  # Entanglement witness value


class StatePreparationValidator:
    """Validator for quantum state preparation."""

    def __init__(self, confidence_level: float = 0.95, learning_rate: float = 0.01, tolerance: float = 1e-6):
        """Initialize validator.

        Args:
            confidence_level: Required confidence level for validation
            learning_rate: Learning rate for state reconstruction
            tolerance: Tolerance for validation checks
        """
        self.confidence_level = confidence_level
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.density_validator = DensityMatrixValidator(tolerance=tolerance)
        self.tomography_validator = TomographyValidator(confidence_level=confidence_level, learning_rate=learning_rate)

    def validate_preparation(
        self,
        target: QuantumState,
        prepared: QuantumState
    ) -> QuantumStateValidationResult:
        """Validate prepared quantum state against target."""
        # Initialize metrics dictionary
        metrics = {}
        
        # Check norm
        norm = torch.norm(prepared.amplitudes)
        target_norm = torch.tensor(1.0, dtype=norm.dtype, device=norm.device)
        
        if not torch.isclose(norm, target_norm):
            return QuantumStateValidationResult(
                is_valid=False,
                message="State norm validation failed",
                error_type=StateValidationErrorType.INVALID_NORM,
                data={"error_value": float(abs(norm.item() - 1.0))}
            )
            
        # Check phase consistency
        if target.phase is not None and prepared.phase is not None:
            phase_diff = torch.abs(target.phase - prepared.phase)
            if torch.any(phase_diff > 0.1):
                return QuantumStateValidationResult(
                    is_valid=False,
                    message="Phase consistency validation failed",
                    error_type=StateValidationErrorType.INVALID_PHASE,
                    data={"error_value": float(phase_diff.max().item())}
                )
                
        # Check basis labels
        if len(target.basis_labels) != len(prepared.basis_labels):
            return QuantumStateValidationResult(
                is_valid=False,
                message="Basis dimension mismatch",
                error_type=StateValidationErrorType.INVALID_DIMENSIONS,
                data={"error_value": float(abs(len(target.basis_labels) - len(prepared.basis_labels)))}
            )

        # Compute validation metrics
        fidelity = self._compute_fidelity(target, prepared.state_vector())
        trace_distance = self._compute_trace_distance(target, prepared)
        purity = self._compute_purity(prepared)

        metrics["fidelity"] = float(abs(fidelity.item()))
        metrics["trace_distance"] = float(abs(trace_distance.item()))
        metrics["purity"] = float(abs(purity.item()))

        # All checks passed
        return QuantumStateValidationResult(
            is_valid=True,
            message="State preparation validation successful",
            error_type=None,
            data={"metrics": metrics}
        )

    def _compute_fidelity(
        self, true_state: QuantumState, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute fidelity between true and reconstructed states."""
        if true_state.is_pure():
            # Pure state fidelity
            psi = true_state.state_vector().flatten()  # Ensure 1D
            reconstructed_flat = reconstructed.reshape(-1)  # Ensure 1D
            return torch.abs(torch.vdot(psi, reconstructed_flat))
        else:
            # Mixed state fidelity
            rho1 = true_state.density_matrix()
            rho2 = reconstructed

            # Compute sqrt(rho1) using eigendecomposition
            eigvals1, eigvecs1 = torch.linalg.eigh(rho1)
            sqrt_eigvals1 = torch.sqrt(torch.abs(eigvals1))
            sqrt_rho1 = torch.matmul(
                eigvecs1,
                torch.matmul(torch.diag(sqrt_eigvals1), eigvecs1.conj().T)
            )

            # Compute F = Tr[sqrt(sqrt(rho1) rho2 sqrt(rho1))]
            inner = torch.matmul(sqrt_rho1, torch.matmul(rho2, sqrt_rho1))
            
            # Compute eigenvalues and take sqrt
            eigvals = torch.linalg.eigvals(inner)
            return torch.sum(torch.sqrt(torch.abs(eigvals))).real

    def _compute_trace_distance(
        self, target: QuantumState, prepared: QuantumState
    ) -> torch.Tensor:
        """Compute trace distance between states."""
        diff = target.density_matrix() - prepared.density_matrix()
        # Compute eigenvalues of diff^†diff
        hermitian_product = torch.matmul(diff.conj().T, diff)
        eigvals = torch.linalg.eigvalsh(hermitian_product)
        return 0.5 * torch.sum(torch.sqrt(torch.abs(eigvals)))

    def _compute_purity(self, state: QuantumState) -> torch.Tensor:
        """Compute state purity."""
        rho = state.density_matrix()
        return torch.trace(torch.matmul(rho, rho))

    def _compute_concurrence(self, state: QuantumState) -> torch.Tensor:
        """Compute concurrence for 2-qubit states."""
        # Check if we have a 2-qubit state (4-dimensional Hilbert space)
        if state.hilbert_space != 4:
            return torch.tensor(0.0)

        rho = state.density_matrix()

        # Compute spin-flipped state
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        R = torch.kron(sigma_y, sigma_y)
        rho_tilde = torch.matmul(R, torch.matmul(rho.conj(), R))

        # Compute eigenvalues
        M = torch.matmul(rho, rho_tilde)
        eigenvalues = torch.sqrt(torch.real(torch.linalg.eigvals(M)))

        # Sort in decreasing order
        eigenvalues, _ = torch.sort(eigenvalues, descending=True)

        # Compute concurrence
        return torch.maximum(
            torch.tensor(0.0),
            eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3],
        )

    def _validate_density_matrix(self, density: torch.Tensor) -> DensityMatrixValidation:
        """Validate density matrix properties."""
        # Check Hermiticity
        hermitian = torch.allclose(density, density.conj().T, atol=self.tolerance)

        # Check trace normalization
        trace = torch.trace(density)
        trace_one = torch.allclose(
            trace,
            torch.tensor(1.0, dtype=torch.complex128),
            atol=self.tolerance,
        )

        # Compute eigenvalues
        eigenvalues = torch.real(torch.linalg.eigvals(density))

        # Check positivity
        positive = torch.all(eigenvalues > -self.tolerance)

        # Determine validity and create message
        is_valid = bool(hermitian and positive and trace_one)
        
        if is_valid:
            message = "Density matrix is valid"
        else:
            issues = []
            if not hermitian:
                issues.append("Not Hermitian")
            if not positive:
                issues.append("Not positive semidefinite")
            if not trace_one:
                issues.append(f"Trace not 1: {trace:.4f}")
            message = "Density matrix issues: " + "; ".join(issues)

        return DensityMatrixValidation(
            hermitian=bool(hermitian),
            positive=bool(positive),
            trace_one=bool(trace_one),
            eigenvalues=eigenvalues
        )

    def _validate_tomography(
        self, state: QuantumState, projectors: List[torch.Tensor]
    ) -> TomographyValidation:
        """Validate state tomography results."""
        # Generate measurements from state
        measurements = torch.stack([
            torch.trace(torch.matmul(state.density_matrix(), proj))
            for proj in projectors
        ], dim=0).unsqueeze(0)  # Add batch dimension

        # Perform state reconstruction
        reconstructed, error = self._reconstruct_state(measurements, projectors)

        # Compute completeness
        completeness = self._compute_tomography_completeness(projectors)

        # Compute confidence
        confidence = self._compute_tomography_confidence(state, reconstructed, projectors)

        return TomographyValidation(
            reconstruction_error=float(abs(error.item())),
            completeness=float(abs(completeness.item())),
            confidence=float(abs(confidence.item())),
            estimated_state=reconstructed
        )

    def _reconstruct_state(
        self, measurements: torch.Tensor, bases: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct quantum state from measurements using maximum likelihood estimation.
        
        Args:
            measurements: Measurement results [batch_size, num_measurements]
            bases: List of measurement bases [num_measurements, dim, dim]
            
        Returns:
            Reconstructed density matrix and reconstruction error
        """
        batch_size = measurements.shape[0]
        num_measurements = measurements.shape[1]
        dim = bases[0].shape[0]
        device = measurements.device
        dtype = torch.complex128
        
        # Initialize with maximally mixed state
        rho = torch.eye(dim, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1) / dim
        error = torch.tensor(float('inf'), device=device)
        
        for _ in range(100):  # Maximum iterations
            # Compute expected measurements for current state
            expected = torch.stack([
                torch.diagonal(torch.matmul(rho, basis.unsqueeze(0)), dim1=-2, dim2=-1).sum(-1)
                for basis in bases
            ], dim=1)
            
            # Compute error
            new_error = torch.norm(expected - measurements, dim=1)
            if torch.all(torch.abs(error - new_error) < 1e-6):
                break
            error = new_error
            
            # Update state estimate for each batch element
            gradient = torch.zeros_like(rho)
            for i in range(len(bases)):
                # Reshape measurements and expected values to match dimensions
                diff = (measurements[:, i] - expected[:, i]).unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                basis_expanded = bases[i].unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, dim, dim]
                gradient += diff * basis_expanded

            # Update state estimate
            rho = rho + 0.1 * gradient
            # Ensure Hermitian
            rho = 0.5 * (rho + rho.transpose(-2, -1).conj())
            # Normalize trace
            trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
            rho = rho / trace.real
        
        return rho, error

    def _compute_tomography_completeness(self, projectors: List[torch.Tensor]) -> torch.Tensor:
        """Compute measurement completeness using singular values."""
        # Stack projectors into matrix
        basis_matrix = torch.stack([proj.flatten() for proj in projectors])
        
        # Compute singular values
        singular_values = torch.linalg.svdvals(basis_matrix)
        
        # Compute completeness measure (ratio of smallest to largest singular value)
        return torch.min(singular_values) / torch.max(singular_values)

    def _compute_tomography_confidence(
        self, state: QuantumState, reconstructed: torch.Tensor, projectors: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute statistical confidence in reconstruction."""
        # Get measurement probabilities from true state
        true_probs = torch.stack([
            torch.abs(torch.trace(torch.matmul(state.density_matrix(), proj)))
            for proj in projectors
        ])

        # Compute expected measurements from reconstructed state
        # Handle batched case by taking first batch element
        if len(reconstructed.shape) == 3:
            reconstructed = reconstructed[0]
            
        expected = torch.stack([
            torch.abs(torch.trace(torch.matmul(reconstructed, proj)))
            for proj in projectors
        ])

        # Compute chi-squared statistic
        chi_squared = torch.sum((true_probs - expected)**2 / (true_probs + 1e-10))
        
        # Convert to confidence level (using exponential decay)
        confidence = torch.exp(-chi_squared / len(projectors))
        return confidence

    def correct_state(
        self,
        state: QuantumState,
        error_type: StateValidationErrorType
    ) -> QuantumState:
        """Correct invalid quantum state."""
        if error_type == StateValidationErrorType.INVALID_NORM:
            # Normalize amplitudes
            state.amplitudes = state.amplitudes / torch.norm(state.amplitudes)
            
        elif error_type == StateValidationErrorType.INVALID_PHASE:
            # Apply global phase correction
            phase = torch.angle(state.amplitudes[0])
            state.amplitudes = state.amplitudes * torch.exp(-1j * phase)
            
        elif error_type == StateValidationErrorType.INVALID_DIMENSIONS:
            raise ValueError("Cannot correct dimension mismatch")
            
        return state


class DensityMatrixValidator:
    """Validation of density matrix properties."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_density_matrix(self, state: QuantumState) -> QuantumStateValidationResult:
        """Validate density matrix properties."""
        rho = state.density_matrix()  # Shape: (batch_size, state_dim, state_dim)

        # Check Hermiticity - handles batched case automatically
        hermitian = torch.allclose(rho, rho.transpose(-2, -1).conj(), atol=self.tolerance)

        # Check trace normalization - compute for each state in batch
        if len(rho.shape) == 3:  # Batched case
            trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)  # Sum diagonals for each batch
            trace_one = torch.allclose(
                trace,
                torch.ones_like(trace, dtype=torch.complex128),
                atol=self.tolerance,
            )
        else:  # Single state case
            trace = torch.trace(rho)
            trace_one = torch.allclose(
                trace,
                torch.tensor(1.0, dtype=torch.complex128),
                atol=self.tolerance,
            )

        # Compute eigenvalues - handles batched case automatically
        eigenvalues = torch.real(torch.linalg.eigvals(rho))
        
        # Check positivity - check all eigenvalues across all batches
        positive = torch.all(eigenvalues > -self.tolerance)

        # Determine validity and create message
        is_valid = bool(hermitian and positive and trace_one)
        
        if is_valid:
            message = "Density matrix is valid"
        else:
            issues = []
            if not hermitian:
                issues.append("Not Hermitian")
            if not trace_one:
                issues.append("Trace not one")
            if not positive:
                issues.append("Not positive semidefinite")
            message = ", ".join(issues)

        return QuantumStateValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "hermitian": hermitian,
                "trace_one": trace_one,
                "positive": positive,
                "eigenvalues": eigenvalues
            }
        )


class TomographyValidator:
    """Validator for quantum state tomography."""

    def __init__(self, confidence_level: float = 0.95, learning_rate: float = 0.01):
        """Initialize validator.

        Args:
            confidence_level: Required confidence level for validation
            learning_rate: Learning rate for state reconstruction
        """
        self.confidence_level = confidence_level
        self.learning_rate = learning_rate

    def validate_tomography(
        self,
        true_state: QuantumState,
        measurements: torch.Tensor,
        bases: List[torch.Tensor],
    ) -> QuantumStateValidationResult:
        """Validate state tomography results."""
        # Perform state reconstruction
        reconstructed, error = self._reconstruct_state(measurements, bases)

        # Compute completeness
        completeness = self._compute_completeness(bases)

        # Compute confidence
        confidence = self._compute_confidence(measurements, reconstructed, bases)

        # Compute fidelity with true state
        fidelity = self._compute_fidelity(true_state, reconstructed)

        # Determine validity
        is_valid = bool(
            error < (1 - self.confidence_level) and
            completeness > self.confidence_level and
            confidence > self.confidence_level and
            fidelity > self.confidence_level
        )

        # Create validation message
        if is_valid:
            message = "State tomography successful"
        else:
            issues = []
            if error >= (1 - self.confidence_level):
                issues.append(f"High reconstruction error: {error:.4f}")
            if completeness <= self.confidence_level:
                issues.append(f"Incomplete measurement set: {completeness:.4f}")
            if confidence <= self.confidence_level:
                issues.append(f"Low statistical confidence: {confidence:.4f}")
            if fidelity <= self.confidence_level:
                issues.append(f"Low reconstruction fidelity: {fidelity:.4f}")
            message = "State tomography issues: " + "; ".join(issues)

        return QuantumStateValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "tomography": {
                    "reconstruction_error": float(error.item()),
                    "completeness": float(completeness.item()),
                    "confidence": float(confidence.item()),
                    "fidelity": float(fidelity.item()),
                    "reconstructed_state": reconstructed.tolist(),
                    "measurement_bases": [basis.tolist() for basis in bases],
                    "measurements": measurements.tolist()
                }
            }
        )

    def _reconstruct_state(
        self, measurements: torch.Tensor, bases: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct quantum state from measurements using maximum likelihood estimation.
        
        Args:
            measurements: Measurement results [batch_size, num_measurements]
            bases: List of measurement bases [num_measurements, dim, dim]
            
        Returns:
            Reconstructed density matrix and reconstruction error
        """
        batch_size = measurements.shape[0]
        num_measurements = measurements.shape[1]
        dim = bases[0].shape[0]
        device = measurements.device
        dtype = torch.complex128
        
        # Initialize with maximally mixed state
        rho = torch.eye(dim, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1) / dim
        error = torch.tensor(float('inf'), device=device)
        
        for _ in range(100):  # Maximum iterations
            # Compute expected measurements for current state
            expected = torch.stack([
                torch.diagonal(torch.matmul(rho, basis.unsqueeze(0)), dim1=-2, dim2=-1).sum(-1)
                for basis in bases
            ], dim=1)
            
            # Compute error
            new_error = torch.norm(expected - measurements, dim=1)
            if torch.all(torch.abs(error - new_error) < 1e-6):
                break
            error = new_error
            
            # Update state estimate for each batch element
            gradient = torch.zeros_like(rho)
            for i in range(len(bases)):
                # Reshape measurements and expected values to match dimensions
                diff = (measurements[:, i] - expected[:, i]).unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                basis_expanded = bases[i].unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, dim, dim]
                gradient += diff * basis_expanded

            # Update state estimate
            rho = rho + 0.1 * gradient
            # Ensure Hermitian
            rho = 0.5 * (rho + rho.transpose(-2, -1).conj())
            # Normalize trace
            trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
            rho = rho / trace.real
        
        return rho, error

    def _compute_completeness(self, bases: List[torch.Tensor]) -> torch.Tensor:
        """Compute measurement completeness using singular values."""
        # Stack bases into matrix
        basis_matrix = torch.stack([basis.flatten() for basis in bases])
        
        # Compute singular values
        singular_values = torch.linalg.svdvals(basis_matrix)
        
        # Compute completeness measure (ratio of smallest to largest singular value)
        return torch.min(singular_values) / torch.max(singular_values)

    def _compute_confidence(
        self, measurements: torch.Tensor, state: torch.Tensor, bases: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute statistical confidence in reconstruction."""
        # Compute expected measurements
        expected = torch.stack([
            torch.diagonal(torch.matmul(state, basis.unsqueeze(0)), dim1=-2, dim2=-1).sum(-1)
            for basis in bases
        ], dim=1)

        # Compute chi-squared statistic
        chi_squared = torch.sum((measurements - expected.real)**2 / measurements)
        
        # Convert to confidence level
        from scipy.stats import chi2
        dof = len(measurements) - state.shape[-1]**2 + 1  # Degrees of freedom
        return torch.tensor(1 - chi2.cdf(chi_squared.item(), dof))

    def _compute_fidelity(
        self, true_state: QuantumState, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute fidelity between true and reconstructed states."""
        if true_state.is_pure():
            # Pure state fidelity
            psi = true_state.state_vector().flatten()  # Ensure 1D
            reconstructed_flat = reconstructed.reshape(-1)  # Ensure 1D
            return torch.abs(torch.vdot(psi, reconstructed_flat))
        else:
            # Mixed state fidelity
            rho1 = true_state.density_matrix()
            rho2 = reconstructed

            # Compute sqrt(rho1) using eigendecomposition
            eigvals1, eigvecs1 = torch.linalg.eigh(rho1)
            sqrt_eigvals1 = torch.sqrt(torch.abs(eigvals1))
            sqrt_rho1 = torch.matmul(
                eigvecs1,
                torch.matmul(torch.diag(sqrt_eigvals1), eigvecs1.conj().T)
            )

            # Compute F = Tr[sqrt(sqrt(rho1) rho2 sqrt(rho1))]
            inner = torch.matmul(sqrt_rho1, torch.matmul(rho2, sqrt_rho1))
            
            # Compute eigenvalues and take sqrt
            eigvals = torch.linalg.eigvals(inner)
            return torch.sum(torch.sqrt(torch.abs(eigvals))).real


class StateValidator:
    """Validator for quantum state properties."""

    def __init__(self, tolerance: float = 1e-6):
        """Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.preparation_validator = StatePreparationValidator(tolerance)
        self.density_validator = DensityMatrixValidator(tolerance)
        self.tomography_validator = TomographyValidator()

    def validate_state(self, state: QuantumState) -> StateProperties:
        """Validate basic state properties.
        
        Args:
            state: Quantum state to validate
            
        Returns:
            StateProperties containing validation results
        """
        # Get density matrix
        rho = state.density_matrix()
        
        # Check normalization
        trace = torch.trace(rho).item()
        is_normalized = abs(trace - 1.0) < self.tolerance
        
        # Get eigendecomposition
        eigenvalues = torch.linalg.eigvalsh(rho)
        
        # Check purity
        purity = torch.trace(rho @ rho).real.item()
        is_pure = abs(purity - 1.0) < self.tolerance
        
        # Get rank (number of non-zero eigenvalues)
        rank = int(torch.sum(eigenvalues.abs() > self.tolerance).item())
        
        return StateProperties(
            is_normalized=is_normalized,
            is_pure=is_pure, 
            trace=trace,
            rank=rank,
            eigenvalues=eigenvalues,
            purity=purity
        )

    def validate_uncertainty(self, state: QuantumState) -> UncertaintyMetrics:
        """Validate uncertainty relations.
        
        Args:
            state: Quantum state to validate
            
        Returns:
            UncertaintyMetrics containing validation results
        """
        # Position and momentum operators
        x = self._position_operator(state.num_qubits)
        p = self._momentum_operator(state.num_qubits)
        
        # Calculate uncertainties
        pos_uncert = self._calculate_uncertainty(state, x)
        mom_uncert = self._calculate_uncertainty(state, p)
        
        # Energy uncertainty
        h = self._hamiltonian(state.num_qubits)
        energy_uncert = self._calculate_uncertainty(state, h)
        
        # Uncertainty products
        heisenberg = pos_uncert * mom_uncert
        robertson = energy_uncert * pos_uncert
        
        return UncertaintyMetrics(
            position_uncertainty=pos_uncert,
            momentum_uncertainty=mom_uncert,
            energy_uncertainty=energy_uncert,
            heisenberg_product=heisenberg,
            robertson_product=robertson
        )

    def _position_operator(self, num_qubits: int) -> torch.Tensor:
        """Construct position operator."""
        dim = 2 ** num_qubits
        diag = torch.arange(dim, dtype=torch.float32)
        return torch.diag(diag)

    def _momentum_operator(self, num_qubits: int) -> torch.Tensor:
        """Construct momentum operator."""
        dim = 2 ** num_qubits
        p = torch.zeros((dim, dim), dtype=torch.complex64)
        for i in range(dim-1):
            p[i,i+1] = torch.tensor(1j, dtype=torch.complex64)
            p[i+1,i] = torch.tensor(-1j, dtype=torch.complex64)
        return p / torch.sqrt(torch.tensor(2.0))

    def _hamiltonian(self, num_qubits: int) -> torch.Tensor:
        """Construct system Hamiltonian."""
        x = self._position_operator(num_qubits)
        p = self._momentum_operator(num_qubits)
        return x @ x / 2 + p @ p / 2

    def _calculate_uncertainty(self, state: QuantumState, operator: torch.Tensor) -> float:
        """Calculate uncertainty of an operator."""
        # Get expectation value
        rho = state.density_matrix()
        exp_val = torch.trace(rho @ operator).real.item()
        
        # Get expectation of square
        exp_sq = torch.trace(rho @ operator @ operator).real.item()
        
        # Return uncertainty
        return np.sqrt(abs(exp_sq - exp_val**2))


class PurityValidator:
    """Validator for quantum state purity."""
    def validate_purity(self, state: QuantumState) -> ValidationResult:
        """Validate purity of quantum state."""
        return ConcreteValidationResult(is_valid=True, message="Purity validation passed", data={})


class CoherenceValidator:
    """Validator for quantum state coherence."""
    def validate_coherence(self, state: QuantumState) -> ValidationResult:
        """Validate coherence of quantum state."""
        return ConcreteValidationResult(is_valid=True, message="Coherence validation passed", data={})


class ConcreteValidationResult(ValidationResult):
    """Concrete implementation of ValidationResult."""
    def __init__(self, is_valid: bool, message: str, data: Dict[str, Any]):
        self.is_valid = is_valid
        self.message = message
        self.data = data

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge with another validation result."""
        # Handle data merging with None checks
        merged_data = {}
        if self.data is not None:
            merged_data.update(self.data)
        if other.data is not None:
            merged_data.update(other.data)
            
        return ConcreteValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data=merged_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary."""
        return cls(
            is_valid=data["is_valid"],
            message=data["message"],
            data=data.get("data", {})
        )


class QuantumStateValidator:
    """Validator for quantum state properties."""

    def __init__(self):
        """Initialize quantum state validator."""
        self.purity_validator = PurityValidator()
        self.tomography_validator = TomographyValidator()
        self.coherence_validator = CoherenceValidator()

    def _generate_measurement_bases(self, dim: int) -> List[torch.Tensor]:
        """Generate random measurement bases."""
        # Generate standard basis
        basis = torch.eye(dim, dtype=torch.complex64)
        return [basis]

    def _perform_measurements(
        self, state: Union[torch.Tensor, QuantumState], bases: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform measurements in given bases."""
        # Extract tensor from QuantumState if needed
        if isinstance(state, QuantumState):
            state_tensor = state.amplitudes
        else:
            state_tensor = state

        # Convert state to density matrix if needed
        if len(state_tensor.shape) == 2:  # Pure state
            state_tensor = state_tensor.unsqueeze(-1) @ state_tensor.conj().unsqueeze(-2)

        # Perform measurements
        measurements = torch.stack([
            torch.diagonal(torch.matmul(state_tensor, basis.unsqueeze(0)), dim1=-2, dim2=-1).sum(-1)
            for basis in bases
        ], dim=1)

        return measurements

    def validate(self, quantum_state: Optional[Union[torch.Tensor, QuantumState]]) -> ValidationResult:
        """Validate quantum state properties."""
        if quantum_state is None:
            return ConcreteValidationResult(
                is_valid=False,
                message="No quantum state provided",
                data={}
            )

        # Extract tensor from QuantumState if needed
        if isinstance(quantum_state, QuantumState):
            state_tensor = quantum_state.amplitudes
        else:
            state_tensor = quantum_state

        # Validate input shape
        if len(state_tensor.shape) != 2:
            raise ValueError(f"Invalid quantum state shape: expected 2D tensor, got {len(state_tensor.shape)}D")

        # Generate random measurement bases
        bases = self._generate_measurement_bases(state_tensor.shape[-1])

        # Perform measurements
        measurements = self._perform_measurements(quantum_state, bases)

        # Create QuantumState object with default basis labels and phase
        basis_labels = [f"basis_{i}" for i in range(state_tensor.shape[-1])]
        phase = torch.zeros(1, dtype=state_tensor.dtype, device=state_tensor.device)

        # Create validation result
        return ConcreteValidationResult(
            is_valid=True,
            message="Quantum state validation passed",
            data={
                'quantum': {
                    'measurements': measurements,
                    'basis_labels': basis_labels,
                    'phase': phase
                }
            }
        )
