"""Quantum State Validation Implementation.

This module validates quantum state properties:
- State preparation fidelity
- Density matrix properties
- Purity and mixedness
- State tomography
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import torch
import numpy as np

from ..base import ValidationResult
from ...core.quantum.types import QuantumState


@dataclass
class QuantumStateValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation results for quantum states.
    
    This class handles validation of:
    - State preparation fidelity
    - Density matrix properties
    - Purity and mixedness
    - State tomography
    """
    
    def __init__(self, is_valid: bool, message: str, data: Optional[Dict[str, Any]] = None):
        """Initialize quantum state validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional validation data containing quantum metrics and tensors
        """
        super().__init__(is_valid, message, data)
    
    def merge(self, other: ValidationResult) -> 'QuantumStateValidationResult':
        """Merge with another validation result.
        
        Args:
            other: Another validation result to merge with
            
        Returns:
            New QuantumStateValidationResult combining both results
            
        Raises:
            TypeError: If other is not a ValidationResult
        """
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge metrics dictionaries carefully
        merged_data = {**(self.data or {})}
        other_data = other.data or {}
        
        # Special handling for quantum metrics
        for key, value in other_data.items():
            if key in merged_data and isinstance(value, dict):
                if isinstance(merged_data[key], dict):
                    merged_data[key].update(value)
                else:
                    merged_data[key] = value
            else:
                merged_data[key] = value
        
        return QuantumStateValidationResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data=merged_data
        )

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
            data=data.get("data", {})
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
    """Validation of quantum state preparation."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.density_validator = DensityMatrixValidator(tolerance)
        self.tomography_validator = TomographyValidator()

    def validate_preparation(
        self, target: QuantumState, prepared: QuantumState
    ) -> QuantumStateValidationResult:
        """Validate state preparation quality."""
        # Compute fidelity
        fidelity = self._compute_fidelity(target, prepared)

        # Compute trace distance
        trace_dist = self._compute_trace_distance(target, prepared)

        # Compute purity
        purity = self._compute_purity(prepared)

        # Compute concurrence (entanglement)
        concurrence = self._compute_concurrence(prepared)

        # Check if preparation is valid
        is_valid = bool(
            fidelity > (1 - self.tolerance) and
            trace_dist < self.tolerance and
            abs(purity - 1.0) < self.tolerance
        )

        # Create validation message
        if is_valid:
            message = "State preparation successful"
        else:
            issues = []
            if fidelity <= (1 - self.tolerance):
                issues.append(f"Low fidelity: {fidelity:.4f}")
            if trace_dist >= self.tolerance:
                issues.append(f"High trace distance: {trace_dist:.4f}")
            if abs(purity - 1.0) >= self.tolerance:
                issues.append(f"Non-unit purity: {purity:.4f}")
            message = "State preparation issues: " + "; ".join(issues)

        return QuantumStateValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "preparation": {
                    "fidelity": float(fidelity.item()),
                    "trace_distance": float(trace_dist.item()),
                    "purity": float(purity.item()),
                    "concurrence": float(concurrence.item())
                }
            }
        )

    def _compute_fidelity(
        self, target: QuantumState, prepared: QuantumState
    ) -> torch.Tensor:
        """Compute quantum state fidelity."""
        if target.is_pure(self.tolerance) and prepared.is_pure(self.tolerance):
            # Pure state fidelity
            overlap = torch.abs(
                torch.vdot(target.state_vector(), prepared.state_vector())
            )
            return overlap**2

        # Mixed state fidelity
        rho1 = target.density_matrix()
        rho2 = prepared.density_matrix()

        # Compute sqrt(rho1) using eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(rho1)
        sqrt_eigvals = torch.sqrt(torch.abs(eigvals))  # Take sqrt of eigenvalues
        sqrt_rho1 = torch.matmul(
            eigvecs,
            torch.matmul(torch.diag(sqrt_eigvals), eigvecs.conj().T)
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
            torch.tensor(1.0, dtype=torch.complex64),
            atol=self.tolerance,
        )

        # Compute eigenvalues
        eigenvalues = torch.real(torch.linalg.eigvals(density))

        # Check positivity
        positive = torch.all(eigenvalues > -self.tolerance)

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
        # Perform state reconstruction
        reconstructed, error = self._reconstruct_state(state, projectors)

        # Compute completeness
        completeness = self._compute_tomography_completeness(projectors)

        # Compute confidence
        confidence = self._compute_tomography_confidence(state, reconstructed, projectors)

        return TomographyValidation(
            reconstruction_error=float(error.item()),
            completeness=float(completeness.item()),
            confidence=float(confidence.item()),
            estimated_state=reconstructed
        )

    def _reconstruct_state(
        self, state: QuantumState, projectors: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform quantum state reconstruction."""
        # Initialize estimated state
        dim = projectors[0].shape[0]
        rho = torch.eye(dim, dtype=torch.complex64) / dim

        # Get measurement probabilities from true state
        true_probs = torch.stack([
            torch.trace(torch.matmul(state.density_matrix(), proj))
            for proj in projectors
        ])

        # Maximum likelihood estimation
        error = torch.tensor(float('inf'))
        for _ in range(1000):  # Maximum iterations
            # Compute expected measurements
            expected = torch.stack([
                torch.trace(torch.matmul(rho, proj))
                for proj in projectors
            ])

            # Compute error
            new_error = torch.norm(expected - true_probs)
            if torch.abs(error - new_error) < 1e-6:
                break
            error = new_error

            # Update state estimate
            gradient = sum(
                (true_probs[i] - expected[i].real) * projectors[i]
                for i in range(len(projectors))
            )
            rho = rho + 0.01 * gradient
            rho = 0.5 * (rho + rho.conj().T)  # Ensure Hermiticity
            rho = rho / torch.trace(rho)  # Normalize

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
            torch.trace(torch.matmul(state.density_matrix(), proj))
            for proj in projectors
        ])

        # Compute expected measurements from reconstructed state
        expected = torch.stack([
            torch.trace(torch.matmul(reconstructed, proj))
            for proj in projectors
        ])

        # Compute chi-squared statistic
        chi_squared = torch.sum((true_probs - expected.real)**2 / (true_probs + 1e-10))
        
        # Convert to confidence level (using exponential decay)
        confidence = torch.exp(-chi_squared / len(projectors))
        return confidence


class DensityMatrixValidator:
    """Validation of density matrix properties."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_density_matrix(self, state: QuantumState) -> QuantumStateValidationResult:
        """Validate density matrix properties."""
        rho = state.density_matrix()

        # Check Hermiticity
        hermitian = torch.allclose(rho, rho.conj().T, atol=self.tolerance)

        # Check trace normalization
        trace = torch.trace(rho)
        trace_one = torch.allclose(
            trace,
            torch.tensor(1.0, dtype=torch.complex64),
            atol=self.tolerance,
        )

        # Compute eigenvalues
        eigenvalues = torch.real(torch.linalg.eigvals(rho))

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

        return QuantumStateValidationResult(
            is_valid=is_valid,
            message=message,
            data={
                "density_matrix": {
                    "hermitian": bool(hermitian),
                    "positive": bool(positive),
                    "trace_one": bool(trace_one),
                    "trace": complex(trace.item()),
                    "eigenvalues": eigenvalues.tolist(),
                    "min_eigenvalue": float(torch.min(eigenvalues).item()),
                    "max_eigenvalue": float(torch.max(eigenvalues).item())
                }
            }
        )


class TomographyValidator:
    """Validation of quantum state tomography."""

    def __init__(self, confidence_level: float = 0.95, max_iterations: int = 1000):
        self.confidence_level = confidence_level
        self.max_iterations = max_iterations

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
        """Perform quantum state reconstruction."""
        # Initialize estimated state
        dim = bases[0].shape[0]
        rho = torch.eye(dim, dtype=torch.complex64) / dim

        # Maximum likelihood estimation
        error = torch.tensor(float('inf'))
        for _ in range(self.max_iterations):
            # Compute expected measurements
            expected = torch.stack([
                torch.trace(torch.matmul(rho, basis))
                for basis in bases
            ])

            # Compute error
            new_error = torch.norm(expected - measurements)
            if torch.abs(error - new_error) < 1e-6:
                break
            error = new_error

            # Update state estimate
            gradient = sum(
                (measurements[i] - expected[i].real) * bases[i]
                for i in range(len(bases))
            )
            rho = rho + 0.01 * gradient
            rho = 0.5 * (rho + rho.conj().T)  # Ensure Hermiticity
            rho = rho / torch.trace(rho)  # Normalize

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
            torch.trace(torch.matmul(state, basis))
            for basis in bases
        ])

        # Compute chi-squared statistic
        chi_squared = torch.sum((measurements - expected.real)**2 / measurements)
        
        # Convert to confidence level
        from scipy.stats import chi2
        dof = len(measurements) - state.shape[0]**2 + 1  # Degrees of freedom
        return torch.tensor(1 - chi2.cdf(chi_squared.item(), dof))

    def _compute_fidelity(
        self, true_state: QuantumState, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute fidelity between true and reconstructed states."""
        if true_state.is_pure():
            # Pure state fidelity
            psi = true_state.state_vector()
            return torch.abs(torch.vdot(psi, torch.matmul(reconstructed, psi)))
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


class QuantumStateValidator:
    """Complete quantum state validation system."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        confidence_level: float = 0.95,
        max_iterations: int = 1000,
    ):
        self.preparation_validator = StatePreparationValidator(tolerance)
        self.density_validator = DensityMatrixValidator(tolerance)
        self.tomography_validator = TomographyValidator(
            confidence_level, max_iterations
        )

    def validate(
        self,
        target: QuantumState,
        prepared: QuantumState,
        measurements: torch.Tensor,
        bases: List[torch.Tensor],
    ) -> QuantumStateValidationResult:
        """Perform complete quantum state validation.
        
        Args:
            target: Target quantum state
            prepared: Prepared quantum state to validate
            measurements: Measurement results
            bases: Measurement bases
            
        Returns:
            Combined validation result from all validators
        """
        # Validate state preparation
        preparation = self.preparation_validator.validate_preparation(target, prepared)

        # Validate density matrix
        density = self.density_validator.validate_density_matrix(prepared)

        # Validate tomography
        tomography = self.tomography_validator.validate_tomography(
            target, measurements, bases
        )

        # Merge all results
        result = preparation.merge(density)
        result = result.merge(tomography)
        return result
