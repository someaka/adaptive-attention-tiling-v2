"""
Unit tests for quantum state validation.

Tests cover:
1. State preparation fidelity
2. Density matrix properties
3. Purity and mixedness
4. State tomography
"""

from typing import Dict, List

import numpy as np
import pytest
import torch

from src.validation.quantum.state import (
    EntanglementMetrics,
    StateProperties,
    StatePreparationValidator,
    StatePreparationValidation,
    DensityMatrixValidation,
    TomographyValidation,
    QuantumStateValidationResult,
)
from src.core.quantum.types import QuantumState


class TestStateValidation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def num_qubits(self) -> int:
        return 4

    @pytest.fixture
    def validator(self) -> StatePreparationValidator:
        return StatePreparationValidator(tolerance=1e-6)

    @pytest.fixture
    def basis_labels(self, num_qubits: int) -> List[str]:
        return [f"|{format(i, f'0{num_qubits}b')}⟩" for i in range(2**num_qubits)]

    def test_state_preparation(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test state preparation validation."""
        # Generate test state
        dim = 2**num_qubits
        amplitudes = torch.randn(dim, dtype=torch.complex128)
        amplitudes = amplitudes / torch.norm(amplitudes)
        phase = torch.exp(1j * torch.rand(1))
        target = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Test preparation
        prepared_amplitudes = amplitudes + 1e-3 * torch.randn_like(amplitudes)
        prepared_amplitudes = prepared_amplitudes / torch.norm(prepared_amplitudes)
        prepared = QuantumState(amplitudes=prepared_amplitudes, basis_labels=basis_labels, phase=phase)
        result = validator.validate_preparation(target, prepared)
        assert result.is_valid
        assert result.data["metrics"]["fidelity"] > 0.9
        assert result.data["metrics"]["trace_distance"] < 0.1
        assert result.data["metrics"]["purity"] > 0.9

    def test_density_matrix_properties(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test density matrix validation."""
        # Generate single pure state
        dim = 2**num_qubits
        amplitudes = torch.randn(dim, dtype=torch.complex128)
        amplitudes = amplitudes / torch.norm(amplitudes)
        phase = torch.exp(1j * torch.rand(1))
        state = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Get density matrix
        density = state.density_matrix()

        # Test density matrix properties
        result = validator._validate_density_matrix(density)
        assert isinstance(result, DensityMatrixValidation)
        assert result.hermitian
        assert result.positive
        assert result.trace_one
        assert torch.all(result.eigenvalues >= -1e-6)

        # Test mixed state
        def generate_mixed_state() -> torch.Tensor:
            # Create a mixed state by mixing two pure states
            state1 = torch.randn(dim, dtype=torch.complex128)
            state1 = state1 / torch.norm(state1)
            state2 = torch.randn(dim, dtype=torch.complex128)
            state2 = state2 / torch.norm(state2)
            
            # Mix states with probabilities p and (1-p)
            p = 0.7
            rho1 = torch.outer(state1, state1.conj())
            rho2 = torch.outer(state2, state2.conj())
            return p * rho1 + (1-p) * rho2

        mixed_density = generate_mixed_state()
        result = validator._validate_density_matrix(mixed_density)
        assert isinstance(result, DensityMatrixValidation)
        assert result.hermitian
        assert result.positive
        assert result.trace_one
        assert torch.all(result.eigenvalues >= -1e-6)

    def test_state_tomography(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test state tomography validation."""
        # Generate test state
        dim = 2**num_qubits
        amplitudes = torch.randn(dim, dtype=torch.complex128)
        amplitudes = amplitudes / torch.norm(amplitudes)
        phase = torch.exp(1j * torch.rand(1))
        state = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Generate measurement operators
        def generate_projector(basis_state: int) -> torch.Tensor:
            """Generate projection operator."""
            proj = torch.zeros(dim, dim, dtype=torch.complex128)
            proj[basis_state, basis_state] = 1
            return proj

        projectors = [generate_projector(i) for i in range(dim)]

        # Test tomography validation
        result = validator._validate_tomography(state, projectors)
        assert isinstance(result, TomographyValidation)
        assert result.reconstruction_error < 0.1
        assert result.completeness > 0.9
        assert result.confidence > 0.9

    def test_validation_integration(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test integrated state validation."""
        # Generate test state
        dim = 2**num_qubits
        amplitudes = torch.randn(dim, dtype=torch.complex128)
        amplitudes = amplitudes / torch.norm(amplitudes)
        phase = torch.exp(1j * torch.rand(1))
        target = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Test preparation
        prepared_amplitudes = amplitudes + 1e-3 * torch.randn_like(amplitudes)
        prepared_amplitudes = prepared_amplitudes / torch.norm(prepared_amplitudes)
        prepared = QuantumState(amplitudes=prepared_amplitudes, basis_labels=basis_labels, phase=phase)
        prep_result = validator.validate_preparation(target, prepared)
        assert prep_result.is_valid

        # Test density matrix
        density_result = validator._validate_density_matrix(prepared.density_matrix())
        assert density_result.hermitian
        assert density_result.positive
        assert density_result.trace_one

        # Test tomography
        def generate_projector(basis_state: int) -> torch.Tensor:
            """Generate projection operator."""
            proj = torch.zeros(dim, dim, dtype=torch.complex128)
            proj[basis_state, basis_state] = 1
            return proj

        projectors = [generate_projector(i) for i in range(dim)]
        tomo_result = validator._validate_tomography(prepared, projectors)
        assert tomo_result.reconstruction_error < 0.1
        assert tomo_result.completeness > 0.9
        assert tomo_result.confidence > 0.9
