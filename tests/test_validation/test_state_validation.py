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
        """Test quantum state preparation validation."""
        # Generate target state
        dim = 2**num_qubits
        amplitudes = torch.randn(batch_size, dim, dtype=torch.complex64)
        amplitudes = amplitudes / torch.norm(amplitudes, dim=1, keepdim=True)
        phase = torch.exp(1j * torch.rand(batch_size, 1))
        target = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Generate prepared state close to target
        prepared_amplitudes = amplitudes + 1e-3 * torch.randn_like(amplitudes)
        prepared_amplitudes = prepared_amplitudes / torch.norm(prepared_amplitudes, dim=1, keepdim=True)
        prepared = QuantumState(amplitudes=prepared_amplitudes, basis_labels=basis_labels, phase=phase)

        # Test preparation validation
        result = validator.validate_preparation(target, prepared)
        assert isinstance(result, QuantumStateValidationResult)
        assert result.is_valid
        assert result.data is not None
        assert "fidelity" in result.data
        assert "trace_distance" in result.data
        assert "purity" in result.data

        # Test with poorly prepared state
        bad_amplitudes = torch.randn_like(amplitudes)
        bad_amplitudes = bad_amplitudes / torch.norm(bad_amplitudes, dim=1, keepdim=True)
        bad_state = QuantumState(amplitudes=bad_amplitudes, basis_labels=basis_labels, phase=phase)
        result = validator.validate_preparation(target, bad_state)
        assert not result.is_valid

    def test_density_matrix_properties(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test density matrix validation."""
        # Generate pure state
        dim = 2**num_qubits
        amplitudes = torch.randn(batch_size, dim, dtype=torch.complex64)
        amplitudes = amplitudes / torch.norm(amplitudes, dim=1, keepdim=True)
        phase = torch.exp(1j * torch.rand(batch_size, 1))
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
            """Generate mixed state density matrix."""
            pure_states = [torch.randn(dim, dtype=torch.complex64) for _ in range(3)]
            pure_states = [s / torch.norm(s) for s in pure_states]
            weights = torch.softmax(torch.rand(3), dim=0)

            return torch.sum(torch.stack([
                w * (s.unsqueeze(-1) @ s.conj().unsqueeze(-2))
                for w, s in zip(weights, pure_states)
            ]), dim=0)

        mixed = generate_mixed_state()
        result = validator._validate_density_matrix(mixed.unsqueeze(0))
        assert result.hermitian
        assert result.positive
        assert result.trace_one

    def test_state_tomography(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test state tomography validation."""
        # Generate test state
        dim = 2**num_qubits
        amplitudes = torch.randn(batch_size, dim, dtype=torch.complex64)
        amplitudes = amplitudes / torch.norm(amplitudes, dim=1, keepdim=True)
        phase = torch.exp(1j * torch.rand(batch_size, 1))
        state = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Generate measurement operators
        def generate_projector(basis_state: int) -> torch.Tensor:
            """Generate projection operator."""
            proj = torch.zeros(dim, dim, dtype=torch.complex64)
            proj[basis_state, basis_state] = 1
            return proj

        projectors = [generate_projector(i) for i in range(dim)]

        # Test tomography validation
        result = validator._validate_tomography(state, projectors)
        assert isinstance(result, TomographyValidation)
        assert result.reconstruction_error < 0.1
        assert result.completeness > 0.9
        assert result.confidence > 0.9
        assert isinstance(result.estimated_state, torch.Tensor)

    def test_validation_integration(
        self, validator: StatePreparationValidator, batch_size: int, num_qubits: int, basis_labels: List[str]
    ):
        """Test integrated state validation."""
        # Generate test state
        dim = 2**num_qubits
        amplitudes = torch.randn(batch_size, dim, dtype=torch.complex64)
        amplitudes = amplitudes / torch.norm(amplitudes, dim=1, keepdim=True)
        phase = torch.exp(1j * torch.rand(batch_size, 1))
        target = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)

        # Test preparation
        prepared_amplitudes = amplitudes + 1e-3 * torch.randn_like(amplitudes)
        prepared_amplitudes = prepared_amplitudes / torch.norm(prepared_amplitudes, dim=1, keepdim=True)
        prepared = QuantumState(amplitudes=prepared_amplitudes, basis_labels=basis_labels, phase=phase)
        prep_result = validator.validate_preparation(target, prepared)
        assert prep_result.is_valid

        # Test density matrix
        density = prepared.density_matrix()
        density_result = validator._validate_density_matrix(density)
        assert density_result.hermitian
        assert density_result.positive

        # Test tomography
        projectors = [
            torch.eye(dim, dtype=torch.complex64)[i:i+1].expand(dim, -1)
            for i in range(dim)
        ]
        tomo_result = validator._validate_tomography(prepared, projectors)
        assert tomo_result.reconstruction_error < 0.1
