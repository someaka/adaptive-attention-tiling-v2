"""
Unit tests for quantum state validation.

Tests cover:
1. Normalization properties
2. Uncertainty relations
3. Entanglement measures
4. State evolution
"""

from typing import Dict

import numpy as np
import pytest
import torch

from src.validation.quantum.state import (
    EntanglementMetrics,
    StateProperties,
    StateValidator,
    UncertaintyMetrics,
    ValidationResult,
)


class TestStateValidation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def num_qubits(self) -> int:
        return 4

    @pytest.fixture
    def validator(self) -> StateValidator:
        return StateValidator(
            normalization_threshold=1e-6,
            uncertainty_threshold=0.5,
            entanglement_threshold=0.1,
        )

    def test_normalization(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test quantum state normalization."""
        # Generate normalized state
        dim = 2**num_qubits
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Test normalization validation
        result = validator.validate_normalization(state)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "norm_deviation" in result.metrics

        # Test unnormalized state
        unnorm_state = 2 * state
        result = validator.validate_normalization(unnorm_state)
        assert not result.is_valid

        # Test state properties
        properties = validator.compute_state_properties(state)
        assert isinstance(properties, StateProperties)
        assert hasattr(properties, "norm")
        assert hasattr(properties, "phase")

        # Test global phase invariance
        phase = torch.exp(1j * torch.rand(batch_size, 1))
        phase_state = phase * state
        result = validator.validate_normalization(phase_state)
        assert result.is_valid

    def test_uncertainty_relations(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test uncertainty relation validation."""
        # Generate test state and observables
        dim = 2**num_qubits
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Generate Pauli observables
        def generate_pauli(type: str) -> torch.Tensor:
            """Generate Pauli matrix."""
            if type == "X":
                return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
            if type == "Y":
                return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
            # Z
            return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        observables = [generate_pauli(type) for type in ["X", "Y", "Z"]]

        # Test uncertainty relations
        result = validator.validate_uncertainty_relations(state, observables)
        assert isinstance(result, ValidationResult)
        assert "uncertainty_product" in result.metrics
        assert "minimum_uncertainty" in result.metrics

        # Test uncertainty metrics
        metrics = validator.compute_uncertainty_metrics(state, observables)
        assert isinstance(metrics, UncertaintyMetrics)
        assert hasattr(metrics, "expectation_values")
        assert hasattr(metrics, "variances")

        # Test Robertson uncertainty relation
        robertson = validator.validate_robertson_relation(
            state, observables[0], observables[1]
        )
        assert isinstance(robertson, bool)

    def test_entanglement_measures(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test entanglement measure validation."""

        # Generate maximally entangled state (Bell state)
        def generate_bell_state() -> torch.Tensor:
            """Generate Bell state."""
            return torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)

        bell_state = generate_bell_state().unsqueeze(0).repeat(batch_size, 1)

        # Test entanglement validation
        result = validator.validate_entanglement(bell_state)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "entanglement_measure" in result.metrics

        # Test separable state
        separable = torch.tensor([1, 0, 0, 0], dtype=torch.complex64)
        separable = separable.unsqueeze(0).repeat(batch_size, 1)
        result = validator.validate_entanglement(separable)
        assert not result.is_valid

        # Test entanglement metrics
        metrics = validator.compute_entanglement_metrics(bell_state)
        assert isinstance(metrics, EntanglementMetrics)
        assert hasattr(metrics, "von_neumann_entropy")
        assert hasattr(metrics, "concurrence")

        # Test partial trace
        reduced = validator.compute_reduced_density_matrix(bell_state)
        assert reduced.shape[-2:] == (2, 2)

    def test_state_evolution(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test quantum state evolution validation."""
        # Generate initial state
        dim = 2**num_qubits
        initial_state = torch.randn(batch_size, dim, dtype=torch.complex64)
        initial_state = initial_state / torch.norm(initial_state, dim=1, keepdim=True)

        # Generate unitary evolution
        def generate_unitary(dim: int) -> torch.Tensor:
            """Generate random unitary matrix."""
            matrix = torch.randn(dim, dim, dtype=torch.complex64)
            q, _ = torch.linalg.qr(matrix)
            return q

        unitary = generate_unitary(dim)
        final_state = initial_state @ unitary.T

        # Test evolution validation
        result = validator.validate_evolution(initial_state, final_state, unitary)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "unitarity_measure" in result.metrics

        # Test non-unitary evolution
        non_unitary = torch.randn(dim, dim, dtype=torch.complex64)
        final_state_nu = initial_state @ non_unitary.T
        result = validator.validate_evolution(
            initial_state, final_state_nu, non_unitary
        )
        assert not result.is_valid

    def test_density_matrix(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test density matrix properties."""
        # Generate pure state
        dim = 2**num_qubits
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Compute density matrix
        density = validator.compute_density_matrix(state)

        # Test density matrix properties
        result = validator.validate_density_matrix(density)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "trace" in result.metrics
        assert "purity" in result.metrics

        # Test mixed state
        def generate_mixed_state() -> torch.Tensor:
            """Generate mixed state density matrix."""
            pure_states = [torch.randn(dim, dtype=torch.complex64) for _ in range(3)]
            pure_states = [s / torch.norm(s) for s in pure_states]
            weights = torch.softmax(torch.rand(3), dim=0)

            return sum(
                w * (s.unsqueeze(-1) @ s.conj().unsqueeze(-2))
                for w, s in zip(weights, pure_states)
            )

        mixed = generate_mixed_state()
        result = validator.validate_density_matrix(mixed.unsqueeze(0))
        assert result.is_valid
        assert result.metrics["purity"] < 1.0

    def test_measurement_statistics(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test measurement statistics validation."""
        # Generate test state
        dim = 2**num_qubits
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Generate measurement operators
        def generate_projector(basis_state: int) -> torch.Tensor:
            """Generate projection operator."""
            proj = torch.zeros(dim, dim, dtype=torch.complex64)
            proj[basis_state, basis_state] = 1
            return proj

        projectors = [generate_projector(i) for i in range(dim)]

        # Test measurement statistics
        result = validator.validate_measurement_statistics(state, projectors)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "probability_sum" in result.metrics
        assert "probability_positivity" in result.metrics

        # Test Born rule
        probs = validator.compute_measurement_probabilities(state, projectors)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), rtol=1e-5)
        assert torch.all(probs >= 0)

    def test_validation_integration(
        self, validator: StateValidator, batch_size: int, num_qubits: int
    ):
        """Test integrated state validation."""
        # Generate test state
        dim = 2**num_qubits
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Run full validation
        result = validator.validate_state(state)
        assert isinstance(result, Dict)
        assert "normalization" in result
        assert "uncertainty" in result
        assert "entanglement" in result
        assert "density_matrix" in result

        # Check validation scores
        assert all(0 <= score <= 1 for score in result.values())
        assert "overall_score" in result

        # Test validation with parameters
        params = torch.linspace(0, 1, 10)
        param_result = validator.validate_state_family(state, params)
        assert "parameter_dependence" in param_result

        # Test validation summary
        summary = validator.get_validation_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0
