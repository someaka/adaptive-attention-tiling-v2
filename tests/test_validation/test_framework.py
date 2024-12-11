"""
Unit tests for the validation framework.

Tests cover:
1. Geometric validation
2. Quantum validation
3. Pattern validation
4. Integration tests
"""

import pytest
import torch

from src.validation.framework import (
    ModelGeometricValidator,
    PatternValidator,
    QuantumValidator,
    ValidationFramework,
    ValidationResult,
)


class TestValidationFramework:
    @pytest.fixture
    def batch_size(self) -> int:
        return 4

    @pytest.fixture
    def dim(self) -> int:
        return 8

    @pytest.fixture
    def manifold_dim(self) -> int:
        return 8

    @pytest.fixture
    def state_dim(self) -> int:
        return 8

    @pytest.fixture
    def pattern_dim(self) -> int:
        return 8

    @pytest.fixture
    def framework(self, manifold_dim) -> ValidationFramework:
        return ValidationFramework(
            geometric_validator=ModelGeometricValidator(manifold_dim=manifold_dim),
            quantum_validator=QuantumValidator(),
            pattern_validator=PatternValidator(),
        )

    @pytest.mark.level0
    def test_geometric_validation(
        self, framework: ValidationFramework, batch_size: int, manifold_dim: int
    ):
        """Test geometric validation methods."""
        metric = framework.get_test_metric(batch_size)
        assert metric.shape == (batch_size, manifold_dim, manifold_dim)
        assert framework.validate_positive_definite(metric)
        
        connection = framework.get_test_connection(batch_size)
        assert connection.shape == (batch_size, manifold_dim, manifold_dim, manifold_dim)
        assert framework.validate_compatibility(metric, connection)
        
        curvature = framework.get_test_curvature(batch_size)
        assert curvature.shape == (batch_size, manifold_dim, manifold_dim, manifold_dim, manifold_dim)
        assert framework.validate_curvature_symmetries(curvature)
        assert framework.validate_bianchi_identities(curvature)

    @pytest.mark.level0
    def test_quantum_validation(
        self, framework: ValidationFramework, batch_size: int, state_dim: int
    ):
        """Test quantum validation methods."""
        state = torch.randn(batch_size, state_dim, dtype=torch.complex64)
        state = state / state.norm(dim=1, keepdim=True)
        
        metrics = framework.validate_quantum_state(state)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    @pytest.mark.level0
    def test_pattern_validation(
        self, framework: ValidationFramework, batch_size: int, pattern_dim: int
    ):
        """Test pattern validation methods."""
        pattern = torch.randn(batch_size, pattern_dim)
        
        metrics = framework.validate_pattern_formation(pattern)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    @pytest.mark.level1
    def test_integrated_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test integrated validation workflow."""
        # Create test data
        data = torch.randn(batch_size, dim)
        param_range = torch.linspace(-1, 1, 10)
        
        # Run validation
        result = framework.validate_all(None, data, param_range)
        
        # Check results
        assert isinstance(result, ValidationResult)
        assert result.framework_accuracy >= 0.0
        assert result.framework_consistency >= 0.0

    @pytest.mark.level0
    def test_error_handling(
        self, framework: ValidationFramework, batch_size: int, manifold_dim: int
    ):
        """Test error handling in validation framework."""
        # Test invalid metric
        invalid_metric = torch.zeros(1, 1)
        assert not framework.validate_metric(invalid_metric)
        
        # Test incompatible shapes
        metric = framework.get_test_metric(batch_size)
        invalid_connection = torch.zeros(1, 1, 1)
        assert not framework.validate_compatibility(metric, invalid_connection)

    @pytest.mark.level0
    def test_validation_metrics(self):
        """Test validation metrics computation and aggregation."""
        metrics = ValidationResult(
            curvature_bounds=(-1.0, 1.0),
            energy_metrics={"total": 0.5},
            bifurcation_points=[torch.tensor([1.0])],
            stability_eigenvalues=torch.tensor([0.1]),
            framework_accuracy=0.95,
            framework_consistency=0.90
        )
        assert metrics.framework_accuracy == 0.95
        assert metrics.framework_consistency == 0.90

    def test_quantum_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test quantum validation components."""
        # Create test quantum state
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Test state validation
        state_result = framework.validate_quantum_state(state)
        assert isinstance(state_result, ValidationResult)
        assert state_result.is_valid
        assert "normalization" in state_result.metrics
        assert "coherence" in state_result.metrics

        # Test evolution validation
        hamiltonian = torch.randn(dim, dim, dtype=torch.complex64)
        hamiltonian = hamiltonian + hamiltonian.conj().T  # Make Hermitian
        evolution = framework.evolve_quantum_state(state, hamiltonian)
        evol_result = framework.validate_quantum_evolution(
            initial_state=state, final_state=evolution, hamiltonian=hamiltonian
        )
        assert isinstance(evol_result, ValidationResult)
        assert "unitarity" in evol_result.metrics
        assert "energy_conservation" in evol_result.metrics

        # Test measurement validation
        observables = [torch.randn(dim, dim, dtype=torch.complex64) for _ in range(3)]
        for obs in observables:
            obs += obs.conj().T  # Make Hermitian
        meas_result = framework.validate_quantum_measurement(
            state=state, observables=observables
        )
        assert isinstance(meas_result, ValidationResult)
        assert "expectation_bounds" in meas_result.metrics
        assert "uncertainty_relations" in meas_result.metrics

    def test_pattern_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test pattern validation components."""
        # Create test pattern configuration
        pattern = torch.randn(batch_size, dim)

        # Test pattern formation validation
        form_result = framework.validate_pattern_formation(pattern)
        assert isinstance(form_result, ValidationResult)
        assert "spatial_coherence" in form_result.metrics
        assert "temporal_stability" in form_result.metrics

        # Test symmetry validation
        symm_result = framework.validate_pattern_symmetry(pattern)
        assert isinstance(symm_result, ValidationResult)
        assert "translation_invariance" in symm_result.metrics
        assert "rotation_invariance" in symm_result.metrics

        # Test stability validation
        perturbation = 0.01 * torch.randn_like(pattern)
        stab_result = framework.validate_pattern_stability(
            pattern=pattern, perturbation=perturbation
        )
        assert isinstance(stab_result, ValidationResult)
        assert "linear_stability" in stab_result.metrics
        assert "nonlinear_stability" in stab_result.metrics

    def test_integrated_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test integrated validation workflow."""
        # Create test configuration
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        pattern = torch.randn(batch_size, dim)

        # Run integrated validation
        result = framework.validate_all(
            metric=metric, quantum_state=state, pattern=pattern
        )

        # Test result structure
        assert isinstance(result, dict)
        assert "geometric" in result
        assert "quantum" in result
        assert "pattern" in result

        # Test metric aggregation
        metrics = framework.aggregate_metrics(result)
        assert isinstance(metrics, ValidationResult)
        assert metrics.overall_score > 0
        assert len(metrics.component_scores) == 3
        assert all(0 <= score <= 1 for score in metrics.component_scores.values())

    def test_error_handling(self, framework: ValidationFramework):
        """Test error handling in validation framework."""
        # Test invalid metric
        with pytest.raises(ValueError):
            framework.validate_metric(torch.ones(1))

        # Test invalid quantum state
        with pytest.raises(ValueError):
            framework.validate_quantum_state(torch.ones(1))

        # Test invalid pattern
        with pytest.raises(ValueError):
            framework.validate_pattern_formation(torch.ones(1))

        # Test incompatible dimensions
        with pytest.raises(ValueError):
            framework.validate_all(
                metric=torch.ones(2, 3, 3),
                quantum_state=torch.ones(2, 4),
                pattern=torch.ones(2, 5),
            )

    def test_validation_metrics(self):
        """Test validation metrics computation and aggregation."""
        # Create test metrics
        metrics = ValidationResult(
            curvature_bounds=(-1.0, 1.0),
            energy_metrics={"total": 0.5},
            bifurcation_points=[torch.tensor([1.0])],
            stability_eigenvalues=torch.tensor([0.1]),
            framework_accuracy=0.85,
            framework_consistency=0.90
        )

        # Test metric properties
        assert metrics.is_valid
        assert len(metrics.component_scores) == 3
        assert all(0 <= score <= 1 for score in metrics.component_scores.values())

        # Test metric serialization
        serialized = metrics.to_dict()
        assert isinstance(serialized, dict)
        assert "overall_score" in serialized
        assert "component_scores" in serialized
        assert "detailed_metrics" in serialized

        # Test metric comparison
        other_metrics = ValidationResult(
            curvature_bounds=(-1.0, 1.0),
            energy_metrics={"total": 0.5},
            bifurcation_points=[torch.tensor([1.0])],
            stability_eigenvalues=torch.tensor([0.1]),
            framework_accuracy=0.75,
            framework_consistency=0.80
        )
        assert metrics > other_metrics
        assert metrics >= other_metrics
        assert not metrics < other_metrics

        # Test metric aggregation
        combined = ValidationResult.aggregate([metrics, other_metrics])
        assert isinstance(combined, ValidationResult)
        assert combined.overall_score == pytest.approx(0.8, rel=1e-2)
        assert all(
            name in combined.component_scores
            for name in ["geometric", "quantum", "pattern"]
        )
