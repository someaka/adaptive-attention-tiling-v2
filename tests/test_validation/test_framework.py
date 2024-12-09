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
import numpy as np
from typing import Dict, List, Optional

from src.validation.framework import (
    ValidationFramework,
    GeometricValidator,
    QuantumValidator,
    PatternValidator,
    ValidationResult,
    ValidationMetrics
)

class TestValidationFramework:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def dim(self) -> int:
        return 64

    @pytest.fixture
    def manifold_dim(self) -> int:
        return 64

    @pytest.fixture
    def state_dim(self) -> int:
        return 64

    @pytest.fixture
    def pattern_dim(self) -> int:
        return 64

    @pytest.fixture
    def framework(self) -> ValidationFramework:
        return ValidationFramework(
            geometric_validator=GeometricValidator(),
            quantum_validator=QuantumValidator(),
            pattern_validator=PatternValidator()
        )

    def test_geometric_validation(
        self, framework: ValidationFramework, batch_size: int, manifold_dim: int
    ):
        """Test geometric validation methods."""
        # Test metric validation
        def test_metric():
            """Test metric validation methods."""
            # Get test metric
            metric = framework.get_test_metric()
            
            # Test positive definiteness
            assert framework.validate_positive_definite(
                metric
            ), "Metric should be positive definite"
            
            # Test smoothness
            assert framework.validate_smoothness(
                metric
            ), "Metric should be smooth"
            
            # Test compatibility
            connection = framework.get_test_connection()
            assert framework.validate_compatibility(
                metric, connection
            ), "Should be compatible"
            
            return metric, connection
            
        metric, connection = test_metric()
        
        # Test curvature validation
        def test_curvature():
            """Test curvature validation methods."""
            # Get test curvature
            curvature = framework.get_test_curvature()
            
            # Test symmetries
            assert framework.validate_curvature_symmetries(
                curvature
            ), "Should have correct symmetries"
            
            # Test Bianchi identities
            assert framework.validate_bianchi_identities(
                curvature
            ), "Should satisfy Bianchi"
            
            # Test sectional bounds
            if framework.has_sectional_bounds():
                bounds = framework.get_sectional_bounds()
                assert framework.validate_sectional_bounds(
                    curvature, bounds
                ), "Should satisfy bounds"
                
            return curvature
            
        curvature = test_curvature()
        
        # Test geodesic validation
        def test_geodesics():
            """Test geodesic validation methods."""
            # Get test geodesic
            geodesic = framework.get_test_geodesic()
            
            # Test geodesic equation
            assert framework.validate_geodesic_equation(
                geodesic
            ), "Should satisfy geodesic equation"
            
            # Test energy conservation
            assert framework.validate_energy_conservation(
                geodesic
            ), "Should conserve energy"
            
            # Test completeness
            if framework.is_complete():
                assert framework.validate_completeness(
                    geodesic
                ), "Should be complete"
                
            return geodesic
            
        geodesic = test_geodesics()

    def test_quantum_validation(
        self, framework: ValidationFramework, batch_size: int, state_dim: int
    ):
        """Test quantum validation methods."""
        # Test state validation
        def test_states():
            """Test quantum state validation."""
            # Get test state
            state = framework.get_test_state()
            
            # Test normalization
            assert framework.validate_normalization(
                state
            ), "State should be normalized"
            
            # Test uncertainty relations
            assert framework.validate_uncertainty_relations(
                state
            ), "Should satisfy uncertainty"
            
            # Test entanglement bounds
            if framework.has_entanglement_bounds():
                bounds = framework.get_entanglement_bounds()
                assert framework.validate_entanglement_bounds(
                    state, bounds
                ), "Should satisfy bounds"
                
            return state
            
        state = test_states()
        
        # Test operator validation
        def test_operators():
            """Test quantum operator validation."""
            # Get test operators
            H = framework.get_test_hamiltonian()
            U = framework.get_test_unitary()
            
            # Test hermiticity
            assert framework.validate_hermiticity(
                H
            ), "Should be Hermitian"
            
            # Test unitarity
            assert framework.validate_unitarity(
                U
            ), "Should be unitary"
            
            # Test group properties
            if framework.has_group_structure():
                assert framework.validate_group_axioms(
                    U
                ), "Should satisfy group axioms"
                
            return H, U
            
        H, U = test_operators()
        
        # Test evolution validation
        def test_evolution():
            """Test quantum evolution validation."""
            # Get test evolution
            evolution = framework.get_test_evolution()
            
            # Test Schrödinger equation
            assert framework.validate_schrodinger(
                evolution
            ), "Should satisfy Schrödinger"
            
            # Test probability conservation
            assert framework.validate_probability_conservation(
                evolution
            ), "Should conserve probability"
            
            # Test reversibility
            if framework.is_reversible():
                assert framework.validate_reversibility(
                    evolution
                ), "Should be reversible"
                
            return evolution
            
        evolution = test_evolution()

    def test_pattern_validation(
        self, framework: ValidationFramework, batch_size: int, pattern_dim: int
    ):
        """Test pattern validation methods."""
        # Test pattern formation
        def test_formation():
            """Test pattern formation validation."""
            # Get test pattern
            pattern = framework.get_test_pattern()
            
            # Test stability
            assert framework.validate_pattern_stability(
                pattern
            ), "Pattern should be stable"
            
            # Test symmetry
            if framework.has_symmetry():
                symmetry = framework.get_pattern_symmetry()
                assert framework.validate_symmetry(
                    pattern, symmetry
                ), "Should respect symmetry"
                
            # Test boundary conditions
            assert framework.validate_boundary_conditions(
                pattern
            ), "Should satisfy boundary conditions"
            
            return pattern
            
        pattern = test_formation()
        
        # Test dynamics validation
        def test_dynamics():
            """Test pattern dynamics validation."""
            # Get test dynamics
            dynamics = framework.get_test_dynamics()
            
            # Test conservation laws
            assert framework.validate_conservation_laws(
                dynamics
            ), "Should conserve quantities"
            
            # Test dissipation inequality
            if framework.is_dissipative():
                assert framework.validate_dissipation(
                    dynamics
                ), "Should satisfy dissipation"
                
            # Test attractors
            if framework.has_attractors():
                attractors = framework.get_attractors()
                assert framework.validate_attractor_properties(
                    dynamics, attractors
                ), "Should have correct attractors"
                
            return dynamics
            
        dynamics = test_dynamics()
        
        # Test bifurcation validation
        def test_bifurcations():
            """Test bifurcation validation."""
            # Get test bifurcation
            bifurcation = framework.get_test_bifurcation()
            
            # Test normal form
            assert framework.validate_normal_form(
                bifurcation
            ), "Should have correct normal form"
            
            # Test stability exchange
            assert framework.validate_stability_exchange(
                bifurcation
            ), "Should exchange stability"
            
            # Test universality
            if framework.is_universal():
                assert framework.validate_universality(
                    bifurcation
                ), "Should be universal"
                
            return bifurcation
            
        bifurcation = test_bifurcations()

    def test_quantum_validation(self, framework: ValidationFramework,
                              batch_size: int, dim: int):
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
            initial_state=state,
            final_state=evolution,
            hamiltonian=hamiltonian
        )
        assert isinstance(evol_result, ValidationResult)
        assert "unitarity" in evol_result.metrics
        assert "energy_conservation" in evol_result.metrics
        
        # Test measurement validation
        observables = [torch.randn(dim, dim, dtype=torch.complex64) for _ in range(3)]
        for obs in observables:
            obs += obs.conj().T  # Make Hermitian
        meas_result = framework.validate_quantum_measurement(
            state=state,
            observables=observables
        )
        assert isinstance(meas_result, ValidationResult)
        assert "expectation_bounds" in meas_result.metrics
        assert "uncertainty_relations" in meas_result.metrics

    def test_pattern_validation(self, framework: ValidationFramework,
                              batch_size: int, dim: int):
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
            pattern=pattern,
            perturbation=perturbation
        )
        assert isinstance(stab_result, ValidationResult)
        assert "linear_stability" in stab_result.metrics
        assert "nonlinear_stability" in stab_result.metrics

    def test_integrated_validation(self, framework: ValidationFramework,
                                 batch_size: int, dim: int):
        """Test integrated validation workflow."""
        # Create test configuration
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        pattern = torch.randn(batch_size, dim)
        
        # Run integrated validation
        result = framework.validate_all(
            metric=metric,
            quantum_state=state,
            pattern=pattern
        )
        
        # Test result structure
        assert isinstance(result, Dict)
        assert "geometric" in result
        assert "quantum" in result
        assert "pattern" in result
        
        # Test metric aggregation
        metrics = framework.aggregate_metrics(result)
        assert isinstance(metrics, ValidationMetrics)
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
                pattern=torch.ones(2, 5)
            )

    def test_validation_metrics(self):
        """Test validation metrics computation and aggregation."""
        # Create test metrics
        metrics = ValidationMetrics(
            overall_score=0.85,
            component_scores={
                "geometric": 0.9,
                "quantum": 0.8,
                "pattern": 0.85
            },
            detailed_metrics={
                "geometric": {
                    "positive_definite": True,
                    "symmetry": True,
                    "curvature_bounds": 0.95
                },
                "quantum": {
                    "normalization": True,
                    "unitarity": 0.85,
                    "energy_conservation": 0.75
                },
                "pattern": {
                    "spatial_coherence": 0.8,
                    "temporal_stability": 0.9,
                    "symmetry": 0.85
                }
            }
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
        other_metrics = ValidationMetrics(
            overall_score=0.75,
            component_scores={
                "geometric": 0.8,
                "quantum": 0.7,
                "pattern": 0.75
            },
            detailed_metrics={}
        )
        assert metrics > other_metrics
        assert metrics >= other_metrics
        assert not metrics < other_metrics
        
        # Test metric aggregation
        combined = ValidationMetrics.aggregate([metrics, other_metrics])
        assert isinstance(combined, ValidationMetrics)
        assert combined.overall_score == pytest.approx(0.8, rel=1e-2)
        assert all(name in combined.component_scores 
                  for name in ["geometric", "quantum", "pattern"])
