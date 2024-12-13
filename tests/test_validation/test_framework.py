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
    ValidationFramework,
    ValidationResult
)
from src.validation.geometric.flow import (
    GeometricFlowValidator,
    FlowStabilityValidator,
    EnergyValidator,
    ConvergenceValidator
)
from src.validation.patterns.stability import (
    PatternStabilityValidator,
    LinearStabilityAnalyzer,
    NonlinearStabilityAnalyzer,
    LyapunovAnalyzer,
    BifurcationValidator
)
from src.validation.quantum.validator import QuantumValidator
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.flow.geometric_flow import GeometricFlow
from src.neural.flow.hamiltonian import HamiltonianSystem


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
    def pattern_dynamics(self) -> PatternDynamics:
        return PatternDynamics(grid_size=32, space_dim=2)

    @pytest.fixture
    def geometric_flow(self) -> GeometricFlow:
        return GeometricFlow()

    @pytest.fixture
    def hamiltonian(self) -> HamiltonianSystem:
        return HamiltonianSystem()

    @pytest.fixture
    def framework(self, pattern_dynamics: PatternDynamics, geometric_flow: GeometricFlow) -> ValidationFramework:
        return ValidationFramework(
            geometric_validator=GeometricFlowValidator(),
            quantum_validator=QuantumValidator(),
            pattern_validator=PatternStabilityValidator(pattern_dynamics)
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
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test quantum validation methods."""
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / state.norm(dim=1, keepdim=True)
        
        result = framework.validate_quantum_state(state)
        assert isinstance(result, ValidationResult)
        assert "quantum" in result.metrics
        assert "metrics" in result.metrics["quantum"]
        assert all(k in result.metrics["quantum"]["metrics"] for k in 
                  ["normalization", "unitarity", "energy_conservation"])
        
        # Test quantum metrics structure
        quantum_metrics = result.metrics["quantum"]
        assert "entanglement" in quantum_metrics
        assert all(k in quantum_metrics["entanglement"] for k in 
                  ["entanglement_entropy", "mutual_information", "relative_entropy"])
        assert "coherence" in quantum_metrics
        assert all(k in quantum_metrics["coherence"] for k in 
                  ["coherence_length", "coherence_time", "decoherence_rate"])

    @pytest.mark.level0
    def test_pattern_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test pattern validation methods."""
        pattern = torch.randn(batch_size, dim)
        
        result = framework.validate_pattern_formation(pattern)
        assert isinstance(result, ValidationResult)
        assert "pattern" in result.metrics
        pattern_metrics = result.metrics["pattern"]
        assert all(k in pattern_metrics for k in [
            "spatial_coherence", "temporal_stability", "translation_invariance",
            "rotation_invariance", "linear_stability", "nonlinear_stability",
            "bifurcation_points", "stability_eigenvalues", "symmetry"
        ])

    @pytest.mark.level1
    def test_integrated_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test integrated validation workflow."""
        # Create test data
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)  # Make symmetric positive definite
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Run validation
        result = framework.validate_all(
            model=None,
            data=state,
            metric=metric
        )
        
        # Test result structure
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'metrics')
        assert all(k in result.metrics for k in ["geometric", "quantum", "pattern"])
        
        # Test metric aggregation
        assert result.framework_accuracy >= 0 and result.framework_accuracy <= 1
        assert result.framework_consistency >= 0 and result.framework_consistency <= 1
        assert all(0 <= score <= 1 for score in result.component_scores.values())

    def test_error_handling(self, framework: ValidationFramework):
        """Test error handling in validation framework."""
        # Test invalid inputs
        with pytest.raises(ValueError, match="Invalid metric shape"):
            framework.validate_metric(torch.ones(1))
        
        # Test invalid quantum state
        with pytest.raises(ValueError, match="Invalid quantum state shape"):
            framework.validate_quantum_state(torch.ones(1))
        
        # Test invalid pattern
        with pytest.raises(ValueError, match="Invalid pattern shape"):
            framework.validate_pattern_formation(torch.ones(1))
        
        # Test incompatible dimensions
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            framework.validate_all(
                model=None,
                data=torch.ones(2, 4),
                metric=torch.ones(3, 3)
            )
            
        # Test invalid parameter types
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            framework.validate_metric([1, 2, 3])
            
        # Test NaN/Inf handling
        with pytest.raises(ValueError, match="Contains NaN or Inf values"):
            framework.validate_metric(torch.tensor([[float('nan'), 0], [0, 1]]))
            
        # Test empty tensors
        with pytest.raises(ValueError, match="Empty tensor"):
            framework.validate_metric(torch.tensor([]))

    def test_validation_metrics(self):
        """Test validation metrics computation and aggregation."""
        metrics = ValidationResult(
            curvature_bounds=(-1.0, 1.0),
            energy_metrics={"total": 0.5},
            bifurcation_points=[torch.tensor([1.0])],
            stability_eigenvalues=torch.tensor([0.1]),
            framework_accuracy=0.95,
            framework_consistency=0.90,
            metrics={
                "geometric": {"curvature": True, "energy": True},
                "quantum": {
                    "metrics": {
                        "normalization": True,
                        "unitarity": True,
                        "energy_conservation": True
                    },
                    "entanglement": {
                        "entanglement_entropy": 0.5,
                        "mutual_information": 0.3,
                        "relative_entropy": 0.2
                    },
                    "coherence": {
                        "coherence_length": 0.8,
                        "coherence_time": 0.7,
                        "decoherence_rate": 0.1
                    }
                },
                "pattern": {
                    "spatial_coherence": True,
                    "temporal_stability": True,
                    "translation_invariance": True,
                    "rotation_invariance": True,
                    "linear_stability": True,
                    "nonlinear_stability": True,
                    "bifurcation_points": [torch.tensor([1.0])],
                    "stability_eigenvalues": torch.tensor([0.1]),
                    "symmetry": True
                }
            },
            component_scores={
                "geometric": 0.95,
                "quantum": 0.90,
                "pattern": 0.85
            }
        )
        
        # Test metric structure
        assert metrics.framework_accuracy == 0.95
        assert metrics.framework_consistency == 0.90
        assert len(metrics.component_scores) == 3
        assert all(0 <= score <= 1 for score in metrics.component_scores.values())
        
        # Test quantum metrics
        quantum = metrics.metrics["quantum"]
        assert all(k in quantum["metrics"] for k in ["normalization", "unitarity", "energy_conservation"])
        assert all(k in quantum["entanglement"] for k in ["entanglement_entropy", "mutual_information", "relative_entropy"])
        assert all(k in quantum["coherence"] for k in ["coherence_length", "coherence_time", "decoherence_rate"])
        
        # Test pattern metrics
        pattern = metrics.metrics["pattern"]
        assert all(k in pattern for k in [
            "spatial_coherence", "temporal_stability", "translation_invariance",
            "rotation_invariance", "linear_stability", "nonlinear_stability",
            "bifurcation_points", "stability_eigenvalues", "symmetry"
        ])

    def test_full_integration(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test full integration of all validation components."""
        # Create test data
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)  # Make symmetric positive definite
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Run validation
        result = framework.validate_all(
            model=None,
            data=state,
            metric=metric
        )
        
        # Test result structure
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'metrics')
        assert all(k in result.metrics for k in ["geometric", "quantum", "pattern"])
        
        # Test component scores
        assert all(k in result.component_scores for k in ["geometric", "quantum", "pattern"])
        assert all(0 <= score <= 1 for score in result.component_scores.values())
        
        # Test framework metrics
        assert 0 <= result.framework_accuracy <= 1
        assert 0 <= result.framework_consistency <= 1
