"""
Unit tests for the validation framework.

Tests cover:
1. Geometric validation
2. Quantum validation
3. Pattern validation
4. Integration tests
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from typing import Dict, List, Optional, Sequence, Union, cast

from src.validation.framework import ValidationFramework, ConcreteValidationResult, FrameworkValidationResult
from src.validation.geometric.model import ModelGeometricValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.patterns.stability import PatternValidator as StabilityValidator
from src.validation.flow.stability import LinearStabilityValidator, NonlinearStabilityValidator
from src.validation.base import ValidationResult
from src.core.models.base import LayerGeometry, ModelGeometry
from src.neural.attention.pattern.dynamics import PatternDynamics as AttentionPatternDynamics
from src.core.tiling.geometric_flow import GeometricFlow
from src.core.patterns.riemannian import RiemannianFramework, PatternRiemannianStructure
from src.core.quantum.types import QuantumState
from src.core.patterns.dynamics import PatternDynamics
from src.core.patterns import (
    BaseRiemannianStructure,
    RiemannianFramework,
    PatternRiemannianStructure,
    MetricTensor,
    ChristoffelSymbols,
    CurvatureTensor,
)


class TestValidationFramework:
    @pytest.fixture
    def batch_size(self) -> int:
        """Batch size for testing."""
        return 16

    @pytest.fixture
    def dim(self) -> int:
        """Dimension for testing."""
        return 8

    @pytest.fixture
    def manifold_dim(self) -> int:
        """Manifold dimension for testing."""
        return 16

    @pytest.fixture
    def state_dim(self) -> int:
        """State dimension for testing."""
        return 8

    @pytest.fixture
    def pattern_dim(self) -> int:
        """Pattern dimension for testing."""
        return 8

    @pytest.fixture
    def mock_layer(self, manifold_dim: int) -> LayerGeometry:
        """Create mock layer geometry."""
        return LayerGeometry(manifold_dim=manifold_dim, pattern_dim=manifold_dim)

    @pytest.fixture
    def model_geometry(self, manifold_dim: int, mock_layer: LayerGeometry) -> ModelGeometry:
        """Create model geometry."""
        return ModelGeometry(
            manifold_dim=manifold_dim,
            query_dim=manifold_dim,
            key_dim=manifold_dim,
            layers={
                'input': mock_layer,
                'hidden': mock_layer,
                'output': mock_layer
            },
            attention_heads=[]
        )

    @pytest.fixture
    def geometric_validator(self, model_geometry: ModelGeometry) -> ModelGeometricValidator:
        """Create geometric validator."""
        return ModelGeometricValidator(
            model_geometry=model_geometry,
            tolerance=1e-6,
            curvature_bounds=(-1.0, 1.0)
        )

    @pytest.fixture
    def linear_stability_validator(self) -> LinearStabilityValidator:
        """Create linear stability validator."""
        return LinearStabilityValidator(
            tolerance=1e-6,
            stability_threshold=0.0
        )

    @pytest.fixture
    def nonlinear_stability_validator(self) -> NonlinearStabilityValidator:
        """Create nonlinear stability validator."""
        return NonlinearStabilityValidator(
            tolerance=1e-6,
            basin_samples=100
        )

    @pytest.fixture
    def quantum_validator(self) -> QuantumStateValidator:
        """Create quantum validator."""
        return QuantumStateValidator()

    @pytest.fixture
    def riemannian_framework(self, manifold_dim: int) -> RiemannianFramework:
        """Create Riemannian framework for testing."""
        return PatternRiemannianStructure(
            manifold_dim=manifold_dim,
            pattern_dim=manifold_dim
        )

    @pytest.fixture
    def pattern_validator(self, model_geometry: ModelGeometry) -> StabilityValidator:
        """Create pattern validator."""
        return StabilityValidator(
            linear_validator=LinearStabilityValidator(stability_threshold=0.0),
            nonlinear_validator=NonlinearStabilityValidator(),
            lyapunov_threshold=1e-6,
            perturbation_threshold=1e-6
        )

    @pytest.fixture
    def validation_framework(
        self,
        geometric_validator: ModelGeometricValidator,
        quantum_validator: QuantumStateValidator,
        pattern_validator: StabilityValidator
    ) -> ValidationFramework:
        """Create validation framework."""
        return ValidationFramework(
            geometric_validator=geometric_validator,
            quantum_validator=quantum_validator,
            pattern_validator=pattern_validator
        )

    @pytest.mark.level0
    def test_geometric_validation(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        batch_size: int,
        manifold_dim: int
    ) -> None:
        """Test geometric validation."""
        # Create test data
        points = torch.randn(batch_size, manifold_dim)
        metric = riemannian_framework.compute_metric(points)

        # Run validation
        result = validation_framework.validate_geometry(
            model=None,
            data=points,
            riemannian=riemannian_framework
        )

        # Check result structure
        assert result.is_valid
        assert result.data is not None, "Validation result data should not be None"
        assert "geometric" in result.data, "Validation result should contain geometric data"
        assert result.data["geometric"]["complete"]

        # Check metric properties
        assert "metric_tensor" in result.data["geometric"]
        metric_tensor = cast(Tensor, result.data["geometric"]["metric_tensor"])
        assert metric_tensor.shape == (batch_size, manifold_dim, manifold_dim)
        assert torch.allclose(metric_tensor, metric_tensor.transpose(-1, -2))

        # Check curvature properties
        assert "curvature" in result.data["geometric"]
        assert "sectional_curvature" in result.data["geometric"]["curvature"]
        assert "ricci_curvature" in result.data["geometric"]["curvature"]

    @pytest.mark.level0
    def test_quantum_validation(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        batch_size: int,
        manifold_dim: int
    ) -> None:
        """Test quantum validation."""
        # Create test quantum state
        state = torch.randn(batch_size, manifold_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Create test measurements and bases
        measurements: List[Union[Tensor, QuantumState]] = [
            torch.randn(batch_size, manifold_dim, dtype=torch.complex64) / torch.sqrt(torch.tensor(manifold_dim))
            for _ in range(3)  # Create 3 test measurements
        ]
        bases = [f"|{i}⟩" for i in range(manifold_dim)]

        # Run validation
        result = validation_framework.validate_quantum_state(
            state=state,
            measurements=measurements
        )

        # Check result structure
        assert result.is_valid is not None
        assert result.data is not None, "Validation result data should not be None"
        assert "quantum" in result.data, "Validation result should contain quantum data"
        assert "metrics" in result.data["quantum"]

        # Check quantum metrics
        metrics = result.data["quantum"]["metrics"]
        assert "normalization" in metrics
        assert "unitarity" in metrics
        assert "energy_conservation" in metrics

        # Check quantum properties
        assert "entanglement" in result.data["quantum"]
        entanglement = result.data["quantum"]["entanglement"]
        assert "entanglement_entropy" in entanglement
        assert "mutual_information" in entanglement
        assert "relative_entropy" in entanglement

    @pytest.mark.level0
    def test_pattern_validation(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        batch_size: int,
        manifold_dim: int
    ):
        """Test pattern validation."""
        # Create test pattern
        pattern = torch.randn(batch_size, manifold_dim)

        # Create test dynamics
        dynamics = AttentionPatternDynamics(
            grid_size=manifold_dim,
            space_dim=2,
            hidden_dim=manifold_dim
        )

        # Run validation
        result = validation_framework.validate_pattern_formation(
            pattern=pattern,
            dynamics=dynamics,
            time_steps=1000
        )

        # Check result structure
        assert result.is_valid is not None
        assert result.data is not None, "Validation result data should not be None"
        assert "pattern" in result.data, "Validation result should contain pattern data"
        assert "metrics" in result.data["pattern"]

        # Check pattern metrics
        metrics = result.data["pattern"]["metrics"]
        assert "stability" in metrics
        assert "linear_stability" in metrics["stability"]
        assert "nonlinear_stability" in metrics["stability"]

        # Check pattern properties
        assert "eigenvalues" in result.data["pattern"]
        assert "eigenvectors" in result.data["pattern"]
        assert "growth_rates" in result.data["pattern"]

    @pytest.mark.level1
    def test_integrated_validation(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        batch_size: int,
        manifold_dim: int
    ):
        """Test integrated validation workflow."""
        # Create test data
        points = torch.randn(batch_size, manifold_dim)
        metric_tensor = riemannian_framework.compute_metric(points)
        metric = metric_tensor.values  # Extract raw tensor values

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data=points,
            metric=metric,
            riemannian=riemannian_framework
        )

        # Check result structure
        assert result.is_valid
        assert result.data is not None, "Validation result data should not be None"
        assert all(k in result.data for k in ["geometric", "quantum", "pattern"])
        assert "framework_metrics" in result.data
        metrics = result.data["framework_metrics"]
        assert all(k in metrics for k in ["accuracy", "consistency", "completeness"])

        # Check component scores
        assert "component_scores" in result.data
        scores = result.data["component_scores"]
        assert all(k in scores for k in ["geometric", "quantum", "pattern"])
        assert all(0 <= score <= 1 for score in scores.values())

    def test_error_handling(
        self,
        validation_framework: ValidationFramework,
        geometric_validator: ModelGeometricValidator
    ):
        """Test error handling in validation framework."""
        # Test invalid metric shape
        with pytest.raises(ValueError, match="Invalid metric shape"):
            geometric_validator.validate_layer_geometry("default", torch.ones(1))
        
        # Test invalid quantum state
        with pytest.raises(ValueError, match="Invalid quantum state shape"):
            validation_framework.validate_quantum_state(torch.ones(1))
        
        # Test invalid pattern
        with pytest.raises(ValueError, match="Invalid pattern shape"):
            validation_framework.validate_pattern_formation(torch.ones(1))
        
        # Test incompatible dimensions
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            validation_framework.validate_all(
                model=None,
                data=torch.ones(2, 4),
                metric=torch.ones(3, 3)
            )
            
        # Test invalid parameter types
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            geometric_validator.validate_layer_geometry("default", torch.tensor([1, 2, 3]))
            
        # Test NaN/Inf handling
        with pytest.raises(ValueError, match="Contains NaN or Inf values"):
            geometric_validator.validate_layer_geometry(
                "default",
                torch.tensor([[float('nan'), 0], [0, 1]])
            )
            
        # Test empty tensors
        with pytest.raises(ValueError, match="Empty tensor"):
            geometric_validator.validate_layer_geometry("default", torch.tensor([]))

    def test_validation_metrics(self):
        """Test validation metrics computation and aggregation."""
        metrics = ConcreteValidationResult(
            is_valid=True,
            message="Test validation metrics",
            data={
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
            curvature_bounds=(-1.0, 1.0),
            energy_metrics={"total": 0.5},
            bifurcation_points=[torch.tensor([1.0])],
            stability_eigenvalues=torch.tensor([0.1]),
            framework_accuracy=0.95,
            framework_consistency=0.90,
            metrics={
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
        self, validation_framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test full integration of all validation components."""
        # Create test data
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)  # Make symmetric positive definite
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data=state,
            metric=metric
        )
        
        # Test result structure
        assert isinstance(result, FrameworkValidationResult)
        assert result.data is not None
        assert all(k in result.data for k in ["geometric", "quantum", "pattern"])
        
        # Test metrics
        assert result.data["geometric"] is not None
        assert result.data["quantum"] is not None
        assert result.data["pattern"] is not None
        
        # Test framework metrics
        assert "framework_metrics" in result.data
        metrics = result.data["framework_metrics"]
        assert all(k in metrics for k in ["accuracy", "consistency", "completeness"])
        assert all(0 <= metrics[k] <= 1 for k in ["accuracy", "consistency", "completeness"])

    def test_validate_all(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        batch_size: int,
        manifold_dim: int
    ):
        """Test full validation pipeline."""
        # Create test data
        points = torch.randn(batch_size, manifold_dim)
        metric_tensor = riemannian_framework.compute_metric(points)
        metric = metric_tensor.values  # Extract raw tensor values

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data=points,
            metric=metric,
            riemannian=riemannian_framework
        )

        # Check result structure
        assert result.is_valid
        assert result.data is not None
        assert all(k in result.data for k in ["geometric", "quantum", "pattern"])

        # Check component results
        assert result.geometric_result is not None
        assert result.quantum_result is not None
        assert result.pattern_result is not None

        # Check framework metrics
        assert "framework_metrics" in result.data
        metrics = result.data["framework_metrics"]
        assert all(k in metrics for k in ["accuracy", "consistency", "completeness"])
        assert all(0 <= metrics[k] <= 1 for k in ["accuracy", "consistency", "completeness"])
