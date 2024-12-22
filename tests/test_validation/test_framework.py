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
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.tiling.geometric_flow import GeometricFlow
from src.core.quantum.types import QuantumState
from src.core.patterns.dynamics import PatternDynamics as CorePatternDynamics
from src.core.patterns import (
    BaseRiemannianStructure,
    RiemannianFramework,
    PatternRiemannianStructure,
    MetricTensor,
    ChristoffelSymbols,
    CurvatureTensor,
)
from src.core.patterns.enriched_structure import PatternTransition, WaveEmergence


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

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'points': points
            }
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
        batch_size: int,
        manifold_dim: int
    ) -> None:
        """Test quantum validation."""
        # Create test quantum state
        state = torch.randn(batch_size, manifold_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'quantum_state': state
            }
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
        batch_size: int,
        manifold_dim: int
    ):
        """Test pattern validation."""
        # Use consistent dimensions for quantum-geometric interaction
        grid_size = 4
        hidden_dim = 8  # 2 * base_quantum_dim for complex phase space
        space_dim = 4   # Minimum for quantum field structure
        batch_size = 2

        # Create quantum state pattern with proper dimensionality
        # Following Ψ(x) = ∫ dk/√(2ω_k) (a(k)e^{-ikx} + a†(k)e^{ikx})
        k_space = torch.randn(batch_size, hidden_dim // 2, dtype=torch.complex64)
        pattern = torch.fft.ifft(k_space, dim=1)  # Create quantum field pattern
        pattern = pattern / torch.norm(pattern, dim=1, keepdim=True)

        # Create geometric pattern using quantum Fisher metric (real part of Q_{μν})
        # Project to real space for geometric operations while preserving structure
        geometric_pattern = torch.real(pattern @ pattern.conj().transpose(-2, -1))
        geometric_pattern = geometric_pattern.to(dtype=torch.float32)  # Ensure real dtype
        geometric_pattern = geometric_pattern / torch.norm(geometric_pattern, dim=1, keepdim=True)

        # Create test dynamics with quantum features enabled
        # Initialize with real tensors for geometric operations
        dynamics = PatternDynamics(
            grid_size=grid_size,
            space_dim=space_dim,  # 4D for quantum field structure
            hidden_dim=hidden_dim,  # Complex phase space dimension
            quantum_enabled=True,
            dt=0.1,
            num_modes=4  # Match quantum degrees of freedom
        )

        # Create pattern transition with wave emergence disabled
        pattern_transition = PatternTransition(
            wave_emergence=WaveEmergence(dt=0.1, num_steps=10)
        )

        # Run validation with both quantum and geometric patterns
        result = validation_framework.validate_all(
            model=None,
            data={
                'patterns': {
                    'initial_state': geometric_pattern,  # Fisher metric for geometric ops
                    'pattern_flow': dynamics,
                    'quantum_dim': space_dim,  # Explicitly specify quantum dimension
                    'berry_phase': torch.imag(pattern @ pattern.conj().transpose(-2, -1)).to(dtype=torch.float32),  # ω_{μν}
                    'pattern_transition': pattern_transition  # Add pattern transition
                },
                'quantum_state': pattern  # Full quantum state with phase
            }
        )

        # Validation checks
        assert result.is_valid is not None
        assert result.data is not None
        assert 'patterns' in result.data
        assert 'quantum' in result.data

        # Check quantum-pattern interaction with proper dimension verification
        if result.data.get('quantum'):
            quantum_data = result.data['quantum']
            assert 'metrics' in quantum_data
            assert 'entanglement' in quantum_data
            # Verify quantum dimension preservation
            if 'state_dim' in quantum_data:
                assert quantum_data['state_dim'] >= space_dim

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
        state = torch.randn(batch_size, manifold_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': points,
                    'pattern_flow': None
                },
                'quantum_state': state
            }
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
        with pytest.raises(ValueError, match="Invalid points shape"):
            geometric_validator.validate_layer_geometry("default", torch.ones(1))
        
        # Test invalid quantum state
        with pytest.raises(ValueError, match="Invalid quantum state shape"):
            validation_framework.validate_quantum(torch.ones(1))
        
        # Test invalid pattern
        with pytest.raises(ValueError, match="Invalid pattern shape"):
            validation_framework.validate_patterns(torch.ones(1))
        
        # Test incompatible dimensions
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            validation_framework.validate_all(
                model=None,
                data={
                    'points': torch.ones(2, 4),
                    'patterns': {
                        'initial_state': torch.ones(3, 3),
                        'pattern_flow': None
                    }
                }
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

    def test_validation_metrics(self):
        """Test validation metrics computation."""
        # Create test data
        batch_size = 16
        dim = 8
        points = torch.randn(batch_size, dim)
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Create validation framework
        framework = ValidationFramework(
            geometric_validator=ModelGeometricValidator(
                model_geometry=ModelGeometry(
                    manifold_dim=dim,
                    query_dim=dim,
                    key_dim=dim,
                    layers={},
                    attention_heads=[]
                ),
                tolerance=1e-6,
                curvature_bounds=(-1.0, 1.0)
            ),
            quantum_validator=QuantumStateValidator(),
            pattern_validator=StabilityValidator(
                linear_validator=LinearStabilityValidator(stability_threshold=0.0),
                nonlinear_validator=NonlinearStabilityValidator(),
                lyapunov_threshold=1e-6,
                perturbation_threshold=1e-6
            )
        )

        # Run validation
        result = framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': points,
                    'pattern_flow': None
                },
                'quantum_state': state
            }
        )

        # Check metrics structure
        assert result.data is not None
        assert "framework_metrics" in result.data
        metrics = result.data["framework_metrics"]
        assert all(k in metrics for k in ["accuracy", "consistency", "completeness"])
        assert all(0 <= metrics[k] <= 1 for k in metrics.keys())

    def test_full_integration(
        self, validation_framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test full integration of all validation components."""
        # Create test data
        points = torch.randn(batch_size, dim)
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': points,
                    'pattern_flow': None
                },
                'quantum_state': state
            }
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
        assert all(0 <= metrics[k] <= 1 for k in metrics.keys())

    def test_validate_all(
        self,
        validation_framework: ValidationFramework,
        batch_size: int,
        manifold_dim: int
    ):
        """Test full validation pipeline."""
        # Create test data
        points = torch.randn(batch_size, manifold_dim)
        state = torch.randn(batch_size, manifold_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': points,
                    'pattern_flow': None
                },
                'quantum_state': state
            }
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
        assert all(0 <= metrics[k] <= 1 for k in metrics.keys())
