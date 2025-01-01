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
from validation.flow.flow_stability import LinearStabilityValidator, NonlinearStabilityValidator
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
from src.core.patterns.riemannian_flow import RiemannianFlow
from src.validation.geometric.metric import MetricValidator


class TestValidationFramework:
    @pytest.fixture
    def batch_size(self, test_config) -> int:
        """Batch size for testing."""
        return test_config['fiber_bundle']['batch_size']

    @pytest.fixture
    def dim(self, test_config) -> int:
        """Dimension for testing."""
        return test_config['geometric_tests']['dimensions']

    @pytest.fixture
    def manifold_dim(self, test_config) -> int:
        """Manifold dimension for testing."""
        return test_config['quantum_geometric']['manifold_dim']

    @pytest.fixture
    def state_dim(self, test_config) -> int:
        """State dimension for testing."""
        return test_config['quantum_geometric']['hidden_dim']

    @pytest.fixture
    def pattern_dim(self, test_config) -> int:
        """Pattern dimension for testing."""
        return test_config['geometric_tests']['dimensions']

    @pytest.fixture
    def mock_layer(self, manifold_dim: int) -> LayerGeometry:
        """Create mock layer geometry."""
        return LayerGeometry(manifold_dim=manifold_dim, pattern_dim=manifold_dim)

    @pytest.fixture
    def model_geometry(self, manifold_dim: int, test_config: Dict) -> ModelGeometry:
        """Create model geometry."""
        return ModelGeometry(
            manifold_dim=manifold_dim,
            query_dim=test_config['geometric_tests']['hidden_dim'],
            key_dim=test_config['geometric_tests']['hidden_dim'],
            layers={
                'input': LayerGeometry(manifold_dim=manifold_dim, pattern_dim=manifold_dim),
                'hidden': LayerGeometry(manifold_dim=manifold_dim, pattern_dim=manifold_dim),
                'output': LayerGeometry(manifold_dim=manifold_dim, pattern_dim=manifold_dim)
            },
            attention_heads=[]
        )

    @pytest.fixture
    def geometric_validator(self, model_geometry: ModelGeometry, test_config: Dict) -> ModelGeometricValidator:
        """Create geometric validator."""
        return ModelGeometricValidator(
            model_geometry=model_geometry,
            tolerance=test_config['fiber_bundle']['tolerance'],
            curvature_bounds=(-1.0, 1.0)
        )

    @pytest.fixture
    def linear_stability_validator(self, test_config: Dict) -> LinearStabilityValidator:
        """Create linear stability validator."""
        return LinearStabilityValidator(
            tolerance=test_config['quantum_arithmetic']['tolerances']['state_norm'],
            stability_threshold=test_config['quantum_arithmetic']['validation']['stability_threshold']
        )

    @pytest.fixture
    def nonlinear_stability_validator(self, test_config: Dict) -> NonlinearStabilityValidator:
        """Create nonlinear stability validator."""
        return NonlinearStabilityValidator(
            tolerance=test_config['quantum_arithmetic']['tolerances']['state_norm'],
            basin_samples=test_config['geometric_tests']['num_heads'] * 10  # Scale with complexity
        )

    @pytest.fixture
    def quantum_validator(self) -> QuantumStateValidator:
        """Create quantum validator."""
        return QuantumStateValidator()

    @pytest.fixture
    def riemannian_framework(self, manifold_dim: int, test_config: Dict) -> RiemannianFramework:
        """Create Riemannian framework for testing."""
        return PatternRiemannianStructure(
            manifold_dim=manifold_dim,
            pattern_dim=test_config['geometric_tests']['dimensions']
        )

    @pytest.fixture
    def pattern_validator(self, model_geometry: ModelGeometry, test_config: Dict) -> StabilityValidator:
        """Create pattern validator."""
        return StabilityValidator(
            linear_validator=LinearStabilityValidator(
                stability_threshold=test_config['quantum_arithmetic']['validation']['stability_threshold']
            ),
            nonlinear_validator=NonlinearStabilityValidator(),
            lyapunov_threshold=test_config['quantum_arithmetic']['validation']['convergence_threshold'],
            perturbation_threshold=test_config['quantum_arithmetic']['tolerances']['state_norm']
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

    @pytest.mark.dependency(name="test_basic_tensor_shapes")
    @pytest.mark.level0
    def test_basic_tensor_shapes(
        self,
        validation_framework: ValidationFramework,
        test_config: Dict
    ):
        """Test basic tensor shape validation. Level 0: Only depends on PyTorch."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        tolerance = float(test_config['fiber_bundle']['tolerance'])
        
        # Create test tensors with proper dtype
        points = torch.randn(batch_size, manifold_dim)
        state = torch.randn(batch_size, manifold_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Test points shape validation
        assert points.shape == (batch_size, manifold_dim)
        assert state.shape == (batch_size, manifold_dim)
        
        # Test basic metric tensor properties with proper tolerances
        metric = torch.eye(manifold_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        assert metric.shape == (batch_size, manifold_dim, manifold_dim)
        assert torch.allclose(
            metric,
            metric.transpose(-1, -2),
            rtol=tolerance
        )  # Symmetry
        assert torch.all(torch.linalg.eigvals(metric).real > 0)  # Positive definiteness

    @pytest.mark.dependency(depends=["test_basic_tensor_shapes"])
    @pytest.mark.level0
    def test_basic_metric_properties(
        self,
        validation_framework: ValidationFramework,
        test_config: Dict
    ):
        """Test basic metric tensor properties. Level 0: Only depends on PyTorch."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        tolerance = float(test_config['fiber_bundle']['tolerance'])
        
        # Create a simple metric tensor (identity matrix for each batch)
        metric = torch.eye(manifold_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Test metric tensor properties
        # 1. Symmetry
        assert torch.allclose(
            metric,
            metric.transpose(-1, -2),
            rtol=tolerance
        )
        
        # 2. Positive definiteness
        eigenvals = torch.linalg.eigvals(metric).real
        assert torch.all(eigenvals > 0)
        
        # 3. Shape consistency
        assert metric.shape == (batch_size, manifold_dim, manifold_dim)
        
        # 4. Batch independence
        for i in range(batch_size):
            assert torch.allclose(
                metric[i],
                torch.eye(manifold_dim),
                rtol=tolerance
            )

    @pytest.mark.dependency(depends=["test_basic_tensor_shapes", "test_basic_metric_properties"])
    @pytest.mark.level2
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_geometric_validation(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        test_config: Dict
    ) -> None:
        """Test geometric validation. Level 2: Depends on metric properties and tensor validation."""
        import time
        from contextlib import contextmanager
        from src.core.crystal.scale_classes.memory_utils import memory_efficient_computation
        from src.utils.memory_management import register_tensor, tensor_manager, memory_optimizer, DEBUG_MODE, clear_memory
        from src.core.performance.cpu.memory_management import MemoryManager
        import logging

        @contextmanager
        def timed_operation(name: str, timeout: float = 10.0):
            """Context manager to time operations and log memory usage with timeout."""
            start = time.perf_counter()
            try:
                yield
                duration = time.perf_counter() - start
                if duration > timeout:
                    logger.warning(f"Operation '{name}' took longer than {timeout}s: {duration:.2f}s")
                memory_stats = memory_optimizer.get_memory_stats()
                logger.info(f"Operation '{name}' completed in {duration:.2f}s")
                logger.info(f"Memory after '{name}': {memory_stats}")
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"Operation '{name}' failed after {duration:.2f}s: {str(e)}")
                raise

        # Set up logging with proper format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Get dimensions from config
        batch_size = min(test_config['fiber_bundle']['batch_size'], 2)  # Limit batch size
        manifold_dim = min(test_config['quantum_geometric']['manifold_dim'], 2)  # Limit manifold dim
        hidden_dim = min(test_config['quantum_geometric']['hidden_dim'], 4)  # Limit hidden dim
        tolerance = float(test_config['fiber_bundle']['tolerance'])
        
        logger.info("Starting geometric validation test with configuration:")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  manifold_dim: {manifold_dim}")
        logger.info(f"  hidden_dim: {hidden_dim}")
        logger.info(f"  tolerance: {tolerance}")
        
        # Clear any existing memory before starting
        clear_memory()
        
        with memory_efficient_computation("geometric_validation"):
            try:
                # Create test data with proper dimensions - using real numbers
                with timed_operation("points_creation", timeout=1.0):
                    logger.info("Creating input points...")
                    points = register_tensor(
                        torch.randn(batch_size, hidden_dim, dtype=torch.float32),
                        operation="points"
                    )
                    logger.info(f"Points tensor shape: {points.shape}, dtype: {points.dtype}")
                    
                    points = register_tensor(
                        points / torch.norm(points, dim=1, keepdim=True),
                        operation="points_normalized"
                    )
                    logger.info("Points normalized")
                
                # Create metric tensor - using real numbers
                with timed_operation("metric_creation", timeout=1.0):
                    logger.info("Creating metric tensor...")
                    metric = register_tensor(
                        torch.eye(manifold_dim, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1),
                        operation="metric_base"
                    )
                    logger.info(f"Base metric tensor shape: {metric.shape}, dtype: {metric.dtype}")
                    
                    # Add small perturbation to make it interesting but stable
                    metric = register_tensor(
                        metric + 0.01 * torch.randn(batch_size, manifold_dim, manifold_dim, dtype=torch.float32),
                        operation="metric_perturbed"
                    )
                    logger.info("Added small random perturbation to metric")
                    
                    # Make metric symmetric and positive definite
                    metric = register_tensor(
                        0.5 * (metric + metric.transpose(-2, -1)),
                        operation="metric_symmetric"
                    )
                    logger.info("Made metric symmetric")
                    
                    # Add small regularization to ensure positive definiteness
                    metric = register_tensor(
                        metric + 0.1 * torch.eye(manifold_dim, dtype=torch.float32).unsqueeze(0),
                        operation="metric_regularized"
                    )
                    logger.info("Added regularization for positive definiteness")

                # Create a metric validator for testing
                with timed_operation("validator_creation", timeout=1.0):
                    logger.info("Creating metric validator...")
                    validator = MetricValidator(
                        manifold_dim=manifold_dim,
                        tolerance=tolerance
                    )
                    logger.info("Validator created")

                # Run validation with error handling
                with timed_operation("validation", timeout=20.0):
                    logger.info("Running validation...")
                    try:
                        # Validate metric properties
                        metric_result = validator.validate_metric(metric)
                        logger.info(f"Metric validation result: {metric_result}")
                        assert metric_result.is_positive_definite, "Metric must be positive definite"
                        
                        # Compute metric properties
                        properties = validator.validate_metric_properties(metric)
                        logger.info(f"Metric properties: {properties}")
                        assert properties.is_positive_definite, "Metric must be positive definite"
                        assert properties.is_compatible, "Metric must be compatible"
                        assert properties.has_bounded_curvature, "Metric must have bounded curvature"
                        
                        # Compute curvature bounds
                        curvature_bounds = validator.validate_curvature_bounds(metric)
                        logger.info(f"Curvature bounds: {curvature_bounds}")
                        assert curvature_bounds.sectional_bounds is not None, "Sectional curvature bounds must exist"
                        assert curvature_bounds.ricci_bounds is not None, "Ricci curvature bounds must exist"
                        assert curvature_bounds.scalar_bounds is not None, "Scalar curvature bounds must exist"
                        
                        # Create validation result
                        result = validation_framework.validate_all(
                            model=None,
                            data={
                                'points': points,
                                'patterns': {
                                    'initial_state': points,
                                    'pattern_flow': None,
                                    'metric_tensor': metric,
                                    'time_steps': 10  # Reduce time steps for testing
                                },
                                'quantum_state': None
                            }
                        )
                        logger.info("Validation completed")
                        logger.info(f"Result valid: {result.is_valid}")
                        if result.data:
                            logger.info("Result data sections:")
                            for key in result.data:
                                logger.info(f"  - {key}")
                    except Exception as e:
                        logger.error(f"Validation failed: {str(e)}")
                        raise

                # Check results with error handling
                with timed_operation("result_verification", timeout=1.0):
                    logger.info("Verifying results...")
                    try:
                        # Basic structure checks
                        assert result.is_valid
                        assert result.data is not None, "Validation result data should not be None"
                        assert "geometric" in result.data, "Validation result should contain geometric data"
                        assert result.data["geometric"]["complete"]

                        # Metric tensor checks
                        assert "metric_tensor" in result.data["geometric"]
                        metric_tensor = cast(Tensor, result.data["geometric"]["metric_tensor"])
                        metric_tensor = register_tensor(metric_tensor, operation="metric_tensor_result")
                        logger.info(f"Result metric tensor shape: {metric_tensor.shape}")
                        
                        # Shape checks
                        assert len(metric_tensor.shape) == 3
                        assert metric_tensor.shape[0] == batch_size
                        assert metric_tensor.shape[-2:] == (manifold_dim, manifold_dim)
                        
                        # Symmetry check
                        assert torch.allclose(
                            metric_tensor,
                            metric_tensor.transpose(-2, -1),
                            rtol=tolerance
                        )

                        # Curvature checks
                        assert "curvature" in result.data["geometric"]
                        curvature_data = result.data["geometric"]["curvature"]
                        assert "sectional_curvature" in curvature_data
                        assert "ricci_curvature" in curvature_data
                        
                        sectional = register_tensor(curvature_data["sectional_curvature"], operation="sectional_curvature")
                        ricci = register_tensor(curvature_data["ricci_curvature"], operation="ricci_curvature")
                        logger.info(f"Sectional curvature shape: {sectional.shape}")
                        logger.info(f"Ricci curvature shape: {ricci.shape}")
                        assert sectional.shape[0] == batch_size
                        assert ricci.shape == (batch_size, manifold_dim, manifold_dim)
                    except Exception as e:
                        logger.error(f"Result verification failed: {str(e)}")
                        raise

            except Exception as e:
                logger.error(f"Test failed with error: {str(e)}")
                raise
            finally:
                # Clean up
                logger.info("Cleaning up resources...")
                clear_memory()
                logger.info("Test completed")

    @pytest.mark.level1
    def test_quantum_validation(
        self,
        validation_framework: ValidationFramework,
        test_config: Dict
    ) -> None:
        """Test quantum validation."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        hidden_dim = test_config['quantum_geometric']['hidden_dim']
        
        # Create test quantum state with proper dimensionality
        state = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Create geometric pattern from quantum state
        geometric_pattern = torch.real(state @ state.conj().transpose(-2, -1))
        geometric_pattern = geometric_pattern.to(dtype=torch.float32)
        geometric_pattern = geometric_pattern / torch.norm(geometric_pattern, dim=1, keepdim=True)

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'quantum_state': state,
                'patterns': {
                    'initial_state': geometric_pattern,
                    'pattern_flow': None,
                    'quantum_dim': manifold_dim
                }
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

        # Check quantum properties with proper tolerances
        assert "entanglement" in result.data["quantum"]
        entanglement = result.data["quantum"]["entanglement"]
        assert "entanglement_entropy" in entanglement
        assert "mutual_information" in entanglement
        assert "relative_entropy" in entanglement

    @pytest.mark.level1
    def test_pattern_validation(
        self,
        validation_framework: ValidationFramework,
        test_config: Dict
    ):
        """Test pattern validation."""
        # Use config dimensions
        grid_size = test_config['geometric_tests']['dimensions']
        hidden_dim = test_config['quantum_geometric']['hidden_dim']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        batch_size = test_config['fiber_bundle']['batch_size']
        dt = float(test_config['quantum_geometric']['dt'])

        # Create quantum state pattern with proper dimensionality
        k_space = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        pattern = torch.fft.ifft(k_space, dim=1)  # Create quantum field pattern
        pattern = pattern / torch.norm(pattern, dim=1, keepdim=True)

        # Create geometric pattern using quantum Fisher metric
        geometric_pattern = torch.real(pattern @ pattern.conj().transpose(-2, -1))
        geometric_pattern = geometric_pattern.to(dtype=torch.float32)
        geometric_pattern = geometric_pattern / torch.norm(geometric_pattern, dim=1, keepdim=True)

        # Create test dynamics with quantum features enabled
        dynamics = PatternDynamics(
            grid_size=grid_size,
            space_dim=manifold_dim,  # Use manifold_dim for space dimension
            hidden_dim=hidden_dim,
            quantum_enabled=True,
            dt=dt,
            num_modes=test_config['geometric_tests']['num_heads']
        )

        # Create pattern transition with wave emergence
        pattern_transition = PatternTransition(
            wave_emergence=WaveEmergence(
                dt=dt,
                num_steps=test_config['geometric_tests']['num_heads']
            )
        )

        # Run validation with both quantum and geometric patterns
        result = validation_framework.validate_all(
            model=None,
            data={
                'patterns': {
                    'initial_state': geometric_pattern,
                    'pattern_flow': dynamics,
                    'quantum_dim': manifold_dim,
                    'berry_phase': torch.imag(pattern @ pattern.conj().transpose(-2, -1)).to(dtype=torch.float32),
                    'pattern_transition': pattern_transition
                },
                'quantum_state': pattern
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
            if 'state_dim' in quantum_data:
                assert quantum_data['state_dim'] >= manifold_dim

    @pytest.mark.level1
    def test_integrated_validation(
        self,
        validation_framework: ValidationFramework,
        riemannian_framework: RiemannianFramework,
        test_config: Dict
    ):
        """Test integrated validation workflow."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        
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

        # Check component scores with proper tolerances
        assert "component_scores" in result.data
        scores = result.data["component_scores"]
        assert all(k in scores for k in ["geometric", "quantum", "pattern"])
        assert all(0 <= score <= 1 for score in scores.values())

    def test_error_handling(
        self,
        validation_framework: ValidationFramework,
        geometric_validator: ModelGeometricValidator,
        test_config: Dict
    ):
        """Test error handling in validation framework."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        hidden_dim = test_config['quantum_geometric']['hidden_dim']
        
        # Test invalid metric shape
        with pytest.raises(ValueError):
            geometric_validator.validate_layer_geometry("default", torch.ones(1))
        
        # Test invalid quantum state
        with pytest.raises(ValueError):
            validation_framework.validate_quantum(torch.ones(1))
        
        # Test invalid pattern
        with pytest.raises(ValueError):
            validation_framework.validate_patterns(torch.ones(1))
        
        # Test incompatible dimensions
        with pytest.raises(ValueError):
            validation_framework.validate_all(
                model=None,
                data={
                    'points': torch.ones(batch_size, hidden_dim + 1, dtype=torch.complex64),  # Wrong hidden_dim
                    'patterns': {
                        'initial_state': torch.ones(batch_size, manifold_dim),
                        'pattern_flow': None
                    }
                }
            )
            
        # Test invalid parameter types
        with pytest.raises(TypeError):
            geometric_validator.validate_layer_geometry(
                "default",
                torch.tensor([1, 2, 3], dtype=torch.float32).view(-1, 1)  # Wrong shape tensor
            )
            
        # Test NaN/Inf handling
        with pytest.raises(ValueError):
            geometric_validator.validate_layer_geometry(
                "default",
                torch.tensor([[float('nan'), 0], [0, 1]], dtype=torch.float32)
            )

    def test_validation_metrics(self, test_config: Dict):
        """Test validation metrics computation."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        hidden_dim = test_config['quantum_geometric']['hidden_dim']
        tolerance = float(test_config['fiber_bundle']['tolerance'])
        stability_threshold = float(test_config['quantum_arithmetic']['validation']['stability_threshold'])
        convergence_threshold = float(test_config['quantum_arithmetic']['validation']['convergence_threshold'])
        state_norm_tolerance = float(test_config['quantum_arithmetic']['tolerances']['state_norm'])
        
        # Create test data with proper dimensionality
        points = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        points = points / torch.norm(points, dim=1, keepdim=True)  # Normalize points
        state = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Create geometric pattern from quantum state
        geometric_pattern = torch.real(state @ state.conj().transpose(-2, -1))
        geometric_pattern = geometric_pattern.to(dtype=torch.float32)
        geometric_pattern = geometric_pattern / torch.norm(geometric_pattern, dim=1, keepdim=True)

        # Create validation framework
        framework = ValidationFramework(
            geometric_validator=ModelGeometricValidator(
                model_geometry=ModelGeometry(
                    manifold_dim=manifold_dim,
                    query_dim=hidden_dim,
                    key_dim=hidden_dim,
                    layers={},
                    attention_heads=[]
                ),
                tolerance=tolerance,
                curvature_bounds=(-1.0, 1.0)
            ),
            quantum_validator=QuantumStateValidator(),
            pattern_validator=StabilityValidator(
                linear_validator=LinearStabilityValidator(
                    stability_threshold=stability_threshold
                ),
                nonlinear_validator=NonlinearStabilityValidator(),
                lyapunov_threshold=convergence_threshold,
                perturbation_threshold=state_norm_tolerance
            )
        )

        # Run validation
        result = framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': geometric_pattern,
                    'pattern_flow': None,
                    'quantum_dim': manifold_dim
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
        self,
        validation_framework: ValidationFramework,
        test_config: Dict
    ):
        """Test full integration of all validation components."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        hidden_dim = test_config['quantum_geometric']['hidden_dim']
        
        # Create test data with proper dimensionality
        points = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        points = points / torch.norm(points, dim=1, keepdim=True)  # Normalize points
        state = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)
        
        # Create geometric pattern from quantum state
        geometric_pattern = torch.real(state @ state.conj().transpose(-2, -1))
        geometric_pattern = geometric_pattern.to(dtype=torch.float32)
        geometric_pattern = geometric_pattern / torch.norm(geometric_pattern, dim=1, keepdim=True)
        
        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': geometric_pattern,
                    'pattern_flow': None,
                    'quantum_dim': manifold_dim
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
        test_config: Dict
    ):
        """Test full validation pipeline."""
        # Get dimensions from config
        batch_size = test_config['fiber_bundle']['batch_size']
        manifold_dim = test_config['quantum_geometric']['manifold_dim']
        hidden_dim = test_config['quantum_geometric']['hidden_dim']
        
        # Create test data with proper dimensionality
        points = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        points = points / torch.norm(points, dim=1, keepdim=True)  # Normalize points
        state = torch.randn(batch_size, hidden_dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Create geometric pattern from quantum state
        geometric_pattern = torch.real(state @ state.conj().transpose(-2, -1))
        geometric_pattern = geometric_pattern.to(dtype=torch.float32)
        geometric_pattern = geometric_pattern / torch.norm(geometric_pattern, dim=1, keepdim=True)

        # Run validation
        result = validation_framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': geometric_pattern,
                    'pattern_flow': None,
                    'quantum_dim': manifold_dim
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
