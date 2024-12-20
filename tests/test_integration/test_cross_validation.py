"""
Integration tests for cross-validation between different components.

Tests cover:
1. Pattern-Quantum interactions
2. Geometric-Pattern coupling
3. Infrastructure-Framework integration
4. End-to-end validation
"""

import pytest
import torch
import numpy as np

from src.validation.framework import ValidationFramework, ValidationResult, FrameworkValidationResult
from src.validation.geometric.model import ModelGeometricValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.patterns.stability import PatternValidator, LinearStabilityValidator, NonlinearStabilityValidator, StructuralStabilityValidator
from src.validation.patterns.formation import PatternFormationValidator
from src.core.tiling.geometric_flow import GeometricFlow
from src.core.performance.cpu.memory import MemoryManager, MemoryStats
from src.core.performance import CPUOptimizer, PerformanceMetrics
from src.core.models.base import LayerGeometry, ModelGeometry


class TestCrossValidation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 8

    @pytest.fixture
    def dim(self) -> int:
        return 16

    @pytest.fixture
    def manifold_dim(self) -> int:
        return 16

    @pytest.fixture
    def mock_layer(self, manifold_dim: int) -> LayerGeometry:
        """Create mock layer geometry."""
        layer = LayerGeometry(manifold_dim=manifold_dim)
        with torch.no_grad():
            layer.metric_tensor.data *= 0.01  # Initialize with small metric factors
        return layer

    @pytest.fixture
    def mock_model(self, manifold_dim: int, mock_layer: LayerGeometry) -> ModelGeometry:
        """Create mock model geometry."""
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
    def geometric_validator(self, mock_model: ModelGeometry) -> ModelGeometricValidator:
        """Create geometric validator."""
        return ModelGeometricValidator(
            model_geometry=mock_model,
            tolerance=1e-6,
            curvature_bounds=(-1.0, 1.0)
        )

    @pytest.fixture
    def quantum_validator(self) -> QuantumStateValidator:
        """Create quantum validator."""
        return QuantumStateValidator()

    @pytest.fixture
    def pattern_validator(self) -> PatternValidator:
        """Create pattern validator."""
        return PatternValidator(
            linear_validator=LinearStabilityValidator(tolerance=1e-6),
            nonlinear_validator=NonlinearStabilityValidator(tolerance=1e-6),
            structural_validator=StructuralStabilityValidator(tolerance=1e-6),
            lyapunov_threshold=0.1,
            perturbation_threshold=0.1
        )

    @pytest.fixture
    def flow(self, manifold_dim: int) -> GeometricFlow:
        """Create geometric flow."""
        return GeometricFlow(
            hidden_dim=manifold_dim * 2,
            manifold_dim=manifold_dim,
            motive_rank=4,
            num_charts=4,
            integration_steps=10
        )

    @pytest.fixture
    def framework(
        self,
        geometric_validator: ModelGeometricValidator,
        quantum_validator: QuantumStateValidator,
        pattern_validator: PatternValidator
    ) -> ValidationFramework:
        """Create validation framework."""
        return ValidationFramework(
            geometric_validator=geometric_validator,
            quantum_validator=quantum_validator,
            pattern_validator=pattern_validator,
            tolerance=1e-6
        )

    def test_pattern_quantum_interaction(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test interaction between pattern and quantum components."""
        # Generate quantum state that represents pattern
        state = torch.randn(batch_size, 1, dim, dtype=torch.complex64)  # Add sequence dimension
        state = state / torch.norm(state, dim=2, keepdim=True)  # Normalize along feature dimension
        points = state.squeeze(1)

        # Extract pattern from quantum state
        pattern = torch.abs(state) ** 2

        # Validate both representations
        result = framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': pattern.squeeze(1),
                    'pattern_flow': None
                },
                'quantum_state': points
            }
        )

        # Verify consistency
        assert result.is_valid
        assert result.quantum_result is not None
        assert result.pattern_result is not None
        assert torch.allclose(
            torch.sum(pattern.squeeze(1), dim=1),  # Sum over feature dimension
            torch.ones(batch_size),
            rtol=1e-5
        )

    def test_geometric_pattern_coupling(
        self, framework: ValidationFramework, flow: GeometricFlow, batch_size: int, dim: int
    ):
        """Test coupling between geometric and pattern components."""
        # Generate pattern
        pattern = torch.randn(batch_size, 1, dim)  # Add sequence dimension for PatternFlow
        pattern = pattern / torch.norm(pattern, dim=2, keepdim=True)  # Normalize along feature dimension
        points = pattern.squeeze(1)

        # Validate geometric consistency
        result = framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': points,
                    'pattern_flow': flow
                }
            }
        )

        # Check results with proper null checks
        assert result.is_valid
        assert result.geometric_result is not None
        assert result.pattern_result is not None
        if result.data is not None:
            assert result.data.get("pattern", {}).get("nonlinear_result", {}).get("lyapunov_function", 0) > 0
            assert result.data.get("pattern", {}).get("nonlinear_result", {}).get("basin_size", 0) > 0
            assert result.data.get("pattern", {}).get("nonlinear_result", {}).get("perturbation_bound", 0) > 0

    def test_infrastructure_framework(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test integration between infrastructure and validation framework."""
        # Initialize infrastructure
        cpu_opt = CPUOptimizer(enable_profiling=True, enable_memory_tracking=True)
        mem_mgr = MemoryManager(pool_size=1024, enable_monitoring=True)  # 1GB pool

        # Generate test data
        data = torch.randn(batch_size, 1, dim)  # Add sequence dimension
        points = data.squeeze(1)  # Create points for geometric validation

        # Test CPU optimization
        @cpu_opt.profile_execution
        def run_validation(data: torch.Tensor) -> FrameworkValidationResult:
            return framework.validate_all(
                model=None,
                data={
                    'points': data,
                    'patterns': {
                        'initial_state': data,
                        'pattern_flow': None
                    },
                    'quantum_state': data
                }
            )

        result = run_validation(points)

        # Check performance metrics
        metrics = cpu_opt.get_performance_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time > 0
        assert metrics.memory_usage > 0
        assert 0 <= metrics.cpu_utilization <= 100
        assert metrics.cache_hits >= 0
        assert 0 <= metrics.vectorization_efficiency <= 1.0

        # Check result structure
        assert isinstance(result, FrameworkValidationResult)
        assert result.geometric_result is not None
        assert result.quantum_result is not None
        assert result.pattern_result is not None
        assert result.is_valid
        assert result.message is not None
        if result.data is not None:
            assert "geometric" in result.data
            assert "quantum" in result.data
            assert "pattern" in result.data

        # Test memory management
        optimized_data = mem_mgr.optimize_tensor(points, access_pattern="sequential")

        @cpu_opt.profile_execution
        def run_optimized_validation(data: torch.Tensor) -> FrameworkValidationResult:
            return framework.validate_all(
                model=None,
                data={
                    'points': data,
                    'patterns': {
                        'initial_state': data,
                        'pattern_flow': None
                    },
                    'quantum_state': data
                }
            )

        optimized_result = run_optimized_validation(optimized_data)

        # Check memory stats
        memory_stats = mem_mgr.get_memory_stats()
        assert len(memory_stats) > 0
        for stat in memory_stats:
            assert isinstance(stat, MemoryStats)
            assert stat.allocation_size > 0
            assert stat.pool_hits >= 0
            assert stat.cache_hits >= 0
            assert 0 <= stat.fragmentation <= 1.0
            assert stat.access_pattern in ["sequential", "random", "interleaved"]

    def test_end_to_end_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test end-to-end validation pipeline."""
        # Generate test data
        data = torch.randn(batch_size, 1, dim)  # Add sequence dimension
        points = data.squeeze(1)

        # Run end-to-end validation
        result = framework.validate_all(
            model=None,
            data={
                'points': points,
                'patterns': {
                    'initial_state': points,
                    'pattern_flow': None
                },
                'quantum_state': points
            }
        )

        # Check validation results
        assert result.is_valid
        assert result.geometric_result is not None
        assert result.quantum_result is not None
        assert result.pattern_result is not None
        assert result.message is not None
        if result.data is not None:
            assert "geometric" in result.data
            assert "quantum" in result.data
            assert "pattern" in result.data

    def test_validation_stability(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test validation framework stability."""
        # Generate test data
        data = torch.randn(batch_size, 1, dim)  # Add sequence dimension
        points = data.squeeze(1)

        # Run multiple validations
        results = []
        for _ in range(5):
            result = framework.validate_all(
                model=None,
                data={
                    'points': points,
                    'patterns': {
                        'initial_state': points,
                        'pattern_flow': None
                    },
                    'quantum_state': points
                }
            )
            results.append(result)

        # Check consistency across runs
        for i in range(1, len(results)):
            assert results[i].is_valid == results[0].is_valid
            assert results[i].message == results[0].message
            if results[i].data is not None and results[0].data is not None:
                assert results[i].data.keys() == results[0].data.keys()
                for key in results[0].data:
                    assert key in results[i].data
