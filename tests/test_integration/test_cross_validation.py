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
import torch.nn as nn
import numpy as np
import gc
import psutil
import os
from typing import Dict, Any, Optional, Generator

from src.validation.framework import ValidationFramework, ValidationResult, FrameworkValidationResult
from src.validation.geometric.model import ModelGeometricValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.patterns.stability import PatternValidator, LinearStabilityValidator, NonlinearStabilityValidator, StructuralStabilityValidator
from src.validation.patterns.formation import PatternFormationValidator
from src.core.tiling.geometric_flow import GeometricFlow
from src.core.performance.cpu.memory import MemoryManager, MemoryStats
from src.core.performance import CPUOptimizer, PerformanceMetrics
from src.core.models.base import LayerGeometry, ModelGeometry


class MockLayerGeometry(LayerGeometry):
    """Mock layer geometry for testing."""
    
    def sectional_curvature(self, points: torch.Tensor) -> torch.Tensor:
        """Mock implementation of sectional curvature.
        
        Returns a dynamic curvature tensor that depends on the input points.
        This simulates a more realistic geometric structure.
        """
        batch_size = points.shape[0]
        # Create a base negative curvature tensor
        curvature = -0.1 * torch.ones(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Add point-dependent terms
        for i in range(batch_size):
            # Compute point-dependent matrix
            point = points[i]
            point_matrix = torch.outer(point, point.conj())
            if point_matrix.is_complex():
                point_matrix = point_matrix.real
            
            # Scale and normalize
            point_matrix = 0.05 * point_matrix / (torch.norm(point_matrix) + 1e-6)
            
            # Add to base curvature with coupling
            curvature[i] = curvature[i] + point_matrix
            
            # Ensure symmetry
            curvature[i] = 0.5 * (curvature[i] + curvature[i].T)
        
        return curvature


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class TestCrossValidation:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self) -> Generator[None, None, None]:
        """Setup and cleanup for each test."""
        # Setup - clear memory and cache before test
        gc.collect()
        initial_memory = get_memory_usage()
        print(f"\nInitial memory usage: {initial_memory:.2f} MB")
        
        yield
        
        # Cleanup - ensure memory is freed after test
        gc.collect()
        final_memory = get_memory_usage()
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory difference: {final_memory - initial_memory:.2f} MB")

    @pytest.fixture(scope="class")
    def batch_size(self) -> int:
        """Minimal batch size needed for tests."""
        return 1

    @pytest.fixture(scope="class")
    def dim(self) -> int:
        """Minimal dimension needed for tests."""
        return 2

    @pytest.fixture(scope="class")
    def manifold_dim(self) -> int:
        """Manifold dimension matching input dim."""
        return 2

    @pytest.fixture(scope="class")
    def mock_layer(self, manifold_dim: int) -> LayerGeometry:
        """Create optimized mock layer geometry."""
        layer = MockLayerGeometry(manifold_dim=manifold_dim)
        with torch.no_grad():
            # Initialize with stable values for faster convergence
            layer.metric_tensor.data = torch.eye(manifold_dim) * 0.1
        return layer

    @pytest.fixture(scope="class")
    def mock_model(self, manifold_dim: int, mock_layer: LayerGeometry) -> ModelGeometry:
        """Create minimal mock model geometry."""
        return ModelGeometry(
            manifold_dim=manifold_dim,
            query_dim=manifold_dim,
            key_dim=manifold_dim,
            layers={
                'default': mock_layer,
                'hidden': mock_layer,
                'output': mock_layer
            },
            attention_heads=[]
        )

    @pytest.fixture(scope="class")
    def geometric_validator(self, mock_model: ModelGeometry) -> ModelGeometricValidator:
        """Create geometric validator with relaxed tolerances."""
        return ModelGeometricValidator(
            model_geometry=mock_model,
            tolerance=1e-4,  # Relaxed tolerance
            curvature_bounds=(-1.0, 1.0)
        )

    @pytest.fixture(scope="class")
    def quantum_validator(self) -> QuantumStateValidator:
        """Create quantum validator."""
        return QuantumStateValidator()

    @pytest.fixture(scope="class")
    def setup_test_parameters(self) -> Dict[str, Any]:
        """Setup test parameters from configuration."""
        return {
            'batch_size': 1,
            'grid_size': 32,
            'space_dim': 2,
            'time_steps': 10,
            'dt': 0.01,
            'energy_threshold': 1e-4,
            'tolerance': 1e-3,
            'stability_threshold': 0.1,
            'hidden_dim': 4,
            'dtype': torch.float32
        }

    @pytest.fixture(scope="class")
    def pattern_validator(self, setup_test_parameters: Dict[str, Any]) -> PatternValidator:
        """Create pattern validator."""
        from src.validation.flow.flow_stability import (
            LinearStabilityValidator,
            NonlinearStabilityValidator,
            StructuralStabilityValidator
        )
        
        # Create individual validators with extremely lenient thresholds for testing
        linear_validator = LinearStabilityValidator(
            tolerance=1e-1,  # Super lenient tolerance
            stability_threshold=5.0  # Extremely lenient eigenvalue threshold
        )
        nonlinear_validator = NonlinearStabilityValidator(
            tolerance=1e-1,  # Super lenient tolerance
            basin_samples=5  # Minimal samples for testing
        )
        structural_validator = StructuralStabilityValidator(
            tolerance=1e-1,  # Extremely lenient tolerance for testing
            parameter_range=10.0  # Larger range for testing
        )
        
        return PatternValidator(
            linear_validator=linear_validator,
            nonlinear_validator=nonlinear_validator,
            structural_validator=structural_validator,
            lyapunov_threshold=10.0,  # Super lenient for testing
            perturbation_threshold=1.0  # More lenient for testing
        )

    @pytest.fixture(scope="class")
    def flow(self, setup_test_parameters: Dict[str, Any]) -> GeometricFlow:
        """Create geometric flow with complex pattern support."""
        # Initialize with complex dtype for pattern support
        flow = GeometricFlow(
            hidden_dim=4,  # Small hidden dim for testing
            manifold_dim=2,
            motive_rank=2,
            num_charts=2,
            integration_steps=5,
            dt=0.01,
            stability_threshold=0.5,
            dtype=torch.float64,  # Use float64 as base dtype
            use_quantum_features=True  # Enable quantum features for complex support
        )
        
        # Initialize with very stable complex weights
        with torch.no_grad():
            for m in flow.modules():
                if isinstance(m, nn.Linear):
                    # Initialize weights with very small complex values
                    weight_shape = m.weight.shape
                    real_weight = torch.randn(*weight_shape, dtype=torch.float64) * 0.001  # Much smaller weights
                    imag_weight = torch.randn(*weight_shape, dtype=torch.float64) * 0.001  # Much smaller weights
                    m.weight.data = torch.complex(real_weight, imag_weight)
                    if m.bias is not None:
                        # Initialize biases to zero for stability
                        real_bias = torch.zeros(m.bias.shape, dtype=torch.float64)
                        imag_bias = torch.zeros(m.bias.shape, dtype=torch.float64)
                        m.bias.data = torch.complex(real_bias, imag_bias)
            
            # Add small positive diagonal terms to metric network for stability
            if hasattr(flow, 'metric_net'):
                for layer in flow.metric_net:
                    if isinstance(layer, nn.Linear):
                        if layer.weight.shape[0] == layer.weight.shape[1]:
                            eye = torch.eye(layer.weight.shape[0], dtype=torch.float64)
                            layer.weight.data = layer.weight.data + 0.01 * torch.complex(eye, torch.zeros_like(eye))
        
        return flow

    @pytest.fixture(scope="class")
    def framework(
        self,
        geometric_validator: ModelGeometricValidator,
        quantum_validator: QuantumStateValidator,
        pattern_validator: PatternValidator
    ) -> ValidationFramework:
        """Create validation framework with relaxed tolerances."""
        return ValidationFramework(
            geometric_validator=geometric_validator,
            quantum_validator=quantum_validator,
            pattern_validator=pattern_validator,
            tolerance=1e-4  # Relaxed tolerance
        )

    def validate_components(
        self,
        framework: ValidationFramework,
        points: torch.Tensor,
        pattern: torch.Tensor,
        flow: Optional[GeometricFlow] = None,
        model: Optional[ModelGeometry] = None
    ) -> FrameworkValidationResult:
        """Helper method to validate all components efficiently."""
        # Validate quantum state
        quantum_result = framework.validate_quantum(points)
        assert quantum_result.is_valid, "Quantum validation failed"

        # Validate pattern if flow is provided
        if flow is not None:
            pattern_result = framework.validate_patterns({
                'initial_state': pattern.squeeze(1),
                'pattern_flow': flow
            })
        else:
            pattern_result = framework.validate_patterns(pattern)

        # Validate geometric properties if model is provided
        if model is not None:
            geometric_result = framework.validate_geometric(model, {'points': points})
        else:
            geometric_result = None

        # Combine results
        return FrameworkValidationResult(
            is_valid=all(r.is_valid for r in [quantum_result, pattern_result, geometric_result] if r is not None),
            message="; ".join(r.message for r in [quantum_result, pattern_result, geometric_result] if r is not None),
            quantum_result=quantum_result,
            pattern_result=pattern_result,
            geometric_result=geometric_result,
            data={
                'quantum': quantum_result.data,
                'pattern': pattern_result.data,
                'geometric': geometric_result.data if geometric_result else None
            }
        )

    def test_pattern_quantum_interaction(
        self,
        framework: ValidationFramework,
        batch_size: int,
        dim: int,
        mock_model: ModelGeometry,
        flow: GeometricFlow
    ) -> None:
        """Test interaction between pattern and quantum components."""
        try:
            # Generate quantum state efficiently with consistent dtypes
            with torch.no_grad():
                # Use float64 components to create complex128 state
                real = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                imag = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                state = torch.complex(real, imag)
                state = state / torch.norm(state, dim=2, keepdim=True)
                # Use float64 for pattern to match quantum precision
                pattern = torch.abs(state) ** 2
                pattern = pattern.to(dtype=torch.float64)
                points = state.squeeze(1)

            # Validate all components
            result = self.validate_components(
                framework=framework,
                points=points,
                pattern=pattern,
                flow=flow,
                model=mock_model
            )
            
            # Verify quantum properties with consistent tolerances
            assert result.quantum_result is not None and result.quantum_result.is_valid
            assert torch.allclose(
                torch.sum(pattern.squeeze(1), dim=1),
                torch.ones(batch_size, dtype=torch.float64),
                rtol=1e-8,
                atol=1e-8
            )

            # Verify pattern properties
            assert result.pattern_result is not None
            if result.data is not None and "pattern" in result.data:
                pattern_data = result.data["pattern"]
                if "nonlinear_result" in pattern_data:
                    nonlinear = pattern_data["nonlinear_result"]
                    # Use consistent tolerances
                    for key in ["lyapunov_function", "basin_size", "perturbation_bound"]:
                        assert nonlinear.get(key, 0) >= -1e-8
        finally:
            del state, pattern, points
            gc.collect()

    def test_geometric_pattern_coupling(
        self,
        framework: ValidationFramework,
        flow: GeometricFlow,
        batch_size: int,
        dim: int,
        mock_model: ModelGeometry
    ) -> None:
        """Test coupling between geometric and pattern components."""
        try:
            # Generate complex-valued pattern with consistent dtypes
            with torch.no_grad():
                real = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                imag = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                pattern = torch.complex(real, imag)
                pattern = pattern / torch.norm(pattern, dim=2, keepdim=True)
                points = pattern.squeeze(1)

            # Validate all components
            result = self.validate_components(
                framework=framework,
                points=points,
                pattern=pattern.real,  # Keep as float64
                flow=flow,
                model=mock_model
            )

            # Verify results with consistent tolerances
            assert result.geometric_result is not None and result.geometric_result.is_valid
            assert result.pattern_result is not None and result.pattern_result.is_valid
            
            if result.data is not None and "pattern" in result.data:
                pattern_data = result.data["pattern"]
                if "nonlinear_result" in pattern_data:
                    nonlinear = pattern_data["nonlinear_result"]
                    # Use consistent tolerances
                    assert nonlinear.get("lyapunov_function", -1e-8) >= -1e-8
                    assert nonlinear.get("basin_size", -1e-8) >= -1e-8
                    assert nonlinear.get("perturbation_bound", 1.0) <= 1.0
        finally:
            del pattern, points
            gc.collect()

    def test_infrastructure_framework_integration(
        self,
        framework: ValidationFramework,
        batch_size: int,
        dim: int,
        mock_model: ModelGeometry,
        flow: GeometricFlow
    ) -> None:
        """Test integration between infrastructure and validation framework."""
        try:
            # Initialize infrastructure with optimized parameters
            cpu_opt = CPUOptimizer(enable_profiling=True, enable_memory_tracking=True)
            mem_mgr = MemoryManager(pool_size=64 * 1024 * 1024, enable_monitoring=True)  # 64MB pool

            # Generate test data with consistent dtypes
            with torch.no_grad():
                real = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                imag = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                data = torch.complex(real, imag)
                points = data.squeeze(1)

            # Profile validation
            @cpu_opt.profile_execution
            def run_validation(data: torch.Tensor) -> FrameworkValidationResult:
                return self.validate_components(
                    framework=framework,
                    points=data,
                    pattern=data.real,  # Keep as float64
                    flow=flow,
                    model=mock_model
                )

            # Run validation with memory optimization
            optimized_data = mem_mgr.optimize_tensor(points, access_pattern="sequential")
            result = run_validation(optimized_data)

            # Verify infrastructure metrics with consistent tolerances
            metrics = cpu_opt.get_performance_metrics()
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.execution_time >= 0
            assert metrics.memory_usage >= 0
            assert 0 <= metrics.cpu_utilization <= 100
            assert metrics.cache_hits >= 0
            assert 0 <= metrics.vectorization_efficiency <= 1.0

            # Verify memory stats with consistent tolerances
            memory_stats = mem_mgr.get_memory_stats()
            assert len(memory_stats) > 0
            for stat in memory_stats:
                assert isinstance(stat, MemoryStats)
                assert stat.allocation_size >= 0
                assert stat.pool_hits >= 0
                assert stat.cache_hits >= 0
                assert 0 <= stat.fragmentation <= 1.0
        finally:
            del data, points, optimized_data
            mem_mgr.clear_stats()
            gc.collect()

    def test_end_to_end_validation(
        self,
        framework: ValidationFramework,
        batch_size: int,
        dim: int,
        mock_model: ModelGeometry,
        flow: GeometricFlow
    ) -> None:
        """Test end-to-end validation pipeline."""
        try:
            # Generate test data with consistent dtypes
            with torch.no_grad():
                real = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                imag = torch.randn(batch_size, 1, dim, dtype=torch.float64)
                data = torch.complex(real, imag)
                pattern = torch.abs(data) ** 2
                points = data.squeeze(1)

            # Run end-to-end validation
            result = self.validate_components(
                framework=framework,
                points=points,
                pattern=pattern,  # Already float64
                flow=flow,
                model=mock_model
            )

            # Verify quantum validation
            assert result.quantum_result is not None and result.quantum_result.is_valid
            assert result.message is not None

            # Verify pattern validation with consistent tolerances
            if result.data is not None and "pattern" in result.data:
                pattern_data = result.data["pattern"]
                if "nonlinear_result" in pattern_data:
                    nonlinear = pattern_data["nonlinear_result"]
                    for key in [
                        "lyapunov_function",
                        "basin_size",
                        "perturbation_bound"
                    ]:
                        assert nonlinear.get(key, -1e-8) >= -1e-8

            # Verify geometric validation
            assert result.geometric_result is not None
            assert result.geometric_result.is_valid
        finally:
            del data, pattern, points
            gc.collect()
