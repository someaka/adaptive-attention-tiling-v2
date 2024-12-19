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
            device=torch.device('cpu')
        )

    def test_pattern_quantum_interaction(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test interaction between pattern and quantum components."""
        # Generate quantum state that represents pattern
        state = torch.randn(batch_size, 1, dim, dtype=torch.complex64)  # Add sequence dimension
        state = state / torch.norm(state, dim=2, keepdim=True)  # Normalize along feature dimension

        # Extract pattern from quantum state
        pattern = torch.abs(state) ** 2

        # Validate both representations
        state_result = framework.validate_quantum_state(state.squeeze(1))  # Remove sequence dimension for validation
        pattern_result = framework.validate_pattern_formation(pattern.squeeze(1))  # Remove sequence dimension for validation

        # Verify consistency
        assert state_result is not None and state_result.is_valid
        assert pattern_result is not None and pattern_result.is_valid
        assert torch.allclose(
            torch.sum(pattern.squeeze(1), dim=1),  # Sum over feature dimension
            torch.ones(batch_size),
            rtol=1e-5
        )

    def test_geometric_pattern_coupling(
        self, framework: ValidationFramework, flow: GeometricFlow, batch_size: int, dim: int
    ):
        """Test coupling between geometric and pattern components."""
        # Generate metric tensor
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)

        # Generate pattern compatible with metric
        pattern = torch.randn(batch_size, 1, dim)  # Add sequence dimension for PatternFlow
        pattern = pattern / torch.norm(pattern, dim=2, keepdim=True)  # Normalize along feature dimension

        # Validate geometric consistency
        metric_result = framework.geometric_validator.validate_model_geometry(
            batch_size=batch_size,
            manifold_dim=dim
        )
        pattern_result = framework.validate_pattern_formation(
            pattern=pattern.squeeze(1),  # Remove sequence dimension for pattern validation
            dynamics=None,
            time_steps=100
        )

        # Test metric-induced evolution
        output, metrics = flow(pattern)
        flow_result = framework.pattern_validator.validate(
            pattern_flow=flow,
            initial_state=pattern.squeeze(1),  # Remove sequence dimension for validation
            time_steps=100
        )

        # Check results with proper null checks
        assert metric_result is not None and metric_result.is_valid
        assert pattern_result is not None and pattern_result.is_valid
        assert flow_result is not None and flow_result.is_valid
        if flow_result.data is not None:
            assert flow_result.data.get("nonlinear_result", {}).get("lyapunov_function", 0) > 0
            assert flow_result.data.get("nonlinear_result", {}).get("basin_size", 0) > 0
            assert flow_result.data.get("nonlinear_result", {}).get("perturbation_bound", 0) > 0

    def test_infrastructure_framework(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test integration between infrastructure and validation framework."""
        # Initialize infrastructure
        cpu_opt = CPUOptimizer(enable_profiling=True, enable_memory_tracking=True)
        mem_mgr = MemoryManager(pool_size=1024, enable_monitoring=True)  # 1GB pool

        # Generate test data
        data = torch.randn(batch_size, 1, dim)  # Add sequence dimension
        metric = data.squeeze(1) @ data.squeeze(1).t()  # Create a valid metric tensor

        # Test CPU optimization
        @cpu_opt.profile_execution
        def run_validation(data: torch.Tensor, metric: torch.Tensor) -> FrameworkValidationResult:
            return framework.validate_all(
                model=None,
                data=data.squeeze(1),  # Remove sequence dimension for validation
                metric=metric,
                riemannian=None
            )

        result = run_validation(data, metric)

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
        assert result.flow_result is not None
        assert result.curvature_bounds is not None
        assert result.is_valid
        assert result.message is not None
        if result.data is not None:
            assert "geometric" in result.data
            assert "quantum" in result.data
            assert "pattern" in result.data
            assert "flow" in result.data

        # Test memory management
        optimized_data = mem_mgr.optimize_tensor(data, access_pattern="sequential")
        optimized_metric = optimized_data.squeeze(1) @ optimized_data.squeeze(1).t()  # Create metric from optimized data

        @cpu_opt.profile_execution
        def run_optimized_validation(data: torch.Tensor, metric: torch.Tensor) -> FrameworkValidationResult:
            return framework.validate_all(
                model=None,
                data=data.squeeze(1),  # Remove sequence dimension for validation
                metric=metric,
                riemannian=None
            )

        metrics = run_optimized_validation(optimized_data, optimized_metric)

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

        # Check metrics structure with proper null checks
        assert metrics is not None
        assert metrics.geometric_result is not None
        assert metrics.quantum_result is not None
        assert metrics.pattern_result is not None
        assert metrics.flow_result is not None
        assert metrics.curvature_bounds is not None
        assert metrics.is_valid
        assert metrics.message is not None
        if metrics.data is not None:
            assert "geometric" in metrics.data
            assert "quantum" in metrics.data
            assert "pattern" in metrics.data
            assert "flow" in metrics.data

    def test_end_to_end_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test complete end-to-end validation pipeline."""
        # Generate test configuration
        config = {
            "metric": torch.randn(batch_size, dim, dim),
            "quantum_state": torch.randn(batch_size, 1, dim, dtype=torch.complex64),  # Add sequence dimension
            "pattern": torch.randn(batch_size, 1, dim),  # Add sequence dimension
            "flow": torch.randn(batch_size, dim, dim),
            "parameters": torch.linspace(0, 1, 10),
        }

        # Normalize and prepare data
        config["metric"] = config["metric"] @ config["metric"].transpose(-1, -2)
        config["quantum_state"] = config["quantum_state"] / torch.norm(
            config["quantum_state"], dim=2, keepdim=True  # Normalize along feature dimension
        )
        config["pattern"] = config["pattern"] / torch.norm(
            config["pattern"], dim=2, keepdim=True  # Normalize along feature dimension
        )

        # Run complete validation
        result = framework.validate_all(
            model=None,
            data=config["pattern"].squeeze(1),  # Remove sequence dimension for validation
            metric=config["metric"]
        )

        # Verify all components with proper null checks
        assert result is not None
        assert result.geometric_result is not None
        assert result.quantum_result is not None
        assert result.pattern_result is not None

        # Check consistency with proper null checks
        assert result.is_valid
        assert result.geometric_result is not None and result.geometric_result.is_valid
        assert result.quantum_result is not None and result.quantum_result.is_valid
        assert result.pattern_result is not None and result.pattern_result.is_valid

    def test_validation_stability(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test stability of validation results under perturbations."""
        # Generate base configuration
        base_config = {
            "metric": torch.randn(batch_size, dim, dim),
            "quantum_state": torch.randn(batch_size, 1, dim, dtype=torch.complex64),  # Add sequence dimension
            "pattern": torch.randn(batch_size, 1, dim),  # Add sequence dimension
        }

        # Normalize base configuration
        base_config["metric"] = base_config["metric"] @ base_config["metric"].transpose(
            -1, -2
        )
        base_config["quantum_state"] = base_config["quantum_state"] / torch.norm(
            base_config["quantum_state"], dim=2, keepdim=True  # Normalize along feature dimension
        )
        base_config["pattern"] = base_config["pattern"] / torch.norm(
            base_config["pattern"], dim=2, keepdim=True  # Normalize along feature dimension
        )

        # Get base validation result
        base_result = framework.validate_all(
            model=None,
            data=base_config["pattern"].squeeze(1),  # Remove sequence dimension for validation
            metric=base_config["metric"]
        )

        # Test perturbations
        perturbation_scales = [1e-4, 1e-3, 1e-2]
        for scale in perturbation_scales:
            # Create perturbed configuration
            perturbed_config = {
                key: value + scale * torch.randn_like(value)
                for key, value in base_config.items()
            }

            # Renormalize perturbed configuration
            perturbed_config["quantum_state"] = perturbed_config[
                "quantum_state"
            ] / torch.norm(perturbed_config["quantum_state"], dim=2, keepdim=True)  # Normalize along feature dimension
            perturbed_config["pattern"] = perturbed_config["pattern"] / torch.norm(
                perturbed_config["pattern"], dim=2, keepdim=True  # Normalize along feature dimension
            )

            # Validate perturbed configuration
            perturbed_result = framework.validate_all(
                model=None,
                data=perturbed_config["pattern"].squeeze(1),  # Remove sequence dimension for validation
                metric=perturbed_config["metric"]
            )

            # Check stability with proper null checks
            assert base_result is not None and perturbed_result is not None
            assert base_result.geometric_result is not None and perturbed_result.geometric_result is not None
            assert base_result.quantum_result is not None and perturbed_result.quantum_result is not None
            assert base_result.pattern_result is not None and perturbed_result.pattern_result is not None
            
            assert perturbed_result.is_valid == base_result.is_valid
            assert perturbed_result.geometric_result.is_valid == base_result.geometric_result.is_valid
            assert perturbed_result.quantum_result.is_valid == base_result.quantum_result.is_valid
            assert perturbed_result.pattern_result.is_valid == base_result.pattern_result.is_valid
