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

from src.infrastructure import CPUOptimizer, MemoryManager, VulkanIntegration
from src.validation.framework import ValidationFramework


class TestCrossValidation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 8

    @pytest.fixture
    def dim(self) -> int:
        return 16

    @pytest.fixture
    def framework(self) -> ValidationFramework:
        return ValidationFramework()

    def test_pattern_quantum_interaction(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test interaction between pattern and quantum components."""
        # Generate quantum state that represents pattern
        state = torch.randn(batch_size, dim, dtype=torch.complex64)
        state = state / torch.norm(state, dim=1, keepdim=True)

        # Extract pattern from quantum state
        pattern = torch.abs(state) ** 2

        # Validate both representations
        state_result = framework.validate_quantum_state(state)
        pattern_result = framework.validate_pattern_formation(pattern)

        # Verify consistency
        assert state_result.is_valid
        assert pattern_result.is_valid
        assert torch.allclose(
            torch.sum(pattern, dim=1), torch.ones(batch_size), rtol=1e-5
        )

    def test_geometric_pattern_coupling(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test coupling between geometric and pattern components."""
        # Generate metric tensor
        metric = torch.randn(batch_size, dim, dim)
        metric = metric @ metric.transpose(-1, -2)

        # Generate pattern compatible with metric
        pattern = torch.randn(batch_size, dim)
        pattern = pattern / torch.norm(pattern, dim=1, keepdim=True)

        # Validate geometric consistency
        metric_result = framework.validate_metric(metric)
        pattern_result = framework.validate_pattern_formation(pattern)

        # Test metric-induced evolution
        flow = framework.compute_metric_flow(metric, pattern)
        flow_result = framework.validate_flow(flow)

        assert metric_result.is_valid
        assert pattern_result.is_valid
        assert flow_result.is_valid

    def test_infrastructure_framework(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test integration between infrastructure and validation framework."""
        # Initialize infrastructure
        cpu_opt = CPUOptimizer()
        mem_mgr = MemoryManager()
        vulkan = VulkanIntegration()

        # Generate test data
        data = torch.randn(batch_size, dim)

        # Test CPU optimization
        with cpu_opt.optimize_cache():
            result = framework.validate_all(
                metric=data @ data.t(),
                quantum_state=torch.randn(batch_size, dim, dtype=torch.complex64),
                pattern=data,
            )

        assert isinstance(result, dict)
        assert all(
            component in result for component in ["geometric", "quantum", "pattern"]
        )

        # Test memory management
        with mem_mgr.optimize_memory():
            metrics = framework.aggregate_metrics(result)

        assert metrics.overall_score > 0

        # Test Vulkan acceleration if available
        if vulkan.is_available():
            gpu_result = vulkan.accelerate_validation(framework, data)
            assert torch.allclose(
                gpu_result["overall_score"], result["overall_score"], rtol=1e-4
            )

    def test_end_to_end_validation(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test complete end-to-end validation pipeline."""
        # Generate test configuration
        config = {
            "metric": torch.randn(batch_size, dim, dim),
            "quantum_state": torch.randn(batch_size, dim, dtype=torch.complex64),
            "pattern": torch.randn(batch_size, dim),
            "flow": torch.randn(batch_size, dim, dim),
            "parameters": torch.linspace(0, 1, 10),
        }

        # Normalize and prepare data
        config["metric"] = config["metric"] @ config["metric"].transpose(-1, -2)
        config["quantum_state"] = config["quantum_state"] / torch.norm(
            config["quantum_state"], dim=1, keepdim=True
        )
        config["pattern"] = config["pattern"] / torch.norm(
            config["pattern"], dim=1, keepdim=True
        )

        # Run complete validation
        result = framework.validate_complete(config)

        # Verify all components
        assert isinstance(result, dict)
        assert all(
            key in result
            for key in [
                "geometric_validation",
                "quantum_validation",
                "pattern_validation",
                "flow_validation",
                "cross_validation",
                "overall_score",
            ]
        )

        # Check consistency
        assert result["overall_score"] <= 1.0
        assert result["overall_score"] >= 0.0
        assert all(0 <= score <= 1 for score in result["cross_validation"].values())

        # Verify no conflicts
        conflicts = framework.detect_validation_conflicts(result)
        assert len(conflicts) == 0, f"Found validation conflicts: {conflicts}"

    def test_validation_stability(
        self, framework: ValidationFramework, batch_size: int, dim: int
    ):
        """Test stability of validation results under perturbations."""
        # Generate base configuration
        base_config = {
            "metric": torch.randn(batch_size, dim, dim),
            "quantum_state": torch.randn(batch_size, dim, dtype=torch.complex64),
            "pattern": torch.randn(batch_size, dim),
        }

        # Normalize base configuration
        base_config["metric"] = base_config["metric"] @ base_config["metric"].transpose(
            -1, -2
        )
        base_config["quantum_state"] = base_config["quantum_state"] / torch.norm(
            base_config["quantum_state"], dim=1, keepdim=True
        )
        base_config["pattern"] = base_config["pattern"] / torch.norm(
            base_config["pattern"], dim=1, keepdim=True
        )

        # Get base validation result
        base_result = framework.validate_complete(base_config)

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
            ] / torch.norm(perturbed_config["quantum_state"], dim=1, keepdim=True)
            perturbed_config["pattern"] = perturbed_config["pattern"] / torch.norm(
                perturbed_config["pattern"], dim=1, keepdim=True
            )

            # Validate perturbed configuration
            perturbed_result = framework.validate_complete(perturbed_config)

            # Check stability
            score_diff = abs(
                perturbed_result["overall_score"] - base_result["overall_score"]
            )
            assert (
                score_diff < 10 * scale
            ), f"Validation unstable for perturbation scale {scale}"
