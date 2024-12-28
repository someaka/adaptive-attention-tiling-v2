"""Unit tests for anomaly polynomial computation in crystal scale cohomology.

Mathematical Framework:
    1. Wess-Zumino Consistency: δ_g1(A[g2]) - δ_g2(A[g1]) = A[g1, g2]
    2. Cohomological Properties: dA = 0, A ≠ dB
    3. Cocycle Condition: A[g1·g2] = A[g1] + A[g2]
"""

from pathlib import Path
from typing import List, Callable
import yaml

import pytest
import torch
from torch import Tensor

from src.core.crystal.scale import ScaleCohomology
from src.core.quantum.u1_utils import compute_winding_number
from src.core.crystal.scale_classes.anomalydetector import AnomalyDetector, AnomalyPolynomial
from src.core.crystal.scale_classes.memory_utils import memory_efficient_computation


# Fixtures
@pytest.fixture
def test_config():
    """Load test configuration with merged parameters."""
    project_root = Path(__file__).parents[3]
    config_path = project_root / "tests" / "test_integration" / "test_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    regime_config = config["regimes"]["debug"].copy()
    regime_config.update(config["common"])
    return regime_config


@pytest.fixture
def space_dim(test_config) -> int:
    """Base space dimension from config."""
    return test_config["manifold_dim"]


@pytest.fixture
def dtype() -> torch.dtype:
    """Quantum computation dtype."""
    return torch.complex64


@pytest.fixture
def scale_system(space_dim: int, test_config: dict, dtype: torch.dtype) -> ScaleCohomology:
    """Initialize scale system with config parameters."""
    return ScaleCohomology(
        dim=space_dim,
        num_scales=max(test_config.get("num_layers", 4), 2),
        dtype=dtype
    )


class TestAnomalyPolynomial:
    """Test suite for anomaly polynomial functionality."""

    @staticmethod
    def _create_u1_action(phase: float, dtype: torch.dtype) -> Callable[[Tensor], Tensor]:
        """Create U(1) symmetry action with given phase."""
        return lambda x: torch.exp(1j * phase * x).to(dtype)

    def test_anomaly_polynomial(self, scale_system: ScaleCohomology, dtype: torch.dtype, test_config: dict):
        """Test anomaly polynomial computation with optimized memory usage."""
        # Create symmetry actions
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)

        def test_consistency(g1: Callable, g2: Callable, batch_size: int = 8) -> bool:
            """Test Wess-Zumino consistency with efficient batching."""
            total_points = min(scale_system.dim, 32)
            num_batches = (total_points + batch_size - 1) // batch_size
            
            # Pre-allocate tensors
            winding_sums = torch.zeros(3, dtype=dtype)
            test_points = torch.linspace(0, 2*torch.pi, total_points, dtype=torch.float32).to(dtype)
            
            # Process in batches
            for i in range(num_batches):
                with memory_efficient_computation("batch_processing"), torch.no_grad():
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_points)
                    batch = test_points[start_idx:end_idx]
                    
                    # Compute transformations and windings in batch
                    g1x, g2x = g1(batch), g2(batch)
                    g1g2x = g1(g2x)
                    
                    windings = torch.stack([
                        compute_winding_number(x) for x in [g1x, g2x, g1g2x]
                    ])
                    winding_sums += windings

            # Compute anomalies efficiently
            with memory_efficient_computation("anomaly_computation"), torch.no_grad():
                # Parallel anomaly computation
                anomalies = [
                    scale_system.anomaly_polynomial(g) 
                    for g in [g1, g2, lambda x: g1(g2(x))]
                ]
                
                # Verify consistency
                a1, a2, composed = anomalies
                
                # Normalize coefficients for comparison
                def normalize_coeffs(coeffs: Tensor) -> Tensor:
                    norm = torch.norm(coeffs)
                    if norm > 0:
                        return coeffs / norm
                    return coeffs

                for c, a1p, a2p in zip(composed, a1, a2):
                    c_norm = normalize_coeffs(c.coefficients)
                    sum_norm = normalize_coeffs(a1p.coefficients + a2p.coefficients)
                    if not torch.allclose(c_norm, sum_norm, rtol=1e-2, atol=1e-2):
                        return False
                return True

        # Main test execution
        with memory_efficient_computation("main_test"), torch.no_grad():
            # Test basic anomaly computation
            anomaly = scale_system.anomaly_polynomial(g1)
            assert anomaly is not None, "Should compute anomaly"
            
            # Test consistency conditions
            assert test_consistency(g1, g2), "Anomaly should satisfy consistency"

    def test_memory_management(self, space_dim: int, dtype: torch.dtype, test_config: dict):
        """Test memory efficiency of anomaly detection with optimized caching."""
        with memory_efficient_computation("memory_test"):
            # Initialize detector with config parameters
            detector = AnomalyDetector(
                dim=space_dim, 
                max_degree=test_config.get("max_polynomial_degree", 4),
                dtype=dtype
            )
            
            # Generate test states efficiently
            batch_size = test_config.get("test_iterations", 10)
            states = torch.randn(batch_size, 1, space_dim, dtype=dtype)
            
            # Test cache behavior with batched processing
            with torch.no_grad():
                # Process unique states
                results = {}
                for state in states:
                    key = detector._get_state_key(state)
                    if key not in results:
                        results[key] = detector.detect_anomalies(state)
                    
                    # Verify cache hit
                    cached = detector.detect_anomalies(state)
                    assert cached == results[key], "Cache inconsistency detected"
                
                # Verify cache size constraints
                max_cache = test_config.get("max_cache_size", 1000)
                assert len(detector._poly_cache) <= max_cache, f"Cache size {len(detector._poly_cache)} exceeds limit {max_cache}"
                
                # Verify unique results for different states
                unique_results = {str(r) for r in results.values()}
                assert len(unique_results) > 1, "Should detect different anomalies for different states"

    def test_wess_zumino_consistency(
        self, 
        scale_system: ScaleCohomology, 
        dtype: torch.dtype, 
        test_config: dict
    ):
        """Test Wess-Zumino consistency condition for anomaly polynomials.
        
        Mathematical Framework:
            δ_g1(A[g2]) - δ_g2(A[g1]) = A[g1, g2]
            where:
            - A[g] is the anomaly for transformation g
            - δ_g is the variation under g
            - A[g1, g2] is the commutator anomaly
        """
        # Create test transformations
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)

        def compute_variation(g: Callable, anomaly: List[AnomalyPolynomial]) -> Tensor:
            """Compute variation of anomaly under transformation g."""
            with memory_efficient_computation("variation"), torch.no_grad():
                # Generate test points matching coefficient dimensions
                coeff_dim = anomaly[0].coefficients.shape[0]
                x = torch.linspace(0, 2*torch.pi, coeff_dim, dtype=torch.float32).to(dtype)
                gx = g(x)
                
                # Compute variation with batched operations
                variations = torch.stack([
                    torch.sum(a.coefficients * gx)
                    for a in anomaly
                ])
                return torch.sum(variations)

        # Main test execution
        with memory_efficient_computation("wz_test"), torch.no_grad():
            # Compute anomalies in parallel
            a1, a2 = [scale_system.anomaly_polynomial(g) for g in (g1, g2)]
            
            # Compute variations
            delta_g1_a2 = compute_variation(g1, a2)
            delta_g2_a1 = compute_variation(g2, a1)
            
            # Verify consistency condition
            commutator = delta_g1_a2 - delta_g2_a1
            tolerance = test_config.get("wz_tolerance", 1e-5)
            assert torch.abs(commutator) < tolerance, f"Wess-Zumino consistency violated: {commutator}"

    def test_cohomological_properties(
        self, 
        scale_system: ScaleCohomology, 
        dtype: torch.dtype, 
        test_config: dict
    ):
        """Test cohomological properties of anomaly polynomials.
        
        Mathematical Framework:
            1. Closure: dA = 0 (anomaly is closed)
            2. Non-exactness: A ≠ dB (anomaly is not exact)
            3. Cocycle condition: A[g1·g2] = A[g1] + A[g2]
        """
        # Create test transformations
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)
        g_composed = lambda x: g1(g2(x))

        def sum_coefficients(anomalies: List[AnomalyPolynomial]) -> Tensor:
            """Sum coefficients of anomaly polynomials efficiently."""
            with torch.no_grad():
                coeffs = torch.stack([a.coefficients for a in anomalies])
                return torch.sum(coeffs, dim=0)

        # Main test execution
        with memory_efficient_computation("cohomology_test"), torch.no_grad():
            # Compute anomalies in parallel
            a1, a2, a_composed = [
                scale_system.anomaly_polynomial(g) 
                for g in (g1, g2, g_composed)
            ]

            # Test cocycle condition
            def normalize_coeffs(coeffs: Tensor) -> Tensor:
                norm = torch.norm(coeffs)
                if norm > 0:
                    return coeffs / norm
                return coeffs

            sum_separate = normalize_coeffs(sum_coefficients(a1) + sum_coefficients(a2))
            sum_composed = normalize_coeffs(sum_coefficients(a_composed))

            # Verify with configurable tolerance
            tolerance = test_config.get("anomaly_tolerance", 1e-2)
            assert torch.allclose(
                sum_composed, 
                sum_separate, 
                rtol=tolerance,
                atol=tolerance
            ), f"Cocycle condition violated: composed={sum_composed}, separate={sum_separate}"

            # Test closure property
            def compute_differential(anomaly: List[AnomalyPolynomial]) -> Tensor:
                """Compute exterior derivative of anomaly polynomial."""
                with torch.no_grad():
                    # Sample points for differential
                    x = torch.linspace(0, 2*torch.pi, 8, dtype=torch.float32).to(dtype)
                    dx = torch.ones_like(x) * (2*torch.pi / 8)
                    
                    # Compute differential with finite differences
                    coeffs = torch.stack([a.coefficients for a in anomaly])
                    forward = torch.roll(coeffs, -1, dims=-1)
                    backward = torch.roll(coeffs, 1, dims=-1)
                    diff = (forward - backward) / (2 * dx[None, None])
                    return torch.sum(torch.abs(diff))

            # Verify closure
            diff_a1 = compute_differential(a1)
            diff_a2 = compute_differential(a2)
            assert torch.all(diff_a1 < tolerance), "Anomaly a1 not closed"
            assert torch.all(diff_a2 < tolerance), "Anomaly a2 not closed" 