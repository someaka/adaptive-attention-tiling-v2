"""Unit tests for anomaly polynomial computation in crystal scale cohomology.

Mathematical Framework:
    1. Wess-Zumino Consistency: δ_g1(A[g2]) - δ_g2(A[g1]) = A[g1, g2]
    2. Cohomological Properties: dA = 0, A ≠ dB
    3. Cocycle Condition: A[g1·g2] = A[g1] + A[g2]
"""

from pathlib import Path
from typing import List, Callable
import yaml
import numpy as np
import math

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
        """Create U(1) symmetry action with given phase.
        
        The action is normalized to ensure proper group composition:
        g1 · g2 = exp(i(θ1 + θ2))
        """
        def action(x: Tensor) -> Tensor:
            # Normalize phase to [0, 2π)
            phase_norm = phase % (2 * torch.pi)
            # Apply U(1) transformation
            return torch.exp(1j * phase_norm * x).to(dtype)
        return action

    def test_anomaly_polynomial(self, scale_system: ScaleCohomology, dtype: torch.dtype):
        """Test basic anomaly polynomial computation."""
        # Create test symmetry actions
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)

        with memory_efficient_computation("basic_test"), torch.no_grad():
            # Test basic anomaly computation
            anomaly = scale_system.anomaly_polynomial(g1)
            assert anomaly is not None, "Should compute anomaly"
            assert hasattr(anomaly[0], 'winding_number'), "Should compute winding number"
            assert hasattr(anomaly[0], 'is_consistent'), "Should check consistency"

    def test_wess_zumino_consistency(self, scale_system: ScaleCohomology, dtype: torch.dtype):
        """Test Wess-Zumino consistency with batched processing."""
        batch_size = 8
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)

        with memory_efficient_computation("wz_test"), torch.no_grad():
            # Generate test points
            x = torch.linspace(0, 2*torch.pi, batch_size, dtype=torch.float32).to(dtype)
            
            # Compute actions
            g1x = g1(x)
            g2x = g2(x)
            g1g2x = g1(g2x)
            g2g1x = g2(g1x)
            
            # Compute winding numbers
            w1 = compute_winding_number(g1x)
            w2 = compute_winding_number(g2x)
            w12 = compute_winding_number(g1g2x)
            w21 = compute_winding_number(g2g1x)
            
            # Check Wess-Zumino consistency
            commutator = torch.abs(w12 - w21)
            assert commutator < 1e-5, f"Wess-Zumino consistency violated: {commutator}"

    def test_cocycle_condition(self, scale_system: ScaleCohomology, dtype: torch.dtype):
        """Test cocycle condition with memory efficiency."""
        # Create test symmetry actions with smaller dimension
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)

        def print_function_values(g, x_points):
            """Print function values at test points."""
            print("\nFunction values at test points:")
            for i, x in enumerate(x_points):
                y = g(x)
                print(f"  x[{i}] = {x:.4f} -> y = {y}")

        def print_composition_values(g1, g2, x_points):
            """Print composition values at test points."""
            print("\nComposition values at test points:")
            for i, x in enumerate(x_points):
                y2 = g2(x)
                y1 = g1(y2)
                print(f"  x[{i}] = {x:.4f} -> g2(x) = {y2} -> g1(g2(x)) = {y1}")

        def compare_coefficients(sum_coeffs: torch.Tensor, composed_coeffs: torch.Tensor) -> bool:
            """Compare coefficients with magnitude sorting and phase alignment."""
            print("\n=== Comparing Coefficients ===")
            print(f"Sum coefficients: {sum_coeffs}")
            print(f"Composed coefficients: {composed_coeffs}")

            # Handle 0-d tensors
            if sum_coeffs.dim() == 0 or composed_coeffs.dim() == 0:
                print("Converting 0-d tensors to 1-d")
                sum_coeffs = sum_coeffs.unsqueeze(0)
                composed_coeffs = composed_coeffs.unsqueeze(0)

            # Get magnitudes and phases
            sum_mags = sum_coeffs.abs()
            comp_mags = composed_coeffs.abs()

            # Normalize before comparing magnitudes
            sum_norm = torch.norm(sum_coeffs)
            comp_norm = torch.norm(composed_coeffs)

            if sum_norm > 1e-6:
                sum_coeffs = sum_coeffs / sum_norm
                sum_mags = sum_mags / sum_norm

            if comp_norm > 1e-6:
                composed_coeffs = composed_coeffs / comp_norm
                comp_mags = comp_mags / comp_norm

            # Find significant terms using relative threshold
            max_mag = torch.max(torch.max(sum_mags), torch.max(comp_mags))
            sig_threshold = max_mag * 0.1  # 10% of max magnitude
            
            sum_sig_mask = sum_mags > sig_threshold
            comp_sig_mask = comp_mags > sig_threshold

            # Get significant coefficients
            sum_sig = sum_coeffs[sum_sig_mask]
            comp_sig = composed_coeffs[comp_sig_mask]

            # If either has no significant terms, they should match
            if len(sum_sig) == 0 and len(comp_sig) == 0:
                return True

            # Get magnitudes and phases of significant coefficients
            sum_mags_sig = sum_sig.abs()
            comp_mags_sig = comp_sig.abs()
            sum_phases_sig = torch.angle(sum_sig)
            comp_phases_sig = torch.angle(comp_sig)

            print("\nSignificant coefficients after normalization:")
            print(f"Sum: {sum_sig}")
            print(f"Composed: {comp_sig}")
            print(f"\nMagnitudes:")
            print(f"Sum: {sum_mags_sig}")
            print(f"Composed: {comp_mags_sig}")
            print(f"\nPhases:")
            print(f"Sum: {sum_phases_sig}")
            print(f"Composed: {comp_phases_sig}")

            # Sort by magnitude
            sum_sorted_idx = torch.argsort(sum_mags_sig, descending=True)
            comp_sorted_idx = torch.argsort(comp_mags_sig, descending=True)

            sum_mags_sorted = sum_mags_sig[sum_sorted_idx]
            comp_mags_sorted = comp_mags_sig[comp_sorted_idx]
            sum_phases_sorted = sum_phases_sig[sum_sorted_idx]
            comp_phases_sorted = comp_phases_sig[comp_sorted_idx]

            print("\nSorted values:")
            print(f"Sum magnitudes: {sum_mags_sorted}")
            print(f"Composed magnitudes: {comp_mags_sorted}")
            print(f"Sum phases: {sum_phases_sorted}")
            print(f"Composed phases: {comp_phases_sorted}")

            # Group coefficients by similar magnitudes
            def group_by_magnitude(mags, phases, base_threshold=0.1):
                """Group coefficients by similar magnitudes with dynamic thresholding."""
                groups = []
                used = set()

                # Sort by magnitude descending for stable grouping
                sorted_idx = torch.argsort(mags, descending=True)
                sorted_mags = mags[sorted_idx]
                sorted_phases = phases[sorted_idx]

                # Compute magnitude differences
                mag_diffs = torch.zeros_like(sorted_mags)
                for i in range(len(sorted_mags)-1):
                    mag_diffs[i] = abs(sorted_mags[i+1] / sorted_mags[i] - 1.0)

                # Find natural breaks in magnitude differences
                if len(mag_diffs[mag_diffs > 0]) > 0:
                    mean_diff = torch.mean(mag_diffs[mag_diffs > 0]).item()
                    std_diff = torch.std(mag_diffs[mag_diffs > 0]).item()
                    dynamic_threshold = min(base_threshold, mean_diff + std_diff)
                else:
                    dynamic_threshold = base_threshold

                # Group using dynamic threshold
                current_group = []
                current_mag = None

                for i, mag in enumerate(sorted_mags):
                    if i in used:
                        continue

                    if current_mag is None:
                        current_mag = mag
                        current_group = [sorted_idx[i]]
                        used.add(i)
                    else:
                        ratio = mag / current_mag
                        if abs(ratio - 1.0) < dynamic_threshold:
                            current_group.append(sorted_idx[i])
                            used.add(i)
                        else:
                            if len(current_group) > 0:
                                groups.append(current_group)
                            current_group = [sorted_idx[i]]
                            current_mag = mag
                            used.add(i)

                if len(current_group) > 0:
                    groups.append(current_group)

                return groups

            sum_groups = group_by_magnitude(sum_mags_sorted, sum_phases_sorted)
            comp_groups = group_by_magnitude(comp_mags_sorted, comp_phases_sorted)

            # Check if we have compatible groups
            if len(sum_groups) < len(comp_groups):
                # Try to merge adjacent sum groups
                merged_sum_groups = []
                i = 0
                while i < len(sum_groups):
                    if i < len(sum_groups) - 1:
                        # Check if merging would match a composed group size
                        merged_size = len(sum_groups[i]) + len(sum_groups[i+1])
                        if any(len(g) == merged_size for g in comp_groups):
                            merged_sum_groups.append(sum_groups[i] + sum_groups[i+1])
                            i += 2
                            continue
                    merged_sum_groups.append(sum_groups[i])
                    i += 1
                sum_groups = merged_sum_groups
            elif len(comp_groups) < len(sum_groups):
                # Try to merge adjacent composed groups
                merged_comp_groups = []
                i = 0
                while i < len(comp_groups):
                    if i < len(comp_groups) - 1:
                        # Check if merging would match a sum group size
                        merged_size = len(comp_groups[i]) + len(comp_groups[i+1])
                        if any(len(g) == merged_size for g in sum_groups):
                            merged_comp_groups.append(comp_groups[i] + comp_groups[i+1])
                            i += 2
                            continue
                    merged_comp_groups.append(comp_groups[i])
                    i += 1
                comp_groups = merged_comp_groups

            if len(sum_groups) != len(comp_groups):
                print(f"Different number of magnitude groups after merging: sum={len(sum_groups)}, composed={len(comp_groups)}")
                return False

            # For each group, check if phases match after potential rotation
            for sum_group, comp_group in zip(sum_groups, comp_groups):
                if len(sum_group) != len(comp_group):
                    print(f"Group size mismatch: sum={len(sum_group)}, composed={len(comp_group)}")
                    return False

                # Get phases for this group
                sum_group_phases = sum_phases_sorted[sum_group]
                comp_group_phases = comp_phases_sorted[comp_group]

                # Try different rotations
                found_match = False
                for phase_shift in [0, torch.pi/4, torch.pi/2, 3*torch.pi/4, torch.pi, 5*torch.pi/4, 3*torch.pi/2, 7*torch.pi/4]:
                    # Shift composed phases
                    shifted_phases = (comp_group_phases + phase_shift) % (2 * torch.pi)

                    # Sort both sets of phases
                    sum_sorted = torch.sort(sum_group_phases % (2 * torch.pi))[0]
                    comp_sorted = torch.sort(shifted_phases)[0]

                    # Check if phases match after sorting
                    if torch.allclose(sum_sorted, comp_sorted, rtol=1e-2, atol=1e-2):
                        found_match = True
                        break

                if not found_match:
                    print(f"No matching phase alignment found for group")
                    return False

            return True

        with memory_efficient_computation("cocycle_test"), torch.no_grad():
            # Create test points
            test_points = torch.linspace(0, 2*torch.pi, 5, dtype=dtype)
            
            print("\n=== Testing Individual Functions ===")
            print("\nFunction g1:")
            print_function_values(g1, test_points)
            print("\nFunction g2:")
            print_function_values(g2, test_points)
            print("\nComposition g1∘g2:")
            print_composition_values(g1, g2, test_points)

            print("\n=== Computing Anomalies ===")
            print("\nComputing a1...")
            # Track intermediate values for g1
            print("Computing coefficients for g1...")
            a1 = scale_system.anomaly_polynomial(g1)
            print("\nAnomaly a1:")
            for i, poly in enumerate(a1):
                print(f"Degree {i+1}: {poly.coefficients}")
                print(f"Type: {poly.type}")
                if hasattr(poly, 'winding_number'):
                    print(f"Winding number: {poly.winding_number}")

            print("\nComputing a2...")
            # Track intermediate values for g2
            print("Computing coefficients for g2...")
            a2 = scale_system.anomaly_polynomial(g2)
            print("\nAnomaly a2:")
            for i, poly in enumerate(a2):
                print(f"Degree {i+1}: {poly.coefficients}")
                print(f"Type: {poly.type}")
                if hasattr(poly, 'winding_number'):
                    print(f"Winding number: {poly.winding_number}")

            print("\nComputing composed anomaly...")
            # Track intermediate values for composition
            print("Computing coefficients for composition...")
            composed = scale_system.anomaly_polynomial(lambda x: g1(g2(x)))
            print("\nComposed anomaly:")
            for i, poly in enumerate(composed):
                print(f"Degree {i+1}: {poly.coefficients}")
                print(f"Type: {poly.type}")
                if hasattr(poly, 'winding_number'):
                    print(f"Winding number: {poly.winding_number}")

            # Check cocycle condition
            for i, (c, a1p, a2p) in enumerate(zip(composed, a1, a2)):
                print(f"\n=== Checking degree {i+1} ===")
                print(f"Polynomial types: c={c.type}, a1={a1p.type}, a2={a2p.type}")
                
                # Use operadic composition to combine anomalies
                operad_composed = scale_system.anomaly_detector.compose_anomalies([a1p], [a2p])[0]
                print("\nOperadic composition result:")
                print(f"Type: {operad_composed.type}")
                print(f"Coefficients: {operad_composed.coefficients}")
                print(f"Winding number: {operad_composed.winding_number}")
                print(f"Berry phase: {operad_composed.berry_phase}")
                print(f"Is consistent: {operad_composed.is_consistent}")
                
                # For U(1) symmetries, compare winding numbers
                if c.type == "U1" and a1p.type == "U1" and a2p.type == "U1":
                    print("U(1) symmetry detected")
                    # The winding number of the composition should be the sum
                    if c.winding_number is not None and a1p.winding_number is not None and a2p.winding_number is not None:
                        print(f"Winding numbers:")
                        print(f"  a1: {a1p.winding_number}")
                        print(f"  a2: {a2p.winding_number}")
                        print(f"  Composed: {c.winding_number}")
                        print(f"  Sum: {a1p.winding_number + a2p.winding_number}")
                        print(f"  Difference: {abs(c.winding_number - (a1p.winding_number + a2p.winding_number))}")
                        
                        assert abs(c.winding_number - (a1p.winding_number + a2p.winding_number)) < 1e-5, \
                            "Cocycle condition violated for U(1) symmetry"
                    else:
                        print("Some winding numbers are None, falling back to coefficient comparison")
                        print(f"Winding numbers (None check):")
                        print(f"  a1: {a1p.winding_number}")
                        print(f"  a2: {a2p.winding_number}")
                        print(f"  Composed: {c.winding_number}")
                        
                        # If any winding numbers are None, fall back to coefficient comparison
                        sum_coeffs = operad_composed.coefficients
                        composed_coeffs = c.coefficients
                        
                        print("\nCoefficient comparison for None winding numbers:")
                        print(f"Operadic composed coefficients: {sum_coeffs}")
                        print(f"Direct composed coefficients: {composed_coeffs}")
                        
                        assert compare_coefficients(sum_coeffs, composed_coeffs), \
                            "Cocycle condition violated"
                else:
                    print("Non-U(1) symmetry")
                    # For non-U(1) symmetries, compare coefficients
                    sum_coeffs = operad_composed.coefficients
                    composed_coeffs = c.coefficients
                    
                    # Ensure tensors have the same size
                    if sum_coeffs.size(0) > composed_coeffs.size(0):
                        sum_coeffs = sum_coeffs[:composed_coeffs.size(0)]
                    elif composed_coeffs.size(0) > sum_coeffs.size(0):
                        composed_coeffs = composed_coeffs[:sum_coeffs.size(0)]

                    print("\nCoefficient comparison for non-U(1):")
                    print(f"Operadic composed coefficients: {sum_coeffs}")
                    print(f"Direct composed coefficients: {composed_coeffs}")
                    print(f"Difference: {sum_coeffs - composed_coeffs}")
                    
                    assert compare_coefficients(sum_coeffs, composed_coeffs), \
                        "Cocycle condition violated"

    def test_memory_management(self, space_dim: int, dtype: torch.dtype, test_config: dict):
        """Test memory efficiency of anomaly detection."""
        with memory_efficient_computation("memory_test"):
            detector = AnomalyDetector(
                dim=space_dim,
                max_degree=test_config.get("max_polynomial_degree", 4),
                dtype=dtype
            )
            
            # Test with batched processing
            batch_size = 8
            states = torch.randn(batch_size, space_dim, dtype=dtype)
            
            # Process batch
            anomalies = detector.detect_anomalies(states)
            
            # Verify results
            assert len(anomalies) > 0, "Should detect anomalies"
            for anomaly in anomalies:
                assert hasattr(anomaly, 'winding_number'), "Missing winding number"
                assert hasattr(anomaly, 'is_consistent'), "Missing consistency check"
                assert anomaly.coefficients.shape[-1] <= detector.max_degree + 1, \
                    "Invalid coefficient dimension" 

    def test_coefficient_grouping(self, scale_system: ScaleCohomology, dtype: torch.dtype):
        """Test coefficient grouping logic in isolation."""
        # Test case 1: Clearly grouped coefficients
        coeffs1 = torch.tensor([1.0+0j, 1.0+0.1j, 0.5+0j, 0.48+0.1j], dtype=dtype)
        
        def group_by_magnitude(coeffs):
            mags = coeffs.abs()
            sorted_idx = torch.argsort(mags, descending=True)
            sorted_mags = mags[sorted_idx]
            
            groups = []
            current_group = [sorted_idx[0]]
            current_mag = sorted_mags[0]
            
            for i in range(1, len(sorted_mags)):
                if abs(sorted_mags[i] / current_mag - 1.0) < 0.1:
                    current_group.append(sorted_idx[i])
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [sorted_idx[i]]
                    current_mag = sorted_mags[i]
            
            if current_group:
                groups.append(current_group)
            
            return groups
        
        groups1 = group_by_magnitude(coeffs1)
        assert len(groups1) == 2, "Should find 2 magnitude groups"
        
        # Test case 2: Coefficients with phase relationships
        coeffs2 = torch.tensor([1.0+0j, -1.0+0j, 0.5+0.5j, -0.5-0.5j], dtype=dtype)
        groups2 = group_by_magnitude(coeffs2)
        assert len(groups2) == 2, "Should find 2 magnitude groups with phase relationships"
        
        # Test case 3: Nearly identical magnitudes
        coeffs3 = torch.tensor([1.0+0j, 0.99+0.01j, 0.98+0.02j], dtype=dtype)
        groups3 = group_by_magnitude(coeffs3)
        assert len(groups3) == 1, "Should group nearly identical magnitudes"

    def test_projection_consistency(self, scale_system: ScaleCohomology, dtype: torch.dtype):
        """Test that projection to consistent space preserves key properties."""
        # Create test coefficients with two distinct magnitude groups
        coeffs = torch.tensor([1.0+0j, 0.5+0.5j, -0.5-0.5j], dtype=dtype)
        print("\nOriginal coefficients:", coeffs)
        print("Original magnitudes:", coeffs.abs())
        
        # The second and third coefficients should form a magnitude group
        similar_mags = coeffs.abs()[1:]  # [0.7071, 0.7071]
        similar_ratio = similar_mags[1] / similar_mags[0]
        print("Ratio between similar magnitudes:", similar_ratio)
        assert abs(similar_ratio - 1.0) < 0.1, "Test input should have similar magnitudes"
        
        # Project coefficients
        projected = scale_system.anomaly_detector._project_to_consistent_space(coeffs, degree=2)
        print("\nProjected coefficients:", projected)
        print("Projected magnitudes:", projected.abs())
        
        # Check properties
        assert torch.allclose(torch.norm(projected), torch.tensor(1.0)), "Should preserve normalization"
        
        # Check that coefficients that had similar magnitudes still have similar magnitudes
        projected_similar = projected.abs()[1:]  # Should correspond to original similar group
        projected_ratio = projected_similar[1] / projected_similar[0]
        print("\nProjected ratio between originally similar magnitudes:", projected_ratio)
        assert abs(projected_ratio - 1.0) < 0.1, "Should preserve similarity between magnitudes"
        
        # Check phase relationships
        phases = torch.angle(projected)
        phase_diffs = torch.diff(phases)
        phase_diffs = (phase_diffs + torch.pi) % (2 * torch.pi) - torch.pi
        print("\nPhase differences:", phase_diffs)
        assert torch.allclose(phase_diffs, phase_diffs[0], rtol=1e-2), "Should preserve phase relationships"

    def test_cocycle_components(self, scale_system: ScaleCohomology, dtype: torch.dtype):
        """Test individual components of the cocycle condition."""
        g1 = self._create_u1_action(1.0, dtype)
        g2 = self._create_u1_action(2.0, dtype)
        
        # Compute individual anomalies
        a1 = scale_system.anomaly_polynomial(g1)
        a2 = scale_system.anomaly_polynomial(g2)
        composed = scale_system.anomaly_polynomial(lambda x: g1(g2(x)))
        
        # Test 1: Check that individual anomalies are consistent
        for poly in a1 + a2:
            assert poly.is_consistent, "Individual anomalies should be consistent"
            
        # Test 2: Check winding number additivity for U(1) components
        for c, a1p, a2p in zip(composed, a1, a2):
            if c.type == "U1" and a1p.type == "U1" and a2p.type == "U1":
                if (c.winding_number is not None and 
                    a1p.winding_number is not None and 
                    a2p.winding_number is not None):
                    sum_winding = float(a1p.winding_number) + float(a2p.winding_number)
                    composed_winding = float(c.winding_number)
                    assert abs(composed_winding - sum_winding) < 1e-5, \
                        f"Winding number mismatch: {composed_winding} vs {sum_winding}"
                        
        # Test 3: Check coefficient structure preservation
        for c, a1p, a2p in zip(composed, a1, a2):
            # Get significant coefficients
            sum_coeffs = a1p.coefficients + a2p.coefficients
            sum_sig = sum_coeffs[sum_coeffs.abs() > 0.1]
            comp_sig = c.coefficients[c.coefficients.abs() > 0.1]
            
            # Check that number of significant terms matches
            assert len(sum_sig) == len(comp_sig), \
                f"Mismatch in number of significant terms: {len(sum_sig)} vs {len(comp_sig)}" 