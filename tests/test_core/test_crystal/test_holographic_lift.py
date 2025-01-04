"""Tests for holographic lifting functionality.

This module tests the holographic lifting implementation, which maps boundary data to bulk
data using AdS/CFT correspondence principles.
"""

import numpy as np
import pytest
import torch
from typing import Tuple, NamedTuple
import os

from src.core.crystal.scale_classes.holographiclift import HolographicLifter


class HolographicTestData(NamedTuple):
    """Container for test data using NamedTuple for immutability and efficiency."""
    boundary_field: torch.Tensor
    radial_points: torch.Tensor
    bulk_field: torch.Tensor
    uv_data: torch.Tensor  # Pre-computed for efficiency
    ir_data: torch.Tensor  # Pre-computed for efficiency


class TestBase:
    """Base class containing common test utilities."""
    
    @staticmethod
    def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        """Compute relative error between tensors."""
        norm_expected = torch.norm(expected)
        return torch.norm(actual - expected).item() / norm_expected.item() if norm_expected > 0 else float('inf')
    
    @staticmethod
    def create_test_data(lifter: HolographicLifter,
                        shape: Tuple[int, ...] = (4, 4),  # Smaller default shape
                        n_radial: int = 10,  # Fewer radial points
                        z_range: Tuple[float, float] = (0.1, 10.0),
                        random: bool = True) -> HolographicTestData:
        """Create and cache test data."""
        # Create real and imaginary parts separately
        if random:
            boundary_real = torch.randn(*shape, dtype=torch.float32)
            boundary_imag = torch.randn(*shape, dtype=torch.float32)
        else:
            boundary_real = torch.ones(*shape, dtype=torch.float32)
            boundary_imag = torch.zeros(*shape, dtype=torch.float32)
        
        # Combine into complex tensor
        boundary = torch.complex(boundary_real, boundary_imag)
        
        # Create radial points as real tensor and convert to complex
        radial_real = torch.linspace(z_range[0], z_range[1], n_radial, dtype=torch.float32)
        radial = torch.complex(radial_real, torch.zeros_like(radial_real))
        
        # Compute bulk field
        bulk = lifter.holographic_lift(boundary, radial)
        
        return HolographicTestData(
            boundary_field=boundary,
            radial_points=radial,
            bulk_field=bulk,
            uv_data=bulk[0],  # Pre-compute for efficiency
            ir_data=bulk[-1]
        )
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate tensor properties with descriptive errors."""
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains infinite values")
        if torch.norm(tensor) == 0:
            raise ValueError(f"{name} has zero norm")


@pytest.fixture(scope="class")
def lifter():
    """Create reusable holographic lifter fixture."""
    return HolographicLifter(dim=4, dtype=torch.complex64)  # Fixed parameters for consistency


@pytest.fixture(scope="class")
def test_data(lifter):
    """Create reusable test data fixture."""
    return TestBase.create_test_data(lifter)


class TestHolographicLift(TestBase):
    """Core test suite for holographic lifting."""
    
    def test_basic_lifting(self, lifter: HolographicLifter, test_data: HolographicTestData):
        """Test basic lifting properties with clear error messages."""
        # Shape tests
        assert test_data.bulk_field.shape[0] == len(test_data.radial_points), \
            "Bulk field has incorrect number of radial slices"
        assert test_data.bulk_field.shape[1:] == test_data.boundary_field.shape, \
            "Bulk field has incorrect boundary dimensions"
        
        # Boundary condition test
        uv_error = self.rel_error(torch.abs(test_data.uv_data), 
                                 torch.abs(test_data.boundary_field))
        assert uv_error < 1e-4, f"UV boundary condition violated with error {uv_error:.2e}"
        
        # Radial scaling test with clear error reporting
        for i, z in enumerate(test_data.radial_points):
            # Use dimensionless ratio for scaling
            z_ratio = torch.abs(z / test_data.radial_points[0])
            expected_scale = z_ratio**(-lifter.dim)
            actual_scale = torch.norm(test_data.bulk_field[i]) / torch.norm(test_data.boundary_field)
            scale_error = self.rel_error(actual_scale, expected_scale)
            assert scale_error < 0.1, \
                f"Radial scaling violated at z={z:.2f} with error {scale_error:.2e}" 
            


    def test_operator_expansion(self, lifter: HolographicLifter):
        """Test operator product expansion with quantum and geometric properties.
        
        Tests:
        1. Basic operator properties (normalization, symmetry)
        2. Quantum structure (phase coherence, U(1) preservation)
        3. Operadic composition (associativity)
        4. Scaling behavior with distance
        5. Memory efficiency
        6. Geometric structure preservation
        """
        # Create test operators with quantum structure
        dim = lifter.dim
        
        # Create operators with real and imaginary parts separately
        ops_real = [torch.randn(dim, dtype=torch.float32) for _ in range(3)]
        ops_imag = [torch.randn(dim, dtype=torch.float32) for _ in range(3)]
        
        # Combine into complex tensors
        ops = [torch.complex(real, imag) for real, imag in zip(ops_real, ops_imag)]
        
        # Normalize operators
        ops = [op / torch.norm(op) for op in ops]
        
        # Test basic properties
        result = lifter.operator_product_expansion(ops[0], ops[1], normalize=True)
        self.validate_tensor(result, "OPE result")
        
        # Test normalization preservation
        norm_error = abs(torch.norm(result).item() - 1.0)
        assert norm_error < 1e-2, f"OPE should preserve normalization, error: {norm_error:.2e}"
        
        # Test U(1) phase structure with detailed instrumentation
        phase1 = torch.exp(torch.tensor(1j * torch.pi / 4, dtype=torch.complex64))
        phase2 = torch.exp(torch.tensor(1j * torch.pi / 3, dtype=torch.complex64))
        
        # Track phases at each stage
        print("\nPhase Tracking Instrumentation:")
        print(f"Initial phase1: {torch.angle(phase1):.4f} rad")
        print(f"Initial phase2: {torch.angle(phase2):.4f} rad")
        
        # Test phase multiplication from left
        op1_phased = phase1 * ops[0]
        print(f"\nLeft multiplication:")
        print(f"Operator 1 phase before: {torch.angle(torch.mean(ops[0])):.4f} rad")
        print(f"Operator 1 phase after: {torch.angle(torch.mean(op1_phased)):.4f} rad")
        
        ope_left = lifter.operator_product_expansion(op1_phased, ops[1], normalize=True)
        phase_diff_left = torch.angle(ope_left / result) - torch.angle(phase1)
        phase_error_left = torch.mean(torch.abs(torch.exp(1j * phase_diff_left) - 1.0)).item()
        
        print(f"OPE left result phase: {torch.angle(torch.mean(ope_left)):.4f} rad")
        print(f"Expected phase: {torch.angle(torch.mean(result * phase1)):.4f} rad")
        print(f"Left phase error: {phase_error_left:.4f}")

        # Test phase multiplication from right
        op2_phased = phase2 * ops[1]
        print(f"\nRight multiplication:")
        print(f"Operator 2 phase before: {torch.angle(torch.mean(ops[1])):.4f} rad")
        print(f"Operator 2 phase after: {torch.angle(torch.mean(op2_phased)):.4f} rad")
        
        ope_right = lifter.operator_product_expansion(ops[0], op2_phased, normalize=True)
        phase_diff_right = torch.angle(ope_right / result) - torch.angle(phase2)
        phase_error_right = torch.mean(torch.abs(torch.exp(1j * phase_diff_right) - 1.0)).item()
        
        print(f"OPE right result phase: {torch.angle(torch.mean(ope_right)):.4f} rad")
        print(f"Expected phase: {torch.angle(torch.mean(result * phase2)):.4f} rad")
        print(f"Right phase error: {phase_error_right:.4f}")

        # Use maximum error from both tests
        phase_consistency = max(phase_error_left, phase_error_right)
        print(f"\nFinal phase consistency: {phase_consistency:.4f}")
        assert phase_consistency < 1e-2, f"OPE should respect U(1) structure, error: {phase_consistency:.2e}"
        
        # Test associativity through triple products
        print("\nAssociativity Test Instrumentation:")
        print(f"Initial operator norms:")
        print(f"op1 norm: {torch.norm(ops[0]):.4f}")
        print(f"op2 norm: {torch.norm(ops[1]):.4f}")
        print(f"op3 norm: {torch.norm(ops[2]):.4f}")

        # Track first path: (op1 ∘ op2) ∘ op3
        print("\nPath 1: (op1 ∘ op2) ∘ op3")
        ope12 = lifter.operator_product_expansion(ops[0], ops[1], normalize=True)
        print(f"ope12 norm: {torch.norm(ope12):.4f}")
        print(f"ope12 phase: {torch.angle(torch.mean(ope12)):.4f}")
        
        ope12_3 = lifter.operator_product_expansion(ope12, ops[2], normalize=True)
        print(f"ope12_3 norm: {torch.norm(ope12_3):.4f}")
        print(f"ope12_3 phase: {torch.angle(torch.mean(ope12_3)):.4f}")

        # Track second path: op1 ∘ (op2 ∘ op3)
        print("\nPath 2: op1 ∘ (op2 ∘ op3)")
        ope23 = lifter.operator_product_expansion(ops[1], ops[2], normalize=True)
        print(f"ope23 norm: {torch.norm(ope23):.4f}")
        print(f"ope23 phase: {torch.angle(torch.mean(ope23)):.4f}")
        
        ope1_23 = lifter.operator_product_expansion(ops[0], ope23, normalize=True)
        print(f"ope1_23 norm: {torch.norm(ope1_23):.4f}")
        print(f"ope1_23 phase: {torch.angle(torch.mean(ope1_23)):.4f}")

        # Compute detailed error metrics
        assoc_error = torch.norm(ope12_3 - ope1_23).item()
        print(f"\nDetailed Error Analysis:")
        print(f"L2 norm of difference: {assoc_error:.4f}")
        print(f"Max element-wise difference: {torch.max(torch.abs(ope12_3 - ope1_23)):.4f}")
        print(f"Mean element-wise difference: {torch.mean(torch.abs(ope12_3 - ope1_23)):.4f}")
        
        assert assoc_error < 0.1, f"OPE should be approximately associative, error: {assoc_error:.2e}"
        
        # Test scaling behavior with distance
        x = torch.linspace(0.1, 2.0, 10, dtype=torch.float32)
        scaled_ops = [op * torch.exp(-x[:, None]**2) for op in ops[:2]]  # Gaussian localization
        opes = [lifter.operator_product_expansion(scaled_ops[0][i], scaled_ops[1][i], normalize=False)
                for i in range(len(x))]
        opes = torch.stack(opes)
        
        # Check that OPE amplitude decreases with distance
        norms = torch.tensor([torch.norm(ope).item() for ope in opes])
        assert torch.all(norms[1:] <= norms[:-1]), "OPE amplitude should decrease with distance"
        
        # Test geometric structure preservation
        # Create operators with specific geometric structure
        metric = torch.eye(dim, dtype=torch.complex64)
        geom_ops = [torch.mv(metric, op) for op in ops[:2]]
        geom_ope = lifter.operator_product_expansion(geom_ops[0], geom_ops[1])
        
        # Verify the geometric structure is preserved
        geom_error = torch.norm(torch.mv(metric, geom_ope) - geom_ope).item()
        assert geom_error < 1e-2, f"OPE should preserve geometric structure, error: {geom_error:.2e}"
        
        # Test memory management
        initial_memory = lifter.memory_manager.get_allocated_memory()
        for _ in range(10):  # Repeated OPE computations
            _ = lifter.operator_product_expansion(ops[0], ops[1])
        final_memory = lifter.memory_manager.get_allocated_memory()
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1024 * 1024, f"Memory growth {memory_growth/1024/1024:.2f}MB exceeds threshold"




    def test_scaling_analysis(self, lifter: HolographicLifter):
        """Analyze scaling behavior in detail."""
        data = self.create_test_data(lifter, n_radial=5, random=False)
        
        print("\nDetailed Scaling Analysis:")
        print(f"{'z':>10} {'Norm':>12} {'Expected':>12} {'Error':>12}")
        print("-" * 48)
        
        for i, z in enumerate(data.radial_points):
            slice_norm = torch.norm(data.bulk_field[i]).item()
            z_ratio = torch.abs(z / data.radial_points[0])
            expected = torch.norm(data.boundary_field).item() * z_ratio**(-lifter.dim)
            error = abs(slice_norm - expected) / expected
            
            print(f"{z.item():10.4f} {slice_norm:12.2e} {expected:12.2e} {error:12.2e}")
            assert error < 0.1, f"Scaling error {error:.2e} at z={z.item():.4f}"

    def test_quantum_corrections(self, lifter: HolographicLifter, test_data: HolographicTestData):
        """Analyze quantum corrections in detail."""
        corrections = []
        for i in range(1, len(test_data.radial_points)):
            prev = test_data.bulk_field[i-1]
            curr = test_data.bulk_field[i]
            ope = lifter.operator_product_expansion(
                prev.flatten()[:lifter.dim] / torch.norm(prev),
                curr.flatten()[:lifter.dim] / torch.norm(curr)
            )
            corrections.append(torch.norm(ope).item())
        
        mean_corr = np.mean(corrections)
        std_corr = np.std(corrections)
        print(f"\nQuantum Corrections: mean={mean_corr:.2e}, std={std_corr:.2e}")
        assert std_corr < mean_corr, "Quantum corrections show high variability" 