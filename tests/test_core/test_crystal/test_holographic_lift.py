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
            


    def test_operator_expansion(self):
        """Test operator product expansion properties with detailed phase tracking."""
        # Initialize test environment
        dtype = torch.complex128
        lifter = HolographicLifter(dim=4, dtype=dtype)
        
        def print_phase_info(tensor: torch.Tensor, name: str):
            """Helper to print detailed phase information."""
            mean = torch.mean(tensor)
            phase = mean / torch.abs(mean) if torch.abs(mean) > lifter.EPSILON else torch.tensor(1.0, dtype=dtype)
            angle = torch.angle(phase)
            norm = torch.norm(tensor)
            print(f"\n{name}:")
            print(f"  Mean: {mean.real:.4f} + {mean.imag:.4f}j")
            print(f"  Phase: {phase.real:.4f} + {phase.imag:.4f}j")
            print(f"  Angle: {angle.item():.4f} rad")
            print(f"  Norm: {norm.item():.4f}")
            return phase, angle, norm

        # Test case with detailed tracking
        theta1 = torch.pi/4  # π/4
        theta2 = torch.pi/3  # π/3
        print(f"\nTest case: θ₁={theta1:.4f} rad, θ₂={theta2:.4f} rad")
        
        # Create operators with specific phases
        op1 = torch.ones(4, dtype=dtype) * torch.exp(1j * torch.tensor(theta1, dtype=torch.float64))
        op2 = torch.ones(4, dtype=dtype) * torch.exp(1j * torch.tensor(theta2, dtype=torch.float64))
        
        print("\nInitial Operators:")
        phase1, angle1, norm1 = print_phase_info(op1, "Operator 1")
        phase2, angle2, norm2 = print_phase_info(op2, "Operator 2")
        
        # Expected combined phase
        expected_phase = torch.exp(1j * torch.tensor(theta1 + theta2, dtype=torch.float64))
        expected_phase = expected_phase / torch.abs(expected_phase)
        expected_angle = torch.angle(expected_phase)
        print("\nExpected Combined Phase:")
        print(f"  Phase: {expected_phase.real:.4f} + {expected_phase.imag:.4f}j")
        print(f"  Angle: {expected_angle.item():.4f} rad")
        
        # Compute OPE
        print("\nComputing OPE...")
        result = lifter.operator_product_expansion(op1, op2)
        result_phase, result_angle, result_norm = print_phase_info(result, "OPE Result")
        
        # Detailed error analysis
        phase_error = torch.abs(result_phase - expected_phase)
        angle_error = torch.abs(result_angle - expected_angle)
        print("\nError Analysis:")
        print(f"  Phase error (|result - expected|): {phase_error.item():.4f}")
        print(f"  Angle error (|result - expected|): {angle_error.item():.4f} rad")
        print(f"  Norm error (|result - 1.0|): {abs(result_norm - 1.0):.4e}")
        
        # Verify phase consistency
        assert phase_error < 0.01, f"Phase error {phase_error} exceeds threshold"
        assert torch.abs(result_norm - 1.0) < 1e-6, f"Result not normalized: {result_norm}"
        
        # Test U(1) structure preservation with detailed tracking
        print("\nTesting U(1) Structure Preservation...")
        
        # Left multiplication
        left_phase = torch.exp(1j * torch.tensor(torch.pi/4, dtype=torch.float64))
        print("\nLeft Multiplication:")
        print(f"  Test phase: {left_phase.real:.4f} + {left_phase.imag:.4f}j")
        print(f"  Test angle: {torch.angle(left_phase).item():.4f} rad")
        
        left_op = left_phase * op1
        print_phase_info(left_op, "Left-multiplied operator")
        
        left_result = lifter.operator_product_expansion(left_op, op2)
        left_actual_phase, left_actual_angle, _ = print_phase_info(left_result, "Left OPE result")
        
        # Expected left result
        left_expected = left_phase * result_phase
        print("\nLeft Expected:")
        print(f"  Phase: {left_expected.real:.4f} + {left_expected.imag:.4f}j")
        print(f"  Angle: {torch.angle(left_expected).item():.4f} rad")
        
        left_error = torch.abs(left_actual_phase - left_expected)
        print(f"Left error: {left_error.item():.4f}")
        
        # Right multiplication
        right_phase = torch.exp(1j * torch.tensor(torch.pi/3, dtype=torch.float64))
        print("\nRight Multiplication:")
        print(f"  Test phase: {right_phase.real:.4f} + {right_phase.imag:.4f}j")
        print(f"  Test angle: {torch.angle(right_phase).item():.4f} rad")
        
        right_op = right_phase * op2
        print_phase_info(right_op, "Right-multiplied operator")
        
        right_result = lifter.operator_product_expansion(op1, right_op)
        right_actual_phase, right_actual_angle, _ = print_phase_info(right_result, "Right OPE result")
        
        # Expected right result
        right_expected = right_phase * result_phase
        print("\nRight Expected:")
        print(f"  Phase: {right_expected.real:.4f} + {right_expected.imag:.4f}j")
        print(f"  Angle: {torch.angle(right_expected).item():.4f} rad")
        
        right_error = torch.abs(right_actual_phase - right_expected)
        print(f"Right error: {right_error.item():.4f}")
        
        # Final assertions
        assert left_error < 0.01, f"Left U(1) error {left_error} exceeds threshold"
        assert right_error < 0.01, f"Right U(1) error {right_error} exceeds threshold"

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