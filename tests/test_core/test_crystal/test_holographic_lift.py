"""Tests for holographic lifting functionality.

This module tests the holographic lifting implementation, which maps boundary data to bulk
data using AdS/CFT correspondence principles.
"""

import pytest
import torch
from typing import Tuple
from src.core.crystal.scale_classes.holographiclift import HolographicLifter
from src.core.crystal.scale_classes.ml.models import HolographicNet
from tests.test_core.test_crystal.test_base import TestHolographicBase
from tests.test_core.test_crystal.test_ml import TestML

@pytest.mark.dependency(depends=["TestML::test_holographic_convergence"])
class TestHolographicLift(TestHolographicBase):
    """Core test suite for holographic lifting."""
    
    @pytest.fixture(scope="class")
    def model(self, config):
        """Create model using configuration."""
        return HolographicNet(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dtype=config.dtype,
            z_uv=config.z_uv,
            z_ir=config.z_ir
        )
    
    @pytest.fixture(scope="class")
    def lifter(self, model, config):
        """Create reusable holographic lifter fixture."""
        return HolographicLifter(
            dim=config.dim,
            dtype=config.dtype
        )

    @pytest.fixture(scope="class")
    def test_data(self, lifter, model):
        """Create reusable test data fixture."""
        return self.create_test_data(lifter)

    def test_basic_lifting(self, lifter: HolographicLifter, test_data, config):
        """Test basic lifting properties with clear error messages."""
        # Shape tests
        assert test_data.bulk_field.shape[0] == len(test_data.radial_points), \
            "Bulk field has incorrect number of radial slices"
        assert test_data.bulk_field.shape[1:] == test_data.boundary_field.shape, \
            "Bulk field has incorrect boundary dimensions"
        
        # Boundary condition test
        uv_error = self.rel_error(
            torch.abs(test_data.uv_data), 
            torch.abs(test_data.boundary_field)
        )
        assert uv_error < config.uv_boundary_threshold, \
            f"UV boundary condition violated with error {uv_error:.2e}"
        
        # Radial scaling test with clear error reporting
        for i, z in enumerate(test_data.radial_points):
            # Use dimensionless ratio for scaling
            z_ratio = torch.abs(z / test_data.radial_points[0])
            expected_scale = z_ratio**(-lifter.dim)
            actual_scale = torch.norm(test_data.bulk_field[i]) / torch.norm(test_data.boundary_field)
            scale_error = self.rel_error(actual_scale, expected_scale)
            assert scale_error < config.radial_scaling_threshold, \
                f"Radial scaling violated at z={z:.2f} with error {scale_error:.2e}"

    @pytest.mark.dependency(depends=["TestML::test_norm_preservation"])
    def test_uv_ir_connection(self, lifter: HolographicLifter, test_data, config):
        """Test UV/IR connection with detailed diagnostics."""
        reconstructed = lifter.reconstruct_from_ir(test_data.ir_data)
        self.validate_tensor(reconstructed, "reconstructed field")
        
        # Compute and log key metrics
        uv_norm = torch.norm(test_data.uv_data).item()
        ir_norm = torch.norm(test_data.ir_data).item()
        rec_norm = torch.norm(reconstructed).item()
        rel_error = self.rel_error(reconstructed, test_data.uv_data)
        
        print(f"\nUV/IR Analysis: UV={uv_norm:.2e}, IR={ir_norm:.2e}, Rec={rec_norm:.2e}, Err={rel_error:.2e}")
        
        assert rel_error < config.reconstruction_threshold, \
            f"UV/IR reconstruction failed with error {rel_error:.2e}"

    def test_c_theorem(self, lifter: HolographicLifter, test_data, config):
        """Test c-theorem with monotonicity checks."""
        c_func = torch.abs(lifter.compute_c_function(test_data.bulk_field, test_data.radial_points))
        self.validate_tensor(c_func, "c-function")
        
        # Test monotonicity with clear violation reporting
        diffs = c_func[1:] - c_func[:-1]
        max_violation = torch.max(diffs).item()
        assert max_violation <= config.c_theorem_threshold, \
            f"C-theorem violated: found increasing region with slope {max_violation:.2e}"
        
        # Test UV/IR ordering
        uv_ir_ratio = (c_func[0] / c_func[-1]).item()
        assert uv_ir_ratio > 1, \
            f"C-theorem UV/IR ordering violated: UV/IR ratio = {uv_ir_ratio:.2e}"

    @pytest.mark.dependency(depends=["TestML::test_quantum_corrections"])
    def test_operator_expansion(self, lifter: HolographicLifter, config):
        """Test OPE properties with normalized operators."""
        # Create test operators
        ops = [torch.randn(lifter.dim, dtype=lifter.dtype) for _ in range(2)]
        ops = [op / torch.norm(op) for op in ops]
        
        # Test basic properties
        result = lifter.operator_product_expansion(*ops)
        self.validate_tensor(result, "OPE result")
        
        # Test normalization
        norm_error = abs(torch.norm(result).item() - 1.0)
        assert norm_error < config.quantum_correction_threshold, \
            f"OPE normalization error: {norm_error:.2e}"
        
        # Test symmetry
        ope12 = lifter.operator_product_expansion(*ops)
        ope21 = lifter.operator_product_expansion(*reversed(ops))
        sym_error = torch.norm(torch.abs(ope12) - torch.abs(ope21)).item()
        assert sym_error < config.quantum_correction_threshold, \
            f"OPE symmetry violated with error {sym_error:.2e}"


@pytest.mark.dependency(depends=["TestHolographicLift::test_basic_lifting"])
class TestHolographicLiftDebug(TestHolographicBase):
    """Debug test suite with detailed diagnostics."""
    
    @pytest.fixture(scope="class")
    def model(self, config):
        """Create model using configuration."""
        return HolographicNet(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dtype=config.dtype,
            z_uv=config.z_uv,
            z_ir=config.z_ir
        )
    
    @pytest.fixture(scope="class")
    def lifter(self, model, config):
        """Create reusable holographic lifter fixture."""
        return HolographicLifter(
            dim=config.dim,
            dtype=config.dtype
        )
    
    def test_scaling_analysis(self, lifter: HolographicLifter, config):
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
            assert error < config.radial_scaling_threshold, \
                f"Scaling error {error:.2e} at z={z.item():.4f}"

    @pytest.mark.dependency(depends=["TestML::test_quantum_corrections"])
    def test_quantum_corrections(self, lifter: HolographicLifter, test_data, config):
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
        
        mean_corr = sum(corrections) / len(corrections)
        std_corr = torch.tensor(corrections).std().item()
        print(f"\nQuantum Corrections: mean={mean_corr:.2e}, std={std_corr:.2e}")
        assert std_corr < mean_corr, "Quantum corrections show high variability" 