"""Tests for holographic lifting functionality.

This module tests the holographic lifting implementation, which maps boundary data to bulk
data using AdS/CFT correspondence principles.
"""

import numpy as np
import pytest
import torch
from typing import Tuple, NamedTuple
import torch.nn as nn
import torch.nn.functional as F
import math
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
                        n_radial: int = 10,  # Fewer radial points by default
                        z_range: Tuple[float, float] = (0.1, 10.0),
                        random: bool = True) -> HolographicTestData:
        """Create and cache test data."""
        boundary = (torch.randn if random else torch.ones)(*shape, dtype=lifter.dtype)
        radial = torch.linspace(z_range[0], z_range[1], n_radial, dtype=lifter.dtype)
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


@pytest.fixture(scope="class")
def pretrained_lifter():
    """Create a pre-trained holographic lifter."""
    lifter = HolographicLifter(dim=4, dtype=torch.complex64)
    
    # Load pre-trained weights if they exist
    weights_path = "pretrained/holographic_net.pt"
    if os.path.exists(weights_path):
        lifter.holographic_net.load_state_dict(torch.load(weights_path))
        return lifter
            
    # If no pre-trained weights, train the network
    print("\nPre-training holographic network...")
    batch_size = 128  # Larger batch for better statistics
    n_epochs = 1000   # Train longer for better convergence
    z_uv = 0.1
    z_ir = 10.0
    
    # Create diverse training data
    uv_data = torch.randn(batch_size, lifter.dim, dtype=lifter.dtype)
    uv_data = uv_data / torch.norm(uv_data, dim=1, keepdim=True)
    
    # Compute expected IR data using analytical scaling
    z_ratio = z_ir / z_uv
    ir_data = uv_data * z_ratio**(-lifter.dim)
    
    # Add quantum corrections with proper scaling
    correction_scale = 0.1 / (1 + z_ratio**2)
    for n in range(1, 4):
        power = -lifter.dim + 2*n
        correction = (-1)**n * uv_data * z_ratio**power / math.factorial(n)
        ir_data = ir_data + correction * correction_scale
    
    # Find optimal learning rate
    def lr_finder(model, x, y, min_lr=1e-5, max_lr=1e-1, num_iters=100):
        init_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        lrs = torch.logspace(math.log10(min_lr), math.log10(max_lr), num_iters)
        losses = []
        
        for lr in lrs:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr.item())
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output.real, y.real) + F.mse_loss(output.imag, y.imag)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if not torch.isfinite(loss):
                break
                    
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(init_state[name])
            
        smoothed_losses = torch.tensor(losses)
        if len(smoothed_losses) > 1:
            derivatives = (smoothed_losses[1:] - smoothed_losses[:-1]) / (lrs[1:len(smoothed_losses)] - lrs[:len(smoothed_losses)-1])
            optimal_idx = torch.argmin(derivatives)
        else:
            optimal_idx = 0
                
        return lrs[optimal_idx].item()
    
    # Train with optimal learning rate and physics-based loss
    optimal_lr = lr_finder(lifter.holographic_net, uv_data, ir_data)
    optimizer = torch.optim.Adam(lifter.parameters(), lr=optimal_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    best_loss = float('inf')
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred_ir = lifter.holographic_net(uv_data)
        
        # Holographic error with conformal weight
        N = z_ratio**lifter.dim
        holographic_error = N*pred_ir - uv_data
        conformal_factor = N**2 + 1
        basic_loss = torch.mean(torch.abs(holographic_error)**2) / conformal_factor
        
        # Quantum corrections with proper scaling
        quantum_pred = lifter.holographic_net.compute_quantum_corrections(uv_data)
        quantum_target = ir_data - uv_data/N  # Remove classical scaling
        quantum_loss = torch.mean(torch.abs(quantum_pred - quantum_target)**2) * N
        
        # Phase coherence with conformal coupling
        phase_diff = torch.angle(pred_ir) - torch.angle(uv_data/N)
        phase_loss = torch.mean(torch.abs(uv_data)**2 * (1 - torch.cos(phase_diff))) / conformal_factor
        
        # Total loss combines all physics terms
        loss = basic_loss + quantum_loss + phase_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.2e}")
            print(f"  Basic: {basic_loss.item():.2e}")
            print(f"  Quantum: {quantum_loss.item():.2e}")
            print(f"  Phase: {phase_loss.item():.2e}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(lifter.holographic_net.state_dict(), weights_path)
    
    return lifter


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

    def test_network_learning(self, lifter: HolographicLifter):
        """Test that the network learns by verifying loss decreases using optimal learning rate."""
        # Create test data with correct shape for holographic network
        # Shape should be [batch_size, seq_len, dim]
        batch_size = 1
        seq_len = 4
        
        print(f"\nNetwork dimensions:")
        print(f"dim: {lifter.dim}")
        print(f"batch_size: {batch_size}")
        print(f"seq_len: {seq_len}")
        
        # Create input with shape [batch_size, seq_len, dim]
        test_input = torch.randn(batch_size, seq_len, lifter.dim, dtype=lifter.dtype)
        test_target = torch.randn(batch_size, seq_len, lifter.dim, dtype=lifter.dtype)
        
        print(f"\nInput shapes:")
        print(f"test_input: {test_input.shape}")
        print(f"test_target: {test_target.shape}")
        
        # Custom complex MSE loss
        def complex_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute MSE loss for complex tensors by treating real and imaginary parts separately."""
            real_loss = F.mse_loss(output.real, target.real)
            imag_loss = F.mse_loss(output.imag, target.imag)
            return real_loss + imag_loss
        
        # Learning rate finder
        def lr_finder(
            model: nn.Module,
            x: torch.Tensor,
            y: torch.Tensor,
            min_lr: float = 1e-5,
            max_lr: float = 1e-1,  # More conservative range
            num_iters: int = 100
        ) -> float:
            # Store initial parameters
            init_state = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }
            
            # Log-spaced learning rates
            lrs = torch.logspace(math.log10(min_lr), math.log10(max_lr), num_iters)
            losses = []
            
            # Test each learning rate
            for lr in lrs:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr.item())
                optimizer.zero_grad()
                
                # Forward pass
                output = model(x)
                loss = complex_mse_loss(output, y)
                losses.append(loss.item())
                
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Check for divergence
                if not torch.isfinite(loss):
                    break
                    
            # Restore initial parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(init_state[name])
            
            # Find learning rate with steepest descent
            smoothed_losses = torch.tensor(losses)  # Include all losses
            if len(smoothed_losses) > 1:  # Check if we have enough losses
                derivatives = (smoothed_losses[1:] - smoothed_losses[:-1]) / (lrs[1:len(smoothed_losses)] - lrs[:len(smoothed_losses)-1])
                optimal_idx = torch.argmin(derivatives)
            else:
                optimal_idx = 0  # Default to first learning rate if we don't have enough data
            
            print(f"\nLR finder results:")
            print(f"Optimal learning rate: {lrs[optimal_idx].item():.2e}")
            print(f"Loss at optimal lr: {losses[optimal_idx]:.2e}")
            
            return lrs[optimal_idx].item()
        
        # Find optimal learning rate
        optimal_lr = lr_finder(lifter.holographic_net, test_input, test_target)
        
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in lifter.holographic_net.named_parameters()
        }
        
        # Create optimizer with found learning rate
        optimizer = torch.optim.Adam(lifter.parameters(), lr=optimal_lr)
        
        # Track losses
        losses = []
        
        # Do several training steps
        n_steps = 10
        for step in range(n_steps):
            optimizer.zero_grad()
            output = lifter.holographic_net(test_input)
            loss = complex_mse_loss(output, test_target)
            losses.append(loss.item())
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(lifter.holographic_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            print(f"Step {step}: loss = {loss.item():.2e}")
        
        # Check if parameters changed
        changed = False
        for name, param in lifter.holographic_net.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                changed = True
                break
                
        assert changed, "Parameters did not change during training"
        assert losses[-1] < losses[0], "Loss did not decrease during training"



    def test_holographic_convergence(self, pretrained_lifter: HolographicLifter):
        """Test that the pre-trained network correctly implements the holographic lifting function."""
        # Test on new data
        test_uv = torch.randn(10, pretrained_lifter.dim, dtype=pretrained_lifter.dtype)
        test_uv = test_uv / torch.norm(test_uv, dim=1, keepdim=True)
        
        # Compute network prediction
        pred_ir = pretrained_lifter.holographic_net(test_uv)
        
        # Compute expected IR using analytical formula
        z_uv = 0.1
        z_ir = 10.0
        z_ratio = z_ir / z_uv
        expected_ir = test_uv * z_ratio**(-pretrained_lifter.dim)
        
        # Add quantum corrections
        correction_scale = 0.1 / (1 + z_ratio**2)
        for n in range(1, 4):
            power = -pretrained_lifter.dim + 2*n
            correction = (-1)**n * test_uv * z_ratio**power / math.factorial(n)
            expected_ir = expected_ir + correction * correction_scale
        
        # Verify accuracy
        rel_error = torch.norm(pred_ir - expected_ir) / torch.norm(expected_ir)
        assert rel_error < 0.1, f"Network predictions deviate from theoretical values: {rel_error:.2e}"
        
        # Verify scaling behavior is preserved
        scale_ratios = torch.norm(pred_ir, dim=1) / torch.norm(test_uv, dim=1)
        expected_ratio = z_ratio**(-pretrained_lifter.dim)
        scale_error = torch.abs(scale_ratios - expected_ratio).mean()
        assert scale_error < 0.1, f"Network failed to preserve scaling behavior: {scale_error:.2e}"



    def test_uv_ir_connection(self, lifter: HolographicLifter, test_data: HolographicTestData):
        """Test UV/IR connection with detailed diagnostics."""
        reconstructed = lifter.reconstruct_from_ir(test_data.ir_data)
        self.validate_tensor(reconstructed, "reconstructed field")
        
        # Compute and log key metrics
        uv_norm = torch.norm(test_data.uv_data).item()
        ir_norm = torch.norm(test_data.ir_data).item()
        rec_norm = torch.norm(reconstructed).item()
        rel_error = self.rel_error(reconstructed, test_data.uv_data)
        
        print(f"\nUV/IR Analysis: UV={uv_norm:.2e}, IR={ir_norm:.2e}, Rec={rec_norm:.2e}, Err={rel_error:.2e}")
        
        assert rel_error < 1e-4, f"UV/IR reconstruction failed with error {rel_error:.2e}"

    def test_c_theorem(self, lifter: HolographicLifter, test_data: HolographicTestData):
        """Test c-theorem with monotonicity checks."""
        c_func = torch.abs(lifter.compute_c_function(test_data.bulk_field, test_data.radial_points))
        self.validate_tensor(c_func, "c-function")
        
        # Test monotonicity with clear violation reporting
        diffs = c_func[1:] - c_func[:-1]
        max_violation = torch.max(diffs).item()
        assert max_violation <= 1e-6, \
            f"C-theorem violated: found increasing region with slope {max_violation:.2e}"
        
        # Test UV/IR ordering
        uv_ir_ratio = (c_func[0] / c_func[-1]).item()
        assert uv_ir_ratio > 1, \
            f"C-theorem UV/IR ordering violated: UV/IR ratio = {uv_ir_ratio:.2e}"

    def test_operator_expansion(self, lifter: HolographicLifter):
        """Test OPE properties with normalized operators."""
        # Create test operators
        ops = [torch.randn(lifter.dim, dtype=lifter.dtype) for _ in range(2)]
        ops = [op / torch.norm(op) for op in ops]
        
        # Test basic properties
        result = lifter.operator_product_expansion(*ops)
        self.validate_tensor(result, "OPE result")
        
        # Test normalization
        norm_error = abs(torch.norm(result).item() - 1.0)
        assert norm_error < 0.1, f"OPE normalization error: {norm_error:.2e}"
        
        # Test symmetry
        ope12 = lifter.operator_product_expansion(*ops)
        ope21 = lifter.operator_product_expansion(*reversed(ops))
        sym_error = torch.norm(torch.abs(ope12) - torch.abs(ope21)).item()
        assert sym_error < 0.1, f"OPE symmetry violated with error {sym_error:.2e}"



class TestHolographicLiftDebug(TestBase):
    """Debug test suite with detailed diagnostics."""
    
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