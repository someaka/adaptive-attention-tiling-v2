"""Tests for holographic neural network implementation."""

import pytest
import torch
import torch.nn.functional as F
from src.core.crystal.scale_classes.ml.models import HolographicNet
from tests.test_core.test_crystal.test_base import TestHolographicBase


class TestML(TestHolographicBase):
    """Test suite for holographic neural network."""
    
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
    
    def test_model_initialization(self, model: HolographicNet, config):
        """Test model initialization and basic properties."""
        assert model.dim == config.dim, "Incorrect dimension"
        assert model.hidden_dim == config.hidden_dim, "Incorrect hidden dimension"
        assert model.n_layers == config.n_layers, "Incorrect number of layers"
        assert model.dtype == config.dtype, "Incorrect dtype"
        
        # Test parameter initialization
        for name, param in model.named_parameters():
            self.validate_tensor(param, name)
            assert param.dtype == config.dtype, f"Parameter {name} has wrong dtype"
    
    def test_forward_shape(self, model: HolographicNet, config):
        """Test forward pass shape preservation."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, config.dim, dtype=config.dtype)
        output = model(input_tensor)
        
        assert output.shape == input_tensor.shape, "Forward pass changed tensor shape"
        assert output.dtype == input_tensor.dtype, "Forward pass changed dtype"
        self.validate_tensor(output, "forward pass output")
    
    def test_quantum_corrections(self, model: HolographicNet, config):
        """Test quantum correction computation."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, config.dim, dtype=config.dtype)
        corrections = model.compute_quantum_corrections(input_tensor)
        
        # Test shape and properties
        assert corrections.shape == input_tensor.shape, "Quantum corrections changed shape"
        assert corrections.dtype == input_tensor.dtype, "Quantum corrections changed dtype"
        self.validate_tensor(corrections, "quantum corrections")
        
        # Test that corrections are smaller than input
        input_norm = torch.norm(input_tensor)
        corr_norm = torch.norm(corrections)
        assert corr_norm < input_norm, "Quantum corrections larger than input"
        
        # Test that corrections scale with input
        scaled_input = input_tensor * 2
        scaled_corr = model.compute_quantum_corrections(scaled_input)
        scale_error = self.rel_error(scaled_corr, corrections * 2)
        assert scale_error < config.quantum_correction_threshold, \
            f"Quantum corrections failed scaling test with error {scale_error:.2e}"
    
    def test_loss_computation(self, model: HolographicNet, config):
        """Test loss function computation."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, config.dim, dtype=config.dtype)
        target = torch.randn(batch_size, config.dim, dtype=config.dtype)
        
        # Basic loss
        basic_loss = F.mse_loss(model(input_tensor).real, target.real) + \
                    F.mse_loss(model(input_tensor).imag, target.imag)
        assert torch.isfinite(basic_loss), "Basic loss is not finite"
        
        # Quantum loss
        quantum_pred = model.compute_quantum_corrections(input_tensor)
        quantum_target = target - input_tensor
        quantum_loss = F.mse_loss(quantum_pred.real, quantum_target.real) + \
                      F.mse_loss(quantum_pred.imag, quantum_target.imag)
        assert torch.isfinite(quantum_loss), "Quantum loss is not finite"
        
        # Phase loss
        phase_diff = torch.angle(model(input_tensor)) - torch.angle(target)
        phase_loss = torch.mean(1 - torch.cos(phase_diff))
        assert torch.isfinite(phase_loss), "Phase loss is not finite"
    
    def test_norm_preservation(self, model: HolographicNet, config):
        """Test that the model approximately preserves input norms."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, config.dim, dtype=config.dtype)
        input_norms = torch.norm(input_tensor, dim=1)
        
        output = model(input_tensor)
        output_norms = torch.norm(output, dim=1)
        
        # Test norm preservation with tolerance
        norm_errors = torch.abs(output_norms - input_norms) / input_norms
        max_error = torch.max(norm_errors).item()
        assert max_error < config.norm_preservation_threshold, \
            f"Model failed to preserve norms with max error {max_error:.2e}"
    
    def test_training_step(self, model: HolographicNet, config):
        """Test that a single training step runs without errors."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, config.dim, dtype=config.dtype)
        target = torch.randn(batch_size, config.dim, dtype=config.dtype)
        
        # Store initial parameters
        init_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Run training step
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer.zero_grad()
        
        output = model(input_tensor)
        loss = F.mse_loss(output.real, target.real) + F.mse_loss(output.imag, target.imag)
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, init_params[name]):
                changed = True
                break
        assert changed, "Parameters did not update during training step"
    
    def test_holographic_convergence(self, model: HolographicNet, config):
        """Test that the network learns the correct holographic mapping."""
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # Create test data
        test_data = self.create_test_data(model)
        
        # Train model with improved optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-2,
            epochs=config.n_epochs,
            steps_per_epoch=1,
            pct_start=0.3,  # Warmup for 30% of training
            div_factor=25.0,  # Initial lr = max_lr/25
            final_div_factor=1e4  # Final lr = max_lr/10000
        )
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.n_epochs):
            optimizer.zero_grad()
            pred_ir = model(test_data.uv_data)
            
            # Compute losses with balanced weights
            basic_loss = F.mse_loss(pred_ir.real, test_data.ir_data.real) + \
                        F.mse_loss(pred_ir.imag, test_data.ir_data.imag)
            with torch.no_grad():
                quantum_pred = model.compute_quantum_corrections(test_data.uv_data.detach())
                quantum_target = test_data.ir_data.detach() - test_data.uv_data.detach()
                quantum_loss = F.mse_loss(quantum_pred.real, quantum_target.real) + \
                              F.mse_loss(quantum_pred.imag, quantum_target.imag)
                phase_diff = torch.angle(pred_ir.detach()) - torch.angle(test_data.ir_data.detach())
                phase_loss = torch.mean(1 - torch.cos(phase_diff))
            
            # Add L2 regularization for stability
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            
            # Total loss with adjusted weights
            loss = basic_loss + \
                   0.01 * quantum_loss + \
                   0.1 * phase_loss + \
                   1e-5 * l2_reg
            
            # Backward pass with retain_graph but detach intermediate tensors
            loss.backward(retain_graph=True)
            # Detach tensors to prevent inplace operations
            pred_ir = pred_ir.detach()
            uv_data = test_data.uv_data.detach()
            ir_data = test_data.ir_data.detach()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Early stopping with improved criteria
            if loss.item() < best_loss - config.min_delta:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.2e}, Basic: {basic_loss.item():.2e}, "
                      f"Quantum: {quantum_loss.item():.2e}, Phase: {phase_loss.item():.2e}")
        
        # Verify convergence
        final_pred = model(test_data.uv_data)
        error = self.rel_error(final_pred, test_data.ir_data)
        assert error < config.convergence_threshold, \
            f"Network failed to converge with error {error:.2e}" 