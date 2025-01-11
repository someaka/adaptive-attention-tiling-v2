"""Test gradient flow through quantum geometric attention.

This module tests gradient flow through various components of the quantum geometric attention
mechanism. Tests are organized by component and verify both forward and backward pass
behavior, ensuring proper gradient propagation and energy conservation.

Key test categories:
1. Metric tensor gradients
2. Quantum bridge component gradients
3. Attention mechanism gradients
4. Energy conservation
5. Shape validation
6. Complex tensor operations
7. Edge cases and error handling
"""

import torch
import torch.nn as nn
import pytest
import logging
from typing import Tuple, Dict, Any, Optional
from pytest import approx

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention
from src.core.patterns.riemannian_flow import RiemannianFlow
from src.metrics.attention import AttentionMetrics, FlowMetrics
from src.core.quantum.types import QuantumState

logger = logging.getLogger(__name__)

def complex_randn(*size, device=None, dtype=torch.complex64):
    """Create random complex tensor with proper initialization."""
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(*size, device=device, dtype=real_dtype)
    imag = torch.randn(*size, device=device, dtype=real_dtype)
    return torch.complex(real, imag).to(dtype=dtype)

@pytest.fixture
def setup_attention() -> Tuple[QuantumGeometricAttention, Dict[str, Any]]:
    """Setup attention layer and parameters for testing."""
    params = {
        "batch_size": 2,
        "hidden_dim": 16,
        "seq_length": 4,
        "num_heads": 2,
        "manifold_dim": 8
    }
    
    layer = QuantumGeometricAttention(
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        manifold_dim=params["manifold_dim"],
        dtype=torch.complex64
    )
    
    return layer, params

class TestGradientFlow:
    """Test gradient flow through quantum geometric attention."""
    
    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Setup test environment."""
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        yield
        torch.autograd.set_detect_anomaly(False)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def verify_gradients(self, tensor: torch.Tensor, name: str):
        """Helper to verify gradient properties."""
        assert tensor.grad is not None, f"{name} should have gradients"
        assert torch.isfinite(tensor.grad).all(), f"{name} has inf/nan gradients"
        assert tensor.grad.abs().mean() > 0, f"{name} has zero gradients"
        
        # Additional checks for complex tensors
        if torch.is_complex(tensor):
            assert torch.isfinite(tensor.grad.real).all(), f"{name} has inf/nan real gradients"
            assert torch.isfinite(tensor.grad.imag).all(), f"{name} has inf/nan imaginary gradients"
            assert tensor.grad.real.abs().mean() > 0, f"{name} has zero real gradients"
            assert tensor.grad.imag.abs().mean() > 0, f"{name} has zero imaginary gradients"

    def verify_tensor_properties(self, tensor: torch.Tensor, name: str):
        """Helper to verify tensor properties."""
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains infinite values"
        if torch.is_complex(tensor):
            assert not torch.isnan(tensor.real).any(), f"{name} contains real NaN values"
            assert not torch.isnan(tensor.imag).any(), f"{name} contains imaginary NaN values"
            assert not torch.isinf(tensor.real).any(), f"{name} contains real infinite values"
            assert not torch.isinf(tensor.imag).any(), f"{name} contains imaginary infinite values"

    class TestMetricFlow:
        """Tests for gradient flow through metric tensors."""
        
        @pytest.mark.timeout(30)
        def test_base_metric_gradient_flow(self, setup_attention):
            """Test gradient flow through base metric."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            base_metric = layer.base_metric
            base_metric.requires_grad_(True)
            base_metric.retain_grad()
            
            output = layer(x)
            loss = output.abs().mean()
            loss.backward()
            
            assert base_metric.grad is not None, "Base metric should receive gradients"
            assert base_metric.grad.abs().mean() > 0, "Base metric gradients should be non-zero"
            assert torch.isfinite(base_metric.grad).all(), "Base metric gradients should be finite"
        
        @pytest.mark.timeout(30)
        def test_pattern_metric_gradient_flow(self, setup_attention):
            """Test gradient flow through pattern metric."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            pattern_metric = layer.pattern_metric
            pattern_metric.requires_grad_(True)
            pattern_metric.retain_grad()
            
            output = layer(x)
            loss = output.abs().mean()
            loss.backward()
            
            assert pattern_metric.grad is not None, "Pattern metric should receive gradients"
            assert pattern_metric.grad.abs().mean() > 0, "Pattern metric gradients should be non-zero"
            assert torch.isfinite(pattern_metric.grad).all(), "Pattern metric gradients should be finite"
        
        @pytest.mark.timeout(30)
        def test_combined_metric_gradient_flow(self, setup_attention):
            """Test gradient flow through combined metric tensor."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            combined_metric = layer.combined_metric
            combined_metric.requires_grad_(True)
            combined_metric.retain_grad()
            
            output = layer(x)
            loss = output.abs().mean()
            loss.backward()
            
            assert combined_metric.grad is not None, "Combined metric should receive gradients"
            assert combined_metric.grad.abs().mean() > 0, "Combined metric gradients should be non-zero"
            assert torch.isfinite(combined_metric.grad).all(), "Combined metric gradients should be finite"
    
    class TestQuantumBridgeFlow:
        """Tests for gradient flow through quantum bridge components."""
        
        @pytest.mark.timeout(30)
        def test_pattern_bundle_metric_flow(self, setup_attention):
            """Test gradient flow through quantum bridge pattern bundle metric."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            pattern_bundle_metric = layer.quantum_bridge.pattern_bundle_metric
            pattern_bundle_metric.requires_grad_(True)
            pattern_bundle_metric.retain_grad()
            
            output = layer(x)
            loss = output.abs().mean()
            loss.backward()
            
            assert pattern_bundle_metric.grad is not None, "Pattern bundle metric should receive gradients"
            assert pattern_bundle_metric.grad.abs().mean() > 0, "Pattern bundle metric gradients should be non-zero"
            assert torch.isfinite(pattern_bundle_metric.grad).all(), "Pattern bundle metric gradients should be finite"
        
        @pytest.mark.timeout(30)
        def test_connection_gradient_flow(self, setup_attention):
            """Test gradient flow through pattern bundle connection."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            connection = layer.quantum_bridge.pattern_bundle.connection
            connection.requires_grad_(True)
            connection.retain_grad()
            
            output = layer(x)
            loss = output.abs().mean()
            loss.backward()
            
            assert connection.grad is not None, "Connection should receive gradients"
            assert connection.grad.abs().mean() > 0, "Connection gradients should be non-zero"
            assert torch.isfinite(connection.grad).all(), "Connection gradients should be finite"
        
        @pytest.mark.timeout(30)
        def test_metric_factors_gradient_flow(self, setup_attention):
            """Test gradient flow through metric factors."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            metric_factors = layer.quantum_bridge.pattern_bundle.riemannian_framework.metric_factors
            metric_factors.requires_grad_(True)
            metric_factors.retain_grad()
            
            output = layer(x)
            loss = output.abs().mean()
            loss.backward()
            
            assert metric_factors.grad is not None, "Metric factors should receive gradients"
            assert metric_factors.grad.abs().mean() > 0, "Metric factors gradients should be non-zero"
            assert torch.isfinite(metric_factors.grad).all(), "Metric factors gradients should be finite"
        
        @pytest.mark.timeout(30)
        def test_connection_coefficients_gradient_flow(self, setup_attention):
            """Test gradient flow through connection coefficients."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            # Get connection coefficients
            connection = layer.quantum_bridge.pattern_bundle.connection
            connection.requires_grad_(True)
            connection.retain_grad()
            
            # Track intermediate tensors
            intermediate_tensors = {}
            computation_steps = []
            
            def save_tensor(name, tensor, step_info=""):
                """Track tensor with computation step info."""
                if tensor.requires_grad:
                    tensor.retain_grad()
                    intermediate_tensors[name] = tensor
                    computation_steps.append(f"Step: {step_info}")
                    
                    print(f"\nTracking tensor: {name}")
                    print(f"Step info: {step_info}")
                    print(f"Shape: {tensor.shape}")
                    print(f"Requires grad: {tensor.requires_grad}")
                    print(f"Is complex: {tensor.is_complex()}")
                    
                    def hook(grad):
                        if grad is not None:
                            print(f"\nGradient for {name}:")
                            print(f"Shape: {grad.shape}")
                            if grad.is_complex():
                                grad_abs = grad.abs()
                                print(f"Complex Gradient stats:")
                                print(f"Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                                print(f"Real mean: {grad.real.mean().item():.6f}")
                                print(f"Imag mean: {grad.imag.mean().item():.6f}")
                            return grad
                        return grad
                    
                    tensor.register_hook(hook)
            
            # Track connection usage
            save_tensor("connection", connection, "Initial connection tensor")
            
            # Forward pass
            output = layer(x)
            save_tensor("output", output, "Final output")
            
            # Compute loss and backward
            loss = output.abs().sum()
            print(f"Loss value: {loss.item():.6f}")
            loss.backward()
            
            # Verify gradients
            assert connection.grad is not None, "Connection should have gradients"
            assert torch.isfinite(connection.grad).all(), "Connection has inf/nan gradients"
            assert connection.grad.abs().mean() > 0, "Connection has zero gradients"
            
            # Check gradient properties
            if connection.grad is not None:
                grad = connection.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    print(f"\nComplex Gradient stats:")
                    print(f"Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    print(f"Real mean: {grad.real.mean().item():.6f}")
                    print(f"Imag mean: {grad.imag.mean().item():.6f}")
                    print(f"Max magnitude: {grad_abs.max().item():.6f}")
                    print(f"Min magnitude: {grad_abs.min().item():.6f}")

        @pytest.mark.timeout(30)
        def test_christoffel_symbols_gradient_flow(self, setup_attention):
            """Test gradient flow through Christoffel symbols."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            # Get pattern bundle
            pattern_bundle = layer.quantum_bridge.pattern_bundle
            
            # Compute Christoffel symbols
            metric = pattern_bundle.metric
            metric.requires_grad_(True)
            metric.retain_grad()
            
            def compute_christoffel(metric):
                """Compute Christoffel symbols."""
                # Compute inverse metric
                metric_inv = torch.inverse(metric)
                
                # Compute metric derivatives
                grad_metric = torch.autograd.grad(
                    metric.sum(), x,
                    create_graph=True, retain_graph=True
                )[0]
                
                # Compute Christoffel symbols
                christoffel = 0.5 * torch.einsum(
                    'ij,jkl->ikl',
                    metric_inv,
                    grad_metric + grad_metric.transpose(-2, -1) - grad_metric.transpose(-1, -2)
                )
                
                return christoffel
            
            # Forward pass with Christoffel computation
            christoffel = compute_christoffel(metric)
            output = layer(x)
            
            # Compute loss and backward
            loss = output.abs().mean() + christoffel.abs().mean()
            loss.backward()
            
            # Verify gradients
            assert metric.grad is not None, "Metric should have gradients"
            assert torch.isfinite(metric.grad).all(), "Metric has inf/nan gradients"
            assert metric.grad.abs().mean() > 0, "Metric has zero gradients"

        @pytest.mark.timeout(30)
        def test_riemann_tensor_gradient_flow(self, setup_attention):
            """Test gradient flow through Riemann curvature tensor."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            # Get pattern bundle
            pattern_bundle = layer.quantum_bridge.pattern_bundle
            
            # Compute Riemann tensor
            metric = pattern_bundle.metric
            metric.requires_grad_(True)
            metric.retain_grad()
            
            def compute_riemann(metric):
                """Compute Riemann curvature tensor."""
                # First compute Christoffel symbols
                metric_inv = torch.inverse(metric)
                grad_metric = torch.autograd.grad(
                    metric.sum(), x,
                    create_graph=True, retain_graph=True
                )[0]
                
                christoffel = 0.5 * torch.einsum(
                    'ij,jkl->ikl',
                    metric_inv,
                    grad_metric + grad_metric.transpose(-2, -1) - grad_metric.transpose(-1, -2)
                )
                
                # Compute Riemann tensor
                riemann = (
                    torch.einsum('...ijk->...kij', christoffel) -
                    torch.einsum('...ikj->...kij', christoffel)
                )
                
                return riemann
            
            # Forward pass with Riemann computation
            riemann = compute_riemann(metric)
            output = layer(x)
            
            # Compute loss and backward
            loss = output.abs().mean() + riemann.abs().mean()
            loss.backward()
            
            # Verify gradients
            assert metric.grad is not None, "Metric should have gradients"
            assert torch.isfinite(metric.grad).all(), "Metric has inf/nan gradients"
            assert metric.grad.abs().mean() > 0, "Metric has zero gradients"

    class TestShapeValidation:
        """Tests for tensor shape validation."""
        
        @pytest.mark.timeout(30)
        def test_input_shape_validation(self, setup_attention):
            """Test input tensor shape validation."""
            layer, params = setup_attention
            
            # Test invalid batch size
            with pytest.raises(ValueError, match="Invalid input shape"):
                invalid_batch = complex_randn(
                    params["batch_size"] + 1,
                    params["seq_length"],
                    params["hidden_dim"]
                )
                layer(invalid_batch)
            
            # Test invalid sequence length
            with pytest.raises(ValueError, match="Invalid input shape"):
                invalid_seq = complex_randn(
                    params["batch_size"],
                    params["seq_length"] + 1,
                    params["hidden_dim"]
                )
                layer(invalid_seq)
            
            # Test invalid hidden dimension
            with pytest.raises(ValueError, match="Invalid input shape"):
                invalid_hidden = complex_randn(
                    params["batch_size"],
                    params["seq_length"],
                    params["hidden_dim"] + 1
                )
                layer(invalid_hidden)

        @pytest.mark.timeout(30)
        def test_intermediate_shape_validation(self, setup_attention):
            """Test intermediate tensor shape validation."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            
            # Track shapes through forward pass
            shapes = {}
            
            def shape_hook(name):
                def hook(module, input, output):
                    shapes[name] = {
                        'input': [t.shape if isinstance(t, torch.Tensor) else None for t in input],
                        'output': output.shape if isinstance(output, torch.Tensor) else 
                                 [t.shape for t in output] if isinstance(output, tuple) else None
                    }
                return hook
            
            # Register hooks
            hooks = []
            for name, module in layer.named_modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    hooks.append(module.register_forward_hook(shape_hook(name)))
            
            # Forward pass
            output = layer(x)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Verify shapes
            expected_shapes = {
                'manifold_proj': {
                    'input': [(params["batch_size"] * params["seq_length"], params["hidden_dim"])],
                    'output': (params["batch_size"] * params["seq_length"], params["manifold_dim"])
                },
                'manifold_proj_inv': {
                    'input': [(params["batch_size"] * params["seq_length"], params["manifold_dim"])],
                    'output': (params["batch_size"] * params["seq_length"], params["hidden_dim"])
                }
            }
            
            for name, expected in expected_shapes.items():
                assert name in shapes, f"Missing shape for {name}"
                assert shapes[name]['input'] == expected['input'], \
                    f"Wrong input shape for {name}: expected {expected['input']}, got {shapes[name]['input']}"
                assert shapes[name]['output'] == expected['output'], \
                    f"Wrong output shape for {name}: expected {expected['output']}, got {shapes[name]['output']}"

    class TestAttentionFlow:
        """Tests for gradient flow through attention mechanism."""
        
        @pytest.mark.timeout(30)
        def test_multi_head_gradient_flow(self, setup_attention):
            """Test gradient flow through multi-head attention mechanism."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            head_gradients = []
            for i, tile in enumerate(layer.tiles):
                def make_hook(tile_idx):
                    def hook(grad):
                        if grad is not None:
                            head_gradients.append((tile_idx, grad.detach().clone()))
                        return grad
                    return hook
                
                tile.query.weight.register_hook(make_hook(i))
                tile.key.weight.register_hook(make_hook(i))
                tile.value.weight.register_hook(make_hook(i))
            
            output = layer(x)
            loss = output.abs().pow(2).sum()
            loss.backward()
            
            assert len(head_gradients) > 0, "Should have received gradients for attention heads"
            for tile_idx, grad in head_gradients:
                assert grad.abs().mean() > 0, f"Head {tile_idx} gradients should be non-zero"
                assert torch.isfinite(grad).all(), f"Head {tile_idx} gradients should be finite"
        
        @pytest.mark.timeout(30)
        def test_geometric_phases(self, setup_attention):
            """Test quantum geometric phases and their gradients."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            quantum_bridge = layer.quantum_bridge
            pattern_bundle = quantum_bridge.pattern_bundle
            riemannian_framework = pattern_bundle.riemannian_framework
            
            if hasattr(pattern_bundle, 'manifold_dim'):
                pattern_bundle.manifold_dim = params['manifold_dim']
            if hasattr(riemannian_framework, 'manifold_dim'):
                riemannian_framework.manifold_dim = params['manifold_dim']
            
            geometric_flow = RiemannianFlow(
                manifold_dim=params['manifold_dim'],
                hidden_dim=params['manifold_dim'],
                num_layers=3,
                dt=0.1,
                stability_threshold=1e-6,
                dtype=layer.dtype,
                device=x.device
            )
            
            # Ensure all parameters require gradients
            for param in geometric_flow.parameters():
                param.requires_grad_(True)
            
            pattern_bundle.geometric_flow = geometric_flow
            
            # Forward pass with direct geometric flow computation
            manifold_proj = layer.manifold_proj(x.reshape(-1, params["hidden_dim"]))
            flow_output = geometric_flow(manifold_proj.reshape(params["batch_size"], params["seq_length"], -1))
            output = layer.manifold_proj_inv(flow_output.reshape(-1, params["manifold_dim"]))
            output = output.reshape(params["batch_size"], params["seq_length"], -1)
            
            # Use a loss that directly depends on geometric phases
            loss = output.abs().pow(2).mean() + flow_output.abs().pow(2).mean()
            loss.backward()
            
            # Check gradients for all parameters in geometric_flow
            for name, param in geometric_flow.named_parameters():
                assert param.grad is not None, f"Parameter {name} should have gradient"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} gradients should be finite"
                assert param.grad.abs().mean() > 0, f"Parameter {name} gradients should be non-zero"
            
            # Verify real metric gradients specifically
            real_metric_params = [(name, p) for name, p in geometric_flow.named_parameters() if 'real_metric_net' in name]
            assert len(real_metric_params) > 0, "Should have real metric parameters"
            for name, param in real_metric_params:
                assert param.grad is not None, f"Real metric parameter {name} should have gradient"
                assert torch.isfinite(param.grad).all(), f"Real metric parameter {name} gradients should be finite"
                assert param.grad.abs().mean() > 0, f"Real metric parameter {name} gradients should be non-zero"
            
            # Verify imaginary metric gradients if they exist
            imag_metric_params = [(name, p) for name, p in geometric_flow.named_parameters() if 'imag_metric_net' in name]
            if imag_metric_params:
                for name, param in imag_metric_params:
                    assert param.grad is not None, f"Imaginary metric parameter {name} should have gradient"
                    assert torch.isfinite(param.grad).all(), f"Imaginary metric parameter {name} gradients should be finite"
                    assert param.grad.abs().mean() > 0, f"Imaginary metric parameter {name} gradients should be non-zero"
    
    class TestEnergyConservation:
        """Tests for energy conservation during gradient flow."""
        
        @pytest.mark.timeout(30)
        def test_energy_conservation(self, setup_attention):
            """Test energy conservation during forward and backward pass."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            def compute_energy(tensor):
                """Compute energy of a complex tensor."""
                return torch.sum(tensor.real ** 2 + tensor.imag ** 2)
            
            initial_energy = compute_energy(x)
            output = layer(x)
            final_energy = compute_energy(output)
            
            assert torch.allclose(initial_energy, final_energy, rtol=1e-2), \
                "Energy should be conserved during forward pass"
            
            loss = output.abs().mean()
            loss.backward()
            
            if x.grad is not None:
                grad_energy = compute_energy(x.grad)
                assert torch.isfinite(grad_energy), "Gradient energy should be finite"
                assert grad_energy > 0, "Gradient energy should be positive"
        
        @pytest.mark.timeout(30)
        def test_quantum_bridge_energy_conservation(self, setup_attention):
            """Test energy conservation in quantum bridge."""
            layer, params = setup_attention
            x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
            x.requires_grad_(True)
            
            initial_energy = torch.sum(x.abs() ** 2)
            
            x_flat = x.reshape(-1, params["hidden_dim"])
            manifold_proj = layer.manifold_proj(x_flat)
            manifold_energy = torch.sum(manifold_proj.abs() ** 2)
            
            assert torch.allclose(initial_energy, manifold_energy, rtol=1e-2), \
                "Energy should be conserved in manifold projection"
            
            output = layer(x)
            final_energy = torch.sum(output.abs() ** 2)
            
            assert torch.allclose(initial_energy, final_energy, rtol=1e-2), \
                "Energy should be conserved through quantum bridge"
            
            loss = output.abs().mean()
            loss.backward()
            
            bridge_params = [
                ('pattern_bundle.metric', layer.quantum_bridge.pattern_bundle.metric),
                ('pattern_bundle.connection', layer.quantum_bridge.pattern_bundle.connection),
                ('pattern_bundle.riemannian_framework.metric_factors',
                 layer.quantum_bridge.pattern_bundle.riemannian_framework.metric_factors)
            ]
            
            for name, param in bridge_params:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has inf/nan gradients"
                assert param.grad.abs().mean() > 0, f"Parameter {name} has zero gradients"